"""
embeddings.py
=============
Stage 2 of the pipeline. Builds the multimodal item embedding space.

Models used (and why):
  - FashionCLIP ("patrickjohncyh/fashion-clip"): image+text -> joint space.
    Fashion-domain-tuned CLIP. This is the PRIMARY item representation,
    used for visual/style compatibility between items.
  - Vanilla CLIP ("openai/clip-vit-base-patch32"): same pipeline, run as an
    ABLATION baseline. Proves FashionCLIP's domain tuning actually helps
    on THIS dataset instead of just asserting it -- real evaluation
    content for your docs/video.
  - MiniLM ("sentence-transformers/all-MiniLM-L6-v2"): separate text-only
    space. Used later for QUERY <-> item-description matching, because a
    user's free-text request ("something stylish for office") doesn't
    read like a product description and CLIP's text tower is weak on
    that mismatch. This is the hybrid-search layer.

Fusion: item_vector = alpha * image_emb + (1-alpha) * text_emb, in the
FashionCLIP space. alpha is grid-searched (not guessed) against leave-
one-out retrieval recall over the 25 ground-truth outfits.

Run:
    python embeddings.py --data_dir . --processed_dir ./processed

REQUIRES REAL INTERNET to huggingface.co on first run (downloads ~600MB
FashionCLIP + ~90MB MiniLM + ~600MB vanilla CLIP). Will fail offline /
in a sandboxed network. Re-runs use the local HF cache, no re-download.

Install:
    pip install torch transformers sentence-transformers faiss-cpu pillow
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image

FASHIONCLIP_NAME = "patrickjohncyh/fashion-clip"
VANILLA_CLIP_NAME = "openai/clip-vit-base-patch32"
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Text composition -- same recipe used for both CLIP text tower and MiniLM
# ---------------------------------------------------------------------------
def build_item_text(row) -> str:
    tags = row["tags"].replace(";", ", ") if isinstance(row["tags"], str) else ""
    return (
        f"{row['name']}. Category: {row['category_label']}. "
        f"Occasion: {row['occasion']}. {row['description']} Tags: {tags}"
    )


# ---------------------------------------------------------------------------
# CLIP embedding (shared code path for FashionCLIP and vanilla CLIP)
# ---------------------------------------------------------------------------
def load_clip(model_name: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def embed_images_texts(model, processor, image_paths, texts, batch_size: int = 16):
    """Returns (image_embeds [N,D], text_embeds [N,D]), both L2-normalized."""
    import torch

    img_embeds, txt_embeds = [], []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_txts = texts[i : i + batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(
                text=batch_txts,
                images=batch_imgs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            out = model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            img_embeds.append(ie.numpy())
            txt_embeds.append(te.numpy())
    return np.vstack(img_embeds), np.vstack(txt_embeds)


def fuse(img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float) -> np.ndarray:
    fused = alpha * img_emb + (1 - alpha) * txt_emb
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return fused / norms


def build_faiss_index(vectors: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    return index


# ---------------------------------------------------------------------------
# Evaluation: leave-one-out recall@k over the 25 ground-truth outfits
# ---------------------------------------------------------------------------
def recall_at_k(fused: np.ndarray, id_to_idx: dict, parsed_outfits: list,
                slot_of: dict, k: int = 5) -> float:
    """
    For every cross-slot pair (a, b) that co-occurs in a ground-truth
    outfit: is b inside the top-k nearest neighbours of a, restricted to
    the candidate pool that shares b's slot? Mean hit-rate over all pairs.
    This is the metric used to pick alpha and to compare FashionCLIP vs
    vanilla CLIP -- NOT held-out in the strict ML sense (n=25 is too small
    for a real train/val split), so treat this as a calibration signal,
    not a generalization claim. State that explicitly in your docs.
    """
    hits, total = 0, 0
    for outfit in parsed_outfits:
        items = outfit["items"]
        for a, b in combinations(items, 2):
            if a["slot"] == b["slot"]:
                continue
            if a["id"] not in id_to_idx or b["id"] not in id_to_idx:
                continue
            pool = [pid for pid, s in slot_of.items() if s == b["slot"] and pid in id_to_idx]
            if b["id"] not in pool:
                continue
            a_vec = fused[id_to_idx[a["id"]]]
            sims = [(pid, float(np.dot(a_vec, fused[id_to_idx[pid]]))) for pid in pool]
            sims.sort(key=lambda x: -x[1])
            top_ids = [pid for pid, _ in sims[:k]]
            total += 1
            hits += int(b["id"] in top_ids)
    return hits / total if total else 0.0


def tune_alpha(img_emb, txt_emb, id_to_idx, parsed_outfits, slot_of, k=5):
    best_alpha, best_recall, history = None, -1.0, []
    for alpha in ALPHA_GRID:
        fused = fuse(img_emb, txt_emb, alpha)
        r = recall_at_k(fused, id_to_idx, parsed_outfits, slot_of, k=k)
        history.append((alpha, r))
        if r > best_recall:
            best_recall, best_alpha = r, alpha
    return best_alpha, best_recall, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".", help="Folder with products.csv (and images/)")
    ap.add_argument("--processed_dir", default="./processed", help="Output of data_pipeline.py")
    ap.add_argument("--k", type=int, default=5, help="k for recall@k tuning")
    ap.add_argument("--skip_ablation", action="store_true", help="Skip vanilla-CLIP comparison run")
    args = ap.parse_args()

    products = pd.read_csv(os.path.join(args.processed_dir, "products_with_slots.csv"))
    with open(os.path.join(args.processed_dir, "outfits_parsed.json")) as f:
        parsed_outfits = json.load(f)

    slot_of = dict(zip(products["id"], products["slot"]))
    id_to_idx = {pid: i for i, pid in enumerate(products["id"])}
    image_paths = [os.path.join(args.data_dir, p) for p in products["image"]]
    texts = [build_item_text(row) for _, row in products.iterrows()]

    # ---- FashionCLIP (primary) ----
    print(f"[1/3] Loading FashionCLIP ({FASHIONCLIP_NAME}) ...")
    model, processor = load_clip(FASHIONCLIP_NAME)
    print("      Embedding images+text ...")
    img_emb, txt_emb = embed_images_texts(model, processor, image_paths, texts)

    print("      Tuning fusion alpha via leave-one-out recall@%d ..." % args.k)
    best_alpha, best_recall, history = tune_alpha(img_emb, txt_emb, id_to_idx, parsed_outfits, slot_of, k=args.k)
    for a, r in history:
        print(f"        alpha={a:.1f}  recall@{args.k}={r:.3f}")
    print(f"      -> best alpha = {best_alpha} (recall@{args.k} = {best_recall:.3f})")

    fused = fuse(img_emb, txt_emb, best_alpha)
    index = build_faiss_index(fused)

    import faiss

    faiss.write_index(index, os.path.join(args.processed_dir, "fashionclip.index"))
    np.save(os.path.join(args.processed_dir, "fashionclip_fused.npy"), fused)

    fashionclip_recall = best_recall

    # ---- Vanilla CLIP ablation ----
    vanilla_recall = None
    if not args.skip_ablation:
        print(f"\n[2/3] Loading vanilla CLIP ({VANILLA_CLIP_NAME}) for ablation ...")
        v_model, v_processor = load_clip(VANILLA_CLIP_NAME)
        v_img_emb, v_txt_emb = embed_images_texts(v_model, v_processor, image_paths, texts)
        v_alpha, v_recall, _ = tune_alpha(v_img_emb, v_txt_emb, id_to_idx, parsed_outfits, slot_of, k=args.k)
        print(f"      vanilla CLIP best alpha={v_alpha}, recall@{args.k}={v_recall:.3f}")
        print(f"      FashionCLIP recall@{args.k}={fashionclip_recall:.3f} vs vanilla CLIP={v_recall:.3f} "
              f"({'+' if fashionclip_recall >= v_recall else ''}{(fashionclip_recall - v_recall):.3f})")
        vanilla_recall = v_recall
    else:
        print("\n[2/3] Skipping vanilla CLIP ablation (--skip_ablation)")

    # ---- MiniLM query encoder ----
    print(f"\n[3/3] Loading MiniLM ({MINILM_NAME}) for query<->description matching ...")
    from sentence_transformers import SentenceTransformer

    minilm = SentenceTransformer(MINILM_NAME)
    minilm_emb = minilm.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    np.save(os.path.join(args.processed_dir, "minilm_item_embeddings.npy"), minilm_emb)

    # ---- persist metadata ----
    meta = {
        "ids_order": products["id"].tolist(),
        "best_alpha": best_alpha,
        "fashionclip_recall_at_k": fashionclip_recall,
        "vanilla_clip_recall_at_k": vanilla_recall,
        "k": args.k,
    }
    with open(os.path.join(args.processed_dir, "embeddings_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Wrote to {args.processed_dir}/: fashionclip.index, fashionclip_fused.npy, "
          f"minilm_item_embeddings.npy, embeddings_meta.json")


if __name__ == "__main__":
    main()
