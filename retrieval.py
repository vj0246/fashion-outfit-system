"""
retrieval.py
============
Stage 4. Given a structured query, assembles complete outfits.

Algorithm:
  1. Filter candidate pool by gender (hard) + occasion (soft -- relaxed
     automatically if it leaves a required slot empty; 68 items is too
     small to afford hard filters that zero out a slot).
  2. Pick hero candidates: either the user's anchor item, or the top-N
     topwear/onepiece items by query-text relevance (MiniLM cosine).
  3. For each hero, work out which slots are still needed from its slot
     type (onepiece needs only footwear; topwear needs bottomwear+footwear)
     and enumerate combinations over the top-K candidates per required
     slot -- candidate pools here are tiny (a handful of items per slot
     after filtering), so exhaustive enumeration is cheap and more
     correct than a greedy heuristic. Optional slots (layer, accessory x2)
     are added greedily only if they clear a compatibility threshold.
  4. Score every full combo = blend(mean pairwise compatibility from
     compatibility.py's calibrator, mean query-relevance). Rank, dedupe,
     return top_n distinct outfits.

Run (interactive smoke test):
    python retrieval.py --processed_dir ./processed --data_dir . \\
        --gender women --occasion office --query "smart formal outfit for a client meeting"

Requires: sentence-transformers, numpy, pandas, scikit-learn
"""

import argparse
import json
import os
import pickle
from itertools import product as iproduct

import numpy as np
import pandas as pd

from compatibility import CompatibilityFeaturizer, mean_feature_breakdown

REQUIRED_SLOTS = {
    "onepiece": ["footwear"],
    "topwear": ["bottomwear", "footwear"],
    "bottomwear": ["topwear", "footwear"],
}
OPTIONAL_SLOTS = ["layer", "accessory"]
MAX_ACCESSORIES = 2
TOP_K_PER_SLOT = 6          # candidate pool size per slot before combo enumeration
OPTIONAL_THRESHOLD = 0.55    # min mean-compat-with-core to include an optional item
COMPAT_WEIGHT = 0.6
RELEVANCE_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_artifacts(processed_dir: str):
    products = pd.read_csv(os.path.join(processed_dir, "products_with_slots.csv"))
    with open(os.path.join(processed_dir, "cooccurrence_graph.json")) as f:
        graph = json.load(f)
    with open(os.path.join(processed_dir, "compatibility_model.pkl"), "rb") as f:
        compat = pickle.load(f)

    fused = np.load(os.path.join(processed_dir, "fashionclip_fused.npy"))
    minilm_emb = np.load(os.path.join(processed_dir, "minilm_item_embeddings.npy"))
    id_to_idx = {pid: i for i, pid in enumerate(products["id"])}

    featurizer = CompatibilityFeaturizer(
        products, graph, fused, id_to_idx, compat["color_vocab"], compat["color_cooc"]
    )
    return products, featurizer, compat["classifier"], minilm_emb, id_to_idx


def encode_query(query_text: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode([query_text], normalize_embeddings=True)[0]


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------
def filter_pool(products: pd.DataFrame, gender: str, occasion: str | None, slot: str) -> pd.DataFrame:
    pool = products[(products["gender"] == gender) & (products["slot"] == slot)]
    if occasion:
        strict = pool[pool["occasion"] == occasion]
        if len(strict) > 0:
            return strict
    return pool  # relaxed: drop occasion constraint if it would empty the slot


def rank_by_relevance(pool_ids: list, id_to_idx: dict, minilm_emb: np.ndarray,
                    query_vec: np.ndarray, top_k: int) -> list:
    scored = [
        (pid, float(np.dot(query_vec, minilm_emb[id_to_idx[pid]])))
        for pid in pool_ids if pid in id_to_idx
    ]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Compatibility scoring helpers
# ---------------------------------------------------------------------------
def compat_prob(featurizer: CompatibilityFeaturizer, clf, a_id: str, b_id: str) -> float:
    feat = featurizer.featurize(a_id, b_id).reshape(1, -1)
    return float(clf.predict_proba(feat)[0, 1])


def mean_pairwise_compat(featurizer, clf, ids: list) -> float:
    if len(ids) < 2:
        return 1.0
    scores = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            scores.append(compat_prob(featurizer, clf, ids[i], ids[j]))
    return float(np.mean(scores))


def mean_relevance(ids: list, id_to_idx: dict, minilm_emb: np.ndarray, query_vec: np.ndarray) -> float:
    sims = [float(np.dot(query_vec, minilm_emb[id_to_idx[i]])) for i in ids if i in id_to_idx]
    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Outfit assembly
# ---------------------------------------------------------------------------
def assemble_for_hero(hero_id: str, hero_slot: str, products, featurizer, clf,
                    id_to_idx, minilm_emb, query_vec, gender, occasion):
    required = REQUIRED_SLOTS.get(hero_slot)
    if required is None:
        return []  # hero must be onepiece/topwear/bottomwear

    # candidate pools per required slot, pre-ranked by relevance, capped
    slot_candidates = {}
    for slot in required:
        pool = filter_pool(products, gender, occasion, slot)
        ranked = rank_by_relevance(pool["id"].tolist(), id_to_idx, minilm_emb, query_vec, TOP_K_PER_SLOT)
        if not ranked:
            return []  # can't complete an outfit without this required slot
        slot_candidates[slot] = [pid for pid, _ in ranked]

    combos = []
    for combo_ids in iproduct(*[slot_candidates[s] for s in required]):
        core_ids = [hero_id] + list(combo_ids)
        combos.append(core_ids)

    results = []
    for core_ids in combos:
        final_ids = list(core_ids)

        # greedily add optional layer / up to 2 accessories if they clear threshold
        for opt_slot in OPTIONAL_SLOTS:
            limit = MAX_ACCESSORIES if opt_slot == "accessory" else 1
            pool = filter_pool(products, gender, occasion, opt_slot)
            ranked = rank_by_relevance(pool["id"].tolist(), id_to_idx, minilm_emb, query_vec, TOP_K_PER_SLOT)
            added = 0
            for pid, _ in ranked:
                if added >= limit:
                    break
                trial_compat = float(np.mean([compat_prob(featurizer, clf, pid, cid) for cid in final_ids]))
                if trial_compat >= OPTIONAL_THRESHOLD:
                    final_ids.append(pid)
                    added += 1

        compat_avg = mean_pairwise_compat(featurizer, clf, final_ids)
        relevance_avg = mean_relevance(final_ids, id_to_idx, minilm_emb, query_vec)
        score = COMPAT_WEIGHT * compat_avg + RELEVANCE_WEIGHT * relevance_avg
        results.append({"items": final_ids, "compat_avg": compat_avg,
                        "relevance_avg": relevance_avg, "score": score,
                        "feature_breakdown": mean_feature_breakdown(featurizer, final_ids)})

    return results


def recommend(query: dict, products, featurizer, clf, id_to_idx, minilm_emb, top_n: int = 3):
    """
    query = {
        "gender": "women" | "men",
        "occasion": "office" | "party" | ... | None,
        "style_text": free-text description used for relevance ranking,
        "anchor_item": optional product id to build around,
    }
    """
    query_vec = encode_query(query["style_text"])
    gender, occasion = query["gender"], query.get("occasion")

    if query.get("anchor_item"):
        hero_ids = [query["anchor_item"]]
    else:
        hero_pool = products[
            (products["gender"] == gender) & (products["slot"].isin(["onepiece", "topwear"]))
        ]
        ranked = rank_by_relevance(hero_pool["id"].tolist(), id_to_idx, minilm_emb, query_vec, top_k=5)
        hero_ids = [pid for pid, _ in ranked]

    all_combos = []
    slot_of = dict(zip(products["id"], products["slot"]))
    for hero_id in hero_ids:
        hero_slot = slot_of.get(hero_id)
        if hero_slot is None:
            continue
        all_combos.extend(
            assemble_for_hero(hero_id, hero_slot, products, featurizer, clf,
                            id_to_idx, minilm_emb, query_vec, gender, occasion)
        )

    all_combos.sort(key=lambda c: -c["score"])

    # dedupe: skip a combo if its item set heavily overlaps a better-ranked one
    chosen, seen_sets = [], []
    for combo in all_combos:
        item_set = set(combo["items"])
        if any(len(item_set & s) / len(item_set | s) > 0.6 for s in seen_sets):
            continue
        chosen.append(combo)
        seen_sets.append(item_set)
        if len(chosen) >= top_n:
            break

    return chosen


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="./processed")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--gender", required=True, choices=["men", "women"])
    ap.add_argument("--occasion", default=None)
    ap.add_argument("--query", required=True, help="Free-text style/occasion description")
    ap.add_argument("--anchor_item", default=None)
    ap.add_argument("--top_n", type=int, default=3)
    args = ap.parse_args()

    products, featurizer, clf, minilm_emb, id_to_idx = load_artifacts(args.processed_dir)

    query = {
        "gender": args.gender,
        "occasion": args.occasion,
        "style_text": args.query,
        "anchor_item": args.anchor_item,
    }
    results = recommend(query, products, featurizer, clf, id_to_idx, minilm_emb, top_n=args.top_n)

    names = dict(zip(products["id"], products["name"]))
    for rank, combo in enumerate(results, 1):
        print(f"\n--- Outfit #{rank} (score={combo['score']:.3f}, "
            f"compat={combo['compat_avg']:.3f}, relevance={combo['relevance_avg']:.3f}) ---")
        for item_id in combo["items"]:
            print(f"  - {names.get(item_id, item_id)}  [{item_id}]")


if __name__ == "__main__":
    main()
