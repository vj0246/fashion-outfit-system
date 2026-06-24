"""
compatibility.py
=================
Stage 3. Item-item compatibility scoring.

Three frozen signals, blended by a learned calibrator:
  1. RULE features:
     - same occasion (1.0) vs different (0.2 baseline -- outfits do mix
       occasion-tagged pieces, it's not a hard veto)
     - wear_type match. NOTE: data_pipeline.py found wear_type is broken
       for footwear/accessory rows (overwritten with their own category
       instead of western/ethnic). Treated as 'unknown' -> neutral 0.5,
       NOT a real mismatch.
     - color harmony: colors extracted via keyword match against a
       vocabulary built directly from outfits.csv's `palette` field
       (15 real color tokens actually used by the stylist), scored by
       how often each color PAIR co-occurs in the 25 real outfits. This
       is data-driven, not invented color theory.
  2. GRAPH feature: normalized item-item co-occurrence weight from
     data_pipeline.py's graph (the graph-rec signal).
  3. EMBEDDING feature: cosine similarity in the fused FashionCLIP space
     from embeddings.py.

A logistic regression CALIBRATOR (5 features above) learns how to weight
these signals. Trained on real cross-slot pairs from the 25 curated
outfits (positives) vs sampled non-co-occurring pairs (negatives).
Validated by LEAVE-ONE-OUTFIT-OUT -- the only defensible scheme at n=25.
This is a calibration step on frozen signals, not a generalizable deep
compatibility model trained from scratch -- say so explicitly in docs.

Run:
    python compatibility.py --processed_dir ./processed

Requires: scikit-learn, numpy, pandas  (pip install scikit-learn)
"""

import argparse
import json
import os
import pickle
import re
from itertools import combinations

import numpy as np
import pandas as pd

NEG_RATIO = 3
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Color vocabulary + co-occurrence, built from outfits.csv `palette` field
# ---------------------------------------------------------------------------
def build_color_vocab_and_cooc(parsed_outfits: list):
    vocab = set()
    palette_lists = []
    for outfit in parsed_outfits:
        tokens = [t.strip().lower() for t in outfit["palette"].split("/") if t.strip()]
        palette_lists.append(tokens)
        vocab.update(tokens)

    cooc = {}
    for tokens in palette_lists:
        for a, b in combinations(set(tokens), 2):
            key = frozenset({a, b})
            cooc[key] = cooc.get(key, 0) + 1
    return vocab, cooc


def extract_colors(text: str, vocab: set) -> set:
    if not isinstance(text, str):
        return set()
    text_low = text.lower()
    found = set()
    for color in vocab:
        if re.search(rf"\b{re.escape(color)}\b", text_low):
            found.add(color)
    return found


def color_score(colors_a: set, colors_b: set, cooc: dict, max_w: int = 3) -> float:
    if not colors_a or not colors_b:
        return 0.3  # unknown -- modest neutral, not penalized to 0
    best = 0.0
    for ca in colors_a:
        for cb in colors_b:
            if ca == cb:
                best = max(best, 1.0)
                continue
            w = cooc.get(frozenset({ca, cb}), 0)
            best = max(best, min(w, max_w) / max_w * 0.9)  # cap below same-color score
    return best


# ---------------------------------------------------------------------------
# Feature vector for a pair of items
# ---------------------------------------------------------------------------
class CompatibilityFeaturizer:
    def __init__(self, products: pd.DataFrame, graph: dict, fused: np.ndarray,
                id_to_idx: dict, color_vocab: set, color_cooc: dict):
        self.rows = products.set_index("id")
        self.graph = graph
        self.fused = fused
        self.id_to_idx = id_to_idx
        self.color_vocab = color_vocab
        self.color_cooc = color_cooc
        self._color_cache = {}

    def _colors_of(self, item_id: str) -> set:
        if item_id not in self._color_cache:
            row = self.rows.loc[item_id]
            blob = f"{row['name']} {row['description']}"
            self._color_cache[item_id] = extract_colors(blob, self.color_vocab)
        return self._color_cache[item_id]

    def featurize(self, a_id: str, b_id: str) -> np.ndarray:
        ra, rb = self.rows.loc[a_id], self.rows.loc[b_id]

        same_occasion = 1.0 if ra["occasion"] == rb["occasion"] else 0.2

        wa, wb = ra["wear_type"], rb["wear_type"]
        valid_wear = {"western", "ethnic"}
        if wa in valid_wear and wb in valid_wear:
            wear_match = 1.0 if wa == wb else 0.0
        else:
            wear_match = 0.5  # unknown / corrupted field -> neutral

        col_score = color_score(self._colors_of(a_id), self._colors_of(b_id), self.color_cooc)

        gw = self.graph.get(a_id, {}).get(b_id, 0)
        graph_feat = min(gw, 3) / 3.0

        if a_id in self.id_to_idx and b_id in self.id_to_idx:
            va, vb = self.fused[self.id_to_idx[a_id]], self.fused[self.id_to_idx[b_id]]
            cos = float(np.dot(va, vb))  # vectors are L2-normalized already
            emb_feat = (cos + 1.0) / 2.0  # rescale [-1,1] -> [0,1]
        else:
            emb_feat = 0.5

        return np.array([emb_feat, col_score, graph_feat, same_occasion, wear_match])


FEATURE_NAMES = ["embedding_cos", "color_score", "graph_score", "same_occasion", "wear_match"]


def mean_feature_breakdown(featurizer: "CompatibilityFeaturizer", ids: list) -> dict:
    """Mean of each of the 5 raw features across every pair in `ids`.
    Used for the UI's per-outfit explainability breakdown (not the
    calibrator's blended probability -- the raw signals behind it)."""
    if len(ids) < 2:
        return {name: None for name in FEATURE_NAMES}
    vecs = [
        featurizer.featurize(ids[i], ids[j])
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    ]
    mean_vec = np.mean(vecs, axis=0)
    return {name: round(float(v), 3) for name, v in zip(FEATURE_NAMES, mean_vec)}


# ---------------------------------------------------------------------------
# Training set construction
# ---------------------------------------------------------------------------
def build_training_pairs(parsed_outfits: list, all_ids: list, graph: dict, rng):
    positives = []  # (a, b, outfit_idx)
    for oi, outfit in enumerate(parsed_outfits):
        ids = [it["id"] for it in outfit["items"]]
        for a, b in combinations(ids, 2):
            positives.append((a, b, oi))

    n_neg = len(positives) * NEG_RATIO
    negatives = []
    attempts = 0
    while len(negatives) < n_neg and attempts < n_neg * 50:
        attempts += 1
        a, b = rng.choice(all_ids), rng.choice(all_ids)
        if a == b:
            continue
        if graph.get(a, {}).get(b, 0) > 0:
            continue  # actually co-occurs somewhere -- not a clean negative
        negatives.append((a, b, -1))  # -1 = no outfit owns this negative

    return positives, negatives


def build_cooc_graph_from_outfits(outfits_subset: list) -> dict:
    """Same logic as data_pipeline.build_cooccurrence_graph, but over a subset
    of outfits -- used to rebuild a leakage-free graph per CV fold."""
    graph = {}
    for outfit in outfits_subset:
        ids = [it["id"] for it in outfit["items"]]
        for a, b in combinations(ids, 2):
            graph.setdefault(a, {})[b] = graph.get(a, {}).get(b, 0) + 1
            graph.setdefault(b, {})[a] = graph.get(b, {}).get(a, 0) + 1
    return graph


def build_color_cooc_from_outfits(outfits_subset: list) -> dict:
    cooc = {}
    for outfit in outfits_subset:
        tokens = [t.strip().lower() for t in outfit["palette"].split("/") if t.strip()]
        for a, b in combinations(set(tokens), 2):
            key = frozenset({a, b})
            cooc[key] = cooc.get(key, 0) + 1
    return cooc


def leave_one_outfit_out_eval(parsed_outfits, positives, negatives, products,
                            fused, id_to_idx, color_vocab):
    """
    IMPORTANT: graph_score and color_score are both derived from the 25
    outfits themselves. If fold N's evaluation used a graph/color-cooc table
    built from ALL 25 outfits, the held-out outfit's own edges would still
    be sitting in that table -- the model would be scored on data it was
    implicitly given the answer to. Each fold rebuilds both tables from the
    24 *other* outfits only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    accs, aucs = [], []
    n_outfits = len(parsed_outfits)
    for held_out in range(n_outfits):
        fold_outfits = [o for i, o in enumerate(parsed_outfits) if i != held_out]
        fold_graph = build_cooc_graph_from_outfits(fold_outfits)
        fold_color_cooc = build_color_cooc_from_outfits(fold_outfits)
        fold_featurizer = CompatibilityFeaturizer(
            products, fold_graph, fused, id_to_idx, color_vocab, fold_color_cooc
        )

        train_pos = [(a, b) for a, b, oi in positives if oi != held_out]
        test_pos = [(a, b) for a, b, oi in positives if oi == held_out]
        if not test_pos:
            continue

        neg_X = np.array([fold_featurizer.featurize(a, b) for a, b, _ in negatives])
        neg_y = np.zeros(len(negatives))

        train_X = np.vstack([
            np.array([fold_featurizer.featurize(a, b) for a, b in train_pos]),
            neg_X,
        ])
        train_y = np.concatenate([np.ones(len(train_pos)), neg_y])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(train_X, train_y)

        test_X = np.array([fold_featurizer.featurize(a, b) for a, b in test_pos])
        test_y = np.ones(len(test_pos))
        # mix in an equal number of negatives for a meaningful eval set
        sample_neg_idx = np.random.choice(len(negatives), size=len(test_pos), replace=False)
        eval_X = np.vstack([test_X, neg_X[sample_neg_idx]])
        eval_y = np.concatenate([test_y, np.zeros(len(test_pos))])

        preds = clf.predict(eval_X)
        accs.append(accuracy_score(eval_y, preds))
        try:
            probs = clf.predict_proba(eval_X)[:, 1]
            aucs.append(roc_auc_score(eval_y, probs))
        except ValueError:
            pass  # can happen with tiny eval sets, skip AUC that fold

    return float(np.mean(accs)), (float(np.mean(aucs)) if aucs else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="./processed")
    args = ap.parse_args()

    products = pd.read_csv(os.path.join(args.processed_dir, "products_with_slots.csv"))
    with open(os.path.join(args.processed_dir, "outfits_parsed.json")) as f:
        parsed_outfits = json.load(f)
    with open(os.path.join(args.processed_dir, "cooccurrence_graph.json")) as f:
        graph = json.load(f)

    fused_path = os.path.join(args.processed_dir, "fashionclip_fused.npy")
    if not os.path.exists(fused_path):
        raise FileNotFoundError(
            f"{fused_path} not found -- run embeddings.py first (stage 2)."
        )
    fused = np.load(fused_path)
    id_to_idx = {pid: i for i, pid in enumerate(products["id"])}

    color_vocab, color_cooc = build_color_vocab_and_cooc(parsed_outfits)
    print(f"Color vocabulary ({len(color_vocab)} tokens): {sorted(color_vocab)}")

    featurizer = CompatibilityFeaturizer(products, graph, fused, id_to_idx, color_vocab, color_cooc)

    rng = np.random.default_rng(RANDOM_SEED)
    all_ids = products["id"].tolist()
    positives, negatives = build_training_pairs(parsed_outfits, all_ids, graph, rng)
    print(f"Training pairs: {len(positives)} positive, {len(negatives)} negative "
          f"(ratio {NEG_RATIO}x)")

    acc, auc = leave_one_outfit_out_eval(parsed_outfits, positives, negatives, products,
                                        fused, id_to_idx, color_vocab)
    print(f"Leave-one-outfit-out: accuracy={acc:.3f}" + (f", AUC={auc:.3f}" if auc else ""))
    print("(n=25 outfits -- treat this as a calibration sanity check, not a")
    print(" generalization guarantee. State that plainly in your docs.)")

    # final calibrator trained on ALL positives + negatives, for production use
    from sklearn.linear_model import LogisticRegression

    pos_X = np.array([featurizer.featurize(a, b) for a, b, _ in positives])
    neg_X = np.array([featurizer.featurize(a, b) for a, b, _ in negatives])
    X = np.vstack([pos_X, neg_X])
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])

    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    final_clf.fit(X, y)
    print("\nLearned feature weights (higher = more important):")
    for name, w in zip(FEATURE_NAMES, final_clf.coef_[0]):
        print(f"  {name:>15s}: {w:+.3f}")

    # persist
    with open(os.path.join(args.processed_dir, "compatibility_model.pkl"), "wb") as f:
        pickle.dump(
            {
                "classifier": final_clf,
                "color_vocab": color_vocab,
                "color_cooc": color_cooc,
                "feature_names": FEATURE_NAMES,
                "leave_one_out_accuracy": acc,
                "leave_one_out_auc": auc,
            },
            f,
        )
    print(f"\n[OK] Wrote {args.processed_dir}/compatibility_model.pkl")


if __name__ == "__main__":
    main()
