"""
make_ablation_chart.py
=======================
Reads processed/embeddings_meta.json (written by embeddings.py) and saves
a bar chart comparing FashionCLIP vs vanilla CLIP recall@k -- the real
ablation number, not an assertion. Drop the PNG into your README/demo
video.

Run (after embeddings.py has been run WITHOUT --skip_ablation):
    python make_ablation_chart.py --processed_dir ./processed

Requires: matplotlib  (pip install matplotlib)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="./processed")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <processed_dir>/ablation_chart.png)")
    args = ap.parse_args()

    meta_path = os.path.join(args.processed_dir, "embeddings_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found -- run embeddings.py first (without --skip_ablation)."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    fc_recall = meta.get("fashionclip_recall_at_k")
    clip_recall = meta.get("vanilla_clip_recall_at_k")
    k = meta.get("k", 5)

    if clip_recall is None:
        raise ValueError(
            "vanilla_clip_recall_at_k is null -- embeddings.py was run with --skip_ablation, "
            "re-run it without that flag to get a real comparison number."
        )

    labels = ["FashionCLIP\n(fashion-domain-tuned)", "Vanilla CLIP\n(openai/clip-vit-base-patch32)"]
    values = [fc_recall, clip_recall]
    colors = ["#7a2331", "#9a9082"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"Leave-one-out recall@{k}")
    ax.set_title("FashionCLIP vs vanilla CLIP\n(real numbers, this 68-item / 25-outfit catalog)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    delta = fc_recall - clip_recall
    ax.text(
        0.5, 0.95,
        f"{'+' if delta >= 0 else ''}{delta:.3f} recall@{k} for the domain-tuned model",
        transform=ax.transAxes, ha="center", fontsize=9, color="#6b6356",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = args.out or os.path.join(args.processed_dir, "ablation_chart.png")
    fig.savefig(out_path, dpi=150)
    print(f"[OK] Wrote {out_path}  (FashionCLIP={fc_recall:.3f}, vanilla CLIP={clip_recall:.3f}, delta={delta:+.3f})")


if __name__ == "__main__":
    main()
