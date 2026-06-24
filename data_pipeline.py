"""
data_pipeline.py
=================
Stage 1 of the outfit recommendation pipeline.

Responsibilities:
  1. Load products.csv + outfits.csv
  2. Map every product category -> a functional SLOT (topwear / bottomwear /
     footwear / layer / accessory / onepiece)
  3. Parse the 25 curated outfits into a clean item-list representation
  4. Build an item-item co-occurrence graph from those 25 outfits
     (this is the "Fashion Graph-Based Recommendation" signal)
  5. Print a dataset analysis report (counts, missing values, palette spread,
     known data-quality issues) -> paste this into your README's
     "Dataset Analysis" section
  6. Persist everything to ./processed/ for the next pipeline stages to consume

Run:
    python data_pipeline.py --data_dir . --out_dir ./processed

Requires: pandas (pip install pandas)
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import pandas as pd

# ---------------------------------------------------------------------------
# 1. CATEGORY -> SLOT MAP
# ---------------------------------------------------------------------------
# Built by hand from the 47 categories present in products.csv.
# ONEPIECE = a single garment that already covers top+bottom (dress, saree,
# sherwani, suit, co-ord set, kurta set...). These items do NOT need a
# separate bottomwear slot filled.
CATEGORY_SLOT_MAP = {
    # ---- ONEPIECE ----
    "casual-dresses": "onepiece",
    "maxi-dresses": "onepiece",
    "party-dresses": "onepiece",
    "co-ord-sets": "onepiece",
    "kurta-sets": "onepiece",
    "salwar-suits": "onepiece",
    "sharara-sets": "onepiece",
    "sherwanis": "onepiece",
    "suits": "onepiece",
    "wedding-sarees": "onepiece",
    # ---- TOPWEAR ----
    "casual-shirts": "topwear",
    "formal-shirts": "topwear",
    "linen-shirts": "topwear",
    "party-shirts": "topwear",
    "polo-tshirts": "topwear",
    "tshirts": "topwear",
    "tops": "topwear",
    "activewear": "topwear",  # context-dependent, default topwear
    # ---- BOTTOMWEAR ----
    "chinos": "bottomwear",
    "jeans": "bottomwear",
    "leggings": "bottomwear",
    "shorts": "bottomwear",
    "skirts": "bottomwear",
    "track-pants": "bottomwear",
    "trousers": "bottomwear",
    # ---- LAYER ----
    "blazers": "layer",
    "denim-jackets": "layer",
    "long-coats": "layer",
    "nehru-jackets": "layer",
    "sweaters": "layer",
    "sweatshirts": "layer",
    # ---- FOOTWEAR ----
    "boots": "footwear",
    "ethnic-footwear": "footwear",
    "flats": "footwear",
    "formal-shoes": "footwear",
    "heels": "footwear",
    "loafers": "footwear",
    "running-shoes": "footwear",
    "sandals": "footwear",
    "sneakers": "footwear",
    # ---- ACCESSORY ----
    "caps": "accessory",
    "clutches": "accessory",
    "earrings": "accessory",
    "handbags": "accessory",
    "necklaces": "accessory",
    "sunglasses": "accessory",
    "watches": "accessory",
}

# Outfit role columns -> generic role name (used when parsing outfits.csv)
OUTFIT_ROLE_COLUMNS = [
    ("hero", "hero_id"),
    ("second", "second_id"),
    ("layer", "layer_id"),
    ("footwear", "footwear_id"),
    ("accessory_1", "accessory_1_id"),
    ("accessory_2", "accessory_2_id"),
]


# ---------------------------------------------------------------------------
# 2. LOAD + SLOT-MAP PRODUCTS
# ---------------------------------------------------------------------------
def load_data(data_dir: str):
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))
    outfits = pd.read_csv(os.path.join(data_dir, "outfits.csv"))
    return products, outfits


def add_slot_column(products: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()
    products["slot"] = products["category"].map(CATEGORY_SLOT_MAP)
    return products


def validate_slot_coverage(products: pd.DataFrame) -> None:
    unmapped = products.loc[products["slot"].isna(), "category"].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"CATEGORY_SLOT_MAP is missing these categories: {sorted(unmapped)}. "
            "Add them to CATEGORY_SLOT_MAP before continuing."
        )


# ---------------------------------------------------------------------------
# 3. PARSE OUTFITS INTO CLEAN ITEM LISTS
# ---------------------------------------------------------------------------
def parse_outfits(outfits: pd.DataFrame, products: pd.DataFrame) -> list:
    """
    Returns a list of dicts, one per outfit:
    {
        "outfit_id": "outfit W1",
        "gender": "women", "occasion": "party", "wear_type": "western",
        "theme": "Black cocktail", "palette": "black / red",
        "stylist_rationale": "...",
        "items": [{"id": "ajio_703182002", "role": "hero", "slot": "onepiece"}, ...]
    }
    """
    slot_lookup = dict(zip(products["id"], products["slot"]))
    parsed = []
    for _, row in outfits.iterrows():
        items = []
        for role_name, id_col in OUTFIT_ROLE_COLUMNS:
            item_id = row.get(id_col)
            if pd.isna(item_id):
                continue
            slot = slot_lookup.get(item_id)
            if slot is None:
                raise ValueError(
                    f"Outfit {row['outfit_id']} references unknown product id {item_id}"
                )
            items.append({"id": item_id, "role": role_name, "slot": slot})

        parsed.append(
            {
                "outfit_id": row["outfit_id"],
                "gender": row["gender"],
                "occasion": row["occasion"],
                "wear_type": row["wear_type"],
                "theme": row["theme"],
                "palette": row["palette"],
                "stylist_rationale": row["stylist_rationale"],
                "items": items,
            }
        )
    return parsed


# ---------------------------------------------------------------------------
# 4. CO-OCCURRENCE GRAPH (graph-based recommendation signal)
# ---------------------------------------------------------------------------
def build_cooccurrence_graph(parsed_outfits: list) -> dict:
    """
    Undirected weighted graph: item_id -> {other_item_id: weight}
    weight = number of curated outfits in which the pair co-occurs.
    """
    graph = defaultdict(lambda: defaultdict(int))
    for outfit in parsed_outfits:
        ids = [it["id"] for it in outfit["items"]]
        for a, b in combinations(ids, 2):
            graph[a][b] += 1
            graph[b][a] += 1
    # convert to plain dicts for JSON serialisation
    return {k: dict(v) for k, v in graph.items()}


def graph_score(graph: dict, item_a: str, item_b: str) -> float:
    """Normalized co-occurrence score in [0, 1]. 0 if pair never co-occurred."""
    max_weight = 3  # observed max in this dataset; clip defensively
    w = graph.get(item_a, {}).get(item_b, 0)
    return min(w, max_weight) / max_weight


# ---------------------------------------------------------------------------
# 5. DATASET ANALYSIS REPORT
# ---------------------------------------------------------------------------
def dataset_report(products: pd.DataFrame, outfits: pd.DataFrame) -> str:
    lines = []
    add = lines.append

    add("=" * 70)
    add("DATASET ANALYSIS REPORT")
    add("=" * 70)

    add(f"\nProducts: {len(products)} | Outfits (ground truth): {len(outfits)}")

    add("\n-- Slot distribution --")
    add(products["slot"].value_counts().to_string())

    add("\n-- Category distribution (raw) --")
    add(products["category"].value_counts().to_string())

    add("\n-- Gender split --")
    add(products["gender"].value_counts().to_string())

    add("\n-- Wear type split --")
    add(products["wear_type"].value_counts().to_string())

    add("\n-- Occasion split --")
    add(products["occasion"].value_counts().to_string())

    add("\n-- Missing values per column --")
    miss = products.isna().sum()
    add(miss[miss > 0].to_string() if miss.any() else "None")

    add("\n-- Outfit size distribution (items per outfit) --")
    add(outfits["items_count"].value_counts().sort_index().to_string())

    add("\n-- Palette distribution (top 10) --")
    add(outfits["palette"].value_counts().head(10).to_string())

    add("\n-- Known data-quality issues / challenges --")
    add("- Only 68 unique products / 25 outfits: too small for training any")
    add("  deep model from scratch. Forces a frozen-embedding + rule/graph")
    add("  hybrid approach rather than end-to-end learning.")
    bad_wear = products.loc[products["wear_type"].isin(["footwear", "accessory"])]
    add(f"- DATA BUG: wear_type should be western/ethnic only, but {len(bad_wear)} "
        "rows have it overwritten with their own category-type ('footwear'/"
        "'accessory') instead of a real style label. wear_type is NOT trustworthy "
        "for footwear/accessory rows -- treat as 'unknown' (neutral score) "
        "rather than a genuine western/ethnic mismatch downstream.")
    add("- No structured color/baseColour field on products.csv. Color only")
    add("  exists as free text inside `description` (e.g. 'White solid...',")
    add("  'Dark grey sweatshirt...'). Color compatibility extracted via")
    add("  keyword matching against a vocabulary built from outfits.csv's")
    add("  `palette` field (data-driven, not invented color theory).")
    add(f"- rating missing on {products['rating'].isna().sum()}/{len(products)} items, "
        f"rating_count missing on {products['rating_count'].isna().sum()}/{len(products)} "
        "-> not usable as a ranking signal, dropped.")
    add("- Outfits have variable composition (3-5 items): some skip 'second'")
    add("  (one-piece hero), most skip 'accessory_2'. Pipeline must treat slots")
    add("  as optional, not fixed-length.")
    add("- Only 2 genders represented (men/women); no unisex/non-binary")
    add("  category — recommendation logic inherits this binary split.")
    add("- 3 source sites (ajio/myntra/nykaa) -> inconsistent description style")
    add("  per site, handled by relying on embeddings over raw text patterns.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".", help="Folder with products.csv/outfits.csv")
    ap.add_argument("--out_dir", default="./processed", help="Output folder")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    products, outfits = load_data(args.data_dir)
    products = add_slot_column(products)
    validate_slot_coverage(products)

    parsed_outfits = parse_outfits(outfits, products)
    graph = build_cooccurrence_graph(parsed_outfits)

    report = dataset_report(products, outfits)
    print(report)

    # persist artifacts
    products.to_csv(os.path.join(args.out_dir, "products_with_slots.csv"), index=False)
    with open(os.path.join(args.out_dir, "outfits_parsed.json"), "w") as f:
        json.dump(parsed_outfits, f, indent=2, default=str)
    with open(os.path.join(args.out_dir, "cooccurrence_graph.json"), "w") as f:
        json.dump(graph, f, indent=2)
    with open(os.path.join(args.out_dir, "category_slot_map.json"), "w") as f:
        json.dump(CATEGORY_SLOT_MAP, f, indent=2)
    with open(os.path.join(args.out_dir, "dataset_report.txt"), "w") as f:
        f.write(report)

    print(f"\n[OK] Wrote artifacts to {args.out_dir}/:")
    print("  - products_with_slots.csv")
    print("  - outfits_parsed.json")
    print("  - cooccurrence_graph.json")
    print("  - category_slot_map.json")
    print("  - dataset_report.txt")


if __name__ == "__main__":
    main()
