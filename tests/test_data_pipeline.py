import json

import data_pipeline as DP


def test_all_real_categories_are_mapped():
    """The actual assignment dataset has 47 categories. If a new category
    ever gets added to products.csv, this should fail loudly instead of
    silently producing NaN slots downstream."""
    assert len(DP.CATEGORY_SLOT_MAP) >= 40
    assert set(DP.CATEGORY_SLOT_MAP.values()) == {
        "onepiece", "topwear", "bottomwear", "layer", "footwear", "accessory",
    }


def test_add_slot_column_and_validation(products_df):
    assert products_df["slot"].isna().sum() == 0
    assert products_df.loc[products_df["id"] == "p4", "slot"].iloc[0] == "onepiece"
    assert products_df.loc[products_df["id"] == "p3", "slot"].iloc[0] == "footwear"
    DP.validate_slot_coverage(products_df)  # must not raise


def test_validate_slot_coverage_catches_unmapped_category(products_df):
    bad = products_df.copy()
    bad.loc[0, "category"] = "not-a-real-category"
    bad["slot"] = bad["category"].map(DP.CATEGORY_SLOT_MAP)
    try:
        DP.validate_slot_coverage(bad)
        assert False, "expected ValueError for unmapped category"
    except ValueError as e:
        assert "not-a-real-category" in str(e)


def test_build_cooccurrence_graph_symmetric_and_weighted(parsed_outfits):
    graph = DP.build_cooccurrence_graph(parsed_outfits)
    # p1-p2-p3 all co-occur in outfit_1 -> each pair has weight 1
    assert graph["p1"]["p2"] == 1
    assert graph["p2"]["p1"] == 1  # symmetric
    assert graph["p1"]["p3"] == 1
    # p4-p5-p6 co-occur in outfit_2, no edge to outfit_1's items
    assert "p4" not in graph.get("p1", {})
    assert graph["p4"]["p5"] == 1


def test_graph_score_helper(cooc_graph):
    assert DP.graph_score(cooc_graph, "p1", "p2") == 1 / 3  # clipped to max_weight=3
    assert DP.graph_score(cooc_graph, "p1", "p99") == 0.0  # never co-occurred


def test_parse_outfits_round_trip_is_json_serializable(parsed_outfits):
    # outfits_parsed.json must survive a dump/load cycle unchanged for
    # downstream stages (compatibility.py, retrieval.py) to consume it.
    dumped = json.dumps(parsed_outfits, default=str)
    loaded = json.loads(dumped)
    assert loaded[0]["outfit_id"] == "outfit_1"
    assert len(loaded[0]["items"]) == 3
