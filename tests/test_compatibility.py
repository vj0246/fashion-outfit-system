import numpy as np

import compatibility as C


def test_build_color_vocab_and_cooc(parsed_outfits):
    vocab, cooc = C.build_color_vocab_and_cooc(parsed_outfits)
    assert {"blue", "black", "red", "gold"}.issubset(vocab)
    # blue/black co-occur in outfit_1's palette, red/gold in outfit_2's
    assert cooc[frozenset({"blue", "black"})] == 1
    assert cooc[frozenset({"red", "gold"})] == 1


def test_extract_colors_matches_whole_words_only():
    vocab = {"red", "tan"}
    # "tan" should not match inside "instant" -- regex must be word-bounded
    assert C.extract_colors("an instant classic", vocab) == set()
    assert C.extract_colors("a red and tan blazer", vocab) == {"red", "tan"}


def test_color_score_same_color_is_best():
    cooc = {}
    assert C.color_score({"red"}, {"red"}, cooc) == 1.0


def test_color_score_unknown_is_neutral_not_penalized():
    assert C.color_score(set(), {"red"}, {}) == 0.3


def test_featurizer_returns_five_features_in_range(products_df, parsed_outfits,
                                                    fused_embeddings, id_to_idx):
    graph = __import__("data_pipeline").build_cooccurrence_graph(parsed_outfits)
    vocab, cooc = C.build_color_vocab_and_cooc(parsed_outfits)
    feat = C.CompatibilityFeaturizer(products_df, graph, fused_embeddings, id_to_idx, vocab, cooc)

    vec = feat.featurize("p1", "p2")
    assert vec.shape == (5,)
    assert np.all(vec >= 0) and np.all(vec <= 1)


def test_featurizer_wear_type_bug_handled_as_neutral(products_df, parsed_outfits,
                                                    fused_embeddings, id_to_idx):
    """Real assignment dataset has wear_type corrupted to 'footwear'/'accessory'
    on those rows instead of western/ethnic (see data_pipeline.py's dataset
    report). The featurizer must treat that as neutral (0.5), not as a
    genuine western/ethnic mismatch."""
    df = products_df.copy()
    df.loc[df["id"] == "p3", "wear_type"] = "footwear"  # simulate the real bug
    graph = __import__("data_pipeline").build_cooccurrence_graph(parsed_outfits)
    vocab, cooc = C.build_color_vocab_and_cooc(parsed_outfits)
    feat = C.CompatibilityFeaturizer(df, graph, fused_embeddings, id_to_idx, vocab, cooc)

    vec = feat.featurize("p1", "p3")
    wear_match_idx = C.FEATURE_NAMES.index("wear_match")
    assert vec[wear_match_idx] == 0.5


def test_mean_feature_breakdown_keys_and_single_item(products_df, parsed_outfits,
                                                    fused_embeddings, id_to_idx):
    graph = __import__("data_pipeline").build_cooccurrence_graph(parsed_outfits)
    vocab, cooc = C.build_color_vocab_and_cooc(parsed_outfits)
    feat = C.CompatibilityFeaturizer(products_df, graph, fused_embeddings, id_to_idx, vocab, cooc)

    breakdown = C.mean_feature_breakdown(feat, ["p1", "p2", "p3"])
    assert set(breakdown.keys()) == set(C.FEATURE_NAMES)
    assert all(0 <= v <= 1 for v in breakdown.values())

    single = C.mean_feature_breakdown(feat, ["p1"])
    assert all(v is None for v in single.values())
