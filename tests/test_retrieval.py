import numpy as np
import pytest

import compatibility as C
import data_pipeline as DP
import retrieval as R


@pytest.fixture
def featurizer_and_clf(products_df, parsed_outfits, fused_embeddings, id_to_idx):
    graph = DP.build_cooccurrence_graph(parsed_outfits)
    vocab, cooc = C.build_color_vocab_and_cooc(parsed_outfits)
    featurizer = C.CompatibilityFeaturizer(products_df, graph, fused_embeddings, id_to_idx, vocab, cooc)

    # tiny trained-on-the-fly calibrator -- same shape as compatibility.py's,
    # just doesn't need a real LogisticRegression fit for these logic tests
    class DummyClf:
        def predict_proba(self, X):
            # higher graph_score -> higher compat probability, deterministic
            graph_col = X[:, C.FEATURE_NAMES.index("graph_score")]
            p = 0.5 + 0.4 * graph_col
            return np.stack([1 - p, p], axis=1)

    return featurizer, DummyClf()


@pytest.fixture(autouse=True)
def mock_encode_query(monkeypatch):
    def fake_encode_query(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(32).astype("float32")
        return v / np.linalg.norm(v)
    monkeypatch.setattr(R, "encode_query", fake_encode_query)


def test_filter_pool_relaxes_occasion_when_it_would_empty_a_slot(products_df):
    # no men's "party" footwear exists in the fixture -> must relax to all men's footwear
    pool = R.filter_pool(products_df, "men", "party", "footwear")
    assert len(pool) > 0
    assert set(pool["id"]) == {"p9"}


def test_filter_pool_keeps_strict_occasion_when_available(products_df):
    pool = R.filter_pool(products_df, "women", "office", "footwear")
    assert set(pool["id"]) == {"p3"}


def test_assemble_for_hero_onepiece_only_needs_footwear(products_df, fused_embeddings,
                                                        id_to_idx, featurizer_and_clf):
    featurizer, clf = featurizer_and_clf
    query_vec = R.encode_query("red party look")
    results = R.assemble_for_hero(
        "p4", "onepiece", products_df, featurizer, clf, id_to_idx,
        fused_embeddings, query_vec, gender="women", occasion="party",
    )
    assert len(results) > 0
    for combo in results:
        assert "p4" in combo["items"]
        # exactly one footwear item must be present (the required slot for onepiece)
        footwear_in_combo = [i for i in combo["items"] if i == "p5"]
        assert len(footwear_in_combo) == 1


def test_assemble_for_hero_topwear_needs_bottomwear_and_footwear(products_df, fused_embeddings,
                                                                id_to_idx, featurizer_and_clf):
    featurizer, clf = featurizer_and_clf
    query_vec = R.encode_query("smart office outfit")
    results = R.assemble_for_hero(
        "p1", "topwear", products_df, featurizer, clf, id_to_idx,
        fused_embeddings, query_vec, gender="women", occasion="office",
    )
    assert len(results) > 0
    combo = results[0]
    assert {"p1", "p2", "p3"}.issubset(set(combo["items"]))
    assert "feature_breakdown" in combo
    assert set(combo["feature_breakdown"].keys()) == set(C.FEATURE_NAMES)


def test_recommend_dedupes_overlapping_combos(products_df, fused_embeddings,
                                            id_to_idx, featurizer_and_clf):
    featurizer, clf = featurizer_and_clf
    query = {"gender": "women", "occasion": "office", "style_text": "office outfit", "anchor_item": None}
    results = R.recommend(query, products_df, featurizer, clf, id_to_idx, fused_embeddings, top_n=3)
    assert len(results) >= 1
    seen = []
    for combo in results:
        item_set = set(combo["items"])
        for prior in seen:
            overlap = len(item_set & prior) / len(item_set | prior)
            assert overlap <= 0.6
        seen.append(item_set)


def test_recommend_with_explicit_anchor_item(products_df, fused_embeddings,
                                            id_to_idx, featurizer_and_clf):
    featurizer, clf = featurizer_and_clf
    query = {"gender": "men", "occasion": "office", "style_text": "office look", "anchor_item": "p7"}
    results = R.recommend(query, products_df, featurizer, clf, id_to_idx, fused_embeddings, top_n=3)
    assert len(results) > 0
    assert all("p7" in combo["items"] for combo in results)
