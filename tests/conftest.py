"""
tests/conftest.py
==================
Shared fixtures: a small SYNTHETIC catalog (10 items, 2 outfits), not the
real 68-item dataset. Keeps CI fast, deterministic, and free of any
network/model dependency -- these are unit tests of the pipeline LOGIC
(slot mapping, graph building, featurization, retrieval filtering), not
of embedding quality, which needs the real dataset and real models to
mean anything (see compatibility.py / embeddings.py docstrings).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_pipeline as DP


@pytest.fixture
def products_df():
    rows = [
        # id, name, category, gender, wear_type, occasion, description
        ("p1", "Blue Formal Shirt", "formal-shirts", "women", "western", "office", "Crisp blue formal shirt"),
        ("p2", "Black Trousers", "trousers", "women", "western", "office", "Tailored black trousers"),
        ("p3", "Black Pumps", "heels", "women", "western", "office", "Black heeled pumps"),
        ("p4", "Red Party Dress", "party-dresses", "women", "western", "party", "Red sequinned party dress"),
        ("p5", "Gold Sandals", "sandals", "women", "western", "party", "Gold strappy sandals"),
        ("p6", "Red Clutch", "clutches", "women", "western", "party", "Small red clutch bag"),
        ("p7", "White Tshirt", "tshirts", "men", "western", "office", "Plain white cotton tshirt"),
        ("p8", "Navy Chinos", "chinos", "men", "western", "office", "Navy slim chinos"),
        ("p9", "Brown Loafers", "loafers", "men", "western", "office", "Brown leather loafers"),
        ("p10", "Denim Jacket", "denim-jackets", "men", "western", "casual", "Blue denim jacket"),
    ]
    df = pd.DataFrame(rows, columns=["id", "name", "category", "gender", "wear_type", "occasion", "description"])
    df["brand"] = "TestBrand"
    df["price_inr"] = 999
    df["rating"] = np.nan
    df["rating_count"] = np.nan
    df["category_label"] = df["category"].str.replace("-", " ").str.title()
    df["tags"] = "test;fixture"
    df["image"] = "images/test/" + df["id"] + ".jpg"
    df = DP.add_slot_column(df)
    return df


@pytest.fixture
def parsed_outfits():
    return [
        {
            "outfit_id": "outfit_1", "gender": "women", "occasion": "office",
            "wear_type": "western", "theme": "Office basics", "palette": "blue / black",
            "stylist_rationale": "Blue shirt with black trousers and black pumps for the office.",
            "items": [
                {"id": "p1", "role": "hero", "slot": "topwear"},
                {"id": "p2", "role": "second", "slot": "bottomwear"},
                {"id": "p3", "role": "footwear", "slot": "footwear"},
            ],
        },
        {
            "outfit_id": "outfit_2", "gender": "women", "occasion": "party",
            "wear_type": "western", "theme": "Red party look", "palette": "red / gold",
            "stylist_rationale": "Red dress with gold sandals and a red clutch for a party.",
            "items": [
                {"id": "p4", "role": "hero", "slot": "onepiece"},
                {"id": "p5", "role": "footwear", "slot": "footwear"},
                {"id": "p6", "role": "accessory_1", "slot": "accessory"},
            ],
        },
    ]


@pytest.fixture
def cooc_graph(parsed_outfits):
    return DP.build_cooccurrence_graph(parsed_outfits)


@pytest.fixture
def fused_embeddings(products_df):
    """Deterministic synthetic embeddings -- shape/index correctness is
    what's under test here, not semantic quality (that needs real
    FashionCLIP weights, see embeddings.py)."""
    rng = np.random.default_rng(0)
    n, d = len(products_df), 32
    vecs = rng.standard_normal((n, d)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def id_to_idx(products_df):
    return {pid: i for i, pid in enumerate(products_df["id"])}
