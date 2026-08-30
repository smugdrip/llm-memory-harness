"""HashEmbedder: deterministic, normalized, and cosine tracks token overlap."""

from __future__ import annotations

import math

from llm.embedder import HashEmbedder
from memory.store import _cosine


def test_deterministic():
    e = HashEmbedder()
    assert e.embed(["hello world"]) == e.embed(["hello world"])


def test_dim_and_model_id():
    e = HashEmbedder(dim=64)
    v = e.embed(["x"])[0]
    assert len(v) == 64
    assert e.dim == 64
    assert e.model_id == "fake/hash-bow-64"


def test_normalized():
    v = HashEmbedder().embed(["some words to embed here"])[0]
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_overlap_scores_higher():
    e = HashEmbedder()
    a, b, c = e.embed(
        [
            "the red fox jumped over the fence",
            "red fox jumped high",
            "quarterly sqlite economics report",
        ]
    )
    assert _cosine(a, b) > _cosine(a, c)
    assert _cosine(a, a) > 0.999


def test_empty_text_is_zero_vector():
    v = HashEmbedder().embed([""])[0]
    assert all(x == 0.0 for x in v)
