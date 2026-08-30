"""Preprocessing: golden files for canonical_text, and the property the design states —
names, dates, and project terms survive."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from memory.preprocess import PREPROCESS_VERSION, preprocess

GOLDEN = Path(__file__).parent / "golden" / "canonical_text.json"


def test_version_is_declared():
    assert PREPROCESS_VERSION == 1


def test_golden_canonical_text():
    for case in json.loads(GOLDEN.read_text()):
        assert preprocess(case["raw"]) == case["canonical"]


def test_collapses_whitespace_only():
    assert preprocess("  a \t b \n\n c ") == "a b c"


MARKERS = ["Meridian", "Priya", "2026-09-05", "X-T5", "libvips", "Falkenstein"]


@given(
    markers=st.lists(st.sampled_from(MARKERS), min_size=1, max_size=4, unique=True),
    fillers=st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8), max_size=10),
    data=st.randoms(use_true_random=False),
)
def test_names_dates_and_terms_survive(markers, fillers, data):
    words = markers + fillers
    data.shuffle(words)
    text = " \t ".join(words)
    out = preprocess(text)
    for marker in markers:
        assert marker in out


@given(st.text(max_size=300))
def test_idempotent(text):
    once = preprocess(text)
    assert preprocess(once) == once
