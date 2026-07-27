"""Shared series_key convention and fail-loud selection helper.

This exists because a key-format mismatch (comparing a tuple against a
bulk-path "ITEM::STORE" uid string) previously produced a silently empty
DataFrame instead of an error -- select_series() must always raise
rather than let that kind of bug hide downstream.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.models.base import SERIES_KEY_SEP, encode_series_key, select_series


def test_encode_series_key_uses_pinned_separator():
    assert encode_series_key("FOODS_3_033", "WI_1") == f"FOODS_3_033{SERIES_KEY_SEP}WI_1"
    assert encode_series_key("FOODS_3_033", "WI_1") == "FOODS_3_033::WI_1"


def test_select_series_returns_matching_rows():
    df = pd.DataFrame({"series_key": ["A::S1", "A::S1", "B::S1"], "y": [1.0, 2.0, 3.0]})
    subset = select_series(df, "A::S1")
    assert len(subset) == 2
    assert subset["y"].tolist() == [1.0, 2.0]


def test_select_series_raises_on_no_match_instead_of_returning_empty():
    df = pd.DataFrame({"series_key": ["A::S1"], "y": [1.0]})
    with pytest.raises(ValueError, match="no rows"):
        select_series(df, "B::S1")


def test_select_series_raises_on_key_format_mismatch():
    """The exact bug this guard exists for: a tuple key against a string column."""
    df = pd.DataFrame({"series_key": ["A::S1"], "y": [1.0]})
    with pytest.raises(ValueError, match="no rows"):
        select_series(df, ("A", "S1"))


def test_select_series_enforces_expected_count():
    df = pd.DataFrame({"series_key": ["A::S1", "A::S1", "A::S1"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="expected 2 rows"):
        select_series(df, "A::S1", expected_count=2)
    assert len(select_series(df, "A::S1", expected_count=3)) == 3
