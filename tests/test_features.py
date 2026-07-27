"""Feature library tests — task 2.2.

Every feature has a dedicated no-leakage test: mutate ONLY rows after a
cutoff, recompute, and assert every row up to and including the cutoff
is byte-for-byte unchanged. If a feature's past values shift just
because the future changed, that feature is leaking.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.features.library import (
    add_active_span_flag,
    add_calendar_features,
    add_event_adstock,
    add_event_snap_flags,
    add_lag_features,
    add_price_features,
    add_rolling_features,
    build_feature_set,
)
from engine.features.profiles import PROFILES, get_profile


def _make_df(n_days: int = 60, n_series: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 1)
    rows = []
    for s in range(n_series):
        for i in range(n_days):
            rows.append(
                {
                    "series_key": f"S{s}",
                    "ds": start + timedelta(days=i),
                    "y": float(rng.integers(0, 10)),
                    "sell_price": 5.0 + 0.1 * i,
                    "event_name_1": "Holiday" if i % 10 == 0 else None,
                    "event_name_2": None,
                    "snap": int(i % 3 == 0),
                }
            )
    return pd.DataFrame(rows)


def _assert_no_leakage(feature_fn, df: pd.DataFrame, cutoff_idx: int, feature_cols: list[str], **kwargs):
    """Mutate only rows strictly after cutoff_idx (per series) and assert
    the feature columns for rows up to and including cutoff_idx never move."""
    before = feature_fn(df, **kwargs).sort_values(["series_key", "ds"]).reset_index(drop=True)

    mutated = df.copy()
    for series_key, group in mutated.groupby("series_key"):
        sorted_idx = group.sort_values("ds").index
        future_idx = sorted_idx[cutoff_idx + 1 :]
        mutated.loc[future_idx, "y"] = mutated.loc[future_idx, "y"] + 1000.0
        if "sell_price" in mutated.columns:
            mutated.loc[future_idx, "sell_price"] = mutated.loc[future_idx, "sell_price"] * 5.0
        if "event_name_1" in mutated.columns:
            mutated.loc[future_idx, "event_name_1"] = "MutatedFutureEvent"
        if "snap" in mutated.columns:
            mutated.loc[future_idx, "snap"] = 1 - mutated.loc[future_idx, "snap"]

    after = feature_fn(mutated, **kwargs).sort_values(["series_key", "ds"]).reset_index(drop=True)

    for series_key in df["series_key"].unique():
        b = before[before["series_key"] == series_key].reset_index(drop=True)
        a = after[after["series_key"] == series_key].reset_index(drop=True)
        for col in feature_cols:
            pd.testing.assert_series_equal(
                b[col].iloc[: cutoff_idx + 1],
                a[col].iloc[: cutoff_idx + 1],
                check_names=False,
                obj=f"{col} (series {series_key}) up to cutoff",
            )


def test_calendar_features_are_pure_functions_of_own_date():
    df = _make_df()
    _assert_no_leakage(add_calendar_features, df, cutoff_idx=30, feature_cols=["dow", "month", "week_of_year", "is_weekend"])


def test_event_snap_flags_no_leakage():
    df = _make_df()
    _assert_no_leakage(add_event_snap_flags, df, cutoff_idx=30, feature_cols=["event_flag", "snap_flag"])


def test_price_features_no_leakage():
    df = _make_df()
    _assert_no_leakage(add_price_features, df, cutoff_idx=30, feature_cols=["log_price", "price_change"])


def test_lag_features_no_leakage():
    df = _make_df()
    _assert_no_leakage(add_lag_features, df, cutoff_idx=30, feature_cols=["y_lag_7", "y_lag_28"])


def test_rolling_features_no_leakage():
    df = _make_df()
    _assert_no_leakage(
        add_rolling_features, df, cutoff_idx=30,
        feature_cols=["y_rolling_mean_7", "y_rolling_std_7", "y_rolling_mean_28", "y_rolling_std_28"],
    )


def test_rolling_feature_excludes_current_row_own_y():
    """The window ending at row t must be [t-window, t-1] -- verify directly
    against a hand-computed value, not just the mutation-based leakage test."""
    df = pd.DataFrame({"series_key": ["S0"] * 5, "ds": pd.date_range("2020-01-01", periods=5), "y": [1.0, 2.0, 3.0, 4.0, 100.0]})
    out = add_rolling_features(df, windows=(3,))
    # row index 4 (y=100) must not appear in its own rolling mean.
    expected_mean_at_4 = np.mean([2.0, 3.0, 4.0])  # rows 1,2,3 (shifted window of size 3 ending at t-1)
    assert out["y_rolling_mean_3"].iloc[4] == pytest.approx(expected_mean_at_4)


def test_event_adstock_no_leakage():
    df = _make_df()
    df_with_flags = add_event_snap_flags(df)
    _assert_no_leakage(add_event_adstock, df_with_flags, cutoff_idx=30, feature_cols=["event_adstock"])


def test_event_adstock_rejects_invalid_decay():
    df = add_event_snap_flags(_make_df())
    with pytest.raises(ValueError, match="decay"):
        add_event_adstock(df, decay=1.0)


def test_event_adstock_matches_hand_computed_recursion():
    df = pd.DataFrame({"series_key": ["S0"] * 4, "ds": pd.date_range("2020-01-01", periods=4), "event_flag": [1, 0, 1, 0]})
    out = add_event_adstock(df, event_col="event_flag", decay=0.5)
    # adstock: 1, 0+0.5*1=0.5, 1+0.5*0.5=1.25, 0+0.5*1.25=0.625
    assert out["event_adstock"].tolist() == pytest.approx([1.0, 0.5, 1.25, 0.625])


def test_active_span_flag_no_leakage():
    df = _make_df()
    _assert_no_leakage(add_active_span_flag, df, cutoff_idx=30, feature_cols=["active_span_flag"])


def test_active_span_flag_marks_only_from_first_nonzero_sale():
    df = pd.DataFrame({"series_key": ["S0"] * 5, "ds": pd.date_range("2020-01-01", periods=5), "y": [0.0, 0.0, 3.0, 0.0, 5.0]})
    out = add_active_span_flag(df)
    assert out["active_span_flag"].tolist() == [0, 0, 1, 1, 1]


def test_build_feature_set_runs_end_to_end():
    df = _make_df()
    out = build_feature_set(df)
    for expected_col in ("dow", "event_flag", "log_price", "y_lag_7", "y_rolling_mean_7", "event_adstock", "active_span_flag"):
        assert expected_col in out.columns


def test_feature_profiles_are_declared_and_retrievable():
    assert "retail_promo_v1" in PROFILES
    profile = get_profile("retail_promo_v1")
    assert profile.lags == (7, 28)


def test_get_profile_raises_on_unknown_name():
    with pytest.raises(KeyError):
        get_profile("not_a_real_profile")
