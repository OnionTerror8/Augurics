"""Bayesian comparator evidence checks — task 2.1 review bar.

Covers what the generic contract-compliance suite doesn't: the
co-firing identifiability check actually triggering (not just the "OK"
branch), and coefficient plausibility being evaluated at observed
exposures rather than the possibly-unattainable feature=1.0 point.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.models.comparators import BayesGLMPooled
from engine.models.base import ModelConfig, PanelData


def _make_panel(n_days: int, event_flag_fn, snap_flag_fn, price_fn=None, series_key=("SKU1", "STORE1"), category="FOODS"):
    start = date(2020, 1, 1)
    rows = []
    for i in range(n_days):
        rows.append(
            {
                "series_key": series_key,
                "ds": start + timedelta(days=i),
                "y": float(1 + i % 7),
                "category": category,
                "event_flag": event_flag_fn(i),
                "snap_flag": snap_flag_fn(i),
                "price": price_fn(i) if price_fn else 5.0,
            }
        )
    return PanelData(df=pd.DataFrame(rows))


def _default_config():
    return ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=28)


def test_identifiability_check_detects_perfectly_co_firing_streams():
    """event_flag and snap_flag identical on every row -- must be flagged
    as not separately identifiable, verbatim."""
    panel = _make_panel(60, event_flag_fn=lambda i: i % 4 == 0, snap_flag_fn=lambda i: i % 4 == 0)
    model = BayesGLMPooled()
    fit_result = model.fit(panel, _default_config())
    assert "NOT separately identifiable" in fit_result.diagnostics["identifiability_note"]
    assert "NOT separately identifiable" in model._identifiability_note


def test_identifiability_check_passes_when_streams_are_not_collinear():
    panel = _make_panel(60, event_flag_fn=lambda i: i % 5 == 0, snap_flag_fn=lambda i: i % 3 == 0)
    model = BayesGLMPooled()
    fit_result = model.fit(panel, _default_config())
    assert fit_result.diagnostics["identifiability_note"] == "event_flag/snap_flag separability OK"


def test_coefficient_plausibility_report_flags_unattainable_feature_eq_1():
    """price is always 5.0 here, so log_price is a constant far from 1.0
    -- feature=1.0 must be reported as not attainable for that column."""
    panel = _make_panel(60, event_flag_fn=lambda i: i % 5 == 0, snap_flag_fn=lambda i: i % 3 == 0, price_fn=lambda i: 20.0)
    model = BayesGLMPooled()
    model.fit(panel, _default_config())

    report = model.coefficient_plausibility_report()
    log_price_row = report[report["feature"] == "log_price"].iloc[0]
    assert log_price_row["feature_eq_1_attainable"] == False  # noqa: E712
    assert log_price_row["effect_at_1"] == "not attainable"

    event_row = report[report["feature"] == "event_flag"].iloc[0]
    assert event_row["feature_eq_1_attainable"] == True  # noqa: E712
    assert isinstance(event_row["effect_at_1"], float)


def test_coefficient_plausibility_report_uses_median_and_p90_of_observed_data():
    panel = _make_panel(90, event_flag_fn=lambda i: i % 5 == 0, snap_flag_fn=lambda i: i % 3 == 0)
    model = BayesGLMPooled()
    model.fit(panel, _default_config())

    report = model.coefficient_plausibility_report()
    trend_row = report[report["feature"] == "trend"].iloc[0]

    fit_df = model._fit_df
    expected_trend = ((fit_df["ds"] - fit_df["ds"].min()).dt.days.astype(float)) / 30.0
    assert trend_row["median_exposure"] == pytest.approx(float(np.median(expected_trend)))
    assert trend_row["p90_exposure"] == pytest.approx(float(np.quantile(expected_trend, 0.9)))


def test_offset_is_computed_within_active_span_only():
    """A series with early zeros before its first sale must have its
    offset computed from the active span (post-first-sale) mean, not the
    full-window mean including the leading zeros."""
    start = date(2020, 1, 1)
    n_days = 60
    # first 20 days are zero (pre-listing), then sales begin.
    y = [0.0] * 20 + [float(3 + i % 5) for i in range(n_days - 20)]
    df = pd.DataFrame(
        {
            "series_key": [("LATE", "S1")] * n_days,
            "ds": [start + timedelta(days=i) for i in range(n_days)],
            "y": y,
            "category": "FOODS",
            "event_flag": 0,
            "snap_flag": 0,
            "price": 5.0,
        }
    )
    model = BayesGLMPooled()
    model.fit(PanelData(df=df), _default_config())

    offset = model._series_offset[("LATE", "S1")]
    active_span_mean = np.mean(y[20:])
    full_window_mean = np.mean(y)
    assert offset == pytest.approx(np.log(active_span_mean))
    assert offset != pytest.approx(np.log(full_window_mean))
