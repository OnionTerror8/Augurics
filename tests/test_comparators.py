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

from engine.models.comparators import (
    ADVIConvergenceError,
    BayesGLMPooled,
    _check_advi_convergence,
)
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


# --- ADVI convergence guard ---


def test_advi_convergence_passes_on_stable_loss():
    stable = np.concatenate([np.linspace(1000, 100, 50), np.full(50, 100.0) + np.random.default_rng(0).normal(0, 0.01, 50)])
    rel_change = _check_advi_convergence(stable)
    assert rel_change < 0.05


def test_advi_convergence_raises_on_nan_in_history():
    bad = np.array([100.0, 90.0, np.nan, 80.0])
    with pytest.raises(ADVIConvergenceError, match="NaN/Inf"):
        _check_advi_convergence(bad)


def test_advi_convergence_raises_when_tail_still_decreasing():
    still_decreasing = np.linspace(1000, 1, 100)  # never plateaus
    with pytest.raises(ADVIConvergenceError, match="not stabilised"):
        _check_advi_convergence(still_decreasing)


# --- NUTS spot-check row cap ---


def _make_multi_series_category_panel(n_series: int, n_days: int, category="FOODS"):
    start = date(2020, 1, 1)
    rows = []
    for s in range(n_series):
        for i in range(n_days):
            rows.append(
                {
                    "series_key": (f"SKU{s}", "STORE1"),
                    "ds": start + timedelta(days=i),
                    "y": float(1 + (i + s) % 7),
                    "category": category,
                    "event_flag": i % 5 == 0,
                    "snap_flag": i % 3 == 0,
                    "price": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_nuts_spot_check_caps_to_max_rows_via_whole_series_subsample():
    df = _make_multi_series_category_panel(n_series=10, n_days=60)  # 600 rows total
    model = BayesGLMPooled()
    fit_result = model.fit(PanelData(df=df), _default_config())

    spot_check = model._run_nuts_spot_check(model._fit_df, "FOODS", n_features=len(model._feature_names), max_rows=200)
    assert spot_check["ran"] is True
    assert spot_check["capped"] is True
    assert spot_check["original_category_rows"] == 600
    assert spot_check["spot_check_rows"] <= 200 + 60  # at most a bit over cap (whole-series granularity)


def test_nuts_spot_check_does_not_cap_when_under_max_rows():
    df = _make_multi_series_category_panel(n_series=3, n_days=60)  # 180 rows
    model = BayesGLMPooled()
    model.fit(PanelData(df=df), _default_config())

    spot_check = model._run_nuts_spot_check(model._fit_df, "FOODS", n_features=len(model._feature_names), max_rows=1_000_000)
    assert spot_check["ran"] is True
    assert spot_check["capped"] is False
    assert spot_check["spot_check_rows"] == spot_check["original_category_rows"] == 180


def test_fit_can_skip_nuts_spot_check_for_per_fold_runs():
    df = _make_multi_series_category_panel(n_series=3, n_days=60)
    model = BayesGLMPooled()
    fit_result = model.fit(PanelData(df=df), _default_config(), run_nuts_spot_check=False)
    assert fit_result.diagnostics["nuts_spot_check"] == {
        "ran": False,
        "reason": "run_nuts_spot_check=False for this fit",
    }


def test_fit_result_reports_advi_convergence_rel_change():
    df = _make_multi_series_category_panel(n_series=3, n_days=60)
    model = BayesGLMPooled()
    fit_result = model.fit(PanelData(df=df), _default_config(), run_nuts_spot_check=False)
    assert "advi_convergence_rel_change" in fit_result.diagnostics
    assert fit_result.diagnostics["advi_convergence_rel_change"] >= 0


# --- bulk multi-series prediction (used by the per-fold backtest orchestration) ---


def test_predict_all_returns_every_fitted_series_with_no_nan():
    df = _make_multi_series_category_panel(n_series=4, n_days=60)
    model = BayesGLMPooled()
    model.fit(PanelData(df=df), _default_config(), run_nuts_spot_check=False)

    preds = model._predict_all(horizon=7)
    assert set(preds["series_key"]) == {(f"SKU{s}", "STORE1") for s in range(4)}
    assert len(preds) == 4 * 7
    assert not preds["yhat"].isna().any()
    assert (preds["yhat"] >= 0).all()


def test_predict_all_applies_date_indexed_covariates_to_every_series():
    df = _make_multi_series_category_panel(n_series=2, n_days=60)
    model = BayesGLMPooled()
    model.fit(PanelData(df=df), _default_config(), run_nuts_spot_check=False)

    last_ds = model._fit_df["ds"].max()
    future_dates = pd.date_range(last_ds + pd.Timedelta(days=1), periods=5, freq="D")
    shared_covariates = pd.DataFrame({"ds": future_dates, "event_flag": [1.0, 0.0, 0.0, 1.0, 0.0], "snap_flag": 1.0})

    preds_with_events = model._predict_all(horizon=5, future_covariates=shared_covariates)
    preds_without_events = model._predict_all(horizon=5)
    # broadcasting the same covariates to every series should change both series identically
    for series_key in preds_with_events["series_key"].unique():
        with_ev = preds_with_events[preds_with_events["series_key"] == series_key]["yhat"].to_numpy()
        without_ev = preds_without_events[preds_without_events["series_key"] == series_key]["yhat"].to_numpy()
        assert not (with_ev == without_ev).all()  # covariates actually changed the forecast


def test_predict_all_applies_per_series_price_covariates():
    df = _make_multi_series_category_panel(n_series=2, n_days=60)
    model = BayesGLMPooled()
    model.fit(PanelData(df=df), _default_config(), run_nuts_spot_check=False)

    last_ds = model._fit_df["ds"].max()
    future_dates = pd.date_range(last_ds + pd.Timedelta(days=1), periods=5, freq="D")
    per_series_price = pd.concat(
        [
            pd.DataFrame({"series_key": [("SKU0", "STORE1")] * 5, "ds": future_dates, "price": 5.0}),
            pd.DataFrame({"series_key": [("SKU1", "STORE1")] * 5, "ds": future_dates, "price": 50.0}),
        ]
    )
    preds = model._predict_all(horizon=5, future_covariates=per_series_price)
    sku0 = preds[preds["series_key"] == ("SKU0", "STORE1")]["yhat"].to_numpy()
    sku1 = preds[preds["series_key"] == ("SKU1", "STORE1")]["yhat"].to_numpy()
    assert not (sku0 == sku1).all()  # different per-series price -> different forecast
