"""Challenger (Rung 2, lgbm) pre-fit memory guard -- task 2.3 step 1.
Training itself is deferred (see KNOWN_ISSUES.md); this only covers the
guard that decided that, mirroring tests/test_comparators.py's coverage
of the analogous PyMC guard."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.models.ml import (
    InsufficientChallengerMemoryError,
    check_challenger_fit_memory,
    estimate_challenger_fit_memory_bytes,
)


class _FakeVirtualMemory:
    def __init__(self, available: int) -> None:
        self.available = available


def test_estimate_scales_linearly_with_row_count():
    est_1x = estimate_challenger_fit_memory_bytes(1_000_000)
    est_2x = estimate_challenger_fit_memory_bytes(2_000_000)
    assert est_2x == pytest.approx(est_1x * 2, rel=1e-9)


def test_check_passes_when_plenty_of_ram_available():
    with patch("psutil.virtual_memory", return_value=_FakeVirtualMemory(available=100_000_000_000)):
        result = check_challenger_fit_memory(n_rows=1_000_000)
    assert result["estimated_bytes"] < result["budget_bytes"]


def test_check_raises_with_numbers_when_ram_insufficient():
    with patch("psutil.virtual_memory", return_value=_FakeVirtualMemory(available=1_000_000_000)):
        with pytest.raises(InsufficientChallengerMemoryError, match="Refusing to start"):
            check_challenger_fit_memory(n_rows=48_110_016)


def test_check_reflects_real_fold1_scale_against_low_ram():
    """The actual finding that deferred challenger training: fold 1's
    real row count (48,110,016) against a modest RAM budget fails,
    consistent with this laptop's measured available RAM this session."""
    with patch("psutil.virtual_memory", return_value=_FakeVirtualMemory(available=5_000_000_000)):
        with pytest.raises(InsufficientChallengerMemoryError):
            check_challenger_fit_memory(n_rows=48_110_016)


# --- LGBMPromoModel: real fit/predict/explain, task 2.3 ---


import numpy as np
import pandas as pd

from engine.models.base import DateRange, ModelConfig, PanelData
from engine.models.ml import LGBMPromoModel, MIN_HISTORY_DAYS


def _synthetic_panel(n_series: int, n_days: int, category="FOODS") -> pd.DataFrame:
    from datetime import date, timedelta

    start = date(2020, 1, 1)
    rows = []
    for s in range(n_series):
        for i in range(n_days):
            rows.append({
                "series_key": f"SKU{s}::STORE1",
                "ds": start + timedelta(days=i),
                "y": float(2 + (i + s) % 7),
                "category": category,
                "event_flag": i % 5 == 0,
                "snap_flag": i % 3 == 0,
                "price": 5.0 + 0.1 * (i % 10),
            })
    return pd.DataFrame(rows)


def _default_config():
    return ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=14)


def test_fit_produces_real_diagnostics_with_default_hyperparams():
    df = _synthetic_panel(n_series=3, n_days=90)
    model = LGBMPromoModel()
    fit_result = model.fit(PanelData(df=df), _default_config())
    assert fit_result.family == "LGBMRegressor"
    assert fit_result.diagnostics["n_series"] == 3
    assert fit_result.diagnostics["hyperparams"] == {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    assert "event_adstock" in fit_result.diagnostics["exogenous_features"]


def test_fit_uses_explicit_hyperparams_from_config():
    df = _synthetic_panel(n_series=3, n_days=90)
    model = LGBMPromoModel()
    fit_result = model.fit(
        PanelData(df=df),
        ModelConfig(hyperparams={"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 50}, feature_set="none", vertical_profile="retail", horizon=14),
    )
    assert fit_result.diagnostics["hyperparams"] == {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 50}


def test_predict_returns_no_nan_and_correct_horizon():
    df = _synthetic_panel(n_series=2, n_days=90)
    model = LGBMPromoModel()
    model.fit(PanelData(df=df), _default_config())
    forecast = model.predict(horizon=14, future_covariates=None)
    assert len(forecast.yhat) == 14
    assert not any(np.isnan(v) for v in forecast.yhat)
    assert all(v >= 0 for v in forecast.yhat)


def test_predict_all_covers_every_fitted_series():
    df = _synthetic_panel(n_series=4, n_days=90)
    model = LGBMPromoModel()
    model.fit(PanelData(df=df), _default_config())
    preds = model._predict_all(horizon=7)
    assert set(preds["series_key"]) == {f"SKU{s}::STORE1" for s in range(4)}
    assert len(preds) == 4 * 7
    assert not preds["yhat"].isna().any()


def test_adstock_seeding_uses_trailing_history_not_a_cold_start():
    """The recursive event_adstock filter must carry the training
    period's decayed history into the future window -- proven by
    comparing predictions when the trailing history has a strong recent
    event vs none; if the seed were dropped (cold restart at 0), the
    two would predict identically once event_flag itself is absent from
    future_covariates."""
    from datetime import date, timedelta

    start = date(2020, 1, 1)
    n_days = 90

    def _make(event_on_last_day: bool):
        rows = []
        for i in range(n_days):
            rows.append({
                "series_key": "SKU0::STORE1", "ds": start + timedelta(days=i),
                "y": float(2 + i % 7), "category": "FOODS",
                "event_flag": event_on_last_day and i == n_days - 1,
                "snap_flag": False, "price": 5.0,
            })
        return pd.DataFrame(rows)

    model_with_event = LGBMPromoModel()
    model_with_event.fit(PanelData(df=_make(True)), _default_config())
    model_without_event = LGBMPromoModel()
    model_without_event.fit(PanelData(df=_make(False)), _default_config())

    uid = "SKU0::STORE1"
    fut_with = model_with_event._future_exogenous_frame(uid, horizon=5, future_covariates=None)
    fut_without = model_without_event._future_exogenous_frame(uid, horizon=5, future_covariates=None)
    assert not np.allclose(fut_with["event_adstock"].to_numpy(), fut_without["event_adstock"].to_numpy())
    assert fut_with["event_adstock"].iloc[0] > fut_without["event_adstock"].iloc[0]


def test_explain_returns_real_feature_importances():
    df = _synthetic_panel(n_series=2, n_days=90)
    model = LGBMPromoModel()
    model.fit(PanelData(df=df), _default_config())
    explanation = model.explain("SKU0::STORE1", DateRange(start=df["ds"].min(), end=df["ds"].max()))
    assert "feature_importances" in explanation.components
    assert len(explanation.drivers) <= 5
    assert all(d["magnitude"] >= 0 for d in explanation.drivers)


def test_explain_raises_for_series_not_in_fitted_panel():
    df = _synthetic_panel(n_series=2, n_days=90)
    model = LGBMPromoModel()
    model.fit(PanelData(df=df), _default_config())
    with pytest.raises(ValueError, match="was not part of the fitted panel"):
        model.explain("NOTREAL::STORE1", DateRange(start=df["ds"].min(), end=df["ds"].max()))


def test_save_and_load_round_trip_predicts_identically():
    df = _synthetic_panel(n_series=2, n_days=90)
    model = LGBMPromoModel()
    model.fit(PanelData(df=df), _default_config())
    forecast_before = model.predict(horizon=7, future_covariates=None)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        model.save(str(path))
        loaded = LGBMPromoModel.load(str(path))
    forecast_after = loaded.predict(horizon=7, future_covariates=None)
    assert forecast_before.yhat == forecast_after.yhat


def test_fit_raises_insufficient_history_below_minimum():
    from engine.models.base import InsufficientHistoryError

    df = _synthetic_panel(n_series=2, n_days=MIN_HISTORY_DAYS - 1)
    model = LGBMPromoModel()
    with pytest.raises(InsufficientHistoryError, match="min_history_days"):
        model.fit(PanelData(df=df), _default_config())
