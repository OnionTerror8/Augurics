"""The keystone suite: every registered model must pass this to be usable.

Parametrised over engine.registry.REGISTRY, so it grows automatically as
models register (task 1.6 onward). With no models registered yet, each
test collects zero parametrised cases and reports as passed/deselected —
this is the "trivial pass" expected before any model exists.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import engine.models.baselines  # noqa: F401  -- side effect: registers Rung 1 baselines
import engine.models.comparators  # noqa: F401  -- side effect: registers bayes_glm_pooled
from engine.models.base import (
    DateRange,
    ForecastModel,
    InsufficientHistoryError,
    ModelConfig,
    PanelData,
)
from engine.registry import REGISTRY

REGISTERED_MODEL_CLASSES = list(REGISTRY.values())


def _make_panel(n_days: int, series_key: tuple[str, str] = ("SKU1", "STORE1")) -> PanelData:
    start = date(2020, 1, 1)
    rows = [
        {
            "series_key": series_key,
            "ds": start + timedelta(days=i),
            "y": float(1 + i % 7),
            "static_features": {},
            "known_future": {},
            "observed_past": {},
        }
        for i in range(n_days)
    ]
    return PanelData(df=pd.DataFrame(rows))


def _default_config(horizon: int = 28) -> ModelConfig:
    return ModelConfig(
        hyperparams={},
        feature_set="none",
        vertical_profile="retail",
        horizon=horizon,
        quantiles=[],
    )


@pytest.mark.parametrize("model_cls", REGISTERED_MODEL_CLASSES, ids=lambda m: m.model_id)
class TestContractCompliance:
    """Every registered model, all rungs, one contract (registry spec §1)."""

    def test_is_forecast_model_subclass(self, model_cls):
        assert issubclass(model_cls, ForecastModel)

    def test_declares_identity_attributes(self, model_cls):
        for attr in (
            "model_id",
            "display_name",
            "rung",
            "tier",
            "supports_quantiles",
            "supports_covariates",
            "min_history_days",
            "max_horizon_days",
        ):
            assert hasattr(model_cls, attr), f"{model_cls.__name__} missing '{attr}'"

    def test_fit_raises_on_insufficient_history(self, model_cls):
        model = model_cls()
        short_panel = _make_panel(max(model_cls.min_history_days - 1, 0))
        with pytest.raises(InsufficientHistoryError):
            model.fit(short_panel, _default_config())

    def test_predict_output_has_no_nan_and_is_non_negative(self, model_cls):
        model = model_cls()
        panel = _make_panel(model_cls.min_history_days + 30)
        model.fit(panel, _default_config())
        result = model.predict(horizon=7, future_covariates=None)
        assert not any(math.isnan(v) for v in result.yhat), "predict() must never return NaN"
        assert all(v >= 0 for v in result.yhat), "demand forecasts must be non-negative"

    def test_explain_is_never_empty(self, model_cls):
        model = model_cls()
        panel = _make_panel(model_cls.min_history_days + 30)
        model.fit(panel, _default_config())
        explanation = model.explain(
            series_key=("SKU1", "STORE1"),
            date_range=DateRange(start=date(2020, 1, 1), end=date(2020, 2, 1)),
        )
        assert explanation.components, "explain() must return a non-empty decomposition"
        assert explanation.narrative_facts, "explain() must emit at least one narrative fact"

    def test_predict_output_serialises_cleanly_to_json(self, model_cls):
        model = model_cls()
        panel = _make_panel(model_cls.min_history_days + 30)
        model.fit(panel, _default_config())
        result = model.predict(horizon=7, future_covariates=None)

        payload = json.dumps(
            {
                "series_key": list(result.series_key),
                "ds": result.ds,
                "yhat": result.yhat,
                "yhat_quantiles": result.yhat_quantiles,
                "model_id": result.model_id,
                "model_version": result.model_version,
                "generated_at": result.generated_at,
                "job_id": result.job_id,
            }
        )
        round_tripped = json.loads(payload)

        for ds_value in round_tripped["ds"]:
            datetime.fromisoformat(ds_value)  # raises if not a valid ISO string
        datetime.fromisoformat(round_tripped["generated_at"])

    def test_save_load_round_trip_predicts_identically(self, model_cls, tmp_path):
        model = model_cls()
        panel = _make_panel(model_cls.min_history_days + 30)
        model.fit(panel, _default_config())
        before = model.predict(horizon=7, future_covariates=None)

        artefact_uri = str(tmp_path / f"{model_cls.model_id}.pkl")
        model.save(artefact_uri)
        loaded = model_cls.load(artefact_uri)
        after = loaded.predict(horizon=7, future_covariates=None)

        assert after.series_key == before.series_key
        assert after.yhat == before.yhat
        assert after.ds == before.ds
