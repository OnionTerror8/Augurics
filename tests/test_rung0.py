"""Rung 0 -- "Incumbent" (EXPORT_CONTRACTS.md §2): StaticForecastModel
wraps an externally-supplied forecast series as a registry citizen.
Contract-suite compliance is covered generically by
tests/test_contract_compliance.py (parametrised over the registry).
This file is the Rung 0-specific demo: register seasonal naive's own
predictions as a fake "incumbent" forecast and show it scores on the
ladder through the SAME harness and the SAME compare_models tool as
every real model -- no special casing anywhere in the pipeline.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

import engine.models.baselines  # noqa: F401 -- side effect: registers seasonal_naive
import engine.models.rung0  # noqa: F401 -- side effect: registers static_incumbent
from engine.backtest.gates import wmape
from engine.backtest.run_baselines import load_and_validate_result, score_model
from engine.models.base import InsufficientHistoryError, ModelConfig, PanelData
from engine.models.rung0 import IncompleteForecastCoverageError, StaticForecastModel
from engine.service import tools

SERIES_KEYS = [("SKU1", "STORE1"), ("SKU2", "STORE1")]
TRAIN_DAYS = 60
HOLDOUT_DAYS = 7


def _true_pattern(series_idx: int, i: int) -> float:
    """A real, non-constant weekly-seasonal pattern -- deterministic so
    the test is reproducible, varied enough that seasonal_naive's own
    forecast (last week's value, per its rule) will genuinely diverge
    from the next window's actuals, giving a non-trivial WMAPE rather
    than a suspiciously perfect one."""
    return 5.0 + series_idx * 2.0 + (i % 7) * 0.8 + (i // 7) * 0.05


def _synthetic_panel(n_days: int, start: date) -> pd.DataFrame:
    rows = []
    for s, series_key in enumerate(SERIES_KEYS):
        for i in range(n_days):
            rows.append({"series_key": series_key, "ds": start + timedelta(days=i), "y": _true_pattern(s, i)})
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_seasonal_naive_forecast():
    """Fits the REAL seasonal_naive baseline on TRAIN_DAYS of history and
    returns its own HOLDOUT_DAYS-ahead forecast -- this becomes the
    "fake incumbent" forecast series Rung 0 wraps. Also returns the
    real holdout actuals (continuing the same generative pattern) to
    score against."""
    from engine.registry import REGISTRY

    start = date(2020, 1, 1)
    train_df = _synthetic_panel(TRAIN_DAYS, start)

    model = REGISTRY["seasonal_naive"]()
    model.fit(PanelData(df=train_df), ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=HOLDOUT_DAYS))
    incumbent_forecast_df = model._predict_all(HOLDOUT_DAYS).rename(columns={"yhat": "y"})

    holdout_start = start + timedelta(days=TRAIN_DAYS)
    holdout_rows = []
    for s, series_key in enumerate(SERIES_KEYS):
        for j in range(HOLDOUT_DAYS):
            i = TRAIN_DAYS + j
            holdout_rows.append({"series_key": series_key, "ds": holdout_start + timedelta(days=j), "y": _true_pattern(s, i)})
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df["ds"] = pd.to_datetime(holdout_df["ds"])

    return incumbent_forecast_df, holdout_df


def test_incumbent_forecast_scores_through_the_same_harness_as_every_model(fitted_seasonal_naive_forecast):
    """The core demo: seasonal_naive's own predictions, wrapped as a
    "fake incumbent" via StaticForecastModel, score a real, non-trivial
    WMAPE through engine.backtest.run_baselines.score_model -- the
    EXACT same function every Rung 1 baseline is scored through, no
    Rung-0-specific code path."""
    incumbent_forecast_df, holdout_df = fitted_seasonal_naive_forecast

    result = score_model("static_incumbent", train_df=incumbent_forecast_df, holdout_df=holdout_df, holdout_days=HOLDOUT_DAYS)

    assert result["n_series"] == len(SERIES_KEYS)
    assert result["n_rows"] == len(SERIES_KEYS) * HOLDOUT_DAYS
    assert result["wmape"] > 0  # the seasonal pattern's slow drift means last-week's value is NOT a perfect forecast
    assert result["wmape"] < 1.0  # but it should still be a reasonably close incumbent forecast, not nonsense

    # cross-check against a hand-computed WMAPE using the exact same
    # merge the harness performs, so a bug in score_model's own
    # aggregation couldn't hide behind this test's own assertions.
    lookup = holdout_df.rename(columns={"y": "y_true"})
    merged = incumbent_forecast_df.rename(columns={"y": "yhat"}).merge(lookup, on=["series_key", "ds"], how="inner")
    expected_wmape = wmape(merged["y_true"].to_numpy(), merged["yhat"].to_numpy())
    assert result["wmape"] == pytest.approx(expected_wmape)


def test_incumbent_result_appears_in_compare_models_output_alongside_a_real_model(fitted_seasonal_naive_forecast, tmp_path):
    """Rung 0 is not a special case in the tool layer either: once its
    result is written in the same validated schema, compare_models()
    (the same tool the agent layer calls) lists it side-by-side with a
    real Rung 1 model's result, with no special casing."""
    incumbent_forecast_df, holdout_df = fitted_seasonal_naive_forecast
    incumbent_result = score_model("static_incumbent", train_df=incumbent_forecast_df, holdout_df=holdout_df, holdout_days=HOLDOUT_DAYS)

    # a second, real model's result in the same ladder -- seasonal_naive
    # scored directly against its own holdout (a different, ordinary
    # Rung 1 evaluation), so the ladder has more than one rung to compare.
    real_train_df = _synthetic_panel(TRAIN_DAYS, date(2020, 1, 1))
    seasonal_naive_result = score_model("seasonal_naive", train_df=real_train_df, holdout_df=holdout_df, holdout_days=HOLDOUT_DAYS)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for fold_id, model_id, result in ((1, "static_incumbent", incumbent_result), (1, "seasonal_naive", seasonal_naive_result)):
        record = {"fold_id": fold_id, "model_id": model_id, "wall_seconds": 1.0, **result}
        pd.DataFrame([record]).to_parquet(results_dir / f"fold{fold_id}_{model_id}.parquet")
        load_and_validate_result(results_dir / f"fold{fold_id}_{model_id}.parquet")  # schema-valid, by content

    comparison = tools.compare_models(scope=None, model_version_ids="all_candidates", results_dirs=[results_dir])

    assert "static_incumbent" in comparison["model_version_ids"]
    assert "seasonal_naive" in comparison["model_version_ids"]
    assert comparison["metrics"]["static_incumbent"]["mean_wmape"] == pytest.approx(incumbent_result["wmape"])
    assert comparison["overall_winner"] in comparison["model_version_ids"]

    # JSON round-trip -- the same contract every tool output must meet
    # (task 2.6), Rung 0 included.
    json.loads(json.dumps(comparison, allow_nan=False))


def test_explain_reports_external_provenance_not_a_fitted_narrative():
    """explain() must be explicit that Rung 0 involves zero internal
    computation -- an agent narrating this to a user must never imply
    Augurics computed or influenced these numbers."""
    from engine.models.base import DateRange

    start = date(2020, 1, 1)
    panel = _synthetic_panel(30, start)
    model = StaticForecastModel()
    model.fit(PanelData(df=panel), ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))

    explanation = model.explain(SERIES_KEYS[0], DateRange(start=start, end=start + timedelta(days=29)))
    facts_text = " ".join(explanation.narrative_facts)
    assert "externally supplied" in facts_text
    assert "never computed by Augurics" in facts_text
    assert explanation.components["source"] == "external"


def test_fit_raises_on_insufficient_supplied_dates():
    start = date(2020, 1, 1)
    short_panel = _synthetic_panel(3, start)  # below MIN_HISTORY_DAYS=7
    model = StaticForecastModel()
    with pytest.raises(InsufficientHistoryError):
        model.fit(PanelData(df=short_panel), ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))


def test_predict_raises_when_horizon_exceeds_supplied_coverage():
    """A static/replay model must never extrapolate beyond what it was
    given -- fail loud instead of padding or repeating values, which
    would misrepresent the externally-supplied forecast."""
    start = date(2020, 1, 1)
    panel = _synthetic_panel(10, start)  # only 10 supplied dates per series
    model = StaticForecastModel()
    model.fit(PanelData(df=panel), ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))

    with pytest.raises(IncompleteForecastCoverageError):
        model.predict(horizon=20, future_covariates=None)


def test_predict_all_replays_stored_values_verbatim_never_transformed():
    start = date(2020, 1, 1)
    panel = _synthetic_panel(20, start)
    model = StaticForecastModel()
    model.fit(PanelData(df=panel), ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))

    preds = model._predict_all(horizon=7)
    for series_key in SERIES_KEYS:
        expected = panel[panel["series_key"] == series_key].sort_values("ds")["y"].to_numpy()[:7]
        actual = preds[preds["series_key"] == series_key].sort_values("ds")["yhat"].to_numpy()
        assert (actual == expected).all()
