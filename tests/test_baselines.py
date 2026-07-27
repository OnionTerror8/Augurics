"""Rung 1 baseline-specific tests — task 1.6.

The generic contract suite (test_contract_compliance.py) covers the
single-series interface. These tests cover the multi-series fit path and
the self-contained backtest() method that the generic suite doesn't
exercise.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from engine.backtest.protocol import HOLDOUT_DAYS
from engine.models.baselines import NaiveModel, SeasonalNaiveModel
from engine.models.base import BacktestProtocol, ModelConfig, PanelData


def _multi_series_panel(n_days: int, n_series: int = 3) -> PanelData:
    start = date(2020, 1, 1)
    rows = []
    for s in range(n_series):
        for i in range(n_days):
            rows.append(
                {
                    "series_key": (f"SKU{s}", "STORE1"),
                    "ds": start + timedelta(days=i),
                    "y": float(1 + (i + s) % 7),
                }
            )
    return PanelData(df=pd.DataFrame(rows))


def test_fit_handles_multiple_series_at_once():
    model = NaiveModel()
    panel = _multi_series_panel(60)
    fit_result = model.fit(panel, ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))
    assert fit_result.diagnostics["n_series"] == 3
    assert fit_result.family == "Naive"


def test_predict_all_returns_every_fitted_series():
    model = NaiveModel()
    panel = _multi_series_panel(60)
    model.fit(panel, ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))
    all_preds = model._predict_all(7)
    assert set(all_preds["series_key"]) == {("SKU0", "STORE1"), ("SKU1", "STORE1"), ("SKU2", "STORE1")}
    assert len(all_preds) == 3 * 7


def test_explain_rule_narrative_mentions_seasonal_period():
    model = SeasonalNaiveModel()
    panel = _multi_series_panel(60, n_series=1)
    model.fit(panel, ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7))
    from engine.models.base import DateRange

    explanation = model.explain(("SKU0", "STORE1"), DateRange(start=date(2020, 1, 1), end=date(2020, 2, 1)))
    assert "7 days prior" in explanation.narrative_facts[0]


def test_backtest_runs_a_genuine_holdout_and_returns_wmape():
    model = NaiveModel()
    panel = _multi_series_panel(model.min_history_days + HOLDOUT_DAYS + 20)
    protocol = BacktestProtocol(protocol_id="test-protocol", holdout_days=HOLDOUT_DAYS)

    result = model.backtest(panel, protocol)

    assert "wmape" in result.metrics
    assert result.metrics["wmape"] >= 0
    assert result.per_fold[0]["n_rows"] == 3 * HOLDOUT_DAYS  # 3 series x holdout days


def test_backtest_raises_when_not_enough_history_for_holdout():
    model = NaiveModel()
    panel = _multi_series_panel(10)  # far too short for a 28-day holdout
    protocol = BacktestProtocol(protocol_id="test-protocol", holdout_days=HOLDOUT_DAYS)
    with pytest.raises(ValueError, match="not enough history"):
        model.backtest(panel, protocol)
