"""Comparator per-fold scoring — tested on synthetic scratch data, never
against real fold data (that launch waits on review)."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from engine.backtest.run_comparator import score_comparator


def _make_frames(n_series: int, train_days: int, holdout_days: int, category="FOODS"):
    start = date(2020, 1, 1)
    train_rows, holdout_rows = [], []
    for s in range(n_series):
        for i in range(train_days):
            train_rows.append(
                {
                    "series_key": f"SKU{s}::STORE1",
                    "ds": start + timedelta(days=i),
                    "y": float(1 + (i + s) % 7),
                    "category": category,
                    "event_flag": i % 5 == 0,
                    "snap_flag": i % 3 == 0,
                    "price": 5.0 + s,
                }
            )
        for j in range(holdout_days):
            i = train_days + j
            holdout_rows.append(
                {
                    "series_key": f"SKU{s}::STORE1",
                    "ds": start + timedelta(days=i),
                    "y": float(1 + (i + s) % 7),
                    "category": category,
                    "event_flag": i % 5 == 0,
                    "snap_flag": i % 3 == 0,
                    "price": 5.0 + s,
                }
            )
    return pd.DataFrame(train_rows), pd.DataFrame(holdout_rows)


def test_score_comparator_returns_valid_wmape_and_diagnostics():
    train_df, holdout_df = _make_frames(n_series=4, train_days=60, holdout_days=7)
    result = score_comparator(train_df, holdout_df, holdout_days=7, run_nuts_spot_check=False)

    assert result["wmape"] >= 0
    assert result["n_series"] == 4
    assert result["n_rows"] == 4 * 7
    assert result["family"] == "NegBinomialRV"
    assert result["nuts_spot_check"]["ran"] is False
    assert "advi_convergence_rel_change" in result


def test_score_comparator_result_is_parquet_round_trippable():
    """The exact concern from the real evidence run: nested dict fields
    (nuts_spot_check) must survive a parquet write/read, since that's how
    run_comparator_one_fold.py persists results."""
    train_df, holdout_df = _make_frames(n_series=3, train_days=60, holdout_days=7)
    result = score_comparator(train_df, holdout_df, holdout_days=7, run_nuts_spot_check=False)
    record = {"fold_id": 1, "model_id": "bayes_glm_pooled", "wall_seconds": 1.0, **result}

    df = pd.DataFrame([record])
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.parquet"
        df.to_parquet(path)
        back = pd.read_parquet(path)

    assert back.iloc[0]["wmape"] == pytest.approx(result["wmape"])
    assert back.iloc[0]["nuts_spot_check"] == result["nuts_spot_check"]


def test_score_comparator_with_nuts_spot_check_enabled():
    train_df, holdout_df = _make_frames(n_series=4, train_days=60, holdout_days=7)
    result = score_comparator(train_df, holdout_df, holdout_days=7, run_nuts_spot_check=True)
    assert result["nuts_spot_check"]["ran"] is True
    assert "mean_abs_coef_diff_advi_vs_nuts" in result["nuts_spot_check"]
    assert result["predictive_agreement"] is not None
    assert "advi_wmape" in result["predictive_agreement"]
    assert "nuts_wmape" in result["predictive_agreement"]
    assert "relative_difference" in result["predictive_agreement"]


def test_predictive_agreement_check_licenses_when_within_tolerance():
    from engine.backtest.run_comparator import run_predictive_agreement_check
    from engine.models.comparators import BayesGLMPooled
    from engine.models.base import PanelData, ModelConfig

    train_df, holdout_df = _make_frames(n_series=4, train_days=60, holdout_days=7)
    model = BayesGLMPooled()
    fit_result = model.fit(
        PanelData(df=train_df),
        ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7),
        run_nuts_spot_check=True,
    )
    spot_check = fit_result.diagnostics["nuts_spot_check"]
    assert spot_check["ran"] is True

    result = run_predictive_agreement_check(model, holdout_df, spot_check)
    assert result["category"] == spot_check["category"]
    assert result["n_series"] == len(spot_check["spot_check_series_keys"])
    assert isinstance(result["licensed"], bool)


def test_predictive_agreement_check_detects_forced_divergence():
    """Force the NUTS beta to something wildly different from ADVI's and
    confirm the check itself reports non-licensed with a large relative
    difference -- the deterministic case for the decisive stop condition."""
    from engine.backtest.run_comparator import run_predictive_agreement_check
    from engine.models.comparators import BayesGLMPooled
    from engine.models.base import PanelData, ModelConfig

    train_df, holdout_df = _make_frames(n_series=4, train_days=60, holdout_days=7)
    model = BayesGLMPooled()
    fit_result = model.fit(
        PanelData(df=train_df),
        ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=7),
        run_nuts_spot_check=True,
    )
    spot_check = dict(fit_result.diagnostics["nuts_spot_check"])
    spot_check["nuts_beta"] = [b + 5.0 for b in spot_check["nuts_beta"]]  # force wildly different predictions

    result = run_predictive_agreement_check(model, holdout_df, spot_check)
    assert result["licensed"] is False
    assert result["relative_difference"] > 0.02


def test_score_comparator_raises_predictive_agreement_error_when_not_licensed():
    """score_comparator itself must stop before returning any result when
    the predictive-agreement check fails -- mocked so this is deterministic
    rather than depending on a real fit happening to diverge."""
    from unittest.mock import patch

    from engine.backtest.run_comparator import PredictiveAgreementError

    train_df, holdout_df = _make_frames(n_series=4, train_days=60, holdout_days=7)
    fake_not_licensed = {
        "advi_wmape": 0.5,
        "nuts_wmape": 0.7,
        "relative_difference": 0.2857,
        "licensed": False,
        "category": "FOODS",
        "n_series": 4,
        "n_holdout_rows": 28,
    }
    with patch("engine.backtest.run_comparator.run_predictive_agreement_check", return_value=fake_not_licensed):
        with pytest.raises(PredictiveAgreementError, match="NOT licensed"):
            score_comparator(train_df, holdout_df, holdout_days=7, run_nuts_spot_check=True)
