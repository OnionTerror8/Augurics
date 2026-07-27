"""BASELINE_TABLE.md arithmetic — task 1.6 exit artefact.

Tested against synthetic scratch data so the volume-weighting formula
and gate-threshold restatement are verified independent of any real run.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from engine.backtest.baseline_table import build_baseline_table, render_baseline_table


def _write_result(results_dir, name, record):
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_parquet(results_dir / name)


@pytest.fixture
def two_fold_results(tmp_path):
    results_dir = tmp_path / "results"
    for fold_id, wmapes in [(1, {"naive": 1.0, "seasonal_naive": 0.9, "auto_ets": 0.7, "theta": 0.72}), (2, {"naive": 1.1, "seasonal_naive": 0.95, "auto_ets": 0.75, "theta": 0.77})]:
        for model_id, wmape in wmapes.items():
            _write_result(
                results_dir, f"fold{fold_id}_{model_id}.parquet",
                {"fold_id": fold_id, "model_id": model_id, "wall_seconds": 10.0, "wmape": wmape, "n_series": 100, "n_rows": 2800},
            )
    return results_dir


def test_volume_weighted_summary_uses_holdout_volume_not_simple_mean(two_fold_results):
    # fold 1 holdout volume much larger than fold 2's -> summary should
    # skew toward fold 1's WMAPE, not the unweighted average.
    with patch("engine.backtest.baseline_table._fold_holdout_volume", side_effect=lambda fold, *a, **k: {1: 900.0, 2: 100.0}[fold.fold_id]), \
         patch("engine.backtest.baseline_table._fold_population", return_value={"n_kept": 90, "n_total": 100, "vol_kept": 900.0, "vol_total": 1000.0}), \
         patch("engine.backtest.baseline_table.make_folds") as mock_make_folds, \
         patch("engine.backtest.baseline_table.dataset_bounds", return_value=(None, None)):
        from engine.backtest.protocol import Fold
        from datetime import date
        mock_make_folds.return_value = [
            Fold(fold_id=1, train_end=date(2020, 1, 1), holdout_start=date(2020, 1, 2), holdout_end=date(2020, 1, 29)),
            Fold(fold_id=2, train_end=date(2020, 2, 1), holdout_start=date(2020, 2, 2), holdout_end=date(2020, 3, 1)),
        ]
        data = build_baseline_table(two_fold_results)

    simple_mean_naive = (1.0 + 1.1) / 2
    expected_weighted_naive = (1.0 * 900.0 + 1.1 * 100.0) / (900.0 + 100.0)
    assert data["summary"]["naive"] == pytest.approx(expected_weighted_naive)
    assert data["summary"]["naive"] != pytest.approx(simple_mean_naive)
    assert data["summary"]["naive"] == pytest.approx(1.01)  # closer to fold 1 (bigger weight)


def test_render_includes_g1_g2_thresholds_and_naive_discrepancy_flag():
    data = {
        "fold_ids": [1],
        "fold_results": {1: {"naive": {"wmape": 1.0, "wall_seconds": 5.0}, "seasonal_naive": {"wmape": 0.9, "wall_seconds": 5.0}, "auto_ets": {"wmape": 0.7, "wall_seconds": 5.0}, "theta": {"wmape": 0.72, "wall_seconds": 5.0}}},
        "population": {1: {"n_kept": 90, "n_total": 100, "vol_kept": 900.0, "vol_total": 1000.0}},
        "holdout_volume": {1: 900.0},
        "train_end": {1: "2020-01-01"},
        "summary": {"naive": 1.0, "seasonal_naive": 0.9, "auto_ets": 0.7, "theta": 0.72},
    }
    report = render_baseline_table(data)
    # G1: 0.85 * 0.9 = 0.765 (Seasonal Naive, Protocol-P1-correct)
    assert "0.7650" in report
    # G2: 0.92 * 0.7 = 0.644
    assert "0.6440" in report
    # flagged discrepancy: 0.85 * 1.0 = 0.85 (plain Naive)
    assert "0.8500" in report
    assert "Flagged discrepancy" in report
    assert "Seasonal Naive" in report
