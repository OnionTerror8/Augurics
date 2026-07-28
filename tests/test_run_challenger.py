"""Challenger scoring + grid-search batch orchestrator -- task 2.3.
score_challenger is tested on real synthetic data through the actual
LGBMPromoModel (fast, small); the batch orchestrator's resumability/
failure-propagation/best-combo-selection logic is tested with mocked
subprocess calls, mirroring test_run_comparator_batch.py, since a real
run is a multi-hour, 32GB-RAM operation that belongs on Kaggle, not in
this test suite."""
from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from engine.backtest.run_challenger import score_challenger


def _make_frames(n_series: int, train_days: int, holdout_days: int, category="FOODS"):
    start = date(2020, 1, 1)
    train_rows, holdout_rows = [], []
    for s in range(n_series):
        for i in range(train_days):
            train_rows.append({
                "series_key": f"SKU{s}::STORE1", "ds": start + timedelta(days=i),
                "y": float(1 + (i + s) % 7), "category": category,
                "event_flag": i % 5 == 0, "snap_flag": i % 3 == 0, "price": 5.0 + s,
            })
        for j in range(holdout_days):
            i = train_days + j
            holdout_rows.append({
                "series_key": f"SKU{s}::STORE1", "ds": start + timedelta(days=i),
                "y": float(1 + (i + s) % 7), "category": category,
                "event_flag": i % 5 == 0, "snap_flag": i % 3 == 0, "price": 5.0 + s,
            })
    return pd.DataFrame(train_rows), pd.DataFrame(holdout_rows)


def test_score_challenger_returns_valid_wmape_and_diagnostics():
    train_df, holdout_df = _make_frames(n_series=4, train_days=90, holdout_days=7)
    result = score_challenger(train_df, holdout_df, holdout_days=7, hyperparams={"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 50})
    assert result["wmape"] >= 0
    assert result["n_series"] == 4
    assert result["family"] == "LGBMRegressor"
    assert result["hyperparams"]["num_leaves"] == 31


def test_score_challenger_result_is_parquet_round_trippable():
    train_df, holdout_df = _make_frames(n_series=3, train_days=90, holdout_days=7)
    result = score_challenger(train_df, holdout_df, holdout_days=7, hyperparams={"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 50})
    record = {
        "fold_id": 1, "model_id": "lgbm_promo_v1", "combo_index": 0, "wall_seconds": 1.0,
        **{k: v for k, v in result.items() if k != "hyperparams"},
        "hyperparams": json.dumps(result["hyperparams"]),
    }
    df = pd.DataFrame([record])
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.parquet"
        df.to_parquet(path)
        back = pd.read_parquet(path)
    assert back.iloc[0]["wmape"] == pytest.approx(result["wmape"])
    assert json.loads(back.iloc[0]["hyperparams"]) == result["hyperparams"]


# --- grid-search batch orchestrator (mocked subprocess) ---


def _write_combo_result(path, fold_id, combo_index, wmape):
    pd.DataFrame([{
        "fold_id": fold_id, "model_id": "lgbm_promo_v1", "combo_index": combo_index,
        "wall_seconds": 1.0, "wmape": wmape, "n_series": 10, "n_rows": 280,
        "fit_seconds": 0.5, "predict_seconds": 0.1, "family": "LGBMRegressor",
        "hyperparams": json.dumps({"num_leaves": 31}),
    }]).to_parquet(path)


def test_select_best_combo_picks_lowest_mean_wmape_across_folds(tmp_path):
    from engine.backtest.run_challenger_batch import select_best_combo_and_write_results

    grid_dir = tmp_path / "grid"
    results_dir = tmp_path / "results"
    grid_dir.mkdir()

    # combo 0: mean wmape 0.70; combo 1: mean wmape 0.60 (wins)
    for fold_id, w in zip([1, 2], [0.70, 0.70]):
        _write_combo_result(grid_dir / f"fold{fold_id}_combo0.parquet", fold_id, 0, w)
    for fold_id, w in zip([1, 2], [0.55, 0.65]):
        _write_combo_result(grid_dir / f"fold{fold_id}_combo1.parquet", fold_id, 1, w)

    with patch("engine.backtest.run_challenger_batch.load_lgbm_grid", return_value={"combos": [{"num_leaves": 31}, {"num_leaves": 63}]}):
        summary = select_best_combo_and_write_results(grid_dir, results_dir, fold_ids=[1, 2])

    assert summary["best_combo_index"] == 1
    assert summary["mean_wmape"] == pytest.approx(0.60)
    assert (results_dir / "fold1_lgbm_promo_v1.parquet").exists()
    assert (results_dir / "fold2_lgbm_promo_v1.parquet").exists()
    written = pd.read_parquet(results_dir / "fold1_lgbm_promo_v1.parquet")
    assert set(written.columns) == {"fold_id", "model_id", "wall_seconds", "wmape", "n_series", "n_rows"}
    assert written.iloc[0]["wmape"] == pytest.approx(0.55)


def test_select_best_combo_raises_if_no_combo_covers_all_folds(tmp_path):
    from engine.backtest.run_challenger_batch import select_best_combo_and_write_results

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    _write_combo_result(grid_dir / "fold1_combo0.parquet", 1, 0, 0.7)  # fold 2 missing for combo 0

    with patch("engine.backtest.run_challenger_batch.load_lgbm_grid", return_value={"combos": [{"num_leaves": 31}]}):
        with pytest.raises(RuntimeError, match="no combo has results for all folds"):
            select_best_combo_and_write_results(grid_dir, tmp_path / "results", fold_ids=[1, 2])


def test_batch_skips_existing_combo_results(tmp_path):
    from engine.backtest.run_challenger_batch import run

    grid_dir = tmp_path / "grid"
    results_dir = tmp_path / "results"
    grid_dir.mkdir()

    folds = _four_folds()
    grid = {"combos": [{"num_leaves": 31}, {"num_leaves": 63}]}
    # pre-populate fold 1's results for both combos
    for combo_index in (0, 1):
        _write_combo_result(grid_dir / f"fold1_combo{combo_index}.parquet", 1, combo_index, 0.7)

    with patch("engine.backtest.run_challenger_batch.load_lgbm_grid", return_value=grid), \
         patch("engine.backtest.run_challenger_batch.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.run_challenger_batch.make_folds", return_value=folds), \
         patch("engine.backtest.run_challenger_batch.subprocess.run") as mock_run:

        def side_effect(cmd, **kwargs):
            fold_id = int(cmd[cmd.index("--fold-id") + 1])
            combo_index = int(cmd[cmd.index("--combo-index") + 1])
            _write_combo_result(grid_dir / f"fold{fold_id}_combo{combo_index}.parquet", fold_id, combo_index, 0.6)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect
        run(grid_results_dir=grid_dir, results_dir=results_dir)

        called = {(int(c.args[0][c.args[0].index("--fold-id") + 1]), int(c.args[0][c.args[0].index("--combo-index") + 1])) for c in mock_run.call_args_list}
        assert (1, 0) not in called and (1, 1) not in called  # already done, skipped
        assert len(called) == 3 * 2  # folds 2,3,4 x 2 combos


def test_batch_raises_and_stops_on_subprocess_failure(tmp_path):
    from engine.backtest.run_challenger_batch import run

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    folds = _four_folds()
    grid = {"combos": [{"num_leaves": 31}]}

    with patch("engine.backtest.run_challenger_batch.load_lgbm_grid", return_value=grid), \
         patch("engine.backtest.run_challenger_batch.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.run_challenger_batch.make_folds", return_value=folds), \
         patch("engine.backtest.run_challenger_batch.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
        with pytest.raises(RuntimeError, match="failed"):
            run(grid_results_dir=grid_dir, results_dir=tmp_path / "results")


def _four_folds():
    from engine.backtest.protocol import Fold

    return [
        Fold(fold_id=i, train_end=date(2020, i, 1), holdout_start=date(2020, i, 2), holdout_end=date(2020, i, 15))
        for i in range(1, 5)
    ]
