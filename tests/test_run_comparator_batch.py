"""Comparator batch orchestrator logic -- mocked subprocess calls, so this
tests resumability/NUTS-flag/failure-propagation without ever running the
actual (multi-hour) fit. Never touches real fold data."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from engine.backtest.run_comparator_batch import run


def _write_result(path, fold_id):
    pd.DataFrame([{"fold_id": fold_id, "model_id": "bayes_glm_pooled", "wall_seconds": 1.0, "wmape": 0.7, "n_series": 10, "n_rows": 280}]).to_parquet(path)


@pytest.fixture
def four_folds():
    from engine.backtest.protocol import Fold

    return [
        Fold(fold_id=i, train_end=date(2020, i, 1), holdout_start=date(2020, i, 2), holdout_end=date(2020, i, 15))
        for i in range(1, 5)
    ]


def test_skips_folds_with_existing_result_files(tmp_path, four_folds):
    _write_result(tmp_path / "fold1_bayes_glm_pooled.parquet", 1)
    _write_result(tmp_path / "fold2_bayes_glm_pooled.parquet", 2)

    with patch("engine.backtest.run_comparator_batch.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.run_comparator_batch.make_folds", return_value=four_folds), \
         patch("engine.backtest.run_comparator_batch.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        def side_effect(cmd, **kwargs):
            fold_id = int(cmd[cmd.index("--fold-id") + 1])
            _write_result(tmp_path / f"fold{fold_id}_bayes_glm_pooled.parquet", fold_id)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect
        run(results_dir=tmp_path)

        # only folds 3 and 4 should have triggered a subprocess call
        called_fold_ids = {int(c.args[0][c.args[0].index("--fold-id") + 1]) for c in mock_run.call_args_list}
        assert called_fold_ids == {3, 4}


def test_only_the_designated_fold_gets_nuts_spot_check_flag(tmp_path, four_folds):
    with patch("engine.backtest.run_comparator_batch.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.run_comparator_batch.make_folds", return_value=four_folds), \
         patch("engine.backtest.run_comparator_batch.subprocess.run") as mock_run:

        def side_effect(cmd, **kwargs):
            fold_id = int(cmd[cmd.index("--fold-id") + 1])
            _write_result(tmp_path / f"fold{fold_id}_bayes_glm_pooled.parquet", fold_id)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect
        run(results_dir=tmp_path, nuts_spot_check_fold_id=1)

        for call in mock_run.call_args_list:
            cmd = call.args[0]
            fold_id = int(cmd[cmd.index("--fold-id") + 1])
            if fold_id == 1:
                assert "--run-nuts-spot-check" in cmd
            else:
                assert "--run-nuts-spot-check" not in cmd


def test_raises_and_logs_on_subprocess_failure(tmp_path, four_folds):
    with patch("engine.backtest.run_comparator_batch.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.run_comparator_batch.make_folds", return_value=[four_folds[0]]), \
         patch("engine.backtest.run_comparator_batch.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ADVIConvergenceError: loss did not stabilise")

        with pytest.raises(RuntimeError, match="ADVIConvergenceError"):
            run(results_dir=tmp_path)

        log_text = (tmp_path / "run_log.txt").read_text()
        assert "FAILED" in log_text
        assert "ADVIConvergenceError" in log_text
        assert not (tmp_path / "fold1_bayes_glm_pooled.parquet").exists()
