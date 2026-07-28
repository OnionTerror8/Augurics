"""validate_kaggle.py -- fabricated zip/parquet fixtures, never real
Kaggle data (none exists yet). Exercises the exact code path
`make validate-kaggle` runs: unzip into the results tree, validate by
content, check run logs, summarise."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from engine.backtest.protocol import Fold
from engine.backtest.validate_kaggle import render_summary, unzip_into_fold_cache, validate


def _write_comparator_result(root: Path, fold_id: int, ok: bool = True) -> None:
    d = root / "comparator" / "results"
    d.mkdir(parents=True, exist_ok=True)
    if ok:
        pd.DataFrame([{
            "fold_id": fold_id, "model_id": "bayes_glm_pooled", "wall_seconds": 100.0,
            "wmape": 0.7, "n_series": 1000, "n_rows": 28000,
        }]).to_parquet(d / f"fold{fold_id}_bayes_glm_pooled.parquet")
    else:
        pd.DataFrame([{"fold_id": fold_id, "model_id": "bayes_glm_pooled", "status": "started"}]).to_parquet(
            d / f"fold{fold_id}_bayes_glm_pooled.parquet"
        )


def _write_challenger_combo(root: Path, fold_id: int, combo_index: int) -> None:
    d = root / "challenger_grid"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "fold_id": fold_id, "model_id": "lgbm_promo_v1", "combo_index": combo_index,
        "wall_seconds": 10.0, "wmape": 0.6, "n_series": 1000, "n_rows": 28000,
        "hyperparams": json.dumps({"num_leaves": 31}),
    }]).to_parquet(d / f"fold{fold_id}_combo{combo_index}.parquet")


def _write_selected_challenger(root: Path, fold_id: int) -> None:
    d = root / "slow_baseline_results"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "fold_id": fold_id, "model_id": "lgbm_promo_v1", "wall_seconds": 10.0,
        "wmape": 0.55, "n_series": 1000, "n_rows": 28000,
    }]).to_parquet(d / f"fold{fold_id}_lgbm_promo_v1.parquet")


def _zip_dir(src_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in src_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src_root))


@pytest.fixture
def four_folds():
    from datetime import date

    return [
        Fold(fold_id=i, train_end=date(2020, i, 1), holdout_start=date(2020, i, 2), holdout_end=date(2020, i, 15))
        for i in range(1, 5)
    ]


def test_unzip_and_validate_clean_comparator_only(tmp_path, four_folds, monkeypatch):
    staging = tmp_path / "staging"
    for fid in range(1, 5):
        _write_comparator_result(staging, fid)
    (staging / "comparator" / "results" / "run_log.txt").write_text("fold 1 bayes_glm_pooled: 100.0s -- ok\n")
    zip_path = tmp_path / "comparator.zip"
    _zip_dir(staging, zip_path)

    fold_cache = tmp_path / "fold_cache_target"
    with patch("engine.backtest.validate_kaggle.FOLD_CACHE_DIR", fold_cache), \
         patch("engine.backtest.validate_kaggle.COMPARATOR_RESULTS_DIR", fold_cache / "comparator" / "results"), \
         patch("engine.backtest.validate_kaggle.CHALLENGER_GRID_DIR", fold_cache / "challenger_grid"), \
         patch("engine.backtest.validate_kaggle.SLOW_BASELINE_RESULTS_DIR", fold_cache / "slow_baseline_results"), \
         patch("engine.backtest.validate_kaggle.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.validate_kaggle.make_folds", return_value=four_folds):
        unzip_into_fold_cache(zip_path)
        report = validate()

    assert sorted(report.comparator_valid_folds) == [1, 2, 3, 4]
    assert report.clean
    summary = render_summary(report)
    assert "4/4" in summary
    assert "No failures" in summary


def test_validate_detects_schema_failure(tmp_path, four_folds):
    staging = tmp_path / "staging"
    _write_comparator_result(staging, 1, ok=True)
    _write_comparator_result(staging, 2, ok=False)  # bad schema

    fold_cache = tmp_path / "fold_cache_target"
    shutil.copytree(staging, fold_cache)

    with patch("engine.backtest.validate_kaggle.FOLD_CACHE_DIR", fold_cache), \
         patch("engine.backtest.validate_kaggle.COMPARATOR_RESULTS_DIR", fold_cache / "comparator" / "results"), \
         patch("engine.backtest.validate_kaggle.CHALLENGER_GRID_DIR", fold_cache / "challenger_grid"), \
         patch("engine.backtest.validate_kaggle.SLOW_BASELINE_RESULTS_DIR", fold_cache / "slow_baseline_results"), \
         patch("engine.backtest.validate_kaggle.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.validate_kaggle.make_folds", return_value=four_folds):
        report = validate()

    assert report.comparator_valid_folds == [1]
    assert not report.clean
    assert any("fold2_bayes_glm_pooled" in f for f in report.failures)


def test_validate_detects_run_log_failure(tmp_path, four_folds):
    staging = tmp_path / "staging"
    _write_comparator_result(staging, 1)
    (staging / "comparator" / "results" / "run_log.txt").write_text(
        "fold 1 bayes_glm_pooled: FAILED after 10.0s (exit 1): boom\n"
    )
    fold_cache = tmp_path / "fold_cache_target"
    shutil.copytree(staging, fold_cache)

    with patch("engine.backtest.validate_kaggle.FOLD_CACHE_DIR", fold_cache), \
         patch("engine.backtest.validate_kaggle.COMPARATOR_RESULTS_DIR", fold_cache / "comparator" / "results"), \
         patch("engine.backtest.validate_kaggle.CHALLENGER_GRID_DIR", fold_cache / "challenger_grid"), \
         patch("engine.backtest.validate_kaggle.SLOW_BASELINE_RESULTS_DIR", fold_cache / "slow_baseline_results"), \
         patch("engine.backtest.validate_kaggle.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.validate_kaggle.make_folds", return_value=four_folds):
        report = validate()

    assert not report.clean
    assert any("FAILED" in f for f in report.log_failures)


def test_validate_challenger_grid_and_selected_results(tmp_path, four_folds):
    staging = tmp_path / "staging"
    for fid in range(1, 5):
        for combo in range(3):
            _write_challenger_combo(staging, fid, combo)
        _write_selected_challenger(staging, fid)
    (staging / "challenger_grid" / "run_log.txt").write_text("all combos: ok\n")
    fold_cache = tmp_path / "fold_cache_target"
    shutil.copytree(staging, fold_cache)

    with patch("engine.backtest.validate_kaggle.FOLD_CACHE_DIR", fold_cache), \
         patch("engine.backtest.validate_kaggle.COMPARATOR_RESULTS_DIR", fold_cache / "comparator" / "results"), \
         patch("engine.backtest.validate_kaggle.CHALLENGER_GRID_DIR", fold_cache / "challenger_grid"), \
         patch("engine.backtest.validate_kaggle.SLOW_BASELINE_RESULTS_DIR", fold_cache / "slow_baseline_results"), \
         patch("engine.backtest.validate_kaggle.dataset_bounds", return_value=(None, None)), \
         patch("engine.backtest.validate_kaggle.make_folds", return_value=four_folds), \
         patch("engine.backtest.validate_kaggle.load_lgbm_grid", return_value={"combos": [{}] * 3}):
        report = validate()

    assert len(report.challenger_valid_units) == 12
    assert sorted(report.challenger_selected_valid_folds) == [1, 2, 3, 4]
    assert report.clean
