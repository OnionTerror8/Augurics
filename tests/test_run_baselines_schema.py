"""Result-file schema validation — verified by content, never by filename.

A crash, a wrong write target, or a stray orchestrator bookkeeping record
at a result path must all be caught before the final table trusts the
file, not silently accepted because the filename looked right.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.backtest.run_baselines import (
    ResultSchemaError,
    load_and_validate_result,
    load_slow_batch_results,
)


def _write(tmp_path, name, record: dict):
    path = tmp_path / name
    pd.DataFrame([record]).to_parquet(path)
    return path


def test_valid_result_schema_passes(tmp_path):
    path = _write(
        tmp_path,
        "fold1_auto_ets.parquet",
        {
            "fold_id": 1,
            "model_id": "auto_ets",
            "wall_seconds": 100.0,
            "wmape": 0.73,
            "n_series": 26304,
            "n_rows": 736512,
            "fit_seconds": 90.0,
            "predict_seconds": 10.0,
        },
    )
    record = load_and_validate_result(path)
    assert record["n_series"] == 26304
    assert record["wmape"] == pytest.approx(0.73)


def test_bookkeeping_shaped_record_is_rejected(tmp_path):
    """Simulates the exact failure mode this check exists to catch: a
    record shaped like orchestrator bookkeeping (no wmape/n_series/n_rows)
    written to a result path."""
    path = _write(
        tmp_path,
        "fold1_auto_ets.parquet",
        {"fold_id": 1, "model_id": "auto_ets", "status": "started", "pid": 12345},
    )
    with pytest.raises(ResultSchemaError, match="missing expected result columns"):
        load_and_validate_result(path)


def test_empty_result_is_rejected(tmp_path):
    path = tmp_path / "fold1_auto_ets.parquet"
    pd.DataFrame(
        columns=["fold_id", "model_id", "wall_seconds", "wmape", "n_series", "n_rows"]
    ).to_parquet(path)
    with pytest.raises(ResultSchemaError, match="expected exactly one row"):
        load_and_validate_result(path)


def test_negative_wmape_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        "fold1_theta.parquet",
        {"fold_id": 1, "model_id": "theta", "wall_seconds": 5.0, "wmape": -0.1, "n_series": 100, "n_rows": 2800},
    )
    with pytest.raises(ResultSchemaError, match="wmape"):
        load_and_validate_result(path)


def test_zero_n_series_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        "fold1_theta.parquet",
        {"fold_id": 1, "model_id": "theta", "wall_seconds": 5.0, "wmape": 0.5, "n_series": 0, "n_rows": 0},
    )
    with pytest.raises(ResultSchemaError, match="n_series"):
        load_and_validate_result(path)


def test_load_slow_batch_results_organises_by_fold_and_model(tmp_path):
    _write(tmp_path, "fold1_auto_ets.parquet", {"fold_id": 1, "model_id": "auto_ets", "wall_seconds": 1.0, "wmape": 0.7, "n_series": 100, "n_rows": 2800})
    _write(tmp_path, "fold1_theta.parquet", {"fold_id": 1, "model_id": "theta", "wall_seconds": 1.0, "wmape": 0.8, "n_series": 100, "n_rows": 2800})
    _write(tmp_path, "fold2_auto_ets.parquet", {"fold_id": 2, "model_id": "auto_ets", "wall_seconds": 1.0, "wmape": 0.75, "n_series": 105, "n_rows": 2940})

    fold_results = load_slow_batch_results(tmp_path)
    assert set(fold_results.keys()) == {1, 2}
    assert set(fold_results[1].keys()) == {"auto_ets", "theta"}
    assert fold_results[2]["auto_ets"]["wmape"] == pytest.approx(0.75)


def test_load_slow_batch_results_propagates_schema_error_for_any_bad_file(tmp_path):
    _write(tmp_path, "fold1_auto_ets.parquet", {"fold_id": 1, "model_id": "auto_ets", "wall_seconds": 1.0, "wmape": 0.7, "n_series": 100, "n_rows": 2800})
    _write(tmp_path, "fold1_theta.parquet", {"fold_id": 1, "model_id": "theta", "status": "started"})

    with pytest.raises(ResultSchemaError):
        load_slow_batch_results(tmp_path)
