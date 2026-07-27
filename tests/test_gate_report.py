"""GATE_REPORT.md assembly tests — against scratch-run outputs, not the
live M5 batch, so this is verified and ready independent of it."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.backtest.gate_report import (
    collect_category_results,
    collect_results,
    render_gate_report,
)
from engine.backtest.run_baselines import ResultSchemaError


def _write_result(tmp_path, name, record):
    path = tmp_path / name
    pd.DataFrame([record]).to_parquet(path)
    return path


def test_collect_results_merges_multiple_directories(tmp_path):
    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    fast_dir.mkdir()
    slow_dir.mkdir()

    _write_result(fast_dir, "fold1_naive.parquet", {"fold_id": 1, "model_id": "naive", "wall_seconds": 1.0, "wmape": 0.98, "n_series": 100, "n_rows": 2800})
    _write_result(slow_dir, "fold1_auto_ets.parquet", {"fold_id": 1, "model_id": "auto_ets", "wall_seconds": 1.0, "wmape": 0.73, "n_series": 100, "n_rows": 2800})

    fold_results = collect_results([fast_dir, slow_dir])
    assert set(fold_results[1].keys()) == {"naive", "auto_ets"}


def test_collect_results_raises_with_gate_report_context_on_bad_file(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    _write_result(bad_dir, "fold1_theta.parquet", {"fold_id": 1, "model_id": "theta", "status": "started"})

    with pytest.raises(ResultSchemaError, match="while assembling GATE_REPORT.md"):
        collect_results([bad_dir])


def test_collect_category_results_returns_empty_shaped_frame_when_none_exist(tmp_path):
    df = collect_category_results([tmp_path])
    assert df.empty
    assert list(df.columns) == ["fold_id", "model_id", "category", "wmape", "n_series", "volume"]


def test_collect_category_results_merges_files(tmp_path):
    cat_dir = tmp_path / "categories"
    cat_dir.mkdir()
    pd.DataFrame(
        [{"fold_id": 1, "model_id": "naive", "category": "FOODS", "wmape": 0.9, "n_series": 10, "volume": 100.0}]
    ).to_parquet(cat_dir / "fold1_naive_by_category.parquet")

    df = collect_category_results([cat_dir])
    assert len(df) == 1
    assert df.iloc[0]["category"] == "FOODS"


def test_render_gate_report_without_category_data_notes_unavailability():
    fold_results = {
        1: {"naive": {"wmape": 0.98}, "auto_ets": {"wmape": 0.73}},
        2: {"naive": {"wmape": 0.99}, "auto_ets": {"wmape": 0.75}},
    }
    report = render_gate_report(fold_results, population_rows=None, category_results=None)
    assert "0.9800" in report
    assert "0.7300" in report
    assert "Not available for this run" in report


def test_render_gate_report_computes_mean_wmape_across_folds():
    fold_results = {1: {"naive": {"wmape": 0.90}}, 2: {"naive": {"wmape": 1.10}}}
    report = render_gate_report(fold_results, population_rows=None, category_results=None)
    assert "1.0000" in report  # mean of 0.90 and 1.10


def test_render_gate_report_handles_model_missing_from_a_fold():
    fold_results = {1: {"naive": {"wmape": 0.9}, "auto_ets": {"wmape": 0.7}}, 2: {"naive": {"wmape": 1.0}}}
    report = render_gate_report(fold_results, population_rows=None, category_results=None)
    lines = [l for l in report.splitlines() if l.startswith("| auto_ets")]
    assert len(lines) == 1
    assert "—" in lines[0]


def test_render_gate_report_includes_population_rows():
    fold_results = {1: {"naive": {"wmape": 0.9}}}
    population_rows = [{"fold_id": 1, "n_series_kept": 26304, "n_series_total": 30490, "volume_kept": 65000.0, "volume_total": 67000.0}]
    report = render_gate_report(fold_results, population_rows=population_rows, category_results=None)
    assert "26,304" in report
    assert "30,490" in report


def test_render_gate_report_includes_category_breakdown_when_present():
    fold_results = {1: {"naive": {"wmape": 0.9}}}
    category_results = pd.DataFrame(
        [
            {"fold_id": 1, "model_id": "naive", "category": "FOODS", "wmape": 0.85, "n_series": 12000, "volume": 4_000_000.0},
            {"fold_id": 1, "model_id": "naive", "category": "HOBBIES", "wmape": 1.10, "n_series": 5000, "volume": 500_000.0},
        ]
    )
    report = render_gate_report(fold_results, population_rows=None, category_results=category_results)
    assert "FOODS" in report
    assert "HOBBIES" in report
    assert "0.8500" in report
