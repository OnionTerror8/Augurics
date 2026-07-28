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


# --- gates evaluation (task 2.4: real G1/G2/G3, or honest PENDING) ---


def _fold_record(model_id: str, wmape: float) -> dict:
    return {"model_id": model_id, "wmape": wmape}


def test_gates_section_pending_when_no_challenger_or_comparator_exist():
    from engine.backtest.gate_report import render_gates_section

    fold_results = {
        1: {"seasonal_naive": _fold_record("seasonal_naive", 0.86), "auto_ets": _fold_record("auto_ets", 0.73)},
        2: {"seasonal_naive": _fold_record("seasonal_naive", 0.87), "auto_ets": _fold_record("auto_ets", 0.73)},
        3: {"seasonal_naive": _fold_record("seasonal_naive", 0.88), "auto_ets": _fold_record("auto_ets", 0.74)},
        4: {"seasonal_naive": _fold_record("seasonal_naive", 0.85), "auto_ets": _fold_record("auto_ets", 0.71)},
    }
    lines = render_gates_section(fold_results)
    text = "\n".join(lines)
    assert "**G1**" in text and "PENDING" in text
    assert "**G2**" in text
    assert "**G3**" in text
    assert "PASS" not in text and "FAIL" not in text


def test_gates_section_evaluates_g1_g2_for_real_when_challenger_present():
    from engine.backtest.gate_report import render_gates_section

    fold_results = {
        fid: {
            "seasonal_naive": _fold_record("seasonal_naive", 0.86),
            "auto_ets": _fold_record("auto_ets", 0.73),
            "lgbm_promo_v1": _fold_record("lgbm_promo_v1", 0.60),  # clears both 15% and 8% improvement
        }
        for fid in (1, 2, 3, 4)
    }
    lines = render_gates_section(fold_results)
    text = "\n".join(lines)
    assert "**G1**: **PASS**" in text or "**G1**" in text and "PASS" in text
    assert "PENDING" in text  # G3 still pending: no comparator result


def test_gates_section_evaluates_g3_for_real_when_comparator_and_challenger_present():
    from engine.backtest.gate_report import render_gates_section

    fold_results = {
        fid: {
            "seasonal_naive": _fold_record("seasonal_naive", 0.86),
            "auto_ets": _fold_record("auto_ets", 0.73),
            "lgbm_promo_v1": _fold_record("lgbm_promo_v1", 0.60),
            "bayes_glm_pooled": _fold_record("bayes_glm_pooled", 0.70),  # challenger wins every fold
        }
        for fid in (1, 2, 3, 4)
    }
    lines = render_gates_section(fold_results)
    text = "\n".join(lines)
    assert "PENDING" not in text
    assert "G3" in text and "PASS" in text
