"""kaggle_morning_report.py -- fabricated results, checks section
content/order and that it is genuinely read-only (asserts on the
module's absence of any write/merge/gate-evaluation call)."""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from engine.backtest import kaggle_morning_report as report_module


def _write_comparator_result(results_dir, fold_id, predictive_agreement=None, nuts_spot_check=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "fold_id": fold_id, "model_id": "bayes_glm_pooled", "wall_seconds": 100.0,
        "wmape": 0.7, "n_series": 1000, "n_rows": 28000,
        "predictive_agreement": predictive_agreement,
        "nuts_spot_check": nuts_spot_check,
    }
    pd.DataFrame([record]).to_parquet(results_dir / f"fold{fold_id}_bayes_glm_pooled.parquet")


def _write_challenger_combo(grid_dir, fold_id, combo_index, wmape):
    grid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "fold_id": fold_id, "model_id": "lgbm_promo_v1", "combo_index": combo_index,
        "wall_seconds": 10.0, "wmape": wmape, "n_series": 1000, "n_rows": 28000,
        "hyperparams": json.dumps({"num_leaves": 31}),
    }]).to_parquet(grid_dir / f"fold{fold_id}_combo{combo_index}.parquet")


def _write_selected(results_dir, fold_id, wmape):
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "fold_id": fold_id, "model_id": "lgbm_promo_v1", "wall_seconds": 10.0,
        "wmape": wmape, "n_series": 1000, "n_rows": 28000,
    }]).to_parquet(results_dir / f"fold{fold_id}_lgbm_promo_v1.parquet")


@pytest.fixture
def dirs(tmp_path):
    comparator = tmp_path / "comparator_results"
    grid = tmp_path / "challenger_grid"
    selected = tmp_path / "slow_baseline_results"
    with patch.object(report_module, "COMPARATOR_RESULTS_DIR", comparator), \
         patch.object(report_module, "CHALLENGER_GRID_DIR", grid), \
         patch.object(report_module, "SLOW_BASELINE_RESULTS_DIR", selected):
        yield comparator, grid, selected


def test_section_1_reports_pass_within_tolerance(dirs):
    comparator, _, _ = dirs
    _write_comparator_result(comparator, 1, predictive_agreement={
        "advi_wmape": 0.70, "nuts_wmape": 0.705, "relative_difference": 0.0071, "licensed": True,
    })
    section = report_module.section_1_predictive_agreement()
    assert "PASS" in section
    assert "0.70" in section and "0.705" in section.replace("0.7050", "0.705")


def test_section_1_reports_fail_beyond_tolerance(dirs):
    comparator, _, _ = dirs
    _write_comparator_result(comparator, 1, predictive_agreement={
        "advi_wmape": 0.60, "nuts_wmape": 0.80, "relative_difference": 0.25, "licensed": False,
    })
    section = report_module.section_1_predictive_agreement()
    assert "FAIL" in section
    assert "NOT licensed" in section


def test_section_1_missing_result_says_not_available(dirs):
    section = report_module.section_1_predictive_agreement()
    assert "NOT AVAILABLE" in section
    assert "STOP" in section


def test_section_2_reports_batch_health_and_log_failures(dirs):
    comparator, grid, _ = dirs
    for fid in range(1, 4):
        _write_comparator_result(comparator, fid)
    (comparator / "run_log.txt").write_text("fold 4 bayes_glm_pooled: FAILED after 5.0s: boom\n")
    with patch("engine.backtest.kaggle_morning_report.load_lgbm_grid", return_value={"combos": [{}] * 2}):
        section = report_module.section_2_batch_health()
    assert "fold 1: wmape=0.7000" in section
    assert "1 FAILED line" in section


def test_section_3_reports_nuts_coefficient_diff(dirs):
    comparator, _, _ = dirs
    _write_comparator_result(comparator, 1, nuts_spot_check={
        "ran": True, "category": "FOODS", "capped": True,
        "spot_check_rows": 100000, "original_category_rows": 22432685,
        "mean_abs_coef_diff_advi_vs_nuts": 0.1706,
    })
    section = report_module.section_3_nuts_coefficient_diff()
    assert "0.1706" in section
    assert "FOODS" in section
    assert "never itself a licensing signal" in section


def test_section_3_reports_not_available_when_not_run(dirs):
    comparator, _, _ = dirs
    _write_comparator_result(comparator, 1, nuts_spot_check={"ran": False, "reason": "run_nuts_spot_check=False"})
    section = report_module.section_3_nuts_coefficient_diff()
    assert "NOT AVAILABLE" in section


def test_section_4_summarises_grid_and_selected_combo(dirs):
    _, grid, selected = dirs
    for fid in (1, 2):
        _write_challenger_combo(grid, fid, combo_index=0, wmape=0.70)
        _write_challenger_combo(grid, fid, combo_index=1, wmape=0.55)
        _write_selected(selected, fid, wmape=0.55)

    with patch("engine.backtest.kaggle_morning_report.load_lgbm_grid", return_value={"combos": [{"num_leaves": 31}, {"num_leaves": 63}]}):
        section = report_module.section_4_challenger_grid_summary()

    assert "combo 0" in section and "combo 1" in section
    assert "mean_wmape=0.7000" in section or "0.7000" in section
    assert "fold 1: wmape=0.5500" in section
    assert "fold 2: wmape=0.5500" in section


def test_render_report_preserves_agreed_order_and_never_merges(dirs):
    with patch("engine.backtest.kaggle_morning_report.load_lgbm_grid", return_value={"combos": []}):
        full = report_module.render_report()

    idx1 = full.index("(1)")
    idx2 = full.index("(2)")
    idx3 = full.index("(3)")
    idx4 = full.index("(4)")
    assert idx1 < idx2 < idx3 < idx4
    assert "Nothing above has been merged" in full
    assert "Quarantine lifts only on explicit confirmation" in full


def test_module_never_imports_gate_report_or_write_functions():
    """The module must not be able to merge/evaluate gates even by
    accident -- it should have no importable reference to gate_report.py
    or any select_best_combo_and_write_results-style writer in its own
    namespace (the docstring mentions them by name to explain what this
    module deliberately does NOT do -- that's fine; an actual import
    binding either name into the module's globals would not be)."""
    assert "gate_report" not in report_module.__dict__
    assert "select_best_combo_and_write_results" not in report_module.__dict__
    assert "write_gate_report" not in report_module.__dict__
    assert not any("gate_report" in str(v) for k, v in vars(report_module).items() if k.startswith("__") is False and callable(v))
