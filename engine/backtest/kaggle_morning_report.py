"""The morning report -- task 2.1/2.3, run after `make validate-kaggle`
reports clean. Read-only: prints a summary in the agreed order and
touches nothing else. Deliberately does NOT regenerate GATE_REPORT.md,
does NOT call select_best_combo_and_write_results, does NOT evaluate
any gate -- per instruction, nothing merges and no gate evaluates
until the fold-1 predictive-agreement number has been seen and
quarantine is explicitly lifted. That is a separate, later step.

Order: (1) fold-1 predictive-agreement, the decisive number, first;
(2) batch health, so a partial/failed batch is obvious before reading
any WMAPE number; (3) the bounded NUTS coefficient-difference figure,
context for (1) but never itself a licensing signal (the gate compares
predictions, not posteriors); (4) challenger grid summary, informational
only until gates actually run.
"""
from __future__ import annotations

from pathlib import Path

from engine.backtest.config import load_lgbm_grid
from engine.backtest.run_baselines import load_and_validate_result
from engine.backtest.run_comparator import PREDICTIVE_AGREEMENT_REL_TOLERANCE
from engine.backtest.validate_kaggle import (
    CHALLENGER_GRID_DIR,
    COMPARATOR_RESULTS_DIR,
    SLOW_BASELINE_RESULTS_DIR,
)

NUTS_SPOT_CHECK_FOLD_ID = 1


def section_1_predictive_agreement() -> str:
    path = COMPARATOR_RESULTS_DIR / f"fold{NUTS_SPOT_CHECK_FOLD_ID}_bayes_glm_pooled.parquet"
    if not path.exists():
        return "(1) FOLD-1 PREDICTIVE AGREEMENT: NOT AVAILABLE -- fold 1's comparator result is missing. STOP."

    record = load_and_validate_result(path)
    pa = record.get("predictive_agreement")
    if pa is None:
        return "(1) FOLD-1 PREDICTIVE AGREEMENT: NOT AVAILABLE -- result exists but has no predictive_agreement field (was --run-nuts-spot-check used?). STOP."

    lines = [
        "(1) FOLD-1 PREDICTIVE AGREEMENT -- the decisive licensing number",
        "",
        "| | WMAPE |",
        "|---|---|",
        f"| ADVI  | {pa['advi_wmape']:.4f} |",
        f"| NUTS  | {pa['nuts_wmape']:.4f} |",
        "",
        f"Relative difference: {pa['relative_difference']:.2%} "
        f"(tolerance: {PREDICTIVE_AGREEMENT_REL_TOLERANCE:.0%})",
        f"Result: {'PASS -- licensed' if pa['licensed'] else 'FAIL -- NOT licensed'}",
    ]
    return "\n".join(lines)


def section_2_batch_health() -> str:
    lines = ["(2) BATCH HEALTH"]

    lines.append("\nComparator:")
    if COMPARATOR_RESULTS_DIR.exists():
        for path in sorted(COMPARATOR_RESULTS_DIR.glob("fold*_bayes_glm_pooled.parquet")):
            record = load_and_validate_result(path)
            lines.append(f"  fold {record['fold_id']}: wmape={record['wmape']:.4f} wall={record['wall_seconds']:.1f}s")
        log_path = COMPARATOR_RESULTS_DIR / "run_log.txt"
        if log_path.exists():
            failed = [l for l in log_path.read_text().splitlines() if "FAILED" in l]
            lines.append(f"  run_log.txt: {len(failed)} FAILED line(s)" if failed else "  run_log.txt: clean")
    else:
        lines.append("  no comparator results directory found")

    lines.append("\nChallenger grid:")
    if CHALLENGER_GRID_DIR.exists():
        n_combos = len(load_lgbm_grid()["combos"])
        units = sorted(CHALLENGER_GRID_DIR.glob("fold*_combo*.parquet"))
        lines.append(f"  {len(units)}/{4 * n_combos} (fold, combo) units present")
        log_path = CHALLENGER_GRID_DIR / "run_log.txt"
        if log_path.exists():
            failed = [l for l in log_path.read_text().splitlines() if "FAILED" in l]
            lines.append(f"  run_log.txt: {len(failed)} FAILED line(s)" if failed else "  run_log.txt: clean")
            for f in failed:
                lines.append(f"    - {f}")
    else:
        lines.append("  no challenger grid directory found")

    return "\n".join(lines)


def section_3_nuts_coefficient_diff() -> str:
    path = COMPARATOR_RESULTS_DIR / f"fold{NUTS_SPOT_CHECK_FOLD_ID}_bayes_glm_pooled.parquet"
    if not path.exists():
        return "(3) BOUNDED NUTS COEFFICIENT DIFFERENCE: NOT AVAILABLE -- fold 1's comparator result is missing."

    record = load_and_validate_result(path)
    nuts = record.get("nuts_spot_check") or {}
    if not nuts.get("ran"):
        return f"(3) BOUNDED NUTS COEFFICIENT DIFFERENCE: NOT AVAILABLE -- {nuts.get('reason', 'nuts_spot_check did not run')}."

    return (
        "(3) BOUNDED NUTS COEFFICIENT DIFFERENCE (context for (1), never itself a licensing signal)\n\n"
        f"category: {nuts['category']}, capped: {nuts['capped']} "
        f"({nuts['spot_check_rows']:,}/{nuts['original_category_rows']:,} rows)\n"
        f"mean_abs_coef_diff_advi_vs_nuts: {nuts['mean_abs_coef_diff_advi_vs_nuts']:.4f}"
    )


def section_4_challenger_grid_summary() -> str:
    grid = load_lgbm_grid()
    lines = ["(4) CHALLENGER GRID SUMMARY (informational -- not merged, no gate evaluated)"]

    if CHALLENGER_GRID_DIR.exists():
        by_combo: dict[int, list[float]] = {}
        for path in sorted(CHALLENGER_GRID_DIR.glob("fold*_combo*.parquet")):
            record = load_and_validate_result(path)
            by_combo.setdefault(int(record["combo_index"]), []).append(record["wmape"])

        lines.append("\nPer-combo mean WMAPE across available folds:")
        for combo_index in sorted(by_combo):
            mean_wmape = sum(by_combo[combo_index]) / len(by_combo[combo_index])
            n_folds = len(by_combo[combo_index])
            lines.append(f"  combo {combo_index} {grid['combos'][combo_index]}: mean_wmape={mean_wmape:.4f} ({n_folds}/4 folds)")

    lines.append("\nSelected best combo (already written by run_challenger_batch.py on Kaggle, per the pre-registered lowest-mean-WMAPE rule):")
    if SLOW_BASELINE_RESULTS_DIR.exists():
        selected = sorted(SLOW_BASELINE_RESULTS_DIR.glob("fold*_lgbm_promo_v1.parquet"))
        if selected:
            wmapes = []
            for path in selected:
                record = load_and_validate_result(path)
                wmapes.append(record["wmape"])
                lines.append(f"  fold {record['fold_id']}: wmape={record['wmape']:.4f}")
            lines.append(f"  mean across {len(wmapes)} fold(s): {sum(wmapes) / len(wmapes):.4f}")
        else:
            lines.append("  none found yet")
    else:
        lines.append("  slow_baseline_results directory not found")

    return "\n".join(lines)


def render_report() -> str:
    return "\n\n".join([
        section_1_predictive_agreement(),
        section_2_batch_health(),
        section_3_nuts_coefficient_diff(),
        section_4_challenger_grid_summary(),
        "---\nNothing above has been merged into GATE_REPORT.md or any gate arithmetic. "
        "Quarantine lifts only on explicit confirmation after reviewing (1).",
    ])


def main() -> None:
    print(render_report())


if __name__ == "__main__":
    main()
