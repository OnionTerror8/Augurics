"""One-command validation for results returned from Kaggle -- unzips
both notebooks' output archives into the results tree, passes every
parquet through load_and_validate_result (schema-checked by content,
same as every other result in this codebase, never trusted by filename
alone), checks both run logs for clean completion, and prints a single
summary. Read-only otherwise: never touches GATE_REPORT.md, never
merges anything, never evaluates a gate -- that is a separate,
explicit step the user takes after reviewing the morning report
(engine.backtest.kaggle_morning_report).
"""
from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from engine.backtest.config import load_lgbm_grid
from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_baselines import ResultSchemaError, load_and_validate_result

FOLD_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "fold_cache"
COMPARATOR_RESULTS_DIR = FOLD_CACHE_DIR / "comparator" / "results"
CHALLENGER_GRID_DIR = FOLD_CACHE_DIR / "challenger_grid"
SLOW_BASELINE_RESULTS_DIR = FOLD_CACHE_DIR / "slow_baseline_results"


@dataclass
class ValidationReport:
    comparator_valid_folds: list[int] = field(default_factory=list)
    comparator_expected_folds: list[int] = field(default_factory=list)
    challenger_valid_units: list[tuple[int, int]] = field(default_factory=list)
    challenger_expected_units: list[tuple[int, int]] = field(default_factory=list)
    challenger_selected_valid_folds: list[int] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    log_failures: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.failures and not self.log_failures


def unzip_into_fold_cache(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"validate_kaggle: zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(FOLD_CACHE_DIR)


def _check_run_log(log_path: Path, failures: list[str]) -> None:
    if not log_path.exists():
        failures.append(f"{log_path}: run_log.txt missing -- cannot confirm clean completion")
        return
    for line in log_path.read_text().splitlines():
        if "FAILED" in line:
            failures.append(f"{log_path.parent.name}/run_log.txt: {line}")


def validate(expected_fold_ids: list[int] | None = None) -> ValidationReport:
    report = ValidationReport()

    _, dataset_end = dataset_bounds()
    fold_ids = expected_fold_ids or [f.fold_id for f in make_folds(dataset_end)]
    grid = load_lgbm_grid()
    n_combos = len(grid["combos"])

    report.comparator_expected_folds = fold_ids
    report.challenger_expected_units = [(fid, ci) for fid in fold_ids for ci in range(n_combos)]

    # comparator
    if COMPARATOR_RESULTS_DIR.exists():
        _check_run_log(COMPARATOR_RESULTS_DIR / "run_log.txt", report.log_failures)
        for path in sorted(COMPARATOR_RESULTS_DIR.glob("fold*_bayes_glm_pooled.parquet")):
            try:
                record = load_and_validate_result(path)
                report.comparator_valid_folds.append(int(record["fold_id"]))
            except ResultSchemaError as exc:
                report.failures.append(f"{path.name}: {exc}")

    # challenger grid (raw per-combo results)
    if CHALLENGER_GRID_DIR.exists():
        _check_run_log(CHALLENGER_GRID_DIR / "run_log.txt", report.log_failures)
        for path in sorted(CHALLENGER_GRID_DIR.glob("fold*_combo*.parquet")):
            try:
                record = load_and_validate_result(path)
                report.challenger_valid_units.append((int(record["fold_id"]), int(record["combo_index"])))
            except ResultSchemaError as exc:
                report.failures.append(f"{path.name}: {exc}")

    # challenger selected best-combo results (what GATE_REPORT.md would read)
    if SLOW_BASELINE_RESULTS_DIR.exists():
        for path in sorted(SLOW_BASELINE_RESULTS_DIR.glob("fold*_lgbm_promo_v1.parquet")):
            try:
                record = load_and_validate_result(path)
                report.challenger_selected_valid_folds.append(int(record["fold_id"]))
            except ResultSchemaError as exc:
                report.failures.append(f"{path.name}: {exc}")

    return report


def render_summary(report: ValidationReport) -> str:
    lines = ["=== validate-kaggle summary ===", ""]

    lines.append(
        f"Comparator: {len(report.comparator_valid_folds)}/{len(report.comparator_expected_folds)} "
        f"folds valid -- {sorted(report.comparator_valid_folds)}"
    )
    lines.append(
        f"Challenger grid: {len(report.challenger_valid_units)}/{len(report.challenger_expected_units)} "
        f"(fold, combo) units valid"
    )
    lines.append(
        f"Challenger selected (best combo, what GATE_REPORT.md reads): "
        f"{len(report.challenger_selected_valid_folds)}/{len(report.comparator_expected_folds)} folds valid -- "
        f"{sorted(report.challenger_selected_valid_folds)}"
    )
    lines.append("")

    if report.failures:
        lines.append(f"SCHEMA FAILURES ({len(report.failures)}):")
        lines.extend(f"  - {f}" for f in report.failures)
    if report.log_failures:
        lines.append(f"RUN LOG FAILURES ({len(report.log_failures)}):")
        lines.extend(f"  - {f}" for f in report.log_failures)
    if report.clean:
        lines.append("No failures. Nothing merged, no gates evaluated -- that remains a separate step.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator-zip", type=Path, default=None)
    parser.add_argument("--challenger-zip", type=Path, default=None)
    args = parser.parse_args()

    if not args.comparator_zip and not args.challenger_zip:
        raise SystemExit("validate_kaggle: provide at least one of --comparator-zip / --challenger-zip")

    if args.comparator_zip:
        unzip_into_fold_cache(args.comparator_zip)
    if args.challenger_zip:
        unzip_into_fold_cache(args.challenger_zip)

    report = validate()
    print(render_summary(report))
    if not report.clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
