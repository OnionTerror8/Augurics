"""GATE_REPORT.md assembly — per-fold, per-category, volume-weighted
tables, AND the real G1/G2/G3 gate evaluation, over validated result
parquets. Task 2.4.

G1/G2 need a Rung 2 challenger (task 2.3); G3 needs the Bayesian
comparator's full backtest (task 2.1). As of this module's task 2.4
run, neither exists (both deferred by real, verified memory
infeasibility -- see KNOWN_ISSUES.md), so all three render as PENDING
with the concrete reason, never as a false pass/fail. The evaluation
logic itself is real and runs the moment either result exists.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.backtest.gates import evaluate_fold_consistent_sign_gate, evaluate_wmape_improvement_gate
from engine.backtest.population import MIN_ACTIVE_SPAN_DAYS, apply_active_span_filter, compute_active_span
from engine.backtest.protocol import Fold
from engine.backtest.run_baselines import ResultSchemaError, load_and_validate_result
from engine.data.ingest import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
GATE_REPORT_MD = REPORTS_DIR / "GATE_REPORT.md"

# Protocol P1 (immutable): G1 vs Seasonal Naive (>=15% improvement), G2
# vs AutoETS (>=8%), G3 vs the Bayesian comparator (>0% improvement,
# fold-consistent sign, wins >= 3/4 folds). Mirrors baseline_table.py's
# constants -- restated here rather than imported, since gate_report.py
# is the module that actually evaluates them and must not depend on the
# BASELINE_TABLE.md-specific assembly module for its own gate logic.
CHALLENGER_MODEL_ID = "lgbm_promo_v1"
COMPARATOR_MODEL_ID = "bayes_glm_pooled"
G1_COMPARATOR_MODEL_ID = "seasonal_naive"
G1_MIN_IMPROVEMENT = 0.15
G2_COMPARATOR_MODEL_ID = "auto_ets"
G2_MIN_IMPROVEMENT = 0.08
G3_MIN_WINS = 3

CHALLENGER_DEFERRED_REASON = (
    "no Rung 2 challenger result yet -- task 2.3 training deferred by a "
    "verified memory-infeasibility check (KNOWN_ISSUES.md: fold-1 scale "
    "estimate 5.13GB vs this laptop's available RAM), awaiting the VM in "
    "PROVISIONING.md. Not a false pass/fail; the evaluation below runs "
    "the moment `lgbm_promo_v1` results exist for all four folds."
)
COMPARATOR_DEFERRED_REASON = (
    "the Bayesian comparator's real 4-fold backtest awaits compute (VM "
    "unavailable, see PROVISIONING.md) and is quarantined regardless: "
    "results are written but not merged into any gate arithmetic until "
    "the fold-1 bounded NUTS + predictive-agreement gate reports and is "
    "reviewed (comparator_evidence.md §7)."
)


def _fold_metric_series(fold_results: dict[int, dict[str, dict]], model_id: str, fold_ids: list[int]) -> list[float] | None:
    """WMAPEs for `model_id` across `fold_ids`, in fold order, or None if
    any fold is missing that model's result -- a partial set is not
    enough to evaluate a gate that must cover the same four folds."""
    values = []
    for fid in fold_ids:
        record = fold_results.get(fid, {}).get(model_id)
        if record is None:
            return None
        values.append(record["wmape"])
    return values


def render_gates_section(fold_results: dict[int, dict[str, dict]]) -> list[str]:
    fold_ids = sorted(fold_results.keys())
    lines = ["## Gates (Protocol P1, immutable thresholds)", ""]

    challenger = _fold_metric_series(fold_results, CHALLENGER_MODEL_ID, fold_ids)
    g1_comparator = _fold_metric_series(fold_results, G1_COMPARATOR_MODEL_ID, fold_ids)
    g2_comparator = _fold_metric_series(fold_results, G2_COMPARATOR_MODEL_ID, fold_ids)
    bayes_comparator = _fold_metric_series(fold_results, COMPARATOR_MODEL_ID, fold_ids)

    if challenger is None or g1_comparator is None:
        lines.append(f"- **G1** (vs Seasonal Naive, >= {G1_MIN_IMPROVEMENT:.0%} improvement): "
                      f"**PENDING** -- {CHALLENGER_DEFERRED_REASON}")
    else:
        g1 = evaluate_wmape_improvement_gate("G1", challenger, g1_comparator, G1_MIN_IMPROVEMENT)
        lines.append(f"- **G1** (vs Seasonal Naive, >= {G1_MIN_IMPROVEMENT:.0%} improvement): "
                      f"**{'PASS' if g1.passed else 'FAIL'}** -- {g1.detail}")

    if challenger is None or g2_comparator is None:
        lines.append(f"- **G2** (vs AutoETS, >= {G2_MIN_IMPROVEMENT:.0%} improvement): "
                      f"**PENDING** -- {CHALLENGER_DEFERRED_REASON}")
    else:
        g2 = evaluate_wmape_improvement_gate("G2", challenger, g2_comparator, G2_MIN_IMPROVEMENT)
        lines.append(f"- **G2** (vs AutoETS, >= {G2_MIN_IMPROVEMENT:.0%} improvement): "
                      f"**{'PASS' if g2.passed else 'FAIL'}** -- {g2.detail}")

    if challenger is None or bayes_comparator is None:
        reason = CHALLENGER_DEFERRED_REASON if challenger is None else COMPARATOR_DEFERRED_REASON
        lines.append(f"- **G3** (vs Bayesian comparator, >0% improvement, wins >= {G3_MIN_WINS}/{len(fold_ids) or 4} folds): "
                      f"**PENDING** -- {reason}")
    else:
        g3 = evaluate_fold_consistent_sign_gate("G3", challenger, bayes_comparator, G3_MIN_WINS)
        lines.append(f"- **G3** (vs Bayesian comparator, >0% improvement, wins >= {G3_MIN_WINS}/{len(fold_ids)} folds): "
                      f"**{'PASS' if g3.passed else 'FAIL'}** -- {g3.detail}")

    lines.append("")
    return lines


def collect_results(results_dirs: list[Path]) -> dict[int, dict[str, dict]]:
    """Merge validated (fold, model) results from one or more directories
    (e.g. the fast in-process run's outputs and the slow batch's outputs).
    Every file is schema-validated by content -- a bad file anywhere
    raises rather than silently entering the table."""
    fold_results: dict[int, dict[str, dict]] = {}
    for results_dir in results_dirs:
        for path in sorted(Path(results_dir).glob("fold*_*.parquet")):
            try:
                record = load_and_validate_result(path)
            except ResultSchemaError as exc:
                raise ResultSchemaError(f"while assembling GATE_REPORT.md: {exc}") from exc
            fold_results.setdefault(record["fold_id"], {})[record["model_id"]] = record
    return fold_results


def collect_category_results(category_results_dirs: list[Path]) -> pd.DataFrame:
    """Merge per-category result parquets (score_model_by_category output),
    if any exist. Returns an empty (but correctly-shaped) frame if none do
    -- callers must not assume category data is always available."""
    frames = []
    for results_dir in category_results_dirs:
        for path in sorted(Path(results_dir).glob("fold*_*_by_category.parquet")):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["fold_id", "model_id", "category", "wmape", "n_series", "volume"])
    return pd.concat(frames, ignore_index=True)


def _population_row(fold: Fold, db_path: Path = DB_PATH, min_days: int = MIN_ACTIVE_SPAN_DAYS) -> dict:
    series_df = apply_active_span_filter(compute_active_span(db_path, as_of_date=fold.train_end), min_days)
    n_total = len(series_df)
    n_kept = int(series_df["kept"].sum())
    vol_total = series_df["total_volume"].sum()
    vol_kept = series_df.loc[series_df["kept"], "total_volume"].sum()
    return {
        "fold_id": fold.fold_id,
        "n_series_kept": n_kept,
        "n_series_total": n_total,
        "volume_kept": vol_kept,
        "volume_total": vol_total,
    }


def render_gate_report(
    fold_results: dict[int, dict[str, dict]],
    population_rows: list[dict] | None = None,
    category_results: pd.DataFrame | None = None,
) -> str:
    fold_ids = sorted(fold_results.keys())
    model_ids = sorted({model_id for fold in fold_results.values() for model_id in fold})

    lines = [
        "# GATE_REPORT (Protocol P1)",
        "",
    ]
    lines.extend(render_gates_section(fold_results))
    lines.extend([
        "## Per-fold, volume-weighted WMAPE",
        "",
        "| Model | " + " | ".join(f"Fold {fid}" for fid in fold_ids) + " | Mean |",
        "|---|" + "---|" * (len(fold_ids) + 1),
    ])
    for model_id in model_ids:
        per_fold = [fold_results[fid][model_id]["wmape"] for fid in fold_ids if model_id in fold_results[fid]]
        mean_wmape = sum(per_fold) / len(per_fold) if per_fold else float("nan")
        cells = " | ".join(
            f"{fold_results[fid][model_id]['wmape']:.4f}" if model_id in fold_results[fid] else "—" for fid in fold_ids
        )
        lines.append(f"| {model_id} | {cells} | {mean_wmape:.4f} |")

    lines.append("")
    lines.append("## Per-fold population (series kept, volume retained)")
    lines.append("")
    if population_rows:
        lines.append("| Fold | Series kept | Series total | Share kept | Volume kept | Volume total | Share of volume retained |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in sorted(population_rows, key=lambda r: r["fold_id"]):
            share_kept = row["n_series_kept"] / row["n_series_total"]
            share_vol = row["volume_kept"] / row["volume_total"]
            lines.append(
                f"| {row['fold_id']} | {row['n_series_kept']:,} | {row['n_series_total']:,} | "
                f"{share_kept:.1%} | {row['volume_kept']:,.0f} | {row['volume_total']:,.0f} | {share_vol:.1%} |"
            )
    else:
        lines.append("_Not provided to render_gate_report()._")

    lines.append("")
    lines.append("## Per-category WMAPE")
    lines.append("")
    if category_results is not None and not category_results.empty:
        for fold_id in sorted(category_results["fold_id"].unique()):
            lines.append(f"### Fold {fold_id}")
            lines.append("")
            lines.append("| Model | Category | WMAPE | Series | Volume |")
            lines.append("|---|---|---|---|---|")
            fold_cat = category_results[category_results["fold_id"] == fold_id]
            for _, row in fold_cat.sort_values(["model_id", "category"]).iterrows():
                lines.append(
                    f"| {row['model_id']} | {row['category']} | {row['wmape']:.4f} | "
                    f"{int(row['n_series']):,} | {row['volume']:,.0f} |"
                )
            lines.append("")
    else:
        lines.append(
            "_Not available for this run: the cached fold frames used by the Rung 1 "
            "baseline batch predate category-aware extraction. `score_model_by_category` "
            "is implemented and tested (tests/test_score_by_category.py) and will populate "
            "this section once fold frames are rebuilt with the `category` column, or for "
            "any future run that uses `run_baselines.score_model_by_category` directly._"
        )

    return "\n".join(lines)


def write_gate_report(
    results_dirs: list[Path],
    folds: list[Fold],
    out_path: Path = GATE_REPORT_MD,
    category_results_dirs: list[Path] | None = None,
    db_path: Path = DB_PATH,
) -> Path:
    fold_results = collect_results(results_dirs)
    population_rows = [_population_row(fold, db_path) for fold in folds]
    category_results = collect_category_results(category_results_dirs or [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_gate_report(fold_results, population_rows, category_results))
    return out_path
