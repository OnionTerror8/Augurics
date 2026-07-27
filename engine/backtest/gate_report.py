"""GATE_REPORT.md assembly — per-fold, per-category, and volume-weighted
tables, over validated result parquets. Task 2.4 groundwork, built ahead
of the baseline batch finishing so it's ready the moment it does.

No gates (G1/G2/G3) are evaluated by this module -- that needs a Rung 2
challenger (task 2.3) and the Bayesian comparator's real backtest (task
2.1, deferred until cores are free). This assembles the per-model metric
tables the eventual gate evaluation will read.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.backtest.population import MIN_ACTIVE_SPAN_DAYS, apply_active_span_filter, compute_active_span
from engine.backtest.protocol import Fold
from engine.backtest.run_baselines import ResultSchemaError, load_and_validate_result
from engine.data.ingest import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
GATE_REPORT_MD = REPORTS_DIR / "GATE_REPORT.md"


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
        "# GATE_REPORT — Rung 1 baselines (Protocol P1)",
        "",
        "No gates (G1/G2/G3) are evaluated here -- G1/G2 need a Rung 2 "
        "challenger (task 2.3), G3 needs the Bayesian comparator's full "
        "backtest (task 2.1, deferred until the baseline batch's cores "
        "are free). This is the per-model metric table the eventual gate "
        "evaluation will read.",
        "",
        "## Per-fold, volume-weighted WMAPE",
        "",
        "| Model | " + " | ".join(f"Fold {fid}" for fid in fold_ids) + " | Mean |",
        "|---|" + "---|" * (len(fold_ids) + 1),
    ]
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
