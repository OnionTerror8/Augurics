"""BASELINE_TABLE.md — task 1.6 exit artefact.

Per fold: train-end, eligible series, volume retained, volume-weighted
WMAPE per model, wall time per (fold, model). A volume-weighted summary
row per model across all four folds, weighted by each fold's HOLDOUT
volume (the same eligible population is scored by every model within a
fold, so that volume is a fold property, not a model property). States
the implied G1/G2 gate thresholds as concrete numbers, without
evaluating them -- no Rung 2 challenger exists yet to evaluate against.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.backtest.gate_report import collect_results
from engine.backtest.population import MIN_ACTIVE_SPAN_DAYS, apply_active_span_filter, compute_active_span
from engine.backtest.protocol import Fold, dataset_bounds, make_folds
from engine.backtest.run_baselines import CACHE_DIR, get_fold_frames_cached
from engine.data.ingest import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
BASELINE_TABLE_MD = REPORTS_DIR / "BASELINE_TABLE.md"

MODEL_IDS = ["naive", "seasonal_naive", "auto_ets", "theta"]

# Protocol P1 (immutable): G1 comparator is Seasonal Naive, G2 comparator
# is AutoETS. Thresholds themselves (15% / 8% improvement) are also
# immutable -- this module only restates them as concrete WMAPE numbers
# once a summary exists; it does not evaluate any gate.
G1_COMPARATOR_MODEL_ID = "seasonal_naive"
G1_MIN_IMPROVEMENT = 0.15
G2_COMPARATOR_MODEL_ID = "auto_ets"
G2_MIN_IMPROVEMENT = 0.08


def _fold_population(fold: Fold, db_path: Path = DB_PATH, min_days: int = MIN_ACTIVE_SPAN_DAYS) -> dict:
    series_df = apply_active_span_filter(compute_active_span(db_path, as_of_date=fold.train_end), min_days)
    n_total = len(series_df)
    n_kept = int(series_df["kept"].sum())
    vol_total = float(series_df["total_volume"].sum())
    vol_kept = float(series_df.loc[series_df["kept"], "total_volume"].sum())
    return {"n_kept": n_kept, "n_total": n_total, "vol_kept": vol_kept, "vol_total": vol_total}


def _fold_holdout_volume(fold: Fold, db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR) -> float:
    """Total actual (holdout) volume for the fold's eligible population --
    the weight used to combine per-fold WMAPE into a cross-fold summary.
    Same for every model within a fold (same population), read from the
    already-cached holdout frame, no re-fitting required."""
    _, holdout_df = get_fold_frames_cached(fold, db_path, cache_dir)
    return float(holdout_df["y"].sum())


def build_baseline_table(
    results_dir: Path,
    model_ids: list[str] = MODEL_IDS,
    db_path: Path = DB_PATH,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Returns the assembled data (not yet rendered) so tests can check
    the arithmetic independent of markdown formatting."""
    fold_results = collect_results([results_dir])  # schema-validated by content

    _, dataset_end = dataset_bounds(db_path)
    folds = make_folds(dataset_end)
    fold_ids = sorted(fold_results.keys())

    population = {fold.fold_id: _fold_population(fold, db_path) for fold in folds if fold.fold_id in fold_ids}
    holdout_volume = {fold.fold_id: _fold_holdout_volume(fold, db_path, cache_dir) for fold in folds if fold.fold_id in fold_ids}
    train_end = {fold.fold_id: fold.train_end for fold in folds}

    summary = {}
    for model_id in model_ids:
        weighted_num = 0.0
        weighted_den = 0.0
        for fold_id in fold_ids:
            record = fold_results[fold_id].get(model_id)
            if record is None:
                continue
            w = holdout_volume[fold_id]
            weighted_num += record["wmape"] * w
            weighted_den += w
        summary[model_id] = weighted_num / weighted_den if weighted_den > 0 else float("nan")

    return {
        "fold_ids": fold_ids,
        "fold_results": fold_results,
        "population": population,
        "holdout_volume": holdout_volume,
        "train_end": train_end,
        "summary": summary,
    }


def render_baseline_table(data: dict) -> str:
    fold_ids = data["fold_ids"]
    fold_results = data["fold_results"]
    population = data["population"]
    train_end = data["train_end"]
    summary = data["summary"]
    model_ids = MODEL_IDS

    lines = [
        "# BASELINE_TABLE — Rung 1 baselines, task 1.6 exit artefact",
        "",
        "## Per fold",
        "",
    ]

    for fold_id in fold_ids:
        pop = population[fold_id]
        share_kept = pop["n_kept"] / pop["n_total"]
        share_vol = pop["vol_kept"] / pop["vol_total"]
        lines.append(f"### Fold {fold_id} (train end {train_end[fold_id]})")
        lines.append("")
        lines.append(
            f"Eligible series: **{pop['n_kept']:,} / {pop['n_total']:,}** ({share_kept:.1%}). "
            f"Volume retained: **{pop['vol_kept']:,.0f} / {pop['vol_total']:,.0f}** units ({share_vol:.1%})."
        )
        lines.append("")
        lines.append("| Model | WMAPE | Wall time |")
        lines.append("|---|---|---|")
        for model_id in model_ids:
            record = fold_results[fold_id].get(model_id)
            if record is None:
                lines.append(f"| {model_id} | — | — |")
            else:
                lines.append(f"| {model_id} | {record['wmape']:.4f} | {record['wall_seconds']:.1f}s |")
        lines.append("")

    lines.append("## Volume-weighted summary across all four folds")
    lines.append("")
    lines.append(
        "Each fold's WMAPE is weighted by that fold's own holdout volume "
        "(same eligible population scored by every model within a fold) -- "
        "this is the number any gate evaluation will reference."
    )
    lines.append("")
    lines.append("| Model | Volume-weighted WMAPE | Total wall time (all folds) |")
    lines.append("|---|---|---|")
    for model_id in model_ids:
        total_wall = sum(
            fold_results[fid][model_id]["wall_seconds"] for fid in fold_ids if model_id in fold_results[fid]
        )
        lines.append(f"| {model_id} | {summary[model_id]:.4f} | {total_wall:,.1f}s |")

    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "Every figure above comes from validated result parquets "
        "(`engine.backtest.run_baselines.load_and_validate_result`, schema-checked "
        "by content at assembly time) produced by the single orchestrator path "
        "(`engine.backtest.run_slow_baselines` for auto_ets/theta, the same "
        "`score_model`/`_fit_predict_merge` core for naive/seasonal_naive) -- no "
        "number here was computed ad hoc outside that harness."
    )

    lines.append("")
    lines.append("## Implied gate arithmetic (stated, not evaluated)")
    lines.append("")
    lines.append(
        "No Rung 2 challenger exists yet (task 2.3), so no gate is evaluated here. "
        "Protocol P1 (immutable) defines G1's comparator as **Seasonal Naive** and "
        "G2's as **AutoETS** -- restated as concrete WMAPE thresholds:"
    )
    lines.append("")
    g1_threshold = (1 - G1_MIN_IMPROVEMENT) * summary[G1_COMPARATOR_MODEL_ID]
    g2_threshold = (1 - G2_MIN_IMPROVEMENT) * summary[G2_COMPARATOR_MODEL_ID]
    lines.append(
        f"- **G1**: challenger WMAPE must be <= `{1 - G1_MIN_IMPROVEMENT}` x "
        f"{summary[G1_COMPARATOR_MODEL_ID]:.4f} (Seasonal Naive summary) = **{g1_threshold:.4f}**"
    )
    lines.append(
        f"- **G2**: challenger WMAPE must be <= `{1 - G2_MIN_IMPROVEMENT}` x "
        f"{summary[G2_COMPARATOR_MODEL_ID]:.4f} (AutoETS summary) = **{g2_threshold:.4f}**"
    )
    if "naive" in summary:
        g1_vs_naive = 0.85 * summary["naive"]
        lines.append(
            f"- _Flagged discrepancy: if G1 were evaluated against plain Naive "
            f"instead (0.85 x {summary['naive']:.4f} = **{g1_vs_naive:.4f}**), the threshold "
            f"would differ from the Protocol-P1-correct Seasonal Naive threshold above. "
            f"PROTOCOL_P1.md names Seasonal Naive as G1's comparator; this line exists "
            f"only to make the discrepancy visible, not as an alternative gate._"
        )

    return "\n".join(lines)


def write_baseline_table(results_dir: Path, out_path: Path = BASELINE_TABLE_MD) -> Path:
    data = build_baseline_table(results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_baseline_table(data))
    return out_path
