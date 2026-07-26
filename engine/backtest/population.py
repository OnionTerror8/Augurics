"""Active-span filter and population report — Protocol P1, task 1.4.

Active span = first nonzero sale -> dataset end (Protocol P1). Series
below MIN_ACTIVE_SPAN_DAYS are excluded from training. The threshold is
declared here, in code, before this report is generated or seen —
not adjusted after the fact.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from engine.data.ingest import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
POPULATION_REPORT_MD = REPORTS_DIR / "population_report.md"

# Protocol P1: "all series with >= 2 years active history (active span =
# first nonzero sale -> dataset end)".
MIN_ACTIVE_SPAN_DAYS = 2 * 365


def compute_active_span(db_path: Path = DB_PATH, as_of_date: date | None = None) -> pd.DataFrame:
    """Per-series active span and total volume.

    If `as_of_date` is given, "first nonzero sale", "dataset end", and
    total volume are all computed using ONLY rows with `ds <= as_of_date`
    -- this is what lets a backtest fold determine its own eligible
    population from train-fold data only (Protocol P1 / E3), instead of
    using a series' future activity to decide today's inclusion.
    """
    predicate = "WHERE ds <= ?" if as_of_date is not None else ""
    params = [as_of_date] if as_of_date is not None else []
    sql = f"""
        WITH train AS (
            SELECT * FROM panel {predicate}
        ),
        bounds AS (
            SELECT MAX(ds) AS dataset_end FROM train
        ),
        first_sale AS (
            SELECT product_id, location_id, category, MIN(ds) AS first_nonzero_sale
            FROM train
            WHERE y > 0
            GROUP BY product_id, location_id, category
        ),
        volume AS (
            SELECT product_id, location_id, category, SUM(y) AS total_volume
            FROM train
            GROUP BY product_id, location_id, category
        )
        SELECT
            v.product_id,
            v.location_id,
            v.category,
            f.first_nonzero_sale,
            b.dataset_end,
            v.total_volume,
            CASE WHEN f.first_nonzero_sale IS NULL THEN NULL
                 ELSE date_diff('day', f.first_nonzero_sale, b.dataset_end)
            END AS active_span_days
        FROM volume v
        LEFT JOIN first_sale f
            ON f.product_id = v.product_id AND f.location_id = v.location_id
        CROSS JOIN bounds b
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def apply_active_span_filter(series_df: pd.DataFrame, min_days: int = MIN_ACTIVE_SPAN_DAYS) -> pd.DataFrame:
    kept = series_df["active_span_days"].fillna(-1) >= min_days
    return series_df.assign(kept=kept)


def render_report(series_df: pd.DataFrame) -> str:
    n_total = len(series_df)
    n_kept = int(series_df["kept"].sum())
    volume_total = series_df["total_volume"].sum()
    volume_kept = series_df.loc[series_df["kept"], "total_volume"].sum()

    lines = [
        "# Population Report — Active-Span Filter (Protocol P1, task 1.4)",
        "",
        f"Filter: active span (first nonzero sale -> dataset end) >= "
        f"{MIN_ACTIVE_SPAN_DAYS} days (2 years), per PROTOCOL_P1.md. Declared "
        "before this report was generated; not adjusted after seeing these numbers.",
        "",
        "## Overall",
        "",
        f"- Series (product x location) kept: **{n_kept:,} / {n_total:,}** "
        f"({n_kept / n_total:.1%})",
        f"- Volume retained: **{volume_kept:,.0f} / {volume_total:,.0f}** units "
        f"({volume_kept / volume_total:.1%})",
        "",
        "## Per category",
        "",
        "| Category | Series kept | Series total | Share kept | Volume kept | "
        "Volume total | Share of volume retained |",
        "|---|---|---|---|---|---|---|",
    ]

    for category, group in series_df.groupby("category"):
        cat_total = len(group)
        cat_kept = int(group["kept"].sum())
        cat_vol_total = group["total_volume"].sum()
        cat_vol_kept = group.loc[group["kept"], "total_volume"].sum()
        vol_share = f"{cat_vol_kept / cat_vol_total:.1%}" if cat_vol_total > 0 else "n/a"
        lines.append(
            f"| {category} | {cat_kept:,} | {cat_total:,} | {cat_kept / cat_total:.1%} | "
            f"{cat_vol_kept:,.0f} | {cat_vol_total:,.0f} | {vol_share} |"
        )

    lines.append("")
    lines.append(
        "Note: this filter is computed on the full ingested history "
        "(2011-01-29 to dataset end). Per Protocol P1 / E3, the backtest "
        "harness (task 1.5) must recompute active span from **train-fold "
        "data only** for each rolling-origin fold, to avoid using future "
        "information to decide which series enter training — the figures "
        "here are a whole-history population view, not the per-fold filter "
        "that will actually gate model fitting."
    )

    return "\n".join(lines)


def render_per_fold_section(folds, db_path: Path = DB_PATH, min_days: int = MIN_ACTIVE_SPAN_DAYS) -> str:
    """Per-fold eligible population, each computed from that fold's OWN
    train-fold data only (Protocol P1 / E3) -- not the whole-history view
    in the section above. `folds` is a sequence of objects with
    `fold_id`, `train_end`, `holdout_start`, `holdout_end` attributes
    (engine.backtest.protocol.Fold)."""
    lines = [
        "",
        "---",
        "",
        "## Per-fold population (train-fold data only, Protocol P1 / E3)",
        "",
        "Each fold's eligible population is computed from ONLY that fold's "
        "training data (`ds <= fold.train_end`) — active span uses the fold's "
        "own train cutoff as \"dataset end\", not the global dataset end. This "
        "avoids leaking a series' future activity into today's inclusion "
        "decision. The eligible population, and therefore volume retained, "
        "differs per fold.",
        "",
        "| Fold | Train end | Holdout | Series kept | Series total | Share kept | "
        "Volume kept | Volume total | Share of volume retained |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for fold in folds:
        series_df = apply_active_span_filter(
            compute_active_span(db_path, as_of_date=fold.train_end), min_days
        )
        n_total = len(series_df)
        n_kept = int(series_df["kept"].sum())
        vol_total = series_df["total_volume"].sum()
        vol_kept = series_df.loc[series_df["kept"], "total_volume"].sum()
        lines.append(
            f"| {fold.fold_id} | {fold.train_end} | {fold.holdout_start} to "
            f"{fold.holdout_end} | {n_kept:,} | {n_total:,} | {n_kept / n_total:.1%} | "
            f"{vol_kept:,.0f} | {vol_total:,.0f} | {vol_kept / vol_total:.1%} |"
        )

    return "\n".join(lines)


def write_report(db_path: Path = DB_PATH, out_path: Path = POPULATION_REPORT_MD, folds=None) -> Path:
    series_df = apply_active_span_filter(compute_active_span(db_path))
    text = render_report(series_df)
    if folds is not None:
        text += "\n" + render_per_fold_section(folds, db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


if __name__ == "__main__":
    from engine.backtest.protocol import dataset_bounds, make_folds

    _, dataset_end = dataset_bounds()
    written = write_report(folds=make_folds(dataset_end))
    print(f"Wrote {written}")
