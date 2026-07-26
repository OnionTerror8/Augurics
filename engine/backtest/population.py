"""Active-span filter and population report — Protocol P1, task 1.4.

Active span = first nonzero sale -> dataset end (Protocol P1). Series
below MIN_ACTIVE_SPAN_DAYS are excluded from training. The threshold is
declared here, in code, before this report is generated or seen —
not adjusted after the fact.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from engine.data.ingest import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
POPULATION_REPORT_MD = REPORTS_DIR / "population_report.md"

# Protocol P1: "all series with >= 2 years active history (active span =
# first nonzero sale -> dataset end)".
MIN_ACTIVE_SPAN_DAYS = 2 * 365

_ACTIVE_SPAN_SQL = """
    WITH bounds AS (
        SELECT MAX(ds) AS dataset_end FROM panel
    ),
    first_sale AS (
        SELECT product_id, location_id, category, MIN(ds) AS first_nonzero_sale
        FROM panel
        WHERE y > 0
        GROUP BY product_id, location_id, category
    ),
    volume AS (
        SELECT product_id, location_id, category, SUM(y) AS total_volume
        FROM panel
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


def compute_active_span(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Per-series active span and total volume, from the full ingested panel."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(_ACTIVE_SPAN_SQL).df()
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


def write_report(db_path: Path = DB_PATH, out_path: Path = POPULATION_REPORT_MD) -> Path:
    series_df = apply_active_span_filter(compute_active_span(db_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(series_df))
    return out_path


if __name__ == "__main__":
    written = write_report()
    print(f"Wrote {written}")
