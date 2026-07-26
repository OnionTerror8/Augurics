"""Coverage diagnostic — Protocol P1, E1. Task 1.3.

Month-by-month profile of price/event coverage and mean sales, overall
and per category, computed on the ingested panel BEFORE any model fit.
This module only computes and reports; any resulting training-window
bound is a documented label-integrity decision, not a silent truncation.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from engine.data.ingest import DB_PATH

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
COVERAGE_REPORT_MD = REPORTS_DIR / "coverage_report.md"

# Below this, >10% of panel rows have no sell_price -- structurally
# not-yet-introduced (product, store) pairs, not real absence-of-demand.
PRICE_COVERAGE_BOUND_THRESHOLD = 0.90

_OVERALL_SQL = """
    SELECT
        strftime(ds, '%Y-%m') AS year_month,
        COUNT(*) AS n_rows,
        AVG(CASE WHEN sell_price IS NOT NULL THEN 1.0 ELSE 0.0 END) AS price_coverage,
        AVG(CASE WHEN event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL THEN 1.0 ELSE 0.0 END) AS event_coverage,
        AVG(y) AS mean_sales
    FROM panel
    GROUP BY 1
    ORDER BY 1
"""

_BY_CATEGORY_SQL = """
    SELECT
        strftime(ds, '%Y-%m') AS year_month,
        category,
        COUNT(*) AS n_rows,
        AVG(CASE WHEN sell_price IS NOT NULL THEN 1.0 ELSE 0.0 END) AS price_coverage,
        AVG(CASE WHEN event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL THEN 1.0 ELSE 0.0 END) AS event_coverage,
        AVG(y) AS mean_sales
    FROM panel
    GROUP BY 1, 2
    ORDER BY 1, 2
"""


def compute_monthly_coverage(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Overall (volume-pooled) month-by-month coverage profile."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(_OVERALL_SQL).df()
    finally:
        con.close()


def compute_monthly_coverage_by_category(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(_BY_CATEGORY_SQL).df()
    finally:
        con.close()


def first_month_reaching(monthly: pd.DataFrame, column: str, threshold: float) -> str | None:
    """First year_month (ascending) where `column` is >= threshold for the
    rest of the series -- a single early fluctuation above the bar does not
    count as the crossing point."""
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    for i in range(len(monthly)):
        if (monthly.loc[i:, column] >= threshold).all():
            return monthly.loc[i, "year_month"]
    return None


def _render_table(df: pd.DataFrame) -> list[str]:
    lines = ["| Month | Rows | Price coverage | Event coverage | Mean sales |", "|---|---|---|---|---|"]
    for row in df.itertuples():
        lines.append(
            f"| {row.year_month} | {row.n_rows:,} | {row.price_coverage:.1%} |"
            f" {row.event_coverage:.1%} | {row.mean_sales:.3f} |"
        )
    return lines


def render_report(db_path: Path = DB_PATH) -> str:
    overall = compute_monthly_coverage(db_path)
    by_category = compute_monthly_coverage_by_category(db_path)
    bound_month = first_month_reaching(overall, "price_coverage", PRICE_COVERAGE_BOUND_THRESHOLD)

    lines = [
        "# Coverage Diagnostic — Protocol P1, E1",
        "",
        "Run BEFORE any model fit, per PROTOCOL_P1.md. Month-by-month price and "
        "event coverage and mean sales, overall and per category, on the full "
        "ingested M5 panel — no filtering applied yet (that is task 1.4).",
        "",
        "## Finding: sustained low price coverage in early years (label-integrity decision)",
        "",
        f"Aggregate (volume-pooled) `price_coverage` does not reach and hold "
        f"{PRICE_COVERAGE_BOUND_THRESHOLD:.0%} until **{bound_month}**. Below that "
        "threshold, more than 10% of panel rows have no `sell_price` — these are "
        "structurally not-yet-introduced (product, store) pairs logging zero sales, "
        "not true absence-of-demand signal. This is a multi-year ramp (the M5 "
        "assortment grows steadily from 2011 to 2016), not a single-month cliff, "
        "but it is the same label-integrity concern the diagnostic exists to catch: "
        "early-period mean_sales and price-covariate availability are not comparable "
        "to later periods.",
        "",
        f"**Decision:** any pooled/aggregate view of the panel (comparator fits, "
        f"category-level summaries) bounds its training window to **{bound_month} "
        f"onward** for price-covariate reliability. This is independent of, and does "
        f"not replace, the per-series active-span filter in task 1.4, which removes "
        f"individual not-yet-introduced series regardless of calendar date.",
        "",
        "## Overall monthly coverage (volume-pooled)",
        "",
        *_render_table(overall),
        "",
        "## Per-category monthly coverage",
        "",
    ]

    for category in sorted(by_category["category"].unique()):
        lines.append(f"### {category}")
        lines.append("")
        lines.extend(_render_table(by_category[by_category["category"] == category]))
        lines.append("")

    return "\n".join(lines)


def write_report(db_path: Path = DB_PATH, out_path: Path = COVERAGE_REPORT_MD) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(db_path))
    return out_path


if __name__ == "__main__":
    written = write_report()
    print(f"Wrote {written}")
