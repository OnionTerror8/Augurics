"""M5 ingestion -> canonical long-format panel in DuckDB. Task 1.2.

Melts the wide sales file, joins calendar (dates, events) and prices,
and maps SNAP/event days into the columns models will read as
`known_future`. Every join is followed by a row-count reconciliation
against the source files — silent fan-out or row loss raises, it is
never absorbed.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "m5"
DB_PATH = DATA_DIR / "augurics.duckdb"

CALENDAR_CSV = DATA_DIR / "calendar.csv"
PRICES_CSV = DATA_DIR / "sell_prices.csv"
SALES_CSV = DATA_DIR / "sales_train_evaluation.csv"


class SourceFileMissingError(Exception):
    pass


class ReconciliationError(Exception):
    """Raised when a melt/join step changes the row or series count unexpectedly."""


def _require_source_files() -> None:
    for path in (CALENDAR_CSV, PRICES_CSV, SALES_CSV):
        if not path.exists():
            raise SourceFileMissingError(
                f"Required M5 source file missing: {path}. Run "
                "`python -m engine.data.download_m5` first."
            )


def ingest(db_path: Path = DB_PATH) -> dict:
    """Build the `panel` table in DuckDB. Returns reconciliation counts."""
    _require_source_files()

    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA threads=4")

        con.execute(f"CREATE OR REPLACE TABLE raw_calendar AS SELECT * FROM read_csv_auto('{CALENDAR_CSV}')")
        con.execute(f"CREATE OR REPLACE TABLE raw_sell_prices AS SELECT * FROM read_csv_auto('{PRICES_CSV}')")
        con.execute(f"CREATE OR REPLACE TABLE raw_sales_wide AS SELECT * FROM read_csv_auto('{SALES_CSV}')")

        n_series = con.execute("SELECT COUNT(*) FROM raw_sales_wide").fetchone()[0]
        day_cols = [
            c for c in con.execute("SELECT * FROM raw_sales_wide LIMIT 0").df().columns if c.startswith("d_")
        ]
        n_days = len(day_cols)

        unpivot_cols = ", ".join(day_cols)
        con.execute(f"""
            CREATE OR REPLACE TABLE sales_long AS
            SELECT id, item_id, dept_id, cat_id, store_id, state_id, d, y
            FROM raw_sales_wide
            UNPIVOT (y FOR d IN ({unpivot_cols}))
        """)

        n_melted = con.execute("SELECT COUNT(*) FROM sales_long").fetchone()[0]
        expected_melted = n_series * n_days
        if n_melted != expected_melted:
            raise ReconciliationError(
                f"Melt row-count mismatch: expected {expected_melted} "
                f"({n_series} series x {n_days} days), got {n_melted}"
            )

        con.execute("""
            CREATE OR REPLACE TABLE panel AS
            SELECT
                s.item_id AS product_id,
                s.store_id AS location_id,
                c.date AS ds,
                s.y AS y,
                s.cat_id AS category,
                s.dept_id AS department,
                s.state_id AS state,
                c.event_name_1 AS event_name_1,
                c.event_type_1 AS event_type_1,
                c.event_name_2 AS event_name_2,
                c.event_type_2 AS event_type_2,
                (CASE s.state_id
                    WHEN 'CA' THEN c.snap_CA
                    WHEN 'TX' THEN c.snap_TX
                    WHEN 'WI' THEN c.snap_WI
                END = 1) AS snap,
                p.sell_price AS sell_price
            FROM sales_long s
            JOIN raw_calendar c ON s.d = c.d
            LEFT JOIN raw_sell_prices p
                ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
        """)

        n_panel = con.execute("SELECT COUNT(*) FROM panel").fetchone()[0]
        if n_panel != n_melted:
            raise ReconciliationError(
                f"Calendar join changed row count: melted {n_melted}, joined panel has {n_panel}. "
                "Calendar must cover every 'd' value in the sales file with no fan-out."
            )

        n_series_out = con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT product_id, location_id FROM panel)"
        ).fetchone()[0]
        if n_series_out != n_series:
            raise ReconciliationError(f"Series count mismatch after ingestion: source {n_series}, panel {n_series_out}")

        return {
            "n_series": n_series,
            "n_days": n_days,
            "n_melted_rows": n_melted,
            "n_panel_rows": n_panel,
        }
    finally:
        con.close()


if __name__ == "__main__":
    print(ingest())
