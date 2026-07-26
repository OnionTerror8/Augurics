"""Row-count reconciliation for M5 ingestion — task 1.2.

Fail-loud tests (missing source files) run everywhere. The full
reconciliation-against-source-CSVs test only runs where the M5 files
have actually been downloaded (they are gitignored and not present in CI).
"""
from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from engine.data.ingest import SALES_CSV, SourceFileMissingError, ingest

requires_m5_data = pytest.mark.skipif(
    not SALES_CSV.exists(),
    reason="M5 source files not present locally; run engine.data.download_m5 first",
)


def test_ingest_raises_on_missing_source_files(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.data.ingest.CALENDAR_CSV", tmp_path / "missing_calendar.csv")
    monkeypatch.setattr("engine.data.ingest.PRICES_CSV", tmp_path / "missing_prices.csv")
    monkeypatch.setattr("engine.data.ingest.SALES_CSV", tmp_path / "missing_sales.csv")

    with pytest.raises(SourceFileMissingError):
        ingest(db_path=tmp_path / "db.duckdb")


def _csv_row_count(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in f) - 1  # minus header


@requires_m5_data
def test_ingest_reconciles_row_counts_against_source(tmp_path):
    counts = ingest(db_path=tmp_path / "test.duckdb")

    n_series_source = _csv_row_count(SALES_CSV)
    assert counts["n_series"] == n_series_source, "melt must preserve the source series count"

    with SALES_CSV.open(newline="") as f:
        header = next(csv.reader(f))
    n_day_cols = len([c for c in header if c.startswith("d_")])
    assert counts["n_days"] == n_day_cols

    assert counts["n_melted_rows"] == n_series_source * n_day_cols, (
        "melted row count must equal series x days, no fan-out or loss"
    )
    assert counts["n_panel_rows"] == counts["n_melted_rows"], (
        "calendar/price joins must not change row count"
    )

    con = duckdb.connect(str(tmp_path / "test.duckdb"))
    try:
        panel_rows = con.execute("SELECT COUNT(*) FROM panel").fetchone()[0]
        assert panel_rows == counts["n_panel_rows"]

        null_ds = con.execute("SELECT COUNT(*) FROM panel WHERE ds IS NULL").fetchone()[0]
        assert null_ds == 0, "every row must join to a calendar date"
    finally:
        con.close()
