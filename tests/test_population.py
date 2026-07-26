"""Active-span filter tests against a small synthetic panel — no M5 data required."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from datetime import date

from engine.backtest.population import (
    MIN_ACTIVE_SPAN_DAYS,
    apply_active_span_filter,
    compute_active_span,
    render_per_fold_section,
    render_report,
)
from engine.backtest.protocol import Fold


@pytest.fixture
def synthetic_db(tmp_path):
    db_path = tmp_path / "synthetic.duckdb"
    con = duckdb.connect(str(db_path))
    # Dataset end fixed at 2020-12-31 for all series.
    # series A (FOODS): first nonzero sale 2018-01-01 -> active span ~3 years, kept.
    # series B (FOODS): first nonzero sale 2020-06-01 -> active span ~7 months, excluded.
    # series C (HOBBIES): never sells (all zero), excluded.
    con.execute("""
        CREATE TABLE panel AS
        SELECT * FROM (VALUES
            ('P_A', 'S1', 'FOODS', DATE '2018-01-01', 5.0),
            ('P_A', 'S1', 'FOODS', DATE '2020-12-31', 5.0),
            ('P_B', 'S1', 'FOODS', DATE '2020-06-01', 2.0),
            ('P_B', 'S1', 'FOODS', DATE '2020-12-31', 2.0),
            ('P_C', 'S1', 'HOBBIES', DATE '2018-01-01', 0.0),
            ('P_C', 'S1', 'HOBBIES', DATE '2020-12-31', 0.0)
        ) AS t(product_id, location_id, category, ds, y)
    """)
    con.close()
    return db_path


def test_compute_active_span(synthetic_db):
    series_df = compute_active_span(synthetic_db)
    assert len(series_df) == 3

    a = series_df[series_df["product_id"] == "P_A"].iloc[0]
    assert a["active_span_days"] == (pd.Timestamp("2020-12-31") - pd.Timestamp("2018-01-01")).days

    c = series_df[series_df["product_id"] == "P_C"].iloc[0]
    assert pd.isna(c["active_span_days"])
    assert c["total_volume"] == 0.0


def test_active_span_filter_keeps_only_long_active_series(synthetic_db):
    series_df = apply_active_span_filter(compute_active_span(synthetic_db))
    kept = dict(zip(series_df["product_id"], series_df["kept"]))
    assert kept["P_A"] is True or kept["P_A"] == True  # noqa: E712
    assert kept["P_B"] == False  # noqa: E712
    assert kept["P_C"] == False  # noqa: E712


def test_active_span_filter_uses_declared_threshold(synthetic_db):
    series_df = compute_active_span(synthetic_db)
    # Loosen the threshold and confirm P_B now qualifies -- proves the
    # threshold is a real, respected parameter, not hardcoded.
    loose = apply_active_span_filter(series_df, min_days=180)
    assert bool(loose.set_index("product_id").loc["P_B", "kept"]) is True
    assert MIN_ACTIVE_SPAN_DAYS == 730


def test_render_report_computes_volume_retained(synthetic_db):
    series_df = apply_active_span_filter(compute_active_span(synthetic_db))
    report = render_report(series_df)
    assert "Series (product x location) kept: **1 / 3**" in report
    # only P_A's 10.0 units of the 14.0 total (5+5+2+2+0+0) are retained
    assert "10 / 14" in report
    # HOBBIES has zero total volume; must report "n/a", never a NaN string
    assert "nan" not in report.lower()


@pytest.fixture
def growing_series_db(tmp_path):
    """P_D: first nonzero sale 2016-01-01, still selling through 2020-12-31.
    Globally that's ~5 years (kept). As of an early fold's train cutoff
    (2017-06-01), it's only ~17 months (excluded) -- proving per-fold
    active span differs from the whole-history view."""
    db_path = tmp_path / "growing.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE panel AS
        SELECT * FROM (VALUES
            ('P_D', 'S1', 'FOODS', DATE '2016-01-01', 1.0),
            ('P_D', 'S1', 'FOODS', DATE '2017-06-01', 1.0),
            ('P_D', 'S1', 'FOODS', DATE '2020-12-31', 1.0)
        ) AS t(product_id, location_id, category, ds, y)
    """)
    con.close()
    return db_path


def test_active_span_as_of_date_uses_train_data_only(growing_series_db):
    global_span = compute_active_span(growing_series_db)
    assert bool(apply_active_span_filter(global_span).iloc[0]["kept"]) is True

    early_fold_span = compute_active_span(growing_series_db, as_of_date=date(2017, 6, 1))
    early_row = early_fold_span.iloc[0]
    assert early_row["dataset_end"] == pd.Timestamp("2017-06-01")
    assert early_row["active_span_days"] == (
        pd.Timestamp("2017-06-01") - pd.Timestamp("2016-01-01")
    ).days
    assert bool(apply_active_span_filter(early_fold_span).iloc[0]["kept"]) is False


def test_render_per_fold_section_differs_from_global_population(growing_series_db):
    folds = [
        Fold(fold_id=1, train_end=date(2017, 6, 1), holdout_start=date(2017, 6, 2), holdout_end=date(2017, 6, 29)),
        Fold(fold_id=2, train_end=date(2020, 12, 31), holdout_start=date(2021, 1, 1), holdout_end=date(2021, 1, 28)),
    ]
    section = render_per_fold_section(folds, growing_series_db)
    assert "| 1 | 2017-06-01 |" in section
    assert "| 2 | 2020-12-31 |" in section
    # fold 1: P_D excluded (0 kept); fold 2: P_D kept (1 kept)
    assert "| 1 | 2017-06-01 | 2017-06-02 to 2017-06-29 | 0 | 1 |" in section
    assert "| 2 | 2020-12-31 | 2021-01-01 to 2021-01-28 | 1 | 1 |" in section
