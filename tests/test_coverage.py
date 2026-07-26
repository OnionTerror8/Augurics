"""Coverage diagnostic tests against small synthetic panels — no M5 data required."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from engine.backtest.coverage import (
    WITHIN_SPAN_LIFT_THRESHOLD,
    compute_calendar_field_completeness,
    compute_monthly_coverage,
    compute_monthly_coverage_by_category,
    compute_within_span_monthly_price_coverage,
    compute_within_span_price_coverage,
    first_month_reaching,
    render_addendum,
)


@pytest.fixture
def synthetic_db(tmp_path):
    db_path = tmp_path / "synthetic.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE panel AS
        SELECT * FROM (VALUES
            (DATE '2020-01-05', 'FOODS', 3.0, NULL, NULL, NULL),
            (DATE '2020-01-06', 'FOODS', 1.0, 2.5, NULL, NULL),
            (DATE '2020-02-05', 'FOODS', 2.0, 2.5, 'SuperBowl', NULL),
            (DATE '2020-02-06', 'FOODS', 4.0, 2.5, NULL, NULL),
            (DATE '2020-01-05', 'HOBBIES', 0.0, NULL, NULL, NULL),
            (DATE '2020-02-05', 'HOBBIES', 5.0, 9.0, NULL, NULL)
        ) AS t(ds, category, y, sell_price, event_name_1, event_name_2)
    """)
    con.close()
    return db_path


def test_compute_monthly_coverage_overall(synthetic_db):
    overall = compute_monthly_coverage(synthetic_db)
    assert list(overall["year_month"]) == ["2020-01", "2020-02"]

    jan = overall[overall["year_month"] == "2020-01"].iloc[0]
    assert jan["n_rows"] == 3
    assert jan["price_coverage"] == pytest.approx(1 / 3)
    assert jan["event_coverage"] == pytest.approx(0.0)
    assert jan["mean_sales"] == pytest.approx((3.0 + 1.0 + 0.0) / 3)

    feb = overall[overall["year_month"] == "2020-02"].iloc[0]
    assert feb["n_rows"] == 3
    assert feb["price_coverage"] == pytest.approx(1.0)
    assert feb["event_coverage"] == pytest.approx(1 / 3)


def test_compute_monthly_coverage_by_category(synthetic_db):
    by_category = compute_monthly_coverage_by_category(synthetic_db)
    foods_jan = by_category[(by_category["category"] == "FOODS") & (by_category["year_month"] == "2020-01")].iloc[0]
    assert foods_jan["n_rows"] == 2
    assert foods_jan["price_coverage"] == pytest.approx(0.5)


def test_first_month_reaching_requires_sustained_threshold():
    monthly = pd.DataFrame(
        {
            "year_month": ["2020-01", "2020-02", "2020-03", "2020-04"],
            "price_coverage": [0.95, 0.40, 0.92, 0.99],
        }
    )
    # 2020-01 clears the bar but 2020-02 doesn't hold it, so the crossing
    # point must be the first month from which every later month also clears it.
    assert first_month_reaching(monthly, "price_coverage", 0.90) == "2020-03"


def test_first_month_reaching_returns_none_if_never_sustained():
    monthly = pd.DataFrame(
        {"year_month": ["2020-01", "2020-02"], "price_coverage": [0.5, 0.99]},
    )
    monthly.loc[len(monthly)] = ["2020-03", 0.4]
    assert first_month_reaching(monthly, "price_coverage", 0.90) is None


def _make_full_panel_db(tmp_path, rows: str, name: str = "full.duckdb"):
    db_path = tmp_path / name
    con = duckdb.connect(str(db_path))
    con.execute(f"""
        CREATE TABLE panel AS
        SELECT * FROM (VALUES {rows}) AS t(
            product_id, location_id, category, ds, y, sell_price,
            event_name_1, event_type_1, event_name_2, event_type_2, snap
        )
    """)
    con.close()
    return db_path


@pytest.fixture
def structural_gap_db(tmp_path):
    """P1's price is only missing BEFORE its first nonzero sale -- structural,
    should yield 100% within-span coverage."""
    rows = """
        ('P1', 'S1', 'FOODS', DATE '2020-01-01', 0.0, NULL, NULL, NULL, NULL, NULL, false),
        ('P1', 'S1', 'FOODS', DATE '2020-01-02', 0.0, NULL, NULL, NULL, NULL, NULL, true),
        ('P1', 'S1', 'FOODS', DATE '2020-01-03', 2.0, 5.0, 'NewYear', 'National', NULL, NULL, false),
        ('P1', 'S1', 'FOODS', DATE '2020-01-04', 1.0, 5.0, NULL, NULL, NULL, NULL, false)
    """
    return _make_full_panel_db(tmp_path, rows, "structural.duckdb")


@pytest.fixture
def genuine_gap_db(tmp_path):
    """P2's price is missing even AFTER its first nonzero sale -- a genuine
    within-span gap that should NOT be lifted."""
    rows = """
        ('P2', 'S1', 'FOODS', DATE '2020-01-01', 3.0, NULL, NULL, NULL, NULL, NULL, false),
        ('P2', 'S1', 'FOODS', DATE '2020-01-02', 1.0, NULL, NULL, NULL, NULL, NULL, false),
        ('P2', 'S1', 'FOODS', DATE '2020-01-03', 2.0, 5.0, NULL, NULL, NULL, NULL, false)
    """
    return _make_full_panel_db(tmp_path, rows, "genuine_gap.duckdb")


def test_within_span_coverage_is_full_for_structural_gap(structural_gap_db):
    result = compute_within_span_price_coverage(structural_gap_db)
    # rows on/after 2020-01-03 (first nonzero sale): both have sell_price set.
    assert result["n_rows"] == 2
    assert result["price_coverage"] == pytest.approx(1.0)


def test_within_span_coverage_reflects_genuine_gap(genuine_gap_db):
    result = compute_within_span_price_coverage(genuine_gap_db)
    # rows on/after 2020-01-01 (first nonzero sale): only 1 of 3 has a price.
    assert result["n_rows"] == 3
    assert result["price_coverage"] == pytest.approx(1 / 3)


def test_addendum_lifts_bound_when_within_span_coverage_is_high(structural_gap_db):
    text = render_addendum(structural_gap_db)
    assert "LIFTED" in text
    assert f"{WITHIN_SPAN_LIFT_THRESHOLD:.0%}" in text


def test_addendum_retains_bound_when_genuine_gap_remains(genuine_gap_db):
    text = render_addendum(genuine_gap_db)
    assert "Decision retained" in text
    assert "LIFTED" not in text


def test_calendar_field_completeness_detects_inconsistent_event_pairing(tmp_path):
    rows = """
        ('P1', 'S1', 'FOODS', DATE '2020-01-01', 1.0, 5.0, 'Xmas', NULL, NULL, NULL, true),
        ('P1', 'S1', 'FOODS', DATE '2020-01-02', 1.0, 5.0, NULL, NULL, NULL, NULL, false)
    """
    db_path = _make_full_panel_db(tmp_path, rows, "broken_events.duckdb")
    result = compute_calendar_field_completeness(db_path)
    assert result["n_rows"] == 2
    assert result["snap_coverage"] == pytest.approx(1.0)
    # row 1 has event_name_1 set but event_type_1 NULL -- inconsistent pairing.
    assert result["event1_consistency"] == pytest.approx(0.5)


def test_within_span_monthly_price_coverage_shape(structural_gap_db):
    monthly = compute_within_span_monthly_price_coverage(structural_gap_db)
    assert list(monthly["year_month"]) == ["2020-01"]
    assert monthly.iloc[0]["price_coverage"] == pytest.approx(1.0)
