"""Coverage diagnostic tests against a small synthetic panel — no M5 data required."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from engine.backtest.coverage import (
    compute_monthly_coverage,
    compute_monthly_coverage_by_category,
    first_month_reaching,
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
