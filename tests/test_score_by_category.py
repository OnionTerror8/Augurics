"""Per-category scoring — Protocol P1's "also reported per category".

Tested against synthetic scratch data (not the live M5 batch, which
predates category-aware fold extraction) so the logic is verified and
ready without needing to touch anything the batch is using.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import engine.models.baselines  # noqa: F401 -- side effect: registers Naive
from engine.backtest.gates import wmape
from engine.backtest.run_baselines import score_model_by_category


def _series(series_key: str, category: str, values: list[float], start: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": series_key,
            "category": category,
            "ds": [start + timedelta(days=i) for i in range(len(values))],
            "y": values,
        }
    )


def test_score_model_by_category_matches_hand_computed_naive_wmape():
    start = date(2020, 1, 1)
    train_days = 20
    holdout_days = 3

    # FOODS: two series, last training value 10.0 and 4.0
    foods_a = _series("FOODS_A", "FOODS", [10.0] * train_days, start)
    foods_b = _series("FOODS_B", "FOODS", [4.0] * train_days, start)
    # HOBBIES: one series, last training value 2.0
    hobbies_a = _series("HOBBIES_A", "HOBBIES", [2.0] * train_days, start)

    train_df = pd.concat([foods_a, foods_b, hobbies_a], ignore_index=True)

    holdout_start = start + timedelta(days=train_days)

    def _holdout(series_key, category, actuals):
        return pd.DataFrame(
            {
                "series_key": series_key,
                "category": category,
                "ds": [holdout_start + timedelta(days=i) for i in range(holdout_days)],
                "y": actuals,
            }
        )

    holdout_df = pd.concat(
        [
            _holdout("FOODS_A", "FOODS", [12.0, 12.0, 12.0]),  # naive predicts 10.0 each day
            _holdout("FOODS_B", "FOODS", [4.0, 4.0, 4.0]),  # naive predicts 4.0 each day -- perfect
            _holdout("HOBBIES_A", "HOBBIES", [1.0, 1.0, 1.0]),  # naive predicts 2.0 each day
        ],
        ignore_index=True,
    )

    result = score_model_by_category("naive", train_df, holdout_df, holdout_days, fold_id=1)

    assert set(result["category"]) == {"FOODS", "HOBBIES"}
    assert (result["fold_id"] == 1).all()
    assert (result["model_id"] == "naive").all()

    foods_row = result[result["category"] == "FOODS"].iloc[0]
    # naive yhat: A=10 (err 2/day), B=4 (err 0/day); wmape = sum|err| / sum|actual|
    expected_foods_wmape = wmape([12.0, 12.0, 12.0, 4.0, 4.0, 4.0], [10.0, 10.0, 10.0, 4.0, 4.0, 4.0])
    assert foods_row["wmape"] == pytest.approx(expected_foods_wmape)
    assert foods_row["n_series"] == 2
    assert foods_row["volume"] == pytest.approx(12.0 * 3 + 4.0 * 3)

    hobbies_row = result[result["category"] == "HOBBIES"].iloc[0]
    expected_hobbies_wmape = wmape([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert hobbies_row["wmape"] == pytest.approx(expected_hobbies_wmape)
    assert hobbies_row["n_series"] == 1
    assert hobbies_row["volume"] == pytest.approx(3.0)


def test_score_model_by_category_requires_category_column():
    df = pd.DataFrame({"series_key": ["A"], "ds": [date(2020, 1, 1)], "y": [1.0]})
    with pytest.raises(ValueError, match="category"):
        score_model_by_category("naive", df, df, holdout_days=1, fold_id=1)
