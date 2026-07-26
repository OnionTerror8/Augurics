"""Metric suite and gate evaluator tests — task 1.5.

Demonstrates the volume-weighted-per-fold-population usage required by
Protocol P1 / E3: each fold's WMAPE is computed over that fold's OWN
eligible series, not a population shared across folds.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.backtest.gates import (
    GateResult,
    evaluate_fold_consistent_sign_gate,
    evaluate_wmape_improvement_gate,
    wmape,
)


def test_wmape_basic_value():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    # |10-12| + |20-18| + |30-33| = 2+2+3 = 7; sum(|y_true|) = 60
    assert wmape(y_true, y_pred) == pytest.approx(7 / 60)


def test_wmape_is_volume_weighted_not_series_averaged():
    # A high-volume series with a small relative error should dominate a
    # low-volume series with a huge relative error, because WMAPE weights
    # by volume rather than averaging per-series MAPE.
    y_true = [1000, 1]
    y_pred = [1010, 100]  # high-vol series off by 1%, low-vol series off by 9900%
    result = wmape(y_true, y_pred)
    naive_mape_average = (abs(1010 - 1000) / 1000 + abs(100 - 1) / 1) / 2
    assert result < naive_mape_average / 10


def test_wmape_raises_on_nan_prediction():
    with pytest.raises(ValueError, match="NaN"):
        wmape([1.0, 2.0], [1.0, float("nan")])


def test_wmape_raises_on_zero_weight_sum():
    with pytest.raises(ValueError, match="zero"):
        wmape([0.0, 0.0], [0.0, 0.0])


def test_wmape_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        wmape([1.0, 2.0, 3.0], [1.0, 2.0])


# --- Gate evaluation, using each fold's own eligible-population WMAPE ---


def _per_fold_wmape(fold_populations, fold_predictions, model_id):
    """Compute WMAPE per fold restricted to that fold's own eligible series
    -- the point being that fold 1 and fold 2 use DIFFERENT populations."""
    results = []
    for eligible_series, predictions in zip(fold_populations, fold_predictions):
        y_true = np.array([predictions[model_id][s]["y_true"] for s in eligible_series])
        y_pred = np.array([predictions[model_id][s]["y_pred"] for s in eligible_series])
        results.append(wmape(y_true, y_pred))
    return results


@pytest.fixture
def two_fold_scenario():
    # Fold 1's eligible population is {"A", "B"}; fold 2's is {"A", "B", "C"}
    # -- different populations per fold, as Protocol P1 / E3 requires.
    fold_populations = [["A", "B"], ["A", "B", "C"]]
    predictions = {
        "challenger": {
            "A": {"y_true": 100.0, "y_pred": 105.0},
            "B": {"y_true": 50.0, "y_pred": 48.0},
            "C": {"y_true": 10.0, "y_pred": 9.0},
        },
        "comparator": {
            "A": {"y_true": 100.0, "y_pred": 130.0},
            "B": {"y_true": 50.0, "y_pred": 40.0},
            "C": {"y_true": 10.0, "y_pred": 15.0},
        },
    }
    fold_predictions = [predictions, predictions]
    return fold_populations, fold_predictions


def test_gate_uses_each_folds_own_population(two_fold_scenario):
    fold_populations, fold_predictions = two_fold_scenario
    challenger_wmape = _per_fold_wmape(fold_populations, fold_predictions, "challenger")
    comparator_wmape = _per_fold_wmape(fold_populations, fold_predictions, "comparator")

    # fold 1 excludes "C"; fold 2 includes it -- the two folds' WMAPEs must
    # differ because they're computed over different populations.
    assert challenger_wmape[0] != challenger_wmape[1]
    assert comparator_wmape[0] != comparator_wmape[1]

    result = evaluate_wmape_improvement_gate("G_TEST", challenger_wmape, comparator_wmape, min_improvement=0.15)
    assert isinstance(result, GateResult)
    assert result.passed is True


def test_wmape_improvement_gate_fails_below_threshold():
    result = evaluate_wmape_improvement_gate("G1", [0.20, 0.21], [0.22, 0.23], min_improvement=0.15)
    assert result.passed is False
    assert "required 15.00%" in result.detail


def test_wmape_improvement_gate_passes_at_threshold():
    challenger = [0.17, 0.17]
    comparator = [0.20, 0.20]  # 15% improvement exactly
    result = evaluate_wmape_improvement_gate("G1", challenger, comparator, min_improvement=0.15)
    assert result.passed is True


def test_fold_consistent_sign_gate_requires_min_wins():
    # challenger wins folds 1 and 2, loses fold 3 -> only 2/3 wins
    challenger = [0.10, 0.10, 0.30]
    comparator = [0.20, 0.20, 0.20]
    result = evaluate_fold_consistent_sign_gate("G3", challenger, comparator, min_wins=3)
    assert result.passed is False
    assert "2/3 folds won" in result.detail


def test_fold_consistent_sign_gate_passes_with_enough_wins():
    challenger = [0.10, 0.15, 0.19]
    comparator = [0.20, 0.20, 0.20]
    result = evaluate_fold_consistent_sign_gate("G3", challenger, comparator, min_wins=3)
    assert result.passed is True


def test_gates_raise_on_mismatched_fold_counts():
    with pytest.raises(ValueError, match="folds"):
        evaluate_wmape_improvement_gate("G1", [0.1, 0.2], [0.1], min_improvement=0.1)
