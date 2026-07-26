"""Metric suite and gate evaluator — Protocol P1, task 1.5.

WMAPE is volume-weighted by construction (high-volume series dominate
both numerator and denominator). Gate evaluation compares a challenger's
per-fold WMAPE against a comparator's, where each fold's WMAPE is
computed over that fold's OWN eligible population (Protocol P1 / E3) --
never a population computed once globally and reused across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wmape(y_true, y_pred, weights=None) -> float:
    """Volume-weighted MAPE: sum(|y_true - y_pred|) / sum(weights).

    weights defaults to |y_true| (the standard WMAPE definition, which is
    inherently volume-weighted: high-volume series/periods dominate).
    Fails loud on NaN predictions or a zero weight sum -- never coalesces.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"y_true shape {y_true_arr.shape} != y_pred shape {y_pred_arr.shape}")
    if np.any(np.isnan(y_pred_arr)):
        raise ValueError("wmape: y_pred contains NaN -- fail loud, never coalesce")
    if np.any(np.isnan(y_true_arr)):
        raise ValueError("wmape: y_true contains NaN")

    w = np.abs(y_true_arr) if weights is None else np.asarray(weights, dtype=float)
    denom = w.sum()
    if denom == 0:
        raise ValueError("wmape: sum of weights is zero -- cannot compute a weighted metric")
    return float(np.abs(y_true_arr - y_pred_arr).sum() / denom)


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    challenger_wmape_per_fold: list[float]
    comparator_wmape_per_fold: list[float]
    improvement_per_fold: list[float]  # (comparator - challenger) / comparator, per fold
    passed: bool
    detail: str


def _fold_improvements(challenger: list[float], comparator: list[float]) -> list[float]:
    if len(challenger) != len(comparator):
        raise ValueError(
            f"challenger has {len(challenger)} folds, comparator has {len(comparator)} -- "
            "both must cover the same folds"
        )
    if not challenger:
        raise ValueError("no folds to evaluate")
    return [(c - x) / c for c, x in zip(comparator, challenger)]


def evaluate_wmape_improvement_gate(
    gate_id: str,
    challenger_wmape_per_fold: list[float],
    comparator_wmape_per_fold: list[float],
    min_improvement: float,
) -> GateResult:
    """G1/G2-style gate: challenger's mean per-fold WMAPE improvement over
    the comparator must be >= min_improvement."""
    improvements = _fold_improvements(challenger_wmape_per_fold, comparator_wmape_per_fold)
    mean_improvement = sum(improvements) / len(improvements)
    passed = mean_improvement >= min_improvement
    detail = f"mean improvement {mean_improvement:.2%} vs required {min_improvement:.2%}"
    return GateResult(
        gate_id, challenger_wmape_per_fold, comparator_wmape_per_fold, improvements, passed, detail
    )


def evaluate_fold_consistent_sign_gate(
    gate_id: str,
    challenger_wmape_per_fold: list[float],
    comparator_wmape_per_fold: list[float],
    min_wins: int,
) -> GateResult:
    """G3-style gate: challenger must win (lower WMAPE) in at least
    min_wins of the folds, AND the mean improvement must be > 0."""
    improvements = _fold_improvements(challenger_wmape_per_fold, comparator_wmape_per_fold)
    wins = [x < c for x, c in zip(challenger_wmape_per_fold, comparator_wmape_per_fold)]
    n_wins = sum(wins)
    mean_improvement = sum(improvements) / len(improvements)
    passed = mean_improvement > 0 and n_wins >= min_wins
    detail = f"{n_wins}/{len(wins)} folds won, mean improvement {mean_improvement:.2%}"
    return GateResult(
        gate_id, challenger_wmape_per_fold, comparator_wmape_per_fold, improvements, passed, detail
    )
