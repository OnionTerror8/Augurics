"""Rolling-origin backtest splitter — Protocol P1, task 1.5.

4 folds, 28-day holdout per fold, expanding training window, no fold's
training data reaching into or past its own holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from engine.data.ingest import DB_PATH

N_FOLDS = 4
HOLDOUT_DAYS = 28


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_end: date  # last date included in training (inclusive)
    holdout_start: date  # first date of the holdout window (inclusive)
    holdout_end: date  # last date of the holdout window (inclusive)


class LeakageError(Exception):
    """Raised when a fold's training window reaches into or past its holdout."""


def dataset_bounds(db_path: Path = DB_PATH) -> tuple[date, date]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        start, end = con.execute("SELECT MIN(ds), MAX(ds) FROM panel").fetchone()
    finally:
        con.close()
    return start, end


def make_folds(dataset_end: date, n_folds: int = N_FOLDS, holdout_days: int = HOLDOUT_DAYS) -> list[Fold]:
    """Rolling-origin, expanding window.

    Fold k's holdout is the `holdout_days` block ending
    `holdout_days * (n_folds - k)` days before `dataset_end`. Training for
    fold k is everything strictly before its holdout start -- expanding as
    k increases. No fold's holdout overlaps another's.
    """
    folds = []
    for k in range(1, n_folds + 1):
        days_before_end = holdout_days * (n_folds - k)
        holdout_end = dataset_end - timedelta(days=days_before_end)
        holdout_start = holdout_end - timedelta(days=holdout_days - 1)
        train_end = holdout_start - timedelta(days=1)
        folds.append(Fold(fold_id=k, train_end=train_end, holdout_start=holdout_start, holdout_end=holdout_end))
    assert_no_leakage(folds)
    return folds


def assert_no_leakage(folds: list[Fold]) -> None:
    """Fail loud if any fold's train window reaches into or past its holdout,
    or if two folds' holdouts overlap."""
    for fold in folds:
        if fold.train_end >= fold.holdout_start:
            raise LeakageError(
                f"Fold {fold.fold_id}: train_end {fold.train_end} >= "
                f"holdout_start {fold.holdout_start}"
            )
        if fold.holdout_start > fold.holdout_end:
            raise LeakageError(f"Fold {fold.fold_id}: holdout_start after holdout_end")
    for a, b in zip(folds, folds[1:]):
        if a.holdout_end >= b.holdout_start:
            raise LeakageError(
                f"Folds {a.fold_id} and {b.fold_id} have overlapping holdout windows"
            )


def fetch_fold_train(fold: Fold, db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM panel WHERE ds <= ?", [fold.train_end]).df()
    finally:
        con.close()


def fetch_fold_holdout(fold: Fold, db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT * FROM panel WHERE ds >= ? AND ds <= ?", [fold.holdout_start, fold.holdout_end]
        ).df()
    finally:
        con.close()
