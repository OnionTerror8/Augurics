"""Rolling-origin splitter tests — task 1.5. Includes the leakage test."""
from __future__ import annotations

from datetime import date

import duckdb
import pytest

from engine.backtest.protocol import (
    HOLDOUT_DAYS,
    N_FOLDS,
    LeakageError,
    assert_no_leakage,
    dataset_bounds,
    fetch_fold_holdout,
    fetch_fold_train,
    make_folds,
)


def test_make_folds_produces_expected_count_and_holdout_length():
    folds = make_folds(date(2020, 12, 31))
    assert len(folds) == N_FOLDS
    for fold in folds:
        assert (fold.holdout_end - fold.holdout_start).days + 1 == HOLDOUT_DAYS


def test_make_folds_last_fold_ends_at_dataset_end():
    dataset_end = date(2020, 12, 31)
    folds = make_folds(dataset_end)
    assert folds[-1].holdout_end == dataset_end


def test_make_folds_train_window_expands():
    folds = make_folds(date(2020, 12, 31))
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_make_folds_holdouts_do_not_overlap():
    folds = make_folds(date(2020, 12, 31))
    for a, b in zip(folds, folds[1:]):
        assert a.holdout_end < b.holdout_start


def test_no_leakage_in_generated_folds():
    """The leakage test: no fold's training window reaches into or past its holdout."""
    folds = make_folds(date(2020, 12, 31))
    assert_no_leakage(folds)  # must not raise
    for fold in folds:
        assert fold.train_end < fold.holdout_start


def test_assert_no_leakage_raises_on_bad_fold():
    from engine.backtest.protocol import Fold

    bad_folds = [Fold(fold_id=1, train_end=date(2020, 1, 10), holdout_start=date(2020, 1, 5), holdout_end=date(2020, 1, 12))]
    with pytest.raises(LeakageError):
        assert_no_leakage(bad_folds)


def test_assert_no_leakage_raises_on_overlapping_holdouts():
    from engine.backtest.protocol import Fold

    folds = [
        Fold(fold_id=1, train_end=date(2020, 1, 1), holdout_start=date(2020, 1, 2), holdout_end=date(2020, 1, 10)),
        Fold(fold_id=2, train_end=date(2020, 1, 5), holdout_start=date(2020, 1, 6), holdout_end=date(2020, 1, 15)),
    ]
    with pytest.raises(LeakageError):
        assert_no_leakage(folds)


@pytest.fixture
def synthetic_db(tmp_path):
    db_path = tmp_path / "synthetic.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE panel AS
        SELECT * FROM (VALUES
            ('P1', 'S1', DATE '2020-01-01', 1.0),
            ('P1', 'S1', DATE '2020-01-15', 2.0),
            ('P1', 'S1', DATE '2020-02-01', 3.0),
            ('P1', 'S1', DATE '2020-02-15', 4.0)
        ) AS t(product_id, location_id, ds, y)
    """)
    con.close()
    return db_path


def test_dataset_bounds(synthetic_db):
    start, end = dataset_bounds(synthetic_db)
    assert start == date(2020, 1, 1)
    assert end == date(2020, 2, 15)


def test_fetch_fold_train_and_holdout_partition_without_overlap(synthetic_db):
    fold = make_folds(date(2020, 2, 15), n_folds=1, holdout_days=14)[0]
    train = fetch_fold_train(fold, synthetic_db)
    holdout = fetch_fold_holdout(fold, synthetic_db)

    assert train["ds"].max().date() <= fold.train_end
    assert holdout["ds"].min().date() >= fold.holdout_start
    assert holdout["ds"].max().date() <= fold.holdout_end
    # no row appears in both: fail-loud proof that fit never sees holdout dates
    assert set(train["ds"]).isdisjoint(set(holdout["ds"]))
