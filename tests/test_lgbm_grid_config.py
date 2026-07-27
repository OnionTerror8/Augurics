"""LightGBM grid pre-registration — task 2.3 config, committed before any
challenger result exists (non-negotiable rule 7)."""
from __future__ import annotations

from itertools import product

from engine.backtest.config import load_lgbm_grid


def test_grid_has_at_most_12_combos():
    grid = load_lgbm_grid()
    assert len(grid["combos"]) <= grid["max_combos"] <= 12


def test_enumerated_combos_match_the_declared_search_space_exactly():
    """The explicit `combos` list must be exactly the cartesian product of
    `search_space` -- proves the two can't silently drift apart."""
    grid = load_lgbm_grid()
    space = grid["search_space"]
    keys = list(space.keys())

    expected = {tuple(sorted(dict(zip(keys, values)).items())) for values in product(*space.values())}
    actual = {tuple(sorted(combo.items())) for combo in grid["combos"]}

    assert actual == expected


def test_no_duplicate_combos():
    grid = load_lgbm_grid()
    seen = {tuple(sorted(combo.items())) for combo in grid["combos"]}
    assert len(seen) == len(grid["combos"])
