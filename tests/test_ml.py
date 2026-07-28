"""Challenger (Rung 2, lgbm) pre-fit memory guard -- task 2.3 step 1.
Training itself is deferred (see KNOWN_ISSUES.md); this only covers the
guard that decided that, mirroring tests/test_comparators.py's coverage
of the analogous PyMC guard."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.models.ml import (
    InsufficientChallengerMemoryError,
    check_challenger_fit_memory,
    estimate_challenger_fit_memory_bytes,
)


class _FakeVirtualMemory:
    def __init__(self, available: int) -> None:
        self.available = available


def test_estimate_scales_linearly_with_row_count():
    est_1x = estimate_challenger_fit_memory_bytes(1_000_000)
    est_2x = estimate_challenger_fit_memory_bytes(2_000_000)
    assert est_2x == pytest.approx(est_1x * 2, rel=1e-9)


def test_check_passes_when_plenty_of_ram_available():
    with patch("psutil.virtual_memory", return_value=_FakeVirtualMemory(available=100_000_000_000)):
        result = check_challenger_fit_memory(n_rows=1_000_000)
    assert result["estimated_bytes"] < result["budget_bytes"]


def test_check_raises_with_numbers_when_ram_insufficient():
    with patch("psutil.virtual_memory", return_value=_FakeVirtualMemory(available=1_000_000_000)):
        with pytest.raises(InsufficientChallengerMemoryError, match="Refusing to start"):
            check_challenger_fit_memory(n_rows=48_110_016)


def test_check_reflects_real_fold1_scale_against_low_ram():
    """The actual finding that deferred challenger training: fold 1's
    real row count (48,110,016) against a modest RAM budget fails,
    consistent with this laptop's measured available RAM this session."""
    with patch("psutil.virtual_memory", return_value=_FakeVirtualMemory(available=5_000_000_000)):
        with pytest.raises(InsufficientChallengerMemoryError):
            check_challenger_fit_memory(n_rows=48_110_016)
