"""Rung 2: lgbm via mlforecast + feature library. Task 2.3.

Training is gated on `check_challenger_fit_memory` first (kickoff §2 /
non-negotiable rule on fail-loud memory guards, same discipline as
`engine.models.comparators._check_fit_memory`) -- estimated empirically
from the feature library's real output on fold 1, not a formula guess.
See KNOWN_ISSUES.md: at fold-1 scale (48.1M rows) this does not clear
this development laptop's available RAM even after the same
category-dtype / int8-flag / float32 optimisations that fixed the
comparator, so training is deferred alongside the comparator batch
until compute (the same VM in PROVISIONING.md) is available. The
estimator below stays in place so any future run -- on the VM or on
this laptop once memory frees up -- gets the same fail-loud refusal
the comparator has, instead of an OOM discovered mid-grid-search.
"""
from __future__ import annotations

import psutil

# Empirically calibrated (engine/models/ml.py's feasibility check, task
# 2.3 step 1): building the full feature set (engine.features.library.
# build_feature_set) on fold 1's real 48,110,016-row train frame, with
# series_key/category cast to `category` dtype, all flag columns
# downcast to int8/bool, and all engineered float columns to float32,
# measures 3,033,770,424 bytes via df.memory_usage(deep=True) --
# 63.06 bytes/row. This constant is that number, not a theoretical one.
OPTIMIZED_FEATURE_BYTES_PER_ROW = 63.06

# mlforecast/lightgbm's own training buffers on top of the feature
# matrix: gradient + hessian arrays (2 x 4 bytes/row, float32) plus
# mlforecast's internal copy during its own preprocessing step. Less
# than the comparator's 3x (no PyMC graph compilation), but real:
# calibrate conservatively at 1.5x total (matrix + lgbm buffers)
# rather than assume mlforecast makes zero additional copies.
LGBM_GRADIENT_HESSIAN_BYTES_PER_ROW = 8.0
LGBM_OVERHEAD_MULTIPLIER = 1.5

MAX_CHALLENGER_FIT_MEMORY_FRACTION = 0.5


class InsufficientChallengerMemoryError(Exception):
    """Raised by the pre-fit memory estimator when the challenger's
    estimated feature-matrix-plus-training footprint would exceed a
    safe fraction of currently available RAM. Mirrors
    engine.models.comparators.InsufficientMemoryError -- a refusal
    with numbers, not an OOM discovered mid-grid-search."""


def estimate_challenger_fit_memory_bytes(n_rows: int) -> int:
    bytes_per_row = OPTIMIZED_FEATURE_BYTES_PER_ROW + LGBM_GRADIENT_HESSIAN_BYTES_PER_ROW
    return int(n_rows * bytes_per_row * LGBM_OVERHEAD_MULTIPLIER)


def check_challenger_fit_memory(n_rows: int, max_fraction: float = MAX_CHALLENGER_FIT_MEMORY_FRACTION) -> dict:
    estimated = estimate_challenger_fit_memory_bytes(n_rows)
    available = psutil.virtual_memory().available
    budget = available * max_fraction
    if estimated > budget:
        raise InsufficientChallengerMemoryError(
            f"Estimated challenger fit memory ({estimated / 1e9:.2f} GB) exceeds {max_fraction:.0%} "
            f"of currently available RAM ({available / 1e9:.2f} GB, budget {budget / 1e9:.2f} GB) "
            f"for n_rows={n_rows:,}. Refusing to start -- this would very likely be OOM-killed "
            f"partway through a grid-search fit, wasting the fit time instead of failing loud immediately."
        )
    return {"estimated_bytes": estimated, "available_bytes": available, "budget_bytes": budget}
