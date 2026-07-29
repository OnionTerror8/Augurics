"""Bayesian comparator batch across all 4 folds -- task 2.1 gate-relevant run.

Same structure as run_slow_baselines.py: each fold runs as its own
subprocess (run_comparator_one_fold), resumable (skips any fold whose
result file already exists), logs wall time as it goes.

The bounded NUTS spot-check (see engine.models.comparators module
docstring for why it's bounded) runs ONCE, on the fold named by
NUTS_SPOT_CHECK_FOLD_ID -- not per fold. It exists to validate ADVI's
approximation quality on real full-scale data, which is a property of
the model/data distribution, not something that needs re-checking on
every fold.

Real fold-1 timing on Kaggle (10.64h wall -- see comparator_evidence.md
§8) is ~3.5-4x the pre-refactor local extrapolation, leaving no margin
under Kaggle's 12h session ceiling for more than one fold per session.
The first real launch confirmed the failure mode directly: after fold 1
finished, the loop moved on to fold 2, which started (cached its fold
frames) and was then ceiling-cancelled mid-fit with ZERO partial
credit -- atomic writes mean an in-progress fit banks nothing. Two
mitigations follow from that:
  1. `--fold-id` restricts a single invocation to exactly one fold, so
     each Kaggle notebook commit runs one fold and exits -- the caller
     (the notebook) controls the fold-per-commit split, not this loop.
  2. `check_session_budget` refuses loud, before starting a fold's
     fit, if the projected time would not fit in the session's
     remaining budget -- cheap insurance against a future multi-fold
     attempt repeating fold 2's fate.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_comparator import CACHE_DIR

RESULTS_DIR = CACHE_DIR / "results"
NUTS_SPOT_CHECK_FOLD_ID = 1

# Kaggle Notebooks' CPU session ceiling (KAGGLE_PROVISIONING.md) and a
# conservative safety margin before it.
SESSION_LIMIT_SECONDS = 12 * 3600
SAFETY_MARGIN_SECONDS = 30 * 60

# Fold 1's REAL measured wall time on Kaggle (38,308.3s = 10.64h --
# comparator_evidence.md §8), used as the conservative per-fold
# projection for the budget guard. Folds without the bounded NUTS
# spot-check (all but NUTS_SPOT_CHECK_FOLD_ID) should be somewhat
# faster, but NUTS is a small fraction of that total (a ~99k-row
# single-chain sampler against a 48M-row ADVI fit) -- using the full
# fold-1 figure for every fold is deliberately conservative, not an
# average or a best case.
MEASURED_FOLD1_WALL_SECONDS = 10.6 * 3600


class SessionBudgetExceededError(Exception):
    """Raised when starting a fold's fit is projected to exceed the
    remaining session time minus SAFETY_MARGIN_SECONDS -- refuses loud
    with the numbers instead of starting a multi-hour fit that gets
    ceiling-cancelled mid-fit with zero partial credit (fold 2's real
    fate in the first Kaggle launch -- see KNOWN_ISSUES.md)."""


def check_session_budget(
    session_started_at: float,
    projected_fit_seconds: float = MEASURED_FOLD1_WALL_SECONDS,
    session_limit_seconds: float = SESSION_LIMIT_SECONDS,
    safety_margin_seconds: float = SAFETY_MARGIN_SECONDS,
    now: float | None = None,
) -> float:
    """Returns remaining session seconds if the budget clears; raises
    SessionBudgetExceededError otherwise. `session_started_at` is a
    unix timestamp -- the notebook's own Cell 1 is the only thing that
    knows when the session actually began (git clone + pip install +
    ingest + make test all consume real session time before any fold
    starts), so it must be passed in, never assumed to be "now"."""
    now = now if now is not None else time.time()
    elapsed = now - session_started_at
    remaining = session_limit_seconds - elapsed
    if projected_fit_seconds > remaining - safety_margin_seconds:
        raise SessionBudgetExceededError(
            f"Starting this fold's fit is projected to take {projected_fit_seconds / 3600:.2f}h, but only "
            f"{remaining / 3600:.2f}h remain in this ~{session_limit_seconds / 3600:.0f}h session "
            f"({elapsed / 3600:.2f}h already elapsed) minus a {safety_margin_seconds / 60:.0f}-min safety "
            f"margin. Refusing to start -- this fold would very likely be ceiling-cancelled mid-fit with "
            f"zero partial credit, exactly as fold 2 was in the first real Kaggle run."
        )
    return remaining


def run(
    results_dir: Path = RESULTS_DIR,
    nuts_spot_check_fold_id: int = NUTS_SPOT_CHECK_FOLD_ID,
    fold_id: int | None = None,
    session_started_at: float | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"

    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)
    if fold_id is not None:
        folds = [f for f in folds if f.fold_id == fold_id]
        if not folds:
            raise ValueError(f"run_comparator_batch: no fold with fold_id={fold_id}")

    for fold in folds:
        result_path = results_dir / f"fold{fold.fold_id}_bayes_glm_pooled.parquet"
        if result_path.exists():
            _log(log_path, f"fold {fold.fold_id} bayes_glm_pooled: already done, skipping ({result_path})")
            continue

        if session_started_at is not None:
            check_session_budget(session_started_at)

        run_nuts = fold.fold_id == nuts_spot_check_fold_id
        cmd = [
            sys.executable,
            "-m",
            "engine.backtest.run_comparator_one_fold",
            "--fold-id",
            str(fold.fold_id),
            "--out",
            str(result_path),
        ]
        if run_nuts:
            cmd.append("--run-nuts-spot-check")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall_seconds = time.time() - t0

        if proc.returncode != 0:
            _log(
                log_path,
                f"fold {fold.fold_id} bayes_glm_pooled: FAILED after {wall_seconds:.1f}s "
                f"(exit {proc.returncode}): {proc.stderr[-2000:]}",
            )
            raise RuntimeError(f"fold {fold.fold_id} bayes_glm_pooled failed: {proc.stderr[-2000:]}")

        _log(
            log_path,
            f"fold {fold.fold_id} bayes_glm_pooled (nuts_spot_check={run_nuts}): "
            f"{wall_seconds:.1f}s -- {proc.stdout.strip()} -- result: {result_path}",
        )


def _log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-id", type=int, default=None,
        help="Run exactly one fold (required for session-constrained hosts like Kaggle, "
             "where more than one fold does not reliably fit in a 12h session). "
             "Omit to run all not-yet-done folds sequentially.",
    )
    parser.add_argument(
        "--session-started-at", type=float, default=None,
        help="Unix timestamp when the current notebook session began (capture this at "
             "Cell 1, e.g. `SESSION_STARTED_AT = time.time()`). Enables the session-budget "
             "guard. Falls back to the SESSION_STARTED_AT_UNIX env var if omitted; if "
             "neither is set, the guard is skipped (e.g. for the VM route, which has no "
             "session ceiling).",
    )
    args = parser.parse_args()

    session_started_at = args.session_started_at
    if session_started_at is None and os.environ.get("SESSION_STARTED_AT_UNIX"):
        session_started_at = float(os.environ["SESSION_STARTED_AT_UNIX"])

    run(fold_id=args.fold_id, session_started_at=session_started_at)


if __name__ == "__main__":
    main()
