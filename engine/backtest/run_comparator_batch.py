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

NOT launched against real fold data yet -- prepared and ready pending
review of BASELINE_TABLE.md and comparator_evidence.md.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_comparator import CACHE_DIR

RESULTS_DIR = CACHE_DIR / "results"
NUTS_SPOT_CHECK_FOLD_ID = 1


def run(results_dir: Path = RESULTS_DIR, nuts_spot_check_fold_id: int = NUTS_SPOT_CHECK_FOLD_ID) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"

    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)

    for fold in folds:
        result_path = results_dir / f"fold{fold.fold_id}_bayes_glm_pooled.parquet"
        if result_path.exists():
            _log(log_path, f"fold {fold.fold_id} bayes_glm_pooled: already done, skipping")
            continue

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
            f"{wall_seconds:.1f}s -- {proc.stdout.strip()}",
        )


def _log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    run()
