"""AutoETS + Theta batch run across all 4 folds -- task 1.6 (slow models).

Each (fold, model) pair runs as its own subprocess (engine.backtest.
run_one_baseline), so a hang or crash in one unit never loses another,
and this script's own job is just to launch them one at a time, wait,
and log wall time -- it holds no long-lived state that could interact
badly with a child process's own multiprocessing pool.

Resumable: any (fold, model) whose result parquet already exists is
skipped, so a rerun after a partial completion picks up where it left off.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_baselines import CACHE_DIR

RESULTS_DIR = CACHE_DIR / "slow_baseline_results"
MODEL_IDS = ["auto_ets", "theta"]


def run(model_ids: list[str] = MODEL_IDS, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"

    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)

    for fold in folds:
        fold_t0 = time.time()
        for model_id in model_ids:
            result_path = results_dir / f"fold{fold.fold_id}_{model_id}.parquet"
            if result_path.exists():
                _log(log_path, f"fold {fold.fold_id} {model_id}: already done, skipping")
                continue

            t0 = time.time()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "engine.backtest.run_one_baseline",
                    "--fold-id",
                    str(fold.fold_id),
                    "--model-id",
                    model_id,
                    "--out",
                    str(result_path),
                ],
                capture_output=True,
                text=True,
            )
            wall_seconds = time.time() - t0

            if proc.returncode != 0:
                _log(
                    log_path,
                    f"fold {fold.fold_id} {model_id}: FAILED after {wall_seconds:.1f}s "
                    f"(exit {proc.returncode}): {proc.stderr[-2000:]}",
                )
                raise RuntimeError(f"fold {fold.fold_id} {model_id} failed: {proc.stderr[-2000:]}")

            _log(log_path, f"fold {fold.fold_id} {model_id}: {wall_seconds:.1f}s -- {proc.stdout.strip()}")

        _log(log_path, f"fold {fold.fold_id}: total wall time {time.time() - fold_t0:.1f}s")


def _log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    run()
