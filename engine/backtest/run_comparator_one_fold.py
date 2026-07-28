"""Score the Bayesian comparator on exactly one fold and exit. Task 2.1.

Same isolation contract as run_one_baseline.py: one fresh OS process per
fold, atomic write (temp path + os.replace) so the result file exists
only on full success, and any exception -- including ADVIConvergenceError
-- propagates as a non-zero exit code rather than being swallowed.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_comparator import get_comparator_fold_frames_cached, score_comparator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-nuts-spot-check", action="store_true")
    args = parser.parse_args()

    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)
    fold = next(f for f in folds if f.fold_id == args.fold_id)

    train_df, holdout_df = get_comparator_fold_frames_cached(fold)
    holdout_days = (fold.holdout_end - fold.holdout_start).days + 1

    t0 = time.time()
    result = score_comparator(train_df, holdout_df, holdout_days, run_nuts_spot_check=args.run_nuts_spot_check)
    wall_seconds = time.time() - t0

    record = {"fold_id": fold.fold_id, "model_id": "bayes_glm_pooled", "wall_seconds": wall_seconds, **result}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pd.DataFrame([record]).to_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    print(f"fold {fold.fold_id} bayes_glm_pooled: {wall_seconds:.1f}s wmape={result['wmape']:.4f}", flush=True)


if __name__ == "__main__":
    main()
