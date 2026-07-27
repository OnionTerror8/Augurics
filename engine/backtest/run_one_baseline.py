"""Score exactly one (fold, model) pair and exit. Task 1.6 (slow models).

Run as a fresh, isolated OS process per unit of work (invoked by
run_slow_baselines.py via subprocess), so a hang or crash in one
(fold, model) combination can never affect any other, and progress is
observable one discrete result file at a time rather than as internal
state in one long-lived process.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

import engine.models.baselines  # noqa: F401 -- side effect: registers Rung 1 baselines
from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_baselines import get_fold_frames_cached, score_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)
    fold = next(f for f in folds if f.fold_id == args.fold_id)

    train_df, holdout_df = get_fold_frames_cached(fold)
    holdout_days = (fold.holdout_end - fold.holdout_start).days + 1

    t0 = time.time()
    result = score_model(args.model_id, train_df, holdout_df, holdout_days)
    wall_seconds = time.time() - t0

    record = {"fold_id": fold.fold_id, "model_id": args.model_id, "wall_seconds": wall_seconds, **result}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: the result file must exist ONLY on full success, so the
    # orchestrator's resume-skip check (`if result_path.exists(): skip`) is
    # correct by construction -- a crash or kill mid-write must never leave
    # a partial/corrupt file at out_path.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pd.DataFrame([record]).to_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    print(f"fold {fold.fold_id} {args.model_id}: {wall_seconds:.1f}s wmape={result['wmape']:.4f}", flush=True)


if __name__ == "__main__":
    main()
