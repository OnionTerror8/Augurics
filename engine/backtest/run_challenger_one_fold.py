"""Score the challenger on exactly one (fold, hyperparameter-combo) pair
and exit. Task 2.3. Same isolation contract as run_comparator_one_fold.py:
one fresh OS process per unit, atomic write (temp path + os.replace),
any exception propagates as a non-zero exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from engine.backtest.config import load_lgbm_grid
from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_challenger import score_challenger
from engine.backtest.run_comparator import get_comparator_fold_frames_cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--combo-index", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    grid = load_lgbm_grid()
    combo = grid["combos"][args.combo_index]

    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)
    fold = next(f for f in folds if f.fold_id == args.fold_id)

    train_df, holdout_df = get_comparator_fold_frames_cached(fold)
    holdout_days = (fold.holdout_end - fold.holdout_start).days + 1

    t0 = time.time()
    result = score_challenger(train_df, holdout_df, holdout_days, hyperparams=combo)
    wall_seconds = time.time() - t0

    record = {
        "fold_id": fold.fold_id,
        "model_id": "lgbm_promo_v1",
        "combo_index": args.combo_index,
        "wall_seconds": wall_seconds,
        **{k: v for k, v in result.items() if k != "hyperparams"},
        "hyperparams": json.dumps(result["hyperparams"]),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pd.DataFrame([record]).to_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    print(f"fold {fold.fold_id} combo {args.combo_index} ({combo}): {wall_seconds:.1f}s wmape={result['wmape']:.4f}", flush=True)


if __name__ == "__main__":
    main()
