"""Challenger grid-search batch across all 4 folds x the pre-registered
12-combo grid (config/lgbm_grid.yaml) -- task 2.3. Same structure as
run_comparator_batch.py: each (fold, combo) unit runs as its own
subprocess (run_challenger_one_fold), resumable (skips any unit whose
result file already exists), logs wall time as it goes.

"Best Rung 2 model" (Protocol P1's G1/G2/G3 language) selection
methodology, stated here since the protocol doesn't specify one:
the combo with the LOWEST MEAN WMAPE across all 4 folds wins -- the
same aggregation convention BASELINE_TABLE.md and GATE_REPORT.md
already use for comparing models. That combo's per-fold WMAPEs become
`lgbm_promo_v1`'s result files in RESULTS_DIR, in the exact schema
gate_report.collect_results()/load_and_validate_result() expect, so
GATE_REPORT.md's G1/G2/G3 evaluation picks them up with no special
casing.

`--fold-id` restricts one invocation to a single fold's 12 combos --
the session-constrained-host split (mirrors run_comparator_batch.py's
`--fold-id`, added after fold 2's real Kaggle wall time turned out
~3.5-4x the pre-launch estimate). When given, this run does NOT
auto-select a winning combo (results for the other folds may not
exist yet) -- call `--finalize` once all 4 folds' combos are present,
as a separate, explicit step.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from engine.backtest.config import load_lgbm_grid
from engine.backtest.protocol import dataset_bounds, make_folds
from engine.backtest.run_baselines import load_and_validate_result

GRID_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "fold_cache" / "challenger_grid"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "fold_cache" / "slow_baseline_results"  # same dir GATE_REPORT.md reads


def run(
    grid_results_dir: Path = GRID_RESULTS_DIR,
    results_dir: Path = RESULTS_DIR,
    fold_id: int | None = None,
) -> None:
    grid_results_dir.mkdir(parents=True, exist_ok=True)
    log_path = grid_results_dir / "run_log.txt"

    grid = load_lgbm_grid()
    _, dataset_end = dataset_bounds()
    folds = make_folds(dataset_end)
    if fold_id is not None:
        folds = [f for f in folds if f.fold_id == fold_id]
        if not folds:
            raise ValueError(f"run_challenger_batch: no fold with fold_id={fold_id}")

    for fold in folds:
        for combo_index in range(len(grid["combos"])):
            result_path = grid_results_dir / f"fold{fold.fold_id}_combo{combo_index}.parquet"
            if result_path.exists():
                _log(log_path, f"fold {fold.fold_id} combo {combo_index}: already done, skipping")
                continue

            cmd = [
                sys.executable, "-m", "engine.backtest.run_challenger_one_fold",
                "--fold-id", str(fold.fold_id), "--combo-index", str(combo_index), "--out", str(result_path),
            ]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            wall_seconds = time.time() - t0

            if proc.returncode != 0:
                _log(log_path, f"fold {fold.fold_id} combo {combo_index}: FAILED after {wall_seconds:.1f}s (exit {proc.returncode}): {proc.stderr[-2000:]}")
                raise RuntimeError(f"fold {fold.fold_id} combo {combo_index} failed: {proc.stderr[-2000:]}")

            _log(log_path, f"fold {fold.fold_id} combo {combo_index}: {wall_seconds:.1f}s -- {proc.stdout.strip()}")

    if fold_id is None:
        # Full, unrestricted run: all 4 folds were just processed in this
        # one invocation, so selection can run immediately.
        _, all_fold_ids_end = dataset_bounds()
        all_fold_ids = [f.fold_id for f in make_folds(all_fold_ids_end)]
        select_best_combo_and_write_results(grid_results_dir, results_dir, all_fold_ids)
    else:
        _log(log_path, f"fold {fold_id} complete -- run with --finalize once all folds are done to select the winning combo")


def select_best_combo_and_write_results(grid_results_dir: Path, results_dir: Path, fold_ids: list[int]) -> dict:
    """Lowest-mean-WMAPE combo across all folds wins (see module
    docstring). Writes that combo's per-fold results as
    fold{N}_lgbm_promo_v1.parquet in `results_dir`, schema-matching
    validated result parquets exactly (atomic write, same as every
    other orchestrator in this codebase)."""
    grid = load_lgbm_grid()
    all_records = []
    for path in sorted(grid_results_dir.glob("fold*_combo*.parquet")):
        record = load_and_validate_result(path)
        record["hyperparams"] = json.loads(record["hyperparams"]) if isinstance(record["hyperparams"], str) else record["hyperparams"]
        all_records.append(record)

    by_combo: dict[int, list[dict]] = {}
    for record in all_records:
        by_combo.setdefault(record["combo_index"], []).append(record)

    complete_combos = {ci: recs for ci, recs in by_combo.items() if {r["fold_id"] for r in recs} == set(fold_ids)}
    if not complete_combos:
        raise RuntimeError("select_best_combo_and_write_results: no combo has results for all folds -- batch incomplete")

    mean_wmape_by_combo = {ci: sum(r["wmape"] for r in recs) / len(recs) for ci, recs in complete_combos.items()}
    best_combo_index = min(mean_wmape_by_combo, key=mean_wmape_by_combo.get)
    best_records = {r["fold_id"]: r for r in complete_combos[best_combo_index]}

    results_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, record in best_records.items():
        out_path = results_dir / f"fold{fold_id}_lgbm_promo_v1.parquet"
        final_record = {
            "fold_id": record["fold_id"],
            "model_id": "lgbm_promo_v1",
            "wall_seconds": record["wall_seconds"],
            "wmape": record["wmape"],
            "n_series": record["n_series"],
            "n_rows": record["n_rows"],
        }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        pd.DataFrame([final_record]).to_parquet(tmp_path)
        os.replace(tmp_path, out_path)

    return {
        "best_combo_index": best_combo_index,
        "best_combo": grid["combos"][best_combo_index],
        "mean_wmape": mean_wmape_by_combo[best_combo_index],
        "all_combo_mean_wmape": mean_wmape_by_combo,
    }


def _log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-id", type=int, default=None,
        help="Run exactly one fold's 12 combos (for session-constrained hosts). "
             "Omit to run all not-yet-done (fold, combo) units, which also "
             "auto-selects the winning combo at the end.",
    )
    parser.add_argument(
        "--finalize", action="store_true",
        help="Skip launching any fits; select the winning combo from whatever "
             "(fold, combo) results already exist in GRID_RESULTS_DIR. Use this "
             "once all 4 folds are done after a series of --fold-id commits.",
    )
    args = parser.parse_args()

    if args.finalize:
        _, dataset_end = dataset_bounds()
        all_fold_ids = [f.fold_id for f in make_folds(dataset_end)]
        summary = select_best_combo_and_write_results(GRID_RESULTS_DIR, RESULTS_DIR, all_fold_ids)
        print(f"Winning combo: index {summary['best_combo_index']} {summary['best_combo']} "
              f"(mean WMAPE {summary['mean_wmape']:.4f})")
        return

    run(fold_id=args.fold_id)


if __name__ == "__main__":
    main()
