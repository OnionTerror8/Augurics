"""Real M5 baseline run — task 1.6.

For each Protocol P1 fold: compute that fold's OWN eligible population
from train-fold data only (Protocol P1 / E3), extract train/holdout rows
for that population in ONE set-based DuckDB query each (never a Python
loop or per-row zip/isin over the full row count -- rows are in the tens
of millions, series in the tens of thousands), cache the result to
parquet so it's built once and reused by every model, then fit each
registered Rung 1 baseline with statsforecast's vectorised API
(n_jobs=-1) and score volume-weighted WMAPE against that fold's holdout.

No gates are evaluated here -- G1/G2 need a Rung 2 challenger (task 2.3)
and G3 needs the Bayesian comparator (task 2.1). This produces the
baseline WMAPE table only.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

import engine.models.baselines  # noqa: F401 -- side effect: registers Rung 1 baselines
from engine.backtest.gates import wmape
from engine.backtest.population import MIN_ACTIVE_SPAN_DAYS
from engine.backtest.protocol import Fold, dataset_bounds, make_folds
from engine.data.ingest import DB_PATH
from engine.models.base import ModelConfig, PanelData
from engine.registry import REGISTRY

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
BASELINE_RESULTS_MD = REPORTS_DIR / "baseline_results.md"

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "fold_cache"

# Both queries compute the fold's eligible population from TRAIN-FOLD DATA
# ONLY (Protocol P1 / E3), then join it back against the panel to extract
# exactly the rows for that population -- one set-based query each, never
# a Python-side filter over the full row count.
_ELIGIBLE_CTE = """
    WITH train_only AS (
        SELECT * FROM panel WHERE ds <= $train_end
    ),
    first_sale AS (
        SELECT product_id, location_id, MIN(ds) AS first_nonzero_sale
        FROM train_only WHERE y > 0
        GROUP BY product_id, location_id
    ),
    eligible AS (
        SELECT product_id, location_id
        FROM first_sale
        WHERE date_diff('day', first_nonzero_sale, $train_end) >= $min_days
    )
"""

_TRAIN_FRAME_SQL = (
    _ELIGIBLE_CTE
    + """
    SELECT (p.product_id || '::' || p.location_id) AS series_key, p.ds, p.y
    FROM panel p
    JOIN eligible e ON e.product_id = p.product_id AND e.location_id = p.location_id
    WHERE p.ds <= $train_end
"""
)

_HOLDOUT_FRAME_SQL = (
    _ELIGIBLE_CTE
    + """
    SELECT (p.product_id || '::' || p.location_id) AS series_key, p.ds, p.y
    FROM panel p
    JOIN eligible e ON e.product_id = p.product_id AND e.location_id = p.location_id
    WHERE p.ds >= $holdout_start AND p.ds <= $holdout_end
"""
)


def build_fold_frames(fold: Fold, db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and holdout frames for `fold`, restricted to the fold's own
    train-data-only eligible population, in two set-based queries."""
    params = {"train_end": fold.train_end, "min_days": MIN_ACTIVE_SPAN_DAYS}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        train_df = con.execute(_TRAIN_FRAME_SQL, params).df()
        holdout_df = con.execute(
            _HOLDOUT_FRAME_SQL, {**params, "holdout_start": fold.holdout_start, "holdout_end": fold.holdout_end}
        ).df()
    finally:
        con.close()
    return train_df, holdout_df


def get_fold_frames_cached(fold: Fold, db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"fold{fold.fold_id}_train.parquet"
    holdout_path = cache_dir / f"fold{fold.fold_id}_holdout.parquet"
    if train_path.exists() and holdout_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(holdout_path)

    train_df, holdout_df = build_fold_frames(fold, db_path)
    train_df.to_parquet(train_path)
    holdout_df.to_parquet(holdout_path)
    return train_df, holdout_df


def score_model(model_id: str, train_df: pd.DataFrame, holdout_df: pd.DataFrame, holdout_days: int) -> dict:
    model_cls = REGISTRY[model_id]
    model = model_cls()

    t0 = time.time()
    model.fit(
        PanelData(df=train_df),
        ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=holdout_days),
    )
    fit_seconds = time.time() - t0

    t0 = time.time()
    preds = model._predict_all(holdout_days)
    predict_seconds = time.time() - t0

    holdout_ds = pd.to_datetime(holdout_df["ds"])
    holdout_lookup = pd.DataFrame(
        {"series_key": holdout_df["series_key"].to_numpy(), "ds": holdout_ds.to_numpy(), "y_true": holdout_df["y"].to_numpy()}
    )
    merged = preds.merge(holdout_lookup, on=["series_key", "ds"], how="inner")
    if merged.empty:
        raise ValueError(f"{model_id}: no predictions aligned with holdout actuals")

    return {
        "wmape": wmape(merged["y_true"].to_numpy(), merged["yhat"].to_numpy()),
        "n_series": train_df["series_key"].nunique(),
        "n_rows": len(merged),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def run_fold(fold: Fold, model_ids: list[str], db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR) -> dict:
    train_df, holdout_df = get_fold_frames_cached(fold, db_path, cache_dir)
    holdout_days = (fold.holdout_end - fold.holdout_start).days + 1
    return {model_id: score_model(model_id, train_df, holdout_df, holdout_days) for model_id in model_ids}


def run_all_folds(model_ids: list[str] | None = None, db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR) -> dict:
    if model_ids is None:
        model_ids = sorted(REGISTRY.keys())
    _, dataset_end = dataset_bounds(db_path)
    folds = make_folds(dataset_end)
    return {fold.fold_id: run_fold(fold, model_ids, db_path, cache_dir) for fold in folds}


def render_report(fold_results: dict) -> str:
    fold_ids = sorted(fold_results.keys())
    model_ids = sorted({model_id for fold in fold_results.values() for model_id in fold})

    lines = [
        "# Baseline Results — Rung 1 (Protocol P1, task 1.6)",
        "",
        "Volume-weighted WMAPE per fold, each fold scored against ITS OWN "
        "eligible population (Protocol P1 / E3 — active span computed from "
        "train-fold data only). No gates (G1/G2/G3) are evaluated here: "
        "G1/G2 need a Rung 2 challenger (task 2.3), G3 needs the Bayesian "
        "comparator (task 2.1).",
        "",
        "| Model | " + " | ".join(f"Fold {fid} WMAPE" for fid in fold_ids) + " | Mean WMAPE |",
        "|---|" + "---|" * (len(fold_ids) + 1),
    ]

    for model_id in model_ids:
        per_fold = [fold_results[fid][model_id]["wmape"] for fid in fold_ids]
        mean_wmape = sum(per_fold) / len(per_fold)
        cells = " | ".join(f"{v:.4f}" for v in per_fold)
        lines.append(f"| {model_id} | {cells} | {mean_wmape:.4f} |")

    lines.append("")
    lines.append("## Fold population sizes")
    lines.append("")
    lines.append("| Fold | Eligible series | Holdout rows scored |")
    lines.append("|---|---|---|")
    for fid in fold_ids:
        any_model = next(iter(fold_results[fid].values()))
        lines.append(f"| {fid} | {any_model['n_series']:,} | {any_model['n_rows']:,} |")

    lines.append("")
    lines.append("## Fit/predict runtime per model per fold (seconds)")
    lines.append("")
    lines.append("| Model | " + " | ".join(f"Fold {fid} fit+predict" for fid in fold_ids) + " |")
    lines.append("|---|" + "---|" * len(fold_ids))
    for model_id in model_ids:
        cells = " | ".join(
            f"{fold_results[fid][model_id]['fit_seconds'] + fold_results[fid][model_id]['predict_seconds']:.1f}s"
            for fid in fold_ids
        )
        lines.append(f"| {model_id} | {cells} |")

    return "\n".join(lines)


def write_report(out_path: Path = BASELINE_RESULTS_MD, db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR) -> Path:
    fold_results = run_all_folds(db_path=db_path, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(fold_results))
    return out_path


if __name__ == "__main__":
    written = write_report()
    print(f"Wrote {written}")
