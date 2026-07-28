"""Challenger (Rung 2, lgbm_promo_v1) per-fold, per-combo scoring --
task 2.3. Same set-based extraction as the comparator (identical
columns: series_key, ds, y, category, event_flag, snap_flag, price) --
reused directly rather than duplicated. Same subprocess-per-unit,
atomic-write, resumable orchestration as run_comparator.py/
run_baselines.py, just at (fold, combo) granularity instead of (fold,)
alone, since the pre-registered grid (config/lgbm_grid.yaml) has 12
combos to try per fold.
"""
from __future__ import annotations

import time

import pandas as pd

import engine.models.ml  # noqa: F401 -- side effect: registers lgbm_promo_v1
from engine.backtest.gates import wmape
from engine.backtest.run_comparator import build_comparator_fold_frames, get_comparator_fold_frames_cached  # noqa: F401 -- re-exported, same fold-frame shape the challenger needs
from engine.models.base import ModelConfig, PanelData
from engine.registry import REGISTRY

CHALLENGER_MODEL_ID = "lgbm_promo_v1"


def score_challenger(train_df: pd.DataFrame, holdout_df: pd.DataFrame, holdout_days: int, hyperparams: dict) -> dict:
    model_cls = REGISTRY[CHALLENGER_MODEL_ID]
    model = model_cls()

    t0 = time.time()
    fit_result = model.fit(
        PanelData(df=train_df),
        ModelConfig(hyperparams=hyperparams, feature_set="none", vertical_profile="retail", horizon=holdout_days),
    )
    fit_seconds = time.time() - t0

    future_covariates = holdout_df[["series_key", "ds", "event_flag", "snap_flag", "price"]].copy()

    t0 = time.time()
    preds = model._predict_all(holdout_days, future_covariates=future_covariates)
    predict_seconds = time.time() - t0

    holdout_lookup = pd.DataFrame({
        "series_key": holdout_df["series_key"].to_numpy(),
        "ds": pd.to_datetime(holdout_df["ds"]).to_numpy(),
        "y_true": holdout_df["y"].to_numpy(),
    })
    merged = preds.merge(holdout_lookup, on=["series_key", "ds"], how="inner")
    if merged.empty:
        raise ValueError(f"{CHALLENGER_MODEL_ID}: no predictions aligned with holdout actuals")

    return {
        "wmape": wmape(merged["y_true"].to_numpy(), merged["yhat"].to_numpy()),
        "n_series": train_df["series_key"].nunique(),
        "n_rows": len(merged),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "family": fit_result.family,
        "hyperparams": hyperparams,
    }
