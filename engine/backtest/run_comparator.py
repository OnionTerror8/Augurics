"""Bayesian comparator per-fold run — task 2.1 gate-relevant numbers.

Mirrors run_baselines.py's set-based extraction and run_slow_baselines.py
/run_one_baseline.py's subprocess-per-unit, atomic-write, resumable
orchestration -- same harness, same guarantees, different model.

Unquarantine conditions (checked once, on the fold running the bounded
NUTS spot-check -- fold 1 by default): (1) the coefficient-level ADVI-
vs-NUTS agreement already reported by the spot-check, and (2) the
DECISIVE predictive-agreement check (run_predictive_agreement_check) --
holdout WMAPE from ADVI's beta vs NUTS's beta on the same bounded
sample. If they diverge beyond PREDICTIVE_AGREEMENT_REL_TOLERANCE,
score_comparator raises PredictiveAgreementError before returning any
result, which stops the batch (see run_comparator_batch.py) before any
comparator figure reaches a report.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import engine.models.comparators  # noqa: F401 -- side effect: registers bayes_glm_pooled
from engine.backtest.gates import wmape
from engine.backtest.population import MIN_ACTIVE_SPAN_DAYS
from engine.backtest.protocol import Fold
from engine.data.ingest import DB_PATH
from engine.models.base import ModelConfig, PanelData
from engine.registry import REGISTRY

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "fold_cache" / "comparator"

COMPARATOR_MODEL_ID = "bayes_glm_pooled"

# Same eligibility CTE as run_baselines.py (train-fold data only, Protocol
# P1 / E3), but selecting the comparator's full feature set (category,
# event/SNAP flags, price) rather than just series_key/ds/y.
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

_COMPARATOR_TRAIN_SQL = (
    _ELIGIBLE_CTE
    + """
    SELECT
        (p.product_id || '::' || p.location_id) AS series_key,
        p.ds, p.y, p.category,
        (p.event_name_1 IS NOT NULL OR p.event_name_2 IS NOT NULL) AS event_flag,
        p.snap AS snap_flag,
        p.sell_price AS price
    FROM panel p
    JOIN eligible e ON e.product_id = p.product_id AND e.location_id = p.location_id
    WHERE p.ds <= $train_end
"""
)

_COMPARATOR_HOLDOUT_SQL = (
    _ELIGIBLE_CTE
    + """
    SELECT
        (p.product_id || '::' || p.location_id) AS series_key,
        p.ds, p.y, p.category,
        (p.event_name_1 IS NOT NULL OR p.event_name_2 IS NOT NULL) AS event_flag,
        p.snap AS snap_flag,
        p.sell_price AS price
    FROM panel p
    JOIN eligible e ON e.product_id = p.product_id AND e.location_id = p.location_id
    WHERE p.ds >= $holdout_start AND p.ds <= $holdout_end
"""
)


def build_comparator_fold_frames(fold: Fold, db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {"train_end": fold.train_end, "min_days": MIN_ACTIVE_SPAN_DAYS}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        train_df = con.execute(_COMPARATOR_TRAIN_SQL, params).df()
        holdout_df = con.execute(
            _COMPARATOR_HOLDOUT_SQL, {**params, "holdout_start": fold.holdout_start, "holdout_end": fold.holdout_end}
        ).df()
    finally:
        con.close()
    train_df["price"] = train_df["price"].fillna(1e-3)
    holdout_df["price"] = holdout_df["price"].fillna(1e-3)
    return train_df, holdout_df


def get_comparator_fold_frames_cached(
    fold: Fold, db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"fold{fold.fold_id}_train.parquet"
    holdout_path = cache_dir / f"fold{fold.fold_id}_holdout.parquet"
    if train_path.exists() and holdout_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(holdout_path)

    train_df, holdout_df = build_comparator_fold_frames(fold, db_path)
    train_df.to_parquet(train_path)
    holdout_df.to_parquet(holdout_path)
    return train_df, holdout_df


PREDICTIVE_AGREEMENT_REL_TOLERANCE = 0.02


class PredictiveAgreementError(Exception):
    """Raised when ADVI-beta and NUTS-beta holdout predictions on the same
    bounded spot-check sample diverge beyond PREDICTIVE_AGREEMENT_REL_TOLERANCE.
    The gate compares predictions, not posteriors -- a coefficient-level gap
    (mean_abs_coef_diff_advi_vs_nuts) does not by itself license or block
    ADVI; this check does, and must stop the batch before any comparator
    figure is used if it fails."""


def run_predictive_agreement_check(model, holdout_df: pd.DataFrame, spot_check: dict) -> dict:
    """Decisive check for licensing ADVI at gate scale: generate holdout
    predictions from BOTH the ADVI fit's beta and the NUTS spot-check's
    own beta, for the exact series the spot-check ran on, and compare
    their WMAPEs. Coefficient-level agreement (or disagreement) doesn't
    by itself say whether the two fits would gate the same way; this does."""
    if not spot_check.get("ran"):
        raise ValueError("run_predictive_agreement_check requires a completed NUTS spot-check")

    series_keys = spot_check["spot_check_series_keys"]
    category = spot_check["category"]
    nuts_beta = {block: np.array(values) for block, values in spot_check["nuts_beta"].items()}
    advi_beta = model._posterior_beta_for_category(category)

    subset_holdout = holdout_df[holdout_df["series_key"].isin(series_keys)]
    future_dates = pd.DatetimeIndex(sorted(pd.to_datetime(subset_holdout["ds"]).unique()))
    future_covariates = subset_holdout[["series_key", "ds", "event_flag", "snap_flag", "price"]].copy()

    lookup = pd.DataFrame(
        {
            "series_key": subset_holdout["series_key"].to_numpy(),
            "ds": pd.to_datetime(subset_holdout["ds"]).to_numpy(),
            "y_true": subset_holdout["y"].to_numpy(),
        }
    )

    advi_frames, nuts_frames = [], []
    for series_key in series_keys:
        advi_yhat = model._predict_series_with_beta(series_key, future_dates, advi_beta, future_covariates)
        nuts_yhat = model._predict_series_with_beta(series_key, future_dates, nuts_beta, future_covariates)
        advi_frames.append(pd.DataFrame({"series_key": series_key, "ds": future_dates, "yhat": advi_yhat}))
        nuts_frames.append(pd.DataFrame({"series_key": series_key, "ds": future_dates, "yhat": nuts_yhat}))

    advi_merged = pd.concat(advi_frames, ignore_index=True).merge(lookup, on=["series_key", "ds"], how="inner")
    nuts_merged = pd.concat(nuts_frames, ignore_index=True).merge(lookup, on=["series_key", "ds"], how="inner")
    if advi_merged.empty or nuts_merged.empty:
        raise ValueError("predictive agreement check: no predictions aligned with holdout actuals")

    advi_wmape = wmape(advi_merged["y_true"].to_numpy(), advi_merged["yhat"].to_numpy())
    nuts_wmape = wmape(nuts_merged["y_true"].to_numpy(), nuts_merged["yhat"].to_numpy())
    rel_diff = abs(advi_wmape - nuts_wmape) / nuts_wmape if nuts_wmape > 0 else float("inf")
    licensed = rel_diff <= PREDICTIVE_AGREEMENT_REL_TOLERANCE

    return {
        "advi_wmape": advi_wmape,
        "nuts_wmape": nuts_wmape,
        "relative_difference": rel_diff,
        "licensed": licensed,
        "category": category,
        "n_series": len(series_keys),
        "n_holdout_rows": len(lookup),
    }


def score_comparator(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, holdout_days: int, run_nuts_spot_check: bool = False
) -> dict:
    model_cls = REGISTRY[COMPARATOR_MODEL_ID]
    model = model_cls()

    t0 = time.time()
    fit_result = model.fit(
        PanelData(df=train_df),
        ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=holdout_days),
        run_nuts_spot_check=run_nuts_spot_check,
    )
    fit_seconds = time.time() - t0

    predictive_agreement = None
    if run_nuts_spot_check and fit_result.diagnostics["nuts_spot_check"].get("ran"):
        predictive_agreement = run_predictive_agreement_check(
            model, holdout_df, fit_result.diagnostics["nuts_spot_check"]
        )
        if not predictive_agreement["licensed"]:
            raise PredictiveAgreementError(
                f"ADVI vs NUTS holdout WMAPE diverge by {predictive_agreement['relative_difference']:.2%} "
                f"(ADVI={predictive_agreement['advi_wmape']:.4f}, NUTS={predictive_agreement['nuts_wmape']:.4f}), "
                f"exceeding the {PREDICTIVE_AGREEMENT_REL_TOLERANCE:.0%} tolerance -- ADVI is NOT licensed for "
                f"gate purposes. Stopping before any comparator figure is used."
            )

    future_covariates = holdout_df[["series_key", "ds", "event_flag", "snap_flag", "price"]].copy()

    t0 = time.time()
    preds = model._predict_all(holdout_days, future_covariates=future_covariates)
    predict_seconds = time.time() - t0

    holdout_lookup = pd.DataFrame(
        {
            "series_key": holdout_df["series_key"].to_numpy(),
            "ds": pd.to_datetime(holdout_df["ds"]).to_numpy(),
            "y_true": holdout_df["y"].to_numpy(),
        }
    )
    merged = preds.merge(holdout_lookup, on=["series_key", "ds"], how="inner")
    if merged.empty:
        raise ValueError(f"{COMPARATOR_MODEL_ID}: no predictions aligned with holdout actuals")

    return {
        "wmape": wmape(merged["y_true"].to_numpy(), merged["yhat"].to_numpy()),
        "n_series": train_df["series_key"].nunique(),
        "n_rows": len(merged),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "family": fit_result.family,
        "overdispersion_ratio": fit_result.diagnostics["overdispersion_ratio"],
        "advi_convergence_rel_change": fit_result.diagnostics["advi_convergence_rel_change"],
        "identifiability_note": fit_result.diagnostics["identifiability_note"],
        "nuts_spot_check": fit_result.diagnostics["nuts_spot_check"],
        "predictive_agreement": predictive_agreement,
    }
