"""Rung 2: lgbm via mlforecast + feature library. Task 2.3.

Training is gated on `check_challenger_fit_memory` first (kickoff §2 /
non-negotiable rule on fail-loud memory guards, same discipline as
`engine.models.comparators._check_fit_memory`) -- estimated empirically
from the feature library's real output on fold 1, not a formula guess.
At fold-1 scale (48.1M rows) this does not clear this development
laptop's available RAM even after the same category-dtype / int8-flag
/ float32 optimisations that fixed the comparator (see KNOWN_ISSUES.md)
-- training runs on Kaggle Notebooks (32GB RAM, see
KAGGLE_PROVISIONING.md) instead. The guard stays in place regardless of
host: a refusal with numbers beats an OOM discovered mid-grid-search
anywhere it runs.

mlforecast owns the autoregressive lag/rolling features (`lags`,
`lag_transforms` below) rather than `engine.features.library`'s
add_lag_features/add_rolling_features -- only mlforecast's recursive
prediction loop correctly substitutes its OWN prior predictions as the
next step's lag during multi-day-ahead forecasting; precomputed lag
columns have no such mechanism for the holdout period. The feature
library's non-autoregressive functions (calendar, event/SNAP flags,
price, event adstock -- all "known future" per Protocol P1/E3, real
values available for the holdout window in a backtest) ARE reused
directly, as exogenous regressors supplied for both train and predict.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd

from engine.features.library import add_calendar_features, add_event_adstock, add_price_features
from engine.models.base import (
    BacktestProtocol,
    BacktestResult,
    DateRange,
    Explanation,
    FitResult,
    ForecastModel,
    ForecastResult,
    InsufficientHistoryError,
    ModelConfig,
    NaNOutputError,
    PanelData,
    SeriesKey,
    encode_series_key,
)
from engine.registry import register

# Empirically calibrated (engine/models/ml.py's feasibility check, task
# 2.3 step 1): building the full feature set (engine.features.library.
# build_feature_set) on fold 1's real 48,110,016-row train frame, with
# series_key/category cast to `category` dtype, all flag columns
# downcast to int8/bool, and all engineered float columns to float32,
# measures 3,033,770,424 bytes via df.memory_usage(deep=True) --
# 63.06 bytes/row. This constant is that number, not a theoretical one.
OPTIMIZED_FEATURE_BYTES_PER_ROW = 63.06

# mlforecast/lightgbm's own training buffers on top of the feature
# matrix: gradient + hessian arrays (2 x 4 bytes/row, float32) plus
# mlforecast's internal copy during its own preprocessing step. Less
# than the comparator's 3x (no PyMC graph compilation), but real:
# calibrate conservatively at 1.5x total (matrix + lgbm buffers)
# rather than assume mlforecast makes zero additional copies.
LGBM_GRADIENT_HESSIAN_BYTES_PER_ROW = 8.0
LGBM_OVERHEAD_MULTIPLIER = 1.5

MAX_CHALLENGER_FIT_MEMORY_FRACTION = 0.5


class InsufficientChallengerMemoryError(Exception):
    """Raised by the pre-fit memory estimator when the challenger's
    estimated feature-matrix-plus-training footprint would exceed a
    safe fraction of currently available RAM. Mirrors
    engine.models.comparators.InsufficientMemoryError -- a refusal
    with numbers, not an OOM discovered mid-grid-search."""


def estimate_challenger_fit_memory_bytes(n_rows: int) -> int:
    bytes_per_row = OPTIMIZED_FEATURE_BYTES_PER_ROW + LGBM_GRADIENT_HESSIAN_BYTES_PER_ROW
    return int(n_rows * bytes_per_row * LGBM_OVERHEAD_MULTIPLIER)


def check_challenger_fit_memory(n_rows: int, max_fraction: float = MAX_CHALLENGER_FIT_MEMORY_FRACTION) -> dict:
    estimated = estimate_challenger_fit_memory_bytes(n_rows)
    available = psutil.virtual_memory().available
    budget = available * max_fraction
    if estimated > budget:
        raise InsufficientChallengerMemoryError(
            f"Estimated challenger fit memory ({estimated / 1e9:.2f} GB) exceeds {max_fraction:.0%} "
            f"of currently available RAM ({available / 1e9:.2f} GB, budget {budget / 1e9:.2f} GB) "
            f"for n_rows={n_rows:,}. Refusing to start -- this would very likely be OOM-killed "
            f"partway through a grid-search fit, wasting the fit time instead of failing loud immediately."
        )
    return {"estimated_bytes": estimated, "available_bytes": available, "budget_bytes": budget}


DEFAULT_CATEGORY = "unknown"
MIN_HISTORY_DAYS = 42
LAGS = (7, 28)
ROLLING_WINDOWS = (7, 28)
DEFAULT_HYPERPARAMS = {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
EXOGENOUS_FEATURE_COLUMNS = (
    "dow", "month", "week_of_year", "is_weekend",
    "event_flag", "snap_flag", "log_price", "price_change", "event_adstock",
)
ADSTOCK_DECAY = 0.5
ADSTOCK_SEED_DAYS = 60  # trailing history carried into the future window so the
                        # recursive adstock filter doesn't restart from 0 at the
                        # holdout boundary (see module docstring: this feature is
                        # "known future" but its own recursion needs a real seed).


def _series_key_to_uid(key: SeriesKey | str) -> str:
    return encode_series_key(*key) if isinstance(key, tuple) else str(key)


@register
class LGBMPromoModel(ForecastModel):
    model_id = "lgbm_promo_v1"
    display_name = "LightGBM (mlforecast, promo/price features)"
    rung = 2
    tier = 1
    supports_quantiles = False
    supports_covariates = True
    min_history_days = MIN_HISTORY_DAYS
    max_horizon_days = 28

    def __init__(self) -> None:
        self._mlf = None
        self._model_version = "v1"
        self._fit_df: pd.DataFrame | None = None
        self._exogenous_cols: list[str] = []
        self._series_category: dict[SeriesKey | str, str] = {}
        self._hyperparams: dict = {}

    def fit(self, panel: PanelData, config: ModelConfig) -> FitResult:
        df = panel.df.copy(deep=False)
        counts = df.groupby("series_key").size()
        too_short = counts[counts < self.min_history_days]
        if len(too_short) > 0:
            raise InsufficientHistoryError(
                f"{self.model_id}: {len(too_short)} series below min_history_days="
                f"{self.min_history_days}: {list(too_short.index)[:5]}"
            )

        df["ds"] = pd.to_datetime(df["ds"])
        if "category" not in df.columns:
            df["category"] = DEFAULT_CATEGORY
        df["category"] = df["category"].fillna(DEFAULT_CATEGORY)
        for col, default in (("event_flag", 0.0), ("snap_flag", 0.0), ("price", np.nan)):
            if col not in df.columns:
                df[col] = default

        check_challenger_fit_memory(len(df))

        df["unique_id"] = df["series_key"].map(_series_key_to_uid)
        self._series_category = {
            row["series_key"]: row["category"] for _, row in df.drop_duplicates("series_key").iterrows()
        }

        # event_flag/snap_flag are already-precomputed booleans by this
        # point (real data: derived from event_name_1/2 by the SQL that
        # built the panel, same convention as run_comparator.py; the
        # generic contract panel: defaulted above). Do NOT call
        # add_event_snap_flags -- it recomputes both from raw
        # event_name_1/event_name_2/snap columns and would overwrite
        # these with 0 whenever those raw columns aren't present.
        df = add_calendar_features(df)
        df = add_price_features(df, price_col="price", group_col="series_key")
        df = add_event_adstock(df, decay=ADSTOCK_DECAY, group_col="series_key")

        exogenous_cols = [c for c in EXOGENOUS_FEATURE_COLUMNS if c in df.columns]
        self._exogenous_cols = exogenous_cols

        hyperparams = dict(config.hyperparams) if config.hyperparams else dict(DEFAULT_HYPERPARAMS)
        self._hyperparams = hyperparams
        lgbm = lgb.LGBMRegressor(**hyperparams, verbosity=-1, random_state=42)

        self._mlf = MLForecast(
            models={self.model_id: lgbm},
            freq="D",
            lags=list(LAGS),
            lag_transforms={1: [RollingMean(window_size=w) for w in ROLLING_WINDOWS] + [RollingStd(window_size=w) for w in ROLLING_WINDOWS]},
        )
        train_cols = ["unique_id", "ds", "y", "category"] + exogenous_cols
        train_frame = df[train_cols].copy()
        train_frame["category"] = train_frame["category"].astype("category")  # lightgbm requires pandas category dtype, not object, for categorical columns
        self._mlf.fit(train_frame, static_features=["category"])
        self._fit_df = df

        return FitResult(
            model_version=self._model_version,
            family=type(lgbm).__name__,
            fitted_at=datetime_now_iso(),
            diagnostics={
                "n_series": len(self._series_category),
                "n_rows": len(df),
                "hyperparams": hyperparams,
                "exogenous_features": exogenous_cols,
            },
        )

    def _future_exogenous_frame(self, uid: str, horizon: int, future_covariates: pd.DataFrame | None) -> pd.DataFrame:
        history = self._fit_df[self._fit_df["unique_id"] == uid].sort_values("ds")
        last_ds = history["ds"].max()
        future_dates = pd.date_range(last_ds + pd.Timedelta(days=1), periods=horizon, freq="D")

        fut = pd.DataFrame({"unique_id": uid, "ds": future_dates})
        if future_covariates is not None:
            cov = future_covariates.copy()
            cov["ds"] = pd.to_datetime(cov["ds"])
            if "series_key" in cov.columns:
                cov = cov[cov["series_key"].map(_series_key_to_uid) == uid].drop(columns=["series_key"])
            fut = fut.merge(cov, on="ds", how="left")
        for col, default in (("event_flag", 0.0), ("snap_flag", 0.0), ("price", np.nan)):
            if col not in fut.columns:
                fut[col] = default
            fut[col] = fut[col].fillna(default)

        fut = add_calendar_features(fut)
        fut = add_price_features(fut, price_col="price", group_col=None)
        # event_adstock is a recursive filter (E3 "known future" but its own
        # recursion needs a real seed) -- seed it with the trailing history's
        # event_flag so the filter doesn't restart from 0 at the fold boundary.
        seed_tail = history.tail(ADSTOCK_SEED_DAYS)[["ds", "event_flag"]]
        combined = pd.concat([seed_tail, fut[["ds", "event_flag"]]], ignore_index=True)
        combined = add_event_adstock(combined, decay=ADSTOCK_DECAY, group_col=None)
        fut["event_adstock"] = combined["event_adstock"].to_numpy()[len(seed_tail):]

        return fut[["unique_id", "ds"] + self._exogenous_cols]

    def predict(self, horizon: int, future_covariates: PanelData | None) -> ForecastResult:
        if self._mlf is None:
            raise RuntimeError(f"{self.model_id}.predict() called before fit()")
        if horizon > self.max_horizon_days:
            raise ValueError(f"{self.model_id}: horizon {horizon} exceeds max_horizon_days {self.max_horizon_days}")

        series_key = next(iter(self._series_category))
        uid = _series_key_to_uid(series_key)
        cov_df = future_covariates.df if future_covariates is not None else None
        X_df = self._future_exogenous_frame(uid, horizon, cov_df)

        preds = self._mlf.predict(horizon, X_df=X_df, ids=[uid])
        yhat = preds[self.model_id].to_numpy(dtype=float)
        if np.any(np.isnan(yhat)):
            raise NaNOutputError(f"{self.model_id}.predict() produced NaN output")
        yhat = np.clip(yhat, 0.0, None)

        return ForecastResult(
            series_key=series_key,
            ds=[d.date().isoformat() for d in preds["ds"]],
            yhat=yhat.tolist(),
            yhat_quantiles={},
            model_id=self.model_id,
            model_version=self._model_version,
            generated_at=datetime_now_iso(),
            job_id=None,
        )

    def _predict_all(self, horizon: int, future_covariates: pd.DataFrame | None = None) -> pd.DataFrame:
        """Bulk multi-series prediction, same convention as baselines/
        comparators -- used by the per-fold backtest orchestration."""
        if self._mlf is None:
            raise RuntimeError(f"{self.model_id}._predict_all() called before fit()")
        uids = [_series_key_to_uid(k) for k in self._series_category]
        frames = []
        for series_key, uid in zip(self._series_category, uids):
            X_df = self._future_exogenous_frame(uid, horizon, future_covariates)
            preds = self._mlf.predict(horizon, X_df=X_df, ids=[uid])
            yhat = np.clip(preds[self.model_id].to_numpy(dtype=float), 0.0, None)
            frames.append(pd.DataFrame({"series_key": [series_key] * horizon, "ds": preds["ds"].to_numpy(), "yhat": yhat}))
        return pd.concat(frames, ignore_index=True)

    def explain(self, series_key: SeriesKey, date_range: DateRange) -> Explanation:
        if self._mlf is None:
            raise RuntimeError(f"{self.model_id}.explain() called before fit()")
        if series_key not in self._series_category:
            raise ValueError(f"{self.model_id}.explain(): series {series_key} was not part of the fitted panel")

        booster = self._mlf.models_[self.model_id]
        importances = dict(zip(booster.feature_name_, booster.feature_importances_.tolist()))
        top_drivers = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return Explanation(
            series_key=series_key,
            date_range=date_range,
            components={"feature_importances": importances},
            drivers=[
                {"name": name, "direction": "n/a", "magnitude": float(score), "confidence": "lgbm split-count importance"}
                for name, score in top_drivers
            ],
            narrative_facts=[
                f"Family: {type(booster).__name__} (mlforecast/lightgbm), hyperparams {self._hyperparams}.",
                f"Top driver by split-count importance: {top_drivers[0][0]}." if top_drivers else "No feature importances available.",
            ],
        )

    def backtest(self, panel: PanelData, protocol: BacktestProtocol) -> BacktestResult:
        from engine.backtest.gates import wmape

        df = panel.df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        dataset_end = pd.Timestamp(df["ds"].max())
        holdout_start = dataset_end - pd.Timedelta(days=protocol.holdout_days - 1)

        train_df = df[df["ds"] < holdout_start]
        holdout_df = df[df["ds"] >= holdout_start]
        if train_df.empty or holdout_df.empty:
            raise ValueError(f"{self.model_id}.backtest(): not enough history for a {protocol.holdout_days}-day holdout")

        self.fit(PanelData(df=train_df), ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=protocol.holdout_days))
        forecast = self.predict(protocol.holdout_days, future_covariates=None)
        series_key = next(iter(self._series_category))
        series_holdout = holdout_df[holdout_df["series_key"] == series_key].sort_values("ds")
        if series_holdout.empty:
            raise ValueError(f"{self.model_id}.backtest(): no holdout rows aligned with the fitted series")

        y_true = series_holdout["y"].to_numpy(dtype=float)[: len(forecast.yhat)]
        y_pred = np.array(forecast.yhat[: len(series_holdout)])
        metric_value = wmape(y_true, y_pred)

        return BacktestResult(
            experiment_id=f"{self.model_id}-{protocol.protocol_id}",
            model_version_id=self._model_version,
            metrics={"wmape": metric_value},
            per_fold=[{"fold_id": 1, "wmape": metric_value, "n_rows": len(y_true)}],
            gates_evaluated=[],
            volume_retained=1.0,
        )

    def save(self, artefact_uri: str) -> None:
        import pickle

        with open(artefact_uri, "wb") as f:
            pickle.dump(
                {
                    "mlf": self._mlf,
                    "model_version": self._model_version,
                    "fit_df": self._fit_df,
                    "exogenous_cols": self._exogenous_cols,
                    "series_category": self._series_category,
                    "hyperparams": self._hyperparams,
                },
                f,
            )

    @classmethod
    def load(cls, artefact_uri: str) -> "ForecastModel":
        import pickle

        with open(artefact_uri, "rb") as f:
            state = pickle.load(f)
        instance = cls()
        instance._mlf = state["mlf"]
        instance._model_version = state["model_version"]
        instance._fit_df = state["fit_df"]
        instance._exogenous_cols = state["exogenous_cols"]
        instance._series_category = state["series_category"]
        instance._hyperparams = state["hyperparams"]
        return instance


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
