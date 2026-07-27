"""Bayesian comparator: bayes_glm_pooled. Task 2.1, PHASE1_KICKOFF.md §3.

"The arm we must beat" for gate G3 -- under-specifying it would make G3
hollow, so every element of §3 is implemented, not stubbed:

- Family: Negative Binomial GLM, family read from the fitted pm.NegativeBinomial
  node's distribution name, never a config string.
- Pooling: hierarchical partial pooling at category level (non-centred
  parameterisation), zero-centred priors on regressor coefficients.
- Regressors: event/SNAP flags, log(price), day-of-week, month, trend;
  active-span offset = log(mean sales over active span), never the
  full-window mean (E3).
- Identifiability: co-firing event/SNAP streams are checked for
  separability; a near-zero coefficient on a co-firing stream is reported
  as "not separately identifiable", never "no effect" (E5).
- Fit budget: ADVI for speed, with a NUTS spot-check on one category,
  documented either way in fit diagnostics.

The generic contract-compliance panel has no category/event/price columns
(registry spec's static_features/known_future are empty dicts in that
fixture) -- fit() degrades gracefully to a single pooled category and
zero-valued covariates rather than raising, so the same model class is
exercised meaningfully on both the tiny contract panel and the real M5
panel where those columns are present.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pymc as pm

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
)
from engine.registry import register

MIN_HISTORY_DAYS = 42  # enough for weekly seasonality + a stable active-span offset
DEFAULT_CATEGORY = "unknown"
ADVI_ITERATIONS = 2000
ADVI_LEARNING_RATE = 0.01
POSTERIOR_DRAWS = 500
CO_FIRING_CORRELATION_THRESHOLD = 0.995


def _active_span_offset(y: np.ndarray) -> float:
    """log(mean sales over active span), never full-window mean (E3):
    active span starts at the series' first nonzero observation."""
    nonzero_idx = np.flatnonzero(y > 0)
    if len(nonzero_idx) == 0:
        return float(np.log(1e-6))
    active = y[nonzero_idx[0] :]
    mean_active = max(float(active.mean()), 1e-6)
    return float(np.log(mean_active))


def _build_features(df: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow").astype(float)
    month_dummies = pd.get_dummies(df["month"], prefix="month").astype(float)
    base = df[["event_flag", "snap_flag", "log_price", "trend"]].astype(float).reset_index(drop=True)
    X = pd.concat([base, dow_dummies.reset_index(drop=True), month_dummies.reset_index(drop=True)], axis=1)
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0.0)
    return X


@register
class BayesGLMPooled(ForecastModel):
    model_id = "bayes_glm_pooled"
    display_name = "Bayesian GLM (pooled, Negative Binomial)"
    rung = 1  # statistical comparator -- not tree/GBM (rung 2) or neural (rung 3)
    tier = 2  # python-svc: pymc fit is heavier than the tier-1 statsforecast baselines
    supports_quantiles = True  # posterior predictive gives a full distribution
    supports_covariates = True
    min_history_days = MIN_HISTORY_DAYS
    max_horizon_days = 28

    def __init__(self) -> None:
        self._trace = None
        self._model_version = "v1"
        self._family = "NegativeBinomial"
        self._fit_df: pd.DataFrame | None = None
        self._feature_names: list[str] = []
        self._category_codes: dict[str, int] = {}
        self._series_offset: dict[SeriesKey, float] = {}
        self._series_category: dict[SeriesKey, str] = {}
        self._identifiability_note = "not evaluated"
        self._nuts_spot_check: dict | None = None

    def fit(self, panel: PanelData, config: ModelConfig) -> FitResult:
        df = panel.df.copy()
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

        # Active-span offset per series (E3): log(mean sales over active span).
        offsets: dict = {}
        categories: dict = {}
        for series_key, group in df.groupby("series_key"):
            y = group.sort_values("ds")["y"].to_numpy(dtype=float)
            offsets[series_key] = _active_span_offset(y)
            categories[series_key] = group["category"].iloc[0]
        self._series_offset = offsets
        self._series_category = categories

        df["offset"] = df["series_key"].map(offsets)
        df["dow"] = df["ds"].dt.dayofweek
        df["month"] = df["ds"].dt.month
        df["trend"] = ((df["ds"] - df["ds"].min()).dt.days.astype(float)) / 30.0
        df["log_price"] = np.log(df["price"].astype(float).clip(lower=1e-3)).fillna(0.0)

        # Co-firing identifiability check (E5): near-zero coefficient on a
        # perfectly collinear stream means "not separately identifiable",
        # never "no effect".
        if df["event_flag"].nunique() > 1 and df["snap_flag"].nunique() > 1:
            corr = np.corrcoef(df["event_flag"].astype(float), df["snap_flag"].astype(float))[0, 1]
        else:
            corr = np.nan
        self._identifiability_note = (
            "event_flag/snap_flag are perfectly co-firing in this fit -- their "
            "individual coefficients are NOT separately identifiable, not evidence of no effect"
            if np.isfinite(corr) and abs(corr) > CO_FIRING_CORRELATION_THRESHOLD
            else "event_flag/snap_flag separability OK"
        )

        categories_sorted = sorted(df["category"].unique())
        self._category_codes = {c: i for i, c in enumerate(categories_sorted)}
        cat_idx = df["category"].map(self._category_codes).to_numpy()
        n_categories = len(categories_sorted)

        X = _build_features(df)
        self._feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
        y_arr = df["y"].to_numpy(dtype=float)
        offset_arr = df["offset"].to_numpy(dtype=float)

        with pm.Model() as model:
            mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=0.5, shape=X_arr.shape[1])
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
            beta_offset_rv = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=(n_categories, X_arr.shape[1]))
            beta = mu_beta + sigma_beta * beta_offset_rv  # non-centred, zero-centred priors

            alpha = pm.HalfNormal("alpha", sigma=2.0)

            eta = offset_arr + pm.math.sum(beta[cat_idx] * X_arr, axis=1)
            eta_clipped = pm.math.clip(eta, -10, 10)
            mu = pm.math.exp(eta_clipped)

            pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y_arr)
            family_from_fitted_object = model.named_vars["y_obs"].owner.op.__class__.__name__

            approx = pm.fit(
                n=ADVI_ITERATIONS,
                method="advi",
                progressbar=False,
                random_seed=42,
                obj_optimizer=pm.adam(learning_rate=ADVI_LEARNING_RATE),
            )
            trace = approx.sample(POSTERIOR_DRAWS)

        self._family = family_from_fitted_object
        self._trace = trace
        self._fit_df = df

        # NUTS spot-check on one category (kickoff §3): documented either way.
        self._nuts_spot_check = self._run_nuts_spot_check(
            df, categories_sorted[0], X_arr.shape[1]
        )

        overdispersion_ratio = float(y_arr.var() / max(y_arr.mean(), 1e-6))

        return FitResult(
            model_version=self._model_version,
            family=self._family,
            fitted_at=datetime.now(timezone.utc).isoformat(),
            diagnostics={
                "n_series": len(offsets),
                "n_rows": len(df),
                "n_categories": n_categories,
                "overdispersion_ratio": overdispersion_ratio,
                "fit_method": "ADVI",
                "identifiability_note": self._identifiability_note,
                "nuts_spot_check": self._nuts_spot_check,
            },
        )

    def _run_nuts_spot_check(self, df: pd.DataFrame, spot_category: str, n_features: int) -> dict:
        """Fit budget per kickoff §3: ADVI for speed, NUTS spot-check on one
        category, documented either way. Runs a small NUTS chain on just
        that category's rows and compares its mean coefficient vector to
        ADVI's for the same category."""
        subset = df[df["category"] == spot_category]
        if len(subset) < self.min_history_days:
            return {"ran": False, "reason": f"category {spot_category!r} too small for a spot-check"}

        X = _build_features(subset, feature_names=self._feature_names).to_numpy(dtype=float)
        y = subset["y"].to_numpy(dtype=float)
        offset = subset["offset"].to_numpy(dtype=float)

        try:
            with pm.Model():
                beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
                alpha = pm.HalfNormal("alpha", sigma=2.0)
                eta = pm.math.clip(offset + pm.math.sum(beta * X, axis=1), -10, 10)
                mu = pm.math.exp(eta)
                pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y)
                nuts_trace = pm.sample(
                    draws=200, tune=200, chains=1, progressbar=False, random_seed=42,
                    compute_convergence_checks=False,
                )
            nuts_beta_mean = nuts_trace.posterior["beta"].mean(dim=("chain", "draw")).values

            cat_idx = self._category_codes[spot_category]
            advi_beta_mean = (
                self._trace.posterior["mu_beta"].mean(dim=("chain", "draw")).values
                + self._trace.posterior["sigma_beta"].mean(dim=("chain", "draw")).values
                * self._trace.posterior["beta_offset"].mean(dim=("chain", "draw")).values[cat_idx]
            )
            mean_abs_diff = float(np.mean(np.abs(nuts_beta_mean - advi_beta_mean)))
            return {"ran": True, "category": spot_category, "mean_abs_coef_diff_advi_vs_nuts": mean_abs_diff}
        except Exception as exc:  # NUTS spot-check is a diagnostic, not load-bearing for the fit itself
            return {"ran": False, "reason": f"NUTS spot-check failed: {exc}"}

    def _posterior_beta_for_category(self, category: str) -> np.ndarray:
        cat_idx = self._category_codes.get(category, 0)
        mu_beta = self._trace.posterior["mu_beta"].mean(dim=("chain", "draw")).values
        sigma_beta = self._trace.posterior["sigma_beta"].mean(dim=("chain", "draw")).values
        beta_offset = self._trace.posterior["beta_offset"].mean(dim=("chain", "draw")).values
        return mu_beta + sigma_beta * beta_offset[cat_idx]

    def _future_feature_frame(self, horizon: int, future_covariates: PanelData | None) -> pd.DataFrame:
        last_ds = self._fit_df["ds"].max()
        future_dates = pd.date_range(last_ds + pd.Timedelta(days=1), periods=horizon, freq="D")

        fut = pd.DataFrame({"ds": future_dates})
        if future_covariates is not None:
            cov = future_covariates.df.copy()
            cov["ds"] = pd.to_datetime(cov["ds"])
            fut = fut.merge(cov, on="ds", how="left")
        for col, default in (("event_flag", 0.0), ("snap_flag", 0.0), ("price", np.nan)):
            if col not in fut.columns:
                fut[col] = default
            fut[col] = fut[col].fillna(default)

        fut["dow"] = fut["ds"].dt.dayofweek
        fut["month"] = fut["ds"].dt.month
        fut["trend"] = ((fut["ds"] - self._fit_df["ds"].min()).dt.days.astype(float)) / 30.0
        fut["log_price"] = np.log(fut["price"].astype(float).clip(lower=1e-3)).fillna(0.0)
        return fut

    def predict(self, horizon: int, future_covariates: PanelData | None) -> ForecastResult:
        if self._trace is None:
            raise RuntimeError(f"{self.model_id}.predict() called before fit()")
        if horizon > self.max_horizon_days:
            raise ValueError(f"{self.model_id}: horizon {horizon} exceeds max_horizon_days {self.max_horizon_days}")

        # Contract-level predict reports the single/first fitted series, the
        # same convention as engine.models.baselines (the ABC's predict()
        # signature has no series_key argument to disambiguate a multi-series fit).
        series_key = next(iter(self._series_offset))
        category = self._series_category[series_key]
        offset = self._series_offset[series_key]

        fut = self._future_feature_frame(horizon, future_covariates)
        X_future = _build_features(fut, feature_names=self._feature_names).to_numpy(dtype=float)
        beta = self._posterior_beta_for_category(category)

        eta = np.clip(offset + X_future @ beta, -10, 10)
        yhat = np.exp(eta)

        if np.any(np.isnan(yhat)):
            raise NaNOutputError(f"{self.model_id}.predict() produced NaN output")
        yhat = np.clip(yhat, 0.0, None)

        yhat_quantiles = self._posterior_predictive_quantiles(eta)

        return ForecastResult(
            series_key=series_key,
            ds=[d.date().isoformat() for d in fut["ds"]],
            yhat=yhat.tolist(),
            yhat_quantiles=yhat_quantiles,
            model_id=self.model_id,
            model_version=self._model_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            job_id=None,
        )

    def _posterior_predictive_quantiles(self, eta: np.ndarray, n_draws: int = 200) -> dict[str, list[float]]:
        rng = np.random.default_rng(42)
        alpha_draws = self._trace.posterior["alpha"].values.reshape(-1)
        alpha_sample = rng.choice(alpha_draws, size=min(n_draws, len(alpha_draws)), replace=False)

        mu = np.exp(eta)
        samples = np.stack(
            [rng.negative_binomial(n=a, p=a / (a + mu)) for a in alpha_sample], axis=0
        ).astype(float)
        return {
            "p10": np.quantile(samples, 0.1, axis=0).tolist(),
            "p50": np.quantile(samples, 0.5, axis=0).tolist(),
            "p90": np.quantile(samples, 0.9, axis=0).tolist(),
        }

    def explain(self, series_key: SeriesKey, date_range: DateRange) -> Explanation:
        if self._fit_df is None:
            raise RuntimeError(f"{self.model_id}.explain() called before fit()")
        if series_key not in self._series_offset:
            raise ValueError(f"{self.model_id}.explain(): series {series_key} was not part of the fitted panel")

        category = self._series_category[series_key]
        offset = self._series_offset[series_key]
        beta = self._posterior_beta_for_category(category)

        series_df = self._fit_df[self._fit_df["series_key"] == series_key].sort_values("ds")
        mask = (series_df["ds"].dt.date >= date_range.start) & (series_df["ds"].dt.date <= date_range.end)
        windowed = series_df.loc[mask] if mask.any() else series_df

        X_window = _build_features(windowed, feature_names=self._feature_names).to_numpy(dtype=float)
        covariate_contribution = float(np.mean(X_window @ beta)) if len(X_window) else 0.0
        level = float(np.exp(offset))

        trend_idx = self._feature_names.index("trend")
        trend_contribution = float(beta[trend_idx] * windowed["trend"].mean()) if len(windowed) else 0.0
        dow_indices = [i for i, name in enumerate(self._feature_names) if name.startswith("dow_")]
        seasonality_contribution = float(np.mean(beta[dow_indices])) if dow_indices else 0.0

        alpha_mean = float(self._trace.posterior["alpha"].mean().values)

        return Explanation(
            series_key=series_key,
            date_range=date_range,
            components={
                "level": level,
                "trend": trend_contribution,
                "seasonality": seasonality_contribution,
                "covariate_effects": covariate_contribution,
                "residual_dispersion_alpha": alpha_mean,
            },
            drivers=[
                {"name": "active_span_offset", "direction": "level", "magnitude": level, "confidence": "posterior mean"},
                {"name": "event_flag", "direction": "n/a", "magnitude": float(beta[self._feature_names.index("event_flag")]), "confidence": self._identifiability_note},
            ],
            narrative_facts=[
                f"Family: {self._family} (read from the fitted object, not a config string).",
                f"Fit method: ADVI ({ADVI_ITERATIONS} iterations); NUTS spot-check: "
                f"{self._nuts_spot_check.get('ran', False)}.",
                self._identifiability_note,
                f"Active-span offset (log mean sales over active span): {offset:.3f} -> level {level:.3f}.",
            ],
        )

    def backtest(self, panel: PanelData, protocol: BacktestProtocol) -> BacktestResult:
        """Self-contained single-holdout backtest, to genuinely exercise the
        ABC contract. The real, multi-fold M5 run is deferred until the
        Rung 1 baseline batch finishes and the cores are free (task 2.1
        instructions) -- this exists so the interface is not a stub."""
        from engine.backtest.gates import wmape

        df = panel.df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        dataset_end = pd.Timestamp(df["ds"].max())
        holdout_start = dataset_end - pd.Timedelta(days=protocol.holdout_days - 1)

        train_df = df[df["ds"] < holdout_start]
        holdout_df = df[df["ds"] >= holdout_start]
        if train_df.empty or holdout_df.empty:
            raise ValueError(
                f"{self.model_id}.backtest(): not enough history for a {protocol.holdout_days}-day holdout"
            )

        self.fit(
            PanelData(df=train_df),
            ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=protocol.holdout_days),
        )

        results = []
        for series_key in self._series_offset:
            forecast = self.predict(protocol.holdout_days, future_covariates=None)
            series_holdout = holdout_df[holdout_df["series_key"] == series_key].sort_values("ds")
            if series_holdout.empty:
                continue
            results.append((series_holdout["y"].to_numpy(dtype=float)[: len(forecast.yhat)], np.array(forecast.yhat[: len(series_holdout)])))
            break  # single/first series, matching predict()'s single-series convention

        if not results:
            raise ValueError(f"{self.model_id}.backtest(): no holdout rows aligned with the fitted series")

        y_true, y_pred = results[0]
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
                    "trace": self._trace,
                    "model_version": self._model_version,
                    "family": self._family,
                    "fit_df": self._fit_df,
                    "feature_names": self._feature_names,
                    "category_codes": self._category_codes,
                    "series_offset": self._series_offset,
                    "series_category": self._series_category,
                    "identifiability_note": self._identifiability_note,
                    "nuts_spot_check": self._nuts_spot_check,
                },
                f,
            )

    @classmethod
    def load(cls, artefact_uri: str) -> "ForecastModel":
        import pickle

        with open(artefact_uri, "rb") as f:
            state = pickle.load(f)
        instance = cls()
        instance._trace = state["trace"]
        instance._model_version = state["model_version"]
        instance._family = state["family"]
        instance._fit_df = state["fit_df"]
        instance._feature_names = state["feature_names"]
        instance._category_codes = state["category_codes"]
        instance._series_offset = state["series_offset"]
        instance._series_category = state["series_category"]
        instance._identifiability_note = state["identifiability_note"]
        instance._nuts_spot_check = state["nuts_spot_check"]
        return instance
