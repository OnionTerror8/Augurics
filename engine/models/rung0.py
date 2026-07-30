"""Rung 0 -- "Incumbent": wraps an already-existing forecast series (any
external source -- a customer's current platform, a manual override, or
another Augurics model's own output used as a stand-in) as a first-class
registry citizen. EXPORT_CONTRACTS.md §2: "the incumbent forecast is a
registry citizen" -- it scores through the identical rolling-origin
backtest harness as every Augurics model and appears on the Gate Ladder
as Rung 0, "your forecast, measured honestly," ahead of Rungs 1-4.

StaticForecastModel is fundamentally different from every other
registered model: it fits no parameters from historical sales. The
panel handed to fit() IS the forecast series itself -- each
(series_key, ds, y) row is an already-produced forecast value for that
date, from whatever external source supplied it. predict()/
_predict_all() serve those values back verbatim: a pure lookup, never
interpolated, extrapolated, or smoothed. That is the entire point --
Rung 0 must be *exactly* what the incumbent said, not Augurics'
interpretation of it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

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
    select_series,
)
from engine.registry import register

MIN_HISTORY_DAYS = 7  # a Rung-0 citizen needs at least a week of supplied forecast values to be usable
MAX_HORIZON_DAYS = 28

SOURCE_NOTE = (
    "externally supplied forecast (Rung 0 / incumbent) -- values are ingested and "
    "replayed verbatim, never computed by Augurics"
)


class IncompleteForecastCoverageError(Exception):
    """Raised when predict()/​_predict_all() is asked for a horizon longer
    than the number of forecast dates actually supplied for a series --
    a static/replay model has no mechanism to extrapolate beyond what it
    was given, and silently padding or repeating values would
    misrepresent the externally-supplied forecast. Fail loud instead."""


@register
class StaticForecastModel(ForecastModel):
    """Rung 0. Wraps an externally-supplied forecast series as a registry
    citizen with zero internal computation -- fit() stores the given
    values verbatim; predict() replays them. See module docstring."""

    model_id = "static_incumbent"
    display_name = "Rung 0 -- Incumbent (externally supplied forecast)"
    rung = 0
    tier = 1  # cheapest tier: a stored-value lookup, no fit compute at all
    supports_quantiles = False
    supports_covariates = False
    min_history_days = MIN_HISTORY_DAYS
    max_horizon_days = MAX_HORIZON_DAYS

    def __init__(self) -> None:
        self._model_version = "v1"
        self._fit_df: pd.DataFrame | None = None
        self._series_forecast: dict[SeriesKey, pd.DataFrame] = {}

    def fit(self, panel: PanelData, config: ModelConfig) -> FitResult:
        df = panel.df.copy()
        counts = df.groupby("series_key").size()
        too_short = counts[counts < self.min_history_days]
        if len(too_short) > 0:
            raise InsufficientHistoryError(
                f"{self.model_id}: {len(too_short)} series have fewer than min_history_days="
                f"{self.min_history_days} supplied forecast dates: {list(too_short.index)[:5]}"
            )

        df["ds"] = pd.to_datetime(df["ds"])
        self._fit_df = df.sort_values(["series_key", "ds"]).reset_index(drop=True)
        self._series_forecast = {
            key: group[["ds", "y"]].reset_index(drop=True) for key, group in self._fit_df.groupby("series_key")
        }

        return FitResult(
            model_version=self._model_version,
            family="StaticReplay",
            fitted_at=datetime.now(timezone.utc).isoformat(),
            diagnostics={
                "n_series": len(self._series_forecast),
                "n_rows": len(self._fit_df),
                "source": SOURCE_NOTE,
            },
        )

    def _series_window(self, series_key: SeriesKey, horizon: int) -> pd.DataFrame:
        if series_key not in self._series_forecast:
            raise ValueError(f"{self.model_id}: series {series_key} was not part of the fitted panel")
        stored = self._series_forecast[series_key]
        if len(stored) < horizon:
            raise IncompleteForecastCoverageError(
                f"{self.model_id}: series {series_key} has only {len(stored)} supplied forecast "
                f"dates, but horizon={horizon} was requested -- a static/replay model cannot "
                f"extrapolate beyond what was supplied"
            )
        return stored.iloc[:horizon]

    def predict(self, horizon: int, future_covariates: PanelData | None) -> ForecastResult:
        if self._fit_df is None:
            raise RuntimeError(f"{self.model_id}.predict() called before fit()")
        if horizon > self.max_horizon_days:
            raise ValueError(f"{self.model_id}: horizon {horizon} exceeds max_horizon_days {self.max_horizon_days}")

        series_key = next(iter(self._series_forecast))
        window = self._series_window(series_key, horizon)
        yhat = window["y"].to_numpy(dtype=float)
        if np.any(np.isnan(yhat)):
            raise NaNOutputError(f"{self.model_id}.predict() produced NaN output for series {series_key}")

        return ForecastResult(
            series_key=series_key,
            ds=[d.date().isoformat() for d in window["ds"]],
            yhat=yhat.tolist(),
            yhat_quantiles={},
            model_id=self.model_id,
            model_version=self._model_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            job_id=None,
        )

    def _predict_all(self, horizon: int, future_covariates: pd.DataFrame | None = None) -> pd.DataFrame:
        """Bulk multi-series replay -- same convention as every other
        model's _predict_all(), used by the backtest/scoring
        orchestration (engine.backtest.run_baselines.score_model and
        equivalents), which is how Rung 0 is actually scored against
        real holdout actuals (see tests/test_rung0.py)."""
        if self._fit_df is None:
            raise RuntimeError(f"{self.model_id}._predict_all() called before fit()")

        frames = []
        for series_key in self._series_forecast:
            window = self._series_window(series_key, horizon)
            frames.append(
                pd.DataFrame(
                    {
                        "series_key": [series_key] * len(window),
                        "ds": window["ds"].to_numpy(),
                        "yhat": window["y"].to_numpy(dtype=float),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def explain(self, series_key: SeriesKey, date_range: DateRange) -> Explanation:
        if self._fit_df is None:
            raise RuntimeError(f"{self.model_id}.explain() called before fit()")
        if series_key not in self._series_forecast:
            raise ValueError(f"{self.model_id}.explain(): series {series_key} was not part of the fitted panel")

        series_df = select_series(self._fit_df, series_key).sort_values("ds")
        mask = (series_df["ds"].dt.date >= date_range.start) & (series_df["ds"].dt.date <= date_range.end)
        windowed = series_df.loc[mask] if mask.any() else series_df.iloc[0:0]
        mean_value = float(windowed["y"].mean()) if len(windowed) else 0.0

        return Explanation(
            series_key=series_key,
            date_range=date_range,
            components={
                "source": "external",
                "n_forecast_dates_in_range": int(len(windowed)),
                "mean_forecast_value_in_range": mean_value,
            },
            drivers=[
                {
                    "name": "external_source",
                    "direction": "n/a",
                    "magnitude": mean_value,
                    "confidence": "not applicable -- externally supplied, no internal driver decomposition",
                }
            ],
            narrative_facts=[
                f"This is an {SOURCE_NOTE}.",
                f"{len(windowed)} forecast date(s) supplied within the requested window "
                f"({date_range.start.isoformat()} to {date_range.end.isoformat()}).",
                "Rung 0 (Incumbent): scored through the identical rolling-origin backtest "
                "harness as every Augurics model, per EXPORT_CONTRACTS.md §2.",
            ],
        )

    def backtest(self, panel: PanelData, protocol: BacktestProtocol) -> BacktestResult:
        """Self-consistency check ONLY -- a static/replay model has no
        genuine fit-vs-holdout split (there are no parameters to fit;
        fit() just stores whatever it is given verbatim), so this
        necessarily scores the model's stored values against themselves
        (WMAPE == 0 by construction). This exists to satisfy the
        ForecastModel ABC's structural contract, not to produce a
        meaningful accuracy number. The MEANINGFUL evaluation of an
        externally-supplied forecast is via the standard scoring harness
        (engine.backtest.run_baselines.score_model or equivalent),
        comparing its stored values against REAL actuals from a separate
        holdout dataset -- exactly how the Rung 0 demo
        (tests/test_rung0.py) exercises it."""
        from engine.backtest.gates import wmape

        df = panel.df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        self.fit(
            PanelData(df=df),
            ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=protocol.holdout_days),
        )

        series_key = next(iter(self._series_forecast))
        window = self._series_window(series_key, protocol.holdout_days)
        y_true = window["y"].to_numpy(dtype=float)
        y_pred = y_true.copy()  # self-consistency: a static model cannot diverge from what it was given
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
                    "model_version": self._model_version,
                    "fit_df": self._fit_df,
                    "series_forecast": self._series_forecast,
                },
                f,
            )

    @classmethod
    def load(cls, artefact_uri: str) -> "ForecastModel":
        import pickle

        with open(artefact_uri, "rb") as f:
            state = pickle.load(f)
        instance = cls()
        instance._model_version = state["model_version"]
        instance._fit_df = state["fit_df"]
        instance._series_forecast = state["series_forecast"]
        return instance
