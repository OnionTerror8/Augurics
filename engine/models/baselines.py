"""Rung 1 baselines: naive, snaive, ets, theta (statsforecast). Task 1.6.

Registry spec §1: "baselines return their rule in words" for explain().
All four wrap statsforecast, fit on possibly-many-series panels at once
(vectorised), but predict()/explain() report a single series per call,
matching the ForecastResult/Explanation schema.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, SeasonalNaive
from statsforecast.models import Theta as SFTheta

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
    select_series,
)
from engine.registry import register

SEASON_LENGTH = 7  # weekly seasonality, daily M5-style data


def _series_key_to_uid(series_key: SeriesKey) -> str:
    return encode_series_key(*series_key)


class _StatsForecastBaseline(ForecastModel):
    """Shared machinery for the Rung 1 statsforecast baselines. Not itself
    registered -- concrete subclasses set model_id/display_name and the
    wrapped statsforecast model."""

    rung = 1
    tier = 1
    supports_quantiles = False
    supports_covariates = False
    min_history_days = 2 * SEASON_LENGTH
    max_horizon_days = 28

    def __init__(self) -> None:
        self._sf: StatsForecast | None = None
        self._uid_to_key: dict[str, SeriesKey] = {}
        self._fit_df: pd.DataFrame | None = None
        self._model_version = "v1"
        self._family = type(self._sf_model()).__name__

    def _sf_model(self):
        raise NotImplementedError

    def _rule_narrative(self) -> str:
        raise NotImplementedError

    def fit(self, panel: PanelData, config: ModelConfig) -> FitResult:
        df = panel.df
        counts = df.groupby("series_key").size()
        too_short = counts[counts < self.min_history_days]
        if len(too_short) > 0:
            raise InsufficientHistoryError(
                f"{self.model_id}: {len(too_short)} series below "
                f"min_history_days={self.min_history_days}: {list(too_short.index)[:5]}"
            )

        # series_key may already be a pre-built uid string (the fast path:
        # callers doing a bulk multi-million-row backtest build this
        # directly in SQL) or a tuple (the small-scale contract-test path).
        # Never run a Python-level .map()/zip() over the full row count --
        # that is O(rows), not O(series), and rows can be in the tens of
        # millions while series are in the tens of thousands.
        sample_key = df["series_key"].iloc[0]
        if isinstance(sample_key, str):
            unique_id_col = df["series_key"]
        else:
            unique_id_col = df["series_key"].map(_series_key_to_uid)

        sf_df = pd.DataFrame(
            {
                "unique_id": unique_id_col,
                "ds": pd.to_datetime(df["ds"]),
                "y": df["y"].astype(float),
            }
        )

        # Build the uid -> original series_key mapping from the DISTINCT
        # series only (tens of thousands), never from every row.
        distinct = df[["series_key"]].drop_duplicates()
        distinct_uid = distinct["series_key"] if isinstance(sample_key, str) else distinct["series_key"].map(_series_key_to_uid)
        self._uid_to_key = dict(zip(distinct_uid, distinct["series_key"]))

        model_instance = self._sf_model()
        self._family = type(model_instance).__name__
        self._sf = StatsForecast(models=[model_instance], freq="D", n_jobs=-1)
        self._sf.fit(sf_df)
        self._fit_df = sf_df

        return FitResult(
            model_version=self._model_version,
            family=self._family,
            fitted_at=datetime.now(timezone.utc).isoformat(),
            diagnostics={"n_series": len(self._uid_to_key), "n_rows": len(sf_df)},
        )

    def _predict_all(self, horizon: int) -> pd.DataFrame:
        """Bulk prediction for every fitted series -- not part of the ABC
        contract, used internally by backtest() and by the real M5
        baseline-run orchestration for efficiency (one vectorised call
        instead of one predict() per series)."""
        if self._sf is None:
            raise RuntimeError(f"{self.model_id}._predict_all() called before fit()")

        preds = self._sf.predict(h=horizon)
        value_col = next(c for c in preds.columns if c not in ("unique_id", "ds"))
        yhat = preds[value_col].to_numpy(dtype=float)

        if np.any(np.isnan(yhat)):
            raise NaNOutputError(f"{self.model_id}.predict() produced NaN output")
        yhat = np.clip(yhat, a_min=0.0, a_max=None)  # demand forecasts are non-negative

        return pd.DataFrame(
            {
                "unique_id": preds["unique_id"].to_numpy(),
                "series_key": [self._uid_to_key[u] for u in preds["unique_id"]],
                "ds": preds["ds"].to_numpy(),
                "yhat": yhat,
            }
        )

    def predict(self, horizon: int, future_covariates: PanelData | None) -> ForecastResult:
        if self._sf is None:
            raise RuntimeError(f"{self.model_id}.predict() called before fit()")
        if horizon > self.max_horizon_days:
            raise ValueError(f"{self.model_id}: horizon {horizon} exceeds max_horizon_days {self.max_horizon_days}")

        all_preds = self._predict_all(horizon)
        first_uid = all_preds["unique_id"].iloc[0]
        mask = all_preds["unique_id"] == first_uid
        series_preds = all_preds.loc[mask]

        return ForecastResult(
            series_key=self._uid_to_key[first_uid],
            ds=[pd.Timestamp(d).date().isoformat() for d in series_preds["ds"]],
            yhat=series_preds["yhat"].tolist(),
            yhat_quantiles={},
            model_id=self.model_id,
            model_version=self._model_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            job_id=None,
        )

    def explain(self, series_key: SeriesKey, date_range: DateRange) -> Explanation:
        if self._fit_df is None:
            raise RuntimeError(f"{self.model_id}.explain() called before fit()")

        uid = _series_key_to_uid(series_key)
        series_df = self._fit_df[self._fit_df["unique_id"] == uid]
        if series_df.empty:
            raise ValueError(f"{self.model_id}.explain(): series {series_key} was not part of the fitted panel")

        mask = (series_df["ds"].dt.date >= date_range.start) & (series_df["ds"].dt.date <= date_range.end)
        windowed = series_df.loc[mask, "y"]
        if windowed.empty:
            windowed = series_df["y"]
        level = float(windowed.mean())

        rule_text = self._rule_narrative()
        return Explanation(
            series_key=series_key,
            date_range=date_range,
            components={"level": level},
            drivers=[
                {"name": "baseline_rule", "direction": "n/a", "magnitude": 0.0, "confidence": "rule-based"}
            ],
            narrative_facts=[rule_text, f"mean observed sales over the window: {level:.2f}"],
        )

    def backtest(self, panel: PanelData, protocol: BacktestProtocol) -> BacktestResult:
        """Self-contained single-holdout backtest over the given panel, to
        genuinely exercise the ABC contract. The real, multi-fold M5 run
        uses engine.backtest.protocol/population externally instead, since
        that is where per-fold eligible-population logic (Protocol P1 / E3)
        lives -- this method does not duplicate that orchestration."""
        from engine.backtest.gates import wmape

        df = panel.df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        dataset_end = pd.Timestamp(df["ds"].max())
        holdout_start = dataset_end - pd.Timedelta(days=protocol.holdout_days - 1)

        train_df = df[df["ds"] < holdout_start]
        holdout_df = df[df["ds"] >= holdout_start]
        if train_df.empty or holdout_df.empty:
            raise ValueError(
                f"{self.model_id}.backtest(): not enough history for a "
                f"{protocol.holdout_days}-day holdout"
            )

        self.fit(
            PanelData(df=train_df),
            ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=protocol.holdout_days),
        )
        preds = self._predict_all(protocol.holdout_days)

        merged = preds.merge(
            holdout_df.rename(columns={"y": "y_true"})[["series_key", "ds", "y_true"]],
            on=["series_key", "ds"],
            how="inner",
        )
        if merged.empty:
            raise ValueError(f"{self.model_id}.backtest(): predictions did not align with any holdout dates")

        metric_value = wmape(merged["y_true"].to_numpy(), merged["yhat"].to_numpy())

        return BacktestResult(
            experiment_id=f"{self.model_id}-{protocol.protocol_id}",
            model_version_id=self._model_version,
            metrics={"wmape": metric_value},
            per_fold=[{"fold_id": 1, "wmape": metric_value, "n_rows": len(merged)}],
            gates_evaluated=[],
            volume_retained=1.0,
        )

    def save(self, artefact_uri: str) -> None:
        with open(artefact_uri, "wb") as f:
            pickle.dump(
                {
                    "sf": self._sf,
                    "uid_to_key": self._uid_to_key,
                    "fit_df": self._fit_df,
                    "model_version": self._model_version,
                    "family": self._family,
                },
                f,
            )

    @classmethod
    def load(cls, artefact_uri: str) -> "ForecastModel":
        with open(artefact_uri, "rb") as f:
            state = pickle.load(f)
        instance = cls()
        instance._sf = state["sf"]
        instance._uid_to_key = state["uid_to_key"]
        instance._fit_df = state["fit_df"]
        instance._model_version = state["model_version"]
        instance._family = state["family"]
        return instance


@register
class NaiveModel(_StatsForecastBaseline):
    model_id = "naive"
    display_name = "Naive"

    def _sf_model(self):
        return Naive()

    def _rule_narrative(self) -> str:
        return "Naive: forecast equals the last observed value, repeated across the horizon."


@register
class SeasonalNaiveModel(_StatsForecastBaseline):
    model_id = "seasonal_naive"
    display_name = "Seasonal Naive"

    def _sf_model(self):
        return SeasonalNaive(season_length=SEASON_LENGTH)

    def _rule_narrative(self) -> str:
        return f"Seasonal Naive: forecast equals the observed value {SEASON_LENGTH} days prior (weekly seasonality)."


@register
class AutoETSModel(_StatsForecastBaseline):
    model_id = "auto_ets"
    display_name = "AutoETS"

    def _sf_model(self):
        return AutoETS(season_length=SEASON_LENGTH)

    def _rule_narrative(self) -> str:
        return "AutoETS: exponential smoothing with automatically selected error/trend/seasonal components."


@register
class ThetaModel(_StatsForecastBaseline):
    model_id = "theta"
    display_name = "Theta"

    def _sf_model(self):
        return SFTheta(season_length=SEASON_LENGTH)

    def _rule_narrative(self) -> str:
        return "Theta: decomposes the series into long-term trend and short-term theta lines, then recombines them."
