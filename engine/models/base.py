"""The Common Model Interface — registry spec §1-2.

Every model, every rung, one contract. This module defines the interface
and canonical data objects only; no concrete models live here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

SeriesKey = tuple[str, str]  # (product_id, location_id)

PANEL_COLUMNS = ("series_key", "ds", "y", "static_features", "known_future", "observed_past")


class InsufficientHistoryError(Exception):
    """Raised by fit() when a series is below min_history_days. Never fit on garbage."""


class NaNOutputError(Exception):
    """Raised when a forecast output contains NaN. Never coalesce silently."""


@dataclass
class DateRange:
    start: date
    end: date


@dataclass
class PanelData:
    """Long-format panel — the ONLY data shape models see."""

    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in ("series_key", "ds", "y") if c not in self.df.columns]
        if missing:
            raise ValueError(f"PanelData missing required columns: {missing}")


@dataclass
class ModelConfig:
    hyperparams: dict[str, Any]
    feature_set: str
    vertical_profile: str
    horizon: int
    quantiles: list[float] = field(default_factory=list)


@dataclass
class FitResult:
    model_version: str
    family: str  # reported from the fitted object, never from config strings
    fitted_at: str  # ISO string
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    series_key: SeriesKey
    ds: list[str]  # ISO date strings
    yhat: list[float]
    yhat_quantiles: dict[str, list[float]]
    model_id: str
    model_version: str
    generated_at: str  # ISO string
    job_id: str | None = None


@dataclass
class Explanation:
    series_key: SeriesKey
    date_range: DateRange
    components: dict[str, Any]
    drivers: list[dict[str, Any]]
    narrative_facts: list[str]


@dataclass
class Gate:
    gate_id: str
    comparator_model_id: str
    metric: str
    pass_condition: str
    scope: str


@dataclass
class BacktestProtocol:
    protocol_id: str
    holdout_days: int
    method: str = "rolling_origin"
    n_folds: int = 4
    metrics: list[str] = field(default_factory=lambda: ["wmape", "bias", "mase", "pinball@0.9"])
    coverage_diagnostic: bool = True
    gates: list[Gate] = field(default_factory=list)


@dataclass
class BacktestResult:
    experiment_id: str
    model_version_id: str
    metrics: dict[str, float]
    per_fold: list[dict[str, Any]]
    gates_evaluated: list[dict[str, Any]]
    volume_retained: float


class ForecastModel(ABC):
    """All models, all rungs, one contract."""

    # --- Identity ---
    model_id: str
    display_name: str
    rung: int  # 1=Baseline, 2=ML, 3=Neural, 4=Foundation
    tier: int  # compute tier: 1=in-db, 2=python-svc, 3=gpu/heavy
    supports_quantiles: bool
    supports_covariates: bool
    min_history_days: int  # fail-loud threshold, checked before fit
    max_horizon_days: int

    @abstractmethod
    def fit(self, panel: PanelData, config: ModelConfig) -> FitResult: ...

    @abstractmethod
    def predict(self, horizon: int, future_covariates: PanelData | None) -> ForecastResult: ...

    @abstractmethod
    def explain(self, series_key: SeriesKey, date_range: DateRange) -> Explanation: ...

    @abstractmethod
    def backtest(self, panel: PanelData, protocol: BacktestProtocol) -> BacktestResult: ...

    @abstractmethod
    def save(self, artefact_uri: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, artefact_uri: str) -> "ForecastModel": ...
