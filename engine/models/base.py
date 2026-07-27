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

# The one place the "ITEM::STORE" bulk-path series_key string convention is
# defined. Every module that builds or matches this uid string (SQL in
# run_baselines.py, engine.models.baselines, ad hoc scripts) must use these,
# never re-derive the separator or join order locally -- a silently
# inconsistent convention is exactly how a key-format mismatch produces a
# filter that matches nothing without raising.
SERIES_KEY_SEP = "::"


def encode_series_key(product_id: str, location_id: str) -> str:
    return f"{product_id}{SERIES_KEY_SEP}{location_id}"


def select_series(df: pd.DataFrame, key, series_key_col: str = "series_key", expected_count: int | None = None) -> pd.DataFrame:
    """Fail loud: selecting a specific, supposedly-real series and getting
    zero rows back is never a legitimate outcome (unlike an iteration that
    tries several candidates) -- it means the key convention doesn't match
    what's actually in `series_key_col` (e.g. a tuple compared against a
    bulk-path uid string), and that must raise immediately, not silently
    produce an empty frame for the caller to trip over later."""
    subset = df[df[series_key_col] == key]
    if subset.empty:
        raise ValueError(
            f"select_series: no rows for {series_key_col}={key!r} -- check the key "
            f"format/convention (e.g. tuple vs '{SERIES_KEY_SEP}'-joined string)"
        )
    if expected_count is not None and len(subset) != expected_count:
        raise ValueError(f"select_series: expected {expected_count} rows for {series_key_col}={key!r}, got {len(subset)}")
    return subset


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
