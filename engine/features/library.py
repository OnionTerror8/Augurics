"""Feature library v1 — task 2.2.

Calendar, SNAP/event flags, price + price-change, lags/rolling stats,
event adstock decay, and active-span handling. Every feature that could
touch future information is built causally (shift/expanding-window
before any lag/rolling/recursive computation) so that a row's feature
value can never change when only LATER rows change -- verified per
feature in tests/test_features.py's no-leakage tests.

All functions take and return a per-series-or-multi-series long panel
DataFrame; when `group_col` is present, every windowed/lag/recursive
operation is computed within each group independently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sorted(df: pd.DataFrame, ds_col: str, group_col: str | None) -> pd.DataFrame:
    sort_cols = [group_col, ds_col] if group_col and group_col in df.columns else [ds_col]
    return df.sort_values(sort_cols).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    """Pure functions of the row's own date -- no leakage is possible."""
    out = df.copy()
    ds = pd.to_datetime(out[ds_col])
    out["dow"] = ds.dt.dayofweek
    out["month"] = ds.dt.month
    out["week_of_year"] = ds.dt.isocalendar().week.astype(int)
    out["is_weekend"] = ds.dt.dayofweek.isin([5, 6]).astype(int)
    return out


def add_event_snap_flags(
    df: pd.DataFrame,
    event_cols: tuple[str, ...] = ("event_name_1", "event_name_2"),
    snap_col: str = "snap",
) -> pd.DataFrame:
    """Row-wise flags from the row's own calendar fields -- no leakage."""
    out = df.copy()
    present_event_cols = [c for c in event_cols if c in out.columns]
    if present_event_cols:
        out["event_flag"] = np.any([out[c].notna().to_numpy() for c in present_event_cols], axis=0).astype(int)
    else:
        out["event_flag"] = 0
    out["snap_flag"] = out[snap_col].astype(int) if snap_col in out.columns else 0
    return out


def add_price_features(
    df: pd.DataFrame,
    price_col: str = "sell_price",
    ds_col: str = "ds",
    group_col: str | None = "series_key",
) -> pd.DataFrame:
    """log(price) is row-wise (safe). price_change is this row's price
    minus the PRIOR row's price (diff), so it depends only on t and t-1,
    never on t+1 or later."""
    out = _sorted(df, ds_col, group_col)
    out["log_price"] = np.log(out[price_col].astype(float).clip(lower=1e-3))
    if group_col and group_col in out.columns:
        out["price_change"] = out.groupby(group_col)[price_col].diff().fillna(0.0)
    else:
        out["price_change"] = out[price_col].diff().fillna(0.0)
    return out


def add_lag_features(
    df: pd.DataFrame,
    y_col: str = "y",
    lags: tuple[int, ...] = (7, 28),
    ds_col: str = "ds",
    group_col: str | None = "series_key",
) -> pd.DataFrame:
    """y_lag_k at row t is y at row t-k -- depends only on the past."""
    out = _sorted(df, ds_col, group_col)
    for lag in lags:
        col = f"{y_col}_lag_{lag}"
        if group_col and group_col in out.columns:
            out[col] = out.groupby(group_col)[y_col].shift(lag)
        else:
            out[col] = out[y_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    y_col: str = "y",
    windows: tuple[int, ...] = (7, 28),
    ds_col: str = "ds",
    group_col: str | None = "series_key",
) -> pd.DataFrame:
    """Rolling mean/std computed on data shifted by 1 BEFORE windowing, so
    the window covering row t is [t-window, t-1] -- the current row's own
    y is never part of its own rolling feature (that would leak the label
    into its own input)."""
    out = _sorted(df, ds_col, group_col)
    for window in windows:
        mean_col = f"{y_col}_rolling_mean_{window}"
        std_col = f"{y_col}_rolling_std_{window}"
        if group_col and group_col in out.columns:
            shifted = out.groupby(group_col)[y_col].shift(1)
            out[mean_col] = shifted.groupby(out[group_col]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            out[std_col] = shifted.groupby(out[group_col]).rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        else:
            shifted = out[y_col].shift(1)
            out[mean_col] = shifted.rolling(window, min_periods=1).mean()
            out[std_col] = shifted.rolling(window, min_periods=1).std()
    return out


def add_event_adstock(
    df: pd.DataFrame,
    event_col: str = "event_flag",
    decay: float = 0.5,
    ds_col: str = "ds",
    group_col: str | None = "series_key",
) -> pd.DataFrame:
    """Adstock: adstock_t = event_t + decay * adstock_(t-1). A pure causal
    recursive filter -- by construction it can only depend on t and earlier."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    out = _sorted(df, ds_col, group_col)

    def _adstock(series: pd.Series) -> pd.Series:
        vals = series.to_numpy(dtype=float)
        result = np.empty(len(vals))
        carry = 0.0
        for i, v in enumerate(vals):
            carry = v + decay * carry
            result[i] = carry
        return pd.Series(result, index=series.index)

    if group_col and group_col in out.columns:
        out["event_adstock"] = out.groupby(group_col)[event_col].transform(_adstock)
    else:
        out["event_adstock"] = _adstock(out[event_col])
    return out


def add_active_span_flag(
    df: pd.DataFrame,
    y_col: str = "y",
    ds_col: str = "ds",
    group_col: str | None = "series_key",
) -> pd.DataFrame:
    """1 from the row of the series' first nonzero sale onward, else 0.
    Computed as a causal expanding flag (cumulative sum of y up to and
    including the current row), NOT from the series' globally-known
    first-sale date -- the latter would use future rows to answer "have
    we sold anything yet" for rows before the actual first sale."""
    out = _sorted(df, ds_col, group_col)
    if group_col and group_col in out.columns:
        out["active_span_flag"] = (out.groupby(group_col)[y_col].cumsum() > 0).astype(int)
    else:
        out["active_span_flag"] = (out[y_col].cumsum() > 0).astype(int)
    return out


def build_feature_set(df: pd.DataFrame, adstock_decay: float = 0.5, group_col: str | None = "series_key") -> pd.DataFrame:
    """Apply the full v1 feature set in one call, in an order where every
    step's inputs are already present."""
    out = add_calendar_features(df)
    out = add_event_snap_flags(out)
    if "sell_price" in out.columns:
        out = add_price_features(out, group_col=group_col)
    out = add_lag_features(out, group_col=group_col)
    out = add_rolling_features(out, group_col=group_col)
    out = add_event_adstock(out, decay=adstock_decay, group_col=group_col)
    out = add_active_span_flag(out, group_col=group_col)
    return out
