"""The six read-tool functions (02_AGENT_TOOL_CONTRACTS.md's "Read /
analyse (auto-execute)" set), as tested pure Python -- task 2.5. No
FastAPI/SSE layer yet (that's Phase 2's engine/service/api.py); these
are the underlying functions the eventual orchestration loop will call
as tools, callable and tested standalone now.

Every output is a JSON-primitive dict/list (ISO date strings, series
keys as "product::location" strings, no NaN -- fails loud instead of
emitting one, per non-negotiable rule 4) so that task 2.6's JSON
round-trip test can assert on every tool's output, not just some.

Phase-1 stand-ins, documented rather than silent:
- No champions table exists yet (Phase 2). `model_version` defaults to
  DEFAULT_CHAMPION_MODEL_ID (the best Rung-1 baseline by volume-weighted
  WMAPE in BASELINE_TABLE.md) when the caller doesn't specify one.
- No model-serving/caching layer exists yet. Each call fits fresh
  against real M5 history -- correct, not cached, per Phase 1's scope.
- No scenarios table exists yet. run_scenario returns a fresh uuid4
  scenario_id each call; nothing persists.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import engine.models.baselines  # noqa: F401 -- side effect: registers Rung 1 baselines
import engine.models.comparators  # noqa: F401 -- side effect: registers bayes_glm_pooled
from engine.backtest.gate_report import collect_results
from engine.backtest.gates import wmape
from engine.backtest.run_baselines import CACHE_DIR as BASELINE_CACHE_DIR
from engine.data.ingest import DB_PATH
from engine.models.base import DateRange, ModelConfig, PanelData, encode_series_key, select_series
from engine.registry import REGISTRY
from engine.service.jobs import get_job

DEFAULT_CHAMPION_MODEL_ID = "auto_ets"  # BASELINE_TABLE.md: lowest volume-weighted WMAPE of Rung 1
BASELINE_RESULTS_DIR = BASELINE_CACHE_DIR / "slow_baseline_results"

SENSITIVITY_Z_THRESHOLDS = {"low": 3.0, "med": 2.5, "high": 2.0}


class AmbiguousScopeError(Exception):
    """Raised when `scope` resolves to more than one series and the
    caller (Phase 1) has no aggregation strategy for that yet -- fail
    loud rather than silently picking one series or averaging."""


class UnsupportedDriverKindError(Exception):
    """Raised by run_scenario for a driver kind Phase 1 doesn't
    implement yet (weather/custom/profile) -- never silently a no-op."""


def _resolve_series(scope: dict) -> str:
    """scope -> exactly one "product::location" series key, or raises.
    Phase 1 requires product_id+location_id; category-level scope needs
    an aggregation strategy this phase doesn't have -- fail loud."""
    product_id = scope.get("product")
    location_id = scope.get("location")
    if not product_id or not location_id:
        raise AmbiguousScopeError(
            f"_resolve_series: scope {scope!r} does not resolve to exactly one series -- "
            "Phase 1 requires both 'product' and 'location'; category-only scope has no "
            "aggregation strategy yet"
        )
    return encode_series_key(product_id, location_id)


def _fetch_history(series_key: str, as_of: date, db_path=DB_PATH) -> pd.DataFrame:
    """All real history up to (and including) `as_of` for one series, via
    one set-based DuckDB query -- never a Python-side filter."""
    product_id, location_id = series_key.split("::")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT (p.product_id || '::' || p.location_id) AS series_key,
                   p.ds, p.y, p.category,
                   (p.event_name_1 IS NOT NULL OR p.event_name_2 IS NOT NULL) AS event_flag,
                   p.snap AS snap_flag, p.sell_price AS price
            FROM panel p
            WHERE p.product_id = $product_id AND p.location_id = $location_id AND p.ds <= $as_of
            ORDER BY p.ds
            """,
            {"product_id": product_id, "location_id": location_id, "as_of": as_of},
        ).df()
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"_fetch_history: no rows for series_key={series_key!r} as of {as_of}")
    df["price"] = df["price"].fillna(1e-3)
    return df


def _fit_champion(history_df: pd.DataFrame, model_id: str, horizon: int):
    if model_id not in REGISTRY:
        raise ValueError(f"unknown model_id={model_id!r} -- registered: {sorted(REGISTRY)}")
    model = REGISTRY[model_id]()
    model.fit(
        PanelData(df=history_df),
        ModelConfig(hyperparams={}, feature_set="none", vertical_profile="retail", horizon=horizon),
    )
    return model


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def get_forecast(scope: dict, date_range: dict, model_version: str | None = None) -> dict:
    """in: scope {product, location}, date_range {start, end} (ISO date
    strings), model_version?  out: forecast series + champion model
    identity + quantiles if available."""
    series_key = _resolve_series(scope)
    start = date.fromisoformat(date_range["start"])
    end = date.fromisoformat(date_range["end"])
    horizon = (end - start).days + 1
    if horizon <= 0:
        raise ValueError(f"get_forecast: date_range end {end} is not after start {start}")

    model_id = model_version or DEFAULT_CHAMPION_MODEL_ID
    history_df = _fetch_history(series_key, as_of=start - pd.Timedelta(days=1))
    model = _fit_champion(history_df, model_id, horizon)
    forecast = model.predict(horizon, future_covariates=None)

    if any(np.isnan(v) for v in forecast.yhat):
        raise ValueError(f"get_forecast: model {model_id!r} produced NaN in yhat -- fail loud, never coalesce")

    return {
        "series_key": series_key,
        "ds": list(forecast.ds),
        "yhat": [float(v) for v in forecast.yhat],
        "yhat_quantiles": {k: [float(v) for v in vs] for k, vs in forecast.yhat_quantiles.items()},
        "champion_model_id": model_id,
        "model_version": forecast.model_version,
        "generated_at": forecast.generated_at,
    }


def explain_forecast(series_key: str, date_range: dict, question_focus: str | None = None, model_version: str | None = None) -> dict:
    """in: series_key ("product::location"), date_range {start, end},
    question_focus?  out: Explanation object (components, drivers,
    narrative_facts), JSON-safe."""
    start = date.fromisoformat(date_range["start"])
    end = date.fromisoformat(date_range["end"])
    model_id = model_version or DEFAULT_CHAMPION_MODEL_ID

    history_df = _fetch_history(series_key, as_of=end)
    model = _fit_champion(history_df, model_id, horizon=max((end - start).days + 1, 1))

    product_id, location_id = series_key.split("::")
    explanation = model.explain((product_id, location_id), DateRange(start=start, end=end))
    result = asdict(explanation)
    result["series_key"] = series_key
    result["date_range"] = {"start": _iso(start), "end": _iso(end)}
    result["question_focus"] = question_focus
    result["model_id"] = model_id
    return result


def detect_anomalies(scope: dict, date_range: dict, sensitivity: str = "med") -> list[dict]:
    """in: scope, date_range, sensitivity: low|med|high  out:
    list[{series_key, ds_range, anomaly_type, severity, narrative_facts}].
    Real statistical detection: z-score of actual y against a trailing
    28-day rolling mean/std computed causally (shifted before windowing,
    same no-leakage discipline as engine.features.library)."""
    if sensitivity not in SENSITIVITY_Z_THRESHOLDS:
        raise ValueError(f"detect_anomalies: sensitivity must be one of {sorted(SENSITIVITY_Z_THRESHOLDS)}, got {sensitivity!r}")
    z_threshold = SENSITIVITY_Z_THRESHOLDS[sensitivity]

    series_key = _resolve_series(scope)
    start = date.fromisoformat(date_range["start"])
    end = date.fromisoformat(date_range["end"])

    history_df = _fetch_history(series_key, as_of=end).sort_values("ds").reset_index(drop=True)
    y = history_df["y"].to_numpy(dtype=float)
    shifted = pd.Series(y).shift(1)
    rolling_mean = shifted.rolling(28, min_periods=7).mean().to_numpy()
    rolling_std = shifted.rolling(28, min_periods=7).std().to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(rolling_std > 0, (y - rolling_mean) / rolling_std, 0.0)

    mask = (history_df["ds"].dt.date >= start) & (history_df["ds"].dt.date <= end) & np.isfinite(z) & (np.abs(z) >= z_threshold)
    anomalies = []
    for _, row in history_df.loc[mask].iterrows():
        idx = row.name
        anomaly_type = "spike" if z[idx] > 0 else "drop"
        anomalies.append({
            "series_key": series_key,
            "ds_range": {"start": _iso(row["ds"].date()), "end": _iso(row["ds"].date())},
            "anomaly_type": anomaly_type,
            "severity": float(abs(z[idx])),
            "narrative_facts": [
                f"y={row['y']:.1f} vs trailing-28-day mean {rolling_mean[idx]:.1f} "
                f"(std {rolling_std[idx]:.1f}) -- z={z[idx]:.2f}, sensitivity={sensitivity!r} (threshold {z_threshold})."
            ],
        })
    return anomalies


def run_scenario(scope: dict, drivers: list[dict], base_model_version: str | None = None) -> dict:
    """in: scope, drivers: [{kind: pct|abs, dates: [start,end], magnitude}],
    base_model_version?  out: scenario_id, base vs scenario series, delta
    summary, assumptions list. Sandboxed: computes in memory only, never
    writes to any forecasts table (none exists yet in Phase 1 anyway)."""
    if not drivers:
        raise ValueError("run_scenario: at least one driver is required")

    all_starts = [date.fromisoformat(d["dates"][0]) for d in drivers]
    all_ends = [date.fromisoformat(d["dates"][1]) for d in drivers]
    date_range = {"start": min(all_starts).isoformat(), "end": max(all_ends).isoformat()}

    base = get_forecast(scope, date_range, model_version=base_model_version)
    ds_dates = [date.fromisoformat(d) for d in base["ds"]]
    scenario_yhat = list(base["yhat"])
    assumptions = []

    for driver in drivers:
        kind = driver["kind"]
        d_start, d_end = date.fromisoformat(driver["dates"][0]), date.fromisoformat(driver["dates"][1])
        magnitude = driver["magnitude"]
        if kind not in ("pct", "abs"):
            raise UnsupportedDriverKindError(f"run_scenario: driver kind {kind!r} not implemented in Phase 1 (pct/abs only)")
        for i, d in enumerate(ds_dates):
            if d_start <= d <= d_end:
                scenario_yhat[i] = scenario_yhat[i] * (1 + magnitude) if kind == "pct" else scenario_yhat[i] + magnitude
        assumptions.append(f"{kind} driver of magnitude {magnitude} applied {d_start.isoformat()}..{d_end.isoformat()}")

    scenario_yhat = [max(v, 0.0) for v in scenario_yhat]
    volume_delta = float(sum(scenario_yhat) - sum(base["yhat"]))

    return {
        "scenario_id": str(uuid.uuid4()),
        "series_key": base["series_key"],
        "base_series": {"ds": base["ds"], "yhat": base["yhat"]},
        "scenario_series": {"ds": base["ds"], "yhat": scenario_yhat},
        "delta_summary": {
            "volume_delta_units": volume_delta,
            "volume_delta_pct": volume_delta / sum(base["yhat"]) if sum(base["yhat"]) else None,
        },
        "assumptions": assumptions,
    }


def compare_models(
    scope: dict | None,
    model_version_ids: list[str] | str = "all_candidates",
    protocol_id: str | None = None,
    results_dirs: list[Path] | None = None,
) -> dict:
    """in: scope, model_version_ids | "all_candidates", protocol_id?
    out: side-by-side backtest metrics, gate outcomes, per-segment
    winners. Reads the SAME validated result parquets GATE_REPORT.md is
    built from (engine.backtest.gate_report.collect_results) -- never a
    number computed ad hoc outside that harness. `results_dirs` defaults
    to BASELINE_RESULTS_DIR but is injectable so callers (and tests) are
    not coupled to whatever happens to be on disk.

    Contract decision (02_AGENT_TOOL_CONTRACTS.md design principle 1,
    "the agent narrates facts, it does not invent them", and principle
    4, "fail loud, explain gently"): "all_candidates" with zero results
    available raises ValueError, the same as asking for an explicit
    model_version_ids that doesn't exist. A well-formed comparison
    object with overall_winner=None and empty metrics would let an
    agent state "no candidates to compare" as if that were a real,
    checked finding, rather than the tool having nothing to read --
    exactly the silent-stub pattern this project's non-negotiable rules
    ban elsewhere (fail loud on missing data, never stub)."""
    fold_results = collect_results(results_dirs if results_dirs is not None else [BASELINE_RESULTS_DIR])
    available_model_ids = sorted({mid for fold in fold_results.values() for mid in fold})

    if model_version_ids == "all_candidates":
        if not available_model_ids:
            raise ValueError("compare_models: model_version_ids='all_candidates' but no validated results exist in results_dirs -- nothing to compare")
        selected = available_model_ids
    else:
        selected = list(model_version_ids)
        missing = [m for m in selected if m not in available_model_ids]
        if missing:
            raise ValueError(f"compare_models: no real results for model_version_ids={missing} -- available: {available_model_ids}")

    fold_ids = sorted(fold_results.keys())
    per_model = {}
    for model_id in selected:
        per_fold_wmape = [fold_results[fid][model_id]["wmape"] for fid in fold_ids if model_id in fold_results[fid]]
        per_model[model_id] = {
            "per_fold_wmape": per_fold_wmape,
            "mean_wmape": float(sum(per_fold_wmape) / len(per_fold_wmape)) if per_fold_wmape else None,
        }

    winner = min((m for m in selected if per_model[m]["mean_wmape"] is not None), key=lambda m: per_model[m]["mean_wmape"], default=None)

    return {
        "protocol_id": protocol_id or "P1",
        "model_version_ids": selected,
        "fold_ids": fold_ids,
        "metrics": per_model,
        "gate_outcomes": "see GATE_REPORT.md -- G1/G2/G3 evaluation needs the challenger/comparator, not covered by this table",
        "per_segment_winners": {},  # per-category not populated yet, see GATE_REPORT.md
        "overall_winner": winner,
    }


def get_run_status(job_id: str) -> dict:
    """in: job_id  out: status, progress, partial results if available.
    Phase-1 stand-in: reads engine.service.jobs' in-memory registry
    (no write-tool populates it yet in this phase; tests seed it
    directly) rather than a real async execution layer (Phase 2)."""
    return get_job(job_id)
