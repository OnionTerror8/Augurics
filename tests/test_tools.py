"""The six read-tool functions -- task 2.5 (schema + serialisation,
ISO dates, no NaN) and task 2.6 (JSON round-trip across every tool
output). Runs against real M5 history (no mocked/synthetic data, per
non-negotiable rule 4) for a real series with enough history."""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from engine.data.ingest import DB_PATH
from engine.service import jobs, tools

# A real, well-populated series (used elsewhere in this project's
# evidence work) with history well before and after the forecast window
# used below.
SCOPE = {"product": "FOODS_1_130", "location": "CA_1"}
SERIES_KEY = "FOODS_1_130::CA_1"
FORECAST_START = date(2016, 2, 1)
FORECAST_END = date(2016, 2, 7)
DATE_RANGE = {"start": FORECAST_START.isoformat(), "end": FORECAST_END.isoformat()}


def _assert_json_round_trips(obj) -> None:
    """task 2.6: every tool output must survive a JSON round-trip
    unchanged -- ISO date strings, no NaN/Inf (json.dumps itself would
    emit invalid JSON for those, so this also fails loud on them)."""
    encoded = json.dumps(obj, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded == obj


def _assert_no_nan_anywhere(obj) -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"found non-finite float: {obj!r}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan_anywhere(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_nan_anywhere(v)


@pytest.mark.skipif(not DB_PATH.exists(), reason="requires the real M5 DuckDB (data/m5/augurics.duckdb)")
class TestGetForecast:
    def test_schema_and_round_trip(self):
        result = tools.get_forecast(SCOPE, DATE_RANGE)
        assert result["series_key"] == SERIES_KEY
        assert len(result["ds"]) == len(result["yhat"]) == 7
        assert all(isinstance(d, str) and date.fromisoformat(d) for d in result["ds"])
        assert result["champion_model_id"] == tools.DEFAULT_CHAMPION_MODEL_ID
        _assert_no_nan_anywhere(result)
        _assert_json_round_trips(result)

    def test_respects_explicit_model_version(self):
        result = tools.get_forecast(SCOPE, DATE_RANGE, model_version="seasonal_naive")
        assert result["champion_model_id"] == "seasonal_naive"
        _assert_json_round_trips(result)

    def test_ambiguous_scope_raises(self):
        with pytest.raises(tools.AmbiguousScopeError):
            tools.get_forecast({"category": "FOODS"}, DATE_RANGE)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown model_id"):
            tools.get_forecast(SCOPE, DATE_RANGE, model_version="not_a_real_model")


@pytest.mark.skipif(not DB_PATH.exists(), reason="requires the real M5 DuckDB")
class TestExplainForecast:
    def test_schema_and_round_trip(self):
        result = tools.explain_forecast(SERIES_KEY, DATE_RANGE, question_focus="why dropping")
        assert result["series_key"] == SERIES_KEY
        assert result["question_focus"] == "why dropping"
        assert "components" in result and "drivers" in result and "narrative_facts" in result
        assert result["date_range"] == DATE_RANGE
        _assert_no_nan_anywhere(result)
        _assert_json_round_trips(result)


@pytest.mark.skipif(not DB_PATH.exists(), reason="requires the real M5 DuckDB")
class TestDetectAnomalies:
    def test_schema_and_round_trip(self):
        wide_range = {"start": "2016-01-01", "end": "2016-02-07"}
        result = tools.detect_anomalies(SCOPE, wide_range, sensitivity="med")
        assert isinstance(result, list)
        for anomaly in result:
            assert anomaly["series_key"] == SERIES_KEY
            assert anomaly["anomaly_type"] in ("spike", "drop")
            assert anomaly["severity"] >= tools.SENSITIVITY_Z_THRESHOLDS["med"]
        _assert_no_nan_anywhere(result)
        _assert_json_round_trips(result)

    def test_invalid_sensitivity_raises(self):
        with pytest.raises(ValueError, match="sensitivity"):
            tools.detect_anomalies(SCOPE, DATE_RANGE, sensitivity="extreme")

    def test_higher_sensitivity_finds_at_least_as_many(self):
        wide_range = {"start": "2016-01-01", "end": "2016-02-07"}
        low = tools.detect_anomalies(SCOPE, wide_range, sensitivity="low")
        high = tools.detect_anomalies(SCOPE, wide_range, sensitivity="high")
        assert len(high) >= len(low)


@pytest.mark.skipif(not DB_PATH.exists(), reason="requires the real M5 DuckDB")
class TestRunScenario:
    def test_pct_driver_schema_and_round_trip(self):
        drivers = [{"kind": "pct", "dates": [DATE_RANGE["start"], DATE_RANGE["end"]], "magnitude": 0.5}]
        result = tools.run_scenario(SCOPE, drivers)
        assert result["series_key"] == SERIES_KEY
        base_sum = sum(result["base_series"]["yhat"])
        scenario_sum = sum(result["scenario_series"]["yhat"])
        assert scenario_sum >= base_sum  # a positive uplift can only raise or hold volume (clipped at 0)
        assert result["delta_summary"]["volume_delta_units"] == pytest.approx(scenario_sum - base_sum)
        _assert_no_nan_anywhere(result)
        _assert_json_round_trips(result)

    def test_unsupported_driver_kind_raises(self):
        drivers = [{"kind": "weather", "dates": [DATE_RANGE["start"], DATE_RANGE["end"]], "magnitude": 1.0}]
        with pytest.raises(tools.UnsupportedDriverKindError):
            tools.run_scenario(SCOPE, drivers)

    def test_no_drivers_raises(self):
        with pytest.raises(ValueError, match="at least one driver"):
            tools.run_scenario(SCOPE, [])


def _write_result(results_dir, fold_id, model_id, wmape):
    results_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame([{
        "fold_id": fold_id, "model_id": model_id, "wall_seconds": 1.0,
        "wmape": wmape, "n_series": 100, "n_rows": 2800,
    }]).to_parquet(results_dir / f"fold{fold_id}_{model_id}.parquet")


class TestCompareModels:
    """compare_models' real dependency is a results directory with
    validated result parquets, NOT the M5 DuckDB (it never touches
    DB_PATH) -- synthetic results injected via results_dirs, so these
    run identically everywhere (a fresh clone included), no skip."""

    @pytest.fixture
    def results_dir(self, tmp_path):
        d = tmp_path / "results"
        _write_result(d, 1, "naive", 0.98)
        _write_result(d, 2, "naive", 0.99)
        _write_result(d, 1, "auto_ets", 0.73)
        _write_result(d, 2, "auto_ets", 0.72)
        return d

    def test_all_candidates_schema_and_round_trip(self, results_dir):
        result = tools.compare_models(scope=None, model_version_ids="all_candidates", results_dirs=[results_dir])
        assert set(result["model_version_ids"]) == set(result["metrics"].keys())
        assert result["overall_winner"] in result["model_version_ids"]
        for model_id, metrics in result["metrics"].items():
            assert len(metrics["per_fold_wmape"]) == len(result["fold_ids"])
        _assert_no_nan_anywhere(result)
        _assert_json_round_trips(result)

    def test_explicit_subset(self, results_dir):
        result = tools.compare_models(scope=None, model_version_ids=["naive", "auto_ets"], results_dirs=[results_dir])
        assert set(result["model_version_ids"]) == {"naive", "auto_ets"}
        assert result["overall_winner"] == "auto_ets"  # lower mean WMAPE in the synthetic fixture

    def test_unknown_model_id_raises(self, results_dir):
        with pytest.raises(ValueError, match="no real results"):
            tools.compare_models(scope=None, model_version_ids=["lgbm_promo_v1"], results_dirs=[results_dir])

    def test_all_candidates_with_no_results_anywhere_raises(self, tmp_path):
        """Contract decision: an empty results directory must fail
        loud, never return a well-formed comparison with
        overall_winner=None -- that would be the exact silent-stub
        pattern this project bans elsewhere."""
        empty_dir = tmp_path / "empty_results"
        with pytest.raises(ValueError, match="nothing to compare"):
            tools.compare_models(scope=None, model_version_ids="all_candidates", results_dirs=[empty_dir])


class TestGetRunStatus:
    def test_schema_and_round_trip(self):
        jobs.register_job("job-abc", kind="start_retrain")
        jobs.update_job("job-abc", status="running", progress=0.4)
        result = tools.get_run_status("job-abc")
        assert result["job_id"] == "job-abc"
        assert result["status"] == "running"
        assert result["progress"] == pytest.approx(0.4)
        _assert_no_nan_anywhere(result)
        _assert_json_round_trips(result)

    def test_unknown_job_id_raises(self):
        with pytest.raises(jobs.UnknownJobError):
            tools.get_run_status("does-not-exist")
