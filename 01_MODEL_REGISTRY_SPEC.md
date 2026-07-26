# Forecast Engine — Model Registry Specification (Phase 1)
## Hand-to-Claude-Code build spec, v1.0

---

## Purpose

A single registry through which every forecasting model — baseline to neural to foundation — is trained, backtested, promoted, and invoked. The registry is the enforcement point for the accuracy gates and the object the agent layer and UI both consume. No model reaches a user except through this registry.

---

## 1. The Common Model Interface

Every model implements `ForecastModel`. No exceptions, including baselines.

```python
class ForecastModel(ABC):
    """All models, all rungs, one contract."""

    # --- Identity ---
    model_id: str            # e.g. "seasonal_naive", "lgbm_promo_v2", "nhits_long"
    display_name: str        # UI-facing name
    rung: int                # 1=Baseline, 2=ML, 3=Neural, 4=Foundation
    tier: int                # compute tier: 1=in-db, 2=python-svc, 3=gpu/heavy
    supports_quantiles: bool
    supports_covariates: bool
    min_history_days: int    # fail-loud threshold, checked before fit
    max_horizon_days: int

    # --- Core contract ---
    def fit(self, panel: PanelData, config: ModelConfig) -> FitResult: ...
    def predict(self, horizon: int, future_covariates: PanelData | None) -> ForecastResult: ...
    def explain(self, series_key: SeriesKey, date_range: DateRange) -> Explanation: ...
    def backtest(self, panel: PanelData, protocol: BacktestProtocol) -> BacktestResult: ...

    # --- Serialisation ---
    def save(self, artefact_uri: str) -> None: ...
    @classmethod
    def load(cls, artefact_uri: str) -> "ForecastModel": ...
```

### Contract rules (fail loud, per Module 08 discipline)

1. `fit` raises `InsufficientHistoryError` if any series is below `min_history_days`. Never silently fit on garbage.
2. `predict` output is range-checked: non-negative for demand, NaN forbidden (raise, do not coalesce).
3. `explain` must always return something renderable. Minimum viable: component decomposition (level / trend / seasonality / covariate effects / residual). TFT returns attention-derived driver weights; tree models return SHAP values; baselines return their rule in words. An empty explanation is a contract violation.
4. All dates cross service boundaries as **ISO strings**. Never raw date objects across any serialisation boundary (the marshalling lesson, applied to our own product).
5. Models are pure with respect to the database: they receive `PanelData` in, return results out. All I/O lives in the service layer, not the model. (Same responsibility split as TqlScript-vs-Python.)

---

## 2. Canonical Data Objects

```python
PanelData:
    # Long-format panel, the ONLY data shape models see
    series_key: (product_id, location_id)      # composite key
    ds: date                                    # ISO
    y: float                                    # target (sales/demand)
    static_features: dict                       # vertical, category, price_band...
    known_future: dict                          # promo flags, calendar, price plans
    observed_past: dict                         # weather actuals, stockout flags

ModelConfig:
    hyperparams: dict
    feature_set: str          # named feature-library profile, e.g. "retail_promo_v1"
    vertical_profile: str     # "retail" | "wholesale" | "manufacturing"
    horizon: int
    quantiles: list[float]    # e.g. [0.1, 0.5, 0.9]; empty = point forecast

ForecastResult:
    series_key, ds, yhat, yhat_quantiles: dict[str, float]
    model_id, model_version, generated_at, job_id

Explanation:
    series_key, date_range
    components: dict[str, timeseries]     # decomposition
    drivers: list[{name, direction, magnitude, confidence}]
    narrative_facts: list[str]            # atomic facts the agent turns into prose
```

`narrative_facts` is deliberate: the engine emits verifiable atomic facts ("promo_flag active 12–18 June", "seasonal component contributes +23%"); the agent composes language from facts, never invents numbers.

---

## 3. Backtest Protocol & Gates (pre-registered, immutable per experiment)

```python
BacktestProtocol:
    protocol_id: str
    method: "rolling_origin"        # k folds, expanding window
    n_folds: int = 4
    holdout_days: int
    metrics: ["wmape", "bias", "mase", "pinball@0.9"]
    coverage_diagnostic: bool = True   # coverage-cliff scan BEFORE any fit (Phase E1)
    gates: list[Gate]

Gate:
    gate_id: str          # "G1", "G2", "G3"
    comparator_model_id: str
    metric: "wmape"
    pass_condition: str   # e.g. "improvement >= 0.15"
    scope: str            # dataset / vertical it applies to
```

Rules carried over from Phase E discipline:

- Thresholds are pre-registered and never lowered after seeing results. Comparators may be corrected if invalid; the threshold survives unchanged.
- The coverage-cliff diagnostic runs before any fit and its findings are stored on the `experiment` record. Unlabellable periods are excluded from training and documented as label-integrity decisions.
- Model family is reported from the fitted object, never from config strings.
- Every backtest reports population filters with volume retained.

---

## 4. Champion / Challenger Governance

- The registry holds exactly one **champion** per `(vertical_profile, segment)` scope. Segments default to category level.
- Any model run that beats the champion on the protocol becomes a **challenger candidate**. Promotion is never automatic: the agent proposes, a human approves (audit-logged).
- Champions are versioned; rollback is one registry write.
- Rung 1 baselines are permanent and can never be deleted, only superseded as champion.

---

## 5. Database Schema (Supabase / Postgres)

```sql
-- Core data
products(id, code, name, category, vertical, attrs jsonb)
locations(id, code, name, type, region, attrs jsonb)
sales_actuals(product_id, location_id, ds date, qty numeric, revenue numeric,
              flags jsonb)                       -- stockout, return, etc.
events(id, name, kind, ds_start, ds_end, scope jsonb, magnitude jsonb)
                                                 -- promos, weather, holidays

-- Registry
models(model_id pk, display_name, rung int, tier int, capabilities jsonb,
       created_at)
model_versions(id pk, model_id fk, version, artefact_uri, config jsonb,
               status text,                       -- draft|candidate|champion|retired
               created_at)
experiments(id pk, protocol jsonb, coverage_report jsonb, created_at, created_by)
backtest_results(id pk, experiment_id fk, model_version_id fk,
                 metrics jsonb, per_fold jsonb, gates_evaluated jsonb,
                 volume_retained numeric)
champions(vertical_profile, segment, model_version_id fk, promoted_at,
          promoted_by, superseded_at,
          primary key (vertical_profile, segment, promoted_at))

-- Forecast outputs
forecasts(id pk, model_version_id fk, product_id, location_id, ds date,
          yhat numeric, quantiles jsonb, generated_at, job_id)
overrides(id pk, forecast_scope jsonb, adjustment jsonb, reason text,
          proposed_by text,                       -- 'agent' | user_id
          approved_by, applied_at, impact jsonb)  -- full audit trail
scenarios(id pk, name, definition jsonb, base_model_version_id fk,
          result_summary jsonb, created_by, created_at)

-- Compute
jobs(id pk, kind text,                            -- fit|backtest|predict|scenario
     tier int, status text,                       -- queued|running|succeeded|failed
     payload jsonb, progress jsonb, result_ref, error text,
     created_at, started_at, finished_at)
audit_log(id pk, actor text, action text, object_ref, detail jsonb, at timestamptz)
```

Storage rules: never delete-before-insert on `forecasts`; supersede by `generated_at` and prune asynchronously. Every agent-initiated write lands in `audit_log`.

---

## 6. Service Layout (repo structure for Claude Code)

```
engine/
  models/            # one module per model, registered via decorator
    baselines.py     # rung 1: naive, snaive, ets, theta (statsforecast)
    ml.py            # rung 2: lgbm via mlforecast + feature library
    neural.py        # rung 3: nhits, tft (neuralforecast) — Phase 3+
    foundation.py    # rung 4: chronos wrapper — Phase 2
  features/
    library.py       # adstock, cold_start, calendar, promo streams
    profiles.py      # retail / wholesale / manufacturing feature profiles
  backtest/
    protocol.py, gates.py, coverage.py
  registry.py        # champion/challenger state machine
  service/
    api.py           # FastAPI: /fit /predict /backtest /explain /jobs
    queue.py         # tier-2 sync vs tier-3 async dispatch
tests/
  test_contract_compliance.py   # EVERY registered model passes the same suite
  test_serialisation_boundary.py # assert on SERIALISED values (ISO dates, no NaN)
  test_gates.py
```

`test_contract_compliance.py` is the keystone: a parametrised suite that loads every registered model and asserts the full interface, including that `explain` is never empty and `predict` output serialises cleanly to JSON. A model that cannot pass this suite cannot be registered.

---

## 7. Phase 1 Definition of Done

- [ ] Rung 1 + Rung 2 + Bayesian comparator registered and passing contract suite
- [ ] Coverage diagnostic implemented and run on the demo dataset
- [ ] Rolling-origin backtest produces G1/G2/G3 evaluations, honestly reported
- [ ] Champion assigned per segment via governance flow (manual approval)
- [ ] FastAPI endpoints live, all responses JSON-serialisation tested
- [ ] Every write path audit-logged
