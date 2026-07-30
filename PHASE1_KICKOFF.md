# Augurics — Phase 1 Kick-off Brief
## Forecast Engine build: dataset, gates, comparator, and the first two weeks
### Companion to: 01_MODEL_REGISTRY_SPEC.md · 02_AGENT_TOOL_CONTRACTS.md · 03_UX_DESIGN_SYSTEM.md
### Status: Ready to hand to Claude Code · v1.0

---

## 0. Decisions Locked

| Decision | Value | Rationale |
|---|---|---|
| Primary gate dataset | **M5 (Walmart)** | Promotions (SNAP events), prices, calendar, 3,049 products × 10 stores, 5+ years daily. Canonical benchmark; results are externally credible. |
| Secondary datasets | Favorita (Retail validation) · synthesised Wholesale + Manufacturing (Phase 3) | One vertical proven first; profiles validated later. |
| Hosting Phase 1 | Local (Claude Code) only. Railway + Supabase from Phase 2. Modal deferred to Rung 3. | Phase 1 is science, not serving. Near-zero cost. |
| Success criterion | Pre-registered gates G1–G3 below. Immutable from commit 1. | Per Phase E discipline: thresholds never move after seeing numbers. |
| Language/runtime | Python 3.11, DuckDB local store, pytest | Fast local iteration; Postgres schema applied at Phase 2 migration. |

---

## 1. The Pre-Registered Gate Protocol (commit this file first)

Registered before any model is fitted. This section is append-only thereafter.

### Protocol P1 — "M5 Retail Gate"

- **Data:** M5, all series with ≥ 2 years active history (active span = first nonzero sale → dataset end). Report PLs kept and **share of volume retained** (expect sparse-series exclusions; per E3, filters use train data only).
- **Coverage diagnostic first (E1):** month-by-month profile of event/price coverage and mean sales BEFORE any fit. Findings recorded in `experiments.coverage_report`. Any cliff bounds the training window and is documented as a label-integrity decision.
- **Backtest:** rolling origin, 4 folds, 28-day holdout per fold, expanding window. Metrics: WMAPE (primary), bias, MASE, pinball@0.9 (where quantiles exist).
- **Aggregation:** volume-weighted across series; also reported per category and per horizon-week.

### Gates (immutable)

| Gate | Challenger vs comparator | Pass condition |
|---|---|---|
| **G1** | Best Rung 2 model vs **Seasonal naive** | WMAPE improvement ≥ 15% |
| **G2** | Best Rung 2 model vs **ETS (AutoETS)** | WMAPE improvement ≥ 8% |
| **G3** | Best Rung 2 model vs **Bayesian comparator (§3)** | WMAPE improvement > 0 with fold-consistent sign (wins ≥ 3 of 4 folds) |

Comparator corrections are permitted with written justification (E2); thresholds are not. A failed gate is reported as failed.

---

## 2. Week 1–2 Build Tasks (dependency order)

### Week 1 — Data, harness, baselines

| # | Task | Depends on | Done when |
|---|---|---|---|
| 1.1 | Repo scaffold per registry spec §6; CI = pytest on every commit | — | `pytest` green on empty suite |
| 1.2 | M5 ingestion → canonical `PanelData` (long format) in DuckDB; melt sales, join calendar + prices; SNAP/event days mapped to `known_future` | 1.1 | Row counts reconciled vs source; schema contract test passes |
| 1.3 | **Coverage diagnostic** module + run on M5; report committed | 1.2 | `coverage_report.md` in repo; training window bounded if needed |
| 1.4 | Active-span + volume filters; population report (PLs kept, volume retained) | 1.3 | Report shows both figures |
| 1.5 | Backtest harness: rolling-origin splitter, metric suite, gate evaluator reading Protocol P1 from config | 1.2 | Harness unit tests pass incl. leakage test (no future data reaches fit) |
| 1.6 | Rung 1 baselines via `statsforecast`: Naive, SeasonalNaive, AutoETS, Theta — registered, contract-compliant | 1.5 | All pass `test_contract_compliance.py`; baseline WMAPEs committed |

### Week 2 — Comparator, challenger, gates

| # | Task | Depends on | Done when |
|---|---|---|---|
| 2.1 | **Bayesian comparator** per §3 — registered as `bayes_glm_pooled` | 1.6 | Family printed from fitted object; comparator WMAPE committed |
| 2.2 | Feature library v1: calendar, SNAP/event flags, price + price-change, lags/rolling stats, event **adstock decay** (reuse verified Biedronka form), active-span handling | 1.4 | Feature unit tests; no-leakage test per feature |
| 2.3 | Rung 2 challenger: LightGBM via `mlforecast` on feature library; modest pre-registered hyperparameter grid (≤ 12 combos, chosen before results) | 2.2 | Contract suite passes |
| 2.4 | **Gate run:** full P1 protocol, all models, one command (`make gates`) | 2.1, 2.3 | `GATE_REPORT.md` auto-generated: per-fold, per-category, volume-weighted; honest pass/fail |
| 2.5 | The six read-tool functions (get_forecast, explain_forecast, detect_anomalies, run_scenario, compare_models, get_run_status) as tested pure-Python, per tool contracts spec | 2.4 | Each has schema + serialisation test (ISO dates, no NaN) |
| 2.6 | Serialisation boundary tests across the whole API surface | 2.5 | JSON round-trip asserted on every tool output |

**Exit review (end week 2):** G1/G2 expected to pass; G3 attempted and honestly reported. If G3 fails, the iteration backlog is: promo/event feature depth → per-category models → quantile objective → then, and only then, Rung 3.

---

## 3. Bayesian Comparator Specification (`bayes_glm_pooled`)

The arm we must beat, built properly so beating it means something. Under-specifying the comparator would make G3 hollow.

- **Family:** Negative Binomial GLM (count data, overdispersion expected — verify ratio and print family from the fitted object, per E3).
- **Pooling:** hierarchical partial pooling at category level (`pymc` or `bambi`); zero-centred priors on regressor coefficients (mirroring the platform-class regulariser).
- **Regressors:** event/SNAP flags, price (log), day-of-week, month, trend; **active-span offset** = log(mean sales over active span), never full-window mean (E3).
- **Identifiability care:** co-firing event streams checked for separability; near-zero coefficient on a co-firing stream reported as "not separately identifiable", never "no effect" (E5). Coefficient plausibility evaluated at observed exposures, not at feature = 1.0 (E4).
- **Fit budget:** ADVI acceptable for speed with a NUTS spot-check on one category; documented either way.

---

## 4. Repo Skeleton (initialise exactly this)

```
augurics-engine/
  README.md                  # points at this brief + the three specs
  PROTOCOL_P1.md             # §1 verbatim — committed first, append-only
  Makefile                   # ingest / diagnose / gates / test
  pyproject.toml             # statsforecast, mlforecast, lightgbm, duckdb,
                             # pymc or bambi, pandas, pytest
  engine/                    # per registry spec §6
  data/                      # gitignored; M5 download script included
  reports/                   # coverage_report.md, GATE_REPORT.md (generated)
  tests/
```

First three commits, in order: (1) this brief + PROTOCOL_P1.md, (2) scaffold + CI, (3) M5 ingestion. The protocol precedes all code — that is the point.

---

## 5. Claude Code Kick-off Prompt (paste to start the build)

```
You are building the Augurics Forecast Engine, Phase 1. The specification
is in PHASE1_KICKOFF.md, 01_MODEL_REGISTRY_SPEC.md and
02_AGENT_TOOL_CONTRACTS.md in this repo — read all three fully before
writing code.

Non-negotiable rules:
1. PROTOCOL_P1.md is committed first and never edited, only appended.
   Gates G1–G3 are immutable.
2. Coverage diagnostic runs BEFORE any model fit. Its findings bound the
   training window.
3. Every model implements the ForecastModel interface and must pass
   test_contract_compliance.py to be registered. No exceptions, including
   baselines.
4. Fail loud: insufficient history, missing columns, NaN outputs all
   raise. Never stub, never coalesce silently.
5. All dates serialise as ISO strings; every tool output has a JSON
   round-trip test asserting the serialised value.
6. The Bayesian comparator is built to §3 exactly — family printed from
   the fitted object, active-span offsets, identifiability checks.
7. Hyperparameter grids and population filters are declared in config
   before results are seen; report volume retained with every filter.
8. Work the Week 1 / Week 2 task tables in dependency order. Each task
   ends with its tests green and a one-paragraph commit note stating what
   was verified and what remains assumed.

Start with task 1.1. After each task, stop and summarise before
proceeding.
```

---

## 6. Risks & Standing Watch

| Item | Note |
|---|---|
| **RISK (High)** | G3 may not pass in week 2. That is a finding, not a failure — the iteration backlog in §2 is the response. Do not soften the comparator or the gate. |
| **RISK (Medium)** | M5's event structure is simpler than real promo data; a G3 pass here is necessary, not sufficient. Favorita validation follows in Phase 3. |
| **RISK (Low)** | PyMC fit time on full M5. Mitigation: pooled fit per category, ADVI, documented. |
| **OPEN** | Product/entity name check for "Augurics" (trademark/domain) before anything public. |
| **OPEN** | Repo visibility / open-core question: `OnionTerror8/Augurics` audited clean for a public switch (no secrets/tokens/emails/customer references in git history) and doing so would remove a real Kaggle-secret failure class (see `KNOWN_ISSUES.md`) and restore the GitHub Actions compute option -- but visibility is a product/business decision, deliberately deferred here, not decided by default. |
| **NEXT AFTER EXIT REVIEW** | Phase 2: FastAPI + Supabase migration, agent loop, Lovable build sequence from the UX spec §5 — with the two-lens (Horizon/Instrument) architecture folded in. |

---

*Version 1.0 · July 2026 · Prepared by the AI Brain for the Augurics build.*
