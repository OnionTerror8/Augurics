# Augurics — Export Contracts & Co-Processor Specification
## Phase 3 feature spec: assess, supplement, and surgically improve incumbent platform forecasts
### v1.0 · Deliberately drafted during Phase 1 so Phase 2 builds nothing that assumes the UI is the only consumer

---

## 1. Intent (confirmed with product owner)

Augurics operates in three modes from ONE engine, with complete flexibility between them:

| Mode | Description |
|---|---|
| **Standalone** | Full product: Horizon + Instrument lenses, own data, own forecasts. (Phases 1–2.) |
| **Co-processor** | Assess an incumbent platform's forecast, find its weak segments, prove challengers on exactly those segments, and inject only winning numbers back — surgically, audibly, reversibly. |
| **Headless API** | Platforms or pipelines pull Augurics predictions directly; no Augurics UI in the loop. |

The co-processor mode exists because displacement is a hard sale and supplementation is an easy one:
*"RELEX's regression is weak on newly-introduced products"* → Augurics proves its cold-start challenger on that segment only. *"o9 underperforms on a category"* → Augurics proves a Neural challenger there and the platform absorbs those numbers. The incumbent keeps everything it is good at.

**Explicitly rejected approach:** building models as Python plugins *inside* target platforms. In-platform plugin architectures (runtime boundaries, schema drift, validator loops) are hostile to model development by construction. Augurics inverts the topology: computation stays in a purpose-built external environment; only proven numbers cross the boundary. This is simultaneously the engineering, UX, and IP position (models are served, never shipped).

---

## 2. The keystone: the incumbent forecast is a registry citizen

The target platform's own forecast is ingested as data and registered as a comparator:

- **Rung 0 — "Incumbent"**: the customer's current platform forecast, scored through the identical rolling-origin backtest harness as every Augurics model. It appears on the Gate Ladder as the first rung: *your forecast, measured honestly*.
- All existing machinery applies unchanged: coverage diagnostic on their extract, per-fold populations, volume-weighted WMAPE, per-segment champions, pre-registered gates.
- **The co-processor gate (G0):** an Augurics challenger may write back into a segment only if it beats the incumbent on that segment under a pre-registered protocol (fold-consistent, volume-weighted, threshold agreed with the customer before any fit). Same E2 discipline: thresholds never move after results are seen.
- The Gate Ladder therefore becomes the sales and governance document in one: Rung 0 (their number) → Rungs 1–4 (Augurics'), with the delta per segment quantified and auditable.

---

## 3. The three adapter contracts (per target platform)

An adapter is thin by design. It implements exactly three contracts; everything else is core engine.

### 3.1 Ingest Adapter — their schema → PanelData
```
IngestAdapter:
  discover_schema() -> SchemaReport          # tables/fields found, mapped, missing
  extract(scope, date_range) -> PanelData    # canonical long format
  extract_incumbent_forecast(scope, date_range) -> ForecastSeries  # Rung 0 input
  reconcile() -> ReconciliationReport        # row counts asserted vs source, fail loud
```
Rules: every extract carries a row-count reconciliation (the 59,181,090-rows discipline, generalised). Encodings, delimiters and export quirks are the adapter's problem, never the engine's. The coverage diagnostic runs on every new source BEFORE any fit — a coverage cliff in their data bounds the engagement exactly as it bounded M5.

### 3.2 Result Adapter — ForecastResult → their write surface
```
ResultAdapter:
  write_mode: "override_layer" | "import_file" | "api_push"
  push(results: list[ForecastResult], provenance: Provenance) -> WriteReceipt
  revert(receipt) -> RevertReceipt           # every write is reversible
```
Rules, non-negotiable regardless of platform:
- **Surgical**: writes are scoped to gate-passing segments only; never a wholesale forecast replacement.
- **Layered**: written as corrections/overrides where the platform supports it — the incumbent baseline is never destroyed.
- **Provenanced**: every number carries model_id, model_version, gate_id passed, backtest reference, generated_at. Planners on the target platform must be able to see WHY a number changed (planner-transparency, exported).
- **Idempotent + upsert semantics**: re-pushing a batch cannot duplicate or leave a window of emptiness (no delete-before-insert, ever).
- **Receipted**: every push returns a receipt logged in Augurics' audit_log; revert is one call.

### 3.3 Sync Contract — the operational agreement
```
SyncContract:
  frequency: cron-like schedule per direction (ingest / push)
  freshness_gate: max acceptable source-data age before a push is refused
  failure_policy: fail loud + alert; NEVER silently skip or fabricate
  monitoring: per-run audit row; drift alarm when live accuracy decays vs backtest
  kill_switch: customer-side toggle halts all pushes instantly, reverts optional
```
The drift alarm matters: a challenger licensed by a backtest must keep earning its place live. If its rolling live WMAPE decays past an agreed band vs the incumbent, pushes pause automatically and the segment reverts to the incumbent — champion/challenger governance, running in production.

---

## 4. The agentic schema interview (the differentiating front door)

The onboarding flow for any new platform/customer, run by the Augurics agent:

1. **Schema in**: customer connects or pastes their platform's export schema / sample extract.
2. **Interrogation**: the agent maps fields to PanelData needs — sales history, product/location masters, promotions/events, prices, the incumbent forecast series, the write-back surface — asking targeted questions where mapping is ambiguous ("Is `qty_adj` post-returns? Where do campaign flags live?").
3. **Diagnostics on THEIR data**: coverage diagnostic, history depth, promo-label integrity, population/volume profile — the same reports Phase 1 produces for M5, pointed at their extract.
4. **Feasibility & effort output** (modelled on the plugin-feasibility matrix pattern): a scored report — data readiness per dataset, mappable vs missing fields, candidate weak segments detected from the incumbent's own accuracy profile, recommended challenger rungs per segment, adapter effort estimate (three-point), and the proposed pre-registered G0 protocol for the engagement.
5. **Generated adapter spec**: the concrete field mapping + transform notes that a build then implements — the interview's output IS the adapter's design document.

This converts what is normally a consulting discovery phase into a product feature. No incumbent platform vendor offers step 4 against their own forecast.

---

## 5. Deployment shapes

| Target style | Ingest | Write-back | Notes |
|---|---|---|---|
| **File-based platforms** (exports/imports, SFTP/S3) | Scheduled extract files | Correction/import files | The universal fallback; also the reference adapter (see §6) |
| **API-first platforms** | REST pull | REST push to forecast-override endpoints | Cleanest; sync contract enforced both ways |
| **Plugin-surface platforms** (RELEX-class) | Platform export | Numbers land via the platform's own override/DSL-write surface as a thin connector | The connector carries numbers + provenance ONLY — no model code crosses. See §7 constraint. |

In all shapes: **models are served, never shipped.** Augurics' compute hosts the models; the customer-side artefact is a connector measured in hundreds of lines, not a model runtime.

---

## 6. Build sequencing (deliberate)

1. **Now (free, during Phase 1 batches):** this document; and one Phase 2 requirement — the FastAPI layer treats "external platform" as a first-class client alongside the UI (auth-scoped API keys per consumer, ForecastResult already JSON-clean by construction).
2. **After the gates:** Rung 0 support in the registry (ingest-a-forecast-as-comparator) — small, high-leverage, and immediately demo-able against M5 by treating any Augurics baseline as a fake "incumbent".
3. **Phase 3 proper:** the reference **File Adapter** (CSV/S3) end-to-end — ingest, G0 protocol, surgical push, revert, drift monitor — proving all three contracts against the simplest possible target.
4. **Then one API-first target**, chosen commercially.
5. **The agentic schema interview** productised (it can be prototyped earlier as a consulting accelerator using the same prompts).
6. **Named-platform connectors last**, each a thin implementation of contracts that already exist.

## 7. Open items & constraints (owner-level)

- **RELEX conflict-of-interest**: any RELEX-specific connector is deferred until the owner's employment/IP position is explicitly resolved. If built, it is built against public documentation and customer-side interfaces only. (Companion to the standing trademark and repo-visibility items.)
- **G0 threshold policy**: per-engagement and pre-registered with the customer — Augurics does not self-certify against a bar it sets alone after seeing data.
- **Liability posture**: write-backs change a customer's operational orders; the sync contract's kill switch, revert, and drift alarm are the minimum viable safety story and are not optional in any deployment.
- **Pricing hook**: the tier ladder maps naturally — co-processor pricing per segment-rung deployed (Signals vs Neural vs Frontier supplementation) with the Gate Ladder as the invoice's justification.

---

*v1.0 · July 2026 · Companion to 01_MODEL_REGISTRY_SPEC.md and 02_AGENT_TOOL_CONTRACTS.md. Nothing here alters Phase 1; one requirement lands in Phase 2 (§6.1); the rest activates after GATE_REPORT.md exists.*
