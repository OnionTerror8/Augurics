# Augurics — Phase 2 Kick-off Brief
## Agent layer, serving API, Supabase migration, and wiring the two-lens UI to real numbers
### Companion to: 01_MODEL_REGISTRY_SPEC.md · 02_AGENT_TOOL_CONTRACTS.md · 03_UX_DESIGN_SYSTEM.md · EXPORT_CONTRACTS.md · PHASE1_KICKOFF.md
### Status: Ready to execute the day GATE_REPORT.md exists · v1.0

---

## 0. Entry Gate (hard, per Phase 1 discipline)

Phase 2 does not begin until ALL of:
- [ ] GATE_REPORT.md exists with G1/G2/G3 evaluated against the pre-registered thresholds (pass OR honest fail — a failed G3 routes the *engine* into its iteration backlog but does NOT block Phase 2: the API and agent serve whatever champion the registry holds)
- [ ] All four comparator folds + challenger grid validated through load_and_validate_result
- [ ] A champion assigned per segment via the governance flow, human-approved

**Carried-in requirements (already decided, do not relitigate):**
1. External platforms are first-class API clients (EXPORT_CONTRACTS §6.1): auth-scoped keys per consumer from day one.
2. The teal-means-agent rule, propose-then-approve writes, visible activity ≤200ms (optimistic render), fact-cited numbers — all from the tool contracts and UX spec, now binding on implementation.
3. No number reaches the UI except via the serving API reading validated results. The UI never computes.

---

## 1. Architecture Recap (what Phase 2 assembles)

```
Lovable UI (Instrument + Horizon)      External platform clients
        │  HTTPS/SSE                        │  HTTPS (API key)
        ▼                                   ▼
FastAPI service  ──  auth (Supabase JWT | API keys) ── rate limits
   ├── /v1/tools/*      (the six read tools, already built & tested)
   ├── /v1/chat         (agent loop: Claude API tool-calling, SSE)
   ├── /v1/proposals/*  (propose → approve/reject → revert)
   └── /v1/jobs/*       (async: retrain, scenario, backtest status)
        │
Engine (Phase 1, unchanged)          Supabase Postgres
   models · registry · harness  ←──  (migrated schema §3)
        │
DuckDB stays for heavy backtest work; Postgres is the serving store
```

Hosting for Phase 2: **Railway** (FastAPI) + **Supabase** (Postgres/auth). Heavy compute (retrains) stays on the Kaggle/notebook path initially; Modal enters only with Rung 3.

---

## 2. Task Plan — dependency-ordered, done-when conditions

### Week A — Serving foundation

| # | Task | Depends on | Done when |
|---|---|---|---|
| A.1 | Repo: `service/` package scaffold (FastAPI app, settings via env, structured logging), CI extended | Entry gate | `uvicorn` serves /healthz; tests green |
| A.2 | **Supabase project + schema migration**: translate registry spec §5 DDL to Postgres migrations (sqitch/alembic/supabase-cli — pick one, document); tables: products, locations, forecasts, models, model_versions, champions, backtest_results, overrides, scenarios, jobs, audit_log, api_clients | A.1 | Migrations apply clean on a fresh project; rollback tested |
| A.3 | **Results publisher**: one command (`make publish-results`) reads the validated Phase 1 parquets + GATE_REPORT data and upserts into Postgres (forecasts for the champion, backtest_results, champions, gate outcomes). Idempotent, row-count reconciled, audit-logged | A.2 + Entry gate | Re-running changes nothing; counts printed and asserted |
| A.4 | **Auth**: Supabase JWT for UI users; `api_clients` table + hashed API keys for external consumers; per-client scopes (read_forecasts, read_gates, push_disabled_v1) and rate limits | A.1 | Both auth paths integration-tested; unauth’d requests 401 |
| A.5 | **/v1/tools/***: mount the six existing tool functions behind the API. Zero logic changes — routing, auth, and pagination only. Every response already JSON-clean by construction; add contract tests hitting the HTTP layer | A.3, A.4 | All six endpoints return validated real numbers from Postgres; OpenAPI doc generated |

Gate A→B: an external client with an API key can pull the champion forecast and the Gate Ladder numbers over HTTPS. (This IS the Export Contracts §6.1 requirement, proven.)

### Week B — The agent loop

| # | Task | Depends on | Done when |
|---|---|---|---|
| B.1 | **/v1/chat**: server-side loop per tool-contracts spec — system prompt assembled from platform state (champion map, vertical profile, user role) + tool schemas; Claude API streaming; max 8 tool calls/turn; conversation persistence | A.5 | Scripted conversation test: a “why” question triggers explain_forecast and returns cited facts |
| B.2 | **SSE activity events**: every tool invocation emits `{tool_activity}` the moment it STARTS (optimistic), then completion/failure. Latency test asserts first event ≤200ms from tool dispatch | B.1 | Latency test green; events observable via curl |
| B.3 | **Fact citation plumbing**: narrative_facts from tool results carry stable ids; the agent’s reply cites by id; /v1/facts/{id} returns the source payload for the hover popover | B.1 | Every number in a test reply resolves to a fact id; uncited numbers fail the response validator |
| B.4 | **Proposals**: propose_override + propose_model_switch produce proposal rows (schema per tool contracts); /v1/proposals approve|reject|revert with audit rows; approved overrides served as a LAYER over base forecasts (base never mutated) | A.5 | Full lifecycle test: propose → approve → forecast reflects layer → revert → base restored; every step in audit_log |
| B.5 | **Red-team pass**: prompt-injection suite — the agent cannot be induced to call uncatalogued actions, write without approval, or fabricate numbers (response validator rejects uncited figures) | B.1–B.4 | Suite of ≥12 adversarial prompts all safely handled, committed as regression tests |

Gate B→C: the agent answers “why is X dropping?” with cited real numbers, and a proposed override completes its full approved-and-reverted lifecycle through the API.

### Week C — Wiring the shell

| # | Task | Depends on | Done when |
|---|---|---|---|
| C.1 | Lovable: swap sampleData.ts for the API client (Supabase auth session → FastAPI); the SAMPLE DATA banner is removed ONLY on surfaces now fed by /v1 endpoints — any remaining mock surface keeps its badge | Gate A→B | Chart + Gate Ladder render real champion numbers; ladder shows real G1/G2/G3 outcomes |
| C.2 | Companion rail wiring: /v1/chat SSE → activity rows (optimistic), streamed replies, citation popovers hitting /v1/facts, proposal cards driving /v1/proposals | B.2–B.4, C.1 | The three demo components from the readiness prompts operate live end-to-end |
| C.3 | Approvals workspace: proposal queue + audit trail from real tables; Insights inbox fed by a scheduled detect_anomalies job (n8n or supabase cron) posting rows | C.2 | An anomaly appears in the inbox from a scheduled run; audit trail paginates real history |
| C.4 | Horizon lens live: the semantic statement generated from narrative_facts (never free-composed), terrain direction/climate from real champion deltas; Step-back transition carries scope | C.1 | Horizon renders a real statement whose every figure traces to a fact id |
| C.5 | UX Quality Bar audit against the LIVE product: the 7 criteria re-tested for real (90-second first-why test with a fresh user, 200ms trace, AA re-check, keyboard pass) | C.1–C.4 | Checklist committed with evidence per criterion |

Gate C→done: a fresh user, real data, no docs: forecast seen and “why?” answered with citations inside 90 seconds.

### Week D — Hardening & release posture

| # | Task | Done when |
|---|---|---|
| D.1 | Observability: request logging, tool-call metrics, Claude token accounting per conversation (the future pricing telemetry) | Dashboards/queries exist; costs per conversation visible |
| D.2 | Backup/restore for Postgres; environment promotion story (dev → prod project) | Restore rehearsed once |
| D.3 | Rung 0 (incumbent-as-comparator) exposed via API + a demo against M5 (seasonal naive as fake incumbent) — the co-processor teaser | /v1/tools/compare_models accepts an ingested forecast; ladder shows Rung 0 |
| D.4 | Phase 3 go/no-go review: co-processor reference File Adapter scoping, owner items (trademark, visibility, RELEX conflict) revisited | Decision minuted |

---

## 3. Schema Migration Notes (DuckDB → Postgres)

- Registry spec §5 DDL is the source of truth; add `api_clients(id, name, key_hash, scopes jsonb, rate_limit, created_at, revoked_at)`.
- Forecasts table gets a serving index `(product_id, location_id, ds)` and a `layer` column (`base` | `override:{proposal_id}`) — overrides are rows, never updates to base.
- Money/quantity as numeric, never float, in Postgres; dates as date; every timestamp timestamptz.
- The publisher (A.3) is the ONLY writer of base forecasts; proposals are the only writer of override layers; both audit-logged. No other write path exists.
- DuckDB remains the backtest workhorse locally/Kaggle; Postgres is serving-only. No dual-write: publish is a batch upsert from validated parquets.

---

## 4. The Paste-Ready Claude Code Prompt (session one of Phase 2)

```
Phase 2 begins. Read PHASE2_KICKOFF.md, 02_AGENT_TOOL_CONTRACTS.md,
EXPORT_CONTRACTS.md and the registry spec fully before writing code.

Standing rules, inherited and binding:
1. The entry gate is confirmed passed (GATE_REPORT.md exists; results
   validated; champion assigned). State the champion and gate outcomes
   back to me before any code.
2. No number reaches any client except via the serving API reading
   validated, published results. The UI never computes. The publisher
   (task A.3) is the only writer of base forecasts.
3. Writes are propose-then-approve, layered, reverted-able, audited —
   no exceptions, including "convenience" endpoints.
4. Agent replies may not contain uncited figures: implement the
   response validator early (B.3) and make it a hard failure.
5. Optimistic activity events: emit tool_activity at dispatch, not at
   completion. The 200ms latency test is part of B.2's definition of
   done.
6. External platforms are first-class clients from A.4 onward: API
   keys, scopes, rate limits — not an afterthought.
7. Fail loud everywhere; every write path audit-logged; secrets via
   env only, never committed.

Work the task tables in order (A.1 → A.5, gate check, B.1 → B.5, gate
check, C, D). Each task ends with its done-when condition demonstrated
and a one-paragraph commit note stating what was verified and what
remains assumed. Stop and summarise after each task. Start with A.1.
```

---

## 5. Risks & Open Items

| Item | Note |
|---|---|
| **RISK (Med)** | Lovable-generated UI meeting a real API: expect friction in auth/session wiring (C.1). Budget a session; keep the API client in one file Lovable is told never to restructure. |
| **RISK (Med)** | Agent cost drift: token accounting (D.1) lands late but per-conversation caps go in B.1 from the start. |
| **RISK (Low)** | Supabase row-level security vs service-role access: publisher and API use the service role; RLS protects direct client access. Decide once in A.4, document. |
| **OPEN (owner)** | Trademark/domain · repo visibility/open-core · RELEX-connector conflict — all pre-date this brief; D.4 forces the revisit. |
| **DEFERRED** | Rung 3 neural + Modal; the co-processor File Adapter (Phase 3, per EXPORT_CONTRACTS §6); write-scoped API keys (push stays disabled in v1 scopes). |

---

*v1.0 · July 2026 · Executes the day the gates land. Until then: nothing in this brief touches Phase 1.*
