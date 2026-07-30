# Agent Layer — Tool Contracts Specification (Phase 2)
## Hand-to-Claude-Code build spec, v1.0

---

## Design Principles

1. **The agent narrates facts, it does not invent them.** Every number in an agent reply must trace to a tool result. The engine emits `narrative_facts`; the agent composes prose from them.
2. **Read tools execute freely; write tools propose.** Any tool that changes state (override, model switch, retrain promotion) returns a **proposal object** rendered as an Approve/Reject card in the UI. Nothing mutates without human approval in v1.
3. **Every tool call is visible.** The UI shows a live activity trace (tool name, scope, status) while the agent works. No hidden actions — this is the product's trust foundation and directly serves the UX requirement of clarity about what the system is doing.
4. **Fail loud, explain gently.** Tool errors surface as plain-language explanations of what could not be done and why, never silent degradation.
5. **Asynchronous honesty.** Tier-3 tools return a `job_id` immediately; the agent says what it started and when to expect results. It never pretends a training run finished.

---

## The Tool Set (v1: ten tools)

### Read / analyse (auto-execute)

```
get_forecast
  in:  scope {product?, location?, category?}, date_range, model_version? 
  out: forecast series + champion model identity + quantiles if available

explain_forecast
  in:  series_key, date_range, question_focus?   # "why dropping", "promo effect"
  out: Explanation object (components, drivers, narrative_facts)

detect_anomalies
  in:  scope, date_range, sensitivity: low|med|high
  out: list[{series_key, ds_range, anomaly_type, severity, narrative_facts}]
  # also runs on schedule via n8n; scheduled findings post to an Insights inbox

run_scenario
  in:  scope, drivers: [{kind: promo|price|weather|custom, dates, magnitude}],
       base_model_version?
  out: scenario_id, base vs scenario series, delta summary, assumptions list
  # sandboxed: writes only to scenarios table, never to forecasts

compare_models
  in:  scope, model_version_ids | "all_candidates", protocol_id?
  out: side-by-side backtest metrics, gate outcomes, per-segment winners

get_run_status
  in:  job_id
  out: status, progress, partial results if available
```

### Write / act (propose → approve)

```
propose_override
  in:  scope, adjustment {kind: pct|abs|profile, value, dates}, reason
  out: PROPOSAL {impact preview (units, revenue, WMAPE effect on recent
       holdout), affected series count, expiry}
  approve → writes overrides row + audit_log, forecasts re-served with
  override applied as a layer (base forecast never destroyed)

propose_model_switch
  in:  scope (vertical_profile, segment), to_model_version_id, rationale
  out: PROPOSAL {champion vs candidate gate table, risk notes}
  approve → champions table write + audit_log

start_retrain
  in:  model_id, scope, config_overrides?, protocol_id
  out: job_id (tier 2 sync if <30s expected, else queued; tier 3 always queued)
  # completion posts an Insights-inbox card; if result beats champion the
  # agent attaches a propose_model_switch proposal automatically

run_python_action
  in:  action_name ∈ registered catalogue, params
  out: action result or job_id
  # The extensibility hatch. v1 catalogue:
  #   adstock_transform(scope, decay, saturation)
  #   cold_start_forecast(product_id, analogue_strategy)
  #   long_horizon_params(scope, horizon_profile)
  # Catalogue is a registry table: name, description, param schema,
  # tier, write_scope. The agent can ONLY call catalogued actions —
  # never arbitrary code. New actions are added by deployment, not chat.
```

---

## Proposal Object (the approve/reject contract)

```json
{
  "proposal_id": "uuid",
  "kind": "override | model_switch | retrain_promotion",
  "summary": "Uplift Dairy category 10% for 12–18 June promotion",
  "impact_preview": {
    "series_affected": 214,
    "volume_delta_units": 18400,
    "estimated_wmape_effect": "+0.4pp on holdout replay",
    "reversible": true
  },
  "rationale_facts": ["..."],
  "expires_at": "ISO",
  "status": "pending | approved | rejected | expired"
}
```

Rendered as a card in chat AND queued on the Overrides/Approvals page. Approval writes `approved_by`; every proposal, whatever its fate, persists in `audit_log`.

---

## Orchestration Loop (FastAPI service)

```
POST /chat  {conversation_id, message, context: {current_page, selected_scope}}
  → server-side loop:
      1. Build system prompt: platform state summary, active vertical profile,
         champion map, user role, tool schemas
      2. Claude API call with tools (streaming)
      3. On tool_use: execute against engine API, emit SSE event
         {type: "tool_activity", tool, scope, status} to UI immediately
      4. Return tool_result to Claude; loop until final text
      5. Stream final text; persist turn + tool trace to conversations table
  → SSE stream: token deltas + tool_activity events + proposal cards
```

Context injection rule: the UI always sends `current_page` and `selected_scope`, so "why is this dropping?" resolves against what the user is looking at. This single detail is worth more UX than any styling.

Guardrails: max 8 tool calls per turn; tier-3 tools never awaited in-turn; conversation memory summarised beyond 20 turns; all Claude calls logged with token counts for future pricing telemetry.

---

## Tier/Rung Awareness in the Agent

The agent knows the user's current tier context (see UX spec) and behaves accordingly:

- On a lower tab, tools are scoped to that rung's models by default; the agent may *mention* that a higher rung could help ("an N-HiTS model would likely improve this 26-week horizon — that is available in the Neural tier") but does not invoke it. This is the built-in, honest upsell surface for the future pricing model.
- Gate results always show the full comparator ladder regardless of tier, because honesty about baselines is the credibility of the product.

---

## Phase 2 Definition of Done

- [ ] All ten tools implemented, each with a contract test (input validation, output schema, serialisation)
- [ ] Proposal flow end-to-end: chat proposal → card → approve → audit_log → forecast layer applied
- [ ] SSE activity trace visible in UI for every tool call
- [ ] run_python_action catalogue table + three v1 actions registered
- [ ] Scheduled anomaly scan (n8n) posting to Insights inbox
- [ ] Red-team pass: agent cannot be prompted into uncatalogued actions or unapproved writes
- [ ] The FastAPI layer treats "external platform" as a first-class client, not an afterthought: auth-scoped API keys per consumer (co-processor/headless-API modes, see `EXPORT_CONTRACTS.md`), not just session auth for the Augurics UI itself
