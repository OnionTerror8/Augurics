# UX & Design System Specification
## Tier-based information architecture, agent visibility, and visual identity, v1.0

---

## 1. Information Architecture — The Ladder

The tiers/rungs are the product's organising metaphor AND its future pricing axis. Present them as an ascending **ladder** the user consciously climbs, not hidden machinery.

### Primary navigation: four tier tabs + three workspaces

```
┌────────────────────────────────────────────────────────────┐
│  [LOGO]   Foundations · Signals · Neural · Frontier        │
│           ───────────────────────────────────────          │
│           Workspace:  Forecasts | Scenarios | Approvals    │
└────────────────────────────────────────────────────────────┘
```

**Tier tabs (rungs, user-facing names):**

| Tab | Rung | Contents | Positioning line shown in UI |
|---|---|---|---|
| **Foundations** | 1 — Baselines | Naive, seasonal, ETS, Theta; instant results | "Trusted statistical baselines. Always on, always honest." |
| **Signals** | 2 — ML | Gradient boosting + feature library (promo, adstock, price, weather) | "Machine learning that reads your demand drivers." |
| **Neural** | 3 — NN | N-HiTS, TFT; training jobs, attention explanations | "Deep models for long horizons and complex interactions." |
| **Frontier** | 4 — Foundation models | Chronos zero-shot, cold-start | "Pre-trained intelligence. Forecasts with zero history." |

Rules:
- Every tab has the identical layout (forecast canvas, model cards, gate ladder, chat). Only the model set and accent behaviour change. Users learn the product once.
- Each tab shows the **Gate Ladder** — a persistent strip comparing this tier's champion to the tiers below on WMAPE. The product proves its own value ladder on the user's data. This is the pricing story made visible without a price list.
- Locked tiers (future pricing) render fully but with a frosted overlay and a live teaser: "Neural champion would improve your Dairy segment by an estimated 6.2%". Estimate from a sample backtest, honestly labelled.
- A small tier badge (I–IV, colour-coded) accompanies every forecast, everywhere in the product, so provenance is never ambiguous.

**Workspaces (cross-tier):** Forecasts (the canvas), Scenarios (what-if builder + comparisons), Approvals (proposal queue + full audit trail + Insights inbox from scheduled anomaly scans).

---

## 2. Agent Experience — "Visible Hands"

The requirement: effortless interaction, and total clarity about when the agent acts and what the system is doing. The design principle: **the agent is a colleague at the same desk, not a chatbot in a drawer.**

### 2.1 The Companion Rail (not a modal, not a popup)

A persistent right rail, 380px, collapsible to a 48px strip. Three stacked zones:

```
┌─────────────────────┐
│  ACTIVITY           │  ← live tool trace
│  ◉ explain_forecast │     each tool call appears the moment it
│    Dairy · Jun 1–30 │     starts: name in plain words, scope,
│  ✓ detect_anomalies │     spinner → tick. Click any row to see
│                     │     the raw result. NOTHING is hidden.
├─────────────────────┤
│  CONVERSATION       │  ← chat. Agent replies cite their facts:
│                     │     hoverable superscripts link each number
│                     │     to the tool result that produced it.
├─────────────────────┤
│  PROPOSALS (2)      │  ← pending approve/reject cards, pinned
└─────────────────────┘
```

### 2.2 Agent state grammar (system-wide, non-negotiable)

One visual language for agent activity, used identically everywhere:

- **Thinking**: soft pulse on the companion-rail edge.
- **Acting (read)**: activity row + a subtle highlight sweep over the chart region being analysed. The user literally sees where the agent is looking.
- **Proposing (write)**: proposal card slides in; affected series on the canvas get a dashed "proposed change" overlay in the accent colour, before/after toggle. Nothing solid until approved.
- **Applied**: overlay becomes solid; an entry pins to the audit strip with actor = "Agent, approved by [user]".
- **Working long (Tier 3)**: job chip docks to the top bar with live progress; completion posts to the Insights inbox and gently pulses the chip. Never a blocking spinner.

### 2.3 Interaction affordances

- **Ask-anything anywhere**: select any region of any chart → a floating "Ask" chip appears → question is automatically scoped to that selection. This is the signature interaction.
- **Command palette (⌘K)**: every agent tool invocable as a typed command with structured inputs, for power users who outgrow chat.
- **Undo as a first-class citizen**: applied overrides show "Revert" for their lifetime. Reversibility is stated on every proposal card. Trust follows reversibility.
- **Empty states teach**: each tier tab's empty state is three example questions tailored to that tier, clickable to run.

---

## 3. Visual Identity — "Instrument"

Direction name: **Instrument**. The feel of a precision scientific instrument: calm, exact, quietly confident. Distinct because forecasting tools are either dense enterprise grey or cartoonish dashboard-ware; this is neither.

### 3.1 Foundations

- **Canvas**: near-white warm paper `#FAFAF7`. Not pure white — reduces glare across long planning sessions and is instantly recognisable.
- **Ink**: deep charcoal `#1A1D21` for data and text. No pure black.
- **The one accent**: a single signal colour, electric teal `#00B3A4`, reserved EXCLUSIVELY for agent presence and agent-proposed change. If it is teal, the agent did it. This rule alone delivers the "clear when that occurs" requirement at a glance, and becomes the brand.
- **Tier hues** (badges, gate ladder, tab underline only — never chart data): Foundations stone `#8A8F98`, Signals amber `#E8A13D`, Neural violet `#7C6FF0`, Frontier cyan `#3DC8E8`.
- **Charts**: actuals as thin charcoal line, forecast as slightly heavier line with quantile band at 8% opacity, events as delicate baseline ticks. Maximum data-ink, zero chartjunk, no gradients, no drop shadows.

### 3.2 Typography

- **Data & UI**: Inter (or Söhne if licensed later). Tabular numerals ON everywhere numbers appear — aligned digits read as precision.
- **Display**: one distinctive serif for page titles and the wordmark, e.g. Tiempos or Fraunces, sparingly. The serif-on-instrument contrast is the "iconic" note; everything else stays strictly functional.
- Type scale: 13px base UI, 12px dense tables, 20/28px titles. Generous 1.5 line height in chat.

### 3.3 Space, shape, motion

- 8px spacing grid; 12px card radius; 1px hairline borders `#E7E5DF` instead of shadows. Flat, crisp, printed-instrument feel.
- Motion budget: 150–200ms ease-out, and motion ONLY for agent-state changes (the pulse, the sweep, the card slide). Static content never animates. Scarcity of motion is what makes agent activity unmistakable.
- Dark mode from day one (planners work early hours): `#101214` canvas, same accent discipline.

### 3.4 The iconic marks

1. **The teal rule**: one colour that always and only means "agent". Ownable, defensible, memorable.
2. **The Gate Ladder** motif: the ascending four-step comparison strip, also the logo geometry (four ascending bars, top bar teal).
3. **The highlight sweep**: the agent visibly "reading" the chart. Nobody else does this; it demos unforgettably.

---

## 4. UX Quality Bar (acceptance criteria)

- [ ] A first-time user generates a forecast and asks "why?" within 90 seconds, no documentation
- [ ] Every agent action is visible in the activity trace within 200ms of starting
- [ ] Every number in an agent reply traces to a hoverable fact source
- [ ] No write occurs without an approval card; every approval is revertible and audit-visible
- [ ] Tier provenance badge visible on every forecast surface
- [ ] All charts readable at 100% zoom on a 13" laptop; core flows keyboard-navigable
- [ ] Lighthouse accessibility ≥ 95; contrast AA minimum including on tier hues

---

## 5. Lovable Prompt Sequence (build order)

Feed one prompt per session, in this order, referencing this document's tokens:

1. **Design system shell**: tokens above as Tailwind config, app frame with tier tabs + workspace nav + collapsed companion rail, dark/light.
2. **Forecast canvas**: chart component (actuals/forecast/quantile band/event ticks), scope selector, tier badge, gate ladder strip. Data from Supabase only.
3. **Companion rail**: three zones, SSE hookup to `{AGENT_URL}/chat`, activity rows, fact-linked citations, proposal cards with approve/reject.
4. **Approvals workspace**: proposal queue, audit trail table, Insights inbox.
5. **Scenarios workspace**: driver builder, base-vs-scenario comparison view.
6. **Selection-to-Ask** interaction + command palette.
7. **Locked-tier treatment** + teaser estimates.

Standing instruction to include in every Lovable prompt: "Never compute forecasts or agent logic in the frontend. All data via Supabase; all agent interaction via the SSE chat endpoint. Follow the Instrument design tokens exactly; the teal accent is reserved for agent activity only."
