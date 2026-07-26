# Augurics Forecast Engine

Phase 1 build. Start here:

- [`PHASE1_KICKOFF.md`](PHASE1_KICKOFF.md) — decisions, gate protocol, week 1-2 task tables
- [`01_MODEL_REGISTRY_SPEC.md`](01_MODEL_REGISTRY_SPEC.md) — `ForecastModel` interface, registry, backtest/gate schema
- [`02_AGENT_TOOL_CONTRACTS.md`](02_AGENT_TOOL_CONTRACTS.md) — agent tool contracts (Phase 2)
- [`03_UX_DESIGN_SYSTEM.md`](03_UX_DESIGN_SYSTEM.md) — UX design system
- [`PROTOCOL_P1.md`](PROTOCOL_P1.md) — pre-registered gate protocol, append-only, immutable

```
make test        # pytest
make ingest       # M5 ingestion (task 1.2)
make diagnose     # coverage diagnostic (task 1.3)
make gates        # full gate run (task 2.4)
```
