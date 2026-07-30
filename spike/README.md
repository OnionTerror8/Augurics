# Bounded agent-loop spike

This directory predates the Phase 2 entry gate (`PHASE2_KICKOFF.md`
§0, which has not cleared: no `GATE_REPORT.md` with G1/G2/G3
evaluated, no champion assigned). It is a bounded, local-only spike of
the agent orchestration loop and its response-validation discipline
(`02_AGENT_TOOL_CONTRACTS.md` design principle 1), built to de-risk
Phase 2's build, not to begin it.

It is promoted into `engine/service/` (or removed) only after review
when the Phase 2 entry gate actually opens. Nothing here is wired to
a UI, deployed, or reachable outside a local test run.

## Scope, enforced in code

- Reads ONLY already-validated results: the accepted Rung 1 baseline
  results (`data/fold_cache/slow_baseline_results/`, all 4 folds) and
  the quarantine-lifted comparator folds 1-2
  (`data/fold_cache/comparator/results/`) -- comparator folds 3-4 are
  explicitly excluded by an allowlist in `data_scope.py`, regardless
  of whether they exist on disk, because quarantine has not lifted on
  them (`comparator_evidence.md` §8).
- No champion is assigned (`reports/GATE_REPORT.md`: G1/G2/G3 all
  PENDING) -- the system prompt states this verbatim, and the
  validator separately rejects any champion/winner/best claim as a
  hard failure, independent of the system prompt actually being
  followed.
- No deployment, no UI wiring, no Supabase, no SSE server. Activity
  events print to stdout.
