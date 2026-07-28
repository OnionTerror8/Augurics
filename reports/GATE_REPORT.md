# GATE_REPORT — DEFERRED (not abandoned)

This file is a stub. It is the intended output of `make gates` (Protocol
P1 §2, task 2.4): the honest per-fold, per-category, volume-weighted
pass/fail evaluation of G1/G2/G3 against the pre-registered thresholds
in `PROTOCOL_P1.md`, once a real Rung 2 challenger (task 2.3) and the
Bayesian comparator's full backtest (task 2.1) both exist.

Neither exists yet as of this commit, so there is nothing to gate. The
assembly code for this file already exists and is tested
(`engine.backtest.gate_report`, `tests/test_gate_report.py`) — it was
built ahead of the baseline batch finishing so it would be ready.

**Do not confuse this with the current task 1.6 exit artefact.** The
Rung 1 baseline table — per fold series count, volume retained,
volume-weighted WMAPE per model, wall time, and the implied (not yet
evaluated) G1/G2 thresholds — is `reports/BASELINE_TABLE.md`, already
assembled, reviewed, and accepted.

This stub will be replaced with the real `GATE_REPORT.md` when `make
gates` actually runs a full gate evaluation, not before.
