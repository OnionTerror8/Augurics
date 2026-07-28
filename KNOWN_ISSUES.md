# Known Issues

Tracked but not yet fixed. Add new entries at the top.

## CLOSED (design catch): NUTS spot-check cost was coupled to fold population size (2026-07-28)

Before launching the comparator's real 4-fold gate-relevant run, timed
the NUTS spot-check against real fold-1 data and found its cost scaled
with the FULL category subset, not a bounded diagnostic sample: FOODS
alone in fold 1 is 22,432,685 rows (12,265 series). Scaling the measured
per-row cost from the 182-series evidence sample (113,398 FOODS rows,
332s for 200 tune + 200 draw NUTS) gives an estimated **~18.2 hours** for
the NUTS spot-check on ONE fold's FOODS category alone, on top of an
estimated ~2.6-2.9h/fold for ADVI itself -- ~20.8h/fold, ~83h (~3.5 days)
for all four folds if run per fold as originally structured.

Caught before any real launch (BASELINE_TABLE.md/comparator_evidence.md
review gated it). Fixed by design, not by brute-forcing through it:

- `engine.models.comparators.NUTS_SPOT_CHECK_MAX_ROWS` caps the
  spot-check to a bounded, seeded, whole-series subsample (~100k rows)
  regardless of fold size -- its job is validating ADVI's approximation
  quality, a property of the model/data distribution, not of how many
  series happen to be in a given fold.
- `fit(..., run_nuts_spot_check=False)` lets per-fold runs skip it
  entirely; the batch orchestrator (`run_comparator_batch.py`) only
  requests it on one designated fold (default fold 1), once, not per fold.
- Gate-relevant numbers are ADVI-only across all four folds: ~2.6-2.9h
  each, ~11h total -- long, but bounded and stated up front, not an
  ~83h near-miss discovered mid-run.

See `tests/test_comparators.py` for the cap/skip/convergence-guard tests.

## CLOSED: Naive fit+predict overhead on the full fold-1 population (2026-07-27)

Closed, not fixed: re-measured through the actual orchestrator
(`run_slow_baselines`) once the AutoETS/Theta batch finished and cores
were free -- `naive` fold 1 took **20.8s** end to end (see
`reports/BASELINE_TABLE.md`), consistent with folds 2-4 (23.1-26.0s).
The original 48.65s figure below was core contention with the
concurrently running AutoETS/Theta batch, not a real defect in Naive's
fit path. No code change was needed.

<details>
<summary>Original entry</summary>

## Naive fit+predict overhead on the full fold-1 population (2026-07-27)

Full fold-1 population (26,304 series, 48.1M train rows), `naive` model:

- `score_model()` total wall time: 48.65s
- `model.fit()`: 32.3s
- `model._predict_all()`: 6.0s
- unaccounted (merge + `wmape()` + DataFrame overhead in `score_model`): ~10s

Naive is arithmetic (last-observed-value repeated across the horizon) --
32s to *fit* it on 48M rows is higher than expected even accounting for
statsforecast's per-series dispatch overhead with `n_jobs=-1`. Worth
profiling later (e.g. whether `StatsForecast.fit()`'s per-series
dispatch/serialisation overhead dominates for models this cheap, or
whether the ~10s unaccounted-for gap is the parquet-backed merge in
`score_model`). Not fixed now: it doesn't block correctness, and Rung 1
baseline WMAPEs from the current implementation are valid regardless of
this overhead.

</details>
