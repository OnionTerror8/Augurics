# Known Issues

Tracked but not yet fixed. Add new entries at the top.

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
