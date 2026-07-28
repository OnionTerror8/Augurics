# BASELINE_TABLE — Rung 1 baselines, task 1.6 exit artefact

## Per fold

### Fold 1 (train end 2016-01-31)

Eligible series: **26,304 / 30,490** (86.3%). Volume retained: **60,324,492 / 62,160,048** units (97.0%).

| Model | WMAPE | Wall time |
|---|---|---|
| naive | 0.9832 | 20.8s |
| seasonal_naive | 0.8572 | 23.0s |
| auto_ets | 0.7302 | 1781.6s |
| theta | 0.7338 | 1184.5s |

### Fold 2 (train end 2016-02-28)

Eligible series: **26,799 / 30,490** (87.9%). Volume retained: **61,743,488 / 63,335,716** units (97.5%).

| Model | WMAPE | Wall time |
|---|---|---|
| naive | 0.9965 | 23.7s |
| seasonal_naive | 0.8656 | 22.7s |
| auto_ets | 0.7300 | 2319.8s |
| theta | 0.7352 | 874.4s |

### Fold 3 (train end 2016-03-27)

Eligible series: **27,366 / 30,490** (89.8%). Volume retained: **63,230,200 / 64,511,783** units (98.0%).

| Model | WMAPE | Wall time |
|---|---|---|
| naive | 0.9658 | 26.0s |
| seasonal_naive | 0.8778 | 23.8s |
| auto_ets | 0.7389 | 2183.0s |
| theta | 0.7427 | 1053.3s |

### Fold 4 (train end 2016-04-24)

Eligible series: **27,684 / 30,490** (90.8%). Volume retained: **64,504,065 / 65,695,409** units (98.2%).

| Model | WMAPE | Wall time |
|---|---|---|
| naive | 0.9386 | 23.1s |
| seasonal_naive | 0.8516 | 25.9s |
| auto_ets | 0.7105 | 2413.3s |
| theta | 0.7144 | 943.7s |

## Volume-weighted summary across all four folds

Each fold's WMAPE is weighted by that fold's own holdout volume (same eligible population scored by every model within a fold) -- this is the number any gate evaluation will reference.

| Model | Volume-weighted WMAPE | Total wall time (all folds) |
|---|---|---|
| naive | 0.9704 | 93.5s |
| seasonal_naive | 0.8630 | 95.3s |
| auto_ets | 0.7272 | 8,697.7s |
| theta | 0.7313 | 4,055.9s |

## Provenance

Every figure above comes from validated result parquets (`engine.backtest.run_baselines.load_and_validate_result`, schema-checked by content at assembly time) produced by the single orchestrator path (`engine.backtest.run_slow_baselines` for auto_ets/theta, the same `score_model`/`_fit_predict_merge` core for naive/seasonal_naive) -- no number here was computed ad hoc outside that harness.

## Implied gate arithmetic (stated, not evaluated)

No Rung 2 challenger exists yet (task 2.3), so no gate is evaluated here. Protocol P1 (immutable) defines G1's comparator as **Seasonal Naive** and G2's as **AutoETS** -- restated as concrete WMAPE thresholds:

- **G1**: challenger WMAPE must be <= `0.85` x 0.8630 (Seasonal Naive summary) = **0.7335** -- confirmed against PROTOCOL_P1.md's text (G1's comparator is Seasonal Naive). Discrepancy flag closed.
- **G2**: challenger WMAPE must be <= `0.92` x 0.7272 (AutoETS summary) = **0.6690**