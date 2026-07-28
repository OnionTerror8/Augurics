# Known Issues

Tracked but not yet fixed. Add new entries at the top.

## DEFERRED (verified infeasible on this laptop): challenger (Rung 2 lgbm) training, task 2.3 (2026-07-28)

Checked feasibility BEFORE training anything, per the same discipline
as the comparator (see the memory entries below). Built the real
feature matrix (`engine.features.library.build_feature_set`) on fold
1's actual 48,110,016-row train frame, applying the same fixes that
worked for the comparator (category dtype for `series_key`/`category`,
int8 for flag columns, float32 for engineered numerics, drop the raw
`event_name_1`/`event_name_2` string columns once `event_flag` is
derived from them). Measured via `df.memory_usage(deep=True)`:
**3.03GB** for the fully-optimized feature matrix (versus 9.58GB
unoptimized) -- 63.06 bytes/row, now the calibration constant in
`engine.models.ml.OPTIMIZED_FEATURE_BYTES_PER_ROW`.

Adding LightGBM's own gradient/hessian buffers and a conservative 1.5x
multiplier for mlforecast's internal preprocessing copy gives an
estimated **5.13GB** peak for a single fit at fold-1 scale
(`engine.models.ml.check_challenger_fit_memory`). This laptop's
available RAM has ranged 1.7-7.4GB over the course of tonight's work
(mostly clustering 2-5GB under normal desktop load) -- unlike the
comparator's post-fix 3.90GB estimate, which was a close call against
this range, the challenger's 5.13GB does not reliably clear even the
best observed windows at the guard's 50%-of-available budget. Checked
live: FAILS at 4.17GB available (budget 2.08GB).

Not a case for relaxing `MAX_CHALLENGER_FIT_MEMORY_FRACTION` -- per
instruction, the guard stands. Challenger training (task 2.3) is
deferred alongside the comparator batch (G3), pending the same VM
(`PROVISIONING.md`). `GATE_REPORT.md` marks G1 and G2 PENDING for this
reason, not evaluated as a false pass/fail. `tests/test_ml.py` covers
the guard itself, which is real, tested, and ready to run the moment
compute is available -- nothing here blocks re-attempting training as
soon as the VM (or more free RAM on this laptop) exists.

## FIXED: the memory estimator's own 3.90GB figure never accounted for the base DataFrame (2026-07-28)

Discovered while measuring real numbers for the GitHub Actions
feasibility check below, not from a failure -- worth catching before
it became one. `_check_fit_memory`/`_estimate_fit_memory_bytes` only
ever estimated the arrays `fit()` builds for PyMC (continuous design
matrix, index arrays); it never accounted for the input pandas
DataFrame itself. Measuring fold 1's actual cached train frame
(`data/fold_cache/comparator/fold1_train.parquet`, 48.1M rows) with
`df.memory_usage(deep=True)` gave **7.19GB** for the base 7-column
frame alone -- `series_key` and `category` as object-dtype string
columns accounted for 5.94GB of that (pandas stores each string as a
separate Python object). `fit()`'s first line, `df = panel.df.copy()`,
would have deep-copied those columns again, doubling it to ~14.4GB
before any feature engineering. On a 16GB VM (`PROVISIONING.md`) that
would have been a genuinely close call, not the comfortable margin the
3.90GB PyMC-array estimate implied.

Fixed at the root, not by inflating the estimate: `fit()` now does
`df = panel.df.copy(deep=False)` and immediately casts `series_key` and
`category` to pandas `category` dtype (each column collapses to a
small integer-code array plus one copy of the distinct values).
Measured effect: base DataFrame footprint 7.19GB -> **1.40GB**; RSS
during the copy+cast didn't move at all (confirmed via `resource.
getrusage`). `_run_nuts_spot_check`'s series-key subsampling also had
to change (`rng.shuffle` on a `Categorical` isn't guaranteed correct --
pandas warns "not a subclass of Sequence"; fixed by converting to a
plain numpy array first). All 147 tests still pass.

## RULED OUT (verified, not assumed): GitHub Actions private-repo runners cannot run the comparator batch (2026-07-28)

Considered as an alternative to a manually-provisioned VM (`PROVISIONING.md`)
for the real 4-fold comparator batch. Verified directly rather than
assumed from docs:

- This repo (`OnionTerror8/Augurics`) is **private**, so it does NOT
  get the free 4-vCPU/16GB public-repo runner upgrade -- confirmed via
  `gh repo view --json visibility`.
- A throwaway `workflow_dispatch` probe job (`free -h`, `/proc/meminfo`,
  `psutil.virtual_memory()`, run once and deleted) confirmed the actual
  standard private-repo runner: **2 vCPU, 8.3GB total RAM, 7.4GB
  available at job start** (before installing any project
  dependencies).
- Measuring fold 1's actual cached train DataFrame
  (`data/fold_cache/comparator/fold1_train.parquet`, 48.1M rows) with
  `df.memory_usage(deep=True)` gives **7.19GB** for the base 7-column
  frame alone -- dominated by `series_key` and `category` as
  object-dtype string columns (3.26GB + 2.68GB). `fit()`'s first line
  (`df = panel.df.copy()`) doubles this to ~14.4GB immediately, before
  any feature engineering, the holdout frame, PyMC's own graph, or the
  `_check_fit_memory` estimate (3.90GB) enter the picture.

Not a close call to tune `MAX_FIT_MEMORY_FRACTION` around: the runner
would be OOM-killed on the DataFrame copy alone, at roughly half the
RAM this workload needs just to hold the data. GitHub's paid "larger
runners" (up to 64 vCPU/256GB) exist but require a Team/Enterprise
plan with billing, not available on the Free plan this repo is on --
functionally the same cost/commitment tradeoff as `PROVISIONING.md`'s
VM route, with less control over the environment. Decision: proceed
with the VM route as already documented.

## CLOSED, working as designed: pre-fit memory estimator's first live catch forced a move to cloud compute (2026-07-28)

The real fold-1 launch (post-OOM-refactor, see the entry below) hit
`_check_fit_memory`'s `InsufficientMemoryError` immediately (23.2s, not
another ~577s SIGKILL): estimated fit footprint 3.90GB against this
machine's available RAM, which fluctuated 1.70-3.14GB across several
rechecks over ~20 minutes of closing applications and never cleared the
budget (50% of available). Swap stayed pinned at 6-13GB used throughout
regardless of how much was freed at the app level -- this laptop's
baseline memory pressure (Chrome/Slack/Teams/Word/Outlook, all normally
open) is high enough that a bare 4GB, single-fit budget is not reliably
available interactively.

This is not a defect in the guard -- it is the guard doing exactly its
job: refusing loud, with numbers, before wasting a multi-hour ADVI fit
on something that would have SIGKILLed partway through again. No change
to `MAX_FIT_MEMORY_FRACTION` or the estimator. Decision: the real 4-fold
batch moves to a provisioned cloud VM (16GB+ RAM, see `PROVISIONING.md`)
rather than continuing to compete with this machine's normal desktop
load. Fold 1's pre-flight check on a 16GB box has ~4x headroom against
the 3.90GB estimate.

## CLOSED (design catch): dense one-hot dow/month encoding caused a real fold-1 OOM (2026-07-27)

Launching the real comparator batch on fold 1 (48.1M rows) was
SIGKILLed (exit -9) after 577.3s -- far too early for the ~2.6h expected
ADVI fit. Diagnosed via `sysctl vm.swapusage` (13.0/14.3GB swap used)
and `psutil.virtual_memory()` (~1.5GB available, 91.3% used) at the time
of the kill. Root cause: `_build_features()`'s `pd.get_dummies`
one-hot encoding of day-of-week (7 levels) and month (12 levels)
materialised 19 dense float64 columns x 48.1M rows (~7.3GB for that
block alone, before pandas' typical 2-3x construction/copy overhead).

Fixed by design, not patched: day-of-week and month became
integer-indexed categorical effects inside the PyMC model itself
(`beta_dow[cat_idx, dow_idx]`, `beta_month[cat_idx, month_idx]`) instead
of dense dummy columns -- confirmed via a standalone pytensor test that
this performs the identical elementwise gather a dense-dummy matmul
would. Continuous features moved to float32; category/dow/month index
arrays use `int8`. A pre-fit memory estimator
(`_estimate_fit_memory_bytes` / `_check_fit_memory` /
`InsufficientMemoryError` in `engine/models/comparators.py`) now refuses
to start a fit whose estimated footprint exceeds `MAX_FIT_MEMORY_FRACTION`
(50%) of currently-available RAM, turning a ~10-minute SIGKILL into a
one-second refusal with numbers (see the entry above for its first live
catch).

An equivalence proof (rerunning the 182-series evidence fit with the
new parameterisation, same seed) incidentally surfaced a second,
unrelated instability in the `trend` feature during this same
investigation -- see `reports/comparator_evidence.md` §7 for the full
writeup and fix (trend rescaled to fraction-of-training-span, its prior
tightened). That fix, not this one, is what made the comparator
reliably beat seasonal naive on the evidence sample; this entry's fix
(indexed categoricals, float32, int8, memory estimator) addressed the
OOM only and was confirmed equivalent to the pre-refactor model once the
trend fix was applied on top.

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
