# Kaggle Notebooks provisioning — comparator + challenger batch (task 2.1/2.3)

Why: this laptop's memory estimator correctly refuses both the
comparator (3.90GB estimate) and the challenger (5.13GB estimate) under
normal desktop load (see `KNOWN_ISSUES.md`). Kaggle Notebooks' CPU
runtime is **32GB RAM, 12-hour session limit**, background execution
persists after closing the browser ("Save & Run All (Commit)"),
verified before committing to this path -- not assumed. M5 is a Kaggle
competition dataset, so it attaches directly to a notebook, eliminating
`PROVISIONING.md`'s data-transfer step entirely.

`PROVISIONING.md` (the VM route) is kept as a fallback in case Kaggle
disappoints in practice; this is the current preferred path.

## 0. One-time setup (your action, everything else is paste-and-click)

1. Confirm/create a free Kaggle account.
2. Accept the M5 Forecasting - Accuracy competition rules (same
   requirement as the local Kaggle API download earlier in this
   project) at
   `kaggle.com/competitions/m5-forecasting-accuracy/rules`.
3. Create a GitHub Personal Access Token with `repo` scope (this repo
   is private -- the notebook needs it to `git clone`). Add it to the
   notebook as a Kaggle **Secret** named `GITHUB_TOKEN` (Add-ons ->
   Secrets, in the notebook editor).
4. Create a new Kaggle Notebook, CPU-only (no GPU/TPU needed). Under
   "Add Data", search and attach the **m5-forecasting-accuracy**
   competition dataset.

Everything below is cell content to paste into that notebook, in order.

## Cell 1 — clone at HEAD, install

```python
import os
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
os.system(f"git clone https://{token}@github.com/OnionTerror8/Augurics.git /kaggle/working/augurics-engine")
os.chdir("/kaggle/working/augurics-engine")
os.system("pip install -e . -q")
print(os.popen("git rev-parse HEAD").read())  # confirms exactly which commit is running
```

## Cell 2 — attach M5 data, ingest, reconcile (fails loud on mismatch)

```python
import shutil
from pathlib import Path

# Kaggle mounts the attached competition dataset read-only under
# /kaggle/input/<dataset-slug>/ -- copy the 3 files ingest.py needs into
# the repo's expected location. No download step, no Kaggle API auth
# inside the notebook: the dataset is already attached.
src = Path("/kaggle/input/m5-forecasting-accuracy")
dest = Path("data/m5")
dest.mkdir(parents=True, exist_ok=True)
for name in ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"):
    shutil.copy(src / name, dest / name)

result = os.system("python -m engine.data.ingest")
assert result == 0, "ingest failed -- row-count reconciliation raised, see output above"

# Belt-and-suspenders: engine.data.ingest.ingest() already asserts this
# internally (fails loud, raises ReconciliationError on any mismatch,
# same check used locally) -- re-asserted explicitly here too, against
# the exact committed figures, so a silent pass on Kaggle's copy of the
# data is never possible even if ingest.py's own check were bypassed.
import duckdb

con = duckdb.connect("data/m5/augurics.duckdb", read_only=True)
row_count = con.execute("SELECT COUNT(*) FROM panel").fetchone()[0]
con.close()
EXPECTED_ROWS = 30_490 * 1_941  # 59,181,090 -- committed figure, reviewed in this project's earlier sessions
assert row_count == EXPECTED_ROWS, f"Kaggle's ingested panel has {row_count:,} rows, expected {EXPECTED_ROWS:,} -- STOP, do not proceed"
print(f"Row-count reconciliation OK: {row_count:,} rows")
```

## Cell 3 — verify the environment

```python
result = os.system("make test")
assert result == 0, "tests failed on this Kaggle environment -- do not proceed"
```

All tests must pass green here, same bar as locally, before any real
run -- if something about the Kaggle environment differs (a wheel that
built differently, a library version mismatch), this is where it
surfaces, not mid-fold.

## Cell 4 — Notebook A only: comparator, fold 1 (bounded NUTS + predictive-agreement gate)

```python
result = os.system("python -m engine.backtest.run_comparator_batch")
assert result == 0, "comparator batch failed or a gate did not clear -- see run_log.txt"
```

Unchanged from `PROVISIONING.md`: fold 1 runs with the bounded NUTS
spot-check and the decisive predictive-agreement check (2% relative
WMAPE tolerance); folds 2-4 are ADVI-only. If fold 1's
`PredictiveAgreementError` fires, the batch halts before folds 2-4 --
same quarantine rule as always, log this run in `comparator_evidence.md`
as the first ADVI/NUTS predictive comparison on the fixed model,
superseding the §6 figures for licensing purposes (per the review-two
instruction already on record).

## Cell 5 — Notebook B only: challenger, 4 folds x the pre-registered 12-combo grid

Do the bounded first-combo check described in "Session-length" below
BEFORE running this cell for the full grid.

```python
result = os.system("python -m engine.backtest.run_challenger_batch")
assert result == 0, "challenger grid-search batch failed -- see run_log.txt"
```

Writes the best combo's (lowest mean WMAPE across folds, see
`run_challenger_batch.py`'s docstring for the selection methodology)
per-fold results as `fold{N}_lgbm_promo_v1.parquet`, in the same
directory and schema the Rung 1 baselines already use -- `make gates`
and `GATE_REPORT.md`'s G1/G2 evaluation pick them up with no special
casing.

## Cell 6 — package results for retrieval (both notebooks)

Archives ONLY the small result/log directories, never the raw
fold-cache train/holdout parquets (comparator/fold1_train.parquet
alone is hundreds of MB at real scale) -- those exist identically on
both ends already (same M5 DB, same extraction logic), so shipping
them over the network would be pure waste.

```python
import shutil
from pathlib import Path

staging = Path("/kaggle/working/results_staging")
staging.mkdir(exist_ok=True)

fold_cache = Path("/kaggle/working/augurics-engine/data/fold_cache")
# present only on Notebook A (comparator) or Notebook B (challenger) --
# copy whichever subset exists here, skip what doesn't.
for rel_path in ("comparator/results", "challenger_grid"):
    src = fold_cache / rel_path
    if src.exists():
        shutil.copytree(src, staging / rel_path)

slow_baseline_src = fold_cache / "slow_baseline_results"
if slow_baseline_src.exists():
    dest = staging / "slow_baseline_results"
    dest.mkdir(parents=True, exist_ok=True)
    for f in slow_baseline_src.glob("*lgbm_promo_v1*"):
        shutil.copy(f, dest / f.name)

shutil.make_archive("/kaggle/working/augurics_results", "zip", root_dir=staging)
print("Result archive at /kaggle/working/augurics_results.zip")
```

Anything under `/kaggle/working/` at the end of a "Save & Run All
(Commit)" becomes a downloadable output of that notebook version.
Notebook A's zip holds `comparator/results/` (+ its `run_log.txt`);
Notebook B's holds `challenger_grid/` (the raw per-combo results, its
`run_log.txt`) and `slow_baseline_results/` (only the selected best
combo's `fold{N}_lgbm_promo_v1.parquet` files, not the whole directory
-- Notebook B never touches the Rung 1 baseline results already
committed locally).

## Retrieval (local) — CLI route (primary)

One command, no manual download:

```bash
cd ~/augurics-engine
make pull-kaggle
```

This runs `kaggle kernels output` for both notebooks (defaults:
`onionterror8/notebook-a` for the comparator, `onionterror8/notebook-b`
for the challenger -- override with `NOTEBOOK_A=`/`NOTEBOOK_B=` if the
slugs ever change), fails loud with a clear message if the `kaggle` CLI
isn't installed, if credentials aren't configured, or if either
notebook's pulled output is missing `augurics_results.zip` (e.g. the
notebook hasn't reached Cell 6 yet, or failed earlier) -- then chains
directly into `make validate-kaggle` against both downloaded zips, so
the command ends with the validation summary printed.

Every result file that lands here still passes through
`load_and_validate_result` the moment anything reads it -- schema
validated by content, exactly as if produced locally. Quarantine rules
unchanged: nothing merges into any gate arithmetic until the fold-1
predictive-agreement number has been reviewed and quarantine is
explicitly lifted (`make morning-report`, run separately, after
`make pull-kaggle` reports clean).

## Retrieval (local) — browser download (fallback)

If the CLI route doesn't work (e.g. no local Kaggle credentials, or
`kaggle kernels output` behaves unexpectedly), download the notebook's
output zip from Kaggle's UI (Output tab of the committed version)
instead, then:

```bash
cd ~/augurics-engine
make validate-kaggle COMPARATOR_ZIP=~/Downloads/augurics_results_a.zip CHALLENGER_ZIP=~/Downloads/augurics_results_b.zip
```

(rename the two downloaded files first if they land with the same
default name -- Kaggle doesn't namespace them by notebook.)

## Session-length: split into two notebooks, not one

Stated before launching, per instruction -- and it changes the plan:

**Comparator alone is already close to the 12-hour ceiling.** The
pre-refactor estimate was ~2.6-2.9h/fold x 4 folds = **10.4-11.6h**,
plus fold 1's bounded NUTS spot-check and predictive-agreement check
(~5-15 min more, based on the 332s measured at 182-series/113k-row
scale, extrapolated to the bounded 100k-row cap). The indexed-
categorical refactor should not have made ADVI slower (gather ops
replace dense matmuls), but there is no fresh measurement at real
fold-1 scale post-refactor -- the estimate carries that uncertainty
forward. Either way, there is little to no headroom left in the same
session for the challenger's 48 fits on top.

**Challenger timing is genuinely unverified** -- no real-fold-scale
LightGBM fit has been timed (only tiny synthetic data locally, which
says nothing about wall time at 48M rows). Gradient boosting is
typically much faster than ADVI per fit, but "typically" is not a
number.

**Recommendation: two separate notebook commits, not one "Save & Run
All" spanning both.**

1. **Notebook A** -- cells 1-4 (clone, ingest+reconcile, test, comparator
   4 folds). Commit and let it run; expect close to the full 12h.
2. **Notebook B** -- cells 1-3 again (fresh session, re-ingest is cheap
   -- it's a few DuckDB queries over already-attached data, not a
   multi-hour operation) + cell 5 (challenger grid) + cell 6 (package).
   **Before committing to the full run**, do a bounded first check:
   run combo 0/fold 1 alone (`python -m engine.backtest.
   run_challenger_one_fold --fold-id 1 --combo-index 0 --out /tmp/probe.parquet`)
   and read its printed wall time. Multiply by 48 for a real estimate
   of the full grid's wall time before committing to "Save & Run All"
   on the rest -- the same verify-before-committing discipline as
   everything else in this project, not a guess.

If Notebook B's extrapolated total exceeds ~10h, split the grid itself
at fold boundaries (e.g. folds 1-2 in one commit, folds 3-4 in
another) -- `run_challenger_batch.py`'s per-(fold,combo) resumability
means a partial run is never wasted; a subsequent commit just skips
what already has a result file.
