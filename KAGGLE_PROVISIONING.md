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
```

`engine.data.ingest.ingest()` already asserts row-count reconciliation
against the source CSVs (30,490 series x 1,941 days = 59,181,090 rows)
-- this is the SAME fail-loud check used locally, not a Kaggle-specific
addition.

## Cell 3 — verify the environment

```python
result = os.system("make test")
assert result == 0, "tests failed on this Kaggle environment -- do not proceed"
```

All tests must pass green here, same bar as locally, before any real
run -- if something about the Kaggle environment differs (a wheel that
built differently, a library version mismatch), this is where it
surfaces, not mid-fold.

## Cell 4 — comparator: fold 1 (bounded NUTS + predictive-agreement gate)

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

## Cell 5 — challenger: 4 folds x the pre-registered 12-combo grid

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

## Cell 6 — package results for retrieval

```python
import shutil

shutil.make_archive(
    "/kaggle/working/augurics_results",
    "zip",
    root_dir="/kaggle/working/augurics-engine/data/fold_cache",
)
print("Result archive at /kaggle/working/augurics_results.zip")
```

Anything under `/kaggle/working/` at the end of a "Save & Run All
(Commit)" becomes a downloadable output of that notebook version --
this zip (comparator results, challenger grid + selected results, both
run_log.txt files) is what comes back.

## Retrieval (local)

Download the notebook's output zip from Kaggle's UI (Output tab of the
committed version), then:

```bash
cd ~/augurics-engine
unzip ~/Downloads/augurics_results.zip -d /tmp/kaggle_results
cp -r /tmp/kaggle_results/comparator/results/* data/fold_cache/comparator/results/
cp -r /tmp/kaggle_results/slow_baseline_results/*lgbm_promo_v1* data/fold_cache/slow_baseline_results/
cp -r /tmp/kaggle_results/challenger_grid data/fold_cache/
```

Every result file that lands here still passes through
`load_and_validate_result` the moment anything reads it (`GATE_REPORT.md`
assembly, `compare_models`) -- schema-validated by content, exactly as
if it had been produced locally. Quarantine rules unchanged: the
comparator's numbers are not merged into any gate arithmetic until the
fold-1 predictive-agreement gate has been reviewed.

## Session-length note

Fold 1's comparator estimate was ~2.6-2.9h pre-refactor; the challenger
grid is 48 fits (4 folds x 12 combos) at unknown-but-likely-much-faster
per-fit time than ADVI (LightGBM, not MCMC). Both should clear the
12-hour session limit individually, but if the notebook is run as ONE
"Save & Run All" spanning cells 4 AND 5, watch the cumulative wall
time -- split into two separate commits (comparator first, challenger
second) if it looks close, rather than risk a mid-run cutoff.
