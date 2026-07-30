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

**Before every "Save & Run All", re-verify the `GITHUB_TOKEN` secret is
attached to THAT notebook** (Add-ons -> Secrets -> toggle ON for this
notebook), not just that it exists somewhere in your account. This is
not a hypothetical -- **confirmed**, not just plausible: both
notebooks failed with an identical `ConnectionError` (`HTTPError: HTTP
Error 400: Bad Request` from `UserSecretsClient().get_secret(...)`)
immediately after a `kaggle kernels push` from the CLI, and both
required manually re-toggling the secret ON before they would run
again. **`kaggle kernels push` from the CLI appears to detach a
notebook's secrets** (or at least does not preserve the attachment
reliably) -- confirmed by direct observation, not fully explained.
Our token is also a 7-day fine-grained GitHub PAT, so it expires
outright on its own schedule regardless and needs regenerating
periodically -- a second, independent reason the same failure can
recur.

**Consequence, stronger than first stated: `kaggle kernels push` is
banned as a deployment method, full stop -- not just "prefer the UI
for the trigger."** `kaggle kernels push` has no "upload without
running" mode (`kaggle kernels push --help` confirms no such flag
exists) -- every push immediately auto-runs the kernel, so there is no
way to use the CLI to *prepare* a version without also triggering the
exact launch this section says to avoid. Editing a notebook's cells
must happen by pasting into the Kaggle UI editor directly, and
launching is "Save & Run All" from that same editor, with the secret
toggle re-checked immediately before clicking it. The `kaggle` CLI's
role in this workflow is retrieval and status only (`kaggle kernels
output`, `kaggle kernels status`, both wrapped by `make pull-kaggle`)
-- never `kaggle kernels push`.

Cell 1 below now fails gracefully instead of surfacing a raw
traceback: it catches any exception from `get_secret` and exits with
one instruction line naming the toggle path, rather than a multi-frame
`ConnectionError`/`HTTPError` stack that's easy to skim past.

Everything below is cell content to paste into that notebook, in order.

## Cell 1 — clone at HEAD, install

```python
import os
from kaggle_secrets import UserSecretsClient

try:
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    raise SystemExit(
        "Secret not attached: open Add-ons -> Secrets and toggle GITHUB_TOKEN "
        "on for this notebook, then re-run."
    ) from None

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

**Notebook B only: this cell is merged with the probe (see Cell 5)
into one cell, not kept separate.** Notebook B's session died twice,
reproducibly, right after this cell's `make test` completed and before
the next cell started -- with `git clone`, ingest, and the full test
suite all having succeeded both times, ruling out a Kaggle-secret/auth
explanation. Whether the death is tied to the cell BOUNDARY itself or
to elapsed time (~11-12 min in) is unresolved; merging removes the
boundary as a variable and shifts the probe's start earlier, which the
log will show either way. See Cell 5 for the merged content and
`KNOWN_ISSUES.md` for the full investigation.

## Cell 4 — Notebook A only: comparator, ONE fold per commit

**Real fold-1 timing on Kaggle was 10.64h (comparator_evidence.md §8)
-- ~3.5-4x the pre-refactor local extrapolation.** This does not
leave room for more than one fold per 12h session (confirmed directly:
the first real run's fold 2 started -- its fold frames got cached --
and was then ceiling-cancelled mid-fit with zero partial credit,
atomic writes meaning nothing was banked). `run_comparator_batch` now
takes `--fold-id` to restrict a single invocation to exactly one fold,
and a session-budget guard that refuses loud, before starting a fit,
if the projected time would not fit in the session's remaining budget
minus a 30-minute margin.

```python
import time

SESSION_STARTED_AT = time.time()  # capture as close to session start as practical --
                                   # ideally move this line to Cell 1 instead; kept
                                   # here for a single copy-paste block

FOLD_ID = 3  # bump this each commit: 2 (done), 3 (this one), then 4

result = os.system(
    f"python -m engine.backtest.run_comparator_batch "
    f"--fold-id {FOLD_ID} --session-started-at {SESSION_STARTED_AT}"
)
assert result == 0, "comparator batch failed, the session-budget guard refused, or a gate did not clear -- see run_log.txt"
```

If accuracy matters more than convenience, capture `SESSION_STARTED_AT
= time.time()` as literally the first line of Cell 1 instead and reuse
it here -- the guard's whole point is knowing how much of the 12h
ceiling Cells 1-3 (clone, ingest, `make test`) have already consumed,
and those are not free (the first real run spent ~700s on them before
Cell 4 even started).

Fold 1 (already banked, see comparator_evidence.md §8) ran with the
bounded NUTS spot-check and the decisive predictive-agreement check
(2% relative WMAPE tolerance) and PASSED (0.31% relative difference) --
ADVI is licensed for gate purposes. Folds 2-4 are ADVI-only (no NUTS;
`NUTS_SPOT_CHECK_FOLD_ID = 1` only). Run folds 2, 3, 4 as **three
separate notebook commits**, confirming via `make pull-kaggle` (or
`kaggle kernels status`) that each fold's result file exists before
starting the next -- never let one commit's Cell 4 attempt more than
one fold's worth of fitting.

## Cell 5 — Notebook B only: challenger, 4 folds x the pre-registered 12-combo grid

**Mandatory: the single-combo probe (below) must run and its
extrapolation must be read BEFORE this cell runs.**

**Merged with Cell 3 (`make test`), not a separate cell.** Notebook
B's session died twice, reproducibly, right at this exact cell
boundary -- after `make test` completed, before the probe (originally
a separate next cell) ever started -- with no traceback either time.
`git clone`, ingest, and the full test suite succeeded both times, so
this is not a Kaggle-secret/auth failure. Whether the actual cause is
the cell transition itself or something time-based (~11-12 minutes in)
is still unresolved (see `KNOWN_ISSUES.md`); merging removes the
boundary as a variable, and the probe now starts about a minute
earlier, which the log will show either way if the real cause is
purely elapsed time.

**The probe's timing line is written into the repo tree, not just
printed to stdout**, so even a post-probe cancel leaves the number
retrievable via the raw-tree pull fallback (`make pull-kaggle` /
`validate_kaggle.py`) -- a hard kill can end a session with zero
further kernel-log lines, so stdout alone is not reliable insurance.

```python
result = os.system("make test")
assert result == 0, "tests failed on this Kaggle environment -- do not proceed"

import time
from pathlib import Path

t0 = time.time()
result = os.system("python -m engine.backtest.run_challenger_one_fold --fold-id 1 --combo-index 0 --out /tmp/probe.parquet")
wall = time.time() - t0
assert result == 0, "probe failed -- see output above"
est_hours = wall * 48 / 3600
timing_line = f"Probe wall time: {wall:.1f}s -- extrapolated full grid (48 fits): {est_hours:.1f} hours"
print(timing_line)

probe_log_path = Path("data/fold_cache/probe_timing.txt")
probe_log_path.parent.mkdir(parents=True, exist_ok=True)
probe_log_path.write_text(timing_line + "\n")

assert est_hours <= 10, f"STOP: extrapolated {est_hours:.1f}h exceeds the 10h safety bound -- do not run the next cell, split the grid by fold instead"
print("Under the 10h bound -- proceed to the full-grid cell.")
```

If the extrapolation exceeds 10h, do not run the full-grid cell as a
single commit -- split by fold instead (12 fits per commit, one commit
per fold), using the same `--fold-id` pattern as the comparator, via
`run_challenger_one_fold` directly for each combo in that fold (a
`--fold-id`-only mode for `run_challenger_batch` itself is a natural
follow-up if this split becomes the normal path, mirroring the
comparator's).

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

**This cell can never run after a hard ceiling-kill.** Confirmed in
practice: when Kaggle's session ceiling hits, the ENTIRE kernel
process is terminated, not gracefully stopped -- no in-notebook code,
`try`/`finally`, or signal handler can guarantee this cell executes
after that. `kaggle kernels output` still captures everything under
`/kaggle/working/` regardless (this is how fold 1's real result
reached the retrieval step even though Cell 6 never ran that session)
-- `make pull-kaggle` / `validate_kaggle.py` handle this: if
`augurics_results.zip` is absent, they build the equivalent archive
client-side from the raw pulled tree, running this exact same staging
logic locally instead of on Kaggle. This cell is still worth keeping
for the normal (non-killed) case -- it just isn't load-bearing.

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

## Session-length: real measured numbers, not the pre-launch estimate

The pre-launch estimate below this line (kept in git history, not
here) assumed ~2.6-2.9h/fold based on local, pre-refactor timing. The
first real Kaggle launch replaced that guess with a measurement:

**Fold 1's real wall time was 38,308.3s = 10.64h** (`fit_seconds`
32,967.6s / 9.16h including the bounded NUTS spot-check;
`predict_seconds` 5,237.5s / 1.45h for `_predict_all` over the full
26,304-series holdout population) -- see `comparator_evidence.md` §8.
This is **~3.5-4x slower than assumed**, almost certainly because
Kaggle's CPU-only runtime (2 vCPU) is markedly weaker for this
workload than the development laptop's extrapolation assumed. There is
no headroom for more than one comparator fold per 12h session, full
stop -- confirmed the hard way when the first run's fold 2 started,
cached its fold frames, and was ceiling-cancelled mid-fit with zero
partial credit.

**Comparator: one fold per commit, mandatory.** `run_comparator_batch
--fold-id N` (Cell 4, above) plus the session-budget guard
(`check_session_budget`, `engine/backtest/run_comparator_batch.py`)
enforce this going forward -- the guard refuses loud, before starting
a fit, if the projected time (conservatively, fold 1's full 10.6h,
applied to every fold regardless of NUTS) would not fit the session's
remaining budget minus 30 minutes.

**Challenger: still genuinely unverified.** No real-fold-scale
LightGBM fit has been timed. Notebook B's committed version already
has the probe-first structure (Cell 5, above: single fold/combo, `x48`
extrapolation, hard `assert est_hours <= 10`) -- read its printed
extrapolation before ever running the full-grid cell. If the
extrapolation exceeds 10h, split the grid by fold (12 fits/commit)
rather than attempt all 48 in one session.
