# Cloud provisioning — comparator batch (task 2.1)

Why: the pre-fit memory estimator (`engine/models/comparators.py`,
`_check_fit_memory`) correctly refused fold 1's real launch on the
development laptop — estimated fit footprint 3.90GB against 1.7-3.1GB
available RAM, which never cleared budget across ~20 minutes of closing
applications (see `KNOWN_ISSUES.md`, "pre-fit memory estimator's first
live catch"). The guard is not the problem; the laptop's baseline
desktop load is. This document is the path to a VM with enough headroom
that the estimator clears with margin instead of refusing.

## 1. Target spec

- **16GB+ RAM, 8 vCPU.** Fold 1's estimated footprint is 3.90GB
  (`MAX_FIT_MEMORY_FRACTION=0.5` of available RAM budgets 8GB+ needed
  headroom); 16GB clears it with ~4x margin even accounting for the OS
  and Python/PyMC's own baseline footprint.
- **Ubuntu 24.04 LTS.**
- **Cost-appropriate options** (destroy after Phase 1's comparator
  batch completes, or after task 2.1 closes, to avoid ongoing spend):
  - Hetzner Cloud **CX42** (8 vCPU, 16GB RAM) — ~€28/month, billed
    hourly; a few hours of real use costs cents.
  - DigitalOcean **Premium AMD/Intel 8vCPU/16GB** droplet — comparable
    hourly pricing.
  Either is ample; this is a CPU-bound ADVI fit, no GPU needed.

Provisioning the VM itself requires the user's own cloud account/API
token — not something this assistant can do without those credentials.
Reference commands below (Hetzner CLI shown; DigitalOcean's `doctl`
equivalent is a straightforward substitution) are for the user to run
or adapt.

```bash
# reference only — requires HCLOUD_TOKEN configured locally
hcloud server create \
  --name augurics-p1-comparator \
  --type cx42 \
  --image ubuntu-24.04 \
  --ssh-key <your-key-name>
```

## 2. Environment setup (on the VM, once it exists)

```bash
ssh root@<vm-ip>

apt-get update && apt-get install -y python3.11 python3.11-venv git build-essential

git clone https://github.com/OnionTerror8/Augurics.git augurics-engine
cd augurics-engine

python3.11 -m venv .venv
.venv/bin/pip install -e .
```

`pyproject.toml` already pins `requires-python = ">=3.11"` and lists
every dependency (`statsforecast`, `mlforecast`, `lightgbm`, `duckdb`,
`pandas`, `pymc`, `pytest`, `kaggle`, `pyyaml`, `psutil`) — `pip
install -e .` is the only install step needed; nothing else to
translate from the local environment.

## 3. Data transfer (local -> VM)

The M5 panel is already ingested and reconciled locally
(`data/m5/augurics.duckdb`, ~795MB) — re-running the Kaggle
download+ingest pipeline on the VM would just reproduce the same
DuckDB file at the cost of re-authenticating Kaggle credentials on a
fresh box. Ship the built DB instead:

```bash
# from the local repo root
make push-data REMOTE=root@<vm-ip>
```

This runs (see Makefile):

```bash
ssh $(REMOTE) 'mkdir -p augurics-engine/data/m5'
rsync -avzP --progress data/m5/augurics.duckdb $(REMOTE):augurics-engine/data/m5/augurics.duckdb
```

Fold caches (`data/fold_cache/`) are deliberately NOT transferred —
they regenerate on the VM from the DB on first access
(`get_comparator_fold_frames_cached`), which is fast (a few DuckDB
queries) and guarantees the VM's cache can't silently diverge from a
stale local copy.

## 4. Verify the environment

```bash
# on the VM
cd augurics-engine
make test
```

All 147 tests must pass green before any real-data run. This is the
same suite that runs locally — if the VM's environment is missing
anything (compiler toolchain for a wheel, a BLAS library, etc.), this
is where it surfaces, not mid-fold-1.

## 5. Launch the batch (survives disconnect)

```bash
# on the VM
cd augurics-engine
tmux new -s comparator-batch
source .venv/bin/activate
nohup python -m engine.backtest.run_comparator_batch > /tmp/comparator_batch_output.log 2>&1 &
disown
# Ctrl-b d to detach; tmux keeps the session (and the nohup'd process)
# alive across SSH disconnects.
```

Reattach any time with `tmux attach -t comparator-batch`. Progress and
failures land in `data/fold_cache/comparator/results/run_log.txt` (same
resumable, atomic-write, subprocess-per-fold contract as the local
orchestrator — nothing about the orchestration logic changes on the
VM, only where it runs).

## 6. Retrieve results (VM -> local)

Once the batch finishes (or halts on a gate failure — check
`run_log.txt` either way):

```bash
# from the local repo root
make pull-results REMOTE=root@<vm-ip>
```

This runs:

```bash
rsync -avzP --progress $(REMOTE):augurics-engine/data/fold_cache/comparator/results/ data/fold_cache/comparator/results/
```

Pulling is safe to run repeatedly (rsync, not overwrite-blind copy) —
useful for checking in on `run_log.txt` while the batch is still
running on the VM, without waiting for full completion.

## 7. Teardown

Once results are pulled and validated locally, destroy the VM to stop
billing:

```bash
hcloud server delete augurics-p1-comparator   # or the DigitalOcean equivalent
```
