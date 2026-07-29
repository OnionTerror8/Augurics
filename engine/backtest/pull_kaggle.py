"""Kaggle CLI retrieval for the two batch notebooks -- task 2.1/2.3.
`kaggle kernels output` fetches whatever a notebook's committed version
produced under /kaggle/working/ (KAGGLE_PROVISIONING.md's Cell 6
result-staging zip, specifically) into a local directory. This module
is the fail-loud checks around that call -- CLI installed, credentials
present, expected output actually there -- kept importable/testable
separately from the `make pull-kaggle` shell invocation itself (which
does the actual subprocess call, since that's a real network operation
this module's tests must never perform).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from engine.data.download_m5 import has_kaggle_credentials

RESULT_ARCHIVE_NAME = "augurics_results.zip"


class KaggleCLIMissingError(Exception):
    """Raised when the `kaggle` command is not on PATH. Fix: `pip
    install kaggle` (already a declared dependency in pyproject.toml --
    this means the venv this command is running under doesn't have it,
    or the CLI shim isn't on PATH)."""


class KaggleCredentialsMissingError(Exception):
    """Raised when neither ~/.kaggle/kaggle.json nor
    ~/.kaggle/access_token (nor the KAGGLE_API_TOKEN /
    KAGGLE_USERNAME+KAGGLE_KEY env vars) are present. Same check
    engine.data.download_m5 already uses for the M5 dataset download --
    reused here, not re-implemented, so there is exactly one definition
    of "are Kaggle credentials configured" in this codebase."""


class KaggleOutputMissingError(Exception):
    """Raised when a notebook's pulled output directory has neither
    augurics_results.zip NOR a raw fold_cache tree to build one from --
    e.g. the notebook hasn't finished a committed run yet, failed
    before any result was written, or the slug pulled the wrong
    kernel. Never silently proceed to validate_kaggle.py against a
    directory that doesn't have what it expects."""


# Where `kaggle kernels output` places the notebook's raw working
# directory when Cell 6 (packaging) never ran -- confirmed in practice
# (first real Kaggle launch): a hard ceiling-kill terminates the whole
# kernel process, which no in-notebook code can defend against, so
# Cell 6 simply never gets a chance to execute. The raw tree still
# lands here regardless, since `kaggle kernels output` captures
# everything under /kaggle/working/, not just Cell 6's staged zip.
RAW_TREE_FOLD_CACHE_REL_PATH = Path("augurics-engine") / "data" / "fold_cache"


def check_kaggle_cli_available() -> None:
    if shutil.which("kaggle") is None:
        raise KaggleCLIMissingError(
            "kaggle CLI not found on PATH. Install it with `pip install kaggle` "
            "(already listed in pyproject.toml -- if this venv is active and it's "
            "still missing, run `pip install -e .` to install project dependencies)."
        )


def check_kaggle_credentials_available() -> None:
    if not has_kaggle_credentials():
        raise KaggleCredentialsMissingError(
            "No Kaggle API credentials found. Set up ONE of:\n"
            "  - ~/.kaggle/kaggle.json (download from kaggle.com -> Account -> "
            "Create New API Token)\n"
            "  - ~/.kaggle/access_token (modern token method: "
            "kaggle.com/settings -> API -> Create Token)\n"
            "  - KAGGLE_USERNAME + KAGGLE_KEY environment variables\n"
            "If using ~/.kaggle/kaggle.json, also run: chmod 600 ~/.kaggle/kaggle.json"
        )


def check_kernel_output_has_result_archive(output_dir: Path, kernel_slug: str) -> Path:
    """Returns a path to a usable augurics_results.zip -- either Cell
    6's own staged archive (the normal case), or one built here,
    client-side, from the raw /kaggle/working tree using the exact
    same staging logic Cell 6 would have run (the fallback case, when
    the notebook's process was killed before Cell 6 could execute).
    Raises only if NEITHER is present -- i.e. there is genuinely
    nothing to validate yet."""
    output_dir = Path(output_dir)
    archive_path = output_dir / RESULT_ARCHIVE_NAME
    if archive_path.exists():
        return archive_path

    raw_fold_cache = output_dir / RAW_TREE_FOLD_CACHE_REL_PATH
    if raw_fold_cache.exists():
        return _build_archive_from_raw_tree(output_dir, raw_fold_cache, kernel_slug)

    found = sorted(p.name for p in output_dir.glob("*")) if output_dir.exists() else []
    raise KaggleOutputMissingError(
        f"{kernel_slug}: pulled output at {output_dir} has neither {RESULT_ARCHIVE_NAME} "
        f"nor a raw fold_cache tree at {raw_fold_cache}. Found instead: "
        f"{found or '(directory does not exist)'}. The notebook may not have reached "
        "the ingest/batch cells yet, failed before writing any result, or the kernel "
        "slug may be wrong."
    )


def _build_archive_from_raw_tree(output_dir: Path, raw_fold_cache: Path, kernel_slug: str) -> Path:
    """Mirrors Cell 6's own staging logic (KAGGLE_PROVISIONING.md)
    against whatever raw tree `kaggle kernels output` actually
    captured -- comparator/results, challenger_grid, and only the
    lgbm_promo_v1 files from slow_baseline_results, never the whole
    directory."""
    staging = output_dir / "_raw_tree_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for rel_path in ("comparator/results", "challenger_grid"):
        src = raw_fold_cache / rel_path
        if src.exists():
            shutil.copytree(src, staging / rel_path)

    slow_baseline_src = raw_fold_cache / "slow_baseline_results"
    if slow_baseline_src.exists():
        dest = staging / "slow_baseline_results"
        dest.mkdir(parents=True, exist_ok=True)
        for f in slow_baseline_src.glob("*lgbm_promo_v1*"):
            shutil.copy(f, dest / f.name)

    if not any(staging.iterdir()):
        raise KaggleOutputMissingError(
            f"{kernel_slug}: raw tree found at {raw_fold_cache} but it contained none "
            "of the expected result directories (comparator/results, challenger_grid, "
            "slow_baseline_results) -- nothing to package"
        )

    archive_path = shutil.make_archive(str(output_dir / "augurics_results_from_raw_tree"), "zip", root_dir=staging)
    return Path(archive_path)
