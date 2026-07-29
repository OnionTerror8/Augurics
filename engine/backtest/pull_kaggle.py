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
    """Raised when a notebook's pulled output directory does not
    contain augurics_results.zip -- e.g. the notebook hasn't finished
    a committed run yet, failed before Cell 6, or the slug pulled the
    wrong kernel. Never silently proceed to validate_kaggle.py against
    a directory that doesn't have what it expects."""


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
    archive_path = Path(output_dir) / RESULT_ARCHIVE_NAME
    if not archive_path.exists():
        found = sorted(p.name for p in Path(output_dir).glob("*")) if Path(output_dir).exists() else []
        raise KaggleOutputMissingError(
            f"{kernel_slug}: pulled output at {output_dir} does not contain "
            f"{RESULT_ARCHIVE_NAME}. Found instead: {found or '(directory does not exist)'}. "
            "The notebook may not have reached Cell 6 (packaging), may have failed "
            "earlier, or the kernel slug may be wrong."
        )
    return archive_path
