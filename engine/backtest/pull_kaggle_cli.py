"""`make pull-kaggle`'s entrypoint -- the real `kaggle kernels output`
subprocess calls, gated by pull_kaggle.py's fail-loud checks. Kept
separate from pull_kaggle.py itself so that module stays importable
and unit-testable without ever touching the network or subprocess.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from engine.backtest.pull_kaggle import (
    check_kaggle_cli_available,
    check_kaggle_credentials_available,
    check_kernel_output_has_result_archive,
)

NOTEBOOK_A_OUTPUT_DIR = Path("/tmp/kaggle_a")
NOTEBOOK_B_OUTPUT_DIR = Path("/tmp/kaggle_b")


def pull_kernel_output(kernel_slug: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels output {kernel_slug}: failed (exit {result.returncode}):\n{result.stderr}"
        )
    print(result.stdout, flush=True)
    return check_kernel_output_has_result_archive(output_dir, kernel_slug)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-a", required=True)
    parser.add_argument("--notebook-b", required=True)
    args = parser.parse_args()

    check_kaggle_cli_available()
    check_kaggle_credentials_available()

    archive_a = pull_kernel_output(args.notebook_a, NOTEBOOK_A_OUTPUT_DIR)
    print(f"Notebook A ({args.notebook_a}): {archive_a}")

    archive_b = pull_kernel_output(args.notebook_b, NOTEBOOK_B_OUTPUT_DIR)
    print(f"Notebook B ({args.notebook_b}): {archive_b}")


if __name__ == "__main__":
    main()
