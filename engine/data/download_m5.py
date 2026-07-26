"""M5 (m5-forecasting-accuracy) download — task 1.2.

Uses the Kaggle API when credentials are available. Fails loud with exact
manual-download instructions when they are not — never falls back to
stubbed or synthetic data (Non-negotiable rule 4).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

COMPETITION = "m5-forecasting-accuracy"
DEST = Path(__file__).resolve().parents[2] / "data" / "m5"
REQUIRED_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")


class CredentialsMissingError(Exception):
    pass


def manual_instructions(dest: Path = DEST) -> str:
    return (
        "Kaggle API credentials not found (no ~/.kaggle/kaggle.json and no "
        "KAGGLE_USERNAME/KAGGLE_KEY environment variables).\n\n"
        "To proceed manually:\n"
        "  1. Go to https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
        "     (sign in, accept the competition rules if prompted).\n"
        "  2. Download these three files:\n"
        "       - calendar.csv\n"
        "       - sell_prices.csv\n"
        "       - sales_train_evaluation.csv\n"
        f"  3. Place them, unzipped, at: {dest}/\n"
        "     i.e.:\n"
        f"       {dest}/calendar.csv\n"
        f"       {dest}/sell_prices.csv\n"
        f"       {dest}/sales_train_evaluation.csv\n"
        "  4. Re-run: make ingest\n"
    )


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download(dest: Path = DEST) -> None:
    if not has_kaggle_credentials():
        print(manual_instructions(dest), file=sys.stderr)
        raise CredentialsMissingError(
            "Kaggle API credentials not found; see manual instructions above."
        )

    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI not installed. `pip install kaggle` first.")

    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(dest)],
        check=True,
    )
    for zip_path in dest.glob("*.zip"):
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(dest)], check=True)
        zip_path.unlink()

    missing = [f for f in REQUIRED_FILES if not (dest / f).exists()]
    if missing:
        raise FileNotFoundError(f"Download completed but missing expected files: {missing}")


if __name__ == "__main__":
    download()
