"""pull_kaggle.py's fail-loud checks -- no network, no subprocess.
Mocks shutil.which, has_kaggle_credentials, and a real tmp_path
directory structure to exercise each guard independently."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.backtest.pull_kaggle import (
    KaggleCLIMissingError,
    KaggleCredentialsMissingError,
    KaggleOutputMissingError,
    check_kaggle_cli_available,
    check_kaggle_credentials_available,
    check_kernel_output_has_result_archive,
)


def test_check_kaggle_cli_available_passes_when_on_path():
    with patch("engine.backtest.pull_kaggle.shutil.which", return_value="/usr/local/bin/kaggle"):
        check_kaggle_cli_available()  # must not raise


def test_check_kaggle_cli_available_raises_when_missing():
    with patch("engine.backtest.pull_kaggle.shutil.which", return_value=None):
        with pytest.raises(KaggleCLIMissingError, match="pip install kaggle"):
            check_kaggle_cli_available()


def test_check_kaggle_credentials_available_passes_when_present():
    with patch("engine.backtest.pull_kaggle.has_kaggle_credentials", return_value=True):
        check_kaggle_credentials_available()  # must not raise


def test_check_kaggle_credentials_available_raises_with_setup_instructions():
    with patch("engine.backtest.pull_kaggle.has_kaggle_credentials", return_value=False):
        with pytest.raises(KaggleCredentialsMissingError, match="kaggle.json"):
            check_kaggle_credentials_available()


def test_check_kernel_output_passes_when_archive_present(tmp_path):
    (tmp_path / "augurics_results.zip").write_bytes(b"fake zip contents")
    archive = check_kernel_output_has_result_archive(tmp_path, "onionterror8/notebook-a")
    assert archive == tmp_path / "augurics_results.zip"


def test_check_kernel_output_raises_when_archive_missing_but_other_files_present(tmp_path):
    (tmp_path / "some_other_output.txt").write_text("not the zip")
    with pytest.raises(KaggleOutputMissingError, match="onionterror8/notebook-a"):
        check_kernel_output_has_result_archive(tmp_path, "onionterror8/notebook-a")


def test_check_kernel_output_raises_when_directory_does_not_exist(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(KaggleOutputMissingError, match="does not exist"):
        check_kernel_output_has_result_archive(missing_dir, "onionterror8/notebook-b")
