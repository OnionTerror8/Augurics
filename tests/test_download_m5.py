"""Fail-loud contract for the M5 downloader: no credentials, no silent stub."""
from __future__ import annotations

import pytest

from engine.data.download_m5 import (
    REQUIRED_FILES,
    CredentialsMissingError,
    has_kaggle_credentials,
    manual_instructions,
)


@pytest.fixture(autouse=True)
def _no_kaggle_env(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)


def test_no_credentials_detected_in_this_environment(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert has_kaggle_credentials() is False


def test_download_raises_credentials_missing_error(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from engine.data.download_m5 import download

    with pytest.raises(CredentialsMissingError):
        download(dest=tmp_path / "m5")


def test_manual_instructions_name_every_required_file(tmp_path):
    text = manual_instructions(tmp_path / "m5")
    for filename in REQUIRED_FILES:
        assert filename in text
