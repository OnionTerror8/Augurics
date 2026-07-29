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


# --- raw-tree fallback: Cell 6 never ran (e.g. hard ceiling-kill), but
# kaggle kernels output still captured the raw /kaggle/working tree ---


def _write_comparator_result(fold_cache_dir, fold_id=1):
    d = fold_cache_dir / "comparator" / "results"
    d.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame([{
        "fold_id": fold_id, "model_id": "bayes_glm_pooled", "wall_seconds": 100.0,
        "wmape": 0.7, "n_series": 1000, "n_rows": 28000,
    }]).to_parquet(d / f"fold{fold_id}_bayes_glm_pooled.parquet")
    (d / "run_log.txt").write_text(f"fold {fold_id} bayes_glm_pooled: 100.0s -- ok\n")


def test_raw_tree_fallback_builds_archive_when_zip_absent(tmp_path):
    import zipfile

    raw_fold_cache = tmp_path / "augurics-engine" / "data" / "fold_cache"
    _write_comparator_result(raw_fold_cache)

    archive = check_kernel_output_has_result_archive(tmp_path, "onionterror8/notebook-a")
    assert archive.exists()
    assert archive.name == "augurics_results_from_raw_tree.zip"

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert any("fold1_bayes_glm_pooled.parquet" in n for n in names)
    assert any("run_log.txt" in n for n in names)


def test_raw_tree_fallback_prefers_staged_zip_when_both_present(tmp_path):
    """Cell 6's own zip is authoritative when it exists -- the raw-tree
    fallback only kicks in when it's absent, never overrides it."""
    (tmp_path / "augurics_results.zip").write_bytes(b"real staged zip")
    raw_fold_cache = tmp_path / "augurics-engine" / "data" / "fold_cache"
    _write_comparator_result(raw_fold_cache)

    archive = check_kernel_output_has_result_archive(tmp_path, "onionterror8/notebook-a")
    assert archive == tmp_path / "augurics_results.zip"
    assert archive.read_bytes() == b"real staged zip"


def test_raw_tree_fallback_raises_when_raw_tree_has_no_expected_dirs(tmp_path):
    raw_fold_cache = tmp_path / "augurics-engine" / "data" / "fold_cache"
    raw_fold_cache.mkdir(parents=True)
    (raw_fold_cache / "m5").mkdir()  # present, but not a result directory

    with pytest.raises(KaggleOutputMissingError, match="none of the expected result directories"):
        check_kernel_output_has_result_archive(tmp_path, "onionterror8/notebook-a")


def test_raw_tree_fallback_only_copies_lgbm_files_from_slow_baseline_results(tmp_path):
    import zipfile

    raw_fold_cache = tmp_path / "augurics-engine" / "data" / "fold_cache"
    slow_dir = raw_fold_cache / "slow_baseline_results"
    slow_dir.mkdir(parents=True)
    (slow_dir / "fold1_lgbm_promo_v1.parquet").write_bytes(b"challenger result")
    (slow_dir / "fold1_auto_ets.parquet").write_bytes(b"unrelated baseline result")

    archive = check_kernel_output_has_result_archive(tmp_path, "onionterror8/notebook-b")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert any("fold1_lgbm_promo_v1.parquet" in n for n in names)
    assert not any("fold1_auto_ets.parquet" in n for n in names)
