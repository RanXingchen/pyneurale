# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import conftest as test_config
import pytest


def test_decoding_integration_suite_requires_native_install() -> None:
    path = Path(test_config.__file__).parent / "integration/decoding/test_decoder_workflow.py"
    assert test_config._classification(path) == "native"


def test_experiment_integration_suite_requires_native_install() -> None:
    path = (
        Path(test_config.__file__).parent / "integration/experiments/test_center_out_closed_loop.py"
    )
    assert test_config._classification(path) == "native"


def test_recording_integration_suite_requires_native_install() -> None:
    path = Path(test_config.__file__).parent / "integration/recording/test_continuous_recording.py"
    assert test_config._classification(path) == "native"


def test_traceability_suite_is_python_only() -> None:
    path = Path(test_config.__file__).parent / "unit/experiments/test_traceability.py"
    assert test_config._classification(path) == "python_only"


def test_required_native_preflight_has_clear_install_error(monkeypatch) -> None:
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(test_config.importlib, "import_module", unavailable)
    with pytest.raises(pytest.UsageError, match="build and install"):
        test_config._require_native_extension()


def test_required_native_preflight_rejects_stale_contract(monkeypatch) -> None:
    package = SimpleNamespace(__version__="1.2.3")
    native = SimpleNamespace(
        build_info=lambda: {"version": "1.2.3", "abi_version": 1},
    )

    def import_module(name: str) -> object:
        return package if name == "neurale" else native

    monkeypatch.setattr(test_config.importlib, "import_module", import_module)
    with pytest.raises(
        pytest.UsageError,
        match=r"incompatible.*models.decomposition\.fit_pca",
    ):
        test_config._require_native_extension()


def test_every_python_only_file_is_claimed_by_a_ci_lane() -> None:
    # `pytest.ini` deselects `python_only` from the default run, so these files
    # only execute where a job names them. Both lanes derive their file list
    # from the sets below -- source-only from `_PYTHON_ONLY_FILES`, and
    # `built-install` from `_PYTHON_ONLY_INSTALLED_FILES` -- so a file that
    # marks itself and lands in neither set runs in no CI job at all. That is
    # how `integration/features/test_feature_workflows.py` went unexecuted.
    tests_root = Path(test_config.__file__).parent
    claimed = test_config._PYTHON_ONLY_FILES | test_config._PYTHON_ONLY_INSTALLED_FILES
    self_marked = {
        path.relative_to(tests_root).as_posix()
        for path in tests_root.rglob("test_*.py")
        if "mark.python_only" in path.read_text(encoding="utf-8")
    }
    assert self_marked <= claimed, sorted(self_marked - claimed)


def test_python_only_lanes_do_not_overlap() -> None:
    # A file cannot be both runnable from `PYTHONPATH=src` and dependent on an
    # install; overlap would mean one lane is running something it cannot.
    assert not (test_config._PYTHON_ONLY_FILES & test_config._PYTHON_ONLY_INSTALLED_FILES)


def test_claimed_python_only_files_exist() -> None:
    tests_root = Path(test_config.__file__).parent
    missing = [
        relative
        for relative in sorted(
            test_config._PYTHON_ONLY_FILES | test_config._PYTHON_ONLY_INSTALLED_FILES
        )
        if not (tests_root / relative).is_file()
    ]
    assert not missing, missing
