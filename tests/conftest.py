# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Repository test-mode classification and installed-native preflight."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
_EXPECTED_NATIVE_ABI = 1
_REQUIRED_NATIVE_ATTRIBUTES = (
    "signal",
    "signal.simulation.SignalGenerator",
    "signal.simulation.NeuralSignalGenerator",
    "devices.simulation.SimulatedNeuralSource",
    "features.online",
    "models.decomposition.fit_pca",
    "models.classification.fit_lda",
    "models.linear_model.fit_linear_model",
    "models.state_space.fit_linear_gaussian",
    "models.alignment.dtw",
    "models.neighbors.knn",
    "models.density.fit_gaussian_kde",
    "models.manifold.fit_lpp",
    "streaming",
    "sorting.detection._detect_threshold",
    "sorting.detection._detect_threshold_waveforms",
    "sorting.detection._extract_threshold_waveforms",
    "sorting.valley_seeking._valley_seeking",
    "experiments.TrialIdentity",
    "experiments.assistance.VelocityVector",
    "experiments.center_out.CenterOutTask",
    "experiments.center_out.CenterOutMachine",
    "experiments.center_out._NativeCenterOutGuidance",
    "experiments.center_out._NativeCenterOutGuidanceConfig",
    "experiments.speech.SpeechCueConfig",
    "experiments.speech.SpeechMachine",
)

_PYTHON_ONLY_FILES = frozenset(
    {
        "integration/signal/test_clean_import.py",
        "unit/models/test_models_import.py",
        "unit/runtime/test_native_probe.py",
        "unit/runtime/test_runtime_import.py",
        "unit/signal/test_signal_import.py",
        "unit/devices/test_devices_import.py",
        # test_experiments_import.py is intentionally NOT here: its
        # exports_present checks resolve native attributes (TrialIdentity,
        # VelocityVector, METRIC_VERSION_1, SpeechContentKind, ...), so it
        # requires an installed native extension and belongs to the native /
        # built-install suite via the unit/experiments/ -> "native" rule.
        "unit/experiments/test_traceability.py",
        "unit/streaming/test_streaming_import.py",
        "unit/test_exceptions.py",
        "unit/test_package_artifacts.py",
        "unit/test_benchmark_regression_policy.py",
        "unit/test_package_import.py",
        "unit/test_suite_classification.py",
    }
)

#: Python-only files that nevertheless need PyNeurale *installed*. They spawn a
#: probe interpreter that does not inherit ``PYTHONPATH``, so the source-only
#: jobs (``PYTHONPATH=src``) cannot run them. ``built-install`` runs them by
#: name, because ``pytest.ini`` deselects ``python_only`` from the full suite
#: and they would otherwise run in no CI job at all.
_PYTHON_ONLY_INSTALLED_FILES = frozenset(
    {
        "integration/features/test_feature_workflows.py",
    }
)


#: Everything under one of these paths resolves native attributes at import or
#: call time, so it belongs to the native / built-install suite.
_NATIVE_PREFIXES = (
    "unit/devices/",
    "unit/experiments/",
    "unit/features/",
    "unit/models/",
    "unit/signal/",
    "unit/streaming/",
    "integration/decoding/",
    "integration/experiments/",
    "integration/recording/",
    "integration/signal/test_native_",
)

#: Single native files that no prefix above covers.
_NATIVE_FILES = frozenset(
    {
        "integration/runtime/test_native_extension.py",
        "test_signal_realtime_api.py",
    }
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-native",
        action="store_true",
        default=False,
        help="fail before collection unless the installed native contract matches",
    )


def _require_native_extension() -> None:
    try:
        package = importlib.import_module("neurale")
        native = importlib.import_module("neurale._native")
    except (ImportError, OSError) as exc:
        raise pytest.UsageError(
            "the primary test mode requires an installed PyNeurale native "
            "extension; build and install a wheel or editable package first"
        ) from exc

    build_info = getattr(native, "build_info", None)
    if not callable(build_info):
        _raise_incompatible_native("build_info() is missing")
    try:
        build = build_info()
    except Exception as exc:
        _raise_incompatible_native("build_info() failed", cause=exc)
    if not isinstance(build, Mapping):
        _raise_incompatible_native("build_info() did not return a mapping")
    if build.get("abi_version") != _EXPECTED_NATIVE_ABI:
        _raise_incompatible_native(
            f"expected ABI {_EXPECTED_NATIVE_ABI}, got {build.get('abi_version')!r}"
        )
    package_version = getattr(package, "__version__", None)
    if build.get("version") != package_version:
        _raise_incompatible_native(
            f"Python version {package_version!r} does not match native version "
            f"{build.get('version')!r}"
        )

    missing = [
        path for path in _REQUIRED_NATIVE_ATTRIBUTES if _resolve_attribute(native, path) is None
    ]
    if missing:
        _raise_incompatible_native("missing required native attributes: " + ", ".join(missing))


def _resolve_attribute(root: object, path: str) -> object | None:
    current = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _raise_incompatible_native(
    reason: str,
    *,
    cause: BaseException | None = None,
) -> None:
    error = pytest.UsageError(
        "the installed PyNeurale native extension is incompatible with the "
        f"Python package ({reason}); rebuild and reinstall the package"
    )
    if cause is None:
        raise error
    raise error from cause


def pytest_sessionstart(session: pytest.Session) -> None:
    if session.config.getoption("--require-native"):
        _require_native_extension()


def _classification(path: Path) -> str | None:
    resolved = path.resolve()
    # A run can collect files outside tests/ -- pytest applies the rootdir's
    # conftest to any path named on the command line, and benchmarks/ carries
    # its own test_harness.py. Those have no classification here; asking
    # relative_to() for one raises and takes the whole session down with an
    # INTERNALERROR.
    if not resolved.is_relative_to(_TESTS_ROOT):
        return None
    relative = resolved.relative_to(_TESTS_ROOT).as_posix()
    if relative in _PYTHON_ONLY_FILES:
        return "python_only"
    if relative in _NATIVE_FILES or relative.startswith(_NATIVE_PREFIXES):
        return "native"
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        classification = _classification(Path(str(item.path)))
        if classification is not None:
            item.add_marker(getattr(pytest.mark, classification))
