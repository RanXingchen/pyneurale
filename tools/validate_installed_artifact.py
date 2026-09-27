# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Validate one installed PyNeurale wheel without using the source tree."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

_OPTIONAL_IMPORTS = (
    "OpenGL",
    "PySide6",
    "breathe",
    "freetype",
    "glfw",
    "jsonschema",
    "matplotlib",
    "myst_parser",
    "pydata_sphinx_theme",
    "pytest",
    "rfc8785",
    "ruff",
    "sphinx",
    "torch",
    "uharfbuzz",
    "zarr",
)
_RUNTIME_REQUIREMENTS = frozenset({"numpy", "scipy"})


class InstalledArtifactError(RuntimeError):
    """The installed wheel does not satisfy its runtime artifact contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InstalledArtifactError(message)


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _runtime_requirement_names(distribution: importlib.metadata.Distribution) -> frozenset[str]:
    names: set[str] = set()
    for requirement in distribution.requires or ():
        if "extra ==" in requirement.lower():
            continue
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        _require(match is not None, f"invalid Requires-Dist value: {requirement!r}")
        names.add(_normalize_name(match.group(1)))
    return frozenset(names)


def _assert_optional_dependencies_absent() -> list[str]:
    present = [name for name in _OPTIONAL_IMPORTS if importlib.util.find_spec(name) is not None]
    _require(not present, f"unexpected optional Python dependencies are installed: {present}")
    return list(_OPTIONAL_IMPORTS)


def _typing_contract(package_root: Path) -> dict[str, object]:
    """Validate the installed PEP 561 marker and Center-Out public stub."""

    marker = package_root / "py.typed"
    stub = package_root / "experiments" / "center_out.pyi"
    _require(marker.is_file(), "installed wheel does not contain neurale/py.typed")
    _require(stub.is_file(), "installed wheel does not contain the Center-Out type stub")

    tree = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    declared.update(
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )

    from neurale.experiments import center_out

    missing = sorted(set(center_out.__all__) - declared)
    _require(not missing, f"Center-Out type stub is missing public names: {missing}")
    return {
        "pep561_marker": marker.name,
        "stub": stub.relative_to(package_root).as_posix(),
        "public_names": len(center_out.__all__),
    }


def _cpu_operation() -> dict[str, object]:
    import numpy as np

    from neurale.models import knn

    values = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    indices, distances = knn(values, 1)
    np.testing.assert_array_equal(indices[:, 0], np.array([2, 0, 0]))
    np.testing.assert_array_equal(distances[:, 0], np.array([1.0, 4.0, 1.0]))
    return {"operation": "neurale.models.knn", "indices": indices[:, 0].tolist()}


def _cuda_contract(profile: str, native: object) -> dict[str, object]:
    import numpy as np

    from neurale.exceptions import DeviceUnavailableError
    from neurale.models import knn
    from neurale.runtime import runtime_context

    _require("neurale._native_cuda" not in sys.modules, "CUDA loaded before explicit probe")
    cuda = native.cuda.info()
    values = np.array([[0.0], [1.0], [3.0]], dtype=np.float64)

    if profile == "cuda":
        _require(cuda["compiled"] is True, "CUDA profile is not compiled with CUDA")
        _require(cuda["available"] is True, f"CUDA profile is unavailable: {cuda['reason']}")
        _require(cuda["device_count"] >= 1, "CUDA profile reports no device")
        with runtime_context(device="cuda"):
            indices, distances = knn(values, 1)
        np.testing.assert_array_equal(indices[:, 0], np.array([1, 0, 1]))
        np.testing.assert_array_equal(distances[:, 0], np.array([1.0, 1.0, 4.0]))
        _require("neurale._native_cuda" in sys.modules, "explicit CUDA probe did not load CUDA")
        return {"compiled": True, "available": True, "device_count": cuda["device_count"]}

    _require(cuda["compiled"] is False, f"{profile} profile unexpectedly compiled CUDA")
    _require(cuda["available"] is False, f"{profile} profile unexpectedly exposes CUDA")
    _require(importlib.util.find_spec("neurale._native_cuda") is None, "unexpected CUDA module")
    try:
        with runtime_context(device="cuda"):
            knn(values, 1)
    except DeviceUnavailableError as exc:
        _require("cuda" in str(exc).lower(), "CUDA failure has no device context")
    else:
        raise InstalledArtifactError("explicit CUDA request silently executed without CUDA")
    return {"compiled": False, "available": False, "reason": cuda["reason"]}


def _presentation_contract(profile: str) -> dict[str, object]:
    import neurale.experiments.presentation as presentation
    from neurale.exceptions import DependencyError

    _require(
        "neurale._native" not in sys.modules,
        "presentation namespace import eagerly loaded the native extension",
    )
    if profile != "presentation":
        try:
            presentation.dependency_versions()
        except DependencyError as exc:
            message = str(exc)
            _require(
                "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON" in message,
                "missing presentation error does not explain how to enable it",
            )
            return {"available": False, "lazy_import": True, "error": message}
        raise InstalledArtifactError("missing presentation support did not fail explicitly")

    versions = presentation.dependency_versions()
    _require(
        set(versions) == {"glfw", "opengl_minimum", "freetype", "harfbuzz"},
        f"unexpected presentation dependency report: {versions}",
    )
    _require(versions["opengl_minimum"] == (3, 3), "unexpected OpenGL baseline")

    return {
        "available": True,
        "lazy_import": True,
        "dependency_versions": versions,
    }


def validate_installed_artifact(profile: str) -> dict[str, object]:
    """Validate and return machine-readable evidence for an installed profile."""

    if profile not in {"core", "presentation", "cuda"}:
        raise InstalledArtifactError(f"unknown artifact profile: {profile!r}")

    distribution = importlib.metadata.distribution("pyneurale")
    requirements = _runtime_requirement_names(distribution)
    _require(
        requirements == _RUNTIME_REQUIREMENTS,
        f"runtime requirements must be {sorted(_RUNTIME_REQUIREMENTS)}, found {sorted(requirements)}",
    )
    optional_absent = _assert_optional_dependencies_absent()

    _require("neurale" not in sys.modules, "neurale was imported before validation")
    import neurale
    import neurale.devices.simulation
    import neurale.pipeline
    import neurale.runtime
    import neurale.signal.simulation
    import neurale.streaming

    package_path = Path(neurale.__file__).resolve()
    environment_root = Path(sys.prefix).resolve()
    _require(
        package_path.is_relative_to(environment_root),
        f"neurale resolved outside the clean environment: {package_path}",
    )
    _require(
        not {
            "neurale._native",
            "neurale._native_cuda",
        }
        & sys.modules.keys(),
        "core/runtime/pipeline import eagerly loaded a native extension",
    )

    typing = _typing_contract(package_path.parent)

    presentation = _presentation_contract(profile)

    native = importlib.import_module("neurale._native")
    _require(
        hasattr(native.experiments.center_out, "_AdaptiveCenterOutSession")
        and hasattr(native.experiments.ssvep, "_SSVEPSession"),
        "the shared native extension is missing experiment sessions",
    )
    build = native.build_info()
    version = importlib.metadata.version("pyneurale")
    _require(neurale.__version__ == version, "package and metadata versions differ")
    _require(build["version"] == version, "native and Python versions differ")
    _require(build["abi_version"] == 1, f"unexpected native ABI: {build['abi_version']}")
    _require(bool(build["compiler"]), "native compiler metadata is empty")
    _require(bool(build["build_type"]), "native build type metadata is empty")
    _require(build["cpu_math_backend"] == "none", "release artifact is not builtin CPU")
    _require(build["cuda_compiled"] is (profile == "cuda"), "CUDA build metadata mismatch")

    pipeline = neurale.pipeline.compile_pipeline(
        neurale.pipeline.PipelinePlan(
            (
                neurale.pipeline.BadChannelRemovalStage(max_impedance_ohm=20_000),
                neurale.pipeline.LineNoiseFilterStage(),
            )
        )
    )
    _require(pipeline.stage_count == 2, "pipeline binding is incomplete")
    device = neurale.devices.simulation.SimulatedNeuralDevice(
        neurale.signal.simulation.SignalGenerator.zeros(2, 1_000.0),
        samples_per_frame=4,
        channel_names=("C1", "C2"),
        channel_impedances_ohm=(10_000.0, 100_000.0),
        paced=False,
    )
    input_schema = device.schema
    output_schema = pipeline.output_schema(input_schema)
    _require(output_schema.signals[0].n_channels == 1, "bad-channel removal was not prepared")
    _require(
        output_schema.signals[0].channel_names == ["C1"],
        "bad-channel removal output schema is incorrect",
    )
    _require(
        output_schema.signals[0].channel_impedances_ohm == [10_000.0],
        "bad-channel removal impedance metadata is incorrect",
    )

    cpu = _cpu_operation()
    cuda = _cuda_contract(profile, native)

    return {
        "profile": profile,
        "version": version,
        "package_path": str(package_path),
        "runtime_requirements": sorted(requirements),
        "optional_dependencies_absent": optional_absent,
        "core_runtime_import_native_lazy": True,
        "pipeline_import_native_lazy": True,
        "typing": typing,
        "pipeline": {
            "available": True,
            "stage_count": pipeline.stage_count,
            "output_channels": output_schema.signals[0].channel_names,
        },
        "build_info": build,
        "cpu": cpu,
        "cuda": cuda,
        "presentation": presentation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "presentation", "cuda"), required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_installed_artifact(args.profile)
    except InstalledArtifactError as exc:
        print(f"installed artifact validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
