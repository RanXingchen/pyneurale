#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Collect runtime, native extension, CPU, CUDA, and threading diagnostics."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from neurale._native_loader import load_native_extension
from neurale.exceptions import NativeUnavailableError

from . import _state
from ._native_probe import (
    probe_native_build,
    probe_native_cpu,
    probe_native_cuda,
)
from ._types import ResolvedDevice
from .config import RuntimeConfig
from .context import get_runtime_config
from .threading import ThreadingInfo, threading_info


@dataclass(frozen=True, slots=True)
class NativeBuildInfo:
    """Describe the installed native extension and its build options.

    Parameters
    ----------
    available : bool
        Whether the native extension is available.
    version : str or None, optional
        Native extension version.
    abi_version : int or None, optional
        Native ABI version.
    compiler : str or None, optional
        Compiler identifier.
    build_type : str or None, optional
        CMake build configuration.
    cpu_math_backend : str or None, optional
        CPU math backend name.
    fft_backend : str or None, optional
        FFT backend name.
    openmp_enabled : bool or None, optional
        Whether OpenMP support is enabled.
    cuda_compiled : bool or None, optional
        Whether CUDA support was compiled.
    cuda_toolkit_version : str or None, optional
        CUDA toolkit version used for compilation.
    reason : str or None, optional
        Explanation when the extension is unavailable.
    """

    available: bool
    version: str | None = None
    abi_version: int | None = None
    compiler: str | None = None
    build_type: str | None = None
    cpu_math_backend: str | None = None
    fft_backend: str | None = None
    openmp_enabled: bool | None = None
    cuda_compiled: bool | None = None
    cuda_toolkit_version: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CpuInfo:
    """Describe the host CPU.

    Parameters
    ----------
    architecture : str
        Host architecture.
    vendor : str or None
        Normalized CPU vendor.
    model : str or None
        CPU model or brand string.
    logical_cores : int or None
        Visible logical processor count.
    physical_cores : int or None
        Detected physical core count.
    """

    architecture: str
    vendor: str | None
    model: str | None
    logical_cores: int | None
    physical_cores: int | None


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    """Describe one CUDA device reported by the native extension.

    Parameters
    ----------
    ordinal : int
        Zero-based device ordinal.
    name : str
        Device name.
    total_memory : int or None, optional
        Total device memory in bytes.
    compute_capability : tuple of int or None, optional
        Compute capability major and minor versions.
    n_multiprocessors : int or None, optional
        Streaming multiprocessor count.
    """

    ordinal: int
    name: str
    total_memory: int | None = None
    compute_capability: tuple[int, int] | None = None
    n_multiprocessors: int | None = None


@dataclass(frozen=True, slots=True)
class CudaInfo:
    """Describe native CUDA build and runtime availability.

    Parameters
    ----------
    compiled : bool
        Whether CUDA support was compiled.
    available : bool
        Whether a usable CUDA runtime and device are available.
    runtime_version : str or None, optional
        CUDA runtime version.
    driver_version : str or None, optional
        CUDA driver version.
    devices : tuple of neurale.runtime.diagnostics.CudaDeviceInfo, optional
        Enumerated devices.
    reason : str or None, optional
        Explanation when CUDA is unavailable.
    """

    compiled: bool
    available: bool
    runtime_version: str | None = None
    driver_version: str | None = None
    devices: tuple[CudaDeviceInfo, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Collect package, configuration, native, CPU, CUDA, and threading data.

    Parameters
    ----------
    package_version : str
        Installed PyNeurale version.
    python_version : str
        Active Python version.
    platform : str
        Host platform description.
    architecture : str
        Host architecture.
    config : neurale.runtime.config.RuntimeConfig
        Active runtime configuration.
    resolved_device : {"cpu", "cuda"} or None
        Resolved execution device after explicit initialization.
    cpu : neurale.runtime.diagnostics.CpuInfo
        CPU diagnostics.
    native : neurale.runtime.diagnostics.NativeBuildInfo
        Native extension diagnostics.
    cuda : neurale.runtime.diagnostics.CudaInfo or None
        CUDA diagnostics when requested.
    threading : neurale.runtime.threading.ThreadingInfo
        Thread configuration diagnostics.
    """

    package_version: str
    python_version: str
    platform: str
    architecture: str
    config: RuntimeConfig
    resolved_device: ResolvedDevice | None
    cpu: CpuInfo
    native: NativeBuildInfo
    cuda: CudaInfo | None
    threading: ThreadingInfo


def _package_version() -> str:
    try:
        return metadata.version("pyneurale")
    except metadata.PackageNotFoundError:
        return "0+unknown"


def _cpu_vendor(model: str | None) -> str | None:
    if model is None:
        return None
    lowered = model.lower()
    if "intel" in lowered:
        return "Intel"
    if "amd" in lowered or "ryzen" in lowered or "epyc" in lowered:
        return "AMD"
    return None


def _normalize_cpu_vendor(vendor: Any, model: str | None) -> str | None:
    if vendor is None:
        return _cpu_vendor(model)
    value = str(vendor)
    lowered = value.lower()
    if lowered in {"genuineintel", "intel"}:
        return "Intel"
    if lowered in {"authenticamd", "amd"}:
        return "AMD"
    return value


def _cpu_info(native_data: dict[str, Any] | None = None) -> CpuInfo:
    model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None
    data = {} if native_data is None else native_data
    selected_model = data.get("model", model)
    return CpuInfo(
        architecture=str(data.get("architecture", platform.machine())),
        vendor=_normalize_cpu_vendor(data.get("vendor"), selected_model),
        model=selected_model,
        logical_cores=data.get("logical_cores", os.cpu_count()),
        physical_cores=data.get("physical_cores"),
    )


def _native_info(data: dict[str, Any] | None, reason: str | None) -> NativeBuildInfo:
    if data is None:
        return NativeBuildInfo(available=False, reason=reason)
    return NativeBuildInfo(
        available=True,
        version=_optional_str(data.get("version")),
        abi_version=_optional_int(data.get("abi_version")),
        compiler=_optional_str(data.get("compiler")),
        build_type=_optional_str(data.get("build_type")),
        cpu_math_backend=_optional_str(data.get("cpu_math_backend")),
        fft_backend=_optional_str(data.get("fft_backend")),
        openmp_enabled=_optional_bool(data.get("openmp_enabled")),
        cuda_compiled=_optional_bool(data.get("cuda_compiled")),
        cuda_toolkit_version=_optional_str(data.get("cuda_toolkit_version")),
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _cuda_version(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"{value // 1000}.{(value % 1000) // 10}"
    return str(value)


def _cuda_info(data: dict[str, Any]) -> CudaInfo:
    devices = []
    for item in data.get("devices", ()) or ():
        capability = item.get("compute_capability")
        devices.append(
            CudaDeviceInfo(
                ordinal=int(item["ordinal"]),
                name=str(item["name"]),
                total_memory=_optional_int(item.get("total_memory")),
                compute_capability=(
                    (int(capability[0]), int(capability[1])) if capability is not None else None
                ),
                n_multiprocessors=_optional_int(item.get("multiprocessor_count")),
            )
        )
    return CudaInfo(
        compiled=bool(data.get("compiled", False)),
        available=bool(data.get("available", False)),
        runtime_version=_cuda_version(data.get("runtime_version")),
        driver_version=_cuda_version(data.get("driver_version")),
        devices=tuple(devices),
        reason=_optional_str(data.get("reason")),
    )


def runtime_info(
    *,
    probe_native: bool = True,
    probe_cuda: bool = False,
) -> RuntimeInfo:
    """Collect runtime diagnostics.

    Parameters
    ----------
    probe_native : bool, default=True
        Attempt to load and inspect the native extension.
    probe_cuda : bool, default=False
        Query CUDA runtime and devices. This may initialize CUDA.
    Returns
    -------
    neurale.runtime.diagnostics.RuntimeInfo
        Current package, platform, native, CUDA, and threading information.
    """

    build_data: dict[str, Any] | None = None
    native_cpu: dict[str, Any] | None = None
    cuda: CudaInfo | None = None
    native_reason: str | None = "native extension was not probed."
    native = None

    if probe_native or probe_cuda:
        try:
            build_data = probe_native_build()
            native_reason = "native extension is not installed." if build_data is None else None
            native = load_native_extension() if build_data is not None else None
            if native is not None:
                native_cpu = probe_native_cpu(native)
                if probe_cuda:
                    cuda = _cuda_info(probe_native_cuda(native))
        except NativeUnavailableError as exc:
            native_reason = str(exc)

    if probe_cuda and native is None:
        cuda = CudaInfo(
            compiled=False,
            available=False,
            reason=native_reason,
        )

    config = get_runtime_config()
    with _state.LOCK:
        resolved_device = _state.RESOLVED_DEVICE if config == _state.INITIALIZED_CONFIG else None

    return RuntimeInfo(
        package_version=_package_version(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        architecture=platform.machine(),
        config=config,
        resolved_device=resolved_device,
        cpu=_cpu_info(native_cpu),
        native=_native_info(build_data, native_reason),
        cuda=cuda,
        threading=threading_info(probe_native=native is not None),
    )


def show_runtime_info(*, probe_cuda: bool = False) -> str:
    """Format runtime diagnostics for display.

    Parameters
    ----------
    probe_cuda : bool, default=False
        Query CUDA runtime and devices before formatting.

    Returns
    -------
    str
        Human-readable multi-line diagnostic report.
    """

    info = runtime_info(probe_native=True, probe_cuda=probe_cuda)
    lines = [
        "PyNeurale runtime",
        "-----------------",
        f"PyNeurale:          {info.package_version}",
        f"Python:             {info.python_version}",
        f"Platform:           {info.platform}",
        f"Architecture:       {info.architecture}",
        f"CPU vendor:         {info.cpu.vendor or 'unknown'}",
        f"CPU model:          {info.cpu.model or 'unknown'}",
        "",
        "Configuration",
        f"  requested device: {info.config.device}",
        f"  resolved device:  {info.resolved_device or 'not initialized'}",
        f"  threads:          {info.config.num_threads or 'default'}",
        f"  random seed:      {info.config.random_seed}",
        f"  deterministic:    {str(info.config.deterministic).lower()}",
        f"  strict:           {str(info.config.strict).lower()}",
        "",
        "Native extension",
        f"  available:        {'yes' if info.native.available else 'no'}",
    ]
    if info.native.available:
        lines.extend(
            [
                f"  ABI:              {info.native.abi_version}",
                f"  compiler:         {info.native.compiler or 'unknown'}",
                f"  build type:       {info.native.build_type or 'unknown'}",
                f"  CPU math backend: {info.native.cpu_math_backend or 'unknown'}",
                f"  FFT backend:      {info.native.fft_backend or 'unknown'}",
                f"  CUDA compiled:    {'yes' if info.native.cuda_compiled else 'no'}",
            ]
        )
    else:
        lines.append(f"  reason:           {info.native.reason}")

    if (
        info.cpu.vendor == "AMD"
        and info.native.cpu_math_backend is not None
        and info.native.cpu_math_backend.lower() == "mkl"
    ):
        lines.extend(
            [
                "",
                "Compatibility",
                "  CPU math note:    MKL is supported, but performance should be",
                "                    validated on this AMD processor.",
            ]
        )

    lines.extend(["", "CUDA"])
    if not probe_cuda:
        lines.append("  probed:           no (pass probe_cuda=True to query the CUDA runtime)")
    elif info.cuda is not None:
        lines.extend(
            [
                f"  compiled:         {'yes' if info.cuda.compiled else 'no'}",
                f"  available:        {'yes' if info.cuda.available else 'no'}",
                f"  devices:          {len(info.cuda.devices)}",
            ]
        )
        if info.cuda.reason:
            lines.append(f"  reason:           {info.cuda.reason}")
    return "\n".join(lines)
