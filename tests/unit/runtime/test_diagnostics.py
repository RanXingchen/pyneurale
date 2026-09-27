#!/usr/bin/env python3

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace

import neurale.runtime.diagnostics as diagnostics
from neurale.runtime import runtime_info, show_runtime_info


def test_runtime_info_exposes_only_implemented_probe_options() -> None:
    assert set(inspect.signature(runtime_info).parameters) == {
        "probe_native",
        "probe_cuda",
    }


def test_runtime_info_without_probe_does_not_load_native() -> None:
    sys.modules.pop("neurale._native", None)

    info = runtime_info(probe_native=False)

    assert not info.native.available
    assert info.native.reason == "native extension was not probed."
    assert info.cuda is None
    assert "neurale._native" not in sys.modules


def test_runtime_info_reads_native_and_cuda_contract(monkeypatch) -> None:
    native = ModuleType("neurale._native")
    native.build_info = lambda: {
        "version": "1.2.3",
        "abi_version": 1,
        "compiler": "test compiler",
        "build_type": "Release",
        "cpu_math_backend": "mkl",
        "fft_backend": "mkl",
        "openmp_enabled": True,
        "cuda_compiled": True,
        "cuda_toolkit_version": "13.0",
    }
    native.cpu_info = lambda: {
        "architecture": "x86_64",
        "vendor": "AMD",
        "model": "Test CPU",
        "logical_cores": 16,
        "physical_cores": 8,
    }
    native.threading_info = lambda: {
        "backend": "mkl",
        "num_threads": 4,
    }
    native.cuda = SimpleNamespace(
        info=lambda: {
            "compiled": True,
            "available": True,
            "runtime_version": "13.0",
            "driver_version": "13.1",
            "devices": [
                {
                    "ordinal": 0,
                    "name": "Test GPU",
                    "total_memory": 1024,
                    "compute_capability": (9, 0),
                    "multiprocessor_count": 8,
                }
            ],
        }
    )
    monkeypatch.setattr(diagnostics, "probe_native_build", native.build_info)
    monkeypatch.setattr(diagnostics, "load_native_extension", lambda: native)
    monkeypatch.setattr(
        diagnostics,
        "probe_native_cpu",
        lambda extension: extension.cpu_info(),
    )
    monkeypatch.setattr(
        diagnostics,
        "probe_native_cuda",
        lambda extension: extension.cuda.info(),
    )

    info = runtime_info(probe_native=True, probe_cuda=True)

    assert info.native.available
    assert info.native.cpu_math_backend == "mkl"
    assert info.cpu.vendor == "AMD"
    assert info.cuda is not None
    assert info.cuda.available
    assert info.cuda.devices[0].name == "Test GPU"

    output = show_runtime_info(probe_cuda=True)
    assert "CPU math backend: mkl" in output
    assert "performance should be" in output
