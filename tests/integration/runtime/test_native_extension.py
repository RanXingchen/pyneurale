#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neurale.runtime._native_probe import EXPECTED_NATIVE_ABI


def _native():
    return importlib.import_module("neurale._native")


def test_native_runtime_contract() -> None:
    native = _native()

    build = native.build_info()
    cpu = native.cpu_info()
    threading = native.threading_info()

    assert build["abi_version"] == EXPECTED_NATIVE_ABI
    assert build["cpu_math_backend"] in {"mkl", "none"}
    assert isinstance(build["cuda_compiled"], bool)
    assert cpu["architecture"]
    assert cpu["logical_cores"] >= 1
    assert threading["backend"] in {"mkl", "openmp", "none"}
    assert threading["num_threads"] >= 1


def test_native_thread_limit_can_be_changed_and_restored() -> None:
    native = _native()
    previous = native.threading.get_num_threads()

    try:
        native.threading.set_num_threads(1)
        assert native.threading.get_num_threads() == 1
    finally:
        native.threading.set_num_threads(previous)


def test_native_build_info_does_not_load_cuda_extension(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    script = """
import sys
import neurale._native as native

native.build_info()
assert "neurale._native_cuda" not in sys.modules
native.cuda.info()
build = native.build_info()
assert ("neurale._native_cuda" in sys.modules) is build["cuda_compiled"]
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        cwd=tmp_path,
    )


@pytest.mark.gpu
def test_native_cuda_probe() -> None:
    native = _native()
    info = native.cuda.info()

    assert isinstance(info["compiled"], bool)
    assert isinstance(info["available"], bool)
    if info["available"]:
        assert info["device_count"] >= 1
        assert info["devices"][0]["name"]
        current = native.cuda.current_device()
        native.cuda.set_device(current)
        assert native.cuda.current_device() == current
