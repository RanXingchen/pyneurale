#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Load and validate native runtime capabilities on explicit request."""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

from neurale._native_loader import load_native_extension
from neurale.exceptions import NativeExtensionError, NativeUnavailableError

EXPECTED_NATIVE_ABI = 1


def _mapping_result(value: Any, operation: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeExtensionError(
            f"native {operation}() must return a mapping, got {type(value).__name__}."
        )
    return dict(value)


def probe_native_build() -> dict[str, Any] | None:
    try:
        native = load_native_extension()
    except NativeUnavailableError:
        return None

    build_info = getattr(native, "build_info", None)
    if not callable(build_info):
        raise NativeExtensionError("native extension does not provide build_info().")
    result = _mapping_result(build_info(), "build_info")
    abi = result.get("abi_version")
    if abi != EXPECTED_NATIVE_ABI:
        raise NativeExtensionError(
            f"Python package expects native ABI {EXPECTED_NATIVE_ABI}, "
            f"but extension provides ABI {abi!r}."
        )
    return result


def probe_native_cpu(native: ModuleType) -> dict[str, Any] | None:
    function = getattr(native, "cpu_info", None)
    if not callable(function):
        return None
    return _mapping_result(function(), "cpu_info")


def probe_native_threading(native: ModuleType) -> dict[str, Any] | None:
    function = getattr(native, "threading_info", None)
    if callable(function):
        return _mapping_result(function(), "threading_info")
    namespace = getattr(native, "threading", None)
    function = getattr(namespace, "info", None)
    if callable(function):
        return _mapping_result(function(), "threading.info")
    return None


def probe_native_cuda(native: ModuleType) -> dict[str, Any]:
    namespace = getattr(native, "cuda", None)
    if namespace is None:
        return {
            "compiled": False,
            "available": False,
            "reason": "native extension does not expose CUDA support.",
        }
    info = getattr(namespace, "info", None)
    if callable(info):
        return _mapping_result(info(), "cuda.info")
    build_info = getattr(namespace, "build_info", None)
    result = _mapping_result(build_info(), "cuda.build_info") if callable(build_info) else {}
    available = getattr(namespace, "is_available", None)
    if callable(available):
        result["available"] = bool(available())
    devices = getattr(namespace, "devices", None)
    if callable(devices):
        result["devices"] = devices()
    return result
