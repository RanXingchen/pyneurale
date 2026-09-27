#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from neurale._native_loader import load_native_namespace
from neurale.exceptions import DeviceUnavailableError, NativeUnavailableError
from neurale.runtime._device import resolve_device
from neurale.runtime._types import ResolvedDevice


def load_models_operation(
    operation: str,
) -> tuple[object, int | None, type[BaseException] | None]:
    resolution = resolve_device(operation=f"models.{operation}")
    if resolution.resolved == "cpu":
        return load_native_namespace(f"models.{operation}"), None, None

    try:
        cuda = load_native_namespace("models.cuda")
        native = getattr(cuda, operation, None)
        if native is None:
            raise NativeUnavailableError(f"CUDA support is unavailable for models.{operation}.")
    except NativeUnavailableError as exc:
        raise DeviceUnavailableError(
            f"CUDA support is unavailable for models.{operation}."
        ) from exc
    device_id = -1 if resolution.cuda_device is None else resolution.cuda_device
    return native, device_id, cuda.DeviceUnavailableError


def load_cpu_models_operation(operation: str) -> object:
    """Load a CPU-only model operation after enforcing the device request."""

    require_cpu_models_operation(operation)
    return load_native_namespace(f"models.{operation}")


def load_restored_models_operation(operation: str) -> object:
    """Load a CPU model kernel for rebuilding an already fitted model.

    Deliberately does *not* resolve the ambient device. Restoring a model from
    parameters a previous fit produced is not a request to run anything: the
    device that fit ran on is a recorded fact, and the runtime the restore
    happens in has no say in it. Resolving here would make loading a saved
    CPU-only decoder fail under an ambient ``device="cuda"`` -- on an artifact
    that is entirely valid and would predict on the CPU either way.
    """

    return load_native_namespace(f"models.{operation}")


def require_cpu_models_operation(operation: str) -> ResolvedDevice:
    """Resolve and enforce CPU execution for one models operation."""

    resolution = resolve_device(
        operation=f"models.{operation}",
        supports_cuda=False,
    )
    return resolution.resolved
