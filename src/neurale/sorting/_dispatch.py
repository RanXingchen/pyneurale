#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from neurale._native_loader import load_native_namespace
from neurale.runtime._device import resolve_device
from neurale.runtime._types import ResolvedDevice


def load_cpu_sorting_operation(operation: str) -> object:
    """Load a CPU-only sorting operation after enforcing the device request."""

    require_cpu_sorting_operation(operation)
    return load_native_namespace(f"sorting.{operation}")


def require_cpu_sorting_operation(operation: str) -> ResolvedDevice:
    """Resolve and enforce CPU execution for one sorting operation."""

    resolution = resolve_device(
        operation=f"sorting.{operation}",
        supports_cuda=False,
    )
    return resolution.resolved
