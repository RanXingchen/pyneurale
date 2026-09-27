#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Runtime policy, optional dependency, threading, and diagnostics tools.

Importing this module does not load the native extension, initialize CUDA,
configure logging, alter thread pools, or seed random number generators.
"""

from ._logging import configure_logging, is_logging_configured, reset_logging
from .capabilities import Capability
from .config import RuntimeConfig
from .context import (
    configure_runtime,
    get_runtime_config,
    reset_runtime,
    runtime_context,
)
from .dependencies import (
    DependencyStatus,
    dependency_status,
    has_dependency,
    require_dependency,
)
from .diagnostics import (
    CpuInfo,
    CudaDeviceInfo,
    CudaInfo,
    NativeBuildInfo,
    RuntimeInfo,
    runtime_info,
    show_runtime_info,
)
from .initialization import initialize_runtime
from .random import RandomState, configure_random_seed
from .threading import (
    ThreadingInfo,
    configure_threading,
    thread_limit,
    threading_info,
)

__all__ = [
    "Capability",
    "CpuInfo",
    "CudaDeviceInfo",
    "CudaInfo",
    "DependencyStatus",
    "NativeBuildInfo",
    "RandomState",
    "RuntimeConfig",
    "RuntimeInfo",
    "ThreadingInfo",
    "configure_logging",
    "configure_random_seed",
    "configure_runtime",
    "configure_threading",
    "dependency_status",
    "get_runtime_config",
    "has_dependency",
    "initialize_runtime",
    "is_logging_configured",
    "require_dependency",
    "reset_logging",
    "reset_runtime",
    "runtime_context",
    "runtime_info",
    "show_runtime_info",
    "thread_limit",
    "threading_info",
]
