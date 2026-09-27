#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Store process-wide and context-local runtime state."""

from __future__ import annotations

from contextvars import ContextVar
from threading import RLock

from ._types import ResolvedDevice
from .config import RuntimeConfig

LOCK = RLock()
GLOBAL_CONFIG = RuntimeConfig.from_environment()
CONTEXT_CONFIG: ContextVar[RuntimeConfig | None] = ContextVar(
    "neurale_runtime_config",
    default=None,
)
INITIALIZED = False
INITIALIZED_CONFIG: RuntimeConfig | None = None
RESOLVED_DEVICE: ResolvedDevice | None = None
