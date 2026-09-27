#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Researcher-facing intent for the native streaming runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any


class RealtimeSchedulingPolicy(StrEnum):
    """Optional operating-system scheduling policy for one native worker."""

    NORMAL = "normal"
    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"


@dataclass(frozen=True, slots=True)
class RealtimeThreadConfig:
    """Advanced native-thread placement and scheduling request."""

    cpu_affinity_mask: int = 0
    scheduling_policy: RealtimeSchedulingPolicy = RealtimeSchedulingPolicy.NORMAL
    priority: int = 0
    stack_size: int = 0
    prefault_stack_bytes: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.cpu_affinity_mask, bool) or not isinstance(self.cpu_affinity_mask, int):
            raise TypeError("cpu_affinity_mask must be an integer")
        if not 0 <= self.cpu_affinity_mask <= (1 << 64) - 1:
            raise ValueError("cpu_affinity_mask must fit an unsigned 64-bit mask")
        if not isinstance(self.scheduling_policy, RealtimeSchedulingPolicy):
            raise TypeError("scheduling_policy must be a RealtimeSchedulingPolicy")
        for name in ("priority", "stack_size", "prefault_stack_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name != "priority" and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.stack_size and self.prefault_stack_bytes > self.stack_size:
            raise ValueError("prefault_stack_bytes cannot exceed stack_size")


@dataclass(frozen=True, slots=True)
class RealtimePlatformConfig:
    """Advanced operating-system configuration for native runtime workers."""

    lock_memory: bool = False
    prefault_pools: bool = False
    acquisition: RealtimeThreadConfig = field(default_factory=RealtimeThreadConfig)
    processing: RealtimeThreadConfig = field(default_factory=RealtimeThreadConfig)
    actuator: RealtimeThreadConfig = field(default_factory=RealtimeThreadConfig)
    observer_dispatch: RealtimeThreadConfig = field(default_factory=RealtimeThreadConfig)
    watchdog: RealtimeThreadConfig = field(default_factory=RealtimeThreadConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.lock_memory, bool):
            raise TypeError("lock_memory must be a bool")
        if not isinstance(self.prefault_pools, bool):
            raise TypeError("prefault_pools must be a bool")
        for name in (
            "acquisition",
            "processing",
            "actuator",
            "observer_dispatch",
            "watchdog",
        ):
            if not isinstance(getattr(self, name), RealtimeThreadConfig):
                raise TypeError(f"{name} must be a RealtimeThreadConfig")


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """High-level latency and source-liveness intent for native streaming."""

    latency_budget_seconds: float = 0.1
    source_timeout_seconds: float = 1.0
    platform: RealtimePlatformConfig | None = None

    def __post_init__(self) -> None:
        for name in ("latency_budget_seconds", "source_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.platform is not None and not isinstance(self.platform, RealtimePlatformConfig):
            raise TypeError("platform must be a RealtimePlatformConfig or None")


def _native_realtime_config(
    config: RealtimeConfig,
    profile: Any,
    *,
    schema_probe: bool = False,
) -> Any:
    from . import _native

    if isinstance(config, _native.RealtimeConfig):
        return config
    if not isinstance(config, RealtimeConfig):
        raise TypeError("config must be a RealtimeConfig")

    native = _native.RealtimeConfig()
    latency = timedelta(seconds=config.latency_budget_seconds)
    native.source_stall_timeout = timedelta(seconds=config.source_timeout_seconds)
    native.max_ingress_dwell = latency
    native.processor_execution_deadline = latency
    native.max_source_to_actuator_age = latency
    native.actuator_deadline = latency
    native.max_output_age = timedelta(0)
    native.watchdog_period = timedelta(0)

    profile_value = getattr(profile, "value", profile)
    platform = config.platform or RealtimePlatformConfig()
    native_platform = _native.RealtimePlatformConfig()
    native_platform.mode = (
        _native.RealtimeConfigMode.STRICT
        if profile_value == "realtime"
        else (
            _native.RealtimeConfigMode.BEST_EFFORT
            if config.platform is not None
            else _native.RealtimeConfigMode.DISABLED
        )
    )
    native_platform.lock_memory = platform.lock_memory
    native_platform.prefault_pools = platform.prefault_pools
    for role in (
        "acquisition",
        "processing",
        "actuator",
        "observer_dispatch",
        "watchdog",
    ):
        request = getattr(platform, role)
        thread = _native.RealtimeThreadConfig()
        thread.cpu_affinity_mask = request.cpu_affinity_mask
        thread.scheduling_policy = getattr(
            _native.RealtimeSchedulingPolicy, request.scheduling_policy.name
        )
        thread.priority = request.priority
        thread.stack_size = request.stack_size
        thread.prefault_stack_bytes = request.prefault_stack_bytes
        setattr(native_platform, role, thread)
    native.platform = native_platform

    if schema_probe:
        native.max_process_outputs = (1 << 63) - 1
        native.max_flush_outputs = (1 << 63) - 1
        native.pool_capacity.processor_owned = (1 << 63) - 1
    else:
        native._automatic_resources = True
    return native
