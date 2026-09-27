#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import FrozenInstanceError
from inspect import signature

import pytest

import neurale.streaming as streaming


def _schema() -> object:
    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT32,
        2,
        4,
        4,
        streaming.RationalRate(1_000, 1),
        7,
    )
    return streaming.StreamSchema(11, [signal])


def test_public_realtime_config_is_valid_minimal_intent() -> None:
    config = streaming.RealtimeConfig()

    assert tuple(signature(streaming.RealtimeConfig).parameters) == (
        "latency_budget_seconds",
        "source_timeout_seconds",
        "platform",
    )
    assert config.latency_budget_seconds == 0.1
    assert config.source_timeout_seconds == 1.0
    assert config.platform is None
    assert not hasattr(config, "pool_capacity")
    assert not hasattr(config, "buffer_size")
    assert not hasattr(config, "validate")
    assert not hasattr(streaming, "PoolCapacityBudget")
    assert not hasattr(streaming, "RealtimeConfigMode")
    with pytest.raises(FrozenInstanceError):
        config.latency_budget_seconds = 1.0


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"latency_budget_seconds": 0}, ValueError),
        ({"source_timeout_seconds": float("inf")}, ValueError),
        ({"latency_budget_seconds": True}, TypeError),
        ({"platform": object()}, TypeError),
    ],
)
def test_public_realtime_config_rejects_invalid_intent(values, error) -> None:
    with pytest.raises(error):
        streaming.RealtimeConfig(**values)


def test_runner_resolves_native_storage_and_profile() -> None:
    schema = _schema()
    sink = streaming.CountingNativeConsumer()
    runner = streaming.StreamRunner(
        schema,
        streaming.RealtimeConfig(
            latency_budget_seconds=0.02,
            source_timeout_seconds=0.5,
        ),
        streaming.SyntheticNativeSource(schema, 3),
        streaming.IdentityNativeProcessor(),
        sink,
        profile=streaming.ExecutionProfile.REALTIME,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert sink.frame_count == 3
