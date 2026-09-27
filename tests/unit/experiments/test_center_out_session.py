#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
from inspect import signature

import pytest

import neurale.streaming as streaming
from neurale.exceptions import ValidationError
from neurale.experiments import center_out as co
from neurale.experiments.center_out_training import AssistanceBlock


def _task(*, movement_timeout_seconds: float = 2.0) -> object:
    request = co.RadialLayoutRequest(radius=0.5, center_id=1, ids=[2, 3], spokes=[0, 4])
    layout = co.build_radial_layout(request)
    return co.CenterOutTask(
        geometry_unit=co.GeometryUnit.NORMALIZED,
        layout=layout,
        acceptance=0.1,
        movement_timeout_seconds=movement_timeout_seconds,
        selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
        seed=17,
    )


def _guidance() -> object:
    return co.CenterOutGuidanceConfig(
        max_speed=1.0,
        acceleration=4.0,
        arrival_radius=0.05,
    )


def _schema(*, rate_hz: int = 100) -> object:
    signal = streaming.SignalSchema(
        9,
        streaming.SignalDType.FLOAT64,
        2,
        1,
        1,
        streaming.RationalRate(rate_hz),
        1,
        physical_unit=streaming.PhysicalUnit.DIMENSIONLESS,
    )
    return streaming.StreamSchema(3, [signal])


def test_protocol_derives_trial_count_from_assistance_schedule() -> None:
    config = co.CenterOutProtocol(
        task=_task(),
        assistance_blocks=(AssistanceBlock(1.0, 1), AssistanceBlock(0.5, 1)),
    )
    assert config.guidance is None
    assert config.initial_position is None
    assert config.trials == 2
    assert tuple(block.assistance for block in config.assistance_blocks) == (1.0, 0.5)


def test_protocol_exposes_only_task_schedule_and_optional_safety() -> None:
    assert tuple(signature(co.CenterOutProtocol).parameters) == (
        "task",
        "assistance_blocks",
        "guidance",
        "initial_position",
        "safety",
    )


def test_session_safety_uses_seconds_and_is_optional() -> None:
    from neurale.experiments.center_out_session import _stale_interval_ns

    assert _stale_interval_ns(co.CenterOutSafetyConfig()) == 0
    assert _stale_interval_ns(co.CenterOutSafetyConfig(stale_after_seconds=0.25)) == 250_000_000
    with pytest.raises(ValueError, match="greater than zero"):
        co.CenterOutSafetyConfig(stale_after_seconds=0.0)


def test_native_config_derives_wiring_trial_count_and_queue_bounds() -> None:
    from neurale._native_loader import load_native_namespace
    from neurale.experiments.center_out_session import (
        _native_controller_config,
        _resolve_guidance,
    )

    config = co.CenterOutProtocol(
        task=_task(),
        assistance_blocks=(AssistanceBlock(1.0, 1), AssistanceBlock(0.5, 1)),
        safety=co.CenterOutSafetyConfig(stale_after_seconds=0.25),
    )
    schema = _schema()
    guidance, resolver_version = _resolve_guidance(config, schema)
    native = _native_controller_config(
        load_native_namespace("experiments.center_out"),
        config,
        schema,
        guidance=guidance,
        guidance_resolver_version=resolver_version,
    )

    assert native.decoded_signal_id == 9
    assert native.paradigm == 1
    assert sum(block.trials for block in native.assistance_blocks) == 2
    assert native.velocity_space.id == 1
    assert native.initial_position.x == config.task.layout.center.pos.x
    assert native.initial_position.y == config.task.layout.center.pos.y
    assert native.cursor_min.x == pytest.approx(-1.175)
    assert native.cursor_min.y == pytest.approx(-1.175)
    assert native.cursor_max.x == pytest.approx(1.175)
    assert native.cursor_max.y == pytest.approx(1.175)
    assert native.trace_capacity >= 4096
    assert native.max_observation_interval_ns == 250_000_000


def test_protocol_rejects_initial_position_outside_cursor_workspace() -> None:
    with pytest.raises(ValidationError, match="inside the Center-Out workspace"):
        co.CenterOutProtocol(
            task=_task(),
            assistance_blocks=(AssistanceBlock(1.0, 1),),
            initial_position=(2.0, 0.0),
        )


def test_trace_capacity_covers_the_declared_high_rate_session() -> None:
    from neurale.experiments.center_out_session import (
        _trace_capacity,
        _training_capture_capacity,
    )

    task = _task()
    schema = _schema(rate_hz=4000)
    rows = _training_capture_capacity(task, schema, 2)

    assert _trace_capacity(task, schema, 2) >= rows
    assert rows > 4096


def test_protocol_rejects_arrival_outside_usable_acceptance() -> None:
    guidance = co.CenterOutGuidanceConfig(
        max_speed=1.0,
        acceleration=4.0,
        arrival_radius=0.11,
    )
    with pytest.raises(ValidationError, match=r"usable acceptance radius \(0.1\)"):
        co.CenterOutProtocol(
            task=_task(),
            guidance=guidance,
            assistance_blocks=(AssistanceBlock(1.0, 1),),
        )


def test_automatic_guidance_is_deterministic_and_uses_half_the_usable_radius() -> None:
    from neurale.experiments.center_out_session import _resolve_guidance

    config = co.CenterOutProtocol(
        task=_task(),
        assistance_blocks=(AssistanceBlock(1.0, 1),),
    )
    first, first_version = _resolve_guidance(config, _schema())
    second, second_version = _resolve_guidance(config, _schema())

    assert first == second
    assert first_version == second_version == 1
    assert first.arrival_radius == pytest.approx(0.05)
    assert first.deceleration == first.acceleration
    assert co.validate_against(first, config.task) == co.validate(first)


def test_explicit_guidance_must_reach_during_full_assistance() -> None:
    from neurale.experiments.center_out_session import _resolve_guidance

    config = co.CenterOutProtocol(
        task=_task(),
        assistance_blocks=(AssistanceBlock(1.0, 1),),
        guidance=co.CenterOutGuidanceConfig(
            max_speed=0.01,
            acceleration=0.01,
            arrival_radius=0.05,
        ),
    )
    with pytest.raises(ValidationError, match="cannot complete the 100%-assisted"):
        _resolve_guidance(config, _schema())


def test_automatic_guidance_rejects_timeout_shorter_than_controlled_reach() -> None:
    from neurale.experiments.center_out_session import _resolve_guidance

    config = co.CenterOutProtocol(
        task=_task(movement_timeout_seconds=0.02),
        assistance_blocks=(AssistanceBlock(1.0, 1),),
    )
    with pytest.raises(ValidationError, match="cannot complete the 100%-assisted"):
        _resolve_guidance(config, _schema())


def test_session_module_import_is_native_and_presentation_lazy() -> None:
    script = """
import sys
import neurale.experiments.center_out_session
assert 'neurale._native' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)


def test_sessions_share_the_core_native_extension() -> None:
    from neurale._native_loader import load_native_namespace

    center_out = load_native_namespace("experiments.center_out")
    ssvep = load_native_namespace("experiments.ssvep")
    assert (
        center_out._AdaptiveCenterOutSession.__module__ == "neurale._native.experiments.center_out"
    )
    assert ssvep._SSVEPSession.__module__ == "neurale._native.experiments.ssvep"
