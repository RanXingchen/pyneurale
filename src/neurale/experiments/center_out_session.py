#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Internal Center-Out protocol validation and native preparation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neurale.exceptions import ValidationError

from ._center_out_geometry import _center_out_cursor_bounds
from .center_out import CenterOutGuidanceConfig
from .center_out_training import AssistanceBlock

if TYPE_CHECKING:
    from .center_out import CenterOutTask


@dataclass(frozen=True, slots=True, kw_only=True)
class CenterOutSafetyConfig:
    """Optional stale-command policy for one Center-Out session."""

    stale_after_seconds: float | None = None

    def __post_init__(self) -> None:
        value = self.stale_after_seconds
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("stale_after_seconds must be a number or None")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("stale_after_seconds must be finite and greater than zero")
        object.__setattr__(self, "stale_after_seconds", value)


@dataclass(frozen=True, slots=True, kw_only=True)
class CenterOutProtocol:
    """Center-Out task and assistance protocol for one native run."""

    task: CenterOutTask
    assistance_blocks: tuple[AssistanceBlock, ...]
    guidance: CenterOutGuidanceConfig | None = None
    initial_position: tuple[float, float] | None = None
    safety: CenterOutSafetyConfig = field(default_factory=CenterOutSafetyConfig)

    def __post_init__(self) -> None:
        from .center_out import CenterOutTask

        if not isinstance(self.task, CenterOutTask):
            raise TypeError("task must be a CenterOutTask")
        if self.guidance is not None and not isinstance(self.guidance, CenterOutGuidanceConfig):
            raise TypeError("guidance must be a CenterOutGuidanceConfig or None")
        if not isinstance(self.safety, CenterOutSafetyConfig):
            raise TypeError("safety must be a CenterOutSafetyConfig")
        blocks = tuple(self.assistance_blocks)
        if not blocks:
            raise ValidationError("assistance_blocks must not be empty.")
        if not all(isinstance(block, AssistanceBlock) for block in blocks):
            raise TypeError("assistance_blocks must contain AssistanceBlock values")
        position = self.initial_position
        if position is not None:
            if not isinstance(position, tuple) or len(position) != 2:
                raise ValidationError("initial_position must be a two-value tuple or None.")
            position = tuple(float(value) for value in position)
            if not all(math.isfinite(value) for value in position):
                raise ValidationError("initial_position values must be finite.")
            left, bottom, right, top = _center_out_cursor_bounds(self.task)
            if not left <= position[0] <= right or not bottom <= position[1] <= top:
                raise ValidationError("initial_position must be inside the Center-Out workspace.")
        usable_radius = _usable_acceptance_radius(self.task)
        if self.guidance is not None and self.guidance.arrival_radius > usable_radius:
            raise ValidationError(
                "guidance.arrival_radius must not exceed the task's usable "
                f"acceptance radius ({usable_radius:g})."
            )
        object.__setattr__(self, "assistance_blocks", blocks)
        object.__setattr__(self, "initial_position", position)

    @property
    def trials(self) -> int:
        """Total trials derived from the assistance schedule."""

        return sum(block.trials for block in self.assistance_blocks)


@dataclass(frozen=True, slots=True)
class CenterOutTrainingObservation:
    """One decoder input row paired with its Center-Out supervision state."""

    sample_idx: int
    features: tuple[float, ...]
    time_ns: int
    trial: Any
    target_position: tuple[float, float]
    cursor_position: tuple[float, float]
    cursor_velocity: tuple[float, float]
    guidance_velocity: tuple[float, float]
    trial_stop: bool


def _native_controller_config(
    native: Any,
    config: CenterOutProtocol,
    decoded_schema: Any,
    *,
    decoded_signal_id: int | None = None,
    guidance: CenterOutGuidanceConfig | None = None,
    guidance_resolver_version: int = 0,
    training_capture_capacity: int = 0,
    presentation_state_capacity: int = 0,
) -> Any:
    from neurale.experiments import (
        AbnormalPolicySet,
        CommandAxis,
        CommandAxisName,
        CommandFrame,
        CommandSpace,
        CommandUnit,
    )
    from neurale.experiments.assistance import AssistanceMethod, LinearAssistance
    from neurale.experiments.center_out import (
        GeometryUnit,
        WorkspacePoint,
        _native_guidance_config,
        _with_trial_limit,
    )

    units = {
        GeometryUnit.DIMENSIONLESS: CommandUnit.DIMENSIONLESS,
        GeometryUnit.NORMALIZED: CommandUnit.NORMALIZED,
    }
    try:
        unit = units[config.task.geometry_unit]
    except KeyError as exc:
        raise ValidationError(
            "Center-Out cursor control supports dimensionless or normalized geometry."
        ) from exc
    signal = _decoded_signal(decoded_schema)
    value = native._CenterOutControllerConfig()
    value.decoded_signal_id = signal.id if decoded_signal_id is None else decoded_signal_id
    value.paradigm = _CENTER_OUT_PARADIGM_ID
    value.task = _with_trial_limit(config.task, config.trials)
    resolved = config.guidance if guidance is None else guidance
    if resolved is None:
        raise RuntimeError("guidance must be resolved before building the native controller")
    value.guidance = _native_guidance_config(resolved, config.task.geometry_unit)
    value.guidance_resolver_version = guidance_resolver_version
    value.velocity_space = CommandSpace(
        id=_CENTER_OUT_COMMAND_SPACE_ID,
        dim=2,
        frame=CommandFrame.WORKSPACE_2D,
        axes=[CommandAxis(CommandAxisName.X, unit), CommandAxis(CommandAxisName.Y, unit)],
    )
    value.linear_assistance = LinearAssistance(config.assistance_blocks[0].assistance)
    blocks = []
    for declared in config.assistance_blocks:
        block = native._CenterOutAssistanceBlock()
        block.linear = LinearAssistance(declared.assistance)
        block.trials = declared.trials
        blocks.append(block)
    value.assistance_blocks = blocks
    value.training_capture_capacity = training_capture_capacity
    value.presentation_state_capacity = presentation_state_capacity
    value.initial_position = WorkspacePoint(*_initial_position(config))
    left, bottom, right, top = _center_out_cursor_bounds(config.task)
    value.cursor_min = WorkspacePoint(left, bottom)
    value.cursor_max = WorkspacePoint(right, top)
    value.trace_capacity = _trace_capacity(config.task, decoded_schema, config.trials)
    value.assistance_method = AssistanceMethod.LINEAR_BLEND
    value.abnormal = AbnormalPolicySet()
    value.max_observation_interval_ns = _stale_interval_ns(config.safety)
    return value


_CENTER_OUT_PARADIGM_ID = 1
_CENTER_OUT_COMMAND_SPACE_ID = 1
_MIN_TRACE_CAPACITY = 4096
_MIN_DRAIN_BUDGET = 256
_BRIDGE_POLL_NANOS = 200_000
_CONTROL_CLOCK_DOMAIN = 0
_AUTO_GUIDANCE_RESOLVER_VERSION = 1
_AUTO_GUIDANCE_TARGET_FRACTION = 0.6
_MAX_GUIDANCE_VERIFICATION_STEPS = 1_000_000


def _native_session_config(native: Any, schema: Any) -> Any:
    signal = _decoded_signal(schema)
    value = native._ExperimentSessionConfig()
    value.drain_budget = max(_MIN_DRAIN_BUDGET, int(signal.max_block_samples))
    value.bridge_poll_nanos = _BRIDGE_POLL_NANOS
    value.control_clock_domain = _CONTROL_CLOCK_DOMAIN
    value.trace_loss_policy = native._TraceLossPolicy.FAULT
    return value


def _resolve_guidance(
    config: CenterOutProtocol, decoded_schema: Any
) -> tuple[CenterOutGuidanceConfig, int]:
    period_ns = _control_period_ns(decoded_schema)
    initial_position = _initial_position(config)
    automatic = config.guidance is None
    guidance = (
        _automatic_guidance(config.task, initial_position, period_ns)
        if automatic
        else config.guidance
    )
    assert guidance is not None
    if any(block.assistance == 1.0 for block in config.assistance_blocks):
        _validate_guidance_reachability(config.task, guidance, initial_position, period_ns)
    return guidance, _AUTO_GUIDANCE_RESOLVER_VERSION if automatic else 0


def _control_period_ns(schema: Any) -> int:
    rate = _decoded_signal(schema).fs
    numerator = int(rate.numerator)
    denominator = int(rate.denominator)
    scaled = denominator * 1_000_000_000
    if numerator <= 0 or denominator <= 0 or scaled % numerator:
        raise ValidationError(
            "Center-Out guidance resolution requires a positive control rate "
            "with an integer-nanosecond period."
        )
    period_ns = scaled // numerator
    if period_ns <= 0:
        raise ValidationError("Center-Out control period must be positive.")
    return period_ns


def _decoded_signal(schema: Any) -> Any:
    signals = tuple(schema.signals)
    if len(signals) != 1:
        raise ValidationError("Center-Out requires a single decoded signal.")
    return signals[0]


def _initial_position(config: CenterOutProtocol) -> tuple[float, float]:
    if config.initial_position is not None:
        return config.initial_position
    center = config.task.layout.center.pos
    return float(center.x), float(center.y)


def _trace_capacity(task: Any, schema: Any, trials: int) -> int:
    signal = _decoded_signal(schema)
    return max(
        _MIN_TRACE_CAPACITY,
        _training_capture_capacity(task, schema, trials) + 2 * int(signal.max_block_samples),
    )


def _stale_interval_ns(config: CenterOutSafetyConfig) -> int:
    if config.stale_after_seconds is None:
        return 0
    value = round(config.stale_after_seconds * 1_000_000_000)
    if value <= 0:
        raise ValueError("stale_after_seconds is too small to represent in nanoseconds")
    return value


def _training_capture_capacity(task: Any, schema: Any, trials: int) -> int:
    period_ns = _control_period_ns(schema)
    center_failure = task.movement_timeout.to_center + task.punish_dwell.to_center
    outward_outcome_dwell = max(
        task.reward_dwell.to_out,
        task.punish_dwell.to_out,
    )
    outward_leg = (
        task.movement_timeout.to_center
        + task.reward_dwell.to_center
        + task.movement_timeout.to_out
        + outward_outcome_dwell
    )
    rows_per_trial = (max(center_failure, outward_leg) + period_ns - 1) // period_ns + 2
    return max(
        int(_decoded_signal(schema).max_block_samples),
        int(rows_per_trial) * trials,
    )


def _automatic_guidance(
    task: Any, initial_position: tuple[float, float], period_ns: int
) -> CenterOutGuidanceConfig:
    usable_radius = _usable_acceptance_radius(task)
    arrival_radius = usable_radius * 0.5
    period_seconds = period_ns / 1_000_000_000.0
    requirements: list[tuple[float, float]] = []
    for _, start, target, timeout_ns in _guidance_legs(task, initial_position):
        distance = math.hypot(target.pos.x - start[0], target.pos.y - start[1])
        available_seconds = (timeout_ns - task.hold_ns) / 1_000_000_000.0
        if available_seconds <= 0.0:
            raise ValidationError(
                "cannot derive automatic guidance because hold_seconds must be "
                "strictly shorter than each movement timeout."
            )
        requirements.append((distance, available_seconds))

    max_speed = max(
        2.0 * distance / (available * _AUTO_GUIDANCE_TARGET_FRACTION)
        for distance, available in requirements
    )
    max_speed = min(max_speed, usable_radius / period_seconds)
    acceleration = max(
        4.0 * distance / (available * _AUTO_GUIDANCE_TARGET_FRACTION) ** 2
        for distance, available in requirements
    )
    shortest_target = min(
        available * _AUTO_GUIDANCE_TARGET_FRACTION for _, available in requirements
    )
    acceleration = max(
        acceleration,
        max_speed / max(period_seconds, shortest_target * 0.25),
    )
    try:
        return CenterOutGuidanceConfig(
            max_speed=max_speed,
            acceleration=acceleration,
            deceleration=acceleration,
            arrival_radius=arrival_radius,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"could not derive finite automatic guidance: {exc}") from exc


def _validate_guidance_reachability(
    task: Any,
    guidance: CenterOutGuidanceConfig,
    initial_position: tuple[float, float],
    period_ns: int,
) -> None:
    for label, start, target, timeout_ns in _guidance_legs(task, initial_position):
        reached_ns = _guidance_acquisition_time_ns(
            task, guidance, start, target, period_ns, timeout_ns
        )
        if reached_ns is None:
            rate_hz = 1_000_000_000.0 / period_ns
            distance = math.hypot(target.pos.x - start[0], target.pos.y - start[1])
            raise ValidationError(
                f"guidance cannot complete the 100%-assisted {label} before its "
                f"movement timeout: target_id={target.id}, distance={distance:g}, "
                f"timeout={timeout_ns / 1e9:g}s, control_rate={rate_hz:g}Hz, "
                f"max_speed={guidance.max_speed:g}, "
                f"acceleration={guidance.acceleration:g}."
            )


def _guidance_legs(
    task: Any, initial_position: tuple[float, float]
) -> tuple[tuple[str, tuple[float, float], Any, int], ...]:
    center = task.layout.center
    center_position = (float(center.pos.x), float(center.pos.y))
    legs = [
        (
            "initial-to-center leg",
            initial_position,
            center,
            int(task.movement_timeout.to_center),
        )
    ]
    for target in task.layout.surrounding:
        target_position = (float(target.pos.x), float(target.pos.y))
        legs.append(
            (
                f"target-{target.id}-to-center leg",
                target_position,
                center,
                int(task.movement_timeout.to_center),
            )
        )
        legs.append(
            (
                f"center-to-target-{target.id} leg",
                center_position,
                target,
                int(task.movement_timeout.to_out),
            )
        )
    return tuple(legs)


def _guidance_acquisition_time_ns(
    task: Any,
    guidance: CenterOutGuidanceConfig,
    start: tuple[float, float],
    target: Any,
    period_ns: int,
    timeout_ns: int,
) -> int | None:
    from neurale.experiments import ContractStatus
    from neurale.experiments.center_out import CenterOutGuidance, WorkspacePoint

    position = WorkspacePoint(*start)
    holding_since = 0 if _cursor_contained(task, target, position) else None
    if holding_since is not None and task.hold_ns == 0:
        return 0

    steps = (timeout_ns + period_ns - 1) // period_ns
    if steps > _MAX_GUIDANCE_VERIFICATION_STEPS:
        raise ValidationError(
            "Center-Out guidance timing verification exceeds its bounded "
            f"{_MAX_GUIDANCE_VERIFICATION_STEPS}-step preparation limit."
        )
    generator = CenterOutGuidance()
    if generator.configure(guidance) != ContractStatus.OK:
        raise ValidationError("Center-Out guidance configuration is invalid.")
    dt_seconds = period_ns / 1_000_000_000.0
    for step in range(1, steps + 1):
        time_ns = step * period_ns
        status, sample = generator.update(target, position, period_ns)
        if status != ContractStatus.OK:
            raise ValidationError(f"Center-Out guidance simulation failed: {status}")
        position = WorkspacePoint(
            position.x + sample.vel.x * dt_seconds,
            position.y + sample.vel.y * dt_seconds,
        )
        if (
            holding_since is not None
            and holding_since + task.hold_ns <= time_ns
            and holding_since + task.hold_ns < timeout_ns
        ):
            return holding_since + task.hold_ns
        if time_ns >= timeout_ns:
            return None
        if _cursor_contained(task, target, position):
            if holding_since is None:
                holding_since = time_ns
                if task.hold_ns == 0:
                    return time_ns
        else:
            holding_since = None
    return None


def _cursor_contained(task: Any, target: Any, position: Any) -> bool:
    extent = task.cursor.extent
    acceptance = task.acceptance
    return (
        position.x - extent >= target.pos.x - acceptance.half_extent_x
        and position.x + extent <= target.pos.x + acceptance.half_extent_x
        and position.y - extent >= target.pos.y - acceptance.half_extent_y
        and position.y + extent <= target.pos.y + acceptance.half_extent_y
    )


def _usable_acceptance_radius(task: Any) -> float:
    return min(
        task.acceptance.half_extent_x - task.cursor.extent,
        task.acceptance.half_extent_y - task.cursor.extent,
    )


__all__ = [
    "CenterOutProtocol",
    "CenterOutSafetyConfig",
]
