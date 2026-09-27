#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Center-Out 2D configuration, geometry, target schedule, machine, and guidance.

This submodule owns the immutable task configuration, target geometry,
containment predicate, outward-target schedule, deterministic state machine,
derived statistics, and optional target-directed guidance. It reads no clock,
starts no timer or thread, touches no device, writes no recording, and draws
nothing.

The machine consumes explicit integer-nanosecond time and observed cursor
positions. Identical configuration and input sequences produce identical
transitions, events, and trials. Deadlines are instants rather than polling
counters, and repeating a settled step at the same instant produces nothing.

``RadialLayoutRequest(radius=...)`` describes the classic eight-direction
layout by default, with centre identifier 1 and outward identifiers 2 through
9. Supplying both ``ids`` and ``spokes`` selects the advanced layout path.
``build_radial_layout`` returns the immutable layout directly and raises
``ValueError`` with the rejected field when the request is invalid.

``CenterOutTask`` accepts keyword-only, validated experiment parameters.
Durations are expressed in seconds; a scalar applies to both trial legs and a
two-item sequence provides the to-centre and to-out values. Session length and
the native sampler implementation version are intentionally not constructor
parameters.

Radial layouts use ``2*pi / max(count, MIN_RADIAL_SPOKES)``. Cursor containment
is an inclusive axis-aligned whole-extent test rather than an overlap or
centre-distance test. Selection policy states explicitly whether a failed
trial repeats the same outward target. Geometry uses ``double`` precision,
receives unclamped semantic coordinates, and never mutates caller-owned values.

The centre target is separate from the outward-target array, so it cannot be
selected for an outward trial. The machine never moves the cursor; every new
leg begins from the next position supplied by the caller.

:class:`CenterOutGuidance` and :func:`evaluate_guidance` generate a
target-directed reference velocity. Guidance is neither assistance nor task
state: orchestration may combine its reference with a decoded velocity through
:mod:`neurale.experiments.assistance`, and the state machine operates without
guidance. Each guidance step receives an explicit elapsed duration and retains
only its previous workspace velocity and target.

Configuration carries no device, screen, decoder, recorder, or streaming
settings. Importing this module is lightweight: the native extension loads on
first use of a name, not on import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_NATIVE_EXPORTS = {
    "AcceptanceRegion",
    "CenterOutTask",
    "CenterOut2DLayout",
    "CenterOutCause",
    "CenterOutGuidancePhase",
    "CenterOutGuidanceSample",
    "CenterOutGuidanceState",
    "CenterOutMachine",
    "CenterOutPhase",
    "CenterOutReason",
    "CenterOutSnapshot",
    "CenterOutState",
    "CenterOutStatistics",
    "CenterOutStepResult",
    "CenterOutTrial",
    "CursorGeometry",
    "GeometryUnit",
    "MAX_STEP_EVENTS",
    "MAX_STEP_TRANSITIONS",
    "MAX_SURROUNDING_TARGETS",
    "MIN_RADIAL_SPOKES",
    "PhaseDurations",
    "RadialLayoutRequest",
    "TARGET_SELECTION_STREAM",
    "TargetPlacement",
    "TargetSelectionPolicy",
    "WorkspacePoint",
    "WorkspaceVelocity",
    "build_radial_layout",
    "center_out_guidance_phase_declared",
    "center_out_phase_declared",
    "center_out_phase_of",
    "center_out_state_declared",
    "center_out_state_is_dwell",
    "center_out_state_is_leg",
    "configuration_fingerprint",
    "contains_cursor",
    "contains_point",
    "geometry_unit_declared",
    "layout_fingerprint",
    "phase_duration",
    "radial_position",
    "radial_spoke_step",
    "select_outward_target",
    "summarize",
    "target_selection_policy_declared",
    "trial_limit_reached",
    "validate",
}

_PUBLIC_EXPORTS = {
    "AssistanceBlock",
    "CenterOutOutcome",
    "CenterOutProtocol",
    "CenterOutSafetyConfig",
    "CenterOutSession",
    "DecoderPublication",
    "CenterOutGuidance",
    "CenterOutGuidanceConfig",
    "CenterOutIntentSnapshot",
    "OnlineDecoderTrainingConfig",
    "OnlineDecoderUpdatePolicy",
    "SupervisionTarget",
    "evaluate_guidance",
    "validate_against",
}

_LAZY_EXPORTS = {
    "AssistanceBlock": (".center_out_training", "AssistanceBlock"),
    "CenterOutOutcome": (".closed_loop", "CenterOutOutcome"),
    "CenterOutProtocol": (".center_out_session", "CenterOutProtocol"),
    "CenterOutSafetyConfig": (".center_out_session", "CenterOutSafetyConfig"),
    "CenterOutSession": (".closed_loop", "CenterOutSession"),
    "CenterOutIntentSnapshot": (".closed_loop", "CenterOutIntentSnapshot"),
    "DecoderPublication": (".closed_loop", "DecoderPublication"),
    "OnlineDecoderTrainingConfig": (
        ".center_out_training",
        "OnlineDecoderTrainingConfig",
    ),
    "OnlineDecoderUpdatePolicy": (
        ".center_out_training",
        "OnlineDecoderUpdatePolicy",
    ),
    "SupervisionTarget": (".center_out_training", "SupervisionTarget"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class CenterOutGuidanceConfig:
    """Target-directed reference-motion settings.

    Values use the geometry unit declared by the paired
    :class:`CenterOutTask`. ``deceleration`` defaults to
    ``acceleration`` for the common symmetric profile.
    """

    max_speed: float
    acceleration: float
    arrival_radius: float
    deceleration: float | None = None

    def __post_init__(self) -> None:
        max_speed = _positive_float(self.max_speed, "max_speed")
        acceleration = _positive_float(self.acceleration, "acceleration")
        arrival_radius = _positive_float(self.arrival_radius, "arrival_radius")
        deceleration = (
            acceleration
            if self.deceleration is None
            else _positive_float(self.deceleration, "deceleration")
        )
        object.__setattr__(self, "max_speed", max_speed)
        object.__setattr__(self, "acceleration", acceleration)
        object.__setattr__(self, "arrival_radius", arrival_radius)
        object.__setattr__(self, "deceleration", deceleration)


class CenterOutGuidance:
    """Stateful guidance generator backed by the native implementation."""

    __slots__ = ("_config", "_native")

    def __init__(self) -> None:
        self._native = _native_namespace()._NativeCenterOutGuidance()
        self._config: CenterOutGuidanceConfig | None = None

    @property
    def configured(self) -> bool:
        return bool(self._native.configured)

    def configure(self, config: CenterOutGuidanceConfig) -> object:
        _require_guidance_config(config)
        status = self._native.configure(_native_guidance_config(config))
        from neurale.experiments import ContractStatus

        if status == ContractStatus.OK:
            self._config = config
        return status

    def reset(self) -> None:
        self._native.reset()

    def update(self, target: object, pos: object, dt_ns: int) -> object:
        return self._native.update(target, pos, dt_ns)

    def configuration(self) -> CenterOutGuidanceConfig:
        if self._config is None:
            raise RuntimeError("CenterOutGuidance is not configured")
        return self._config

    def state(self) -> object:
        return self._native.state()


def evaluate_guidance(
    config: object,
    previous: object,
    target: object,
    pos: object,
    dt_ns: int,
) -> object:
    """Evaluate one native guidance step without retaining state."""

    if isinstance(config, CenterOutGuidanceConfig):
        config = _native_guidance_config(config)
    return _native_namespace().evaluate_guidance(config, previous, target, pos, dt_ns)


def validate_against(guidance: object, config: object) -> object:
    """Validate guidance against a Center-Out task's usable target radius."""

    native = _native_namespace()
    if isinstance(guidance, CenterOutGuidanceConfig):
        guidance = _native_guidance_config(guidance, config.geometry_unit)
    return native.validate_against(guidance, config)


def validate(value: object) -> object:
    """Validate native Center-Out values; public guidance is valid on construction."""

    native = _native_namespace()
    if isinstance(value, CenterOutGuidanceConfig):
        value = _native_guidance_config(value)
    return native.validate(value)


def _native_namespace() -> object:
    from neurale._native_loader import load_native_namespace

    return load_native_namespace("experiments.center_out")


def _require_guidance_config(config: object) -> None:
    if not isinstance(config, CenterOutGuidanceConfig):
        raise TypeError("config must be a CenterOutGuidanceConfig")


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _native_guidance_config(
    config: CenterOutGuidanceConfig, geometry_unit: object | None = None
) -> object:
    """Resolve a public config into the native execution record."""

    _require_guidance_config(config)
    native = _native_namespace()
    unit = native.GeometryUnit.DIMENSIONLESS if geometry_unit is None else geometry_unit
    return native._NativeCenterOutGuidanceConfig(
        geometry_unit=unit,
        max_speed=config.max_speed,
        acceleration=config.acceleration,
        deceleration=config.deceleration,
        precision=config.arrival_radius,
    )


def _raw_guidance_config(**fields: object) -> object:
    """Build the native record for replay and low-level contract tests."""

    native = _native_namespace()
    values = {
        "geometry_unit": native.GeometryUnit.UNSPECIFIED,
        "max_speed": 0.0,
        "acceleration": 0.0,
        "deceleration": 0.0,
        "precision": 0.0,
    }
    values.update(fields)
    return native._NativeCenterOutGuidanceConfig(**values)


def _raw_config(**fields: object) -> object:
    """Build a native contract record for replay and low-level contract tests."""

    from neurale._native_loader import load_native_namespace

    return load_native_namespace("experiments.center_out")._raw_center_out_config(**fields)


def _with_trial_limit(config: object, trials: int) -> object:
    """Attach the session-owned trial count to the native task record."""

    from neurale._native_loader import load_native_namespace

    return load_native_namespace("experiments.center_out")._with_trial_limit(config, trials)


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name, __package__), attribute)
        globals()[name] = value
        return value
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from neurale._native_loader import load_native_namespace

    value = getattr(load_native_namespace("experiments.center_out"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS | _PUBLIC_EXPORTS)


__all__ = sorted(_NATIVE_EXPORTS | _PUBLIC_EXPORTS)
