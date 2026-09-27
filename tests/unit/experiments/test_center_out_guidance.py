#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Center-Out target-directed guidance and reference velocity checks."""

from __future__ import annotations

import math

import pytest

from neurale.experiments import (
    CommandAxis,
    CommandAxisName,
    CommandFrame,
    CommandSpace,
    CommandUnit,
    ContractStatus,
    TrialOutcome,
)
from neurale.experiments import center_out as co
from neurale.experiments.assistance import (
    LinearAssistance,
    VelocityVector,
    blend_velocity,
)

# ---------------------------------------------------------------------------
# Constants

STEP_NS = 15_625_000  # one sixty-fourth of a second, exactly
STEP_S = 0.015625

MAX_SPEED = 100.0
ACCELERATION = 200.0  # 3.125 per step
DECELERATION = 400.0  # 6.25 per step
PRECISION = 2.0

TARGET = 5
OTHER_TARGET = 6

SPEEDING_UP = co.CenterOutGuidancePhase.SPEEDING_UP
SLOWING_DOWN = co.CenterOutGuidancePhase.SLOWING_DOWN
AT_TARGET = co.CenterOutGuidancePhase.AT_TARGET
IDLE = co.CenterOutGuidancePhase.IDLE


# ---------------------------------------------------------------------------
# Independent reference


def _profile(
    speed: float,
    distance: float,
    dt_s: float,
    *,
    max_speed: float = MAX_SPEED,
    acceleration: float = ACCELERATION,
    deceleration: float = DECELERATION,
    precision: float = PRECISION,
) -> tuple[float, object]:
    """Model a 1D approach along ``+x``: the retained
    velocity is ``(speed, 0)`` and the target is ``(distance, 0)``, so the speed
    towards the target is just ``speed``. Returns the speed after the step and
    the phase it was decided in.
    """
    if distance <= precision:
        return 0.0, AT_TARGET

    s = speed  # dot((speed, 0), (1, 0))
    if s <= 0:
        # Speeding up from rest: nothing to project, accelerate along the target.
        new = acceleration * dt_s
        return (max_speed if new > max_speed else new), SPEEDING_UP

    # The braking test compares two times: time to reach the target at the
    # current speed against time to brake to rest. Braking begins at
    # distance <= s * s / deceleration, measured to the target itself.
    if distance / s <= s / deceleration:
        braked = s - deceleration * dt_s
        return (braked if braked > 0.0 else 0.0), SLOWING_DOWN

    new = s + acceleration * dt_s
    return (max_speed if new > max_speed else new), SPEEDING_UP


# ---------------------------------------------------------------------------
# Builders


def _config(
    *,
    max_speed: float = MAX_SPEED,
    acceleration: float = ACCELERATION,
    deceleration: float = DECELERATION,
    arrival_radius: float = PRECISION,
) -> object:
    return co.CenterOutGuidanceConfig(
        max_speed=max_speed,
        acceleration=acceleration,
        deceleration=deceleration,
        arrival_radius=arrival_radius,
    )


def _raw_guidance(
    *,
    unit: object = None,
    max_speed: float = MAX_SPEED,
    acceleration: float = ACCELERATION,
    deceleration: float = DECELERATION,
    precision: float = PRECISION,
) -> object:
    return co._raw_guidance_config(
        geometry_unit=co.GeometryUnit.MILLIMETRES if unit is None else unit,
        max_speed=max_speed,
        acceleration=acceleration,
        deceleration=deceleration,
        precision=precision,
    )


def _target(x: float = 100.0, y: float = 0.0, identifier: int = TARGET) -> object:
    return co.TargetPlacement(id=identifier, pos=co.WorkspacePoint(x, y))


def _state(vx: float, vy: float, phase: object, target: int = TARGET) -> object:
    return co.CenterOutGuidanceState(target=target, vel=co.WorkspaceVelocity(vx, vy), phase=phase)


def _rest() -> object:
    return co.CenterOutGuidanceState()


def _evaluate(
    config: object,
    previous: object,
    target: object,
    pos: tuple[float, float],
    dt_ns: int = STEP_NS,
) -> object:
    status, sample = co.evaluate_guidance(config, previous, target, co.WorkspacePoint(*pos), dt_ns)
    assert status == ContractStatus.OK
    return sample


def _session(
    *,
    half_x: float = 5.0,
    half_y: float = 4.0,
    cursor: float = 1.0,
    unit: object = None,
) -> object:
    request = co.RadialLayoutRequest(
        radius=100.0,
        center_id=1,
        ids=[idx + 2 for idx in range(8)],
        spokes=list(range(8)),
    )
    layout = co.build_radial_layout(request)
    return co.CenterOutTask(
        geometry_unit=co.GeometryUnit.MILLIMETRES if unit is None else unit,
        layout=layout,
        acceptance=co.AcceptanceRegion(half_x, half_y),
        cursor_extent=cursor,
        movement_timeout_seconds=1.0,
        selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
        seed=0xBEEF,
    )


# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_required_parameters_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            co.CenterOutGuidanceConfig()
        with pytest.raises(TypeError):
            co.CenterOutGuidanceConfig(MAX_SPEED, ACCELERATION, PRECISION)

    def test_every_rate_must_be_finite_and_positive(self) -> None:
        for field in ("max_speed", "acceleration", "deceleration"):
            for value in (0.0, -1.0, math.inf, math.nan):
                with pytest.raises(ValueError, match=field):
                    _config(**{field: value})

    def test_arrival_radius_must_be_finite_and_positive(self) -> None:
        for value in (0.0, -0.5, math.inf, math.nan):
            with pytest.raises(ValueError, match="arrival_radius"):
                _config(arrival_radius=value)

    def test_deceleration_defaults_to_acceleration(self) -> None:
        config = co.CenterOutGuidanceConfig(
            max_speed=MAX_SPEED,
            acceleration=ACCELERATION,
            arrival_radius=PRECISION,
        )
        assert config.deceleration == ACCELERATION
        assert co.validate(config) == ContractStatus.OK

    def test_configuration_carries_no_assistance_or_rendering(self) -> None:
        # This is guidance, not assistance: a coefficient here would be the
        # blend leaking back into the paradigm that rule 9 separates it from.
        # And nothing in it may be derived from something that is drawn.
        fields = {name for name in dir(_config()) if not name.startswith("_")}
        assert fields == {
            "max_speed",
            "acceleration",
            "deceleration",
            "arrival_radius",
        }

    def test_guidance_name_never_mentions_assistance_or_device(self) -> None:
        for name in co.__all__:
            if "guidance" not in name.lower() and "Guidance" not in name:
                continue
            lowered = name.lower()
            for forbidden in (
                "assist",
                "impedance",
                "blend",
                "decoder",
                "actuator",
                "device",
                "render",
                "recorder",
            ):
                assert forbidden not in lowered, name


class TestState:
    def test_default_state_is_valid(self) -> None:
        assert co.validate(_rest()) == ContractStatus.OK

    def test_retained_velocity_must_be_finite(self) -> None:
        # The retained velocity is a 2D vector. Its components may
        # be negative -- a heading -- but a non-finite one is a corrupt record.
        assert co.validate(_state(math.inf, 0.0, SPEEDING_UP)) == ContractStatus.VALUE_NOT_FINITE
        assert co.validate(_state(0.0, math.nan, SPEEDING_UP)) == ContractStatus.VALUE_NOT_FINITE
        # A negative component is a legal heading, not a corrupt value.
        assert co.validate(_state(10.0, -3.0, SPEEDING_UP)) == ContractStatus.OK

    @pytest.mark.parametrize(
        "target,vx,vy,phase",
        [
            (0, 1.0, 0.0, IDLE),
            (0, 0.0, 0.0, SPEEDING_UP),
            (TARGET, 0.0, 0.0, IDLE),
        ],
    )
    def test_individually_legal_fields_can_conflict(
        self, target: int, vx: float, vy: float, phase: object
    ) -> None:
        state = co.CenterOutGuidanceState(
            target=target, vel=co.WorkspaceVelocity(vx, vy), phase=phase
        )
        assert co.validate(state) == ContractStatus.OUTCOME_INVALID

    def test_declared_phases_are_complete(self) -> None:
        for phase in (IDLE, SPEEDING_UP, SLOWING_DOWN, AT_TARGET):
            assert co.center_out_guidance_phase_declared(phase)


class TestReferenceVelocity:
    @pytest.mark.parametrize("distance", [0.5, 2.0, 2.5, 5.0, 14.5, 15.0, 40.0, 100.0])
    @pytest.mark.parametrize("speed", [0.0, 3.125, 25.0, 62.5, 100.0])
    @pytest.mark.parametrize("dt_ns", [0, STEP_NS, 4 * STEP_NS, 62_500_000])
    def test_policy_matches_reference_profile(
        self, distance: float, speed: float, dt_ns: int
    ) -> None:
        previous = _rest() if speed == 0.0 else _state(speed, 0.0, SPEEDING_UP)
        target = _target(distance, 0.0)
        sample = _evaluate(_config(), previous, target, (0.0, 0.0), dt_ns)

        expected_speed, expected_phase = _profile(speed, distance, dt_ns / 1e9)
        assert sample.state.vel.x == expected_speed
        assert sample.state.phase == expected_phase
        assert sample.distance == distance
        assert sample.vel.x == expected_speed
        assert sample.vel.y == 0.0

    @pytest.mark.parametrize(
        "offset,unit",
        [
            ((3.0, 4.0), (0.6, 0.8)),
            ((-3.0, 4.0), (-0.6, 0.8)),
            ((0.0, -5.0), (0.0, -1.0)),
            ((-5.0, 0.0), (-1.0, 0.0)),
        ],
    )
    def test_direction_is_observed_offset(
        self, offset: tuple[float, float], unit: tuple[float, float]
    ) -> None:
        # The direction comes from the observed offset, so the velocity points
        # along it whatever the entering velocity's component along it was --
        # including when that component is negative and the approach restarts
        # from rest.
        sample = _evaluate(
            _raw_guidance(precision=0.0),
            _state(3.125, 0.0, SPEEDING_UP),
            _target(*offset),
            (0.0, 0.0),
        )
        assert sample.distance == pytest.approx(5.0)
        speed = math.hypot(sample.vel.x, sample.vel.y)
        assert sample.vel.x == pytest.approx(speed * unit[0], abs=1e-15)
        assert sample.vel.y == pytest.approx(speed * unit[1], abs=1e-15)

    def test_velocity_never_points_away_from_target(self) -> None:
        # Including from behind it: a profile that overshot is turned round by
        # the next observation, not by a remembered direction.
        for pos in (-50.0, 0.0, 50.0, 150.0, 400.0):
            sample = _evaluate(_config(), _state(50.0, 0.0, SPEEDING_UP), _target(), (pos, 0.0))
            towards = 100.0 - pos
            if sample.state.phase != AT_TARGET:
                assert sample.vel.x * towards > 0.0

    def test_zero_step_leaves_along_direction_velocity(self) -> None:
        # A zero step recomputes the direction and re-projects the retained
        # velocity onto it. The component along the direction is kept; the
        # perpendicular part is discarded. Here the retained velocity is already
        # along the direction, so the speed is unchanged.
        previous = _state(0.0, 50.0, SPEEDING_UP)
        sample = _evaluate(_config(), previous, _target(0.0, 100.0), (0.0, 0.0), 0)
        assert sample.state.vel.x == 0.0
        assert sample.state.vel.y == 50.0
        assert sample.state.phase == SPEEDING_UP


class TestRateBoundaries:
    def test_acceleration_reaches_maximum_in_expected_steps(self) -> None:
        state = _rest()
        for step in range(1, 41):
            sample = _evaluate(_config(), state, _target(), (0.0, 0.0))
            state = sample.state
            if step < 32:
                assert state.vel.x == ACCELERATION * STEP_S * step
                assert state.phase == SPEEDING_UP
            else:
                assert state.vel.x == MAX_SPEED
                # There is no separate cruising phase: a profile pinned at the
                # clamp is still speeding up.
                assert state.phase == SPEEDING_UP

    def test_braking_comparison_is_closed(self) -> None:
        # The braking test is `distance / speed <= speed / deceleration`, so at
        # the maximum it switches at distance 100 * 100 / 400 = 25 -- twice what
        # a constant-deceleration stop needs, and measured to the target itself
        # rather than to the edge of the precision region. Exactly there, the
        # profile brakes; the comparison is closed on the braking side.
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        at_switch = _evaluate(_config(), fast, _target(), (75.0, 0.0))
        assert at_switch.distance == 25.0
        assert at_switch.speed_towards_target == MAX_SPEED
        assert at_switch.state.phase == SLOWING_DOWN
        assert at_switch.state.vel.x == MAX_SPEED - DECELERATION * STEP_S

        # A quarter of a millimetre further out it is still speeding up, and the
        # clamp holds it at the maximum.
        outside = _evaluate(_config(), fast, _target(), (74.75, 0.0))
        assert outside.distance == 25.25
        assert outside.state.phase == SPEEDING_UP
        assert outside.state.vel.x == MAX_SPEED

    def test_reported_speed_is_entering_projection(self) -> None:
        # It is the entering velocity's component along the direction, not the
        # updated one: the whole phase decision is made from it, so reporting it
        # is what makes that decision checkable from outside.
        for speed in (0.0, 6.25, 50.0, MAX_SPEED):
            previous = _rest() if speed == 0.0 else _state(speed, 0.0, SPEEDING_UP)
            sample = _evaluate(_config(), previous, _target(), (0.0, 0.0))
            assert sample.speed_towards_target == speed

    def test_braking_never_drives_speed_below_zero(self) -> None:
        # A step of braking takes 6.25 off a speed of 3.125. A standstill outside
        # the precision region is still braking, not arrival. Reaching that at
        # all needs a precision radius smaller than the switch distance the speed
        # implies, so this uses a zero radius.
        exact = _raw_guidance(precision=0.0)
        crawling = _state(ACCELERATION * STEP_S, 0.0, SLOWING_DOWN)
        sample = _evaluate(exact, crawling, _target(), (99.98, 0.0))
        assert sample.distance == pytest.approx(0.02, abs=1e-12)
        assert sample.state.phase == SLOWING_DOWN
        assert sample.state.vel.x == 0.0
        assert sample.vel.y == 0.0
        # With the ordinary radius the same geometry is simply arrival.
        assert _evaluate(_config(), crawling, _target(), (99.98, 0.0)).state.phase == AT_TARGET

    def test_no_maximum_speed_clamp_is_applied_while_braking(self) -> None:
        # A braking run above the maximum is already on its way down, and the
        # clamp applies only while speeding up. One step of braking is all that
        # happens to it.
        fast = _state(1000.0, 0.0, SLOWING_DOWN)
        sample = _evaluate(_config(), fast, _target(), (0.0, 0.0))
        assert sample.state.phase == SLOWING_DOWN
        assert sample.state.vel.x == 1000.0 - DECELERATION * STEP_S

    def test_two_rates_are_independent(self) -> None:
        gentle = _config(deceleration=50.0)
        # A gentler brake has a longer switch distance, so it brakes here where
        # the standard one would still be speeding up.
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        sample = _evaluate(gentle, fast, _target(), (0.0, 0.0))
        assert sample.state.phase == SLOWING_DOWN
        assert sample.state.vel.x == MAX_SPEED - 50.0 * STEP_S
        # The acceleration is untouched by that change.
        far = _evaluate(gentle, _rest(), _target(1000.0, 0.0), (0.0, 0.0))
        assert far.state.vel.x == ACCELERATION * STEP_S
        assert far.state.phase == SPEEDING_UP


class TestPrecisionRegion:
    def test_region_is_closed_and_inner_velocity_is_zero(self) -> None:
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        for pos in (98.0, 99.0, 100.0, 101.0, 102.0):
            sample = _evaluate(_config(), fast, _target(), (pos, 0.0))
            assert sample.state.phase == AT_TARGET
            assert sample.vel.x == 0.0
            assert sample.vel.y == 0.0
            # Dropped, not decayed: a reference mover that arrived has stopped.
            assert sample.state.vel.x == 0.0
            assert sample.speed_towards_target == 0.0

        outside = _evaluate(_config(), fast, _target(), (97.5, 0.0))
        assert outside.state.phase != AT_TARGET
        assert outside.state.vel.x > 0.0

    def test_cursor_drifting_out_accelerates_from_rest(self) -> None:
        arrived = _evaluate(_config(), _state(MAX_SPEED, 0.0, SPEEDING_UP), _target(), (99.0, 0.0))
        resumed = _evaluate(_config(), arrived.state, _target(), (0.0, 0.0))
        assert resumed.state.phase == SPEEDING_UP
        assert resumed.state.vel.x == ACCELERATION * STEP_S

    def test_zero_radius_makes_only_target_arrival(self) -> None:
        exact = _raw_guidance(precision=0.0)
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        on_it = _evaluate(exact, fast, _target(), (100.0, 0.0))
        assert on_it.state.phase == AT_TARGET
        beside_it = _evaluate(exact, fast, _target(), (99.9999, 0.0))
        assert beside_it.state.phase == SLOWING_DOWN


class TestTargetChangeAndReset:
    def test_different_target_discards_retained_velocity(self) -> None:
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        sample = _evaluate(_config(), fast, _target(0.0, 100.0, OTHER_TARGET), (0.0, 0.0))
        assert sample.retargeted
        assert sample.speed_towards_target == 0.0
        assert sample.state.target == OTHER_TARGET
        assert sample.state.vel.y == ACCELERATION * STEP_S
        assert sample.state.vel.x == 0.0

    def test_moving_same_target_is_not_target_change(self) -> None:
        # Moved along the direction the retained velocity already points, so the
        # projection keeps it: this is the case that distinguishes a retarget
        # (which discards) from a moved target (which does not).
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        moved = _target(200.0, 0.0)
        sample = _evaluate(_config(), fast, moved, (0.0, 0.0))
        assert not sample.retargeted
        assert sample.speed_towards_target == MAX_SPEED
        assert sample.state.vel.x == MAX_SPEED

    def test_first_update_after_reset_is_not_target_change(self) -> None:
        fresh = _evaluate(_config(), _rest(), _target(identifier=OTHER_TARGET), (0.0, 0.0))
        assert not fresh.retargeted

        changed = _evaluate(
            _config(),
            _state(MAX_SPEED, 0.0, SPEEDING_UP),
            _target(identifier=OTHER_TARGET),
            (0.0, 0.0),
        )
        # A target change produces exactly what an explicit reset would.
        assert changed.state.vel.x == fresh.state.vel.x
        assert changed.state.vel.y == fresh.state.vel.y
        assert changed.state.phase == fresh.state.phase
        assert changed.vel.x == fresh.vel.x
        assert changed.vel.y == fresh.vel.y

    def test_generator_refuses_to_run_unconfigured(self) -> None:
        generator = co.CenterOutGuidance()
        assert not generator.configured
        status, _ = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
        assert status == ContractStatus.NOT_RUNNING

    def test_reset_keeps_configuration_drops_run(self) -> None:
        generator = co.CenterOutGuidance()
        assert generator.configure(_config()) == ContractStatus.OK
        for _ in range(4):
            status, _ = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
            assert status == ContractStatus.OK
        assert generator.state().vel.x == ACCELERATION * STEP_S * 4.0

        generator.reset()
        assert generator.state().target == 0
        assert generator.state().vel.x == 0.0
        assert generator.state().vel.y == 0.0
        assert generator.state().phase == IDLE
        assert generator.configuration().max_speed == MAX_SPEED

    def test_refused_update_leaves_run_unchanged(self) -> None:
        generator = co.CenterOutGuidance()
        assert generator.configure(_config()) == ContractStatus.OK
        status, _ = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
        assert status == ContractStatus.OK
        before = generator.state()

        status, _ = generator.update(
            co.TargetPlacement(id=0, pos=co.WorkspacePoint(1.0, 0.0)),
            co.WorkspacePoint(0.0, 0.0),
            STEP_NS,
        )
        assert status == ContractStatus.IDENTITY_MISSING
        assert generator.state().target == before.target
        assert generator.state().vel.x == before.vel.x
        assert generator.state().phase == before.phase

    def test_reported_state_is_copy(self) -> None:
        generator = co.CenterOutGuidance()
        assert generator.configure(_config()) == ContractStatus.OK
        status, _ = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
        assert status == ContractStatus.OK
        held = generator.state()
        status, _ = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
        assert status == ContractStatus.OK
        # A view onto the generator's own field would have moved with it.
        assert held.vel.x == ACCELERATION * STEP_S
        assert generator.state().vel.x == ACCELERATION * STEP_S * 2


class TestCadenceAndDeterminism:
    def test_regime_is_linear_in_elapsed_time(self) -> None:
        dense = _rest()
        for _ in range(4):
            dense = _evaluate(_config(), dense, _target(), (0.0, 0.0)).state
        sparse = _evaluate(_config(), _rest(), _target(), (0.0, 0.0), 4 * STEP_NS)
        assert sparse.state.vel.x == dense.vel.x

    def test_same_observation_sequence_reproduces(self) -> None:
        script = [
            (0.0, 0.0),
            (10.0, 0.0),
            (40.0, 0.0),
            (70.0, 0.0),
            (85.5, 0.0),
            (92.0, 0.0),
            (97.0, 0.0),
            (98.5, 0.0),
            (60.0, 0.0),
            (20.0, 0.0),
        ]

        def run() -> list[tuple[float, float, object, bool]]:
            generator = co.CenterOutGuidance()
            assert generator.configure(_config()) == ContractStatus.OK
            out = []
            for pos in script:
                status, sample = generator.update(_target(), co.WorkspacePoint(*pos), STEP_NS)
                assert status == ContractStatus.OK
                out.append(
                    (
                        sample.state.vel.x,
                        sample.state.vel.y,
                        sample.state.phase,
                        sample.retargeted,
                    )
                )
            return out

        first = run()
        second = run()
        assert first == second

    def test_invocations_are_not_counted(self) -> None:
        # A hundred unrelated evaluations between two updates change neither.
        generator = co.CenterOutGuidance()
        assert generator.configure(_config()) == ContractStatus.OK
        status, first = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
        assert status == ContractStatus.OK
        for noise in range(100):
            _evaluate(_config(), _rest(), _target(0.0, 50.0), (float(noise), 3.0))
        status, second = generator.update(_target(), co.WorkspacePoint(0.0, 0.0), STEP_NS)
        assert status == ContractStatus.OK
        assert second.state.vel.x == first.state.vel.x + ACCELERATION * STEP_S

    def test_elapsed_time_partition_is_not_promised(self) -> None:
        # Stated rather than hidden. The profile integrates its step, and the
        # accelerate-or-brake decision is remade from the speed each step
        # entered with, so a partition that lets the regime switch part way is
        # not the same run as one that does not.
        #
        # Sitting exactly at the braking point from the maximum
        # (distance = speed**2 / deceleration = 25): four steps integrate the
        # cursor, and one braking step is enough to put the time comparison back
        # on the speeding-up side, so the profile chatters -- brake, accelerate,
        # brake, accelerate. One step of four times the duration brakes the
        # whole way, because the decision is made once.
        fast = _state(MAX_SPEED, 0.0, SPEEDING_UP)
        target = _target()
        dense_state = fast
        dense_pos = 75.0
        for _ in range(4):
            sample = _evaluate(_config(), dense_state, target, (dense_pos, 0.0))
            dense_state = sample.state
            dense_pos += sample.vel.x * STEP_S
        sparse = _evaluate(_config(), fast, target, (75.0, 0.0), 4 * STEP_NS)

        assert dense_state.vel.x == 93.75
        assert sparse.state.vel.x == MAX_SPEED - DECELERATION * 4 * STEP_S
        assert dense_state.vel.x != sparse.state.vel.x


class TestRejectionsAndInputIntegrity:
    def test_unset_target_is_rejected(self) -> None:
        status, _ = co.evaluate_guidance(
            _config(),
            _rest(),
            co.TargetPlacement(id=0, pos=co.WorkspacePoint(1.0, 0.0)),
            co.WorkspacePoint(0.0, 0.0),
            STEP_NS,
        )
        assert status == ContractStatus.IDENTITY_MISSING

    @pytest.mark.parametrize("pos", [(math.inf, 0.0), (0.0, math.nan), (-math.inf, 1.0)])
    def test_non_finite_position_is_rejected(self, pos: tuple[float, float]) -> None:
        status, _ = co.evaluate_guidance(
            _config(), _rest(), _target(), co.WorkspacePoint(*pos), STEP_NS
        )
        assert status == ContractStatus.VALUE_NOT_FINITE

    def test_overflowing_separation_is_reported(self) -> None:
        huge = 1.7976931348623157e308
        status, _ = co.evaluate_guidance(
            _config(),
            _rest(),
            _target(huge, 0.0),
            co.WorkspacePoint(-huge, 0.0),
            STEP_NS,
        )
        assert status == ContractStatus.VALUE_NOT_FINITE

    def test_invalid_state_is_rejected(self) -> None:
        status, _ = co.evaluate_guidance(
            _config(),
            co.CenterOutGuidanceState(target=0, vel=co.WorkspaceVelocity(4.0, 0.0), phase=IDLE),
            _target(),
            co.WorkspacePoint(0.0, 0.0),
            STEP_NS,
        )
        assert status == ContractStatus.OUTCOME_INVALID

    def test_inputs_are_not_modified(self) -> None:
        config = _config()
        previous = _state(12.5, -3.0, SPEEDING_UP)
        target = _target()
        pos = co.WorkspacePoint(30.0, -7.5)

        before = (
            config.max_speed,
            config.acceleration,
            config.deceleration,
            config.arrival_radius,
            previous.target,
            previous.vel.x,
            previous.vel.y,
            previous.phase,
            target.id,
            target.pos.x,
            target.pos.y,
            pos.x,
            pos.y,
        )
        status, _ = co.evaluate_guidance(config, previous, target, pos, STEP_NS)
        assert status == ContractStatus.OK
        after = (
            config.max_speed,
            config.acceleration,
            config.deceleration,
            config.arrival_radius,
            previous.target,
            previous.vel.x,
            previous.vel.y,
            previous.phase,
            target.id,
            target.pos.x,
            target.pos.y,
            pos.x,
            pos.y,
        )
        assert before == after


class TestAgainstTheSession:
    def test_task_supplies_the_geometry_unit(self) -> None:
        session = _session()
        assert co.validate_against(_config(), session) == ContractStatus.OK
        assert not hasattr(_config(), "geometry_unit")

    def test_arriving_must_imply_being_contained(self) -> None:
        session = _session(half_x=5.0, half_y=4.0, cursor=1.0)
        margin = 4.0 - 1.0

        assert co.validate_against(_config(arrival_radius=margin), session) == ContractStatus.OK
        assert (
            co.validate_against(_config(arrival_radius=margin + 1 / 1024), session)
            == ContractStatus.PARAMETER_OUT_OF_RANGE
        )

        # The bound earns its place: at the admissible radius, every point at
        # exactly that distance from a target is inside its acceptance region.
        centre = session.layout.center.pos
        for step in range(33):
            angle = 2.0 * math.pi * step / 32.0
            probe = co.WorkspacePoint(
                centre.x + margin * math.cos(angle), centre.y + margin * math.sin(angle)
            )
            status, inside = co.contains_cursor(session.acceptance, session.cursor, centre, probe)
            assert status == ContractStatus.OK
            assert inside

    def test_records_are_validated_individually_first(self) -> None:
        broken_session = co._raw_config()
        assert co.validate_against(_config(), broken_session) == ContractStatus.IDENTITY_MISSING
        assert (
            co.validate_against(co._raw_guidance_config(), _session())
            == ContractStatus.IDENTITY_MISSING
        )


class TestComposition:
    """Guidance in, transform blind -- rules 8 and 9, exercised end to end."""

    def test_orchestration_feeds_guidance_and_velocity(self) -> None:
        # This is the composition the ledger describes, and it is done *here*,
        # in the test standing in for orchestration. Neither module imports the
        # other: guidance produces a velocity in the workspace frame, and the
        # caller is what declares the command space and applies the blend.
        space = CommandSpace(
            id=1,
            dim=2,
            frame=CommandFrame.WORKSPACE_2D,
            axes=[
                CommandAxis(CommandAxisName.X, CommandUnit.METRES_PER_SECOND),
                CommandAxis(CommandAxisName.Y, CommandUnit.METRES_PER_SECOND),
            ],
        )
        config = _config()
        sample = _evaluate(config, _rest(), _target(3.0, 4.0), (0.0, 0.0))

        reference = VelocityVector(space=space.id, dim=2, values=[sample.vel.x, sample.vel.y])
        decoded = VelocityVector(space=space.id, dim=2, values=[0.5, -0.25])

        status, assisted = blend_velocity(space, decoded, reference, LinearAssistance(0.25))
        assert status == ContractStatus.OK
        assert assisted.values[0] == pytest.approx(0.75 * 0.5 + 0.25 * sample.vel.x, abs=1e-15)
        assert assisted.values[1] == pytest.approx(0.75 * -0.25 + 0.25 * sample.vel.y, abs=1e-15)

    def test_machine_runs_whole_trial_without_guidance(self) -> None:
        # Center-Out remains fully usable without assistance and without
        # guidance: a successful trial needs neither, and the machine's own API
        # has nowhere to put one.
        base = _session()
        session = co._with_trial_limit(base, 1)
        machine = co.CenterOutMachine()
        status, result = machine.start(7, session, 0)
        assert status == ContractStatus.OK

        centre = session.layout.center.pos
        status, result = machine.step(1_000_000, co.WorkspacePoint(centre.x, centre.y))
        assert status == ContractStatus.OK
        outward = session.layout.target(result.snapshot.outward_idx).pos
        status, result = machine.step(2_000_000, co.WorkspacePoint(outward.x, outward.y))
        assert status == ContractStatus.OK
        assert result.trial_decided
        assert result.trial.record.outcome == TrialOutcome.SUCCESS

        machine_surface = {name for name in dir(machine) if not name.startswith("_")}
        assert not any("guidance" in name.lower() for name in machine_surface)
