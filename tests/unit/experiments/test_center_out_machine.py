#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The deterministic Center-Out 2D task state machine.

Every expected instant here is computed from the configuration in the test
rather than read back out of the machine, so a rule that changed would have to
be changed in two places to keep the suite green.

The boundary cases follow one frozen rule and are not separate preferences: a
``TimeInterval`` is half-open, so at the instant a window ends the window is
already over. Every precedence assertion below is a consequence of that.
"""

from __future__ import annotations

import pytest

from neurale.experiments import ContractStatus, ExperimentEventKind, TrialOutcome
from neurale.experiments import center_out as co

MS = 1_000_000
PARADIGM = 7
NOWHERE = (500.0, 500.0)
CENTRE = (0.0, 0.0)


def _layout(count: int = 8, radius: float = 100.0) -> object:
    request = co.RadialLayoutRequest(
        radius=radius,
        center_id=1,
        ids=[idx + 2 for idx in range(count)],
        spokes=list(range(count)),
    )
    return co.build_radial_layout(request)


def _config(
    hold_ns: int = 100 * MS,
    dwell_ns: int = 50 * MS,
    timeout_ns: int = 1000 * MS,
    trial_limit: int = 0,
    selection: object | None = None,
    seed: int = 0xBEEF,
    count: int = 8,
) -> object:
    return co._raw_config(
        geometry_unit=co.GeometryUnit.MILLIMETRES,
        layout=_layout(count),
        acceptance=co.AcceptanceRegion(half_extent_x=5.0, half_extent_y=5.0),
        cursor=co.CursorGeometry(extent=0.0),
        movement_timeout=co.PhaseDurations(to_center=timeout_ns, to_out=timeout_ns),
        hold_ns=hold_ns,
        reward_dwell=co.PhaseDurations(to_center=dwell_ns, to_out=dwell_ns),
        punish_dwell=co.PhaseDurations(to_center=dwell_ns, to_out=dwell_ns),
        selection=selection
        if selection is not None
        else co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
        seed=seed,
        trial_limit=trial_limit,
    )


def _point(pos: tuple[float, float]) -> object:
    return co.WorkspacePoint(x=pos[0], y=pos[1])


def _started(config: object, time_ns: int = 0) -> tuple[object, object]:
    machine = co.CenterOutMachine()
    status, result = machine.start(PARADIGM, config, time_ns)
    assert status == ContractStatus.OK
    return machine, result


def _step(machine: object, time_ns: int, cursor: tuple[float, float]) -> object:
    status, result = machine.step(time_ns, _point(cursor))
    assert status == ContractStatus.OK
    return result


def _outward_position(config: object, idx: int) -> tuple[float, float]:
    placement = config.layout.target(idx)
    return (placement.pos.x, placement.pos.y)


def _transition_tuple(transition: object) -> tuple:
    return (
        transition.time_ns,
        transition.sequence,
        transition.trial.ordinal,
        transition.trial.target_id,
        transition.paradigm,
        transition.from_state,
        transition.to_state,
        transition.cause,
    )


def _trial_tuple(trial: object) -> tuple:
    return (
        trial.record.trial.ordinal,
        trial.record.interval.start_ns,
        trial.record.interval.end_ns,
        trial.record.outcome,
        trial.record.reason,
        trial.outward_target,
        trial.outward_idx,
        trial.decided_phase,
        trial.center_acquire_ns,
        trial.outward_acquire_ns,
    )


class _Run:
    """One driven session, with everything it emitted kept for comparison."""

    def __init__(self, config: object, script: list[tuple[int, tuple[float, float]]]) -> None:
        self.transitions: list[tuple] = []
        self.events: list[tuple] = []
        self.trials: list[tuple] = []
        machine = co.CenterOutMachine()
        status, result = machine.start(PARADIGM, config, script[0][0] if script else 0)
        assert status == ContractStatus.OK
        self._collect(result)
        for time_ns, cursor in script:
            # Drain rather than carry: a step stops after deciding one trial, so
            # a caller that fell behind steps again at the same instant.
            for _ in range(8):
                if machine.complete:
                    break
                result = _step(machine, time_ns, cursor)
                self._collect(result)
                if result.settled:
                    break
        self.machine = machine
        self.snapshot = machine.snapshot()

    def _collect(self, result: object) -> None:
        self.transitions.extend(_transition_tuple(t) for t in result.transitions)
        self.events.extend((e.time_ns, e.sequence, e.kind, e.trial.ordinal) for e in result.events)
        if result.trial_decided:
            self.trials.append(_trial_tuple(result.trial))


def _scripted_cursor(config: object, time_ns: int) -> tuple[float, float]:
    """A piecewise-constant trajectory: centre, then a target, alternating.

    Piecewise constant so that two cadences whose instants both land on every
    breakpoint observe the same thing, which is what makes their outcomes
    comparable at all.
    """
    slot = time_ns // (400 * MS)
    if slot % 2 == 0:
        return CENTRE
    return _outward_position(config, slot % config.layout.count)


# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_unstarted_machine_refuses_to_step(self) -> None:
        machine = co.CenterOutMachine()
        assert machine.state == co.CenterOutState.IDLE
        status, _ = machine.step(0, _point(CENTRE))
        assert status == ContractStatus.NOT_RUNNING

    def test_start_requires_paradigm_and_valid_configuration(self) -> None:
        machine = co.CenterOutMachine()
        assert machine.start(0, _config(), 0)[0] == ContractStatus.IDENTITY_MISSING
        broken = _config()
        broken = co._raw_config(
            geometry_unit=broken.geometry_unit,
            layout=broken.layout,
            acceptance=broken.acceptance,
            cursor=broken.cursor,
            movement_timeout=co.PhaseDurations(to_center=0, to_out=1),
            hold_ns=broken.hold_ns,
            reward_dwell=broken.reward_dwell,
            punish_dwell=broken.punish_dwell,
            selection=broken.selection,
            seed=broken.seed,
            trial_limit=0,
        )
        assert machine.start(PARADIGM, broken, 0)[0] == ContractStatus.PARAMETER_OUT_OF_RANGE
        # A refused start leaves the machine where it was.
        assert machine.state == co.CenterOutState.IDLE

    def test_start_opens_first_trial(self) -> None:
        config = _config()
        machine, result = _started(config, 5 * MS)
        assert machine.state == co.CenterOutState.MOVE_TO_CENTER
        assert machine.paradigm == PARADIGM
        assert [e.kind for e in result.events] == [
            ExperimentEventKind.SESSION_START,
            ExperimentEventKind.TRIAL_START,
        ]
        assert len(result.transitions) == 1
        assert result.transitions[0].from_state == int(co.CenterOutState.IDLE)
        assert result.transitions[0].to_state == int(co.CenterOutState.MOVE_TO_CENTER)
        assert result.transitions[0].cause == int(co.CenterOutCause.SESSION_STARTED)
        assert result.snapshot.leg.start_ns == 5 * MS
        assert result.snapshot.leg.end_ns == 5 * MS + 1000 * MS
        assert result.snapshot.active_target == config.layout.center.id
        # The outward target is decided at trial start, is set, and is never the
        # centre.
        assert result.snapshot.outward_target != config.layout.center.id
        assert result.snapshot.outward_target != 0
        assert (
            result.snapshot.outward_target == config.layout.target(result.snapshot.outward_idx).id
        )

    def test_configuration_is_captured_not_referenced(self) -> None:
        config = _config()
        machine, _ = _started(config)
        assert machine.configuration().seed == config.seed
        assert machine.configuration().layout.count == config.layout.count

    def test_backwards_time_is_refused_without_effect(self) -> None:
        machine, _ = _started(_config(), 10 * MS)
        before = machine.snapshot()
        status, _ = machine.step(9 * MS, _point(CENTRE))
        assert status == ContractStatus.TIME_REGRESSED
        assert machine.snapshot().time_ns == before.time_ns
        assert machine.snapshot().state == before.state

    def test_non_finite_observation_is_reported(self) -> None:
        machine, _ = _started(_config())
        status, _ = machine.step(10 * MS, co.WorkspacePoint(x=float("nan"), y=0.0))
        assert status == ContractStatus.VALUE_NOT_FINITE

    def test_reset_returns_machine_to_fresh_state(self) -> None:
        machine, _ = _started(_config())
        _step(machine, 100 * MS, CENTRE)
        machine.reset()
        assert machine.state == co.CenterOutState.IDLE
        assert machine.paradigm == 0
        assert machine.configuration().layout.count == 0
        assert machine.step(0, _point(CENTRE))[0] == ContractStatus.NOT_RUNNING

    def test_start_refuses_running_session(self) -> None:
        machine, _ = _started(_config(), 10 * MS)
        _step(machine, 100 * MS, CENTRE)
        before = machine.snapshot()
        status, result = machine.start(PARADIGM, _config(), 0)
        assert status == ContractStatus.ALREADY_RUNNING
        # The running session is untouched, and the refused start wrote nothing.
        assert machine.snapshot().state == before.state
        assert machine.snapshot().completed == before.completed
        assert machine.snapshot().successes == before.successes
        # The refused start wrote nothing into the caller's result.
        assert result.transitions == []
        assert result.events == []
        assert result.snapshot.state == co.CenterOutState.IDLE
        assert not result.trial_decided

    def test_start_refuses_completed_session_until_reset(self) -> None:
        config = _config(trial_limit=1)
        machine, _ = _started(config, 0)
        # The centre leg times out, deciding the one allowed trial, and the step
        # stops at that decision leaving the failure dwell running.
        _step(machine, 2000 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.CENTER_FAILURE_DWELL
        # The dwell has already ended at this instant, so the next step ends it
        # and the trial limit completes the session.
        _step(machine, 2000 * MS, NOWHERE)
        assert machine.complete
        assert machine.start(PARADIGM, config, 0)[0] == ContractStatus.ALREADY_RUNNING
        machine.reset()
        assert machine.state == co.CenterOutState.IDLE
        status, _ = machine.start(PARADIGM, config, 0)
        assert status == ContractStatus.OK
        assert machine.state == co.CenterOutState.MOVE_TO_CENTER


class TestSuccessfulCycle:
    def test_out_and_back_cycle_lands_on_computed_instants(self) -> None:
        config = _config()
        machine, result = _started(config)
        idx = result.snapshot.outward_idx
        target = _outward_position(config, idx)

        # The hold begins when containment is first observed, not before.
        result = _step(machine, 100 * MS, CENTRE)
        assert machine.state == co.CenterOutState.HOLD_CENTER
        assert (result.snapshot.hold.start_ns, result.snapshot.hold.end_ns) == (
            100 * MS,
            200 * MS,
        )
        assert result.snapshot.contained
        assert result.transitions[0].cause == int(co.CenterOutCause.CONTAINMENT_GAINED)

        # Short of the hold nothing happens at all.
        result = _step(machine, 150 * MS, CENTRE)
        assert result.transitions == []
        assert machine.state == co.CenterOutState.HOLD_CENTER

        # The acquisition is stamped at the hold's own instant, not at the
        # observation that noticed it.
        result = _step(machine, 210 * MS, CENTRE)
        assert machine.state == co.CenterOutState.CENTER_SUCCESS_DWELL
        assert result.transitions[0].time_ns == 200 * MS
        assert result.transitions[0].cause == int(co.CenterOutCause.HOLD_COMPLETED)
        assert not result.trial_decided
        assert (result.snapshot.dwell.start_ns, result.snapshot.dwell.end_ns) == (
            200 * MS,
            250 * MS,
        )

        # The outward leg opens when the dwell ends, again on its own instant.
        result = _step(machine, 260 * MS, target)
        assert machine.state == co.CenterOutState.HOLD_OUT
        assert [t.to_state for t in result.transitions] == [
            int(co.CenterOutState.MOVE_TO_OUT),
            int(co.CenterOutState.HOLD_OUT),
        ]
        assert result.transitions[0].time_ns == 250 * MS
        assert result.snapshot.leg.start_ns == 250 * MS
        assert result.snapshot.active_target == config.layout.target(idx).id

        result = _step(machine, 400 * MS, target)
        assert machine.state == co.CenterOutState.OUT_SUCCESS_DWELL
        assert result.trial_decided
        trial = result.trial
        assert trial.record.outcome == TrialOutcome.SUCCESS
        assert trial.record.reason == int(co.CenterOutReason.OUTWARD_ACQUIRED)
        assert trial.record.trial.ordinal == 0
        assert (trial.record.interval.start_ns, trial.record.interval.end_ns) == (0, 360 * MS)
        assert trial.decided_phase == co.CenterOutPhase.TO_OUT
        assert trial.outward_target == config.layout.target(idx).id
        assert trial.center_acquire_ns == 200 * MS
        assert trial.outward_acquire_ns == 110 * MS
        assert (result.snapshot.completed, result.snapshot.successes) == (1, 1)
        assert [e.kind for e in result.events] == [ExperimentEventKind.TRIAL_STOP]

        # The next trial opens when the outcome dwell ends.
        result = _step(machine, 420 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.MOVE_TO_CENTER
        assert result.snapshot.trial.ordinal == 1
        assert [e.kind for e in result.events] == [ExperimentEventKind.TRIAL_START]

    def test_cycles_number_trials_in_order(self) -> None:
        config = _config()
        machine, result = _started(config)
        decided = []
        time_ns = 0
        for _ in range(3):
            idx = machine.snapshot().outward_idx
            target = _outward_position(config, idx)
            time_ns += 100 * MS
            _step(machine, time_ns, CENTRE)
            time_ns += 100 * MS
            _step(machine, time_ns, CENTRE)
            time_ns += 100 * MS
            _step(machine, time_ns, target)
            time_ns += 150 * MS
            result = _step(machine, time_ns, target)
            assert result.trial_decided, time_ns
            decided.append(result.trial)
            time_ns += 100 * MS
            _step(machine, time_ns, NOWHERE)
        assert [t.record.trial.ordinal for t in decided] == [0, 1, 2]
        assert all(t.record.outcome == TrialOutcome.SUCCESS for t in decided)


class TestFailures:
    def test_centre_leg_times_out_at_own_instant(self) -> None:
        config = _config()
        machine, _ = _started(config)
        result = _step(machine, 1200 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.CENTER_FAILURE_DWELL
        assert result.trial_decided
        trial = result.trial
        assert trial.record.outcome == TrialOutcome.TIMEOUT
        assert trial.decided_phase == co.CenterOutPhase.TO_CENTER
        assert trial.record.reason == int(co.CenterOutReason.CENTER_MOVEMENT_TIMEOUT)
        assert trial.record.interval.end_ns == 1000 * MS
        assert (trial.center_acquire_ns, trial.outward_acquire_ns) == (0, 0)
        assert (result.snapshot.completed, result.snapshot.successes) == (1, 0)
        # More had already elapsed than this step consumed.
        assert not result.settled

    def test_outward_timeout_keeps_acquired_centre(self) -> None:
        config = _config()
        machine, _ = _started(config)
        _step(machine, 100 * MS, CENTRE)
        _step(machine, 200 * MS, CENTRE)
        result = _step(machine, 2000 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.OUT_FAILURE_DWELL
        trial = result.trial
        assert trial.record.outcome == TrialOutcome.TIMEOUT
        assert trial.decided_phase == co.CenterOutPhase.TO_OUT
        assert trial.record.reason == int(co.CenterOutReason.OUTWARD_MOVEMENT_TIMEOUT)
        # The outward leg opened at 250 ms and was allowed 1000 ms.
        assert trial.record.interval.end_ns == 1250 * MS
        assert trial.center_acquire_ns == 200 * MS
        assert trial.outward_acquire_ns == 0

    def test_failed_centre_leg_does_not_move_cursor(self) -> None:
        # Cursor position is caller-owned, so a second trial also times out.
        config = _config()
        machine, _ = _started(config)
        result = _step(machine, 1200 * MS, NOWHERE)
        assert result.trial_decided
        assert not hasattr(result.snapshot, "cursor")
        assert not hasattr(result.trial, "cursor")
        result = _step(machine, 1200 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.MOVE_TO_CENTER
        assert result.snapshot.leg.start_ns == 1050 * MS
        result = _step(machine, 2100 * MS, NOWHERE)
        assert result.trial_decided
        assert result.trial.record.outcome == TrialOutcome.TIMEOUT


class TestHold:
    def test_leaving_discards_hold_and_returning_restarts(self) -> None:
        config = _config()
        machine, _ = _started(config)
        result = _step(machine, 100 * MS, CENTRE)
        assert result.snapshot.hold.end_ns == 200 * MS

        result = _step(machine, 150 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.MOVE_TO_CENTER
        assert result.transitions[0].cause == int(co.CenterOutCause.CONTAINMENT_LOST)
        assert result.snapshot.hold.start_ns == result.snapshot.hold.end_ns
        assert not result.snapshot.contained
        # Leaving the target does not buy time: the movement window is untouched.
        assert result.snapshot.leg.end_ns == 1000 * MS

        result = _step(machine, 160 * MS, CENTRE)
        assert machine.state == co.CenterOutState.HOLD_CENTER
        assert (result.snapshot.hold.start_ns, result.snapshot.hold.end_ns) == (
            160 * MS,
            260 * MS,
        )

        assert _step(machine, 259 * MS, CENTRE).transitions == []
        result = _step(machine, 260 * MS, CENTRE)
        assert machine.state == co.CenterOutState.CENTER_SUCCESS_DWELL
        assert result.transitions[0].time_ns == 260 * MS

    def test_zero_hold_acquires_on_first_inside_observation(self) -> None:
        config = _config(hold_ns=0, dwell_ns=0)
        machine, _ = _started(config)
        result = _step(machine, 10 * MS, CENTRE)
        # Centre acquired and the outward leg opened, all on one observation.
        assert machine.state == co.CenterOutState.MOVE_TO_OUT
        assert result.snapshot.leg.start_ns == 10 * MS
        assert result.settled


class TestBoundaryPrecedence:
    def test_hold_completing_at_deadline_loses_to_timeout(self) -> None:
        # The movement window is half-open, so at its end instant it is over.
        config = _config()
        machine, _ = _started(config)
        result = _step(machine, 900 * MS, CENTRE)
        assert result.snapshot.hold.end_ns == 1000 * MS == result.snapshot.leg.end_ns
        result = _step(machine, 1000 * MS, CENTRE)
        assert machine.state == co.CenterOutState.CENTER_FAILURE_DWELL
        assert result.trial.record.outcome == TrialOutcome.TIMEOUT

    def test_hold_completing_one_nanosecond_earlier_wins(self) -> None:
        config = _config()
        machine, _ = _started(config)
        _step(machine, 900 * MS - 1, CENTRE)
        result = _step(machine, 1000 * MS, CENTRE)
        assert machine.state == co.CenterOutState.CENTER_SUCCESS_DWELL
        assert result.transitions[0].time_ns == 1000 * MS - 1

    def test_containment_at_deadline_starts_no_hold(self) -> None:
        config = _config()
        machine, _ = _started(config)
        result = _step(machine, 1000 * MS, CENTRE)
        assert machine.state == co.CenterOutState.CENTER_FAILURE_DWELL
        assert result.trial.record.interval.end_ns == 1000 * MS
        assert [t.cause for t in result.transitions] == [int(co.CenterOutCause.MOVEMENT_TIMED_OUT)]

    def test_hold_completed_before_deadline_wins_late_news(self) -> None:
        # It happened first, so a later observation showing the cursor gone
        # cannot undo it.
        config = _config()
        machine, _ = _started(config)
        _step(machine, 100 * MS, CENTRE)
        result = _step(machine, 5000 * MS, NOWHERE)
        assert result.transitions[0].to_state == int(co.CenterOutState.CENTER_SUCCESS_DWELL)
        assert result.transitions[0].time_ns == 200 * MS

    def test_unfinishable_hold_still_records_containment(self) -> None:
        # Entering at 950 ms with a 100 ms hold cannot succeed, but the cursor
        # really was inside, so the transition is emitted and the leg then times
        # out at its own instant.
        config = _config()
        machine, _ = _started(config)
        result = _step(machine, 950 * MS, CENTRE)
        assert machine.state == co.CenterOutState.HOLD_CENTER
        assert result.snapshot.hold.end_ns == 1050 * MS
        result = _step(machine, 1010 * MS, CENTRE)
        assert machine.state == co.CenterOutState.CENTER_FAILURE_DWELL
        assert result.trial.record.interval.end_ns == 1000 * MS


class TestCadenceAndIdempotence:
    def test_repeated_settled_step_produces_nothing(self) -> None:
        config = _config()
        machine, result = _started(config)
        idx = result.snapshot.outward_idx
        target = _outward_position(config, idx)
        for time_ns, cursor in [
            (100 * MS, CENTRE),
            (200 * MS, CENTRE),
            (250 * MS, CENTRE),
            (300 * MS, target),
            (400 * MS, target),
        ]:
            result = _step(machine, time_ns, cursor)
        assert result.trial_decided
        assert result.settled

        for _ in range(5):
            again = _step(machine, 400 * MS, target)
            assert again.transitions == []
            assert again.events == []
            assert not again.trial_decided
            assert again.settled
            assert again.snapshot.completed == 1
            assert again.snapshot.state == co.CenterOutState.OUT_SUCCESS_DWELL

    def test_timeout_uses_elapsed_time_not_invocations(self) -> None:
        config = _config()
        sparse, _ = _started(config)
        one = _step(sparse, 1000 * MS, NOWHERE)
        assert one.trial_decided

        dense, _ = _started(config)
        decided = [_step(dense, tick * MS, NOWHERE) for tick in range(1, 1001)]
        matched = [r for r in decided if r.trial_decided]
        assert len(matched) == 1
        assert _trial_tuple(matched[0].trial) == _trial_tuple(one.trial)
        assert dense.state == sparse.state

    def test_two_cadences_decide_same_trials(self) -> None:
        config = _config(trial_limit=6)
        dense = _Run(
            config, [(i * 10 * MS, _scripted_cursor(config, i * 10 * MS)) for i in range(1200)]
        )
        sparse = _Run(
            config, [(i * 50 * MS, _scripted_cursor(config, i * 50 * MS)) for i in range(240)]
        )
        assert dense.trials
        assert dense.trials == sparse.trials
        assert dense.snapshot.completed == sparse.snapshot.completed


class TestDeterminism:
    def test_same_script_emits_same_records(self) -> None:
        config = _config(trial_limit=5)
        script = [(i * 20 * MS, _scripted_cursor(config, i * 20 * MS)) for i in range(600)]
        left = _Run(config, script)
        right = _Run(config, script)
        assert left.transitions == right.transitions
        assert left.events == right.events
        assert left.trials == right.trials
        assert left.trials

    def test_reset_then_replay_reproduces_run(self) -> None:
        config = _config(trial_limit=5)
        script = [(i * 20 * MS, _scripted_cursor(config, i * 20 * MS)) for i in range(600)]
        first = _Run(config, script)
        first.machine.reset()
        assert first.machine.state == co.CenterOutState.IDLE
        second = _Run(config, script)
        assert first.trials == second.trials
        assert first.transitions == second.transitions

    def test_outward_target_matches_schedule(self) -> None:
        config = _config(trial_limit=6)
        run = _Run(
            config, [(i * 20 * MS, _scripted_cursor(config, i * 20 * MS)) for i in range(800)]
        )
        successes = 0
        for ordinal, trial in enumerate(run.trials):
            status, expected = co.select_outward_target(config, ordinal, successes)
            assert status == ContractStatus.OK
            assert trial[6] == expected, ordinal
            assert trial[5] != config.layout.center.id
            if trial[3] == TrialOutcome.SUCCESS:
                successes += 1

    def test_repeat_until_success_represents_failed_target(self) -> None:
        config = _config(trial_limit=4, selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS)
        # Never inside anything, so every trial times out on the centre leg and
        # the success count never advances.
        run = _Run(config, [(i * 100 * MS, NOWHERE) for i in range(60)])
        assert len(run.trials) == 4
        assert {t[3] for t in run.trials} == {TrialOutcome.TIMEOUT}
        assert {t[6] for t in run.trials} == {0}

    def test_different_seed_gives_different_target_sequence(self) -> None:
        script_config = _config(trial_limit=8, seed=1)
        other_config = _config(trial_limit=8, seed=2)
        left = _Run(
            script_config,
            [(i * 20 * MS, _scripted_cursor(script_config, i * 20 * MS)) for i in range(900)],
        )
        right = _Run(
            other_config,
            [(i * 20 * MS, _scripted_cursor(other_config, i * 20 * MS)) for i in range(900)],
        )
        assert [t[6] for t in left.trials] != [t[6] for t in right.trials]


class TestSessionEnd:
    def test_trial_limit_completes_session(self) -> None:
        config = _config(trial_limit=2)
        run = _Run(config, [(i * 100 * MS, NOWHERE) for i in range(40)])
        assert len(run.trials) == 2
        assert run.snapshot.state == co.CenterOutState.COMPLETE
        assert run.snapshot.completed == 2
        assert ExperimentEventKind.SESSION_STOP in {e[2] for e in run.events}
        # A finished session refuses to step rather than quietly doing nothing.
        assert run.machine.step(10**12, _point(CENTRE))[0] == ContractStatus.NOT_RUNNING
        assert run.machine.complete

    def test_last_trial_outcome_dwell_still_runs(self) -> None:
        # Cutting the dwell short on the last trial would make the last trial a
        # different trial from every other one.
        config = _config(trial_limit=1, dwell_ns=50 * MS)
        machine, _ = _started(config)
        result = _step(machine, 1000 * MS, NOWHERE)
        assert result.trial_decided
        assert machine.state == co.CenterOutState.CENTER_FAILURE_DWELL
        assert result.snapshot.dwell.end_ns == 1050 * MS
        assert _step(machine, 1049 * MS, NOWHERE).snapshot.state == (
            co.CenterOutState.CENTER_FAILURE_DWELL
        )
        result = _step(machine, 1050 * MS, NOWHERE)
        assert machine.state == co.CenterOutState.COMPLETE
        assert result.transitions[0].cause == int(co.CenterOutCause.TRIAL_LIMIT_REACHED)


class TestStatistics:
    def test_summary_agrees_with_its_machine(self) -> None:
        config = _config(trial_limit=6)
        run = _Run(
            config, [(i * 20 * MS, _scripted_cursor(config, i * 20 * MS)) for i in range(900)]
        )
        trials = []
        machine, _ = _started(config)
        # Re-drive the same script to collect the CenterOutTrial objects rather
        # than the tuples the comparison helper keeps.
        for time_ns, cursor in [
            (i * 20 * MS, _scripted_cursor(config, i * 20 * MS)) for i in range(900)
        ]:
            for _ in range(8):
                if machine.complete:
                    break
                result = _step(machine, time_ns, cursor)
                if result.trial_decided:
                    trials.append(result.trial)
                if result.settled:
                    break
        status, statistics = co.summarize(trials)
        assert status == ContractStatus.OK
        snapshot = machine.snapshot()
        assert statistics.decided == snapshot.completed == len(run.trials)
        assert statistics.successes == snapshot.successes
        assert (
            statistics.decided
            == statistics.successes + statistics.center_timeouts + statistics.outward_timeouts
        )

    def test_empty_summary_has_no_mean(self) -> None:
        status, statistics = co.summarize([])
        assert status == ContractStatus.OK
        assert statistics.decided == 0
        assert statistics.mean_time_to_target_ns == 0

    def test_mean_is_truncated_integer_over_successes(self) -> None:
        from neurale.experiments import TimeInterval, TrialIdentity, TrialRecord

        def trial(
            outcome: object,
            interval_end: int,
            center_acquire: int,
            outward_acquire: int,
            phase: object,
        ) -> object:
            if outcome == TrialOutcome.SUCCESS:
                reason = int(co.CenterOutReason.OUTWARD_ACQUIRED)
            elif phase == co.CenterOutPhase.TO_CENTER:
                reason = int(co.CenterOutReason.CENTER_MOVEMENT_TIMEOUT)
            else:
                reason = int(co.CenterOutReason.OUTWARD_MOVEMENT_TIMEOUT)
            return co.CenterOutTrial(
                record=TrialRecord(
                    trial=TrialIdentity(target_id=2),
                    interval=TimeInterval(start_ns=0, end_ns=interval_end),
                    paradigm=PARADIGM,
                    outcome=outcome,
                    reason=reason,
                ),
                outward_target=2,
                outward_idx=0,
                decided_phase=phase,
                center_acquire_ns=center_acquire,
                outward_acquire_ns=outward_acquire,
            )

        # Each interval covers the acquisition durations it reports, so the
        # trials are machine-like rather than records that happen to pass a
        # field-by-field check. The timeouts carry no outward acquisition; the
        # outward timeout keeps the centre it did acquire.
        trials = [
            trial(TrialOutcome.SUCCESS, 100, 0, 100, co.CenterOutPhase.TO_OUT),
            trial(TrialOutcome.SUCCESS, 201, 0, 201, co.CenterOutPhase.TO_OUT),
            trial(TrialOutcome.TIMEOUT, 999, 0, 0, co.CenterOutPhase.TO_CENTER),
            trial(TrialOutcome.TIMEOUT, 999, 200, 0, co.CenterOutPhase.TO_OUT),
        ]
        status, statistics = co.summarize(trials)
        assert status == ContractStatus.OK
        assert statistics.decided == 4
        assert statistics.successes == 2
        assert statistics.center_timeouts == 1
        assert statistics.outward_timeouts == 1
        assert statistics.total_time_to_target_ns == 301
        # Truncated integer division, exactly reproducible.
        assert statistics.mean_time_to_target_ns == 150

    def test_summary_reports_unrepresentable_total(self) -> None:
        from neurale.experiments import TimeInterval, TrialIdentity, TrialRecord

        # The acquisition fits inside its own trial -- the interval runs the full
        # width -- so the trial is valid; it is the *sum* of two such trials that
        # does not fit, which is what duration_overflow reports.
        huge = co.CenterOutTrial(
            record=TrialRecord(
                trial=TrialIdentity(target_id=2),
                interval=TimeInterval(start_ns=0, end_ns=2**64 - 1),
                paradigm=PARADIGM,
                outcome=TrialOutcome.SUCCESS,
                reason=int(co.CenterOutReason.OUTWARD_ACQUIRED),
            ),
            outward_target=2,
            decided_phase=co.CenterOutPhase.TO_OUT,
            center_acquire_ns=0,
            outward_acquire_ns=2**64 - 1,
        )
        status, _ = co.summarize([huge, huge])
        assert status == ContractStatus.DURATION_OVERFLOW

    def test_summary_rejects_undecided_trials(self) -> None:
        from neurale.experiments import TimeInterval, TrialIdentity, TrialRecord

        def trial(
            outcome: object = TrialOutcome.SUCCESS,
            phase: object = co.CenterOutPhase.TO_OUT,
            reason: int = int(co.CenterOutReason.OUTWARD_ACQUIRED),
            outward_target: int = 2,
            record_target: int = 2,
            empty_interval: bool = False,
        ) -> object:
            return co.CenterOutTrial(
                record=TrialRecord(
                    trial=TrialIdentity(target_id=record_target),
                    interval=TimeInterval(start_ns=0, end_ns=0 if empty_interval else 1),
                    paradigm=PARADIGM,
                    outcome=outcome,
                    reason=reason,
                ),
                outward_target=outward_target,
                decided_phase=phase,
            )

        def rejected(trial_obj: object, status: object) -> None:
            result, statistics = co.summarize([trial_obj])
            assert result == status
            # A refused summary leaves the statistics untouched.
            assert statistics.decided == 0

        # An undeclared outcome is caught by the shared record validator.
        rejected(trial(outcome=TrialOutcome(255)), ContractStatus.ENUM_UNDECLARED)
        # An unset outward target.
        rejected(trial(outward_target=0), ContractStatus.IDENTITY_MISSING)
        # A record target id that disagrees with the outward target.
        rejected(trial(record_target=3), ContractStatus.OUTCOME_INVALID)
        # A pending outcome is not a decided trial; the empty interval keeps the
        # record itself valid so the combination check is what refuses it.
        rejected(
            trial(outcome=TrialOutcome.PENDING, reason=0, empty_interval=True),
            ContractStatus.OUTCOME_INVALID,
        )
        # A success on the wrong leg.
        rejected(trial(phase=co.CenterOutPhase.TO_CENTER), ContractStatus.OUTCOME_INVALID)
        # A timeout with the wrong reason for its leg.
        rejected(
            trial(
                outcome=TrialOutcome.TIMEOUT,
                phase=co.CenterOutPhase.TO_CENTER,
                reason=int(co.CenterOutReason.OUTWARD_MOVEMENT_TIMEOUT),
            ),
            ContractStatus.OUTCOME_INVALID,
        )
        # A valid trial is still summarized.
        status, statistics = co.summarize([trial()])
        assert status == ContractStatus.OK
        assert statistics.decided == 1
        assert statistics.successes == 1

    def test_summary_rejects_acquisition_index_inconsistency(self) -> None:
        from neurale.experiments import TimeInterval, TrialIdentity, TrialRecord

        def trial(
            outcome: object,
            phase: object,
            interval_end: int,
            center_acquire: int,
            outward_acquire: int,
            outward_idx: int = 0,
        ) -> object:
            if outcome == TrialOutcome.SUCCESS:
                reason = int(co.CenterOutReason.OUTWARD_ACQUIRED)
            elif phase == co.CenterOutPhase.TO_CENTER:
                reason = int(co.CenterOutReason.CENTER_MOVEMENT_TIMEOUT)
            else:
                reason = int(co.CenterOutReason.OUTWARD_MOVEMENT_TIMEOUT)
            return co.CenterOutTrial(
                record=TrialRecord(
                    trial=TrialIdentity(target_id=2),
                    interval=TimeInterval(start_ns=0, end_ns=interval_end),
                    paradigm=PARADIGM,
                    outcome=outcome,
                    reason=reason,
                ),
                outward_target=2,
                outward_idx=outward_idx,
                decided_phase=phase,
                center_acquire_ns=center_acquire,
                outward_acquire_ns=outward_acquire,
            )

        def rejected(trial_obj: object, status: object) -> None:
            result, statistics = co.summarize([trial_obj])
            assert result == status
            assert statistics.decided == 0

        def accepted(trial_obj: object) -> None:
            result, statistics = co.summarize([trial_obj])
            assert result == ContractStatus.OK
            assert statistics.decided == 1

        # A success whose outward acquisition outlasts the trial it is reported
        # in: an interval of 100 ns cannot contain a 101 ns time to target.
        rejected(
            trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 100, 0, 101),
            ContractStatus.OUTCOME_INVALID,
        )
        # A success whose centre acquisition outlasts the trial.
        rejected(
            trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 100, 101, 100),
            ContractStatus.OUTCOME_INVALID,
        )
        # A centre timeout that nonetheless reports a centre acquisition: the
        # centre never acquired, so the duration must be zero.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_CENTER, 1000, 1, 0),
            ContractStatus.OUTCOME_INVALID,
        )
        # A centre timeout that reports an outward acquisition.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_CENTER, 1000, 0, 1),
            ContractStatus.OUTCOME_INVALID,
        )
        # An outward timeout that reports an outward acquisition: the outward leg
        # never acquired, so the duration must be zero.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_OUT, 1000, 200, 1),
            ContractStatus.OUTCOME_INVALID,
        )
        # An outward timeout whose centre acquisition outlasts the trial.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_OUT, 100, 101, 0),
            ContractStatus.OUTCOME_INVALID,
        )
        # An outward timeout whose centre acquisition fills the whole trial: the
        # outward movement window is strictly positive, so the trial must end
        # strictly after the centre was acquired.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_OUT, 100, 100, 0),
            ContractStatus.OUTCOME_INVALID,
        )
        # An outward timeout cannot be a zero-length trial either.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_OUT, 0, 0, 0),
            ContractStatus.OUTCOME_INVALID,
        )
        # A centre timeout cannot be a zero-length trial: the centre movement
        # window is strictly positive, so the machine always runs it before it
        # times out.
        rejected(
            trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_CENTER, 0, 0, 0),
            ContractStatus.OUTCOME_INVALID,
        )
        # An outward index no layout can hold: 16 is at the capacity, not inside.
        rejected(
            trial(
                TrialOutcome.SUCCESS,
                co.CenterOutPhase.TO_OUT,
                100,
                0,
                100,
                outward_idx=co.MAX_SURROUNDING_TARGETS,
            ),
            ContractStatus.OUTCOME_INVALID,
        )

        # Cross-stage consistency on a success: the outward leg cannot begin
        # before the centre is acquired, so the two acquisitions must sum to no
        # more than the trial. A centre reward dwell only makes the trial longer.
        # These two each pass the per-duration bound yet fail the sum.
        rejected(
            trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 100, 50, 51),
            ContractStatus.OUTCOME_INVALID,
        )
        rejected(
            trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 100, 100, 1),
            ContractStatus.OUTCOME_INVALID,
        )

        # The boundary invariants admit the trials they should. A success may
        # have both acquisitions zero (a zero hold, an overlapping geometry), so
        # only an upper -- not a positive lower -- bound is enforced.
        accepted(trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 1, 0, 0))
        # A zero-duration success is admissible: zero hold, zero centre dwell, and
        # a geometry that lets both legs complete at the same instant.
        accepted(trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 0, 0, 0))
        # The two acquisitions sum exactly to the trial duration, each staying
        # inside it: the outward leg begins the instant the centre is acquired.
        accepted(trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 100, 50, 50))
        # A centre acquisition equal to the trial duration, with the outward leg
        # completing at the same instant it began.
        accepted(trial(TrialOutcome.SUCCESS, co.CenterOutPhase.TO_OUT, 100, 100, 0))
        # The last admissible surrounding index is one past the largest in use.
        accepted(
            trial(
                TrialOutcome.SUCCESS,
                co.CenterOutPhase.TO_OUT,
                100,
                0,
                100,
                outward_idx=co.MAX_SURROUNDING_TARGETS - 1,
            )
        )
        # An outward timeout keeps the centre it did acquire, ending strictly
        # after.
        accepted(trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_OUT, 1000, 200, 0))
        # A centre timeout runs for its strictly positive movement window.
        accepted(trial(TrialOutcome.TIMEOUT, co.CenterOutPhase.TO_CENTER, 1000, 0, 0))


class TestStateVocabulary:
    @pytest.mark.parametrize(
        ("state", "phase"),
        [
            (co.CenterOutState.MOVE_TO_CENTER, co.CenterOutPhase.TO_CENTER),
            (co.CenterOutState.HOLD_CENTER, co.CenterOutPhase.TO_CENTER),
            (co.CenterOutState.CENTER_SUCCESS_DWELL, co.CenterOutPhase.TO_CENTER),
            (co.CenterOutState.CENTER_FAILURE_DWELL, co.CenterOutPhase.TO_CENTER),
            (co.CenterOutState.MOVE_TO_OUT, co.CenterOutPhase.TO_OUT),
            (co.CenterOutState.HOLD_OUT, co.CenterOutPhase.TO_OUT),
            (co.CenterOutState.OUT_SUCCESS_DWELL, co.CenterOutPhase.TO_OUT),
            (co.CenterOutState.OUT_FAILURE_DWELL, co.CenterOutPhase.TO_OUT),
        ],
    )
    def test_states_belong_to_their_leg(self, state: object, phase: object) -> None:
        assert co.center_out_phase_of(state) == phase

    def test_leg_and_dwell_predicates_partition_states(self) -> None:
        running = [
            co.CenterOutState.MOVE_TO_CENTER,
            co.CenterOutState.HOLD_CENTER,
            co.CenterOutState.CENTER_SUCCESS_DWELL,
            co.CenterOutState.CENTER_FAILURE_DWELL,
            co.CenterOutState.MOVE_TO_OUT,
            co.CenterOutState.HOLD_OUT,
            co.CenterOutState.OUT_SUCCESS_DWELL,
            co.CenterOutState.OUT_FAILURE_DWELL,
        ]
        for state in running:
            assert co.center_out_state_is_leg(state) != co.center_out_state_is_dwell(state)
        for state in (co.CenterOutState.IDLE, co.CenterOutState.COMPLETE):
            assert not co.center_out_state_is_leg(state)
            assert not co.center_out_state_is_dwell(state)
        assert co.center_out_state_declared(co.CenterOutState.COMPLETE)

    def test_new_status_has_own_message(self) -> None:
        from neurale.experiments import contract_status_message

        message = contract_status_message(ContractStatus.NOT_RUNNING)
        assert message
        assert message != contract_status_message(ContractStatus.OUTCOME_INVALID)
