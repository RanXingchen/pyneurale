#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The Speech cue trial as a pure deterministic timed state machine.

Every assertion here is about what the machine does with explicit time and
prepared schedules. Nothing constructs a renderer, a microphone, a decoder, or a
clock, because the machine reads none of them and a test that supplied one would
be testing something else.
"""

from __future__ import annotations

import pytest

from neurale.experiments import (
    UNSET_STIMULUS_ID,
    ContractStatus,
    CueKind,
    ExperimentEventKind,
    TrialOutcome,
    speech,
)

PARADIGM = 909
T1 = 500_000_000
T2 = 300_000_000
T3 = 2_000_000_000
SEED = 0xC0FFEE1234567890
STIMULI = [11, 22, 33, 44]


# ---------------------------------------------------------------------------
# Fixtures


def _config(*, cross_enabled: bool = True, n_trials: int = 3, **overrides: object):
    values: dict[str, object] = {
        "black_bound_ns": T1,
        "cross_bound_ns": T2 if cross_enabled else 0,
        "content_bound_ns": T3,
        "cross_enabled": cross_enabled,
        "inter_trial_ns": 0,
        "schedule": speech.SpeechScheduleKind.SEEDED,
        "stimulus_order": speech.StimulusOrderPolicy.SEQUENTIAL,
        "seed": SEED,
        "n_trials": n_trials,
        "stimuli": STIMULI,
    }
    values.update(overrides)
    return speech.SpeechCueConfig(**values)


def _schedule(config, ordinal: int):
    status, schedule = speech.prepare_trial(config, ordinal)
    assert status == ContractStatus.OK
    return schedule


def _total(schedule) -> int:
    status, total = speech.trial_duration(schedule)
    assert status == ContractStatus.OK
    return total


def _started(config, start_ns: int = 0):
    machine = speech.SpeechMachine()
    status, result = machine.start(PARADIGM, config, _schedule(config, 0), start_ns)
    assert status == ContractStatus.OK
    return machine, result


def _markers(result) -> list[object]:
    # ``ExperimentEvent.code`` is a paradigm-owned integer, so it is read back
    # through the paradigm's own enumeration rather than compared to one.
    return [
        speech.SpeechMarker(event.code)
        for event in result.events
        if event.kind == ExperimentEventKind.PARADIGM_MARKER
    ]


def _kinds(result) -> list[object]:
    return [event.kind for event in result.events]


def _step(machine, time_ns: int, *schedule):
    status, result = machine.step(time_ns, *schedule)
    assert status == ContractStatus.OK
    return result


# ---------------------------------------------------------------------------


class TestOrdinarySequence:
    def test_black_then_cross_then_content(self) -> None:
        config = _config()
        first = _schedule(config, 0)
        machine, started = _started(config, 1_000)

        assert machine.state == speech.SpeechState.BLACK
        assert started.settled
        assert _kinds(started)[:2] == [
            ExperimentEventKind.SESSION_START,
            ExperimentEventKind.TRIAL_START,
        ]
        assert _markers(started) == [speech.SpeechMarker.BLACK_ONSET]
        assert started.n_requests == 1
        assert started.requests[0].cue == CueKind.BLACK
        assert started.requests[0].stimulus_id == UNSET_STIMULUS_ID
        assert started.requests[0].onset_ns == 1_000
        assert started.requests[0].duration_ns == first.black_duration_ns

        black_end = 1_000 + first.black_duration_ns
        cross_end = black_end + first.cross_duration_ns

        crossed = _step(machine, black_end)
        assert machine.state == speech.SpeechState.FIXATION_CROSS
        assert _markers(crossed) == [
            speech.SpeechMarker.BLACK_OFFSET,
            speech.SpeechMarker.CROSS_ONSET,
        ]
        assert crossed.requests[0].cue == CueKind.FIXATION_CROSS
        assert crossed.requests[0].onset_ns == black_end
        assert crossed.snapshot.presentation.cue == CueKind.FIXATION_CROSS

        content = _step(machine, cross_end)
        assert machine.state == speech.SpeechState.CONTENT
        assert _markers(content) == [
            speech.SpeechMarker.CROSS_OFFSET,
            speech.SpeechMarker.CONTENT_ONSET,
        ]
        assert content.requests[0].cue == CueKind.TEXT_CONTENT
        assert content.requests[0].stimulus_id == first.stimulus_id
        assert content.requests[0].duration_ns == first.content_duration_ns

    def test_trial_completes_at_content_end_boundary(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        machine, _ = _started(config, 1_000)
        end = 1_000 + _total(only)

        finished = _step(machine, end)
        assert finished.trial_decided
        assert speech.validate(finished.trial) == ContractStatus.OK
        assert finished.trial.record.interval.start_ns == 1_000
        assert finished.trial.record.interval.end_ns == end
        assert finished.trial.record.outcome == TrialOutcome.SUCCESS
        assert finished.trial.record.reason == int(speech.SpeechReason.CONTENT_ELAPSED)
        assert machine.complete

    def test_emitted_records_carry_paradigm_and_trial(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        _, started = _started(config, 4_000)
        for event in started.events:
            assert event.paradigm == PARADIGM
            assert event.trial.ordinal == 0
            assert event.trial.stimulus_id == only.stimulus_id
            assert event.time_ns == 4_000
        assert started.transitions[0].paradigm == PARADIGM
        assert started.transitions[0].from_state == int(speech.SpeechState.IDLE)
        assert started.transitions[0].to_state == int(speech.SpeechState.BLACK)
        assert started.transitions[0].cause == int(speech.SpeechCause.SESSION_STARTED)

    def test_stamped_identity_matches_draw_provenance(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        _, started = _started(config)
        # make_schedule_draws checks the identity rather than copying it, so this
        # passing is the machine and the schedule agreeing about the trial.
        status, draws = speech.make_schedule_draws(only, started.snapshot.trial, 0)
        assert status == ContractStatus.OK
        assert draws.count == 3


class TestDisabledCross:
    def test_no_state_no_marker_no_request_and_no_interval(self) -> None:
        config = _config(cross_enabled=False, n_trials=1)
        only = _schedule(config, 0)
        assert only.cross_duration_ns == 0
        machine, _ = _started(config)

        content = _step(machine, only.black_duration_ns)
        # CONTENT begins exactly at BLACK's end, with nothing in between.
        assert machine.state == speech.SpeechState.CONTENT
        assert _markers(content) == [
            speech.SpeechMarker.BLACK_OFFSET,
            speech.SpeechMarker.CONTENT_ONSET,
        ]
        assert content.n_requests == 1
        assert content.requests[0].cue == CueKind.TEXT_CONTENT
        assert content.requests[0].onset_ns == only.black_duration_ns
        assert all(
            transition.to_state != int(speech.SpeechState.FIXATION_CROSS)
            for transition in content.transitions
        )

        finished = _step(machine, _total(only))
        # A zero-length cross is persisted nowhere: the phase is absent from the
        # recorded timeline rather than present with no length.
        assert finished.trial.timeline.count == 2
        assert all(
            phase.phase != speech.SpeechPhase.CROSS for phase in finished.trial.timeline.phases
        )
        assert speech.validate(finished.trial) == ContractStatus.OK

    def test_toggling_cross_keeps_other_boundaries(self) -> None:
        with_cross = _schedule(_config(n_trials=1), 0)
        without = _schedule(_config(cross_enabled=False, n_trials=1), 0)
        # The draw ordinals are coordinates rather than a running counter, so the
        # machine's BLACK and CONTENT windows are the same either way.
        assert with_cross.black_duration_ns == without.black_duration_ns
        assert with_cross.content_duration_ns == without.content_duration_ns


class TestBoundaries:
    def test_phase_owns_start_not_end(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        start = 7_777
        black_end = start + only.black_duration_ns
        cross_end = black_end + only.cross_duration_ns
        content_end = cross_end + only.content_duration_ns
        machine, _ = _started(config, start)

        for time_ns, expected in (
            (start, speech.SpeechState.BLACK),
            (black_end - 1, speech.SpeechState.BLACK),
            (black_end, speech.SpeechState.FIXATION_CROSS),
            (cross_end - 1, speech.SpeechState.FIXATION_CROSS),
            (cross_end, speech.SpeechState.CONTENT),
            (content_end - 1, speech.SpeechState.CONTENT),
        ):
            _step(machine, time_ns)
            assert machine.state == expected
        assert _step(machine, content_end).trial_decided
        assert machine.complete

    def test_reported_timeline_is_running_timeline(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        _, started = _started(config, 100)
        status, expected = speech.build_timeline(config, only, 100)
        assert status == ContractStatus.OK
        actual = started.snapshot.timeline
        assert actual.count == expected.count
        assert [phase.interval.start_ns for phase in actual.phases] == [
            phase.interval.start_ns for phase in expected.phases
        ]
        assert [phase.interval.end_ns for phase in actual.phases] == [
            phase.interval.end_ns for phase in expected.phases
        ]


class TestLargeJumps:
    def test_step_crosses_every_phase_once(self) -> None:
        config = _config(n_trials=2)
        first = _schedule(config, 0)
        second = _schedule(config, 1)
        end = _total(first)
        machine, _ = _started(config)

        jumped = _step(machine, end + _total(second) + 1_000_000, second)
        assert _markers(jumped) == [
            speech.SpeechMarker.BLACK_OFFSET,
            speech.SpeechMarker.CROSS_ONSET,
            speech.SpeechMarker.CROSS_OFFSET,
            speech.SpeechMarker.CONTENT_ONSET,
            speech.SpeechMarker.CONTENT_OFFSET,
            speech.SpeechMarker.BLACK_ONSET,
        ]
        assert _kinds(jumped).count(ExperimentEventKind.TRIAL_STOP) == 1
        assert _kinds(jumped).count(ExperimentEventKind.TRIAL_START) == 1
        # Boundaries carry the instant they actually occurred, not the instant
        # the caller happened to ask at.
        assert jumped.trial.record.interval.end_ns == end
        # A trial was decided while more had already elapsed, so the step stopped
        # rather than inferring an unbounded run of trials.
        assert not jumped.settled
        assert machine.snapshot().trial.ordinal == 1

    def test_sequence_ordinals_increase_within_step(self) -> None:
        config = _config(n_trials=2)
        machine, _ = _started(config)
        jumped = _step(machine, _total(_schedule(config, 0)), _schedule(config, 1))
        emitted = (
            [(record.sequence, "event") for record in jumped.events]
            + [(record.sequence, "transition") for record in jumped.transitions]
            + [(record.sequence, "request") for record in jumped.requests]
        )
        ordinals = sorted(sequence for sequence, _ in emitted)
        assert len(set(ordinals)) == len(ordinals)
        assert ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))


class TestIdempotence:
    def test_repeated_timestamp_emits_nothing(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        machine, _ = _started(config)
        crossed = _step(machine, only.black_duration_ns)
        assert crossed.n_transitions == 1

        before = machine.snapshot()
        for _ in range(4):
            repeated = _step(machine, only.black_duration_ns)
            assert repeated.n_transitions == 0
            assert repeated.n_events == 0
            assert repeated.n_requests == 0
            assert not repeated.trial_decided
            assert repeated.settled
            assert machine.snapshot().state == before.state
            assert machine.snapshot().active.end_ns == before.active.end_ns

    def test_time_never_moves_backwards(self) -> None:
        config = _config(n_trials=1)
        machine, _ = _started(config, 1_000)
        _step(machine, 2_000)
        status, _ = machine.step(1_999)
        assert status == ContractStatus.TIME_REGRESSED
        assert machine.snapshot().time_ns == 2_000


class TestConsecutiveTrials:
    def test_next_trial_begins_where_last_ended(self) -> None:
        config = _config(n_trials=4)
        machine, _ = _started(config)
        cursor = 0
        for ordinal in range(4):
            cursor += _total(_schedule(config, ordinal))
            if ordinal + 1 < 4:
                result = _step(machine, cursor, _schedule(config, ordinal + 1))
                # No fourth interval: the next trial's BLACK is the blank period
                # between them.
                assert machine.snapshot().active.start_ns == cursor
                assert machine.state == speech.SpeechState.BLACK
            else:
                result = _step(machine, cursor)
            assert result.trial_decided
            assert result.trial.record.trial.ordinal == ordinal
            assert result.trial.record.interval.end_ns == cursor
            assert speech.validate(result.trial) == ContractStatus.OK
        assert machine.complete
        assert machine.snapshot().completed == 4

    def test_ended_session_refuses_to_step(self) -> None:
        config = _config(n_trials=1)
        machine, _ = _started(config)
        final = _step(machine, _total(_schedule(config, 0)))
        assert ExperimentEventKind.SESSION_STOP in _kinds(final)
        assert final.transitions[-1].to_state == int(speech.SpeechState.COMPLETE)
        assert final.transitions[-1].cause == int(speech.SpeechCause.TRIAL_LIMIT_REACHED)
        assert machine.step(10**12)[0] == ContractStatus.NOT_RUNNING
        # The machine stops asking for anything rather than leaving a cue up.
        assert machine.snapshot().presentation.cue == CueKind.NONE


class TestScheduleOwnership:
    def test_boundary_without_next_schedule_moves_nothing(self) -> None:
        config = _config(n_trials=2)
        machine, _ = _started(config)
        end = _total(_schedule(config, 0))
        before = machine.snapshot()

        status, _ = machine.step(end)
        assert status == ContractStatus.IDENTITY_MISSING
        assert machine.snapshot().state == before.state
        assert machine.snapshot().completed == before.completed
        assert machine.snapshot().trial.ordinal == before.trial.ordinal
        # The retry loses nothing.
        assert _step(machine, end, _schedule(config, 1)).trial_decided

    def test_schedule_must_name_its_trial(self) -> None:
        config = _config(n_trials=2)
        machine, _ = _started(config)
        end = _total(_schedule(config, 0))
        assert machine.step(end, _schedule(config, 0))[0] == ContractStatus.OUTCOME_INVALID
        assert machine.state == speech.SpeechState.BLACK

    def test_foreign_schedule_never_reaches_transition(self) -> None:
        config = _config(n_trials=2)
        machine, _ = _started(config)
        end = _total(_schedule(config, 0))
        original = _schedule(config, 1)
        foreign = speech.SpeechTrialSchedule(
            ordinal=original.ordinal,
            stimulus_id=999,
            cross_enabled=original.cross_enabled,
            black_duration_ns=original.black_duration_ns,
            cross_duration_ns=original.cross_duration_ns,
            content_duration_ns=original.content_duration_ns,
            sampler_version=original.sampler_version,
        )
        # build_timeline runs validate_against, so the ownership checks gate
        # every trial before it can produce a single record.
        assert machine.step(end, foreign)[0] == ContractStatus.TARGET_SET_INVALID
        assert machine.snapshot().completed == 0


class TestStartRefusals:
    def test_record_must_name_producing_paradigm(self) -> None:
        config = _config()
        machine = speech.SpeechMachine()
        status, _ = machine.start(0, config, _schedule(config, 0), 0)
        assert status == ContractStatus.IDENTITY_MISSING
        assert machine.state == speech.SpeechState.IDLE

    def test_configured_gap_is_rejected(self) -> None:
        config = _config(inter_trial_ns=1_000)
        machine = speech.SpeechMachine()
        status, _ = machine.start(PARADIGM, config, _schedule(config, 0), 0)
        # This machine has no state to spend a gap in, and running the session
        # anyway would disagree with the timeline the same configuration builds.
        assert status == ContractStatus.OUTCOME_INVALID
        assert machine.state == speech.SpeechState.IDLE

    def test_first_trial_is_trial_zero(self) -> None:
        config = _config()
        machine = speech.SpeechMachine()
        status, _ = machine.start(PARADIGM, config, _schedule(config, 1), 0)
        assert status == ContractStatus.OUTCOME_INVALID

    def test_second_start_requires_reset(self) -> None:
        config = _config()
        machine, _ = _started(config)
        status, _ = machine.start(PARADIGM, config, _schedule(config, 0), 0)
        assert status == ContractStatus.ALREADY_RUNNING

    def test_idle_machine_has_no_step(self) -> None:
        assert speech.SpeechMachine().step(0)[0] == ContractStatus.NOT_RUNNING


class TestResetAndReplay:
    def _run(self, config) -> list[tuple[object, ...]]:
        machine, _ = _started(config, 500)
        cursor = 500
        decided = []
        count = config.n_trials
        for ordinal in range(count):
            cursor += _total(_schedule(config, ordinal))
            result = (
                _step(machine, cursor, _schedule(config, ordinal + 1))
                if ordinal + 1 < count
                else _step(machine, cursor)
            )
            decided.append(
                (
                    result.trial.record.trial.ordinal,
                    result.trial.record.interval.start_ns,
                    result.trial.record.interval.end_ns,
                    result.trial.schedule.stimulus_id,
                    result.trial.timeline.count,
                )
            )
        assert machine.complete
        return decided

    def test_reset_then_replay_reproduces_run(self) -> None:
        config = _config(n_trials=3)
        first = self._run(config)
        second = self._run(config)
        assert first == second

    def test_reset_returns_machine_to_fresh_state(self) -> None:
        config = _config()
        machine, _ = _started(config)
        machine.reset()
        assert machine.state == speech.SpeechState.IDLE
        assert machine.paradigm == 0
        assert machine.configuration().n_trials == 0
        assert machine.snapshot().completed == 0


class TestSemanticTimeIsNotPresentationTime:
    def test_request_carries_decision_not_appearance(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        _, started = _started(config, 2_000)
        request = started.requests[0]
        # Both are the instant the paradigm decided. Nothing here is a
        # presenter-reported software observation point, and the machine has no
        # way to produce one.
        assert request.requested_ns == 2_000
        assert request.onset_ns == 2_000
        # Presenting a phase after it is over is pointless, so the phase's own
        # end is where the request expires.
        assert request.valid_until_ns == 2_000 + only.black_duration_ns
        assert request.duration_ns == only.black_duration_ns

    def test_believed_state_advances_on_request(self) -> None:
        config = _config(n_trials=1)
        only = _schedule(config, 0)
        machine, _ = _started(config, 2_000)
        black_end = 2_000 + only.black_duration_ns
        crossed = _step(machine, black_end)
        believed = crossed.snapshot.presentation
        # It moved because the machine issued a request, not because anything
        # reported a cue had appeared -- nothing did, and nothing can here.
        assert believed.since_ns == black_end
        assert believed.cue == CueKind.FIXATION_CROSS
        assert believed.stimulus_id == UNSET_STIMULUS_ID
        assert believed.paradigm == PARADIGM
        # PresentationOutcome is where a presenter would report an actual
        # appearance. This module defines its shape and produces none.
        assert not hasattr(speech, "PresentationOutcome")


class TestTrialValidation:
    def test_disagreeing_artefacts_are_rejected(self) -> None:
        config = _config(n_trials=1)
        machine, _ = _started(config)
        finished = _step(machine, _total(_schedule(config, 0)))
        trial = finished.trial
        assert speech.validate(trial) == ContractStatus.OK
        status, _shifted = speech.build_timeline(config, trial.schedule, 10**9)
        assert status == ContractStatus.OK

        for changed in (
            speech.SpeechTrial(
                record=trial.record,
                schedule=speech.SpeechTrialSchedule(
                    ordinal=trial.schedule.ordinal,
                    stimulus_id=trial.schedule.stimulus_id,
                    cross_enabled=trial.schedule.cross_enabled,
                    black_duration_ns=trial.schedule.black_duration_ns,
                    cross_duration_ns=trial.schedule.cross_duration_ns,
                    content_duration_ns=trial.schedule.content_duration_ns + 1,
                    sampler_version=trial.schedule.sampler_version,
                ),
                timeline=trial.timeline,
            ),
            # The same trial's phases, shifted: a timeline that describes some
            # other instant is not this trial's, however well formed it is.
            speech.SpeechTrial(record=trial.record, schedule=trial.schedule, timeline=_shifted),
        ):
            assert speech.validate(changed) == ContractStatus.OUTCOME_INVALID


class TestOwnershipBoundary:
    @pytest.mark.parametrize(
        "name",
        [
            "Renderer",
            "Microphone",
            "AudioCapture",
            "Decoder",
            "Clock",
            "Timer",
            "Thread",
            "PresentationOutcome",
        ],
    )
    def test_machine_owns_no_presentation_concerns(self, name: str) -> None:
        assert not hasattr(speech, name)

    def test_states_are_exactly_five(self) -> None:
        # No inter-trial state, and no trial-boundary state: the next trial's
        # BLACK is the blank period between two trials.
        assert list(speech.SpeechState.__members__) == [
            "IDLE",
            "BLACK",
            "FIXATION_CROSS",
            "CONTENT",
            "COMPLETE",
        ]
        for state in speech.SpeechState.__members__.values():
            assert speech.speech_state_declared(state)
        assert [
            state
            for state in speech.SpeechState.__members__.values()
            if speech.speech_state_is_phase(state)
        ] == [
            speech.SpeechState.BLACK,
            speech.SpeechState.FIXATION_CROSS,
            speech.SpeechState.CONTENT,
        ]
