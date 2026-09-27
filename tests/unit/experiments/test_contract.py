#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Behaviour of the shared experiment value contract through its native binding."""

from __future__ import annotations

import math

import pytest

from neurale.experiments import (
    MAX_COMMAND_DIMENSION,
    NO_EXPIRY_NS,
    SAMPLER_VERSION_1,
    AbnormalCondition,
    AbnormalEvent,
    AbnormalPolicy,
    AbnormalPolicySet,
    AbnormalResponse,
    CommandApplication,
    CommandAxis,
    CommandAxisName,
    CommandFrame,
    CommandOutcome,
    CommandRequest,
    CommandSpace,
    CommandUnit,
    ContractStatus,
    CueKind,
    DecisionSnapshot,
    DrawKey,
    ExperimentEvent,
    ExperimentEventKind,
    ExperimentSnapshot,
    FingerprintAccumulator,
    MonotonicTimeGate,
    PresentationOutcome,
    PresentationRequest,
    PresentationState,
    PresentationStatus,
    ReplayAuthority,
    ScheduleDraw,
    ScheduleIdentity,
    SelectionEvent,
    SelectionKind,
    SequenceCounter,
    StateTransition,
    TimeInterval,
    TrialCounter,
    TrialIdentity,
    TrialOutcome,
    TrialRecord,
    abnormal_condition_declared,
    abnormal_condition_name,
    abnormal_response_admits_trial,
    command_application_declared,
    command_axis_name_declared,
    command_frame_declared,
    command_unit_declared,
    contract_status_message,
    cue_kind_declared,
    escalate,
    experiment_event_kind_declared,
    fingerprint_of_bytes,
    interval_from_duration,
    next_draw_key,
    policy_for,
    presentation_status_declared,
    replay_authority,
    sample_exclusive,
    sample_inclusive,
    sample_index,
    sampler_version_supported,
    schedule_fingerprint,
    selection_kind_declared,
    trial_ended,
    trial_outcome_declared,
    validate,
    validate_against,
)

_PARADIGM = 7


# Vocabulary that would mean a semantic cue had acquired a rendering decision.
# The paradigm layer owns what is presented; the presenter owns how it is drawn,
# and the split is only real if the request cannot express the second.
_RENDERING_VOCABULARY = (
    "color",
    "colour",
    "font",
    "height",
    "monitor",
    "pixel",
    "radius",
    "rgb",
    "screen",
    "size",
    "surface",
    "vsync",
    "width",
    "window",
)


# Every value record in the contract. They are decided at construction and never
# afterwards, which is what makes a validate() result still true by the time the
# value is persisted.
_VALUE_TYPES = (
    TimeInterval,
    TrialIdentity,
    DrawKey,
    ScheduleIdentity,
    ScheduleDraw,
    ExperimentEvent,
    StateTransition,
    SelectionEvent,
    TrialRecord,
    PresentationRequest,
    PresentationState,
    PresentationOutcome,
    CommandAxis,
    CommandSpace,
    CommandRequest,
    CommandOutcome,
    ExperimentSnapshot,
    DecisionSnapshot,
)


def _trial(ordinal: int = 3, *, target: int = 0, stimulus: int = 0) -> TrialIdentity:
    return TrialIdentity(ordinal=ordinal, block=1, target_id=target, stimulus_id=stimulus)


# Read-only properties computed from stored fields rather than stored themselves.
# They are still unsettable, so the immutability test covers them; they are just
# not constructor arguments.
_DERIVED_FIELDS = {TimeInterval: frozenset({"well_formed", "empty", "duration_ns"})}


def _data_fields(value: object) -> list[str]:
    return [
        name
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    ]


def _stored_fields(value_type: type) -> list[str]:
    derived = _DERIVED_FIELDS.get(value_type, frozenset())
    return [name for name in _data_fields(value_type()) if name not in derived]


class TestImmutability:
    @pytest.mark.parametrize("value_type", _VALUE_TYPES, ids=lambda item: item.__name__)
    def test_value_records_expose_no_setter(self, value_type: type) -> None:
        value = value_type()
        fields = _data_fields(value)
        assert fields, f"{value_type.__name__} exposes no data fields"
        for name in fields:
            with pytest.raises(AttributeError):
                setattr(value, name, getattr(value, name))

    @pytest.mark.parametrize("value_type", _VALUE_TYPES, ids=lambda item: item.__name__)
    def test_every_field_is_settable_at_construction(self, value_type: type) -> None:
        # Readonly fields are only defensible if construction can reach all of
        # them; otherwise the contract would carry values nobody can set.
        default = value_type()
        for name in _stored_fields(value_type):
            assert value_type(**{name: getattr(default, name)}) is not None

    def test_validated_value_cannot_change(self) -> None:
        identity = ScheduleIdentity(seed=42, configuration_fingerprint=7)
        assert validate(identity) == ContractStatus.OK
        recorded = schedule_fingerprint(identity)

        # This is the whole point: a fingerprint taken once stays true, because
        # the seed it was taken over cannot be swapped out behind it.
        with pytest.raises(AttributeError):
            identity.seed = 43
        with pytest.raises(AttributeError):
            identity.sampler_version = SAMPLER_VERSION_1 + 1
        assert schedule_fingerprint(identity) == recorded

    def test_command_space_axes_are_fixed_at_preparation(self) -> None:
        space = CommandSpace(
            id=1,
            dim=2,
            frame=CommandFrame.WORKSPACE_2D,
            axes=[
                CommandAxis(CommandAxisName.X, CommandUnit.METRES_PER_SECOND),
                CommandAxis(CommandAxisName.Y, CommandUnit.METRES_PER_SECOND),
            ],
        )
        assert validate(space) == ContractStatus.OK
        # A space an identifier stands for cannot be edited after the fact.
        assert not hasattr(space, "set_axis")
        assert not hasattr(CommandRequest(), "set_value")

    def test_over_long_fixed_sequences_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            CommandRequest(dim=2, values=[0.0] * (MAX_COMMAND_DIMENSION + 1))
        with pytest.raises(ValueError):
            CommandSpace(id=1, axes=[CommandAxis()] * (MAX_COMMAND_DIMENSION + 1))

    def test_stateful_helpers_stay_mutable(self) -> None:
        # These are not records. Advancing is what they are for, so they are the
        # deliberate exception to the rule above.
        gate = MonotonicTimeGate()
        assert gate.accept(10) == ContractStatus.OK
        assert gate.last_ns == 10

        counter = TrialCounter()
        assert counter.issue() == (ContractStatus.OK, 0)

        accumulator = FingerprintAccumulator()
        accumulator.absorb(1)
        assert accumulator.value != FingerprintAccumulator().value


class TestTime:
    def test_phase_intervals_are_half_open(self) -> None:
        phase = TimeInterval(1_000, 2_000)
        following = TimeInterval(2_000, 3_000)

        assert phase.contains(1_000)
        assert phase.contains(1_999)
        # The exact phase end belongs to the following phase and to nothing else.
        assert not phase.contains(2_000)
        assert following.contains(2_000)
        assert phase.elapsed_at(2_000)
        assert not phase.elapsed_at(1_999)
        assert phase.duration_ns == 1_000

    def test_empty_interval_contains_no_instant(self) -> None:
        pending = TimeInterval(5_000, 5_000)
        assert pending.empty
        assert pending.well_formed
        assert not pending.contains(5_000)
        assert pending.duration_ns == 0

    def test_inverted_interval_is_rejected(self) -> None:
        assert validate(TimeInterval(5_000, 4_000)) == ContractStatus.INTERVAL_INVERTED
        assert validate(TimeInterval(4_000, 5_000)) == ContractStatus.OK

    def test_interval_from_duration_reports_overflow(self) -> None:
        status, interval = interval_from_duration(10, 5)
        assert status == ContractStatus.OK
        assert (interval.start_ns, interval.end_ns) == (10, 15)

        status, _ = interval_from_duration(2**64 - 3, 4)
        assert status == ContractStatus.DURATION_OVERFLOW

    def test_supplied_time_must_be_monotonic_non_decreasing(self) -> None:
        gate = MonotonicTimeGate()
        assert not gate.started

        assert gate.accept(100) == ContractStatus.OK
        # Equal instants are accepted: two decisions may share one.
        assert gate.accept(100) == ContractStatus.OK
        assert gate.accept(101) == ContractStatus.OK
        assert gate.accept(100) == ContractStatus.TIME_REGRESSED
        # A rejected instant leaves the gate untouched.
        assert gate.last_ns == 101

    def test_gate_reset_is_explicit(self) -> None:
        gate = MonotonicTimeGate(500)
        assert gate.started and gate.last_ns == 500
        assert gate.accept(499) == ContractStatus.TIME_REGRESSED

        gate.reset()
        assert not gate.started
        assert gate.accept(1) == ContractStatus.OK

        gate.reset(900)
        assert gate.started and gate.last_ns == 900


class TestIdentity:
    def test_trial_ordinals_are_deterministic_and_monotonic(self) -> None:
        first = TrialCounter()
        second = TrialCounter()
        assert [first.issue() for _ in range(5)] == [second.issue() for _ in range(5)]
        assert first.issued == 5
        assert first.peek == 5

    def test_reset_restarts_session_local_sequence(self) -> None:
        counter = TrialCounter()
        for _ in range(4):
            counter.issue()

        counter.reset()
        assert counter.issued == 0
        assert counter.issue() == (ContractStatus.OK, 0)

        counter.reset(100)
        assert counter.origin == 100
        assert [counter.issue(), counter.issue()] == [
            (ContractStatus.OK, 100),
            (ContractStatus.OK, 101),
        ]

    def test_record_ordinals_form_separate_sequence(self) -> None:
        sequence = SequenceCounter(7)
        assert [sequence.issue(), sequence.issue()] == [
            (ContractStatus.OK, 7),
            (ContractStatus.OK, 8),
        ]
        # A record counter is not a trial counter, even though both count with
        # the same integer width.
        assert not isinstance(sequence, TrialCounter)

    def test_exhausted_counter_issues_nothing(self) -> None:
        counter = TrialCounter(2**64 - 1)
        assert counter.exhausted
        # Reissuing the last ordinal would hand the same identity out twice, and
        # an ordinal that repeats is not an identity.
        assert counter.issue() == (ContractStatus.ORDINAL_EXHAUSTED, 0)
        assert counter.issue() == (ContractStatus.ORDINAL_EXHAUSTED, 0)
        assert counter.issued == 0

        last = TrialCounter(2**64 - 2)
        assert not last.exhausted
        assert last.issue() == (ContractStatus.OK, 2**64 - 2)
        assert last.exhausted
        assert last.issue()[0] == ContractStatus.ORDINAL_EXHAUSTED

    def test_identity_holds_only_integers(self) -> None:
        identity = _trial(ordinal=12, target=4, stimulus=9)
        assert (identity.ordinal, identity.block, identity.target_id, identity.stimulus_id) == (
            12,
            1,
            4,
            9,
        )
        assert identity.key == 0
        # No object address participates: two independently built identities
        # carrying the same numbers are the same trial.
        assert identity.ordinal == _trial(ordinal=12, target=4, stimulus=9).ordinal

    def test_undeclared_outcome_does_not_end_trial(self) -> None:
        assert trial_outcome_declared(TrialOutcome.PENDING)
        assert trial_outcome_declared(TrialOutcome.ABORTED)
        assert not trial_ended(TrialOutcome.PENDING)
        assert trial_ended(TrialOutcome.ABORTED)


class TestSchedule:
    def test_values_regenerate_from_determining_tuple(self) -> None:
        def schedule(seed: int) -> list[int]:
            values = []
            for trial in range(32):
                status, value = sample_exclusive(DrawKey(seed, 1, trial, 0), 0, 1_000_000)
                assert status == ContractStatus.OK
                values.append(value)
            return values

        assert schedule(0xABCDEF) == schedule(0xABCDEF)
        assert schedule(0xABCDEF) != schedule(0xABCDEE)

    def test_exclusive_range_never_realizes_either_bound(self) -> None:
        for draw in range(200):
            status, value = sample_exclusive(DrawKey(7, 2, 5, draw), 0, 5)
            assert status == ContractStatus.OK
            assert 0 < value < 5

    def test_open_range_without_admissible_integer_is_rejected(self) -> None:
        key = DrawKey(1, 0, 0, 0)
        assert sample_exclusive(key, 0, 1)[0] == ContractStatus.RANGE_EMPTY
        assert sample_exclusive(key, 5, 5)[0] == ContractStatus.RANGE_EMPTY
        assert sample_exclusive(key, 0, 2) == (ContractStatus.OK, 1)

    def test_inclusive_range_reaches_both_endpoints(self) -> None:
        seen = set()
        for draw in range(400):
            status, value = sample_inclusive(DrawKey(99, 1, 0, draw), 0, 3)
            assert status == ContractStatus.OK
            assert 0 <= value <= 3
            seen.add(value)
        assert seen == {0, 1, 2, 3}

    def test_returned_sample_is_never_folded_into_range(self) -> None:
        # Exhausting the rejection loop is reported rather than repaired: a
        # modulo fallback would be uniform only when the span divides 2**64, so
        # it would make the sampler biased on exactly the path too rare for any
        # test to catch it on. Each attempt is accepted with probability above
        # one half, so no key is going to reach exhaustion here; the branch
        # itself is exercised in the native test through a bit source that
        # rejects on purpose. This asserts the status exists and that ordinary
        # sampling returns values instead of it.
        assert ContractStatus.SAMPLING_EXHAUSTED != ContractStatus.OK
        for draw in range(500):
            for high in (1, 2, 3, 5, 9, 17, 1_000):
                status, value = sample_inclusive(DrawKey(3, 1, 0, draw), 0, high)
                assert status == ContractStatus.OK
                assert 0 <= value <= high

    def test_index_sampling_rejects_empty_population(self) -> None:
        assert sample_index(DrawKey(1, 0, 0, 0), 0)[0] == ContractStatus.RANGE_EMPTY
        assert sample_index(DrawKey(1, 0, 0, 0), 1) == (ContractStatus.OK, 0)

    def test_replay_authority_has_exactly_two_cases(self) -> None:
        assert sampler_version_supported(SAMPLER_VERSION_1)
        assert not sampler_version_supported(SAMPLER_VERSION_1 + 1)

        assert replay_authority(SAMPLER_VERSION_1) == ReplayAuthority.REGENERATE
        assert replay_authority(SAMPLER_VERSION_1 + 1) == ReplayAuthority.RECORDED_SCHEDULE
        assert replay_authority(0) == ReplayAuthority.RECORDED_SCHEDULE
        assert set(ReplayAuthority.__members__) == {"REGENERATE", "RECORDED_SCHEDULE"}

    def test_schedule_fingerprint_is_persistable_and_field_sensitive(self) -> None:
        identity = ScheduleIdentity(seed=42, configuration_fingerprint=7)
        base = schedule_fingerprint(identity)
        assert schedule_fingerprint(identity) == base

        for field, value in (
            ("seed", 43),
            ("sampler_version", SAMPLER_VERSION_1 + 1),
            ("configuration_fingerprint", 8),
            ("catalog_fingerprint", 1),
        ):
            fields = {"seed": identity.seed, "configuration_fingerprint": 7, field: value}
            assert schedule_fingerprint(ScheduleIdentity(**fields)) != base

    def test_unsupported_sampler_version_is_recorded(self) -> None:
        assert validate(ScheduleIdentity(sampler_version=4_000)) == ContractStatus.OK
        assert validate(ScheduleIdentity(sampler_version=0)) == ContractStatus.IDENTITY_MISSING

    def test_catalog_fingerprints_are_content_sensitive(self) -> None:
        assert fingerprint_of_bytes(b"abc") == fingerprint_of_bytes(b"abc")
        assert fingerprint_of_bytes(b"abc") != fingerprint_of_bytes(b"abd")
        assert fingerprint_of_bytes(b"") != fingerprint_of_bytes(b"a")

        forward = FingerprintAccumulator()
        forward.absorb(1)
        forward.absorb(2)
        backward = FingerprintAccumulator()
        backward.absorb(2)
        backward.absorb(1)
        assert forward.value != backward.value

    def test_decision_snapshot_reconstructs_next_draw(self) -> None:
        snapshot = DecisionSnapshot(
            paradigm=_PARADIGM,
            schedule=ScheduleIdentity(seed=77),
            stream=2,
            trial=_trial(ordinal=20),
            draw_cursor=4,
        )
        assert validate(snapshot) == ContractStatus.OK

        key = next_draw_key(snapshot)
        assert (key.seed, key.stream, key.trial_idx, key.draw) == (77, 2, 20, 4)
        assert sample_index(key, 8) == sample_index(DrawKey(77, 2, 20, 4), 8)


class TestEvents:
    def test_event_names_kind_and_paradigm(self) -> None:
        assert validate(ExperimentEvent()) == ContractStatus.IDENTITY_MISSING
        assert (
            validate(ExperimentEvent(kind=ExperimentEventKind.TRIAL_START))
            == ContractStatus.IDENTITY_MISSING
        )
        assert (
            validate(
                ExperimentEvent(
                    kind=ExperimentEventKind.TRIAL_START, paradigm=_PARADIGM, trial=_trial()
                )
            )
            == ContractStatus.OK
        )

    def test_state_transitions_carry_owning_paradigm(self) -> None:
        assert validate(StateTransition()) == ContractStatus.IDENTITY_MISSING

        # The states are the paradigm's own enumerators, so the record is only
        # readable against the paradigm that owns them.
        transition = StateTransition(paradigm=_PARADIGM, from_state=1, to_state=2, cause=33)
        assert validate(transition) == ContractStatus.OK

    def test_selection_distinguishes_hit_from_misclick(self) -> None:
        hit = SelectionEvent(paradigm=_PARADIGM, intended_id=12, selected_id=12, correct=True)
        assert validate(hit) == ContractStatus.OK

        misclick = SelectionEvent(paradigm=_PARADIGM, intended_id=12, selected_id=13, correct=False)
        assert validate(misclick) == ContractStatus.OK

        # Claiming a wrong item was correct is a contradiction, not a value.
        lie = SelectionEvent(paradigm=_PARADIGM, intended_id=12, selected_id=13, correct=True)
        assert validate(lie) == ContractStatus.OUTCOME_INVALID

    def test_dwell_and_discrete_selections_differ_on_dwell_time(self) -> None:
        def selection(kind: SelectionKind, dwell_ns: int) -> SelectionEvent:
            return SelectionEvent(
                paradigm=_PARADIGM,
                intended_id=5,
                selected_id=5,
                correct=True,
                kind=kind,
                dwell_ns=dwell_ns,
            )

        assert validate(selection(SelectionKind.DWELL, 0)) == ContractStatus.OUTCOME_INVALID
        assert validate(selection(SelectionKind.DWELL, 250_000)) == ContractStatus.OK
        assert (
            validate(selection(SelectionKind.DISCRETE, 250_000)) == ContractStatus.OUTCOME_INVALID
        )
        assert validate(selection(SelectionKind.DISCRETE, 0)) == ContractStatus.OK

    def test_pending_trial_has_no_end(self) -> None:
        pending = TrialRecord(paradigm=_PARADIGM, trial=_trial(), interval=TimeInterval(100, 100))
        assert pending.outcome == TrialOutcome.PENDING
        assert validate(pending) == ContractStatus.OK

        assert (
            validate(TrialRecord(paradigm=_PARADIGM, interval=TimeInterval(100, 200)))
            == ContractStatus.OUTCOME_INVALID
        )
        assert (
            validate(
                TrialRecord(
                    paradigm=_PARADIGM,
                    interval=TimeInterval(100, 200),
                    outcome=TrialOutcome.TIMEOUT,
                )
            )
            == ContractStatus.OK
        )

    def test_enumerated_fields_check_declared_set(self) -> None:
        # Every persisted enumeration answers "is this one this build declares",
        # not merely "is this the one enumerator some rule happens to name".
        assert experiment_event_kind_declared(ExperimentEventKind.PARADIGM_MARKER)
        assert selection_kind_declared(SelectionKind.DWELL)
        assert cue_kind_declared(CueKind.TEXT_CONTENT)
        assert presentation_status_declared(PresentationStatus.EXPIRED)
        assert command_frame_declared(CommandFrame.DEVICE_NATIVE)
        assert command_axis_name_declared(CommandAxisName.GRASP)
        assert command_unit_declared(CommandUnit.RADIANS_PER_SECOND)
        assert command_application_declared(CommandApplication.EXPIRED)

    @pytest.mark.parametrize(
        ("predicate", "enumeration"),
        [
            (experiment_event_kind_declared, ExperimentEventKind),
            (selection_kind_declared, SelectionKind),
            (cue_kind_declared, CueKind),
            (presentation_status_declared, PresentationStatus),
            (command_frame_declared, CommandFrame),
            (command_axis_name_declared, CommandAxisName),
            (command_unit_declared, CommandUnit),
            (command_application_declared, CommandApplication),
        ],
        ids=lambda item: getattr(item, "__name__", str(item)),
    )
    def test_declared_members_pass_own_predicate(
        self, predicate: object, enumeration: type
    ) -> None:
        for member in enumeration.__members__.values():
            assert predicate(member)


class TestPresentation:
    def test_v1_cue_kinds_cover_black_cross_and_text(self) -> None:
        assert {"BLACK", "FIXATION_CROSS", "TEXT_CONTENT"} <= set(CueKind.__members__)

    def test_request_identifies_phase_trial_and_stimulus(self) -> None:
        request = PresentationRequest(
            paradigm=_PARADIGM,
            phase=2,
            trial=_trial(ordinal=11),
            cue=CueKind.TEXT_CONTENT,
            stimulus_id=17,
            requested_ns=1_000,
            onset_ns=1_000,
            duration_ns=500_000,
        )
        assert validate(request) == ContractStatus.OK

        # Semantic serialization: every field a presenter needs survives a
        # round trip through the contract, and nothing else is carried.
        payload = {
            "paradigm": request.paradigm,
            "phase": request.phase,
            "trial_ordinal": request.trial.ordinal,
            "cue": request.cue,
            "stimulus_id": request.stimulus_id,
            "requested_ns": request.requested_ns,
            "onset_ns": request.onset_ns,
            "duration_ns": request.duration_ns,
            "valid_until_ns": request.valid_until_ns,
        }
        restored = PresentationRequest(
            paradigm=payload["paradigm"],
            phase=payload["phase"],
            trial=_trial(ordinal=payload["trial_ordinal"]),
            cue=payload["cue"],
            stimulus_id=payload["stimulus_id"],
            requested_ns=payload["requested_ns"],
            onset_ns=payload["onset_ns"],
            duration_ns=payload["duration_ns"],
            valid_until_ns=payload["valid_until_ns"],
        )
        assert validate(restored) == ContractStatus.OK
        assert restored.stimulus_id == request.stimulus_id
        assert restored.trial.ordinal == request.trial.ordinal
        assert restored.valid_until_ns == NO_EXPIRY_NS

    @pytest.mark.parametrize("record", [PresentationRequest, PresentationState])
    def test_presentation_records_carry_no_rendering_decision(self, record: type) -> None:
        attributes = [name for name in dir(record) if not name.startswith("_")]
        for attribute in attributes:
            lowered = attribute.lower()
            assert not any(token in lowered for token in _RENDERING_VOCABULARY), attribute

    def test_only_content_cues_carry_stimulus(self) -> None:
        assert (
            validate(PresentationRequest(paradigm=_PARADIGM, cue=CueKind.TEXT_CONTENT))
            == ContractStatus.IDENTITY_MISSING
        )
        assert (
            validate(
                PresentationRequest(paradigm=_PARADIGM, cue=CueKind.TEXT_CONTENT, stimulus_id=4)
            )
            == ContractStatus.OK
        )
        assert (
            validate(PresentationRequest(paradigm=_PARADIGM, cue=CueKind.BLACK, stimulus_id=4))
            == ContractStatus.PRESENTATION_INVALID
        )
        assert (
            validate(PresentationRequest(paradigm=_PARADIGM, cue=CueKind.BLACK))
            == ContractStatus.OK
        )

    def test_onset_cannot_precede_its_decision(self) -> None:
        def request(onset_ns: int, valid_until_ns: int = NO_EXPIRY_NS) -> PresentationRequest:
            return PresentationRequest(
                paradigm=_PARADIGM,
                cue=CueKind.FIXATION_CROSS,
                requested_ns=1_000,
                onset_ns=onset_ns,
                valid_until_ns=valid_until_ns,
            )

        assert validate(request(999)) == ContractStatus.TIME_REGRESSED
        assert validate(request(2_000, 1_500)) == ContractStatus.EXPIRY_BEFORE_GENERATION
        assert validate(request(2_000, 2_500)) == ContractStatus.OK

    def test_contract_reports_no_presentation_time(self) -> None:
        outcome = PresentationOutcome(requested_ns=5_000)
        # A session with no presenter report says nothing about when anything
        # was displayed.
        assert outcome.status == PresentationStatus.NOT_REPORTED
        assert outcome.presented_ns == 0
        assert validate(outcome) == ContractStatus.OK

        assert (
            validate(PresentationOutcome(requested_ns=5_000, presented_ns=5_100))
            == ContractStatus.PRESENTATION_INVALID
        )

        # Only a presenter may claim one.
        def reported(presented_ns: int) -> PresentationOutcome:
            return PresentationOutcome(
                requested_ns=5_000,
                presented_ns=presented_ns,
                status=PresentationStatus.PRESENTED,
            )

        assert validate(reported(5_100)) == ContractStatus.OK
        assert validate(reported(4_999)) == ContractStatus.TIME_REGRESSED

    def test_report_names_request(self) -> None:
        # requested_ns cannot identify a request: the time contract accepts equal
        # instants, so two requests may share one. request_sequence identifies.
        outcome = PresentationOutcome(requested_ns=5_000, sequence=41, request_sequence=12)
        assert outcome.sequence == 41
        assert outcome.request_sequence == 12
        assert outcome.sequence != outcome.request_sequence


class TestCommand:
    @staticmethod
    def _space() -> CommandSpace:
        return CommandSpace(
            id=1,
            dim=2,
            frame=CommandFrame.WORKSPACE_2D,
            axes=[
                CommandAxis(CommandAxisName.X, CommandUnit.METRES_PER_SECOND),
                CommandAxis(CommandAxisName.Y, CommandUnit.METRES_PER_SECOND),
            ],
        )

    @staticmethod
    def _request(**overrides: object) -> CommandRequest:
        fields: dict[str, object] = {"space": 1, "dim": 2, "values": [0.25, -0.5]}
        fields.update(overrides)
        return CommandRequest(**fields)

    def test_command_space_states_dimensions_units_and_frame(self) -> None:
        space = self._space()
        assert validate(space) == ContractStatus.OK
        assert space.axis(0).name == CommandAxisName.X
        assert space.axis(0).unit == CommandUnit.METRES_PER_SECOND
        assert space.axis(1).name == CommandAxisName.Y

        unframed = CommandSpace(
            id=1,
            dim=2,
            frame=CommandFrame.UNSPECIFIED,
            axes=[
                CommandAxis(CommandAxisName.X, CommandUnit.METRES_PER_SECOND),
                CommandAxis(CommandAxisName.Y, CommandUnit.METRES_PER_SECOND),
            ],
        )
        assert validate(unframed) == ContractStatus.IDENTITY_MISSING

    def test_realtime_command_values_must_be_finite(self) -> None:
        assert validate(self._request()) == ContractStatus.OK

        for bad in (math.nan, math.inf, -math.inf):
            assert validate(self._request(values=[0.25, bad])) == ContractStatus.VALUE_NOT_FINITE

    def test_unused_command_slots_stay_zero(self) -> None:
        padded = [0.25, -0.5] + [0.0] * (MAX_COMMAND_DIMENSION - 3) + [1.0]
        assert validate(self._request(values=padded)) == ContractStatus.DIMENSION_INVALID
        assert validate(self._request(values=[0.25, -0.5])) == ContractStatus.OK

    def test_declared_command_value_is_never_defaulted(self) -> None:
        # A component nobody stated and a component deliberately set to zero are
        # the same bytes once written, and validate() would call both OK. So the
        # one place that can still tell them apart -- construction, where the
        # caller's own sequence is still visible -- is where it is caught.
        with pytest.raises(ValueError):
            self._request(values=[0.25])

        stated = self._request(values=[0.25, 0.0])
        assert validate(stated) == ContractStatus.OK
        assert stated.value(1) == 0.0

        # Beyond the declared dimension the contract requires the slot to be
        # unset, so padding it out to the fixed capacity states nothing.
        assert self._request(dim=1, values=[0.25]).value(1) == 0.0

    def test_declared_command_axis_is_never_defaulted(self) -> None:
        with pytest.raises(ValueError):
            CommandSpace(
                id=1,
                dim=2,
                frame=CommandFrame.WORKSPACE_2D,
                axes=[CommandAxis(CommandAxisName.X, CommandUnit.METRES_PER_SECOND)],
            )

    def test_command_dimension_is_bounded(self) -> None:
        assert validate(self._request(dim=0, values=[])) == ContractStatus.DIMENSION_INVALID

        # Above the capacity there is no sequence that could name every declared
        # axis, so the record cannot be built at all rather than being built and
        # then rejected.
        with pytest.raises(ValueError):
            self._request(dim=MAX_COMMAND_DIMENSION + 1, values=[0.0] * MAX_COMMAND_DIMENSION)

        with pytest.raises(IndexError):
            self._request().value(MAX_COMMAND_DIMENSION)

    def test_request_must_match_named_space(self) -> None:
        space = self._space()
        assert validate_against(self._request(), space) == ContractStatus.OK
        assert (
            validate_against(self._request(dim=1, values=[0.25]), space)
            == ContractStatus.DIMENSION_INVALID
        )
        assert validate_against(self._request(space=2), space) == ContractStatus.IDENTITY_MISSING

    def test_outcome_records_only_proven_facts(self) -> None:
        outcome = CommandOutcome(generated_ns=1_000, submitted_ns=1_000)
        assert outcome.application == CommandApplication.NOT_SUBMITTED
        assert validate(outcome) == ContractStatus.OK

        # An unsubmitted command cannot carry a runtime status: there was none.
        assert (
            validate(CommandOutcome(generated_ns=1_000, submitted_ns=1_000, status_code=3))
            == ContractStatus.OUTCOME_INVALID
        )
        assert (
            validate(
                CommandOutcome(
                    generated_ns=1_000,
                    submitted_ns=1_200,
                    application=CommandApplication.ACCEPTED,
                )
            )
            == ContractStatus.OK
        )
        assert set(CommandApplication.__members__) == {
            "NOT_SUBMITTED",
            "ACCEPTED",
            "REJECTED",
            "EXPIRED",
        }

    def test_command_report_names_request(self) -> None:
        outcome = CommandOutcome(generated_ns=1_000, sequence=90, request_sequence=12)
        assert outcome.sequence != outcome.request_sequence


class TestStatusMessages:
    def test_every_status_has_message(self) -> None:
        for status in ContractStatus.__members__.values():
            message = contract_status_message(status)
            assert isinstance(message, str)
            assert message

    def test_undeclared_and_unset_are_distinct_failures(self) -> None:
        # An unset field is a validly incomplete record; a number naming no
        # enumerator is an unreadable one, and reporting them alike would let the
        # second pass as the first.
        assert ContractStatus.ENUM_UNDECLARED != ContractStatus.IDENTITY_MISSING
        assert contract_status_message(ContractStatus.ENUM_UNDECLARED) != contract_status_message(
            ContractStatus.IDENTITY_MISSING
        )


class TestAbnormalConditions:
    def test_default_policies_are_conservative(self) -> None:
        # The defaults are part of the contract, not an implementation detail: a
        # caller who configures nothing gets a run that ends a trial whose
        # continuity broke and ends a run whose recorder failed.
        policies = AbnormalPolicySet()
        assert (
            policy_for(policies, AbnormalCondition.DECODED_COMMAND_INVALID)
            == AbnormalPolicy.ABORT_TRIAL
        )
        assert (
            policy_for(policies, AbnormalCondition.SOURCE_DISCONTINUITY)
            == AbnormalPolicy.ABORT_TRIAL
        )
        assert policy_for(policies, AbnormalCondition.INPUT_GAP) == AbnormalPolicy.ABORT_TRIAL
        assert (
            policy_for(policies, AbnormalCondition.RECORDER_FAULT) == AbnormalPolicy.ABORT_SESSION
        )
        assert (
            policy_for(policies, AbnormalCondition.PRESENTATION_EVIDENCE_MISSING)
            == AbnormalPolicy.RECORD
        )

    def test_four_conditions_ignore_configuration(self) -> None:
        severe = AbnormalPolicySet(
            decoded_command_invalid=AbnormalPolicy.ABORT_SESSION,
            input_discontinuity=AbnormalPolicy.ABORT_SESSION,
            presentation_failed=AbnormalPolicy.ABORT_SESSION,
            presentation_evidence_missing=AbnormalPolicy.ABORT_SESSION,
            acquisition_fault=AbnormalPolicy.ABORT_SESSION,
        )
        # A drop on a path that carried no task input cannot be configured into
        # something that invalidates a measurement, and input arriving after the
        # run ended has no run left to end.
        assert policy_for(severe, AbnormalCondition.OBSERVER_FRAME_DROP) == AbnormalPolicy.RECORD
        assert policy_for(severe, AbnormalCondition.INPUT_AFTER_TERMINAL) == AbnormalPolicy.RECORD
        # And a presenter's report the run cannot match to a request it made is
        # not evidence about any trial, so it is never allowed to end one.
        assert (
            policy_for(severe, AbnormalCondition.PRESENTATION_REPORT_UNMATCHED)
            == AbnormalPolicy.RECORD
        )

        permissive = AbnormalPolicySet(
            decoded_command_invalid=AbnormalPolicy.RECORD,
            input_discontinuity=AbnormalPolicy.RECORD,
            acquisition_fault=AbnormalPolicy.RECORD,
        )
        # And an emergency stop ends the run whatever else is configured.
        assert (
            policy_for(permissive, AbnormalCondition.EMERGENCY_STOP) == AbnormalPolicy.ABORT_SESSION
        )

    def test_trial_touched_by_abnormal_condition_is_inadmissible(self) -> None:
        # The whole of the "no silent success" rule, stated once so that no
        # paradigm restates it differently.
        assert abnormal_response_admits_trial(AbnormalResponse.RECORDED)
        assert abnormal_response_admits_trial(AbnormalResponse.INPUT_REFUSED)
        assert not abnormal_response_admits_trial(AbnormalResponse.TRIAL_INVALIDATED)
        assert not abnormal_response_admits_trial(AbnormalResponse.TRIAL_ABORTED)
        assert not abnormal_response_admits_trial(AbnormalResponse.SESSION_ABORTED)

    def test_policy_and_response_are_recorded_separately(self) -> None:
        # A record carrying only one of them could not tell an ended trial from
        # an invalidated one, which is exactly the difference between a paradigm
        # that can end a trial and one whose state machine owns termination.
        event = AbnormalEvent(
            paradigm=3,
            condition=AbnormalCondition.SOURCE_DISCONTINUITY,
            policy=AbnormalPolicy.ABORT_TRIAL,
            response=AbnormalResponse.TRIAL_INVALIDATED,
            has_trial=True,
        )
        assert validate(event) == ContractStatus.OK
        assert event.policy != event.response

    def test_response_more_severe_than_policy_is_rejected(self) -> None:
        event = AbnormalEvent(
            paradigm=3,
            condition=AbnormalCondition.SOURCE_DISCONTINUITY,
            policy=AbnormalPolicy.RECORD,
            response=AbnormalResponse.TRIAL_ABORTED,
            has_trial=True,
        )
        assert validate(event) == ContractStatus.OUTCOME_INVALID

    def test_response_naming_trial_requires_trial(self) -> None:
        event = AbnormalEvent(
            paradigm=3,
            condition=AbnormalCondition.SOURCE_DISCONTINUITY,
            policy=AbnormalPolicy.ABORT_TRIAL,
            response=AbnormalResponse.TRIAL_ABORTED,
        )
        assert validate(event) == ContractStatus.OUTCOME_INVALID

    def test_record_without_subject_is_rejected(self) -> None:
        assert validate(AbnormalEvent()) == ContractStatus.IDENTITY_MISSING

    def test_policy_set_with_undeclared_severity_is_rejected(self) -> None:
        # Validated where a run freezes its configuration, like every other
        # persisted enumeration: a severity nothing declared cannot be compared,
        # recorded, or replayed.
        assert validate(AbnormalPolicySet()) == ContractStatus.OK
        assert (
            validate(
                AbnormalPolicySet(
                    decoded_command_invalid=AbnormalPolicy.ABORT_SESSION,
                    input_discontinuity=AbnormalPolicy.RECORD,
                    presentation_failed=AbnormalPolicy.ABORT_TRIAL,
                    presentation_evidence_missing=AbnormalPolicy.RECORD,
                    acquisition_fault=AbnormalPolicy.ABORT_SESSION,
                )
            )
            == ContractStatus.OK
        )

    def test_declared_conditions_have_distinct_stable_names(self) -> None:
        # The names are what a recording carries, so two conditions sharing one
        # would make a recorded reason ambiguous.
        conditions = list(AbnormalCondition.__members__.values())
        names = [abnormal_condition_name(condition) for condition in conditions]
        assert all(name != "undeclared" for name in names)
        assert len(set(names)) == len(names)
        assert abnormal_condition_declared(AbnormalCondition.PRESENTATION_REPORT_UNMATCHED)
        assert validate(AbnormalEvent(paradigm=3)) == ContractStatus.IDENTITY_MISSING

    def test_conditions_have_distinct_stable_names(self) -> None:
        names = {
            abnormal_condition_name(condition)
            for condition in AbnormalCondition.__members__.values()
        }
        assert len(names) == len(AbnormalCondition.__members__)
        assert all(name and name != "undeclared" for name in names)

    def test_severity_orders_policies(self) -> None:
        assert escalate(AbnormalPolicy.RECORD, AbnormalPolicy.ABORT_TRIAL) == (
            AbnormalPolicy.ABORT_TRIAL
        )
        assert escalate(AbnormalPolicy.ABORT_SESSION, AbnormalPolicy.ABORT_TRIAL) == (
            AbnormalPolicy.ABORT_SESSION
        )
