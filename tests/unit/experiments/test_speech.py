#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Speech cue configuration, stimulus catalog, and per-trial timing schedule.

The realized durations are not restated from the implementation. ``_sample``
below is an independent transcription of the sampler the schedule contract
specifies -- the three-round mixing function, the counter-slot addressing, and
the masked-rejection mapping -- written from that specification in Python
integer arithmetic, and every duration the native schedule produces is checked
against it. A constant that moved on one side and not the other has nowhere to
hide, and because the transcription is pure Python it is also the cross-platform
reproduction claim: a bounded mapping that agreed with the C++ only because both
called the same standard-library distribution would prove nothing.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from neurale.experiments import (
    DRAW_SLOT_STRIDE,
    UNSET_STIMULUS_ID,
    ContractStatus,
    CueKind,
    ReplayAuthority,
    TrialIdentity,
    speech,
)

# ---------------------------------------------------------------------------
# Constants

MASK64 = (1 << 64) - 1

T1 = 500_000_000
T2 = 300_000_000
T3 = 2_000_000_000
SEED = 0xC0FFEE1234567890
STIMULI = [11, 22, 33, 44]


# ---------------------------------------------------------------------------
# An independent transcription of the specified sampler


def _mix64(value: int) -> int:
    z = (value + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def _slot_bits(seed: int, stream: int, trial: int, draw: int, slot: int) -> int:
    state = _mix64(seed ^ 0xA0761D6478BD642F)
    state = _mix64((state + trial) & MASK64)
    state = _mix64(state ^ ((stream * 0x9E3779B97F4A7C15) & MASK64))
    return _mix64((state + draw * DRAW_SLOT_STRIDE + slot) & MASK64)


def _sample_inclusive(seed: int, stream: int, trial: int, draw: int, low: int, high: int) -> int:
    span = high - low
    mask = span
    for shift in (1, 2, 4, 8, 16, 32):
        mask |= mask >> shift
    for attempt in range(64):
        candidate = _slot_bits(seed, stream, trial, draw, attempt) & mask
        if candidate <= span:
            return low + candidate
    raise AssertionError("the transcription exhausted its attempts")


def _sample(seed: int, stream: int, trial: int, draw: int, bound: int) -> int:
    """The open range ``(0, bound)``, which is the closed ``[1, bound - 1]``."""
    return _sample_inclusive(seed, stream, trial, draw, 1, bound - 1)


def _reference_durations(config: speech.SpeechCueConfig, ordinal: int) -> tuple[int, int, int]:
    stream = speech.TRIAL_TIMING_STREAM
    black = _sample(config.seed, stream, ordinal, speech.BLACK_DRAW, config.black_bound_ns)
    cross = (
        _sample(config.seed, stream, ordinal, speech.CROSS_DRAW, config.cross_bound_ns)
        if config.cross_enabled
        else 0
    )
    content = _sample(config.seed, stream, ordinal, speech.CONTENT_DRAW, config.content_bound_ns)
    return black, cross, content


def _reference_stimulus(config: speech.SpeechCueConfig, ordinal: int) -> int:
    stimuli = list(config.stimuli)
    count = len(stimuli)
    stream = speech.STIMULUS_ORDER_STREAM
    if config.stimulus_order == speech.StimulusOrderPolicy.SEQUENTIAL:
        return stimuli[ordinal % count]
    if config.stimulus_order == speech.StimulusOrderPolicy.RANDOM_WITH_REPLACEMENT:
        return stimuli[_sample_inclusive(config.seed, stream, ordinal, 0, 0, count - 1)]
    block, pos = divmod(ordinal, count)
    order = list(stimuli)
    draw = 0
    for i in range(count - 1, 0, -1):
        chosen = _sample_inclusive(config.seed, stream, block, draw, 0, i)
        order[i], order[chosen] = order[chosen], order[i]
        draw += 1
    return order[pos]


# ---------------------------------------------------------------------------
# Fixtures


def _catalog(count: int = 4, ids: list[int] | None = None) -> speech.SpeechCatalog:
    identifiers = ids if ids is not None else STIMULI[:count]
    return speech.SpeechCatalog(
        [
            speech.SpeechStimulus(
                id=identifier,
                content=speech.SpeechContentKind.TEXT,
                text=f"prompt-{idx}",
                label=idx % 3 + 1,
                metadata=idx,
            )
            for idx, identifier in enumerate(identifiers)
        ]
    )


def _seeded(**overrides: object) -> speech.SpeechCueConfig:
    values: dict[str, object] = {
        "black_bound_ns": T1,
        "cross_bound_ns": T2,
        "content_bound_ns": T3,
        "cross_enabled": True,
        "inter_trial_ns": 0,
        "schedule": speech.SpeechScheduleKind.SEEDED,
        "stimulus_order": speech.StimulusOrderPolicy.SEQUENTIAL,
        "seed": SEED,
        "n_trials": 16,
        "stimuli": STIMULI,
    }
    values.update(overrides)
    return speech.SpeechCueConfig(**values)


def _frozen(seeded: speech.SpeechCueConfig) -> speech.SpeechCueConfig:
    entries = []
    for ordinal in range(seeded.n_trials):
        status, schedule = speech.prepare_trial(seeded, ordinal)
        assert status == ContractStatus.OK
        entries.append(schedule)
    return speech.SpeechCueConfig(
        black_bound_ns=seeded.black_bound_ns,
        cross_bound_ns=seeded.cross_bound_ns,
        content_bound_ns=seeded.content_bound_ns,
        cross_enabled=seeded.cross_enabled,
        inter_trial_ns=seeded.inter_trial_ns,
        schedule=speech.SpeechScheduleKind.EXPLICIT_SEQUENCE,
        stimulus_order=speech.StimulusOrderPolicy.UNSPECIFIED,
        seed=seeded.seed,
        sampler_version=seeded.sampler_version,
        n_trials=seeded.n_trials,
        stimuli=[],
        explicit_schedule=entries,
    )


def _edited(schedule: speech.SpeechTrialSchedule, **changes: object) -> speech.SpeechTrialSchedule:
    values: dict[str, object] = {
        "ordinal": schedule.ordinal,
        "stimulus_id": schedule.stimulus_id,
        "cross_enabled": schedule.cross_enabled,
        "black_duration_ns": schedule.black_duration_ns,
        "cross_duration_ns": schedule.cross_duration_ns,
        "content_duration_ns": schedule.content_duration_ns,
        "sampler_version": schedule.sampler_version,
    }
    values.update(changes)
    return speech.SpeechTrialSchedule(**values)


def _fields(schedule: speech.SpeechTrialSchedule) -> tuple[object, ...]:
    return (
        schedule.ordinal,
        schedule.black_duration_ns,
        schedule.cross_duration_ns,
        schedule.content_duration_ns,
        schedule.stimulus_id,
        schedule.sampler_version,
        schedule.cross_enabled,
    )


# ---------------------------------------------------------------------------


class TestSampledDurations:
    def test_durations_lie_strictly_inside_bound(self) -> None:
        config = _seeded(n_trials=256)
        assert speech.validate(config) == ContractStatus.OK
        for ordinal in range(config.n_trials):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            # Open at both ends, on every trial: never zero, never the bound.
            assert 0 < schedule.black_duration_ns < T1
            assert 0 < schedule.cross_duration_ns < T2
            assert 0 < schedule.content_duration_ns < T3
            assert speech.validate_against(schedule, config) == ContractStatus.OK

    def test_durations_match_transcribed_sampler(self) -> None:
        config = _seeded(n_trials=64)
        for ordinal in range(config.n_trials):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            expected = _reference_durations(config, ordinal)
            assert (
                schedule.black_duration_ns,
                schedule.cross_duration_ns,
                schedule.content_duration_ns,
            ) == expected

    def test_narrowest_bound_forces_single_value(self) -> None:
        config = _seeded(black_bound_ns=2, cross_bound_ns=2, content_bound_ns=2, n_trials=4)
        assert speech.validate(config) == ContractStatus.OK
        for ordinal in range(config.n_trials):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            # (0, 2) holds exactly one integer, so the mapping is pinned rather
            # than merely constrained.
            assert schedule.black_duration_ns == 1
            assert schedule.cross_duration_ns == 1
            assert schedule.content_duration_ns == 1

    def test_range_is_used_not_merely_respected(self) -> None:
        config = _seeded(
            stimulus_order=speech.StimulusOrderPolicy.RANDOM_WITH_REPLACEMENT, n_trials=512
        )
        realized = []
        for ordinal in range(config.n_trials):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            realized.append(schedule.black_duration_ns)
        # Not a uniformity test -- the sampler's own suite owns that -- but a
        # constant sampler passes every bound assertion above.
        assert min(realized) < T1 // 4
        assert max(realized) > T1 * 3 // 4
        assert len(set(realized)) > config.n_trials // 2


class TestCrossIsConfiguration:
    def test_disabled_cross_is_zero_and_absent_from_timeline(self) -> None:
        config = _seeded(cross_enabled=False, cross_bound_ns=0, n_trials=32)
        assert speech.validate(config) == ContractStatus.OK
        for ordinal in range(config.n_trials):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            assert schedule.cross_enabled is False
            # Unambiguous precisely because the enabled case cannot produce it.
            assert schedule.cross_duration_ns == 0

            status, timeline = speech.build_timeline(config, schedule, 1_000)
            assert status == ContractStatus.OK
            assert timeline.count == 2
            assert [phase.phase for phase in timeline.phases] == [
                speech.SpeechPhase.BLACK,
                speech.SpeechPhase.CONTENT,
            ]
            assert speech.SpeechPhase.CROSS not in [phase.phase for phase in timeline.phases]

    def test_enabled_cross_is_strictly_positive_on_every_trial(self) -> None:
        config = _seeded(n_trials=256)
        for ordinal in range(config.n_trials):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            assert schedule.cross_enabled is True
            assert schedule.cross_duration_ns > 0
            status, timeline = speech.build_timeline(config, schedule, 0)
            assert status == ContractStatus.OK
            assert timeline.count == 3
            assert timeline.phases[1].phase == speech.SpeechPhase.CROSS
            assert timeline.phases[1].cue == CueKind.FIXATION_CROSS
            assert timeline.phases[1].interval.duration_ns == schedule.cross_duration_ns

    def test_toggling_cross_rerolls_nothing_else(self) -> None:
        enabled = _seeded(n_trials=64)
        disabled = _seeded(n_trials=64, cross_enabled=False, cross_bound_ns=0)
        for ordinal in range(enabled.n_trials):
            _, with_cross = speech.prepare_trial(enabled, ordinal)
            _, without_cross = speech.prepare_trial(disabled, ordinal)
            # The draw ordinals are coordinates, not a running counter.
            assert with_cross.black_duration_ns == without_cross.black_duration_ns
            assert with_cross.content_duration_ns == without_cross.content_duration_ns
            assert with_cross.stimulus_id == without_cross.stimulus_id

    def test_disabled_cross_leaves_no_inert_bound(self) -> None:
        inert = _seeded(cross_enabled=False)
        assert inert.cross_bound_ns == T2
        # A bound naming a domain nothing samples is a field no reader can
        # interpret, so it is rejected rather than ignored.
        assert speech.validate(inert) == ContractStatus.OUTCOME_INVALID
        assert speech.validate(_seeded(cross_enabled=False, cross_bound_ns=0)) == ContractStatus.OK


class TestTimeline:
    def test_phases_are_contiguous_and_half_open(self) -> None:
        config = _seeded(inter_trial_ns=250_000_000)
        status, schedule = speech.prepare_trial(config, 0)
        assert status == ContractStatus.OK
        start = 7_000_000_000
        status, timeline = speech.build_timeline(config, schedule, start)
        assert status == ContractStatus.OK

        assert timeline.count == 4
        assert [phase.phase for phase in timeline.phases] == [
            speech.SpeechPhase.BLACK,
            speech.SpeechPhase.CROSS,
            speech.SpeechPhase.CONTENT,
            speech.SpeechPhase.INTER_TRIAL,
        ]
        assert [phase.cue for phase in timeline.phases] == [
            CueKind.BLACK,
            CueKind.FIXATION_CROSS,
            CueKind.TEXT_CONTENT,
            CueKind.NONE,
        ]
        assert timeline.phases[2].stimulus_id == schedule.stimulus_id
        assert timeline.phases[0].stimulus_id == UNSET_STIMULUS_ID

        assert timeline.phases[0].interval.start_ns == start
        for earlier, later in pairwise(timeline.phases):
            assert later.interval.start_ns == earlier.interval.end_ns
            # The exact boundary belongs to the phase that starts there, and to
            # nothing else.
            assert not earlier.interval.contains(later.interval.start_ns)
            assert later.interval.contains(later.interval.start_ns)
            assert earlier.interval.elapsed_at(later.interval.start_ns)

    def test_gap_is_not_part_of_trial(self) -> None:
        config = _seeded(inter_trial_ns=250_000_000)
        _, schedule = speech.prepare_trial(config, 1)
        status, timeline = speech.build_timeline(config, schedule, 0)
        assert status == ContractStatus.OK

        status, length = speech.trial_duration(schedule)
        assert status == ContractStatus.OK
        assert length == (
            schedule.black_duration_ns + schedule.cross_duration_ns + schedule.content_duration_ns
        )
        assert timeline.trial.duration_ns == length
        assert timeline.trial.end_ns == timeline.phases[2].interval.end_ns
        assert timeline.next_trial_start_ns == timeline.trial.end_ns + config.inter_trial_ns

    def test_zero_gap_is_absent_not_empty(self) -> None:
        config = _seeded(inter_trial_ns=0)
        _, schedule = speech.prepare_trial(config, 1)
        status, timeline = speech.build_timeline(config, schedule, 0)
        assert status == ContractStatus.OK
        assert timeline.count == 3
        assert speech.SpeechPhase.INTER_TRIAL not in [phase.phase for phase in timeline.phases]
        assert timeline.next_trial_start_ns == timeline.trial.end_ns

    def test_chaining_trials_reproduces_session_clock(self) -> None:
        config = _seeded(n_trials=8, inter_trial_ns=1_000_000)
        cursor = 5_000
        expected = 5_000
        for ordinal in range(config.n_trials):
            _, schedule = speech.prepare_trial(config, ordinal)
            status, timeline = speech.build_timeline(config, schedule, cursor)
            assert status == ContractStatus.OK
            assert timeline.trial.start_ns == expected
            expected += (
                schedule.black_duration_ns
                + schedule.cross_duration_ns
                + schedule.content_duration_ns
                + config.inter_trial_ns
            )
            cursor = timeline.next_trial_start_ns
        assert cursor == expected

    def test_overflow_is_reported_not_wrapped(self) -> None:
        config = _seeded()
        _, schedule = speech.prepare_trial(config, 0)
        status, _ = speech.build_timeline(config, schedule, (1 << 64) - 11)
        assert status == ContractStatus.DURATION_OVERFLOW


class TestDeterminism:
    def test_same_seed_reproduces_same_schedule(self) -> None:
        config = _seeded(n_trials=64, stimulus_order=speech.StimulusOrderPolicy.SHUFFLED_BLOCKS)
        first = [speech.prepare_trial(config, ordinal)[1] for ordinal in range(64)]
        again = [
            speech.prepare_trial(
                _seeded(n_trials=64, stimulus_order=speech.StimulusOrderPolicy.SHUFFLED_BLOCKS),
                ordinal,
            )[1]
            for ordinal in range(64)
        ]
        assert [_fields(entry) for entry in first] == [_fields(entry) for entry in again]

    def test_different_seed_diverges(self) -> None:
        config = _seeded(n_trials=64)
        other = _seeded(n_trials=64, seed=SEED ^ 1)
        mine = [_fields(speech.prepare_trial(config, ordinal)[1]) for ordinal in range(64)]
        theirs = [_fields(speech.prepare_trial(other, ordinal)[1]) for ordinal in range(64)]
        assert mine != theirs

    def test_trial_does_not_depend_on_earlier_trials(self) -> None:
        config = _seeded(n_trials=64)
        _, alone = speech.prepare_trial(config, 40)
        for ordinal in reversed(range(config.n_trials)):
            speech.prepare_trial(config, ordinal)
        for ordinal in range(config.n_trials):
            speech.prepare_trial(config, ordinal)
        _, after = speech.prepare_trial(config, 40)
        # Addressed, not advanced: a sparse caller and a dense one see the same
        # values, whatever order they ask in.
        assert _fields(alone) == _fields(after)

    def test_ordinal_past_session_is_rejected(self) -> None:
        config = _seeded(n_trials=4)
        assert speech.prepare_trial(config, 4)[0] == ContractStatus.RANGE_EMPTY
        assert speech.select_stimulus(config, 4)[0] == ContractStatus.RANGE_EMPTY
        assert speech.prepare_trial(config, 3)[0] == ContractStatus.OK

    def test_preparation_does_not_mutate_inputs(self) -> None:
        config = _seeded(n_trials=8)
        before = (
            config.black_bound_ns,
            config.cross_bound_ns,
            config.content_bound_ns,
            config.cross_enabled,
            config.seed,
            config.sampler_version,
            config.n_trials,
            list(config.stimuli),
            config.stimulus_order,
        )
        for ordinal in range(config.n_trials):
            _, schedule = speech.prepare_trial(config, ordinal)
            speech.build_timeline(config, schedule, 0)
            speech.select_stimulus(config, ordinal)
        assert before == (
            config.black_bound_ns,
            config.cross_bound_ns,
            config.content_bound_ns,
            config.cross_enabled,
            config.seed,
            config.sampler_version,
            config.n_trials,
            list(config.stimuli),
            config.stimulus_order,
        )

    def test_records_cannot_be_edited_after_decision(self) -> None:
        config = _seeded()
        _, schedule = speech.prepare_trial(config, 0)
        with pytest.raises(AttributeError):
            schedule.cross_duration_ns = 0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            config.cross_enabled = False  # type: ignore[misc]


class TestStimulusOrder:
    def test_sequential_sampler_steps_and_wraps(self) -> None:
        config = _seeded(n_trials=4 * len(STIMULI))
        for ordinal in range(config.n_trials):
            status, stimulus_id = speech.select_stimulus(config, ordinal)
            assert status == ContractStatus.OK
            assert stimulus_id == STIMULI[ordinal % len(STIMULI)]
            assert stimulus_id == _reference_stimulus(config, ordinal)

    def test_with_replacement_covers_set_and_may_repeat(self) -> None:
        config = _seeded(
            n_trials=400, stimulus_order=speech.StimulusOrderPolicy.RANDOM_WITH_REPLACEMENT
        )
        drawn = []
        for ordinal in range(config.n_trials):
            status, stimulus_id = speech.select_stimulus(config, ordinal)
            assert status == ContractStatus.OK
            assert stimulus_id == _reference_stimulus(config, ordinal)
            drawn.append(stimulus_id)
        assert set(drawn) == set(STIMULI)
        # With replacement really does mean with replacement.
        assert any(left == right for left, right in pairwise(drawn))

    def test_shuffled_blocks_are_balanced_at_boundaries(self) -> None:
        config = _seeded(
            n_trials=8 * len(STIMULI),
            stimulus_order=speech.StimulusOrderPolicy.SHUFFLED_BLOCKS,
        )
        reordered = False
        for block in range(8):
            within = []
            for i in range(len(STIMULI)):
                ordinal = block * len(STIMULI) + i
                status, stimulus_id = speech.select_stimulus(config, ordinal)
                assert status == ContractStatus.OK
                assert stimulus_id == _reference_stimulus(config, ordinal)
                within.append(stimulus_id)
                if stimulus_id != STIMULI[i]:
                    reordered = True
            assert sorted(within) == sorted(STIMULI)
        assert reordered

    def test_schedule_stimulus_is_selected_stimulus(self) -> None:
        for policy in (
            speech.StimulusOrderPolicy.SEQUENTIAL,
            speech.StimulusOrderPolicy.RANDOM_WITH_REPLACEMENT,
            speech.StimulusOrderPolicy.SHUFFLED_BLOCKS,
        ):
            config = _seeded(n_trials=16, stimulus_order=policy)
            for ordinal in range(config.n_trials):
                _, schedule = speech.prepare_trial(config, ordinal)
                assert schedule.stimulus_id == speech.select_stimulus(config, ordinal)[1]


class TestExplicitSchedule:
    def test_frozen_schedule_reproduces_seeded_schedule(self) -> None:
        seeded = _seeded(n_trials=32, stimulus_order=speech.StimulusOrderPolicy.SHUFFLED_BLOCKS)
        frozen = _frozen(seeded)
        assert speech.validate(frozen) == ContractStatus.OK
        assert frozen.n_explicit == seeded.n_trials
        for ordinal in range(seeded.n_trials):
            _, regenerated = speech.prepare_trial(seeded, ordinal)
            status, recorded = speech.prepare_trial(frozen, ordinal)
            assert status == ContractStatus.OK
            assert _fields(regenerated) == _fields(recorded)
            assert speech.select_stimulus(frozen, ordinal)[1] == regenerated.stimulus_id

    def test_replay_authority_names_deciding_artefact(self) -> None:
        seeded = _seeded()
        assert speech.replay_authority(seeded) == ReplayAuthority.REGENERATE
        frozen = _frozen(_seeded(n_trials=4))
        # Nothing to regenerate, whatever the sampler version says.
        assert speech.replay_authority(frozen) == ReplayAuthority.RECORDED_SCHEDULE

    def test_unknown_sampler_replays_without_regenerating(self) -> None:
        future = _seeded(n_trials=4, sampler_version=4242)
        assert speech.validate(future) == ContractStatus.OK
        assert speech.prepare_trial(future, 0)[0] == ContractStatus.VERSION_UNSUPPORTED
        frozen = speech.SpeechCueConfig(
            black_bound_ns=T1,
            cross_bound_ns=T2,
            content_bound_ns=T3,
            cross_enabled=True,
            schedule=speech.SpeechScheduleKind.EXPLICIT_SEQUENCE,
            seed=SEED,
            sampler_version=4242,
            n_trials=1,
            explicit_schedule=[
                speech.SpeechTrialSchedule(
                    ordinal=0,
                    stimulus_id=11,
                    cross_enabled=True,
                    black_duration_ns=1_000,
                    cross_duration_ns=2_000,
                    content_duration_ns=3_000,
                    sampler_version=4242,
                )
            ],
        )
        assert speech.validate(frozen) == ContractStatus.OK
        assert speech.replay_authority(frozen) == ReplayAuthority.RECORDED_SCHEDULE
        status, schedule = speech.prepare_trial(frozen, 0)
        assert status == ContractStatus.OK
        assert schedule.content_duration_ns == 3_000

    def test_supplied_schedule_is_checked_not_trusted(self) -> None:
        seeded = _seeded(n_trials=4)
        frozen = _frozen(seeded)
        entries = list(frozen.explicit_schedule)

        def rebuilt(idx: int, **changes: object) -> speech.SpeechCueConfig:
            replaced = list(entries)
            original = entries[idx]
            values: dict[str, object] = {
                "ordinal": original.ordinal,
                "stimulus_id": original.stimulus_id,
                "cross_enabled": original.cross_enabled,
                "black_duration_ns": original.black_duration_ns,
                "cross_duration_ns": original.cross_duration_ns,
                "content_duration_ns": original.content_duration_ns,
                "sampler_version": original.sampler_version,
            }
            values.update(changes)
            replaced[idx] = speech.SpeechTrialSchedule(**values)
            return speech.SpeechCueConfig(
                black_bound_ns=frozen.black_bound_ns,
                cross_bound_ns=frozen.cross_bound_ns,
                content_bound_ns=frozen.content_bound_ns,
                cross_enabled=frozen.cross_enabled,
                schedule=speech.SpeechScheduleKind.EXPLICIT_SEQUENCE,
                seed=frozen.seed,
                n_trials=frozen.n_trials,
                explicit_schedule=replaced,
            )

        assert speech.validate(rebuilt(1, cross_duration_ns=0)) == ContractStatus.OUTCOME_INVALID
        assert speech.validate(rebuilt(1, ordinal=9)) == ContractStatus.OUTCOME_INVALID
        assert (
            speech.validate(rebuilt(0, black_duration_ns=T1))
            == ContractStatus.PARAMETER_OUT_OF_RANGE
        )
        assert (
            speech.validate(rebuilt(0, black_duration_ns=0))
            == ContractStatus.PARAMETER_OUT_OF_RANGE
        )
        assert (
            speech.validate(rebuilt(2, content_duration_ns=T3 + 1))
            == ContractStatus.PARAMETER_OUT_OF_RANGE
        )
        assert (
            speech.validate(rebuilt(2, stimulus_id=UNSET_STIMULUS_ID))
            == ContractStatus.IDENTITY_MISSING
        )
        assert speech.validate(rebuilt(2, sampler_version=0)) == ContractStatus.IDENTITY_MISSING
        # A version the configuration does not share. The session identity takes
        # its sampler from the configuration, so this is not an entry recorded
        # under an older build -- it is one record of a session disagreeing with
        # the session about what produced it.
        assert speech.validate(rebuilt(2, sampler_version=4242)) == ContractStatus.OUTCOME_INVALID

    def test_explicit_cross_must_match_configuration(self) -> None:
        off = _frozen(_seeded(n_trials=4, cross_enabled=False, cross_bound_ns=0))
        assert speech.validate(off) == ContractStatus.OK
        assert all(entry.cross_duration_ns == 0 for entry in off.explicit_schedule)

        sneaked = speech.SpeechCueConfig(
            black_bound_ns=T1,
            cross_bound_ns=0,
            content_bound_ns=T3,
            cross_enabled=False,
            schedule=speech.SpeechScheduleKind.EXPLICIT_SEQUENCE,
            n_trials=1,
            explicit_schedule=[
                speech.SpeechTrialSchedule(
                    ordinal=0,
                    stimulus_id=11,
                    cross_enabled=True,
                    black_duration_ns=1_000,
                    cross_duration_ns=2_000,
                    content_duration_ns=3_000,
                )
            ],
        )
        # A cross the configuration says does not exist.
        assert speech.validate(sneaked) == ContractStatus.OUTCOME_INVALID

    def test_only_one_side_decides_realized_value(self) -> None:
        frozen = _frozen(_seeded(n_trials=4))
        entries = list(frozen.explicit_schedule)

        def with_extra(**changes: object) -> speech.SpeechCueConfig:
            values: dict[str, object] = {
                "black_bound_ns": T1,
                "cross_bound_ns": T2,
                "content_bound_ns": T3,
                "cross_enabled": True,
                "schedule": speech.SpeechScheduleKind.EXPLICIT_SEQUENCE,
                "seed": SEED,
                "n_trials": 4,
                "explicit_schedule": entries,
            }
            values.update(changes)
            return speech.SpeechCueConfig(**values)

        assert speech.validate(with_extra()) == ContractStatus.OK
        assert (
            speech.validate(with_extra(stimulus_order=speech.StimulusOrderPolicy.SEQUENTIAL))
            == ContractStatus.OUTCOME_INVALID
        )
        assert speech.validate(with_extra(stimuli=[11])) == ContractStatus.OUTCOME_INVALID
        assert speech.validate(with_extra(n_trials=3)) == ContractStatus.OUTCOME_INVALID
        # The seed is provenance, and provenance is not a second answer.
        assert with_extra().seed == SEED

        seeded = _seeded(n_trials=4)
        assert speech.validate(seeded) == ContractStatus.OK
        strayed = speech.SpeechCueConfig(
            black_bound_ns=T1,
            cross_bound_ns=T2,
            content_bound_ns=T3,
            cross_enabled=True,
            schedule=speech.SpeechScheduleKind.SEEDED,
            stimulus_order=speech.StimulusOrderPolicy.SEQUENTIAL,
            seed=SEED,
            n_trials=4,
            stimuli=STIMULI,
            explicit_schedule=entries,
        )
        assert speech.validate(strayed) == ContractStatus.OUTCOME_INVALID

    def test_explicit_schedule_is_bounded(self) -> None:
        entry = speech.SpeechTrialSchedule(
            ordinal=0,
            stimulus_id=11,
            cross_enabled=False,
            black_duration_ns=1_000,
            content_duration_ns=3_000,
        )
        with pytest.raises(ValueError, match="explicit schedule"):
            speech.SpeechCueConfig(
                explicit_schedule=[entry] * (speech.MAX_SPEECH_EXPLICIT_TRIALS + 1)
            )


class TestScheduleOwnership:
    """Belonging, not plausibility.

    Fitting inside the bounds is not evidence that a schedule came from this
    session, and these are the checks that separate the two.
    """

    def test_seeded_schedule_cannot_introduce_foreign_stimulus(self) -> None:
        config = _seeded(n_trials=4)
        _, schedule = speech.prepare_trial(config, 0)
        foreign = _edited(schedule, stimulus_id=999)
        # Well-formed on its own, and presenting content this session does not
        # have. Were it admitted, the public timeline would carry a stimulus the
        # deterministic order could not have selected under any seed.
        assert speech.validate(foreign) == ContractStatus.OK
        assert speech.validate_against(foreign, config) == ContractStatus.TARGET_SET_INVALID
        status, _ = speech.build_timeline(config, foreign, 0)
        assert status == ContractStatus.TARGET_SET_INVALID

    def test_validation_checks_membership_not_identity(self) -> None:
        config = _seeded(n_trials=4)
        _, schedule = speech.prepare_trial(config, 0)
        assert schedule.stimulus_id != STIMULI[2]
        # Which member a given ordinal draws is for replay verification to
        # answer; this gate asks only whether the session can present it.
        assert (
            speech.validate_against(_edited(schedule, stimulus_id=STIMULI[2]), config)
            == ContractStatus.OK
        )

    def test_one_session_names_one_sampler(self) -> None:
        config = _seeded(n_trials=4)
        _, schedule = speech.prepare_trial(config, 0)
        reversioned = _edited(schedule, sampler_version=4242)
        # The session identity takes its sampler version from the configuration,
        # so an entry naming another one is not extra provenance -- it is two
        # claims about one session.
        assert speech.validate(reversioned) == ContractStatus.OK
        assert speech.validate_against(reversioned, config) == ContractStatus.OUTCOME_INVALID
        assert speech.schedule_identity(config, _catalog()).sampler_version == (
            config.sampler_version
        )

    def test_explicit_entry_is_only_schedule_for_ordinal(self) -> None:
        frozen = _frozen(_seeded(n_trials=4))
        status, recorded = speech.prepare_trial(frozen, 1)
        assert status == ContractStatus.OK
        assert speech.validate_against(recorded, frozen) == ContractStatus.OK

        # Inside every bound, and still not this trial's schedule. The
        # configuration already answered the question; a substitute that merely
        # fits is a second answer nobody would notice.
        for substitute in (
            _edited(recorded, black_duration_ns=1),
            _edited(recorded, content_duration_ns=recorded.content_duration_ns + 1),
            _edited(recorded, stimulus_id=frozen.explicit_schedule[0].stimulus_id),
        ):
            assert speech.validate(substitute) == ContractStatus.OK
            assert speech.validate_against(substitute, frozen) == ContractStatus.OUTCOME_INVALID
            assert speech.build_timeline(frozen, substitute, 0)[0] == ContractStatus.OUTCOME_INVALID
        # And the entry itself still passes, so what was refused is substitution.
        assert speech.build_timeline(frozen, recorded, 0)[0] == ContractStatus.OK


class TestConfigurationValidation:
    @pytest.mark.parametrize("bound", [0, 1])
    @pytest.mark.parametrize("field", ["black_bound_ns", "cross_bound_ns", "content_bound_ns"])
    def test_empty_timing_domain_is_rejected(self, field: str, bound: int) -> None:
        # (0, 0) and (0, 1) hold no integer, so they are empty domains rather
        # than bounds to be widened.
        assert speech.validate(_seeded(**{field: bound})) == ContractStatus.RANGE_EMPTY

    def test_rejected_bound_is_not_clamped(self) -> None:
        config = _seeded(black_bound_ns=1)
        assert speech.validate(config) == ContractStatus.RANGE_EMPTY
        assert config.black_bound_ns == 1

    def test_structural_requirements_hold(self) -> None:
        assert speech.validate(_seeded(n_trials=0)) == ContractStatus.RANGE_EMPTY
        assert (
            speech.validate(_seeded(schedule=speech.SpeechScheduleKind.UNSPECIFIED))
            == ContractStatus.IDENTITY_MISSING
        )
        assert (
            speech.validate(_seeded(stimulus_order=speech.StimulusOrderPolicy.UNSPECIFIED))
            == ContractStatus.IDENTITY_MISSING
        )
        assert speech.validate(_seeded(sampler_version=0)) == ContractStatus.IDENTITY_MISSING

    def test_stimulus_set_has_no_duplicates(self) -> None:
        assert speech.validate(_seeded(stimuli=[])) == ContractStatus.TARGET_SET_INVALID
        assert speech.validate(_seeded(stimuli=[11, 22, 11])) == ContractStatus.TARGET_SET_INVALID
        assert (
            speech.validate(_seeded(stimuli=[11, UNSET_STIMULUS_ID]))
            == ContractStatus.TARGET_SET_INVALID
        )
        with pytest.raises(ValueError, match="stimulus set"):
            _seeded(stimuli=list(range(1, speech.MAX_SPEECH_STIMULI + 2)))

    def test_declared_predicates_cover_enumerations(self) -> None:
        for value in speech.SpeechScheduleKind.__members__.values():
            assert speech.speech_schedule_kind_declared(value)
        for value in speech.StimulusOrderPolicy.__members__.values():
            assert speech.stimulus_order_policy_declared(value)
        for value in speech.SpeechContentKind.__members__.values():
            assert speech.speech_content_kind_declared(value)
        for value in speech.SpeechPhase.__members__.values():
            assert speech.speech_phase_declared(value)


class TestCatalog:
    def test_catalog_resolves_identifier(self) -> None:
        catalog = _catalog()
        assert speech.validate(catalog) == ContractStatus.OK
        assert catalog.count == len(STIMULI)
        status, stimulus = speech.find_stimulus(catalog, 33)
        assert status == ContractStatus.OK
        assert stimulus.id == 33
        assert stimulus.text == b"prompt-2"
        assert stimulus.text_length == len(b"prompt-2")
        assert stimulus.content == speech.SpeechContentKind.TEXT
        assert stimulus.metadata == 2

        assert speech.find_stimulus(catalog, 99)[0] == ContractStatus.IDENTITY_MISSING
        assert (
            speech.find_stimulus(catalog, UNSET_STIMULUS_ID)[0] == ContractStatus.IDENTITY_MISSING
        )

    def test_entry_must_carry_content(self) -> None:
        assert (
            speech.validate(speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT))
            == ContractStatus.IDENTITY_MISSING
        )
        assert (
            speech.validate(speech.SpeechStimulus(id=1, text="hello"))
            == ContractStatus.IDENTITY_MISSING
        )
        assert (
            speech.validate(
                speech.SpeechStimulus(
                    id=UNSET_STIMULUS_ID, content=speech.SpeechContentKind.TEXT, text="hello"
                )
            )
            == ContractStatus.IDENTITY_MISSING
        )
        with pytest.raises(ValueError, match="stimulus text"):
            speech.SpeechStimulus(
                id=1,
                content=speech.SpeechContentKind.TEXT,
                text="x" * (speech.MAX_SPEECH_TEXT_BYTES + 1),
            )

    def test_catalog_has_no_duplicates(self) -> None:
        assert speech.validate(speech.SpeechCatalog([])) == ContractStatus.TARGET_SET_INVALID
        duplicated = speech.SpeechCatalog(
            [
                speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text="a"),
                speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text="b"),
            ]
        )
        assert speech.validate(duplicated) == ContractStatus.TARGET_SET_INVALID

    def test_configuration_presents_only_catalog_entries(self) -> None:
        catalog = _catalog()
        config = _seeded()
        assert speech.validate_against(config, catalog) == ContractStatus.OK
        assert (
            speech.validate_against(_seeded(stimuli=[11, 22, 33, 99]), catalog)
            == ContractStatus.IDENTITY_MISSING
        )
        assert speech.validate_against(_frozen(_seeded(n_trials=4)), catalog) == (ContractStatus.OK)

    def test_text_is_bytes_and_carries_no_padding(self) -> None:
        short = speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text="hi")
        assert short.text == b"hi"
        assert len(short.text) == 2
        # Two entries with the same prompt are the same bytes, which is what
        # makes the catalog fingerprint stable.
        again = speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text="hi")
        assert speech.catalog_fingerprint(
            speech.SpeechCatalog([short])
        ) == speech.catalog_fingerprint(speech.SpeechCatalog([again]))

    def test_text_is_canonical_utf_8(self) -> None:
        # The contract imposes exactly one encoding. A str is encoded as UTF-8
        # and raw bytes are admitted only when already well-formed UTF-8, so a
        # multi-script prompt is one entry like any other.
        chinese = speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text="中文")
        assert chinese.text == "中文".encode()
        assert chinese.text_length == 6
        assert speech.validate(chinese) == ContractStatus.OK
        # The same bytes supplied directly are accepted too.
        from_bytes = speech.SpeechStimulus(
            id=1, content=speech.SpeechContentKind.TEXT, text="中文".encode()
        )
        assert from_bytes.text == chinese.text
        assert speech.catalog_fingerprint(
            speech.SpeechCatalog([chinese])
        ) == speech.catalog_fingerprint(speech.SpeechCatalog([from_bytes]))

    @pytest.mark.parametrize(
        "bad",
        [
            b"\x80",  # lone continuation byte
            b"\xc2",  # truncated two-byte sequence
            b"\xc0\x80",  # overlong encoding of NUL
            b"\xed\xa0\x80",  # lone surrogate U+D800
            b"\xf4\x90\x80\x80",  # code point beyond U+10FFFF
        ],
    )
    def test_malformed_utf_8_is_rejected_at_construction(self, bad: bytes) -> None:
        # The constructor and the native validator share one decoder, so a
        # stimulus the contract rejects can never be built from Python.
        with pytest.raises(ValueError, match="UTF-8"):
            speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text=bad)

    @pytest.mark.parametrize(
        "good",
        [
            "中文".encode(),  # three-byte CJK
            b"\xc2\x80",  # U+0080, smallest two-byte
            b"\xed\x9f\xbf",  # U+D3FF, last valid before surrogates
            b"\xf4\x8f\xbf\xbf",  # U+10FFFF, the maximum
        ],
    )
    def test_valid_utf_8_boundaries_are_accepted(self, good: bytes) -> None:
        # The boundary code points adjacent to the rejected ranges are valid,
        # and the constrained first-continuation window must not leak into the
        # later bytes of the same sequence.
        entry = speech.SpeechStimulus(id=1, content=speech.SpeechContentKind.TEXT, text=good)
        assert entry.text == good
        assert speech.validate(entry) == ContractStatus.OK


class TestFingerprints:
    def test_fingerprint_is_stable_for_one_value(self) -> None:
        catalog = _catalog()
        config = _seeded()
        assert speech.configuration_fingerprint(config) == speech.configuration_fingerprint(
            _seeded()
        )
        assert speech.catalog_fingerprint(catalog) == speech.catalog_fingerprint(_catalog())

        identity = speech.schedule_identity(config, catalog)
        assert identity.seed == config.seed
        assert identity.sampler_version == config.sampler_version
        assert identity.configuration_fingerprint == speech.configuration_fingerprint(config)
        assert identity.catalog_fingerprint == speech.catalog_fingerprint(catalog)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"seed": SEED ^ 1},
            {"sampler_version": 2},
            {"cross_enabled": False, "cross_bound_ns": 0},
            {"black_bound_ns": T1 + 1},
            {"content_bound_ns": T3 + 1},
            {"inter_trial_ns": 1},
            {"n_trials": 15},
            {"stimuli": [11, 22, 33, 55]},
            {"stimulus_order": speech.StimulusOrderPolicy.SHUFFLED_BLOCKS},
        ],
    )
    def test_schedule_change_changes_fingerprint(self, overrides: dict[str, object]) -> None:
        base = speech.configuration_fingerprint(_seeded())
        assert speech.configuration_fingerprint(_seeded(**overrides)) != base

    def test_catalog_fingerprint_covers_content(self) -> None:
        base = speech.catalog_fingerprint(_catalog())
        reworded = speech.SpeechCatalog(
            [
                speech.SpeechStimulus(
                    id=identifier, content=speech.SpeechContentKind.TEXT, text="same"
                )
                for identifier in STIMULI
            ]
        )
        assert speech.catalog_fingerprint(reworded) != base
        relabelled = speech.SpeechCatalog(
            [
                speech.SpeechStimulus(
                    id=identifier,
                    content=speech.SpeechContentKind.TEXT,
                    text=f"prompt-{idx}",
                    label=9,
                    metadata=idx,
                )
                for idx, identifier in enumerate(STIMULI)
            ]
        )
        assert speech.catalog_fingerprint(relabelled) != base


class TestDrawProvenance:
    def test_realized_durations_are_emitted_as_shared_records(self) -> None:
        config = _seeded()
        _, schedule = speech.prepare_trial(config, 2)
        trial = TrialIdentity(ordinal=2, key=7, stimulus_id=schedule.stimulus_id)
        status, draws = speech.make_schedule_draws(schedule, trial, 5_000)
        assert status == ContractStatus.OK
        assert draws.count == 3
        assert [record.draw for record in draws.draws] == [
            speech.BLACK_DRAW,
            speech.CROSS_DRAW,
            speech.CONTENT_DRAW,
        ]
        assert [record.value for record in draws.draws] == [
            schedule.black_duration_ns,
            schedule.cross_duration_ns,
            schedule.content_duration_ns,
        ]
        for record in draws.draws:
            assert record.stream == speech.TRIAL_TIMING_STREAM
            assert record.sampler_version == schedule.sampler_version
            assert record.time_ns == 5_000
            assert record.trial.ordinal == 2
            # Provenance a reader can check against a regeneration without
            # knowing anything about this paradigm.
            bound = {
                speech.BLACK_DRAW: T1,
                speech.CROSS_DRAW: T2,
                speech.CONTENT_DRAW: T3,
            }[record.draw]
            assert record.value == _sample(
                config.seed, record.stream, schedule.ordinal, record.draw, bound
            )

    def test_disabled_cross_emits_no_draw(self) -> None:
        config = _seeded(cross_enabled=False, cross_bound_ns=0)
        _, schedule = speech.prepare_trial(config, 2)
        trial = TrialIdentity(ordinal=2, stimulus_id=schedule.stimulus_id)
        status, draws = speech.make_schedule_draws(schedule, trial, 0)
        assert status == ContractStatus.OK
        # A reader counting records sees the phase was never sampled.
        assert draws.count == 2
        assert [record.draw for record in draws.draws] == [
            speech.BLACK_DRAW,
            speech.CONTENT_DRAW,
        ]

    def test_values_cannot_be_filed_under_another_trial(self) -> None:
        config = _seeded()
        _, schedule = speech.prepare_trial(config, 2)
        # The identity is the one field these records exist to carry, so it is
        # checked rather than copied: values drawn for one trial filed under
        # another are provenance that says the wrong thing.
        assert (
            speech.make_schedule_draws(
                schedule, TrialIdentity(ordinal=9, stimulus_id=schedule.stimulus_id), 0
            )[0]
            == ContractStatus.OUTCOME_INVALID
        )
        assert (
            speech.make_schedule_draws(
                schedule, TrialIdentity(ordinal=2, stimulus_id=schedule.stimulus_id + 1), 0
            )[0]
            == ContractStatus.OUTCOME_INVALID
        )
        assert (
            speech.make_schedule_draws(schedule, TrialIdentity(ordinal=2), 0)[0]
            == ContractStatus.IDENTITY_MISSING
        )

    def test_draw_record_binds_only_its_own_values(self) -> None:
        config = _seeded()
        _, schedule = speech.prepare_trial(config, 2)
        trial = TrialIdentity(ordinal=2, key=11, block=4, stimulus_id=schedule.stimulus_id)
        status, draws = speech.make_schedule_draws(schedule, trial, 0)
        assert status == ContractStatus.OK
        assert draws.count == 3
        assert all(record.trial.block == 4 for record in draws.draws)


class TestLargeSchedules:
    def test_long_session_schedules_without_prepared_table(self) -> None:
        config = _seeded(
            n_trials=1_000_000,
            stimulus_order=speech.StimulusOrderPolicy.RANDOM_WITH_REPLACEMENT,
        )
        assert speech.validate(config) == ContractStatus.OK
        # Nothing was prepared to reach the last trial, and reaching it costs
        # what reaching the first one costs.
        for ordinal in (0, 999, 500_000, config.n_trials - 1):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            assert schedule.ordinal == ordinal
            assert 0 < schedule.black_duration_ns < T1
            assert 0 < schedule.cross_duration_ns < T2
            assert 0 < schedule.content_duration_ns < T3

    def test_dense_sweep_of_large_block_stays_in_range(self) -> None:
        config = _seeded(n_trials=20_000, stimulus_order=speech.StimulusOrderPolicy.SHUFFLED_BLOCKS)
        for ordinal in range(0, config.n_trials, 7):
            status, schedule = speech.prepare_trial(config, ordinal)
            assert status == ContractStatus.OK
            assert 0 < schedule.black_duration_ns < T1
            assert 0 < schedule.cross_duration_ns < T2
            assert 0 < schedule.content_duration_ns < T3
            assert schedule.stimulus_id in STIMULI


class TestOwnershipBoundary:
    def test_module_owns_no_presentation_or_audio(self) -> None:
        exported = set(speech.__all__)
        # ``SpeechMachine`` is not on this list: it is a pure timed state
        # machine over explicit time and prepared schedules rather than a
        # presenter. What stays forbidden is anything that would draw,
        # capture, or decode.
        forbidden = {
            "SpeechRenderer",
            "Microphone",
            "AudioDevice",
            "SpeechDecoder",
            "record_audio",
            "present",
            "render",
        }
        assert exported & forbidden == set()
        assert exported == set(dir(speech)) & exported

    def test_undeclared_name_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            speech.SpeechRenderer  # type: ignore[attr-defined]  # noqa: B018
