#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from neurale.experiments import (
    UNSET_TARGET_ID,
    ContractStatus,
    SelectionEvent,
    SelectionKind,
    TrialIdentity,
    webgrid,
)

PARADIGM = 17
SECOND = 1_000_000_000


def _explicit_config(**overrides: object) -> webgrid.WebGridConfig:
    values: dict[str, object] = {
        "rows": 1,
        "columns": 2,
        "bounds": webgrid.TaskBounds(0.0, 2.0, 0.0, 1.0),
        "candidates": [1, 2],
        "schedule": webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE,
        "immediate_repetition": webgrid.ImmediateRepetitionPolicy.FORBID,
        "correct_selection": webgrid.CorrectSelectionPolicy.ADVANCE_TARGET,
        "incorrect_selection": webgrid.IncorrectSelectionPolicy.KEEP_CURRENT_TARGET,
        "explicit_targets": [1, 2, 1],
        "initial_target": 1,
        "target_count_limit": 2,
        "metric_version": webgrid.METRIC_VERSION_1,
    }
    values.update(overrides)
    return webgrid.WebGridConfig(**values)


def _selection(
    machine: webgrid.WebGridMachine,
    pointer: webgrid.PointerPosition,
    time_ns: int,
    sequence: int,
) -> object:
    snapshot = machine.snapshot()
    status, event = webgrid.make_selection_event(
        machine.configuration(),
        pointer,
        snapshot.active_target,
        snapshot.trial,
        machine.paradigm,
        time_ns,
        sequence,
    )
    assert status == ContractStatus.OK
    return event


def _raw_selection(
    trial: TrialIdentity,
    *,
    target_onset_ns: int,
    time_ns: int,
    sequence: int,
    correct: bool,
    selected_id: int,
    paradigm: int = PARADIGM,
) -> webgrid.WebGridSelectionRecord:
    event = SelectionEvent(
        time_ns=time_ns,
        sequence=sequence,
        trial=trial,
        paradigm=paradigm,
        kind=SelectionKind.DISCRETE,
        correct=correct,
        selected_id=selected_id,
        intended_id=trial.target_id,
    )
    return webgrid.WebGridSelectionRecord(
        event=event,
        target_onset_ns=target_onset_ns,
        elapsed_since_target_onset_ns=time_ns - target_onset_ns,
    )


def test_misselection_is_recorded_without_changing_target() -> None:
    machine = webgrid.WebGridMachine()
    status, started = machine.start(PARADIGM, _explicit_config(), 0)
    assert status == ContractStatus.OK
    assert started.snapshot.state == webgrid.WebGridState.ACTIVE_TARGET
    assert started.snapshot.active_target == 1

    # The exact x=1 boundary belongs to the half-open second cell.
    boundary = webgrid.PointerPosition(1.0, 0.5)
    event = _selection(machine, boundary, 10 * SECOND, 1)
    assert not event.correct
    assert event.selected_id == 2

    status, result = machine.step(10 * SECOND, boundary, event)
    assert status == ContractStatus.OK
    assert result.selection_processed
    assert not result.selection.event.correct
    assert result.selection.target_onset_ns == 0
    assert result.selection.elapsed_since_target_onset_ns == 10 * SECOND
    assert not result.trial_decided
    assert result.snapshot.active_target == 1
    assert result.snapshot.target_onset_ns == 0
    assert result.snapshot.metrics.incorrect_selections == 1


def test_successes_advance_targets_and_terminate_at_count() -> None:
    machine = webgrid.WebGridMachine()
    assert machine.start(PARADIGM, _explicit_config(), 0)[0] == ContractStatus.OK
    cell_one = webgrid.PointerPosition(0.5, 0.5)
    first = _selection(machine, cell_one, 20 * SECOND, 1)

    status, result = machine.step(20 * SECOND, cell_one, first)
    assert status == ContractStatus.OK
    assert result.trial_decided
    assert result.trial.selection.event.correct
    assert (
        result.trial.selection.event.selected_id,
        result.trial.selection.event.intended_id,
        result.trial.selection.event.time_ns,
    ) == (1, 1, 20 * SECOND)
    assert result.trial.selection.target_onset_ns == 0
    assert result.trial.acquisition_ns == 20 * SECOND
    assert result.trial.record.trial.target_id == 1
    assert (result.trial.record.interval.start_ns, result.trial.record.interval.end_ns) == (
        0,
        20 * SECOND,
    )
    assert result.snapshot.active_target == 2
    assert result.snapshot.target_onset_ns == 20 * SECOND

    cell_two = webgrid.PointerPosition(1.5, 0.5)
    second = _selection(machine, cell_two, 50 * SECOND, 2)
    status, result = machine.step(50 * SECOND, cell_two, second)
    assert status == ContractStatus.OK
    assert result.trial_decided
    assert result.trial.acquisition_ns == 30 * SECOND
    assert result.snapshot.state == webgrid.WebGridState.COMPLETE
    assert result.snapshot.active_target == UNSET_TARGET_ID
    assert machine.complete


def test_repeated_selection_id_is_rejected_without_double_count() -> None:
    machine = webgrid.WebGridMachine()
    assert machine.start(PARADIGM, _explicit_config(), 0)[0] == ContractStatus.OK
    wrong_pointer = webgrid.PointerPosition(1.5, 0.5)
    event = _selection(machine, wrong_pointer, 10, 5)
    assert machine.step(10, wrong_pointer, event)[0] == ContractStatus.OK

    before = machine.snapshot()
    assert machine.step(10, wrong_pointer, event)[0] == ContractStatus.OUTCOME_INVALID
    assert machine.snapshot().metrics.incorrect_selections == before.metrics.incorrect_selections
    regressed = _selection(machine, wrong_pointer, 11, 4)
    assert machine.step(11, wrong_pointer, regressed)[0] == ContractStatus.OUTCOME_INVALID
    assert machine.snapshot().time_ns == before.time_ns


def test_metrics_match_independent_raw_recomputation() -> None:
    machine = webgrid.WebGridMachine()
    assert machine.start(PARADIGM, _explicit_config(), 0)[0] == ContractStatus.OK
    records = []

    wrong_pointer = webgrid.PointerPosition(1.5, 0.5)
    wrong = _selection(machine, wrong_pointer, 10 * SECOND, 1)
    _, result = machine.step(10 * SECOND, wrong_pointer, wrong)
    records.append(result.selection)

    cell_one = webgrid.PointerPosition(0.5, 0.5)
    correct_one = _selection(machine, cell_one, 20 * SECOND, 2)
    _, result = machine.step(20 * SECOND, cell_one, correct_one)
    records.append(result.selection)

    cell_two = webgrid.PointerPosition(1.5, 0.5)
    correct_two = _selection(machine, cell_two, 50 * SECOND, 3)
    _, result = machine.step(50 * SECOND, cell_two, correct_two)
    records.append(result.selection)
    metrics = result.snapshot.metrics

    correct = sum(record.event.correct for record in records)
    incorrect = len(records) - correct
    acquisitions = [
        record.elapsed_since_target_onset_ns for record in records if record.event.correct
    ]
    elapsed = 50 * SECOND
    assert (metrics.correct_selections, metrics.incorrect_selections) == (correct, incorrect)
    assert metrics.elapsed_active_ns == elapsed
    assert metrics.correct_targets_per_minute == pytest.approx(correct * 60 * SECOND / elapsed)
    assert metrics.net_correct_targets_per_minute == pytest.approx(
        (correct - incorrect) * 60 * SECOND / elapsed
    )
    assert metrics.n_acquisitions == len(acquisitions)
    assert metrics.total_acquisition_ns == sum(acquisitions)
    assert metrics.minimum_acquisition_ns == min(acquisitions)
    assert metrics.maximum_acquisition_ns == max(acquisitions)
    assert metrics.mean_acquisition_ns == sum(acquisitions) // len(acquisitions)

    status, recomputed = webgrid.summarize(records, 0, elapsed, webgrid.METRIC_VERSION_1)
    assert status == ContractStatus.OK
    for field in (
        "correct_selections",
        "incorrect_selections",
        "elapsed_active_ns",
        "rates_defined",
        "correct_targets_per_minute",
        "net_correct_targets_per_minute",
        "n_acquisitions",
        "total_acquisition_ns",
        "minimum_acquisition_ns",
        "maximum_acquisition_ns",
        "mean_acquisition_ns",
    ):
        assert getattr(recomputed, field) == pytest.approx(getattr(metrics, field))


def test_offline_summary_enforces_trial_lifecycle() -> None:
    trial_zero = TrialIdentity(ordinal=0, target_id=1)
    trial_one = TrialIdentity(ordinal=1, target_id=2)
    valid = [
        _raw_selection(
            trial_zero,
            target_onset_ns=0,
            time_ns=5,
            sequence=1,
            correct=False,
            selected_id=2,
        ),
        _raw_selection(
            trial_zero,
            target_onset_ns=0,
            time_ns=7,
            sequence=2,
            correct=False,
            selected_id=2,
        ),
        _raw_selection(
            trial_zero,
            target_onset_ns=0,
            time_ns=10,
            sequence=3,
            correct=True,
            selected_id=1,
        ),
        _raw_selection(
            trial_one,
            target_onset_ns=10,
            time_ns=12,
            sequence=4,
            correct=False,
            selected_id=1,
        ),
        _raw_selection(
            trial_one,
            target_onset_ns=10,
            time_ns=20,
            sequence=5,
            correct=True,
            selected_id=2,
        ),
    ]
    status, metrics = webgrid.summarize(valid, 0, 20, webgrid.METRIC_VERSION_1)
    assert status == ContractStatus.OK
    assert (metrics.correct_selections, metrics.incorrect_selections) == (2, 3)
    assert (metrics.total_acquisition_ns, metrics.mean_acquisition_ns) == (20, 10)

    two_correct_same_trial = [
        _raw_selection(
            trial_zero,
            target_onset_ns=0,
            time_ns=10,
            sequence=1,
            correct=True,
            selected_id=1,
        ),
        _raw_selection(
            trial_zero,
            target_onset_ns=0,
            time_ns=20,
            sequence=2,
            correct=True,
            selected_id=1,
        ),
    ]
    assert (
        webgrid.summarize(two_correct_same_trial, 0, 20, webgrid.METRIC_VERSION_1)[0]
        == ContractStatus.OUTCOME_INVALID
    )

    changed_onset_after_incorrect = [
        _raw_selection(
            trial_zero,
            target_onset_ns=0,
            time_ns=5,
            sequence=1,
            correct=False,
            selected_id=2,
        ),
        _raw_selection(
            trial_zero,
            target_onset_ns=7,
            time_ns=10,
            sequence=2,
            correct=True,
            selected_id=1,
        ),
    ]
    assert (
        webgrid.summarize(changed_onset_after_incorrect, 0, 10, webgrid.METRIC_VERSION_1)[0]
        == ContractStatus.OUTCOME_INVALID
    )


def test_offline_summary_requires_supported_metric_version() -> None:
    with pytest.raises(TypeError):
        webgrid.summarize([], 0, 0)

    status, metrics = webgrid.summarize([], 0, 0, webgrid.METRIC_VERSION_1)
    assert status == ContractStatus.OK
    assert metrics.metric_version == webgrid.METRIC_VERSION_1

    status, metrics = webgrid.summarize([], 0, 0, 999)
    assert status == ContractStatus.VERSION_UNSUPPORTED
    assert metrics.correct_selections == 0

    assert webgrid.summarize([], 0, 0, 0)[0] == ContractStatus.IDENTITY_MISSING


def test_zero_elapsed_rates_are_explicitly_undefined() -> None:
    machine = webgrid.WebGridMachine()
    status, result = machine.start(PARADIGM, _explicit_config(), 100)
    assert status == ContractStatus.OK
    metrics = result.snapshot.metrics
    assert not metrics.rates_defined
    assert metrics.correct_targets_per_minute == 0.0
    assert metrics.net_correct_targets_per_minute == 0.0


def test_duration_endpoint_terminates_without_selection() -> None:
    machine = webgrid.WebGridMachine()
    config = _explicit_config(target_count_limit=0, session_duration_ns=10)
    assert machine.start(PARADIGM, config, 100)[0] == ContractStatus.OK
    pointer = webgrid.PointerPosition(0.5, 0.5)
    event = _selection(machine, pointer, 110, 1)

    status, result = machine.step(110, pointer, event)
    assert status == ContractStatus.OK
    assert result.snapshot.state == webgrid.WebGridState.COMPLETE
    assert not result.selection_processed
    assert not result.trial_decided
    assert result.snapshot.metrics.elapsed_active_ns == 10


def test_seeded_schedule_and_reset_ignore_extra_pointer_only_steps() -> None:
    config = _explicit_config(
        schedule=webgrid.TargetScheduleKind.SEEDED,
        explicit_targets=[],
        initial_target=UNSET_TARGET_ID,
        target_count_limit=4,
        seed=123456,
    )
    machine = webgrid.WebGridMachine()
    sequences: list[list[int]] = []
    for run in range(2):
        assert machine.start(PARADIGM, config, 0)[0] == ContractStatus.OK
        targets = []
        for ordinal in range(4):
            snapshot = machine.snapshot()
            targets.append(snapshot.active_target)
            _, cell = webgrid.grid_cell_by_id(config, snapshot.active_target)
            pointer = webgrid.PointerPosition(
                (cell.bounds.min_x + cell.bounds.max_x) / 2,
                (cell.bounds.min_y + cell.bounds.max_y) / 2,
            )
            event_time = (ordinal + 1) * 10
            if run == 0:
                assert machine.step(event_time - 1, pointer)[0] == ContractStatus.OK
            event = _selection(machine, pointer, event_time, ordinal + 1)
            assert machine.step(event_time, pointer, event)[0] == ContractStatus.OK
        sequences.append(targets)
        assert machine.complete
        machine.reset()
        assert machine.state == webgrid.WebGridState.IDLE
    assert sequences[0] == sequences[1]


def test_metric_version_and_records_are_immutable() -> None:
    assert webgrid.CURRENT_METRIC_VERSION == webgrid.METRIC_VERSION_1
    future = _explicit_config(metric_version=999)
    assert webgrid.validate(future) == ContractStatus.OK
    assert webgrid.WebGridMachine().start(PARADIGM, future, 0)[0] == (
        ContractStatus.VERSION_UNSUPPORTED
    )

    machine = webgrid.WebGridMachine()
    _, result = machine.start(PARADIGM, _explicit_config(), 0)
    with pytest.raises(AttributeError):
        result.snapshot.state = webgrid.WebGridState.COMPLETE
    with pytest.raises(AttributeError):
        result.snapshot.metrics.correct_selections = 1
