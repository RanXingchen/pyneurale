# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Explicit-time SSVEP trials: this is not display or EEG timing evidence."""

from __future__ import annotations

import pytest
from test_ssvep import make_config

from neurale.experiments import ContractStatus, CueKind, SelectionEvent, TrialOutcome, ssvep


def started(n_trials=12, **kwargs):
    machine = ssvep.SSVEPMachine()
    status, output = machine.start(7, make_config(**kwargs), 0, trials=n_trials)
    assert status == ContractStatus.OK
    return machine, output


def choice(machine, at, correct=True, sequence=0):
    trial = machine.snapshot().trial
    return SelectionEvent(
        time_ns=at,
        sequence=sequence,
        trial=trial,
        paradigm=7,
        selected_id=trial.target_id if correct else trial.target_id % 4 + 1,
        intended_id=trial.target_id,
        correct=correct,
    )


@pytest.mark.parametrize("at", [10, 29, 30, 31, 39, 40, 41])
@pytest.mark.parametrize("correct", [True, False])
def test_outcomes(at, correct):
    machine, _ = started(n_trials=1)
    status, output = machine.step(at, choice(machine, at, correct))
    assert status == ContractStatus.OK
    assert output.selection_disposition == (
        ssvep.SSVEPSelectionDisposition.ACCEPTED
        if at < 40
        else ssvep.SSVEPSelectionDisposition.EXPIRED
    )
    feedback = max(30, min(at, 40))
    status, output = machine.step(feedback + 5)
    assert status == ContractStatus.OK and output.trial_decided
    expected = (
        TrialOutcome.TIMEOUT
        if at >= 40
        else TrialOutcome.SUCCESS
        if correct
        else TrialOutcome.FAILURE
    )
    assert output.trial.record.outcome == expected
    assert output.trial.decision_ns == feedback
    assert len(output.trial.phases) == (3 if at <= 30 else 4)
    assert ssvep.validate(output.trial) == ContractStatus.OK
    for request in output.requests:
        assert ssvep.validate(request) == ContractStatus.OK
    status, output = machine.step(feedback + 10)
    assert status == ContractStatus.OK and machine.complete
    assert output.requests[-1].request.cue == CueKind.BLACK


def test_early_choice_does_not_shorten_stimulation():
    machine, _ = started()
    assert machine.step(10, choice(machine, 10))[0] == ContractStatus.OK
    status, output = machine.step(29)
    assert status == ContractStatus.OK
    assert output.snapshot.state == ssvep.SSVEPState.STIMULATION
    assert output.snapshot.outcome == TrialOutcome.PENDING
    assert not output.trial_decided
    status, output = machine.step(30)
    assert output.snapshot.state == ssvep.SSVEPState.FEEDBACK
    assert output.requests[-1].selected_id == machine.snapshot().trial.target_id
    assert output.requests[-1].request.cue == CueKind.SSVEP_TARGETS


def test_rejection_reset_and_stop():
    machine, _ = started()
    assert machine.step(9, choice(machine, 9))[0] == ContractStatus.OUTCOME_INVALID
    assert machine.snapshot().time_ns == 0
    event = choice(machine, 10)
    assert machine.step(10, event)[0] == ContractStatus.OK
    assert machine.step(10, event)[0] == ContractStatus.OUTCOME_INVALID
    assert machine.step(9)[0] == ContractStatus.TIME_REGRESSED
    status, output = machine.stop(11)
    assert status == ContractStatus.OK and machine.complete
    assert output.trial.record.outcome == TrialOutcome.ABORTED
    assert ssvep.validate(output.trial) == ContractStatus.OK
    assert machine.step(12)[0] == ContractStatus.NOT_RUNNING
    machine.reset()
    assert machine.state == ssvep.SSVEPState.IDLE
    assert machine.start(7, make_config(), 0, trials=12)[1].snapshot.trial.target_id == 4


def test_large_jump_and_no_duplicate_trial_on_stop():
    machine, _ = started(n_trials=2)
    status, output = machine.step(100)
    assert status == ContractStatus.OK and output.trial_decided and not output.settled
    assert output.trial.record.trial.ordinal == 0
    status, output = machine.stop(100)
    assert status == ContractStatus.OK and not output.trial_decided
    assert output.snapshot.completed == 1


def test_task_reuse_and_start_trial_count():
    task = make_config()
    machine = ssvep.SSVEPMachine()
    with pytest.raises(ValueError, match="trials"):
        machine.start(7, task, 0, trials=0)
    assert machine.state == ssvep.SSVEPState.IDLE
    with pytest.raises(TypeError):
        machine.start(7, task, 0)
    assert machine.start(7, task, 0, trials=1)[0] == ContractStatus.OK
    machine.step(45)
    assert machine.step(50)[0] == ContractStatus.OK and machine.complete
    machine.reset()
    assert machine.start(7, task, 0, trials=2)[0] == ContractStatus.OK
    machine.step(45)
    assert machine.step(50)[0] == ContractStatus.OK and not machine.complete


def test_seconds_are_converted_to_native_timing():
    task = make_config(
        cue_duration=1,
        stimulation_duration=2.5,
        decision_timeout=1,
        feedback_duration=0.5,
        inter_trial=0.5,
    )
    machine = ssvep.SSVEPMachine()
    assert machine.start(7, task, 0, trials=1)[0] == ContractStatus.OK
    assert machine.step(999_999_999)[1].snapshot.state == ssvep.SSVEPState.CUE
    assert machine.step(1_000_000_000)[1].snapshot.state == ssvep.SSVEPState.STIMULATION
    assert machine.step(3_500_000_000)[1].snapshot.state == ssvep.SSVEPState.AWAIT_DECISION
    assert machine.step(4_500_000_000)[1].snapshot.state == ssvep.SSVEPState.FEEDBACK
    assert machine.step(5_000_000_000)[1].trial_decided
    assert machine.step(5_500_000_000)[0] == ContractStatus.OK and machine.complete
