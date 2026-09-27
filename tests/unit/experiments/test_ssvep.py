# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""SSVEP public values and deterministic target schedule, without a renderer."""

from __future__ import annotations

import pytest
from _subprocess_probe import probe_json

from neurale.experiments import ContractStatus, ssvep


def make_config(**overrides):
    values = dict(
        targets=[ssvep.SSVEPTarget(i + 1, f) for i, f in enumerate((8, 10, 12, 15))],
        stimulus_id=5,
        seed=42,
        cue_duration=10e-9,
        stimulation_duration=20e-9,
        decision_timeout=10e-9,
        feedback_duration=5e-9,
        inter_trial=5e-9,
    )
    values.update(overrides)
    return ssvep.SSVEPTask(**values)


def test_import_is_lazy():
    result = probe_json("""
import json, sys
from neurale.experiments import ssvep
print(json.dumps({'native': 'neurale._native' in sys.modules,
                  'machine': 'SSVEPMachine' in dir(ssvep)}))
""")
    assert result == {"native": False, "machine": True}


def test_balanced_schedule_and_partial_cycle():
    config = make_config()
    targets = []
    for ordinal in range(13):
        status, schedule = ssvep.prepare_trial(config, ordinal)
        assert status == ContractStatus.OK
        assert schedule.ordinal == ordinal
        targets.append(schedule.target_id)
    assert targets[:12] == [4, 3, 1, 2, 4, 3, 1, 2, 1, 4, 3, 2]


@pytest.mark.parametrize(
    "field",
    [
        "cue_duration",
        "stimulation_duration",
        "decision_timeout",
        "feedback_duration",
        "inter_trial",
    ],
)
def test_positive_configuration(field):
    with pytest.raises(ValueError, match=field):
        make_config(**{field: 0})


def test_value_validation_and_readonly():
    with pytest.raises(ValueError, match="frequency_hz"):
        ssvep.SSVEPTarget(1, float("nan"))
    with pytest.raises(ValueError, match="id"):
        ssvep.SSVEPTarget(0, 8)
    with pytest.raises(ValueError, match="targets"):
        make_config(targets=[ssvep.SSVEPTarget(1, 8)])
    with pytest.raises(ValueError, match="unique"):
        make_config(targets=[ssvep.SSVEPTarget(1, 8), ssvep.SSVEPTarget(2, 8)])
    with pytest.raises(ValueError, match="unique"):
        make_config(targets=[ssvep.SSVEPTarget(1, 8), ssvep.SSVEPTarget(1, 10)])
    with pytest.raises(ValueError, match="targets"):
        make_config(targets=[ssvep.SSVEPTarget(i + 1, i + 1) for i in range(65)])
    with pytest.raises(ValueError, match="stimulus_id"):
        make_config(stimulus_id=0)
    config = make_config()
    with pytest.raises(AttributeError):
        config.seed = 5
    items = config.targets
    items.clear()
    assert len(config.targets) == 4


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 1e30, 1e-12])
@pytest.mark.parametrize(
    "field",
    [
        "cue_duration",
        "stimulation_duration",
        "decision_timeout",
        "feedback_duration",
        "inter_trial",
    ],
)
def test_duration_errors_identify_field(field, value):
    with pytest.raises(ValueError, match=field):
        make_config(**{field: value})


def test_public_task_boundary():
    task = make_config(inter_trial=0.5)
    assert task.inter_trial == 0.5
    assert ssvep.validate(task) == ContractStatus.OK
    for field in ("sampler_version", "n_trials", "inter_trial_ns", "n_targets"):
        assert not hasattr(task, field)
        with pytest.raises(TypeError):
            make_config(**{field: 1})
    with pytest.raises(TypeError):
        ssvep.SSVEPTask(task.targets, task.stimulus_id, 1, 2, 1, 0.5, 0.5)
    for name in (
        "SSVEPConfig",
        "MAX_STEP_EVENTS",
        "MAX_STEP_TRANSITIONS",
        "MAX_STEP_REQUESTS",
        "MAX_SSVEP_PHASES",
        "contains_target",
    ):
        assert name not in ssvep.__all__
        assert not hasattr(ssvep, name)
    for name in (
        "SSVEPTrial",
        "SSVEPTrialSchedule",
        "SSVEPPhaseInterval",
        "SSVEPPresentationRequest",
    ):
        with pytest.raises(TypeError):
            getattr(ssvep, name)()
