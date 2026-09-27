#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The verdict vocabulary a deterministic semantic replay reports in.

Driving a replay is native: it steps a paradigm's own machine over a recorded
timeline and compares as it goes, and doing that through a binding would put a
Python call between every regenerated transition and the recorded one it is
checked against. What is bound, and what is tested here, is the *answer* -- read
by whoever decides whether a recording is trustworthy, which is where Python is.
"""

from __future__ import annotations

import pytest

from neurale import experiments


def test_verdict_is_never_boolean() -> None:
    # Four values, and the two that are neither match nor mismatch are the
    # interesting ones: a recording that never held the outputs, and a recording
    # of a different experiment, are both answerable and neither is a match.
    assert {
        experiments.ReplayVerdict.MATCH,
        experiments.ReplayVerdict.MISMATCH,
        experiments.ReplayVerdict.INCOMPLETE,
        experiments.ReplayVerdict.REJECTED,
    } == set(experiments.ReplayVerdict.__members__.values())
    for verdict in experiments.ReplayVerdict.__members__.values():
        assert experiments.replay_verdict_declared(verdict)


def test_empty_report_verifies_nothing() -> None:
    report = experiments.ReplayReport()
    # The default is `rejected` rather than `match` so that forgetting to run a
    # replay cannot read as having run one.
    assert report.verdict == experiments.ReplayVerdict.REJECTED
    assert not experiments.replay_reproduced(report)
    assert report.rejection == experiments.ReplayRejection.NONE
    assert report.completeness == experiments.ReplayCompleteness.COMPLETE
    assert report.items_compared == 0
    assert report.fields_compared == 0
    assert report.inputs_replayed == 0
    assert report.first_mismatch.item == experiments.ReplayItem.NONE
    assert report.first_mismatch.field == ""


def test_only_match_counts_as_reproduction() -> None:
    # `replay_reproduced` exists so that a caller cannot spell the question as
    # "not a mismatch", which is true of a rejected replay and of one that
    # compared nothing at all.
    assert experiments.replay_reproduced(experiments.ReplayReport()) is False


@pytest.mark.parametrize(
    ("declared", "enumeration"),
    [
        (experiments.replay_verdict_declared, experiments.ReplayVerdict),
        (experiments.replay_rejection_declared, experiments.ReplayRejection),
        (experiments.replay_item_declared, experiments.ReplayItem),
        (experiments.replay_completeness_declared, experiments.ReplayCompleteness),
        (
            experiments.replay_incomplete_policy_declared,
            experiments.ReplayIncompletePolicy,
        ),
    ],
)
def test_every_declared_value_is_declared(declared, enumeration) -> None:
    for value in enumeration.__members__.values():
        assert declared(value)


@pytest.mark.parametrize(
    ("name_of", "enumeration"),
    [
        (experiments.replay_verdict_name, experiments.ReplayVerdict),
        (experiments.replay_rejection_name, experiments.ReplayRejection),
        (experiments.replay_item_name, experiments.ReplayItem),
        (experiments.replay_completeness_name, experiments.ReplayCompleteness),
    ],
)
def test_values_have_distinct_stable_names(name_of, enumeration) -> None:
    # A report is read by a person, so two findings must not print the same way.
    names = [name_of(value) for value in enumeration.__members__.values()]
    assert len(set(names)) == len(names)
    assert all(name and name != "undeclared" for name in names)


def _provenance(**overrides: int) -> experiments.ReplayProvenance:
    fields = {
        "paradigm": 41,
        "experiment_version": 1,
        "configuration_fingerprint": 111,
        "schedule_fingerprint": 222,
        "realized_schedule_fingerprint": 333,
        "seed": 17,
        "sampler_version": experiments.SAMPLER_VERSION_1,
        "metric_version": 0,
        "policy_version": 1,
    }
    fields.update(overrides)
    return experiments.ReplayProvenance(**fields)


def test_recordings_of_same_experiment_agree() -> None:
    assert (
        experiments.check_provenance(_provenance(), _provenance())
        == experiments.ReplayRejection.NONE
    )


@pytest.mark.parametrize(
    ("field", "value", "rejection"),
    [
        ("paradigm", 42, experiments.ReplayRejection.PARADIGM_MISMATCH),
        (
            "experiment_version",
            2,
            experiments.ReplayRejection.EXPERIMENT_VERSION_MISMATCH,
        ),
        ("seed", 18, experiments.ReplayRejection.SEED_MISMATCH),
        (
            "configuration_fingerprint",
            112,
            experiments.ReplayRejection.CONFIGURATION_FINGERPRINT_MISMATCH,
        ),
        (
            "schedule_fingerprint",
            223,
            experiments.ReplayRejection.SCHEDULE_FINGERPRINT_MISMATCH,
        ),
        (
            "realized_schedule_fingerprint",
            334,
            experiments.ReplayRejection.REALIZED_SCHEDULE_MISMATCH,
        ),
        ("metric_version", 1, experiments.ReplayRejection.METRIC_VERSION_MISMATCH),
        ("policy_version", 2, experiments.ReplayRejection.POLICY_VERSION_MISMATCH),
    ],
)
def test_recording_of_other_experiment_is_named(field: str, value: int, rejection) -> None:
    assert experiments.check_provenance(_provenance(), _provenance(**{field: value})) == rejection


def test_empty_provenance_is_not_agreement() -> None:
    # Two provenances that both name no paradigm agree field by field, and a
    # replay that took that as permission would proceed against provenance that
    # identifies nothing.
    blank = experiments.ReplayProvenance()
    assert (
        experiments.check_provenance(blank, blank)
        == experiments.ReplayRejection.PROVENANCE_INCOMPLETE
    )
    assert (
        experiments.check_provenance(
            _provenance(paradigm=experiments.UNSET_PARADIGM_ID), _provenance()
        )
        == experiments.ReplayRejection.PROVENANCE_INCOMPLETE
    )


def test_provenance_does_not_guess_sampler_need() -> None:
    unsupported = experiments.SAMPLER_VERSION_1 + 7
    assert not experiments.sampler_version_supported(unsupported)
    # With a realized schedule to fall back to, an unsupported sampler is not a
    # rejection: ScheduleIdentity says such a version is legal, and
    # ReplayAuthority.RECORDED_SCHEDULE is the route through it.
    with_schedule = _provenance(sampler_version=unsupported)
    assert (
        experiments.check_provenance(with_schedule, with_schedule)
        == experiments.ReplayRejection.NONE
    )
    assert (
        experiments.replay_authority(unsupported) == experiments.ReplayAuthority.RECORDED_SCHEDULE
    )
    # The absence of a realized-schedule digest is not enough for this generic
    # function to decide that a draw is required. Explicit WebGrid schedules and
    # Center-Out repeat-until-success both have a zero digest and execute no
    # sampler; each replay engine owns the policy-specific decision.
    without_schedule = _provenance(sampler_version=unsupported, realized_schedule_fingerprint=0)
    assert (
        experiments.check_provenance(without_schedule, without_schedule)
        == experiments.ReplayRejection.NONE
    )


def test_incomplete_policy_covers_partial_evidence() -> None:
    assert {
        experiments.ReplayIncompletePolicy.VERIFY_AVAILABLE,
        experiments.ReplayIncompletePolicy.REFUSE,
    } == set(experiments.ReplayIncompletePolicy.__members__.values())
    # Verifying what is there is the default, because a partial verification is
    # worth having and the verdict is what stops it reading as a whole one.
    assert int(experiments.ReplayIncompletePolicy.VERIFY_AVAILABLE) == 0
