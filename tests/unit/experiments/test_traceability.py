#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from neurale.exceptions import ValidationError
from neurale.experiments.traceability import (
    AcquisitionMarkerEvidence,
    AssistanceDecisionEvidence,
    CenterOutCommandEvidence,
    CenterOutProvenanceIndex,
    CommandApplicationEvidence,
    DecoderOutputEvidence,
    EvidenceGapReason,
    EvidenceId,
    EvidenceKind,
    ExperimentStateEvidence,
    FeatureObservationEvidence,
    GuidanceDecisionEvidence,
    MetricContributionEvidence,
    PointerSourceEvidence,
    PointerUpdateEvidence,
    ProvenanceVerdict,
    RecordingEvidence,
    SourceRangeEvidence,
    SpeechAcquisitionEvidence,
    SpeechPhaseEvidence,
    SpeechPresentationOutcomeEvidence,
    SpeechPresentationRequestEvidence,
    SpeechProvenanceIndex,
    SpeechScheduleEvidence,
    SpeechTrialEvidence,
    TargetOnsetEvidence,
    TrialIdentityEvidence,
    WebGridProvenanceIndex,
    WebGridSelectionEvidence,
    WebGridTrialEvidence,
    center_out_provenance_from_records,
    webgrid_provenance_from_records,
)


def eid(kind: EvidenceKind, ordinal: int) -> EvidenceId:
    return EvidenceId(kind, ordinal)


def complete_recording(*raw: str) -> RecordingEvidence:
    return RecordingEvidence(
        session_id="session-1",
        complete=True,
        accounting_verified=True,
        trace_complete=True,
        raw_stream_ids=raw,
    )


def trial_identity(
    ordinal: int,
    *,
    key: int = 0,
    block: int = 0,
    target_id: int = 0,
    stimulus_id: int = 0,
) -> TrialIdentityEvidence:
    return TrialIdentityEvidence(ordinal, key, block, target_id, stimulus_id)


def center_out_index(
    recording: RecordingEvidence,
    *,
    faulted: bool = False,
) -> CenterOutProvenanceIndex:
    trial = trial_identity(4, key=44, block=2, target_id=3)
    source_a = SourceRangeEvidence(
        eid(EvidenceKind.SOURCE_RANGE, 1),
        "neural-a",
        2,
        400,
        800,
        20,
        23,
    )
    source_b = SourceRangeEvidence(
        eid(EvidenceKind.SOURCE_RANGE, 2),
        "neural-b",
        2,
        1000,
        1400,
    )
    feature = FeatureObservationEvidence(
        eid(EvidenceKind.FEATURE_OBSERVATION, 8),
        observation_index=8,
        time_ns=1_000_000,
        source_ranges=(source_a.id, source_b.id),
    )
    decoded = DecoderOutputEvidence(
        eid(EvidenceKind.DECODER_OUTPUT, 8),
        feature.id,
        output_index=8,
        time_ns=1_000_000,
        frame_sequence=27,
        sample_index=408,
    )
    guidance = GuidanceDecisionEvidence(
        eid(EvidenceKind.GUIDANCE_DECISION, 8), decoded.id, target_id=3
    )
    assistance = AssistanceDecisionEvidence(
        eid(EvidenceKind.ASSISTANCE_DECISION, 8), decoded.id, guidance.id, trial
    )
    state = ExperimentStateEvidence(
        eid(EvidenceKind.EXPERIMENT_STATE, 12), trial=trial, target_id=3, state_code=2
    )
    command = CenterOutCommandEvidence(
        eid(EvidenceKind.COMMAND_REQUEST, 41),
        decoded.id,
        state.id,
        generated_ns=1_000_000,
        trial=trial,
        guidance=guidance.id,
        assistance=assistance.id,
        guidance_applicable=True,
        assistance_configured=True,
    )
    application = CommandApplicationEvidence(
        eid(EvidenceKind.COMMAND_APPLICATION, 42),
        command.id,
        generated_ns=1_000_000,
        submitted_ns=1_000_010,
        application_code=2 if faulted else 1,
        status_code=17 if faulted else 0,
        trial=trial,
    )
    return CenterOutProvenanceIndex(
        recording,
        sources=(source_b, source_a),
        features=(feature,),
        decoder_outputs=(decoded,),
        guidance=(guidance,),
        assistance=(assistance,),
        states=(state,),
        commands=(command,),
        applications=(application,),
    )


def test_center_out_trace_preserves_ranges_and_frame_identity() -> None:
    idx = center_out_index(complete_recording("neural-a", "neural-b"))

    first = idx.resolve(41)
    second = idx.resolve(41)

    assert first == second
    assert first.verdict is ProvenanceVerdict.COMPLETE
    assert [source.stream_id for source in first.source_ranges] == ["neural-a", "neural-b"]
    assert first.source_ranges[0].first_frame_sequence == 20
    assert first.source_ranges[1].first_frame_sequence is None
    assert first.feature is not None and first.feature.observation_index == 8
    assert first.decoder_output is not None and first.decoder_output.output_index == 8
    assert first.guidance is not None
    assert first.assistance is not None
    assert first.experiment_state is not None and first.experiment_state.target_id == 3
    assert first.application is not None and first.application.request == first.request_id
    assert first.gaps == ()


def test_evidence_records_copy_sequences_and_are_immutable() -> None:
    source_refs = [eid(EvidenceKind.SOURCE_RANGE, 1)]
    feature = FeatureObservationEvidence(
        eid(EvidenceKind.FEATURE_OBSERVATION, 1),
        1,
        10,
        source_refs,  # type: ignore[arg-type]
    )
    source_refs.append(eid(EvidenceKind.SOURCE_RANGE, 2))

    assert feature.source_ranges == (eid(EvidenceKind.SOURCE_RANGE, 1),)
    with pytest.raises(FrozenInstanceError):
        feature.time_ns = 20  # type: ignore[misc]


def test_missing_raw_stream_is_partial_without_evidence() -> None:
    result = center_out_index(complete_recording("neural-a")).resolve(41)

    assert result.verdict is ProvenanceVerdict.PARTIAL
    assert [source.stream_id for source in result.source_ranges] == ["neural-a"]
    assert any(gap.reason is EvidenceGapReason.RAW_STREAM_NOT_RECORDED for gap in result.gaps)


def test_incomplete_recording_overrides_complete_trace() -> None:
    recording = RecordingEvidence(
        session_id="session-1",
        complete=False,
        accounting_verified=True,
        trace_complete=True,
        raw_stream_ids=("neural-a", "neural-b"),
    )
    result = center_out_index(recording).resolve(41)

    assert result.verdict is ProvenanceVerdict.INCOMPLETE
    assert result.application is not None
    assert result.gaps[0].reason is EvidenceGapReason.RECORDING_INCOMPLETE
    assert result.gaps[0].kind is EvidenceKind.RECORDING


def test_faulted_command_preserves_application_outcome() -> None:
    result = center_out_index(complete_recording("neural-a", "neural-b"), faulted=True).resolve(41)

    assert result.verdict is ProvenanceVerdict.FAULTED
    assert result.application is not None
    assert result.application.application_code == 2
    assert result.application.status_code == 17


def webgrid_index(*, correct: bool) -> WebGridProvenanceIndex:
    trial_identity_value = trial_identity(3, key=33, block=2, target_id=7)
    source = PointerSourceEvidence(eid(EvidenceKind.POINTER_SOURCE, 4), "mouse", 90)
    update = PointerUpdateEvidence(
        eid(EvidenceKind.POINTER_UPDATE, 9), 2_000, source.id, trial_identity_value
    )
    target = TargetOnsetEvidence(eid(EvidenceKind.TARGET_ONSET, 5), trial_identity_value, 7, 1_000)
    selection = WebGridSelectionEvidence(
        eid(EvidenceKind.SELECTION, 99),
        update.id,
        target.id,
        trial_identity_value,
        8,
        7 if correct else 6,
        7,
        correct,
        2_000,
        1_000,
    )
    trials = (
        (
            WebGridTrialEvidence(
                eid(EvidenceKind.TRIAL, 3),
                selection.id,
                target.id,
                trial_identity_value,
                8,
                1,
                7,
                7,
                True,
                1_000,
                1_000,
                True,
            ),
        )
        if correct
        else ()
    )
    metric = MetricContributionEvidence(
        eid(EvidenceKind.METRIC_CONTRIBUTION, 99),
        selection.id,
        correct_delta=int(correct),
        incorrect_delta=int(not correct),
        acquisition_ns=1_000 if correct else None,
    )
    return WebGridProvenanceIndex(
        complete_recording(),
        pointer_sources=(source,),
        pointer_updates=(update,),
        targets=(target,),
        selections=(selection,),
        trials=trials,
        metrics=(metric,),
    )


@pytest.mark.parametrize("correct", [False, True])
def test_webgrid_selection_trace_resolves_target_outcome_and_metric(correct: bool) -> None:
    result = webgrid_index(correct=correct).resolve(99)

    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert result.pointer_source is not None and result.pointer_source.source_name == "mouse"
    assert result.target is not None and result.target.target_id == 7
    assert result.selection is not None and result.selection.correct is correct
    assert (result.trial is not None) is correct
    assert result.metric is not None
    assert result.metric.correct_delta == int(correct)
    assert result.metric.incorrect_delta == int(not correct)


def test_webgrid_rejects_acquisition_contradiction() -> None:
    baseline = webgrid_index(correct=True).resolve(99)
    assert baseline.pointer_source is not None
    assert baseline.pointer_update is not None
    assert baseline.target is not None
    assert baseline.selection is not None
    assert baseline.trial is not None

    with pytest.raises(ValidationError, match="trial acquisition time"):
        WebGridProvenanceIndex(
            complete_recording(),
            pointer_sources=(baseline.pointer_source,),
            pointer_updates=(baseline.pointer_update,),
            targets=(baseline.target,),
            selections=(baseline.selection,),
            trials=(replace(baseline.trial, acquisition_ns=999),),
        ).resolve(99)


def speech_index(
    *,
    include_black_outcome: bool = True,
    include_content_outcome: bool = True,
    software_evidence: bool = False,
    software_evidence_required: bool = False,
) -> SpeechProvenanceIndex:
    trial = trial_identity(7, key=70, stimulus_id=12)
    schedule = SpeechScheduleEvidence(eid(EvidenceKind.SPEECH_SCHEDULE, 7), trial, 12)
    phase_a = SpeechPhaseEvidence(
        eid(EvidenceKind.SPEECH_PHASE, 71), schedule.id, phase_code=0, start_ns=100, end_ns=200
    )
    phase_b = SpeechPhaseEvidence(
        eid(EvidenceKind.SPEECH_PHASE, 72), schedule.id, phase_code=2, start_ns=200, end_ns=500
    )
    request_a = SpeechPresentationRequestEvidence(
        eid(EvidenceKind.PRESENTATION_REQUEST, 21), phase_a.id, trial, 90, 100, 0
    )
    request_b = SpeechPresentationRequestEvidence(
        eid(EvidenceKind.PRESENTATION_REQUEST, 22), phase_b.id, trial, 190, 200, 12
    )
    black_outcome = SpeechPresentationOutcomeEvidence(
        eid(EvidenceKind.PRESENTATION_OUTCOME, 31), request_a.id, trial, False, 90, 1, 113, 0
    )
    content_outcome = SpeechPresentationOutcomeEvidence(
        eid(EvidenceKind.PRESENTATION_OUTCOME, 32),
        request_b.id,
        trial,
        True,
        190,
        1,
        219,
        12,
        software_evidence_available=software_evidence,
        intended_ns=200 if software_evidence else None,
        submitted_renderer_ns=1_000 if software_evidence else None,
        submitted_ns=210 if software_evidence else None,
        presented_renderer_ns=1_009 if software_evidence else None,
        implementation_status=0 if software_evidence else None,
    )
    outcomes = (
        *((black_outcome,) if include_black_outcome else ()),
        *((content_outcome,) if include_content_outcome else ()),
    )
    marker_a = AcquisitionMarkerEvidence(eid(EvidenceKind.ACQUISITION_MARKER, 51), trial, 100, 1)
    marker_b = AcquisitionMarkerEvidence(eid(EvidenceKind.ACQUISITION_MARKER, 52), trial, 500, 2)
    acquisition = SpeechAcquisitionEvidence(
        eid(EvidenceKind.ACQUISITION_INTERVAL, 7), trial, 100, 500, (marker_a.id, marker_b.id)
    )
    trial_record = SpeechTrialEvidence(eid(EvidenceKind.TRIAL, 7), schedule.id, trial, 1)
    return SpeechProvenanceIndex(
        complete_recording(),
        schedules=(schedule,),
        phases=(phase_b, phase_a),
        requests=(request_b, request_a),
        outcomes=outcomes,
        acquisitions=(acquisition,),
        markers=(marker_b, marker_a),
        trials=(trial_record,),
        software_evidence_required=software_evidence_required,
    )


def test_speech_trace_requires_matched_content_outcome() -> None:
    result = speech_index().resolve(7)

    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert [phase.start_ns for phase in result.phases] == [100, 200]
    assert [request.intended_onset_ns for request in result.requests] == [100, 200]
    assert [outcome.actual_presented_ns for outcome in result.outcomes] == [219]
    assert all(outcome.matched_request for outcome in result.outcomes)
    assert result.acquisition is not None
    assert [marker.time_ns for marker in result.markers] == [100, 500]
    assert result.trial is not None and not result.trial.invalidated


def test_speech_software_timing_stays_distinct() -> None:
    complete = speech_index(software_evidence=True, software_evidence_required=True).resolve(7)
    outcome = complete.outcomes[0]

    assert complete.verdict is ProvenanceVerdict.COMPLETE
    assert outcome.requested_ns == 190
    assert outcome.intended_ns == 200
    assert outcome.submitted_ns == 210
    assert outcome.actual_presented_ns == 219
    assert outcome.submitted_renderer_ns == 1_000
    assert outcome.presented_renderer_ns == 1_009

    partial = speech_index(software_evidence_required=True).resolve(7)
    assert partial.verdict is ProvenanceVerdict.PARTIAL
    assert any(gap.kind is EvidenceKind.PRESENTATION_OUTCOME for gap in partial.gaps)


def test_speech_cues_use_native_stimulus_contract() -> None:
    trial = trial_identity(1, stimulus_id=12)
    schedule = SpeechScheduleEvidence(eid(EvidenceKind.SPEECH_SCHEDULE, 1), trial, 12)
    phases = tuple(
        SpeechPhaseEvidence(eid(EvidenceKind.SPEECH_PHASE, idx), schedule.id, code, start, stop)
        for idx, code, start, stop in ((1, 0, 0, 10), (2, 1, 10, 20), (3, 2, 20, 30))
    )
    requests = tuple(
        SpeechPresentationRequestEvidence(
            eid(EvidenceKind.PRESENTATION_REQUEST, idx),
            phase.id,
            trial,
            phase.start_ns,
            phase.start_ns,
            12 if phase.phase_code == 2 else 0,
        )
        for idx, phase in enumerate(phases, start=1)
    )
    result = SpeechProvenanceIndex(
        complete_recording(),
        schedules=(schedule,),
        phases=phases,
        requests=requests,
        trials=(SpeechTrialEvidence(eid(EvidenceKind.TRIAL, 1), schedule.id, trial, 1),),
    ).resolve(1)

    assert [request.stimulus_id for request in result.requests] == [0, 0, 12]


@pytest.mark.parametrize("phase_codes", [(0, 2), (0, 1, 2), (0, 2, 3), (0, 1, 2, 3)])
def test_speech_accepts_only_each_declared_phase_topology(
    phase_codes: tuple[int, ...],
) -> None:
    trial = trial_identity(1, stimulus_id=12)
    schedule = SpeechScheduleEvidence(eid(EvidenceKind.SPEECH_SCHEDULE, 1), trial, 12)
    phases = tuple(
        SpeechPhaseEvidence(
            eid(EvidenceKind.SPEECH_PHASE, idx + 1),
            schedule.id,
            code,
            idx * 10,
            (idx + 1) * 10,
        )
        for idx, code in enumerate(phase_codes)
    )
    requests = tuple(
        SpeechPresentationRequestEvidence(
            eid(EvidenceKind.PRESENTATION_REQUEST, idx + 1),
            phase.id,
            trial,
            phase.start_ns,
            phase.start_ns,
            12 if phase.phase_code == 2 else 0,
        )
        for idx, phase in enumerate(phases)
        if phase.phase_code != 3
    )

    result = SpeechProvenanceIndex(
        complete_recording(), schedules=(schedule,), phases=phases, requests=requests
    ).resolve(1)

    assert tuple(phase.phase_code for phase in result.phases) == phase_codes


def test_speech_rejects_inter_trial_and_impossible_times() -> None:
    schedule = SpeechScheduleEvidence(
        eid(EvidenceKind.SPEECH_SCHEDULE, 1), trial_identity(1, stimulus_id=12), 12
    )
    black = SpeechPhaseEvidence(eid(EvidenceKind.SPEECH_PHASE, 1), schedule.id, 0, 0, 10)
    content = SpeechPhaseEvidence(eid(EvidenceKind.SPEECH_PHASE, 2), schedule.id, 2, 10, 20)
    inter_trial = SpeechPhaseEvidence(eid(EvidenceKind.SPEECH_PHASE, 3), schedule.id, 3, 20, 30)
    requests = (
        SpeechPresentationRequestEvidence(
            eid(EvidenceKind.PRESENTATION_REQUEST, 1), black.id, schedule.trial, 0, 0, 0
        ),
        SpeechPresentationRequestEvidence(
            eid(EvidenceKind.PRESENTATION_REQUEST, 2), content.id, schedule.trial, 10, 10, 12
        ),
        SpeechPresentationRequestEvidence(
            eid(EvidenceKind.PRESENTATION_REQUEST, 3),
            inter_trial.id,
            schedule.trial,
            20,
            20,
            0,
        ),
    )
    with pytest.raises(ValidationError, match="INTER_TRIAL"):
        SpeechProvenanceIndex(
            complete_recording(),
            schedules=(schedule,),
            phases=(black, content, inter_trial),
            requests=requests,
        ).resolve(1)

    content_request = requests[1]
    outcome = SpeechPresentationOutcomeEvidence(
        eid(EvidenceKind.PRESENTATION_OUTCOME, 2),
        content_request.id,
        content_request.trial,
        True,
        10,
        1,
        9,
        12,
    )
    with pytest.raises(ValidationError, match="precedes its request"):
        SpeechProvenanceIndex(
            complete_recording(),
            schedules=(schedule,),
            phases=(black, content),
            requests=requests[:2],
            outcomes=(outcome,),
        ).resolve(1)


def test_speech_unmatched_black_report_keeps_trace_complete() -> None:
    with_report = speech_index(include_black_outcome=True).resolve(7)
    without_report = speech_index(include_black_outcome=False).resolve(7)

    assert with_report.verdict is ProvenanceVerdict.COMPLETE
    assert without_report.verdict is ProvenanceVerdict.COMPLETE
    assert with_report.outcomes == without_report.outcomes


def test_speech_unmatched_cross_report_keeps_trace_complete() -> None:
    baseline = speech_index().resolve(7)
    assert baseline.schedule is not None
    assert baseline.acquisition is not None
    assert baseline.trial is not None
    black, content = baseline.phases
    black_request, content_request = baseline.requests
    cross = SpeechPhaseEvidence(
        eid(EvidenceKind.SPEECH_PHASE, 73), baseline.schedule.id, 1, 200, 250
    )
    content = replace(content, start_ns=250)
    cross_request = SpeechPresentationRequestEvidence(
        eid(EvidenceKind.PRESENTATION_REQUEST, 23), cross.id, baseline.schedule.trial, 190, 200, 0
    )
    content_request = replace(
        content_request,
        phase=content.id,
        requested_ns=240,
        intended_onset_ns=250,
    )
    cross_outcome = SpeechPresentationOutcomeEvidence(
        eid(EvidenceKind.PRESENTATION_OUTCOME, 33),
        cross_request.id,
        baseline.schedule.trial,
        False,
        190,
        1,
        213,
        0,
    )
    content_outcome = replace(
        baseline.outcomes[0],
        request=content_request.id,
        requested_ns=240,
        actual_presented_ns=269,
    )

    result = SpeechProvenanceIndex(
        complete_recording(),
        schedules=(baseline.schedule,),
        phases=(black, cross, content),
        requests=(black_request, cross_request, content_request),
        outcomes=(cross_outcome, content_outcome),
        acquisitions=(baseline.acquisition,),
        markers=baseline.markers,
        trials=(baseline.trial,),
    ).resolve(7)

    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert result.outcomes == (content_outcome,)


def test_speech_missing_matched_content_outcome_is_partial() -> None:
    result = speech_index(include_content_outcome=False).resolve(7)

    assert result.verdict is ProvenanceVerdict.PARTIAL
    assert result.outcomes == ()
    assert sum(gap.reason is EvidenceGapReason.LINK_OMITTED for gap in result.gaps) == 1


def test_speech_unmatched_report_cannot_support_request() -> None:
    baseline = speech_index().resolve(7)
    assert baseline.schedule is not None
    assert baseline.acquisition is not None
    assert baseline.trial is not None
    unmatched = replace(
        baseline.outcomes[0],
        id=eid(EvidenceKind.PRESENTATION_OUTCOME, 99),
        matched_request=False,
    )
    common = {
        "schedules": (baseline.schedule,),
        "phases": baseline.phases,
        "requests": baseline.requests,
        "acquisitions": (baseline.acquisition,),
        "markers": baseline.markers,
        "trials": (baseline.trial,),
    }

    partial = SpeechProvenanceIndex(
        complete_recording(),
        outcomes=(unmatched,),
        **common,
    ).resolve(7)
    assert partial.verdict is ProvenanceVerdict.PARTIAL
    assert partial.outcomes == ()
    assert any(
        gap.kind is EvidenceKind.PRESENTATION_OUTCOME
        and gap.reason is EvidenceGapReason.LINK_OMITTED
        for gap in partial.gaps
    )

    complete = SpeechProvenanceIndex(
        complete_recording(),
        outcomes=(unmatched, *baseline.outcomes),
        **common,
    ).resolve(7)
    assert complete.verdict is ProvenanceVerdict.COMPLETE
    assert complete.outcomes == baseline.outcomes


def test_speech_matched_outcome_trial_must_corroborate_request() -> None:
    baseline = speech_index().resolve(7)
    assert baseline.schedule is not None
    contradictory = replace(
        baseline.outcomes[0],
        trial=replace(baseline.outcomes[0].trial, key=999),
    )

    with pytest.raises(ValidationError, match="outcome trial contradicts request"):
        SpeechProvenanceIndex(
            complete_recording(),
            schedules=(baseline.schedule,),
            phases=baseline.phases,
            requests=baseline.requests,
            outcomes=(contradictory,),
        ).resolve(7)


def test_speech_rejects_illegal_phase_topology() -> None:
    trial = trial_identity(1, stimulus_id=12)
    schedule = SpeechScheduleEvidence(eid(EvidenceKind.SPEECH_SCHEDULE, 1), trial, 12)
    content = SpeechPhaseEvidence(eid(EvidenceKind.SPEECH_PHASE, 1), schedule.id, 2, 0, 10)
    request = SpeechPresentationRequestEvidence(
        eid(EvidenceKind.PRESENTATION_REQUEST, 1), content.id, trial, 0, 0, 12
    )

    with pytest.raises(ValidationError, match="phase topology"):
        SpeechProvenanceIndex(
            complete_recording(),
            schedules=(schedule,),
            phases=(content,),
            requests=(request,),
        ).resolve(1)

    with pytest.raises(ValidationError, match="must have outcome ABORTED"):
        SpeechTrialEvidence(eid(EvidenceKind.TRIAL, 1), schedule.id, trial, 1, True)


def test_speech_trial_lookup_includes_block_identity() -> None:
    first = trial_identity(1, block=0, stimulus_id=12)
    second = trial_identity(1, block=2, stimulus_id=13)
    schedules = (
        SpeechScheduleEvidence(eid(EvidenceKind.SPEECH_SCHEDULE, 1), first, 12),
        SpeechScheduleEvidence(eid(EvidenceKind.SPEECH_SCHEDULE, 2), second, 13),
    )
    idx = SpeechProvenanceIndex(complete_recording(), schedules=schedules)

    assert first.same_trial(trial_identity(1, key=99, block=0, stimulus_id=99))
    assert not first.same_trial(second)
    assert idx.resolve(1).trial_identity == first
    assert idx.resolve(1, block=2).trial_identity == second


def test_speech_rejects_acquisition_identity_contradiction() -> None:
    trial = trial_identity(1, key=2, stimulus_id=13)
    acquisition_trial = trial_identity(1, key=1, stimulus_id=12)
    missing_schedule = eid(EvidenceKind.SPEECH_SCHEDULE, 1)

    with pytest.raises(ValidationError, match="trial and acquisition identities"):
        SpeechProvenanceIndex(
            complete_recording(),
            trials=(SpeechTrialEvidence(eid(EvidenceKind.TRIAL, 1), missing_schedule, trial, 1),),
            acquisitions=(
                SpeechAcquisitionEvidence(
                    eid(EvidenceKind.ACQUISITION_INTERVAL, 1),
                    acquisition_trial,
                    0,
                    10,
                ),
            ),
        ).resolve(1)


def test_invalid_cross_record_claims_are_rejected() -> None:
    idx = webgrid_index(correct=True)
    with pytest.raises(ValidationError, match="contradicts the active target"):
        trial = trial_identity(0, target_id=3)
        source = PointerSourceEvidence(eid(EvidenceKind.POINTER_SOURCE, 1), "mouse", 1)
        update = PointerUpdateEvidence(eid(EvidenceKind.POINTER_UPDATE, 1), 1, source.id, trial)
        target = TargetOnsetEvidence(eid(EvidenceKind.TARGET_ONSET, 1), trial, 3, 0)
        selection = WebGridSelectionEvidence(
            eid(EvidenceKind.SELECTION, 1), update.id, target.id, trial, 1, 2, 3, True, 1, 0
        )
        WebGridProvenanceIndex(
            complete_recording(),
            pointer_sources=(source,),
            pointer_updates=(update,),
            targets=(target,),
            selections=(selection,),
            metrics=(
                MetricContributionEvidence(
                    eid(EvidenceKind.METRIC_CONTRIBUTION, 1), selection.id, 1, 0, 1
                ),
            ),
        ).resolve(1)
    assert idx.resolve(99).verdict is ProvenanceVerdict.COMPLETE


def test_webgrid_rejects_completed_trial_on_incorrect_selection() -> None:
    trial_identity_value = trial_identity(4, target_id=7)
    source = PointerSourceEvidence(eid(EvidenceKind.POINTER_SOURCE, 1), "mouse", 1)
    update = PointerUpdateEvidence(
        eid(EvidenceKind.POINTER_UPDATE, 1), 20, source.id, trial_identity_value
    )
    target = TargetOnsetEvidence(eid(EvidenceKind.TARGET_ONSET, 1), trial_identity_value, 7, 10)
    selection = WebGridSelectionEvidence(
        eid(EvidenceKind.SELECTION, 1),
        update.id,
        target.id,
        trial_identity_value,
        1,
        6,
        7,
        False,
        20,
        10,
    )
    trial = WebGridTrialEvidence(
        eid(EvidenceKind.TRIAL, 4),
        selection.id,
        target.id,
        trial_identity_value,
        1,
        1,
        6,
        7,
        False,
        10,
        10,
        True,
    )
    metric = MetricContributionEvidence(
        eid(EvidenceKind.METRIC_CONTRIBUTION, 1), selection.id, 0, 1, None
    )
    with pytest.raises(ValidationError, match="incorrect selection cannot complete"):
        WebGridProvenanceIndex(
            complete_recording(),
            pointer_sources=(source,),
            pointer_updates=(update,),
            targets=(target,),
            selections=(selection,),
            trials=(trial,),
            metrics=(metric,),
        ).resolve(1)


def test_webgrid_preserves_invalidated_measurement() -> None:
    trial_identity_value = trial_identity(4, target_id=7)
    source = PointerSourceEvidence(eid(EvidenceKind.POINTER_SOURCE, 1), "mouse", 1)
    update = PointerUpdateEvidence(
        eid(EvidenceKind.POINTER_UPDATE, 1), 20, source.id, trial_identity_value
    )
    target = TargetOnsetEvidence(eid(EvidenceKind.TARGET_ONSET, 1), trial_identity_value, 7, 10)
    selection = WebGridSelectionEvidence(
        eid(EvidenceKind.SELECTION, 1),
        update.id,
        target.id,
        trial_identity_value,
        1,
        7,
        7,
        True,
        20,
        10,
        False,
    )
    result = WebGridProvenanceIndex(
        complete_recording(),
        pointer_sources=(source,),
        pointer_updates=(update,),
        targets=(target,),
        selections=(selection,),
        trials=(
            WebGridTrialEvidence(
                eid(EvidenceKind.TRIAL, 4),
                selection.id,
                target.id,
                trial_identity_value,
                1,
                4,
                7,
                7,
                True,
                10,
                10,
                False,
            ),
        ),
        metrics=(
            MetricContributionEvidence(
                eid(EvidenceKind.METRIC_CONTRIBUTION, 1), selection.id, 1, 0, 10
            ),
        ),
    ).resolve(1)

    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert result.selection is not None and not result.selection.acquisition_timing_valid
    assert result.trial is not None and result.trial.outcome_code == 4
    assert result.metric is not None and result.metric.acquisition_ns == 10


def test_center_out_rejects_early_application_and_unknown_enums() -> None:
    idx = center_out_index(complete_recording("neural-a", "neural-b"))
    command = idx.resolve(41).command
    assert command is not None
    with pytest.raises(ValidationError, match="precedes command generation"):
        CenterOutProvenanceIndex(
            complete_recording(),
            commands=(command,),
            applications=(
                CommandApplicationEvidence(
                    eid(EvidenceKind.COMMAND_APPLICATION, 1),
                    command.id,
                    1_000_000,
                    999_999,
                    1,
                    0,
                    command.trial,
                ),
            ),
        ).resolve(41)
    with pytest.raises(ValidationError, match="declared CommandApplication"):
        CommandApplicationEvidence(
            eid(EvidenceKind.COMMAND_APPLICATION, 1),
            command.id,
            1_000_000,
            1_000_000,
            255,
            0,
            command.trial,
        )


def test_interpreted_shared_enum_codes_reject_undeclared_values() -> None:
    schedule = SpeechScheduleEvidence(
        eid(EvidenceKind.SPEECH_SCHEDULE, 1), trial_identity(1, stimulus_id=12), 12
    )
    with pytest.raises(ValidationError, match="declared SpeechPhase"):
        SpeechPhaseEvidence(eid(EvidenceKind.SPEECH_PHASE, 1), schedule.id, 255, 0, 1)
    with pytest.raises(ValidationError, match="declared PresentationStatus"):
        SpeechPresentationOutcomeEvidence(
            eid(EvidenceKind.PRESENTATION_OUTCOME, 1),
            eid(EvidenceKind.PRESENTATION_REQUEST, 1),
            schedule.trial,
            True,
            0,
            255,
            None,
            0,
        )
    with pytest.raises(ValidationError, match="declared TrialOutcome"):
        WebGridTrialEvidence(
            eid(EvidenceKind.TRIAL, 1),
            eid(EvidenceKind.SELECTION, 1),
            eid(EvidenceKind.TARGET_ONSET, 1),
            trial_identity(1, target_id=1),
            1,
            255,
            1,
            1,
            True,
            0,
            0,
            True,
        )


def _row(name: str, time_ns: int, body: dict[str, object]) -> dict[str, object]:
    return {"name": name, "time_ns": time_ns, "text": json.dumps(body)}


def test_center_out_rows_keep_explicit_parent_links() -> None:
    source = SourceRangeEvidence(eid(EvidenceKind.SOURCE_RANGE, 0), "neural", 0, 0, 4)
    feature = FeatureObservationEvidence(
        eid(EvidenceKind.FEATURE_OBSERVATION, 0), 0, 100, (source.id,)
    )
    decoded = DecoderOutputEvidence(eid(EvidenceKind.DECODER_OUTPUT, 0), feature.id, 0, 100, 5, 9)
    trial = {"ordinal": 0, "key": 0, "block": 0, "target_id": 3, "stimulus_id": 0}
    records = (
        _row(
            "center_out.transition",
            0,
            {"sequence": 4, "to_state": 2, "active_target": 3, "trial": trial},
        ),
        _row(
            "center_out.observation_provenance",
            100,
            {
                "observation_ordinal": 0,
                "decoder_output_ordinal": 0,
                "frame_sequence": 5,
                "sample_index": 9,
                "state_sequence": 4,
                "guidance_ordinal": 0,
                "assistance_ordinal": 0,
                "guidance_applicable": True,
                "assistance_configured": True,
            },
        ),
        _row(
            "center_out.guidance_sample",
            100,
            {
                "observation_ordinal": 0,
                "decoder_output_ordinal": 0,
                "guidance_applicable": True,
                "state_target_id": 3,
            },
        ),
        _row(
            "center_out.assisted_velocity",
            100,
            {
                "observation_ordinal": 0,
                "decoder_output_ordinal": 0,
                "guidance_ordinal": 0,
                "guidance_applicable": True,
                "assistance_configured": True,
                "trial": trial,
            },
        ),
        _row(
            "center_out.command",
            100,
            {
                "sequence": 0,
                "observation_ordinal": 0,
                "decoder_output_ordinal": 0,
                "state_sequence": 4,
                "guidance_ordinal": 0,
                "assistance_ordinal": 0,
                "guidance_applicable": True,
                "assistance_configured": True,
                "generated_ns": 100,
                "trial": trial,
            },
        ),
        _row(
            "center_out.command_outcome",
            100,
            {
                "sequence": 0,
                "request_sequence": 0,
                "generated_ns": 100,
                "submitted_ns": 100,
                "application": 1,
                "status_code": 0,
                "application_scope": "center_out.headless_cursor",
                "trial": trial,
            },
        ),
        _row(
            "center_out.transition",
            100,
            {"sequence": 5, "to_state": 3, "active_target": 2, "trial": trial},
        ),
    )
    result = center_out_provenance_from_records(
        complete_recording("neural"),
        records,
        sources=(source,),
        features=(feature,),
        decoder_outputs=(decoded,),
    ).resolve(0)

    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert result.command is not None and result.command.decoder_output == decoded.id
    assert result.application is not None and result.application.request == result.command.id
    assert result.application.application_scope == "center_out.headless_cursor"
    assert result.experiment_state is not None and result.experiment_state.target_id == 3

    missing_presentation = center_out_provenance_from_records(
        complete_recording("neural"),
        records,
        sources=(source,),
        features=(feature,),
        decoder_outputs=(decoded,),
        presentation_evidence_required=True,
    ).resolve(0)
    assert missing_presentation.verdict is ProvenanceVerdict.PARTIAL
    assert any(
        gap.kind is EvidenceKind.PRESENTATION_OUTCOME
        and gap.reason is EvidenceGapReason.LINK_OMITTED
        for gap in missing_presentation.gaps
    )

    with_presentation = (
        *records,
        _row(
            "center_out.presentation",
            130,
            {
                "event": 2,
                "has_software_times": True,
                "has_source_ordinal": True,
                "source_ordinal": 0,
                "has_trial": True,
                "trial": trial,
                "presented_ns": 130,
            },
        ),
    )
    presented = center_out_provenance_from_records(
        complete_recording("neural"),
        with_presentation,
        sources=(source,),
        features=(feature,),
        decoder_outputs=(decoded,),
        presentation_evidence_required=True,
    ).resolve(0)
    assert presented.verdict is ProvenanceVerdict.COMPLETE

    mismatched_presentation = (
        *records,
        _row(
            "center_out.presentation",
            130,
            {
                "event": 2,
                "has_software_times": True,
                "has_source_ordinal": True,
                "source_ordinal": 999,
                "has_trial": True,
                "trial": trial,
            },
        ),
    )
    mismatch = center_out_provenance_from_records(
        complete_recording("neural"),
        mismatched_presentation,
        sources=(source,),
        features=(feature,),
        decoder_outputs=(decoded,),
        presentation_evidence_required=True,
    ).resolve(0)
    assert mismatch.verdict is ProvenanceVerdict.PARTIAL
    assert any(gap.kind is EvidenceKind.PRESENTATION_OUTCOME for gap in mismatch.gaps)

    omitted_source = (
        *records,
        _row(
            "center_out.presentation",
            130,
            {"event": 2, "has_software_times": True, "has_source_ordinal": False},
        ),
    )
    omitted = center_out_provenance_from_records(
        complete_recording("neural"),
        omitted_source,
        sources=(source,),
        features=(feature,),
        decoder_outputs=(decoded,),
        presentation_evidence_required=True,
    ).resolve(0)
    assert omitted.verdict is ProvenanceVerdict.PARTIAL

    wrong_trial = (
        *records,
        _row(
            "center_out.presentation",
            130,
            {
                "event": 2,
                "has_software_times": True,
                "has_source_ordinal": True,
                "source_ordinal": 0,
                "has_trial": True,
                "trial": {**trial, "ordinal": 1},
            },
        ),
    )
    with pytest.raises(ValidationError, match="presentation trial contradicts"):
        center_out_provenance_from_records(
            complete_recording("neural"),
            wrong_trial,
            sources=(source,),
            features=(feature,),
            decoder_outputs=(decoded,),
            presentation_evidence_required=True,
        )

    post_step_link = tuple(
        (
            _row(
                "center_out.observation_provenance",
                100,
                {**json.loads(str(row["text"])), "state_sequence": 5},
            )
            if row.get("name") == "center_out.observation_provenance"
            else row
        )
        for row in records
    )
    with pytest.raises(ValidationError, match="guidance target contradicts"):
        center_out_provenance_from_records(
            complete_recording("neural"),
            post_step_link,
            sources=(source,),
            features=(feature,),
            decoder_outputs=(decoded,),
        ).resolve(0)

    without_explicit_link = center_out_provenance_from_records(
        complete_recording("neural"),
        tuple(row for row in records if row["name"] != "center_out.observation_provenance"),
        sources=(source,),
        features=(feature,),
        decoder_outputs=(decoded,),
    ).resolve(0)
    assert without_explicit_link.verdict is ProvenanceVerdict.PARTIAL
    assert without_explicit_link.command is None

    mismatched_decoder = DecoderOutputEvidence(
        decoded.id,
        decoded.feature,
        decoded.output_index,
        decoded.time_ns,
        decoded.frame_sequence,
        999,
    )
    with pytest.raises(ValidationError, match="decoder frame/sample identity"):
        center_out_provenance_from_records(
            complete_recording("neural"),
            records,
            sources=(source,),
            features=(feature,),
            decoder_outputs=(mismatched_decoder,),
        )

    for record_name, changed_time, message in (
        ("center_out.observation_provenance", 999, "decoder frame/sample identity or time"),
        ("center_out.command", 777, "command row time"),
        ("center_out.command_outcome", 888, "outcome row time"),
    ):
        mutated = tuple(
            ({**row, "time_ns": changed_time} if row.get("name") == record_name else row)
            for row in records
        )
        with pytest.raises(ValidationError, match=message):
            center_out_provenance_from_records(
                complete_recording("neural"),
                mutated,
                sources=(source,),
                features=(feature,),
                decoder_outputs=(decoded,),
            ).resolve(0)

    different_trial = {**trial, "key": 99}
    for record_name in ("center_out.command", "center_out.command_outcome"):
        mutated = tuple(
            (
                _row(
                    str(row["name"]),
                    int(row["time_ns"]),
                    {**json.loads(str(row["text"])), "trial": different_trial},
                )
                if row.get("name") == record_name
                else row
            )
            for row in records
        )
        with pytest.raises(ValidationError, match="trial identit"):
            center_out_provenance_from_records(
                complete_recording("neural"),
                mutated,
                sources=(source,),
                features=(feature,),
                decoder_outputs=(decoded,),
            ).resolve(0)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("guidance_ordinal", "guidance parent ordinal contradicts child identity"),
        ("assistance_ordinal", "assistance parent ordinal contradicts child identity"),
    ),
)
def test_center_out_child_identity_cannot_be_renamed_by_parent_link(
    field: str, message: str
) -> None:
    trial = {"ordinal": 0, "key": 0, "block": 0, "target_id": 3, "stimulus_id": 0}
    provenance = {
        "observation_ordinal": 0,
        "decoder_output_ordinal": 0,
        "frame_sequence": 5,
        "sample_index": 9,
        "state_sequence": 4,
        "guidance_ordinal": 0,
        "assistance_ordinal": 0,
        "guidance_applicable": True,
        "assistance_configured": True,
    }
    provenance[field] = 1
    records = (
        _row("center_out.observation_provenance", 100, provenance),
        _row(
            "center_out.guidance_sample",
            100,
            {
                "observation_ordinal": 0,
                "decoder_output_ordinal": 0,
                "guidance_applicable": True,
                "state_target_id": 3,
            },
        ),
        _row(
            "center_out.assisted_velocity",
            100,
            {
                "observation_ordinal": 0,
                "decoder_output_ordinal": 0,
                "guidance_ordinal": 0,
                "guidance_applicable": True,
                "assistance_configured": True,
                "trial": trial,
            },
        ),
    )

    with pytest.raises(ValidationError, match=message):
        center_out_provenance_from_records(complete_recording(), records)


def test_webgrid_rows_preserve_pointer_and_trial_links() -> None:
    trial = {"ordinal": 2, "key": 0, "block": 0, "target_id": 7, "stimulus_id": 0}
    records = (
        _row(
            "webgrid.pointer_source",
            0,
            {"recorded_here": True, "stream": "", "source_ordinal": 0},
        ),
        _row(
            "webgrid.target_onset",
            10,
            {"target_onset_ordinal": 3, "target_id": 7, "trial": trial},
        ),
        _row(
            "webgrid.pointer",
            20,
            {
                "pointer_update_ordinal": 5,
                "source_ordinal": 0,
                "stream": "",
                "trial": trial,
            },
        ),
        _row(
            "webgrid.selection_correct",
            20,
            {
                "sequence": 9,
                "pointer_update_ordinal": 5,
                "target_onset_ordinal": 3,
                "paradigm": 8,
                "correct": True,
                "selected_id": 7,
                "intended_id": 7,
                "target_onset_ns": 10,
                "elapsed_since_target_onset_ns": 10,
                "acquisition_timing_valid": False,
                "trial": trial,
            },
        ),
        {
            "label": json.dumps(
                {
                    "trial": trial,
                    "selection_sequence": 9,
                    "target_onset_ordinal": 3,
                    "paradigm": 8,
                    "acquisition_ns": 10,
                    "selected_id": 7,
                    "intended_id": 7,
                    "correct": True,
                    "target_onset_ns": 10,
                    "acquisition_timing_valid": False,
                }
            ),
            "outcome": json.dumps({"outcome": 4, "reason": 0}),
        },
    )
    result = webgrid_provenance_from_records(complete_recording(), records).resolve(9)

    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert result.pointer_update is not None and result.pointer_update.id.ordinal == 5
    assert result.target is not None and result.target.id.ordinal == 3
    assert result.selection is not None and not result.selection.acquisition_timing_valid
    assert result.trial is not None and result.trial.outcome_code == 4

    for field, value in (
        ("target_onset_ordinal", 999),
        ("selected_id", 999),
        ("intended_id", 999),
        ("correct", False),
        ("target_onset_ns", 999),
        ("acquisition_ns", 999),
        ("acquisition_timing_valid", True),
        ("paradigm", 999),
    ):
        mutated = tuple(
            (
                {**row, "label": json.dumps({**json.loads(str(row["label"])), field: value})}
                if "label" in row
                else row
            )
            for row in records
        )
        with pytest.raises(ValidationError):
            webgrid_provenance_from_records(complete_recording(), mutated).resolve(9)

    mutated_trial = tuple(
        (
            {
                **row,
                "label": json.dumps(
                    {
                        **json.loads(str(row["label"])),
                        "trial": {**trial, "key": 99},
                    }
                ),
            }
            if "label" in row
            else row
        )
        for row in records
    )
    with pytest.raises(ValidationError, match="trial identity"):
        webgrid_provenance_from_records(complete_recording(), mutated_trial).resolve(9)

    next_snapshot_trial = {**trial, "ordinal": 3, "key": 1, "target_id": 2}
    post_step_pointer = tuple(
        (
            _row(
                str(row["name"]),
                int(row["time_ns"]),
                {**json.loads(str(row["text"])), "trial": next_snapshot_trial},
            )
            if row.get("name") == "webgrid.pointer"
            else row
        )
        for row in records
    )
    post_step_result = webgrid_provenance_from_records(
        complete_recording(), post_step_pointer
    ).resolve(9)
    assert post_step_result.verdict is ProvenanceVerdict.COMPLETE
    assert post_step_result.pointer_update is not None
    assert post_step_result.pointer_update.trial.ordinal == 3


def test_webgrid_input_identity_is_queryable() -> None:
    trial = {"ordinal": 0, "key": 0, "block": 0, "target_id": 1, "stimulus_id": 0}
    pointer = {
        "pointer_update_ordinal": 0,
        "source_ordinal": 0,
        "stream": "",
        "trial": trial,
        "presentation_input_available": True,
        "presentation_input_ordinal": 17,
        "renderer_time_ns": 900,
        "presentation_input_kind": 1,
        "inside_presentation": True,
        "button": 0,
        "modifiers": 2,
    }
    records = (
        _row(
            "webgrid.pointer_source",
            0,
            {"recorded_here": True, "stream": "", "source_ordinal": 0},
        ),
        _row(
            "webgrid.target_onset",
            10,
            {"target_onset_ordinal": 0, "target_id": 1, "trial": trial},
        ),
        _row("webgrid.pointer", 20, pointer),
        _row(
            "webgrid.selection_incorrect",
            20,
            {
                "sequence": 4,
                "pointer_update_ordinal": 0,
                "target_onset_ordinal": 0,
                "paradigm": 3,
                "correct": False,
                "selected_id": 0,
                "intended_id": 1,
                "target_onset_ns": 10,
                "elapsed_since_target_onset_ns": 10,
                "acquisition_timing_valid": True,
                "trial": trial,
            },
        ),
    )

    result = webgrid_provenance_from_records(
        complete_recording(), records, presentation_input_required=True
    ).resolve(4)
    assert result.verdict is ProvenanceVerdict.COMPLETE
    assert result.pointer_update is not None
    assert result.pointer_update.presentation_input_ordinal == 17
    assert result.pointer_update.renderer_time_ns == 900
    assert result.pointer_update.presentation_input_kind == 1

    missing_render = webgrid_provenance_from_records(
        complete_recording(), records, presentation_evidence_required=True
    ).resolve(4)
    assert missing_render.verdict is ProvenanceVerdict.PARTIAL
    rendered_records = (
        *records,
        _row(
            "webgrid.presentation",
            25,
            {
                "event": 2,
                "has_software_times": True,
                "has_source_ordinal": True,
                "source_ordinal": 0,
                "has_trial": True,
                "trial": trial,
            },
        ),
    )
    rendered = webgrid_provenance_from_records(
        complete_recording(), rendered_records, presentation_evidence_required=True
    ).resolve(4)
    assert rendered.verdict is ProvenanceVerdict.COMPLETE

    mismatched_render = (
        *records,
        _row(
            "webgrid.presentation",
            25,
            {
                "event": 2,
                "has_software_times": True,
                "has_source_ordinal": True,
                "source_ordinal": 999,
                "has_trial": True,
                "trial": trial,
            },
        ),
    )
    mismatched = webgrid_provenance_from_records(
        complete_recording(), mismatched_render, presentation_evidence_required=True
    ).resolve(4)
    assert mismatched.verdict is ProvenanceVerdict.PARTIAL

    without_input = tuple(
        _row(
            "webgrid.pointer",
            20,
            {
                "pointer_update_ordinal": 0,
                "source_ordinal": 0,
                "stream": "",
                "trial": trial,
                "presentation_input_available": False,
            },
        )
        if row.get("name") == "webgrid.pointer"
        else row
        for row in records
    )
    partial = webgrid_provenance_from_records(
        complete_recording(), without_input, presentation_input_required=True
    ).resolve(4)
    assert partial.verdict is ProvenanceVerdict.PARTIAL
    assert any(gap.kind is EvidenceKind.PRESENTATION_INPUT for gap in partial.gaps)


@pytest.mark.parametrize(
    ("record_name", "correct"),
    (("webgrid.selection_correct", False), ("webgrid.selection_incorrect", True)),
)
def test_webgrid_rejects_name_correctness_contradiction(record_name: str, correct: bool) -> None:
    trial = {"ordinal": 2, "key": 0, "block": 0, "target_id": 7, "stimulus_id": 0}
    records = (
        _row(
            "webgrid.pointer_source",
            0,
            {"recorded_here": True, "stream": "", "source_ordinal": 0},
        ),
        _row(
            "webgrid.target_onset",
            10,
            {"target_onset_ordinal": 3, "target_id": 7, "trial": trial},
        ),
        _row(
            "webgrid.pointer",
            20,
            {
                "pointer_update_ordinal": 5,
                "source_ordinal": 0,
                "stream": "",
                "trial": trial,
            },
        ),
        _row(
            record_name,
            20,
            {
                "sequence": 9,
                "pointer_update_ordinal": 5,
                "target_onset_ordinal": 3,
                "paradigm": 8,
                "correct": correct,
                "selected_id": 7 if correct else 6,
                "intended_id": 7,
                "target_onset_ns": 10,
                "elapsed_since_target_onset_ns": 10,
                "acquisition_timing_valid": True,
                "trial": trial,
            },
        ),
    )

    with pytest.raises(ValidationError, match="record name contradicts its correctness"):
        webgrid_provenance_from_records(complete_recording(), records)


@pytest.mark.parametrize(
    ("source_body", "pointer_name", "pointer_stream", "message"),
    (
        (
            {"recorded_here": True, "stream": "cursor", "source_ordinal": 0},
            None,
            None,
            "embedded pointer sources require an empty stream",
        ),
        (
            {"recorded_here": True, "stream": "", "source_ordinal": 0},
            "webgrid.pointer",
            "cursor",
            "pointer row name contradicts its stream field",
        ),
        (
            {"recorded_here": False, "stream": "cursor", "source_ordinal": 0},
            "webgrid.pointer_reference",
            "",
            "pointer row name contradicts its stream field",
        ),
    ),
)
def test_webgrid_rejects_pointer_mode_and_stream_conflicts(
    source_body: dict[str, object],
    pointer_name: str | None,
    pointer_stream: str | None,
    message: str,
) -> None:
    records = [_row("webgrid.pointer_source", 0, source_body)]
    if pointer_name is not None:
        records.append(
            _row(
                pointer_name,
                10,
                {
                    "pointer_update_ordinal": 1,
                    "source_ordinal": 0,
                    "stream": pointer_stream,
                    "trial": {
                        "ordinal": 0,
                        "key": 0,
                        "block": 0,
                        "target_id": 1,
                        "stimulus_id": 0,
                    },
                },
            )
        )

    with pytest.raises(ValidationError, match=message):
        webgrid_provenance_from_records(complete_recording("cursor"), records)


def test_webgrid_rejects_elapsed_time_contradiction() -> None:
    trial = {"ordinal": 2, "key": 0, "block": 0, "target_id": 7, "stimulus_id": 0}
    records = (
        _row(
            "webgrid.selection_incorrect",
            20,
            {
                "sequence": 9,
                "pointer_update_ordinal": 5,
                "target_onset_ordinal": 3,
                "paradigm": 8,
                "correct": False,
                "selected_id": 6,
                "intended_id": 7,
                "target_onset_ns": 10,
                "elapsed_since_target_onset_ns": 999,
                "acquisition_timing_valid": True,
                "trial": trial,
            },
        ),
    )

    with pytest.raises(ValidationError, match="elapsed time contradicts its timestamps"):
        webgrid_provenance_from_records(complete_recording(), records)


def test_webgrid_external_pointer_requires_stream_identity() -> None:
    trial = {"ordinal": 2, "key": 0, "block": 0, "target_id": 7, "stimulus_id": 0}
    base_records = (
        _row(
            "webgrid.pointer_source",
            0,
            {"recorded_here": False, "stream": "cursor", "source_ordinal": 0},
        ),
        _row(
            "webgrid.target_onset",
            10,
            {"target_onset_ordinal": 3, "target_id": 7, "trial": trial},
        ),
        _row(
            "webgrid.pointer_reference",
            20,
            {
                "pointer_update_ordinal": 5,
                "source_ordinal": 0,
                "stream": "cursor",
                "trial": trial,
            },
        ),
        _row(
            "webgrid.selection_correct",
            20,
            {
                "sequence": 9,
                "pointer_update_ordinal": 5,
                "target_onset_ordinal": 3,
                "paradigm": 8,
                "correct": True,
                "selected_id": 7,
                "intended_id": 7,
                "target_onset_ns": 10,
                "elapsed_since_target_onset_ns": 10,
                "acquisition_timing_valid": True,
                "trial": trial,
            },
        ),
        {
            "label": json.dumps(
                {
                    "trial": trial,
                    "paradigm": 8,
                    "acquisition_ns": 10,
                    "selected_id": 7,
                    "intended_id": 7,
                    "correct": True,
                    "target_onset_ns": 10,
                    "selection_sequence": 9,
                    "target_onset_ordinal": 3,
                    "acquisition_timing_valid": True,
                }
            ),
            "outcome": json.dumps({"outcome": 1, "reason": 0}),
        },
    )

    missing_stream = webgrid_provenance_from_records(complete_recording(), base_records).resolve(9)
    assert missing_stream.verdict is ProvenanceVerdict.PARTIAL
    assert any(
        gap.reason is EvidenceGapReason.RAW_STREAM_NOT_RECORDED for gap in missing_stream.gaps
    )

    missing_sample_identity = webgrid_provenance_from_records(
        complete_recording("cursor"), base_records
    ).resolve(9)
    assert missing_sample_identity.verdict is ProvenanceVerdict.PARTIAL
    assert any(gap.reason is EvidenceGapReason.LINK_OMITTED for gap in missing_sample_identity.gaps)

    identified_records = tuple(
        (
            _row(
                "webgrid.pointer_reference",
                20,
                {
                    "pointer_update_ordinal": 5,
                    "source_ordinal": 0,
                    "stream": "cursor",
                    "frame_sequence": 12,
                    "sample_index": 90,
                    "trial": trial,
                },
            )
            if row.get("name") == "webgrid.pointer_reference"
            else row
        )
        for row in base_records
    )
    identified = webgrid_provenance_from_records(
        complete_recording("cursor"), identified_records
    ).resolve(9)
    assert identified.verdict is ProvenanceVerdict.PARTIAL
    assert any(gap.reason is EvidenceGapReason.LINK_OMITTED for gap in identified.gaps)

    external_sample = SourceRangeEvidence(
        eid(EvidenceKind.SOURCE_RANGE, 33), "cursor", 0, 90, 91, 12, 12
    )
    verified = webgrid_provenance_from_records(
        complete_recording("cursor"),
        identified_records,
        external_sources=(external_sample,),
    ).resolve(9)
    assert verified.verdict is ProvenanceVerdict.COMPLETE
    assert verified.pointer_update is not None
    assert verified.pointer_update.external_frame_sequence == 12
    assert verified.pointer_update.external_sample_index == 90
    assert verified.external_source == external_sample
