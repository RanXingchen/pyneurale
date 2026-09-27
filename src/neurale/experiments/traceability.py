#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline experiment provenance resolution.

The records in this module are an immutable, normalized view of evidence a
reader has actually recovered.  They do not infer links from timestamps.  In
particular, a feature names every source interval it depends on, a report names
the request ordinal it reports on, and an absent raw stream remains absent.

Resolution is deliberately offline and allocation-using.  Nothing here is a
streaming callback, a recorder, or an NRF parser, and importing this module does
not load the native extension.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from itertools import pairwise
from typing import Any, Protocol, TypeVar

from neurale.exceptions import ValidationError

_COMMAND_APPLICATION_CODES = frozenset(range(4))
_PRESENTATION_STATUS_CODES = frozenset(range(4))
_TRIAL_OUTCOME_CODES = frozenset(range(5))
_SPEECH_PHASE_BLACK = 0
_SPEECH_PHASE_CROSS = 1
_SPEECH_PHASE_CONTENT = 2
_SPEECH_PHASE_INTER_TRIAL = 3
_SPEECH_PRESENTATION_PHASES = frozenset(
    (_SPEECH_PHASE_BLACK, _SPEECH_PHASE_CROSS, _SPEECH_PHASE_CONTENT)
)
_SPEECH_PHASE_TOPOLOGIES = frozenset(
    (
        (_SPEECH_PHASE_BLACK, _SPEECH_PHASE_CONTENT),
        (_SPEECH_PHASE_BLACK, _SPEECH_PHASE_CROSS, _SPEECH_PHASE_CONTENT),
        (_SPEECH_PHASE_BLACK, _SPEECH_PHASE_CONTENT, _SPEECH_PHASE_INTER_TRIAL),
        (
            _SPEECH_PHASE_BLACK,
            _SPEECH_PHASE_CROSS,
            _SPEECH_PHASE_CONTENT,
            _SPEECH_PHASE_INTER_TRIAL,
        ),
    )
)
_UNSET_STIMULUS_ID = 0


class EvidenceKind(IntEnum):
    """Stable kind component of one compact ``(kind, ordinal)`` identity."""

    SOURCE_RANGE = 1
    FEATURE_OBSERVATION = 2
    DECODER_OUTPUT = 3
    GUIDANCE_DECISION = 4
    ASSISTANCE_DECISION = 5
    EXPERIMENT_STATE = 6
    COMMAND_REQUEST = 7
    COMMAND_APPLICATION = 8
    POINTER_SOURCE = 9
    POINTER_UPDATE = 10
    TARGET_ONSET = 11
    SELECTION = 12
    TRIAL = 13
    METRIC_CONTRIBUTION = 14
    SPEECH_SCHEDULE = 15
    SPEECH_PHASE = 16
    PRESENTATION_REQUEST = 17
    PRESENTATION_OUTCOME = 18
    ACQUISITION_INTERVAL = 19
    ACQUISITION_MARKER = 20
    RECORDING = 21
    PRESENTATION_INPUT = 22


class ProvenanceVerdict(IntEnum):
    """What the available evidence establishes, in conservative precedence."""

    COMPLETE = 0
    PARTIAL = 1
    INCOMPLETE = 2
    FAULTED = 3


class EvidenceGapReason(IntEnum):
    """Why a stage could not participate in a resolved trace."""

    MISSING_RECORD = 1
    LINK_OMITTED = 2
    RAW_STREAM_NOT_RECORDED = 3
    RECORDING_UNVERIFIED = 4
    RECORDING_INCOMPLETE = 5
    TRACE_INCOMPLETE = 6


@dataclass(frozen=True, slots=True, order=True)
class EvidenceId:
    kind: EvidenceKind
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            raise ValidationError("EvidenceId.kind must be an EvidenceKind")
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise ValidationError("EvidenceId.ordinal must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class EvidenceGap:
    kind: EvidenceKind
    reason: EvidenceGapReason
    expected: EvidenceId | None = None


@dataclass(frozen=True, slots=True)
class RecordingEvidence:
    """Reader-derived completeness facts supplied without an I/O dependency."""

    session_id: str
    complete: bool | None
    accounting_verified: bool
    trace_complete: bool
    raw_stream_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValidationError("RecordingEvidence.session_id must be non-empty")
        streams = tuple(self.raw_stream_ids)
        if any(not value for value in streams) or len(set(streams)) != len(streams):
            raise ValidationError("RecordingEvidence.raw_stream_ids must be unique and non-empty")
        if self.complete is True and not self.accounting_verified:
            raise ValidationError("complete=True requires verified accounting")
        object.__setattr__(self, "raw_stream_ids", streams)


@dataclass(frozen=True, slots=True)
class TrialIdentityEvidence:
    """Dependency-light mirror of the complete ``TrialIdentity`` value."""

    ordinal: int
    key: int
    block: int
    target_id: int
    stimulus_id: int

    def __post_init__(self) -> None:
        values = (self.ordinal, self.key, self.block, self.target_id, self.stimulus_id)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
        ):
            raise ValidationError("trial identity fields must be non-negative integers")

    def same_trial(self, other: TrialIdentityEvidence) -> bool:
        """Match the native contract: session-local ordinal plus block."""

        return self.ordinal == other.ordinal and self.block == other.block


def _require_ref(ref: EvidenceId, kind: EvidenceKind, field: str) -> None:
    if ref.kind != kind:
        raise ValidationError(f"{field} must reference {kind.name.lower()}")


@dataclass(frozen=True, slots=True)
class SourceRangeEvidence:
    """One recorded source dependency over ``[sample_start, sample_stop)``.

    Frame sequences are optional because not every recording path promises a unique parent
    frame for a window.  They are either both present or both absent.
    """

    id: EvidenceId
    stream_id: str
    segment_id: int
    sample_start: int
    sample_stop: int
    first_frame_sequence: int | None = None
    last_frame_sequence: int | None = None

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.SOURCE_RANGE, "SourceRangeEvidence.id")
        if not self.stream_id:
            raise ValidationError("SourceRangeEvidence.stream_id must be non-empty")
        if self.segment_id < 0 or self.sample_start < 0 or self.sample_stop <= self.sample_start:
            raise ValidationError(
                "source ranges must be non-empty non-negative [start, stop) ranges"
            )
        first, last = self.first_frame_sequence, self.last_frame_sequence
        if (first is None) != (last is None):
            raise ValidationError("source frame bounds must both be present or both be absent")
        if first is not None and (first < 0 or last is None or last < first):
            raise ValidationError("source frame bounds must be ordered non-negative ordinals")


@dataclass(frozen=True, slots=True)
class FeatureObservationEvidence:
    id: EvidenceId
    observation_index: int
    time_ns: int
    source_ranges: tuple[EvidenceId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ranges", tuple(self.source_ranges))
        _require_ref(self.id, EvidenceKind.FEATURE_OBSERVATION, "FeatureObservationEvidence.id")
        if self.observation_index < 0 or self.time_ns < 0 or not self.source_ranges:
            raise ValidationError("a feature observation requires time, index, and source ranges")
        for ref in self.source_ranges:
            _require_ref(ref, EvidenceKind.SOURCE_RANGE, "source_ranges")
        if len(set(self.source_ranges)) != len(self.source_ranges):
            raise ValidationError("feature source ranges must be unique")


@dataclass(frozen=True, slots=True)
class DecoderOutputEvidence:
    id: EvidenceId
    feature: EvidenceId
    output_index: int
    time_ns: int
    frame_sequence: int
    sample_index: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.DECODER_OUTPUT, "DecoderOutputEvidence.id")
        _require_ref(self.feature, EvidenceKind.FEATURE_OBSERVATION, "feature")
        if min(self.output_index, self.time_ns, self.frame_sequence, self.sample_index) < 0:
            raise ValidationError("decoder output identity, index, and time must be non-negative")


@dataclass(frozen=True, slots=True)
class GuidanceDecisionEvidence:
    id: EvidenceId
    decoder_output: EvidenceId
    target_id: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.GUIDANCE_DECISION, "GuidanceDecisionEvidence.id")
        _require_ref(self.decoder_output, EvidenceKind.DECODER_OUTPUT, "decoder_output")
        if self.target_id < 0:
            raise ValidationError("target_id must be non-negative")


@dataclass(frozen=True, slots=True)
class AssistanceDecisionEvidence:
    id: EvidenceId
    decoder_output: EvidenceId
    guidance: EvidenceId | None
    trial: TrialIdentityEvidence

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.ASSISTANCE_DECISION, "AssistanceDecisionEvidence.id")
        _require_ref(self.decoder_output, EvidenceKind.DECODER_OUTPUT, "decoder_output")
        if self.guidance is not None:
            _require_ref(self.guidance, EvidenceKind.GUIDANCE_DECISION, "guidance")


@dataclass(frozen=True, slots=True)
class ExperimentStateEvidence:
    id: EvidenceId
    trial: TrialIdentityEvidence
    target_id: int
    state_code: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.EXPERIMENT_STATE, "ExperimentStateEvidence.id")
        if min(self.target_id, self.state_code) < 0:
            raise ValidationError("experiment state fields must be non-negative")


@dataclass(frozen=True, slots=True)
class CenterOutCommandEvidence:
    id: EvidenceId
    decoder_output: EvidenceId
    experiment_state: EvidenceId
    generated_ns: int
    trial: TrialIdentityEvidence
    guidance: EvidenceId | None = None
    assistance: EvidenceId | None = None
    guidance_applicable: bool = False
    assistance_configured: bool = False

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.COMMAND_REQUEST, "CenterOutCommandEvidence.id")
        _require_ref(self.decoder_output, EvidenceKind.DECODER_OUTPUT, "decoder_output")
        _require_ref(self.experiment_state, EvidenceKind.EXPERIMENT_STATE, "experiment_state")
        if self.guidance is not None:
            _require_ref(self.guidance, EvidenceKind.GUIDANCE_DECISION, "guidance")
        if self.assistance is not None:
            _require_ref(self.assistance, EvidenceKind.ASSISTANCE_DECISION, "assistance")
        if self.generated_ns < 0:
            raise ValidationError("generated_ns must be non-negative")
        if not self.guidance_applicable and self.guidance is not None:
            raise ValidationError("guidance evidence requires guidance_applicable=True")
        if not self.assistance_configured and self.assistance is not None:
            raise ValidationError("assistance evidence requires assistance_configured=True")


@dataclass(frozen=True, slots=True)
class CommandApplicationEvidence:
    """Synchronous application evidence, not an asynchronous hardware ACK."""

    id: EvidenceId
    request: EvidenceId
    generated_ns: int
    submitted_ns: int
    application_code: int
    status_code: int
    trial: TrialIdentityEvidence
    application_scope: str = "runtime_submission"

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.COMMAND_APPLICATION, "CommandApplicationEvidence.id")
        _require_ref(self.request, EvidenceKind.COMMAND_REQUEST, "request")
        if min(self.generated_ns, self.submitted_ns, self.application_code, self.status_code) < 0:
            raise ValidationError("command application fields must be non-negative")
        if self.application_code not in _COMMAND_APPLICATION_CODES:
            raise ValidationError("application_code is not a declared CommandApplication value")
        if not isinstance(self.application_scope, str) or not self.application_scope:
            raise ValidationError("application_scope must be non-empty")
        if self.application_code == 0 and (
            self.submitted_ns != self.generated_ns or self.status_code != 0
        ):
            raise ValidationError("a not-submitted command cannot carry submission status")

    @property
    def faulted(self) -> bool:
        """Whether synchronous submission failed; never a hardware ACK verdict."""

        return self.application_code != 1 or self.status_code != 0


@dataclass(frozen=True, slots=True)
class CenterOutCommandProvenance:
    verdict: ProvenanceVerdict
    session_id: str
    request_id: EvidenceId
    source_ranges: tuple[SourceRangeEvidence, ...]
    feature: FeatureObservationEvidence | None
    decoder_output: DecoderOutputEvidence | None
    guidance: GuidanceDecisionEvidence | None
    assistance: AssistanceDecisionEvidence | None
    experiment_state: ExperimentStateEvidence | None
    command: CenterOutCommandEvidence | None
    application: CommandApplicationEvidence | None
    gaps: tuple[EvidenceGap, ...]


@dataclass(frozen=True, slots=True)
class PointerSourceEvidence:
    id: EvidenceId
    source_name: str
    source_ordinal: int
    recorded_here: bool = True

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.POINTER_SOURCE, "PointerSourceEvidence.id")
        if (
            not self.source_name
            or self.source_ordinal < 0
            or not isinstance(self.recorded_here, bool)
        ):
            raise ValidationError("pointer source identity must be non-empty and non-negative")


@dataclass(frozen=True, slots=True)
class PointerUpdateEvidence:
    """Accepted pointer value plus the post-step WebGrid snapshot identity."""

    id: EvidenceId
    time_ns: int
    source: EvidenceId | None
    trial: TrialIdentityEvidence
    external_frame_sequence: int | None = None
    external_sample_index: int | None = None
    external_source: EvidenceId | None = None
    presentation_input_ordinal: int | None = None
    renderer_time_ns: int | None = None
    presentation_input_kind: int | None = None
    inside_presentation: bool | None = None
    button: int | None = None
    modifiers: int | None = None

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.POINTER_UPDATE, "PointerUpdateEvidence.id")
        if self.source is not None:
            _require_ref(self.source, EvidenceKind.POINTER_SOURCE, "source")
        if self.time_ns < 0:
            raise ValidationError("pointer update time must be non-negative")
        frame, sample = self.external_frame_sequence, self.external_sample_index
        if (frame is None) != (sample is None):
            raise ValidationError(
                "external pointer frame and sample identities must appear together"
            )
        if frame is not None and (frame < 0 or sample is None or sample < 0):
            raise ValidationError("external pointer identity must be non-negative")
        if self.external_source is not None:
            _require_ref(self.external_source, EvidenceKind.SOURCE_RANGE, "external_source")
            if frame is None:
                raise ValidationError("external_source requires an external pointer identity")
        presentation_fields = (
            self.presentation_input_ordinal,
            self.renderer_time_ns,
            self.presentation_input_kind,
            self.inside_presentation,
            self.button,
            self.modifiers,
        )
        if any(value is not None for value in presentation_fields) and any(
            value is None for value in presentation_fields
        ):
            raise ValidationError("presentation input provenance must be complete when present")
        if self.presentation_input_ordinal is not None:
            if (
                min(
                    self.presentation_input_ordinal,
                    self.renderer_time_ns,
                    self.presentation_input_kind,
                )
                < 0
            ):
                raise ValidationError("presentation input identity must be non-negative")
            if not isinstance(self.inside_presentation, bool):
                raise ValidationError("inside_presentation must be boolean")


@dataclass(frozen=True, slots=True)
class TargetOnsetEvidence:
    id: EvidenceId
    trial: TrialIdentityEvidence
    target_id: int
    onset_ns: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.TARGET_ONSET, "TargetOnsetEvidence.id")
        if min(self.target_id, self.onset_ns) < 0:
            raise ValidationError("target onset fields must be non-negative")
        if self.target_id != self.trial.target_id:
            raise ValidationError("target onset contradicts its trial target")


@dataclass(frozen=True, slots=True)
class WebGridSelectionEvidence:
    id: EvidenceId
    pointer_update: EvidenceId
    target: EvidenceId
    trial: TrialIdentityEvidence
    paradigm_id: int
    selected_cell: int
    intended_cell: int
    correct: bool
    time_ns: int
    target_onset_ns: int
    acquisition_timing_valid: bool = True

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.SELECTION, "WebGridSelectionEvidence.id")
        _require_ref(self.pointer_update, EvidenceKind.POINTER_UPDATE, "pointer_update")
        _require_ref(self.target, EvidenceKind.TARGET_ONSET, "target")
        if (
            min(
                self.paradigm_id,
                self.selected_cell,
                self.intended_cell,
                self.time_ns,
                self.target_onset_ns,
            )
            < 0
        ):
            raise ValidationError("selection cell and time must be non-negative")
        if self.intended_cell != self.trial.target_id:
            raise ValidationError("selection intended cell contradicts its trial target")


@dataclass(frozen=True, slots=True)
class WebGridTrialEvidence:
    id: EvidenceId
    selection: EvidenceId
    target: EvidenceId
    trial: TrialIdentityEvidence
    paradigm_id: int
    outcome_code: int
    selected_cell: int
    intended_cell: int
    correct: bool
    target_onset_ns: int
    acquisition_ns: int
    acquisition_timing_valid: bool

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.TRIAL, "WebGridTrialEvidence.id")
        _require_ref(self.selection, EvidenceKind.SELECTION, "selection")
        _require_ref(self.target, EvidenceKind.TARGET_ONSET, "target")
        if (
            min(
                self.paradigm_id,
                self.outcome_code,
                self.selected_cell,
                self.intended_cell,
                self.target_onset_ns,
                self.acquisition_ns,
            )
            < 0
        ):
            raise ValidationError("trial fields must be non-negative")
        if self.outcome_code not in _TRIAL_OUTCOME_CODES:
            raise ValidationError("outcome_code is not a declared TrialOutcome value")
        if self.intended_cell != self.trial.target_id:
            raise ValidationError("completed trial intended cell contradicts its trial target")


@dataclass(frozen=True, slots=True)
class MetricContributionEvidence:
    id: EvidenceId
    selection: EvidenceId
    correct_delta: int
    incorrect_delta: int
    acquisition_ns: int | None

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.METRIC_CONTRIBUTION, "MetricContributionEvidence.id")
        _require_ref(self.selection, EvidenceKind.SELECTION, "selection")
        if self.correct_delta not in (0, 1) or self.incorrect_delta not in (0, 1):
            raise ValidationError("metric count deltas must be zero or one")
        if self.correct_delta + self.incorrect_delta != 1:
            raise ValidationError("one selection must contribute to exactly one count")
        if self.acquisition_ns is not None and self.acquisition_ns < 0:
            raise ValidationError("acquisition_ns must be non-negative when present")


@dataclass(frozen=True, slots=True)
class WebGridSelectionProvenance:
    verdict: ProvenanceVerdict
    session_id: str
    selection_id: EvidenceId
    pointer_source: PointerSourceEvidence | None
    pointer_update: PointerUpdateEvidence | None
    external_source: SourceRangeEvidence | None
    target: TargetOnsetEvidence | None
    selection: WebGridSelectionEvidence | None
    trial: WebGridTrialEvidence | None
    metric: MetricContributionEvidence | None
    gaps: tuple[EvidenceGap, ...]


@dataclass(frozen=True, slots=True)
class SpeechScheduleEvidence:
    id: EvidenceId
    trial: TrialIdentityEvidence
    stimulus_id: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.SPEECH_SCHEDULE, "SpeechScheduleEvidence.id")
        if self.stimulus_id < 0:
            raise ValidationError("speech schedule fields must be non-negative")
        if self.stimulus_id != self.trial.stimulus_id:
            raise ValidationError("speech schedule stimulus contradicts its trial identity")


@dataclass(frozen=True, slots=True)
class SpeechPhaseEvidence:
    id: EvidenceId
    schedule: EvidenceId
    phase_code: int
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.SPEECH_PHASE, "SpeechPhaseEvidence.id")
        _require_ref(self.schedule, EvidenceKind.SPEECH_SCHEDULE, "schedule")
        if self.phase_code < 0 or self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValidationError("speech phases require a non-empty [start, end) interval")
        if self.phase_code not in range(_SPEECH_PHASE_INTER_TRIAL + 1):
            raise ValidationError("phase_code is not a declared SpeechPhase value")


@dataclass(frozen=True, slots=True)
class SpeechPresentationRequestEvidence:
    id: EvidenceId
    phase: EvidenceId
    trial: TrialIdentityEvidence
    requested_ns: int
    intended_onset_ns: int
    stimulus_id: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.PRESENTATION_REQUEST, "request id")
        _require_ref(self.phase, EvidenceKind.SPEECH_PHASE, "phase")
        if min(self.requested_ns, self.intended_onset_ns, self.stimulus_id) < 0:
            raise ValidationError("presentation request fields must be non-negative")


@dataclass(frozen=True, slots=True)
class SpeechPresentationOutcomeEvidence:
    id: EvidenceId
    request: EvidenceId
    trial: TrialIdentityEvidence
    matched_request: bool
    requested_ns: int
    status_code: int
    actual_presented_ns: int | None
    stimulus_id: int
    software_evidence_available: bool = False
    intended_ns: int | None = None
    submitted_renderer_ns: int | None = None
    submitted_ns: int | None = None
    presented_renderer_ns: int | None = None
    implementation_status: int | None = None

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.PRESENTATION_OUTCOME, "outcome id")
        _require_ref(self.request, EvidenceKind.PRESENTATION_REQUEST, "request")
        if not isinstance(self.matched_request, bool):
            raise ValidationError("matched_request must be boolean")
        if min(self.requested_ns, self.status_code, self.stimulus_id) < 0 or (
            self.actual_presented_ns is not None and self.actual_presented_ns < 0
        ):
            raise ValidationError("presentation outcome fields must be non-negative")
        if self.status_code not in _PRESENTATION_STATUS_CODES:
            raise ValidationError("status_code is not a declared PresentationStatus value")
        if (self.status_code == 1) != (self.actual_presented_ns is not None):
            raise ValidationError(
                "actual_presented_ns is required exactly for PresentationStatus.PRESENTED"
            )
        if not isinstance(self.software_evidence_available, bool):
            raise ValidationError("software_evidence_available must be boolean")
        software_fields = (
            self.intended_ns,
            self.submitted_renderer_ns,
            self.submitted_ns,
            self.presented_renderer_ns,
            self.implementation_status,
        )
        if self.software_evidence_available != all(value is not None for value in software_fields):
            raise ValidationError(
                "software timing fields are required exactly when software evidence is available"
            )
        if self.software_evidence_available and min(software_fields) < 0:
            raise ValidationError("software timing fields must be non-negative")


@dataclass(frozen=True, slots=True)
class AcquisitionMarkerEvidence:
    id: EvidenceId
    trial: TrialIdentityEvidence
    time_ns: int
    marker_code: int

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.ACQUISITION_MARKER, "marker id")
        if min(self.time_ns, self.marker_code) < 0:
            raise ValidationError("acquisition marker fields must be non-negative")


@dataclass(frozen=True, slots=True)
class SpeechAcquisitionEvidence:
    id: EvidenceId
    trial: TrialIdentityEvidence
    start_ns: int
    end_ns: int
    markers: tuple[EvidenceId, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "markers", tuple(self.markers))
        _require_ref(self.id, EvidenceKind.ACQUISITION_INTERVAL, "acquisition id")
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValidationError("acquisition requires a non-empty [start, end) interval")
        for marker in self.markers:
            _require_ref(marker, EvidenceKind.ACQUISITION_MARKER, "markers")
        if len(set(self.markers)) != len(self.markers):
            raise ValidationError("acquisition markers must be unique")


@dataclass(frozen=True, slots=True)
class SpeechTrialEvidence:
    id: EvidenceId
    schedule: EvidenceId
    trial: TrialIdentityEvidence
    outcome_code: int
    invalidated: bool = False

    def __post_init__(self) -> None:
        _require_ref(self.id, EvidenceKind.TRIAL, "SpeechTrialEvidence.id")
        _require_ref(self.schedule, EvidenceKind.SPEECH_SCHEDULE, "schedule")
        if self.outcome_code < 0:
            raise ValidationError("speech trial fields must be non-negative")
        if self.outcome_code not in _TRIAL_OUTCOME_CODES:
            raise ValidationError("outcome_code is not a declared TrialOutcome value")
        if self.invalidated and self.outcome_code != 4:
            raise ValidationError("an invalidated speech trial must have outcome ABORTED")


@dataclass(frozen=True, slots=True)
class SpeechTrialProvenance:
    verdict: ProvenanceVerdict
    session_id: str
    trial_ordinal: int
    trial_block: int
    trial_identity: TrialIdentityEvidence | None
    schedule: SpeechScheduleEvidence | None
    phases: tuple[SpeechPhaseEvidence, ...]
    requests: tuple[SpeechPresentationRequestEvidence, ...]
    outcomes: tuple[SpeechPresentationOutcomeEvidence, ...]
    acquisition: SpeechAcquisitionEvidence | None
    markers: tuple[AcquisitionMarkerEvidence, ...]
    trial: SpeechTrialEvidence | None
    gaps: tuple[EvidenceGap, ...]


class _HasId(Protocol):
    id: EvidenceId


class _HasTrialIdentity(Protocol):
    id: EvidenceId
    trial: TrialIdentityEvidence


T = TypeVar("T", bound=_HasId)
TrialT = TypeVar("TrialT", bound=_HasTrialIdentity)


def _by_id(values: Iterable[T]) -> dict[EvidenceId, T]:
    result: dict[EvidenceId, T] = {}
    for value in values:
        ref = value.id
        if ref in result:
            raise ValidationError(f"duplicate evidence ID: {ref.kind.name}:{ref.ordinal}")
        result[ref] = value
    return result


def _base_gaps(recording: RecordingEvidence) -> list[EvidenceGap]:
    gaps: list[EvidenceGap] = []
    if recording.complete is False:
        gaps.append(EvidenceGap(EvidenceKind.RECORDING, EvidenceGapReason.RECORDING_INCOMPLETE))
    elif recording.complete is None or not recording.accounting_verified:
        gaps.append(EvidenceGap(EvidenceKind.RECORDING, EvidenceGapReason.RECORDING_UNVERIFIED))
    if not recording.trace_complete:
        gaps.append(EvidenceGap(EvidenceKind.RECORDING, EvidenceGapReason.TRACE_INCOMPLETE))
    return gaps


def _verdict(
    recording: RecordingEvidence, gaps: list[EvidenceGap], *, faulted: bool = False
) -> ProvenanceVerdict:
    if recording.complete is False or not recording.trace_complete:
        return ProvenanceVerdict.INCOMPLETE
    if gaps:
        return ProvenanceVerdict.PARTIAL
    if faulted:
        return ProvenanceVerdict.FAULTED
    return ProvenanceVerdict.COMPLETE


def _get(values: dict[EvidenceId, T], ref: EvidenceId, gaps: list[EvidenceGap]) -> T | None:
    value = values.get(ref)
    if value is None:
        gaps.append(EvidenceGap(ref.kind, EvidenceGapReason.MISSING_RECORD, ref))
    return value


class CenterOutProvenanceIndex:
    """Resolve one final Center-Out command by its request ordinal."""

    def __init__(
        self,
        recording: RecordingEvidence,
        *,
        sources: Iterable[SourceRangeEvidence] = (),
        features: Iterable[FeatureObservationEvidence] = (),
        decoder_outputs: Iterable[DecoderOutputEvidence] = (),
        guidance: Iterable[GuidanceDecisionEvidence] = (),
        assistance: Iterable[AssistanceDecisionEvidence] = (),
        states: Iterable[ExperimentStateEvidence] = (),
        commands: Iterable[CenterOutCommandEvidence] = (),
        applications: Iterable[CommandApplicationEvidence] = (),
        presentation_evidence_required: bool = False,
        presentation_source_ordinals: Iterable[int] = (),
    ) -> None:
        self._recording = recording
        self._sources = _by_id(sources)
        self._features = _by_id(features)
        self._decoders = _by_id(decoder_outputs)
        self._guidance = _by_id(guidance)
        self._assistance = _by_id(assistance)
        self._states = _by_id(states)
        self._commands = _by_id(commands)
        self._applications = _by_id(applications)
        self._presentation_evidence_required = presentation_evidence_required
        self._presentation_source_ordinals = frozenset(presentation_source_ordinals)
        self._application_by_request: dict[EvidenceId, CommandApplicationEvidence] = {}
        for application in self._applications.values():
            if application.request in self._application_by_request:
                raise ValidationError("a command request has more than one application outcome")
            self._application_by_request[application.request] = application

    def resolve(self, request_ordinal: int) -> CenterOutCommandProvenance:
        request_id = EvidenceId(EvidenceKind.COMMAND_REQUEST, request_ordinal)
        gaps = _base_gaps(self._recording)
        if (
            self._presentation_evidence_required
            and request_ordinal not in self._presentation_source_ordinals
        ):
            gaps.append(
                EvidenceGap(EvidenceKind.PRESENTATION_OUTCOME, EvidenceGapReason.LINK_OMITTED)
            )
        command = _get(self._commands, request_id, gaps)
        feature = decoder = guidance = assistance = state = application = None
        source_ranges: list[SourceRangeEvidence] = []
        if command is not None:
            decoder = _get(self._decoders, command.decoder_output, gaps)
            state = _get(self._states, command.experiment_state, gaps)
            if command.guidance_applicable:
                if command.guidance is None:
                    gaps.append(
                        EvidenceGap(EvidenceKind.GUIDANCE_DECISION, EvidenceGapReason.LINK_OMITTED)
                    )
                else:
                    guidance = _get(self._guidance, command.guidance, gaps)
                    if guidance is not None and guidance.decoder_output != command.decoder_output:
                        raise ValidationError("guidance refers to a different decoder output")
            if command.assistance_configured:
                if command.assistance is None:
                    gaps.append(
                        EvidenceGap(
                            EvidenceKind.ASSISTANCE_DECISION, EvidenceGapReason.LINK_OMITTED
                        )
                    )
                else:
                    assistance = _get(self._assistance, command.assistance, gaps)
                    if assistance is not None:
                        if assistance.decoder_output != command.decoder_output:
                            raise ValidationError("assistance refers to a different decoder output")
                        if assistance.guidance != command.guidance:
                            raise ValidationError(
                                "assistance refers to a different guidance decision"
                            )
                        if assistance.trial != command.trial:
                            raise ValidationError(
                                "assistance and command trial identities disagree"
                            )
            if decoder is not None:
                if decoder.time_ns != command.generated_ns:
                    raise ValidationError("command time contradicts its decoder output")
                feature = _get(self._features, decoder.feature, gaps)
            if guidance is not None and state is not None and guidance.target_id != state.target_id:
                raise ValidationError("guidance target contradicts experiment state")
            if feature is not None:
                for ref in feature.source_ranges:
                    source = _get(self._sources, ref, gaps)
                    if source is None:
                        continue
                    if source.stream_id not in self._recording.raw_stream_ids:
                        gaps.append(
                            EvidenceGap(
                                EvidenceKind.SOURCE_RANGE,
                                EvidenceGapReason.RAW_STREAM_NOT_RECORDED,
                                ref,
                            )
                        )
                    else:
                        source_ranges.append(source)
            application = self._application_by_request.get(command.id)
            if application is None:
                gaps.append(
                    EvidenceGap(EvidenceKind.COMMAND_APPLICATION, EvidenceGapReason.MISSING_RECORD)
                )
            elif application.generated_ns != command.generated_ns:
                raise ValidationError("command application contradicts command generation time")
            elif application.submitted_ns < command.generated_ns:
                raise ValidationError("command application precedes command generation")
            elif application.trial != command.trial:
                raise ValidationError("command application trial identity disagrees with request")
        return CenterOutCommandProvenance(
            verdict=_verdict(
                self._recording, gaps, faulted=bool(application and application.faulted)
            ),
            session_id=self._recording.session_id,
            request_id=request_id,
            source_ranges=tuple(source_ranges),
            feature=feature,
            decoder_output=decoder,
            guidance=guidance,
            assistance=assistance,
            experiment_state=state,
            command=command,
            application=application,
            gaps=tuple(gaps),
        )


class WebGridProvenanceIndex:
    """Resolve one WebGrid selection by ``SelectionEvent.id``."""

    def __init__(
        self,
        recording: RecordingEvidence,
        *,
        pointer_sources: Iterable[PointerSourceEvidence] = (),
        pointer_updates: Iterable[PointerUpdateEvidence] = (),
        external_sources: Iterable[SourceRangeEvidence] = (),
        targets: Iterable[TargetOnsetEvidence] = (),
        selections: Iterable[WebGridSelectionEvidence] = (),
        trials: Iterable[WebGridTrialEvidence] = (),
        metrics: Iterable[MetricContributionEvidence] = (),
        presentation_input_required: bool = False,
        presentation_evidence_required: bool = False,
        presentation_source_ordinals: Iterable[int] = (),
    ) -> None:
        self._recording = recording
        self._sources = _by_id(pointer_sources)
        self._updates = _by_id(pointer_updates)
        self._external_sources = _by_id(external_sources)
        self._targets = _by_id(targets)
        self._selections = _by_id(selections)
        self._trial_by_selection = _one_by_parent(trials, "selection")
        self._metric_by_selection = _one_by_parent(metrics, "selection")
        self._presentation_input_required = presentation_input_required
        self._presentation_evidence_required = presentation_evidence_required
        self._presentation_source_ordinals = frozenset(presentation_source_ordinals)

    def resolve(self, selection_event_id: int) -> WebGridSelectionProvenance:
        selection_id = EvidenceId(EvidenceKind.SELECTION, selection_event_id)
        gaps = _base_gaps(self._recording)
        selection = _get(self._selections, selection_id, gaps)
        update = target = source = external_source = trial = metric = None
        if selection is not None:
            update = _get(self._updates, selection.pointer_update, gaps)
            target = _get(self._targets, selection.target, gaps)
            if update is not None:
                if update.time_ns != selection.time_ns:
                    raise ValidationError("selection and pointer update must share one timestamp")
                if update.source is None:
                    gaps.append(
                        EvidenceGap(EvidenceKind.POINTER_SOURCE, EvidenceGapReason.LINK_OMITTED)
                    )
                else:
                    source = _get(self._sources, update.source, gaps)
                    if source is not None and source.recorded_here:
                        if (
                            update.external_frame_sequence is not None
                            or update.external_source is not None
                        ):
                            raise ValidationError(
                                "an embedded pointer update cannot name an external source"
                            )
                    elif source is not None:
                        if source.source_name not in self._recording.raw_stream_ids:
                            gaps.append(
                                EvidenceGap(
                                    EvidenceKind.POINTER_SOURCE,
                                    EvidenceGapReason.RAW_STREAM_NOT_RECORDED,
                                    source.id,
                                )
                            )
                        elif update.external_frame_sequence is None:
                            gaps.append(
                                EvidenceGap(
                                    EvidenceKind.POINTER_UPDATE,
                                    EvidenceGapReason.LINK_OMITTED,
                                    update.id,
                                )
                            )
                        elif update.external_source is None:
                            gaps.append(
                                EvidenceGap(
                                    EvidenceKind.SOURCE_RANGE,
                                    EvidenceGapReason.LINK_OMITTED,
                                )
                            )
                        else:
                            external_source = _get(
                                self._external_sources, update.external_source, gaps
                            )
                            if external_source is not None and (
                                external_source.stream_id != source.source_name
                                or external_source.sample_start != update.external_sample_index
                                or external_source.sample_stop != update.external_sample_index + 1
                                or external_source.first_frame_sequence
                                != update.external_frame_sequence
                                or external_source.last_frame_sequence
                                != update.external_frame_sequence
                            ):
                                raise ValidationError(
                                    "external pointer evidence contradicts its recorded identity"
                                )
                if self._presentation_input_required and update.presentation_input_ordinal is None:
                    gaps.append(
                        EvidenceGap(
                            EvidenceKind.PRESENTATION_INPUT,
                            EvidenceGapReason.LINK_OMITTED,
                        )
                    )
                if (
                    self._presentation_evidence_required
                    and update.id.ordinal not in self._presentation_source_ordinals
                ):
                    gaps.append(
                        EvidenceGap(
                            EvidenceKind.PRESENTATION_OUTCOME,
                            EvidenceGapReason.LINK_OMITTED,
                        )
                    )
            metric = self._metric_by_selection.get(selection.id)
            if metric is None:
                gaps.append(
                    EvidenceGap(EvidenceKind.METRIC_CONTRIBUTION, EvidenceGapReason.MISSING_RECORD)
                )
            trial = self._trial_by_selection.get(selection.id)
            if selection.correct and trial is None:
                gaps.append(EvidenceGap(EvidenceKind.TRIAL, EvidenceGapReason.MISSING_RECORD))
            if not selection.correct and trial is not None:
                raise ValidationError("an incorrect selection cannot complete a trial")
            if (
                target is not None
                and (selection.selected_cell == target.target_id) != selection.correct
            ):
                raise ValidationError("selection correctness contradicts the active target")
            if target is not None and selection.time_ns < target.onset_ns:
                raise ValidationError("selection precedes the active target onset")
            if target is not None:
                if selection.target_onset_ns != target.onset_ns:
                    raise ValidationError("selection target onset contradicts the active target")
                if selection.intended_cell != target.target_id:
                    raise ValidationError("selection intended cell contradicts the active target")
                if selection.trial != target.trial:
                    raise ValidationError("selection and active target trial identities disagree")
            if metric is not None and metric.correct_delta != int(selection.correct):
                raise ValidationError("metric contribution contradicts selection correctness")
            if metric is not None and target is not None:
                expected_acquisition = selection.time_ns - target.onset_ns
                if selection.correct and metric.acquisition_ns != expected_acquisition:
                    raise ValidationError(
                        "metric acquisition time contradicts the selection interval"
                    )
                if not selection.correct and metric.acquisition_ns is not None:
                    raise ValidationError(
                        "an incorrect selection cannot contribute acquisition time"
                    )
            if trial is not None:
                expected_outcome = 1 if selection.acquisition_timing_valid else 4
                if trial.outcome_code != expected_outcome:
                    raise ValidationError(
                        "completed trial outcome contradicts acquisition timing validity"
                    )
                if trial.trial != selection.trial:
                    raise ValidationError("completed trial identity contradicts selection")
                if trial.paradigm_id != selection.paradigm_id:
                    raise ValidationError("completed trial paradigm contradicts selection")
                if trial.target != selection.target:
                    raise ValidationError("completed trial target link contradicts selection")
                if trial.selected_cell != selection.selected_cell:
                    raise ValidationError("completed trial selected cell contradicts selection")
                if trial.intended_cell != selection.intended_cell:
                    raise ValidationError("completed trial intended cell contradicts selection")
                if trial.correct != selection.correct:
                    raise ValidationError("completed trial correctness contradicts selection")
                if trial.target_onset_ns != selection.target_onset_ns:
                    raise ValidationError("completed trial target onset contradicts selection")
                if trial.acquisition_timing_valid != selection.acquisition_timing_valid:
                    raise ValidationError("completed trial timing validity contradicts selection")
                if (
                    target is not None
                    and trial.acquisition_ns != selection.time_ns - target.onset_ns
                ):
                    raise ValidationError(
                        "completed trial acquisition time contradicts the selection interval"
                    )
                if metric is not None and trial.acquisition_ns != metric.acquisition_ns:
                    raise ValidationError("completed trial acquisition time contradicts metric")
            if (
                trial is not None
                and target is not None
                and not trial.trial.same_trial(target.trial)
            ):
                raise ValidationError("completed trial does not match the active target trial")
        return WebGridSelectionProvenance(
            verdict=_verdict(self._recording, gaps),
            session_id=self._recording.session_id,
            selection_id=selection_id,
            pointer_source=source,
            pointer_update=update,
            external_source=external_source,
            target=target,
            selection=selection,
            trial=trial,
            metric=metric,
            gaps=tuple(gaps),
        )


def _one_by_parent(values: Iterable[T], field: str) -> dict[EvidenceId, T]:
    result: dict[EvidenceId, T] = {}
    seen_ids: set[EvidenceId] = set()
    for value in values:
        ref = value.id
        parent = getattr(value, field)
        if ref in seen_ids:
            raise ValidationError(f"duplicate evidence ID: {ref.kind.name}:{ref.ordinal}")
        if parent in result:
            raise ValidationError(f"more than one evidence record refers to {parent}")
        seen_ids.add(ref)
        result[parent] = value
    return result


class SpeechProvenanceIndex:
    """Resolve one Speech trial while keeping intended and actual timing apart."""

    def __init__(
        self,
        recording: RecordingEvidence,
        *,
        schedules: Iterable[SpeechScheduleEvidence] = (),
        phases: Iterable[SpeechPhaseEvidence] = (),
        requests: Iterable[SpeechPresentationRequestEvidence] = (),
        outcomes: Iterable[SpeechPresentationOutcomeEvidence] = (),
        acquisitions: Iterable[SpeechAcquisitionEvidence] = (),
        markers: Iterable[AcquisitionMarkerEvidence] = (),
        trials: Iterable[SpeechTrialEvidence] = (),
        software_evidence_required: bool = False,
    ) -> None:
        self._recording = recording
        self._schedules = _by_id(schedules)
        self._phases = _by_id(phases)
        self._requests = _by_id(requests)
        self._outcomes = _by_id(outcomes)
        self._acquisitions = _by_id(acquisitions)
        self._markers = _by_id(markers)
        self._trials = _by_id(trials)
        self._schedule_by_trial = _unique_trial(self._schedules.values(), "schedule")
        self._trial_by_identity = _unique_trial(self._trials.values(), "trial")
        self._acquisition_by_trial = _unique_trial(self._acquisitions.values(), "acquisition")
        self._outcome_by_request: dict[EvidenceId, SpeechPresentationOutcomeEvidence] = {}
        self._software_evidence_required = software_evidence_required
        for outcome in self._outcomes.values():
            if not outcome.matched_request:
                continue
            if outcome.request in self._outcome_by_request:
                raise ValidationError("a presentation request has more than one matched outcome")
            self._outcome_by_request[outcome.request] = outcome

    def resolve(self, trial_ordinal: int, *, block: int = 0) -> SpeechTrialProvenance:
        if min(trial_ordinal, block) < 0:
            raise ValidationError("trial_ordinal and block must be non-negative")
        trial_key = (trial_ordinal, block)
        gaps = _base_gaps(self._recording)
        schedule = self._schedule_by_trial.get(trial_key)
        trial = self._trial_by_identity.get(trial_key)
        acquisition = self._acquisition_by_trial.get(trial_key)
        if schedule is None:
            gaps.append(EvidenceGap(EvidenceKind.SPEECH_SCHEDULE, EvidenceGapReason.MISSING_RECORD))
        if trial is None:
            gaps.append(EvidenceGap(EvidenceKind.TRIAL, EvidenceGapReason.MISSING_RECORD))
        elif schedule is not None and trial.schedule != schedule.id:
            raise ValidationError("speech trial refers to a different schedule")
        elif schedule is not None and trial.trial != schedule.trial:
            raise ValidationError("speech trial identity contradicts its schedule")
        if trial is not None and acquisition is not None and trial.trial != acquisition.trial:
            raise ValidationError("speech trial and acquisition identities disagree")
        if acquisition is not None and schedule is not None and acquisition.trial != schedule.trial:
            raise ValidationError("speech acquisition identity contradicts its schedule")
        if acquisition is None:
            gaps.append(
                EvidenceGap(EvidenceKind.ACQUISITION_INTERVAL, EvidenceGapReason.LINK_OMITTED)
            )

        phase_values = tuple(
            sorted(
                (
                    phase
                    for phase in self._phases.values()
                    if schedule and phase.schedule == schedule.id
                ),
                key=lambda value: (value.start_ns, value.id.ordinal),
            )
        )
        if schedule is not None and not phase_values:
            gaps.append(EvidenceGap(EvidenceKind.SPEECH_PHASE, EvidenceGapReason.MISSING_RECORD))
        elif (
            schedule is not None
            and tuple(phase.phase_code for phase in phase_values) not in _SPEECH_PHASE_TOPOLOGIES
        ):
            raise ValidationError("speech phase evidence does not form a declared phase topology")
        for previous, current in pairwise(phase_values):
            if previous.end_ns != current.start_ns:
                raise ValidationError("speech phase evidence must form one contiguous timeline")
        phase_ids = {phase.id for phase in phase_values}
        requests_by_phase: dict[EvidenceId, list[SpeechPresentationRequestEvidence]] = {
            phase_id: [] for phase_id in phase_ids
        }
        for request in self._requests.values():
            if request.phase in requests_by_phase:
                requests_by_phase[request.phase].append(request)
        ordered_requests: list[SpeechPresentationRequestEvidence] = []
        for phase in phase_values:
            if phase.phase_code == _SPEECH_PHASE_INTER_TRIAL:
                if any(request.phase == phase.id for request in self._requests.values()):
                    raise ValidationError("INTER_TRIAL cannot carry a PresentationRequest")
                continue
            matching = requests_by_phase[phase.id]
            if not matching:
                gaps.append(
                    EvidenceGap(EvidenceKind.PRESENTATION_REQUEST, EvidenceGapReason.MISSING_RECORD)
                )
                continue
            if len(matching) != 1:
                raise ValidationError("a speech phase has more than one presentation request")
            request = matching[0]
            if request.intended_onset_ns != phase.start_ns:
                raise ValidationError("presentation request does not use its intended phase onset")
            if schedule is not None and request.trial != schedule.trial:
                raise ValidationError("presentation request trial contradicts its schedule")
            ordered_requests.append(request)
        request_values = tuple(ordered_requests)
        outcome_values: list[SpeechPresentationOutcomeEvidence] = []
        for request in request_values:
            phase = self._phases[request.phase]
            if phase.phase_code not in _SPEECH_PRESENTATION_PHASES:
                raise ValidationError("the speech phase cannot emit a PresentationRequest")
            expected_stimulus = (
                schedule.stimulus_id
                if schedule is not None and phase.phase_code == _SPEECH_PHASE_CONTENT
                else _UNSET_STIMULUS_ID
            )
            if request.stimulus_id != expected_stimulus:
                raise ValidationError("presentation request stimulus contradicts its phase")
            if phase.phase_code != _SPEECH_PHASE_CONTENT:
                continue
            outcome = self._outcome_by_request.get(request.id)
            if outcome is None:
                gaps.append(
                    EvidenceGap(EvidenceKind.PRESENTATION_OUTCOME, EvidenceGapReason.LINK_OMITTED)
                )
            else:
                if outcome.trial != request.trial:
                    raise ValidationError("matched presentation outcome trial contradicts request")
                if outcome.requested_ns != request.requested_ns:
                    raise ValidationError("presentation outcome contradicts request time")
                if outcome.stimulus_id != request.stimulus_id:
                    raise ValidationError("presentation outcome contradicts request stimulus")
                if (
                    outcome.actual_presented_ns is not None
                    and outcome.actual_presented_ns < request.requested_ns
                ):
                    raise ValidationError("presentation outcome precedes its request")
                if outcome.software_evidence_available:
                    if outcome.intended_ns != request.intended_onset_ns:
                        raise ValidationError(
                            "software presentation evidence contradicts intended onset"
                        )
                    if outcome.submitted_ns < request.requested_ns:
                        raise ValidationError("software submission precedes its request")
                    if (
                        outcome.actual_presented_ns is not None
                        and outcome.presented_renderer_ns < outcome.submitted_renderer_ns
                    ):
                        raise ValidationError("software presentation time precedes submission")
                elif self._software_evidence_required:
                    gaps.append(
                        EvidenceGap(
                            EvidenceKind.PRESENTATION_OUTCOME,
                            EvidenceGapReason.LINK_OMITTED,
                            outcome.id,
                        )
                    )
                outcome_values.append(outcome)

        marker_values: list[AcquisitionMarkerEvidence] = []
        if acquisition is not None:
            for ref in acquisition.markers:
                marker = _get(self._markers, ref, gaps)
                if marker is not None:
                    if marker.trial != acquisition.trial:
                        raise ValidationError("acquisition marker belongs to a different trial")
                    marker_values.append(marker)
            marker_values.sort(key=lambda value: (value.time_ns, value.id.ordinal))

        return SpeechTrialProvenance(
            verdict=_verdict(self._recording, gaps),
            session_id=self._recording.session_id,
            trial_ordinal=trial_ordinal,
            trial_block=block,
            trial_identity=(
                schedule.trial
                if schedule is not None
                else trial.trial
                if trial is not None
                else acquisition.trial
                if acquisition is not None
                else None
            ),
            schedule=schedule,
            phases=phase_values,
            requests=request_values,
            outcomes=tuple(outcome_values),
            acquisition=acquisition,
            markers=tuple(marker_values),
            trial=trial,
            gaps=tuple(gaps),
        )


def _unique_trial(values: Iterable[TrialT], label: str) -> dict[tuple[int, int], TrialT]:
    result: dict[tuple[int, int], TrialT] = {}
    for value in values:
        key = (value.trial.ordinal, value.trial.block)
        if key in result:
            raise ValidationError(
                f"more than one {label} record names trial {key[0]} in block {key[1]}"
            )
        result[key] = value
    return result


def _json_object(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValidationError(f"record field {field!r} must contain a JSON object")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"record field {field!r} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValidationError(f"record field {field!r} must contain a JSON object")
    return decoded


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"record field {field!r} must be a non-negative integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"record field {field!r} must be boolean")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"record field {field!r} must be a non-empty string")
    return value


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _trial_identity(body: Mapping[str, Any]) -> TrialIdentityEvidence:
    trial = body.get("trial")
    if not isinstance(trial, Mapping):
        raise ValidationError("record field 'trial' must be an object")
    return TrialIdentityEvidence(
        _integer(trial.get("ordinal"), "trial.ordinal"),
        _integer(trial.get("key"), "trial.key"),
        _integer(trial.get("block"), "trial.block"),
        _integer(trial.get("target_id"), "trial.target_id"),
        _integer(trial.get("stimulus_id"), "trial.stimulus_id"),
    )


def center_out_provenance_from_records(
    recording: RecordingEvidence,
    records: Iterable[Mapping[str, Any]],
    *,
    sources: Iterable[SourceRangeEvidence] = (),
    features: Iterable[FeatureObservationEvidence] = (),
    decoder_outputs: Iterable[DecoderOutputEvidence] = (),
    presentation_evidence_required: bool = False,
) -> CenterOutProvenanceIndex:
    """Build an index from committed Center-Out NRF control rows.

    ``records`` are rows returned by :meth:`NrfReader.iter_records`, combined
    across the relevant control record sets.  Only explicit persisted ordinals
    are joined; row position and timestamps never create parent links.  Source,
    feature, and decoder evidence remain caller-supplied because those belong to
    the recorded data/feature/decoder streams, not the experiment control rows.
    """

    rows = tuple(records)
    presentation_trials: dict[int, TrialIdentityEvidence] = {}
    for row in rows:
        if row.get("name") != "center_out.presentation":
            continue
        body = _json_object(row, "text")
        if (
            _integer(body.get("event"), "event") == 2
            and _boolean(body.get("has_software_times"), "has_software_times")
            and _boolean(body.get("has_source_ordinal", False), "has_source_ordinal")
            and _boolean(body.get("has_trial", False), "has_trial")
        ):
            source_ordinal = _integer(body.get("source_ordinal"), "source_ordinal")
            trial = _trial_identity(body)
            previous = presentation_trials.get(source_ordinal)
            if previous is not None and previous != trial:
                raise ValidationError(
                    "Center-Out presentations for one observation disagree on trial identity"
                )
            presentation_trials[source_ordinal] = trial
    decoder_rows = tuple(decoder_outputs)
    decoders_by_id = _by_id(decoder_rows)
    links: dict[int, tuple[Mapping[str, Any], int]] = {}
    for row in rows:
        if row.get("name") != "center_out.observation_provenance":
            continue
        body = _json_object(row, "text")
        ordinal = _integer(body.get("observation_ordinal"), "observation_ordinal")
        if ordinal in links:
            raise ValidationError("duplicate Center-Out observation provenance")
        if _integer(body.get("decoder_output_ordinal"), "decoder_output_ordinal") != ordinal:
            raise ValidationError("Center-Out observation and decoder ordinals disagree")
        decoder_id = EvidenceId(EvidenceKind.DECODER_OUTPUT, ordinal)
        decoder = decoders_by_id.get(decoder_id)
        row_time_ns = _integer(row.get("time_ns"), "time_ns")
        if decoder is not None and (
            decoder.frame_sequence != _integer(body.get("frame_sequence"), "frame_sequence")
            or decoder.sample_index != _integer(body.get("sample_index"), "sample_index")
            or decoder.time_ns != row_time_ns
        ):
            raise ValidationError(
                "Center-Out observation provenance contradicts decoder frame/sample identity or time"
            )
        links[ordinal] = (body, row_time_ns)

    states: list[ExperimentStateEvidence] = []
    guidance: list[GuidanceDecisionEvidence] = []
    assistance: list[AssistanceDecisionEvidence] = []
    commands: list[CenterOutCommandEvidence] = []
    applications: list[CommandApplicationEvidence] = []
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name.startswith("center_out."):
            continue
        body = _json_object(row, "text")
        if name == "center_out.transition":
            trial = _trial_identity(body)
            target_id = (
                _integer(body.get("active_target"), "active_target")
                if "active_target" in body
                else trial.target_id
            )
            states.append(
                ExperimentStateEvidence(
                    EvidenceId(
                        EvidenceKind.EXPERIMENT_STATE, _integer(body.get("sequence"), "sequence")
                    ),
                    trial,
                    target_id,
                    _integer(body.get("to_state"), "to_state"),
                )
            )
        elif name == "center_out.guidance_sample":
            ordinal = _integer(body.get("observation_ordinal"), "observation_ordinal")
            link_entry = links.get(ordinal)
            if link_entry is None:
                continue
            link, link_time_ns = link_entry
            if _integer(row.get("time_ns"), "time_ns") != link_time_ns:
                raise ValidationError("Center-Out guidance time contradicts its observation")
            if not _boolean(link.get("guidance_applicable"), "guidance_applicable"):
                continue
            if _integer(link.get("guidance_ordinal"), "guidance_ordinal") != ordinal:
                raise ValidationError(
                    "Center-Out guidance parent ordinal contradicts child identity"
                )
            guidance.append(
                GuidanceDecisionEvidence(
                    EvidenceId(EvidenceKind.GUIDANCE_DECISION, ordinal),
                    EvidenceId(
                        EvidenceKind.DECODER_OUTPUT,
                        _integer(link.get("decoder_output_ordinal"), "decoder_output_ordinal"),
                    ),
                    _integer(body.get("state_target_id"), "state_target_id"),
                )
            )
        elif name == "center_out.assisted_velocity":
            ordinal = _integer(body.get("observation_ordinal"), "observation_ordinal")
            link_entry = links.get(ordinal)
            if link_entry is None:
                continue
            link, link_time_ns = link_entry
            if _integer(row.get("time_ns"), "time_ns") != link_time_ns:
                raise ValidationError("Center-Out assistance time contradicts its observation")
            if not _boolean(link.get("assistance_configured"), "assistance_configured"):
                continue
            if _integer(link.get("assistance_ordinal"), "assistance_ordinal") != ordinal:
                raise ValidationError(
                    "Center-Out assistance parent ordinal contradicts child identity"
                )
            guidance_ref = (
                EvidenceId(
                    EvidenceKind.GUIDANCE_DECISION,
                    _integer(link.get("guidance_ordinal"), "guidance_ordinal"),
                )
                if _boolean(link.get("guidance_applicable"), "guidance_applicable")
                else None
            )
            assistance.append(
                AssistanceDecisionEvidence(
                    EvidenceId(EvidenceKind.ASSISTANCE_DECISION, ordinal),
                    EvidenceId(
                        EvidenceKind.DECODER_OUTPUT,
                        _integer(link.get("decoder_output_ordinal"), "decoder_output_ordinal"),
                    ),
                    guidance_ref,
                    _trial_identity(body),
                )
            )
        elif name == "center_out.command":
            ordinal = _integer(body.get("observation_ordinal"), "observation_ordinal")
            if _integer(body.get("sequence"), "sequence") != ordinal:
                raise ValidationError("Center-Out command and observation ordinals disagree")
            link_entry = links.get(ordinal)
            if link_entry is None:
                continue
            link, link_time_ns = link_entry
            guidance_applicable = _boolean(link.get("guidance_applicable"), "guidance_applicable")
            assistance_configured = _boolean(
                link.get("assistance_configured"), "assistance_configured"
            )
            generated_ns = _integer(body.get("generated_ns"), "generated_ns")
            if _integer(row.get("time_ns"), "time_ns") != generated_ns:
                raise ValidationError("Center-Out command row time contradicts generated_ns")
            if generated_ns != link_time_ns:
                raise ValidationError("Center-Out command time contradicts its observation")
            commands.append(
                CenterOutCommandEvidence(
                    EvidenceId(EvidenceKind.COMMAND_REQUEST, ordinal),
                    EvidenceId(
                        EvidenceKind.DECODER_OUTPUT,
                        _integer(link.get("decoder_output_ordinal"), "decoder_output_ordinal"),
                    ),
                    EvidenceId(
                        EvidenceKind.EXPERIMENT_STATE,
                        _integer(link.get("state_sequence"), "state_sequence"),
                    ),
                    generated_ns,
                    _trial_identity(body),
                    guidance=(
                        EvidenceId(
                            EvidenceKind.GUIDANCE_DECISION,
                            _integer(link.get("guidance_ordinal"), "guidance_ordinal"),
                        )
                        if guidance_applicable
                        else None
                    ),
                    assistance=(
                        EvidenceId(
                            EvidenceKind.ASSISTANCE_DECISION,
                            _integer(link.get("assistance_ordinal"), "assistance_ordinal"),
                        )
                        if assistance_configured
                        else None
                    ),
                    guidance_applicable=guidance_applicable,
                    assistance_configured=assistance_configured,
                )
            )
        elif name == "center_out.command_outcome":
            submitted_ns = _integer(body.get("submitted_ns"), "submitted_ns")
            if _integer(row.get("time_ns"), "time_ns") != submitted_ns:
                raise ValidationError("Center-Out outcome row time contradicts submitted_ns")
            applications.append(
                CommandApplicationEvidence(
                    EvidenceId(
                        EvidenceKind.COMMAND_APPLICATION,
                        _integer(body.get("sequence"), "sequence"),
                    ),
                    EvidenceId(
                        EvidenceKind.COMMAND_REQUEST,
                        _integer(body.get("request_sequence"), "request_sequence"),
                    ),
                    _integer(body.get("generated_ns"), "generated_ns"),
                    submitted_ns,
                    _integer(body.get("application"), "application"),
                    _integer(body.get("status_code"), "status_code"),
                    _trial_identity(body),
                    _string(body.get("application_scope"), "application_scope"),
                )
            )
    commands_by_ordinal = {command.id.ordinal: command for command in commands}
    for source_ordinal, trial in presentation_trials.items():
        command = commands_by_ordinal.get(source_ordinal)
        if command is not None and command.trial != trial:
            raise ValidationError(
                "Center-Out presentation trial contradicts its source observation"
            )
    return CenterOutProvenanceIndex(
        recording,
        sources=sources,
        features=features,
        decoder_outputs=decoder_rows,
        guidance=guidance,
        assistance=assistance,
        states=states,
        commands=commands,
        applications=applications,
        presentation_evidence_required=presentation_evidence_required,
        presentation_source_ordinals=presentation_trials,
    )


def webgrid_provenance_from_records(
    recording: RecordingEvidence,
    records: Iterable[Mapping[str, Any]],
    *,
    external_sources: Iterable[SourceRangeEvidence] = (),
    presentation_input_required: bool = False,
    presentation_evidence_required: bool = False,
) -> WebGridProvenanceIndex:
    """Build a WebGrid index from committed NRF control/trial rows."""

    rows = tuple(records)
    presentation_trials: dict[int, TrialIdentityEvidence] = {}
    for row in rows:
        if row.get("name") != "webgrid.presentation":
            continue
        body = _json_object(row, "text")
        if (
            _integer(body.get("event"), "event") == 2
            and _boolean(body.get("has_software_times"), "has_software_times")
            and _boolean(body.get("has_source_ordinal", False), "has_source_ordinal")
            and _boolean(body.get("has_trial", False), "has_trial")
        ):
            source_ordinal = _integer(body.get("source_ordinal"), "source_ordinal")
            trial = _trial_identity(body)
            previous = presentation_trials.get(source_ordinal)
            if previous is not None and previous != trial:
                raise ValidationError(
                    "WebGrid presentations for one pointer update disagree on trial identity"
                )
            presentation_trials[source_ordinal] = trial
    external_source_rows = tuple(external_sources)
    external_sources_by_coordinate: dict[tuple[str, int, int], SourceRangeEvidence] = {}
    for source in external_source_rows:
        if (
            source.sample_stop != source.sample_start + 1
            or source.first_frame_sequence is None
            or source.first_frame_sequence != source.last_frame_sequence
        ):
            raise ValidationError(
                "external pointer evidence must identify exactly one sample in one frame"
            )
        key = (source.stream_id, source.first_frame_sequence, source.sample_start)
        if key in external_sources_by_coordinate:
            raise ValidationError("duplicate external pointer sample identity")
        external_sources_by_coordinate[key] = source
    pointer_sources: list[PointerSourceEvidence] = []
    pointer_updates: list[PointerUpdateEvidence] = []
    targets: list[TargetOnsetEvidence] = []
    selections: list[WebGridSelectionEvidence] = []
    trials: list[WebGridTrialEvidence] = []
    metrics: list[MetricContributionEvidence] = []
    for row in rows:
        if row.get("name") != "webgrid.pointer_source":
            continue
        body = _json_object(row, "text")
        ordinal = _integer(body.get("source_ordinal"), "source_ordinal")
        stream = body.get("stream")
        if not isinstance(stream, str):
            raise ValidationError("record field 'stream' must be a string")
        recorded_here = _boolean(body.get("recorded_here"), "recorded_here")
        if recorded_here != (stream == ""):
            raise ValidationError(
                "embedded pointer sources require an empty stream and external sources require one"
            )
        pointer_sources.append(
            PointerSourceEvidence(
                EvidenceId(EvidenceKind.POINTER_SOURCE, ordinal),
                stream or "webgrid.pointer",
                ordinal,
                recorded_here,
            )
        )

    source_by_id = _by_id(pointer_sources)

    for row in rows:
        name = row.get("name")
        if name in ("webgrid.pointer", "webgrid.pointer_reference"):
            body = _json_object(row, "text")
            ordinal = _integer(body.get("pointer_update_ordinal"), "pointer_update_ordinal")
            pointer_source_ordinal = _integer(body.get("source_ordinal"), "source_ordinal")
            source_id = EvidenceId(EvidenceKind.POINTER_SOURCE, pointer_source_ordinal)
            source = source_by_id.get(source_id)
            external = name == "webgrid.pointer_reference"
            if source is not None and source.recorded_here == external:
                raise ValidationError("WebGrid pointer row contradicts its source recording mode")
            stream = body.get("stream")
            if not isinstance(stream, str):
                raise ValidationError("record field 'stream' must be a string")
            if external != bool(stream):
                raise ValidationError("WebGrid pointer row name contradicts its stream field")
            if source is not None and external and stream != source.source_name:
                raise ValidationError("WebGrid pointer reference contradicts its source stream")
            frame_sequence = _optional_integer(body.get("frame_sequence"), "frame_sequence")
            sample_idx = _optional_integer(body.get("sample_index"), "sample_index")
            external_source = (
                external_sources_by_coordinate.get((stream, frame_sequence, sample_idx))
                if external and frame_sequence is not None and sample_idx is not None
                else None
            )
            presentation_input_available = _boolean(
                body.get("presentation_input_available", False),
                "presentation_input_available",
            )
            pointer_updates.append(
                PointerUpdateEvidence(
                    EvidenceId(EvidenceKind.POINTER_UPDATE, ordinal),
                    _integer(row.get("time_ns"), "time_ns"),
                    source_id,
                    _trial_identity(body),
                    frame_sequence,
                    sample_idx,
                    external_source.id if external_source is not None else None,
                    (
                        _integer(
                            body.get("presentation_input_ordinal"), "presentation_input_ordinal"
                        )
                        if presentation_input_available
                        else None
                    ),
                    (
                        _integer(body.get("renderer_time_ns"), "renderer_time_ns")
                        if presentation_input_available
                        else None
                    ),
                    (
                        _integer(body.get("presentation_input_kind"), "presentation_input_kind")
                        if presentation_input_available
                        else None
                    ),
                    (
                        _boolean(body.get("inside_presentation"), "inside_presentation")
                        if presentation_input_available
                        else None
                    ),
                    (
                        _integer(body.get("button"), "button")
                        if presentation_input_available
                        else None
                    ),
                    (
                        _integer(body.get("modifiers"), "modifiers")
                        if presentation_input_available
                        else None
                    ),
                )
            )
        elif name == "webgrid.target_onset":
            body = _json_object(row, "text")
            trial = _trial_identity(body)
            targets.append(
                TargetOnsetEvidence(
                    EvidenceId(
                        EvidenceKind.TARGET_ONSET,
                        _integer(body.get("target_onset_ordinal"), "target_onset_ordinal"),
                    ),
                    trial,
                    _integer(body.get("target_id"), "target_id"),
                    _integer(row.get("time_ns"), "time_ns"),
                )
            )
        elif name in ("webgrid.selection_correct", "webgrid.selection_incorrect"):
            body = _json_object(row, "text")
            sequence = _integer(body.get("sequence"), "sequence")
            correct = _boolean(body.get("correct"), "correct")
            if correct != (name == "webgrid.selection_correct"):
                raise ValidationError(
                    "WebGrid selection record name contradicts its correctness field"
                )
            trial_identity = _trial_identity(body)
            selection_time_ns = _integer(row.get("time_ns"), "time_ns")
            target_onset_ns = _integer(body.get("target_onset_ns"), "target_onset_ns")
            elapsed_ns = _integer(
                body.get("elapsed_since_target_onset_ns"),
                "elapsed_since_target_onset_ns",
            )
            if target_onset_ns > selection_time_ns or elapsed_ns != (
                selection_time_ns - target_onset_ns
            ):
                raise ValidationError("WebGrid selection elapsed time contradicts its timestamps")
            selection = WebGridSelectionEvidence(
                EvidenceId(EvidenceKind.SELECTION, sequence),
                EvidenceId(
                    EvidenceKind.POINTER_UPDATE,
                    _integer(body.get("pointer_update_ordinal"), "pointer_update_ordinal"),
                ),
                EvidenceId(
                    EvidenceKind.TARGET_ONSET,
                    _integer(body.get("target_onset_ordinal"), "target_onset_ordinal"),
                ),
                trial_identity,
                _integer(body.get("paradigm"), "paradigm"),
                _integer(body.get("selected_id"), "selected_id"),
                _integer(body.get("intended_id"), "intended_id"),
                correct,
                selection_time_ns,
                target_onset_ns,
                _boolean(body.get("acquisition_timing_valid"), "acquisition_timing_valid"),
            )
            selections.append(selection)
            metrics.append(
                MetricContributionEvidence(
                    EvidenceId(EvidenceKind.METRIC_CONTRIBUTION, sequence),
                    selection.id,
                    int(correct),
                    int(not correct),
                    elapsed_ns if correct else None,
                )
            )
        elif "label" in row:
            label = _json_object(row, "label")
            if "selection_sequence" not in label:
                continue
            outcome = _json_object(row, "outcome")
            trial = _trial_identity(label)
            selection_sequence = _integer(label.get("selection_sequence"), "selection_sequence")
            target_onset_ordinal = _integer(
                label.get("target_onset_ordinal"), "target_onset_ordinal"
            )
            trials.append(
                WebGridTrialEvidence(
                    EvidenceId(EvidenceKind.TRIAL, trial.ordinal),
                    EvidenceId(EvidenceKind.SELECTION, selection_sequence),
                    EvidenceId(EvidenceKind.TARGET_ONSET, target_onset_ordinal),
                    trial,
                    _integer(label.get("paradigm"), "paradigm"),
                    _integer(outcome.get("outcome"), "outcome"),
                    _integer(label.get("selected_id"), "selected_id"),
                    _integer(label.get("intended_id"), "intended_id"),
                    _boolean(label.get("correct"), "correct"),
                    _integer(label.get("target_onset_ns"), "target_onset_ns"),
                    _integer(label.get("acquisition_ns"), "acquisition_ns"),
                    _boolean(label.get("acquisition_timing_valid"), "acquisition_timing_valid"),
                )
            )

    updates_by_ordinal = {update.id.ordinal: update for update in pointer_updates}
    for source_ordinal, trial in presentation_trials.items():
        update = updates_by_ordinal.get(source_ordinal)
        if update is not None and update.trial != trial:
            raise ValidationError(
                "WebGrid presentation trial contradicts its source pointer update"
            )

    return WebGridProvenanceIndex(
        recording,
        pointer_sources=pointer_sources,
        pointer_updates=pointer_updates,
        external_sources=external_source_rows,
        targets=targets,
        selections=selections,
        trials=trials,
        metrics=metrics,
        presentation_input_required=presentation_input_required,
        presentation_evidence_required=presentation_evidence_required,
        presentation_source_ordinals=presentation_trials,
    )


__all__ = [
    "AcquisitionMarkerEvidence",
    "AssistanceDecisionEvidence",
    "CenterOutCommandEvidence",
    "CenterOutCommandProvenance",
    "CenterOutProvenanceIndex",
    "CommandApplicationEvidence",
    "DecoderOutputEvidence",
    "EvidenceGap",
    "EvidenceGapReason",
    "EvidenceId",
    "EvidenceKind",
    "ExperimentStateEvidence",
    "FeatureObservationEvidence",
    "GuidanceDecisionEvidence",
    "MetricContributionEvidence",
    "PointerSourceEvidence",
    "PointerUpdateEvidence",
    "ProvenanceVerdict",
    "RecordingEvidence",
    "SourceRangeEvidence",
    "SpeechAcquisitionEvidence",
    "SpeechPhaseEvidence",
    "SpeechPresentationOutcomeEvidence",
    "SpeechPresentationRequestEvidence",
    "SpeechProvenanceIndex",
    "SpeechScheduleEvidence",
    "SpeechTrialEvidence",
    "SpeechTrialProvenance",
    "TargetOnsetEvidence",
    "TrialIdentityEvidence",
    "WebGridProvenanceIndex",
    "WebGridSelectionEvidence",
    "WebGridSelectionProvenance",
    "WebGridTrialEvidence",
    "center_out_provenance_from_records",
    "webgrid_provenance_from_records",
]
