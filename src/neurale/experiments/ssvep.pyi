# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
from collections.abc import Sequence
from typing import ClassVar, Final, overload

from neurale.experiments import (
    ContractStatus,
    ExperimentEvent,
    PresentationRequest,
    SelectionEvent,
    StateTransition,
    TimeInterval,
    TrialIdentity,
    TrialOutcome,
    TrialRecord,
)

from ._ssvep_session import SSVEPOutcome as SSVEPOutcome
from ._ssvep_session import SSVEPProtocol as SSVEPProtocol
from ._ssvep_session import SSVEPSession as SSVEPSession

MAX_SSVEP_TARGETS: Final[int]

class SSVEPState:
    IDLE: ClassVar[SSVEPState]
    CUE: ClassVar[SSVEPState]
    STIMULATION: ClassVar[SSVEPState]
    AWAIT_DECISION: ClassVar[SSVEPState]
    FEEDBACK: ClassVar[SSVEPState]
    INTER_TRIAL: ClassVar[SSVEPState]
    COMPLETE: ClassVar[SSVEPState]
    __members__: ClassVar[dict[str, SSVEPState]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SSVEPPhase:
    NONE: ClassVar[SSVEPPhase]
    CUE: ClassVar[SSVEPPhase]
    STIMULATION: ClassVar[SSVEPPhase]
    AWAIT_DECISION: ClassVar[SSVEPPhase]
    FEEDBACK: ClassVar[SSVEPPhase]
    INTER_TRIAL: ClassVar[SSVEPPhase]
    __members__: ClassVar[dict[str, SSVEPPhase]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SSVEPCause:
    UNSPECIFIED: ClassVar[SSVEPCause]
    SESSION_STARTED: ClassVar[SSVEPCause]
    CUE_ELAPSED: ClassVar[SSVEPCause]
    STIMULATION_ELAPSED: ClassVar[SSVEPCause]
    SELECTION_RECEIVED: ClassVar[SSVEPCause]
    DECISION_TIMEOUT: ClassVar[SSVEPCause]
    FEEDBACK_ELAPSED: ClassVar[SSVEPCause]
    REST_ELAPSED: ClassVar[SSVEPCause]
    TRIAL_LIMIT_REACHED: ClassVar[SSVEPCause]
    STOPPED: ClassVar[SSVEPCause]
    __members__: ClassVar[dict[str, SSVEPCause]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SSVEPReason:
    UNSPECIFIED: ClassVar[SSVEPReason]
    CORRECT_SELECTION: ClassVar[SSVEPReason]
    INCORRECT_SELECTION: ClassVar[SSVEPReason]
    DECISION_TIMEOUT: ClassVar[SSVEPReason]
    STOPPED: ClassVar[SSVEPReason]
    __members__: ClassVar[dict[str, SSVEPReason]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SSVEPMarker:
    UNSPECIFIED: ClassVar[SSVEPMarker]
    CUE_ONSET: ClassVar[SSVEPMarker]
    CUE_OFFSET: ClassVar[SSVEPMarker]
    STIMULATION_ONSET: ClassVar[SSVEPMarker]
    STIMULATION_OFFSET: ClassVar[SSVEPMarker]
    WAIT_ONSET: ClassVar[SSVEPMarker]
    WAIT_OFFSET: ClassVar[SSVEPMarker]
    FEEDBACK_ONSET: ClassVar[SSVEPMarker]
    FEEDBACK_OFFSET: ClassVar[SSVEPMarker]
    REST_ONSET: ClassVar[SSVEPMarker]
    REST_OFFSET: ClassVar[SSVEPMarker]
    __members__: ClassVar[dict[str, SSVEPMarker]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SSVEPSelectionDisposition:
    NONE: ClassVar[SSVEPSelectionDisposition]
    ACCEPTED: ClassVar[SSVEPSelectionDisposition]
    EXPIRED: ClassVar[SSVEPSelectionDisposition]
    __members__: ClassVar[dict[str, SSVEPSelectionDisposition]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SSVEPTarget:
    def __init__(self, id: int, frequency_hz: float) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def frequency_hz(self) -> float: ...

class SSVEPTask:
    """Immutable task configuration. All durations are in seconds."""

    def __init__(
        self,
        *,
        targets: Sequence[SSVEPTarget],
        stimulus_id: int = 1,
        cue_duration: float,
        stimulation_duration: float,
        decision_timeout: float,
        feedback_duration: float,
        inter_trial: float,
        seed: int = 0,
    ) -> None: ...
    @property
    def targets(self) -> list[SSVEPTarget]: ...
    @property
    def stimulus_id(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def cue_duration(self) -> float: ...
    @property
    def stimulation_duration(self) -> float: ...
    @property
    def decision_timeout(self) -> float: ...
    @property
    def feedback_duration(self) -> float: ...
    @property
    def inter_trial(self) -> float: ...

class SSVEPTrialSchedule:
    @property
    def ordinal(self) -> int: ...
    @property
    def target_id(self) -> int: ...

class SSVEPPhaseInterval:
    @property
    def phase(self) -> SSVEPPhase: ...
    @property
    def interval(self) -> TimeInterval: ...

class SSVEPPresentationRequest:
    @property
    def request(self) -> PresentationRequest: ...
    @property
    def selected_id(self) -> int: ...
    @property
    def outcome(self) -> TrialOutcome: ...

class SSVEPTrial:
    @property
    def phases(self) -> list[SSVEPPhaseInterval]: ...
    @property
    def record(self) -> TrialRecord: ...
    @property
    def schedule(self) -> SSVEPTrialSchedule: ...
    @property
    def has_selection(self) -> bool: ...
    @property
    def selection(self) -> SelectionEvent: ...
    @property
    def has_decision(self) -> bool: ...
    @property
    def decision_ns(self) -> int: ...

class SSVEPSnapshot:
    @property
    def time_ns(self) -> int: ...
    @property
    def state(self) -> SSVEPState: ...
    @property
    def trial(self) -> TrialIdentity: ...
    @property
    def active(self) -> TimeInterval: ...
    @property
    def stimulation(self) -> TimeInterval: ...
    @property
    def decision_deadline_ns(self) -> int: ...
    @property
    def has_selection(self) -> bool: ...
    @property
    def selection(self) -> SelectionEvent: ...
    @property
    def outcome(self) -> TrialOutcome: ...
    @property
    def completed(self) -> int: ...

class SSVEPStepResult:
    @property
    def snapshot(self) -> SSVEPSnapshot: ...
    @property
    def selection_disposition(self) -> SSVEPSelectionDisposition: ...
    @property
    def trial_decided(self) -> bool: ...
    @property
    def trial(self) -> SSVEPTrial: ...
    @property
    def settled(self) -> bool: ...
    @property
    def transitions(self) -> list[StateTransition]: ...
    @property
    def events(self) -> list[ExperimentEvent]: ...
    @property
    def requests(self) -> list[SSVEPPresentationRequest]: ...

class SSVEPMachine:
    def __init__(self) -> None: ...
    def start(
        self, paradigm: int, task: SSVEPTask, time_ns: int, *, trials: int
    ) -> tuple[ContractStatus, SSVEPStepResult]: ...
    @overload
    def step(self, time_ns: int) -> tuple[ContractStatus, SSVEPStepResult]: ...
    @overload
    def step(
        self, time_ns: int, selection: SelectionEvent
    ) -> tuple[ContractStatus, SSVEPStepResult]: ...
    def stop(self, time_ns: int) -> tuple[ContractStatus, SSVEPStepResult]: ...
    def reset(self) -> None: ...
    def snapshot(self) -> SSVEPSnapshot: ...
    def configuration(self) -> SSVEPTask: ...
    @property
    def paradigm(self) -> int: ...
    @property
    def state(self) -> SSVEPState: ...
    @property
    def complete(self) -> bool: ...

def prepare_trial(config: SSVEPTask, ordinal: int) -> tuple[ContractStatus, SSVEPTrialSchedule]: ...
def validate(
    value: SSVEPTarget | SSVEPTask | SSVEPTrialSchedule | SSVEPTrial | SSVEPPresentationRequest,
) -> ContractStatus: ...
