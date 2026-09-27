# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Final

from neurale.decoding import KalmanDecoder, LinearDecoder
from neurale.experiments.presentation import CenterOutPresentationConfig
from neurale.pipeline import PipelinePlan
from neurale.recording import SessionRecorder
from neurale.streaming import RealtimeConfig

MAX_STEP_EVENTS: Final[int]
MAX_STEP_TRANSITIONS: Final[int]
MAX_SURROUNDING_TARGETS: Final[int]
MIN_RADIAL_SPOKES: Final[int]
TARGET_SELECTION_STREAM: Final[int]

class GeometryUnit:
    UNSPECIFIED: ClassVar[GeometryUnit]
    DIMENSIONLESS: ClassVar[GeometryUnit]
    NORMALIZED: ClassVar[GeometryUnit]
    MILLIMETRES: ClassVar[GeometryUnit]
    METRES: ClassVar[GeometryUnit]
    __members__: ClassVar[dict[str, GeometryUnit]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CenterOutPhase:
    TO_CENTER: ClassVar[CenterOutPhase]
    TO_OUT: ClassVar[CenterOutPhase]
    __members__: ClassVar[dict[str, CenterOutPhase]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class TargetSelectionPolicy:
    UNSPECIFIED: ClassVar[TargetSelectionPolicy]
    REPEAT_UNTIL_SUCCESS: ClassVar[TargetSelectionPolicy]
    SAMPLE_EACH_TRIAL: ClassVar[TargetSelectionPolicy]
    BALANCED_SHUFFLED_CYCLES: ClassVar[TargetSelectionPolicy]
    __members__: ClassVar[dict[str, TargetSelectionPolicy]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CenterOutState:
    IDLE: ClassVar[CenterOutState]
    MOVE_TO_CENTER: ClassVar[CenterOutState]
    HOLD_CENTER: ClassVar[CenterOutState]
    CENTER_SUCCESS_DWELL: ClassVar[CenterOutState]
    CENTER_FAILURE_DWELL: ClassVar[CenterOutState]
    MOVE_TO_OUT: ClassVar[CenterOutState]
    HOLD_OUT: ClassVar[CenterOutState]
    OUT_SUCCESS_DWELL: ClassVar[CenterOutState]
    OUT_FAILURE_DWELL: ClassVar[CenterOutState]
    COMPLETE: ClassVar[CenterOutState]
    __members__: ClassVar[dict[str, CenterOutState]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CenterOutCause:
    UNSPECIFIED: ClassVar[CenterOutCause]
    SESSION_STARTED: ClassVar[CenterOutCause]
    CONTAINMENT_GAINED: ClassVar[CenterOutCause]
    CONTAINMENT_LOST: ClassVar[CenterOutCause]
    HOLD_COMPLETED: ClassVar[CenterOutCause]
    MOVEMENT_TIMED_OUT: ClassVar[CenterOutCause]
    DWELL_ELAPSED: ClassVar[CenterOutCause]
    TRIAL_LIMIT_REACHED: ClassVar[CenterOutCause]
    __members__: ClassVar[dict[str, CenterOutCause]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CenterOutReason:
    UNSPECIFIED: ClassVar[CenterOutReason]
    OUTWARD_ACQUIRED: ClassVar[CenterOutReason]
    CENTER_MOVEMENT_TIMEOUT: ClassVar[CenterOutReason]
    OUTWARD_MOVEMENT_TIMEOUT: ClassVar[CenterOutReason]
    __members__: ClassVar[dict[str, CenterOutReason]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CenterOutGuidancePhase:
    IDLE: ClassVar[CenterOutGuidancePhase]
    SPEEDING_UP: ClassVar[CenterOutGuidancePhase]
    SLOWING_DOWN: ClassVar[CenterOutGuidancePhase]
    AT_TARGET: ClassVar[CenterOutGuidancePhase]
    __members__: ClassVar[dict[str, CenterOutGuidancePhase]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class WorkspacePoint:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None: ...
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...

class WorkspaceVelocity:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None: ...
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...

class TargetPlacement:
    def __init__(self, id: int = 0, pos: WorkspacePoint = ...) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def pos(self) -> WorkspacePoint: ...

class CenterOut2DLayout:
    def __init__(
        self,
        center: TargetPlacement = ...,
        surrounding: Sequence[TargetPlacement] = ...,
    ) -> None: ...
    @property
    def center(self) -> TargetPlacement: ...
    @property
    def count(self) -> int: ...
    @property
    def surrounding(self) -> list[TargetPlacement]: ...
    def target(self, idx: int) -> TargetPlacement: ...

class AcceptanceRegion:
    def __init__(self, half_extent_x: float = 0.0, half_extent_y: float = 0.0) -> None: ...
    @property
    def half_extent_x(self) -> float: ...
    @property
    def half_extent_y(self) -> float: ...

class CursorGeometry:
    def __init__(self, extent: float = 0.0) -> None: ...
    @property
    def extent(self) -> float: ...

class PhaseDurations:
    def __init__(self, to_center: int = 0, to_out: int = 0) -> None: ...
    @property
    def to_center(self) -> int: ...
    @property
    def to_out(self) -> int: ...
    def of(self, phase: CenterOutPhase) -> int: ...

class RadialLayoutRequest:
    def __init__(
        self,
        radius: float,
        center_id: int = 1,
        ids: Sequence[int] | None = None,
        spokes: Sequence[int] | None = None,
    ) -> None: ...
    @property
    def radius(self) -> float: ...
    @property
    def count(self) -> int: ...
    @property
    def center_id(self) -> int: ...
    @property
    def ids(self) -> list[int]: ...
    @property
    def spokes(self) -> list[int]: ...

class CenterOutTask:
    def __init__(
        self,
        *,
        geometry_unit: GeometryUnit,
        layout: CenterOut2DLayout,
        acceptance: float | AcceptanceRegion,
        movement_timeout_seconds: float | Sequence[float],
        selection: TargetSelectionPolicy,
        cursor_extent: float = 0.0,
        hold_seconds: float = 0.0,
        reward_dwell_seconds: float | Sequence[float] = 0.0,
        punish_dwell_seconds: float | Sequence[float] = 0.0,
        seed: int = 0,
    ) -> None: ...
    @property
    def geometry_unit(self) -> GeometryUnit: ...
    @property
    def layout(self) -> CenterOut2DLayout: ...
    @property
    def acceptance(self) -> AcceptanceRegion: ...
    @property
    def cursor(self) -> CursorGeometry: ...
    @property
    def movement_timeout(self) -> PhaseDurations: ...
    @property
    def hold_ns(self) -> int: ...
    @property
    def reward_dwell(self) -> PhaseDurations: ...
    @property
    def punish_dwell(self) -> PhaseDurations: ...
    @property
    def selection(self) -> TargetSelectionPolicy: ...
    @property
    def seed(self) -> int: ...
    @property
    def sampler_version(self) -> int: ...
    @property
    def movement_timeout_seconds(self) -> list[float]: ...
    @property
    def hold_seconds(self) -> float: ...
    @property
    def reward_dwell_seconds(self) -> list[float]: ...
    @property
    def punish_dwell_seconds(self) -> list[float]: ...

class CenterOutTrial:
    def __init__(
        self,
        record: Any = ...,
        outward_target: int = 0,
        outward_idx: int = 0,
        decided_phase: CenterOutPhase = ...,
        center_acquire_ns: int = 0,
        outward_acquire_ns: int = 0,
    ) -> None: ...
    @property
    def record(self) -> Any: ...
    @property
    def outward_target(self) -> int: ...
    @property
    def outward_idx(self) -> int: ...
    @property
    def decided_phase(self) -> CenterOutPhase: ...
    @property
    def center_acquire_ns(self) -> int: ...
    @property
    def outward_acquire_ns(self) -> int: ...

class CenterOutSnapshot:
    @property
    def time_ns(self) -> int: ...
    @property
    def state(self) -> CenterOutState: ...
    @property
    def phase(self) -> CenterOutPhase: ...
    @property
    def trial(self) -> Any: ...
    @property
    def active_target(self) -> int: ...
    @property
    def active_position(self) -> WorkspacePoint: ...
    @property
    def outward_target(self) -> int: ...
    @property
    def outward_idx(self) -> int: ...
    @property
    def contained(self) -> bool: ...
    @property
    def leg(self) -> Any: ...
    @property
    def hold(self) -> Any: ...
    @property
    def dwell(self) -> Any: ...
    @property
    def completed(self) -> int: ...
    @property
    def successes(self) -> int: ...

class CenterOutStepResult:
    @property
    def snapshot(self) -> CenterOutSnapshot: ...
    @property
    def settled(self) -> bool: ...
    @property
    def trial_decided(self) -> bool: ...
    @property
    def trial(self) -> CenterOutTrial: ...
    @property
    def transitions(self) -> list[Any]: ...
    @property
    def events(self) -> list[Any]: ...

class CenterOutStatistics:
    def __init__(
        self,
        decided: int = 0,
        successes: int = 0,
        center_timeouts: int = 0,
        outward_timeouts: int = 0,
        total_time_to_target_ns: int = 0,
        mean_time_to_target_ns: int = 0,
    ) -> None: ...
    @property
    def decided(self) -> int: ...
    @property
    def successes(self) -> int: ...
    @property
    def center_timeouts(self) -> int: ...
    @property
    def outward_timeouts(self) -> int: ...
    @property
    def total_time_to_target_ns(self) -> int: ...
    @property
    def mean_time_to_target_ns(self) -> int: ...

class CenterOutMachine:
    def __init__(self) -> None: ...
    def start(
        self, paradigm: int, config: CenterOutTask, time_ns: int
    ) -> tuple[Any, CenterOutStepResult]: ...
    def reset(self) -> None: ...
    def step(self, time_ns: int, cursor: WorkspacePoint) -> tuple[Any, CenterOutStepResult]: ...
    def snapshot(self) -> CenterOutSnapshot: ...
    def configuration(self) -> CenterOutTask: ...
    @property
    def paradigm(self) -> int: ...
    @property
    def state(self) -> CenterOutState: ...
    @property
    def complete(self) -> bool: ...

class CenterOutGuidanceState:
    def __init__(
        self,
        target: int = 0,
        vel: WorkspaceVelocity = ...,
        phase: CenterOutGuidancePhase = ...,
    ) -> None: ...
    @property
    def target(self) -> int: ...
    @property
    def vel(self) -> WorkspaceVelocity: ...
    @property
    def phase(self) -> CenterOutGuidancePhase: ...

class CenterOutGuidanceSample:
    @property
    def vel(self) -> WorkspaceVelocity: ...
    @property
    def state(self) -> CenterOutGuidanceState: ...
    @property
    def distance(self) -> float: ...
    @property
    def speed_towards_target(self) -> float: ...
    @property
    def retargeted(self) -> bool: ...

@dataclass(frozen=True, slots=True, kw_only=True)
class CenterOutGuidanceConfig:
    max_speed: float
    acceleration: float
    arrival_radius: float
    deceleration: float | None = None

class CenterOutGuidance:
    def __init__(self) -> None: ...
    @property
    def configured(self) -> bool: ...
    def configure(self, config: CenterOutGuidanceConfig) -> Any: ...
    def reset(self) -> None: ...
    def update(
        self, target: TargetPlacement, pos: WorkspacePoint, dt_ns: int
    ) -> tuple[Any, CenterOutGuidanceSample]: ...
    def configuration(self) -> CenterOutGuidanceConfig: ...
    def state(self) -> CenterOutGuidanceState: ...

class SupervisionTarget(StrEnum):
    VELOCITY = "velocity"
    POSITION = "position"
    POSITION_VELOCITY = "position_velocity"

class OnlineDecoderUpdatePolicy(StrEnum):
    INITIAL_ONLY = "initial_only"
    PERIODIC = "periodic"

@dataclass(frozen=True, slots=True)
class AssistanceBlock:
    assistance: float
    trials: int

@dataclass(frozen=True, slots=True, kw_only=True)
class OnlineDecoderTrainingConfig:
    target: SupervisionTarget | str = SupervisionTarget.VELOCITY
    update_interval: int | None = None
    training_window: int | None = None
    label_lag_seconds: float = 0.0
    update_policy: OnlineDecoderUpdatePolicy | str = OnlineDecoderUpdatePolicy.PERIODIC

@dataclass(frozen=True, slots=True, kw_only=True)
class CenterOutSafetyConfig:
    stale_after_seconds: float | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class CenterOutProtocol:
    task: CenterOutTask
    assistance_blocks: tuple[AssistanceBlock, ...]
    guidance: CenterOutGuidanceConfig | None = None
    initial_position: tuple[float, float] | None = None
    safety: CenterOutSafetyConfig = ...
    @property
    def trials(self) -> int: ...

@dataclass(frozen=True, slots=True)
class DecoderPublication:
    version: int
    plan_fingerprint: str
    training_blocks: int
    training_trials: int
    completed_trials_at_publication: int
    fit_duration_ns: int = 0

@dataclass(frozen=True, slots=True)
class CenterOutIntentSnapshot:
    sequence: int
    intent_x: float
    intent_y: float
    context_ordinal: int
    source_time_ns: int
    source_frame_sequence: int
    source_sample_index: int
    trial_key: int
    target_id: int
    valid: bool

@dataclass(frozen=True, slots=True)
class CenterOutOutcome:
    state: int
    experiment_trace_complete: bool
    trace_losses: int
    producer_trace_drops: int
    abnormal_conditions: int
    abnormal_trials_affected: int
    abnormal_session_aborted: bool
    terminal_abort: bool
    runtime_status: Any
    publications: tuple[DecoderPublication, ...]
    active_decoder_version: int
    completed_trials: int
    training_capture_drops: tuple[int, int]
    presented_frames: int
    presentation_state_drops: int
    recording: Any | None

class CenterOutSession:
    def __init__(
        self,
        protocol: CenterOutProtocol,
        feature_plan: PipelinePlan,
        device: Any,
        decoder: LinearDecoder | KalmanDecoder,
        *,
        runtime_config: RealtimeConfig | None = None,
        training: OnlineDecoderTrainingConfig | None = None,
        presentation_config: CenterOutPresentationConfig | None = None,
        recording: SessionRecorder | None = None,
    ) -> None: ...
    @property
    def snapshot(self) -> CenterOutSnapshot: ...
    @property
    def position(self) -> WorkspacePoint: ...
    @property
    def active_decoder_version(self) -> int: ...
    @property
    def publications(self) -> tuple[DecoderPublication, ...]: ...
    @property
    def resolved_guidance(self) -> CenterOutGuidanceConfig: ...
    def get_intent(self) -> CenterOutIntentSnapshot: ...
    @property
    def intent_source(self) -> object: ...
    def run(self) -> CenterOutOutcome: ...

def build_radial_layout(request: RadialLayoutRequest) -> CenterOut2DLayout: ...
def center_out_guidance_phase_declared(phase: CenterOutGuidancePhase) -> bool: ...
def center_out_phase_declared(phase: CenterOutPhase) -> bool: ...
def center_out_phase_of(state: CenterOutState) -> CenterOutPhase: ...
def center_out_state_declared(state: CenterOutState) -> bool: ...
def center_out_state_is_dwell(state: CenterOutState) -> bool: ...
def center_out_state_is_leg(state: CenterOutState) -> bool: ...
def configuration_fingerprint(config: CenterOutTask) -> int: ...
def contains_cursor(
    region: AcceptanceRegion,
    cursor_geometry: CursorGeometry,
    target: WorkspacePoint,
    cursor: WorkspacePoint,
) -> tuple[Any, bool]: ...
def contains_point(
    region: AcceptanceRegion, target: WorkspacePoint, cursor: WorkspacePoint
) -> tuple[Any, bool]: ...
def evaluate_guidance(
    config: CenterOutGuidanceConfig,
    previous: CenterOutGuidanceState,
    target: TargetPlacement,
    pos: WorkspacePoint,
    dt_ns: int,
) -> tuple[Any, CenterOutGuidanceSample]: ...
def geometry_unit_declared(unit: GeometryUnit) -> bool: ...
def layout_fingerprint(layout: CenterOut2DLayout) -> int: ...
def phase_duration(durations: PhaseDurations, phase: CenterOutPhase) -> int: ...
def radial_position(radius: float, count: int, spoke: int) -> tuple[Any, WorkspacePoint]: ...
def radial_spoke_step(count: int) -> float: ...
def select_outward_target(config: CenterOutTask, trial: int, successes: int) -> tuple[Any, int]: ...
def summarize(trials: Sequence[CenterOutTrial]) -> tuple[Any, CenterOutStatistics]: ...
def target_selection_policy_declared(policy: TargetSelectionPolicy) -> bool: ...
def trial_limit_reached(config: CenterOutTask, completed: int) -> bool: ...
def validate(value: object) -> Any: ...
def validate_against(guidance: CenterOutGuidanceConfig, config: CenterOutTask) -> Any: ...
def _native_guidance_config(
    config: CenterOutGuidanceConfig, geometry_unit: GeometryUnit | None = None
) -> Any: ...
def _raw_config(**fields: object) -> CenterOutTask: ...
def _raw_guidance_config(**fields: object) -> Any: ...
def _with_trial_limit(config: CenterOutTask, trials: int) -> CenterOutTask: ...

__all__: list[str]
