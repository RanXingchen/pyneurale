#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Deterministic simulated neural acquisition.

The signal generator defines values at absolute sample positions. This module
adds acquisition framing, clock metadata, pacing, cancellation, and the native
``NativeFrameSource`` boundary. Frame payloads use
``(n_samples, n_channels)`` sample-major layout in the configured
``sample_dtype``: ``float64`` values as generated, or ``int16`` counts rounded
and saturated from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeAlias, final

from neurale._native_loader import load_native_namespace
from neurale._validation import validate_number, validate_positive_float
from neurale.exceptions import StreamStateError, ValidationError

if TYPE_CHECKING:
    from neurale.signal.simulation import NeuralSignalConfig, SignalGenerator

_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UNIT_NAMES = {
    "unspecified": "UNSPECIFIED",
    "volts": "VOLTS",
    "amperes": "AMPERES",
    "dimensionless": "DIMENSIONLESS",
}

# Payload element types the simulator writes. ``int32`` and ``float32`` exist in
# the streaming schema but are not produced here, so they are rejected by name
# rather than silently approximated.
_DTYPE_NAMES = {
    "float64": "FLOAT64",
    "int16": "INT16",
}


def _integer(value: int, name: str, *, minimum: int, maximum: int) -> int:
    return int(
        validate_number(
            value,
            name,
            kind="integer",
            minimum=minimum,
            maximum=maximum,
        )
    )


def _identifier(
    value: int,
    name: str,
    *,
    allow_zero: bool = False,
    maximum: int = _UINT64_MAX,
) -> int:
    return _integer(value, name, minimum=0 if allow_zero else 1, maximum=maximum)


def _frame_samples(value: int, name: str) -> int:
    return _integer(value, name, minimum=1, maximum=_UINT32_MAX)


def _frame_ordinal(value: int) -> int:
    return _integer(value, "frame_ordinal", minimum=0, maximum=_UINT64_MAX)


@dataclass(frozen=True, slots=True)
class KnownSampleLoss:
    """Omit an exact number of samples before a data-frame ordinal."""

    frame_ordinal: int
    n_samples: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))
        object.__setattr__(
            self,
            "n_samples",
            _integer(self.n_samples, "n_samples", minimum=1, maximum=_UINT64_MAX),
        )


@dataclass(frozen=True, slots=True)
class TransientWouldBlock:
    """Return ``would_block`` once before a data-frame ordinal."""

    frame_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))


@dataclass(frozen=True, slots=True)
class BoundedStall:
    """Delay a data-frame ordinal by a finite device-clock duration."""

    frame_ordinal: int
    duration_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))
        object.__setattr__(
            self,
            "duration_ns",
            _integer(self.duration_ns, "duration_ns", minimum=1, maximum=_UINT64_MAX),
        )


@dataclass(frozen=True, slots=True)
class Disconnect:
    """Terminate acquisition through the existing source-failure path."""

    frame_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))


@dataclass(frozen=True, slots=True)
class SourceFault:
    """Inject the existing native ``source_failure`` status."""

    frame_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))


@dataclass(frozen=True, slots=True)
class DeviceTickJump:
    """Apply a nonzero signed jump to the sample-counter device tick."""

    frame_ordinal: int
    delta: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))
        object.__setattr__(
            self,
            "delta",
            _integer(self.delta, "delta", minimum=_INT64_MIN, maximum=_INT64_MAX),
        )
        if self.delta == 0:
            raise ValidationError("delta must be nonzero.")


@dataclass(frozen=True, slots=True)
class DeviceClockRestart:
    """Start a new device-clock generation at an explicit tick origin."""

    frame_ordinal: int
    device_tick_origin: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ordinal", _frame_ordinal(self.frame_ordinal))
        object.__setattr__(
            self,
            "device_tick_origin",
            _integer(
                self.device_tick_origin,
                "device_tick_origin",
                minimum=0,
                maximum=_UINT64_MAX,
            ),
        )


AcquisitionEvent: TypeAlias = (
    KnownSampleLoss
    | TransientWouldBlock
    | BoundedStall
    | Disconnect
    | SourceFault
    | DeviceTickJump
    | DeviceClockRestart
)

_ACQUISITION_EVENT_TYPES = (
    KnownSampleLoss,
    TransientWouldBlock,
    BoundedStall,
    Disconnect,
    SourceFault,
    DeviceTickJump,
    DeviceClockRestart,
)


@final
class ManualHostClock:
    """Manually advanced host clock for pacing and clock-sync references."""

    __slots__ = ("_native",)

    def __init__(self, initial_time_ns: int = 0) -> None:
        value = _integer(initial_time_ns, "initial_time_ns", minimum=0, maximum=_UINT64_MAX)
        native = load_native_namespace("devices.simulation")
        self._native = native.ManualHostClock(value)

    @property
    def now_ns(self) -> int:
        return int(self._native.now_ns)

    def set(self, time_ns: int) -> None:
        value = _integer(time_ns, "time_ns", minimum=0, maximum=_UINT64_MAX)
        if not self._native.set(value):
            raise ValidationError("time_ns cannot move a ManualHostClock backwards.")

    def advance(self, duration_ns: int) -> None:
        self._native.advance(_integer(duration_ns, "duration_ns", minimum=0, maximum=_UINT64_MAX))


@dataclass(frozen=True, slots=True)
class SimulationTimingConfig:
    """Advanced device-clock and absolute-position simulation controls."""

    initial_sample_idx: int = 0
    device_ticks: bool = True
    initial_device_tick: int | None = None
    clock_offset_ns: int = 0
    clock_drift_ppm: int = 0
    clock_sync_uncertainty_ns: int = 0
    clock: ManualHostClock | None = None

    def __post_init__(self) -> None:
        initial_sample_idx = _identifier(
            self.initial_sample_idx, "initial_sample_idx", allow_zero=True
        )
        initial_device_tick = self.initial_device_tick
        if initial_device_tick is not None:
            initial_device_tick = _identifier(
                initial_device_tick, "initial_device_tick", allow_zero=True
            )
        offset = _integer(
            self.clock_offset_ns,
            "clock_offset_ns",
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )
        drift = _integer(
            self.clock_drift_ppm,
            "clock_drift_ppm",
            minimum=-999_999,
            maximum=_INT64_MAX,
        )
        uncertainty = _identifier(
            self.clock_sync_uncertainty_ns,
            "clock_sync_uncertainty_ns",
            allow_zero=True,
        )
        if not isinstance(self.device_ticks, bool):
            raise ValidationError("device_ticks must be a bool value.")
        if not self.device_ticks and (offset != 0 or drift != 0 or uncertainty != 0):
            raise ValidationError(
                "device-clock configuration requires sample-counter device ticks."
            )
        if self.clock is not None and not isinstance(self.clock, ManualHostClock):
            raise ValidationError("clock must be a ManualHostClock or None.")
        object.__setattr__(self, "initial_sample_idx", initial_sample_idx)
        object.__setattr__(self, "initial_device_tick", initial_device_tick)
        object.__setattr__(self, "clock_offset_ns", offset)
        object.__setattr__(self, "clock_drift_ppm", drift)
        object.__setattr__(self, "clock_sync_uncertainty_ns", uncertainty)


@dataclass(frozen=True, slots=True)
class SimulationFaultPlan:
    """Immutable deterministic acquisition-event schedule."""

    events: tuple[AcquisitionEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.events, (tuple, list)):
            raise ValidationError("events must be a tuple or list of typed acquisition events.")
        events = tuple(self.events)
        if any(not isinstance(event, _ACQUISITION_EVENT_TYPES) for event in events):
            raise ValidationError("events must contain typed acquisition events.")
        object.__setattr__(self, "events", events)


_DEFAULT_TIMING = SimulationTimingConfig()
_DEFAULT_FAULTS = SimulationFaultPlan()


def _native_event(event: AcquisitionEvent, native: Any) -> object:
    if not isinstance(event, _ACQUISITION_EVENT_TYPES):
        raise ValidationError("events must contain typed acquisition events.")
    value = native.AcquisitionEvent()
    value.frame_ordinal = event.frame_ordinal
    if isinstance(event, KnownSampleLoss):
        value.kind = native.AcquisitionEventKind.SAMPLE_LOSS
        value.n_samples = event.n_samples
    elif isinstance(event, TransientWouldBlock):
        value.kind = native.AcquisitionEventKind.WOULD_BLOCK
    elif isinstance(event, BoundedStall):
        value.kind = native.AcquisitionEventKind.STALL
        value.duration_ns = event.duration_ns
    elif isinstance(event, Disconnect):
        value.kind = native.AcquisitionEventKind.DISCONNECT
    elif isinstance(event, SourceFault):
        value.kind = native.AcquisitionEventKind.SOURCE_FAULT
    elif isinstance(event, DeviceTickJump):
        value.kind = native.AcquisitionEventKind.DEVICE_TICK_JUMP
        value.device_tick_delta = event.delta
    elif isinstance(event, DeviceClockRestart):
        value.kind = native.AcquisitionEventKind.DEVICE_RESTART
        value.restart_tick = event.device_tick_origin
    return value


def _rational_rate(fs: float, streaming: Any) -> object:
    rate = validate_positive_float(fs, "fs")
    rational = Fraction(str(rate))
    if rational.numerator > _UINT64_MAX or rational.denominator > _UINT64_MAX:
        raise ValidationError("fs cannot be represented by the native rational rate.")
    return streaming.RationalRate(rational.numerator, rational.denominator)


def _physical_unit(value: str, streaming: Any) -> object:
    if not isinstance(value, str):
        raise ValidationError("physical_unit must be a supported unit name.")
    try:
        return getattr(streaming.PhysicalUnit, _UNIT_NAMES[value.lower()])
    except (KeyError, AttributeError) as exc:
        names = ", ".join(_UNIT_NAMES)
        raise ValidationError(f"physical_unit must be one of: {names}.") from exc


def _sample_dtype(value: str, streaming: Any) -> object:
    if not isinstance(value, str):
        raise ValidationError("sample_dtype must be a supported dtype name.")
    try:
        return getattr(streaming.SignalDType, _DTYPE_NAMES[value.lower()])
    except (KeyError, AttributeError) as exc:
        names = ", ".join(_DTYPE_NAMES)
        raise ValidationError(f"sample_dtype must be one of: {names}.") from exc


@final
class SimulatedNeuralDevice:
    """Prepared NSP-class simulator exposing one native frame source.

    ``generator`` is the sole owner of sample-generation mathematics. The
    constructor freezes the stream schema and all frame bounds. ``source`` may
    be passed directly to :class:`neurale.streaming.StreamRunner`.

    The native runtime stamps the host receive timestamp after ``read``
    succeeds. Advanced absolute-position, device-clock, and deterministic fault
    controls live in ``timing`` and ``faults`` rather than the common path.
    Each new source receives an internal nonzero session ID; ``reset()`` keeps
    that ID for the lifetime of the source.

    ``sample_dtype`` selects the payload element type. ``"float64"`` writes the
    generated values unchanged. ``"int16"`` rounds each value to the nearest
    integer and saturates it to the 16-bit range, which models a device that
    ships raw ADC counts; configure the generator in count units and use an
    honest ``physical_unit`` rather than labelling uncalibrated counts as volts.
    """

    __slots__ = (
        "_channel_names",
        "_faults",
        "_generator",
        "_recording_capacity_hint",
        "_schema",
        "_source",
        "_timing",
    )

    def __init__(
        self,
        generator: SignalGenerator,
        *,
        samples_per_frame: int,
        channel_names: tuple[str, ...] | list[str] | None = None,
        channel_impedances_ohm: tuple[float, ...] | list[float] | None = None,
        sample_dtype: str = "float64",
        physical_unit: str = "volts",
        paced: bool = True,
        timing: SimulationTimingConfig | None = None,
        faults: SimulationFaultPlan | None = None,
    ) -> None:
        self._initialize(
            generator,
            samples_per_frame=samples_per_frame,
            total_sample_count=None,
            channel_names=channel_names,
            channel_impedances_ohm=channel_impedances_ohm,
            sample_dtype=sample_dtype,
            physical_unit=physical_unit,
            paced=paced,
            timing=timing,
            faults=faults,
        )

    def _initialize(
        self,
        generator: SignalGenerator,
        *,
        samples_per_frame: int,
        total_sample_count: int | None,
        channel_names: tuple[str, ...] | list[str] | None,
        channel_impedances_ohm: tuple[float, ...] | list[float] | None,
        sample_dtype: str,
        physical_unit: str,
        paced: bool,
        timing: SimulationTimingConfig | None,
        faults: SimulationFaultPlan | None,
    ) -> None:
        from neurale.signal.simulation import SignalGenerator, _native_generator

        if not isinstance(generator, SignalGenerator):
            raise ValidationError("generator must be a SignalGenerator.")
        frame_samples = _frame_samples(samples_per_frame, "samples_per_frame")
        if timing is None:
            timing = _DEFAULT_TIMING
        elif not isinstance(timing, SimulationTimingConfig):
            raise ValidationError("timing must be a SimulationTimingConfig.")
        if faults is None:
            faults = _DEFAULT_FAULTS
        elif not isinstance(faults, SimulationFaultPlan):
            raise ValidationError("faults must be a SimulationFaultPlan.")
        if total_sample_count is not None:
            total_sample_count = _identifier(
                total_sample_count, "total_sample_count", allow_zero=True
            )
        if not isinstance(paced, bool):
            raise ValidationError("paced must be a bool value.")
        initial_device_tick = timing.initial_device_tick
        if initial_device_tick is None:
            initial_device_tick = timing.initial_sample_idx

        if channel_names is None:
            names = tuple(f"channel_{idx}" for idx in range(generator.n_channels))
        else:
            if not isinstance(channel_names, (tuple, list)):
                raise ValidationError("channel_names must be a tuple or list of strings.")
            names = tuple(channel_names)
            if len(names) != generator.n_channels or any(
                not isinstance(name, str) or not name for name in names
            ):
                raise ValidationError(
                    "channel_names must contain one non-empty string per generator channel."
                )
            if len(set(names)) != len(names):
                raise ValidationError("channel_names must be unique.")

        if channel_impedances_ohm is None:
            impedances: tuple[float, ...] = ()
        else:
            if not isinstance(channel_impedances_ohm, (tuple, list)):
                raise ValidationError("channel_impedances_ohm must be a tuple or list of numbers.")
            impedances = tuple(
                float(
                    validate_number(
                        value,
                        "channel_impedances_ohm",
                        kind="real",
                        minimum=0,
                        coerce=True,
                    )
                )
                for value in channel_impedances_ohm
            )
            if len(impedances) != generator.n_channels:
                raise ValidationError(
                    "channel_impedances_ohm must contain one value per generator channel."
                )

        native = load_native_namespace("devices.simulation")
        streaming = load_native_namespace("streaming")
        try:
            source = native.SimulatedNeuralSource(
                _native_generator(generator),
                frame_samples,
                _rational_rate(generator.sample_rate, streaming),
                total_sample_count,
                timing.initial_sample_idx,
                _sample_dtype(sample_dtype, streaming),
                _physical_unit(physical_unit, streaming),
                timing.device_ticks,
                initial_device_tick,
                timing.clock_offset_ns,
                timing.clock_drift_ppm,
                timing.clock_sync_uncertainty_ns,
                paced,
                [_native_event(event, native) for event in faults.events],
                names,
                impedances,
                None if timing.clock is None else timing.clock._native,
            )
        except (ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc
        self._generator = generator
        self._timing = timing
        self._faults = faults
        self._source = source
        self._schema = source.schema
        self._channel_names = names
        self._recording_capacity_hint = total_sample_count

    @property
    def generator(self) -> SignalGenerator:
        """Prepared signal generator owned by this device."""
        return self._generator

    @property
    def source(self) -> object:
        """Native ``NativeFrameSource`` accepted by ``StreamRunner``."""
        return self._source

    @property
    def schema(self) -> object:
        """Immutable native stream schema fixed at construction."""
        return self._schema

    @property
    def _native_session_id(self) -> int:
        """Private run identity consumed by native recorder integration."""
        return int(self._source._session_id)

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Control-plane channel names corresponding to payload columns."""
        return self._channel_names

    @property
    def timing(self) -> SimulationTimingConfig:
        """Advanced timing configuration fixed at construction."""
        return self._timing

    @property
    def faults(self) -> SimulationFaultPlan:
        """Deterministic fault plan fixed at construction."""
        return self._faults

    @property
    def events(self) -> tuple[AcquisitionEvent, ...]:
        """Immutable deterministic schedule fixed at construction."""
        return self._faults.events

    @property
    def frames_emitted(self) -> int:
        return int(self._source.frames_emitted)

    @property
    def samples_emitted(self) -> int:
        return int(self._source.samples_emitted)

    @property
    def closed(self) -> bool:
        """Whether this device has entered its terminal closed state."""
        return bool(self._source.closed)

    def cancel(self) -> None:
        """Request cancellation and wake a paced read promptly."""
        self._source.cancel()

    def reset(self) -> None:
        """Restore acquisition state after completion or cancellation.

        The source must be quiescent: a reset issued while a read is still in
        flight is refused rather than racing it, so cancel and then join the
        runner first. A closed device is terminal and cannot be reopened by
        reset at all.
        """
        status = self._source.reset()
        if status != load_native_namespace("streaming").StreamStatus.OK:
            raise StreamStateError(
                "a closed or actively reading SimulatedNeuralDevice cannot be reset"
            )

    def close(self) -> None:
        """Cancel acquisition and enter an idempotent terminal state."""
        self._source.close()


@dataclass(frozen=True, slots=True)
class NeuralDriftSchedule:
    """Map an intent context ordinal to a deterministic drift progress."""

    start_ordinal: int
    end_ordinal: int
    start_progress: float = 0.0
    end_progress: float = 1.0

    def __post_init__(self) -> None:
        start = _identifier(self.start_ordinal, "start_ordinal", allow_zero=True)
        end = _identifier(self.end_ordinal, "end_ordinal", allow_zero=True)
        if end <= start:
            raise ValidationError("end_ordinal must be greater than start_ordinal.")
        start_progress = float(
            validate_number(
                self.start_progress, "start_progress", kind="real", minimum=0.0, maximum=1.0
            )
        )
        end_progress = float(
            validate_number(
                self.end_progress, "end_progress", kind="real", minimum=0.0, maximum=1.0
            )
        )
        object.__setattr__(self, "start_ordinal", start)
        object.__setattr__(self, "end_ordinal", end)
        object.__setattr__(self, "start_progress", start_progress)
        object.__setattr__(self, "end_progress", end_progress)


@dataclass(frozen=True, slots=True)
class AppliedNeuralControl:
    """Exact intent and drift used to generate one emitted frame."""

    intent_sequence: int
    intent_x: float
    intent_y: float
    context_ordinal: int
    intent_valid: bool
    drift_progress: float
    first_sample_index: int
    last_sample_index: int
    generated_frame_sequence: int


@final
class IntentDrivenNeuralDevice:
    """Opt-in native feedback simulator driven by an in-memory intent source.

    Manual ``publish_control`` and ``bind_intent_source`` modes are mutually
    exclusive. Binding consumes a native read capability such as
    ``CenterOutSession.intent_source``; the acquisition thread never calls
    Python while generating frames.
    """

    __slots__ = ("_channel_names", "_config", "_schema", "_source")

    def __init__(
        self,
        config: NeuralSignalConfig,
        *,
        samples_per_frame: int,
        paced: bool = True,
        drift_schedule: NeuralDriftSchedule | None = None,
    ) -> None:
        from neurale.signal.simulation import NeuralSignalConfig, _native_neural_config

        if not isinstance(config, NeuralSignalConfig):
            raise ValidationError("config must be a NeuralSignalConfig.")
        if not isinstance(paced, bool):
            raise ValidationError("paced must be a bool value.")
        if drift_schedule is not None and not isinstance(drift_schedule, NeuralDriftSchedule):
            raise ValidationError("drift_schedule must be a NeuralDriftSchedule or None.")
        native = load_native_namespace("devices.simulation")
        streaming = load_native_namespace("streaming")
        native_schedule = None
        if drift_schedule is not None:
            native_schedule = native.NeuralDriftSchedule()
            native_schedule.start_ordinal = drift_schedule.start_ordinal
            native_schedule.end_ordinal = drift_schedule.end_ordinal
            native_schedule.start_progress = drift_schedule.start_progress
            native_schedule.end_progress = drift_schedule.end_progress
        names = tuple(f"channel_{idx}" for idx in range(config.n_channels))
        try:
            source = native.IntentDrivenNeuralSource(
                _native_neural_config(config),
                _frame_samples(samples_per_frame, "samples_per_frame"),
                _rational_rate(config.fs, streaming),
                paced,
                native_schedule,
                names,
                4096,
            )
        except (ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc
        self._config = config
        self._source = source
        self._schema = source.schema
        self._channel_names = names

    @property
    def config(self) -> NeuralSignalConfig:
        return self._config

    @property
    def source(self) -> object:
        return self._source

    @property
    def schema(self) -> object:
        return self._schema

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._channel_names

    @property
    def _native_session_id(self) -> int:
        return int(self._source._session_id)

    @property
    def frames_emitted(self) -> int:
        return int(self._source.frames_emitted)

    @property
    def samples_emitted(self) -> int:
        return int(self._source.samples_emitted)

    @property
    def drift_fingerprint(self) -> int:
        return int(self._source.drift_fingerprint)

    @property
    def dropped_applied_control_count(self) -> int:
        return int(self._source.dropped_applied_control_count)

    def bind_intent_source(self, source: object) -> None:
        status = self._source.bind_intent_source(source)
        if status != load_native_namespace("streaming").StreamStatus.OK:
            raise StreamStateError("intent source binding is invalid or control mode is fixed")

    def publish_control(
        self,
        intent: tuple[float, float] | list[float],
        *,
        drift_progress: float,
        context_ordinal: int,
    ) -> None:
        if not isinstance(intent, (tuple, list)) or len(intent) != 2:
            raise ValidationError("intent must contain exactly two values.")
        x = float(validate_number(intent[0], "intent[0]", kind="real"))
        y = float(validate_number(intent[1], "intent[1]", kind="real"))
        progress = float(
            validate_number(drift_progress, "drift_progress", kind="real", minimum=0.0, maximum=1.0)
        )
        ordinal = _identifier(context_ordinal, "context_ordinal", allow_zero=True)
        status = self._source.publish_control(x, y, progress, ordinal)
        if status != load_native_namespace("streaming").StreamStatus.OK:
            raise StreamStateError("manual control cannot be published in bound mode")

    def pop_applied_control(self) -> AppliedNeuralControl | None:
        value = self._source.pop_applied_control()
        return None if value is None else AppliedNeuralControl(**value)

    def cancel(self) -> None:
        self._source.cancel()

    def reset(self) -> None:
        status = self._source.reset()
        if status != load_native_namespace("streaming").StreamStatus.OK:
            raise StreamStateError("a closed or actively reading device cannot be reset")

    def close(self) -> None:
        self._source.close()


def _finite_simulated_neural_device(
    generator: SignalGenerator,
    *,
    samples_per_frame: int,
    total_sample_count: int,
    channel_names: tuple[str, ...] | list[str] | None = None,
    channel_impedances_ohm: tuple[float, ...] | list[float] | None = None,
    sample_dtype: str = "float64",
    physical_unit: str = "volts",
    paced: bool = True,
    timing: SimulationTimingConfig | None = None,
    faults: SimulationFaultPlan | None = None,
) -> SimulatedNeuralDevice:
    """Build a finite source for deterministic internal validation."""

    device = object.__new__(SimulatedNeuralDevice)
    device._initialize(
        generator,
        samples_per_frame=samples_per_frame,
        total_sample_count=total_sample_count,
        channel_names=channel_names,
        channel_impedances_ohm=channel_impedances_ohm,
        sample_dtype=sample_dtype,
        physical_unit=physical_unit,
        paced=paced,
        timing=timing,
        faults=faults,
    )
    return device


__all__ = [
    "AcquisitionEvent",
    "AppliedNeuralControl",
    "BoundedStall",
    "DeviceClockRestart",
    "DeviceTickJump",
    "Disconnect",
    "IntentDrivenNeuralDevice",
    "KnownSampleLoss",
    "ManualHostClock",
    "NeuralDriftSchedule",
    "SimulatedNeuralDevice",
    "SimulationFaultPlan",
    "SimulationTimingConfig",
    "SourceFault",
    "TransientWouldBlock",
]
