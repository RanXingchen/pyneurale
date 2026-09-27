#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The public replay configuration (contract section 8).

One value object states everything a replay run is: which mode, which streams,
which part of the session, how it is paced, and which faults are injected. It
is validated in two stages, and the split is not an implementation convenience:

* **Here**, everything decidable from the request alone -- the mode name, which
  range field a mode accepts, the range unit, the pacing/`speed_factor`
  exclusivity, and the fault rules that depend only on the target's kind. A
  request that fails here never touches a session directory.
* **In the image builder**, everything that needs the session -- whether the
  ledgers exist, whether the plan covered every signal, whether a stream has a
  compatible block index, whether a boundary falls between data messages, and
  whether a ``sequence_gap`` target has an emitted frame on either side.

Both stages raise :class:`~neurale.recording.ReplayConfigError`. Contract
section 8 admits no third answer: a request is never served by a different
mode, a widened range, or a silently dropped stream.

**Pacing, ``speed_factor``, the injected faults, and ``replay_run_session_id``
are deliberately not inputs to the replay image.** Section 8.3 makes the
emitted item sequence, the run-wide numbering, and the reported metadata a pure
function of ``(session, mode, selected streams, range)``, and section 8.12 puts
the replay-run session id outside any source-derived fingerprint. Folding a
pacing choice into the image would make two byte-different images out of one
identical run, and would let a cache miss on a field that changes nothing the
image holds. They travel with the run, not with the artifact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

from ._errors import ReplayConfigError
from ._plan import REPLAY_MODES

#: The two modes that read the native ledgers and speak in data-message
#: ordinals (sections 8.1 and 8.2).
LEDGER_MODES: Final[tuple[str, ...]] = ("exact_frames", "recorded_projection")

#: The three mutually exclusive pacing modes of section 8.7.
PACING_MODES: Final[tuple[str, ...]] = ("as_fast_as_possible", "recorded", "step")

#: Units a :class:`StreamReplayRange` accepts in v1 (section 8.4).
RANGE_UNITS: Final[tuple[str, ...]] = ("block_ordinal",)

#: Unit names section 8.4 reserves. They are rejected by name, never
#: reinterpreted, so a caller who asks for one gets told it is not defined yet
#: rather than getting block ordinals under another spelling.
RESERVED_RANGE_UNITS: Final[tuple[str, ...]] = ("sample_index", "observation_index")

#: The four declared fault effects of section 8.11.
FAULT_EFFECTS: Final[tuple[str, ...]] = (
    "stall",
    "read_failure",
    "sequence_gap",
    "abnormal_end",
)

#: Upper bound of ``speed_factor``; the interval is ``(0, 1000]`` (section 8.7).
MAX_SPEED_FACTOR: Final = 1000.0

#: Largest value any ``uint64`` field of the image or the native contract holds.
UINT64_MAX: Final = (1 << 64) - 1


def _index(value: Any, label: str) -> int:
    """Return *value* as a non-negative integer, or raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayConfigError(f"{label} must be an integer, not {type(value).__name__}")
    if value < 0:
        raise ReplayConfigError(f"{label} must not be negative; got {value}")
    if value > UINT64_MAX:
        raise ReplayConfigError(f"{label} does not fit a uint64; got {value}")
    return int(value)


@dataclass(frozen=True, slots=True)
class MessageRange:
    """A half-open range of global data-message ordinals (section 8.4).

    ``[start, stop)`` in the section 1.1 ordinal, which counts frames *and*
    discontinuities. ``start == stop`` is a legal empty range; a boundary that
    does not fall between two committed data messages is rejected by the
    builder, never widened to the nearest legal one.
    """

    start: int
    stop: int

    def __post_init__(self) -> None:
        start = _index(self.start, "message_range.start")
        stop = _index(self.stop, "message_range.stop")
        if start > stop:
            raise ReplayConfigError(f"message_range start {start} is past its stop {stop}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def empty(self) -> bool:
        return self.start == self.stop

    def document(self) -> dict[str, Any]:
        return {"start": self.start, "stop": self.stop}


@dataclass(frozen=True, slots=True)
class StreamReplayRange:
    """A per-stream range for ``stream_frames`` (section 8.4).

    ``unit`` is required and explicit. v1 accepts ``block_ordinal`` only, and a
    reserved name is refused by name so that no caller can believe a sample
    range was honoured when block ordinals were used.
    """

    unit: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str):
            raise ReplayConfigError(
                f"stream range unit must be a string, not {type(self.unit).__name__}"
            )
        if self.unit in RESERVED_RANGE_UNITS:
            raise ReplayConfigError(
                f"stream range unit {self.unit!r} is reserved and not defined in v1; "
                f"the only unit v1 accepts is {RANGE_UNITS[0]!r}"
            )
        if self.unit not in RANGE_UNITS:
            raise ReplayConfigError(
                f"unknown stream range unit {self.unit!r}; expected one of {RANGE_UNITS}"
            )
        start = _index(self.start, "stream range start")
        stop = _index(self.stop, "stream range stop")
        if start > stop:
            raise ReplayConfigError(f"stream range start {start} is past its stop {stop}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def empty(self) -> bool:
        return self.start == self.stop

    def document(self) -> dict[str, Any]:
        return {"unit": self.unit, "start": self.start, "stop": self.stop}


@dataclass(frozen=True, slots=True)
class MessageFaultTarget:
    """A fault position in a ledger-based mode (section 8.11).

    ``data_message_ordinal`` already names one item; ``kind`` is kept beside it
    so configuration validation can reject a ``sequence_gap`` aimed at a
    discontinuity without first reading the session.
    """

    kind: str
    data_message_ordinal: int

    def __post_init__(self) -> None:
        if self.kind not in ("frame", "discontinuity"):
            raise ReplayConfigError(
                f"fault target kind {self.kind!r} must be 'frame' or 'discontinuity'"
            )
        object.__setattr__(
            self,
            "data_message_ordinal",
            _index(self.data_message_ordinal, "fault target data_message_ordinal"),
        )

    def document(self) -> dict[str, Any]:
        return {
            "form": "data_message",
            "kind": self.kind,
            "data_message_ordinal": self.data_message_ordinal,
        }


@dataclass(frozen=True, slots=True)
class StreamFrameFaultTarget:
    """A synthesized frame's position: one committed block (section 8.11)."""

    stream_id: str
    block_ordinal: int

    kind: ClassVar[str] = "frame"

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ReplayConfigError("a stream fault target must name a stream")
        object.__setattr__(
            self, "block_ordinal", _index(self.block_ordinal, "fault target block_ordinal")
        )

    def document(self) -> dict[str, Any]:
        return {
            "form": "stream_frame",
            "kind": self.kind,
            "stream_id": self.stream_id,
            "block_ordinal": self.block_ordinal,
        }


@dataclass(frozen=True, slots=True)
class StreamDiscontinuityFaultTarget:
    """A synthesized discontinuity's position (section 8.11).

    One block can have more than one discontinuity ahead of it, so the record
    ordinal -- not the block ordinal -- is what names this item.
    """

    stream_id: str
    discontinuity_record_ordinal: int

    kind: ClassVar[str] = "discontinuity"

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ReplayConfigError("a stream fault target must name a stream")
        object.__setattr__(
            self,
            "discontinuity_record_ordinal",
            _index(self.discontinuity_record_ordinal, "fault target discontinuity_record_ordinal"),
        )

    def document(self) -> dict[str, Any]:
        return {
            "form": "stream_discontinuity",
            "kind": self.kind,
            "stream_id": self.stream_id,
            "discontinuity_record_ordinal": self.discontinuity_record_ordinal,
        }


FaultTarget = MessageFaultTarget | StreamFrameFaultTarget | StreamDiscontinuityFaultTarget

_STREAM_TARGETS = (StreamFrameFaultTarget, StreamDiscontinuityFaultTarget)
_FAULT_TARGETS = (MessageFaultTarget, *_STREAM_TARGETS)


@dataclass(frozen=True, slots=True)
class InjectedFault:
    """One deterministic, bounded fault (section 8.11).

    ``stall_ns`` belongs to ``stall`` and to nothing else: the bound has to be
    declared up front for the run's total injected delay to be finite and known
    before it starts, and an effect that does not wait has nothing to declare.
    """

    target: FaultTarget
    effect: str
    stall_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, _FAULT_TARGETS):
            raise ReplayConfigError(
                f"fault target {type(self.target).__name__} is not a replay fault target"
            )
        if self.effect not in FAULT_EFFECTS:
            raise ReplayConfigError(
                f"unknown fault effect {self.effect!r}; expected one of {FAULT_EFFECTS}"
            )
        if self.effect == "stall":
            if self.stall_ns is None:
                raise ReplayConfigError(
                    "a stall fault must declare stall_ns; an undeclared delay is not bounded"
                )
            stall = _index(self.stall_ns, "stall_ns")
            if stall == 0:
                raise ReplayConfigError("stall_ns must be positive; a zero stall delays nothing")
            object.__setattr__(self, "stall_ns", stall)
        elif self.stall_ns is not None:
            raise ReplayConfigError(
                f"stall_ns belongs to the 'stall' effect; the {self.effect!r} effect does not wait"
            )
        if self.effect == "sequence_gap" and self.target.kind != "frame":
            raise ReplayConfigError(
                "a sequence_gap fault targets a frame and only a frame; dropping a "
                "discontinuity produces no frame-sequence hole, so this fault would do nothing"
            )

    @property
    def targets_stream_position(self) -> bool:
        return isinstance(self.target, _STREAM_TARGETS)

    def document(self) -> dict[str, Any]:
        return {
            "target": self.target.document(),
            "effect": self.effect,
            "stall_ns": self.stall_ns,
        }


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Everything one replay run is, validated as far as the request allows.

    The mode is always explicit: section 8 has no default, because a default
    would silently pick between "the original topology", "the recorded part,
    projected" and "a documented synthesis".
    """

    mode: str
    selected_streams: tuple[str, ...] = ()
    message_range: MessageRange | None = None
    stream_ranges: Mapping[str, StreamReplayRange] = field(default_factory=dict)
    pacing: str = "as_fast_as_possible"
    speed_factor: float | None = None
    allow_incomplete: bool = False
    faults: tuple[InjectedFault, ...] = ()
    replay_run_session_id: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in REPLAY_MODES:
            raise ReplayConfigError(
                f"unknown replay mode {self.mode!r}; expected one of {REPLAY_MODES}"
            )
        if not isinstance(self.allow_incomplete, bool):
            raise ReplayConfigError(
                f"allow_incomplete must be a bool, not {type(self.allow_incomplete).__name__}; a "
                "truthy non-bool such as the string 'false' would silently enable the prefix mode"
            )
        object.__setattr__(self, "selected_streams", _stream_tuple(self.selected_streams))
        object.__setattr__(self, "stream_ranges", _stream_range_map(self.stream_ranges))
        object.__setattr__(self, "faults", tuple(self.faults))
        self._check_pacing()
        self._check_range_fields()
        self._check_selection()
        self._check_faults()
        self._check_session_id()

    # --- section 8.7 ------------------------------------------------------

    def _check_pacing(self) -> None:
        if self.pacing not in PACING_MODES:
            raise ReplayConfigError(
                f"unknown pacing mode {self.pacing!r}; expected one of {PACING_MODES}"
            )
        if self.pacing != "recorded":
            if self.speed_factor is not None:
                raise ReplayConfigError(
                    f"speed_factor applies to 'recorded' pacing only and is rejected for "
                    f"{self.pacing!r} pacing rather than ignored"
                )
            return
        factor = 1.0 if self.speed_factor is None else self.speed_factor
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise ReplayConfigError(
                f"speed_factor must be a number, not {type(self.speed_factor).__name__}"
            )
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0 or factor > MAX_SPEED_FACTOR:
            raise ReplayConfigError(
                f"speed_factor must be a finite double in (0, {MAX_SPEED_FACTOR:g}]; got {factor!r}"
            )
        object.__setattr__(self, "speed_factor", factor)

    # --- section 8.4 ------------------------------------------------------

    def _check_range_fields(self) -> None:
        if self.mode in LEDGER_MODES:
            if self.stream_ranges:
                raise ReplayConfigError(
                    f"stream_ranges is the range of 'stream_frames'; {self.mode!r} ranges over "
                    "data-message ordinals, and a message ordinal is not derivable from a "
                    "per-stream sample position. Use message_range"
                )
            return
        if self.message_range is not None:
            raise ReplayConfigError(
                "message_range is the range of the ledger-based modes; 'stream_frames' does "
                "not read the ledgers and has no message ordinals. Use stream_ranges"
            )
        if not self.stream_ranges:
            raise ReplayConfigError(
                "'stream_frames' requires stream_ranges to name at least one stream; a run "
                "with no stream has no legal union schema. Replay nothing by giving one "
                "stream an empty range"
            )
        units = {value.unit for value in self.stream_ranges.values()}
        if len(units) > 1:
            raise ReplayConfigError(
                f"all stream ranges of one request must share a unit; got {sorted(units)}"
            )

    # --- section 8.4, one selection surface per mode ----------------------

    def _check_selection(self) -> None:
        if self.mode not in LEDGER_MODES and self.selected_streams:
            keys = tuple(sorted(self.stream_ranges))
            if self.selected_streams != keys:
                raise ReplayConfigError(
                    "under 'stream_frames' the streams are selected by the keys of "
                    f"stream_ranges; selected_streams {list(self.selected_streams)} disagrees "
                    f"with {list(keys)}"
                )

    # --- section 8.11 -----------------------------------------------------

    def _check_faults(self) -> None:
        seen: set[tuple[Any, ...]] = set()
        for fault in self.faults:
            if not isinstance(fault, InjectedFault):
                raise ReplayConfigError(
                    f"a replay fault must be an InjectedFault, not {type(fault).__name__}"
                )
            if self.mode in LEDGER_MODES and fault.targets_stream_position:
                raise ReplayConfigError(
                    f"a stream fault target positions a fault in 'stream_frames'; {self.mode!r} "
                    "positions one by data_message_ordinal"
                )
            if self.mode not in LEDGER_MODES and not fault.targets_stream_position:
                raise ReplayConfigError(
                    "a data_message_ordinal positions a fault in a ledger-based mode; "
                    "'stream_frames' has no data-message ordinals"
                )
            key = tuple(sorted(fault.target.document().items()))
            if key in seen:
                raise ReplayConfigError(
                    f"two faults name the same position {fault.target.document()}; a position "
                    "fires at most once per run, so which effect it would have is undefined"
                )
            seen.add(key)

    def _check_session_id(self) -> None:
        if self.replay_run_session_id is None:
            return
        value = _index(self.replay_run_session_id, "replay_run_session_id")
        if value == 0:
            raise ReplayConfigError("replay_run_session_id must be non-zero; 0 is 'no session'")
        object.__setattr__(self, "replay_run_session_id", value)

    # --- derived ----------------------------------------------------------

    @property
    def ledger_based(self) -> bool:
        """Whether this run reads the native ledgers rather than a block index."""
        return self.mode in LEDGER_MODES

    @property
    def total_injected_delay_ns(self) -> int:
        """The run's total injected delay: finite and known before it starts."""
        return sum(fault.stall_ns or 0 for fault in self.faults)

    def document(self) -> dict[str, Any]:
        """Return the full request, for reporting -- **not** for the image key."""
        return {
            "mode": self.mode,
            "selected_streams": list(self.selected_streams),
            "message_range": None if self.message_range is None else self.message_range.document(),
            "stream_ranges": {
                stream_id: value.document()
                for stream_id, value in sorted(self.stream_ranges.items())
            },
            "pacing": self.pacing,
            "speed_factor": self.speed_factor,
            "allow_incomplete": self.allow_incomplete,
            "faults": [fault.document() for fault in self.faults],
            "replay_run_session_id": self.replay_run_session_id,
        }


def _stream_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ReplayConfigError("selected_streams is a sequence of stream ids, not one string")
    streams = tuple(value)
    for stream_id in streams:
        if not isinstance(stream_id, str) or not stream_id:
            raise ReplayConfigError(f"selected_streams holds an invalid stream id {stream_id!r}")
    if len(set(streams)) != len(streams):
        raise ReplayConfigError(f"selected_streams names a stream twice: {list(streams)}")
    return tuple(sorted(streams))


def _stream_range_map(
    value: Mapping[str, StreamReplayRange] | None,
) -> dict[str, StreamReplayRange]:
    if not value:
        return {}
    ranges: dict[str, StreamReplayRange] = {}
    for stream_id, entry in value.items():
        if not isinstance(stream_id, str) or not stream_id:
            raise ReplayConfigError(f"stream_ranges holds an invalid stream id {stream_id!r}")
        if not isinstance(entry, StreamReplayRange):
            raise ReplayConfigError(
                f"stream_ranges[{stream_id!r}] must be a StreamReplayRange, not "
                f"{type(entry).__name__}"
            )
        ranges[stream_id] = entry
    return dict(sorted(ranges.items()))


__all__ = [
    "FAULT_EFFECTS",
    "LEDGER_MODES",
    "MAX_SPEED_FACTOR",
    "PACING_MODES",
    "RANGE_UNITS",
    "RESERVED_RANGE_UNITS",
    "InjectedFault",
    "MessageFaultTarget",
    "MessageRange",
    "ReplayConfig",
    "StreamDiscontinuityFaultTarget",
    "StreamFrameFaultTarget",
    "StreamReplayRange",
]
