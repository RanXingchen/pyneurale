#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""What a streaming session records, declared before it starts.

NRF v1 freezes every registry before the first commit, so a recorder cannot
discover a stream, a channel, or a descriptor while data is already arriving.
Everything here is therefore stated up front and validated before the session
directory exists.

The common API describes recording intent. Native identifiers, schema metadata,
time-axis origins, and NRF storage layout are resolved into a private stream
specification before recording starts.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ._errors import RecorderConfigError

#: NRF v1 identifier grammar (``manifest.schema.json`` ``$defs/id``). Stream and
#: unit identifiers reach the manifest verbatim, so they are checked here rather
#: than surfacing later as a schema failure with no obvious cause.
_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")

#: NRF stream kinds this recorder can write. ``spike`` is deliberately absent:
#: fixed-capacity sparse blocks have a payload layout of their own and belong to
#: the spike work, not to this integration.
STREAM_KINDS = frozenset({"neural", "behavioral", "feature"})

TIMING_MODES = frozenset({"explicit", "regular"})

_DEFAULT_STREAM_CAPACITY = 1 << 20

# Capture resource sizing, not an NRF format or backend-selection constant.


class RecorderState(StrEnum):
    """Lifecycle position of a :class:`~neurale.recording.SessionRecorder`."""

    CREATED = "created"
    PREPARED = "prepared"
    READY = "ready"
    RECORDING = "recording"
    DRAINING = "draining"
    STOPPED = "stopped"
    FINALIZING = "finalizing"
    FINALIZATION_FAILED = "finalization_failed"
    CLOSED = "closed"
    FAILED = "failed"


class StreamRole(StrEnum):
    """Scientific role of a sampled stream in the recording."""

    NEURAL = "neural"
    BEHAVIORAL = "behavioral"


class StreamTimingMode(StrEnum):
    """How sample observation times are represented in NRF."""

    EXPLICIT = "explicit"
    REGULAR = "regular"


def _require_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise RecorderConfigError(
            f"{label} {value!r} is not an NRF identifier: lowercase letters, digits, and "
            "single '-', '_', or '.' separators, starting with a letter"
        )
    return value


#: How each accepted lower bound is spelled in a rejection message.
_INTEGER_BOUNDS: Mapping[int | None, str] = {
    None: "an integer",
    0: "a non-negative integer",
    1: "a positive integer",
}


def _require_int(label: str, value: object, *, minimum: int | None = None) -> None:
    """Reject *value* unless it is an integer at or above *minimum*.

    ``bool`` is excluded deliberately: it is an ``int`` subclass, so ``True``
    would otherwise pass as a capacity of one.
    """
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or (minimum is not None and value < minimum)
    ):
        raise RecorderConfigError(f"{label} must be {_INTEGER_BOUNDS[minimum]}, got {value!r}")


@dataclass(frozen=True, slots=True)
class StreamMetadata:
    """Metadata absent from a native signal descriptor.

    ``unit`` is only accepted when the native schema leaves the physical unit
    unspecified. Feature names, feature units, and algorithm provenance always
    come from the native feature-set descriptor.
    """

    role: StreamRole | None = None
    name: str | None = None
    unit: str | Sequence[str] | None = None
    channel_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.role is not None and not isinstance(self.role, StreamRole):
            raise RecorderConfigError("metadata.role must be a StreamRole")
        if self.unit is not None and not isinstance(self.unit, (str, tuple, list)):
            raise RecorderConfigError("metadata.unit must be a symbol or a sequence of symbols")
        if self.channel_names is not None:
            names = tuple(self.channel_names)
            if any(not isinstance(name, str) or not name for name in names):
                raise RecorderConfigError("metadata.channel_names must contain non-empty strings")
            object.__setattr__(self, "channel_names", names)
        if self.unit is not None and not isinstance(self.unit, str):
            object.__setattr__(self, "unit", tuple(self.unit))


@dataclass(frozen=True, slots=True)
class StreamStorageConfig:
    """Advanced bounded NRF storage layout for one stream."""

    capacity: int | None = None
    chunk_length: int = 1024
    block_index_chunk_length: int = 256

    def __post_init__(self) -> None:
        if self.capacity is not None:
            _require_int("storage.capacity", self.capacity, minimum=1)
        _require_int("storage.chunk_length", self.chunk_length, minimum=1)
        _require_int("storage.block_index_chunk_length", self.block_index_chunk_length, minimum=1)
        if self.capacity is not None and self.capacity < self.chunk_length:
            raise RecorderConfigError("storage.capacity cannot be below storage.chunk_length")


@dataclass(frozen=True, slots=True)
class StreamTimingConfig:
    """Advanced time-axis representation for one stream."""

    mode: StreamTimingMode = StreamTimingMode.EXPLICIT
    segment_start_index: int = 0
    segment_start_time_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, StreamTimingMode):
            raise RecorderConfigError("timing.mode must be a StreamTimingMode")
        _require_int("timing.segment_start_index", self.segment_start_index, minimum=0)
        _require_int("timing.segment_start_time_ns", self.segment_start_time_ns)
        if self.mode is StreamTimingMode.EXPLICIT and (
            self.segment_start_index != 0 or self.segment_start_time_ns != 0
        ):
            raise RecorderConfigError("explicit timing does not accept a declared segment origin")


@dataclass(frozen=True, slots=True)
class StreamProvenanceConfig:
    """Advanced provenance retained alongside one recorded stream."""

    block_index: bool = True
    sources: Sequence[StreamRecording] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_index, bool):
            raise RecorderConfigError("provenance.block_index must be a bool")
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True, slots=True)
class StreamRecording:
    """User intent to record one prepared signal.

    A single-signal source needs no selector. For a multi-signal schema, pass
    the selected ``SignalSchema`` object through ``signal``; native integer IDs
    are deliberately not accepted.
    """

    stream_id: str | None = None
    signal: object | None = None
    metadata: StreamMetadata | None = None
    storage: StreamStorageConfig | None = None
    timing: StreamTimingConfig | None = None
    provenance: StreamProvenanceConfig | None = None

    def __post_init__(self) -> None:
        if self.stream_id is not None:
            _require_id(self.stream_id, "stream_id")
        if isinstance(self.signal, (int, bool)):
            raise RecorderConfigError(
                "signal must be a SignalSchema object, not a native integer signal ID"
            )
        for value, expected, label in (
            (self.metadata, StreamMetadata, "metadata"),
            (self.storage, StreamStorageConfig, "storage"),
            (self.timing, StreamTimingConfig, "timing"),
            (self.provenance, StreamProvenanceConfig, "provenance"),
        ):
            if value is not None and not isinstance(value, expected):
                raise RecorderConfigError(f"{label} must be a {expected.__name__}")


@dataclass(frozen=True, slots=True)
class _ResolvedStreamRecording:
    """Private stream declaration compiled from public recording intent."""

    signal_id: int
    stream_id: str
    capacity: int
    kind: str
    name: str
    unit: tuple[str, ...]
    channel_names: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    chunk_length: int = 1024
    timing: str = "explicit"
    segment_start_index: int = 0
    segment_start_time_ns: int = 0
    block_index: bool = True
    block_index_chunk_length: int = 256
    algorithm_name: str = ""
    algorithm_version: str = ""
    source_stream_ids: tuple[str, ...] = ()

    @property
    def block_index_stream_id(self) -> str:
        return f"{self.stream_id}.blocks"


@dataclass(frozen=True, slots=True)
class RecorderLimits:
    """Advanced bounds for a native recording session.

    Queue capacities are items, not bytes, and both queues are strictly bounded:
    a recorder that grew its own buffers would convert a machine that cannot
    keep up into one that runs out of memory, which is a worse failure and a
    later one.
    """

    #: Frames the observer edge may hand over before the writer must catch up.
    frame_queue_capacity: int = 256
    #: Control-plane records that may await the writer.
    control_queue_capacity: int = 1024
    #: Optional disk-spool byte ceiling; ``None`` grows until filesystem/file
    #: offset limits. This is not allocated or locked in RAM. Queue capacities
    #: determine capture buffers. An explicit ceiling faults when reached.
    spool_capacity_bytes: int | None = None
    #: Commits between automatic checkpoints. ``0`` disables them; recovery then
    #: starts from journal sequence 1 rather than from a checkpoint.
    checkpoint_interval: int = 32
    #: Maximum number of records in each control-plane record set. The NRF chunk
    #: layout is derived from this intent rather than exposed to callers.
    max_control_records: int = 16 * 1024

    def __post_init__(self) -> None:
        _require_int("frame_queue_capacity", self.frame_queue_capacity, minimum=1)
        _require_int("control_queue_capacity", self.control_queue_capacity, minimum=1)
        if self.spool_capacity_bytes is not None:
            _require_int("spool_capacity_bytes", self.spool_capacity_bytes, minimum=1)
            if self.spool_capacity_bytes > (1 << 53) - 1:
                raise RecorderConfigError(
                    "spool_capacity_bytes exceeds the NRF exact-integer limit"
                )
        _require_int("checkpoint_interval", self.checkpoint_interval, minimum=0)
        _require_int("max_control_records", self.max_control_records, minimum=1)

    @property
    def control_chunk_length(self) -> int:
        """NRF chunk length needed to hold ``max_control_records`` rows."""
        return (self.max_control_records + 1023) // 1024

    @property
    def control_capacity(self) -> int:
        """Actual rows available after rounding to the NRF chunk layout."""
        return self.control_chunk_length * 1024


@dataclass(frozen=True, slots=True)
class RecorderSessionMetadata:
    """Optional NRF session description, excluding recorder-owned identity."""

    subject: Mapping[str, object] | None = None
    experiment: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("subject", "experiment", "metadata", "extensions"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise RecorderConfigError(f"session.{name} must be a mapping")

    def document(self) -> dict[str, object]:
        document: dict[str, object] = {}
        if self.subject is not None:
            document["subject"] = dict(self.subject)
        if self.experiment is not None:
            document["experiment"] = dict(self.experiment)
        if self.metadata:
            document["metadata"] = dict(self.metadata)
        if self.extensions:
            document["extensions"] = dict(self.extensions)
        return document


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    """Public recording intent compiled against a prepared source."""

    #: Compressed single file to create. By convention it ends in ``.nrf``.
    path: str | os.PathLike[str]
    #: Signals selected for recording. Omission selects a sole prepared signal.
    streams: Sequence[StreamRecording] = field(default_factory=lambda: (StreamRecording(),))
    #: Optional session description. Identity and writer provenance are generated
    #: by the recorder and cannot be overridden here.
    session: RecorderSessionMetadata | None = None
    #: Free-form top-level NRF manifest metadata.
    metadata: Mapping[str, object] = field(default_factory=dict)
    #: Advanced bounds. Omission sizes acquisition queues from nominal cadence;
    #: CenterOutSession also sizes control queues. Explicit limits are unchanged.
    limits: RecorderLimits | None = None
    #: Keep recovery input only when explicitly requested for debugging.
    spool_retention: str = "delete_after_validated_finalization"

    def __post_init__(self) -> None:
        if self.spool_retention not in {"retain", "delete_after_validated_finalization"}:
            raise RecorderConfigError("invalid spool_retention policy")
        if not self.streams:
            raise RecorderConfigError("a recording must declare at least one stream")
        if self.session is not None and not isinstance(self.session, RecorderSessionMetadata):
            raise RecorderConfigError("session must be RecorderSessionMetadata")
        if self.limits is not None and not isinstance(self.limits, RecorderLimits):
            raise RecorderConfigError("limits must be RecorderLimits")
        if not isinstance(self.metadata, Mapping):
            raise RecorderConfigError("metadata must be a mapping")

        seen_streams: set[str] = set()
        for stream in self.streams:
            if not isinstance(stream, StreamRecording):
                raise RecorderConfigError("streams must contain StreamRecording values")
            if stream.stream_id is not None and stream.stream_id in seen_streams:
                raise RecorderConfigError(f"stream id {stream.stream_id!r} is used twice")
            if stream.stream_id is not None:
                seen_streams.add(stream.stream_id)
