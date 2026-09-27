#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Turn a prepared native schema into a frozen NRF v1 session.

Everything a recorded session declares is derived here, once, before the writer
freezes. Two rules shape the result:

* **The native schema is authoritative for anything it knows.** Dtype, channel
  count, sample rate, payload layout, clock domain, and feature-set metadata are
  read from the prepared :class:`~neurale.streaming.StreamSchema` rather than
  restated by the caller, so a recording cannot describe a signal differently
  from the runtime that produced it.
* **Per-block provenance is a stream, not a record set.** A signal block carries
  a frame sequence, a sample index, a device tick, and an arrival time, and
  those arrive at block rate. Recording them as an ordinary stream costs one
  Zarr array instead of one array per column, and keeps the ``events`` record
  kind free for the experiment control plane, which is what a reader
  reconstructing a session actually expects to find there. NRF v1 has no
  ``provenance`` stream kind; the index is registered as ``behavioral`` with one
  named channel per column, because that is the kind whose columns are named
  through the channel registry and the only one that needs no fabricated
  feature window and shift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurale.io.nrf import ManifestBuilder, record_field
from neurale.io.nrf._paths import record_set_path, stream_discontinuities_path

from ._errors import RecorderConfigError
from ._plan import (
    _channel_names,
    _descriptor_text,
    _feature_names,
    _resolve_feature_source,
    _resolve_stream_recordings,
    _unit_symbols,
    require_feature_descriptor,
    validate_stream_specs,
)
from ._spec import StreamRecording, _ResolvedStreamRecording

#: Native scalar type -> (NRF dtype, NumPy dtype). NRF v1 has no implicit
#: conversion, so every native dtype must have an exact NRF spelling.
_DTYPES: Mapping[str, tuple[str, str]] = {
    "INT16": ("int16", "<i2"),
    "INT32": ("int32", "<i4"),
    "FLOAT32": ("float32", "<f4"),
    "FLOAT64": ("float64", "<f8"),
}

#: Columns of the per-block provenance stream, in stored order.
BLOCK_IDX_COLUMNS: tuple[str, ...] = (
    "frame_sequence",
    "sample_idx_start",
    "last_sample_idx",
    "n_samples",
    "row_offset",
    "device_tick_start",
    "host_received_ns",
)

NANOSECONDS = 1_000_000_000

#: Every control-plane record kind this recorder writes, with its primary key.
#: The uniform ``name``/``value``/``text`` shape is deliberate: NRF fixes the
#: *kinds* a session may declare but not their fields, and one shape a reader
#: can consume without a per-kind special case is worth more than nine bespoke
#: ones. Structured payloads go in ``text`` as JSON.
CONTROL_PRIMARY_KEYS: Mapping[str, str] = {
    "events": "event_id",
    "experiment_state": "state_transition_id",
    "commands": "command_id",
    "task_variables": "variable_id",
    "labels": "label_id",
    "targets": "target_id",
    "assistance": "assistance_id",
    "drops": "drop_id",
}

_SESSION_CLOCK = "session.clock"


def segment_id(idx: int) -> str:
    """Return the stable identifier of a stream's *index*-th segment.

    Segments are numbered from one within each stream. The recorder and the
    manifest have to agree on the spelling, so both take it from here rather
    than restating the format.
    """
    return f"segment-{idx:04d}"


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """How one native signal's blocks become NRF rows."""

    signal_id: int
    stream_id: str
    numpy_dtype: np.dtype
    n_channels: int
    channel_major: bool
    explicit_timing: bool
    #: Sample period as an exact rational in nanoseconds, or ``None`` when the
    #: signal declares no usable rate. Explicit timestamps then repeat the
    #: block's observation time rather than inventing a grid.
    period: tuple[int, int] | None
    block_index_stream_id: str | None
    #: Largest block the prepared schema allows for this signal. A block that
    #: claims more samples than the runtime could ever have produced is not a
    #: large block; it is a frame that does not belong to this schema.
    max_block_samples: int
    #: Declared origin of the first segment, used only by ``regular`` timing.
    segment_start_index: int
    segment_start_time_ns: int

    def decode(self, payload: np.ndarray, offset: int, n_bytes: int, samples: int) -> np.ndarray:
        """Return ``samples`` rows of this signal from one frame payload.

        The result is always a fresh, C-contiguous, sample-major array: a view
        into the frame payload would keep the whole frame alive inside the
        writer tail, and a channel-major transpose is not contiguous.
        """
        raw = payload[offset : offset + n_bytes]
        values = np.frombuffer(raw.tobytes(), dtype=self.numpy_dtype)
        if self.channel_major:
            return np.ascontiguousarray(values.reshape(self.n_channels, samples).T)
        return np.ascontiguousarray(values.reshape(samples, self.n_channels))

    def timestamps(self, start_ns: int, samples: int) -> np.ndarray:
        """Return one observation time per sample, in nanoseconds.

        Integer arithmetic throughout: a float seconds axis would lose
        nanosecond resolution within an hour of session time, which is exactly
        the interval a recording is expected to keep.
        """
        if self.period is None:
            return np.full(samples, start_ns, dtype="int64")
        num, den = self.period
        offsets = np.arange(samples, dtype="int64") * num // den
        return offsets + np.int64(start_ns)


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """The frozen mapping from native signals to NRF targets."""

    #: ``StreamSchema.id`` this session was prepared against. A frame from any
    #: other schema is refused rather than decoded: signal identifiers are only
    #: unique within a schema, so a foreign frame whose payload happens to be
    #: the right size would otherwise be stored under the wrong dtype, layout,
    #: and channel names, and the session would still read back complete.
    schema_id: int
    streams: Mapping[int, StreamPlan]
    control_schema_ids: Mapping[str, str]
    faults_schema_id: str
    trials_schema_id: str
    discontinuity_schema_ids: Mapping[str, str]

    def plan_for(self, signal_id: int) -> StreamPlan | None:
        return self.streams.get(signal_id)


def _signal_dtype(signal: Any, stream_id: str) -> tuple[str, np.dtype]:
    name = getattr(signal.dtype, "name", str(signal.dtype))
    entry = _DTYPES.get(name)
    if entry is None:  # pragma: no cover - guarded by the native enum
        raise RecorderConfigError(f"stream {stream_id!r} has unsupported native dtype {name!r}")
    nrf, numpy_name = entry
    return nrf, np.dtype(numpy_name)


def _period(signal: Any) -> tuple[int, int] | None:
    """Return the sample period in nanoseconds as ``(numerator, denominator)``."""
    rate = signal.fs
    num = int(rate.numerator)
    den = int(rate.denominator)
    if num <= 0 or den <= 0:
        return None
    return NANOSECONDS * den, num


def _identifier(value: str, fallback: str) -> str:
    """Return an NRF identifier derived from a free-form name."""
    slug = "".join(character if character.isalnum() else "-" for character in value.lower())
    slug = "-".join(part for part in slug.split("-") if part)
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}-{slug}" if slug else fallback
    return slug


def _control_fields(path: str, primary_key: str) -> list[dict[str, Any]]:
    return [
        record_field(path, primary_key, "utf8"),
        record_field(path, "time_ns", "int64"),
        record_field(path, "name", "utf8"),
        record_field(path, "value", "float64", nullable=True),
        record_field(path, "text", "utf8", nullable=True),
    ]


def _register_record_sets(registry: ManifestBuilder, control_chunk_length: int) -> dict[str, str]:
    """Register every session-wide record set and return the control-kind ids.

    The trials, faults, and termination sets are not control-plane kinds, so
    they are registered here but not returned: the recorder addresses them by
    their fixed schema identifier.
    """
    control_schema_ids: dict[str, str] = {}
    for kind, primary_key in CONTROL_PRIMARY_KEYS.items():
        schema_id = f"{kind.replace('_', '-')}-v1"
        path = record_set_path(kind, schema_id)
        registry.register_record_schema(
            schema_id,
            kind=kind,
            primary_key=primary_key,
            clock_id=_SESSION_CLOCK,
            chunk_length=control_chunk_length,
            fields=_control_fields(path, primary_key),
        )
        control_schema_ids[kind] = schema_id

    trials_path = record_set_path("trials", "trials-v1")
    registry.register_record_schema(
        "trials-v1",
        kind="trials",
        primary_key="trial_id",
        clock_id=_SESSION_CLOCK,
        chunk_length=control_chunk_length,
        fields=[
            record_field(trials_path, "trial_id", "utf8"),
            record_field(trials_path, "start_ns", "int64"),
            record_field(trials_path, "stop_ns", "int64"),
            record_field(trials_path, "label", "utf8", nullable=True),
            record_field(trials_path, "outcome", "utf8", nullable=True),
        ],
    )

    faults_path = record_set_path("faults", "faults-v1")
    registry.register_record_schema(
        "faults-v1",
        kind="faults",
        primary_key="fault_id",
        clock_id=_SESSION_CLOCK,
        chunk_length=control_chunk_length,
        fields=[
            record_field(faults_path, "fault_id", "utf8"),
            record_field(faults_path, "time_ns", "int64"),
            record_field(faults_path, "code", "utf8"),
            record_field(faults_path, "stage", "utf8"),
            record_field(faults_path, "frame_sequence", "int64", nullable=True),
            record_field(faults_path, "signal_id", "int64", nullable=True),
            record_field(faults_path, "text", "utf8", nullable=True),
        ],
    )

    termination_path = record_set_path("session_termination", "termination-v1")
    registry.register_record_schema(
        "termination-v1",
        kind="session_termination",
        primary_key="termination_id",
        clock_id=_SESSION_CLOCK,
        chunk_length=1,
        fields=[
            record_field(termination_path, "termination_id", "utf8"),
            record_field(termination_path, "time_ns", "int64"),
            record_field(termination_path, "termination_kind", "utf8"),
            record_field(termination_path, "reason", "utf8"),
            record_field(termination_path, "fault_id", "utf8", nullable=True, reference="fault"),
            record_field(termination_path, "last_transaction_id", "utf8", reference="transaction"),
        ],
    )
    return control_schema_ids


def _register_discontinuities(registry: ManifestBuilder, stream_id: str, clock_id: str) -> str:
    """Register the per-stream discontinuity record set and return its id."""
    path = stream_discontinuities_path(stream_id)
    schema_id = f"discontinuities.{stream_id}"
    registry.register_record_schema(
        schema_id,
        kind="discontinuities",
        stream_id=stream_id,
        primary_key="discontinuity_id",
        clock_id=clock_id,
        chunk_length=1,
        fields=[
            record_field(path, "discontinuity_id", "utf8"),
            record_field(path, "time_ns", "int64"),
            record_field(path, "reason", "utf8"),
            record_field(path, "stream_id", "utf8", reference="stream"),
            # The segment transition is the point of the record. Without it a
            # reader knows a gap exists but cannot say which segment the data on
            # either side belongs to, and NRF forbids inferring a grid, a
            # window, or an alignment across the boundary -- which is only
            # enforceable if the boundary has an identity.
            record_field(path, "previous_segment_id", "utf8", reference="segment"),
            record_field(path, "next_segment_id", "utf8", reference="segment"),
            record_field(path, "previous_frame_sequence", "int64"),
            record_field(path, "actual_frame_sequence", "int64"),
            record_field(path, "expected_sample_index", "int64", nullable=True),
            record_field(path, "actual_sample_index", "int64", nullable=True),
            record_field(path, "missing_samples", "int64", nullable=True),
            record_field(path, "missing_device_ticks", "int64", nullable=True),
        ],
    )
    return schema_id


def build_session(
    registry: ManifestBuilder,
    schema: Any,
    streams: Sequence[StreamRecording | _ResolvedStreamRecording],
    *,
    control_chunk_length: int,
) -> SessionPlan:
    """Register every clock, unit, channel, stream, and record set of a session.

    Called once, before ``freeze()``. Raises :class:`RecorderConfigError` rather
    than registering a partial session: a manifest is immutable once frozen, so
    a stream that could not be described has no later chance to appear.
    """
    schema, resolved_specs = _resolve_stream_recordings(streams, schema)
    signals = {int(signal.id): signal for signal in schema.signals}
    native_feature_sets = {int(item.id): item for item in schema.feature_sets}

    specs: list[_ResolvedStreamRecording] = list(resolved_specs)
    validate_stream_specs(specs, signals)
    signal_to_stream = {int(spec.signal_id): spec.stream_id for spec in specs}

    # --- clocks -----------------------------------------------------------
    # One NRF clock per native clock domain, plus the session clock every
    # control record is timestamped on. Both are host-monotonic nanosecond
    # clocks, because that is what the native frames actually carry.
    registry.register_clock(
        _SESSION_CLOCK,
        clock_type="host_monotonic",
        name="session host clock",
        rate={"numerator": NANOSECONDS, "denominator": 1},
    )
    clock_ids: dict[int, str] = {}
    for spec in specs:
        domain = int(signals[spec.signal_id].clock_domain)
        if domain in clock_ids:
            continue
        clock_id = f"clock.domain-{domain}"
        clock_ids[domain] = clock_id
        registry.register_clock(
            clock_id,
            clock_type="host_monotonic",
            name=f"native clock domain {domain}",
            rate={"numerator": NANOSECONDS, "denominator": 1},
            synchronization_domain=f"domain-{domain}",
            native_id=domain,
        )

    # --- units ------------------------------------------------------------
    unit_ids: dict[str, str] = {}

    def unit_for(symbol: str) -> str:
        if symbol not in unit_ids:
            candidate = _identifier(symbol or "dimensionless", "unit")
            while candidate in unit_ids.values():
                candidate = f"{candidate}-x"
            registry.register_unit(candidate, symbol=symbol or "1")
            unit_ids[symbol] = candidate
        return unit_ids[symbol]

    count_unit = unit_for("1")

    # --- channels ---------------------------------------------------------
    channel_ids: dict[str, list[str]] = {}
    next_idx = 0
    for spec in specs:
        if spec.kind == "feature":
            continue
        signal = signals[spec.signal_id]
        width = int(signal.n_channels)
        symbols = _unit_symbols(spec, width)
        names = _channel_names(spec, width)
        ids: list[str] = []
        for i in range(width):
            channel_id = f"{spec.stream_id}.ch-{i:04d}"
            registry.register_channel(
                channel_id,
                idx=next_idx,
                unit_id=unit_for(symbols[i]),
                name=names[i],
                electrode_id=None,
            )
            ids.append(channel_id)
            next_idx += 1
        channel_ids[spec.stream_id] = ids

    # --- the frame schema grouping ---------------------------------------
    frame_schema_id = f"schema-{int(schema.id)}"
    stream_ids: list[str] = []
    for spec in specs:
        stream_ids.append(spec.stream_id)
        if spec.block_index:
            stream_ids.append(spec.block_index_stream_id)
    registry.register_schema(frame_schema_id, stream_ids=stream_ids, native_id=int(schema.id))

    # --- record sets ------------------------------------------------------
    control_schema_ids = _register_record_sets(registry, control_chunk_length)

    # --- streams ----------------------------------------------------------
    discontinuity_schema_ids: dict[str, str] = {}

    plans: dict[int, StreamPlan] = {}
    for spec in specs:
        signal = signals[spec.signal_id]
        clock_id = clock_ids[int(signal.clock_domain)]
        nrf_dtype, numpy_dtype = _signal_dtype(signal, spec.stream_id)
        width = int(signal.n_channels)
        if width <= 0:  # pragma: no cover - guarded by native schema validation
            raise RecorderConfigError(f"stream {spec.stream_id!r} records a zero-width signal")
        symbols = _unit_symbols(spec, width)
        unit_ids_for_columns = [unit_for(symbol) for symbol in symbols]
        period = _period(signal)

        feature_set_id: str | None = None
        source_stream_ids = list(spec.source_stream_ids)
        if spec.kind == "feature":
            feature_set_id = f"{spec.stream_id}.set"
            descriptor = require_feature_descriptor(spec, signal, native_feature_sets)
            # The descriptor's numeric source_stream_id is the authoritative
            # native linkage; map it to the NRF stream that records that signal.
            source_stream_ids, source = _resolve_feature_source(spec, descriptor, signal_to_stream)
            names = _feature_names(spec, descriptor, width)
            window = int(descriptor.window_length_ns)
            shift = int(descriptor.shift_ns)
            registry.register_feature_set(
                feature_set_id,
                feature_names=names,
                unit_ids=unit_ids_for_columns,
                source_stream_id=source,
                algorithm_name=_descriptor_text(descriptor, "algorithm_name", spec.algorithm_name),
                algorithm_version=_descriptor_text(
                    descriptor, "algorithm_version", spec.algorithm_version
                ),
                window_length_ns=window,
                shift_ns=shift,
                native_id=int(signal.feature_set_id) or None,
            )

        discontinuity_schema_ids[spec.stream_id] = _register_discontinuities(
            registry, spec.stream_id, clock_id
        )
        registry.register_stream(
            spec.stream_id,
            kind=spec.kind,
            dtype=nrf_dtype,
            channel_ids=channel_ids.get(spec.stream_id, []),
            unit_ids=unit_ids_for_columns,
            clock_id=clock_id,
            schema_id=frame_schema_id,
            chunk_length=spec.chunk_length,
            capacity=spec.capacity,
            discontinuity_record_schema_id=discontinuity_schema_ids[spec.stream_id],
            initial_segment_id=segment_id(1),
            timing_mode=spec.timing,
            rate=(
                {
                    "numerator": int(signal.fs.numerator) or 1,
                    "denominator": int(signal.fs.denominator) or 1,
                }
                if spec.timing == "regular"
                else None
            ),
            # A regular stream stores no timestamps, so this origin is the only
            # thing anchoring its time axis. It is the caller's declaration, and
            # the recorder refuses a first block that disagrees with it.
            segment_start_idx=spec.segment_start_index,
            segment_start_time_ns=spec.segment_start_time_ns,
            source_stream_ids=source_stream_ids,
            feature_set_id=feature_set_id,
            name=spec.name or spec.stream_id,
            native_id=spec.signal_id,
        )

        block_stream_id: str | None = None
        if spec.block_index:
            block_stream_id = spec.block_index_stream_id
            block_channel_ids: list[str] = []
            for i, column in enumerate(BLOCK_IDX_COLUMNS):
                channel_id = f"{block_stream_id}.ch-{i:04d}"
                registry.register_channel(
                    channel_id,
                    idx=next_idx,
                    unit_id=count_unit,
                    name=column,
                    # NRF leaves the channel type free-form, but the typed
                    # reconstruction in neurale.data does not: this must be a
                    # kind ChannelInfo accepts, or to_recording() refuses the
                    # session the recorder just wrote.
                    channel_type="aux",
                )
                block_channel_ids.append(channel_id)
                next_idx += 1
            discontinuity_schema_ids[block_stream_id] = _register_discontinuities(
                registry, block_stream_id, clock_id
            )
            registry.register_stream(
                block_stream_id,
                kind="behavioral",
                dtype="int64",
                channel_ids=block_channel_ids,
                unit_ids=[count_unit] * len(BLOCK_IDX_COLUMNS),
                clock_id=clock_id,
                schema_id=frame_schema_id,
                chunk_length=spec.block_index_chunk_length,
                # A block can never hold fewer than one sample, so the recorded
                # stream's capacity is also the exact upper bound on the number
                # of blocks. A Zarr shape reserves nothing on disk.
                capacity=max(spec.capacity, spec.block_index_chunk_length),
                discontinuity_record_schema_id=discontinuity_schema_ids[block_stream_id],
                initial_segment_id=segment_id(1),
                timing_mode="explicit",
                source_stream_ids=[spec.stream_id],
                name=f"{spec.name or spec.stream_id} block index",
            )

        plans[spec.signal_id] = StreamPlan(
            signal_id=spec.signal_id,
            stream_id=spec.stream_id,
            numpy_dtype=numpy_dtype,
            n_channels=width,
            channel_major=getattr(signal.layout, "name", "") == "CHANNEL_MAJOR",
            explicit_timing=spec.timing == "explicit",
            period=period,
            block_index_stream_id=block_stream_id,
            max_block_samples=int(signal.max_block_samples),
            segment_start_index=spec.segment_start_index,
            segment_start_time_ns=spec.segment_start_time_ns,
        )

    return SessionPlan(
        schema_id=int(schema.id),
        streams=plans,
        control_schema_ids=control_schema_ids,
        faults_schema_id="faults-v1",
        trials_schema_id="trials-v1",
        discontinuity_schema_ids=discontinuity_schema_ids,
    )


def block_index_row(
    *,
    frame_sequence: int,
    sample_idx_start: int,
    last_sample_idx: int,
    n_samples: int,
    row_offset: int,
    device_tick_start: int,
    host_received_ns: int,
) -> list[int]:
    """Return one block-index row in :data:`BLOCK_INDEX_COLUMNS` order."""
    return [
        frame_sequence,
        sample_idx_start,
        last_sample_idx,
        n_samples,
        row_offset,
        device_tick_start,
        host_received_ns,
    ]
