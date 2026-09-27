#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Compile a recorder declaration into the plan a session records under.

A recorder declaration is convenient to write and awkward to reason about: it is
ordered by whoever typed it, it names signals the prepared schema may not have,
and it says nothing about which replay modes the resulting session can answer.
The compiled plan is the opposite -- checked against the prepared native schema,
and carrying its own coverage and replay capability.

The plan is the **one authoritative prepared statement** a recorder and its
finalizer share. It freezes, before the first append, everything that decides
what a session records and what exact replay can later recover: the complete
native schema (every signal, feature set, and unit, with every field
``StreamSchema::equivalent()`` compares, in the order the prepared schema
declared them), the stream mappings with every piece of stream metadata that
reaches the NRF manifest, the resource bounds, the session identity, and the
free-form session and manifest metadata. Nothing recompiles a second plan from
the config and schema behind the recorder's back: there is one plan, and the
manifest extension carries it so a reader can rebuild and check it.

This module is the *compiler* half only. The plan's value objects, its document
codec, its fingerprint, and every invariant it must satisfy live in
:mod:`neurale.io.nrf._plan_document`, because the document's spelling belongs to
the NRF native-replay extension and a reader must be able to rebuild a plan with
no recorder present. Both halves share one set of rules, so a plan this compiler
accepts is one that module will validate, and vice versa.

Nothing here writes.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from neurale import __version__
from neurale.io.nrf._plan_document import (
    _KIND_COMPATIBILITY,
    REPLAY_MODES,
    NativeSchema,
    PlannedFeatureSet,
    PlannedSignal,
    PlannedUnit,
    PreparedMetadata,
    PreparedStream,
    PreparedStreamBounds,
    RecordingPlan,
    ReplayCapability,
    ResourceBounds,
    SessionIdentity,
    _descriptor_text,
    _validate_prepared_streams_against_native_schema,
    plan_from_manifest_extension,
    plan_from_spool_document,
)

from ._errors import RecorderConfigError
from ._spec import (
    _DEFAULT_STREAM_CAPACITY,
    RecorderConfig,
    RecorderLimits,
    StreamMetadata,
    StreamRecording,
    StreamRole,
    StreamStorageConfig,
    StreamTimingConfig,
    _ResolvedStreamRecording,
)

_WRITER_NAME = "pyneurale"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


#: The plan API the rest of :mod:`neurale.recording` imports from here.
#: The value objects, the codec, and the reader entry points are defined in
#: :mod:`neurale.io.nrf._plan_document` and re-exported so that package-internal
#: callers keep one import site for "the plan".
__all__ = [
    "REPLAY_MODES",
    "NativeSchema",
    "PlannedFeatureSet",
    "PlannedSignal",
    "PlannedUnit",
    "PreparedMetadata",
    "PreparedStream",
    "PreparedStreamBounds",
    "RecordingPlan",
    "ReplayCapability",
    "ResourceBounds",
    "SessionIdentity",
    "_channel_names",
    "_descriptor_text",
    "_feature_names",
    "_resolve_feature_source",
    "_resolve_stream_recordings",
    "_unit_symbols",
    "compile_recording_plan",
    "plan_from_manifest_extension",
    "plan_from_spool_document",
    "require_feature_descriptor",
    "validate_stream_specs",
]


def _enum_name(value: Any) -> str:
    """Return the stable member name of a pybind11 enum value.

    Native schema enums are serialized by name so the fingerprint document is
    portable and independent of the integer a particular build happened to
    assign. pybind11 enums expose ``__members__`` on their class; the reverse
    lookup is robust to differences in ``__str__`` across versions.
    """
    members = getattr(type(value), "__members__", None)
    if isinstance(members, Mapping):
        for name, member in members.items():
            if member == value:
                return name
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


# --- shared stream-resolution and acceptability rules ---------------------
#
# These helpers resolve a :class:`StreamRecording` declaration against a
# prepared native signal into the exact values the NRF manifest will carry.
# They live in the plan module because the plan is the authority on what a
# recording declares; the session builder (:mod:`._registry`) imports them
# so the two never maintain separate acceptability rules.


_PHYSICAL_UNIT_SYMBOLS = {
    "UNSPECIFIED": "1",
    "DIMENSIONLESS": "1",
    "VOLTS": "V",
    "AMPERES": "A",
}


def _recording_schema(source: Any) -> Any:
    schema = getattr(source, "schema", source)
    if not hasattr(schema, "signals") or not hasattr(schema, "feature_sets"):
        raise RecorderConfigError("recording source must expose a prepared StreamSchema")
    return schema


def _selected_signal(spec: StreamRecording, signals: Sequence[Any]) -> Any:
    if spec.signal is None:
        if len(signals) != 1:
            raise RecorderConfigError(
                "a multi-signal source requires signal=<one of source.schema.signals>"
            )
        return signals[0]
    signal_id = getattr(spec.signal, "id", None)
    if signal_id is None:
        raise RecorderConfigError("signal must be one of the source StreamSchema signals")
    matches = [signal for signal in signals if int(signal.id) == int(signal_id)]
    if not matches:
        raise RecorderConfigError(
            f"selected signal {int(signal_id)} is not declared by the recording source schema"
        )
    return matches[0]


def _metadata_units(
    metadata: StreamMetadata, signal: Any, descriptor: Any, schema: Any, width: int
) -> tuple[str, ...]:
    if descriptor is not None:
        if metadata.unit is not None:
            raise RecorderConfigError("feature units come from the native feature-set descriptor")
        unit_by_id = {int(unit.id): str(unit.symbol) for unit in schema.units}
        try:
            symbols = tuple(unit_by_id[int(unit_id)] for unit_id in descriptor.unit_ids)
        except KeyError as exc:
            raise RecorderConfigError(
                f"feature descriptor references missing native unit {int(exc.args[0])}"
            ) from exc
        if len(symbols) != width:
            raise RecorderConfigError("feature descriptor unit count does not match its width")
        return symbols

    physical_name = _enum_name(signal.physical_unit)
    native_symbol = _PHYSICAL_UNIT_SYMBOLS.get(physical_name)
    if native_symbol is None:
        raise RecorderConfigError(f"unsupported native physical unit {physical_name!r}")
    if metadata.unit is None:
        return (native_symbol,) * width
    if physical_name != "UNSPECIFIED":
        raise RecorderConfigError(
            "metadata.unit is only allowed when the native physical unit is unspecified"
        )
    if isinstance(metadata.unit, str):
        return (metadata.unit,) * width
    symbols = tuple(metadata.unit)
    if len(symbols) != width:
        raise RecorderConfigError(
            f"metadata declares {len(symbols)} unit symbols for {width} columns"
        )
    return symbols


def _resolve_stream_recordings(
    specs: Sequence[StreamRecording | _ResolvedStreamRecording], source: Any
) -> tuple[Any, tuple[_ResolvedStreamRecording, ...]]:
    """Compile public stream intent against its authoritative prepared source."""

    schema = _recording_schema(source)
    if all(isinstance(spec, _ResolvedStreamRecording) for spec in specs):
        return schema, tuple(specs)  # type: ignore[return-value]
    if any(isinstance(spec, _ResolvedStreamRecording) for spec in specs):
        raise RecorderConfigError("public and private stream declarations cannot be mixed")

    signals = tuple(schema.signals)
    feature_sets = {int(item.id): item for item in schema.feature_sets}
    source_names = getattr(source, "channel_names", None)
    source_total = getattr(source, "_recording_capacity_hint", None)
    resolved: list[_ResolvedStreamRecording] = []
    selected_ids: set[int] = set()
    public_specs = tuple(specs)
    for spec in public_specs:
        assert isinstance(spec, StreamRecording)
        signal = _selected_signal(spec, signals)
        signal_id = int(signal.id)
        if signal_id in selected_ids:
            raise RecorderConfigError(f"signal {signal_id} is recorded more than once")
        selected_ids.add(signal_id)

        native_kind = _enum_name(signal.kind)
        descriptor = (
            feature_sets.get(int(signal.feature_set_id)) if native_kind == "FEATURE" else None
        )
        metadata = spec.metadata or StreamMetadata()
        if native_kind == "FEATURE":
            if metadata.role is not None:
                raise RecorderConfigError("feature stream role comes from its native descriptor")
            kind = "feature"
        elif native_kind == "SAMPLED":
            kind = str(metadata.role or StreamRole.NEURAL)
        else:
            raise RecorderConfigError(
                f"signal {signal_id} is {native_kind.lower()!r}; only sampled and feature signals can be recorded"
            )

        stream_id = spec.stream_id or kind
        width = int(signal.n_channels)
        units = _metadata_units(metadata, signal, descriptor, schema, width)
        if descriptor is None:
            if metadata.channel_names is not None:
                channel_names = tuple(metadata.channel_names)
            elif len(signals) == 1 and source_names is not None:
                channel_names = tuple(source_names)
            else:
                channel_names = tuple(f"{stream_id}-{idx}" for idx in range(width))
            if len(channel_names) != width:
                raise RecorderConfigError(
                    f"stream {stream_id!r} has {len(channel_names)} names for {width} channels"
                )
            feature_names: tuple[str, ...] = ()
            algorithm_name = ""
            algorithm_version = ""
        else:
            channel_names = ()
            feature_names = tuple(descriptor.feature_names)
            algorithm_name = str(descriptor.algorithm_name)
            algorithm_version = str(descriptor.algorithm_version)

        storage = spec.storage or StreamStorageConfig()
        inferred_capacity = (
            int(source_total)
            if storage.capacity is None and len(signals) == 1 and source_total is not None
            else _DEFAULT_STREAM_CAPACITY
        )
        capacity = storage.capacity or max(inferred_capacity, storage.chunk_length)
        timing = spec.timing or StreamTimingConfig()
        provenance = spec.provenance
        resolved.append(
            _ResolvedStreamRecording(
                signal_id=signal_id,
                stream_id=stream_id,
                capacity=capacity,
                kind=kind,
                name=metadata.name or stream_id,
                unit=units,
                channel_names=channel_names,
                feature_names=feature_names,
                chunk_length=storage.chunk_length,
                timing=str(timing.mode),
                segment_start_index=timing.segment_start_index,
                segment_start_time_ns=timing.segment_start_time_ns,
                block_index=True if provenance is None else provenance.block_index,
                block_index_chunk_length=storage.block_index_chunk_length,
                algorithm_name=algorithm_name,
                algorithm_version=algorithm_version,
            )
        )

    by_object = {
        id(spec): stream.stream_id for spec, stream in zip(public_specs, resolved, strict=True)
    }
    stream_ids = [stream.stream_id for stream in resolved]
    if len(set(stream_ids)) != len(stream_ids):
        raise RecorderConfigError("resolved recording stream IDs must be unique")
    for idx, spec in enumerate(public_specs):
        assert isinstance(spec, StreamRecording)
        if spec.provenance is None:
            continue
        source_stream_ids: list[str] = []
        for declared_source in spec.provenance.sources:
            source_stream_id = by_object.get(id(declared_source))
            if source_stream_id is None:
                missing_id = getattr(declared_source, "stream_id", None)
                raise RecorderConfigError(
                    f"stream {resolved[idx].stream_id!r} references source stream "
                    f"{missing_id!r}, which this plan does not record"
                )
            source_stream_ids.append(source_stream_id)
        resolved[idx] = replace(resolved[idx], source_stream_ids=tuple(source_stream_ids))

    return schema, tuple(resolved)


def validate_stream_specs(
    specs: Sequence[_ResolvedStreamRecording], signals: Mapping[int, Any]
) -> None:
    """Refuse a declaration the prepared schema cannot support.

    Shared by the plan compiler and the session builder. Checked before
    anything is frozen: a manifest is immutable once committed, so a stream
    that could not be described has no later chance to appear.
    """
    for spec in specs:
        if spec.signal_id not in signals:
            raise RecorderConfigError(
                f"stream {spec.stream_id!r} records signal {spec.signal_id}, which the prepared "
                f"schema does not define (it defines {sorted(signals)})"
            )
        native_kind = _enum_name(signals[spec.signal_id].kind)
        allowed = _KIND_COMPATIBILITY.get(native_kind)
        if allowed is None:
            raise RecorderConfigError(
                f"stream {spec.stream_id!r} records a {native_kind.lower()!r} signal, which this "
                "recorder does not write; it records sampled and feature signals"
            )
        if spec.kind not in allowed:
            raise RecorderConfigError(
                f"stream {spec.stream_id!r} is declared {spec.kind!r} but signal "
                f"{spec.signal_id} is {native_kind.lower()!r}; expected {sorted(allowed)}"
            )


def require_feature_descriptor(
    spec: _ResolvedStreamRecording, signal: Any, feature_sets: Mapping[int, Any]
) -> Any:
    """Return the native feature-set descriptor a ``feature`` declaration records.

    A feature stream is described entirely from its native descriptor: the
    feature names, the algorithm, the window, the shift, and the authoritative
    source linkage all come from there, and none of them may be invented. A
    window this session never used, written into a descriptor a later analysis
    is entitled to trust, is worse than no session at all.

    Resolution is expected to succeed rather than merely hoped for:
    :func:`validate_stream_specs` has already established that a ``feature``
    declaration records a native ``FEATURE`` signal, and a native
    ``StreamSchema`` cannot be constructed unless every ``FEATURE`` signal
    names a descriptor the schema carries whose feature count matches the
    signal and whose window and shift are positive
    (``cpp/src/streaming/schema.cpp``). It is still checked, because the
    schema arrives here duck-typed and a violation must surface as a refused
    configuration rather than as an ``AttributeError`` deep inside the
    compiler.
    """
    descriptor = feature_sets.get(int(signal.feature_set_id))
    if descriptor is None:
        raise RecorderConfigError(
            f"feature stream {spec.stream_id!r} records native signal {int(spec.signal_id)}, "
            f"which references a missing feature-set descriptor {int(signal.feature_set_id)}"
        )
    window = int(descriptor.window_length_ns)
    shift = int(descriptor.shift_ns)
    if window <= 0 or shift <= 0:
        # NRF-SEM-007 requires both to be positive.
        raise RecorderConfigError(
            f"feature stream {spec.stream_id!r} derives from feature set "
            f"{int(descriptor.id)}, whose window length and shift are {window} and {shift}; "
            "both must be positive"
        )
    return descriptor


def _unit_symbols(spec: _ResolvedStreamRecording, width: int) -> list[str]:
    """One unit symbol per column, expanding a scalar to every column."""
    unit = spec.unit
    if isinstance(unit, str):
        return [unit] * width
    symbols = list(unit)
    if len(symbols) != width:
        raise RecorderConfigError(
            f"stream {spec.stream_id!r} declares {len(symbols)} unit symbols for {width} columns"
        )
    return symbols


def _channel_names(spec: _ResolvedStreamRecording, width: int) -> list[str]:
    """One channel name per channel, generated when the caller did not name them."""
    names = list(spec.channel_names) if spec.channel_names is not None else []
    if names and len(names) != width:
        raise RecorderConfigError(
            f"stream {spec.stream_id!r} declares {len(names)} channel names for {width} channels"
        )
    if not names:
        names = [f"{spec.stream_id}-{idx}" for idx in range(width)]
    return names


def _feature_names(spec: _ResolvedStreamRecording, descriptor: Any, width: int) -> list[str]:
    """One name per feature: the caller's, else the native descriptor's.

    There is no generated fallback the way there is for channels: a feature
    column means nothing without the name of the feature, and the descriptor
    always carries one per column (``cpp/src/streaming/schema.cpp`` refuses a
    descriptor with an empty or short name list).
    """
    names = list(spec.feature_names) if spec.feature_names is not None else []
    if not names:
        names = list(descriptor.feature_names)
    if len(names) != width:
        raise RecorderConfigError(
            f"stream {spec.stream_id!r} declares {len(names)} feature names for {width} features"
        )
    return names


def _resolve_feature_source(
    spec: _ResolvedStreamRecording,
    descriptor: Any,
    signal_to_stream: Mapping[int, str],
) -> tuple[list[str], str]:
    """Resolve a feature stream's source linkage from the authoritative native
    descriptor.

    The native ``FeatureSetDescriptor.source_stream_id`` is a native ``SignalId``;
    it is mapped to the NRF ``stream_id`` of the recording declaration that
    records that signal. The descriptor's ``source_stream`` text is provenance
    only and does not participate in the mapping, so two NRF stream ids that
    spell a native source differently do not change the linkage. Explicit
    ``source_stream_ids`` on the declaration are additional links and must each
    resolve to a stream this plan records. There is no "first sampled stream"
    fallback: a feature stream whose authoritative source is not recorded by the
    plan cannot be described, and the plan is refused rather than linked to an
    unrelated stream -- so the same declarations in a different order resolve to
    the same linkage and the same fingerprint.

    Returns the resolved source stream ids (declared, then the descriptor
    source) and the single authoritative source for the feature-set descriptor.
    """
    recorded_stream_ids = set(signal_to_stream.values())
    declared = list(spec.source_stream_ids)
    for source_id in declared:
        if source_id not in recorded_stream_ids:
            raise RecorderConfigError(
                f"feature stream {spec.stream_id!r} declares source stream {source_id!r}, "
                "which is not recorded by this plan"
            )
    native_source_id = int(descriptor.source_stream_id)
    descriptor_source = signal_to_stream.get(native_source_id)
    if descriptor_source is None:
        raise RecorderConfigError(
            f"feature stream {spec.stream_id!r} derives from native signal "
            f"{native_source_id}, which the plan does not record"
        )
    sources = list(declared)
    if descriptor_source not in sources:
        sources.append(descriptor_source)
    return sources, descriptor_source


def _planned_signal(signal: Any) -> PlannedSignal:
    return PlannedSignal(
        id=int(signal.id),
        clock_domain=int(signal.clock_domain),
        n_channels=int(signal.n_channels),
        nominal_block_samples=int(signal.nominal_block_samples),
        max_block_samples=int(signal.max_block_samples),
        fs=(int(signal.fs.numerator), int(signal.fs.denominator)),
        dtype=_enum_name(signal.dtype),
        layout=_enum_name(signal.layout),
        device_tick_tracking=_enum_name(signal.device_tick_tracking),
        kind=_enum_name(signal.kind),
        physical_unit=_enum_name(signal.physical_unit),
        channel_set_id=int(signal.channel_set_id),
        calibration_id=int(signal.calibration_id),
        reference_id=int(signal.reference_id),
        feature_set_id=int(signal.feature_set_id),
        observation_timing=_enum_name(signal.observation_timing),
        fixed_block_bytes=int(signal.fixed_block_bytes),
        max_block_bytes=int(signal.max_block_bytes),
    )


def _planned_feature_set(descriptor: Any) -> PlannedFeatureSet:
    return PlannedFeatureSet(
        id=int(descriptor.id),
        feature_names=tuple(str(name) for name in descriptor.feature_names),
        unit_ids=tuple(int(unit_id) for unit_id in descriptor.unit_ids),
        source_stream_id=int(descriptor.source_stream_id),
        source_stream=str(descriptor.source_stream),
        algorithm_name=str(descriptor.algorithm_name),
        algorithm_version=str(descriptor.algorithm_version),
        window_length_ns=int(descriptor.window_length_ns),
        shift_ns=int(descriptor.shift_ns),
        timestamp_reference=_enum_name(descriptor.timestamp_reference),
    )


def _planned_unit(unit: Any) -> PlannedUnit:
    return PlannedUnit(
        id=int(unit.id),
        symbol=str(unit.symbol),
        description=str(unit.description),
    )


def _prepare_stream(
    spec: _ResolvedStreamRecording,
    signal: Any,
    descriptor: Any,
    signal_to_stream: Mapping[int, str],
) -> PreparedStream:
    """Resolve one declaration into the frozen stream metadata the manifest carries."""
    width = int(signal.n_channels)
    is_feature = spec.kind == "feature"
    unit_symbols = tuple(_unit_symbols(spec, width))
    if is_feature:
        channel_names: tuple[str, ...] = ()
        feature_names = tuple(_feature_names(spec, descriptor, width))
        algorithm_name = _descriptor_text(descriptor, "algorithm_name", spec.algorithm_name)
        algorithm_version = _descriptor_text(
            descriptor, "algorithm_version", spec.algorithm_version
        )
        window = int(descriptor.window_length_ns)
        shift = int(descriptor.shift_ns)
        sources, _primary = _resolve_feature_source(spec, descriptor, signal_to_stream)
        source_stream_ids = tuple(sources)
    else:
        channel_names = tuple(_channel_names(spec, width))
        feature_names = ()
        algorithm_name = ""
        algorithm_version = ""
        window = 0
        shift = 0
        source_stream_ids = tuple(spec.source_stream_ids)
    is_regular = spec.timing == "regular"
    return PreparedStream(
        native_signal_id=int(spec.signal_id),
        stream_id=spec.stream_id,
        kind=spec.kind,
        timing=spec.timing,
        block_index=bool(spec.block_index),
        name=spec.name or spec.stream_id,
        unit=unit_symbols,
        channel_names=channel_names,
        feature_names=feature_names,
        # A regular stream's time axis is this origin plus the declared rate, so
        # the origin is recording semantics and is fingerprinted. An explicit
        # stream stores the timestamps that arrived and ignores the origin, so
        # it is not part of what is recorded and is fixed at zero for the
        # fingerprint.
        segment_start_index=int(spec.segment_start_index) if is_regular else 0,
        segment_start_time_ns=int(spec.segment_start_time_ns) if is_regular else 0,
        algorithm_name=algorithm_name,
        algorithm_version=algorithm_version,
        window_length_ns=window,
        shift_ns=shift,
        source_stream_ids=source_stream_ids,
    )


def compile_recording_plan(config: RecorderConfig, source: Any) -> RecordingPlan:
    """Compile *config* against a prepared device or native schema.

    Raises :class:`RecorderConfigError` rather than returning a plan the
    prepared schema cannot support: the manifest freezes before the first
    append, so a plan that cannot be described has no later chance to be fixed.
    """
    schema, specs = _resolve_stream_recordings(config.streams, source)
    signals = {int(signal.id): signal for signal in schema.signals}
    if not signals:
        raise RecorderConfigError("the prepared schema declares no signals")

    validate_stream_specs(specs, signals)

    # The native schema is kept in prepared order: StreamSchema::equivalent()
    # compares signals, feature sets, and units by position, so the plan must
    # preserve that order for an exact schema reconstruction to be possible.
    native_schema = NativeSchema(
        schema_id=int(schema.id),
        signals=tuple(_planned_signal(signal) for signal in schema.signals),
        feature_sets=tuple(_planned_feature_set(descriptor) for descriptor in schema.feature_sets),
        units=tuple(_planned_unit(unit) for unit in schema.units),
    )
    # The native StreamSchema constructor has already enforced every invariant
    # _validate_native_schema_invariants re-derives (cpp/src/streaming/schema.cpp),
    # so they are not re-run here: on this path the schema is a live prepared
    # object, not a document read back from a file. The parser runs them because
    # there the schema arrives as untrusted JSON.

    native_feature_sets = {int(item.id): item for item in schema.feature_sets}
    # native SignalId -> NRF stream_id, so a feature set's authoritative
    # source_stream_id maps to the stream that records it, independent of the
    # order the caller declared the streams in.
    signal_to_stream = {int(spec.signal_id): spec.stream_id for spec in specs}
    streams = tuple(
        _prepare_stream(
            spec,
            signals[int(spec.signal_id)],
            require_feature_descriptor(spec, signals[int(spec.signal_id)], native_feature_sets)
            if spec.kind == "feature"
            else None,
            signal_to_stream,
        )
        # Sorted by native signal ID so the fingerprint does not depend on the
        # order the caller happened to declare the streams in.
        for spec in sorted(specs, key=lambda item: int(item.signal_id))
    )
    _validate_prepared_streams_against_native_schema(native_schema, streams)

    limits = config.limits or RecorderLimits()
    resource_bounds = ResourceBounds(
        frame_queue_capacity=int(limits.frame_queue_capacity),
        control_queue_capacity=int(limits.control_queue_capacity),
        # NRF v1 requires an I-JSON integer bound; use its exact-integer ceiling
        # for an otherwise unlimited, continuously appended spool.
        spool_capacity_bytes=(
            (1 << 53) - 1 if limits.spool_capacity_bytes is None else limits.spool_capacity_bytes
        ),
        control_chunk_length=int(limits.control_chunk_length),
        checkpoint_interval=int(limits.checkpoint_interval),
        overflow_policy="fault",
        streams=tuple(
            PreparedStreamBounds(
                stream_id=spec.stream_id,
                capacity=int(spec.capacity),
                chunk_length=int(spec.chunk_length),
                block_index_chunk_length=int(spec.block_index_chunk_length),
            )
            for spec in sorted(specs, key=lambda item: str(item.stream_id))
        ),
    )
    session = SessionIdentity(
        session_id=str(uuid.uuid4()),
        created_at=_utc_now(),
        writer_name=_WRITER_NAME,
        writer_version=__version__,
    )
    return RecordingPlan(
        native_schema=native_schema,
        streams=streams,
        session=session,
        resource_bounds=resource_bounds,
        session_metadata=PreparedMetadata.of(
            {} if config.session is None else config.session.document()
        ),
        metadata=PreparedMetadata.of(config.metadata),
    )
