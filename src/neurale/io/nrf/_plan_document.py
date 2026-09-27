#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The native-replay plan document: its value objects, codec, and checks.

A recording plan has two forms. The recorder compiles one from a
``RecorderConfig`` and a prepared native schema; a reader rebuilds one from the
``neurale.native_replay`` manifest extension or from a spool's plan document,
with no recorder in sight. This module owns the second form and everything both
forms share -- the frozen value objects, the document codec, the fingerprint,
and every invariant a plan must satisfy. :mod:`neurale.recording._plan` owns the
first and imports what it needs from here.

The split follows ownership rather than convenience. The document's spelling is
fixed by ``specifications/nrf/v1/extensions/native-replay-v1/``, which is part
of the NRF specification, and :mod:`neurale.io.nrf` is what implements that
specification. Validating the extension is NRF-SEM-018 -- a rule the semantic
layer must enforce for any session that declares it, whether or not
:mod:`neurale.recording` was ever imported. Keeping the parser above the layer
that needs it forced ``_semantics`` to reach upward through a lazy import to
dodge a cycle at module load; the parser lives here instead, so the dependency
runs one way and the rule is enforced by the layer that owns it.

There is still exactly one parser. The semantic layer and the recorder both
call the functions below, which is what keeps a plan the recorder wrote and a
plan a reader rebuilds from being judged by two drifting sets of rules.

The **fingerprint** is the identity of what a plan records: the SHA-256 of the
RFC 8785 canonical JSON of the plan document. Two processes that compiled the
same plan agree on it without comparing objects, and a finalizer can refuse a
target that belongs to a different plan. The document carries the native schema
*content* (in prepared order, not just a numeric id) and the resolved stream
metadata, and deliberately excludes everything that changes how a session is
stored rather than what it records: capacities, chunk lengths, queue sizes,
session identity, creation time, writer version, and free-form metadata live in
the plan but not in the fingerprinted document.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from neurale.exceptions import RecorderConfigError

from ._canonical import canonical_json_bytes
from ._ledgers import (
    NATIVE_REPLAY_EXTENSION_VERSION,
    NATIVE_REPLAY_MINOR_VERSION,
    NATIVE_REPLAY_NAMESPACE,
    ledger_schema_ids,
)

#: Default spool capacity, and the value ``ResourceBounds.spool_capacity_bytes``
#: is omitted from the document for. It lives here rather than with the recorder
#: config because the document format froze it: changing it would change how
#: every historical plan document round-trips. :mod:`neurale.recording._spec`
#: imports it from here for the config default.
_DEFAULT_SPOOL_CAPACITY_BYTES = 64 << 20

REPLAY_MODES: tuple[str, ...] = ("exact_frames", "recorded_projection", "stream_frames")

_FULL = "full"
_PARTIAL = "partial"

#: Native signal kind -> the NRF stream kinds it may be recorded as. The one
#: acceptability rule the plan compiler and the session builder share, so a
#: plan the compiler accepts is one the registry can build and vice versa.
_KIND_COMPATIBILITY: Mapping[str, frozenset[str]] = {
    "SAMPLED": frozenset({"neural", "behavioral"}),
    "FEATURE": frozenset({"feature"}),
}


def _sorted_tuple(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted(values))


def _descriptor_text(descriptor: Any, attribute: str, fallback: str) -> str:
    """Return the descriptor's *attribute*, falling back when it is empty.

    Native validation does not require the algorithm name and version to be
    non-empty, so a descriptor that leaves them blank is legal and the
    declaration's own text is used instead.
    """
    return str(getattr(descriptor, attribute, "")) or fallback


# --- the document codec ----------------------------------------------------
#
# Every prepared value object below is serialized to a plain document and read
# back from one, and the two directions must stay exact inverses: the document
# is what the plan fingerprint is taken over, and the rebuilt plan is checked
# by recomputing that fingerprint. Writing both by hand meant one field spelled
# twice per class and a third time in the rebuild, with nothing but review
# holding the three together.
#
# So both directions are derived from the dataclass fields instead. A field's
# annotation says how it travels; the two tables below carry the handful of
# fields whose document form is not simply the field value. A field added to a
# dataclass is therefore serialized and rebuilt without being named again.


def _field_types(cls: type) -> Mapping[str, Any]:
    """Return *cls*'s resolved field annotations, cached per class."""
    cached = _FIELD_TYPE_CACHE.get(cls)
    if cached is None:
        hints = get_type_hints(cls)
        cached = {field.name: hints[field.name] for field in fields(cls)}
        _FIELD_TYPE_CACHE[cls] = cached
    return cached


_FIELD_TYPE_CACHE: dict[type, Mapping[str, Any]] = {}


def _encode(value: Any, kind: Any) -> Any:
    """Return the document form of *value*, whose annotation is *kind*."""
    if get_origin(kind) is tuple:
        (item,) = (arg for arg in get_args(kind) if arg is not Ellipsis)
        return [_encode(entry, item) for entry in value]
    if is_dataclass(kind):
        return _document_of(value)
    return value


def _decode(value: Any, kind: Any, label: str, name: str) -> Any:
    """Return the field value *value* stands for, or raise if it cannot."""
    if get_origin(kind) is tuple:
        _expect(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
            f"{label}.{name} must be an array",
        )
        (item,) = (arg for arg in get_args(kind) if arg is not Ellipsis)
        return tuple(_decode(entry, item, f"{label}.{name}", "item") for entry in value)
    if is_dataclass(kind):
        return _rebuild(kind, value, f"{label}.{name}")
    if kind is bool:
        # Checked rather than coerced: `bool("false")` is True, and a plan that
        # read a string as an enabled flag would be wrong in the safe-looking
        # direction.
        _expect(isinstance(value, bool), f"{label}.{name} must be a boolean")
        return value
    if kind is int:
        _expect(
            isinstance(value, int) and not isinstance(value, bool),
            f"{label}.{name} must be an integer",
        )
        return value
    return str(value)


def _document_of(value: Any) -> dict[str, Any]:
    """Return the document form of one prepared value object."""
    cls = type(value)
    types = _field_types(cls)
    document: dict[str, Any] = {}
    for name, kind in types.items():
        field_value = getattr(value, name)
        omitted = _OMITTED_WHEN.get((cls.__name__, name), _NO_DEFAULT)
        if omitted is not _NO_DEFAULT and field_value == omitted:
            continue
        override = _FIELD_CODECS.get((cls.__name__, name))
        document[name] = override[0](field_value) if override else _encode(field_value, kind)
    return document


def _rebuild(cls: type, value: Any, label: str) -> Any:
    """Rebuild one prepared value object from its document form."""
    _expect(isinstance(value, Mapping), f"a {label} entry must be an object")
    arguments: dict[str, Any] = {}
    for name, kind in _field_types(cls).items():
        omitted = _OMITTED_WHEN.get((cls.__name__, name), _NO_DEFAULT)
        if omitted is not _NO_DEFAULT and name not in value:
            arguments[name] = omitted
            continue
        raw = _require_field(value, name, label)
        override = _FIELD_CODECS.get((cls.__name__, name))
        arguments[name] = (
            override[1](raw, label, name) if override else _decode(raw, kind, label, name)
        )
    return cls(**arguments)


def _encode_rate(value: tuple[int, int]) -> dict[str, int]:
    return {"numerator": value[0], "denominator": value[1]}


def _decode_rate(value: Any, label: str, name: str) -> tuple[int, int]:
    _expect(
        isinstance(value, Mapping) and "numerator" in value and "denominator" in value,
        f"{label}.{name} must have numerator and denominator",
    )
    return (int(value["numerator"]), int(value["denominator"]))


class _NoDefault:
    __slots__ = ()


_NO_DEFAULT = _NoDefault()

#: Fields whose document form is not the field value, by class and field name.
_FIELD_CODECS: dict[tuple[str, str], tuple[Any, Any]] = {
    ("PlannedSignal", "fs"): (_encode_rate, _decode_rate),
}

#: Fields left out of the document when they hold this value, and restored to
#: it when the document does not carry them. This is what keeps historical plan
#: documents byte-compatible while a caller-selected bound stays reconstructible.
_OMITTED_WHEN: dict[tuple[str, str], Any] = {
    ("ResourceBounds", "spool_capacity_bytes"): _DEFAULT_SPOOL_CAPACITY_BYTES,
}


class _Documented:
    """Serializes itself from its dataclass fields.

    Mixed into the value objects whose document is exactly their fields. The
    three that select, rename, or compute keys -- :class:`PreparedMetadata`,
    :class:`ReplayCapability`, and :class:`RecordingPlan` -- define their own
    and are deliberately not here.
    """

    __slots__ = ()

    def document(self) -> dict[str, Any]:
        return _document_of(self)


# --- frozen prepared value objects ---------------------------------------


@dataclass(frozen=True, slots=True)
class PlannedUnit(_Documented):
    """One unit descriptor, frozen exactly as the prepared schema declared it."""

    id: int
    symbol: str
    description: str


@dataclass(frozen=True, slots=True)
class PlannedFeatureSet(_Documented):
    """One feature-set descriptor, with every field native equivalence compares."""

    id: int
    feature_names: tuple[str, ...]
    unit_ids: tuple[int, ...]
    source_stream_id: int
    source_stream: str
    algorithm_name: str
    algorithm_version: str
    window_length_ns: int
    shift_ns: int
    timestamp_reference: str


@dataclass(frozen=True, slots=True)
class PlannedSignal(_Documented):
    """One signal, with every field ``StreamSchema::equivalent()`` compares.

    ``fs`` is stored as a ``(numerator, denominator)`` pair so the plan
    is a plain immutable value with no live native object held.
    """

    id: int
    clock_domain: int
    n_channels: int
    nominal_block_samples: int
    max_block_samples: int
    fs: tuple[int, int]
    dtype: str
    layout: str
    device_tick_tracking: str
    kind: str
    physical_unit: str
    channel_set_id: int
    calibration_id: int
    reference_id: int
    feature_set_id: int
    observation_timing: str
    fixed_block_bytes: int
    max_block_bytes: int


@dataclass(frozen=True, slots=True)
class NativeSchema(_Documented):
    """The complete prepared native schema, frozen for the plan.

    Signals, feature sets, and units are kept in the order the prepared schema
    declared them: ``StreamSchema::equivalent()`` compares by position, so two
    schemas that carry the same entries in a different order are not equivalent,
    and the fingerprint must reflect that. Order is therefore part of the
    fingerprinted document, not normalized away.
    """

    schema_id: int
    signals: tuple[PlannedSignal, ...]
    feature_sets: tuple[PlannedFeatureSet, ...]
    units: tuple[PlannedUnit, ...]

    @property
    def signal_ids(self) -> tuple[int, ...]:
        return tuple(signal.id for signal in self.signals)


@dataclass(frozen=True, slots=True)
class PreparedStream(_Documented):
    """One native signal, bound to one NRF stream, with all recording semantics.

    Every field that reaches the canonical NRF manifest is carried resolved and
    frozen, so a finalizer that holds only the plan knows which unit, channel
    names, feature names, regular-timing origin, feature algorithm, and source
    links to write without re-reading the original declaration. Fields that do
    not reach the manifest for a given kind are resolved to a canonical empty
    value (an empty tuple, an empty string, or zero) so the fingerprint depends
    on what is recorded, never on how a declaration happened to spell a value
    the manifest does not carry: a scalar ``unit`` and a per-column ``unit``
    that expand to the same symbols fingerprint identically, and a ``regular``
    timing origin that an ``explicit`` stream ignores does not move the
    fingerprint. Storage sizing (``capacity``, ``chunk_length``,
    ``block_index_chunk_length``) is deliberately absent: it lives in
    :class:`PreparedStreamBounds` under :class:`ResourceBounds`, not here.
    """

    native_signal_id: int
    stream_id: str
    kind: str
    timing: str
    block_index: bool
    name: str
    unit: tuple[str, ...]
    channel_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    segment_start_index: int
    segment_start_time_ns: int
    algorithm_name: str
    algorithm_version: str
    window_length_ns: int
    shift_ns: int
    source_stream_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionIdentity(_Documented):
    """Who recorded this session. Identity, not what is recorded."""

    session_id: str
    created_at: str
    writer_name: str
    writer_version: str


@dataclass(frozen=True, slots=True)
class PreparedStreamBounds(_Documented):
    """The storage sizing of one stream: how it is written, not what it records.

    A frozen value object, not a dictionary, so a compiled plan is deeply
    immutable: no caller can mutate a capacity after the plan is prepared and
    have the change surface in the manifest extension.
    """

    stream_id: str
    capacity: int
    chunk_length: int
    block_index_chunk_length: int


@dataclass(frozen=True, slots=True)
class ResourceBounds(_Documented):
    """Storage sizing: how a session is written, not what it records.

    Carried in the plan so the recorder and finalizer share one source of truth,
    but excluded from the fingerprinted document: two plans that differ only in
    bounds record the same thing.
    """

    frame_queue_capacity: int
    control_queue_capacity: int
    spool_capacity_bytes: int
    control_chunk_length: int
    checkpoint_interval: int
    overflow_policy: str
    streams: tuple[PreparedStreamBounds, ...]


@dataclass(frozen=True, slots=True)
class PreparedMetadata:
    """Free-form session or manifest metadata, frozen and non-fingerprinted.

    Stored as canonical JSON text so the plan holds no mutable dict or list: a
    prepared plan is deeply immutable. The text round-trips through the manifest
    extension because metadata is free-form JSON, and it is excluded from the
    fingerprint -- metadata describes the session, not what it records.
    """

    json_text: str

    def document(self) -> dict[str, Any]:
        return json.loads(self.json_text) if self.json_text else {}

    @classmethod
    def of(cls, value: Mapping[str, Any] | None) -> PreparedMetadata:
        if not value:
            return cls(json_text="{}")
        return cls(json_text=json.dumps(dict(value), sort_keys=True, ensure_ascii=False))


@dataclass(frozen=True, slots=True)
class ReplayCapability:
    """Whether one replay mode is available, and why not when it is not."""

    mode: str
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and self.reason is not None:
            raise RecorderConfigError(f"available capability {self.mode!r} carries a reason")
        if not self.available and not self.reason:
            raise RecorderConfigError(f"unavailable capability {self.mode!r} states no reason")

    def document(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RecordingPlan:
    """An immutable, fingerprinted statement of what a session records."""

    native_schema: NativeSchema
    #: The streams this session records, ordered by native signal ID so the
    #: fingerprint does not depend on the order the caller declared them in.
    streams: tuple[PreparedStream, ...]
    #: Session identity -- in the plan, not the fingerprint.
    session: SessionIdentity
    #: Storage sizing -- in the plan, not the fingerprint.
    resource_bounds: ResourceBounds
    #: Free-form session metadata (``session.metadata``) -- frozen, not fingerprinted.
    session_metadata: PreparedMetadata
    #: Free-form manifest metadata (``metadata``) -- frozen, not fingerprinted.
    metadata: PreparedMetadata

    @property
    def native_schema_id(self) -> int:
        return self.native_schema.schema_id

    @property
    def planned_signal_ids(self) -> tuple[int, ...]:
        """Every signal the prepared schema declared, ascending."""
        return _sorted_tuple(self.native_schema.signal_ids)

    @property
    def recorded_signal_ids(self) -> tuple[int, ...]:
        return tuple(stream.native_signal_id for stream in self.streams)

    @property
    def coverage(self) -> str:
        """``full`` when every declared signal is recorded, else ``partial``."""
        return _FULL if self.recorded_signal_ids == self.planned_signal_ids else _PARTIAL

    @property
    def covers_every_signal(self) -> bool:
        return self.coverage == _FULL

    def stream_for(self, native_signal_id: int) -> PreparedStream | None:
        for stream in self.streams:
            if stream.native_signal_id == native_signal_id:
                return stream
        return None

    @property
    def max_block_count(self) -> int:
        """Most blocks one frame of this schema can carry: one per signal."""
        return len(self.native_schema.signals)

    @property
    def max_frame_bytes(self) -> int:
        """Upper bound on one frame's payload: the sum of signal block bytes."""
        return sum(signal.max_block_bytes for signal in self.native_schema.signals)

    # --- identity ---------------------------------------------------------

    def document(self) -> dict[str, Any]:
        """Return the canonical plan document the fingerprint is taken over.

        The native schema content (in prepared order) and the resolved stream
        metadata are in it, so the fingerprint identifies what is recorded, not
        just a numeric schema id. Capabilities, the stored fingerprint, session
        identity, resource bounds, and free-form metadata are not: they are
        derived, the value being computed, or storage and description rather
        than content.
        """
        return {
            "extension_version": NATIVE_REPLAY_EXTENSION_VERSION,
            "native_schema": self.native_schema.document(),
            "coverage": self.coverage,
            "planned_signal_ids": list(self.planned_signal_ids),
            "recorded_signal_ids": list(self.recorded_signal_ids),
            "streams": [stream.document() for stream in self.streams],
            "ledgers": ledger_schema_ids(),
        }

    @property
    def fingerprint(self) -> str:
        """Lower-case hexadecimal SHA-256 of the canonical plan document."""
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    # --- replay capability ------------------------------------------------

    @property
    def replay_capabilities(self) -> tuple[ReplayCapability, ...]:
        """What a replay of a session recorded under this plan may claim."""
        exact = ReplayCapability(
            "exact_frames",
            self.covers_every_signal,
            None
            if self.covers_every_signal
            else (
                f"the recording plan covered {len(self.streams)} of "
                f"{len(self.planned_signal_ids)} signals; replay this session as "
                "recorded_projection"
            ),
        )
        synthesized = any(stream.block_index for stream in self.streams)
        return (
            exact,
            ReplayCapability("recorded_projection", True),
            ReplayCapability(
                "stream_frames",
                synthesized,
                None if synthesized else "no recorded stream declares a committed block index",
            ),
        )

    def capability(self, mode: str) -> ReplayCapability:
        """Return one mode's capability, by name."""
        for capability in self.replay_capabilities:
            if capability.mode == mode:
                return capability
        raise RecorderConfigError(f"unknown replay mode {mode!r}; expected one of {REPLAY_MODES}")

    # --- serialization ----------------------------------------------------

    def manifest_extension(self) -> dict[str, Any]:
        """Return the ``extensions['neurale.native_replay']`` manifest object."""
        document = self.document()
        return {
            "extension_version": document["extension_version"],
            "plan_fingerprint": self.fingerprint,
            "native_schema": document["native_schema"],
            "coverage": document["coverage"],
            "planned_signal_ids": document["planned_signal_ids"],
            "recorded_signal_ids": document["recorded_signal_ids"],
            # The same entries the fingerprint was taken over, so a reader can
            # rebuild the plan from the manifest and recompute the value.
            "streams": document["streams"],
            "ledgers": document["ledgers"],
            "replay_capabilities": {
                capability.mode: capability.document() for capability in self.replay_capabilities
            },
            # Identity, storage sizing, and free-form metadata are carried for
            # the recorder and finalizer but are not part of the fingerprint.
            "session": self.session.document(),
            "resource_bounds": self.resource_bounds.document(),
            "session_metadata": self.session_metadata.document(),
            "metadata": self.metadata.document(),
        }

    def manifest_extensions(self) -> dict[str, Any]:
        """Return the full ``extensions`` object a manifest carries for a plan."""
        return {NATIVE_REPLAY_NAMESPACE: self.manifest_extension()}

    @property
    def nrf_minor_version(self) -> int:
        """NRF minor version a session recorded under this plan declares."""
        return NATIVE_REPLAY_MINOR_VERSION


def _require_field(value: Mapping[str, Any], name: str, label: str) -> Any:
    if name not in value:
        raise RecorderConfigError(
            f"the {NATIVE_REPLAY_NAMESPACE} extension is not a recording plan: {label} is missing {name!r}"
        )
    return value[name]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RecorderConfigError(
            f"the {NATIVE_REPLAY_NAMESPACE} extension is not a recording plan: {message}"
        )


def _require_consistent(condition: bool, message: str) -> None:
    """Reject a plan whose streams and native schema disagree.

    Separate from :func:`_expect` because these rules are shared by the
    compiler and the extension parser: the same contradiction can arrive from
    a ``RecorderConfig`` a caller just wrote or from a manifest read back off
    disk, and telling the first caller that their configuration "is not a
    recording plan extension" names a document they never touched.
    """
    if not condition:
        raise RecorderConfigError(f"the recording plan is not consistent: {message}")


def _rebuild_native_schema(value: Mapping[str, Any]) -> NativeSchema:
    schema = _rebuild(NativeSchema, value, "native_schema")
    # Field shapes are the codec's; these two are the schema's own rules and
    # have no field to hang off.
    _expect(schema.schema_id >= 0, "native_schema.schema_id must be a non-negative integer")
    _expect(
        len({signal.id for signal in schema.signals}) == len(schema.signals),
        "native_schema has duplicate signal ids",
    )
    return schema


def _rebuild_stream(entry: Any) -> PreparedStream:
    stream = _rebuild(PreparedStream, entry, "stream")
    _expect(
        stream.kind in {"neural", "behavioral", "feature", "spike"},
        f"stream kind {stream.kind!r} is not supported",
    )
    _expect(
        stream.timing in {"explicit", "regular"},
        f"stream timing {stream.timing!r} is not supported",
    )
    return stream


_NATIVE_SIGNAL_DTYPES = frozenset({"INT16", "INT32", "FLOAT32", "FLOAT64"})
_NATIVE_SIGNAL_LAYOUTS = frozenset({"SAMPLE_MAJOR", "CHANNEL_MAJOR"})
_NATIVE_DEVICE_TICK_TRACKING = frozenset({"UNAVAILABLE", "SAMPLE_COUNTER"})
_NATIVE_SIGNAL_KINDS = frozenset({"SAMPLED", "EVENT", "FEATURE", "SPIKE"})
_NATIVE_PHYSICAL_UNITS = frozenset({"UNSPECIFIED", "VOLTS", "AMPERES", "DIMENSIONLESS"})
_NATIVE_OBSERVATION_TIMINGS = frozenset({"NOT_APPLICABLE", "REGULAR", "IRREGULAR"})
_NATIVE_TIMESTAMP_REFERENCES = frozenset({"WINDOW_CENTER"})
_NATIVE_DTYPE_BYTES: Mapping[str, int] = {
    "INT16": 2,
    "INT32": 4,
    "FLOAT32": 4,
    "FLOAT64": 8,
}
_NANOSECONDS = 1_000_000_000


def _validate_native_schema_invariants(schema: NativeSchema) -> None:
    """Reject a native_schema that no legal native ``StreamSchema`` could carry.

    A fingerprint only proves an extension was not modified after it was
    written; it does not prove the extension describes a schema the runtime
    could have constructed. These are the ``StreamSchema`` construction
    invariants -- legal enum members, positive channel counts and block limits,
    a positive sample-rate denominator, block byte counts that match the dtype
    and block limits, and a feature registry whose descriptors resolve and whose
    feature count and observation rate match their signal -- so a forged or
    corrupted extension is refused rather than fingerprinted into acceptance.
    """
    signals = schema.signals
    _expect(len(signals) > 0, "native_schema must declare at least one signal")
    signal_ids = [signal.id for signal in signals]
    _expect(len(set(signal_ids)) == len(signal_ids), "native_schema has duplicate signal ids")
    feature_sets_by_id: dict[int, PlannedFeatureSet] = {}
    for feature_set in schema.feature_sets:
        _expect(
            feature_set.id not in feature_sets_by_id,
            f"native_schema has duplicate feature-set id {feature_set.id}",
        )
        feature_sets_by_id[feature_set.id] = feature_set
    unit_ids: set[int] = set()
    unit_symbols: set[str] = set()
    for unit in schema.units:
        _expect(unit.id > 0, f"unit id {unit.id} must be nonzero")
        _expect(bool(unit.symbol), f"unit {unit.id} symbol must be non-empty")
        _expect(unit.id not in unit_ids, f"native_schema has duplicate unit id {unit.id}")
        _expect(
            unit.symbol not in unit_symbols,
            f"native_schema has duplicate unit symbol {unit.symbol!r}",
        )
        unit_ids.add(unit.id)
        unit_symbols.add(unit.symbol)
    for feature_set in schema.feature_sets:
        _expect(feature_set.id > 0, f"feature set {feature_set.id} id must be nonzero")
        _expect(
            bool(feature_set.feature_names),
            f"feature set {feature_set.id} must name at least one feature",
        )
        _expect(
            len(feature_set.feature_names) == len(set(feature_set.feature_names)),
            f"feature set {feature_set.id} feature names must be unique",
        )
        _expect(
            len(feature_set.feature_names) == len(feature_set.unit_ids),
            f"feature set {feature_set.id} feature names and unit ids must have equal length",
        )
        for name in feature_set.feature_names:
            _expect(bool(name), f"feature set {feature_set.id} has an empty feature name")
        for unit_id in feature_set.unit_ids:
            _expect(unit_id > 0, f"feature set {feature_set.id} has a zero unit id")
            _expect(
                unit_id in unit_ids,
                f"feature set {feature_set.id} references a missing unit id {unit_id}",
            )
        _expect(
            feature_set.source_stream_id > 0,
            f"feature set {feature_set.id} source_stream_id must be nonzero",
        )
        _expect(
            bool(feature_set.source_stream),
            f"feature set {feature_set.id} source_stream must be non-empty",
        )
        _expect(
            feature_set.window_length_ns > 0 and feature_set.shift_ns >= 0,
            f"feature set {feature_set.id} requires positive window length and nonnegative shift",
        )
        _expect(
            feature_set.timestamp_reference in _NATIVE_TIMESTAMP_REFERENCES,
            f"feature set {feature_set.id} timestamp_reference is not supported",
        )
    for signal in signals:
        _validate_signal_invariants(signal, feature_sets_by_id)


def _validate_signal_invariants(
    signal: PlannedSignal, feature_sets_by_id: Mapping[int, PlannedFeatureSet]
) -> None:
    label = f"signal {signal.id}"
    _expect(
        signal.dtype in _NATIVE_SIGNAL_DTYPES, f"{label} dtype {signal.dtype!r} is not supported"
    )
    _expect(
        signal.layout in _NATIVE_SIGNAL_LAYOUTS,
        f"{label} layout {signal.layout!r} is not supported",
    )
    _expect(
        signal.device_tick_tracking in _NATIVE_DEVICE_TICK_TRACKING,
        f"{label} device_tick_tracking {signal.device_tick_tracking!r} is not supported",
    )
    _expect(signal.kind in _NATIVE_SIGNAL_KINDS, f"{label} kind {signal.kind!r} is not supported")
    _expect(
        signal.physical_unit in _NATIVE_PHYSICAL_UNITS,
        f"{label} physical_unit {signal.physical_unit!r} is not supported",
    )
    _expect(
        signal.observation_timing in _NATIVE_OBSERVATION_TIMINGS,
        f"{label} observation_timing {signal.observation_timing!r} is not supported",
    )
    _expect(signal.n_channels > 0, f"{label} channel count must be positive")
    _expect(signal.nominal_block_samples > 0, f"{label} nominal block samples must be positive")
    _expect(signal.max_block_samples > 0, f"{label} max block samples must be positive")
    _expect(
        signal.nominal_block_samples <= signal.max_block_samples,
        f"{label} nominal block samples exceed the maximum",
    )
    num, den = signal.fs
    _expect(num >= 0, f"{label} sample rate numerator must be non-negative")
    _expect(den > 0, f"{label} sample rate denominator must be positive")
    kind = signal.kind
    if kind == "SAMPLED":
        _expect(num > 0, f"{label} sampled signal sample rate must be positive")
        _expect(
            signal.feature_set_id == 0, f"{label} sampled signal cannot reference feature metadata"
        )
        _expect(
            signal.observation_timing == "NOT_APPLICABLE",
            f"{label} sampled signal observation timing must be not_applicable",
        )
    elif kind == "EVENT":
        _expect(num == 0 and den == 1, f"{label} event signal requires a zero rate")
        _expect(
            signal.nominal_block_samples == 1 and signal.max_block_samples == 1,
            f"{label} event signal requires one event per block",
        )
        _expect(
            signal.feature_set_id == 0, f"{label} event signal cannot reference feature metadata"
        )
        _expect(
            signal.observation_timing == "NOT_APPLICABLE",
            f"{label} event signal observation timing must be not_applicable",
        )
    elif kind == "FEATURE":
        _expect(
            signal.feature_set_id != 0,
            f"{label} feature signal requires a feature-set descriptor id",
        )
        _expect(
            signal.observation_timing in {"REGULAR", "IRREGULAR"},
            f"{label} feature signal requires regular observation timing",
        )
        if signal.observation_timing == "IRREGULAR":
            _expect(
                num == 0
                and den == 1
                and signal.nominal_block_samples == signal.max_block_samples == 1,
                f"{label} irregular feature signal requires zero rate and one observation",
            )
        else:
            _expect(num > 0, f"{label} feature signal observation rate must be positive")
        _expect(
            signal.layout == "SAMPLE_MAJOR", f"{label} feature signal must use sample-major layout"
        )
        _expect(
            signal.physical_unit == "UNSPECIFIED",
            f"{label} feature signal physical unit must be unspecified",
        )
    elif kind == "SPIKE":
        _expect(num == 0 and den == 1, f"{label} spike signal requires a zero rate")
        _expect(
            signal.feature_set_id == 0, f"{label} spike signal cannot reference feature metadata"
        )
        _expect(
            signal.observation_timing == "NOT_APPLICABLE",
            f"{label} spike signal observation timing must be not_applicable",
        )
        _expect(
            signal.fixed_block_bytes > 0,
            f"{label} spike signal requires a fixed sparse-block payload size",
        )
    if kind == "SPIKE":
        expected_bytes = signal.fixed_block_bytes
    else:
        expected_bytes = (
            signal.n_channels * signal.max_block_samples * _NATIVE_DTYPE_BYTES[signal.dtype]
        )
    _expect(
        signal.max_block_bytes == expected_bytes,
        f"{label} max_block_bytes {signal.max_block_bytes} does not match the dtype and block limits ({expected_bytes})",
    )
    if kind == "FEATURE":
        descriptor = feature_sets_by_id.get(signal.feature_set_id)
        _expect(
            descriptor is not None,
            f"{label} references a missing feature-set descriptor {signal.feature_set_id}",
        )
        assert descriptor is not None
        _expect(
            len(descriptor.feature_names) == signal.n_channels,
            f"{label} feature count does not match its descriptor",
        )
        _expect(
            descriptor.shift_ns == 0
            if signal.observation_timing == "IRREGULAR"
            else num * descriptor.shift_ns == den * _NANOSECONDS,
            f"{label} feature observation rate must equal the descriptor shift",
        )


def _validate_prepared_streams_against_native_schema(
    native_schema: NativeSchema, streams: Sequence[PreparedStream]
) -> None:
    """A prepared stream mapping must be consistent with the native schema it records.

    A fingerprint proves the extension was not modified; it does not prove the
    streams and the native schema agree. Each stream's native signal must exist,
    its NRF kind must be one the native signal's kind allows, its per-column unit
    and channel/feature name counts must match the signal's channel count, a
    feature stream's algorithm, window, and shift must match its descriptor and
    its source linkage must include the descriptor's source, every declared
    source must be a stream this plan records, and an explicit stream's segment
    origin must be normalized to zero. The compiler and the extension parser
    share this one set of acceptability rules.
    """
    signals_by_id = {signal.id: signal for signal in native_schema.signals}
    feature_sets_by_id = {fs.id: fs for fs in native_schema.feature_sets}
    signal_to_stream = {stream.native_signal_id: stream.stream_id for stream in streams}
    recorded_stream_ids = {stream.stream_id for stream in streams}
    for stream in streams:
        # Source linkage is not a feature-stream privilege: a behavioral stream
        # may declare the neural stream it was recorded alongside. Whatever the
        # kind, a source that this plan does not record cannot be written into a
        # legal manifest, and the plan is refused now -- before the manifest
        # freezes -- rather than at the first append.
        for source_stream_id in stream.source_stream_ids:
            _require_consistent(
                source_stream_id in recorded_stream_ids,
                f"stream {stream.stream_id!r} references source stream "
                f"{source_stream_id!r}, which this plan does not record",
            )
        signal = signals_by_id.get(stream.native_signal_id)
        _require_consistent(
            signal is not None,
            f"stream {stream.stream_id!r} records native signal {stream.native_signal_id}, which the schema does not declare",
        )
        assert signal is not None
        native_kind = signal.kind
        allowed = _KIND_COMPATIBILITY.get(native_kind)
        _require_consistent(
            allowed is not None,
            f"stream {stream.stream_id!r} records a {native_kind.lower()!r} signal, which this recorder does not write",
        )
        _require_consistent(
            stream.kind in allowed,
            f"stream {stream.stream_id!r} is {stream.kind!r} but its native signal is {native_kind.lower()!r}; expected {sorted(allowed)}",
        )
        _require_consistent(
            len(stream.unit) == signal.n_channels,
            f"stream {stream.stream_id!r} unit count does not match its channel count",
        )
        if stream.kind == "feature":
            _require_consistent(
                len(stream.feature_names) == signal.n_channels,
                f"stream {stream.stream_id!r} feature name count does not match its channel count",
            )
            _require_consistent(
                stream.channel_names == (),
                f"feature stream {stream.stream_id!r} must not name channels",
            )
            descriptor = feature_sets_by_id.get(signal.feature_set_id)
            _require_consistent(
                descriptor is not None,
                f"stream {stream.stream_id!r} references a missing feature-set descriptor "
                f"{signal.feature_set_id}",
            )
            assert descriptor is not None  # narrowing for the type checker
            _require_consistent(
                stream.algorithm_name
                == _descriptor_text(descriptor, "algorithm_name", stream.algorithm_name),
                f"stream {stream.stream_id!r} algorithm name does not match its descriptor",
            )
            _require_consistent(
                stream.algorithm_version
                == _descriptor_text(descriptor, "algorithm_version", stream.algorithm_version),
                f"stream {stream.stream_id!r} algorithm version does not match its descriptor",
            )
            _require_consistent(
                stream.window_length_ns == descriptor.window_length_ns,
                f"stream {stream.stream_id!r} window length does not match its descriptor",
            )
            _require_consistent(
                stream.shift_ns == descriptor.shift_ns,
                f"stream {stream.stream_id!r} shift does not match its descriptor",
            )
            source_stream = signal_to_stream.get(descriptor.source_stream_id)
            _require_consistent(
                source_stream is not None,
                f"stream {stream.stream_id!r} descriptor source signal {descriptor.source_stream_id} is not recorded",
            )
            _require_consistent(
                source_stream in stream.source_stream_ids,
                f"stream {stream.stream_id!r} source linkage does not include its descriptor source {source_stream!r}",
            )
        else:
            _require_consistent(
                len(stream.channel_names) == signal.n_channels,
                f"stream {stream.stream_id!r} channel name count does not match its channel count",
            )
            _require_consistent(
                stream.feature_names == (),
                f"non-feature stream {stream.stream_id!r} must not name features",
            )
        if stream.timing == "explicit":
            _require_consistent(
                stream.segment_start_index == 0,
                f"explicit stream {stream.stream_id!r} segment start index must be zero",
            )
            _require_consistent(
                stream.segment_start_time_ns == 0,
                f"explicit stream {stream.stream_id!r} segment start time must be zero",
            )


def plan_from_manifest_extension(extension: Mapping[str, Any]) -> RecordingPlan:
    """Rebuild and verify a plan from a manifest extension object.

    Used to answer "what can this recorded session be replayed as?" without the
    prepared schema that produced it. The extension carries every member the
    fingerprint is taken over, plus the identity, bounds, and metadata the
    recorder froze, and nothing that was excluded from the fingerprint. The
    rebuilt plan fingerprints identically to the plan the session was recorded
    under -- which is what makes the stored fingerprint checkable rather than
    merely stored -- so the stored fingerprint is recomputed and compared, and
    every derived field (coverage, signal-id sets, replay capabilities) is
    re-derived and checked against the stored copy. A forged or corrupted
    extension raises rather than being silently regularized.
    """
    try:
        _expect(isinstance(extension, Mapping), "extension must be an object")
        _expect(
            int(_require_field(extension, "extension_version", "extension"))
            == NATIVE_REPLAY_EXTENSION_VERSION,
            "extension_version is not supported",
        )

        native_schema = _rebuild_native_schema(
            _require_field(extension, "native_schema", "extension")
        )
        _validate_native_schema_invariants(native_schema)

        planned = tuple(
            int(value) for value in _require_field(extension, "planned_signal_ids", "extension")
        )
        recorded = tuple(
            int(value) for value in _require_field(extension, "recorded_signal_ids", "extension")
        )
        _expect(
            list(planned) == sorted(planned) and len(set(planned)) == len(planned),
            "planned_signal_ids must be sorted and unique",
        )
        _expect(
            list(recorded) == sorted(recorded) and len(set(recorded)) == len(recorded),
            "recorded_signal_ids must be sorted and unique",
        )
        _expect(
            set(recorded) <= set(planned),
            "recorded_signal_ids must be a subset of planned_signal_ids",
        )
        _expect(
            set(planned) == set(native_schema.signal_ids),
            "planned_signal_ids must equal the native schema's signal ids",
        )

        raw_streams = _require_field(extension, "streams", "extension")
        _expect(
            isinstance(raw_streams, Sequence) and not isinstance(raw_streams, (str, bytes)),
            "streams must be an array",
        )
        streams = tuple(_rebuild_stream(entry) for entry in raw_streams)
        _expect(
            tuple(stream.native_signal_id for stream in streams) == recorded,
            "streams must name exactly the recorded signals in recorded order",
        )
        stream_ids = [stream.stream_id for stream in streams]
        _expect(len(set(stream_ids)) == len(stream_ids), "stream ids must be unique")
        _validate_prepared_streams_against_native_schema(native_schema, streams)

        coverage = str(_require_field(extension, "coverage", "extension"))
        _expect(coverage in {_FULL, _PARTIAL}, "coverage must be full or partial")

        session = _rebuild_session(_require_field(extension, "session", "extension"))
        resource_bounds = _rebuild_resource_bounds(
            _require_field(extension, "resource_bounds", "extension")
        )
        session_metadata = _rebuild_metadata(
            _require_field(extension, "session_metadata", "extension")
        )
        metadata = _rebuild_metadata(_require_field(extension, "metadata", "extension"))

        plan = RecordingPlan(
            native_schema=native_schema,
            streams=streams,
            session=session,
            resource_bounds=resource_bounds,
            session_metadata=session_metadata,
            metadata=metadata,
        )

        # Recompute and compare the stored fingerprint: a reader must not trust
        # a stored value the rebuilt plan does not reproduce.
        stored_fingerprint = str(_require_field(extension, "plan_fingerprint", "extension"))
        _expect(
            plan.fingerprint == stored_fingerprint,
            "plan_fingerprint does not match the rebuilt plan",
        )

        # Re-derive and compare the fields that are conclusions, not inputs.
        _expect(plan.coverage == coverage, "coverage does not match the recorded signal set")
        _expect(
            plan.planned_signal_ids == planned,
            "planned_signal_ids does not match the native schema",
        )
        _expect(
            plan.recorded_signal_ids == recorded, "recorded_signal_ids does not match the streams"
        )

        _verify_replay_capabilities(extension, plan)

        # ledgers is part of the fingerprinted document, so it must reproduce.
        stored_ledgers = _require_field(extension, "ledgers", "extension")
        _expect(isinstance(stored_ledgers, Mapping), "ledgers must be an object")
        _expect(
            dict(stored_ledgers) == ledger_schema_ids(),
            "ledgers do not match the native replay ledger set",
        )

        return plan
    except RecorderConfigError:
        # Already stated by _expect, in the vocabulary of the rule that failed.
        # RecorderConfigError is a ValueError, so without this the handler below
        # would wrap the parser's own diagnostics in a second copy of the prefix.
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise RecorderConfigError(
            f"the {NATIVE_REPLAY_NAMESPACE} manifest extension is not a recording plan: {error}"
        ) from error


def _rebuild_session(value: Any) -> SessionIdentity:
    return _rebuild(SessionIdentity, value, "session")


def _rebuild_metadata(value: Any) -> PreparedMetadata:
    # Not field-derived: metadata is free-form JSON, and the document *is* the
    # value rather than a serialization of one field.
    _expect(isinstance(value, Mapping), "metadata must be an object")
    return PreparedMetadata.of(value)


def _rebuild_resource_bounds(value: Any) -> ResourceBounds:
    return _rebuild(ResourceBounds, value, "resource_bounds")


def _verify_replay_capabilities(extension: Mapping[str, Any], plan: RecordingPlan) -> None:
    stored = _require_field(extension, "replay_capabilities", "extension")
    _expect(isinstance(stored, Mapping), "replay_capabilities must be an object")
    for capability in plan.replay_capabilities:
        entry = stored.get(capability.mode)
        _expect(isinstance(entry, Mapping), f"replay_capabilities is missing {capability.mode!r}")
        _expect(
            entry.get("available") == capability.available,
            f"replay_capabilities.{capability.mode}.available does not match the plan",
        )
        _expect(
            entry.get("reason") == capability.reason,
            f"replay_capabilities.{capability.mode}.reason does not match the plan",
        )


def plan_from_spool_document(
    document: bytes,
    *,
    session: SessionIdentity,
    resource_bounds: ResourceBounds,
    session_metadata: PreparedMetadata | None = None,
    metadata: PreparedMetadata | None = None,
) -> RecordingPlan:
    """Rebuild a plan from the canonical document a native spool stores.

    The spool carries the **fingerprinted** document (section 2.1 of the spool
    specification), which is deliberately narrower than the manifest extension:
    identity, storage sizing, and free-form metadata are not part of what a
    fingerprint identifies, so they are not in it and a caller supplies them.
    That is what makes the offline finalizer possible at all -- it decides the
    output's chunk lengths and capacities from what the committed prefix
    actually holds, rather than inheriting a live recorder's guesses.

    Verification is by reproduction rather than by a second parser: the rebuilt
    plan is run through :func:`plan_from_manifest_extension`, which owns every
    invariant, and its canonical document is then compared with the spool's
    bytes. Byte equality is the strongest available statement that nothing was
    invented, and it makes the superblock's ``plan_fingerprint`` -- the SHA-256
    of exactly these bytes -- match by construction rather than by assertion.
    """
    try:
        value = json.loads(bytes(document).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RecorderConfigError(f"the spool's plan document is not JSON: {error}") from error
    _expect(isinstance(value, Mapping), "the plan document must be an object")
    _expect(
        int(_require_field(value, "extension_version", "plan document"))
        == NATIVE_REPLAY_EXTENSION_VERSION,
        "the plan document's extension_version is not supported",
    )

    native_schema = _rebuild_native_schema(_require_field(value, "native_schema", "plan document"))
    raw_streams = _require_field(value, "streams", "plan document")
    _expect(
        isinstance(raw_streams, Sequence) and not isinstance(raw_streams, (str, bytes)),
        "the plan document's streams must be an array",
    )
    candidate = RecordingPlan(
        native_schema=native_schema,
        streams=tuple(_rebuild_stream(entry) for entry in raw_streams),
        session=session,
        resource_bounds=resource_bounds,
        session_metadata=session_metadata or PreparedMetadata.of(None),
        metadata=metadata or PreparedMetadata.of(None),
    )
    plan = plan_from_manifest_extension(candidate.manifest_extension())
    if canonical_json_bytes(plan.document()) != bytes(document):
        raise RecorderConfigError(
            "the plan rebuilt from the spool does not reproduce the document the spool stored; "
            "the spool's plan_fingerprint would not identify this plan"
        )
    return plan
