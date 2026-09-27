#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Write a typed :class:`~neurale.data.Recording` to an NRF v1 session.

Specification reference: ``README.md`` section 10.

This is the writer half of the typed round trip whose reader half is
:meth:`neurale.io.nrf.NrfReader.to_recording`. It derives every registry a
session needs -- clocks, units, channels, electrodes, streams, feature sets,
and the mandatory record kinds -- from the recording itself, so a caller does
not have to restate in NRF vocabulary what the typed objects already say.

What it deliberately does not do is invent identity or reshape time. Where a
recording carries no NRF-stable ID, one is derived from the name it does carry
and then persisted; where a value is genuinely absent, it stays absent rather
than acquiring a plausible default.

Two properties are what make the round trip faithful rather than merely
lossless in the payload:

* **Every distinct clock is registered as a distinct clock.** Signals that were
  recorded on different timing sources stay on different sources, with their
  rate, epoch, offset, drift, and synchronization domain intact. A signal that
  declared no clock references a placeholder the reader restores as ``None``.
* **Timing mode follows the data, not the declaration.** A stream is registered
  as regular only when its timestamps are exactly the grid a regular reader
  would rebuild; anything else -- an irregular signal, a feature matrix with
  ``fs=None`` -- is registered with explicit timestamps.
"""

from __future__ import annotations

import re
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from neurale.data import Clock, FeatureMatrix, Recording, SignalArray

from ._canonical import SAFE_INTEGER_MAX, SAFE_INTEGER_MIN
from ._errors import NrfSemanticError
from ._fields import record_field
from ._manifest import ManifestBuilder
from ._paths import record_set_path, stream_discontinuities_path
from ._typed import (
    CLOCK_DOMAIN_KEY,
    CLOCK_TYPE_KEY,
    RECORDING_METADATA_KEY,
    RESERVED_METADATA_PREFIX,
    SYNTHETIC_CLOCK_KEY,
)
from ._writer import NrfWriter

NANOSECONDS = 1_000_000_000

_DEFAULT_CHUNK_LENGTH = 1024
_RECORD_CHUNK_LENGTH = 64

#: NumPy kind + itemsize -> NRF dtype. The format has no implicit conversion, so
#: an unsupported dtype is an error rather than something to coerce.
_NRF_DTYPES: Mapping[tuple[str, int], str] = {
    ("i", 2): "int16",
    ("i", 4): "int32",
    ("i", 8): "int64",
    ("u", 1): "uint8",
    ("u", 4): "uint32",
    ("u", 8): "uint64",
    ("f", 4): "float32",
    ("f", 8): "float64",
    ("b", 1): "bool",
}

_ID_CHARACTERS = re.compile(r"[^a-z0-9]+")

#: NRF v1 fixes the clock-type vocabulary; :class:`neurale.data.Clock` does not.
#: Anything outside the NRF enum is recorded as ``external`` -- the honest
#: answer for a clock this format has no category for -- rather than guessed
#: into a specific kind, and the original spelling is kept in the descriptor
#: metadata so the reader restores it exactly.
_NRF_CLOCK_TYPES = frozenset({"host_monotonic", "device_counter", "utc", "external", "derived"})
_CLOCK_TYPE_ALIASES: Mapping[str, str] = {
    "device": "device_counter",
    "monotonic": "host_monotonic",
    "host": "host_monotonic",
    "wall": "utc",
}


def _nrf_clock_type(value: str | None) -> str:
    if value is None:
        return "host_monotonic"
    if value in _NRF_CLOCK_TYPES:
        return value
    return _CLOCK_TYPE_ALIASES.get(value, "external")


def _identifier(value: str, *, fallback: str) -> str:
    """Return a stable NRF identifier derived from a human-facing name."""
    slug = _ID_CHARACTERS.sub("-", str(value).strip().lower()).strip("-")
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}-{slug}" if slug else fallback
    return slug


def _unique(preferred: str, taken: Container[str]) -> str:
    """Return *preferred*, suffixed until it is not already *taken*.

    Two human-facing names can slug to the same identifier. NRF registry IDs
    must be unique, so the collision is resolved by extending the identifier
    rather than by letting the second entry overwrite the first.
    """
    identifier = preferred
    while identifier in taken:
        identifier = f"{identifier}-x"
    return identifier


def _source_stream_id(source_signal: str | None) -> str | None:
    """Return the stream ID a feature matrix names as its source, if any."""
    if not source_signal:
        return None
    return _identifier(source_signal, fallback="stream")


def _nrf_dtype(arr: np.ndarray, label: str) -> str:
    key = (arr.dtype.kind, arr.dtype.itemsize)
    dtype = _NRF_DTYPES.get(key)
    if dtype is None:
        raise NrfSemanticError(
            f"{label} has dtype {arr.dtype.name!r}, which NRF v1 does not define; "
            "cast it to a supported dtype before writing"
        )
    return dtype


def _rational_rate(rate: float, label: str = "rate") -> dict[str, int]:
    """Return an exact-as-possible integer ratio for a positive rate.

    NRF stores rates as integer ratios because a binary float is not the
    normative value. The best rational approximation within a bounded
    denominator is used rather than a fixed decimal scaling, so rates a decimal
    scaling cannot represent -- ``1/3``, ``2500/7`` -- reproduce the original
    float exactly instead of drifting across a long stream. The denominator
    bound steps down until both terms fit the format's exact integer domain.
    """
    if not np.isfinite(rate) or rate <= 0:
        raise NrfSemanticError(f"{label} must be a positive finite rate, got {rate!r}")
    for limit in (1_000_000_000, 1_000_000, 1_000, 1):
        ratio = Fraction(float(rate)).limit_denominator(limit)
        if ratio <= 0:
            continue
        if max(ratio.numerator, ratio.denominator) <= SAFE_INTEGER_MAX:
            return {"numerator": ratio.numerator, "denominator": ratio.denominator}
    return {"numerator": max(round(rate), 1), "denominator": 1}


def _seconds_to_ns(value: float | None) -> int:
    return 0 if value is None else round(float(value) * NANOSECONDS)


def _json_safe(value: Any, label: str) -> Any:
    """Return *value* as JSON a manifest can carry, or refuse it by name.

    ``Recording.metadata``, ``Recording.subject``, and ``Recording.session`` are
    open dicts, while a manifest is canonical JSON with an exact integer domain.
    A value with no unambiguous JSON image is rejected and named rather than
    coerced: a silently reshaped value round-trips into something the caller
    never wrote, which is the failure this whole payload exists to prevent.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if not SAFE_INTEGER_MIN <= value <= SAFE_INTEGER_MAX:
            raise NrfSemanticError(
                f"{label} is outside the exact integer range NRF canonical JSON defines"
            )
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise NrfSemanticError(f"{label} is {value!r}, which canonical JSON cannot express")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NrfSemanticError(f"{label} has the non-string key {key!r}")
            result[key] = _json_safe(item, f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{label}[{idx}]") for idx, item in enumerate(value)]
    raise NrfSemanticError(
        f"{label} has type {type(value).__name__!r}, which NRF cannot store as JSON; "
        "convert it to a JSON value before writing"
    )


# --- timing ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Regular:
    """A stream whose timestamps are exactly the declared regular grid."""

    rate: dict[str, int]
    start_ns: int


@dataclass(frozen=True, slots=True)
class _Explicit:
    """A stream whose timestamps are stored one per item."""

    timestamps: np.ndarray


def _timing_plan(
    *,
    time: np.ndarray | None,
    fs: float | None,
    t0: float | None,
    rows: int,
    label: str,
) -> _Regular | _Explicit:
    """Decide how one stream's time axis is stored.

    Regular timing is chosen only when the timestamps are, to the nanosecond,
    the grid a reader rebuilds from the rational rate that will actually be
    written. Comparing against the *written* rate rather than the declared float
    is what keeps the decision honest: a rate the format cannot represent
    exactly would otherwise be "regular" here and drift on the way back.
    """
    stamps: np.ndarray | None = None
    if time is not None:
        stamps = np.rint(np.asarray(time, dtype=float) * NANOSECONDS).astype("int64")
        if stamps.shape[0] != rows:
            raise NrfSemanticError(f"{label} carries {stamps.shape[0]} timestamps for {rows} items")

    if fs is None:
        if stamps is None:
            raise NrfSemanticError(f"{label} declares neither a sampling rate nor timestamps")
        return _Explicit(stamps)

    rate = _rational_rate(fs, f"{label} sampling rate")
    effective = rate["numerator"] / rate["denominator"]
    start = 0.0 if t0 is None else float(t0)
    grid = np.rint((start + np.arange(rows, dtype=float) / effective) * NANOSECONDS).astype("int64")
    if stamps is None or np.array_equal(stamps, grid):
        return _Regular(rate, round(start * NANOSECONDS))
    return _Explicit(stamps)


def _timing_hint(fs: float | None, t0: float | None) -> dict[str, Any]:
    """Return what an explicitly timed stream still needs to round-trip.

    NRF's explicit timing block stores timestamps and nothing else, but a
    :class:`~neurale.data.SignalArray` requires a sampling rate and both typed
    containers carry ``t0``. Those declarations are recorded beside the payload
    instead of being re-derived, because an average of the timestamps is a
    different number from the rate the recording declared.
    """
    return {
        "fs": None if fs is None else float(fs),
        "t0": float(t0 or 0.0),
    }


# --- clocks ---------------------------------------------------------------


class _ClockRegistry:
    """Register one NRF clock per distinct :class:`~neurale.data.Clock`.

    Two typed objects that share a clock share a clock ID; two that do not stay
    separate, so a session recorded across several timing domains does not
    collapse into one. Objects that declare no clock at all reference a single
    placeholder marked as synthetic, which the reader restores as ``None``
    rather than as a clock the recording never had.
    """

    def __init__(self, registry: ManifestBuilder, default_id: str) -> None:
        self._registry = registry
        self._default_id = default_id
        self._by_key: dict[Any, str] = {}
        self._taken: set[str] = set()

    def resolve(self, clock: Clock | None) -> str:
        """Return the clock ID for *clock*, registering it on first sight."""
        key = self._key(clock)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        preferred = self._default_id if clock is None else _identifier(clock.name, fallback="clock")
        clock_id = self._reserve(preferred)
        self._register(clock_id, clock)
        self._by_key[key] = clock_id
        return clock_id

    def default(self) -> str:
        """Return the placeholder clock used by anything that declares none."""
        return self.resolve(None)

    @staticmethod
    def _key(clock: Clock | None) -> Any:
        if clock is None:
            return None
        return (
            clock.name,
            clock.type,
            clock.rate,
            clock.epoch,
            clock.offset,
            clock.drift,
            clock.synchronization_domain,
        )

    def _reserve(self, preferred: str) -> str:
        clock_id = _unique(preferred, self._taken)
        self._taken.add(clock_id)
        return clock_id

    def _register(self, clock_id: str, clock: Clock | None) -> None:
        if clock is None:
            self._registry.register_clock(
                clock_id,
                clock_type="host_monotonic",
                name=clock_id,
                epoch=None,
                metadata={SYNTHETIC_CLOCK_KEY: True},
            )
            return

        metadata: dict[str, Any] = {}
        clock_type = _nrf_clock_type(clock.type)
        if clock_type != clock.type:
            metadata[CLOCK_TYPE_KEY] = clock.type

        domain: str | None = None
        if clock.synchronization_domain is not None:
            # A synchronization domain is an NRF identifier, not free text.
            domain = _identifier(clock.synchronization_domain, fallback="domain")
            if domain != clock.synchronization_domain:
                metadata[CLOCK_DOMAIN_KEY] = clock.synchronization_domain

        self._registry.register_clock(
            clock_id,
            clock_type=clock_type,
            name=clock.name,
            rate=None
            if clock.rate is None
            else _rational_rate(clock.rate, f"clock {clock.name!r}"),
            epoch=clock.epoch,
            synchronization_domain=domain,
            offset_ns=_seconds_to_ns(clock.offset),
            drift_ppb=round(float(clock.drift) * 1e9),
            metadata=metadata or None,
        )


# --- record registries ----------------------------------------------------


def _mandatory_record_fields() -> dict[str, tuple[str, list[tuple[str, str, bool]]]]:
    """Return the record kinds every NRF v1 session must declare."""
    return {
        "events": (
            "event_id",
            [
                ("event_id", "utf8", False),
                ("time_ns", "int64", False),
                ("duration_ns", "int64", False),
                ("label", "utf8", True),
                ("code", "int64", True),
                ("source", "utf8", True),
                ("sample_index", "int64", True),
            ],
        ),
        "trials": (
            "trial_id",
            [
                ("trial_id", "utf8", False),
                ("start_ns", "int64", False),
                ("stop_ns", "int64", False),
                ("label", "utf8", True),
                ("target_id", "int64", True),
                ("outcome", "utf8", True),
                ("block", "int64", True),
            ],
        ),
        "experiment_state": (
            "state_transition_id",
            [
                ("state_transition_id", "utf8", False),
                ("time_ns", "int64", False),
                ("new_state", "utf8", False),
            ],
        ),
        "commands": (
            "command_id",
            [
                ("command_id", "utf8", False),
                ("time_ns", "int64", False),
                ("command_type", "utf8", False),
            ],
        ),
        "faults": (
            "fault_id",
            [("fault_id", "utf8", False), ("time_ns", "int64", False), ("code", "utf8", False)],
        ),
    }


def _register_records(
    registry: ManifestBuilder, default_clock_id: str, clock_ids: Mapping[str, str]
) -> None:
    """Register the mandatory record kinds, each on its own clock.

    ``clock_ids`` maps a record kind to the clock the typed object it came from
    declared; every other kind falls back to the session default.
    """
    for kind, (primary_key, fields) in _mandatory_record_fields().items():
        schema_id = f"{kind.replace('_', '-')}-v1"
        path = record_set_path(kind, schema_id)
        registry.register_record_schema(
            schema_id,
            kind=kind,
            primary_key=primary_key,
            clock_id=clock_ids.get(kind, default_clock_id),
            chunk_length=_RECORD_CHUNK_LENGTH,
            fields=[
                record_field(path, name, dtype, nullable=nullable)
                for name, dtype, nullable in fields
            ],
        )
    termination = record_set_path("session_termination", "termination-v1")
    registry.register_record_schema(
        "termination-v1",
        kind="session_termination",
        primary_key="termination_id",
        clock_id=default_clock_id,
        chunk_length=1,
        fields=[
            record_field(termination, "termination_id", "utf8"),
            record_field(termination, "time_ns", "int64"),
            record_field(termination, "termination_kind", "utf8"),
            record_field(termination, "reason", "utf8"),
            record_field(termination, "fault_id", "utf8", nullable=True, reference="fault"),
            record_field(termination, "last_transaction_id", "utf8", reference="transaction"),
        ],
    )


def _register_units(registry: ManifestBuilder, symbols: Sequence[str]) -> dict[str, str]:
    """Register one unit per distinct symbol and return symbol -> unit ID."""
    mapping: dict[str, str] = {}
    for symbol in symbols:
        if symbol in mapping:
            continue
        unit_id = _unique(_identifier(symbol or "dimensionless", fallback="unit"), mapping.values())
        registry.register_unit(unit_id, symbol=symbol or "1")
        mapping[symbol] = unit_id
    return mapping


def _unit_symbols(container: SignalArray | FeatureMatrix, label: str, column: str) -> list[str]:
    """Return one unit symbol per column of a typed container.

    A single symbol applies to every column; a sequence must name each one, so
    a mismatched length is an error rather than something to pad or truncate.
    """
    unit = container.unit
    width = int(container.data.shape[1])
    if isinstance(unit, str):
        return [unit] * width
    symbols = list(unit)
    if len(symbols) != width:
        raise NrfSemanticError(f"{label} declares {len(symbols)} units for {width} {column}")
    return symbols


def _signal_units(signal: SignalArray) -> list[str]:
    return _unit_symbols(signal, f"signal {signal.name!r}", "channels")


def _feature_units(features: FeatureMatrix) -> list[str]:
    return _unit_symbols(features, "feature matrix", "features")


def _session_metadata(recording: Recording) -> dict[str, Any]:
    """Return the NRF session members a typed recording can fill.

    The manifest ``session`` object is a closed vocabulary whose ``id`` is a
    UUID, so an arbitrary ``Recording.session`` dict cannot be merged into it.
    Its contents go to ``session.metadata``, which is exactly the free-form
    member the format provides, and the subject to ``session.subject`` so an
    NRF reader that knows nothing about :mod:`neurale.data` still finds it.
    """
    session = dict(recording.session or {})
    subject = recording.subject if recording.subject is not None else session.get("subject")
    descriptor: dict[str, Any] = {}
    if subject is not None:
        descriptor["subject"] = _json_safe(dict(subject), "Recording.subject")
    extra = {key: value for key, value in session.items() if key != "subject"}
    if extra:
        descriptor["metadata"] = _json_safe(extra, "Recording.session")
    return descriptor


def _recording_payload(recording: Recording, timing_hints: Mapping[str, Any]) -> dict[str, Any]:
    """Return everything a :class:`Recording` carries that no registry holds.

    Reader-generated ``nrf_*`` keys are dropped on the way in so that writing a
    recording that was just read back does not accumulate bookkeeping.
    """
    metadata = {
        key: value
        for key, value in dict(recording.metadata).items()
        if not key.startswith(RESERVED_METADATA_PREFIX)
    }
    payload: dict[str, Any] = {
        "metadata": _json_safe(metadata, "Recording.metadata"),
        "streams": dict(timing_hints),
    }
    if recording.subject is not None:
        payload["subject"] = _json_safe(dict(recording.subject), "Recording.subject")
    if recording.session is not None:
        payload["session"] = _json_safe(dict(recording.session), "Recording.session")
    return payload


def write_recording(
    path: str | Path,
    recording: Recording,
    *,
    session_id: str,
    created_at: str,
    writer_name: str = "pyneurale",
    writer_version: str = "1",
    clock_id: str = "recording-clock",
) -> Path:
    """Write *recording* to a new NRF v1 session and finalize it.

    Every signal becomes a stream, every feature matrix becomes a feature
    stream with its descriptor, and events and trials become their record sets.
    Record kinds the recording does not carry are still registered, because a
    normal termination must seal every declared target.

    Timing is preserved rather than normalized. A stream whose timestamps are
    the exact grid of its declared rate is written with regular timing; every
    other stream -- an irregularly sampled signal, a ``FeatureMatrix`` with
    ``fs=None`` -- is written with explicit per-item timestamps.

    Parameters
    ----------
    clock_id
        ID for the placeholder clock referenced by objects that declare none.
        Clocks the recording does declare are registered under their own names.

    Returns
    -------
    pathlib.Path
        The finalized single-file session package.

    Raises
    ------
    NrfSemanticError
        If the recording carries a dtype, unit layout, timing declaration, or
        metadata value NRF v1 cannot express. Nothing is coerced silently.
    """
    root = Path(path)
    writer = NrfWriter.create(
        root,
        session_id=session_id,
        created_at=created_at,
        writer_name=writer_name,
        writer_version=writer_version,
        session_metadata=_session_metadata(recording),
        metadata={"nrf_source": "neurale.data.Recording"},
    )
    registry = writer.registry
    clocks = _ClockRegistry(registry, clock_id)

    stream_ids = [_identifier(name, fallback="stream") for name in recording.signals]
    feature_ids = [_identifier(name, fallback="features") for name in recording.features]

    # Clocks first: every later registration references one, and resolving them
    # up front is what keeps two signals on one timing source from becoming two
    # registry entries (or two sources from collapsing into one).
    signal_clocks = {
        stream_id: clocks.resolve(signal.clock)
        for stream_id, signal in zip(stream_ids, recording.signals.values(), strict=True)
    }
    # A FeatureMatrix declares no clock of its own. It inherits the clock of the
    # signal it was derived from when that signal is in this recording, and the
    # placeholder otherwise -- never a guessed one.
    feature_clocks = {
        feature_id: signal_clocks.get(_source_stream_id(features.source_signal), clocks.default())
        for feature_id, features in zip(feature_ids, recording.features.values(), strict=True)
    }
    record_clocks: dict[str, str] = {}
    if recording.events is not None and recording.events.clock is not None:
        record_clocks["events"] = clocks.resolve(recording.events.clock)
    default_clock_id = clocks.default()

    unit_symbols: list[str] = []
    for signal in recording.signals.values():
        unit_symbols.extend(_signal_units(signal))
    for features in recording.features.values():
        unit_symbols.extend(_feature_units(features))
    units = _register_units(registry, unit_symbols or [""])

    # Channels are shared by name across signals: the same electrode contact
    # must not become two registry entries.
    channel_ids: dict[str, str] = {}
    stream_channels: dict[str, list[str]] = {}
    for name, signal in recording.signals.items():
        symbols = _signal_units(signal)
        ids: list[str] = []
        for i, channel in enumerate(signal.channels):
            key = channel.name
            if key not in channel_ids:
                channel_id = _unique(
                    _identifier(key, fallback="channel"), set(channel_ids.values())
                )
                registry.register_channel(
                    channel_id,
                    idx=len(channel_ids),
                    unit_id=units[symbols[i]],
                    name=channel.name,
                    channel_type=channel.type,
                    contact=channel.contact,
                    valid=channel.valid,
                    bad=channel.bad,
                )
                channel_ids[key] = channel_id
            ids.append(channel_ids[key])
        stream_channels[name] = ids

    registry.register_schema("recording-schema", stream_ids=[*stream_ids, *feature_ids])
    _register_records(registry, default_clock_id, record_clocks)

    stream_clocks = {**signal_clocks, **feature_clocks}
    for stream_id in [*stream_ids, *feature_ids]:
        path_ = stream_discontinuities_path(stream_id)
        registry.register_record_schema(
            f"discontinuities-{stream_id}-v1",
            kind="discontinuities",
            stream_id=stream_id,
            primary_key="discontinuity_id",
            clock_id=stream_clocks[stream_id],
            chunk_length=1,
            fields=[
                record_field(path_, "discontinuity_id", "utf8"),
                record_field(path_, "time_ns", "int64"),
                record_field(path_, "reason", "utf8"),
            ],
        )

    plans: dict[str, _Regular | _Explicit] = {}
    timing_hints: dict[str, dict[str, Any]] = {}

    for stream_id, (name, signal) in zip(stream_ids, recording.signals.items(), strict=True):
        symbols = _signal_units(signal)
        rows = int(signal.data.shape[0])
        plan = _timing_plan(
            time=signal.time,
            fs=signal.fs,
            t0=signal.t0,
            rows=rows,
            label=f"signal {name!r}",
        )
        plans[stream_id] = plan
        if isinstance(plan, _Explicit):
            timing_hints[stream_id] = _timing_hint(signal.fs, signal.t0)
        registry.register_stream(
            stream_id,
            kind="neural",
            dtype=_nrf_dtype(signal.data, f"signal {name!r}"),
            channel_ids=stream_channels[name],
            unit_ids=[units[symbol] for symbol in symbols],
            clock_id=signal_clocks[stream_id],
            schema_id="recording-schema",
            chunk_length=min(_DEFAULT_CHUNK_LENGTH, max(rows, 1)),
            capacity=max(rows, 1),
            discontinuity_record_schema_id=f"discontinuities-{stream_id}-v1",
            name=name,
            **_timing_arguments(plan),
        )

    for feature_id, (name, features) in zip(feature_ids, recording.features.items(), strict=True):
        symbols = _feature_units(features)
        rows = int(features.data.shape[0])
        feature_set_id = f"{feature_id}-set"
        # A feature set must name a source stream. When the matrix declares none,
        # the first signal in the recording stands in, and a recording with no
        # signals at all can only point the feature stream at itself.
        source_stream_id = _source_stream_id(features.source_signal)
        if source_stream_id is None:
            source_stream_id = stream_ids[0] if stream_ids else feature_id
        plan = _timing_plan(
            time=features.time,
            fs=features.fs,
            t0=features.t0,
            rows=rows,
            label=f"feature matrix {name!r}",
        )
        plans[feature_id] = plan
        if isinstance(plan, _Explicit):
            timing_hints[feature_id] = _timing_hint(features.fs, features.t0)
        algorithm = features.attrs.get("nrf_algorithm", {})
        registry.register_feature_set(
            feature_set_id,
            feature_names=list(features.feature_names)
            or [f"{name}:{idx}" for idx in range(features.data.shape[1])],
            unit_ids=[units[symbol] for symbol in symbols],
            source_stream_id=source_stream_id,
            algorithm_name=str(algorithm.get("name", "unspecified")),
            algorithm_version=str(algorithm.get("version", "unspecified")),
            window_length_ns=max(_seconds_to_ns(features.window_size), 1),
            shift_ns=max(_seconds_to_ns(features.shift), 1),
        )
        registry.register_stream(
            feature_id,
            kind="feature",
            dtype=_nrf_dtype(features.data, f"feature matrix {name!r}"),
            channel_ids=[],
            unit_ids=[units[symbol] for symbol in symbols],
            clock_id=feature_clocks[feature_id],
            schema_id="recording-schema",
            chunk_length=min(_DEFAULT_CHUNK_LENGTH, max(rows, 1)),
            capacity=max(rows, 1),
            discontinuity_record_schema_id=f"discontinuities-{feature_id}-v1",
            source_stream_ids=[source_stream_id],
            feature_set_id=feature_set_id,
            name=name,
            **_timing_arguments(plan),
        )

    # The registries are still open, and the manifest is not built until
    # freeze(), so the typed payload is attached once the timing hints exist.
    registry.metadata[RECORDING_METADATA_KEY] = _recording_payload(recording, timing_hints)

    writer.freeze()

    payloads = [*recording.signals.values(), *recording.features.values()]
    for target_id, container in zip([*stream_ids, *feature_ids], payloads, strict=True):
        writer.append_stream(
            target_id, np.asarray(container.data), **_append_arguments(plans[target_id])
        )

    if recording.events is not None and len(recording.events):
        writer.append_records("events-v1", _event_columns(recording.events))
    if recording.trials is not None and len(recording.trials):
        writer.append_records("trials-v1", _trial_columns(recording.trials))

    writer.finalize()
    writer.close()
    return root


def _timing_arguments(plan: _Regular | _Explicit) -> dict[str, Any]:
    """Return the ``register_stream`` timing arguments for *plan*."""
    if isinstance(plan, _Regular):
        return {
            "timing_mode": "regular",
            "rate": plan.rate,
            "segment_start_time_ns": plan.start_ns,
        }
    return {"timing_mode": "explicit"}


def _append_arguments(plan: _Regular | _Explicit) -> dict[str, Any]:
    """Return the ``append_stream`` timing arguments for *plan*."""
    if isinstance(plan, _Regular):
        return {}
    return {"timestamps": plan.timestamps}


def _event_columns(events: Any) -> dict[str, list[Any]]:
    """Return ``events`` record columns for a typed :class:`EventSeries`."""
    rows = list(events)
    return {
        "event_id": [
            str(event.attrs.get("nrf_event_id", f"event-{idx:08d}"))
            for idx, event in enumerate(rows)
        ],
        "time_ns": [_seconds_to_ns(event.onset) for event in rows],
        "duration_ns": [_seconds_to_ns(event.duration) for event in rows],
        "label": [event.label for event in rows],
        "code": [event.code for event in rows],
        "source": [event.source for event in rows],
        "sample_index": [event.sample_index for event in rows],
    }


def _trial_columns(trials: Any) -> dict[str, list[Any]]:
    """Return ``trials`` record columns for a typed :class:`TrialTable`."""
    rows = list(trials)
    return {
        "trial_id": [
            str(trial.attrs.get("nrf_trial_id", f"trial-{trial.trial_id:08d}")) for trial in rows
        ],
        "start_ns": [_seconds_to_ns(trial.start) for trial in rows],
        "stop_ns": [_seconds_to_ns(trial.stop) for trial in rows],
        "label": [trial.label for trial in rows],
        "target_id": [trial.target_id for trial in rows],
        "outcome": [trial.outcome for trial in rows],
        "block": [trial.block for trial in rows],
    }
