#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Manifest assembly and the frozen per-target contract.

Specification reference: ``README.md`` sections 4, 5, 6, and 8.2.

The manifest is written once, before the first committed transaction, and every
registry in it is immutable for the life of the session. That is what lets a
stale manifest cache still describe every committed extent: the descriptors a
reader needs were all present before any data existed.

Descriptor helpers here take the specification's own vocabulary rather than
inventing a parallel one, so there is a single spelling of a clock, a stream, or
a record field across the format, the writer, and the reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._errors import NrfSemanticError, NrfStateError
from ._fields import BYTES_LE_CODEC, UTF8_CODEC, codec_for, endianness_for
from ._paths import (
    record_set_path,
    stream_data_path,
    stream_discontinuities_path,
    stream_timestamps_path,
)
from ._schemas import MANIFEST_SCHEMA, validate_document
from ._semantics import validate_manifest
from ._zarr import ArraySpec, array_spec

DEFAULT_CODECS: tuple[dict[str, Any], ...] = (
    {"id": BYTES_LE_CODEC, "name": "bytes", "configuration": {"endian": "little"}},
    {"id": UTF8_CODEC, "name": "vlen-utf8", "configuration": {}},
)


def descriptor_by_id(
    entries: Sequence[Mapping[str, Any]], identifier: str, label: str
) -> Mapping[str, Any]:
    """Return the registry entry with *identifier*, or raise naming *label*."""
    for entry in entries:
        if entry["id"] == identifier:
            return entry
    raise NrfSemanticError(f"unknown {label} {identifier!r}")


def stream_data_spec(stream: Mapping[str, Any]) -> ArraySpec:
    """Return the physical data array one stream descriptor freezes."""
    return array_spec(stream["data"], dtype=stream["dtype"], endianness=stream["endianness"])


def timestamps_spec(timing: Mapping[str, Any]) -> ArraySpec:
    """Return the physical timestamp array an explicit timing block freezes.

    NRF stores timestamps as little-endian ``int64`` regardless of the stream's
    own byte order, so the endianness is fixed here rather than inherited.
    """
    return array_spec(timing["timestamps"], dtype=timing["timestamp_dtype"], endianness="little")


def _with_optional(descriptor: dict[str, Any], **members: Any) -> dict[str, Any]:
    """Return *descriptor* extended with every optional member that is present.

    The format distinguishes an omitted member from one carrying a default, so
    a ``None`` is dropped rather than written out.
    """
    descriptor.update({name: value for name, value in members.items() if value is not None})
    return descriptor


def zarr_array(
    path: str, shape: Sequence[int], chunk_shape: Sequence[int], dtype: str
) -> dict[str, Any]:
    """Return a manifest Zarr array descriptor."""
    return {
        "path": path,
        "zarr_format": 3,
        "shape": list(shape),
        "chunk_shape": list(chunk_shape),
        "codec_ids": [codec_for(dtype)],
    }


@dataclass(frozen=True, slots=True)
class TargetContract:
    """The frozen contract for one appendable target.

    A target is the unit an extent advances in. A record set is one target
    whose every column and validity array must advance in the same
    transaction.

    A stream with explicit timing owns two *separate* extent targets -- its data
    array and its timestamp array -- because the journal tracks them under
    their own paths. They must nevertheless always hold the same extent, since
    the format requires explicit timestamps to cover every committed item.
    ``linked_target_path`` records that pairing so a writer advances and seals
    them together instead of hoping two independent decisions coincide.
    """

    target_path: str
    target_kind: str
    chunk_length: int
    arrays: tuple[ArraySpec, ...]
    record_schema_id: str | None = None
    #: Declared leading-axis size. A Zarr shape is fixed at registration, so an
    #: extent can never exceed it.
    capacity: int = 0
    #: The timestamp target this data target advances with, if any.
    linked_target_path: str | None = None
    #: True for the timestamp target itself, which is never planned on its own.
    follows_linked_target: bool = False

    @property
    def required_array_paths(self) -> tuple[str, ...]:
        return tuple(spec.path for spec in self.arrays)

    def as_replay_contract(self) -> dict[str, Any]:
        """Return the contract shape journal replay compares against."""
        return {
            "required_array_paths": list(self.required_array_paths),
            "record_schema_id": self.record_schema_id,
            "chunk_length": self.chunk_length,
            "trailing_chunk_grid": {
                spec.path: list(spec.trailing_chunk_grid()) for spec in self.arrays
            },
        }


@dataclass
class ManifestBuilder:
    """Accumulate registries and emit a validated manifest.

    Every ``register_*`` call must happen before :meth:`freeze`; afterwards the
    registries are immutable, which is the session-wide freeze the format
    requires rather than a convenience restriction.
    """

    session_id: str
    created_at: str
    writer_name: str
    writer_version: str
    session_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Namespaced extension objects the session declares. A minor version is
    #: what makes them legible: NRF v1.0 defines no extension, so a v1.0 reader
    #: meeting one has no rule for it, and the writer that adds one has to say
    #: which minor it is writing.
    extensions: dict[str, Any] = field(default_factory=dict)
    #: The manifest's declared minor version. It is a writer decision rather
    #: than something derived from the registries: a session that declares an
    #: extension must raise it, and nothing here may raise it silently.
    minor_version: int = 0

    clocks: list[dict[str, Any]] = field(default_factory=list)
    units: list[dict[str, Any]] = field(default_factory=list)
    channels: list[dict[str, Any]] = field(default_factory=list)
    electrodes: list[dict[str, Any]] = field(default_factory=list)
    schemas: list[dict[str, Any]] = field(default_factory=list)
    streams: list[dict[str, Any]] = field(default_factory=list)
    feature_sets: list[dict[str, Any]] = field(default_factory=list)
    record_schemas: list[dict[str, Any]] = field(default_factory=list)
    frozen: bool = False

    # --- registration -----------------------------------------------------

    def _require_open(self) -> None:
        if self.frozen:
            raise NrfStateError(
                "the session registries are frozen; NRF v1 forbids registering a clock, "
                "unit, channel, electrode, schema, stream, feature set, codec, or record "
                "schema after the first commit"
            )

    def register_clock(
        self,
        clock_id: str,
        *,
        clock_type: str,
        name: str | None = None,
        rate: Mapping[str, int] | None = None,
        epoch: str | None = "session start",
        synchronization_domain: str | None = None,
        synchronization_source_clock_id: str | None = None,
        offset_ns: int = 0,
        drift_ppb: int = 0,
        native_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Register one clock.

        ``epoch`` is optional in the format. Passing ``None`` omits it, which is
        the honest record for a clock whose epoch was never stated; it is not
        the same as claiming an epoch of ``"session start"``.
        """
        self._require_open()
        self.clocks.append(
            _with_optional(
                {
                    "id": clock_id,
                    "type": clock_type,
                    "name": name or clock_id,
                    "offset_ns": offset_ns,
                    "drift_ppb": drift_ppb,
                },
                epoch=epoch,
                rate=None if rate is None else dict(rate),
                metadata=dict(metadata) if metadata else None,
                synchronization_domain=synchronization_domain,
                synchronization_source_clock_id=synchronization_source_clock_id,
                native_id=native_id,
            )
        )

    def register_unit(
        self, unit_id: str, *, symbol: str, description: str = "", native_id: int | None = None
    ) -> None:
        self._require_open()
        self.units.append(
            _with_optional(
                {"id": unit_id, "symbol": symbol},
                description=description or None,
                native_id=native_id,
            )
        )

    def register_channel(
        self,
        channel_id: str,
        *,
        idx: int,
        unit_id: str,
        name: str | None = None,
        channel_type: str = "analog",
        electrode_id: str | None = None,
        contact: int | None = None,
        valid: bool = True,
        bad: bool = False,
        native_id: int | None = None,
    ) -> None:
        self._require_open()
        self.channels.append(
            _with_optional(
                {
                    "id": channel_id,
                    "index": idx,
                    "name": name or channel_id,
                    "type": channel_type,
                    "unit_id": unit_id,
                    "valid": valid,
                    "bad": bad,
                },
                electrode_id=electrode_id,
                contact=contact,
                native_id=native_id,
            )
        )

    def register_electrode(
        self,
        electrode_id: str,
        *,
        channel_ids: Sequence[str],
        name: str | None = None,
        electrode_type: str = "electrode_array",
        shape: Sequence[int] | None = None,
        contact_ids: Sequence[int] | None = None,
    ) -> None:
        self._require_open()
        # One contact per channel unless the caller maps them explicitly; the
        # schema requires the pairing to be stated, not inferred later.
        contacts = range(len(channel_ids)) if contact_ids is None else contact_ids
        self.electrodes.append(
            _with_optional(
                {
                    "id": electrode_id,
                    "name": name or electrode_id,
                    "type": electrode_type,
                    "channel_ids": list(channel_ids),
                    "contact_ids": list(contacts),
                },
                shape=None if shape is None else list(shape),
            )
        )

    def register_schema(
        self, schema_id: str, *, stream_ids: Sequence[str], native_id: int | None = None
    ) -> None:
        self._require_open()
        self.schemas.append(
            _with_optional({"id": schema_id, "stream_ids": list(stream_ids)}, native_id=native_id)
        )

    def register_stream(
        self,
        stream_id: str,
        *,
        kind: str,
        dtype: str,
        channel_ids: Sequence[str],
        unit_ids: Sequence[str],
        clock_id: str,
        schema_id: str,
        chunk_length: int,
        capacity: int,
        discontinuity_record_schema_id: str,
        timing_mode: str = "regular",
        rate: Mapping[str, int] | None = None,
        segment_start_idx: int = 0,
        segment_start_time_ns: int = 0,
        timestamp_reference: str | None = None,
        initial_segment_id: str = "segment-0001",
        source_stream_ids: Sequence[str] = (),
        feature_set_id: str | None = None,
        name: str | None = None,
        native_id: int | None = None,
        timestamp_chunk_length: int | None = None,
    ) -> None:
        """Register one stream and its frozen physical arrays.

        ``capacity`` is the declared leading-axis size of the data array. A Zarr
        shape is fixed at registration, so a session must declare how many
        samples or observations it can hold.
        """
        self._require_open()
        width = len(unit_ids)
        # A feature observation is timestamped at its window centre, not at a
        # sample; the format fixes this per stream kind rather than leaving it
        # to the caller.
        if timestamp_reference is None:
            timestamp_reference = "window_center" if kind == "feature" else "sample"
        axes = {
            "neural": ["sample", "channel"],
            "behavioral": ["sample", "channel"],
            "feature": ["observation", "feature"],
            "spike": ["spike", "sample", "channel"],
        }[kind]

        data_path = stream_data_path(stream_id)
        descriptor = _with_optional(
            {
                "id": stream_id,
                "name": name or stream_id,
                "kind": kind,
                "dtype": dtype,
                "endianness": endianness_for(dtype),
                "axes": axes,
                "schema_id": schema_id,
                "clock_id": clock_id,
                "channel_ids": list(channel_ids),
                "unit_ids": list(unit_ids),
                "source_stream_ids": list(source_stream_ids),
                "committed_extent": 0,
                "data": zarr_array(data_path, (capacity, width), (chunk_length, width), dtype),
                "segment_policy": {
                    "mode": "explicit",
                    "initial_segment_id": initial_segment_id,
                    "discontinuity_record_schema_id": discontinuity_record_schema_id,
                },
            },
            native_id=native_id,
            feature_set_id=feature_set_id,
        )

        if timing_mode == "regular":
            descriptor["timing"] = {
                "mode": "regular",
                "rate": dict(rate or {"numerator": 1, "denominator": 1}),
                "segment_start_index": segment_start_idx,
                "segment_start_time_ns": segment_start_time_ns,
                "timestamp_reference": timestamp_reference,
            }
        else:
            stamp_chunk = timestamp_chunk_length or chunk_length
            descriptor["timing"] = {
                "mode": "explicit",
                "timestamp_dtype": "int64",
                "timestamp_unit": "ns",
                "timestamp_reference": timestamp_reference,
                "timestamps": zarr_array(
                    stream_timestamps_path(stream_id), (capacity,), (stamp_chunk,), "int64"
                ),
            }
        self.streams.append(descriptor)

    def register_feature_set(
        self,
        feature_set_id: str,
        *,
        feature_names: Sequence[str],
        unit_ids: Sequence[str],
        source_stream_id: str,
        algorithm_name: str,
        algorithm_version: str,
        window_length_ns: int,
        shift_ns: int,
        timestamp_reference: str = "window_center",
        parameters: Mapping[str, Any] | None = None,
        native_id: int | None = None,
    ) -> None:
        self._require_open()
        self.feature_sets.append(
            _with_optional(
                {
                    "id": feature_set_id,
                    "feature_names": list(feature_names),
                    "unit_ids": list(unit_ids),
                    "source_stream_id": source_stream_id,
                    "algorithm": {
                        "name": algorithm_name,
                        "version": algorithm_version,
                        "parameters": dict(parameters or {}),
                    },
                    "window_length_ns": window_length_ns,
                    "shift_ns": shift_ns,
                    "timestamp_reference": timestamp_reference,
                },
                native_id=native_id,
            )
        )

    def register_record_schema(
        self,
        schema_id: str,
        *,
        kind: str,
        fields: Sequence[Mapping[str, Any]],
        primary_key: str,
        clock_id: str,
        chunk_length: int,
        stream_id: str | None = None,
    ) -> str:
        """Register one record set and return its frozen path.

        ``discontinuities`` schemas live under their owning stream; every other
        kind lives under ``records/<kind>/<id>``.
        """
        self._require_open()
        if kind == "discontinuities":
            if stream_id is None:
                raise NrfStateError("a discontinuities record schema must name its stream")
            path = stream_discontinuities_path(stream_id)
        else:
            path = record_set_path(kind, schema_id)
        self.record_schemas.append(
            {
                "id": schema_id,
                "kind": kind,
                "path": path,
                "primary_key": primary_key,
                "clock_id": clock_id,
                "chunk_length": chunk_length,
                "committed_extent": 0,
                "fields": [dict(item) for item in fields],
            }
        )
        return path

    # --- assembly ---------------------------------------------------------

    def build(
        self,
        *,
        committed_extents: Mapping[str, int] | None = None,
        sealed_targets: Sequence[str] = (),
        last_transaction_id: str | None = None,
        last_checkpoint_id: str | None = None,
        journal_sequence: int = 0,
    ) -> dict[str, Any]:
        """Return the manifest document for the current committed state."""
        extents = dict(committed_extents or {})
        streams = [
            dict(stream, committed_extent=extents.get(stream["data"]["path"], 0))
            for stream in self.streams
        ]
        record_schemas = [
            dict(schema, committed_extent=extents.get(schema["path"], 0))
            for schema in self.record_schemas
        ]
        session = {"id": self.session_id, "created_at": self.created_at}
        session.update(self.session_metadata)
        manifest: dict[str, Any] = {
            "format": "nrf",
            "version": {"major": 1, "minor": self.minor_version},
            "session": session,
            "writer": {"name": self.writer_name, "version": self.writer_version},
            "checksum": "sha256",
            "canonical_json": "rfc8785",
            "default_endianness": "little",
            "codecs": [dict(codec) for codec in DEFAULT_CODECS],
            "clocks": self.clocks,
            "units": self.units,
            "channels": self.channels,
            "electrodes": self.electrodes,
            "schemas": self.schemas,
            "streams": streams,
            "feature_sets": self.feature_sets,
            "record_schemas": record_schemas,
            "commit": {
                "journal_sequence": journal_sequence,
                "last_transaction_id": last_transaction_id,
                "last_checkpoint_id": last_checkpoint_id,
                "committed_extents": extents,
                "sealed_targets": sorted(sealed_targets),
            },
        }
        if self.metadata:
            manifest["metadata"] = self.metadata
        if self.extensions:
            manifest["extensions"] = self.extensions
        return manifest

    def validate(self, manifest: Mapping[str, Any]) -> None:
        """Run both mandatory validation layers, in the required order."""
        validate_document(manifest, MANIFEST_SCHEMA)
        validate_manifest(manifest)


def target_contracts(manifest: Mapping[str, Any]) -> dict[str, TargetContract]:
    """Derive the frozen per-target contract from a manifest.

    This is the shape journal replay compares each prepare record against, so
    it is computed from the manifest rather than tracked separately by the
    writer: a writer bug cannot make its own transactions look legal.
    """
    contracts: dict[str, TargetContract] = {}

    for stream in manifest["streams"]:
        data = stream["data"]
        spec = stream_data_spec(stream)
        timing = stream["timing"]
        stamps_path: str | None = None
        if timing["mode"] == "explicit":
            stamps = timing["timestamps"]
            stamp_spec = timestamps_spec(timing)
            if stamp_spec.chunk_length != spec.chunk_length:
                raise NrfSemanticError(
                    f"{stream['id']} timestamp chunk length {stamp_spec.chunk_length} differs "
                    f"from its data chunk length {spec.chunk_length}; explicit timestamps must "
                    "advance with their data"
                )
            stamps_path = stamps["path"]
            contracts[stamps_path] = TargetContract(
                target_path=stamps_path,
                target_kind="array",
                chunk_length=stamp_spec.chunk_length,
                arrays=(stamp_spec,),
                capacity=stamp_spec.shape[0],
                follows_linked_target=True,
            )

        contracts[data["path"]] = TargetContract(
            target_path=data["path"],
            target_kind="array",
            chunk_length=spec.chunk_length,
            arrays=(spec,),
            capacity=spec.shape[0],
            linked_target_path=stamps_path,
        )

    for schema in manifest["record_schemas"]:
        capacity = _record_capacity(schema)
        specs: list[ArraySpec] = []
        for item in schema["fields"]:
            trailing = tuple(item.get("shape") or ())
            specs.append(
                ArraySpec(
                    path=item["array_path"],
                    shape=(capacity, *trailing),
                    chunk_shape=(schema["chunk_length"], *trailing),
                    dtype=item["dtype"],
                    endianness=item["endianness"],
                    codec_ids=tuple(item["codec_ids"]),
                )
            )
            if item["nullable"]:
                specs.append(
                    ArraySpec(
                        path=item["validity_path"],
                        shape=(capacity,),
                        chunk_shape=(schema["chunk_length"],),
                        dtype="uint8",
                        endianness="not_applicable",
                        codec_ids=(BYTES_LE_CODEC,),
                    )
                )
        contracts[schema["path"]] = TargetContract(
            target_path=schema["path"],
            target_kind="record_set",
            chunk_length=schema["chunk_length"],
            arrays=tuple(specs),
            record_schema_id=schema["id"],
            capacity=capacity,
        )
    return contracts


def replay_contracts(contracts: Mapping[str, TargetContract]) -> dict[str, dict[str, Any]]:
    """Return the per-target contracts in the shape journal replay compares against."""
    return {path: contract.as_replay_contract() for path, contract in contracts.items()}


def _record_capacity(schema: Mapping[str, Any]) -> int:
    """Return the declared leading-axis capacity of a record set."""
    return int(schema.get("capacity", schema["chunk_length"] * 1024))
