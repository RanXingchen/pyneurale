#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Generate the normative, language-neutral NRF v1 test vectors.

Run it from anywhere; it writes ``test-vectors.json`` next to the specification
it belongs to. The sibling ``jcs.py`` is loaded by path rather than imported by
name: this directory is deliberately not a Python package, so there is no import
root to depend on and no chance of picking up an unrelated installed module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
SPEC_DIR = _HERE.parent
OUTPUT = SPEC_DIR / "test-vectors.json"


def _load_sibling(name: str) -> Any:
    """Import a module that sits next to this file, by path."""
    spec = importlib.util.spec_from_file_location(f"nrf_v1_{name}", _HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {name}.py next to {__file__}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_jcs = _load_sibling("jcs")
canonical_json = _jcs.canonical_json
canonical_json_bytes = _jcs.canonical_json_bytes


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jcs_vectors() -> list[dict[str, Any]]:
    """Return fixed RFC 8785 coverage vectors with independently stated outputs."""

    cases: list[tuple[str, Any, str]] = [
        (
            "ascii_integer_baseline",
            {"kind": "example", "sequence": 1, "text": "NRF v1"},
            '{"kind":"example","sequence":1,"text":"NRF v1"}',
        ),
        (
            "rfc8785_primitives",
            {
                "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27],
                "string": '€$\u000f\nA\'B"\\\\"/',
                "literals": [None, True, False],
            },
            '{"literals":[null,true,false],"numbers":[333333333.3333333,'
            '1e+30,4.5,0.002,1e-27],"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}',
        ),
        (
            "utf16_property_order",
            {
                "€": "Euro Sign",
                "\r": "Carriage Return",
                "דּ": "Hebrew Letter Dalet With Dagesh",
                "1": "One",
                "😀": "Emoji: Grinning Face",
                "\u0080": "Control",
                "ö": "Latin Small Letter O With Diaeresis",
            },
            '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
            '"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
            '"😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"}',
        ),
        (
            "numeric_boundaries",
            [
                -0.0,
                5e-324,
                -5e-324,
                1e30,
                1e-7,
                1e-6,
                9007199254740991,
                -9007199254740991,
            ],
            "[0,5e-324,-5e-324,1e+30,1e-7,0.000001,9007199254740991,-9007199254740991]",
        ),
        (
            "nested_property_order",
            {
                "z": {
                    "😀": 1,
                    "€": 2,
                    "a": 'line\nquote"slash\\',
                },
                "a": [{"b": 2, "a": 1}],
            },
            '{"a":[{"a":1,"b":2}],"z":{"a":"line\\nquote\\"slash\\\\","€":2,"😀":1}}',
        ),
    ]

    vectors = []
    for name, value, expected in cases:
        actual = canonical_json(value)
        if actual != expected:
            raise AssertionError(f"RFC 8785 implementation disagrees with fixed {name} vector")
        vectors.append(
            {
                "name": name,
                "input": value,
                "canonical_utf8": expected,
                "sha256": sha256_bytes(expected.encode("utf-8")),
            }
        )
    return vectors


def with_record_checksum(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["record_checksum"] = sha256_bytes(canonical_json_bytes(value))
    return value


def object_write(
    transaction_id: str,
    path: str,
    payload: bytes,
    kind: str,
    *,
    target_path: str,
    array_path: str,
    logical_role: str,
    chunk_coordinate: list[int] | None = None,
    record_schema_id: str | None = None,
    column_name: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path,
        "staged_path": f".staging/{transaction_id}/{path}",
        "byte_length": len(payload),
        "sha256": sha256_bytes(payload),
        "disposition": "create",
        "content_kind": kind,
        "target_path": target_path,
        "array_path": array_path,
        "logical_role": logical_role,
    }
    if chunk_coordinate is not None:
        value["chunk_coordinate"] = chunk_coordinate
    if record_schema_id is not None:
        value["record_schema_id"] = record_schema_id
    if column_name is not None:
        value["column_name"] = column_name
    return value


def field(
    name: str,
    dtype: str,
    *,
    nullable: bool = False,
    unit_id: str | None = None,
    reference: str | None = None,
    shape: list[int] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "dtype": dtype, "nullable": nullable}
    if unit_id is not None:
        value["unit_id"] = unit_id
    if reference is not None:
        value["reference"] = reference
    if shape is not None:
        value["shape"] = shape
    return value


def record_schema(
    schema_id: str,
    kind: str,
    fields: list[dict[str, Any]],
    *,
    extent: int = 0,
    clock_id: str | None = "host-clock",
    path: str | None = None,
) -> dict[str, Any]:
    if path is None:
        path = f"records/{kind}/{schema_id}"
    physical_fields: list[dict[str, Any]] = []
    for item in fields:
        physical = dict(item)
        physical["array_path"] = f"{path}/columns/{item['name']}"
        physical["endianness"] = (
            "not_applicable" if item["dtype"] in {"bool", "uint8", "utf8"} else "little"
        )
        physical["codec_ids"] = ["utf8-vlen" if item["dtype"] == "utf8" else "bytes-le"]
        if item["nullable"]:
            physical["validity_path"] = f"{path}/validity/{item['name']}"
        physical_fields.append(physical)
    value: dict[str, Any] = {
        "id": schema_id,
        "kind": kind,
        "path": path,
        "primary_key": fields[0]["name"],
        "fields": physical_fields,
        "chunk_length": 1,
        "committed_extent": extent,
    }
    if clock_id is not None:
        value["clock_id"] = clock_id
    return value


def discontinuity_fields() -> list[dict[str, Any]]:
    """Fields for a per-stream discontinuity record set (README section 5).

    A discontinuity record carries the owning stream, previous/next segment IDs,
    expected and actual sample indices, optional missing sample/device-tick
    counts, the reason, and the clock time. The record-set path is per-stream
    (``streams/<stream-id>/discontinuities``); each stream's segment_policy
    references its discontinuity record schema by ID.
    """
    return [
        field("discontinuity_id", "utf8"),
        field("stream_id", "utf8", reference="stream"),
        field("previous_segment_id", "utf8", nullable=True, reference="segment"),
        field("next_segment_id", "utf8", nullable=True, reference="segment"),
        field("expected_sample_index", "int64"),
        field("actual_sample_index", "int64"),
        field("missing_samples", "int32", nullable=True),
        field("missing_device_ticks", "int64", nullable=True),
        field("reason", "utf8"),
        field("time_ns", "int64", unit_id="nanosecond"),
    ]


def discontinuity_schema(stream_id: str, *, extent: int = 0) -> dict[str, Any]:
    schema_id = f"discontinuities-{stream_id}-v1"
    return record_schema(
        schema_id,
        "discontinuities",
        discontinuity_fields(),
        extent=extent,
        path=f"streams/{stream_id}/discontinuities",
    )


def manifest(committed_extents: dict[str, int]) -> dict[str, Any]:
    stream_base = {
        "source_stream_ids": [],
        "segment_policy": {
            "mode": "explicit",
            "initial_segment_id": "segment-0001",
            "discontinuity_record_schema_id": "discontinuities-neural-v1",
        },
    }
    return {
        "format": "nrf",
        "version": {"major": 1, "minor": 0},
        "session": {
            "id": "018f4f30-6f9d-7b36-8c61-3f96cf5c5a40",
            "created_at": "2026-07-30T09:00:00Z",
            "subject": {"id": "subject-01"},
            "experiment": {"name": "nrf-v1-vector"},
        },
        "writer": {"name": "PyNeurale vector generator", "version": "1.0"},
        "checksum": "sha256",
        "canonical_json": "rfc8785",
        "default_endianness": "little",
        "codecs": [
            {"id": "bytes-le", "name": "bytes", "configuration": {"endian": "little"}},
            {"id": "utf8-vlen", "name": "vlen-utf8", "configuration": {}},
        ],
        "clocks": [
            {
                "id": "host-clock",
                "native_id": 1,
                "name": "host monotonic",
                "type": "host_monotonic",
                "rate": {"numerator": 1000000000, "denominator": 1},
                "epoch": "session start",
                "offset_ns": 0,
                "drift_ppb": 0,
                "synchronization_domain": "session-domain",
            },
            {
                "id": "device-clock",
                "native_id": 2,
                "name": "acquisition sample clock",
                "type": "device_counter",
                "rate": {"numerator": 4000, "denominator": 1},
                "epoch": "device start",
                "offset_ns": 0,
                "drift_ppb": 0,
                "synchronization_domain": "session-domain",
                "synchronization_source_clock_id": "host-clock",
            },
        ],
        "units": [
            {"id": "volt", "native_id": 1, "symbol": "V", "description": "volt"},
            {"id": "meter", "symbol": "m", "description": "metre"},
            {"id": "volt-squared", "symbol": "V^2", "description": "integrated power"},
            {"id": "dimensionless", "native_id": 3, "symbol": "1"},
            {"id": "nanosecond", "symbol": "ns", "description": "nanosecond"},
        ],
        "channels": [
            {
                "id": "channel-0000",
                "native_id": 0,
                "name": "E1",
                "index": 0,
                "type": "ecog",
                "unit_id": "volt",
                "valid": True,
                "bad": False,
                "electrode_id": "grid-a",
                "contact": 0,
            },
            {
                "id": "channel-0001",
                "native_id": 1,
                "name": "E2",
                "index": 1,
                "type": "ecog",
                "unit_id": "volt",
                "valid": True,
                "bad": False,
                "electrode_id": "grid-a",
                "contact": 1,
            },
        ],
        "electrodes": [
            {
                "id": "grid-a",
                "name": "grid A",
                "type": "ecog_grid",
                "contact_ids": [0, 1],
                "channel_ids": ["channel-0000", "channel-0001"],
                "shape": [1, 2],
            }
        ],
        "schemas": [
            {"id": "schema-source", "native_id": 1, "stream_ids": ["neural", "cursor"]},
            {"id": "schema-features", "native_id": 2, "stream_ids": ["bandpower"]},
        ],
        "streams": [
            {
                **stream_base,
                "id": "neural",
                "native_id": 1,
                "name": "neural",
                "kind": "neural",
                "schema_id": "schema-source",
                "dtype": "int16",
                "endianness": "little",
                "axes": ["sample", "channel"],
                "data": {
                    "path": "streams/neural/data",
                    "zarr_format": 3,
                    "shape": [4, 2],
                    "chunk_shape": [4, 2],
                    "codec_ids": ["bytes-le"],
                },
                "timing": {
                    "mode": "regular",
                    "rate": {"numerator": 4000, "denominator": 1},
                    "timestamp_reference": "sample",
                    "segment_start_time_ns": 0,
                    "segment_start_index": 0,
                },
                "clock_id": "device-clock",
                "channel_ids": ["channel-0000", "channel-0001"],
                "unit_ids": ["volt", "volt"],
                "committed_extent": 4,
            },
            {
                "id": "cursor",
                "native_id": 2,
                "name": "cursor position",
                "kind": "behavioral",
                "schema_id": "schema-source",
                "dtype": "float32",
                "endianness": "little",
                "axes": ["sample", "channel"],
                "data": {
                    "path": "streams/cursor/data",
                    "zarr_format": 3,
                    "shape": [2, 2],
                    "chunk_shape": [2, 2],
                    "codec_ids": ["bytes-le"],
                },
                "timing": {
                    "mode": "explicit",
                    "timestamps": {
                        "path": "streams/cursor/timestamps",
                        "zarr_format": 3,
                        "shape": [2],
                        "chunk_shape": [2],
                        "codec_ids": ["bytes-le"],
                    },
                    "timestamp_dtype": "int64",
                    "timestamp_unit": "ns",
                    "timestamp_reference": "sample",
                },
                "clock_id": "host-clock",
                "source_stream_ids": [],
                "channel_ids": [],
                "unit_ids": ["meter", "meter"],
                "segment_policy": {
                    "mode": "explicit",
                    "initial_segment_id": "segment-0001",
                    "discontinuity_record_schema_id": "discontinuities-cursor-v1",
                },
                "committed_extent": 2,
                "metadata": {"dimension_names": ["x", "y"]},
            },
            {
                "id": "bandpower",
                "native_id": 3,
                "name": "bandpower",
                "kind": "feature",
                "schema_id": "schema-features",
                "dtype": "float64",
                "endianness": "little",
                "axes": ["observation", "feature"],
                "data": {
                    "path": "streams/bandpower/data",
                    "zarr_format": 3,
                    "shape": [1, 2],
                    "chunk_shape": [1, 2],
                    "codec_ids": ["bytes-le"],
                },
                "timing": {
                    "mode": "regular",
                    "rate": {"numerator": 100, "denominator": 1},
                    "timestamp_reference": "window_center",
                    "segment_start_time_ns": 50000000,
                    "segment_start_index": 0,
                },
                "clock_id": "host-clock",
                "source_stream_ids": ["neural"],
                "channel_ids": [],
                "unit_ids": ["volt-squared", "volt-squared"],
                "feature_set_id": "bandpower-70-200",
                "segment_policy": {
                    "mode": "explicit",
                    "initial_segment_id": "segment-0001",
                    "discontinuity_record_schema_id": "discontinuities-bandpower-v1",
                },
                "committed_extent": 1,
            },
        ],
        "feature_sets": [
            {
                "id": "bandpower-70-200",
                "native_id": 1,
                "feature_names": ["70-200:E1", "70-200:E2"],
                "unit_ids": ["volt-squared", "volt-squared"],
                "source_stream_id": "neural",
                "algorithm": {
                    "name": "multitaper_bandpower",
                    "version": "1",
                    "parameters": {"band_semantics": "[low,high)", "fft_length": 512},
                },
                "window_length_ns": 100000000,
                "shift_ns": 10000000,
                "timestamp_reference": "window_center",
            }
        ],
        "record_schemas": [
            record_schema(
                "events-v1",
                "events",
                [
                    field("event_id", "utf8"),
                    field("onset_ns", "int64", unit_id="nanosecond"),
                    field("duration_ns", "int64", unit_id="nanosecond"),
                    field("label", "utf8", nullable=True),
                    field("stream_id", "utf8", nullable=True, reference="stream"),
                    field("segment_id", "utf8", nullable=True, reference="segment"),
                ],
            ),
            record_schema(
                "trials-v1",
                "trials",
                [
                    field("trial_id", "utf8"),
                    field("start_ns", "int64", unit_id="nanosecond"),
                    field("stop_ns", "int64", unit_id="nanosecond"),
                    field("label", "utf8", nullable=True),
                    field("target_id", "utf8", nullable=True),
                    field("block", "int64", nullable=True),
                ],
            ),
            record_schema(
                "experiment-state-v1",
                "experiment_state",
                [
                    field("state_transition_id", "utf8"),
                    field("time_ns", "int64", unit_id="nanosecond"),
                    field("previous_state", "utf8", nullable=True),
                    field("new_state", "utf8"),
                    field("trial_id", "utf8", nullable=True, reference="trial"),
                ],
                extent=1,
            ),
            record_schema(
                "commands-v1",
                "commands",
                [
                    field("command_id", "utf8"),
                    field("time_ns", "int64", unit_id="nanosecond"),
                    field("command_type", "utf8"),
                    field("value", "float64", shape=[2]),
                    field("trial_id", "utf8", nullable=True, reference="trial"),
                ],
                extent=1,
            ),
            record_schema(
                "task-variables-v1",
                "task_variables",
                [
                    field("observation_id", "utf8"),
                    field("time_ns", "int64"),
                    field("name", "utf8"),
                    field("value", "float64"),
                ],
            ),
            record_schema(
                "labels-v1",
                "labels",
                [field("label_id", "utf8"), field("time_ns", "int64"), field("value", "int64")],
            ),
            record_schema(
                "targets-v1",
                "targets",
                [
                    field("target_id", "utf8"),
                    field("time_ns", "int64"),
                    field("value", "float64", shape=[2]),
                ],
            ),
            record_schema(
                "assistance-v1",
                "assistance",
                [
                    field("assistance_id", "utf8"),
                    field("time_ns", "int64"),
                    field("value", "float64"),
                ],
            ),
            record_schema(
                "drops-v1",
                "drops",
                [
                    field("drop_id", "utf8"),
                    field("time_ns", "int64"),
                    field("first_sequence", "uint64"),
                    field("last_sequence", "uint64"),
                    field("policy", "utf8"),
                ],
            ),
            record_schema(
                "faults-v1",
                "faults",
                [
                    field("fault_id", "utf8"),
                    field("detected_at_ns", "int64"),
                    field("code", "utf8"),
                    field("status", "utf8"),
                    field("stage", "utf8"),
                    field("stream_id", "utf8", nullable=True, reference="stream"),
                    field("sample_index", "uint64", nullable=True),
                ],
                extent=1,
            ),
            record_schema(
                "termination-v1",
                "session_termination",
                [
                    field("termination_id", "utf8"),
                    field("time_ns", "int64"),
                    field("termination_kind", "utf8"),
                    field("reason", "utf8"),
                    field("last_transaction_id", "utf8", reference="transaction"),
                    field("fault_id", "utf8", nullable=True, reference="fault"),
                ],
                extent=committed_extents.get(
                    "records/session_termination/termination-v1",
                    0,
                ),
                clock_id="host-clock",
            ),
            discontinuity_schema("neural", extent=1),
            discontinuity_schema("cursor"),
            discontinuity_schema("bandpower"),
        ],
        "commit": {
            "journal_sequence": 14,
            "last_transaction_id": "tx-0000000000000006",
            "last_checkpoint_id": "checkpoint-0000000000000001",
            "committed_extents": committed_extents,
            "sealed_targets": sorted(committed_extents),
        },
        "metadata": {"interval_semantics": "[start,end)"},
    }


def prepare(
    sequence: int,
    number: int,
    previous: str | None,
    objects: list[dict[str, Any]],
    extents: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_extents = []
    for extent in extents:
        normalized = dict(extent)
        target_path = normalized.pop("array_path", normalized.get("target_path", None))
        if target_path is None:
            raise ValueError("extent transition requires target_path")
        normalized["target_path"] = target_path
        target_objects = [item for item in objects if item["target_path"] == target_path]
        record_objects = [
            item
            for item in target_objects
            if item["logical_role"] in {"record_column_chunk", "record_validity_chunk"}
        ]
        normalized.setdefault(
            "target_kind",
            "record_set" if record_objects else "array",
        )
        if record_objects:
            normalized.setdefault(
                "record_schema_id",
                record_objects[0]["record_schema_id"],
            )
        normalized.setdefault(
            "required_array_paths",
            list(
                dict.fromkeys(
                    item["array_path"]
                    for item in target_objects
                    if item["logical_role"]
                    in {
                        "array_chunk",
                        "record_column_chunk",
                        "record_validity_chunk",
                    }
                )
            )
            or [target_path],
        )
        normalized.setdefault(
            "object_paths",
            [item["path"] for item in target_objects],
        )
        normalized.setdefault("chunk_length", normalized["after"] - normalized["before"])
        normalized.setdefault("seals_target", False)
        normalized_extents.append(normalized)
    return with_record_checksum(
        {
            "kind": "prepare",
            "sequence": sequence,
            "transaction_id": f"tx-{number:016d}",
            "previous_committed_transaction_id": previous,
            "objects": objects,
            "extents": normalized_extents,
        }
    )


def commit(sequence: int, number: int, prepare_sequence: int) -> dict[str, Any]:
    return with_record_checksum(
        {
            "kind": "commit",
            "sequence": sequence,
            "transaction_id": f"tx-{number:016d}",
            "prepare_sequence": prepare_sequence,
        }
    )


def zarr_metadata_snapshot(
    array_path: str,
    shape: list[int],
    chunk_shape: list[int],
) -> bytes:
    return canonical_json(
        {
            "target_path": array_path,
            "zarr_metadata": {
                "zarr_format": 3,
                "node_type": "array",
                "shape": shape,
                "data_type": "int16",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": chunk_shape},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {},
                "dimension_names": [f"axis-{i}" for i in range(len(shape))],
                "storage_transformers": [],
            },
        }
    ).encode("utf-8")


def array_append(
    transaction_id: str,
    *,
    array_path: str,
    before: int,
    after: int,
    shape: list[int],
    chunk_shape: list[int],
    chunk_coordinate: list[int],
    payload: bytes,
    seals_target: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_path = f"{array_path}/c/" + "/".join(str(item) for item in chunk_coordinate)
    metadata_path = f"metadata/transactions/{transaction_id}/{array_path.replace('/', '-')}.json"
    if not shape or shape[0] != after or len(shape) != len(chunk_shape):
        raise ValueError("array append shape must match after extent and chunk rank")
    objects = [
        object_write(
            transaction_id,
            chunk_path,
            payload,
            "zarr_chunk",
            target_path=array_path,
            array_path=array_path,
            logical_role="array_chunk",
            chunk_coordinate=chunk_coordinate,
        ),
        object_write(
            transaction_id,
            metadata_path,
            zarr_metadata_snapshot(array_path, shape, chunk_shape),
            "zarr_metadata",
            target_path=array_path,
            array_path=array_path,
            logical_role="metadata_snapshot",
        ),
    ]
    transition = {
        "target_path": array_path,
        "target_kind": "array",
        "before": before,
        "after": after,
        "chunk_length": chunk_shape[0],
        "seals_target": seals_target,
        "required_array_paths": [array_path],
        "object_paths": [item["path"] for item in objects],
    }
    return objects, transition


def record_set_append(
    transaction_id: str,
    *,
    schema: dict[str, Any],
    before: int,
    after: int,
    seals_target: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_path = schema["path"]
    chunk_length = schema["chunk_length"]
    leading_coordinate = before // chunk_length
    objects: list[dict[str, Any]] = []
    required_array_paths: list[str] = []
    for field_descriptor in schema["fields"]:
        column_name = field_descriptor["name"]
        trailing_shape = field_descriptor.get("shape", [])
        coordinate = [leading_coordinate, *([0] * len(trailing_shape))]
        array_path = field_descriptor["array_path"]
        required_array_paths.append(array_path)
        objects.append(
            object_write(
                transaction_id,
                f"{array_path}/c/" + "/".join(str(item) for item in coordinate),
                f"{schema['id']}:{column_name}:data:{before}:{after}".encode(),
                "zarr_chunk",
                target_path=target_path,
                array_path=array_path,
                logical_role="record_column_chunk",
                chunk_coordinate=coordinate,
                record_schema_id=schema["id"],
                column_name=column_name,
            )
        )
        objects.append(
            object_write(
                transaction_id,
                (f"metadata/transactions/{transaction_id}/{schema['id']}-{column_name}-data.json"),
                zarr_metadata_snapshot(
                    array_path,
                    [after, *trailing_shape],
                    [chunk_length, *trailing_shape],
                ),
                "zarr_metadata",
                target_path=target_path,
                array_path=array_path,
                logical_role="metadata_snapshot",
            )
        )
        if field_descriptor["nullable"]:
            validity_path = field_descriptor["validity_path"]
            required_array_paths.append(validity_path)
            objects.append(
                object_write(
                    transaction_id,
                    f"{validity_path}/c/{leading_coordinate}",
                    f"{schema['id']}:{column_name}:validity:{before}:{after}".encode(),
                    "zarr_chunk",
                    target_path=target_path,
                    array_path=validity_path,
                    logical_role="record_validity_chunk",
                    chunk_coordinate=[leading_coordinate],
                    record_schema_id=schema["id"],
                    column_name=column_name,
                )
            )
            objects.append(
                object_write(
                    transaction_id,
                    (
                        f"metadata/transactions/{transaction_id}/"
                        f"{schema['id']}-{column_name}-validity.json"
                    ),
                    zarr_metadata_snapshot(
                        validity_path,
                        [after],
                        [chunk_length],
                    ),
                    "zarr_metadata",
                    target_path=target_path,
                    array_path=validity_path,
                    logical_role="metadata_snapshot",
                )
            )
    transition = {
        "target_path": target_path,
        "target_kind": "record_set",
        "record_schema_id": schema["id"],
        "before": before,
        "after": after,
        "chunk_length": chunk_length,
        "seals_target": seals_target,
        "required_array_paths": required_array_paths,
        "object_paths": [item["path"] for item in objects],
    }
    return objects, transition


def seal_only_transition(
    *,
    target_path: str,
    target_kind: str,
    before: int,
    chunk_length: int,
    required_array_paths: list[str],
    record_schema_id: str | None = None,
) -> dict[str, Any]:
    transition: dict[str, Any] = {
        "target_path": target_path,
        "target_kind": target_kind,
        "before": before,
        "after": before,
        "chunk_length": chunk_length,
        "seals_target": True,
        "required_array_paths": required_array_paths,
        "object_paths": [],
    }
    if record_schema_id is not None:
        transition["record_schema_id"] = record_schema_id
    return transition


def partial_chunk_vectors() -> dict[str, Any]:
    full_chunk = bytes.fromhex("0100020003000400")
    final_partial = bytes.fromhex("05000600")
    first_transaction = "tx-0000000000000001"
    full_objects, full_transition = array_append(
        first_transaction,
        array_path="streams/example/data",
        before=0,
        after=4,
        shape=[4],
        chunk_shape=[4],
        chunk_coordinate=[0],
        payload=full_chunk,
    )
    full_prepare = prepare(
        1,
        1,
        None,
        full_objects,
        [full_transition],
    )
    full_commit = commit(2, 1, 1)
    final_objects, final_transition = array_append(
        "tx-0000000000000002",
        array_path="streams/example/data",
        before=4,
        after=6,
        shape=[6],
        chunk_shape=[4],
        chunk_coordinate=[1],
        payload=final_partial,
        seals_target=True,
    )
    final_prepare = prepare(
        3,
        2,
        first_transaction,
        final_objects,
        [final_transition],
    )
    normal_termination = with_record_checksum(
        {
            "kind": "termination",
            "sequence": 5,
            "termination_record_id": "termination-0001",
            "termination_kind": "normal",
            "time_ns": 2000000,
            "reason": "completed",
            "fault_id": None,
            "last_transaction_id": "tx-0000000000000002",
        }
    )
    next_objects, next_transition = array_append(
        "tx-0000000000000002",
        array_path="streams/example/data",
        before=4,
        after=8,
        shape=[8],
        chunk_shape=[4],
        chunk_coordinate=[1],
        payload=bytes.fromhex("0500060007000800"),
    )
    next_full_prepare = prepare(
        3,
        2,
        first_transaction,
        next_objects,
        [next_transition],
    )
    crash_prepare = deepcopy_record(full_prepare)
    # Empty-target finalization: a target that never received an append stays
    # at extent 0 for the whole session. Normal finalization must still seal it.
    # A seal-only transaction (no staged objects, after == before) expresses
    # that without inventing a chunk.
    empty_seal_prepare = prepare(
        1,
        1,
        None,
        [],
        [
            {
                "target_path": "streams/example/data",
                "target_kind": "array",
                "before": 0,
                "after": 0,
                "chunk_length": 4,
                "seals_target": True,
                "required_array_paths": ["streams/example/data"],
                "object_paths": [],
            }
        ],
    )
    empty_seal_commit = commit(2, 1, 1)
    empty_seal_termination = with_record_checksum(
        {
            "kind": "termination",
            "sequence": 3,
            "termination_record_id": "termination-0001",
            "termination_kind": "normal",
            "time_ns": 1000000,
            "reason": "completed",
            "fault_id": None,
            "last_transaction_id": "tx-0000000000000001",
        }
    )
    return {
        "append_smaller_than_chunk": {
            "chunk_length": 4,
            "initial_committed_extent": 0,
            "append_sizes": [2],
            "writer_tail_extent": 2,
            "journal_records": [],
            "expected_committed_extent": 0,
        },
        "multiple_appends_fill_one_chunk": {
            "chunk_length": 4,
            "initial_committed_extent": 0,
            "append_sizes": [2, 2],
            "writer_tail_after_each_append": [2, 0],
            "journal_records": [full_prepare, full_commit],
            "expected_committed_extent": 4,
        },
        "crash_before_commit": {
            "crash_point": "after_prepare_before_object_promotion",
            "journal_records": [crash_prepare],
            "staged_objects_ignored": [item["staged_path"] for item in crash_prepare["objects"]],
            "expected_committed_extent": 0,
        },
        "crash_after_object_visibility_before_commit": {
            "crash_point": "after_object_promotion_before_commit",
            "journal_records": [
                deepcopy_record(full_prepare),
                deepcopy_record(full_commit),
                next_full_prepare,
            ],
            "previous_committed_objects": [item["path"] for item in full_prepare["objects"]],
            "orphan_objects_ignored": [item["path"] for item in next_full_prepare["objects"]],
            "expected_committed_extent": 4,
            "previous_committed_objects_modified": False,
        },
        "final_partial_chunk": {
            "chunk_length": 4,
            "initial_committed_extent": 4,
            "append_sizes": [2],
            "journal_records": [final_prepare, commit(4, 2, 3), normal_termination],
            "expected_committed_extent": 6,
            "sealed_targets": ["streams/example/data"],
            "session_terminated": True,
            "further_append_allowed": False,
        },
        "empty_target_seal": {
            "chunk_length": 4,
            "initial_committed_extent": 0,
            "append_sizes": [],
            "journal_records": [empty_seal_prepare, empty_seal_commit, empty_seal_termination],
            "expected_committed_extent": 0,
            "sealed_targets": ["streams/example/data"],
            "session_terminated": True,
            "further_append_allowed": False,
        },
    }


def deepcopy_record(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record))


def build_vectors() -> dict[str, Any]:
    payloads = {
        "neural": bytes.fromhex("01000200030004000500060007000800"),
        "behavior": bytes.fromhex("000000000000803f0000004000004040"),
        "behavior_times": bytes.fromhex("00000000000000008096980000000000"),
        "feature": bytes.fromhex("000000000000f03f0000000000000040"),
        "state": b"state-transition-column-chunks",
        "command": b"command-column-chunks",
        "discontinuity": b"segment-0001->segment-0002@4",
        "fault": b"queue_overflow@4",
        "tail": b"uncommitted-tail",
    }
    committed_before_termination = {
        "streams/neural/data": 4,
        "streams/cursor/data": 2,
        "streams/cursor/timestamps": 2,
        "streams/bandpower/data": 1,
        "records/experiment_state/experiment-state-v1": 1,
        "records/commands/commands-v1": 1,
        "streams/neural/discontinuities": 1,
        "records/faults/faults-v1": 1,
    }
    preliminary_manifest = manifest(committed_before_termination)
    appendable_targets = {stream["data"]["path"] for stream in preliminary_manifest["streams"]}
    appendable_targets.update(
        stream["timing"]["timestamps"]["path"]
        for stream in preliminary_manifest["streams"]
        if stream["timing"]["mode"] == "explicit"
    )
    appendable_targets.update(schema["path"] for schema in preliminary_manifest["record_schemas"])
    final_extents = {
        path: committed_before_termination.get(path, 0) for path in sorted(appendable_targets)
    }
    termination_target = "records/session_termination/termination-v1"
    final_extents[termination_target] = 1
    manifest_value = manifest(final_extents)
    streams_by_id = {descriptor["id"]: descriptor for descriptor in manifest_value["streams"]}
    records_by_path = {schema["path"]: schema for schema in manifest_value["record_schemas"]}

    records: list[dict[str, Any]] = []
    tx1 = "tx-0000000000000001"
    neural_data = streams_by_id["neural"]["data"]
    tx1_objects, tx1_transition = array_append(
        tx1,
        array_path=neural_data["path"],
        before=0,
        after=4,
        shape=neural_data["shape"],
        chunk_shape=neural_data["chunk_shape"],
        chunk_coordinate=[0, 0],
        payload=payloads["neural"],
    )
    records.extend(
        [
            prepare(
                1,
                1,
                None,
                tx1_objects,
                [tx1_transition],
            ),
            commit(2, 1, 1),
        ]
    )
    tx2 = "tx-0000000000000002"
    cursor = streams_by_id["cursor"]
    cursor_data_objects, cursor_data_transition = array_append(
        tx2,
        array_path=cursor["data"]["path"],
        before=0,
        after=2,
        shape=cursor["data"]["shape"],
        chunk_shape=cursor["data"]["chunk_shape"],
        chunk_coordinate=[0, 0],
        payload=payloads["behavior"],
    )
    cursor_time_objects, cursor_time_transition = array_append(
        tx2,
        array_path=cursor["timing"]["timestamps"]["path"],
        before=0,
        after=2,
        shape=cursor["timing"]["timestamps"]["shape"],
        chunk_shape=cursor["timing"]["timestamps"]["chunk_shape"],
        chunk_coordinate=[0],
        payload=payloads["behavior_times"],
    )
    records.extend(
        [
            prepare(
                3,
                2,
                tx1,
                [
                    *cursor_data_objects,
                    *cursor_time_objects,
                ],
                [
                    cursor_data_transition,
                    cursor_time_transition,
                ],
            ),
            commit(4, 2, 3),
        ]
    )
    tx3 = "tx-0000000000000003"
    bandpower_data = streams_by_id["bandpower"]["data"]
    tx3_objects, tx3_transition = array_append(
        tx3,
        array_path=bandpower_data["path"],
        before=0,
        after=1,
        shape=bandpower_data["shape"],
        chunk_shape=bandpower_data["chunk_shape"],
        chunk_coordinate=[0, 0],
        payload=payloads["feature"],
    )
    records.extend(
        [
            prepare(
                5,
                3,
                tx2,
                tx3_objects,
                [tx3_transition],
            ),
            commit(6, 3, 5),
        ]
    )
    tx4 = "tx-0000000000000004"
    state_objects, state_transition = record_set_append(
        tx4,
        schema=records_by_path["records/experiment_state/experiment-state-v1"],
        before=0,
        after=1,
    )
    command_objects, command_transition = record_set_append(
        tx4,
        schema=records_by_path["records/commands/commands-v1"],
        before=0,
        after=1,
    )
    records.extend(
        [
            prepare(
                7,
                4,
                tx3,
                [
                    *state_objects,
                    *command_objects,
                ],
                [
                    state_transition,
                    command_transition,
                ],
            ),
            commit(8, 4, 7),
        ]
    )
    tx5 = "tx-0000000000000005"
    discontinuity_objects, discontinuity_transition = record_set_append(
        tx5,
        schema=records_by_path["streams/neural/discontinuities"],
        before=0,
        after=1,
    )
    fault_objects, fault_transition = record_set_append(
        tx5,
        schema=records_by_path["records/faults/faults-v1"],
        before=0,
        after=1,
    )
    records.extend(
        [
            prepare(
                9,
                5,
                tx4,
                [
                    *discontinuity_objects,
                    *fault_objects,
                ],
                [
                    discontinuity_transition,
                    fault_transition,
                ],
            ),
            commit(10, 5, 9),
        ]
    )

    committed_objects = [
        item for record in records if record["kind"] == "prepare" for item in record["objects"]
    ]
    checkpoint = with_record_checksum(
        {
            "format": "nrf-checkpoint",
            "version": {"major": 1, "minor": 0},
            "checkpoint_id": "checkpoint-0000000000000001",
            "journal_sequence": 10,
            "last_committed_transaction_id": tx5,
            "committed_extents": committed_before_termination,
            "sealed_targets": [],
            "committed_objects": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"staged_path", "disposition"}
                }
                for item in committed_objects
            ],
            "index_extents": {},
        }
    )
    checkpoint_bytes = canonical_json(checkpoint).encode("utf-8")
    checkpoint_record = with_record_checksum(
        {
            "kind": "checkpoint",
            "sequence": 11,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_path": "journal/checkpoints/checkpoint-0000000000000001.json",
            "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
        }
    )
    records.append(checkpoint_record)
    active_records = list(records)

    # Normal finalization is one final transaction. It appends and seals the
    # sole typed session_termination row and seals every other frozen
    # appendable target. The journal termination record follows the commit and
    # references that typed row.
    tx6 = "tx-0000000000000006"
    termination_schema = records_by_path[termination_target]
    termination_objects, termination_transition = record_set_append(
        tx6,
        schema=termination_schema,
        before=0,
        after=1,
        seals_target=True,
    )
    final_seals: list[dict[str, Any]] = []
    for stream in manifest_value["streams"]:
        data = stream["data"]
        final_seals.append(
            seal_only_transition(
                target_path=data["path"],
                target_kind="array",
                before=committed_before_termination.get(data["path"], 0),
                chunk_length=data["chunk_shape"][0],
                required_array_paths=[data["path"]],
            )
        )
        if stream["timing"]["mode"] == "explicit":
            timestamps = stream["timing"]["timestamps"]
            final_seals.append(
                seal_only_transition(
                    target_path=timestamps["path"],
                    target_kind="array",
                    before=committed_before_termination.get(timestamps["path"], 0),
                    chunk_length=timestamps["chunk_shape"][0],
                    required_array_paths=[timestamps["path"]],
                )
            )
    for schema in manifest_value["record_schemas"]:
        if schema["path"] == termination_target:
            continue
        required_array_paths = [
            path
            for field_descriptor in schema["fields"]
            for path in (
                [field_descriptor["array_path"], field_descriptor["validity_path"]]
                if field_descriptor["nullable"]
                else [field_descriptor["array_path"]]
            )
        ]
        final_seals.append(
            seal_only_transition(
                target_path=schema["path"],
                target_kind="record_set",
                before=committed_before_termination.get(schema["path"], 0),
                chunk_length=schema["chunk_length"],
                required_array_paths=required_array_paths,
                record_schema_id=schema["id"],
            )
        )
    final_prepare = prepare(
        12,
        6,
        tx5,
        termination_objects,
        [*final_seals, termination_transition],
    )
    final_commit = commit(13, 6, 12)
    termination_row = {
        "termination_id": "termination-0001",
        "termination_kind": "normal",
        "time_ns": 2000000,
        "reason": "completed",
        "fault_id": None,
        "last_transaction_id": tx6,
    }
    journal_termination = with_record_checksum(
        {
            "kind": "termination",
            "sequence": 14,
            "termination_record_id": termination_row["termination_id"],
            "termination_kind": termination_row["termination_kind"],
            "time_ns": termination_row["time_ns"],
            "reason": termination_row["reason"],
            "fault_id": termination_row["fault_id"],
            "last_transaction_id": termination_row["last_transaction_id"],
        }
    )
    records.extend([final_prepare, final_commit, journal_termination])

    tail_objects, tail_transition = array_append(
        "tx-0000000000000006",
        array_path=neural_data["path"],
        before=4,
        after=8,
        shape=[8, *neural_data["shape"][1:]],
        chunk_shape=neural_data["chunk_shape"],
        chunk_coordinate=[1, 0],
        payload=payloads["tail"],
    )
    tail = prepare(
        12,
        6,
        tx5,
        tail_objects,
        [tail_transition],
    )
    recovery_report = with_record_checksum(
        {
            "format": "nrf-recovery-report",
            "version": {"major": 1, "minor": 0},
            "recovery_id": "recovery-0001",
            "created_at": "2026-07-30T09:01:00Z",
            "base_checkpoint_id": checkpoint["checkpoint_id"],
            "last_valid_journal_sequence": 12,
            "last_committed_transaction_id": tx5,
            "committed_extents": committed_before_termination,
            "sealed_targets": [],
            "retained_transaction_ids": [f"tx-{number:016d}" for number in range(1, 6)],
            "ignored_transaction_ids": ["tx-0000000000000006"],
            "ignored_records": [{"sequence": 12, "reason": "uncommitted_prepare"}],
            "missing_objects": [],
            "corrupt_objects": [],
            "quarantined_paths": [],
            "rebuilt_indexes": [],
            "committed_data_modified": False,
        }
    )
    manifest_vector = manifest_value
    # journal/head.json is the rebuildable cache pointing at the current
    # committed state; it mirrors the manifest commit cache and replay through
    # the terminal journal record (sequence 14). It carries no
    # record_checksum because it is a cache, not a journal record.
    head = {
        "format": "nrf-head",
        "version": {"major": 1, "minor": 0},
        "journal_sequence": 14,
        "last_transaction_id": tx6,
        "last_checkpoint_id": checkpoint["checkpoint_id"],
        "committed_extents": final_extents,
        "sealed_targets": sorted(final_extents),
    }
    # feature_sets/<feature-set-id>.json files are byte-for-value copies of the
    # manifest feature_sets registry entries; the manifest is authoritative.
    feature_set_files = {item["id"]: item for item in manifest_vector["feature_sets"]}
    return {
        "format": "nrf-v1-test-vectors",
        "version": 1,
        "canonical_json": {
            "algorithm": "rfc8785",
            "cases": jcs_vectors(),
        },
        "payload_vectors": [
            {"name": name, "payload_hex": payload.hex(), "sha256": sha256_bytes(payload)}
            for name, payload in payloads.items()
        ],
        "manifest": manifest_vector,
        "head": head,
        "feature_set_files": feature_set_files,
        "append_cases": {
            "regular_neural": records[0:2],
            "explicit_behavioral": records[2:4],
            "feature": records[4:6],
            "experiment_state_and_command": records[6:8],
            "discontinuity_and_fault": records[8:10],
        },
        "logical_record_vectors": {
            "experiment_state": {
                "state_transition_id": "state-transition-0001",
                "time_ns": 100000000,
                "previous_state": "inter_trial",
                "new_state": "move",
                "trial_id": "trial-0001",
            },
            "command": {
                "command_id": "command-0001",
                "time_ns": 100000000,
                "command_type": "external_velocity",
                "value": [1, -1],
                "trial_id": "trial-0001",
            },
            "discontinuity": {
                "discontinuity_id": "discontinuity-0001",
                "stream_id": "neural",
                "previous_segment_id": "segment-0001",
                "next_segment_id": "segment-0002",
                "expected_sample_index": 4,
                "actual_sample_index": 8,
                "missing_samples": 4,
                "reason": "source_gap",
                "time_ns": 1000000,
            },
            "fault": {
                "fault_id": "fault-0001",
                "detected_at_ns": 1000000,
                "code": "queue_overrun",
                "status": "queue_overflow",
                "stage": "observer",
                "stream_id": "neural",
                "sample_index": 4,
            },
            "session_termination": termination_row,
        },
        "complete_sequence": records,
        "checkpoint": checkpoint,
        "recoverable_uncommitted_tail": {
            "journal_records": [*active_records, tail],
            "expected_report": recovery_report,
        },
        "partial_chunk_cases": partial_chunk_vectors(),
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # This container is human-readable JSON. Canonical bytes and checksums are
    # generated separately through RFC 8785 above.
    OUTPUT.write_text(
        json.dumps(build_vectors(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
