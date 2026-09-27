#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Regenerate this extension's schema and fixtures from its README tables.

``README.md`` is the source. This tool parses the ledger tables out of it and
emits ``native-replay.schema.json`` and ``fixtures/``, so a field cannot be added
to the specification text and quietly missed by the machine-readable form. It is
informative rather than normative: where the tool and the document differ, the
document governs.

It imports nothing from ``neurale``. The production implementation is checked
*against* these artifacts, which is only evidence while the two are independent.

Run ``python generate_fixtures.py --check`` to verify the checked-in files are
exactly what the tables produce.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

SPEC_DIR = Path(__file__).resolve().parent.parent


def _load_jcs():
    """Load the specification's own RFC 8785 implementation, by path.

    ``specifications/nrf/v1/tools`` is not a package, and this tool must not
    import ``neurale``: the fixtures are evidence only while they were produced
    without the implementation they check.
    """
    path = SPEC_DIR.parent.parent / "tools" / "jcs.py"
    spec = importlib.util.spec_from_file_location("_nrf_v1_jcs", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise SystemExit(f"generate_fixtures: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jcs = _load_jcs()
README = SPEC_DIR / "README.md"
SCHEMA_PATH = SPEC_DIR / "native-replay.schema.json"
FIXTURE_DIR = SPEC_DIR / "fixtures"

NAMESPACE = "neurale.native_replay"
EXTENSION_VERSION = 1

#: Codec IDs of the writer that produced the fixtures. Session-scoped registry
#: entries, not values this extension fixes; see README section 7.
BYTES_LE_CODEC = "bytes-le"
UTF8_CODEC = "utf8-vlen"

DTYPES = frozenset(
    {"int16", "int32", "int64", "uint8", "uint32", "uint64", "float32", "float64", "bool", "utf8"}
)

#: Stable member names of the native schema enums, exactly as the pybind11
#: bindings expose them. The manifest extension carries these by name so the
#: plan fingerprint is portable and independent of a particular build's integer
#: assignment.
SIGNAL_DTYPES = ("INT16", "INT32", "FLOAT32", "FLOAT64")
SIGNAL_LAYOUTS = ("SAMPLE_MAJOR", "CHANNEL_MAJOR")
DEVICE_TICK_TRACKING = ("UNAVAILABLE", "SAMPLE_COUNTER")
SIGNAL_KINDS = ("SAMPLED", "EVENT", "FEATURE", "SPIKE")
PHYSICAL_UNITS = ("UNSPECIFIED", "VOLTS", "AMPERES", "DIMENSIONLESS")
OBSERVATION_TIMINGS = ("NOT_APPLICABLE", "REGULAR")
TIMESTAMP_REFERENCES = ("WINDOW_CENTER",)

#: Ledger kind -> (schema id, primary key, chunk length). Mirrors the section 2
#: table; the field lists come from the section 3 tables.
LEDGERS: dict[str, tuple[str, str, int]] = {
    "native_frames": ("native-frames-v1", "data_message_ordinal", 1024),
    "native_signal_blocks": ("native-signal-blocks-v1", "signal_block_ordinal", 1024),
    "native_discontinuities": ("native-discontinuities-v1", "data_message_ordinal", 64),
    "native_signal_gaps": ("native-signal-gaps-v1", "signal_gap_ordinal", 64),
    "session_accounting": ("session-accounting-v1", "accounting_id", 1),
}

#: Heading that introduces each ledger's field table in section 3.
_HEADING = re.compile(r"^### 3\.\d+ `([a-z_]+)`$")
_CODE = re.compile(r"^`([a-z0-9_]+)`$")


def _fail(message: str) -> None:
    raise SystemExit(f"generate_fixtures: {message}")


def parse_field_tables(text: str) -> dict[str, list[dict[str, Any]]]:
    """Return ``{kind: [field, ...]}`` parsed from the section 3 tables."""
    tables: dict[str, list[dict[str, Any]]] = {}
    kind: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        heading = _HEADING.match(line)
        if heading is not None:
            kind = heading.group(1)
            if kind not in LEDGERS:
                _fail(f"README names an unknown ledger kind {kind!r}")
            tables[kind] = []
            continue
        if kind is None or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5 or cells[0] in {"#", "---"}:
            continue
        pos, name_cell, dtype, nullable, reference = cells
        if not pos.isdigit():
            continue
        name = _CODE.match(name_cell)
        if name is None:
            _fail(f"{kind}: field name {name_cell!r} is not a code span")
        if dtype not in DTYPES:
            _fail(f"{kind}: unknown dtype {dtype!r}")
        if nullable not in {"yes", "no"}:
            _fail(f"{kind}: nullable must be yes or no, got {nullable!r}")
        if int(pos) != len(tables[kind]) + 1:
            _fail(f"{kind}: field {name.group(1)!r} is numbered out of order")
        field: dict[str, Any] = {
            "name": name.group(1),
            "dtype": dtype,
            "nullable": nullable == "yes",
        }
        if reference not in {"—", "-"}:
            field["reference"] = reference
        tables[kind].append(field)
    missing = sorted(set(LEDGERS) - set(tables))
    if missing:
        _fail(f"README has no field table for {missing}")
    empty = sorted(kind for kind, fields in tables.items() if not fields)
    if empty:
        _fail(f"README field table is empty for {empty}")
    return tables


# --- descriptors ---------------------------------------------------------


def record_set_path(kind: str, schema_id: str) -> str:
    return f"records/{kind}/{schema_id}"


def _endianness(dtype: str) -> str:
    return "not_applicable" if dtype in {"bool", "uint8", "utf8"} else "little"


def _codec(dtype: str) -> str:
    return UTF8_CODEC if dtype == "utf8" else BYTES_LE_CODEC


def record_field(record_path: str, field: dict[str, Any]) -> dict[str, Any]:
    """Return one NRF v1 record-field descriptor for a ledger column."""
    name = field["name"]
    dtype = field["dtype"]
    descriptor: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "endianness": _endianness(dtype),
        "nullable": field["nullable"],
        "array_path": f"{record_path}/columns/{name}",
        "codec_ids": [_codec(dtype)],
    }
    if field["nullable"]:
        descriptor["validity_path"] = f"{record_path}/validity/{name}"
    if "reference" in field:
        descriptor["reference"] = field["reference"]
    return descriptor


def record_schema(kind: str, fields: list[dict[str, Any]], *, clock_id: str) -> dict[str, Any]:
    """Return the manifest ``record_schemas`` entry for one ledger."""
    schema_id, primary_key, chunk_length = LEDGERS[kind]
    path = record_set_path(kind, schema_id)
    return {
        "id": schema_id,
        "kind": kind,
        "path": path,
        "primary_key": primary_key,
        "clock_id": clock_id,
        "chunk_length": chunk_length,
        "committed_extent": 0,
        "fields": [record_field(path, field) for field in fields],
    }


# --- machine-readable schema ---------------------------------------------


def _field_schema(record_path: str, field: dict[str, Any]) -> dict[str, Any]:
    descriptor = record_field(record_path, field)
    properties = {key: {"const": value} for key, value in descriptor.items()}
    return {
        "type": "object",
        "required": sorted(descriptor),
        "properties": properties,
    }


def _ledger_schema(kind: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    schema_id, primary_key, _chunk_length = LEDGERS[kind]
    path = record_set_path(kind, schema_id)
    return {
        "type": "object",
        "required": ["id", "kind", "path", "primary_key", "fields", "chunk_length"],
        "properties": {
            "id": {"const": schema_id},
            "kind": {"const": kind},
            "path": {"const": path},
            "primary_key": {"const": primary_key},
            "chunk_length": {"type": "integer", "minimum": 1},
            "fields": {
                "type": "array",
                "minItems": len(fields),
                "maxItems": len(fields),
                "prefixItems": [_field_schema(path, field) for field in fields],
            },
        },
    }


def _capability_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["available", "reason"],
        "additionalProperties": False,
        "properties": {
            "available": {"type": "boolean"},
            "reason": {"type": ["string", "null"], "minLength": 1},
        },
        "allOf": [
            {
                "if": {"properties": {"available": {"const": True}}, "required": ["available"]},
                "then": {"properties": {"reason": {"type": "null"}}},
                "else": {"properties": {"reason": {"type": "string", "minLength": 1}}},
            }
        ],
    }


def _signal_schema_object() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "id",
            "clock_domain",
            "n_channels",
            "nominal_block_samples",
            "max_block_samples",
            "fs",
            "dtype",
            "layout",
            "device_tick_tracking",
            "kind",
            "physical_unit",
            "channel_set_id",
            "calibration_id",
            "reference_id",
            "feature_set_id",
            "observation_timing",
            "fixed_block_bytes",
            "max_block_bytes",
        ],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer", "minimum": 0},
            "clock_domain": {"type": "integer", "minimum": 0},
            "n_channels": {"type": "integer", "minimum": 0},
            "nominal_block_samples": {"type": "integer", "minimum": 0},
            "max_block_samples": {"type": "integer", "minimum": 0},
            "fs": {
                "type": "object",
                "required": ["numerator", "denominator"],
                "additionalProperties": False,
                "properties": {
                    "numerator": {"type": "integer", "minimum": 0},
                    "denominator": {"type": "integer", "minimum": 1},
                },
            },
            "dtype": {"enum": list(SIGNAL_DTYPES)},
            "layout": {"enum": list(SIGNAL_LAYOUTS)},
            "device_tick_tracking": {"enum": list(DEVICE_TICK_TRACKING)},
            "kind": {"enum": list(SIGNAL_KINDS)},
            "physical_unit": {"enum": list(PHYSICAL_UNITS)},
            "channel_set_id": {"type": "integer", "minimum": 0},
            "calibration_id": {"type": "integer", "minimum": 0},
            "reference_id": {"type": "integer", "minimum": 0},
            "feature_set_id": {"type": "integer", "minimum": 0},
            "observation_timing": {"enum": list(OBSERVATION_TIMINGS)},
            "fixed_block_bytes": {"type": "integer", "minimum": 0},
            "max_block_bytes": {"type": "integer", "minimum": 0},
        },
    }


def _feature_set_schema_object() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "id",
            "feature_names",
            "unit_ids",
            "source_stream_id",
            "source_stream",
            "algorithm_name",
            "algorithm_version",
            "window_length_ns",
            "shift_ns",
            "timestamp_reference",
        ],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer", "minimum": 1},
            "feature_names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "unit_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "source_stream_id": {"type": "integer", "minimum": 0},
            "source_stream": {"type": "string"},
            "algorithm_name": {"type": "string"},
            "algorithm_version": {"type": "string"},
            "window_length_ns": {"type": "integer", "minimum": 1},
            "shift_ns": {"type": "integer", "minimum": 1},
            "timestamp_reference": {"enum": list(TIMESTAMP_REFERENCES)},
        },
    }


def _unit_schema_object() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["id", "symbol", "description"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer", "minimum": 1},
            "symbol": {"type": "string"},
            "description": {"type": "string"},
        },
    }


def _native_schema_object() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["schema_id", "signals", "feature_sets", "units"],
        "additionalProperties": False,
        "properties": {
            "schema_id": {"type": "integer", "minimum": 0},
            "signals": {
                "type": "array",
                "minItems": 1,
                "items": _signal_schema_object(),
            },
            "feature_sets": {"type": "array", "items": _feature_set_schema_object()},
            "units": {"type": "array", "items": _unit_schema_object()},
        },
    }


def _session_object() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["session_id", "created_at", "writer_name", "writer_version"],
        "additionalProperties": False,
        "properties": {
            "session_id": {"type": "string"},
            "created_at": {"type": "string"},
            "writer_name": {"type": "string"},
            "writer_version": {"type": "string"},
        },
    }


def _resource_bounds_object() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "frame_queue_capacity",
            "control_queue_capacity",
            "control_chunk_length",
            "checkpoint_interval",
            "overflow_policy",
            "streams",
        ],
        "additionalProperties": False,
        "properties": {
            "frame_queue_capacity": {"type": "integer", "minimum": 1},
            "control_queue_capacity": {"type": "integer", "minimum": 1},
            "spool_capacity_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": 9007199254740991,
                "description": "Optional spool byte ceiling. Omission retains the legacy 64 MiB bound; continuous-file writers use the I-JSON integer ceiling when no disk limit is configured.",
            },
            "control_chunk_length": {"type": "integer", "minimum": 1},
            "checkpoint_interval": {"type": "integer", "minimum": 0},
            "overflow_policy": {"enum": ["fault", "continue"]},
            "streams": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "stream_id",
                        "capacity",
                        "chunk_length",
                        "block_index_chunk_length",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "stream_id": {"type": "string"},
                        "capacity": {"type": "integer", "minimum": 1},
                        "chunk_length": {"type": "integer", "minimum": 1},
                        "block_index_chunk_length": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
    }


def _manifest_extension_schema() -> dict[str, Any]:
    signal_ids = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
        "uniqueItems": True,
    }
    return {
        "type": "object",
        "required": [
            "extension_version",
            "plan_fingerprint",
            "native_schema",
            "coverage",
            "planned_signal_ids",
            "recorded_signal_ids",
            "streams",
            "ledgers",
            "replay_capabilities",
            "session",
            "resource_bounds",
            "session_metadata",
            "metadata",
        ],
        "additionalProperties": False,
        "properties": {
            "extension_version": {"const": EXTENSION_VERSION},
            "plan_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "native_schema": _native_schema_object(),
            "coverage": {"enum": ["full", "partial"]},
            "planned_signal_ids": signal_ids,
            "recorded_signal_ids": signal_ids,
            "streams": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "native_signal_id",
                        "stream_id",
                        "kind",
                        "timing",
                        "block_index",
                        "name",
                        "unit",
                        "channel_names",
                        "feature_names",
                        "segment_start_index",
                        "segment_start_time_ns",
                        "algorithm_name",
                        "algorithm_version",
                        "window_length_ns",
                        "shift_ns",
                        "source_stream_ids",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "native_signal_id": {"type": "integer", "minimum": 0},
                        "stream_id": {
                            "type": "string",
                            "pattern": "^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$",
                        },
                        "kind": {"enum": ["neural", "behavioral", "feature", "spike"]},
                        "timing": {"enum": ["explicit", "regular"]},
                        "block_index": {"type": "boolean"},
                        "name": {"type": "string"},
                        "unit": {"type": "array", "items": {"type": "string"}},
                        "channel_names": {"type": "array", "items": {"type": "string"}},
                        "feature_names": {"type": "array", "items": {"type": "string"}},
                        "segment_start_index": {"type": "integer", "minimum": 0},
                        "segment_start_time_ns": {"type": "integer"},
                        "algorithm_name": {"type": "string"},
                        "algorithm_version": {"type": "string"},
                        "window_length_ns": {"type": "integer", "minimum": 0},
                        "shift_ns": {"type": "integer", "minimum": 0},
                        "source_stream_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "ledgers": {
                "type": "object",
                "required": sorted(LEDGERS),
                "additionalProperties": False,
                "properties": {
                    kind: {"const": schema_id} for kind, (schema_id, _, _) in LEDGERS.items()
                },
            },
            "replay_capabilities": {
                "type": "object",
                "required": ["exact_frames", "recorded_projection", "stream_frames"],
                "additionalProperties": False,
                "properties": {
                    "exact_frames": {"$ref": "#/$defs/replayCapability"},
                    "recorded_projection": {"$ref": "#/$defs/replayCapability"},
                    "stream_frames": {"$ref": "#/$defs/replayCapability"},
                },
            },
            "session": _session_object(),
            "resource_bounds": _resource_bounds_object(),
            "session_metadata": {"type": "object"},
            "metadata": {"type": "object"},
        },
    }


def _journal_extension_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "extension_version",
            "plan_fingerprint",
            "termination_origin",
            "requested_terminal_intent",
            "capture_outcome",
            "effective_session_outcome",
            "finalization_status",
            "primary_fault_row_committed",
        ],
        "properties": {
            "extension_version": {"const": EXTENSION_VERSION},
            "plan_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "termination_origin": {"const": "recorder"},
            "requested_terminal_intent": {"enum": ["normal", "aborted", "fault"]},
            "capture_outcome": {"enum": ["normal", "aborted", "faulted", "unknown"]},
            "effective_session_outcome": {"enum": ["normal", "aborted", "faulted"]},
            "finalization_status": {
                "enum": ["not_started", "running", "failed_retryable", "succeeded", "abandoned"]
            },
            "primary_fault_row_committed": {"type": "boolean"},
        },
    }


def build_schema(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://pyneurale.org/specifications/nrf/v1/extensions/native-replay-v1.schema.json",
        "title": "NRF v1 native replay ledger extension",
        "description": (
            "Machine-readable form of the native-replay-v1 extension: the five ledger "
            "record schemas, the manifest extension object, and the journal termination "
            "extension object. Generated from README.md by tools/generate_fixtures.py."
        ),
        "type": "object",
        "$defs": {
            "namespace": {"const": NAMESPACE},
            "replayCapability": _capability_schema(),
            "ledgerRecordSchemas": {
                kind: _ledger_schema(kind, fields) for kind, fields in tables.items()
            },
            "manifestExtension": _manifest_extension_schema(),
            "journalTerminationExtension": _journal_extension_schema(),
        },
    }


# --- fixtures ------------------------------------------------------------


def _capability(available: bool, reason: str | None) -> dict[str, Any]:
    return {"available": available, "reason": reason}


def plan_fingerprint(document: dict[str, Any]) -> str:
    """Return the section 5 fingerprint of one plan document.

    Uses the specification's own RFC 8785 implementation, so the fixtures do not
    inherit the canonicalization of the code they are used to check.
    """
    return hashlib.sha256(jcs.canonical_json_bytes(document)).hexdigest()


def _example_native_schema() -> dict[str, Any]:
    """A small, self-consistent native schema the fixture plan was recorded under.

    Both the full- and partial-coverage fixtures share it: the schema declares
    signals 1 and 2, and the partial plan records only signal 1. Block byte
    counts are the products of max samples, channel count, and dtype width, so
    the example is internally consistent.
    """
    return {
        "schema_id": 7,
        "signals": [
            {
                "id": 1,
                "clock_domain": 1,
                "n_channels": 4,
                "nominal_block_samples": 8,
                "max_block_samples": 16,
                "fs": {"numerator": 1000, "denominator": 1},
                "dtype": "INT16",
                "layout": "SAMPLE_MAJOR",
                "device_tick_tracking": "SAMPLE_COUNTER",
                "kind": "SAMPLED",
                "physical_unit": "VOLTS",
                "channel_set_id": 0,
                "calibration_id": 0,
                "reference_id": 0,
                "feature_set_id": 0,
                "observation_timing": "NOT_APPLICABLE",
                "fixed_block_bytes": 0,
                "max_block_bytes": 128,
            },
            {
                "id": 2,
                "clock_domain": 1,
                "n_channels": 2,
                "nominal_block_samples": 2,
                "max_block_samples": 4,
                "fs": {"numerator": 100, "denominator": 1},
                "dtype": "FLOAT32",
                "layout": "SAMPLE_MAJOR",
                "device_tick_tracking": "UNAVAILABLE",
                "kind": "SAMPLED",
                "physical_unit": "DIMENSIONLESS",
                "channel_set_id": 0,
                "calibration_id": 0,
                "reference_id": 0,
                "feature_set_id": 0,
                "observation_timing": "NOT_APPLICABLE",
                "fixed_block_bytes": 0,
                "max_block_bytes": 32,
            },
        ],
        "feature_sets": [],
        "units": [],
    }


def _example_session() -> dict[str, Any]:
    return {
        "session_id": "018f4f30-6f9d-736-8c61-3f96cf5c5a40",
        "created_at": "2026-08-01T00:00:00Z",
        "writer_name": "pyneurale",
        "writer_version": "1",
    }


def _example_resource_bounds(*, full: bool) -> dict[str, Any]:
    streams = [
        {
            "stream_id": "neural",
            "capacity": 4096,
            "chunk_length": 1024,
            "block_index_chunk_length": 256,
        }
    ]
    if full:
        streams.append(
            {
                "stream_id": "emg",
                "capacity": 1024,
                "chunk_length": 1024,
                "block_index_chunk_length": 256,
            }
        )
    return {
        "frame_queue_capacity": 256,
        "control_queue_capacity": 1024,
        "control_chunk_length": 16,
        "checkpoint_interval": 32,
        "overflow_policy": "fault",
        "streams": streams,
    }


def _stream_entry(
    signal_id: int, stream_id: str, kind: str, n_channels: int, unit: str
) -> dict[str, Any]:
    """One resolved, non-feature, explicit-timing stream entry for the fixture.

    Carries every field the plan document fingerprints: the per-column unit and
    channel names a declaration resolved to, the regular-timing origin fixed at
    zero for explicit timing, and the feature fields emptied for a non-feature
    stream. Two fixtures that differ only in these record different things and
    fingerprint differently; two that differ only in storage bounds do not.
    """
    return {
        "native_signal_id": signal_id,
        "stream_id": stream_id,
        "kind": kind,
        "timing": "explicit",
        "block_index": True,
        "name": stream_id,
        "unit": [unit] * n_channels,
        "channel_names": [f"{stream_id}-{i}" for i in range(n_channels)],
        "feature_names": [],
        "segment_start_index": 0,
        "segment_start_time_ns": 0,
        "algorithm_name": "",
        "algorithm_version": "",
        "window_length_ns": 0,
        "shift_ns": 0,
        "source_stream_ids": [],
    }


def _manifest_extension(*, coverage: str) -> dict[str, Any]:
    full = coverage == "full"
    native_schema = _example_native_schema()
    streams = [
        _stream_entry(1, "neural", "neural", 4, "V"),
    ]
    if full:
        streams.append(_stream_entry(2, "emg", "behavioral", 2, "1"))
    ledgers = {kind: schema_id for kind, (schema_id, _, _) in LEDGERS.items()}
    document = {
        "extension_version": EXTENSION_VERSION,
        "native_schema": native_schema,
        "coverage": coverage,
        "planned_signal_ids": [1, 2],
        "recorded_signal_ids": [1, 2] if full else [1],
        "streams": streams,
        "ledgers": ledgers,
    }
    return {
        "extension_version": EXTENSION_VERSION,
        "plan_fingerprint": plan_fingerprint(document),
        "native_schema": native_schema,
        "coverage": coverage,
        "planned_signal_ids": [1, 2],
        "recorded_signal_ids": [1, 2] if full else [1],
        "streams": streams,
        "ledgers": ledgers,
        "replay_capabilities": {
            "exact_frames": _capability(
                full,
                None
                if full
                else (
                    "the recording plan covered 1 of 2 signals; replay this session as "
                    "recorded_projection"
                ),
            ),
            "recorded_projection": _capability(True, None),
            "stream_frames": _capability(True, None),
        },
        "session": _example_session(),
        "resource_bounds": _example_resource_bounds(full=full),
        "session_metadata": {},
        "metadata": {},
    }


def build_fixtures(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    fixtures: dict[str, Any] = {
        f"{LEDGERS[kind][0]}.json": record_schema(kind, fields, clock_id="session.clock")
        for kind, fields in tables.items()
    }
    fixtures["manifest-extension.full-coverage.json"] = _manifest_extension(coverage="full")
    fixtures["manifest-extension.partial-coverage.json"] = _manifest_extension(coverage="partial")
    fixtures["journal-termination-extension.json"] = {
        "extension_version": EXTENSION_VERSION,
        "plan_fingerprint": fixtures["manifest-extension.full-coverage.json"]["plan_fingerprint"],
        "termination_origin": "recorder",
        "requested_terminal_intent": "normal",
        "capture_outcome": "normal",
        "effective_session_outcome": "normal",
        "finalization_status": "succeeded",
        "primary_fault_row_committed": True,
    }
    return fixtures


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when a generated file differs",
    )
    arguments = parser.parse_args(argv)

    tables = parse_field_tables(README.read_text(encoding="utf-8"))
    outputs = {SCHEMA_PATH: _dump(build_schema(tables))}
    for name, value in build_fixtures(tables).items():
        outputs[FIXTURE_DIR / name] = _dump(value)

    differences = []
    for path, text in outputs.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if arguments.check:
            differences.append(path.relative_to(SPEC_DIR).as_posix())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if differences:
        print("stale generated files: " + ", ".join(sorted(differences)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
