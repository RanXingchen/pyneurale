#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Executable reference for NRF v1 manifest semantic validation."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

SEMANTIC_VALIDATION_VERSION = "1.0"

ID = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
TX_ID = re.compile(r"^tx-[0-9]{16}$")
CHECKPOINT_ID = re.compile(r"^checkpoint-[0-9]{16}$")
# Strict RFC 3339 UTC date-time with a mandatory trailing Z and no offset. The
# schema declares this pattern structurally and ``format: "date-time`` for
# range validation; this recheck makes the executable reference validator
# reject a malformed timestamp independently of any format checker.
UTC_DATE_TIME = re.compile(
    r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
    r"T([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d+)?Z$"
)
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _is_valid_utc_date_time(value: str) -> bool:
    """Full RFC 3339 UTC validation: shape plus calendar/clock ranges.

    The schema pattern enforces shape, and ``format: "date-time`` enforces
    ranges only when a format checker with an RFC 3339 backend is available
    (an optional dependency a conforming validator need not install). This
    routine validates ranges directly so the executable reference validator
    rejects values such as ``2026-13-30T09:00:00Z`` on its own. NRF v1 does
    not accept leap-second timestamps, so seconds are limited to 00 through 59.
    """
    if UTC_DATE_TIME.fullmatch(value) is None:
        return False
    year = int(value[0:4])
    month = int(value[5:7])
    day = int(value[8:10])
    hour = int(value[11:13])
    minute = int(value[14:16])
    second = int(value[17:19])
    if not 1 <= year <= 9999:
        return False
    if not 1 <= month <= 12:
        return False
    days = _DAYS_IN_MONTH[month - 1]
    if month == 2 and _is_leap_year(year):
        days = 29
    if not 1 <= day <= days:
        return False
    if not 0 <= hour <= 23:
        return False
    if not 0 <= minute <= 59:
        return False
    if not 0 <= second <= 59:
        return False
    return True


DTYPES = {
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "bool",
    "utf8",
}
CODECS = {"bytes", "vlen-utf8", "blosc", "gzip", "zstd", "crc32c", "sharding_indexed"}
PRIMARY_KEY_DTYPES = {"utf8", "int64", "uint64"}
CANONICAL_RELATIVE_PATH = re.compile(r"^[a-z0-9._-]+(?:/[a-z0-9._-]+)*$")
WINDOWS_RESERVED_BASENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
CONTROL_FILES = {"manifest.json", "zarr.json"}
CONTROL_NAMESPACES = {
    ".staging",
    "journal",
    "metadata",
    "recovery",
    "checksums",
    "indexes",
    "feature_sets",
}
PAYLOAD_NAMESPACES = {"streams", "records"}

# RFC 8785 / I-JSON exact integer range. JSON Schema bounds the integers it
# names via $defs/safeInteger, but free-form objects (metadata, extensions,
# Zarr fill_value) can carry an out-of-range integer the schema cannot bound;
# the recursive walk in validate_manifest_semantics enforces the README
# section 7 contract over the whole manifest.
SAFE_INTEGER_MIN = -9007199254740991
SAFE_INTEGER_MAX = 9007199254740991
STREAM_AXES = {
    "neural": ["sample", "channel"],
    "behavioral": ["sample", "channel"],
    "feature": ["observation", "feature"],
    "spike": ["spike", "sample", "channel"],
}


class NrfSemanticValidationError(ValueError):
    """Raised when an NRF v1 cross-object semantic rule is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NrfSemanticValidationError(message)


def _require_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    _require(fields <= value.keys(), f"{label} is missing required fields")


def _require_canonical_relative_path(path: Any, label: str) -> None:
    segments = path.split("/") if isinstance(path, str) else []
    _require(
        isinstance(path, str)
        and CANONICAL_RELATIVE_PATH.fullmatch(path) is not None
        and PurePosixPath(path).as_posix() == path
        and not PurePosixPath(path).is_absolute()
        and not PureWindowsPath(path).drive,
        f"{label} is not a canonical relative POSIX path",
    )
    _require(
        all(
            segment not in {".", ".."}
            and not segment.endswith(".")
            and segment.split(".", maxsplit=1)[0] not in WINDOWS_RESERVED_BASENAMES
            for segment in segments
        ),
        f"{label} is not a portable relative path",
    )


def portable_path_key(path: str) -> tuple[str, ...]:
    """Return the cross-platform comparison key for a validated NRF path."""
    return tuple(segment.rstrip(" .").casefold() for segment in path.split("/"))


def _require_payload_path(path: str, label: str) -> None:
    _require_canonical_relative_path(path, label)
    top_level = path.split("/", maxsplit=1)[0]
    _require(
        path not in CONTROL_FILES
        and top_level not in CONTROL_NAMESPACES
        and top_level in PAYLOAD_NAMESPACES,
        f"{label} occupies a reserved NRF control path",
    )


def _claim_physical_array_path(
    owners: dict[str, str],
    path: str,
    owner: str,
) -> None:
    _require_payload_path(path, "physical array path")
    portable_path = portable_path_key(path)
    for existing_path in owners:
        _require(
            path != existing_path,
            "physical array path has multiple owners",
        )
        existing_portable_path = portable_path_key(existing_path)
        _require(
            portable_path != existing_portable_path,
            "portable physical array path collision",
        )
        _require(
            portable_path[: len(existing_portable_path)] != existing_portable_path
            and existing_portable_path[: len(portable_path)] != portable_path,
            "physical array paths overlap",
        )
    owners[path] = owner


def _check_ijson_domain(value: Any, path: str) -> None:
    """Recursively reject integers outside the I-JSON exact range and
    non-finite floats.

    JSON Schema bounds every integer it names via ``$defs/safeInteger`` and
    its non-negative/positive variants, but free-form objects (``metadata``,
    ``extensions``, and Zarr ``fill_value``) can carry an out-of-range integer
    the schema cannot bound. This walk enforces the README section 7 I-JSON
    contract over the whole manifest, so a value such as
    ``native_id = 9007199254740992`` is rejected even when it hides in an
    extension. ``bool`` is a Python ``int`` subclass but a JSON boolean, so it
    is excluded from the integer check.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        _require(
            SAFE_INTEGER_MIN <= value <= SAFE_INTEGER_MAX,
            f"integer at {path} is outside the I-JSON exact range",
        )
    elif isinstance(value, float):
        _require(math.isfinite(value), f"non-finite number at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_ijson_domain(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_ijson_domain(item, f"{path}[{i}]")


def _ids(values: list[dict[str, Any]], label: str) -> set[str]:
    result = [value.get("id") for value in values]
    _require(
        all(isinstance(value, str) and ID.fullmatch(value) for value in result),
        f"invalid {label} ID",
    )
    _require(len(result) == len(set(result)), f"duplicate {label} ID")
    return set(result)


def _validate_zarr_array(value: dict[str, Any], codec_ids: set[str], axes: list[str]) -> None:
    _require_fields(
        value,
        {"path", "zarr_format", "shape", "chunk_shape", "codec_ids"},
        "Zarr array",
    )
    _require_canonical_relative_path(value["path"], "Zarr array path")
    _require(value["zarr_format"] == 3, "only Zarr v3 is supported")
    _require(len(value["shape"]) == len(axes), "Zarr shape does not match axes")
    _require(
        len(value["chunk_shape"]) == len(axes),
        "Zarr chunk shape does not match axes",
    )
    _require(
        all(isinstance(item, int) and item >= 0 for item in value["shape"]),
        "invalid Zarr shape",
    )
    _require(
        all(isinstance(item, int) and item > 0 for item in value["chunk_shape"]),
        "invalid Zarr chunk shape",
    )
    _require(
        bool(value["codec_ids"]) and set(value["codec_ids"]) <= codec_ids,
        "unknown codec reference",
    )


def _validate_stream_descriptor(
    stream: dict[str, Any],
    *,
    schema_ids: set[str],
    stream_ids: set[str],
    clock_ids: set[str],
    channel_ids: set[str],
    unit_ids: set[str],
    feature_descriptors: dict[str, dict[str, Any]],
    codec_ids: set[str],
    committed_extents: dict[str, int],
) -> None:
    _require(stream["kind"] in STREAM_AXES, "unsupported stream kind")
    expected_axes = STREAM_AXES[stream["kind"]]
    _require(stream["axes"] == expected_axes, "unsupported payload layout")
    _require(stream["dtype"] in DTYPES, "unsupported dtype")
    if stream["dtype"] in {"bool", "uint8", "utf8"}:
        _require(stream["endianness"] == "not_applicable", "invalid endianness")
    else:
        _require(stream["endianness"] in {"little", "big"}, "invalid endianness")
    _require(stream["schema_id"] in schema_ids, "broken schema reference")
    _require(stream["clock_id"] in clock_ids, "broken clock reference")
    _require(
        set(stream["source_stream_ids"]) <= stream_ids,
        "broken source stream reference",
    )
    _require(set(stream["channel_ids"]) <= channel_ids, "broken channel reference")
    _require(set(stream["unit_ids"]) <= unit_ids, "broken unit reference")
    _validate_zarr_array(stream["data"], codec_ids, expected_axes)
    width = stream["data"]["shape"][-1]
    _require(
        len(stream["unit_ids"]) == width,
        "unit count does not match payload width",
    )
    if stream["channel_ids"]:
        _require(
            len(stream["channel_ids"]) == width,
            "channel count does not match payload width",
        )
    extent = stream["committed_extent"]
    _require(
        isinstance(extent, int) and 0 <= extent <= stream["data"]["shape"][0],
        "invalid stream committed extent",
    )
    _require(
        committed_extents.get(stream["data"]["path"], 0) == extent,
        "manifest extent disagrees with stream",
    )

    timing = stream["timing"]
    if timing["mode"] == "regular":
        rate = timing["rate"]
        _require(
            rate["numerator"] > 0 and rate["denominator"] > 0,
            "invalid regular rate",
        )
    else:
        _validate_zarr_array(timing["timestamps"], codec_ids, [expected_axes[0]])
        _require(
            timing["timestamps"]["shape"][0] >= extent,
            "timestamp array is too short",
        )
        _require(
            committed_extents.get(timing["timestamps"]["path"], 0) == extent,
            "timestamp extent disagrees with stream",
        )

    if stream["kind"] == "feature":
        feature_set_id = stream["feature_set_id"]
        _require(
            feature_set_id in feature_descriptors,
            "feature stream lacks a valid descriptor",
        )
        descriptor = feature_descriptors[feature_set_id]
        _require(
            len(descriptor["feature_names"]) == width,
            "feature descriptor width does not match payload width",
        )
        _require(
            descriptor["unit_ids"] == stream["unit_ids"],
            "feature descriptor units do not match stream units",
        )
        _require(
            descriptor["source_stream_id"] in stream["source_stream_ids"],
            "feature descriptor source is not linked by the stream",
        )


def _validate_feature_descriptor(
    descriptor: dict[str, Any],
    stream_ids: set[str],
    unit_ids: set[str],
) -> None:
    names = descriptor["feature_names"]
    _require(
        bool(names) and len(names) == len(set(names)),
        "feature names must be non-empty and unique",
    )
    _require(
        len(names) == len(descriptor["unit_ids"]),
        "feature name/unit count mismatch",
    )
    _require(
        set(descriptor["unit_ids"]) <= unit_ids,
        "broken feature unit reference",
    )
    _require(
        descriptor["source_stream_id"] in stream_ids,
        "broken feature source reference",
    )
    _require(
        descriptor["window_length_ns"] > 0 and descriptor["shift_ns"] > 0,
        "invalid feature window",
    )


# --- native replay extension (neurale.native_replay) ---------------------
#
# The five native replay ledgers and the manifest extension are a minor NRF v1
# extension (specifications/nrf/v1/extensions/native-replay-v1). The core
# semantic validator does not know what the ledgers are *for* -- that is the
# lifecycle contract -- but it does enforce the cross-field rules that make the
# extension a self-consistent statement rather than a claim a session can make
# about itself: the all-or-nothing ledger set, the v1.1 minor version, the
# derived coverage and capabilities, and a plan fingerprint a reader can
# recompute. This reference implementation imports nothing from the package; it
# reconstructs the plan document and fingerprint with the specification's own
# JCS, the same way the package's plan parser does.

NATIVE_REPLAY_NAMESPACE = "neurale.native_replay"
NATIVE_REPLAY_EXTENSION_VERSION = 1
NATIVE_REPLAY_LEDGER_IDS: dict[str, str] = {
    "native_frames": "native-frames-v1",
    "native_signal_blocks": "native-signal-blocks-v1",
    "native_discontinuities": "native-discontinuities-v1",
    "native_signal_gaps": "native-signal-gaps-v1",
    "session_accounting": "session-accounting-v1",
}

#: kind -> ordered (name, dtype, nullable, reference) for each ledger field.
#: An independent transcription of the extension's README section 3 tables; the
#: package's ``_ledgers.py`` is the other. The conformance suite runs both
#: validators over the same broken vectors so the two cannot drift apart.
NATIVE_REPLAY_LEDGER_FIELDS: dict[str, list[tuple[str, str, bool, str | None]]] = {
    "native_frames": [
        ("data_message_ordinal", "uint64", False, None),
        ("native_session_id", "uint64", False, None),
        ("frame_sequence", "uint64", False, None),
        ("frame_ordinal", "uint64", False, None),
        ("host_received_ns", "uint64", False, None),
        ("source_tick", "uint64", True, None),
        ("valid_until_ns", "uint64", True, None),
        ("native_schema_id", "uint32", False, None),
        ("source_clock_domain", "uint32", False, None),
        ("frame_flags", "uint32", False, None),
        ("signal_block_count", "uint32", False, None),
        ("recorded_signal_block_count", "uint32", False, None),
        ("total_payload_byte_count", "uint64", False, None),
        ("first_signal_block_ordinal", "uint64", True, None),
    ],
    "native_signal_blocks": [
        ("signal_block_ordinal", "uint64", False, None),
        ("data_message_ordinal", "uint64", False, None),
        ("frame_ordinal", "uint64", False, None),
        ("block_index_in_frame", "uint32", False, None),
        ("native_signal_id", "uint32", False, None),
        ("stream_id", "utf8", False, "stream"),
        ("sample_idx_start", "uint64", False, None),
        ("last_sample_idx", "uint64", False, None),
        ("n_samples", "uint32", False, None),
        ("device_tick_start", "uint64", False, None),
        ("observation_time_start_ns", "uint64", False, None),
        ("row_offset", "uint64", False, None),
        ("payload_byte_count", "uint64", False, None),
        ("payload_offset", "uint64", False, None),
        ("clock_sync_device_tick_reference", "uint64", False, None),
        ("clock_sync_host_time_reference_ns", "uint64", False, None),
        ("clock_sync_rate_numerator", "uint64", False, None),
        ("clock_sync_rate_denominator", "uint64", False, None),
        ("clock_sync_uncertainty_ns", "uint64", False, None),
        ("clock_sync_clock_domain", "uint32", False, None),
        ("clock_sync_generation", "uint32", False, None),
        ("clock_sync_flags", "uint32", False, None),
    ],
    "native_discontinuities": [
        ("data_message_ordinal", "uint64", False, None),
        ("native_session_id", "uint64", False, None),
        ("previous_frame_sequence", "uint64", False, None),
        ("actual_frame_sequence", "uint64", False, None),
        ("reason", "utf8", False, None),
        ("first_signal_gap_ordinal", "uint64", True, None),
        ("signal_gap_count", "uint32", False, None),
        ("runtime_accepted_host_time_ns", "uint64", False, None),
    ],
    "native_signal_gaps": [
        ("signal_gap_ordinal", "uint64", False, None),
        ("data_message_ordinal", "uint64", False, None),
        ("gap_index_in_message", "uint32", False, None),
        ("native_signal_id", "uint32", False, None),
        ("expected_sample_index", "uint64", False, None),
        ("actual_sample_index", "uint64", False, None),
        ("missing_samples", "uint64", True, None),
        ("expected_device_tick", "uint64", False, None),
        ("actual_device_tick", "uint64", False, None),
        ("reason", "utf8", False, None),
        ("gap_flags", "uint32", False, None),
    ],
    "session_accounting": [
        ("accounting_id", "utf8", False, None),
        ("accounting_origin", "utf8", False, None),
        ("termination_origin", "utf8", False, None),
        ("producer_acceptance_known", "bool", False, None),
        ("runtime_accepted", "uint64", False, None),
        ("recorder_accepted", "uint64", False, None),
        ("spool_committed", "uint64", True, None),
        ("nrf_committed", "uint64", False, None),
        ("rejected_before_runtime_acceptance", "uint64", False, None),
        ("failed_between_runtime_and_recorder", "uint64", False, None),
        ("lost_between_recorder_and_spool", "uint64", True, None),
        ("lost_during_finalization", "uint64", False, None),
        ("control_offered", "uint64", True, None),
        ("control_accepted", "uint64", False, None),
        ("control_spool_committed", "uint64", True, None),
        ("control_nrf_committed", "uint64", False, None),
        ("control_rejected", "uint64", False, None),
        ("lost_between_control_acceptance_and_spool", "uint64", True, None),
        ("control_lost_during_finalization", "uint64", False, None),
        ("data_first_lost_ordinal", "uint64", True, None),
        ("data_first_rejected_message_kind", "utf8", True, None),
        ("data_first_rejected_frame_sequence", "uint64", True, None),
        ("control_first_lost_ordinal", "uint64", True, None),
        ("control_first_rejected_kind", "utf8", True, None),
        ("control_first_rejected_identity", "utf8", True, None),
        ("data_rejected_after_close", "uint64", False, None),
        ("control_rejected_after_close", "uint64", False, None),
    ],
}
#: Fields a native ledger column descriptor carries. ``codec_ids`` is included
#: because the core rules require it to be present and to reference a session
#: codec, but its value is session-scoped (README section 7) and is not compared
#: to a fixed spelling here.
_LEDGER_FIELD_FIXED_KEYS = ("name", "dtype", "endianness", "nullable", "array_path")
_LEDGER_ENDIanness_FREE = frozenset({"bool", "uint8", "utf8"})


def _ledger_expected_fields(kind: str) -> list[tuple[str, str, bool, str | None]]:
    return NATIVE_REPLAY_LEDGER_FIELDS[kind]


_NATIVE_REPLAY_LEDGER_PRIMARY_KEYS: dict[str, str] = {
    "native_frames": "data_message_ordinal",
    "native_signal_blocks": "signal_block_ordinal",
    "native_discontinuities": "data_message_ordinal",
    "native_signal_gaps": "signal_gap_ordinal",
    "session_accounting": "accounting_id",
}


def _validate_native_ledger_descriptor(kind: str, schema: dict[str, Any]) -> None:
    """Bind a native ledger's schema id to its exact field structure."""
    schema_id = NATIVE_REPLAY_LEDGER_IDS[kind]
    path = f"records/{kind}/{schema_id}"
    _require(
        schema.get("id") == schema_id, f"native replay ledger {kind!r} has the wrong schema id"
    )
    _require(schema.get("kind") == kind, f"native replay ledger {kind!r} has the wrong kind")
    _require(schema.get("path") == path, f"native replay ledger {kind!r} has a non-canonical path")
    _require(
        schema.get("primary_key") == _NATIVE_REPLAY_LEDGER_PRIMARY_KEYS[kind],
        f"native replay ledger {kind!r} has the wrong primary key",
    )
    fields = schema.get("fields")
    _require(isinstance(fields, list), f"native replay ledger {kind!r} fields must be an array")
    expected = _ledger_expected_fields(kind)
    _require(
        len(fields) == len(expected),
        f"native replay ledger {kind!r} has {len(fields)} fields, expected {len(expected)}",
    )
    for i, (actual, (name, dtype, nullable, reference)) in enumerate(
        zip(fields, expected, strict=True)
    ):
        label = f"native replay ledger {kind!r} field {i} ({name!r})"
        _require(actual.get("name") == name, f"{label} has the wrong name")
        _require(actual.get("dtype") == dtype, f"{label} has the wrong dtype")
        endianness = "not_applicable" if dtype in _LEDGER_ENDIanness_FREE else "little"
        _require(actual.get("endianness") == endianness, f"{label} has the wrong endianness")
        _require(actual.get("nullable") == nullable, f"{label} has the wrong nullability")
        _require(
            actual.get("array_path") == f"{path}/columns/{name}",
            f"{label} has the wrong array path",
        )
        if nullable:
            _require(
                actual.get("validity_path") == f"{path}/validity/{name}",
                f"{label} is missing its validity path",
            )
        else:
            _require("validity_path" not in actual, f"{label} must not carry a validity path")
        if reference is not None:
            _require(actual.get("reference") == reference, f"{label} has the wrong reference")
        else:
            _require("reference" not in actual, f"{label} must not carry a reference")
        allowed = set(_LEDGER_FIELD_FIXED_KEYS) | {"codec_ids"}
        if nullable:
            allowed = allowed | {"validity_path"}
        if reference is not None:
            allowed = allowed | {"reference"}
        _require(set(actual) <= allowed, f"{label} carries an unexpected key")


_NATIVE_REPLAY_SIGNAL_FIELDS = (
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
)
_NATIVE_REPLAY_FEATURE_SET_FIELDS = (
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
)
_NATIVE_REPLAY_UNIT_FIELDS = ("id", "symbol", "description")
_NATIVE_REPLAY_SESSION_FIELDS = ("session_id", "created_at", "writer_name", "writer_version")
_NATIVE_REPLAY_RESOURCE_BOUNDS_FIELDS = (
    "frame_queue_capacity",
    "control_queue_capacity",
    "control_chunk_length",
    "checkpoint_interval",
    "overflow_policy",
    "streams",
)


def _load_jcs():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "jcs.py"
    spec = importlib.util.spec_from_file_location("_nrf_v1_jcs_for_native_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_JCS = _load_jcs()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _native_replay_rebuild_signal(entry: Any) -> dict[str, Any]:
    _require(isinstance(entry, dict), "a native schema signal must be an object")
    rebuilt = {}
    for field in _NATIVE_REPLAY_SIGNAL_FIELDS:
        _require(field in entry, f"native schema signal is missing {field!r}")
        value = entry[field]
        if field == "fs":
            _require(
                isinstance(value, dict)
                and _is_int(value.get("numerator"))
                and _is_int(value.get("denominator"))
                and value["denominator"] > 0,
                "signal fs must have numerator and a positive denominator",
            )
            rebuilt[field] = {"numerator": value["numerator"], "denominator": value["denominator"]}
        elif field in {
            "id",
            "clock_domain",
            "n_channels",
            "nominal_block_samples",
            "max_block_samples",
            "channel_set_id",
            "calibration_id",
            "reference_id",
            "feature_set_id",
            "fixed_block_bytes",
            "max_block_bytes",
        }:
            _require(
                _is_int(value) and value >= 0, f"signal {field!r} must be a non-negative integer"
            )
            rebuilt[field] = value
        else:
            _require(
                isinstance(value, str) and value, f"signal {field!r} must be a non-empty string"
            )
            rebuilt[field] = value
    return rebuilt


def _native_replay_rebuild_feature_set(entry: Any) -> dict[str, Any]:
    _require(isinstance(entry, dict), "a native schema feature set must be an object")
    rebuilt = {}
    for field in _NATIVE_REPLAY_FEATURE_SET_FIELDS:
        _require(field in entry, f"native schema feature set is missing {field!r}")
        value = entry[field]
        if field == "feature_names":
            _require(isinstance(value, list) and value, "feature_names must be a non-empty array")
            rebuilt[field] = list(value)
        elif field == "unit_ids":
            _require(isinstance(value, list), "unit_ids must be an array")
            rebuilt[field] = list(value)
        elif field in {"id", "source_stream_id", "window_length_ns", "shift_ns"}:
            _require(
                _is_int(value) and value >= 0,
                f"feature set {field!r} must be a non-negative integer",
            )
            rebuilt[field] = value
        else:
            _require(isinstance(value, str), f"feature set {field!r} must be a string")
            rebuilt[field] = value
    return rebuilt


def _native_replay_rebuild_unit(entry: Any) -> dict[str, Any]:
    _require(isinstance(entry, dict), "a native schema unit must be an object")
    rebuilt = {}
    for field in _NATIVE_REPLAY_UNIT_FIELDS:
        _require(field in entry, f"native schema unit is missing {field!r}")
        value = entry[field]
        if field == "id":
            _require(_is_int(value) and value >= 0, "unit id must be a non-negative integer")
        else:
            _require(isinstance(value, str), f"unit {field!r} must be a string")
        rebuilt[field] = value
    return rebuilt


def _native_replay_rebuild_session(entry: Any) -> None:
    _require(isinstance(entry, dict), "extension session must be an object")
    for field in _NATIVE_REPLAY_SESSION_FIELDS:
        _require(field in entry, f"extension session is missing {field!r}")
        _require(isinstance(entry[field], str), f"extension session.{field} must be a string")


def _native_replay_rebuild_resource_bounds(entry: Any) -> None:
    _require(isinstance(entry, dict), "extension resource_bounds must be an object")
    for field in _NATIVE_REPLAY_RESOURCE_BOUNDS_FIELDS:
        _require(field in entry, f"extension resource_bounds is missing {field!r}")
    for field in ("frame_queue_capacity", "control_queue_capacity", "control_chunk_length"):
        _require(
            _is_int(entry[field]) and entry[field] >= 1,
            f"resource_bounds.{field} must be a positive integer",
        )
    _require(
        _is_int(entry["checkpoint_interval"]) and entry["checkpoint_interval"] >= 0,
        "resource_bounds.checkpoint_interval must be a non-negative integer",
    )
    _require(
        entry["overflow_policy"] in {"fault", "continue"},
        "resource_bounds.overflow_policy must be fault or continue",
    )
    _require(isinstance(entry["streams"], list), "resource_bounds.streams must be an array")
    for stream in entry["streams"]:
        _require(isinstance(stream, dict), "a resource_bounds stream must be an object")
        for field in ("stream_id", "capacity", "chunk_length", "block_index_chunk_length"):
            _require(field in stream, f"resource_bounds stream is missing {field!r}")
        _require(isinstance(stream["stream_id"], str), "resource_bounds stream_id must be a string")
        for field in ("capacity", "chunk_length", "block_index_chunk_length"):
            _require(
                _is_int(stream[field]) and stream[field] >= 1,
                f"resource_bounds stream.{field} must be a positive integer",
            )


_NATIVE_SIGNAL_DTYPES = frozenset({"INT16", "INT32", "FLOAT32", "FLOAT64"})
_NATIVE_SIGNAL_LAYOUTS = frozenset({"SAMPLE_MAJOR", "CHANNEL_MAJOR"})
_NATIVE_DEVICE_TICK_TRACKING = frozenset({"UNAVAILABLE", "SAMPLE_COUNTER"})
_NATIVE_SIGNAL_KINDS = frozenset({"SAMPLED", "EVENT", "FEATURE", "SPIKE"})
_NATIVE_PHYSICAL_UNITS = frozenset({"UNSPECIFIED", "VOLTS", "AMPERES", "DIMENSIONLESS"})
_NATIVE_OBSERVATION_TIMINGS = frozenset({"NOT_APPLICABLE", "REGULAR"})
_NATIVE_TIMESTAMP_REFERENCES = frozenset({"WINDOW_CENTER"})
_NATIVE_DTYPE_BYTES = {"INT16": 2, "INT32": 4, "FLOAT32": 4, "FLOAT64": 8}
_NANOSECONDS = 1_000_000_000


def _validate_native_schema_invariants(
    signals: list[dict[str, Any]],
    feature_sets: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> None:
    """The StreamSchema construction invariants, checked on the rebuilt schema.

    A fingerprint only proves an extension was not modified; it does not prove
    the extension describes a schema the runtime could have constructed. These
    are the same invariants the package parser enforces, independently
    implemented here so the specification reference and the package cannot
    drift apart on what a legal native schema is.
    """
    _require(len(signals) > 0, "native_schema must declare at least one signal")
    signal_ids = [signal["id"] for signal in signals]
    _require(len(set(signal_ids)) == len(signal_ids), "native_schema has duplicate signal ids")
    feature_sets_by_id: dict[int, dict[str, Any]] = {}
    for feature_set in feature_sets:
        _require(
            feature_set["id"] not in feature_sets_by_id,
            f"native_schema has duplicate feature-set id {feature_set['id']}",
        )
        feature_sets_by_id[feature_set["id"]] = feature_set
    unit_id_set: set[int] = set()
    unit_symbol_set: set[str] = set()
    for unit in units:
        _require(unit["id"] > 0, f"unit id {unit['id']} must be nonzero")
        _require(unit["symbol"], f"unit {unit['id']} symbol must be non-empty")
        _require(unit["id"] not in unit_id_set, f"native_schema has duplicate unit id {unit['id']}")
        _require(
            unit["symbol"] not in unit_symbol_set,
            f"native_schema has duplicate unit symbol {unit['symbol']!r}",
        )
        unit_id_set.add(unit["id"])
        unit_symbol_set.add(unit["symbol"])
    for feature_set in feature_sets:
        fid = feature_set["id"]
        _require(fid > 0, f"feature set {fid} id must be nonzero")
        _require(feature_set["feature_names"], f"feature set {fid} must name at least one feature")
        _require(
            len(set(feature_set["feature_names"])) == len(feature_set["feature_names"]),
            f"feature set {fid} feature names must be unique",
        )
        _require(
            len(feature_set["feature_names"]) == len(feature_set["unit_ids"]),
            f"feature set {fid} feature names and unit ids must have equal length",
        )
        for name in feature_set["feature_names"]:
            _require(name, f"feature set {fid} has an empty feature name")
        for unit_id in feature_set["unit_ids"]:
            _require(unit_id > 0, f"feature set {fid} has a zero unit id")
            _require(
                unit_id in unit_id_set, f"feature set {fid} references a missing unit id {unit_id}"
            )
        _require(
            feature_set["source_stream_id"] > 0,
            f"feature set {fid} source_stream_id must be nonzero",
        )
        _require(feature_set["source_stream"], f"feature set {fid} source_stream must be non-empty")
        _require(
            feature_set["window_length_ns"] > 0 and feature_set["shift_ns"] > 0,
            f"feature set {fid} window length and shift must be positive",
        )
        _require(
            feature_set["timestamp_reference"] in _NATIVE_TIMESTAMP_REFERENCES,
            f"feature set {fid} timestamp_reference is not supported",
        )
    for signal in signals:
        _validate_signal_invariants(signal, feature_sets_by_id)


def _validate_signal_invariants(
    signal: dict[str, Any], feature_sets_by_id: dict[int, dict[str, Any]]
) -> None:
    label = f"signal {signal['id']}"
    _require(
        signal["dtype"] in _NATIVE_SIGNAL_DTYPES,
        f"{label} dtype {signal['dtype']!r} is not supported",
    )
    _require(
        signal["layout"] in _NATIVE_SIGNAL_LAYOUTS,
        f"{label} layout {signal['layout']!r} is not supported",
    )
    _require(
        signal["device_tick_tracking"] in _NATIVE_DEVICE_TICK_TRACKING,
        f"{label} device_tick_tracking {signal['device_tick_tracking']!r} is not supported",
    )
    _require(
        signal["kind"] in _NATIVE_SIGNAL_KINDS, f"{label} kind {signal['kind']!r} is not supported"
    )
    _require(
        signal["physical_unit"] in _NATIVE_PHYSICAL_UNITS,
        f"{label} physical_unit {signal['physical_unit']!r} is not supported",
    )
    _require(
        signal["observation_timing"] in _NATIVE_OBSERVATION_TIMINGS,
        f"{label} observation_timing {signal['observation_timing']!r} is not supported",
    )
    _require(signal["n_channels"] > 0, f"{label} channel count must be positive")
    _require(signal["nominal_block_samples"] > 0, f"{label} nominal block samples must be positive")
    _require(signal["max_block_samples"] > 0, f"{label} max block samples must be positive")
    _require(
        signal["nominal_block_samples"] <= signal["max_block_samples"],
        f"{label} nominal block samples exceed the maximum",
    )
    numerator = signal["fs"]["numerator"]
    denominator = signal["fs"]["denominator"]
    _require(numerator >= 0, f"{label} sample rate numerator must be non-negative")
    _require(denominator > 0, f"{label} sample rate denominator must be positive")
    kind = signal["kind"]
    if kind == "SAMPLED":
        _require(numerator > 0, f"{label} sampled signal sample rate must be positive")
        _require(
            signal["feature_set_id"] == 0,
            f"{label} sampled signal cannot reference feature metadata",
        )
        _require(
            signal["observation_timing"] == "NOT_APPLICABLE",
            f"{label} sampled signal observation timing must be not_applicable",
        )
    elif kind == "EVENT":
        _require(numerator == 0 and denominator == 1, f"{label} event signal requires a zero rate")
        _require(
            signal["nominal_block_samples"] == 1 and signal["max_block_samples"] == 1,
            f"{label} event signal requires one event per block",
        )
        _require(
            signal["feature_set_id"] == 0, f"{label} event signal cannot reference feature metadata"
        )
        _require(
            signal["observation_timing"] == "NOT_APPLICABLE",
            f"{label} event signal observation timing must be not_applicable",
        )
    elif kind == "FEATURE":
        _require(
            signal["feature_set_id"] != 0,
            f"{label} feature signal requires a feature-set descriptor id",
        )
        _require(
            signal["observation_timing"] == "REGULAR",
            f"{label} feature signal requires regular observation timing",
        )
        _require(numerator > 0, f"{label} feature signal observation rate must be positive")
        _require(
            signal["layout"] == "SAMPLE_MAJOR",
            f"{label} feature signal must use sample-major layout",
        )
        _require(
            signal["physical_unit"] == "UNSPECIFIED",
            f"{label} feature signal physical unit must be unspecified",
        )
    elif kind == "SPIKE":
        _require(numerator == 0 and denominator == 1, f"{label} spike signal requires a zero rate")
        _require(
            signal["feature_set_id"] == 0, f"{label} spike signal cannot reference feature metadata"
        )
        _require(
            signal["observation_timing"] == "NOT_APPLICABLE",
            f"{label} spike signal observation timing must be not_applicable",
        )
        _require(
            signal["fixed_block_bytes"] > 0,
            f"{label} spike signal requires a fixed sparse-block payload size",
        )
    if kind == "SPIKE":
        expected_bytes = signal["fixed_block_bytes"]
    else:
        expected_bytes = (
            signal["n_channels"]
            * signal["max_block_samples"]
            * _NATIVE_DTYPE_BYTES[signal["dtype"]]
        )
    _require(
        signal["max_block_bytes"] == expected_bytes,
        f"{label} max_block_bytes {signal['max_block_bytes']} does not match the dtype and block limits ({expected_bytes})",
    )
    if kind == "FEATURE":
        descriptor = feature_sets_by_id.get(signal["feature_set_id"])
        _require(
            descriptor is not None,
            f"{label} references a missing feature-set descriptor {signal['feature_set_id']}",
        )
        _require(
            len(descriptor["feature_names"]) == signal["n_channels"],
            f"{label} feature count does not match its descriptor",
        )
        _require(
            numerator * descriptor["shift_ns"] == denominator * _NANOSECONDS,
            f"{label} feature observation rate must equal the descriptor shift",
        )


def _cross_validate_extension_vs_manifest(
    manifest: dict[str, Any], raw_streams: list[dict[str, Any]], extension: dict[str, Any]
) -> None:
    """The extension must agree with the core manifest it rides on."""
    session = extension["session"]
    manifest_session = manifest["session"]
    _require(
        session["session_id"] == manifest_session["id"],
        "extension session_id does not match the manifest session",
    )
    _require(
        session["created_at"] == manifest_session["created_at"],
        "extension session created_at does not match the manifest session",
    )

    units_by_id = {unit["id"]: unit for unit in manifest["units"]}
    channels_by_id = {channel["id"]: channel for channel in manifest["channels"]}
    feature_sets_by_id = {fs["id"]: fs for fs in manifest["feature_sets"]}
    streams_by_id = {stream["id"]: stream for stream in manifest["streams"]}
    data_stream_ids = {stream["id"] for stream in manifest["streams"] if "native_id" in stream}

    _require(
        {entry["stream_id"] for entry in raw_streams} == data_stream_ids,
        "extension stream ids do not match the manifest's recorded data streams",
    )
    for entry in raw_streams:
        stream = streams_by_id[entry["stream_id"]]
        _require(
            entry["native_signal_id"] == stream["native_id"],
            f"extension stream {entry['stream_id']!r} native_signal_id does not match the manifest stream",
        )
        _require(
            entry["kind"] == stream["kind"],
            f"extension stream {entry['stream_id']!r} kind does not match the manifest stream",
        )
        _require(
            entry["timing"] == stream["timing"]["mode"],
            f"extension stream {entry['stream_id']!r} timing does not match the manifest stream",
        )
        _require(
            entry["name"] == stream["name"],
            f"extension stream {entry['stream_id']!r} name does not match the manifest stream",
        )
        expected_unit = [units_by_id[unit_id]["symbol"] for unit_id in stream["unit_ids"]]
        _require(
            entry["unit"] == expected_unit,
            f"extension stream {entry['stream_id']!r} unit does not match the manifest stream",
        )
        if stream["kind"] == "feature":
            feature_set = feature_sets_by_id[stream["feature_set_id"]]
            _require(
                entry["channel_names"] == [],
                f"extension feature stream {entry['stream_id']!r} must not name channels",
            )
            _require(
                entry["feature_names"] == feature_set["feature_names"],
                f"extension stream {entry['stream_id']!r} feature names do not match the manifest feature set",
            )
            _require(
                entry["algorithm_name"] == feature_set["algorithm"]["name"],
                f"extension stream {entry['stream_id']!r} algorithm name does not match the manifest feature set",
            )
            _require(
                entry["algorithm_version"] == feature_set["algorithm"]["version"],
                f"extension stream {entry['stream_id']!r} algorithm version does not match the manifest feature set",
            )
            _require(
                entry["window_length_ns"] == feature_set["window_length_ns"],
                f"extension stream {entry['stream_id']!r} window length does not match the manifest feature set",
            )
            _require(
                entry["shift_ns"] == feature_set["shift_ns"],
                f"extension stream {entry['stream_id']!r} shift does not match the manifest feature set",
            )
        else:
            expected_channels = [channels_by_id[cid]["name"] for cid in stream["channel_ids"]]
            _require(
                entry["channel_names"] == expected_channels,
                f"extension stream {entry['stream_id']!r} channel names do not match the manifest stream",
            )
            _require(
                entry["feature_names"] == [],
                f"extension non-feature stream {entry['stream_id']!r} must not name features",
            )

    _cross_validate_native_schema_vs_manifest(
        manifest, raw_streams, streams_by_id, feature_sets_by_id
    )

    session_metadata = {
        key: value for key, value in manifest_session.items() if key not in {"id", "created_at"}
    }
    _require(
        extension["session_metadata"] == session_metadata,
        "extension session_metadata does not match the manifest session metadata",
    )
    _require(
        extension["metadata"] == manifest.get("metadata", {}),
        "extension metadata does not match the manifest metadata",
    )


def _cross_validate_native_schema_vs_manifest(
    manifest: dict[str, Any],
    raw_streams: list[dict[str, Any]],
    streams_by_id: dict[str, dict[str, Any]],
    feature_sets_by_id: dict[str, dict[str, Any]],
) -> None:
    """The extension's native schema must match the manifest's data description."""
    native_signals = {
        sig["id"]: sig
        for sig in manifest["extensions"][NATIVE_REPLAY_NAMESPACE]["native_schema"]["signals"]
    }
    native_feature_sets = {
        fs["id"]: fs
        for fs in manifest["extensions"][NATIVE_REPLAY_NAMESPACE]["native_schema"]["feature_sets"]
    }
    signal_to_stream = {entry["native_signal_id"]: entry["stream_id"] for entry in raw_streams}
    manifest_stream_id_set = {s["id"] for s in manifest["streams"]}
    for entry in raw_streams:
        stream = streams_by_id[entry["stream_id"]]
        signal = native_signals[entry["native_signal_id"]]
        _require(
            _NATIVE_TO_NRF_DTYPE.get(signal["dtype"]) == stream["dtype"],
            f"extension native signal {entry['native_signal_id']} dtype does not match manifest stream {entry['stream_id']!r}",
        )
        _require(
            signal["n_channels"] == stream["data"]["shape"][-1],
            f"extension native signal {entry['native_signal_id']} channel count does not match manifest stream {entry['stream_id']!r}",
        )
        _require(
            signal["clock_domain"] == _clock_native_id(manifest, stream["clock_id"]),
            f"extension native signal {entry['native_signal_id']} clock domain does not match manifest stream {entry['stream_id']!r}",
        )
        timing = stream["timing"]
        if timing["mode"] == "regular":
            _require(
                signal["fs"]["numerator"] == timing["rate"]["numerator"]
                and signal["fs"]["denominator"] == timing["rate"]["denominator"],
                f"extension native signal {entry['native_signal_id']} sample rate does not match manifest stream {entry['stream_id']!r}",
            )
            _require(
                entry["segment_start_index"] == timing.get("segment_start_index", 0),
                f"extension stream {entry['stream_id']!r} segment start index does not match the manifest stream",
            )
            _require(
                entry["segment_start_time_ns"] == timing.get("segment_start_time_ns", 0),
                f"extension stream {entry['stream_id']!r} segment start time does not match the manifest stream",
            )
        _require(
            entry["source_stream_ids"] == list(stream["source_stream_ids"]),
            f"extension stream {entry['stream_id']!r} source stream ids do not match the manifest stream",
        )
        _require(
            entry["block_index"] == (f"{entry['stream_id']}.blocks" in manifest_stream_id_set),
            f"extension stream {entry['stream_id']!r} block index declaration does not match the manifest",
        )
        if stream["kind"] == "feature":
            manifest_fs = feature_sets_by_id[stream["feature_set_id"]]
            native_fs = native_feature_sets[signal["feature_set_id"]]
            _require(
                signal_to_stream.get(native_fs["source_stream_id"])
                == manifest_fs["source_stream_id"],
                f"extension native feature set source does not match manifest feature set for stream {entry['stream_id']!r}",
            )


def _clock_native_id(manifest: dict[str, Any], clock_id: str) -> Any:
    for clock in manifest["clocks"]:
        if clock["id"] == clock_id:
            return clock.get("native_id", 0)
    return 0


_NATIVE_TO_NRF_DTYPE = {
    "INT16": "int16",
    "INT32": "int32",
    "FLOAT32": "float32",
    "FLOAT64": "float64",
}

_NATIVE_PREPARED_KIND_COMPATIBILITY = {
    "SAMPLED": frozenset({"neural", "behavioral"}),
    "FEATURE": frozenset({"feature"}),
}


def _validate_prepared_streams_against_native_schema(
    signals: list[dict[str, Any]],
    feature_sets: list[dict[str, Any]],
    streams: list[dict[str, Any]],
) -> None:
    """A prepared stream mapping must be consistent with the native schema.

    Independent implementation of the same acceptability rules the package
    parser and compiler share, so the specification reference and the package
    cannot drift on what a consistent prepared plan is.
    """
    signals_by_id = {signal["id"]: signal for signal in signals}
    feature_sets_by_id = {fs["id"]: fs for fs in feature_sets}
    signal_to_stream = {stream["native_signal_id"]: stream["stream_id"] for stream in streams}
    recorded_stream_ids = {stream["stream_id"] for stream in streams}
    for stream in streams:
        # Every declared source, on a stream of any kind, must resolve to a
        # stream this plan records.
        for source_stream_id in stream["source_stream_ids"]:
            _require(
                source_stream_id in recorded_stream_ids,
                f"stream {stream['stream_id']!r} references source stream "
                f"{source_stream_id!r}, which this plan does not record",
            )
        signal = signals_by_id.get(stream["native_signal_id"])
        _require(
            signal is not None,
            f"stream {stream['stream_id']!r} records native signal {stream['native_signal_id']}, which the schema does not declare",
        )
        native_kind = signal["kind"]
        allowed = _NATIVE_PREPARED_KIND_COMPATIBILITY.get(native_kind)
        _require(
            allowed is not None,
            f"stream {stream['stream_id']!r} records a {native_kind.lower()!r} signal, which this recorder does not write",
        )
        _require(
            stream["kind"] in allowed,
            f"stream {stream['stream_id']!r} is {stream['kind']!r} but its native signal is {native_kind.lower()!r}; expected {sorted(allowed)}",
        )
        _require(
            len(stream["unit"]) == signal["n_channels"],
            f"stream {stream['stream_id']!r} unit count does not match its channel count",
        )
        if stream["kind"] == "feature":
            _require(
                len(stream["feature_names"]) == signal["n_channels"],
                f"stream {stream['stream_id']!r} feature name count does not match its channel count",
            )
            _require(
                stream["channel_names"] == [],
                f"feature stream {stream['stream_id']!r} must not name channels",
            )
            descriptor = feature_sets_by_id.get(signal["feature_set_id"])
            _require(
                descriptor is not None,
                f"stream {stream['stream_id']!r} references a missing feature-set descriptor",
            )
            _require(
                stream["algorithm_name"]
                == (descriptor["algorithm_name"] or stream["algorithm_name"]),
                f"stream {stream['stream_id']!r} algorithm name does not match its descriptor",
            )
            _require(
                stream["algorithm_version"]
                == (descriptor["algorithm_version"] or stream["algorithm_version"]),
                f"stream {stream['stream_id']!r} algorithm version does not match its descriptor",
            )
            _require(
                stream["window_length_ns"] == descriptor["window_length_ns"],
                f"stream {stream['stream_id']!r} window length does not match its descriptor",
            )
            _require(
                stream["shift_ns"] == descriptor["shift_ns"],
                f"stream {stream['stream_id']!r} shift does not match its descriptor",
            )
            source_stream = signal_to_stream.get(descriptor["source_stream_id"])
            _require(
                source_stream is not None,
                f"stream {stream['stream_id']!r} descriptor source signal {descriptor['source_stream_id']} is not recorded",
            )
            _require(
                source_stream in stream["source_stream_ids"],
                f"stream {stream['stream_id']!r} source linkage does not include its descriptor source {source_stream!r}",
            )
        else:
            _require(
                len(stream["channel_names"]) == signal["n_channels"],
                f"stream {stream['stream_id']!r} channel name count does not match its channel count",
            )
            _require(
                stream["feature_names"] == [],
                f"non-feature stream {stream['stream_id']!r} must not name features",
            )
        if stream["timing"] == "explicit":
            _require(
                stream["segment_start_index"] == 0,
                f"explicit stream {stream['stream_id']!r} segment start index must be zero",
            )
            _require(
                stream["segment_start_time_ns"] == 0,
                f"explicit stream {stream['stream_id']!r} segment start time must be zero",
            )


def validate_native_replay_extension(manifest: dict[str, Any]) -> None:
    """Validate the neurale.native_replay cross-field rules.

    Called from :func:`validate_manifest_semantics` after the core rules. A
    session that declares none of the extension's record kinds and no extension
    object is untouched. A session that declares any of them must declare all
    five ledgers, the extension object, and minor version 1, and the extension
    must be internally consistent with a recomputable plan fingerprint.
    """
    record_schemas = manifest["record_schemas"]
    present_kinds = {schema["kind"] for schema in record_schemas}
    ledger_kinds = present_kinds & set(NATIVE_REPLAY_LEDGER_IDS)
    extensions = manifest.get("extensions") or {}
    extension = extensions.get(NATIVE_REPLAY_NAMESPACE)
    declared = bool(ledger_kinds) or extension is not None

    if not declared:
        return

    # All five ledgers or none, each with the exact schema id and path the
    # extension fixes. The core path rule already checks records/<kind>/<id>.
    _require(
        set(NATIVE_REPLAY_LEDGER_IDS) <= ledger_kinds,
        "the native replay ledgers must be declared all together or not at all",
    )
    by_kind: dict[str, dict[str, Any]] = {}
    for schema in record_schemas:
        if schema["kind"] in NATIVE_REPLAY_LEDGER_IDS:
            _require(
                schema["kind"] not in by_kind,
                f"native replay ledger kind {schema['kind']!r} is declared more than once",
            )
            by_kind[schema["kind"]] = schema
    for kind in NATIVE_REPLAY_LEDGER_IDS:
        _require(kind in by_kind, f"native replay ledger {kind!r} is missing")
        _validate_native_ledger_descriptor(kind, by_kind[kind])

    _require(
        manifest["version"]["minor"] == 1, "a session declaring native replay must be NRF v1.1"
    )
    _require(isinstance(extension, dict), "the neurale.native_replay extension must be an object")

    _require(
        extension.get("extension_version") == NATIVE_REPLAY_EXTENSION_VERSION,
        "unsupported native replay extension_version",
    )
    _require(
        isinstance(extension.get("plan_fingerprint"), str)
        and re.fullmatch(r"[0-9a-f]{64}", extension["plan_fingerprint"]) is not None,
        "plan_fingerprint must be 64 lower-case hex characters",
    )

    raw_schema = extension.get("native_schema")
    _require(isinstance(raw_schema, dict), "native_schema must be an object")
    _require(
        _is_int(raw_schema.get("schema_id")) and raw_schema["schema_id"] >= 0,
        "native_schema.schema_id must be a non-negative integer",
    )
    raw_signals = raw_schema.get("signals")
    _require(
        isinstance(raw_signals, list) and raw_signals,
        "native_schema.signals must be a non-empty array",
    )
    signals = [_native_replay_rebuild_signal(entry) for entry in raw_signals]
    raw_feature_sets = raw_schema.get("feature_sets")
    _require(isinstance(raw_feature_sets, list), "native_schema.feature_sets must be an array")
    feature_sets = [_native_replay_rebuild_feature_set(entry) for entry in raw_feature_sets]
    raw_units = raw_schema.get("units")
    _require(isinstance(raw_units, list), "native_schema.units must be an array")
    units = [_native_replay_rebuild_unit(entry) for entry in raw_units]
    _validate_native_schema_invariants(signals, feature_sets, units)
    signal_ids = [signal["id"] for signal in signals]

    planned = extension.get("planned_signal_ids")
    recorded = extension.get("recorded_signal_ids")
    _require(isinstance(planned, list), "planned_signal_ids must be an array")
    _require(isinstance(recorded, list), "recorded_signal_ids must be an array")
    _require(
        all(_is_int(value) and value >= 0 for value in planned),
        "planned_signal_ids must be non-negative integers",
    )
    _require(
        all(_is_int(value) and value >= 0 for value in recorded),
        "recorded_signal_ids must be non-negative integers",
    )
    _require(
        planned == sorted(planned) and len(set(planned)) == len(planned),
        "planned_signal_ids must be sorted and unique",
    )
    _require(
        recorded == sorted(recorded) and len(set(recorded)) == len(recorded),
        "recorded_signal_ids must be sorted and unique",
    )
    _require(
        set(recorded) <= set(planned), "recorded_signal_ids must be a subset of planned_signal_ids"
    )
    _require(
        set(planned) == set(signal_ids),
        "planned_signal_ids must equal the native schema signal ids",
    )

    raw_streams = extension.get("streams")
    _require(isinstance(raw_streams, list), "extension streams must be an array")
    stream_signal_ids = []
    stream_ids = []
    manifest_stream_ids = {stream["id"] for stream in manifest["streams"]}
    _STREAM_REQUIRED_STR_ARRAYS = ("unit", "channel_names", "feature_names", "source_stream_ids")
    for entry in raw_streams:
        _require(isinstance(entry, dict), "an extension stream must be an object")
        for field in (
            "native_signal_id",
            "stream_id",
            "kind",
            "timing",
            "block_index",
            "name",
            "segment_start_index",
            "segment_start_time_ns",
            "algorithm_name",
            "algorithm_version",
            "window_length_ns",
            "shift_ns",
        ):
            _require(field in entry, f"extension stream is missing {field!r}")
        for field in _STREAM_REQUIRED_STR_ARRAYS:
            _require(
                isinstance(entry.get(field), list), f"extension stream.{field} must be an array"
            )
        _require(
            _is_int(entry["native_signal_id"]) and entry["native_signal_id"] >= 0,
            "stream native_signal_id must be a non-negative integer",
        )
        _require(isinstance(entry["stream_id"], str), "stream stream_id must be a string")
        _require(
            entry["stream_id"] in manifest_stream_ids,
            "extension stream_id does not resolve to a manifest stream",
        )
        _require(isinstance(entry["block_index"], bool), "stream block_index must be a boolean")
        _require(isinstance(entry["name"], str), "stream name must be a string")
        _require(
            entry["kind"] in ("neural", "behavioral", "feature", "spike"),
            "stream kind is not supported",
        )
        _require(entry["timing"] in ("explicit", "regular"), "stream timing is not supported")
        _require(
            _is_int(entry["segment_start_index"]) and entry["segment_start_index"] >= 0,
            "stream segment_start_index must be a non-negative integer",
        )
        _require(
            _is_int(entry["segment_start_time_ns"]),
            "stream segment_start_time_ns must be an integer",
        )
        _require(isinstance(entry["algorithm_name"], str), "stream algorithm_name must be a string")
        _require(
            isinstance(entry["algorithm_version"], str), "stream algorithm_version must be a string"
        )
        _require(
            _is_int(entry["window_length_ns"]) and entry["window_length_ns"] >= 0,
            "stream window_length_ns must be a non-negative integer",
        )
        _require(
            _is_int(entry["shift_ns"]) and entry["shift_ns"] >= 0,
            "stream shift_ns must be a non-negative integer",
        )
        stream_signal_ids.append(entry["native_signal_id"])
        stream_ids.append(entry["stream_id"])
    _require(
        stream_signal_ids == recorded,
        "extension streams must name exactly the recorded signals in recorded order",
    )
    _require(len(set(stream_ids)) == len(stream_ids), "extension stream ids must be unique")
    _validate_prepared_streams_against_native_schema(signals, feature_sets, raw_streams)

    coverage = extension.get("coverage")
    _require(coverage in {"full", "partial"}, "coverage must be full or partial")
    derived_coverage = "full" if recorded == planned else "partial"
    _require(coverage == derived_coverage, "coverage does not match the recorded signal set")

    _native_replay_rebuild_session(extension.get("session"))
    _native_replay_rebuild_resource_bounds(extension.get("resource_bounds"))
    _require(
        isinstance(extension.get("session_metadata"), dict),
        "extension session_metadata must be an object",
    )
    _require(isinstance(extension.get("metadata"), dict), "extension metadata must be an object")
    _cross_validate_extension_vs_manifest(manifest, raw_streams, extension)

    capabilities = extension.get("replay_capabilities")
    _require(isinstance(capabilities, dict), "replay_capabilities must be an object")
    expected_capabilities = {
        "exact_frames": (
            derived_coverage == "full",
            None if derived_coverage == "full" else "unavailable",
        ),
        "recorded_projection": (True, None),
        "stream_frames": (any(entry["block_index"] for entry in raw_streams), None),
    }
    for mode, (available, _) in expected_capabilities.items():
        _require(mode in capabilities, f"replay_capabilities is missing {mode!r}")
        entry = capabilities[mode]
        _require(isinstance(entry, dict), f"replay_capabilities.{mode} must be an object")
        _require(
            entry.get("available") is available,
            f"replay_capabilities.{mode}.available does not match the plan",
        )
        if available:
            _require(
                entry.get("reason") is None,
                f"replay_capabilities.{mode} must carry no reason when available",
            )
        else:
            _require(
                isinstance(entry.get("reason"), str) and entry["reason"],
                f"replay_capabilities.{mode} must state a reason when unavailable",
            )

    stored_ledgers = extension.get("ledgers")
    _require(isinstance(stored_ledgers, dict), "extension ledgers must be an object")
    _require(
        dict(stored_ledgers) == NATIVE_REPLAY_LEDGER_IDS,
        "extension ledgers do not match the native replay ledger set",
    )

    # Recompute the plan fingerprint over the canonical document and compare.
    document = {
        "extension_version": extension["extension_version"],
        "native_schema": {
            "schema_id": raw_schema["schema_id"],
            "signals": signals,
            "feature_sets": feature_sets,
            "units": units,
        },
        "coverage": coverage,
        "planned_signal_ids": list(planned),
        "recorded_signal_ids": list(recorded),
        "streams": list(raw_streams),
        "ledgers": dict(stored_ledgers),
    }
    import hashlib

    fingerprint = hashlib.sha256(_JCS.canonical_json_bytes(document)).hexdigest()
    _require(
        fingerprint == extension["plan_fingerprint"],
        "plan_fingerprint does not match the recomputed value",
    )


def validate_manifest_semantics(value: dict[str, Any]) -> None:
    """Validate NRF v1 relationships after JSON Schema validation.

    Callers must first validate *value* against ``manifest.schema.json``.
    """

    required = {
        "format",
        "version",
        "session",
        "writer",
        "checksum",
        "canonical_json",
        "default_endianness",
        "codecs",
        "clocks",
        "units",
        "channels",
        "electrodes",
        "schemas",
        "streams",
        "feature_sets",
        "record_schemas",
        "commit",
    }
    allowed = required | {"metadata", "extensions"}
    _require_fields(value, required, "manifest")
    _require(
        set(value) <= allowed,
        "unknown manifest field outside extensions",
    )
    _require(value["format"] == "nrf", "invalid format")
    _require(value["version"]["major"] == 1, "unsupported major version")
    _require(
        isinstance(value["version"]["minor"], int) and value["version"]["minor"] >= 0,
        "invalid minor version",
    )
    _require(
        value["checksum"] == "sha256" and value["canonical_json"] == "rfc8785",
        "unsupported checksum contract",
    )

    # Enforce the README section 7 I-JSON contract over the whole manifest,
    # including free-form objects the JSON Schema cannot bound (metadata,
    # extensions, Zarr fill_value). Runs before registry checks so an
    # out-of-range integer is reported as an I-JSON domain violation.
    _check_ijson_domain(value, "manifest")

    # Defensive recheck of the session created_at timestamp: the schema
    # declares the strict UTC pattern and format annotation, but a validator
    # without a format checker would otherwise accept any Z-terminated string.
    session = value["session"]
    _require(
        isinstance(session, dict)
        and isinstance(session.get("created_at"), str)
        and _is_valid_utc_date_time(session["created_at"]),
        "invalid session created_at timestamp",
    )

    codec_ids = _ids(value["codecs"], "codec")
    _require(
        all(codec["name"] in CODECS for codec in value["codecs"]),
        "unsupported codec",
    )
    clock_ids = _ids(value["clocks"], "clock")
    _require(clock_ids, "clock registry is empty")
    unit_ids = _ids(value["units"], "unit")
    channel_ids = _ids(value["channels"], "channel")
    electrode_ids = _ids(value["electrodes"], "electrode")
    schema_ids = _ids(value["schemas"], "schema")
    stream_ids = _ids(value["streams"], "stream")
    _ids(value["feature_sets"], "feature set")
    feature_descriptors = {descriptor["id"]: descriptor for descriptor in value["feature_sets"]}
    _ids(value["record_schemas"], "record schema")
    record_schema_paths = [schema["path"] for schema in value["record_schemas"]]
    _require(
        len(record_schema_paths) == len(set(record_schema_paths)),
        "duplicate record schema path",
    )
    record_schemas_by_id = {schema["id"]: schema for schema in value["record_schemas"]}

    for clock in value["clocks"]:
        if "rate" in clock:
            _require(
                clock["rate"]["numerator"] > 0 and clock["rate"]["denominator"] > 0,
                "invalid clock rate",
            )
        _require(
            clock.get("synchronization_source_clock_id", clock["id"]) in clock_ids,
            "broken synchronization clock reference",
        )
    for channel in value["channels"]:
        _require(channel["unit_id"] in unit_ids, "broken channel unit reference")
        _require(
            channel.get("electrode_id") is None or channel["electrode_id"] in electrode_ids,
            "broken electrode reference",
        )
    for electrode in value["electrodes"]:
        _require(
            set(electrode["channel_ids"]) <= channel_ids,
            "broken electrode channel reference",
        )
    for schema in value["schemas"]:
        _require(
            set(schema["stream_ids"]) <= stream_ids,
            "broken schema stream reference",
        )

    for descriptor in value["feature_sets"]:
        _validate_feature_descriptor(descriptor, stream_ids, unit_ids)
    committed_extents = value["commit"]["committed_extents"]
    sealed_targets = set(value["commit"]["sealed_targets"])
    for path in committed_extents:
        _require_canonical_relative_path(path, "committed extent path")
    for path in sealed_targets:
        _require_canonical_relative_path(path, "sealed target path")
    _require(
        sealed_targets <= set(committed_extents),
        "sealed target lacks a committed extent",
    )
    owned_paths: set[str] = set()
    physical_array_owners: dict[str, str] = {}
    claimed_discontinuity_schema_ids: set[str] = set()
    for stream in value["streams"]:
        _validate_stream_descriptor(
            stream,
            schema_ids=schema_ids,
            stream_ids=stream_ids,
            clock_ids=clock_ids,
            channel_ids=channel_ids,
            unit_ids=unit_ids,
            feature_descriptors=feature_descriptors,
            codec_ids=codec_ids,
            committed_extents=committed_extents,
        )
        _require(
            stream["data"]["path"] == f"streams/{stream['id']}/data",
            "stream data path does not match stream ID",
        )
        owned_paths.add(stream["data"]["path"])
        _claim_physical_array_path(
            physical_array_owners,
            stream["data"]["path"],
            f"stream:{stream['id']}:data",
        )
        if stream["timing"]["mode"] == "explicit":
            _require(
                stream["timing"]["timestamps"]["path"] == f"streams/{stream['id']}/timestamps",
                "stream timestamp path does not match stream ID",
            )
            owned_paths.add(stream["timing"]["timestamps"]["path"])
            _claim_physical_array_path(
                physical_array_owners,
                stream["timing"]["timestamps"]["path"],
                f"stream:{stream['id']}:timestamps",
            )
        # Each stream's segment policy must reference a discontinuities-kind
        # record schema; that schema's path is the per-stream discontinuity
        # record-set location and the committed-extent key for its chunks.
        disc_schema_id = stream["segment_policy"]["discontinuity_record_schema_id"]
        disc_schema = record_schemas_by_id.get(disc_schema_id)
        _require(disc_schema is not None, "broken discontinuity record schema reference")
        _require(
            disc_schema is not None and disc_schema["kind"] == "discontinuities",
            "segment policy does not reference a discontinuity record schema",
        )
        _require(
            disc_schema_id not in claimed_discontinuity_schema_ids,
            "discontinuity record schema is shared by multiple streams",
        )
        _require(
            disc_schema is not None
            and disc_schema["path"] == f"streams/{stream['id']}/discontinuities",
            "discontinuity record schema path does not belong to stream",
        )
        claimed_discontinuity_schema_ids.add(disc_schema_id)
        data_path = stream["data"]["path"]
        if data_path not in sealed_targets:
            _require(
                stream["committed_extent"] % stream["data"]["chunk_shape"][0] == 0,
                "open stream extent must be chunk aligned",
            )

    record_kinds = set()
    for schema in value["record_schemas"]:
        record_kinds.add(schema["kind"])
        _require_payload_path(schema["path"], "record schema path")
        if schema["kind"] == "discontinuities":
            _require(
                re.fullmatch(
                    r"streams/[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*/discontinuities",
                    schema["path"],
                )
                is not None,
                "discontinuity record schema is outside its stream namespace",
            )
        else:
            _require(
                schema["path"] == f"records/{schema['kind']}/{schema['id']}",
                "record schema path does not match kind and ID",
            )
        owned_paths.add(schema["path"])
        field_names = [item["name"] for item in schema["fields"]]
        _require(
            len(field_names) == len(set(field_names)),
            "duplicate record field",
        )
        _require(
            schema["primary_key"] in field_names,
            "record primary key is not a field",
        )
        primary_key_field = next(
            item for item in schema["fields"] if item["name"] == schema["primary_key"]
        )
        _require(
            primary_key_field["nullable"] is False,
            "record primary key must be non-nullable",
        )
        _require(
            not primary_key_field.get("shape"),
            "record primary key must be scalar",
        )
        _require(
            primary_key_field["dtype"] in PRIMARY_KEY_DTYPES,
            "record primary key has unsupported dtype",
        )
        _require("clock_id" in schema, "record schema is missing clock_id")
        _require(schema["clock_id"] in clock_ids, "broken record clock reference")
        _require(
            committed_extents.get(schema["path"], 0) == schema["committed_extent"],
            "record extent mismatch",
        )
        if schema["path"] not in sealed_targets:
            _require(
                schema["committed_extent"] % schema["chunk_length"] == 0,
                "open record extent must be chunk aligned",
            )
        field_paths = [item["array_path"] for item in schema["fields"]]
        _require(
            len(field_paths) == len(set(field_paths)),
            "duplicate record column path",
        )
        for item in schema["fields"]:
            _require_canonical_relative_path(item["array_path"], "record column path")
            expected_array_path = f"{schema['path']}/columns/{item['name']}"
            _require(
                item["array_path"] == expected_array_path,
                "record column path does not match schema field",
            )
            _claim_physical_array_path(
                physical_array_owners,
                item["array_path"],
                f"record:{schema['id']}:column:{item['name']}",
            )
            if item["nullable"]:
                _require_canonical_relative_path(
                    item["validity_path"],
                    "record validity path",
                )
                expected_validity_path = f"{schema['path']}/validity/{item['name']}"
                _require(
                    item["validity_path"] == expected_validity_path,
                    "record validity path does not match schema field",
                )
                _claim_physical_array_path(
                    physical_array_owners,
                    item["validity_path"],
                    f"record:{schema['id']}:validity:{item['name']}",
                )
            _require(
                set(item["codec_ids"]) <= codec_ids,
                "broken record codec reference",
            )
            if item["dtype"] in {"bool", "uint8", "utf8"}:
                _require(
                    item["endianness"] == "not_applicable",
                    "invalid record endianness",
                )
            else:
                _require(
                    item["endianness"] in {"little", "big"},
                    "invalid record endianness",
                )
            _require(
                item.get("unit_id") is None or item["unit_id"] in unit_ids,
                "broken record unit reference",
            )
    _require(
        {
            "events",
            "trials",
            "experiment_state",
            "commands",
            "faults",
            "session_termination",
        }
        <= record_kinds,
        "required record schemas are missing",
    )
    termination_schemas = [
        schema for schema in value["record_schemas"] if schema["kind"] == "session_termination"
    ]
    _require(
        len(termination_schemas) == 1,
        "manifest must define exactly one session_termination schema",
    )
    termination_schema = termination_schemas[0]
    _require(
        termination_schema["path"] == "records/session_termination/termination-v1",
        "session_termination schema path is not canonical",
    )
    _require(
        termination_schema["primary_key"] == "termination_id",
        "session_termination primary key must be termination_id",
    )
    termination_fields = {item["name"]: item for item in termination_schema["fields"]}
    _require(
        set(termination_fields)
        == {
            "termination_id",
            "time_ns",
            "termination_kind",
            "reason",
            "fault_id",
            "last_transaction_id",
        },
        "session_termination fields do not match the NRF v1 contract",
    )
    expected_termination_fields = {
        "termination_id": ("utf8", False, None),
        "time_ns": ("int64", False, None),
        "termination_kind": ("utf8", False, None),
        "reason": ("utf8", False, None),
        "fault_id": ("utf8", True, "fault"),
        "last_transaction_id": ("utf8", False, "transaction"),
    }
    for field_name, (dtype, nullable, reference) in expected_termination_fields.items():
        descriptor = termination_fields[field_name]
        _require(
            (
                descriptor["dtype"],
                descriptor["nullable"],
                descriptor.get("reference"),
            )
            == (dtype, nullable, reference),
            f"invalid session_termination field contract: {field_name}",
        )
    _require(
        termination_schema["committed_extent"] in {0, 1},
        "session_termination extent must be zero or one",
    )
    _require(
        (termination_schema["committed_extent"] == 1)
        == (termination_schema["path"] in sealed_targets),
        "session_termination extent and seal state disagree",
    )

    # Every committed-extent key must be owned by exactly one stream data
    # array, stream timestamp array, or record schema path. Without this, an
    # arbitrary unowned extent (e.g. "unknown/unowned") could be added to the
    # commit cache and a reader could not attribute the chunk to any decodable
    # object, including per-stream discontinuity record sets.
    _require(
        set(committed_extents) <= owned_paths,
        "committed extent is not owned by a stream or record schema",
    )

    last_transaction = value["commit"]["last_transaction_id"]
    _require(
        last_transaction is None or TX_ID.fullmatch(last_transaction),
        "invalid last transaction ID",
    )
    checkpoint = value["commit"]["last_checkpoint_id"]
    _require(
        checkpoint is None or CHECKPOINT_ID.fullmatch(checkpoint),
        "invalid checkpoint ID",
    )

    # The native replay extension is optional; when any part of it is declared,
    # its cross-field rules are part of manifest semantics.
    validate_native_replay_extension(value)
