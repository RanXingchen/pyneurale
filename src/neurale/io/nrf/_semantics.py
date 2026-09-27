#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Cross-object semantic validation, the second mandatory NRF v1 layer.

Specification reference: ``specifications/nrf/v1/semantic-validation.md``
rules NRF-SEM-001 through NRF-SEM-018.

This is deliberately a **second, independent implementation** of those rules.
``specifications/nrf/v1/tools/semantic_validation.py`` remains the
specification's reference validator and is not imported here: if the package
reused it, the specification tests would be checking the implementation against
itself instead of against an independent oracle.
``tests/specification/test_nrf_v1_package_conformance.py`` runs both validators
over the same inputs and requires identical verdicts, so the two stay pinned
together without either depending on the other.

The functions below are grouped by rule ID rather than by manifest section so
that coverage of the rule table is auditable by reading the module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from neurale.exceptions import RecorderConfigError

from ._canonical import check_ijson_domain, is_utc_date_time
from ._errors import NrfSchemaError, NrfSemanticError
from ._ledgers import (
    LEDGERS,
    NATIVE_REPLAY_NAMESPACE,
    declares_native_replay,
    ledger_schema_ids,
)
from ._paths import (
    portable_path_key,
    record_column_path,
    record_set_path,
    record_validity_path,
    require_canonical_relative_path,
    require_payload_path,
    stream_data_path,
    stream_discontinuities_path,
    stream_timestamps_path,
)
from ._plan_document import plan_from_manifest_extension

SEMANTIC_VALIDATION_VERSION = "1.0"

_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_TRANSACTION_ID = re.compile(r"^tx-[0-9]{16}$")
_CHECKPOINT_ID = re.compile(r"^checkpoint-[0-9]{16}$")

_DTYPES = frozenset(
    {
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
)
#: dtypes for which a byte order is meaningless.
_ENDIAN_FREE_DTYPES = frozenset({"bool", "uint8", "utf8"})
_NATIVE_TO_NRF_DTYPE: Mapping[str, str] = {
    "INT16": "int16",
    "INT32": "int32",
    "FLOAT32": "float32",
    "FLOAT64": "float64",
}
_CODECS = frozenset({"bytes", "vlen-utf8", "blosc", "gzip", "zstd", "crc32c", "sharding_indexed"})
_PRIMARY_KEY_DTYPES = frozenset({"utf8", "int64", "uint64"})

_STREAM_AXES: Mapping[str, list[str]] = {
    "neural": ["sample", "channel"],
    "behavioral": ["sample", "channel"],
    "feature": ["observation", "feature"],
    "spike": ["spike", "sample", "channel"],
}

_REQUIRED_RECORD_KINDS = frozenset(
    {
        "events",
        "trials",
        "experiment_state",
        "commands",
        "faults",
        "session_termination",
    }
)

_TERMINATION_PATH = "records/session_termination/termination-v1"
#: name -> (dtype, nullable, reference) exactly as fixed by README section 8.1.
_TERMINATION_FIELDS: Mapping[str, tuple[str, bool, str | None]] = {
    "termination_id": ("utf8", False, None),
    "time_ns": ("int64", False, None),
    "termination_kind": ("utf8", False, None),
    "reason": ("utf8", False, None),
    "fault_id": ("utf8", True, "fault"),
    "last_transaction_id": ("utf8", False, "transaction"),
}

_MANIFEST_REQUIRED = frozenset(
    {
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
)
_MANIFEST_OPTIONAL = frozenset({"metadata", "extensions"})

_DISCONTINUITY_PATH = re.compile(r"^streams/[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*/discontinuities$")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise NrfSemanticError(message)


class _PathOwners:
    """Collect physical array paths and enforce NRF-SEM-015 ownership.

    A physical array may not be an ancestor or descendant of another, and two
    paths that would alias on a case-insensitive or trailing-dot-stripping
    filesystem count as the same object.
    """

    def __init__(self) -> None:
        #: Claimed path -> (owner, portable comparison key).
        self._owners: dict[str, tuple[str, tuple[str, ...]]] = {}

    def claim(self, path: str, owner: str) -> None:
        require_payload_path(path, "physical array path")
        key = portable_path_key(path)
        for existing, (_, existing_key) in self._owners.items():
            _check(path != existing, f"physical array path has multiple owners: {path!r}")
            _check(key != existing_key, f"portable physical array path collision: {path!r}")
            shortest = min(len(key), len(existing_key))
            _check(
                key[:shortest] != existing_key[:shortest],
                f"physical array paths overlap: {path!r} and {existing!r}",
            )
        self._owners[path] = (owner, key)


# --- NRF-SEM-001: unique IDs within each registry -------------------------


def _registry_ids(entries: Any, label: str) -> set[str]:
    _check(isinstance(entries, Sequence) and not isinstance(entries, str), f"{label} registry")
    ids: list[Any] = [entry.get("id") for entry in entries]
    for value in ids:
        _check(
            isinstance(value, str) and _ID.fullmatch(value) is not None,
            f"invalid {label} ID: {value!r}",
        )
    _check(len(ids) == len(set(ids)), f"duplicate {label} ID")
    return set(ids)


# --- NRF-SEM-003 / 006: Zarr array shape, rank, codecs --------------------


def _validate_zarr_array(arr: Mapping[str, Any], codec_ids: set[str], axes: list[str]) -> None:
    for field in ("path", "zarr_format", "shape", "chunk_shape", "codec_ids"):
        _check(field in arr, f"Zarr array is missing {field}")
    require_canonical_relative_path(arr["path"], "Zarr array path")
    _check(arr["zarr_format"] == 3, "only Zarr v3 is supported")
    shape = arr["shape"]
    chunk_shape = arr["chunk_shape"]
    _check(len(shape) == len(axes), "Zarr shape rank does not match the declared axes")
    _check(len(chunk_shape) == len(axes), "Zarr chunk shape rank does not match the declared axes")
    _check(
        all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape),
        "invalid Zarr shape",
    )
    _check(
        all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in chunk_shape
        ),
        "invalid Zarr chunk shape",
    )
    _check(
        bool(arr["codec_ids"]) and set(arr["codec_ids"]) <= codec_ids,
        "unknown codec reference",
    )


def _validate_endianness(dtype: Any, endianness: Any, label: str) -> None:
    _check(dtype in _DTYPES, f"unsupported {label} dtype: {dtype!r}")
    if dtype in _ENDIAN_FREE_DTYPES:
        _check(endianness == "not_applicable", f"invalid {label} endianness for {dtype}")
    else:
        _check(endianness in {"little", "big"}, f"invalid {label} endianness for {dtype}")


# --- NRF-SEM-007: feature descriptors -------------------------------------


def _validate_feature_set(
    descriptor: Mapping[str, Any], stream_ids: set[str], unit_ids: set[str]
) -> None:
    names = descriptor["feature_names"]
    _check(bool(names), "feature names must be non-empty")
    _check(len(names) == len(set(names)), "feature names must be unique")
    _check(len(names) == len(descriptor["unit_ids"]), "feature name and unit counts disagree")
    _check(set(descriptor["unit_ids"]) <= unit_ids, "broken feature unit reference")
    _check(descriptor["source_stream_id"] in stream_ids, "broken feature source reference")
    _check(
        descriptor["window_length_ns"] > 0 and descriptor["shift_ns"] > 0,
        "invalid feature window length or shift",
    )


# --- NRF-SEM-004 / 005 / 009 / 010 / 011: streams -------------------------


def _validate_stream(
    stream: Mapping[str, Any],
    *,
    registries: Mapping[str, set[str]],
    feature_sets: Mapping[str, Mapping[str, Any]],
    record_schemas: Mapping[str, Mapping[str, Any]],
    committed_extents: Mapping[str, int],
    sealed_targets: set[str],
    owners: _PathOwners,
    claimed_discontinuities: set[str],
    owned_paths: set[str],
) -> None:
    kind = stream["kind"]
    _check(kind in _STREAM_AXES, f"unsupported stream kind: {kind!r}")
    axes = _STREAM_AXES[kind]
    _check(stream["axes"] == axes, f"unsupported payload layout for {kind} stream")
    _validate_endianness(stream["dtype"], stream["endianness"], "stream")

    _check(stream["schema_id"] in registries["schemas"], "broken schema reference")
    _check(stream["clock_id"] in registries["clocks"], "broken clock reference")
    _check(
        set(stream["source_stream_ids"]) <= registries["streams"],
        "broken source stream reference",
    )
    _check(set(stream["channel_ids"]) <= registries["channels"], "broken channel reference")
    _check(set(stream["unit_ids"]) <= registries["units"], "broken unit reference")

    _validate_zarr_array(stream["data"], registries["codecs"], axes)

    # NRF-SEM-004: registry widths must equal the payload width.
    width = stream["data"]["shape"][-1]
    _check(len(stream["unit_ids"]) == width, "unit count does not match payload width")
    if stream["channel_ids"]:
        _check(len(stream["channel_ids"]) == width, "channel count does not match payload width")

    # NRF-SEM-005 / 009: extent agreement with the physical bound and the cache.
    extent = stream["committed_extent"]
    _check(
        isinstance(extent, int)
        and not isinstance(extent, bool)
        and 0 <= extent <= stream["data"]["shape"][0],
        "invalid stream committed extent",
    )
    _check(
        committed_extents.get(stream["data"]["path"], 0) == extent,
        "manifest commit cache disagrees with the stream extent",
    )

    timing = stream["timing"]
    if timing["mode"] == "regular":
        rate = timing["rate"]
        _check(rate["numerator"] > 0 and rate["denominator"] > 0, "invalid regular rate")
    else:
        _validate_zarr_array(timing["timestamps"], registries["codecs"], [axes[0]])
        _check(
            timing["timestamps"]["shape"][0] >= extent,
            "explicit timestamps do not cover every committed item",
        )
        _check(
            committed_extents.get(timing["timestamps"]["path"], 0) == extent,
            "timestamp extent disagrees with the stream extent",
        )

    if kind == "feature":
        feature_set_id = stream["feature_set_id"]
        _check(feature_set_id in feature_sets, "feature stream lacks a valid descriptor")
        descriptor = feature_sets[feature_set_id]
        _check(
            len(descriptor["feature_names"]) == width,
            "feature descriptor width does not match payload width",
        )
        _check(
            descriptor["unit_ids"] == stream["unit_ids"],
            "feature descriptor units do not match stream units",
        )
        _check(
            descriptor["source_stream_id"] in stream["source_stream_ids"],
            "feature descriptor source is not linked by the stream",
        )

    # NRF-SEM-016: the frozen stream layout.
    stream_id = stream["id"]
    data_path = stream["data"]["path"]
    _check(data_path == stream_data_path(stream_id), "stream data path does not match stream ID")
    owned_paths.add(data_path)
    owners.claim(data_path, f"stream:{stream_id}:data")

    if timing["mode"] == "explicit":
        timestamps_path = timing["timestamps"]["path"]
        _check(
            timestamps_path == stream_timestamps_path(stream_id),
            "stream timestamp path does not match stream ID",
        )
        owned_paths.add(timestamps_path)
        owners.claim(timestamps_path, f"stream:{stream_id}:timestamps")

    # NRF-SEM-011: a dedicated, unshared discontinuities schema per stream.
    schema_id = stream["segment_policy"]["discontinuity_record_schema_id"]
    schema = record_schemas.get(schema_id)
    _check(schema is not None, "broken discontinuity record schema reference")
    assert schema is not None
    _check(
        schema["kind"] == "discontinuities",
        "segment policy does not reference a discontinuities record schema",
    )
    _check(
        schema_id not in claimed_discontinuities,
        "discontinuity record schema is shared by multiple streams",
    )
    _check(
        schema["path"] == stream_discontinuities_path(stream_id),
        "discontinuity record schema path does not belong to its stream",
    )
    claimed_discontinuities.add(schema_id)

    # NRF-SEM-009: an open target ends on a chunk boundary.
    if data_path not in sealed_targets:
        _check(
            extent % stream["data"]["chunk_shape"][0] == 0,
            "open stream extent must be chunk aligned",
        )


# --- NRF-SEM-008 / 015: record schemas ------------------------------------


def _validate_record_schema(
    schema: Mapping[str, Any],
    *,
    registries: Mapping[str, set[str]],
    committed_extents: Mapping[str, int],
    sealed_targets: set[str],
    owners: _PathOwners,
    owned_paths: set[str],
) -> None:
    path = require_payload_path(schema["path"], "record schema path")
    kind = schema["kind"]
    if kind == "discontinuities":
        _check(
            _DISCONTINUITY_PATH.fullmatch(path) is not None,
            "discontinuity record schema is outside its stream namespace",
        )
    else:
        _check(
            path == record_set_path(kind, schema["id"]),
            "record schema path does not match its kind and ID",
        )
    owned_paths.add(path)

    fields = schema["fields"]
    names = [field["name"] for field in fields]
    _check(len(names) == len(set(names)), "duplicate record field name")

    # NRF-SEM-008: the primary key must be able to carry stable identity.
    primary_key = schema["primary_key"]
    _check(primary_key in names, "record primary key is not one of the fields")
    key_field = next(field for field in fields if field["name"] == primary_key)
    _check(key_field["nullable"] is False, "record primary key must be non-nullable")
    _check(not key_field.get("shape"), "record primary key must be scalar")
    _check(
        key_field["dtype"] in _PRIMARY_KEY_DTYPES,
        "record primary key has an unsupported dtype",
    )

    _check("clock_id" in schema, "record schema is missing clock_id")
    _check(schema["clock_id"] in registries["clocks"], "broken record clock reference")

    _check(
        committed_extents.get(path, 0) == schema["committed_extent"],
        "manifest commit cache disagrees with the record extent",
    )
    if path not in sealed_targets:
        _check(
            schema["committed_extent"] % schema["chunk_length"] == 0,
            "open record extent must be chunk aligned",
        )

    array_paths = [field["array_path"] for field in fields]
    _check(len(array_paths) == len(set(array_paths)), "duplicate record column path")

    for field in fields:
        name = field["name"]
        require_canonical_relative_path(field["array_path"], "record column path")
        _check(
            field["array_path"] == record_column_path(path, name),
            "record column path does not match its schema field",
        )
        owners.claim(field["array_path"], f"record:{schema['id']}:column:{name}")
        if field["nullable"]:
            require_canonical_relative_path(field["validity_path"], "record validity path")
            _check(
                field["validity_path"] == record_validity_path(path, name),
                "record validity path does not match its schema field",
            )
            owners.claim(field["validity_path"], f"record:{schema['id']}:validity:{name}")
        _check(set(field["codec_ids"]) <= registries["codecs"], "broken record codec reference")
        _validate_endianness(field["dtype"], field["endianness"], "record")
        _check(
            field.get("unit_id") is None or field["unit_id"] in registries["units"],
            "broken record unit reference",
        )


# --- NRF-SEM-017: the session_termination contract ------------------------


def _validate_termination_schema(
    record_schemas: Sequence[Mapping[str, Any]], sealed_targets: set[str]
) -> None:
    schemas = [schema for schema in record_schemas if schema["kind"] == "session_termination"]
    _check(len(schemas) == 1, "manifest must define exactly one session_termination schema")
    schema = schemas[0]
    _check(schema["path"] == _TERMINATION_PATH, "session_termination schema path is not canonical")
    _check(
        schema["primary_key"] == "termination_id",
        "session_termination primary key must be termination_id",
    )
    fields = {field["name"]: field for field in schema["fields"]}
    _check(
        set(fields) == set(_TERMINATION_FIELDS),
        "session_termination fields do not match the NRF v1 contract",
    )
    for name, contract in _TERMINATION_FIELDS.items():
        field = fields[name]
        actual = (field["dtype"], field["nullable"], field.get("reference"))
        _check(actual == contract, f"invalid session_termination field contract: {name}")
    extent = schema["committed_extent"]
    _check(extent in {0, 1}, "session_termination extent must be zero or one")
    _check(
        (extent == 1) == (schema["path"] in sealed_targets),
        "session_termination extent and seal state disagree",
    )


# --- entry point ----------------------------------------------------------


def validate_manifest(manifest: Any) -> None:
    """Validate NRF v1 cross-object rules for an already schema-valid manifest.

    The caller MUST first run
    :func:`neurale.io.nrf._schemas.validate_document`; passing this layer alone
    never compensates for failing the schema layer.

    Raises
    ------
    NrfSemanticError
        If any NRF-SEM rule is violated.
    NrfSchemaError
        If the manifest leaves the I-JSON domain (NRF-SEM-013), which is a
        canonicalization problem rather than a relationship problem.
    """
    _check(isinstance(manifest, Mapping), "manifest must be a JSON object")

    missing = _MANIFEST_REQUIRED - set(manifest)
    _check(not missing, f"manifest is missing required fields: {sorted(missing)}")
    unknown = set(manifest) - _MANIFEST_REQUIRED - _MANIFEST_OPTIONAL
    _check(not unknown, f"unknown manifest field outside extensions: {sorted(unknown)}")

    _check(manifest["format"] == "nrf", "invalid format name")
    _check(manifest["version"]["major"] == 1, "unsupported major version")
    minor = manifest["version"]["minor"]
    _check(
        isinstance(minor, int) and not isinstance(minor, bool) and minor >= 0,
        "invalid minor version",
    )
    _check(
        manifest["checksum"] == "sha256" and manifest["canonical_json"] == "rfc8785",
        "unsupported checksum or canonical-JSON contract",
    )

    # NRF-SEM-013 runs before the registry rules so an uninteroperable integer
    # is reported as a domain violation rather than as a broken reference.
    check_ijson_domain(manifest, "manifest")

    # NRF-SEM-014: defensive recheck. A JSON Schema validator without a format
    # backend would otherwise accept any Z-terminated string.
    session = manifest["session"]
    _check(
        isinstance(session, Mapping) and is_utc_date_time(session.get("created_at")),
        "invalid session created_at timestamp",
    )

    # NRF-SEM-001.
    registries = {
        "codecs": _registry_ids(manifest["codecs"], "codec"),
        "clocks": _registry_ids(manifest["clocks"], "clock"),
        "units": _registry_ids(manifest["units"], "unit"),
        "channels": _registry_ids(manifest["channels"], "channel"),
        "electrodes": _registry_ids(manifest["electrodes"], "electrode"),
        "schemas": _registry_ids(manifest["schemas"], "schema"),
        "streams": _registry_ids(manifest["streams"], "stream"),
        "feature_sets": _registry_ids(manifest["feature_sets"], "feature set"),
        "record_schemas": _registry_ids(manifest["record_schemas"], "record schema"),
    }
    _check(
        all(codec["name"] in _CODECS for codec in manifest["codecs"]),
        "unsupported codec",
    )
    # NRF-SEM-008: the clock registry is never empty in NRF v1.
    _check(bool(registries["clocks"]), "clock registry is empty")

    record_paths = [schema["path"] for schema in manifest["record_schemas"]]
    _check(len(record_paths) == len(set(record_paths)), "duplicate record schema path")

    feature_sets = {entry["id"]: entry for entry in manifest["feature_sets"]}
    record_schemas = {entry["id"]: entry for entry in manifest["record_schemas"]}

    # NRF-SEM-002 / 010 for the small registries.
    for clock in manifest["clocks"]:
        if "rate" in clock:
            _check(
                clock["rate"]["numerator"] > 0 and clock["rate"]["denominator"] > 0,
                "invalid clock rate",
            )
        _check(
            clock.get("synchronization_source_clock_id", clock["id"]) in registries["clocks"],
            "broken synchronization clock reference",
        )
    for channel in manifest["channels"]:
        _check(channel["unit_id"] in registries["units"], "broken channel unit reference")
        _check(
            channel.get("electrode_id") is None
            or channel["electrode_id"] in registries["electrodes"],
            "broken electrode reference",
        )
    for electrode in manifest["electrodes"]:
        _check(
            set(electrode["channel_ids"]) <= registries["channels"],
            "broken electrode channel reference",
        )
    for schema in manifest["schemas"]:
        _check(
            set(schema["stream_ids"]) <= registries["streams"],
            "broken schema stream reference",
        )

    # NRF-SEM-007.
    for descriptor in manifest["feature_sets"]:
        _validate_feature_set(descriptor, registries["streams"], registries["units"])

    commit = manifest["commit"]
    committed_extents = commit["committed_extents"]
    sealed_targets = set(commit["sealed_targets"])
    for path in committed_extents:
        require_canonical_relative_path(path, "committed extent path")
    for path in sealed_targets:
        require_canonical_relative_path(path, "sealed target path")
    _check(
        sealed_targets <= set(committed_extents),
        "sealed target lacks a committed extent",
    )

    owners = _PathOwners()
    owned_paths: set[str] = set()
    claimed_discontinuities: set[str] = set()
    for stream in manifest["streams"]:
        _validate_stream(
            stream,
            registries=registries,
            feature_sets=feature_sets,
            record_schemas=record_schemas,
            committed_extents=committed_extents,
            sealed_targets=sealed_targets,
            owners=owners,
            claimed_discontinuities=claimed_discontinuities,
            owned_paths=owned_paths,
        )

    kinds: set[str] = set()
    for schema in manifest["record_schemas"]:
        kinds.add(schema["kind"])
        _validate_record_schema(
            schema,
            registries=registries,
            committed_extents=committed_extents,
            sealed_targets=sealed_targets,
            owners=owners,
            owned_paths=owned_paths,
        )
    missing_kinds = _REQUIRED_RECORD_KINDS - kinds
    _check(not missing_kinds, f"required record schemas are missing: {sorted(missing_kinds)}")

    # NRF-SEM-017.
    _validate_termination_schema(manifest["record_schemas"], sealed_targets)

    # NRF-SEM-011: no committed extent may name an unowned, non-decodable path.
    unowned = set(committed_extents) - owned_paths
    _check(
        not unowned,
        f"committed extent is not owned by a stream or record schema: {sorted(unowned)}",
    )

    # NRF-SEM-010: the commit cache identifiers themselves.
    last_transaction = commit["last_transaction_id"]
    _check(
        last_transaction is None or _TRANSACTION_ID.fullmatch(last_transaction) is not None,
        "invalid last transaction ID",
    )
    last_checkpoint = commit["last_checkpoint_id"]
    _check(
        last_checkpoint is None or _CHECKPOINT_ID.fullmatch(last_checkpoint) is not None,
        "invalid checkpoint ID",
    )

    # NRF-SEM-018: the optional native replay extension, when declared.
    _validate_native_replay_extension(manifest)


def _validate_native_ledger_descriptors(
    manifest: Mapping[str, Any], ledgers: Sequence[Any]
) -> None:
    """NRF-SEM-018: a native ledger's descriptor must match its schema id exactly.

    A session may declare ``kind = native_frames`` and ``id = native-frames-v1``
    but lay the columns out differently. The core record-schema rules check
    that a descriptor is internally valid (primary key, dtypes, paths, codec
    references); this binds the schema id to the *exact* field structure the
    extension fixes -- primary key, field count, and each field's name, dtype,
    endianness, nullability, array path, validity path, and reference -- so a
    later stage that interprets a ledger by its fixed schema id cannot read the
    wrong type or the wrong meaning. Codec ids, clock id, and chunk length are
    session-scoped (README section 7) and are left to the core rules.
    """
    ledger_kinds = {ledger.kind for ledger in ledgers}
    by_kind: dict[str, Mapping[str, Any]] = {}
    for schema in manifest["record_schemas"]:
        if schema["kind"] in ledger_kinds:
            _check(
                schema["kind"] not in by_kind,
                f"native replay ledger kind {schema['kind']!r} is declared more than once",
            )
            by_kind[schema["kind"]] = schema
    for ledger in ledgers:
        schema = by_kind.get(ledger.kind)
        if schema is None:
            continue
        _check(
            schema["id"] == ledger.schema_id,
            f"native replay ledger {ledger.kind!r} has the wrong schema id",
        )
        _check(
            schema["kind"] == ledger.kind,
            f"native replay ledger {ledger.kind!r} has the wrong kind",
        )
        _check(
            schema["path"] == ledger.path,
            f"native replay ledger {ledger.kind!r} has a non-canonical path",
        )
        _check(
            schema["primary_key"] == ledger.primary_key,
            f"native replay ledger {ledger.kind!r} has the wrong primary key",
        )
        actual = schema["fields"]
        expected = ledger.fields
        _check(
            len(actual) == len(expected),
            f"native replay ledger {ledger.kind!r} has {len(actual)} fields, expected {len(expected)}",
        )
        for i, (actual_field, ledger_field) in enumerate(zip(actual, expected, strict=True)):
            _check_ledger_field(ledger.kind, ledger.path, i, actual_field, ledger_field)


def _check_ledger_field(
    kind: str, ledger_path: str, idx: int, actual: Mapping[str, Any], field: Any
) -> None:
    label = f"native replay ledger {kind!r} field {idx} ({field.name!r})"
    _check(actual.get("name") == field.name, f"{label} has the wrong name")
    _check(actual.get("dtype") == field.dtype, f"{label} has the wrong dtype")
    expected_endianness = "not_applicable" if field.dtype in _ENDIAN_FREE_DTYPES else "little"
    _check(actual.get("endianness") == expected_endianness, f"{label} has the wrong endianness")
    _check(actual.get("nullable") == field.nullable, f"{label} has the wrong nullability")
    _check(
        actual.get("array_path") == f"{ledger_path}/columns/{field.name}",
        f"{label} has the wrong array path",
    )
    if field.nullable:
        _check(
            actual.get("validity_path") == f"{ledger_path}/validity/{field.name}",
            f"{label} is missing its validity path",
        )
    else:
        _check("validity_path" not in actual, f"{label} must not carry a validity path")
    if field.reference is not None:
        _check(actual.get("reference") == field.reference, f"{label} has the wrong reference")
    else:
        _check("reference" not in actual, f"{label} must not carry a reference")
    # The schema id fixes which keys a field carries; a stray ``unit_id`` or
    # ``shape`` is a descriptor that is not the one the schema id names.
    allowed_keys = {"name", "dtype", "endianness", "nullable", "array_path", "codec_ids"}
    if field.nullable:
        allowed_keys = allowed_keys | {"validity_path"}
    if field.reference is not None:
        allowed_keys = allowed_keys | {"reference"}
    _check(set(actual) <= allowed_keys, f"{label} carries an unexpected key")


def _validate_native_replay_extension(manifest: Mapping[str, Any]) -> None:
    """NRF-SEM-018: the neurale.native_replay extension is self-consistent.

    A session that declares none of the extension's record kinds and carries no
    extension object is untouched. A session that declares any of them must
    declare all five ledgers, the extension object, and minor version 1, and the
    extension must rebuild into a plan whose recomputed fingerprint equals the
    stored one. This reuses the plan parser
    (:func:`neurale.io.nrf._plan_document.plan_from_manifest_extension`) for the
    extension-internal checks so the semantic layer and the plan parser cannot
    drift apart, then adds the one cross-field check the parser cannot make
    without the manifest: every extension stream id resolves to a manifest
    stream.
    """
    record_schemas = manifest["record_schemas"]
    ledger_kinds = {
        schema["kind"] for schema in record_schemas if schema["kind"] in ledger_schema_ids()
    }
    extensions = manifest.get("extensions") or {}
    extension = extensions.get(NATIVE_REPLAY_NAMESPACE)
    if not ledger_kinds and extension is None:
        return

    _check(
        declares_native_replay(record_schemas),
        "the native replay ledgers must be declared all together or not at all",
    )
    _validate_native_ledger_descriptors(manifest, LEDGERS)

    _check(manifest["version"]["minor"] == 1, "a session declaring native replay must be NRF v1.1")
    _check(isinstance(extension, Mapping), "the neurale.native_replay extension must be an object")

    try:
        plan_from_manifest_extension(extension)
    except RecorderConfigError as error:
        raise NrfSemanticError(str(error)) from error

    _cross_validate_extension_vs_manifest(manifest, extension)


def _cross_validate_extension_vs_manifest(
    manifest: Mapping[str, Any], extension: Mapping[str, Any]
) -> None:
    """NRF-SEM-018: the extension must agree with the core manifest it rides on.

    The extension restates session identity, the recorded streams, and their
    resolved metadata so a finalizer can use the plan alone; each restated fact
    must equal the one the core manifest carries, or a session could declare two
    different truths and leave a later stage unable to pick the authoritative one.
    """
    session = extension["session"]
    manifest_session = manifest["session"]
    _check(
        session["session_id"] == manifest_session["id"],
        "extension session_id does not match the manifest session",
    )
    _check(
        session["created_at"] == manifest_session["created_at"],
        "extension session created_at does not match the manifest session",
    )

    units_by_id = {unit["id"]: unit for unit in manifest["units"]}
    channels_by_id = {channel["id"]: channel for channel in manifest["channels"]}
    feature_sets_by_id = {fs["id"]: fs for fs in manifest["feature_sets"]}
    streams_by_id = {stream["id"]: stream for stream in manifest["streams"]}
    data_stream_ids = {stream["id"] for stream in manifest["streams"] if "native_id" in stream}

    extension_streams = extension["streams"]
    _check(
        {entry["stream_id"] for entry in extension_streams} == data_stream_ids,
        "extension stream ids do not match the manifest's recorded data streams",
    )
    for entry in extension_streams:
        stream = streams_by_id.get(entry["stream_id"])
        assert stream is not None  # the set check above guarantees this
        _check(
            entry["native_signal_id"] == stream["native_id"],
            f"extension stream {entry['stream_id']!r} native_signal_id does not match the manifest stream",
        )
        _check(
            entry["kind"] == stream["kind"],
            f"extension stream {entry['stream_id']!r} kind does not match the manifest stream",
        )
        _check(
            entry["timing"] == stream["timing"]["mode"],
            f"extension stream {entry['stream_id']!r} timing does not match the manifest stream",
        )
        _check(
            entry["name"] == stream["name"],
            f"extension stream {entry['stream_id']!r} name does not match the manifest stream",
        )
        expected_unit = [units_by_id[unit_id]["symbol"] for unit_id in stream["unit_ids"]]
        _check(
            entry["unit"] == expected_unit,
            f"extension stream {entry['stream_id']!r} unit does not match the manifest stream",
        )
        if stream["kind"] == "feature":
            feature_set = feature_sets_by_id[stream["feature_set_id"]]
            _check(
                entry["channel_names"] == [],
                f"extension feature stream {entry['stream_id']!r} must not name channels",
            )
            _check(
                entry["feature_names"] == feature_set["feature_names"],
                f"extension stream {entry['stream_id']!r} feature names do not match the manifest feature set",
            )
            _check(
                entry["algorithm_name"] == feature_set["algorithm"]["name"],
                f"extension stream {entry['stream_id']!r} algorithm name does not match the manifest feature set",
            )
            _check(
                entry["algorithm_version"] == feature_set["algorithm"]["version"],
                f"extension stream {entry['stream_id']!r} algorithm version does not match the manifest feature set",
            )
            _check(
                entry["window_length_ns"] == feature_set["window_length_ns"],
                f"extension stream {entry['stream_id']!r} window length does not match the manifest feature set",
            )
            _check(
                entry["shift_ns"] == feature_set["shift_ns"],
                f"extension stream {entry['stream_id']!r} shift does not match the manifest feature set",
            )
        else:
            expected_channels = [channels_by_id[cid]["name"] for cid in stream["channel_ids"]]
            _check(
                entry["channel_names"] == expected_channels,
                f"extension stream {entry['stream_id']!r} channel names do not match the manifest stream",
            )
            _check(
                entry["feature_names"] == [],
                f"extension non-feature stream {entry['stream_id']!r} must not name features",
            )

    _cross_validate_native_schema_vs_manifest(
        manifest, extension, streams_by_id, feature_sets_by_id, data_stream_ids
    )

    session_metadata = {
        key: value for key, value in manifest_session.items() if key not in {"id", "created_at"}
    }
    _check(
        extension["session_metadata"] == session_metadata,
        "extension session_metadata does not match the manifest session metadata",
    )
    _check(
        extension["metadata"] == manifest.get("metadata", {}),
        "extension metadata does not match the manifest metadata",
    )


def _cross_validate_native_schema_vs_manifest(
    manifest: Mapping[str, Any],
    extension: Mapping[str, Any],
    streams_by_id: Mapping[str, Mapping[str, Any]],
    feature_sets_by_id: Mapping[str, Mapping[str, Any]],
    data_stream_ids: set[str],
) -> None:
    """NRF-SEM-018: the extension's native schema must match the manifest's data.

    The native schema lets exact replay rebuild the original StreamSchema, so
    each native signal's dtype, channel count, clock domain, and (for a regular
    stream) sample rate must equal the manifest stream's data description, and
    each prepared stream's segment origin, source links, and block-index
    declaration must equal the manifest stream's. A mismatch would let replay
    rebuild a schema unlike the one the payload was actually written under.
    """
    native_signals = {sig["id"]: sig for sig in extension["native_schema"]["signals"]}
    native_feature_sets = {fs["id"]: fs for fs in extension["native_schema"]["feature_sets"]}
    signal_to_stream = {
        entry["native_signal_id"]: entry["stream_id"] for entry in extension["streams"]
    }
    for entry in extension["streams"]:
        stream = streams_by_id[entry["stream_id"]]
        signal = native_signals.get(entry["native_signal_id"])
        assert signal is not None  # plan_from_manifest_extension guarantees this
        _check(
            _NATIVE_TO_NRF_DTYPE.get(signal["dtype"]) == stream["dtype"],
            f"extension native signal {entry['native_signal_id']} dtype does not match manifest stream {entry['stream_id']!r}",
        )
        _check(
            signal["n_channels"] == stream["data"]["shape"][-1],
            f"extension native signal {entry['native_signal_id']} channel count does not match manifest stream {entry['stream_id']!r}",
        )
        _check(
            signal["clock_domain"] == _clock_native_id(manifest, stream["clock_id"]),
            f"extension native signal {entry['native_signal_id']} clock domain does not match manifest stream {entry['stream_id']!r}",
        )
        timing = stream["timing"]
        if timing["mode"] == "regular":
            _check(
                signal["fs"]["numerator"] == timing["rate"]["numerator"]
                and signal["fs"]["denominator"] == timing["rate"]["denominator"],
                f"extension native signal {entry['native_signal_id']} sample rate does not match manifest stream {entry['stream_id']!r}",
            )
            _check(
                entry["segment_start_index"] == timing.get("segment_start_index", 0),
                f"extension stream {entry['stream_id']!r} segment start index does not match the manifest stream",
            )
            _check(
                entry["segment_start_time_ns"] == timing.get("segment_start_time_ns", 0),
                f"extension stream {entry['stream_id']!r} segment start time does not match the manifest stream",
            )
        _check(
            entry["source_stream_ids"] == list(stream["source_stream_ids"]),
            f"extension stream {entry['stream_id']!r} source stream ids do not match the manifest stream",
        )
        _check(
            entry["block_index"]
            == (f"{entry['stream_id']}.blocks" in {s["id"] for s in manifest["streams"]}),
            f"extension stream {entry['stream_id']!r} block index declaration does not match the manifest",
        )
        if stream["kind"] == "feature":
            manifest_fs = feature_sets_by_id[stream["feature_set_id"]]
            native_fs = native_feature_sets.get(signal["feature_set_id"])
            assert native_fs is not None
            _check(
                signal_to_stream.get(native_fs["source_stream_id"])
                == manifest_fs["source_stream_id"],
                f"extension native feature set source does not match manifest feature set for stream {entry['stream_id']!r}",
            )


def _clock_native_id(manifest: Mapping[str, Any], clock_id: str) -> Any:
    for clock in manifest["clocks"]:
        if clock["id"] == clock_id:
            return clock.get("native_id", 0)
    return 0


__all__ = ["SEMANTIC_VALIDATION_VERSION", "NrfSchemaError", "NrfSemanticError", "validate_manifest"]
