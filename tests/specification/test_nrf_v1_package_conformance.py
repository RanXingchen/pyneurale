#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""``neurale.io.nrf`` against the specification's own reference implementations.

The package validates NRF v1 twice over -- semantics in
:mod:`neurale.io.nrf._semantics`, journal replay in
:mod:`neurale.io.nrf._replay` -- and the specification carries an independent
implementation of each: ``specifications/nrf/v1/tools/semantic_validation.py``
and the replay oracle in :mod:`test_nrf_v1_specification`. Neither side imports
the other. If the package reused the reference, this suite would be checking an
implementation against itself.

What is compared is the verdict, and the committed state a reader derives from
it. Diagnostic wording is deliberately not compared -- ``semantic-validation.md``
section 2 explicitly permits it to differ.

This is the only place the specification suite touches the installed package.
Rules the package enforces beyond the reference, and the wording of its own
errors, belong to ``tests/unit/io/nrf/``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest
from _nrf_v1_reference import (
    SEMANTIC_VALIDATION_VERSION as REFERENCE_SEMANTIC_VALIDATION_VERSION,
)
from _nrf_v1_reference import (
    SPEC_DIR,
    VECTORS,
    NrfSemanticValidationError,
    canonical_json_bytes,
    validate_manifest_semantics,
)

# The replay oracle is specification tooling that lives with the rest of the
# specification's self-check, one module over.
from test_nrf_v1_specification import replay_journal

from neurale.io.nrf import (
    NATIVE_REPLAY_EXTENSION_VERSION,
    ledger_record_schemas,
    ledger_schema_ids,
)
from neurale.io.nrf._errors import NrfSchemaError, NrfSemanticError
from neurale.io.nrf._replay import replay
from neurale.io.nrf._semantics import SEMANTIC_VALIDATION_VERSION, validate_manifest

Mutation = Callable[[dict[str, Any]], None]


# --- the semantic layer ---------------------------------------------------


def _reference_verdict(manifest: dict[str, Any]) -> bool:
    """Return whether the specification's reference validator accepts."""
    try:
        validate_manifest_semantics(manifest)
    except NrfSemanticValidationError:
        return False
    return True


def _package_verdict(manifest: dict[str, Any]) -> bool:
    """Return whether the package validator accepts."""
    try:
        validate_manifest(manifest)
    except (NrfSemanticError, NrfSchemaError):
        return False
    return True


#: Every mutation the specification suite exercises, plus reference breakage for
#: each registry the package resolves independently.
MANIFEST_MUTATIONS: dict[str, Mutation] = {
    "duplicate_clock_id": lambda v: v["clocks"].append(deepcopy(v["clocks"][0])),
    "empty_clock_registry": lambda v: v["clocks"].clear(),
    "broken_stream_clock": lambda v: v["streams"][0].update(clock_id="missing-clock"),
    "record_schema_without_clock": lambda v: v["record_schemas"][0].pop("clock_id"),
    "zero_clock_denominator": lambda v: v["clocks"][0]["rate"].update(denominator=0),
    "unsupported_dtype": lambda v: v["streams"][0].update(dtype="complex128"),
    "transposed_axes": lambda v: v["streams"][0].update(axes=["channel", "sample"]),
    "unknown_codec": lambda v: v["codecs"][0].update(name="unknown"),
    "stream_extent_beyond_shape": lambda v: v["streams"][0].update(committed_extent=5),
    "broken_feature_source": lambda v: v["feature_sets"][0].update(source_stream_id="missing"),
    "duplicate_feature_name": lambda v: v["feature_sets"][0]["feature_names"].__setitem__(
        1, v["feature_sets"][0]["feature_names"][0]
    ),
    "broken_record_codec": lambda v: v["record_schemas"][0]["fields"][0].update(
        codec_ids=["missing-codec"]
    ),
    "unsupported_major_version": lambda v: v["version"].update(major=2),
    "malformed_created_at": lambda v: v["session"].update(created_at="not-a-dateZ"),
    "leap_second_created_at": lambda v: v["session"].update(created_at="2026-12-31T23:59:60Z"),
    "impossible_calendar_day": lambda v: v["session"].update(created_at="2026-02-30T00:00:00Z"),
    "non_canonical_stream_path": lambda v: v["streams"][0]["data"].update(
        path="Streams/Neural/Data"
    ),
    "unowned_committed_extent": lambda v: v["commit"]["committed_extents"].update(
        {"unknown/unowned": 1}
    ),
    "uninteroperable_metadata_integer": lambda v: v.setdefault("metadata", {}).update(
        big=9007199254740992
    ),
    "broken_channel_unit": lambda v: v["channels"][0].update(unit_id="missing-unit"),
    "broken_schema_stream": lambda v: v["schemas"][0].update(stream_ids=["missing-stream"]),
    "unknown_manifest_field": lambda v: v.update(surprise=1),
    "missing_manifest_field": lambda v: v.pop("units"),
    "wrong_format_name": lambda v: v.update(format="not-nrf"),
    "wrong_checksum_algorithm": lambda v: v.update(checksum="sha1"),
    "sealed_target_without_extent": lambda v: v["commit"]["sealed_targets"].append(
        "records/events/unknown-v1"
    ),
    "malformed_last_transaction_id": lambda v: v["commit"].update(last_transaction_id="tx-1"),
    "duplicate_record_field": lambda v: v["record_schemas"][0]["fields"].append(
        deepcopy(v["record_schemas"][0]["fields"][0])
    ),
    "unsealed_committed_termination": lambda v: v["commit"].update(
        sealed_targets=[
            path
            for path in v["commit"]["sealed_targets"]
            if path != "records/session_termination/termination-v1"
        ]
    ),
    "shared_discontinuity_schema": lambda v: v["streams"][1]["segment_policy"].update(
        discontinuity_record_schema_id=v["streams"][0]["segment_policy"][
            "discontinuity_record_schema_id"
        ]
    ),
}


def test_semantic_validation_version_tracks_specification() -> None:
    specification = (SPEC_DIR / "semantic-validation.md").read_text(encoding="utf-8")
    assert SEMANTIC_VALIDATION_VERSION == "1.0"
    assert f"Semantic validation version: {SEMANTIC_VALIDATION_VERSION}" in specification
    assert SEMANTIC_VALIDATION_VERSION == REFERENCE_SEMANTIC_VALIDATION_VERSION


def test_normative_manifest_is_accepted_by_both_validators() -> None:
    assert _package_verdict(deepcopy(VECTORS["manifest"]))
    assert _reference_verdict(deepcopy(VECTORS["manifest"]))


@pytest.mark.parametrize("name", sorted(MANIFEST_MUTATIONS))
def test_package_and_reference_validators_agree(name: str) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    MANIFEST_MUTATIONS[name](manifest)
    package = _package_verdict(deepcopy(manifest))
    reference = _reference_verdict(deepcopy(manifest))
    assert package == reference, (
        f"validators disagree on {name}: package accepted={package}, reference accepted={reference}"
    )


@pytest.mark.parametrize("name", sorted(MANIFEST_MUTATIONS))
def test_every_manifest_mutation_is_actually_rejected(name: str) -> None:
    # Guards the differential test above: two validators that both accept
    # everything would also "agree".
    manifest = deepcopy(VECTORS["manifest"])
    MANIFEST_MUTATIONS[name](manifest)
    assert not _package_verdict(manifest)


# --- native replay extension (NRF-SEM-018) -------------------------------


def _native_replay_extension(manifest: dict[str, Any]) -> dict[str, Any]:
    """A valid neurale.native_replay extension for *manifest*'s streams.

    The extension is built FROM the manifest and cross-checked against it: its
    native schema matches the manifest's data description (dtype, channel count,
    clock domain, regular rate), its streams match the manifest's stream
    metadata, and the two agree on session identity and metadata. The fingerprint
    is computed with the specification's own JCS, the same canonicalization both
    validators recompute with, so the legal vector is accepted and a mutation that
    changes fingerprinted content is rejected by both for the same reason.
    """
    units_by_id = {unit["id"]: unit for unit in manifest["units"]}
    channels_by_id = {channel["id"]: channel for channel in manifest["channels"]}
    feature_sets_by_id = {fs["id"]: fs for fs in manifest["feature_sets"]}
    clocks_by_id = {clock["id"]: clock for clock in manifest["clocks"]}
    manifest_stream_ids = {stream["id"] for stream in manifest["streams"]}
    data_streams = [stream for stream in manifest["streams"] if "native_id" in stream]
    stream_to_native = {stream["id"]: stream["native_id"] for stream in data_streams}

    # Native units: one stable native id per distinct manifest unit symbol that a
    # feature stream's descriptor names, so the feature-set unit ids resolve.
    native_units: list[dict[str, Any]] = []
    unit_symbol_to_native_id: dict[str, int] = {}
    for stream in data_streams:
        if stream["kind"] != "feature":
            continue
        feature_set = feature_sets_by_id[stream["feature_set_id"]]
        for unit_id in feature_set["unit_ids"]:
            symbol = units_by_id[unit_id]["symbol"]
            if symbol not in unit_symbol_to_native_id:
                native_id = len(unit_symbol_to_native_id) + 1
                unit_symbol_to_native_id[symbol] = native_id
                native_units.append(
                    {
                        "id": native_id,
                        "symbol": symbol,
                        "description": units_by_id[unit_id].get("description", symbol),
                    }
                )

    # Native feature sets: one per feature data stream, matching the manifest's
    # feature descriptor and mapping its source to the recording's native signal.
    native_feature_sets: list[dict[str, Any]] = []
    manifest_fs_to_native_id: dict[str, int] = {}
    for stream in data_streams:
        if stream["kind"] != "feature":
            continue
        feature_set = feature_sets_by_id[stream["feature_set_id"]]
        native_id = feature_set.get("native_id") or (len(manifest_fs_to_native_id) + 1)
        manifest_fs_to_native_id[feature_set["id"]] = native_id
        native_feature_sets.append(
            {
                "id": native_id,
                "feature_names": list(feature_set["feature_names"]),
                "unit_ids": [
                    unit_symbol_to_native_id[units_by_id[uid]["symbol"]]
                    for uid in feature_set["unit_ids"]
                ],
                "source_stream_id": stream_to_native[feature_set["source_stream_id"]],
                "source_stream": feature_set["source_stream_id"],
                "algorithm_name": feature_set["algorithm"]["name"],
                "algorithm_version": feature_set["algorithm"]["version"],
                "window_length_ns": feature_set["window_length_ns"],
                "shift_ns": feature_set["shift_ns"],
                "timestamp_reference": "WINDOW_CENTER",
            }
        )

    nominal_block_samples = 8
    max_block_samples = 16

    def _signal_for(stream: dict[str, Any]) -> dict[str, Any]:
        is_feature = stream["kind"] == "feature"
        clock = clocks_by_id[stream["clock_id"]]
        dtype = _NRF_DTYPE_TO_NATIVE[stream["dtype"]]
        width = stream["data"]["shape"][-1]
        if is_feature:
            kind = "FEATURE"
            observation = "REGULAR"
            feature_set_id = manifest_fs_to_native_id[stream["feature_set_id"]]
            physical_unit = "UNSPECIFIED"
            rate = stream["timing"]["rate"]
        else:
            kind = "SAMPLED"
            observation = "NOT_APPLICABLE"
            feature_set_id = 0
            physical_unit = "UNSPECIFIED"
            rate = (
                stream["timing"]["rate"]
                if stream["timing"]["mode"] == "regular"
                else {"numerator": 1000, "denominator": 1}
            )
        return {
            "id": stream["native_id"],
            "clock_domain": clock.get("native_id", 0),
            "n_channels": width,
            "nominal_block_samples": nominal_block_samples,
            "max_block_samples": max_block_samples,
            "fs": {"numerator": rate["numerator"], "denominator": rate["denominator"]},
            "dtype": dtype,
            "layout": "SAMPLE_MAJOR",
            "device_tick_tracking": "UNAVAILABLE",
            "kind": kind,
            "physical_unit": physical_unit,
            "channel_set_id": 0,
            "calibration_id": 0,
            "reference_id": 0,
            "feature_set_id": feature_set_id,
            "observation_timing": observation,
            "fixed_block_bytes": 0,
            "max_block_bytes": width * max_block_samples * _NATIVE_DTYPE_BYTES[dtype],
        }

    def _stream_entry(stream: dict[str, Any]) -> dict[str, Any]:
        is_feature = stream["kind"] == "feature"
        timing = stream["timing"]
        unit = [units_by_id[unit_id]["symbol"] for unit_id in stream["unit_ids"]]
        if is_feature:
            feature_set = feature_sets_by_id[stream["feature_set_id"]]
            channel_names: list[str] = []
            feature_names = list(feature_set["feature_names"])
            algorithm_name = feature_set["algorithm"]["name"]
            algorithm_version = feature_set["algorithm"]["version"]
            window = feature_set["window_length_ns"]
            shift = feature_set["shift_ns"]
        else:
            channel_names = [channels_by_id[cid]["name"] for cid in stream["channel_ids"]]
            feature_names = []
            algorithm_name = ""
            algorithm_version = ""
            window = 0
            shift = 0
        if timing["mode"] == "regular":
            segment_start_idx = timing.get("segment_start_index", 0)
            segment_start_time_ns = timing.get("segment_start_time_ns", 0)
        else:
            segment_start_idx = 0
            segment_start_time_ns = 0
        return {
            "native_signal_id": stream["native_id"],
            "stream_id": stream["id"],
            "kind": stream["kind"],
            "timing": timing["mode"],
            "block_index": f"{stream['id']}.blocks" in manifest_stream_ids,
            "name": stream["name"],
            "unit": unit,
            "channel_names": channel_names,
            "feature_names": feature_names,
            "segment_start_index": segment_start_idx,
            "segment_start_time_ns": segment_start_time_ns,
            "algorithm_name": algorithm_name,
            "algorithm_version": algorithm_version,
            "window_length_ns": window,
            "shift_ns": shift,
            "source_stream_ids": list(stream["source_stream_ids"]),
        }

    signals = [_signal_for(stream) for stream in data_streams]
    native_schema = {
        "schema_id": 7,
        "signals": signals,
        "feature_sets": native_feature_sets,
        "units": native_units,
    }
    planned = sorted(stream["native_id"] for stream in data_streams)
    streams = [_stream_entry(stream) for stream in data_streams]
    ledgers = ledger_schema_ids()
    document = {
        "extension_version": NATIVE_REPLAY_EXTENSION_VERSION,
        "native_schema": native_schema,
        "coverage": "full",
        "planned_signal_ids": planned,
        "recorded_signal_ids": planned,
        "streams": streams,
        "ledgers": ledgers,
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    any_block_idx = any(stream["block_index"] for stream in streams)
    capabilities = {
        "exact_frames": {"available": True, "reason": None},
        "recorded_projection": {"available": True, "reason": None},
        "stream_frames": {
            "available": any_block_idx,
            "reason": None
            if any_block_idx
            else "no recorded stream declares a committed block index",
        },
    }
    session = {
        "session_id": manifest["session"]["id"],
        "created_at": manifest["session"]["created_at"],
        "writer_name": "pyneurale",
        "writer_version": "1",
    }
    resource_bounds = {
        "frame_queue_capacity": 256,
        "control_queue_capacity": 1024,
        "control_chunk_length": 16,
        "checkpoint_interval": 32,
        "overflow_policy": "fault",
        "streams": [
            {
                "stream_id": stream["id"],
                "capacity": 4096,
                "chunk_length": 1024,
                "block_index_chunk_length": 256,
            }
            for stream in data_streams
        ],
    }
    extension = dict(document)
    extension["plan_fingerprint"] = fingerprint
    extension["replay_capabilities"] = capabilities
    extension["session"] = session
    extension["resource_bounds"] = resource_bounds
    extension["session_metadata"] = {
        key: value for key, value in manifest["session"].items() if key not in {"id", "created_at"}
    }
    extension["metadata"] = dict(manifest.get("metadata", {}))
    return extension


_NRF_DTYPE_TO_NATIVE = {
    "int16": "INT16",
    "int32": "INT32",
    "float32": "FLOAT32",
    "float64": "FLOAT64",
}
_NATIVE_DTYPE_BYTES = {"INT16": 2, "INT32": 4, "FLOAT32": 4, "FLOAT64": 8}


def _add_cursor_channels(manifest: dict[str, Any]) -> None:
    """Give the shared vector's cursor stream the two channels its width implies.

    A real recorder registers a channel for every column; the shared manifest
    vector omits them, which the cross-checks (channel count == width == recorded
    channels) would otherwise reject. This is local to the native-replay vectors.
    """
    cursor = next(stream for stream in manifest["streams"] if stream["id"] == "cursor")
    if cursor["channel_ids"]:
        return
    next_idx = max((channel["index"] for channel in manifest["channels"]), default=-1) + 1
    new_ids: list[str] = []
    for offset, name in enumerate(("cursor-x", "cursor-y")):
        channel_id = f"cursor-{offset:04d}"
        manifest["channels"].append(
            {
                "id": channel_id,
                "name": name,
                "index": next_idx + offset,
                "type": "behavioral",
                "unit_id": "meter",
                "valid": True,
                "bad": False,
            }
        )
        new_ids.append(channel_id)
    cursor["channel_ids"] = new_ids


def _native_replay_manifest() -> dict[str, Any]:
    manifest = deepcopy(VECTORS["manifest"])
    _add_cursor_channels(manifest)
    manifest["version"]["minor"] = 1
    clock_id = manifest["clocks"][0]["id"]
    manifest["record_schemas"] = [
        *manifest["record_schemas"],
        *ledger_record_schemas(clock_id=clock_id),
    ]
    manifest["extensions"] = {
        **manifest.get("extensions", {}),
        "neurale.native_replay": _native_replay_extension(manifest),
    }
    return manifest


def _drop_fingerprint(manifest: dict[str, Any]) -> None:
    """Force a fingerprint recomputation by removing the stored value."""
    manifest["extensions"]["neurale.native_replay"].pop("plan_fingerprint")


def _ledger(manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    """Return the native-replay ledger record schema of *kind*."""
    return next(schema for schema in manifest["record_schemas"] if schema["kind"] == kind)


def _swap_native_frames_first_two(manifest: dict[str, Any]) -> None:
    """Swap the first two columns of native_frames so field order disagrees."""
    fields = _ledger(manifest, "native_frames")["fields"]
    fields[0], fields[1] = fields[1], fields[0]


def _recompute_fingerprint(manifest: dict[str, Any]) -> None:
    """Recompute the extension fingerprint over its current document.

    Used by the native-schema invariant vectors so the invariant check, not a
    stale fingerprint, is what rejects a corrupted schema.
    """
    ext = manifest["extensions"]["neurale.native_replay"]
    document = {
        "extension_version": ext["extension_version"],
        "native_schema": ext["native_schema"],
        "coverage": ext["coverage"],
        "planned_signal_ids": ext["planned_signal_ids"],
        "recorded_signal_ids": ext["recorded_signal_ids"],
        "streams": ext["streams"],
        "ledgers": ext["ledgers"],
    }
    ext["plan_fingerprint"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _break_native_signal(manifest: dict[str, Any], field: str, value: Any, idx: int = 0) -> None:
    manifest["extensions"]["neurale.native_replay"]["native_schema"]["signals"][idx][field] = value
    _recompute_fingerprint(manifest)


def _duplicate_native_frames(manifest: dict[str, Any]) -> None:
    """Add a second, structurally valid native_frames schema with a different id.

    The core rules accept it (its path matches its kind and id); the duplicate
    kind is what the native-replay validator must refuse.
    """
    duplicate = deepcopy(_ledger(manifest, "native_frames"))
    duplicate["id"] = "native-frames-v2"
    duplicate["path"] = "records/native_frames/native-frames-v2"
    for field in duplicate["fields"]:
        field["array_path"] = field["array_path"].replace("native-frames-v1", "native-frames-v2")
        if "validity_path" in field:
            field["validity_path"] = field["validity_path"].replace(
                "native-frames-v1", "native-frames-v2"
            )
    manifest["record_schemas"].append(duplicate)


def _ext(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["extensions"]["neurale.native_replay"]


def _native_signal(manifest: dict[str, Any], idx: int = 0) -> dict[str, Any]:
    return _ext(manifest)["native_schema"]["signals"][idx]


def _native_channel_count_mismatch(manifest: dict[str, Any]) -> None:
    """The native schema claims a width the recorded payload does not have.

    A width is stated in several places at once -- the native signal's channel
    count, the extension stream's per-column unit list, and the manifest
    stream's payload shape, unit ids, and channel ids -- so a file cannot say
    two different things about it and be accepted. Which rule speaks first is
    not the point of the vector; that none of them lets it through is. Here the
    prepared-stream rule (unit count == channel count) rejects, before the
    native-schema/manifest width comparison is reached, which is why that
    comparison is deliberate redundancy rather than the only guard.
    """
    _native_signal(manifest, 0).update(n_channels=3, max_block_bytes=3 * 16 * 2)
    _recompute_fingerprint(manifest)


def _native_feature_source_mismatch(manifest: dict[str, Any]) -> None:
    """The native feature set derives from a different stream than the manifest's.

    Exact replay rebuilds the feature descriptor from the native schema, so a
    native ``source_stream_id`` that resolves to a stream other than the one the
    manifest feature set names would reconstruct a feature computed from the
    wrong signal. The extension's own source links are moved with it, so the
    extension stays internally consistent and the contradiction is purely
    against the manifest.
    """
    _ext(manifest)["native_schema"]["feature_sets"][0]["source_stream_id"] = 2
    _ext(manifest)["streams"][2]["source_stream_ids"] = ["cursor"]
    _recompute_fingerprint(manifest)


def _block_index_mismatch(manifest: dict[str, Any]) -> None:
    ext = _ext(manifest)
    ext["streams"][0]["block_index"] = True
    ext["replay_capabilities"]["stream_frames"] = {"available": True, "reason": None}
    _recompute_fingerprint(manifest)


NATIVE_REPLAY_MUTATIONS: dict[str, Mutation] = {
    "minor_version_not_bumped": lambda v: v["version"].update(minor=0),
    "four_ledgers_only": lambda v: v.__setitem__(
        "record_schemas",
        [s for s in v["record_schemas"] if s["kind"] != "session_accounting"],
    ),
    "ledgers_without_extension": lambda v: v["extensions"].pop("neurale.native_replay"),
    "extension_without_ledgers": lambda v: v.__setitem__(
        "record_schemas",
        [s for s in v["record_schemas"] if s["kind"] not in ledger_schema_ids()],
    ),
    "coverage_full_with_recorded_subset": lambda v: (
        v["extensions"]["neurale.native_replay"]["recorded_signal_ids"].__setitem__(0, 99),
        _drop_fingerprint(v),
    ),
    "bad_plan_fingerprint": lambda v: v["extensions"]["neurale.native_replay"].update(
        plan_fingerprint="0" * 64
    ),
    "exact_frames_unavailable_under_full": lambda v: (
        v["extensions"]["neurale.native_replay"]["replay_capabilities"]["exact_frames"].update(
            available=False, reason="forged"
        ),
        _drop_fingerprint(v),
    ),
    "extension_stream_not_in_manifest": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(stream_id="nope"),
        _drop_fingerprint(v),
    ),
    "wrong_ledger_schema_id": lambda v: next(
        s.update(id="wrong") for s in v["record_schemas"] if s["kind"] == "native_frames"
    ),
    # BLOCKER 2: a ledger may declare the right schema id and kind but lay its
    # columns out differently. Each of these is accepted by the core
    # record-schema rules and rejected only by the exact-descriptor binding.
    "ledger_wrong_dtype": lambda v: next(
        f.update(dtype="int64")
        for f in _ledger(v, "native_frames")["fields"]
        if f["name"] == "frame_ordinal"
    ),
    "ledger_wrong_nullability": lambda v: next(
        f.update(nullable=False)
        for f in _ledger(v, "native_frames")["fields"]
        if f["name"] == "source_tick"
    ),
    "ledger_renamed_field": lambda v: next(
        f.update(name="frame_ord", array_path=f["array_path"].replace("frame_ordinal", "frame_ord"))
        for f in _ledger(v, "native_frames")["fields"]
        if f["name"] == "frame_ordinal"
    ),
    "ledger_extra_field": lambda v: _ledger(v, "native_frames")["fields"].append(
        deepcopy(_ledger(v, "native_frames")["fields"][0])
    ),
    "ledger_missing_field": lambda v: _ledger(v, "native_frames")["fields"].pop(13),
    "ledger_wrong_field_order": _swap_native_frames_first_two,
    "ledger_wrong_primary_key": lambda v: _ledger(v, "native_frames").update(
        primary_key="native_session_id"
    ),
    "ledger_wrong_reference": lambda v: next(
        f.update(reference="event")
        for f in _ledger(v, "native_signal_blocks")["fields"]
        if f["name"] == "stream_id"
    ),
    # BLOCKER 2.2: a fixed native ledger kind may be declared at most once.
    "duplicate_ledger_kind": _duplicate_native_frames,
    # BLOCKER 2.3: a native_schema that no legal StreamSchema could carry is
    # refused even when its fingerprint is recomputed to match.
    "native_bad_dtype": lambda v: _break_native_signal(v, "dtype", "BOGUS"),
    "native_zero_channel_count": lambda v: _break_native_signal(v, "n_channels", 0),
    "native_bad_max_block_bytes": lambda v: _break_native_signal(v, "max_block_bytes", 999),
    "native_zero_denominator": lambda v: _break_native_signal(
        v, "fs", {"numerator": 1000, "denominator": 0}
    ),
    "native_sampled_with_feature_set": lambda v: _break_native_signal(v, "feature_set_id", 1),
    # BLOCKER 2.1: the extension must agree with the core manifest. Mutations
    # that change fingerprinted stream metadata recompute the fingerprint so the
    # cross-check, not a stale fingerprint, is what rejects.
    "ext_session_id_mismatch": lambda v: v["extensions"]["neurale.native_replay"]["session"].update(
        session_id="different"
    ),
    "ext_stream_name_mismatch": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(name="different"),
        _recompute_fingerprint(v),
    ),
    "ext_stream_unit_mismatch": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(unit=["x"]),
        _recompute_fingerprint(v),
    ),
    "ext_stream_kind_mismatch": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(kind="behavioral"),
        _recompute_fingerprint(v),
    ),
    "ext_session_metadata_mismatch": lambda v: v["extensions"]["neurale.native_replay"].update(
        session_metadata={"unmatched": 1}
    ),
    "ext_metadata_mismatch": lambda v: v["extensions"]["neurale.native_replay"].update(
        metadata={"unmatched": 2}
    ),
    # BLOCKER 1: a prepared stream mapping must be consistent with its native
    # schema. Each mutation recomputes the fingerprint so the consistency check,
    # not a stale fingerprint, is what rejects.
    "prepared_kind_mismatch": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(
            kind="feature",
            feature_names=["x"],
            channel_names=[],
            algorithm_name="a",
            algorithm_version="1",
            window_length_ns=1,
            shift_ns=1,
            source_stream_ids=[],
        ),
        _recompute_fingerprint(v),
    ),
    "prepared_explicit_origin": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][1].update(
            segment_start_idx=5, segment_start_time_ns=7
        ),
        _recompute_fingerprint(v),
    ),
    # BLOCKER 2: the extension native schema must match the manifest's data
    # description, and the prepared stream's origin, source, and block-index
    # declaration must match the manifest stream's.
    "native_dtype_mismatch": lambda v: (
        _native_signal(v, 0).update(dtype="FLOAT32", max_block_bytes=128),
        _recompute_fingerprint(v),
    ),
    "native_clock_domain_mismatch": lambda v: (
        _native_signal(v, 0).update(clock_domain=99),
        _recompute_fingerprint(v),
    ),
    "native_rate_mismatch": lambda v: (
        _native_signal(v, 0).update(fs={"numerator": 999, "denominator": 1}),
        _recompute_fingerprint(v),
    ),
    "native_source_mismatch": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][2].update(
            source_stream_ids=["neural", "cursor"]
        ),
        _recompute_fingerprint(v),
    ),
    "native_segment_origin_mismatch": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(segment_start_time_ns=999),
        _recompute_fingerprint(v),
    ),
    # Every explicit source link, on a stream of any kind, must resolve to a
    # stream the plan records. This one is reached by the prepared-stream rules
    # themselves: both validators check the plan before comparing it with the
    # manifest, so the dangling reference is refused as an incoherent plan
    # rather than as a disagreement with the manifest.
    "prepared_dangling_source": lambda v: (
        v["extensions"]["neurale.native_replay"]["streams"][0].update(source_stream_ids=["ghost"]),
        _recompute_fingerprint(v),
    ),
    "native_channel_count_mismatch": _native_channel_count_mismatch,
    "native_feature_source_mismatch": _native_feature_source_mismatch,
    "native_block_index_mismatch": _block_index_mismatch,
}


def test_normative_replay_manifest_passes_both_validators() -> None:
    assert _package_verdict(_native_replay_manifest())
    assert _reference_verdict(_native_replay_manifest())


@pytest.mark.parametrize("name", sorted(NATIVE_REPLAY_MUTATIONS))
def test_native_replay_validators_agree(name: str) -> None:
    manifest = _native_replay_manifest()
    NATIVE_REPLAY_MUTATIONS[name](manifest)
    package = _package_verdict(deepcopy(manifest))
    reference = _reference_verdict(deepcopy(manifest))
    assert package == reference, (
        f"validators disagree on native replay {name}: "
        f"package accepted={package}, reference accepted={reference}"
    )


@pytest.mark.parametrize("name", sorted(NATIVE_REPLAY_MUTATIONS))
def test_every_native_replay_mutation_is_actually_rejected(name: str) -> None:
    manifest = _native_replay_manifest()
    NATIVE_REPLAY_MUTATIONS[name](manifest)
    assert not _package_verdict(manifest)


# --- journal replay -------------------------------------------------------

_SEQUENCE: list[dict[str, Any]] = VECTORS["complete_sequence"]
_TAIL: list[dict[str, Any]] = VECTORS["recoverable_uncommitted_tail"]["journal_records"]


def _oracle_replay_verdict(records: list[dict[str, Any]]) -> bool:
    try:
        replay_journal(records, allow_uncommitted_tail=True)
    except Exception:
        return False
    return True


def _package_replay_verdict(records: list[dict[str, Any]]) -> bool:
    try:
        replay(records)
    except NrfSemanticError:
        return False
    return True


#: Journal mutations that must be rejected. Each targets a distinct NRF-TX rule.
JOURNAL_MUTATIONS: dict[str, Any] = {
    "sequence_gap": lambda r: r.__setitem__(2, {**r[2], "sequence": 99}),
    "commit_without_prepare": lambda r: r.pop(0),
    "prepare_does_not_follow_last_commit": lambda r: r.__setitem__(
        2, {**r[2], "previous_committed_transaction_id": None}
    ),
    "unassigned_object": lambda r: r[0].__setitem__(
        "extents", [dict(t, object_paths=[]) for t in r[0]["extents"]]
    ),
    "object_claimed_twice": lambda r: r[0].__setitem__(
        "extents",
        [r[0]["extents"][0], dict(r[0]["extents"][0], target_path="streams/cursor/data")],
    ),
    "staged_path_not_derived": lambda r: r[0]["objects"][0].__setitem__(
        "staged_path", ".staging/elsewhere/object"
    ),
    "extent_goes_backwards": lambda r: r[0]["extents"][0].__setitem__("after", -1),
    "before_does_not_continue_extent": lambda r: r[0]["extents"][0].__setitem__("before", 7),
}


def test_oracle_and_package_accept_normative_sequence() -> None:
    assert _package_replay_verdict(deepcopy(_SEQUENCE))
    assert _oracle_replay_verdict(deepcopy(_SEQUENCE))


def test_oracle_and_package_accept_uncommitted_tail() -> None:
    assert _package_replay_verdict(deepcopy(_TAIL))
    assert _oracle_replay_verdict(deepcopy(_TAIL))


def test_committed_state_matches_oracle_on_normative_sequence() -> None:
    mine = replay(deepcopy(_SEQUENCE))
    theirs = replay_journal(deepcopy(_SEQUENCE), allow_uncommitted_tail=False)
    assert mine.committed_extents == theirs.extents
    assert mine.sealed_targets == theirs.sealed_targets
    assert mine.last_transaction_id == theirs.last_committed_transaction_id
    assert mine.journal_sequence == theirs.journal_sequence
    assert mine.last_commit_sequence == theirs.last_committed_sequence
    assert mine.termination is not None
    assert mine.termination.record_id == theirs.termination_record_id
    assert mine.termination.kind == theirs.termination_kind


def test_committed_state_matches_oracle_on_uncommitted_tail() -> None:
    mine = replay(deepcopy(_TAIL))
    theirs = replay_journal(deepcopy(_TAIL), allow_uncommitted_tail=True)
    assert mine.committed_extents == theirs.extents
    assert mine.sealed_targets == theirs.sealed_targets
    assert mine.last_transaction_id == theirs.last_committed_transaction_id


@pytest.mark.parametrize("name", sorted(JOURNAL_MUTATIONS))
def test_package_and_oracle_agree_on_malformed_journals(name: str) -> None:
    records = deepcopy(_SEQUENCE)
    JOURNAL_MUTATIONS[name](records)
    package = _package_replay_verdict(deepcopy(records))
    reference = _oracle_replay_verdict(deepcopy(records))
    assert package == reference, (
        f"replay implementations disagree on {name}: package accepted={package}, "
        f"oracle accepted={reference}"
    )


@pytest.mark.parametrize("name", sorted(JOURNAL_MUTATIONS))
def test_every_journal_mutation_is_actually_rejected(name: str) -> None:
    """Guards the differential test above, as for the manifest mutations."""
    records = deepcopy(_SEQUENCE)
    JOURNAL_MUTATIONS[name](records)
    assert not _package_replay_verdict(records)
