#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Canonical JSON, checksum, and path rules for NRF v1 sessions.

These tests bind the implementation to the checked-in normative vectors rather
than to a round trip through the implementation itself, so an encoding change
that is self-consistent but non-conforming still fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from neurale.exceptions import NeuraleError
from neurale.io.nrf import (
    NrfCorruptionError,
    NrfError,
    NrfSchemaError,
    NrfSemanticError,
    NrfStateError,
)
from neurale.io.nrf._canonical import (
    SAFE_INTEGER_MAX,
    SAFE_INTEGER_MIN,
    canonical_json,
    canonical_json_bytes,
    check_ijson_domain,
    encode_journal_line,
    is_sha256_hex,
    is_utc_date_time,
    record_checksum,
    sha256_hex,
    sign_record,
    verify_record_checksum,
)
from neurale.io.nrf._paths import (
    chunk_path,
    is_canonical_relative_path,
    metadata_snapshot_path,
    paths_overlap,
    portable_path_key,
    record_column_path,
    record_validity_path,
    require_canonical_relative_path,
    require_payload_path,
    staged_path,
    stream_data_path,
    transaction_id,
)

from .nrf_support import VECTORS


def _checksummed_records(node: Any) -> list[dict[str, Any]]:
    """Collect every object in the vectors that carries a ``record_checksum``."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "record_checksum" in node:
            found.append(node)
        for value in node.values():
            found.extend(_checksummed_records(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_checksummed_records(value))
    return found


# --- canonical JSON -------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["canonical_json"]["cases"], ids=lambda c: c["name"])
def test_canonical_json_matches_normative_vectors(case: dict[str, Any]) -> None:
    assert canonical_json(case["input"]) == case["canonical_utf8"]
    assert sha256_hex(canonical_json_bytes(case["input"])) == case["sha256"]


def test_canonical_json_is_not_merely_key_sorted_json() -> None:
    # RFC 8785 sorts by UTF-16 code unit, which differs from Python's default
    # code-point ordering once astral characters are involved.
    value = {"\U0001f600": 1, "\uffff": 2}
    assert canonical_json(value) != json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_canonical_json_serializes_numbers_per_rfc8785() -> None:
    assert canonical_json({"a": -0.0}) == '{"a":0}'
    assert canonical_json({"a": 1.0}) == '{"a":1}'


# --- checksums ------------------------------------------------------------


def test_every_checksummed_vector_record_verifies() -> None:
    records = _checksummed_records(VECTORS)
    assert len(records) >= 50
    assert [record for record in records if not verify_record_checksum(record)] == []


def test_record_checksum_omits_checksum_member() -> None:
    record = {"record_type": "commit", "sequence": 3}
    signed = sign_record(record)
    assert signed["record_checksum"] == record_checksum(record)
    assert verify_record_checksum(signed)
    # Re-signing an already signed record is stable.
    assert sign_record(signed) == signed


def test_record_checksum_detects_any_mutation() -> None:
    signed = sign_record({"record_type": "commit", "sequence": 3})
    tampered = dict(signed, sequence=4)
    assert not verify_record_checksum(tampered)


def test_missing_or_malformed_checksum_does_not_verify() -> None:
    assert not verify_record_checksum({"record_type": "commit"})
    assert not verify_record_checksum({"record_type": "commit", "record_checksum": "NOTHEX"})
    assert not verify_record_checksum({"record_type": "commit", "record_checksum": "ab" * 32})


def test_sha256_hex_recognizes_only_lower_case_digests() -> None:
    assert is_sha256_hex("0" * 64)
    assert not is_sha256_hex("A" * 64)
    assert not is_sha256_hex("0" * 63)


def test_journal_line_is_canonical_json_plus_one_lf() -> None:
    signed = sign_record({"record_type": "commit", "sequence": 1})
    line = encode_journal_line(signed)
    assert line.endswith(b"\n")
    assert b"\n" not in line[:-1]
    assert json.loads(line.decode("utf-8")) == signed


# --- I-JSON domain --------------------------------------------------------


@pytest.mark.parametrize("value", [SAFE_INTEGER_MIN, SAFE_INTEGER_MAX, 0, -1, 1.5, True, "x"])
def test_ijson_domain_accepts_interoperable_values(value: Any) -> None:
    check_ijson_domain({"value": value})


@pytest.mark.parametrize(
    "value",
    [SAFE_INTEGER_MIN - 1, SAFE_INTEGER_MAX + 1, float("nan"), float("inf"), float("-inf")],
)
def test_ijson_domain_rejects_uninteroperable_numbers(value: Any) -> None:
    with pytest.raises(NrfSchemaError):
        check_ijson_domain({"metadata": {"nested": [value]}})


def test_ijson_domain_reaches_free_form_members() -> None:
    # The point of the recursive walk: JSON Schema cannot bound these.
    with pytest.raises(NrfSchemaError, match="I-JSON"):
        check_ijson_domain({"extensions": {"vendor": {"count": SAFE_INTEGER_MAX + 1}}})


def test_record_checksum_rejects_uninteroperable_payloads() -> None:
    with pytest.raises(NrfSchemaError):
        record_checksum({"metadata": {"count": SAFE_INTEGER_MAX + 1}})


# --- date-time ------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z", "2024-02-29T12:00:00.500Z"],
)
def test_utc_date_time_accepts_conforming_values(value: str) -> None:
    assert is_utc_date_time(value)


@pytest.mark.parametrize(
    "value",
    [
        "2026-12-31T23:59:60Z",  # leap second
        "2026-02-30T00:00:00Z",  # impossible calendar day
        "2025-02-29T00:00:00Z",  # not a leap year
        "2026-01-01T00:00:00+01:00",  # offset instead of Z
        "2026-01-01T00:00:00",  # missing Z
        "0000-01-01T00:00:00Z",  # year zero
        "not-a-dateZ",
    ],
)
def test_utc_date_time_rejects_unsupported_values(value: str) -> None:
    assert not is_utc_date_time(value)


# --- paths ----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "streams/cursor/./data",
        "streams/cursor/../data",
        "streams/cursor//data",
        "streams/cursor/data/",
        "/streams/cursor/data",
        "C:/outside/data",
        "C:outside/data",
        "streams/cursor\\data",
        "streams/cursor/\ndata",
        "streams/cursor/data with space",
        "con/data",
        "records/nul.txt/x",
        "streams/neural/data.",
        "Streams/Neural/Data",
        "",
    ],
)
def test_non_canonical_paths_are_rejected_never_normalized(path: str) -> None:
    assert not is_canonical_relative_path(path)
    with pytest.raises(NrfSemanticError, match="canonical relative NRF path"):
        require_canonical_relative_path(path, "stream data path")


@pytest.mark.parametrize(
    "path",
    [
        "manifest.json",
        "zarr.json",
        "journal/transactions.jsonl",
        ".staging/evil/data",
        "metadata/transactions/tx-0000000000000001/x",
        "recovery/example/data",
        "checksums/example/data",
        "indexes/example/data",
        "feature_sets/example/data",
    ],
)
def test_payload_arrays_cannot_occupy_control_paths(path: str) -> None:
    with pytest.raises(NrfSemanticError):
        require_payload_path(path, "stream data path")


@pytest.mark.parametrize(
    "path",
    [
        "streams/neural/data",
        "streams/neural/timestamps",
        "streams/neural/discontinuities",
        "records/events/events-v1",
        "records/session_termination/termination-v1",
    ],
)
def test_frozen_layout_paths_are_valid_payload_paths(path: str) -> None:
    assert require_payload_path(path, "payload") == path


def test_portable_key_collapses_cross_platform_aliases() -> None:
    assert portable_path_key("streams/neural/data") == ("streams", "neural", "data")
    # Case and trailing dots/spaces alias on some filesystems even though a
    # conforming path never carries them.
    assert portable_path_key("STREAMS/Neural/data. ") == portable_path_key("streams/neural/data")


def test_overlapping_physical_arrays_are_detected() -> None:
    assert paths_overlap("streams/n/data", "streams/n/data/c/0")
    assert paths_overlap("streams/n/data", "streams/n/data")
    assert not paths_overlap("streams/a/data", "streams/b/data")
    assert not paths_overlap("records/events/e1", "records/events/e2")


def test_derived_paths_follow_frozen_layout() -> None:
    assert stream_data_path("neural") == "streams/neural/data"
    assert record_column_path("records/events/events-v1", "onset_ns") == (
        "records/events/events-v1/columns/onset_ns"
    )
    assert record_validity_path("records/events/events-v1", "value") == (
        "records/events/events-v1/validity/value"
    )
    assert chunk_path("streams/neural/data", (0, 0)) == "streams/neural/data/c/0/0"
    assert transaction_id(1) == "tx-0000000000000001"
    assert metadata_snapshot_path("tx-0000000000000001", "neural-data") == (
        "metadata/transactions/tx-0000000000000001/neural-data.json"
    )


def test_staged_path_derives_from_final_path() -> None:
    final = "streams/neural/data/c/0/0"
    staged = staged_path("tx-0000000000000001", final)
    assert staged == f".staging/tx-0000000000000001/{final}"
    assert staged != final


# --- packaging boundaries -------------------------------------------------


def test_errors_derive_from_public_hierarchy() -> None:
    for error in (NrfSchemaError, NrfSemanticError, NrfCorruptionError, NrfStateError):
        assert issubclass(error, NrfError)
        assert issubclass(error, NeuraleError)
    # Layer identity is observable: a schema failure is not a semantic failure.
    assert not issubclass(NrfSchemaError, NrfSemanticError)
    assert not issubclass(NrfSemanticError, NrfSchemaError)


def _loaded_modules(imported: str, candidates: tuple[str, ...]) -> list[str]:
    """Return which of *candidates* a fresh interpreter loads with *imported*."""
    code = (
        f"import sys; import {imported}; "
        f"print(' '.join(name for name in {candidates!r} if name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def test_import_does_not_load_optional_dependencies() -> None:
    assert _loaded_modules("neurale.io.nrf", ("zarr", "rfc8785", "jsonschema")) == []


def test_importing_neurale_io_does_not_import_nrf() -> None:
    assert _loaded_modules("neurale.io", ("neurale.io.nrf",)) == []
