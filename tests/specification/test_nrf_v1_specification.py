#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Executable acceptance tests for the normative NRF v1 specification.

This module checks the specification against itself: the schemas, the semantic
rule table, the reference validator, and the normative vectors must all describe
the same format. Nothing here imports :mod:`neurale`. Whether the shipped
package agrees with any of it is a separate question, asked in
``test_nrf_v1_package_conformance.py``.

The journal replay oracle defined below is part of that self-check, and the
conformance module imports it from here as its reference for NRF-TX.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any

import pytest
import rfc8785
from _nrf_v1_reference import (
    SEMANTIC_VALIDATION_VERSION,
    SPEC_DIR,
    VECTORS,
    NrfSemanticValidationError,
    canonical_json,
    canonical_json_bytes,
    portable_path_key,
    validate_manifest_semantics,
)
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

SPEC = SPEC_DIR

TX_ID = re.compile(r"^tx-[0-9]{16}$")
CHECKPOINT_ID = re.compile(r"^checkpoint-[0-9]{16}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDX_OBJECT_PATH = re.compile(
    r"^indexes/[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*/"
    r"[a-z0-9._-]+(?:/[a-z0-9._-]+)*$"
)


class NrfValidationError(ValueError):
    """Raised by the specification-only validator used by these tests."""


def record_checksum(value: dict[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "record_checksum"}
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NrfValidationError(message)


def transaction_number(transaction_id: str) -> int:
    """Numeric value of a ``tx-NNNNNNNNNNNNNNNN`` identifier."""
    return int(transaction_id[len("tx-") :])


@dataclass(frozen=True, slots=True)
class ReplayState:
    """Complete logical state produced by replaying a journal prefix.

    The checkpoint oracle compares every field below against a checkpoint
    object, not just extents and sealed targets. ``journal_sequence`` is the
    last replayed record sequence (the replay horizon); ``last_committed_sequence``
    is the sequence of the most recent commit record.
    """

    extents: dict[str, int]
    last_committed_transaction_id: str | None
    sealed_targets: set[str]
    journal_sequence: int
    last_committed_sequence: int
    committed_objects: list[dict[str, Any]]
    idx_extents: dict[str, int]
    termination_record_id: str | None
    termination_kind: str | None


def manifest_target_contracts(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build frozen physical-array contracts from an NRF v1 manifest."""

    contracts: dict[str, dict[str, Any]] = {}

    def add_array(arr: dict[str, Any]) -> None:
        array_path = arr["path"]
        contracts[array_path] = {
            "target_kind": "array",
            "record_schema_id": None,
            "arrays": {
                array_path: {
                    "logical_role": "array_chunk",
                    "rank": len(arr["shape"]),
                    "trailing_chunk_counts": [
                        (size + chunk - 1) // chunk
                        for size, chunk in zip(
                            arr["shape"][1:],
                            arr["chunk_shape"][1:],
                            strict=True,
                        )
                    ],
                }
            },
        }

    for stream in manifest["streams"]:
        add_array(stream["data"])
        if stream["timing"]["mode"] == "explicit":
            add_array(stream["timing"]["timestamps"])

    for schema in manifest["record_schemas"]:
        arrays: dict[str, dict[str, Any]] = {}
        for field in schema["fields"]:
            trailing_shape = field.get("shape", [])
            arrays[field["array_path"]] = {
                "logical_role": "record_column_chunk",
                "rank": 1 + len(trailing_shape),
                "trailing_chunk_counts": [1] * len(trailing_shape),
                "column_name": field["name"],
            }
            if field["nullable"]:
                arrays[field["validity_path"]] = {
                    "logical_role": "record_validity_chunk",
                    "rank": 1,
                    "trailing_chunk_counts": [],
                    "column_name": field["name"],
                }
        contracts[schema["path"]] = {
            "target_kind": "record_set",
            "record_schema_id": schema["id"],
            "arrays": arrays,
        }
    return contracts


def expected_chunk_coordinates(
    *,
    before: int,
    after: int,
    chunk_length: int,
    trailing_chunk_counts: list[int],
) -> set[tuple[int, ...]]:
    if after == before:
        return set()
    leading = range(before // chunk_length, (after - 1) // chunk_length + 1)
    trailing = [range(count) for count in trailing_chunk_counts]
    return {
        (leading_coordinate, *trailing_coordinate)
        for leading_coordinate in leading
        for trailing_coordinate in product(*trailing)
    }


def validate_prepare_object_ownership(
    record: dict[str, Any],
    target_contracts: dict[str, dict[str, Any]] | None,
) -> None:
    """Validate extent dependencies, object ownership, and chunk coverage."""

    objects_by_path = {item["path"]: item for item in record["objects"]}
    require(
        len(objects_by_path) == len(record["objects"]),
        "duplicate object path in transaction",
    )
    staged_paths = [item["staged_path"] for item in record["objects"]]
    require(
        len(staged_paths) == len(set(staged_paths)),
        "duplicate staged object path in transaction",
    )
    for item in record["objects"]:
        expected_staged_path = f".staging/{record['transaction_id']}/{item['path']}"
        require(
            item["staged_path"] == expected_staged_path,
            "staged object path does not match its final path",
        )
        require(
            item["staged_path"] != item["path"],
            "staged object path equals its final path",
        )
    claimed_object_paths: set[str] = set()
    manifest_array_paths = (
        set()
        if target_contracts is None
        else {
            array_path
            for contract in target_contracts.values()
            for array_path in contract["arrays"]
        }
    )
    target_paths = [item["target_path"] for item in record["extents"]]
    require(
        len(target_paths) == len(set(target_paths)),
        "duplicate extent transition",
    )

    for transition in record["extents"]:
        target_path = transition["target_path"]
        required_arrays = set(transition["required_array_paths"])
        dependency_paths = transition["object_paths"]
        require(
            len(dependency_paths) == len(set(dependency_paths)),
            "duplicate extent object dependency",
        )
        require(
            set(dependency_paths) <= set(objects_by_path),
            "extent references a missing object",
        )
        require(
            claimed_object_paths.isdisjoint(dependency_paths),
            "object supports multiple extent targets",
        )
        claimed_object_paths.update(dependency_paths)

        contract = None if target_contracts is None else target_contracts.get(target_path)
        if target_contracts is not None:
            require(contract is not None, "extent target is absent from the frozen manifest")
            require(
                transition["target_kind"] == contract["target_kind"],
                "extent target kind disagrees with the manifest",
            )
            require(
                transition.get("record_schema_id") == contract["record_schema_id"],
                "extent record schema disagrees with the manifest",
            )
            require(
                required_arrays == set(contract["arrays"]),
                "extent required arrays disagree with the manifest",
            )

        dependencies = [objects_by_path[path] for path in dependency_paths]
        for item in dependencies:
            require(
                item["target_path"] == target_path,
                "object target does not match its extent",
            )
            require(
                item["array_path"] in required_arrays,
                "object array is not required by its extent",
            )
            role = item["logical_role"]
            if role in {
                "array_chunk",
                "record_column_chunk",
                "record_validity_chunk",
            }:
                coordinate = item["chunk_coordinate"]
                require(coordinate, "chunk coordinate is empty")
                expected_path = f"{item['array_path']}/c/" + "/".join(
                    str(value) for value in coordinate
                )
                require(
                    item["path"] == expected_path,
                    "chunk path does not match its coordinate",
                )
                require(
                    item["content_kind"] == "zarr_chunk",
                    "chunk role does not name a Zarr chunk",
                )
            elif role == "metadata_snapshot":
                require(
                    item["content_kind"] == "zarr_metadata",
                    "metadata role does not name Zarr metadata",
                )
                require(
                    all(
                        item["path"] != array_path
                        and not item["path"].startswith(f"{array_path}/")
                        and not array_path.startswith(f"{item['path']}/")
                        for array_path in manifest_array_paths
                    ),
                    "metadata snapshot path conflicts with a physical array",
                )
                require(
                    item["path"].startswith(f"metadata/transactions/{record['transaction_id']}/"),
                    "metadata snapshot path is outside its transaction namespace",
                )
            elif role == "index":
                require(
                    item["content_kind"] == "index",
                    "index role does not name an index object",
                )
                require(
                    IDX_OBJECT_PATH.fullmatch(item["path"]) is not None,
                    "index object path is outside the indexes namespace",
                )
            else:
                raise NrfValidationError("unsupported object logical role")

        advancing = transition["after"] > transition["before"]
        if not advancing:
            require(
                transition["seals_target"] and not dependency_paths,
                "non-advancing transition must be objectless seal-only",
            )
            continue
        require(dependency_paths, "advancing extent has no dependent objects")

        for array_path in required_arrays:
            chunk_objects = [
                item
                for item in dependencies
                if item["array_path"] == array_path
                and item["logical_role"]
                in {
                    "array_chunk",
                    "record_column_chunk",
                    "record_validity_chunk",
                }
            ]
            coordinates = [tuple(item["chunk_coordinate"]) for item in chunk_objects]
            require(
                len(coordinates) == len(set(coordinates)),
                "duplicate chunk coordinate for array",
            )
            if contract is None:
                expected_leading = set(
                    range(
                        transition["before"] // transition["chunk_length"],
                        (transition["after"] - 1) // transition["chunk_length"] + 1,
                    )
                )
                require(
                    {coordinate[0] for coordinate in coordinates} == expected_leading,
                    "chunk coordinates do not cover the extent",
                )
            else:
                array_contract = contract["arrays"][array_path]
                expected = expected_chunk_coordinates(
                    before=transition["before"],
                    after=transition["after"],
                    chunk_length=transition["chunk_length"],
                    trailing_chunk_counts=array_contract["trailing_chunk_counts"],
                )
                require(
                    set(coordinates) == expected,
                    "chunk coordinates do not exactly cover the extent",
                )
                for item in chunk_objects:
                    require(
                        item["logical_role"] == array_contract["logical_role"],
                        "chunk logical role disagrees with the manifest",
                    )
                    require(
                        len(item["chunk_coordinate"]) == array_contract["rank"],
                        "chunk coordinate rank disagrees with the manifest",
                    )
                    if contract["target_kind"] == "record_set":
                        require(
                            item["record_schema_id"] == contract["record_schema_id"],
                            "record chunk schema association is invalid",
                        )
                        require(
                            item["column_name"] == array_contract["column_name"],
                            "record chunk column association is invalid",
                        )
                metadata_objects = [
                    item
                    for item in dependencies
                    if item["array_path"] == array_path
                    and item["logical_role"] == "metadata_snapshot"
                ]
                require(
                    len(metadata_objects) == 1,
                    "advancing array requires one metadata snapshot",
                )

    require(
        claimed_object_paths == set(objects_by_path),
        "transaction contains an unassigned object",
    )


def validate_record_checksum(value: dict[str, Any]) -> None:
    require(
        SHA256.fullmatch(value.get("record_checksum", "")) is not None,
        "invalid record checksum encoding",
    )
    require(value["record_checksum"] == record_checksum(value), "record checksum mismatch")


# The journal-record schema is not re-validated by replay_journal (the schema
# layer is exercised separately); these builders construct checksum-valid
# records whose fields satisfy replay's direct accesses, so adversarial tests
# can target a single invariant without triggering unrelated failures.
_PLACEHOLDER_DIGEST = "0" * 64


def _journal_record(sequence: int, kind: str, **fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"sequence": sequence, "kind": kind, **fields}
    record["record_checksum"] = record_checksum(record)
    return record


def _staged_object(
    path: str,
    transaction_id: str,
    *,
    target_path: str | None = None,
    array_path: str | None = None,
    logical_role: str = "array_chunk",
    chunk_coordinate: list[int] | None = None,
    record_schema_id: str | None = None,
    column_name: str | None = None,
) -> dict[str, Any]:
    inferred_array_path, _, encoded_coordinate = path.rpartition("/c/")
    physical_array_path = array_path or inferred_array_path
    coordinate = chunk_coordinate or [int(item) for item in encoded_coordinate.split("/") if item]
    value: dict[str, Any] = {
        "path": path,
        "staged_path": f".staging/{transaction_id}/{path}",
        "byte_length": 16,
        "sha256": _PLACEHOLDER_DIGEST,
        "disposition": "create",
        "content_kind": "zarr_chunk",
        "target_path": target_path or physical_array_path,
        "array_path": physical_array_path,
        "logical_role": logical_role,
        "chunk_coordinate": coordinate,
    }
    if record_schema_id is not None:
        value["record_schema_id"] = record_schema_id
    if column_name is not None:
        value["column_name"] = column_name
    return value


def _extent_transition(
    target_path: str,
    before: int,
    after: int,
    *,
    seals: bool = False,
    chunk: int = 4,
    target_kind: str = "array",
    record_schema_id: str | None = None,
    required_array_paths: list[str] | None = None,
    object_paths: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "target_path": target_path,
        "target_kind": target_kind,
        "before": before,
        "after": after,
        "chunk_length": chunk,
        "seals_target": seals,
        "required_array_paths": required_array_paths or [target_path],
    }
    if record_schema_id is not None:
        value["record_schema_id"] = record_schema_id
    if object_paths is not None:
        value["object_paths"] = object_paths
    return value


def _prepare(
    sequence: int,
    transaction_id: str,
    *,
    previous_committed_transaction_id: str | None,
    objects: list[dict[str, Any]],
    extents: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_extents = []
    for transition in extents:
        normalized = dict(transition)
        normalized.setdefault(
            "object_paths",
            [item["path"] for item in objects if item["target_path"] == normalized["target_path"]],
        )
        normalized_extents.append(normalized)
    return _journal_record(
        sequence,
        "prepare",
        transaction_id=transaction_id,
        previous_committed_transaction_id=previous_committed_transaction_id,
        objects=objects,
        extents=normalized_extents,
    )


def _commit(sequence: int, transaction_id: str, prepare_sequence: int) -> dict[str, Any]:
    return _journal_record(
        sequence, "commit", transaction_id=transaction_id, prepare_sequence=prepare_sequence
    )


def replay_journal(
    records: list[dict[str, Any]],
    *,
    allow_uncommitted_tail: bool,
    manifest: dict[str, Any] | None = None,
    typed_termination_rows: dict[str, dict[str, Any]] | None = None,
    committed_fault_transactions: dict[str, str] | None = None,
) -> ReplayState:
    target_contracts = None if manifest is None else manifest_target_contracts(manifest)
    prepared: dict[str, dict[str, Any]] = {}
    committed: set[str] = set()
    extents: dict[str, int] = {}
    sealed_targets: set[str] = set()
    seen_object_paths: set[str] = set()
    committed_objects: list[dict[str, Any]] = []
    last_committed: str | None = None
    last_committed_number = 0
    max_transaction_number = 0
    journal_sequence = 0
    last_committed_sequence = 0
    terminated = False
    termination_record_id: str | None = None
    termination_kind: str | None = None
    for expected_sequence, record in enumerate(records, start=1):
        require(not terminated, "journal record follows session termination")
        validate_record_checksum(record)
        require(record["sequence"] == expected_sequence, "journal sequence is not monotonic")
        journal_sequence = record["sequence"]
        kind = record["kind"]
        if kind == "prepare":
            transaction_id = record["transaction_id"]
            require(TX_ID.fullmatch(transaction_id) is not None, "invalid transaction ID")
            require(
                transaction_id not in prepared and transaction_id not in committed,
                "duplicate transaction",
            )
            # Transaction IDs are allocated in strictly increasing numeric
            # order; a prepare whose number does not exceed every prior
            # transaction's number is out of allocation order.
            tx_number = transaction_number(transaction_id)
            require(
                tx_number > max_transaction_number,
                "transaction IDs are not monotonic",
            )
            max_transaction_number = tx_number
            require(
                record["previous_committed_transaction_id"] == last_committed,
                "broken transaction chain",
            )
            paths = [item["path"] for item in record["objects"]]
            require(
                seen_object_paths.isdisjoint(paths),
                "transaction reuses an earlier object path",
            )
            seen_object_paths.update(paths)
            for item in record["objects"]:
                require(SHA256.fullmatch(item["sha256"]) is not None, "invalid object checksum")
                require(item["disposition"] == "create", "transaction may not replace an object")
                require(
                    item["staged_path"].startswith(f".staging/{transaction_id}/"),
                    "invalid staged path",
                )
            # A transaction with no staged objects is a seal-only finalization:
            # every extent transition must seal an already-open target without
            # advancing it (after == before). This lets a session seal a target
            # that stayed empty (extent 0) for its whole life, which the
            # finalization rule requires but the previous "extent must increase"
            # rule forbade.
            if not record["objects"]:
                for transition in record["extents"]:
                    require(
                        transition["seals_target"] and transition["after"] == transition["before"],
                        "objectless transaction must only seal targets",
                    )
            for transition in record["extents"]:
                target = transition["target_path"]
                require(target not in sealed_targets, "transaction targets a sealed extent")
                require(
                    transition["before"] == extents.get(target, 0),
                    "extent before does not match committed state",
                )
                require(
                    transition["after"] >= transition["before"],
                    "extent must not decrease",
                )
                chunk_length = transition["chunk_length"]
                require(chunk_length > 0, "chunk length must be positive")
                require(
                    transition["before"] % chunk_length == 0,
                    "open committed extent must be chunk aligned",
                )
                if not transition["seals_target"]:
                    require(
                        transition["after"] > transition["before"],
                        "extent must increase",
                    )
                    require(
                        transition["after"] % chunk_length == 0,
                        "unsealed committed extent must be chunk aligned",
                    )
            validate_prepare_object_ownership(record, target_contracts)
            prepared[transaction_id] = record
        elif kind == "commit":
            transaction_id = record["transaction_id"]
            require(transaction_id in prepared, "commit without prepare")
            require(transaction_id not in committed, "duplicate commit")
            require(
                record["prepare_sequence"] == prepared[transaction_id]["sequence"],
                "commit references wrong prepare",
            )
            # Re-verify the chain at the visibility point: the prepared
            # transaction must still name the current last committed
            # transaction as its predecessor. This rejects out-of-order
            # commits (e.g. commit tx-2 then tx-1) that would otherwise let an
            # earlier transaction become the last committed transaction.
            require(
                prepared[transaction_id]["previous_committed_transaction_id"] == last_committed,
                "commit breaks the previous committed transaction chain",
            )
            tx_number = transaction_number(transaction_id)
            require(
                tx_number > last_committed_number,
                "commits are not in transaction-number order",
            )
            for transition in prepared[transaction_id]["extents"]:
                extents[transition["target_path"]] = transition["after"]
                if transition["seals_target"]:
                    sealed_targets.add(transition["target_path"])
            for item in prepared[transaction_id]["objects"]:
                committed_object = {
                    key: value
                    for key, value in item.items()
                    if key not in {"staged_path", "disposition"}
                }
                committed_objects.append(committed_object)
            committed.add(transaction_id)
            last_committed = transaction_id
            last_committed_number = tx_number
            last_committed_sequence = record["sequence"]
        elif kind == "checkpoint":
            require(
                CHECKPOINT_ID.fullmatch(record["checkpoint_id"]) is not None,
                "invalid checkpoint record",
            )
            require(
                SHA256.fullmatch(record["checkpoint_sha256"]) is not None,
                "invalid checkpoint object checksum",
            )
        elif kind == "termination":
            require(
                record["last_transaction_id"] == last_committed,
                "termination references wrong transaction",
            )
            require(
                record["sequence"] == last_committed_sequence + 1,
                "termination must immediately follow its final commit",
            )
            require(
                last_committed is not None,
                "termination requires a committed final transaction",
            )
            if manifest is not None:
                termination_schemas = [
                    schema
                    for schema in manifest["record_schemas"]
                    if schema["kind"] == "session_termination"
                ]
                require(
                    len(termination_schemas) == 1,
                    "manifest must define exactly one session_termination schema",
                )
                termination_schema = termination_schemas[0]
                termination_target = termination_schema["path"]
                require(
                    extents.get(termination_target) == 1,
                    "session_termination extent must equal one",
                )
                final_transitions = prepared[last_committed]["extents"]
                termination_transitions = [
                    transition
                    for transition in final_transitions
                    if transition["target_path"] == termination_target
                ]
                require(
                    len(termination_transitions) == 1,
                    "final transaction must own the session_termination row",
                )
                termination_transition = termination_transitions[0]
                require(
                    termination_transition["before"] == 0
                    and termination_transition["after"] == 1
                    and termination_transition["seals_target"],
                    "final transaction must append and seal exactly one session_termination row",
                )
                require(
                    typed_termination_rows is not None,
                    "typed session_termination row is required for termination validation",
                )
                require(
                    set(typed_termination_rows) == {record["termination_record_id"]},
                    "journal termination must reference the sole typed termination row",
                )
                typed_row = typed_termination_rows[record["termination_record_id"]]
                for field_name in (
                    "termination_kind",
                    "time_ns",
                    "reason",
                    "fault_id",
                    "last_transaction_id",
                ):
                    require(
                        typed_row[field_name] == record[field_name],
                        f"typed and journal termination disagree on {field_name}",
                    )
                require(
                    typed_row["termination_id"] == record["termination_record_id"],
                    "typed termination ID disagrees with journal reference",
                )
                if record["termination_kind"] == "normal":
                    require(
                        record["fault_id"] is None,
                        "normal termination must not reference a fault",
                    )
                    require(
                        sealed_targets == set(target_contracts),
                        "normal termination requires every appendable target to be sealed",
                    )
                elif record["termination_kind"] == "faulted":
                    require(
                        record["fault_id"] is not None,
                        "faulted termination must reference a fault",
                    )
                if record["fault_id"] is not None:
                    require(
                        committed_fault_transactions is not None
                        and record["fault_id"] in committed_fault_transactions,
                        "termination fault does not resolve to a committed fault",
                    )
                    fault_transaction_id = committed_fault_transactions[record["fault_id"]]
                    require(
                        fault_transaction_id in committed,
                        "termination fault was not committed",
                    )
                    require(
                        transaction_number(fault_transaction_id)
                        < transaction_number(last_committed),
                        "termination fault must precede the final transaction",
                    )
                require(
                    termination_target in sealed_targets,
                    "session_termination target must be sealed",
                )
            termination_record_id = record["termination_record_id"]
            termination_kind = record["termination_kind"]
            terminated = True
        else:
            raise NrfValidationError("unknown journal record kind")
    uncommitted = set(prepared) - committed
    require(allow_uncommitted_tail or not uncommitted, "uncommitted transaction tail")
    return ReplayState(
        extents=extents,
        last_committed_transaction_id=last_committed,
        sealed_targets=sealed_targets,
        journal_sequence=journal_sequence,
        last_committed_sequence=last_committed_sequence,
        committed_objects=committed_objects,
        # The v1 journal-record schema carries no index-extent transitions, so
        # replay cannot advance index build positions; a checkpoint must agree
        # with that empty replay state.
        idx_extents={},
        termination_record_id=termination_record_id,
        termination_kind=termination_kind,
    )


def validate_checkpoint(value: dict[str, Any], replay_state: ReplayState) -> None:
    """Verify a checkpoint equals replay state through its journal sequence.

    ``replay_state`` MUST be the result of replaying the journal prefix ending
    at ``value["journal_sequence"]``; a checkpoint is valid only when every
    state field equals that replay, not just extents and sealed targets.
    """
    validate_record_checksum(value)
    require(value["format"] == "nrf-checkpoint", "invalid checkpoint format")
    require(value["version"]["major"] == 1, "unsupported checkpoint major version")
    require(CHECKPOINT_ID.fullmatch(value["checkpoint_id"]) is not None, "invalid checkpoint ID")
    require(
        value["journal_sequence"] == replay_state.journal_sequence,
        "checkpoint journal sequence does not equal replay state",
    )
    require(
        value["last_committed_transaction_id"] == replay_state.last_committed_transaction_id,
        "checkpoint last committed transaction does not equal replay state",
    )
    require(value["committed_extents"] == replay_state.extents, "checkpoint extent mismatch")
    require(
        set(value["sealed_targets"]) == replay_state.sealed_targets,
        "checkpoint sealed-target mismatch",
    )
    require(
        value["index_extents"] == replay_state.idx_extents,
        "checkpoint index extents do not equal replay state",
    )
    paths = [item["path"] for item in value["committed_objects"]]
    require(len(paths) == len(set(paths)), "duplicate checkpoint object")
    require(
        all(SHA256.fullmatch(item["sha256"]) for item in value["committed_objects"]),
        "invalid checkpoint object checksum",
    )
    for item in value["committed_objects"]:
        if item["logical_role"] == "index":
            require(
                IDX_OBJECT_PATH.fullmatch(item["path"]) is not None,
                "checkpoint index object path is outside the indexes namespace",
            )
    require(
        value["committed_objects"] == replay_state.committed_objects,
        "checkpoint committed objects do not equal replay state",
    )


def validate_recovery_report(
    value: dict[str, Any], extents: dict[str, int], sealed_targets: set[str]
) -> None:
    validate_record_checksum(value)
    require(value["format"] == "nrf-recovery-report", "invalid recovery report format")
    require(value["version"]["major"] == 1, "unsupported recovery report major version")
    require(value["committed_data_modified"] is False, "recovery modified committed data")
    require(value["committed_extents"] == extents, "recovery extent mismatch")
    require(set(value["sealed_targets"]) == sealed_targets, "recovery sealed-target mismatch")
    require(
        not set(value["retained_transaction_ids"]) & set(value["ignored_transaction_ids"]),
        "transaction is both retained and ignored",
    )


def resolve_local_ref(schema: dict[str, Any], reference: str) -> Any:
    require(reference.startswith("#/"), "schema contains a non-local data reference")
    current: Any = schema
    for token in reference[2:].split("/"):
        current = current[token.replace("~1", "/").replace("~0", "~")]
    return current


def walk_json(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for item in value.values():
            values.extend(walk_json(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(walk_json(item))
    return values


def test_machine_readable_schemas_are_versioned_and_self_contained() -> None:
    for name in (
        "manifest.schema.json",
        "journal-record.schema.json",
        "checkpoint.schema.json",
        "recovery-report.schema.json",
        "head.schema.json",
    ):
        schema = json.loads((SPEC / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].endswith(f"/nrf/v1/{name}")
        assert schema.get("type") == "object" or "oneOf" in schema
        for item in walk_json(schema):
            if isinstance(item, dict) and "$ref" in item:
                assert resolve_local_ref(schema, item["$ref"]) is not None


def schema_validator(name: str) -> Draft202012Validator:
    schema = json.loads((SPEC / name).read_text(encoding="utf-8"))
    # format: "date-time" is annotation-only without a format checker, so a
    # malformed value such as "not-a-dateZ" would otherwise pass as long as it
    # ends with Z. A strict UTC/no-leap-second pattern is also declared in the
    # schemas, while the format checker adds calendar/range validation.
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_checked_in_vectors_pass_draft_2020_12_json_schema() -> None:
    schema_validator("manifest.schema.json").validate(VECTORS["manifest"])

    journal = schema_validator("journal-record.schema.json")
    journal_records = [
        *VECTORS["complete_sequence"],
        *VECTORS["recoverable_uncommitted_tail"]["journal_records"],
    ]
    assert len(journal_records) == 26
    for record in journal_records:
        journal.validate(record)
    for records in VECTORS["append_cases"].values():
        for record in records:
            journal.validate(record)
    for case in VECTORS["partial_chunk_cases"].values():
        for record in case["journal_records"]:
            journal.validate(record)

    schema_validator("checkpoint.schema.json").validate(VECTORS["checkpoint"])
    schema_validator("head.schema.json").validate(VECTORS["head"])
    schema_validator("recovery-report.schema.json").validate(
        VECTORS["recoverable_uncommitted_tail"]["expected_report"]
    )


def test_journal_schema_allows_extensions_rejects_unknown_properties() -> None:
    journal = schema_validator("journal-record.schema.json")
    extended = deepcopy(VECTORS["complete_sequence"][0])
    extended["extensions"] = {"future.example": {"value": 1}}
    journal.validate(extended)

    invalid = deepcopy(extended)
    invalid["unexpected"] = True
    errors = list(journal.iter_errors(invalid))
    nested_errors = [nested for error in errors for nested in error.context]
    assert errors
    assert any(error.validator == "unevaluatedProperties" for error in nested_errors)


def test_journal_schema_requires_machine_readable_object_ownership() -> None:
    journal = schema_validator("journal-record.schema.json")
    missing_target = deepcopy(VECTORS["complete_sequence"][0])
    del missing_target["objects"][0]["target_path"]
    with pytest.raises(JsonSchemaValidationError):
        journal.validate(missing_target)

    record_prepare = deepcopy(VECTORS["append_cases"]["experiment_state_and_command"][0])
    record_chunk = next(
        item for item in record_prepare["objects"] if item["logical_role"] == "record_column_chunk"
    )
    del record_chunk["column_name"]
    with pytest.raises(JsonSchemaValidationError):
        journal.validate(record_prepare)


@pytest.mark.parametrize(
    "timestamp",
    [
        "0000-01-01T00:00:00Z",
        "2026-12-31T23:59:60Z",
    ],
)
def test_json_date_time_policy_rejects_unsupported_values(timestamp: str) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["session"]["created_at"] = timestamp
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("manifest.schema.json").validate(manifest)
    with pytest.raises(NrfSemanticValidationError, match="invalid session created_at"):
        validate_manifest_semantics(manifest)

    report = deepcopy(VECTORS["recoverable_uncommitted_tail"]["expected_report"])
    report["created_at"] = timestamp
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("recovery-report.schema.json").validate(report)


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
    ],
)
def test_documents_reject_noncanonical_relative_paths(path: str) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["streams"][0]["data"]["path"] = path
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("manifest.schema.json").validate(manifest)
    with pytest.raises(
        NrfSemanticValidationError,
        match=r"relative(?: POSIX)? path",
    ):
        validate_manifest_semantics(manifest)

    journal_record = deepcopy(VECTORS["complete_sequence"][0])
    journal_record["objects"][0]["path"] = path
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("journal-record.schema.json").validate(journal_record)

    checkpoint = deepcopy(VECTORS["checkpoint"])
    checkpoint["committed_objects"][0]["path"] = path
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("checkpoint.schema.json").validate(checkpoint)

    head = deepcopy(VECTORS["head"])
    extent = head["committed_extents"].pop(next(iter(head["committed_extents"])))
    head["committed_extents"][path] = extent
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("head.schema.json").validate(head)

    report = deepcopy(VECTORS["recoverable_uncommitted_tail"]["expected_report"])
    report["missing_objects"] = [path]
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("recovery-report.schema.json").validate(report)


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
def test_stream_payload_cannot_occupy_nrf_control_paths(path: str) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    move_stream_data_path(
        manifest,
        stream_id="neural",
        path=path,
    )
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("manifest.schema.json").validate(manifest)
    with pytest.raises(
        NrfSemanticValidationError,
        match="stream data path does not match stream ID",
    ):
        validate_manifest_semantics(manifest)


def test_stream_and_record_paths_follow_frozen_layout() -> None:
    manifest = deepcopy(VECTORS["manifest"])
    move_stream_data_path(
        manifest,
        stream_id="neural",
        path="streams/other/data",
    )
    schema_validator("manifest.schema.json").validate(manifest)
    with pytest.raises(
        NrfSemanticValidationError,
        match="stream data path does not match stream ID",
    ):
        validate_manifest_semantics(manifest)

    manifest = deepcopy(VECTORS["manifest"])
    cursor = next(stream for stream in manifest["streams"] if stream["id"] == "cursor")
    old_timestamp_path = cursor["timing"]["timestamps"]["path"]
    new_timestamp_path = "streams/other/timestamps"
    cursor["timing"]["timestamps"]["path"] = new_timestamp_path
    extent = manifest["commit"]["committed_extents"].pop(old_timestamp_path)
    manifest["commit"]["committed_extents"][new_timestamp_path] = extent
    manifest["commit"]["sealed_targets"] = [
        new_timestamp_path if target == old_timestamp_path else target
        for target in manifest["commit"]["sealed_targets"]
    ]
    schema_validator("manifest.schema.json").validate(manifest)
    with pytest.raises(
        NrfSemanticValidationError,
        match="stream timestamp path does not match stream ID",
    ):
        validate_manifest_semantics(manifest)

    manifest = deepcopy(VECTORS["manifest"])
    events = next(schema for schema in manifest["record_schemas"] if schema["id"] == "events-v1")
    events["path"] = "records/events/other-v1"
    schema_validator("manifest.schema.json").validate(manifest)
    with pytest.raises(
        NrfSemanticValidationError,
        match="record schema path does not match kind and ID",
    ):
        validate_manifest_semantics(manifest)


def test_prepare_rejects_duplicate_staged_object_paths() -> None:
    prepare = deepcopy(VECTORS["complete_sequence"][0])
    prepare["objects"][1]["staged_path"] = prepare["objects"][0]["staged_path"]
    prepare["record_checksum"] = record_checksum(prepare)
    schema_validator("journal-record.schema.json").validate(prepare)

    with pytest.raises(
        NrfValidationError,
        match="duplicate staged object path in transaction",
    ):
        replay_journal([prepare], allow_uncommitted_tail=True)


def test_staged_path_derives_from_final_path() -> None:
    prepare = deepcopy(VECTORS["complete_sequence"][0])
    prepare["objects"][0]["staged_path"] = f".staging/{prepare['transaction_id']}/different-object"
    prepare["record_checksum"] = record_checksum(prepare)
    schema_validator("journal-record.schema.json").validate(prepare)

    with pytest.raises(
        NrfValidationError,
        match="staged object path does not match its final path",
    ):
        replay_journal([prepare], allow_uncommitted_tail=True)


def _prepare_with_index_object(path: str) -> dict[str, Any]:
    prepare = deepcopy(VECTORS["complete_sequence"][0])
    transaction_id = prepare["transaction_id"]
    idx_object = {
        "path": path,
        "staged_path": f".staging/{transaction_id}/{path}",
        "byte_length": 16,
        "sha256": _PLACEHOLDER_DIGEST,
        "disposition": "create",
        "content_kind": "index",
        "target_path": "streams/neural/data",
        "array_path": "streams/neural/data",
        "logical_role": "index",
    }
    prepare["objects"].append(idx_object)
    prepare["extents"][0]["object_paths"].append(path)
    prepare["record_checksum"] = record_checksum(prepare)
    return prepare


def test_index_object_uses_reserved_namespace() -> None:
    prepare = _prepare_with_index_object("indexes/neural-samples/by-sample.bin")

    schema_validator("journal-record.schema.json").validate(prepare)
    replay_journal(
        [prepare],
        allow_uncommitted_tail=True,
        manifest=VECTORS["manifest"],
    )


@pytest.mark.parametrize(
    "path",
    [
        "manifest.json",
        "journal/transactions.jsonl",
        "recovery/report.json",
        ".staging/foreign",
        "metadata/not-a-transaction/index.bin",
    ],
)
def test_index_object_cannot_occupy_control_paths(path: str) -> None:
    prepare = _prepare_with_index_object(path)

    with pytest.raises(JsonSchemaValidationError):
        schema_validator("journal-record.schema.json").validate(prepare)
    with pytest.raises(
        NrfValidationError,
        match="index object path is outside the indexes namespace",
    ):
        replay_journal(
            [prepare],
            allow_uncommitted_tail=True,
            manifest=VECTORS["manifest"],
        )


def test_checkpoint_rejects_index_object_outside_reserved_namespace() -> None:
    checkpoint = deepcopy(VECTORS["checkpoint"])
    idx_object = {
        key: value
        for key, value in _prepare_with_index_object("manifest.json")["objects"][-1].items()
        if key not in {"staged_path", "disposition"}
    }
    checkpoint["committed_objects"].append(idx_object)
    checkpoint["record_checksum"] = record_checksum(checkpoint)

    with pytest.raises(JsonSchemaValidationError):
        schema_validator("checkpoint.schema.json").validate(checkpoint)

    replay_state = replay_journal(
        VECTORS["complete_sequence"][: VECTORS["checkpoint"]["journal_sequence"]],
        allow_uncommitted_tail=False,
    )
    replay_state.committed_objects.append(idx_object)
    with pytest.raises(
        NrfValidationError,
        match="checkpoint index object path is outside the indexes namespace",
    ):
        validate_checkpoint(checkpoint, replay_state)


def test_generic_json_transaction_objects_are_not_supported() -> None:
    prepare = deepcopy(VECTORS["complete_sequence"][0])
    prepare["objects"][0]["content_kind"] = "json"
    prepare["record_checksum"] = record_checksum(prepare)
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("journal-record.schema.json").validate(prepare)

    checkpoint = deepcopy(VECTORS["checkpoint"])
    checkpoint["committed_objects"][0]["content_kind"] = "json"
    checkpoint["record_checksum"] = record_checksum(checkpoint)
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("checkpoint.schema.json").validate(checkpoint)


def test_portable_path_key_collapses_cross_platform_aliases() -> None:
    canonical = portable_path_key("streams/neural/data")
    assert portable_path_key("Streams/Neural/Data") == canonical
    assert portable_path_key("streams/neural/data.") == canonical
    assert portable_path_key("streams/neural/data ") == canonical


def test_normative_manifest_and_descriptors_are_valid() -> None:
    schema_validator("manifest.schema.json").validate(VECTORS["manifest"])
    validate_manifest_semantics(VECTORS["manifest"])


def test_discontinuity_is_per_stream_typed_record_set() -> None:
    # Discontinuities are a typed record set (kind "discontinuities"), not an
    # untyped array: each stream's segment_policy references a discontinuities
    # record schema whose path is the committed-extent key and whose fields
    # make every chunk decodable (README section 5).
    manifest = VECTORS["manifest"]
    schemas_by_id = {schema["id"]: schema for schema in manifest["record_schemas"]}
    expected_fields = {
        "discontinuity_id",
        "stream_id",
        "previous_segment_id",
        "next_segment_id",
        "expected_sample_index",
        "actual_sample_index",
        "missing_samples",
        "missing_device_ticks",
        "reason",
        "time_ns",
    }
    for stream in manifest["streams"]:
        schema_id = stream["segment_policy"]["discontinuity_record_schema_id"]
        schema = schemas_by_id[schema_id]
        assert schema["kind"] == "discontinuities"
        assert schema["path"] == f"streams/{stream['id']}/discontinuities"
        assert schema["primary_key"] == "discontinuity_id"
        assert {field["name"] for field in schema["fields"]} == expected_fields
        # The discontinuity path is owned by the record schema, so it is a
        # decodable committed extent rather than an unowned array.
        assert (
            manifest["commit"]["committed_extents"].get(schema["path"], 0)
            == schema["committed_extent"]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["streams"][1]["timing"].pop("timestamp_dtype"),
        lambda value: value["streams"][1]["timing"].pop("timestamp_unit"),
        lambda value: value["streams"][0].update(axes=["channel", "sample"]),
        lambda value: value["streams"][1].update(axes=["channel", "sample"]),
        lambda value: value["streams"][2].pop("feature_set_id"),
        lambda value: value["streams"][2].update(axes=["feature", "observation"]),
        lambda value: value["streams"][0].update(kind="spike"),
        lambda value: value["streams"][0].update(feature_set_id="bandpower-70-200"),
        lambda value: value["record_schemas"][0]["fields"][0].update(
            validity_path="records/events/columns/event_id_valid"
        ),
        lambda value: value["record_schemas"][0].pop("clock_id"),
        lambda value: value["clocks"].clear(),
        lambda value: value["clocks"][0].update(native_id=9007199254740992),
        lambda value: value["clocks"][0].update(offset_ns=-9007199254740992),
        lambda value: value["session"].update(created_at="not-a-dateZ"),
        lambda value: value["session"].update(created_at="2026-13-30T09:00:00Z"),
        lambda value: value["session"].update(created_at="2026-12-31T23:59:60Z"),
    ],
)
def test_manifest_schema_rejects_local_contract_violations(
    mutation: Any,
) -> None:
    value = deepcopy(VECTORS["manifest"])
    mutation(value)

    with pytest.raises(JsonSchemaValidationError):
        schema_validator("manifest.schema.json").validate(value)


def set_stream_discontinuity_schema(
    value: dict[str, Any],
    *,
    stream_id: str,
    record_schema_id: str,
) -> None:
    stream = next(item for item in value["streams"] if item["id"] == stream_id)
    stream["segment_policy"]["discontinuity_record_schema_id"] = record_schema_id


def set_record_field_storage_path(
    value: dict[str, Any],
    *,
    record_schema_id: str,
    field_name: str,
    key: str,
    path: str,
) -> None:
    schema = next(item for item in value["record_schemas"] if item["id"] == record_schema_id)
    field = next(item for item in schema["fields"] if item["name"] == field_name)
    field[key] = path


def move_stream_data_path(
    value: dict[str, Any],
    *,
    stream_id: str,
    path: str,
) -> None:
    stream = next(item for item in value["streams"] if item["id"] == stream_id)
    old_path = stream["data"]["path"]
    committed_extent = value["commit"]["committed_extents"].pop(old_path)
    stream["data"]["path"] = path
    value["commit"]["committed_extents"][path] = committed_extent
    value["commit"]["sealed_targets"] = [
        path if target == old_path else target for target in value["commit"]["sealed_targets"]
    ]


def update_record_primary_key_field(
    value: dict[str, Any],
    *,
    record_schema_id: str,
    updates: dict[str, Any],
) -> None:
    schema = next(item for item in value["record_schemas"] if item["id"] == record_schema_id)
    primary_key_field = next(
        item for item in schema["fields"] if item["name"] == schema["primary_key"]
    )
    primary_key_field.update(updates)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["clocks"].append(deepcopy(value["clocks"][0])),
            "duplicate clock ID",
        ),
        (
            lambda value: value["streams"][0].update(clock_id="missing-clock"),
            "broken clock reference",
        ),
        (
            lambda value: value["record_schemas"][0].update(clock_id="missing-clock"),
            "broken record clock reference",
        ),
        (
            lambda value: update_record_primary_key_field(
                value,
                record_schema_id="events-v1",
                updates={
                    "nullable": True,
                    "validity_path": "records/events/events-v1/validity/event_id",
                },
            ),
            "record primary key must be non-nullable",
        ),
        (
            lambda value: update_record_primary_key_field(
                value,
                record_schema_id="events-v1",
                updates={"shape": [2]},
            ),
            "record primary key must be scalar",
        ),
        (
            lambda value: update_record_primary_key_field(
                value,
                record_schema_id="events-v1",
                updates={
                    "dtype": "float64",
                    "endianness": "little",
                    "codec_ids": ["bytes-le"],
                },
            ),
            "record primary key has unsupported dtype",
        ),
        (
            lambda value: next(
                schema
                for schema in value["record_schemas"]
                if schema["kind"] == "session_termination"
            )["fields"].pop(3),
            "session_termination fields do not match",
        ),
        (
            lambda value: value["commit"]["sealed_targets"].remove(
                "records/session_termination/termination-v1"
            ),
            "session_termination extent and seal state disagree",
        ),
        (
            lambda value: value["streams"][0]["data"].update(shape=[4, 1, 1]),
            "Zarr shape does not match axes",
        ),
        (
            lambda value: value["feature_sets"][0].update(
                feature_names=["70-200:E1"],
                unit_ids=["volt-squared"],
            ),
            "feature descriptor width does not match payload width",
        ),
        (
            lambda value: value["commit"]["committed_extents"].update({"streams/neural/data": 3}),
            "manifest extent disagrees with stream",
        ),
        (
            lambda value: value["commit"]["committed_extents"].update({"unknown/unowned": 4}),
            "committed extent is not owned",
        ),
        (
            lambda value: value["streams"][0]["segment_policy"].update(
                discontinuity_record_schema_id="missing-discontinuity-schema"
            ),
            "broken discontinuity record schema reference",
        ),
        (
            lambda value: value["streams"][0]["segment_policy"].update(
                discontinuity_record_schema_id="events-v1"
            ),
            "segment policy does not reference a discontinuity record schema",
        ),
        (
            lambda value: set_stream_discontinuity_schema(
                value,
                stream_id="neural",
                record_schema_id="discontinuities-cursor-v1",
            ),
            "discontinuity record schema path does not belong to stream",
        ),
        (
            lambda value: set_stream_discontinuity_schema(
                value,
                stream_id="cursor",
                record_schema_id="discontinuities-neural-v1",
            ),
            "discontinuity record schema is shared by multiple streams",
        ),
        (
            lambda value: set_record_field_storage_path(
                value,
                record_schema_id="events-v1",
                field_name="event_id",
                key="array_path",
                path="records/trials/trials-v1/columns/trial_id",
            ),
            "record column path does not match schema field",
        ),
        (
            lambda value: set_record_field_storage_path(
                value,
                record_schema_id="events-v1",
                field_name="event_id",
                key="array_path",
                path="streams/neural/data",
            ),
            "record column path does not match schema field",
        ),
        (
            lambda value: set_record_field_storage_path(
                value,
                record_schema_id="events-v1",
                field_name="label",
                key="validity_path",
                path="records/events/events-v1/columns/event_id",
            ),
            "record validity path does not match schema field",
        ),
        (
            lambda value: set_record_field_storage_path(
                value,
                record_schema_id="events-v1",
                field_name="stream_id",
                key="validity_path",
                path="records/events/events-v1/validity/label",
            ),
            "record validity path does not match schema field",
        ),
        (
            lambda value: value["record_schemas"][1].update(
                path=value["record_schemas"][0]["path"]
            ),
            "duplicate record schema path",
        ),
        (
            lambda value: value["streams"][1]["timing"]["timestamps"].update(
                path=value["streams"][1]["data"]["path"]
            ),
            "stream timestamp path does not match stream ID",
        ),
        (
            lambda value: move_stream_data_path(
                value,
                stream_id="cursor",
                path="streams/cursor",
            ),
            "stream data path does not match stream ID",
        ),
        (
            lambda value: value["streams"][0]["data"].update(fill_value=9007199254740992),
            "outside the I-JSON exact range",
        ),
    ],
)
def test_semantic_validator_rejects_cross_object_violations(
    mutation: Any,
    message: str,
) -> None:
    value = deepcopy(VECTORS["manifest"])
    mutation(value)
    schema_validator("manifest.schema.json").validate(value)

    with pytest.raises(NrfSemanticValidationError, match=message):
        validate_manifest_semantics(value)


def test_semantic_validation_contract_is_versioned_and_external() -> None:
    specification = (SPEC / "semantic-validation.md").read_text(encoding="utf-8")
    assert SEMANTIC_VALIDATION_VERSION == "1.0"
    assert f"Semantic validation version: {SEMANTIC_VALIDATION_VERSION}" in specification
    assert "tools/semantic_validation.py" in specification


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["clocks"].append(deepcopy(value["clocks"][0])), "duplicate clock ID"),
        (lambda value: value["clocks"].clear(), "clock registry is empty"),
        (
            lambda value: value["streams"][0].update(clock_id="missing-clock"),
            "broken clock reference",
        ),
        (
            lambda value: value["record_schemas"][0].pop("clock_id"),
            "record schema is missing clock_id",
        ),
        (lambda value: value["clocks"][0]["rate"].update(denominator=0), "invalid clock rate"),
        (lambda value: value["streams"][0].update(dtype="complex128"), "unsupported dtype"),
        (
            lambda value: value["streams"][0].update(axes=["channel", "sample"]),
            "unsupported payload layout",
        ),
        (lambda value: value["codecs"][0].update(name="unknown"), "unsupported codec"),
        (
            lambda value: value["streams"][0].update(committed_extent=5),
            "invalid stream committed extent",
        ),
        (
            lambda value: value["feature_sets"][0].update(source_stream_id="missing"),
            "broken feature source reference",
        ),
        (
            lambda value: value["feature_sets"][0]["feature_names"].__setitem__(1, "70-200:E1"),
            "feature names",
        ),
        (
            lambda value: value["record_schemas"][0]["fields"][0].update(
                codec_ids=["missing-codec"]
            ),
            "broken record codec reference",
        ),
        (lambda value: value["version"].update(major=2), "unsupported major version"),
        (
            lambda value: value["session"].update(created_at="not-a-dateZ"),
            "invalid session created_at timestamp",
        ),
        (
            lambda value: value["session"].update(created_at="2026-12-31T23:59:60Z"),
            "invalid session created_at timestamp",
        ),
    ],
)
def test_invalid_manifests_and_descriptors_are_rejected(mutation: Any, message: str) -> None:
    value = deepcopy(VECTORS["manifest"])
    mutation(value)
    with pytest.raises(NrfSemanticValidationError, match=message):
        validate_manifest_semantics(value)


def test_unknown_extension_preserved_unknown_root_rejected() -> None:
    value = deepcopy(VECTORS["manifest"])
    value["version"]["minor"] = 7
    value["extensions"] = {"future.example": {"opaque": [1, 2, 3]}}
    schema_validator("manifest.schema.json").validate(value)
    validate_manifest_semantics(value)
    assert value["extensions"]["future.example"]["opaque"] == [1, 2, 3]

    value["future_required_field"] = True
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("manifest.schema.json").validate(value)


def test_complete_transaction_and_checkpoint_sequence_is_valid() -> None:
    records = VECTORS["complete_sequence"]
    # The full sequence (including the trailing checkpoint journal record)
    # replays cleanly to the fifth committed transaction.
    full = replay_journal(
        records,
        allow_uncommitted_tail=False,
        manifest=VECTORS["manifest"],
        typed_termination_rows={
            VECTORS["logical_record_vectors"]["session_termination"]["termination_id"]: VECTORS[
                "logical_record_vectors"
            ]["session_termination"]
        },
    )
    assert full.last_committed_transaction_id == "tx-0000000000000006"
    assert full.extents == VECTORS["manifest"]["commit"]["committed_extents"]
    assert full.termination_record_id == "termination-0001"
    assert full.termination_kind == "normal"
    # A checkpoint is valid only against replay through its own journal-sequence
    # horizon, so replay exactly that prefix before comparing state.
    prefix = replay_journal(
        records[: VECTORS["checkpoint"]["journal_sequence"]],
        allow_uncommitted_tail=False,
        manifest=VECTORS["manifest"],
    )
    validate_checkpoint(VECTORS["checkpoint"], prefix)
    checkpoint_bytes = canonical_json(VECTORS["checkpoint"]).encode("utf-8")
    checkpoint_record = next(record for record in records if record["kind"] == "checkpoint")
    assert checkpoint_record["checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()


def test_typed_row_and_journal_record_share_one_contract() -> None:
    records = VECTORS["complete_sequence"]
    termination = records[-1]
    typed_row = VECTORS["logical_record_vectors"]["session_termination"]
    assert termination["kind"] == "termination"
    assert termination["sequence"] == records[-2]["sequence"] + 1
    assert records[-2]["kind"] == "commit"
    assert records[-2]["transaction_id"] == termination["last_transaction_id"]
    assert termination["termination_record_id"] == typed_row["termination_id"]
    for field_name in (
        "termination_kind",
        "time_ns",
        "reason",
        "fault_id",
        "last_transaction_id",
    ):
        assert termination[field_name] == typed_row[field_name]

    termination_schema = next(
        schema
        for schema in VECTORS["manifest"]["record_schemas"]
        if schema["kind"] == "session_termination"
    )
    assert termination_schema["committed_extent"] == 1
    assert VECTORS["manifest"]["commit"]["committed_extents"][termination_schema["path"]] == 1
    assert termination_schema["path"] in VECTORS["manifest"]["commit"]["sealed_targets"]


def _termination_variant(
    termination_kind: str,
    fault_id: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = deepcopy(VECTORS["complete_sequence"])
    termination = records[-1]
    termination["termination_kind"] = termination_kind
    termination["fault_id"] = fault_id
    termination["record_checksum"] = record_checksum(termination)
    typed_row = deepcopy(VECTORS["logical_record_vectors"]["session_termination"])
    typed_row["termination_kind"] = termination_kind
    typed_row["fault_id"] = fault_id
    return records, {typed_row["termination_id"]: typed_row}


@pytest.mark.parametrize(
    ("termination_kind", "fault_id"),
    [
        ("normal", "fault-0001"),
        ("faulted", None),
    ],
)
def test_journal_schema_rejects_invalid_termination_fault_contract(
    termination_kind: str,
    fault_id: str | None,
) -> None:
    termination = deepcopy(VECTORS["complete_sequence"][-1])
    termination["termination_kind"] = termination_kind
    termination["fault_id"] = fault_id
    termination["record_checksum"] = record_checksum(termination)
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("journal-record.schema.json").validate(termination)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("termination_kind", "aborted"),
        ("time_ns", 2000001),
        ("reason", "different reason"),
        ("fault_id", "fault-0001"),
    ],
)
def test_replay_rejects_typed_and_journal_termination_disagreement(
    field_name: str,
    replacement: Any,
) -> None:
    records = deepcopy(VECTORS["complete_sequence"])
    records[-1][field_name] = replacement
    records[-1]["record_checksum"] = record_checksum(records[-1])
    typed_row = VECTORS["logical_record_vectors"]["session_termination"]
    with pytest.raises(
        NrfValidationError,
        match=f"typed and journal termination disagree on {field_name}",
    ):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows={typed_row["termination_id"]: typed_row},
        )


def test_replay_rejects_termination_without_committed_typed_row() -> None:
    termination = deepcopy(VECTORS["complete_sequence"][-1])
    termination["sequence"] = 1
    termination["last_transaction_id"] = "tx-0000000000000001"
    termination["record_checksum"] = record_checksum(termination)
    typed_row = VECTORS["logical_record_vectors"]["session_termination"]
    with pytest.raises(NrfValidationError, match="references wrong transaction"):
        replay_journal(
            [termination],
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows={typed_row["termination_id"]: typed_row},
        )


def test_replay_rejects_multiple_typed_termination_rows() -> None:
    typed_row = VECTORS["logical_record_vectors"]["session_termination"]
    duplicate = deepcopy(typed_row)
    duplicate["termination_id"] = "termination-0002"
    with pytest.raises(
        NrfValidationError,
        match="sole typed termination row",
    ):
        replay_journal(
            VECTORS["complete_sequence"],
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows={
                typed_row["termination_id"]: typed_row,
                duplicate["termination_id"]: duplicate,
            },
        )


@pytest.mark.parametrize(
    ("termination_kind", "fault_id", "committed_fault_transactions"),
    [
        ("faulted", "fault-0001", {"fault-0001": "tx-0000000000000005"}),
        ("aborted", "fault-0001", {"fault-0001": "tx-0000000000000005"}),
        ("aborted", None, None),
    ],
)
def test_replay_accepts_valid_abnormal_fault_contract(
    termination_kind: str,
    fault_id: str | None,
    committed_fault_transactions: dict[str, str] | None,
) -> None:
    records, typed_rows = _termination_variant(termination_kind, fault_id)

    replay_journal(
        records,
        allow_uncommitted_tail=False,
        manifest=VECTORS["manifest"],
        typed_termination_rows=typed_rows,
        committed_fault_transactions=committed_fault_transactions,
    )


@pytest.mark.parametrize("termination_kind", ["faulted", "aborted"])
def test_replay_rejects_unresolved_termination_fault(
    termination_kind: str,
) -> None:
    records, typed_rows = _termination_variant(
        termination_kind,
        "fault-does-not-exist",
    )

    with pytest.raises(
        NrfValidationError,
        match="does not resolve to a committed fault",
    ):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows=typed_rows,
            committed_fault_transactions={
                "fault-0001": "tx-0000000000000005",
            },
        )


@pytest.mark.parametrize(
    ("fault_transaction_id", "message"),
    [
        ("tx-0000000000000007", "fault was not committed"),
        ("tx-0000000000000006", "fault must precede the final transaction"),
    ],
)
def test_replay_rejects_uncommitted_fault_before_final_commit(
    fault_transaction_id: str,
    message: str,
) -> None:
    records, typed_rows = _termination_variant("faulted", "fault-0001")

    with pytest.raises(NrfValidationError, match=message):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows=typed_rows,
            committed_fault_transactions={"fault-0001": fault_transaction_id},
        )


def test_normal_termination_requires_all_targets_sealed() -> None:
    records = deepcopy(VECTORS["complete_sequence"])
    final_prepare = records[-3]
    removed = next(
        transition
        for transition in final_prepare["extents"]
        if transition["target_path"] == "records/events/events-v1"
    )
    final_prepare["extents"].remove(removed)
    final_prepare["record_checksum"] = record_checksum(final_prepare)
    typed_row = VECTORS["logical_record_vectors"]["session_termination"]
    with pytest.raises(
        NrfValidationError,
        match="normal termination requires every appendable target to be sealed",
    ):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows={typed_row["termination_id"]: typed_row},
        )


def test_termination_must_immediately_follow_final_commit() -> None:
    records = deepcopy(VECTORS["complete_sequence"])
    termination = records.pop()
    intervening_checkpoint = deepcopy(
        next(record for record in records if record["kind"] == "checkpoint")
    )
    intervening_checkpoint["sequence"] = 14
    intervening_checkpoint["record_checksum"] = record_checksum(intervening_checkpoint)
    termination["sequence"] = 15
    termination["record_checksum"] = record_checksum(termination)
    records.extend([intervening_checkpoint, termination])
    typed_row = VECTORS["logical_record_vectors"]["session_termination"]
    with pytest.raises(
        NrfValidationError,
        match="termination must immediately follow its final commit",
    ):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
            typed_termination_rows={typed_row["termination_id"]: typed_row},
        )


def test_faulted_termination_may_leave_unrelated_targets_open() -> None:
    records = deepcopy(VECTORS["complete_sequence"])
    final_prepare = records[-3]
    removed = next(
        transition
        for transition in final_prepare["extents"]
        if transition["target_path"] == "records/events/events-v1"
    )
    final_prepare["extents"].remove(removed)
    final_prepare["record_checksum"] = record_checksum(final_prepare)
    termination = records[-1]
    termination["termination_kind"] = "faulted"
    termination["fault_id"] = "fault-0001"
    termination["reason"] = "queue overflow"
    termination["record_checksum"] = record_checksum(termination)
    typed_row = deepcopy(VECTORS["logical_record_vectors"]["session_termination"])
    typed_row.update(
        termination_kind="faulted",
        fault_id="fault-0001",
        reason="queue overflow",
    )
    state = replay_journal(
        records,
        allow_uncommitted_tail=False,
        manifest=VECTORS["manifest"],
        typed_termination_rows={typed_row["termination_id"]: typed_row},
        committed_fault_transactions={
            "fault-0001": "tx-0000000000000005",
        },
    )
    assert "records/events/events-v1" not in state.sealed_targets
    assert state.termination_kind == "faulted"


@pytest.mark.parametrize(
    "case_name",
    [
        "regular_neural",
        "explicit_behavioral",
        "feature",
        "experiment_state_and_command",
        "discontinuity_and_fault",
    ],
)
def test_required_append_vector_has_valid_record_checksums(case_name: str) -> None:
    records = VECTORS["append_cases"][case_name]
    assert [record["kind"] for record in records] == ["prepare", "commit"]
    for record in records:
        validate_record_checksum(record)


@pytest.mark.parametrize(
    "case_name",
    ["experiment_state_and_command", "discontinuity_and_fault"],
)
def test_record_append_vectors_use_manifest_column_arrays(case_name: str) -> None:
    contracts = manifest_target_contracts(VECTORS["manifest"])
    prepare = VECTORS["append_cases"][case_name][0]

    for transition in prepare["extents"]:
        target_path = transition["target_path"]
        contract = contracts[target_path]
        assert transition["target_kind"] == "record_set"
        assert transition["record_schema_id"] == contract["record_schema_id"]
        assert set(transition["required_array_paths"]) == set(contract["arrays"])

        dependencies = [
            item for item in prepare["objects"] if item["path"] in transition["object_paths"]
        ]
        chunk_objects = [
            item
            for item in dependencies
            if item["logical_role"] in {"record_column_chunk", "record_validity_chunk"}
        ]
        metadata_objects = [
            item for item in dependencies if item["logical_role"] == "metadata_snapshot"
        ]

        assert {item["array_path"] for item in chunk_objects} == set(contract["arrays"])
        assert {item["array_path"] for item in metadata_objects} == set(contract["arrays"])
        assert not any(item["path"].startswith(f"{target_path}/c/") for item in dependencies)

        expected_validity_paths = {
            array_path
            for array_path, array_contract in contract["arrays"].items()
            if array_contract["logical_role"] == "record_validity_chunk"
        }
        actual_validity_paths = {
            item["array_path"]
            for item in chunk_objects
            if item["logical_role"] == "record_validity_chunk"
        }
        assert actual_validity_paths == expected_validity_paths


def test_extent_cannot_be_backed_by_unrelated_objects() -> None:
    records = deepcopy(VECTORS["append_cases"]["regular_neural"])
    prepare = records[0]
    for item in prepare["objects"]:
        item["target_path"] = "unrelated/target"
    prepare["record_checksum"] = record_checksum(prepare)

    with pytest.raises(NrfValidationError, match="object target does not match"):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
        )


def _isolated_prepare_case(case_name: str) -> dict[str, Any]:
    prepare = deepcopy(VECTORS["append_cases"][case_name][0])
    prepare["sequence"] = 1
    prepare["previous_committed_transaction_id"] = None
    prepare["record_checksum"] = record_checksum(prepare)
    return prepare


def test_advancing_record_set_requires_own_objects() -> None:
    prepare = _isolated_prepare_case("experiment_state_and_command")
    missing_transition = prepare["extents"][1]
    missing_paths = set(missing_transition["object_paths"])
    prepare["objects"] = [item for item in prepare["objects"] if item["path"] not in missing_paths]
    missing_transition["object_paths"] = []
    prepare["record_checksum"] = record_checksum(prepare)

    with pytest.raises(NrfValidationError, match="advancing extent has no dependent objects"):
        replay_journal(
            [prepare],
            allow_uncommitted_tail=True,
            manifest=VECTORS["manifest"],
        )


def test_record_set_requires_manifest_columns_and_validity() -> None:
    prepare = _isolated_prepare_case("experiment_state_and_command")
    transition = prepare["extents"][0]
    omitted_array = transition["required_array_paths"][-1]
    omitted_paths = {
        item["path"]
        for item in prepare["objects"]
        if item["target_path"] == transition["target_path"] and item["array_path"] == omitted_array
    }
    prepare["objects"] = [item for item in prepare["objects"] if item["path"] not in omitted_paths]
    transition["required_array_paths"].remove(omitted_array)
    transition["object_paths"] = [
        path for path in transition["object_paths"] if path not in omitted_paths
    ]
    prepare["record_checksum"] = record_checksum(prepare)

    with pytest.raises(NrfValidationError, match="required arrays disagree with the manifest"):
        replay_journal(
            [prepare],
            allow_uncommitted_tail=True,
            manifest=VECTORS["manifest"],
        )


def test_nullable_record_column_requires_validity_chunk_coverage() -> None:
    prepare = _isolated_prepare_case("experiment_state_and_command")
    transition = prepare["extents"][0]
    validity_object = next(
        item
        for item in prepare["objects"]
        if item["target_path"] == transition["target_path"]
        and item["logical_role"] == "record_validity_chunk"
    )
    prepare["objects"].remove(validity_object)
    transition["object_paths"].remove(validity_object["path"])
    prepare["record_checksum"] = record_checksum(prepare)

    with pytest.raises(NrfValidationError, match="chunk coordinates do not exactly cover"):
        replay_journal(
            [prepare],
            allow_uncommitted_tail=True,
            manifest=VECTORS["manifest"],
        )


def test_chunk_coordinates_must_exactly_cover_extent() -> None:
    records = deepcopy(VECTORS["append_cases"]["regular_neural"])
    prepare = records[0]
    transition = prepare["extents"][0]
    chunk = next(item for item in prepare["objects"] if item["logical_role"] == "array_chunk")
    old_path = chunk["path"]
    chunk["chunk_coordinate"][0] = 1
    chunk["path"] = f"{chunk['array_path']}/c/1/0"
    chunk["staged_path"] = f".staging/{prepare['transaction_id']}/{chunk['path']}"
    transition["object_paths"] = [
        chunk["path"] if path == old_path else path for path in transition["object_paths"]
    ]
    prepare["record_checksum"] = record_checksum(prepare)

    with pytest.raises(NrfValidationError, match="chunk coordinates do not exactly cover"):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
        )


def test_metadata_snapshot_path_cannot_overlap_physical_array() -> None:
    records = deepcopy(VECTORS["append_cases"]["regular_neural"])
    prepare = records[0]
    transition = prepare["extents"][0]
    metadata = next(
        item for item in prepare["objects"] if item["logical_role"] == "metadata_snapshot"
    )
    old_path = metadata["path"]
    metadata["path"] = transition["required_array_paths"][0]
    metadata["staged_path"] = f".staging/{prepare['transaction_id']}/{metadata['path']}"
    transition["object_paths"] = [
        metadata["path"] if path == old_path else path for path in transition["object_paths"]
    ]
    prepare["record_checksum"] = record_checksum(prepare)

    with pytest.raises(NrfValidationError, match="metadata snapshot path conflicts"):
        replay_journal(
            records,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
        )


def test_prepare_rejects_unassigned_and_multiply_assigned_objects() -> None:
    unassigned = deepcopy(VECTORS["append_cases"]["regular_neural"])
    prepare = unassigned[0]
    extra = deepcopy(
        next(item for item in prepare["objects"] if item["logical_role"] == "array_chunk")
    )
    extra["path"] = f"{extra['array_path']}/c/9/0"
    extra["staged_path"] = f".staging/{prepare['transaction_id']}/{extra['path']}"
    extra["chunk_coordinate"] = [9, 0]
    prepare["objects"].append(extra)
    prepare["record_checksum"] = record_checksum(prepare)
    with pytest.raises(NrfValidationError, match="unassigned object"):
        replay_journal(
            unassigned,
            allow_uncommitted_tail=False,
            manifest=VECTORS["manifest"],
        )

    prepare = _isolated_prepare_case("experiment_state_and_command")
    prepare["extents"][1]["object_paths"].append(prepare["extents"][0]["object_paths"][0])
    prepare["record_checksum"] = record_checksum(prepare)
    with pytest.raises(NrfValidationError, match="object supports multiple extent targets"):
        replay_journal(
            [prepare],
            allow_uncommitted_tail=True,
            manifest=VECTORS["manifest"],
        )


def test_discontinuity_and_fault_vectors_are_explicit() -> None:
    logical = VECTORS["logical_record_vectors"]
    assert logical["experiment_state"]["previous_state"] == "inter_trial"
    assert logical["experiment_state"]["new_state"] == "move"
    assert logical["command"]["command_type"] == "external_velocity"
    assert logical["command"]["value"] == [1, -1]
    assert logical["discontinuity"]["previous_segment_id"] == "segment-0001"
    assert logical["discontinuity"]["next_segment_id"] == "segment-0002"
    assert logical["fault"]["status"] == "queue_overflow"


def test_partial_append_and_crash_preserve_committed_bytes() -> None:
    cases = VECTORS["partial_chunk_cases"]
    smaller = cases["append_smaller_than_chunk"]
    assert smaller["journal_records"] == []
    assert smaller["writer_tail_extent"] == 2
    assert smaller["expected_committed_extent"] == 0

    filled = cases["multiple_appends_fill_one_chunk"]
    filled_state = replay_journal(filled["journal_records"], allow_uncommitted_tail=False)
    assert filled_state.extents == {"streams/example/data": 4}
    assert filled_state.sealed_targets == set()
    assert {item["content_kind"] for item in filled["journal_records"][0]["objects"]} == {
        "zarr_chunk",
        "zarr_metadata",
    }
    assert all(item["disposition"] == "create" for item in filled["journal_records"][0]["objects"])

    before_commit = cases["crash_before_commit"]
    before_state = replay_journal(before_commit["journal_records"], allow_uncommitted_tail=True)
    assert before_state.extents == {}
    assert before_state.last_committed_transaction_id is None
    assert before_state.sealed_targets == set()

    after_visibility = cases["crash_after_object_visibility_before_commit"]
    after_state = replay_journal(after_visibility["journal_records"], allow_uncommitted_tail=True)
    assert after_state.extents == {"streams/example/data": 4}
    assert after_state.last_committed_transaction_id == "tx-0000000000000001"
    assert after_state.sealed_targets == set()
    assert after_visibility["expected_committed_extent"] == 4
    assert after_visibility["previous_committed_objects_modified"] is False
    assert set(after_visibility["previous_committed_objects"]).isdisjoint(
        after_visibility["orphan_objects_ignored"]
    )


def test_final_partial_chunk_is_committed_once_and_seals_target() -> None:
    cases = VECTORS["partial_chunk_cases"]
    records = [
        *cases["multiple_appends_fill_one_chunk"]["journal_records"],
        *cases["final_partial_chunk"]["journal_records"],
    ]
    state = replay_journal(records, allow_uncommitted_tail=False)
    assert state.last_committed_transaction_id == "tx-0000000000000002"
    assert state.extents == {"streams/example/data": 6}
    assert state.sealed_targets == {"streams/example/data"}
    assert cases["final_partial_chunk"]["session_terminated"] is True
    assert cases["final_partial_chunk"]["further_append_allowed"] is False


def test_empty_target_seals_with_objectless_transaction() -> None:
    # A target that stays at extent 0 for the whole session (e.g. a record set
    # registered for an experiment that never produced that record kind) must
    # still be sealed at normal finalization. A seal-only extent transition
    # (before == after == 0, seals_target true) with no staged objects expresses
    # that without inventing a chunk (README section 8.1).
    cases = VECTORS["partial_chunk_cases"]
    case = cases["empty_target_seal"]
    assert case["journal_records"][0]["objects"] == []
    state = replay_journal(case["journal_records"], allow_uncommitted_tail=False)
    assert state.last_committed_transaction_id == "tx-0000000000000001"
    assert state.extents == {"streams/example/data": 0}
    assert state.sealed_targets == {"streams/example/data"}
    assert case["session_terminated"] is True
    assert case["further_append_allowed"] is False

    # A further append to the now-sealed empty target is rejected, whether or
    # not it carries objects: the target is sealed and may never advance.
    sealed = case["journal_records"]
    append_after_seal = _prepare(
        3,
        "tx-0000000000000002",
        previous_committed_transaction_id="tx-0000000000000001",
        objects=[_staged_object("streams/example/data/c/0", "tx-0000000000000002")],
        extents=[_extent_transition("streams/example/data", 0, 4)],
    )
    with pytest.raises(NrfValidationError, match="sealed extent"):
        replay_journal([*sealed[:-1], append_after_seal], allow_uncommitted_tail=True)
    # An objectless transaction that tries to advance an empty target without
    # sealing is rejected: no objects means every transition must be a seal.
    phantom = _prepare(
        3,
        "tx-0000000000000002",
        previous_committed_transaction_id="tx-0000000000000001",
        objects=[],
        extents=[_extent_transition("streams/example/data", 0, 4, seals=False)],
    )
    with pytest.raises(NrfValidationError, match="objectless transaction must only seal targets"):
        replay_journal([*sealed[:-1], phantom], allow_uncommitted_tail=True)


def test_extent_replacement_and_append_after_seal_are_rejected() -> None:
    cases = VECTORS["partial_chunk_cases"]
    unaligned = deepcopy(cases["multiple_appends_fill_one_chunk"]["journal_records"])
    unaligned[0]["extents"][0]["after"] = 2
    unaligned[0]["record_checksum"] = record_checksum(unaligned[0])
    with pytest.raises(NrfValidationError, match="unsealed committed extent"):
        replay_journal(unaligned, allow_uncommitted_tail=False)

    replacement = deepcopy(cases["multiple_appends_fill_one_chunk"]["journal_records"][0])
    replacement["objects"][0]["disposition"] = "replace"
    with pytest.raises(JsonSchemaValidationError):
        schema_validator("journal-record.schema.json").validate(replacement)

    committed_and_sealed = [
        *cases["multiple_appends_fill_one_chunk"]["journal_records"],
        *cases["final_partial_chunk"]["journal_records"][:-1],
    ]
    append_after_seal = deepcopy(cases["final_partial_chunk"]["journal_records"][0])
    append_after_seal["sequence"] = 5
    append_after_seal["transaction_id"] = "tx-0000000000000003"
    append_after_seal["previous_committed_transaction_id"] = "tx-0000000000000002"
    for item in append_after_seal["objects"]:
        item["path"] = item["path"].replace("tx-0000000000000002", "tx-0000000000000003")
        item["staged_path"] = item["staged_path"].replace(
            "tx-0000000000000002", "tx-0000000000000003"
        )
    append_after_seal["objects"][0]["path"] = "streams/example/data/c/2"
    append_after_seal["objects"][0]["staged_path"] = (
        ".staging/tx-0000000000000003/streams/example/data/c/2"
    )
    append_after_seal["objects"][0]["chunk_coordinate"] = [2]
    append_after_seal["extents"][0]["object_paths"] = [
        item["path"] for item in append_after_seal["objects"]
    ]
    append_after_seal["extents"][0].update(before=6, after=8, seals_target=False)
    append_after_seal["record_checksum"] = record_checksum(append_after_seal)
    with pytest.raises(NrfValidationError, match="sealed extent"):
        replay_journal(
            [*committed_and_sealed, append_after_seal],
            allow_uncommitted_tail=True,
        )


def test_recoverable_uncommitted_tail_is_visible_only_in_report() -> None:
    case = VECTORS["recoverable_uncommitted_tail"]
    state = replay_journal(case["journal_records"], allow_uncommitted_tail=True)
    assert state.last_committed_transaction_id == "tx-0000000000000005"
    assert state.extents["streams/neural/data"] == 4
    report = case["expected_report"]
    validate_recovery_report(report, state.extents, state.sealed_targets)
    assert report["ignored_transaction_ids"] == ["tx-0000000000000006"]
    assert report["committed_data_modified"] is False


def test_non_monotonic_and_malformed_journals_are_rejected() -> None:
    records = deepcopy(VECTORS["complete_sequence"])
    records[1]["sequence"] = 7
    records[1]["record_checksum"] = record_checksum(records[1])
    with pytest.raises(NrfValidationError, match="sequence is not monotonic"):
        replay_journal(records, allow_uncommitted_tail=False)

    records = deepcopy(VECTORS["complete_sequence"])
    records[0], records[1] = records[1], records[0]
    records[0]["sequence"] = 1
    records[1]["sequence"] = 2
    for record in records[:2]:
        record["record_checksum"] = record_checksum(record)
    with pytest.raises(NrfValidationError, match="commit without prepare"):
        replay_journal(records, allow_uncommitted_tail=False)


def test_invalid_extent_transition_and_record_checksum_are_rejected() -> None:
    records = deepcopy(VECTORS["complete_sequence"])
    records[2]["extents"][0]["before"] = 9
    records[2]["record_checksum"] = record_checksum(records[2])
    with pytest.raises(NrfValidationError, match="extent before"):
        replay_journal(records, allow_uncommitted_tail=False)

    records = deepcopy(VECTORS["complete_sequence"])
    records[0]["record_checksum"] = "0" * 64
    with pytest.raises(NrfValidationError, match="record checksum mismatch"):
        replay_journal(records, allow_uncommitted_tail=False)


def test_invalid_checkpoint_and_recovery_report_are_rejected() -> None:
    records = VECTORS["complete_sequence"]
    prefix = replay_journal(
        records[: VECTORS["checkpoint"]["journal_sequence"]], allow_uncommitted_tail=False
    )
    checkpoint = deepcopy(VECTORS["checkpoint"])
    checkpoint["committed_extents"]["streams/neural/data"] = 99
    checkpoint["record_checksum"] = record_checksum(checkpoint)
    with pytest.raises(NrfValidationError, match="checkpoint extent mismatch"):
        validate_checkpoint(checkpoint, prefix)

    extents = VECTORS["manifest"]["commit"]["committed_extents"]
    report = deepcopy(VECTORS["recoverable_uncommitted_tail"]["expected_report"])
    report["committed_data_modified"] = True
    report["record_checksum"] = record_checksum(report)
    with pytest.raises(NrfValidationError, match="modified committed data"):
        validate_recovery_report(report, extents, set())


def test_out_of_order_commit_breaks_transaction_chain() -> None:
    # Spec section 8: a transaction "commits in transaction-number order" and
    # each prepare names the previous committed transaction. With two
    # independent arrays, prepare tx-1 then prepare tx-2 (both name None, since
    # nothing is committed yet) is accepted at prepare time, but committing
    # tx-2 then tx-1 must be rejected: the second commit would otherwise let an
    # earlier transaction become the last committed transaction.
    records = [
        _prepare(
            1,
            "tx-0000000000000001",
            previous_committed_transaction_id=None,
            objects=[_staged_object("streams/a/data/c/0/0", "tx-0000000000000001")],
            extents=[_extent_transition("streams/a/data", 0, 4)],
        ),
        _prepare(
            2,
            "tx-0000000000000002",
            previous_committed_transaction_id=None,
            objects=[_staged_object("streams/b/data/c/0/0", "tx-0000000000000002")],
            extents=[_extent_transition("streams/b/data", 0, 4)],
        ),
        _commit(3, "tx-0000000000000002", prepare_sequence=2),
        _commit(4, "tx-0000000000000001", prepare_sequence=1),
    ]
    with pytest.raises(
        NrfValidationError, match="commit breaks the previous committed transaction chain"
    ):
        replay_journal(records, allow_uncommitted_tail=False)


def test_non_monotonic_transaction_ids_are_rejected() -> None:
    # Transaction IDs are allocated in strictly increasing numeric order, so a
    # prepare whose number does not exceed every earlier transaction's number
    # is rejected at prepare time even when its chain field is consistent.
    records = [
        _prepare(
            1,
            "tx-0000000000000001",
            previous_committed_transaction_id=None,
            objects=[_staged_object("streams/a/data/c/0/0", "tx-0000000000000001")],
            extents=[_extent_transition("streams/a/data", 0, 4)],
        ),
        _commit(2, "tx-0000000000000001", prepare_sequence=1),
        _prepare(
            3,
            "tx-0000000000000003",
            previous_committed_transaction_id="tx-0000000000000001",
            objects=[_staged_object("streams/b/data/c/0/0", "tx-0000000000000003")],
            extents=[_extent_transition("streams/b/data", 0, 4)],
        ),
        _commit(4, "tx-0000000000000003", prepare_sequence=3),
        _prepare(
            5,
            "tx-0000000000000002",
            previous_committed_transaction_id="tx-0000000000000003",
            objects=[_staged_object("streams/c/data/c/0/0", "tx-0000000000000002")],
            extents=[_extent_transition("streams/c/data", 0, 4)],
        ),
    ]
    with pytest.raises(NrfValidationError, match="transaction IDs are not monotonic"):
        replay_journal(records, allow_uncommitted_tail=True)


def test_cross_transaction_object_path_reuse_is_rejected() -> None:
    # Spec section 8: object paths are unique across transactions. A second
    # transaction that reuses a path already promoted by an earlier committed
    # transaction is rejected at prepare time.
    records = [
        _prepare(
            1,
            "tx-0000000000000001",
            previous_committed_transaction_id=None,
            objects=[_staged_object("streams/a/data/c/0/0", "tx-0000000000000001")],
            extents=[_extent_transition("streams/a/data", 0, 4)],
        ),
        _commit(2, "tx-0000000000000001", prepare_sequence=1),
        _prepare(
            3,
            "tx-0000000000000002",
            previous_committed_transaction_id="tx-0000000000000001",
            objects=[_staged_object("streams/a/data/c/0/0", "tx-0000000000000002")],
            extents=[_extent_transition("streams/a/data", 4, 8)],
        ),
    ]
    with pytest.raises(NrfValidationError, match="transaction reuses an earlier object path"):
        replay_journal(records, allow_uncommitted_tail=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cp: cp.__setitem__("journal_sequence", 9), "checkpoint journal sequence"),
        (
            lambda cp: cp.__setitem__("last_committed_transaction_id", "tx-0000000000000004"),
            "checkpoint last committed transaction",
        ),
        (
            lambda cp: cp["committed_extents"].__setitem__("streams/neural/data", 99),
            "checkpoint extent mismatch",
        ),
        (
            lambda cp: cp["sealed_targets"].append("streams/neural/data"),
            "checkpoint sealed-target mismatch",
        ),
        (lambda cp: cp["committed_objects"].pop(), "checkpoint committed objects"),
        (
            lambda cp: cp["index_extents"].__setitem__("indexes/foo", 1),
            "checkpoint index extents",
        ),
    ],
)
def test_checkpoint_must_equal_replay_state(mutation: Any, message: str) -> None:
    # Spec section 9: a checkpoint "is valid only when its checksum is valid
    # and its state equals replay through its journal sequence." Each state
    # field is compared independently, so a checkpoint that lies about any one
    # of journal sequence, last transaction, extents, sealed targets, index
    # extents, or committed objects is rejected.
    records = VECTORS["complete_sequence"]
    prefix = replay_journal(
        records[: VECTORS["checkpoint"]["journal_sequence"]], allow_uncommitted_tail=False
    )
    checkpoint = deepcopy(VECTORS["checkpoint"])
    mutation(checkpoint)
    # Recompute the record checksum so the failure is attributed to the state
    # mismatch, not to a stale checksum.
    checkpoint["record_checksum"] = record_checksum(checkpoint)
    with pytest.raises(NrfValidationError, match=message):
        validate_checkpoint(checkpoint, prefix)


def test_head_pointer_is_consistent_with_replay_and_manifest() -> None:
    # Spec section 8.5: head.json is a rebuildable cache of the current
    # committed state. It must not lead replayed journal state, and its
    # last_checkpoint_id must reference a checksum-valid checkpoint journal
    # record. It carries no record_checksum. The checked-in vector is a clean
    # (no-crash) session, so in this canonical case head.json and the manifest
    # commit cache agree; the spec does not require that agreement at a crash
    # boundary (see test_head_cache_may_lag_manifest_after_crash).
    head = VECTORS["head"]
    assert "record_checksum" not in head
    manifest_commit = VECTORS["manifest"]["commit"]
    assert head["journal_sequence"] == manifest_commit["journal_sequence"]
    assert head["last_transaction_id"] == manifest_commit["last_transaction_id"]
    assert head["last_checkpoint_id"] == manifest_commit["last_checkpoint_id"]
    assert head["committed_extents"] == manifest_commit["committed_extents"]
    assert head["sealed_targets"] == manifest_commit["sealed_targets"]
    # head.json must not lead replay: the full-sequence replay state equals
    # the cache. (The checkpoint journal record is stateless, so replaying the
    # full sequence yields the same committed state as the cache's horizon.)
    state = replay_journal(VECTORS["complete_sequence"], allow_uncommitted_tail=False)
    assert head["journal_sequence"] == state.journal_sequence
    assert head["last_transaction_id"] == state.last_committed_transaction_id
    assert head["committed_extents"] == state.extents
    assert set(head["sealed_targets"]) == state.sealed_targets
    checkpoint_records = [
        record
        for record in VECTORS["complete_sequence"]
        if record["kind"] == "checkpoint" and record["checkpoint_id"] == head["last_checkpoint_id"]
    ]
    assert len(checkpoint_records) == 1


def test_head_cache_may_lag_manifest_after_crash() -> None:
    # Spec section 8.5: head.json and manifest.json are independent
    # non-authoritative caches; the update protocol replaces each by a single
    # rename, and a crash between the renames may leave them at different journal
    # sequences. The spec tolerates this: each cache only needs to not lead
    # replayed journal state. A head cache that lags the manifest cache is a valid
    # cache of an earlier replay horizon even though it diverges from manifest.
    records = VECTORS["complete_sequence"]
    manifest_commit = VECTORS["manifest"]["commit"]
    lagging_horizon = 8  # after the tx-4 commit, before the tx-5 commit
    lagging_state = replay_journal(records[:lagging_horizon], allow_uncommitted_tail=False)
    lagging_head = {
        "format": "nrf-head",
        "version": {"major": 1, "minor": 0},
        "journal_sequence": lagging_state.journal_sequence,
        "last_transaction_id": lagging_state.last_committed_transaction_id,
        "last_checkpoint_id": None,
        "committed_extents": lagging_state.extents,
        "sealed_targets": sorted(lagging_state.sealed_targets),
    }
    schema_validator("head.schema.json").validate(lagging_head)
    # The lagging head cache diverges from the manifest cache but does not lead it.
    assert lagging_head["journal_sequence"] < manifest_commit["journal_sequence"]
    assert lagging_head["last_transaction_id"] != manifest_commit["last_transaction_id"]
    assert lagging_head["last_transaction_id"] == "tx-0000000000000004"


def test_standalone_descriptors_mirror_manifest_registry() -> None:
    # Spec section 8.3: each feature_sets/<feature-set-id>.json file is a
    # byte-for-value copy of the manifest feature_sets entry with the same id;
    # the manifest registry is authoritative. The standalone descriptors are
    # not transaction objects and must not diverge from the manifest.
    manifest_feature_sets = {
        feature_set["id"]: feature_set for feature_set in VECTORS["manifest"]["feature_sets"]
    }
    standalone = VECTORS["feature_set_files"]
    assert set(standalone) == set(manifest_feature_sets)
    for feature_set_id, descriptor in standalone.items():
        assert descriptor == manifest_feature_sets[feature_set_id]


def test_registries_are_frozen_before_first_transaction() -> None:
    # Spec section 8: all registries are frozen before the first committed
    # transaction. The writer registers every clock, unit, channel, electrode,
    # schema, stream, feature set, codec, and record schema, freezes the
    # session schema, then begins appending; after the first commit no registry
    # entry may be added. The journal carries no registry content, so the
    # manifest written before tx-1 already contains every descriptor needed to
    # decode any committed extent. This reconstructs that pre-tx-1 manifest (all
    # registries, empty commit cache and zero extents) and proves it owns every
    # path the journal later commits to: a manifest cache that never advanced
    # past the pre-tx-1 rename still decodes the whole replayed session, so a
    # crash between a later commit and the manifest rename can never leave the
    # journal pointing at an undescribed target.
    frozen = deepcopy(VECTORS["manifest"])
    frozen["commit"] = {
        "journal_sequence": 0,
        "last_transaction_id": None,
        "last_checkpoint_id": None,
        "committed_extents": {},
        "sealed_targets": [],
    }
    for stream in frozen["streams"]:
        stream["committed_extent"] = 0
    for schema in frozen["record_schemas"]:
        schema["committed_extent"] = 0
    # The pre-tx-1 manifest is a valid registry snapshot: schema-valid and
    # semantically valid with empty committed state.
    schema_validator("manifest.schema.json").validate(frozen)
    validate_manifest_semantics(frozen)
    assert frozen["commit"]["journal_sequence"] == 0
    assert frozen["commit"]["committed_extents"] == {}
    # Every descriptor path the manifest can own.
    frozen_owners: set[str] = set()
    for stream in frozen["streams"]:
        frozen_owners.add(stream["data"]["path"])
        if stream["timing"]["mode"] == "explicit":
            frozen_owners.add(stream["timing"]["timestamps"]["path"])
    for schema in frozen["record_schemas"]:
        frozen_owners.add(schema["path"])
    # Replay the full session; every committed extent is owned by a descriptor
    # already present before the first commit. This is the freeze guarantee: the
    # set of owners did not grow across the session, so a stale pre-tx-1 manifest
    # paired with journal replay decodes every committed path.
    state = replay_journal(VECTORS["complete_sequence"], allow_uncommitted_tail=False)
    assert set(state.extents) <= frozen_owners


def test_canonical_json_and_payload_checksums_are_deterministic() -> None:
    canonical_vectors = VECTORS["canonical_json"]
    assert canonical_vectors["algorithm"] == "rfc8785"
    assert {case["name"] for case in canonical_vectors["cases"]} == {
        "ascii_integer_baseline",
        "rfc8785_primitives",
        "utf16_property_order",
        "numeric_boundaries",
        "nested_property_order",
    }
    for case in canonical_vectors["cases"]:
        canonical = canonical_json_bytes(case["input"])
        assert canonical.decode("utf-8") == case["canonical_utf8"]
        assert hashlib.sha256(canonical).hexdigest() == case["sha256"]

    by_name = {case["name"]: case for case in canonical_vectors["cases"]}
    assert by_name["rfc8785_primitives"]["canonical_utf8"] == (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,'
        '1e+30,4.5,0.002,1e-27],"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert by_name["utf16_property_order"]["canonical_utf8"].startswith(
        '{"\\r":"Carriage Return","1":"One","\u0080":"Control","ö":'
    )
    assert by_name["numeric_boundaries"]["canonical_utf8"] == (
        "[0,5e-324,-5e-324,1e+30,1e-7,0.000001,9007199254740991,-9007199254740991]"
    )

    for payload in VECTORS["payload_vectors"]:
        raw = bytes.fromhex(payload["payload_hex"])
        assert hashlib.sha256(raw).hexdigest() == payload["sha256"]


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        9007199254740992,
        -9007199254740992,
        "\ud800",
    ],
)
def test_jcs_rejects_non_ijson_domains(value: Any) -> None:
    with pytest.raises(rfc8785.CanonicalizationError):
        canonical_json_bytes(value)
