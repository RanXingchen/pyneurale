#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The NRF v1 transaction journal: record construction and JSONL storage.

Specification reference: ``README.md`` sections 7 and 8.

The journal is the authority for visibility. ``manifest.json`` and
``journal/head.json`` are caches that may lag it and must never lead it, so
everything here is written before the corresponding cache is refreshed.

Two storage rules drive the implementation:

* A journal line is canonical JSON plus exactly one LF. A partial line -- the
  normal outcome of a crash mid-append -- is invalid and invisible, so reading
  stops at the last complete line rather than trying to repair the tail.
* The complete, checksum-valid ``commit`` line is the single logical visibility
  point. Data becomes readable at that byte, not when its chunks are promoted.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._canonical import encode_journal_line, sign_record, verify_record_checksum
from ._errors import NrfSchemaError
from ._paths import staged_path as _staged_path
from ._schemas import JOURNAL_RECORD_SCHEMA, validate_document

PREPARE = "prepare"
COMMIT = "commit"
CHECKPOINT = "checkpoint"
TERMINATION = "termination"

TERMINATION_KINDS = frozenset({"normal", "aborted", "faulted"})


@dataclass(frozen=True, slots=True)
class JournalTail:
    """What reading a journal file found beyond its last usable record.

    ``partial_line_bytes`` is the length of an unterminated trailing line. It is
    an expected crash outcome, not corruption: the writer had not finished the
    record, so it never became visible.

    ``invalid_schema_at`` is a complete, correctly signed line that is not a
    well-formed journal record. Its checksum proves only that the bytes are the
    bytes someone signed, not that they mean anything.
    """

    partial_line_bytes: int = 0
    invalid_checksum_at: int | None = None
    invalid_schema_at: int | None = None
    schema_error: str = ""

    @property
    def clean(self) -> bool:
        return (
            self.partial_line_bytes == 0
            and self.invalid_checksum_at is None
            and self.invalid_schema_at is None
        )


def build_prepare(
    *,
    sequence: int,
    transaction_id: str,
    previous_committed_transaction_id: str | None,
    extents: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a signed ``prepare`` record.

    Preparation is not reader-visible data: it names staged objects and the
    extent transitions they back, but nothing it describes is readable until
    the matching commit line lands.
    """
    return sign_record(
        {
            "kind": PREPARE,
            "sequence": sequence,
            "transaction_id": transaction_id,
            "previous_committed_transaction_id": previous_committed_transaction_id,
            "extents": [dict(extent) for extent in extents],
            "objects": [dict(entry) for entry in objects],
        }
    )


def build_commit(*, sequence: int, transaction_id: str, prepare_sequence: int) -> dict[str, Any]:
    """Return a signed ``commit`` record, the transaction's visibility point."""
    return sign_record(
        {
            "kind": COMMIT,
            "sequence": sequence,
            "transaction_id": transaction_id,
            "prepare_sequence": prepare_sequence,
        }
    )


def build_checkpoint(
    *,
    sequence: int,
    checkpoint_id: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Return a signed ``checkpoint`` record referencing a written checkpoint."""
    return sign_record(
        {
            "kind": CHECKPOINT,
            "sequence": sequence,
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )


def build_termination(
    *,
    sequence: int,
    termination_record_id: str,
    termination_kind: str,
    time_ns: int,
    reason: str,
    fault_id: str | None,
    last_transaction_id: str,
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a signed ``termination`` record.

    This is the terminal *reference* to the sole committed typed
    ``session_termination`` row, not an independent representation of it. A
    reader resolves ``termination_record_id`` to that row and requires all six
    values to agree.
    """
    if termination_kind not in TERMINATION_KINDS:
        raise NrfSchemaError(f"unsupported termination kind {termination_kind!r}")
    if not reason:
        raise NrfSchemaError("termination reason must be non-empty")
    if termination_kind == "normal" and fault_id is not None:
        raise NrfSchemaError("normal termination must not reference a fault")
    if termination_kind == "faulted" and fault_id is None:
        raise NrfSchemaError("faulted termination must reference exactly one fault")
    record = {
        "kind": TERMINATION,
        "sequence": sequence,
        "termination_record_id": termination_record_id,
        "termination_kind": termination_kind,
        "time_ns": time_ns,
        "reason": reason,
        "fault_id": fault_id,
        "last_transaction_id": last_transaction_id,
    }
    if extensions:
        record["extensions"] = dict(extensions)
    return sign_record(record)


def prepared_object(
    *,
    path: str,
    transaction_id: str,
    target_path: str,
    array_path: str,
    logical_role: str,
    content_kind: str,
    sha256: str,
    byte_length: int,
    chunk_coordinate: Sequence[int] | None = None,
    record_schema_id: str | None = None,
    column_name: str | None = None,
) -> dict[str, Any]:
    """Return one prepared-object entry.

    ``staged_path`` is derived rather than accepted from the caller: the
    specification allows exactly one staging path per final path so that
    promotion is a one-to-one create-if-absent rename.
    """
    entry: dict[str, Any] = {
        "path": path,
        "staged_path": _staged_path(transaction_id, path),
        "target_path": target_path,
        "array_path": array_path,
        "logical_role": logical_role,
        "content_kind": content_kind,
        "disposition": "create",
        "sha256": sha256,
        "byte_length": byte_length,
    }
    if chunk_coordinate is not None:
        entry["chunk_coordinate"] = list(chunk_coordinate)
    if record_schema_id is not None:
        entry["record_schema_id"] = record_schema_id
    if column_name is not None:
        entry["column_name"] = column_name
    return entry


def extent_transition(
    *,
    target_path: str,
    target_kind: str,
    before: int,
    after: int,
    chunk_length: int,
    required_array_paths: Sequence[str],
    object_paths: Sequence[str],
    seals_target: bool,
    record_schema_id: str | None = None,
) -> dict[str, Any]:
    """Return one extent transition for a prepare record."""
    transition: dict[str, Any] = {
        "target_path": target_path,
        "target_kind": target_kind,
        "before": before,
        "after": after,
        "chunk_length": chunk_length,
        "required_array_paths": list(required_array_paths),
        "object_paths": list(object_paths),
        "seals_target": seals_target,
    }
    if record_schema_id is not None:
        transition["record_schema_id"] = record_schema_id
    return transition


def append_records(journal_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append signed records to the JSONL journal and flush them durably.

    Each record is flushed before the next is written so that a crash truncates
    the journal at a record boundary rather than interleaving two records.
    """
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("ab") as handle:
        for record in records:
            handle.write(encode_journal_line(record))
            handle.flush()
            os.fsync(handle.fileno())


def read_journal(journal_path: Path) -> tuple[list[dict[str, Any]], JournalTail]:
    """Return the usable records of a journal plus a description of its tail.

    Reading stops at the first unusable line and never looks past it: a
    checksum-invalid record means everything after it is unverifiable, and an
    unterminated final line is a record the writer never finished.

    A record is usable only once it has passed *both* checks NRF requires of
    local structure: the checksum, which says the bytes were not altered, and
    ``journal-record.schema.json``, which says the bytes are a record at all. A
    signed line missing a required member would otherwise reach replay and fail
    there as a raw ``KeyError`` instead of a stable NRF diagnosis.
    """
    if not journal_path.exists():
        return [], JournalTail()

    partial_bytes = 0
    records: list[dict[str, Any]] = []
    invalid_at: int | None = None
    schema_invalid_at: int | None = None
    schema_error = ""
    with journal_path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                partial_bytes = len(line)
                continue
            if invalid_at is not None or schema_invalid_at is not None:
                continue
            line = line[:-1]
            if not line:
                continue
            record = _decode_line(line)
            if record is None or not verify_record_checksum(record):
                invalid_at = len(records) + 1
                continue
            try:
                validate_document(record, JOURNAL_RECORD_SCHEMA)
            except NrfSchemaError as error:
                schema_invalid_at = len(records) + 1
                schema_error = str(error)
                continue
            records.append(record)

    return records, JournalTail(
        partial_line_bytes=partial_bytes,
        invalid_checksum_at=invalid_at,
        invalid_schema_at=schema_invalid_at,
        schema_error=schema_error,
    )


def iter_records(journal_path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete, checksum-valid records from a journal file."""
    yield from read_journal(journal_path)[0]


def _decode_line(line: bytes) -> dict[str, Any] | None:
    """Return one decoded record, or ``None`` when the line is not even JSON.

    Decoding is all this does. Whether a decoded object is a *record* -- its
    kind, its members, their types -- belongs to ``journal-record.schema.json``,
    so that a signed line with an unknown ``kind`` is reported as the schema
    failure it is rather than as a checksum failure it is not.
    """
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return record if isinstance(record, dict) else None
