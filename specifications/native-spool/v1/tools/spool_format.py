#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Reference encoder and scanner for the private native spool, version 1.

This module is the executable form of ``specifications/native-spool/v1/README.md``.
It is the specification's own implementation, not the product's: it imports
nothing from ``neurale``, it is not installed into any wheel, and it is not a
Python package. A conforming writer or reader is checked *against* this file;
disagreement is resolved by reading the specification, and if the specification
and this module disagree, the specification is what a third implementation
follows.

Nothing here is tuned. It reads a whole spool into memory and verifies every
checksum in Python, which is exactly wrong for a real writer on the recorder's
worker thread and exactly right for a reference: the byte layout is the
deliverable, and code that is obvious about it is easier to check than code
that is fast.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Final

# --- format identity ------------------------------------------------------

#: File magic. Eight bytes so the superblock stays 8-aligned from byte zero.
MAGIC: Final[bytes] = b"NRLSPOOL"
#: Transaction framing magic. Both markers are searched for by repair tooling,
#: so they are distinct from each other and from the file magic.
TRANSACTION_BEGIN_MAGIC: Final[bytes] = b"NNSTXBEG"
TRANSACTION_END_MAGIC: Final[bytes] = b"NNSTXEND"

#: Container version. A reader MUST refuse a major it does not implement; a
#: minor it does not know is readable, because a minor may only add record
#: kinds and may never change a layout defined here.
VERSION_MAJOR: Final[int] = 1
VERSION_MINOR: Final[int] = 1

#: Every offset and length in the container is a multiple of this.
ALIGNMENT: Final[int] = 8

SUPERBLOCK_BYTES: Final[int] = 256
TRANSACTION_HEADER_BYTES: Final[int] = 48
TRANSACTION_TRAILER_BYTES: Final[int] = 48
RECORD_HEADER_BYTES: Final[int] = 32

#: ``checksum_algorithm`` in the superblock. One value is defined in v1.
CHECKSUM_CRC32C: Final[int] = 1

#: ``plan_encoding`` in the superblock. The blob is opaque to this format: the
#: container verifies its length, its CRC, and its SHA-256 against the recorded
#: plan fingerprint, and never parses it.
PLAN_ENCODING_RECORDING_PLAN_JCS: Final[int] = 1

#: Durability policies, named exactly as section 6 of the lifecycle contract
#: names them. The container stores the policy so no reader has to infer a
#: guarantee from the presence of a sync it cannot see.
DURABILITY_BUFFERED: Final[int] = 0
DURABILITY_CHECKPOINT_SYNC: Final[int] = 1
DURABILITY_TRANSACTION_SYNC: Final[int] = 2

DURABILITY_NAMES: Final[dict[int, str]] = {
    DURABILITY_BUFFERED: "buffered",
    DURABILITY_CHECKPOINT_SYNC: "checkpoint_sync",
    DURABILITY_TRANSACTION_SYNC: "transaction_sync",
}

# --- record kinds ---------------------------------------------------------

KIND_FRAME: Final[int] = 1
KIND_SIGNAL_BLOCK: Final[int] = 2
KIND_DISCONTINUITY: Final[int] = 3
KIND_SIGNAL_GAP: Final[int] = 4
KIND_CONTROL: Final[int] = 5
KIND_FAULT: Final[int] = 6
KIND_ACCOUNTING_SNAPSHOT: Final[int] = 7
KIND_SESSION_END: Final[int] = 8
KIND_CHECKPOINT: Final[int] = 9

RECORD_KIND_NAMES: Final[dict[int, str]] = {
    KIND_FRAME: "frame",
    KIND_SIGNAL_BLOCK: "signal_block",
    KIND_DISCONTINUITY: "discontinuity",
    KIND_SIGNAL_GAP: "signal_gap",
    KIND_CONTROL: "control",
    KIND_FAULT: "fault",
    KIND_ACCOUNTING_SNAPSHOT: "accounting_snapshot",
    KIND_SESSION_END: "session_end",
    KIND_CHECKPOINT: "checkpoint",
}

#: Record kinds whose payload interior this specification defines. Everything
#: else is framed but opaque here: the interior of a frame, block,
#: discontinuity, gap, control, or fault payload belongs to the recorder core
#: (M6-05), and a container that also defined it would give those fields two
#: specifications.
CONTAINER_OWNED_KINDS: Final[frozenset[int]] = frozenset(
    {KIND_ACCOUNTING_SNAPSHOT, KIND_SESSION_END, KIND_CHECKPOINT}
)

#: What one committed record contributes to the item counts of section 1.1 of
#: the lifecycle contract. A signal block is part of the frame that owns it and
#: a signal gap is part of its discontinuity, so neither is counted again.
DATA_PLANE_ITEM_KINDS: Final[frozenset[int]] = frozenset({KIND_FRAME, KIND_DISCONTINUITY})
CONTROL_PLANE_ITEM_KINDS: Final[frozenset[int]] = frozenset({KIND_CONTROL})

CAPTURE_OUTCOME_NORMAL: Final[int] = 0
CAPTURE_OUTCOME_ABORTED: Final[int] = 1
CAPTURE_OUTCOME_FAULTED: Final[int] = 2

CAPTURE_OUTCOME_NAMES: Final[dict[int, str]] = {
    CAPTURE_OUTCOME_NORMAL: "normal",
    CAPTURE_OUTCOME_ABORTED: "aborted",
    CAPTURE_OUTCOME_FAULTED: "faulted",
}

#: ``requested_terminal_intent``: what the first exit from ``recording``
#: latched -- the terminal provenance the lifecycle contract (section 3.2)
#: requires the spool to preserve alongside ``capture_outcome``. The two are
#: separate fields because a caller may abort and a recorder fault may then
#: escalate the capture to ``faulted`` while the request stays ``aborted``;
#: neither may be recovered from the other, and the finalizer (M6-08) needs
#: both to write the NRF termination record faithfully.
REQUESTED_TERMINAL_INTENT_NORMAL: Final[int] = 0
REQUESTED_TERMINAL_INTENT_ABORTED: Final[int] = 1
REQUESTED_TERMINAL_INTENT_FAULT: Final[int] = 2

REQUESTED_TERMINAL_INTENT_NAMES: Final[dict[int, str]] = {
    REQUESTED_TERMINAL_INTENT_NORMAL: "normal",
    REQUESTED_TERMINAL_INTENT_ABORTED: "aborted",
    REQUESTED_TERMINAL_INTENT_FAULT: "fault",
}

#: ``accounting_origin``. A spool is written by the recorder that captured it,
#: so this is the only legal value inside a spool; ``recovery_rebuilt``
#: accounting exists in the NRF session a finalizer writes, never here.
ACCOUNTING_ORIGIN_RECORDER: Final[int] = 0

POSITION_ABSENT: Final[int] = 0
POSITION_ORDINAL: Final[int] = 1
POSITION_PRODUCER_IDENTITY: Final[int] = 2

#: The stable numeric registry for a position's ``identity_kind`` when it
#: carries a producer identity (``tag == POSITION_PRODUCER_IDENTITY``). The
#: lifecycle contract names these kinds in prose; this table freezes them to
#: numbers so two writers cannot use different digits for the same kind and the
#: finalizer (M6-08) can map each number to one canonical UTF-8 name for the NRF
#: accounting row. Data positions name a ``StreamMessageKind`` (the singular
#: message kind of a rejected data message); control positions name a control
#: record-set kind (the plural NRF ``record_schema.kind``). Adding a kind is a
#: minor-version change (section 9); a value not in the table is rejected.
IDENTITY_KIND_REGISTRY: Final[dict[int, str]] = {
    1: "frame",
    2: "discontinuity",
    3: "events",
    4: "trials",
    5: "experiment_states",
    6: "commands",
    7: "targets",
    8: "labels",
    9: "assistance",
    10: "faults",
    11: "task_variables",
}
#: The kinds a data-plane rejection may name. Lifecycle signals
#: (``end_of_stream``/``shutdown``/``abort``) are not recorded and never reach a
#: first-rejection position, so only the two recorded message kinds appear.
DATA_IDENTITY_KINDS: Final[frozenset[int]] = frozenset({1, 2})
#: The kinds a control-plane rejection may name (the contract's control plane:
#: events, trials, experiment states, commands, targets, labels, assistance,
#: fault rows, task variables).
CONTROL_IDENTITY_KINDS: Final[frozenset[int]] = frozenset({3, 4, 5, 6, 7, 8, 9, 10, 11})

ACCOUNTING_PAYLOAD_BYTES: Final[int] = 208
#: The bounded terminal reason carried in the session-end record: a fixed
#: UTF-8 field, NUL-padded, so the finalizer can copy it into the NRF
#: termination record's ``reason`` without the spool growing an unbounded
#: string. A value that does not fit MUST be rejected by the writer.
TERMINAL_REASON_BYTES: Final[int] = 64
#: 1 requested_terminal_intent + 1 capture_outcome + 1 primary_fault_committed
#: + 5 reserved + TERMINAL_REASON_BYTES + 8 end_unix_nanos.
SESSION_END_PAYLOAD_BYTES: Final[int] = 8 + TERMINAL_REASON_BYTES + 8
CHECKPOINT_PAYLOAD_BYTES: Final[int] = 24
POSITION_BYTES: Final[int] = 24

#: The four first-failed-or-lost positions, each with: its byte offset within
#: an accounting payload, the tag form the lifecycle contract fixes for it (a
#: loss after acceptance names the accepted item's ordinal; a rejection before
#: acceptance names the producer identity), and the counters whose first event
#: it names. A ``*_first_loss`` position covers both post-acceptance loss
#: handoffs -- ``failed_between_runtime_and_recorder`` and
#: ``lost_between_recorder_and_spool`` -- because an item that failed between
#: runtime and recorder was already accepted and already has a data-message
#: ordinal; binding the position to only one handoff would miss that loss. A
#: position is present iff the sum of its counters is nonzero (section 5.1): the
#: first loss or rejection is latched in bounded storage, so a zero sum with a
#: present position -- or the reverse -- is an inconsistency a reader must catch.
POSITION_LAYOUT: Final[tuple[tuple[str, int, int, tuple[str, ...], frozenset[int] | None], ...]] = (
    (
        "data_first_loss",
        112,
        POSITION_ORDINAL,
        ("failed_between_runtime_and_recorder", "lost_between_recorder_and_spool"),
        None,
    ),
    (
        "data_first_rejection",
        136,
        POSITION_PRODUCER_IDENTITY,
        ("rejected_before_runtime_acceptance",),
        DATA_IDENTITY_KINDS,
    ),
    (
        "control_first_loss",
        160,
        POSITION_ORDINAL,
        ("lost_between_control_acceptance_and_spool",),
        None,
    ),
    (
        "control_first_rejection",
        184,
        POSITION_PRODUCER_IDENTITY,
        ("control_rejected",),
        CONTROL_IDENTITY_KINDS,
    ),
)

#: Record kinds that have no owning item and therefore MUST carry a
#: ``logical_ordinal`` of 0 (section 2.3).
ORDINAL_ZERO_KINDS: Final[frozenset[int]] = frozenset(
    {KIND_FAULT, KIND_ACCOUNTING_SNAPSHOT, KIND_SESSION_END, KIND_CHECKPOINT}
)

#: The capture-side counters, in the order they occupy the payload. The
#: finalization-side counters of section 5.1 (``nrf_committed``,
#: ``lost_during_finalization``, and their control-plane twins) are absent by
#: construction: the spool is written before finalization runs, so a spool that
#: carried them would be carrying numbers nothing had measured.
DATA_COUNTERS: Final[tuple[str, ...]] = (
    "runtime_accepted",
    "recorder_accepted",
    "spool_committed",
    "rejected_before_runtime_acceptance",
    "failed_between_runtime_and_recorder",
    "lost_between_recorder_and_spool",
)
CONTROL_COUNTERS: Final[tuple[str, ...]] = (
    "control_offered",
    "control_accepted",
    "control_spool_committed",
    "control_rejected",
    "lost_between_control_acceptance_and_spool",
)
CLOSE_COUNTERS: Final[tuple[str, ...]] = (
    "rejected_after_close_data",
    "rejected_after_close_control",
)
POSITION_FIELDS: Final[tuple[str, ...]] = (
    "data_first_loss",
    "data_first_rejection",
    "control_first_loss",
    "control_first_rejection",
)

# --- diagnostic codes -----------------------------------------------------
#
# Stable identifiers. Wording may change; a code may not. Vectors and tests
# compare codes, never sentences.

CODE_BAD_MAGIC: Final[str] = "SPOOL-001"
CODE_UNSUPPORTED_MAJOR: Final[str] = "SPOOL-002"
CODE_SUPERBLOCK_CHECKSUM: Final[str] = "SPOOL-003"
CODE_TRUNCATED_SUPERBLOCK: Final[str] = "SPOOL-004"
CODE_PLAN_MISMATCH: Final[str] = "SPOOL-005"
CODE_SUPERBLOCK_FIELD: Final[str] = "SPOOL-006"

CODE_TORN_TAIL: Final[str] = "SPOOL-010"
CODE_TRANSACTION_HEADER_CHECKSUM: Final[str] = "SPOOL-011"
CODE_TRANSACTION_TRAILER_CHECKSUM: Final[str] = "SPOOL-012"
CODE_TRANSACTION_BODY_CHECKSUM: Final[str] = "SPOOL-013"
CODE_TRANSACTION_ID_SEQUENCE: Final[str] = "SPOOL-014"
CODE_BACK_LINK: Final[str] = "SPOOL-015"
CODE_RECORD_COUNT: Final[str] = "SPOOL-016"
CODE_RECORD_HEADER_CHECKSUM: Final[str] = "SPOOL-017"
CODE_RECORD_PAYLOAD_CHECKSUM: Final[str] = "SPOOL-018"
CODE_NONZERO_PADDING: Final[str] = "SPOOL-019"
CODE_EMPTY_TRANSACTION: Final[str] = "SPOOL-020"
CODE_TRAILER_ITEM_COUNTS: Final[str] = "SPOOL-021"

CODE_UNKNOWN_RECORD_KIND: Final[str] = "SPOOL-030"
CODE_RECORDS_AFTER_SESSION_END: Final[str] = "SPOOL-031"
CODE_SESSION_END_WITHOUT_ACCOUNTING: Final[str] = "SPOOL-032"
CODE_ACCOUNTING_TRANSACTION_NOT_QUIET: Final[str] = "SPOOL-033"
CODE_ACCOUNTING_IDENTITY: Final[str] = "SPOOL-034"
CODE_ACCOUNTING_EXCEEDS_PREFIX: Final[str] = "SPOOL-035"
CODE_CHECKPOINT_EXTENT: Final[str] = "SPOOL-036"
CODE_ACCOUNTING_ORIGIN: Final[str] = "SPOOL-037"
CODE_CONTAINER_PAYLOAD_SIZE: Final[str] = "SPOOL-038"
CODE_ORDINAL: Final[str] = "SPOOL-039"
CODE_FLAGS_NONZERO: Final[str] = "SPOOL-040"
CODE_SESSION_END_FIELD: Final[str] = "SPOOL-041"
CODE_CHECKPOINT_POLICY: Final[str] = "SPOOL-042"
CODE_ACCOUNTING_FLAG: Final[str] = "SPOOL-043"
CODE_POSITION_TAG: Final[str] = "SPOOL-044"
CODE_POSITION_CONSISTENCY: Final[str] = "SPOOL-045"
CODE_MULTIPLE_ACCOUNTING: Final[str] = "SPOOL-046"
CODE_MULTIPLE_SESSION_END: Final[str] = "SPOOL-047"
CODE_ACCOUNTING_WITHOUT_SESSION_END: Final[str] = "SPOOL-048"
CODE_CORRUPT_TRANSACTION_HEADER: Final[str] = "SPOOL-049"
CODE_POSITION_TAG_FORM: Final[str] = "SPOOL-050"
CODE_IDENTITY_KIND: Final[str] = "SPOOL-051"

#: Codes that end the committed prefix where they are found. Everything from
#: that byte on is an invalid tail and MUST NOT be promoted by any code path.
#: A torn or corrupt tail does not by itself block finalization: recovery
#: starts from the last valid committed transaction, which is the ordinary
#: crash path, so the tail codes are here alongside the framing failures.
PREFIX_ENDING_CODES: Final[frozenset[str]] = frozenset(
    {
        CODE_TORN_TAIL,
        CODE_TRANSACTION_HEADER_CHECKSUM,
        CODE_TRANSACTION_TRAILER_CHECKSUM,
        CODE_TRANSACTION_BODY_CHECKSUM,
        CODE_TRANSACTION_ID_SEQUENCE,
        CODE_BACK_LINK,
        CODE_RECORD_COUNT,
        CODE_RECORD_HEADER_CHECKSUM,
        CODE_RECORD_PAYLOAD_CHECKSUM,
        CODE_NONZERO_PADDING,
        CODE_EMPTY_TRANSACTION,
        CODE_TRAILER_ITEM_COUNTS,
        CODE_CORRUPT_TRANSACTION_HEADER,
    }
)

STATUS_OK: Final[str] = "ok"
STATUS_TORN_TAIL: Final[str] = "torn_tail"
STATUS_CORRUPT_TAIL: Final[str] = "corrupt_tail"
STATUS_REJECTED: Final[str] = "rejected"


# --- CRC-32C --------------------------------------------------------------


def _crc32c_table() -> tuple[int, ...]:
    """Build the reflected CRC-32C table for polynomial 0x1EDC6F41."""
    reflected_polynomial = 0x82F63B78
    table = []
    for i in range(256):
        value = i
        for _ in range(8):
            value = (value >> 1) ^ (reflected_polynomial if value & 1 else 0)
        table.append(value)
    return tuple(table)


_CRC32C_TABLE: Final[tuple[int, ...]] = _crc32c_table()


def crc32c(data: bytes, seed: int = 0) -> int:
    """Return the CRC-32C (Castagnoli) of *data*.

    Reflected input and output, initial and final value 0xFFFFFFFF -- the
    iSCSI/ext4 parameterization, which is what hardware CRC32C instructions
    compute. Known answers are in the specification so an implementation can
    check its own before it checks a spool.
    """
    crc = (seed ^ 0xFFFFFFFF) & 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


def padded_length(length: int) -> int:
    """Return *length* rounded up to the container alignment."""
    remainder = length % ALIGNMENT
    return length if remainder == 0 else length + (ALIGNMENT - remainder)


# --- encoding -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuperblockFields:
    """Everything the fixed superblock carries, before it is encoded."""

    session_id: str
    session_uuid: bytes
    plan_document: bytes
    durability_policy: int
    created_unix_nanos: int
    version_major: int = VERSION_MAJOR
    version_minor: int = VERSION_MINOR
    plan_encoding: int = PLAN_ENCODING_RECORDING_PLAN_JCS
    checksum_algorithm: int = CHECKSUM_CRC32C


def encode_superblock_region(fields: SuperblockFields) -> bytes:
    """Encode the superblock and the plan document that follows it.

    The plan document is stored whole rather than by reference. Section 5 of
    the lifecycle contract requires session identity, the plan fingerprint, and
    the schemas, descriptors, and clocks to survive a process crash *in the
    spool*, and a spool that pointed at a file somebody else owned would not
    survive anything on its own.
    """
    session_id_bytes = fields.session_id.encode("utf-8")
    if len(session_id_bytes) > 128:
        raise ValueError("session_id does not fit the fixed 128-byte superblock field")
    if len(fields.session_uuid) != 16:
        raise ValueError("session_uuid must be 16 bytes")
    plan_bytes = len(fields.plan_document)
    first_transaction_offset = SUPERBLOCK_BYTES + padded_length(plan_bytes)

    superblock = bytearray(SUPERBLOCK_BYTES)
    superblock[0:8] = MAGIC
    struct.pack_into(
        "<HHI",
        superblock,
        8,
        fields.version_major,
        fields.version_minor,
        SUPERBLOCK_BYTES,
    )
    struct.pack_into(
        "<BBBB",
        superblock,
        16,
        fields.durability_policy,
        fields.checksum_algorithm,
        fields.plan_encoding,
        0,
    )
    struct.pack_into("<I", superblock, 20, ALIGNMENT)
    struct.pack_into("<I", superblock, 24, plan_bytes)
    struct.pack_into("<I", superblock, 28, crc32c(fields.plan_document))
    superblock[32:64] = hashlib.sha256(fields.plan_document).digest()
    superblock[64:80] = fields.session_uuid
    struct.pack_into("<Q", superblock, 80, fields.created_unix_nanos)
    struct.pack_into("<Q", superblock, 88, first_transaction_offset)
    superblock[96 : 96 + len(session_id_bytes)] = session_id_bytes
    struct.pack_into("<I", superblock, 252, crc32c(bytes(superblock[:252])))

    padding = b"\x00" * (padded_length(plan_bytes) - plan_bytes)
    return bytes(superblock) + fields.plan_document + padding


def encode_record(
    kind: int,
    payload: bytes,
    *,
    logical_ordinal: int = 0,
    record_unix_nanos: int = 0,
    record_flags: int = 0,
) -> bytes:
    """Encode one record: fixed header, payload, zero padding to alignment."""
    header = bytearray(RECORD_HEADER_BYTES)
    struct.pack_into("<HHI", header, 0, kind, record_flags, len(payload))
    struct.pack_into("<QQ", header, 8, logical_ordinal, record_unix_nanos)
    struct.pack_into("<I", header, 24, crc32c(payload))
    struct.pack_into("<I", header, 28, crc32c(bytes(header[:28])))
    padding = b"\x00" * (padded_length(len(payload)) - len(payload))
    return bytes(header) + payload + padding


def encode_transaction(
    transaction_id: int,
    records: list[bytes],
    *,
    previous_transaction_offset: int,
    begin_unix_nanos: int = 0,
    data_items: int | None = None,
    control_items: int | None = None,
    n_records: int | None = None,
) -> bytes:
    """Encode one transaction: header, records, commit trailer.

    The trailer is what makes the records visible, so it is written last and
    carries the checksum over everything before it. A writer that is
    interrupted anywhere in this byte range leaves a tail no reader will
    promote, which is the entire point of writing it in this order.

    The ``data_items``, ``control_items``, and ``n_records`` overrides exist
    for the corrupt vectors; a writer never uses them.
    """
    body = bytearray(TRANSACTION_HEADER_BYTES)
    body[0:8] = TRANSACTION_BEGIN_MAGIC
    struct.pack_into("<QQ", body, 8, transaction_id, previous_transaction_offset)
    counted = len(records) if n_records is None else n_records
    struct.pack_into("<II", body, 24, counted, TRANSACTION_HEADER_BYTES)
    struct.pack_into("<Q", body, 32, begin_unix_nanos)
    struct.pack_into("<I", body, 40, 0)
    struct.pack_into("<I", body, 44, crc32c(bytes(body[:44])))
    for record in records:
        body += record

    if data_items is None or control_items is None:
        counted_data = 0
        counted_control = 0
        for record in records:
            kind = struct.unpack_from("<H", record, 0)[0]
            counted_data += 1 if kind in DATA_PLANE_ITEM_KINDS else 0
            counted_control += 1 if kind in CONTROL_PLANE_ITEM_KINDS else 0
        data_items = counted_data if data_items is None else data_items
        control_items = counted_control if control_items is None else control_items

    trailer = bytearray(TRANSACTION_TRAILER_BYTES)
    trailer[0:8] = TRANSACTION_END_MAGIC
    struct.pack_into("<QQ", trailer, 8, transaction_id, len(body))
    struct.pack_into("<III", trailer, 24, counted, data_items, control_items)
    struct.pack_into("<I", trailer, 36, crc32c(bytes(body)))
    struct.pack_into("<I", trailer, 40, 0)
    struct.pack_into("<I", trailer, 44, crc32c(bytes(trailer[:44])))
    return bytes(body) + bytes(trailer)


def encode_position(
    tag: int = POSITION_ABSENT,
    *,
    ordinal: int = 0,
    identity_kind: int = 0,
    identity_value: int = 0,
) -> bytes:
    """Encode one first-failed-or-lost position (a tagged union, section 5.1)."""
    pos = bytearray(POSITION_BYTES)
    struct.pack_into("<B", pos, 0, tag)
    struct.pack_into("<I", pos, 4, identity_kind)
    struct.pack_into("<QQ", pos, 8, ordinal, identity_value)
    return bytes(pos)


def encode_accounting(
    counters: dict[str, int],
    *,
    positions: dict[str, bytes] | None = None,
    control_offered_present: bool = True,
    producer_acceptance_known: bool = True,
    accounting_origin: int = ACCOUNTING_ORIGIN_RECORDER,
) -> bytes:
    """Encode an accounting snapshot payload.

    The counters are the capture-side ones only, and the layout has no
    ``accounting_verified`` field. Section 5.1 of the contract is explicit that
    the field is a judgement a reader makes about a record, not a fact a writer
    holds, and storing it would create two values that can disagree.
    """
    positions = positions or {}
    payload = bytearray()
    for name in DATA_COUNTERS + CONTROL_COUNTERS + CLOSE_COUNTERS:
        payload += struct.pack("<Q", counters.get(name, 0))
    payload += struct.pack(
        "<BBBBBBBB",
        1 if control_offered_present else 0,
        1 if producer_acceptance_known else 0,
        accounting_origin,
        0,
        0,
        0,
        0,
        0,
    )
    for name in POSITION_FIELDS:
        payload += positions.get(name, encode_position())
    if len(payload) != ACCOUNTING_PAYLOAD_BYTES:  # pragma: no cover - layout guard
        raise AssertionError(f"accounting payload is {len(payload)} bytes, expected 208")
    return bytes(payload)


def encode_session_end(
    capture_outcome: int,
    end_unix_nanos: int,
    *,
    requested_terminal_intent: int = REQUESTED_TERMINAL_INTENT_NORMAL,
    primary_fault_committed: bool = False,
    terminal_reason: bytes = b"",
) -> bytes:
    """Encode the spool session-end payload.

    The session-end record is what *freezes* the capture outcome (section 3.2),
    and it is the only place the terminal provenance survives before
    finalization: ``requested_terminal_intent`` (what the first exit from
    ``recording`` latched), ``capture_outcome`` (how taking the data ended),
    ``primary_fault_committed`` (whether the primary fault row reached a
    committed extent -- false when it could not, so the finalizer can declare
    the row missing), and a bounded ``terminal_reason`` the finalizer copies
    into the NRF termination record. These are separate fields because the
    contract forbids recovering any one from the others.

    There is no ``unknown`` outcome here on purpose: ``unknown`` is what a
    *missing* session-end record means (section 3.2), and a value for it would
    let a writer state the one thing only an absence can say.
    """
    reason = terminal_reason
    if len(reason) > TERMINAL_REASON_BYTES:
        raise ValueError(
            f"terminal_reason is {len(reason)} bytes, expected at most {TERMINAL_REASON_BYTES}"
        )
    payload = bytearray(SESSION_END_PAYLOAD_BYTES)
    struct.pack_into(
        "<BBB",
        payload,
        0,
        requested_terminal_intent,
        capture_outcome,
        1 if primary_fault_committed else 0,
    )
    # Bytes 3..7 are reserved: written zero, ignored on read (section 1).
    payload[8 : 8 + len(reason)] = reason
    struct.pack_into("<Q", payload, 8 + TERMINAL_REASON_BYTES, end_unix_nanos)
    if len(payload) != SESSION_END_PAYLOAD_BYTES:  # pragma: no cover - layout guard
        raise AssertionError(
            f"session_end payload is {len(payload)} bytes, expected {SESSION_END_PAYLOAD_BYTES}"
        )
    return bytes(payload)


def encode_checkpoint(
    durable_extent_bytes: int, synced_unix_nanos: int, durability_policy: int
) -> bytes:
    """Encode a checkpoint payload.

    ``durable_extent_bytes`` is an extent -- how far synchronization has been
    acknowledged -- never a count of records that reached a durable stage
    (section 1.2).
    """
    payload = bytearray(CHECKPOINT_PAYLOAD_BYTES)
    struct.pack_into("<QQ", payload, 0, durable_extent_bytes, synced_unix_nanos)
    struct.pack_into("<B", payload, 16, durability_policy)
    return bytes(payload)


# --- scanning -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    """One diagnostic, with the byte offset that produced it."""

    code: str
    offset: int
    detail: str

    def document(self) -> dict[str, Any]:
        return {"code": self.code, "offset": self.offset, "detail": self.detail}


@dataclass(slots=True)
class ScanResult:
    """What a read of a spool establishes, and nothing more.

    ``committed_prefix_end`` is the byte offset one past the last committed
    transaction. Everything at or after it is an invalid tail: it MUST NOT be
    promoted, and an ordinary read MUST NOT remove it.
    """

    status: str = STATUS_OK
    committed_prefix_end: int = 0
    committed_transactions: int = 0
    last_transaction_id: int = 0
    data_items: int = 0
    control_items: int = 0
    session_end_present: bool = False
    capture_outcome: str | None = None
    requested_terminal_intent: str | None = None
    primary_fault_committed: bool | None = None
    terminal_reason: str | None = None
    durability_policy: str | None = None
    durable_extent_bytes: int = 0
    session_id: str | None = None
    plan_fingerprint: str | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def codes(self) -> list[str]:
        return [finding.code for finding in self.findings]

    @property
    def readable(self) -> bool:
        """Whether the superblock validated and a committed prefix exists."""
        return self.status != STATUS_REJECTED

    @property
    def finalizable(self) -> bool:
        """Whether a finalizer may promote this spool's committed prefix.

        A torn or corrupt tail does not block promotion of the prefix before
        it -- that case is the ordinary crash, and section 4.5 says recovery
        starts from the last valid committed transaction. A conformance
        finding does block it: promoting a prefix a reader cannot account for
        would be exactly the silent loss this contract does not tolerate.
        """
        if self.status == STATUS_REJECTED:
            return False
        return not any(code not in PREFIX_ENDING_CODES for code in self.codes)

    def document(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "committed_prefix_end": self.committed_prefix_end,
            "committed_transactions": self.committed_transactions,
            "last_transaction_id": self.last_transaction_id,
            "data_items": self.data_items,
            "control_items": self.control_items,
            "session_end_present": self.session_end_present,
            "capture_outcome": self.capture_outcome,
            "requested_terminal_intent": self.requested_terminal_intent,
            "primary_fault_committed": self.primary_fault_committed,
            "terminal_reason": self.terminal_reason,
            "durability_policy": self.durability_policy,
            "durable_extent_bytes": self.durable_extent_bytes,
            "finalizable": self.finalizable,
            "codes": self.codes,
        }


@dataclass(frozen=True, slots=True)
class _Record:
    kind: int
    payload: bytes
    logical_ordinal: int


def _decode_accounting(payload: bytes) -> dict[str, int]:
    values: dict[str, int] = {}
    offset = 0
    for name in DATA_COUNTERS + CONTROL_COUNTERS + CLOSE_COUNTERS:
        values[name] = struct.unpack_from("<Q", payload, offset)[0]
        offset += 8
    flags = struct.unpack_from("<BBBBBBBB", payload, offset)
    values["control_offered_present"] = flags[0]
    values["producer_acceptance_known"] = flags[1]
    values["accounting_origin"] = flags[2]
    return values


def scan_spool(data: bytes) -> ScanResult:
    """Read a spool image and report what it establishes.

    Read-only by definition: this function never writes, never truncates, and
    never repairs. Section 4.5 requires ordinary diagnosis to be read-only and
    repair to be a separate, explicitly requested, auditable operation.
    """
    result = ScanResult()
    header = _scan_superblock(data, result)
    if header is None:
        result.status = STATUS_REJECTED
        return result

    first_transaction_offset, policy = header
    result.durability_policy = DURABILITY_NAMES[policy]
    result.committed_prefix_end = first_transaction_offset

    offset = first_transaction_offset
    expected_id = 1
    previous_offset = 0
    session_end_count = 0
    n_accounting = 0
    checkpoint_extent = 0

    while offset < len(data):
        transaction = _scan_transaction(data, offset, expected_id, previous_offset, result)
        if transaction is None:
            break
        end_offset, records, data_items, control_items = transaction

        if session_end_count > 0:
            result.findings.append(
                Finding(
                    CODE_RECORDS_AFTER_SESSION_END,
                    offset,
                    f"transaction {expected_id} was committed after the session-end record",
                )
            )

        kinds = [record.kind for record in records]
        transaction_has_accounting = False
        transaction_has_session_end = False
        transaction_accountings: list[tuple[bytes, int]] = []
        frame_ordinal: int | None = None
        discontinuity_ordinal: int | None = None

        for i, record in enumerate(records):
            _scan_record_semantics(record, offset, result)
            kind = record.kind

            # logical_ordinal ownership (section 2.3): a block carries its
            # owning frame's ordinal, a gap its owning discontinuity's, and a
            # record with no owning item carries 0. The owner is the nearest
            # preceding owner of its kind in the same transaction.
            if kind == KIND_FRAME:
                frame_ordinal = record.logical_ordinal
            elif kind == KIND_DISCONTINUITY:
                discontinuity_ordinal = record.logical_ordinal
            elif kind == KIND_SIGNAL_BLOCK:
                if frame_ordinal is None or record.logical_ordinal != frame_ordinal:
                    result.findings.append(
                        Finding(
                            CODE_ORDINAL,
                            offset,
                            f"a signal_block carries logical_ordinal {record.logical_ordinal}, "
                            f"which does not match its owning frame's ordinal "
                            f"({frame_ordinal if frame_ordinal is not None else 'none'})",
                        )
                    )
            elif kind == KIND_SIGNAL_GAP:
                if discontinuity_ordinal is None or record.logical_ordinal != discontinuity_ordinal:
                    result.findings.append(
                        Finding(
                            CODE_ORDINAL,
                            offset,
                            f"a signal_gap carries logical_ordinal {record.logical_ordinal}, "
                            f"which does not match its owning discontinuity's ordinal "
                            f"({discontinuity_ordinal if discontinuity_ordinal is not None else 'none'})",
                        )
                    )
            elif kind in ORDINAL_ZERO_KINDS:
                if record.logical_ordinal != 0:
                    result.findings.append(
                        Finding(
                            CODE_ORDINAL,
                            offset,
                            f"{RECORD_KIND_NAMES[kind]} carries a nonzero logical_ordinal "
                            f"({record.logical_ordinal}); a record with no owning item carries 0",
                        )
                    )

            if kind == KIND_SESSION_END:
                # A session end must be preceded, in its own transaction, by the
                # accounting snapshot it seals (section 4.5).
                if KIND_ACCOUNTING_SNAPSHOT not in kinds[:i]:
                    result.findings.append(
                        Finding(
                            CODE_SESSION_END_WITHOUT_ACCOUNTING,
                            offset,
                            "a session-end record must be preceded, in its own transaction, by "
                            "the accounting snapshot it seals",
                        )
                    )
                session_end_count += 1
                if session_end_count == 1:
                    result.session_end_present = True
                    if len(record.payload) == SESSION_END_PAYLOAD_BYTES:
                        _decode_session_end(record, offset, result)
                else:
                    result.findings.append(
                        Finding(
                            CODE_MULTIPLE_SESSION_END,
                            offset,
                            "a second session_end record was committed; a session has at most "
                            "one capture outcome",
                        )
                    )
                transaction_has_session_end = True
            elif kind == KIND_ACCOUNTING_SNAPSHOT:
                if any(k in DATA_PLANE_ITEM_KINDS or k in CONTROL_PLANE_ITEM_KINDS for k in kinds):
                    result.findings.append(
                        Finding(
                            CODE_ACCOUNTING_TRANSACTION_NOT_QUIET,
                            offset,
                            "an accounting snapshot must be committed in a transaction that "
                            "carries no data or control items, so its counters name a settled "
                            "prefix",
                        )
                    )
                n_accounting += 1
                if n_accounting > 1:
                    result.findings.append(
                        Finding(
                            CODE_MULTIPLE_ACCOUNTING,
                            offset,
                            "a second accounting_snapshot was committed; the summary is written "
                            "once, in the transaction that seals the session",
                        )
                    )
                transaction_has_accounting = True
                if len(record.payload) == ACCOUNTING_PAYLOAD_BYTES:
                    transaction_accountings.append((record.payload, offset))
                    _validate_accounting_payload(record.payload, offset, result)
            elif kind == KIND_CHECKPOINT and len(record.payload) == CHECKPOINT_PAYLOAD_BYTES:
                if record.payload[16] != policy:
                    result.findings.append(
                        Finding(
                            CODE_CHECKPOINT_POLICY,
                            offset,
                            f"checkpoint durability policy {record.payload[16]} does not match "
                            f"the superblock's {policy}",
                        )
                    )
                claimed = struct.unpack_from("<Q", record.payload, 0)[0]
                if claimed > offset:
                    result.findings.append(
                        Finding(
                            CODE_CHECKPOINT_EXTENT,
                            offset,
                            f"checkpoint claims {claimed} synced bytes, which is beyond the start "
                            f"of the transaction that carries it ({offset})",
                        )
                    )
                else:
                    checkpoint_extent = max(checkpoint_extent, claimed)

        # An accounting snapshot must be sealed by a session_end in the same
        # transaction: the summary and the end are one record pair, and a
        # snapshot written with nothing to seal is unaccountable.
        if transaction_has_accounting and not transaction_has_session_end:
            result.findings.append(
                Finding(
                    CODE_ACCOUNTING_WITHOUT_SESSION_END,
                    offset,
                    "an accounting_snapshot was committed without a session_end in its "
                    "transaction; the summary must seal the session it is written with",
                )
            )

        result.committed_transactions += 1
        result.last_transaction_id = expected_id
        result.data_items += data_items
        result.control_items += control_items
        result.committed_prefix_end = end_offset
        previous_offset = offset
        offset = end_offset
        expected_id += 1

        # Layer 2 -- the committed counters match the committed prefix -- is
        # checked at the prefix the snapshot seals, not after later transactions
        # have inflated the totals. The sealing transaction is quiet, so its
        # own item contribution is zero.
        for payload, accounting_offset in transaction_accountings:
            _check_accounting_exceeds_prefix(payload, accounting_offset, result)

    if result.committed_prefix_end < len(data):
        torn = any(code == CODE_TORN_TAIL for code in result.codes)
        result.status = STATUS_TORN_TAIL if torn else STATUS_CORRUPT_TAIL

    result.durable_extent_bytes = _durable_extent(
        policy, checkpoint_extent, first_transaction_offset, result
    )
    return result


def _decode_session_end(record: _Record, offset: int, result: ScanResult) -> None:
    """Read the terminal provenance the session-end record freezes (section 4.5).

    Reserved bytes are ignored on read (section 1); the three defined fields
    are validated against their enums so a writer cannot store a value no
    finalizer could interpret.
    """
    payload = record.payload
    requested = payload[0]
    outcome = payload[1]
    primary_fault = payload[2]
    result.requested_terminal_intent = REQUESTED_TERMINAL_INTENT_NAMES.get(requested, None)
    result.capture_outcome = CAPTURE_OUTCOME_NAMES.get(outcome, None)
    result.primary_fault_committed = bool(primary_fault) if primary_fault in (0, 1) else None
    result.terminal_reason = (
        payload[8 : 8 + TERMINAL_REASON_BYTES].rstrip(b"\x00").decode("utf-8", errors="replace")
    )
    if requested not in REQUESTED_TERMINAL_INTENT_NAMES:
        result.findings.append(
            Finding(
                CODE_SESSION_END_FIELD,
                offset,
                f"requested_terminal_intent {requested} is not defined "
                f"(0 normal, 1 aborted, 2 fault)",
            )
        )
    if outcome not in CAPTURE_OUTCOME_NAMES:
        result.findings.append(
            Finding(
                CODE_SESSION_END_FIELD,
                offset,
                f"capture_outcome {outcome} is not defined (0 normal, 1 aborted, 2 faulted)",
            )
        )
    if primary_fault not in (0, 1):
        result.findings.append(
            Finding(
                CODE_SESSION_END_FIELD,
                offset,
                f"primary_fault_committed {primary_fault} is not 0 or 1",
            )
        )


def _scan_superblock(data: bytes, result: ScanResult) -> tuple[int, int] | None:
    """Validate the superblock region; return the first offset and the policy."""
    if len(data) < SUPERBLOCK_BYTES:
        result.findings.append(
            Finding(
                CODE_TRUNCATED_SUPERBLOCK, 0, f"file is {len(data)} bytes, expected at least 256"
            )
        )
        return None
    if data[0:8] != MAGIC:
        result.findings.append(Finding(CODE_BAD_MAGIC, 0, "file magic is not NRLSPOOL"))
        return None
    stored_crc = struct.unpack_from("<I", data, 252)[0]
    if crc32c(data[:252]) != stored_crc:
        result.findings.append(
            Finding(CODE_SUPERBLOCK_CHECKSUM, 0, "superblock checksum does not verify")
        )
        return None

    # The minor version is read and deliberately not acted on: a minor may only
    # add record kinds, so an unknown one is refused where such a kind is met,
    # not here (section 9).
    major, _minor, superblock_bytes = struct.unpack_from("<HHI", data, 8)
    if major != VERSION_MAJOR:
        result.findings.append(
            Finding(
                CODE_UNSUPPORTED_MAJOR,
                8,
                f"container major version {major} is not implemented by this reader",
            )
        )
        return None
    policy, checksum_algorithm, plan_encoding, _reserved = struct.unpack_from("<BBBB", data, 16)
    alignment = struct.unpack_from("<I", data, 20)[0]
    if (
        superblock_bytes != SUPERBLOCK_BYTES
        or alignment != ALIGNMENT
        or checksum_algorithm != CHECKSUM_CRC32C
        or plan_encoding != PLAN_ENCODING_RECORDING_PLAN_JCS
        or policy not in DURABILITY_NAMES
    ):
        result.findings.append(
            Finding(
                CODE_SUPERBLOCK_FIELD,
                16,
                "superblock declares a fixed field this version does not define",
            )
        )
        return None

    plan_bytes, plan_crc = struct.unpack_from("<II", data, 24)
    plan_sha256 = data[32:64]
    first_transaction_offset = struct.unpack_from("<Q", data, 88)[0]
    if first_transaction_offset != SUPERBLOCK_BYTES + padded_length(plan_bytes):
        result.findings.append(
            Finding(
                CODE_SUPERBLOCK_FIELD,
                88,
                "first_transaction_offset does not follow the declared plan document",
            )
        )
        return None
    if len(data) < first_transaction_offset:
        result.findings.append(
            Finding(CODE_TRUNCATED_SUPERBLOCK, SUPERBLOCK_BYTES, "the plan document is truncated")
        )
        return None

    plan = data[SUPERBLOCK_BYTES : SUPERBLOCK_BYTES + plan_bytes]
    plan_padding = data[SUPERBLOCK_BYTES + plan_bytes : first_transaction_offset]
    if plan_padding.strip(b"\x00"):
        # No checksum covers this gap -- the plan CRC stops at plan_bytes and
        # the superblock CRC stops at 252 -- so the zero rule is what covers
        # it, exactly as it covers record padding.
        result.findings.append(
            Finding(
                CODE_NONZERO_PADDING,
                SUPERBLOCK_BYTES + plan_bytes,
                "the padding behind the plan document is not zero-filled",
            )
        )
        return None
    if crc32c(plan) != plan_crc or hashlib.sha256(plan).digest() != plan_sha256:
        result.findings.append(
            Finding(
                CODE_PLAN_MISMATCH,
                SUPERBLOCK_BYTES,
                "the stored plan document does not match its checksum and fingerprint",
            )
        )
        return None

    result.session_id = data[96:224].rstrip(b"\x00").decode("utf-8", errors="replace")
    result.plan_fingerprint = plan_sha256.hex()
    return first_transaction_offset, policy


def _scan_transaction(
    data: bytes,
    offset: int,
    expected_id: int,
    previous_offset: int,
    result: ScanResult,
) -> tuple[int, list[_Record], int, int] | None:
    """Validate one transaction; return its end, records, and item counts."""
    if len(data) - offset < TRANSACTION_HEADER_BYTES:
        result.findings.append(
            Finding(CODE_TORN_TAIL, offset, "the file ends inside a transaction header")
        )
        return None
    if data[offset : offset + 8] != TRANSACTION_BEGIN_MAGIC:
        result.findings.append(
            Finding(
                CODE_CORRUPT_TRANSACTION_HEADER,
                offset,
                "the tail begins with a complete header whose begin magic is not NNSTXBEG; "
                "present and wrong is corruption, not truncation",
            )
        )
        return None
    header_crc = struct.unpack_from("<I", data, offset + 44)[0]
    if crc32c(data[offset : offset + 44]) != header_crc:
        result.findings.append(
            Finding(
                CODE_TRANSACTION_HEADER_CHECKSUM, offset, "transaction header checksum mismatch"
            )
        )
        return None

    transaction_id, back_link = struct.unpack_from("<QQ", data, offset + 8)
    declared_records, header_bytes = struct.unpack_from("<II", data, offset + 24)
    if transaction_id != expected_id:
        result.findings.append(
            Finding(
                CODE_TRANSACTION_ID_SEQUENCE,
                offset,
                f"transaction id {transaction_id} is not the successor {expected_id} of the "
                "previous committed transaction",
            )
        )
        return None
    if back_link != previous_offset:
        result.findings.append(
            Finding(
                CODE_BACK_LINK,
                offset,
                f"back link {back_link} does not name the previous transaction at {previous_offset}",
            )
        )
        return None
    if header_bytes != TRANSACTION_HEADER_BYTES:
        result.findings.append(
            Finding(
                CODE_CORRUPT_TRANSACTION_HEADER,
                offset,
                f"transaction header_bytes is {header_bytes}, not 48; a complete header with the "
                "wrong size is corruption, not truncation",
            )
        )
        return None
    if declared_records == 0:
        result.findings.append(
            Finding(CODE_EMPTY_TRANSACTION, offset, "a transaction must carry at least one record")
        )
        return None

    cursor = offset + TRANSACTION_HEADER_BYTES
    records: list[_Record] = []
    for _ in range(declared_records):
        record = _scan_record(data, cursor, result)
        if record is None:
            return None
        parsed, cursor = record
        records.append(parsed)

    if len(data) - cursor < TRANSACTION_TRAILER_BYTES:
        result.findings.append(
            Finding(CODE_TORN_TAIL, cursor, "the file ends before the commit trailer")
        )
        return None
    if data[cursor : cursor + 8] != TRANSACTION_END_MAGIC:
        result.findings.append(
            Finding(
                CODE_RECORD_COUNT, cursor, "the declared records do not end at a commit trailer"
            )
        )
        return None
    trailer_crc = struct.unpack_from("<I", data, cursor + 44)[0]
    if crc32c(data[cursor : cursor + 44]) != trailer_crc:
        result.findings.append(
            Finding(CODE_TRANSACTION_TRAILER_CHECKSUM, cursor, "commit trailer checksum mismatch")
        )
        return None

    trailer_id, body_bytes = struct.unpack_from("<QQ", data, cursor + 8)
    trailer_records, data_items, control_items = struct.unpack_from("<III", data, cursor + 24)
    body_crc = struct.unpack_from("<I", data, cursor + 36)[0]
    trailer_flags = struct.unpack_from("<I", data, cursor + 40)[0]
    if trailer_flags != 0:
        result.findings.append(
            Finding(CODE_FLAGS_NONZERO, cursor, f"trailer flags is {trailer_flags}, not 0")
        )
    if trailer_id != transaction_id or body_bytes != cursor - offset:
        result.findings.append(
            Finding(
                CODE_TRANSACTION_TRAILER_CHECKSUM,
                cursor,
                "the commit trailer does not describe the transaction it closes",
            )
        )
        return None
    if trailer_records != declared_records:
        result.findings.append(
            Finding(CODE_RECORD_COUNT, cursor, "header and trailer disagree on the record count")
        )
        return None
    if crc32c(data[offset:cursor]) != body_crc:
        result.findings.append(
            Finding(CODE_TRANSACTION_BODY_CHECKSUM, offset, "transaction body checksum mismatch")
        )
        return None

    counted_data = sum(1 for record in records if record.kind in DATA_PLANE_ITEM_KINDS)
    counted_control = sum(1 for record in records if record.kind in CONTROL_PLANE_ITEM_KINDS)
    if data_items != counted_data or control_items != counted_control:
        result.findings.append(
            Finding(
                CODE_TRAILER_ITEM_COUNTS,
                cursor,
                f"the trailer claims {data_items} data and {control_items} control items; the "
                f"transaction carries {counted_data} and {counted_control}",
            )
        )
        return None

    return cursor + TRANSACTION_TRAILER_BYTES, records, data_items, control_items


def _scan_record(data: bytes, offset: int, result: ScanResult) -> tuple[_Record, int] | None:
    if len(data) - offset < RECORD_HEADER_BYTES:
        result.findings.append(
            Finding(CODE_TORN_TAIL, offset, "the file ends inside a record header")
        )
        return None
    header_crc = struct.unpack_from("<I", data, offset + 28)[0]
    if crc32c(data[offset : offset + 28]) != header_crc:
        result.findings.append(
            Finding(CODE_RECORD_HEADER_CHECKSUM, offset, "record header checksum mismatch")
        )
        return None
    kind, record_flags, payload_bytes = struct.unpack_from("<HHI", data, offset)
    logical_ordinal = struct.unpack_from("<Q", data, offset + 8)[0]
    payload_crc = struct.unpack_from("<I", data, offset + 24)[0]
    if record_flags != 0:
        result.findings.append(
            Finding(CODE_FLAGS_NONZERO, offset, f"record_flags is {record_flags}, not 0")
        )

    payload_start = offset + RECORD_HEADER_BYTES
    padded = padded_length(payload_bytes)
    if len(data) - payload_start < padded:
        result.findings.append(
            Finding(CODE_TORN_TAIL, offset, "the file ends inside a record payload")
        )
        return None
    payload = data[payload_start : payload_start + payload_bytes]
    if crc32c(payload) != payload_crc:
        result.findings.append(
            Finding(CODE_RECORD_PAYLOAD_CHECKSUM, offset, "record payload checksum mismatch")
        )
        return None
    padding = data[payload_start + payload_bytes : payload_start + padded]
    if padding.strip(b"\x00"):
        result.findings.append(
            Finding(CODE_NONZERO_PADDING, offset, "record padding is not zero-filled")
        )
        return None
    return _Record(kind, payload, logical_ordinal), payload_start + padded


def _scan_record_semantics(record: _Record, offset: int, result: ScanResult) -> None:
    """The per-record rules that do not need cross-record context.

    Cross-record and cross-transaction rules -- the logical-ordinal ownership,
    the accounting/session-end pairing, and the accounting verification -- live
    in ``scan_spool``, which has the whole transaction and the running totals.
    """
    if record.kind not in RECORD_KIND_NAMES:
        result.findings.append(
            Finding(
                CODE_UNKNOWN_RECORD_KIND,
                offset,
                f"record kind {record.kind} is not defined by this container version; promoting "
                "a prefix that skipped it would lose a record the writer committed",
            )
        )
        return
    expected_sizes = {
        KIND_ACCOUNTING_SNAPSHOT: ACCOUNTING_PAYLOAD_BYTES,
        KIND_SESSION_END: SESSION_END_PAYLOAD_BYTES,
        KIND_CHECKPOINT: CHECKPOINT_PAYLOAD_BYTES,
    }
    if record.kind in expected_sizes and len(record.payload) != expected_sizes[record.kind]:
        result.findings.append(
            Finding(
                CODE_CONTAINER_PAYLOAD_SIZE,
                offset,
                f"{RECORD_KIND_NAMES[record.kind]} payload is {len(record.payload)} bytes, "
                f"expected {expected_sizes[record.kind]}",
            )
        )


def _validate_accounting_payload(payload: bytes, offset: int, result: ScanResult) -> None:
    """Section 7 layer 1 plus the structural rules for the payload itself.

    Layer 1 (internal identities) and the payload structure -- booleans, the
    recorder origin, and the four tagged-union positions -- are all facts about
    the snapshot, independent of where in the spool it was committed, so they
    are checked here. Layer 2 (consistency with the committed prefix) is checked
    separately, at the prefix the snapshot seals.
    """
    values = _decode_accounting(payload)

    if values["control_offered_present"] not in (0, 1):
        result.findings.append(
            Finding(
                CODE_ACCOUNTING_FLAG,
                offset,
                f"control_offered_present is {values['control_offered_present']}, not 0 or 1",
            )
        )
    if values["producer_acceptance_known"] not in (0, 1):
        result.findings.append(
            Finding(
                CODE_ACCOUNTING_FLAG,
                offset,
                f"producer_acceptance_known is {values['producer_acceptance_known']}, not 0 or 1",
            )
        )
    if values["accounting_origin"] != ACCOUNTING_ORIGIN_RECORDER:
        result.findings.append(
            Finding(
                CODE_ACCOUNTING_ORIGIN,
                offset,
                "spool accounting is always written by the recorder; recovery-rebuilt "
                "accounting belongs to the NRF session",
            )
        )

    for name, base, required_tag, related_counters, allowed_identity_kinds in POSITION_LAYOUT:
        tag = payload[base]
        identity_kind = struct.unpack_from("<I", payload, base + 4)[0]
        ordinal = struct.unpack_from("<Q", payload, base + 8)[0]
        identity_value = struct.unpack_from("<Q", payload, base + 16)[0]
        if tag not in (POSITION_ABSENT, POSITION_ORDINAL, POSITION_PRODUCER_IDENTITY):
            result.findings.append(
                Finding(
                    CODE_POSITION_TAG,
                    offset,
                    f"{name} position tag {tag} is not 0, 1, or 2",
                )
            )
            continue
        if tag == POSITION_ORDINAL and (identity_kind != 0 or identity_value != 0):
            result.findings.append(
                Finding(
                    CODE_POSITION_TAG,
                    offset,
                    f"{name} is an ordinal position but carries a nonzero identity member",
                )
            )
        elif tag == POSITION_PRODUCER_IDENTITY and ordinal != 0:
            result.findings.append(
                Finding(
                    CODE_POSITION_TAG,
                    offset,
                    f"{name} is a producer-identity position but carries a nonzero ordinal",
                )
            )
        elif tag == POSITION_ABSENT and (identity_kind != 0 or ordinal != 0 or identity_value != 0):
            result.findings.append(
                Finding(
                    CODE_POSITION_TAG,
                    offset,
                    f"{name} is absent but carries a nonzero member",
                )
            )

        present = tag != POSITION_ABSENT
        # The contract fixes the tag form for each position: a loss after
        # acceptance names the accepted item's ordinal; a rejection before
        # acceptance names the producer identity. A loss recorded as a producer
        # identity would read like a pre-acceptance event, and a rejection
        # recorded as an ordinal would name an ordinal the item never had.
        if present and tag != required_tag:
            if required_tag == POSITION_ORDINAL:
                want, event = "an ordinal (tag 1)", "loss after acceptance"
            else:
                want, event = "a producer identity (tag 2)", "rejection before acceptance"
            result.findings.append(
                Finding(
                    CODE_POSITION_TAG_FORM,
                    offset,
                    f"{name} records the first {event} and must carry {want}, not tag {tag}",
                )
            )

        # When the position carries a producer identity, ``identity_kind`` is the
        # active member and MUST be a registered kind for this position's plane:
        # a data rejection names a data message kind, a control rejection a
        # control record-set kind. A value outside the registry (or a control
        # kind in a data position, or vice versa) is one no finalizer can map to
        # a single NRF kind string.
        if (
            allowed_identity_kinds is not None
            and tag == POSITION_PRODUCER_IDENTITY
            and identity_kind not in allowed_identity_kinds
        ):
            result.findings.append(
                Finding(
                    CODE_IDENTITY_KIND,
                    offset,
                    f"{name} carries identity_kind {identity_kind}, which is not a "
                    f"registered producer kind for its plane",
                )
            )

        # The first loss/rejection is latched in bounded storage (section 4.3),
        # so a position is present exactly when the sum of the counters whose
        # first event it names is nonzero.
        related_sum = sum(values[counter_name] for counter_name in related_counters)
        if present and related_sum == 0:
            result.findings.append(
                Finding(
                    CODE_POSITION_CONSISTENCY,
                    offset,
                    f"{name} is present but the counters it names all sum to zero",
                )
            )
        if not present and related_sum > 0:
            result.findings.append(
                Finding(
                    CODE_POSITION_CONSISTENCY,
                    offset,
                    f"an accepted-item loss or rejection occurred (the counters {name} "
                    f"names sum to {related_sum}) but {name} is absent; the first event "
                    "must be latched",
                )
            )

    identities = [
        (
            "runtime_accepted = recorder_accepted + failed_between_runtime_and_recorder",
            values["runtime_accepted"],
            values["recorder_accepted"] + values["failed_between_runtime_and_recorder"],
        ),
        (
            "recorder_accepted = spool_committed + lost_between_recorder_and_spool",
            values["recorder_accepted"],
            values["spool_committed"] + values["lost_between_recorder_and_spool"],
        ),
        (
            "control_accepted = control_spool_committed + "
            "lost_between_control_acceptance_and_spool",
            values["control_accepted"],
            values["control_spool_committed"] + values["lost_between_control_acceptance_and_spool"],
        ),
    ]
    if values["control_offered_present"]:
        identities.append(
            (
                "control_offered = control_accepted + control_rejected",
                values["control_offered"],
                values["control_accepted"] + values["control_rejected"],
            )
        )
    for statement, left, right in identities:
        if left != right:
            result.findings.append(
                Finding(
                    CODE_ACCOUNTING_IDENTITY,
                    offset,
                    f"{statement} does not hold: {left} != {right}",
                )
            )


def _check_accounting_exceeds_prefix(payload: bytes, offset: int, result: ScanResult) -> None:
    """Section 7 layer 2: the committed counters match the committed prefix.

    Called at the prefix the snapshot seals, with the running totals at that
    point, so a transaction committed after the session end cannot mask a
    summary that claimed more than the sealing prefix held.
    """
    values = _decode_accounting(payload)
    if values["spool_committed"] != result.data_items:
        result.findings.append(
            Finding(
                CODE_ACCOUNTING_EXCEEDS_PREFIX,
                offset,
                f"the summary states {values['spool_committed']} committed data items; the "
                f"committed prefix carries {result.data_items}",
            )
        )
    if values["control_spool_committed"] != result.control_items:
        result.findings.append(
            Finding(
                CODE_ACCOUNTING_EXCEEDS_PREFIX,
                offset,
                f"the summary states {values['control_spool_committed']} committed control "
                f"items; the committed prefix carries {result.control_items}",
            )
        )


def _durable_extent(
    policy: int, checkpoint_extent: int, first_transaction_offset: int, result: ScanResult
) -> int:
    """Report the durable extent the selected policy actually supports.

    Section 6: no implementation, document, or status field may promise more
    than the policy delivers. The floor is the superblock region, which is
    synced before the spool is reported ready under every policy -- a spool
    whose own identity did not survive the crash could not be matched to a
    finalization target at all. Under ``buffered`` that floor is also the
    ceiling: no record is covered, which is what "guarantees nothing against
    power loss" means expressed as an extent.
    """
    if policy == DURABILITY_TRANSACTION_SYNC:
        return result.committed_prefix_end
    if policy == DURABILITY_CHECKPOINT_SYNC:
        return max(first_transaction_offset, min(checkpoint_extent, result.committed_prefix_end))
    return first_transaction_offset
