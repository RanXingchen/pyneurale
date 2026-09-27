#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Generate the normative binary vectors for the private native spool v1.

Every byte a vector contains is decided here, from constants in this file:
there is no clock, no random source, no filesystem state, and no dictionary
iteration order in the output. Running this tool twice, on any platform, with
any Python that runs it at all, must produce identical files -- that is what
makes ``--check`` a meaningful gate rather than a formality.

Usage::

    python generate_vectors.py            # rewrite the vectors and the index
    python generate_vectors.py --check    # verify the committed ones, byte for byte

The vectors are language-neutral. An implementation in any language reads the
``.spool`` files and the expectations in ``index.json``; nothing but this
generator needs Python.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

_SPEC_DIR = Path(__file__).resolve().parents[1]
_VECTOR_DIR = _SPEC_DIR / "vectors"
_INDEX_PATH = _VECTOR_DIR / "index.json"


def _load_spool_format() -> Any:
    """Load the sibling reference module by path.

    The tools directory is deliberately not a Python package -- it is
    specification material that happens to be executable -- so the import is
    done the way the test suite does it.
    """
    path = Path(__file__).resolve().parent / "spool_format.py"
    spec = importlib.util.spec_from_file_location("native_spool_v1_format", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the spool reference at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fmt = _load_spool_format()

# --- fixed inputs ---------------------------------------------------------
#
# Chosen once and never derived from the environment. The plan document stands
# in for the fingerprinted RecordingPlan document of the native replay
# extension: the container never parses it, so a short one is as good as a
# real one for exercising the container, and a short one keeps the vectors
# readable in a hex dump.

SESSION_ID = "spool-vector-session"
SESSION_UUID = bytes.fromhex("4e524c53504f4f4c5645435430303031")
CREATED_UNIX_NANOS = 1_767_225_600_000_000_000
PLAN_DOCUMENT = (
    b'{"coverage":"full","extension_version":1,"native_schema_id":11,'
    b'"planned_signal_ids":[1,2],"recorded_signal_ids":[1,2]}'
)

FRAME_PAYLOAD = bytes.fromhex("01000000000000000200000000000000e803000000000000")
BLOCK_PAYLOAD_A = bytes(range(0, 24))
BLOCK_PAYLOAD_B = bytes(range(24, 44))
DISCONTINUITY_PAYLOAD = bytes.fromhex("03000000000000000400000000000000")
GAP_PAYLOAD = bytes.fromhex("0500000000000000")
CONTROL_PAYLOAD = b"trial-start"
FAULT_PAYLOAD = b"writer stalled"

NANOS = 1_767_225_601_000_000_000


class SpoolBuilder:
    """Assemble a spool image and remember where every part landed.

    The offsets are what the corrupt vectors need: a mutation that says "flip
    the record payload byte in the second transaction" has to be expressed as
    an offset somewhere, and computing it here beats re-deriving it from a
    scan of the finished file.
    """

    def __init__(self, policy: int, *, minor: int = fmt.VERSION_MINOR) -> None:
        region = fmt.encode_superblock_region(
            fmt.SuperblockFields(
                session_id=SESSION_ID,
                session_uuid=SESSION_UUID,
                plan_document=PLAN_DOCUMENT,
                durability_policy=policy,
                created_unix_nanos=CREATED_UNIX_NANOS,
                version_minor=minor,
            )
        )
        self.image = bytearray(region)
        self.first_transaction_offset = len(region)
        self.previous_offset = 0
        self.next_id = 1
        self.transactions: list[tuple[int, int]] = []

    def commit(self, records: list[bytes], **overrides: Any) -> int:
        """Append one committed transaction; return the offset it starts at."""
        offset = len(self.image)
        transaction_id = overrides.pop("transaction_id", self.next_id)
        back_link = overrides.pop("previous_transaction_offset", self.previous_offset)
        encoded = fmt.encode_transaction(
            transaction_id,
            records,
            previous_transaction_offset=back_link,
            begin_unix_nanos=NANOS + transaction_id,
            **overrides,
        )
        self.image += encoded
        self.transactions.append((offset, len(encoded)))
        self.previous_offset = offset
        self.next_id = transaction_id + 1
        return offset

    def bytes(self) -> bytes:
        return bytes(self.image)


def _data_transaction_records() -> list[bytes]:
    return [
        fmt.encode_record(
            fmt.KIND_FRAME, FRAME_PAYLOAD, logical_ordinal=1, record_unix_nanos=NANOS
        ),
        fmt.encode_record(fmt.KIND_SIGNAL_BLOCK, BLOCK_PAYLOAD_A, logical_ordinal=1),
        fmt.encode_record(fmt.KIND_SIGNAL_BLOCK, BLOCK_PAYLOAD_B, logical_ordinal=1),
        fmt.encode_record(
            fmt.KIND_DISCONTINUITY,
            DISCONTINUITY_PAYLOAD,
            logical_ordinal=2,
            record_unix_nanos=NANOS,
        ),
        fmt.encode_record(fmt.KIND_SIGNAL_GAP, GAP_PAYLOAD, logical_ordinal=2),
        fmt.encode_record(fmt.KIND_CONTROL, CONTROL_PAYLOAD, logical_ordinal=1),
    ]


def _accounting_payload(**overrides: Any) -> bytes:
    counters = {
        "runtime_accepted": 2,
        "recorder_accepted": 2,
        "spool_committed": 2,
        "rejected_before_runtime_acceptance": 0,
        "failed_between_runtime_and_recorder": 0,
        "lost_between_recorder_and_spool": 0,
        "control_offered": 1,
        "control_accepted": 1,
        "control_spool_committed": 1,
        "control_rejected": 0,
        "lost_between_control_acceptance_and_spool": 0,
        "rejected_after_close_data": 0,
        "rejected_after_close_control": 0,
    }
    counters.update(overrides.pop("counters", {}))
    return fmt.encode_accounting(counters, **overrides)


def _complete(policy: int, *, minor: int = fmt.VERSION_MINOR) -> SpoolBuilder:
    """A spool that recorded two data items and one control item and ended cleanly."""
    builder = SpoolBuilder(policy, minor=minor)
    builder.commit(_data_transaction_records())
    checkpoint_extent = len(builder.image)
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_CHECKPOINT,
                fmt.encode_checkpoint(checkpoint_extent, NANOS + 100, policy),
            )
        ]
    )
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder


def _patch(image: bytes, offset: int, data: bytes) -> bytes:
    patched = bytearray(image)
    patched[offset : offset + len(data)] = data
    return bytes(patched)


def _reseal_transaction(image: bytes, offset: int, length: int) -> bytes:
    """Recompute the body and trailer checksums of one transaction in place.

    Used by the vectors whose point is a *different* violation: without this
    they would trip a checksum first and never reach the rule they exist to
    exercise.
    """
    patched = bytearray(image)
    trailer_start = offset + length - fmt.TRANSACTION_TRAILER_BYTES
    struct.pack_into(
        "<I", patched, trailer_start + 36, fmt.crc32c(bytes(patched[offset:trailer_start]))
    )
    struct.pack_into(
        "<I",
        patched,
        trailer_start + 44,
        fmt.crc32c(bytes(patched[trailer_start : trailer_start + 44])),
    )
    return bytes(patched)


# --- the vectors ----------------------------------------------------------


def _valid_empty() -> bytes:
    """A prepared spool: superblock region committed, nothing recorded.

    Section 7's one discardable case. A reader must be able to say "this holds
    no committed record" without guessing.
    """
    return SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC).bytes()


def _valid_minimal() -> bytes:
    """One committed data transaction and no session end -- a spool still open."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    return builder.bytes()


def _valid_complete() -> bytes:
    return _complete(fmt.DURABILITY_TRANSACTION_SYNC).bytes()


def _valid_checkpoint_sync() -> bytes:
    return _complete(fmt.DURABILITY_CHECKPOINT_SYNC).bytes()


def _valid_buffered() -> bytes:
    return _complete(fmt.DURABILITY_BUFFERED).bytes()


def _valid_unknown_minor() -> bytes:
    """A future minor version this reader has never seen, using only v1.0 kinds."""
    return _complete(fmt.DURABILITY_TRANSACTION_SYNC, minor=7).bytes()


def _valid_aborted_with_losses() -> bytes:
    """A legal spool for a session that lost data: the counters say so."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "runtime_accepted": 5,
                        "recorder_accepted": 4,
                        "spool_committed": 2,
                        "failed_between_runtime_and_recorder": 1,
                        "lost_between_recorder_and_spool": 2,
                        "rejected_before_runtime_acceptance": 3,
                        "control_offered": 2,
                        "control_accepted": 1,
                        "control_spool_committed": 1,
                        "control_rejected": 1,
                    },
                    positions={
                        "data_first_loss": fmt.encode_position(fmt.POSITION_ORDINAL, ordinal=3),
                        "data_first_rejection": fmt.encode_position(
                            fmt.POSITION_PRODUCER_IDENTITY, identity_kind=1, identity_value=97
                        ),
                        "control_first_rejection": fmt.encode_position(
                            fmt.POSITION_PRODUCER_IDENTITY, identity_kind=3, identity_value=5
                        ),
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(
                    fmt.CAPTURE_OUTCOME_ABORTED,
                    NANOS + 300,
                    requested_terminal_intent=fmt.REQUESTED_TERMINAL_INTENT_ABORTED,
                    terminal_reason=b"operator abort",
                ),
            ),
        ]
    )
    return builder.bytes()


def _valid_fault_record() -> bytes:
    """A faulted capture: the primary fault record is committed before the end."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit([fmt.encode_record(fmt.KIND_FAULT, FAULT_PAYLOAD, record_unix_nanos=NANOS + 5)])
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(
                    fmt.CAPTURE_OUTCOME_FAULTED,
                    NANOS + 400,
                    requested_terminal_intent=fmt.REQUESTED_TERMINAL_INTENT_FAULT,
                    primary_fault_committed=True,
                    terminal_reason=b"recorder fault",
                ),
            ),
        ]
    )
    return builder.bytes()


def _valid_aborted_then_faulted() -> bytes:
    """An abort that a recorder fault during the drain escalated to faulted.

    The contract's key escalation case: the caller aborted first (so
    ``requested_terminal_intent`` latched ``aborted``), and a primary recorder
    fault arrived during the drain, so ``capture_outcome`` is ``faulted`` and
    the fault row committed (``primary_fault_committed``). The request and the
    outcome are separate fields, and neither may be recovered from the other.
    """
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit([fmt.encode_record(fmt.KIND_FAULT, FAULT_PAYLOAD, record_unix_nanos=NANOS + 5)])
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(
                    fmt.CAPTURE_OUTCOME_FAULTED,
                    NANOS + 400,
                    requested_terminal_intent=fmt.REQUESTED_TERMINAL_INTENT_ABORTED,
                    primary_fault_committed=True,
                    terminal_reason=b"abort then fault",
                ),
            ),
        ]
    )
    return builder.bytes()


def _torn_inside_body() -> bytes:
    """The process died mid-transaction: the last commit trailer never landed."""
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    last_offset, last_length = builder.transactions[-1]
    return builder.bytes()[: last_offset + last_length // 2]


def _torn_missing_trailer() -> bytes:
    """Every record of the last transaction is on disk; the trailer is not."""
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    last_offset, last_length = builder.transactions[-1]
    return builder.bytes()[: last_offset + last_length - fmt.TRANSACTION_TRAILER_BYTES]


def _torn_inside_record_header() -> bytes:
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    last_offset, _ = builder.transactions[-1]
    return builder.bytes()[: last_offset + fmt.TRANSACTION_HEADER_BYTES + 12]


def _torn_superblock() -> bytes:
    return _valid_complete()[:100]


def _torn_plan_document() -> bytes:
    """The superblock landed; the plan document behind it did not."""
    return _valid_complete()[: fmt.SUPERBLOCK_BYTES + 8]


def _corrupt_magic() -> bytes:
    return _patch(_valid_complete(), 0, b"NOTSPOOL")


def _corrupt_superblock_checksum() -> bytes:
    return _patch(_valid_complete(), 80, b"\xff")


def _corrupt_plan_document() -> bytes:
    return _patch(_valid_complete(), fmt.SUPERBLOCK_BYTES + 4, b"\x00")


def _nonzero_plan_padding() -> bytes:
    """A byte hiding in the gap between the plan document and the first transaction.

    No checksum reaches it: the plan CRC stops at ``plan_bytes`` and the
    superblock CRC stops at byte 252. The zero rule is what covers it, and a
    reader that skipped the check would accept a spool carrying a byte nobody
    can account for.
    """
    image = _valid_complete()
    first_transaction_offset = struct.unpack_from("<Q", image, 88)[0]
    return _patch(image, first_transaction_offset - 2, b"\x01")


def _unknown_major_version() -> bytes:
    image = bytearray(_valid_complete())
    struct.pack_into("<H", image, 8, 2)
    struct.pack_into("<I", image, 252, fmt.crc32c(bytes(image[:252])))
    return bytes(image)


def _undefined_superblock_field() -> bytes:
    """A checksum algorithm this version does not define."""
    image = bytearray(_valid_complete())
    struct.pack_into("<B", image, 17, 9)
    struct.pack_into("<I", image, 252, fmt.crc32c(bytes(image[:252])))
    return bytes(image)


def _corrupt_record_payload() -> bytes:
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    first_offset, _ = builder.transactions[0]
    payload_offset = first_offset + fmt.TRANSACTION_HEADER_BYTES + fmt.RECORD_HEADER_BYTES
    return _patch(builder.bytes(), payload_offset, b"\xa5")


def _corrupt_record_header() -> bytes:
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    first_offset, _ = builder.transactions[0]
    return _patch(builder.bytes(), first_offset + fmt.TRANSACTION_HEADER_BYTES + 2, b"\x07")


def _corrupt_body_checksum() -> bytes:
    """The trailer's body checksum disagrees with a body that is otherwise intact.

    Every single-byte flip inside a body is caught by a narrower checksum
    first, so the only way to exercise the body checksum on its own is to
    corrupt the stored value and reseal the trailer around it -- which is also
    what a partially rewritten trailer would look like.
    """
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    first_offset, first_length = builder.transactions[0]
    trailer_start = first_offset + first_length - fmt.TRANSACTION_TRAILER_BYTES
    image = bytearray(_patch(builder.bytes(), trailer_start + 36, b"\x00\x00\x00\x00"))
    struct.pack_into(
        "<I",
        image,
        trailer_start + 44,
        fmt.crc32c(bytes(image[trailer_start : trailer_start + 44])),
    )
    return bytes(image)


def _corrupt_trailer_checksum() -> bytes:
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    first_offset, first_length = builder.transactions[0]
    trailer_start = first_offset + first_length - fmt.TRANSACTION_TRAILER_BYTES
    return _patch(builder.bytes(), trailer_start + 44, b"\x00\x00\x00\x00")


def _corrupt_transaction_header_checksum() -> bytes:
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    first_offset, _ = builder.transactions[0]
    return _patch(builder.bytes(), first_offset + 44, b"\x00\x00\x00\x00")


def _nonzero_padding() -> bytes:
    """Padding bytes carry no information, so a nonzero one is unexplained."""
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    first_offset, first_length = builder.transactions[0]
    # The control record's payload is 11 bytes, so it is followed by five
    # padding bytes; it is the last record of the first transaction.
    trailer_start = first_offset + first_length - fmt.TRANSACTION_TRAILER_BYTES
    image = _patch(builder.bytes(), trailer_start - 1, b"\x01")
    return _reseal_transaction(image, first_offset, first_length)


def _duplicate_transaction_id() -> bytes:
    """The same transaction id committed twice -- a replayed or doubled append."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(_data_transaction_records(), transaction_id=1)
    return builder.bytes()


def _transaction_id_gap() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(_data_transaction_records(), transaction_id=7)
    return builder.bytes()


def _bad_back_link() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(_data_transaction_records(), previous_transaction_offset=0)
    return builder.bytes()


def _record_count_disagreement() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    offset, length = builder.transactions[0]
    trailer_start = offset + length - fmt.TRANSACTION_TRAILER_BYTES
    patched = bytearray(_patch(builder.bytes(), trailer_start + 24, struct.pack("<I", 5)))
    # Reseal only the trailer: the body is untouched, so the record count is
    # the single thing this vector is about.
    struct.pack_into(
        "<I",
        patched,
        trailer_start + 44,
        fmt.crc32c(bytes(patched[trailer_start : trailer_start + 44])),
    )
    return bytes(patched)


def _empty_transaction() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit([], n_records=0)
    return builder.bytes()


def _trailer_item_counts() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records(), data_items=9)
    return builder.bytes()


def _unknown_record_kind() -> bytes:
    """A committed record this container version cannot name.

    Not a tolerable skip: the writer committed it, so a prefix that stepped
    over it would be promoting a session that lost a record nobody counted.
    """
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_FRAME, FRAME_PAYLOAD, logical_ordinal=1),
            fmt.encode_record(4095, b"from a later minor version"),
        ]
    )
    return builder.bytes()


def _records_after_session_end() -> bytes:
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    return builder.bytes()


def _session_end_without_accounting() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            )
        ]
    )
    return builder.bytes()


def _accounting_with_items() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_FRAME, FRAME_PAYLOAD, logical_ordinal=1),
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "runtime_accepted": 1,
                        "recorder_accepted": 1,
                        "spool_committed": 1,
                        "control_offered": 0,
                        "control_accepted": 0,
                        "control_spool_committed": 0,
                    }
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_identity_violation() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(counters={"recorder_accepted": 5}),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_uint64_wraparound() -> bytes:
    """Counters that only satisfy the identities after unsigned wraparound."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "runtime_accepted": 0,
                        "recorder_accepted": (1 << 64) - 1,
                        "spool_committed": 0,
                        "failed_between_runtime_and_recorder": 1,
                        "lost_between_recorder_and_spool": (1 << 64) - 1,
                        "control_offered": 0,
                        "control_accepted": 0,
                        "control_spool_committed": 0,
                    }
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_ABORTED, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_exceeds_prefix() -> bytes:
    """Internally consistent counters that claim more than the spool holds.

    This is the failure layer 1 cannot see: every identity holds, and the
    session contains two data items rather than the ninety-nine claimed.
    """
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "runtime_accepted": 99,
                        "recorder_accepted": 99,
                        "spool_committed": 99,
                    }
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_origin_not_recorder() -> bytes:
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload(accounting_origin=1)
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _checkpoint_extent_beyond_transaction() -> bytes:
    """A checkpoint claiming bytes that were not yet written when it was."""
    builder = SpoolBuilder(fmt.DURABILITY_CHECKPOINT_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_CHECKPOINT,
                fmt.encode_checkpoint(1 << 40, NANOS + 100, fmt.DURABILITY_CHECKPOINT_SYNC),
            )
        ]
    )
    return builder.bytes()


def _container_payload_size() -> bytes:
    """A session-end record whose payload is not the fixed 80 bytes."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(fmt.KIND_SESSION_END, b"\x00" * 8),
        ]
    )
    return builder.bytes()


def _ordinal_block_mismatch() -> bytes:
    """A signal_block whose logical_ordinal is not its owning frame's."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_FRAME, FRAME_PAYLOAD, logical_ordinal=1),
            fmt.encode_record(fmt.KIND_SIGNAL_BLOCK, BLOCK_PAYLOAD_A, logical_ordinal=5),
        ]
    )
    return builder.bytes()


def _ordinal_gap_mismatch() -> bytes:
    """A signal_gap whose logical_ordinal is not its owning discontinuity's."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_DISCONTINUITY, DISCONTINUITY_PAYLOAD, logical_ordinal=2),
            fmt.encode_record(fmt.KIND_SIGNAL_GAP, GAP_PAYLOAD, logical_ordinal=7),
        ]
    )
    return builder.bytes()


def _ordinal_session_end_nonzero() -> bytes:
    """A session_end record that carries a nonzero logical_ordinal."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
                logical_ordinal=3,
            ),
        ]
    )
    return builder.bytes()


def _record_flags_nonzero() -> bytes:
    """A record whose record_flags is not the version-1 zero."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    records = _data_transaction_records()
    records[0] = fmt.encode_record(
        fmt.KIND_FRAME, FRAME_PAYLOAD, logical_ordinal=1, record_unix_nanos=NANOS, record_flags=1
    )
    builder.commit(records)
    checkpoint_extent = len(builder.image)
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_CHECKPOINT,
                fmt.encode_checkpoint(
                    checkpoint_extent, NANOS + 100, fmt.DURABILITY_TRANSACTION_SYNC
                ),
            )
        ]
    )
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _session_end_undefined_outcome() -> bytes:
    """A session_end whose capture_outcome is not one of the three defined values."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(9, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _checkpoint_policy_mismatch() -> bytes:
    """A checkpoint whose durability_policy is not the superblock's."""
    builder = SpoolBuilder(fmt.DURABILITY_CHECKPOINT_SYNC)
    builder.commit(_data_transaction_records())
    extent = len(builder.image)
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_CHECKPOINT,
                fmt.encode_checkpoint(extent, NANOS + 100, fmt.DURABILITY_TRANSACTION_SYNC),
            )
        ]
    )
    return builder.bytes()


def _accounting_flag_invalid() -> bytes:
    """An accounting flag byte that is neither 0 nor 1."""
    accounting = bytearray(_accounting_payload())
    accounting[104] = 2  # control_offered_present, not 0 or 1
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, bytes(accounting)),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_tag_invalid() -> bytes:
    """An accounting position whose tag is not 0, 1, or 2."""
    accounting = bytearray(_accounting_payload())
    # Keep the identities holding while a loss is recorded, then give its
    # first-loss position an undefined tag.
    struct.pack_into("<Q", accounting, 0, 4)  # runtime_accepted
    struct.pack_into("<Q", accounting, 8, 4)  # recorder_accepted
    struct.pack_into("<Q", accounting, 40, 2)  # lost_between_recorder_and_spool
    accounting[112] = 9  # data_first_loss tag, undefined
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, bytes(accounting)),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_consistency() -> bytes:
    """A first-loss position present while its loss counter is zero."""
    accounting = bytearray(_accounting_payload())
    accounting[112] = fmt.POSITION_ORDINAL
    struct.pack_into("<Q", accounting, 120, 3)  # the position's ordinal
    # lost_between_recorder_and_spool stays zero, so a present loss position is
    # inconsistent with the counter.
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, bytes(accounting)),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _multiple_accounting() -> bytes:
    """Two accounting snapshots committed; the summary is written once."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _multiple_session_end() -> bytes:
    """Two session_end records committed; a session has one capture outcome."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_FAULTED, NANOS + 201),
            ),
        ]
    )
    return builder.bytes()


def _accounting_without_session_end() -> bytes:
    """An accounting snapshot committed with no session_end to seal it."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit([fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, _accounting_payload())])
    return builder.bytes()


def _corrupt_tail_wrong_magic() -> bytes:
    """A complete header with the wrong begin magic: corruption, not truncation."""
    image = bytearray(_valid_complete())
    image.extend(b"\x00" * fmt.TRANSACTION_HEADER_BYTES)
    return bytes(image)


def _corrupt_tail_bad_header_bytes() -> bytes:
    """A checksum-valid header whose header_bytes is not 48: corruption, not truncation."""
    builder = _complete(fmt.DURABILITY_TRANSACTION_SYNC)
    image = bytearray(builder.bytes())
    previous_start = builder.transactions[-1][0]
    header = bytearray(fmt.TRANSACTION_HEADER_BYTES)
    header[0:8] = fmt.TRANSACTION_BEGIN_MAGIC
    struct.pack_into("<QQ", header, 8, builder.next_id, previous_start)
    struct.pack_into("<II", header, 24, 1, 64)
    struct.pack_into("<Q", header, 32, NANOS + 999)
    struct.pack_into("<I", header, 40, 0)
    struct.pack_into("<I", header, 44, fmt.crc32c(bytes(header[:44])))
    image.extend(header)
    return bytes(image)


def _accounting_position_loss_missing() -> bytes:
    """An accepted-item loss with no first-loss ordinal latched.

    ``failed_between_runtime_and_recorder`` counts an item that was accepted
    (and so has a data-message ordinal) but failed before the recorder accepted
    it; ``data_first_loss`` names the first such loss across both post-acceptance
    handoffs, so a nonzero failure with the position absent is inconsistent.
    """
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "runtime_accepted": 3,
                        "failed_between_runtime_and_recorder": 1,
                    }
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_data_loss_tag() -> bytes:
    """A first data loss recorded as a producer identity, not an ordinal."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "runtime_accepted": 4,
                        "recorder_accepted": 4,
                        "lost_between_recorder_and_spool": 2,
                    },
                    positions={
                        "data_first_loss": fmt.encode_position(
                            fmt.POSITION_PRODUCER_IDENTITY, identity_kind=2, identity_value=97
                        )
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_data_rejection_tag() -> bytes:
    """A first data rejection recorded as an ordinal, not a producer identity."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={"rejected_before_runtime_acceptance": 3},
                    positions={
                        "data_first_rejection": fmt.encode_position(fmt.POSITION_ORDINAL, ordinal=7)
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_control_loss_tag() -> bytes:
    """A first control loss recorded as a producer identity, not an ordinal."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "control_offered": 3,
                        "control_accepted": 3,
                        "lost_between_control_acceptance_and_spool": 2,
                    },
                    positions={
                        "control_first_loss": fmt.encode_position(
                            fmt.POSITION_PRODUCER_IDENTITY, identity_kind=1, identity_value=5
                        )
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_control_rejection_tag() -> bytes:
    """A first control rejection recorded as an ordinal, not a producer identity."""
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "control_offered": 3,
                        "control_accepted": 1,
                        "control_rejected": 2,
                    },
                    positions={
                        "control_first_rejection": fmt.encode_position(
                            fmt.POSITION_ORDINAL, ordinal=5
                        )
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_identity_kind_unknown() -> bytes:
    """A first control rejection whose identity_kind is not in the registry.

    The number is the only bridge between the spool and the NRF kind string; a
    value no registry defines is one the finalizer cannot map, so two writers
    must not be free to pick different digits for the same kind.
    """
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={
                        "control_offered": 3,
                        "control_accepted": 1,
                        "control_rejected": 2,
                    },
                    positions={
                        "control_first_rejection": fmt.encode_position(
                            fmt.POSITION_PRODUCER_IDENTITY, identity_kind=999, identity_value=123
                        )
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


def _accounting_position_identity_kind_wrong_plane() -> bytes:
    """A first data rejection that names a control record-set kind.

    A data rejection names a data message kind; a control kind in a data
    position would let the finalizer write a control record-set kind into the
    data first-rejected-message-kind field, so the registry is partitioned by
    plane.
    """
    builder = SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(_data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(
                fmt.KIND_ACCOUNTING_SNAPSHOT,
                _accounting_payload(
                    counters={"rejected_before_runtime_acceptance": 3},
                    positions={
                        "data_first_rejection": fmt.encode_position(
                            fmt.POSITION_PRODUCER_IDENTITY, identity_kind=3, identity_value=97
                        )
                    },
                ),
            ),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(fmt.CAPTURE_OUTCOME_NORMAL, NANOS + 200),
            ),
        ]
    )
    return builder.bytes()


#: Every vector, with the one sentence that says why it exists. Order is the
#: order of the index file; it is fixed here rather than sorted, so related
#: cases read together.
VECTORS: tuple[tuple[str, Any, str], ...] = (
    ("valid-empty", _valid_empty, "A prepared spool holding no committed record."),
    ("valid-minimal", _valid_minimal, "One committed data transaction, no session end."),
    ("valid-complete", _valid_complete, "Data, a checkpoint, and a clean session end."),
    (
        "valid-checkpoint-sync",
        _valid_checkpoint_sync,
        "The same session under checkpoint sync: the durable extent stops at the checkpoint.",
    ),
    (
        "valid-buffered",
        _valid_buffered,
        "The same session under buffered writes: nothing is claimed as durable.",
    ),
    (
        "valid-unknown-minor",
        _valid_unknown_minor,
        "A future minor version using only defined record kinds; readable.",
    ),
    (
        "valid-aborted-with-losses",
        _valid_aborted_with_losses,
        "A legally incomplete session: the counters and first positions record the losses.",
    ),
    ("valid-fault-record", _valid_fault_record, "A faulted capture with a committed fault record."),
    (
        "valid-aborted-then-faulted",
        _valid_aborted_then_faulted,
        "An abort escalated to faulted by a recorder fault during the drain: request and outcome differ.",
    ),
    ("torn-inside-body", _torn_inside_body, "The last transaction was cut in half by a crash."),
    ("torn-missing-trailer", _torn_missing_trailer, "Records landed; the commit trailer did not."),
    (
        "torn-inside-record-header",
        _torn_inside_record_header,
        "The file ends part way through a record header.",
    ),
    ("torn-superblock", _torn_superblock, "The file is shorter than the fixed superblock."),
    (
        "torn-plan-document",
        _torn_plan_document,
        "The superblock survived; the plan document did not.",
    ),
    ("corrupt-magic", _corrupt_magic, "The file does not begin with the spool magic."),
    (
        "corrupt-superblock-checksum",
        _corrupt_superblock_checksum,
        "A flipped superblock byte, detected by the superblock checksum.",
    ),
    (
        "corrupt-plan-document",
        _corrupt_plan_document,
        "The stored plan no longer hashes to the recorded plan fingerprint.",
    ),
    (
        "nonzero-plan-padding",
        _nonzero_plan_padding,
        "A nonzero byte in the unchecksummed gap behind the plan document.",
    ),
    (
        "unknown-major-version",
        _unknown_major_version,
        "A container major version this reader refuses.",
    ),
    (
        "undefined-superblock-field",
        _undefined_superblock_field,
        "A checksum algorithm identifier version 1 does not define.",
    ),
    (
        "corrupt-record-payload",
        _corrupt_record_payload,
        "A flipped payload byte, detected by the record checksum.",
    ),
    (
        "corrupt-record-header",
        _corrupt_record_header,
        "A flipped record header byte, detected by the record header checksum.",
    ),
    (
        "corrupt-body-checksum",
        _corrupt_body_checksum,
        "A resealed trailer whose body checksum no longer matches the body.",
    ),
    (
        "corrupt-trailer-checksum",
        _corrupt_trailer_checksum,
        "A flipped trailer byte, detected by the trailer checksum.",
    ),
    (
        "corrupt-transaction-header-checksum",
        _corrupt_transaction_header_checksum,
        "A flipped transaction header byte, detected by its checksum.",
    ),
    ("nonzero-padding", _nonzero_padding, "Record padding that carries something."),
    (
        "duplicate-transaction-id",
        _duplicate_transaction_id,
        "The same transaction id committed twice.",
    ),
    ("transaction-id-gap", _transaction_id_gap, "A transaction id that skips its predecessor."),
    ("bad-back-link", _bad_back_link, "A back link that does not name the previous transaction."),
    (
        "record-count-disagreement",
        _record_count_disagreement,
        "Header and trailer disagree on how many records were committed.",
    ),
    ("empty-transaction", _empty_transaction, "A transaction that commits nothing."),
    (
        "trailer-item-counts",
        _trailer_item_counts,
        "A trailer claiming more data items than the transaction carries.",
    ),
    (
        "unknown-record-kind",
        _unknown_record_kind,
        "A committed record kind this version cannot name.",
    ),
    (
        "records-after-session-end",
        _records_after_session_end,
        "A transaction committed after the session had ended.",
    ),
    (
        "session-end-without-accounting",
        _session_end_without_accounting,
        "A session end that seals no accounting snapshot.",
    ),
    (
        "accounting-with-items",
        _accounting_with_items,
        "An accounting snapshot sharing its transaction with a data item.",
    ),
    (
        "accounting-identity-violation",
        _accounting_identity_violation,
        "Counters that do not satisfy the per-handoff identities.",
    ),
    (
        "accounting-uint64-wraparound",
        _accounting_uint64_wraparound,
        "Counters that satisfy the identities only after unsigned 64-bit wraparound.",
    ),
    (
        "accounting-exceeds-prefix",
        _accounting_exceeds_prefix,
        "Self-consistent counters claiming more than the committed prefix holds.",
    ),
    (
        "accounting-origin-not-recorder",
        _accounting_origin_not_recorder,
        "Spool accounting claiming an origin only a finalizer could have.",
    ),
    (
        "checkpoint-extent-beyond-transaction",
        _checkpoint_extent_beyond_transaction,
        "A checkpoint claiming more synced bytes than existed when it was written.",
    ),
    (
        "container-payload-size",
        _container_payload_size,
        "A container-owned record whose payload is not its fixed size.",
    ),
    (
        "ordinal-block-mismatch",
        _ordinal_block_mismatch,
        "A signal_block whose logical_ordinal is not its owning frame's.",
    ),
    (
        "ordinal-gap-mismatch",
        _ordinal_gap_mismatch,
        "A signal_gap whose logical_ordinal is not its owning discontinuity's.",
    ),
    (
        "ordinal-session-end-nonzero",
        _ordinal_session_end_nonzero,
        "A record with no owning item that carries a nonzero logical_ordinal.",
    ),
    (
        "record-flags-nonzero",
        _record_flags_nonzero,
        "A record or trailer whose flags field is not the version-1 zero.",
    ),
    (
        "session-end-undefined-outcome",
        _session_end_undefined_outcome,
        "A session_end whose capture_outcome is not a defined value.",
    ),
    (
        "checkpoint-policy-mismatch",
        _checkpoint_policy_mismatch,
        "A checkpoint whose durability policy is not the superblock's.",
    ),
    (
        "accounting-flag-invalid",
        _accounting_flag_invalid,
        "An accounting flag byte that is neither 0 nor 1.",
    ),
    (
        "accounting-position-tag-invalid",
        _accounting_position_tag_invalid,
        "An accounting position whose tag is not 0, 1, or 2.",
    ),
    (
        "accounting-position-consistency",
        _accounting_position_consistency,
        "A first-loss position present while its loss counter is zero.",
    ),
    (
        "multiple-accounting",
        _multiple_accounting,
        "Two accounting snapshots committed; the summary is written once.",
    ),
    (
        "multiple-session-end",
        _multiple_session_end,
        "Two session_end records committed; a session has one outcome.",
    ),
    (
        "accounting-without-session-end",
        _accounting_without_session_end,
        "An accounting snapshot committed with no session_end to seal it.",
    ),
    (
        "corrupt-tail-wrong-magic",
        _corrupt_tail_wrong_magic,
        "A complete header with the wrong begin magic: corruption, not truncation.",
    ),
    (
        "corrupt-tail-bad-header-bytes",
        _corrupt_tail_bad_header_bytes,
        "A checksum-valid header whose header_bytes is not 48.",
    ),
    (
        "accounting-position-loss-missing",
        _accounting_position_loss_missing,
        "An accepted-item loss with no first-loss ordinal latched.",
    ),
    (
        "accounting-position-data-loss-tag",
        _accounting_position_data_loss_tag,
        "A first data loss recorded as a producer identity, not an ordinal.",
    ),
    (
        "accounting-position-data-rejection-tag",
        _accounting_position_data_rejection_tag,
        "A first data rejection recorded as an ordinal, not a producer identity.",
    ),
    (
        "accounting-position-control-loss-tag",
        _accounting_position_control_loss_tag,
        "A first control loss recorded as a producer identity, not an ordinal.",
    ),
    (
        "accounting-position-control-rejection-tag",
        _accounting_position_control_rejection_tag,
        "A first control rejection recorded as an ordinal, not a producer identity.",
    ),
    (
        "accounting-position-identity-kind-unknown",
        _accounting_position_identity_kind_unknown,
        "A first rejection whose identity_kind is not in the registry.",
    ),
    (
        "accounting-position-identity-kind-wrong-plane",
        _accounting_position_identity_kind_wrong_plane,
        "A first data rejection that names a control record-set kind.",
    ),
)


def build_all() -> dict[str, Any]:
    """Build every vector and the index that describes them."""
    entries = []
    files: dict[str, bytes] = {}
    for name, build, description in VECTORS:
        image = build()
        files[f"{name}.spool"] = image
        entries.append(
            {
                "name": name,
                "file": f"{name}.spool",
                "description": description,
                "bytes": len(image),
                "sha256": hashlib.sha256(image).hexdigest(),
                "expect": fmt.scan_spool(image).document(),
            }
        )
    idx = {
        "format": "neurale-native-spool",
        "version_major": fmt.VERSION_MAJOR,
        "version_minor": fmt.VERSION_MINOR,
        "checksum": "crc32c",
        "generator": "specifications/native-spool/v1/tools/generate_vectors.py",
        "vectors": entries,
    }
    return {"index": idx, "files": files}


def _index_bytes(idx: dict[str, Any]) -> bytes:
    return (json.dumps(idx, indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed vectors reproduce byte for byte instead of rewriting them",
    )
    arguments = parser.parse_args(argv)

    built = build_all()
    expected_idx = _index_bytes(built["index"])

    if arguments.check:
        problems: list[str] = []
        for name, image in built["files"].items():
            path = _VECTOR_DIR / name
            if not path.exists():
                problems.append(f"{name}: missing")
            elif path.read_bytes() != image:
                problems.append(f"{name}: differs from the generator output")
        committed = {path.name for path in _VECTOR_DIR.glob("*.spool")}
        for extra in sorted(committed - set(built["files"])):
            problems.append(f"{extra}: not produced by the generator")
        if not _INDEX_PATH.exists() or _INDEX_PATH.read_bytes() != expected_idx:
            problems.append("index.json: differs from the generator output")
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0

    _VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    for path in _VECTOR_DIR.glob("*.spool"):
        path.unlink()
    for name, image in built["files"].items():
        (_VECTOR_DIR / name).write_bytes(image)
    _INDEX_PATH.write_bytes(expected_idx)
    print(f"wrote {len(built['files'])} vectors and index.json to {_VECTOR_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
