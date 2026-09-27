#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Read a native spool container, version 1.

The normative source is ``specifications/native-spool/v1/README.md``. The reader
itself is ``cpp/src/recording/spool_scanner.cpp``, reached through
``neurale._native.recording``; what is left here is the naming layer between the
container's numbers and the vocabulary the NRF session is written in.

**There is one reader.** This module used to hold a second one -- its own
CRC-32C, its own scanner, and its own payload decoders -- and the two were kept
in step only by both happening to pass the same frozen vectors. Two readers of
one private layout is the arrangement where a layout change followed by one side
and not the other leaves both self-consistent and disagreeing, and a spool is the
only copy of a recording. The names below are the part that genuinely belongs
here: a gap reason and a control kind are spelled into NRF fields, so their
spelling is this layer's, not the container's.

Three properties survive the move and are still the whole point:

* **It never writes.** Reading, diagnosing, and finalizing are read-only over a
  spool (contract section 4.5). A spool on disk is opened ``read_only``, so the
  refusal is structural rather than a matter of discipline.
* **It promotes only what a commit trailer covers.** A record is visible if and
  only if the transaction holding it passed every framing and checksum check.
  The first failure ends the committed prefix at the start of that transaction.
* **It reports rather than repairs.** Every rule violation becomes a diagnostic
  code, and a spool carrying any of them is not finalizable.

Records are walked with a cursor rather than materialized. A finalization reads
the committed prefix twice -- once to count what the session will hold, once to
write it -- and two bounded passes over a file cost a fixed buffer, where one
materialized pass costs the whole spool resident in Python objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

# --- container constants ---------------------------------------------------

TRANSACTION_BEGIN_MAGIC: Final = b"NNSTXBEG"
TRANSACTION_END_MAGIC: Final = b"NNSTXEND"

SUPERBLOCK_BYTES: Final = 256
TRANSACTION_HEADER_BYTES: Final = 48
TRANSACTION_TRAILER_BYTES: Final = 48
RECORD_HEADER_BYTES: Final = 32
ALIGNMENT: Final = 8

VERSION_MAJOR: Final = 1

#: Record kinds, in the numbering section 4 fixes.
RECORD_FRAME: Final = 1
RECORD_SIGNAL_BLOCK: Final = 2
RECORD_DISCONTINUITY: Final = 3
RECORD_SIGNAL_GAP: Final = 4
RECORD_CONTROL: Final = 5
RECORD_FAULT: Final = 6
RECORD_ACCOUNTING_SNAPSHOT: Final = 7
RECORD_SESSION_END: Final = 8

#: Which plane a record kind counts against, for the trailer's item counts.
DATA_ITEM_KINDS: Final = frozenset({RECORD_FRAME, RECORD_DISCONTINUITY})
CONTROL_ITEM_KINDS: Final = frozenset({RECORD_CONTROL})

DURABILITY_POLICY_NAMES: Final = ("buffered", "checkpoint_sync", "transaction_sync")
CAPTURE_OUTCOME_NAMES: Final = ("normal", "aborted", "faulted")
TERMINAL_INTENT_NAMES: Final = ("normal", "aborted", "fault")

#: Producer identity kinds, in the registry section 4.4 fixes. The names are
#: what a first-rejection position is written as in the NRF accounting ledger,
#: which is why the table lives on this side of the boundary rather than in the
#: container: the container stores the number, the session stores the name.
PRODUCER_IDENTITY_KINDS: Final = {
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

DATA_IDENTITY_KINDS: Final = frozenset({1, 2})
CONTROL_IDENTITY_KINDS: Final = frozenset(range(3, 12))

#: Gap reasons, in the order the streaming ``GapReason`` enumerates them. Also a
#: name the NRF session stores rather than a number the container does.
GAP_REASONS: Final = (
    "frame_sequence_gap",
    "sample_gap",
    "device_tick_gap",
    "device_restart",
    "source_gap",
    "buffer_exhausted",
    "queue_overflow",
)

GAP_FLAG_MISSING_SAMPLES_KNOWN: Final = 1 << 0
GAP_FLAG_DEVICE_TICKS_AVAILABLE: Final = 1 << 1

FRAME_FLAG_SOURCE_TICK_VALID: Final = 1 << 0
FRAME_FLAG_DEADLINE_VALID: Final = 1 << 1


def pad8(value: int) -> int:
    """Round *value* up to the container's 8-byte alignment."""
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def gap_reason_name(reason: int) -> str:
    """Name a gap reason, or render the number when this build has no name.

    An unknown reason is not guessed at and not dropped: the number is what the
    container stored, and a session that recorded it is more honest than one
    that recorded a neighbouring reason or nothing at all.
    """
    return GAP_REASONS[reason] if reason < len(GAP_REASONS) else str(reason)


def producer_identity_name(kind: int) -> str | None:
    """Name a producer identity kind, or ``None`` when it is not registered."""
    return PRODUCER_IDENTITY_KINDS.get(kind)


def _native() -> Any:
    """Return the native spool bindings, loading them on first use.

    Loaded here rather than at module scope for the reason
    :mod:`neurale.recording._native` states: ``import neurale.recording`` must
    keep working on a build with no native extension. Writing a spool needs no
    native code on this side; only reading one does.
    """
    global _NATIVE
    if _NATIVE is None:
        from neurale._native_loader import load_native_namespace

        _NATIVE = load_native_namespace("recording")
    return _NATIVE


_NATIVE: Any = None


def crc32c(data: bytes | memoryview, seed: int = 0) -> int:
    """CRC-32C (Castagnoli) over *data*, continuing from *seed*.

    Streaming a byte range in chunks yields the same value as one call over the
    whole range, which is what lets a writer that never holds its whole output
    in memory check it with the same kernel the container is read with.
    """
    return int(_native().spool_crc32c(data, seed))


# --- payload types ----------------------------------------------------------
#
# The decoded record payloads are the native reader's own value types
# (``SpoolFramePayload`` and the rest). They are named here as aliases so a
# caller can annotate against them without importing the extension at module
# scope, which the import contract forbids: ``import neurale.recording`` must
# keep working on a build with no native extension.

#: One decoded ``frame`` record.
FramePayload = Any
#: One decoded ``signal_block`` record, including its ``samples`` bytes.
SignalBlockPayload = Any
#: One decoded ``discontinuity`` record. Name its ``reason`` with
#: :func:`gap_reason_name`.
DiscontinuityPayload = Any
#: One decoded ``signal_gap`` record.
SignalGapPayload = Any
#: One decoded ``control`` record, including its ``body`` bytes. Name its
#: ``control_kind`` with :func:`producer_identity_name`.
ControlPayload = Any
#: One decoded recorder ``fault`` record.
FaultPayload = Any
#: One first-loss or first-rejection position from the accounting snapshot.
SpoolPosition = Any
#: One committed record: ``kind``, ``offset``, ``logical_ordinal``,
#: ``record_unix_nanos``, ``payload_bytes``, and a decoded ``payload``.
SpoolRecord = Any


# --- the naming layer over one scan -----------------------------------------


class SpoolSessionEnd:
    """The record that freezes the capture outcome (section 4.5).

    A field is ``None`` where the record stored a value this version does not
    define. That is not the same as the record being absent, which is what
    :attr:`SpoolScan.session_end` being ``None`` means: an undefined outcome is
    a record that exists and cannot be read, and an unrecoverable crash is a
    record that was never written.
    """

    __slots__ = ("_scan",)

    def __init__(self, scan: Any) -> None:
        self._scan = scan

    @property
    def capture_outcome_name(self) -> str | None:
        if not self._scan.has_capture_outcome:
            return None
        return CAPTURE_OUTCOME_NAMES[self._scan.capture_outcome]

    @property
    def requested_terminal_intent_name(self) -> str | None:
        if not self._scan.has_requested_terminal_intent:
            return None
        return TERMINAL_INTENT_NAMES[self._scan.requested_terminal_intent]

    @property
    def primary_fault_committed(self) -> bool | None:
        if not self._scan.has_primary_fault_committed:
            return None
        return bool(self._scan.primary_fault_committed)

    @property
    def terminal_reason(self) -> str:
        return str(self._scan.terminal_reason)

    @property
    def end_unix_nanos(self) -> int:
        return int(self._scan.session_end_unix_nanos)


class SpoolSuperblock:
    """The fixed superblock plus the plan document that follows it."""

    __slots__ = ("_scan",)

    def __init__(self, scan: Any) -> None:
        self._scan = scan

    #: Always 1: a spool whose major version is anything else is rejected before
    #: a superblock exists to report, so a superblock that exists has this one.
    version_major: Final = VERSION_MAJOR

    @property
    def version_minor(self) -> int:
        return int(self._scan.version_minor)

    @property
    def durability_policy(self) -> int:
        return int(self._scan.durability_policy)

    @property
    def durability_policy_name(self) -> str:
        return DURABILITY_POLICY_NAMES[self.durability_policy]

    @property
    def plan_fingerprint(self) -> str:
        return bytes(self._scan.plan_fingerprint).hex()

    @property
    def session_uuid(self) -> bytes:
        return bytes(self._scan.session_uuid)

    @property
    def created_unix_nanos(self) -> int:
        return int(self._scan.created_unix_nanos)

    @property
    def first_transaction_offset(self) -> int:
        return int(self._scan.first_transaction_offset)

    @property
    def session_id(self) -> str:
        return str(self._scan.session_id)

    @property
    def plan_document(self) -> bytes:
        """The plan document, read from the range the superblock declares.

        Read on demand rather than held: it is the one unbounded field, and the
        scan has already checked these bytes against the stored CRC and the
        stored fingerprint.
        """
        return bytes(self._scan.plan_document)


class SpoolScan:
    """Everything a read of one spool establishes.

    ``finalizable`` is the question the finalizer asks, and it is deliberately not the
    same as "the tail is clean": a torn tail is the ordinary crash path and
    recovery starts from the last valid committed transaction, while a committed
    prefix a reader cannot account for blocks finalization outright.
    """

    __slots__ = ("_scan",)

    def __init__(self, scan: Any) -> None:
        self._scan = scan

    @property
    def superblock(self) -> SpoolSuperblock | None:
        """The superblock, or ``None`` when it did not validate.

        ``None`` means nothing in the spool is readable: every offset a reader
        would use comes from here.
        """
        return SpoolSuperblock(self._scan) if self._scan.superblock_valid else None

    @property
    def status(self) -> str:
        return str(self._scan.status)

    @property
    def codes(self) -> tuple[str, ...]:
        """Findings in the order they were made, not deduplicated.

        Two accounting identities that both fail are two findings; collapsing
        them would report a spool with one broken identity and a spool with four
        as the same thing.
        """
        return tuple(self._scan.codes)

    @property
    def finalizable(self) -> bool:
        return bool(self._scan.finalizable)

    @property
    def committed_prefix_end(self) -> int:
        return int(self._scan.committed_prefix_end)

    @property
    def committed_transactions(self) -> int:
        return int(self._scan.committed_transactions)

    @property
    def last_transaction_id(self) -> int:
        return int(self._scan.last_transaction_id)

    @property
    def data_items(self) -> int:
        return int(self._scan.data_items)

    @property
    def control_items(self) -> int:
        return int(self._scan.control_items)

    @property
    def durable_extent_bytes(self) -> int:
        return int(self._scan.durable_extent_bytes)

    @property
    def session_end_present(self) -> bool:
        return bool(self._scan.session_end_present)

    @property
    def session_end(self) -> SpoolSessionEnd | None:
        """The decoded session-end payload, or ``None`` when there is none.

        Separate from :attr:`session_end_present`: a record whose payload is not
        its fixed size still *exists*, and reporting the session as having no
        end would turn a malformed record into a missing one -- which is what an
        unrecoverable crash means and is not what happened.
        """
        return SpoolSessionEnd(self._scan) if self._scan.session_end_decoded else None

    @property
    def accounting(self) -> Any | None:
        """The capture-side accounting snapshot, or ``None`` when none was
        committed. Its fields are the container's; naming a first-rejection
        identity is :func:`producer_identity_name`."""
        return self._scan.accounting if self._scan.has_accounting else None

    @property
    def rejected(self) -> bool:
        return self.status == "rejected"

    @property
    def empty(self) -> bool:
        """Whether the committed prefix holds no record at all.

        The one spool contract section 7 allows to be discarded, because
        discarding it deletes no reconstruction input.
        """
        return self.committed_transactions == 0

    def records(self) -> Iterator[Any]:
        """Walk the committed records, in the order they were committed.

        Each walk is independent of every other: a finalization makes one to
        count and another to write, and neither disturbs the other's position.
        Nothing is buffered between records, so the cost of a walk does not grow
        with the spool.

        A record's ``payload`` is ``None`` for a checkpoint, an accounting
        snapshot, and a session-end record -- the scan already read all three --
        and for any record whose payload is shorter than its kind fixes, which
        the scan has already reported as a finding.
        """
        return iter(self._scan.records())

    def close(self) -> bool:
        """Release a path-backed scan's native file handle.

        The bounded scan report remains readable, but the plan document and
        record cursor require the file and must be consumed before this call.
        """
        return bool(self._scan.close())


def scan_spool(data: bytes) -> SpoolScan:
    """Scan a spool held in memory."""
    return SpoolScan(_native().scan_spool_bytes(data))


def scan_spool_path(path: str) -> SpoolScan:
    """Scan a spool on disk, without reading it into memory first.

    Opened read-only, so the mutating and durability operations are not merely
    unused but unavailable.
    """
    return SpoolScan(_native().scan_spool_path(str(path)))
