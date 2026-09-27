#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The five native replay ledgers, as NRF v1.1 record schemas.

Exact frame replay needs the original frame and block topology, and a session's
completeness can only be checked from the artifact when the artifact holds one
row per accepted data message. Neither is recoverable from the standard
per-stream records, so a native session declares five additional record sets.
What they mean is fixed by ``docs/development/native_recording_replay.md``; how
they are spelled on disk is fixed by
``specifications/nrf/v1/extensions/native-replay-v1/``.

This module builds descriptors. It writes nothing: the native writer, the
finalizer, and the replay source are what these schemas are implemented
against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ._fields import record_field
from ._paths import record_set_path

#: Extension namespace key, in the manifest and in journal record extensions.
NATIVE_REPLAY_NAMESPACE: Final = "neurale.native_replay"

#: Extension namespace key for recovery provenance on a journal termination
#: record. The contract (native_recording_replay.md section 4.5) requires a
#: termination the finalizer writes for a spool with no session-end record to
#: declare its own origin in the record's ``extensions`` under this key, and
#: ``termination_origin`` to be present on a recovery termination in the record
#: itself. This constant only names the key the contract fixes, so the
#: finalizer and the recovery task spell one namespace, not two.
RECOVERY_NAMESPACE: Final = "neurale.recovery"

#: Version of the extension this module implements.
NATIVE_REPLAY_EXTENSION_VERSION: Final = 1

#: NRF minor version a session must declare once it carries the ledgers. A
#: session without them stays at minor 0 and is readable by a v1.0 reader.
NATIVE_REPLAY_MINOR_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class LedgerField:
    """One ledger column: the part of a record field the extension fixes.

    Endianness, codecs, and array paths are not here because NRF v1 derives them
    from the dtype and the record-set path. Fixing them twice would create a
    second place for them to disagree.
    """

    name: str
    dtype: str
    nullable: bool = False
    reference: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerDefinition:
    """One ledger: its record kind, identity, primary key, and columns."""

    kind: str
    schema_id: str
    primary_key: str
    default_chunk_length: int
    fields: tuple[LedgerField, ...]

    @property
    def path(self) -> str:
        """Frozen record-set path this ledger's columns live under."""
        return record_set_path(self.kind, self.schema_id)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)


def _fields(*rows: tuple[str, str, bool] | tuple[str, str, bool, str]) -> tuple[LedgerField, ...]:
    return tuple(LedgerField(*row) for row in rows)


NATIVE_FRAMES: Final = LedgerDefinition(
    kind="native_frames",
    schema_id="native-frames-v1",
    primary_key="data_message_ordinal",
    default_chunk_length=1024,
    fields=_fields(
        ("data_message_ordinal", "uint64", False),
        ("native_session_id", "uint64", False),
        ("frame_sequence", "uint64", False),
        # Contiguous frame-row identity, distinct from data_message_ordinal:
        # that ordinal counts every data item (frames and discontinuities),
        # this counts only frame rows, zero-based and gap-free. Both are needed
        # and they are not interchangeable (section 1.1).
        ("frame_ordinal", "uint64", False),
        ("host_received_ns", "uint64", False),
        # Nullable because the native header carries them only when its flags
        # say so. A null is "the frame had none", never a zero standing in.
        ("source_tick", "uint64", True),
        ("valid_until_ns", "uint64", True),
        ("native_schema_id", "uint32", False),
        ("source_clock_domain", "uint32", False),
        ("frame_flags", "uint32", False),
        ("signal_block_count", "uint32", False),
        # Lower than signal_block_count under a partial plan. This is what makes
        # a projection visible per frame instead of only per session.
        ("recorded_signal_block_count", "uint32", False),
        # Total bytes of the original frame payload, so a reader can verify the
        # recorded blocks cover the whole frame with no trailing or overlapping
        # bytes -- the check is against the original, not a reconstruction.
        ("total_payload_byte_count", "uint64", False),
        # Nullable under the zero-child anchor rule: a partial-plan frame that
        # records no block has no first child. Null means "none"; a count > 0
        # requires a non-null ordinal and the range [first, first + count).
        ("first_signal_block_ordinal", "uint64", True),
    ),
)

NATIVE_SIGNAL_BLOCKS: Final = LedgerDefinition(
    kind="native_signal_blocks",
    schema_id="native-signal-blocks-v1",
    primary_key="signal_block_ordinal",
    default_chunk_length=1024,
    fields=_fields(
        ("signal_block_ordinal", "uint64", False),
        ("data_message_ordinal", "uint64", False),
        # The owning frame's frame_ordinal (in native_frames), so a reader
        # groups blocks by their original frame without re-deriving it from the
        # message ordinal.
        ("frame_ordinal", "uint64", False),
        ("block_index_in_frame", "uint32", False),
        ("native_signal_id", "uint32", False),
        ("stream_id", "utf8", False, "stream"),
        ("sample_idx_start", "uint64", False),
        ("last_sample_idx", "uint64", False),
        ("n_samples", "uint32", False),
        ("device_tick_start", "uint64", False),
        ("observation_time_start_ns", "uint64", False),
        ("row_offset", "uint64", False),
        ("payload_byte_count", "uint64", False),
        # The block's byte offset in the ORIGINAL frame payload, taken from the
        # native SignalBlockHeader. This is not row_offset (where the block's
        # samples land in the per-stream NRF array): the two are different
        # spaces, and the original offset is what verifies overlapping payload
        # ranges against the recording rather than against a reconstruction.
        ("payload_offset", "uint64", False),
        # The complete clock-sync snapshot the block carried, per block. A
        # reader never has to fall back to the session clock registration.
        ("clock_sync_device_tick_reference", "uint64", False),
        ("clock_sync_host_time_reference_ns", "uint64", False),
        ("clock_sync_rate_numerator", "uint64", False),
        ("clock_sync_rate_denominator", "uint64", False),
        ("clock_sync_uncertainty_ns", "uint64", False),
        ("clock_sync_clock_domain", "uint32", False),
        ("clock_sync_generation", "uint32", False),
        ("clock_sync_flags", "uint32", False),
    ),
)

NATIVE_DISCONTINUITIES: Final = LedgerDefinition(
    kind="native_discontinuities",
    schema_id="native-discontinuities-v1",
    primary_key="data_message_ordinal",
    default_chunk_length=64,
    fields=_fields(
        ("data_message_ordinal", "uint64", False),
        ("native_session_id", "uint64", False),
        ("previous_frame_sequence", "uint64", False),
        ("actual_frame_sequence", "uint64", False),
        ("reason", "utf8", False),
        # Nullable under the zero-child anchor rule: a frame-level discontinuity
        # (signal_gap_count == 0) has no first gap. Null means "none"; a count
        # > 0 requires a non-null ordinal and the range [first, first + count).
        ("first_signal_gap_ordinal", "uint64", True),
        # Zero means frame-level, which is information rather than its absence.
        ("signal_gap_count", "uint32", False),
        ("runtime_accepted_host_time_ns", "uint64", False),
    ),
)

NATIVE_SIGNAL_GAPS: Final = LedgerDefinition(
    kind="native_signal_gaps",
    schema_id="native-signal-gaps-v1",
    primary_key="signal_gap_ordinal",
    default_chunk_length=64,
    fields=_fields(
        ("signal_gap_ordinal", "uint64", False),
        ("data_message_ordinal", "uint64", False),
        ("gap_index_in_message", "uint32", False),
        ("native_signal_id", "uint32", False),
        ("expected_sample_index", "uint64", False),
        ("actual_sample_index", "uint64", False),
        ("missing_samples", "uint64", True),
        ("expected_device_tick", "uint64", False),
        ("actual_device_tick", "uint64", False),
        ("reason", "utf8", False),
        ("gap_flags", "uint32", False),
    ),
)

SESSION_ACCOUNTING: Final = LedgerDefinition(
    kind="session_accounting",
    schema_id="session-accounting-v1",
    primary_key="accounting_id",
    default_chunk_length=1,
    fields=_fields(
        ("accounting_id", "utf8", False),
        ("accounting_origin", "utf8", False),
        ("termination_origin", "utf8", False),
        ("producer_acceptance_known", "bool", False),
        ("runtime_accepted", "uint64", False),
        ("recorder_accepted", "uint64", False),
        # The four spool-adjacent counters are nullable: a path with no spool
        # has no stage-3 numbers and must write null rather than invent one.
        ("spool_committed", "uint64", True),
        ("nrf_committed", "uint64", False),
        ("rejected_before_runtime_acceptance", "uint64", False),
        ("failed_between_runtime_and_recorder", "uint64", False),
        ("lost_between_recorder_and_spool", "uint64", True),
        ("lost_during_finalization", "uint64", False),
        ("control_offered", "uint64", True),
        ("control_accepted", "uint64", False),
        ("control_spool_committed", "uint64", True),
        ("control_nrf_committed", "uint64", False),
        ("control_rejected", "uint64", False),
        ("lost_between_control_acceptance_and_spool", "uint64", True),
        ("control_lost_during_finalization", "uint64", False),
        # First-failed position is a tagged pair per plane. An item refused
        # before acceptance never received an ordinal, so the two forms are
        # separate fields and are never compared with each other.
        ("data_first_lost_ordinal", "uint64", True),
        ("data_first_rejected_message_kind", "utf8", True),
        ("data_first_rejected_frame_sequence", "uint64", True),
        ("control_first_lost_ordinal", "uint64", True),
        ("control_first_rejected_kind", "utf8", True),
        ("control_first_rejected_identity", "utf8", True),
        ("data_rejected_after_close", "uint64", False),
        ("control_rejected_after_close", "uint64", False),
    ),
)

#: Every ledger, in declaration order. A session declares all five or none.
LEDGERS: Final[tuple[LedgerDefinition, ...]] = (
    NATIVE_FRAMES,
    NATIVE_SIGNAL_BLOCKS,
    NATIVE_DISCONTINUITIES,
    NATIVE_SIGNAL_GAPS,
    SESSION_ACCOUNTING,
)

LEDGERS_BY_KIND: Final[Mapping[str, LedgerDefinition]] = {ledger.kind: ledger for ledger in LEDGERS}

#: Record kinds this extension adds to NRF v1.
LEDGER_KINDS: Final[tuple[str, ...]] = tuple(ledger.kind for ledger in LEDGERS)

#: The field the accounting summary deliberately does not have. Whether the
#: accounting checks out is derived by the reader answering the call; a stored
#: copy would let an artifact assert a verdict its reader disagrees with.
FORBIDDEN_ACCOUNTING_FIELD: Final = "accounting_verified"


def ledger_record_schema(
    ledger: LedgerDefinition,
    *,
    clock_id: str,
    chunk_length: int | None = None,
) -> dict[str, Any]:
    """Return the manifest ``record_schemas`` entry for one ledger."""
    path = ledger.path
    return {
        "id": ledger.schema_id,
        "kind": ledger.kind,
        "path": path,
        "primary_key": ledger.primary_key,
        "clock_id": clock_id,
        "chunk_length": ledger.default_chunk_length if chunk_length is None else chunk_length,
        "committed_extent": 0,
        "fields": [
            record_field(
                path,
                field.name,
                field.dtype,
                nullable=field.nullable,
                reference=field.reference,
            )
            for field in ledger.fields
        ],
    }


def ledger_record_schemas(
    *,
    clock_id: str,
    chunk_lengths: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return all five ledger descriptors, in declaration order."""
    lengths = dict(chunk_lengths or {})
    return tuple(
        ledger_record_schema(ledger, clock_id=clock_id, chunk_length=lengths.get(ledger.kind))
        for ledger in LEDGERS
    )


def ledger_schema_ids() -> dict[str, str]:
    """Return the ``kind -> schema id`` map the manifest extension carries."""
    return {ledger.kind: ledger.schema_id for ledger in LEDGERS}


def ledger_target_paths() -> tuple[str, ...]:
    """Return every ledger record-set path, in declaration order."""
    return tuple(ledger.path for ledger in LEDGERS)


def declares_native_replay(record_schemas: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether *record_schemas* declares the native replay ledgers.

    True only when all five are present. Four ledgers is not a reduced
    capability -- it is a session whose replay capability cannot be evaluated --
    so a caller gets one answer rather than a per-ledger guess.
    """
    kinds = {schema.get("kind") for schema in record_schemas}
    return set(LEDGER_KINDS) <= kinds
