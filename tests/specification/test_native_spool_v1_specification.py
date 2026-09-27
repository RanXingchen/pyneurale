#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The private native spool v1 specification, checked against itself.

Nothing here imports ``neurale``. What this suite establishes is exactly this:
the format is completely specified, its reference implementation agrees with
the document, and its binary vectors are deterministic and say what they
claim. Package conformance -- whether the installed reader agrees with these
vectors -- is a separate question, answered in
``test_native_spool_v1_package_conformance.py``.

The three questions each group answers:

- do the bytes reproduce, on any machine, from constants alone;
- does a corrupted, torn, duplicated, or unknown-version image produce the
  verdict the specification names, rather than an accident;
- does the prose agree with the layout, for the rules that live in prose.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _native_spool_v1_reference import (
    SPOOL_README,
    SPOOL_SPEC_DIR,
    SPOOL_TOOLS_DIR,
    SPOOL_VECTOR_DIR,
    VECTOR_BYTES,
    VECTOR_EXPECTATIONS,
    VECTOR_IDX,
    generate_vectors,
    specification_text,
    spool_format,
)

fmt = spool_format

VALID_VECTORS = tuple(name for name in VECTOR_BYTES if name.startswith("valid-"))


def _flowed(text: str) -> str:
    """Collapse the document's line wrapping so a sentence can be searched for.

    The specification is hard-wrapped; a required statement is a statement, not
    a particular place for the line to break.
    """
    return " ".join(text.split())


# --- the checksum ---------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", 0x00000000),
        (b"123456789", 0xE3069283),
        (b"\x00" * 32, 0x8A9136AA),
        (b"\xff" * 32, 0x62A8AB43),
    ],
)
def test_crc32c_matches_published_known_answers(data: bytes, expected: int) -> None:
    """The Castagnoli parameterization, not some other CRC-32.

    These are the standard vectors an implementation checks itself against
    before it checks a spool; a reflected/unreflected or wrong-polynomial
    implementation fails the second one immediately.
    """
    assert fmt.crc32c(data) == expected


def test_specification_publishes_known_answers() -> None:
    text = specification_text()
    for expected in ("0xE3069283", "0x8A9136AA", "0x62A8AB43", "0x82F63B78", "0x1EDC6F41"):
        assert expected in text


# --- deterministic vectors ------------------------------------------------


def test_vectors_regenerate_byte_for_byte() -> None:
    """The gate that makes every other vector assertion mean something.

    The generator takes no clock, no random source, and no filesystem state,
    so a difference here is a change in the format or the reference -- never
    an environment.
    """
    completed = subprocess.run(
        [sys.executable, str(SPOOL_TOOLS_DIR / "generate_vectors.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_repeated_vector_build_is_byte_identical() -> None:
    first = generate_vectors.build_all()["files"]
    second = generate_vectors.build_all()["files"]

    assert first == second


@pytest.mark.parametrize("name", sorted(VECTOR_BYTES))
def test_vector_matches_recorded_digest_and_length(name: str) -> None:
    entry = next(item for item in VECTOR_IDX["vectors"] if item["name"] == name)
    image = VECTOR_BYTES[name]

    assert len(image) == entry["bytes"]
    assert hashlib.sha256(image).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("name", sorted(VECTOR_BYTES))
def test_vector_scans_to_indexed_verdict(name: str) -> None:
    """The index is the language-neutral contract; the reference must meet it."""
    assert fmt.scan_spool(VECTOR_BYTES[name]).document() == VECTOR_EXPECTATIONS[name]


def test_index_covers_exactly_committed_vectors() -> None:
    committed = {path.name for path in SPOOL_VECTOR_DIR.glob("*.spool")}
    described = {entry["file"] for entry in VECTOR_IDX["vectors"]}

    assert committed == described


def test_reading_spool_never_modifies_it() -> None:
    """Ordinary diagnosis is read-only (contract section 4.5)."""
    for name, image in VECTOR_BYTES.items():
        buffer = bytearray(image)
        fmt.scan_spool(bytes(buffer))
        assert bytes(buffer) == image, f"{name} was modified by a read"


# --- what a valid spool establishes ---------------------------------------


@pytest.mark.parametrize("name", sorted(VALID_VECTORS))
def test_valid_vector_is_readable_and_finalizable(name: str) -> None:
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.status == fmt.STATUS_OK
    assert result.readable
    assert result.finalizable
    assert result.codes == []


def test_complete_spool_reports_session_end_and_outcome() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["valid-complete"])

    assert result.session_end_present
    assert result.capture_outcome == "normal"
    assert result.session_id == "spool-vector-session"


def test_empty_spool_holds_no_committed_record() -> None:
    """Section 7's one discardable case has to be recognizable, not inferred."""
    result = fmt.scan_spool(VECTOR_BYTES["valid-empty"])

    assert result.status == fmt.STATUS_OK
    assert result.committed_transactions == 0
    assert (result.data_items, result.control_items) == (0, 0)
    assert result.committed_prefix_end == len(VECTOR_BYTES["valid-empty"])


def test_blocks_and_gaps_are_not_counted_as_items() -> None:
    """One frame with two blocks, one discontinuity with one gap, one control.

    Six committed records, three items: counting a block again would
    double-count the frame that owns it, and every accounting identity in the
    contract is stated over items.
    """
    result = fmt.scan_spool(VECTOR_BYTES["valid-minimal"])

    assert (result.data_items, result.control_items) == (2, 1)


def test_capture_outcome_has_no_unknown_value() -> None:
    """``unknown`` is what a missing session-end record means (section 3.2).

    A representable ``unknown`` would let a writer state the one thing only an
    absence can say, so the enum stops at three values and a spool with no
    session end reports no outcome at all.
    """
    assert set(fmt.CAPTURE_OUTCOME_NAMES.values()) == {"normal", "aborted", "faulted"}
    open_spool = fmt.scan_spool(VECTOR_BYTES["valid-minimal"])
    assert not open_spool.session_end_present
    assert open_spool.capture_outcome is None


# --- checksum coverage ----------------------------------------------------


@pytest.mark.parametrize("name", ["valid-minimal", "valid-complete", "valid-aborted-with-losses"])
def test_single_byte_flip_is_detected(name: str) -> None:
    """Exhaustive over whole spools, not a sample of them.

    A checksum layout is only as good as its coverage: a byte nobody sums is a
    byte that can change without anybody noticing. Every offset is flipped in
    turn, and the result must differ from the clean verdict -- either the spool
    is rejected, or the committed prefix ends earlier, or a rule fires. This is
    how the unchecksummed gap behind the plan document was found; without the
    zero rule that now covers it, two bytes in every spool were free.
    """
    image = VECTOR_BYTES[name]
    clean = fmt.scan_spool(image).document()

    undetected = []
    for offset in range(len(image)):
        mutated = bytearray(image)
        mutated[offset] ^= 0xFF
        if fmt.scan_spool(bytes(mutated)).document() == clean:
            undetected.append(offset)

    assert undetected == []


def test_plan_document_bit_flip_breaks_fingerprint() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["corrupt-plan-document"])

    assert result.status == fmt.STATUS_REJECTED
    assert result.codes == [fmt.CODE_PLAN_MISMATCH]


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("corrupt-record-payload", fmt.CODE_RECORD_PAYLOAD_CHECKSUM),
        ("corrupt-record-header", fmt.CODE_RECORD_HEADER_CHECKSUM),
        ("corrupt-body-checksum", fmt.CODE_TRANSACTION_BODY_CHECKSUM),
        ("corrupt-trailer-checksum", fmt.CODE_TRANSACTION_TRAILER_CHECKSUM),
        ("corrupt-transaction-header-checksum", fmt.CODE_TRANSACTION_HEADER_CHECKSUM),
        ("corrupt-superblock-checksum", fmt.CODE_SUPERBLOCK_CHECKSUM),
        ("nonzero-padding", fmt.CODE_NONZERO_PADDING),
    ],
)
def test_corruption_site_has_own_diagnostic(name: str, code: str) -> None:
    """Which structure failed, not merely that something did."""
    assert fmt.scan_spool(VECTOR_BYTES[name]).codes == [code]


# --- tails ----------------------------------------------------------------


def test_truncation_never_promotes_partial_transaction() -> None:
    """The property, over every truncation point, not three chosen ones.

    A reader may end its committed prefix earlier than the intact file's, and
    never later, and never anywhere except a transaction boundary. This is the
    rule the whole crash path rests on.
    """
    image = VECTOR_BYTES["valid-complete"]
    intact = fmt.scan_spool(image)
    boundaries = {intact.committed_prefix_end}
    builder_offsets = fmt.scan_spool(image)
    assert builder_offsets.committed_transactions == 3

    legal_ends = set()
    for length in range(len(image) + 1):
        result = fmt.scan_spool(image[:length])
        if result.status == fmt.STATUS_REJECTED:
            continue
        assert result.committed_prefix_end <= length
        assert result.committed_prefix_end <= intact.committed_prefix_end
        assert result.committed_transactions <= intact.committed_transactions
        legal_ends.add(result.committed_prefix_end)

    # Exactly four resting points: the empty prefix and the three transactions.
    assert len(legal_ends) == 4
    assert boundaries <= legal_ends


@pytest.mark.parametrize(
    "name", ["torn-inside-body", "torn-missing-trailer", "torn-inside-record-header"]
)
def test_torn_tail_keeps_prefix_and_stays_finalizable(name: str) -> None:
    """The ordinary crash: recovery starts from the last valid transaction."""
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.status == fmt.STATUS_TORN_TAIL
    assert result.codes == [fmt.CODE_TORN_TAIL]
    assert result.committed_transactions == 2
    assert result.finalizable
    assert not result.session_end_present


def test_torn_and_corrupt_tails_are_reported_differently() -> None:
    """Same promotion rule, different human response.

    A file that ends mid-write is what a crash looks like; bytes that are
    present and wrong are evidence, and section 8 sends those to quarantine.
    """
    torn = fmt.scan_spool(VECTOR_BYTES["torn-missing-trailer"])
    corrupt = fmt.scan_spool(VECTOR_BYTES["corrupt-record-payload"])

    assert torn.status == fmt.STATUS_TORN_TAIL
    assert corrupt.status == fmt.STATUS_CORRUPT_TAIL


def test_truncated_superblock_is_rejected() -> None:
    for name in ("torn-superblock", "torn-plan-document"):
        result = fmt.scan_spool(VECTOR_BYTES[name])
        assert result.status == fmt.STATUS_REJECTED
        assert not result.readable
        assert not result.finalizable


# --- duplicates and sequence ----------------------------------------------


def test_duplicate_transaction_id_ends_prefix() -> None:
    """An append that ran twice does not silently double the recording."""
    result = fmt.scan_spool(VECTOR_BYTES["duplicate-transaction-id"])

    assert result.codes == [fmt.CODE_TRANSACTION_ID_SEQUENCE]
    assert result.committed_transactions == 1
    assert (result.data_items, result.control_items) == (2, 1)


def test_skipped_transaction_id_is_rejected() -> None:
    """Both are the same failure: the successor rule did not hold."""
    result = fmt.scan_spool(VECTOR_BYTES["transaction-id-gap"])

    assert result.codes == [fmt.CODE_TRANSACTION_ID_SEQUENCE]
    assert result.committed_transactions == 1


def test_broken_back_link_ends_prefix() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["bad-back-link"])

    assert result.codes == [fmt.CODE_BACK_LINK]
    assert result.committed_transactions == 1


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("record-count-disagreement", fmt.CODE_RECORD_COUNT),
        ("empty-transaction", fmt.CODE_EMPTY_TRANSACTION),
        ("trailer-item-counts", fmt.CODE_TRAILER_ITEM_COUNTS),
    ],
)
def test_self_inconsistent_transaction_is_not_committed(name: str, code: str) -> None:
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.codes == [code]
    assert result.committed_transactions == 0


# --- versioning -----------------------------------------------------------


def test_unknown_major_version_is_rejected_before_reading() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["unknown-major-version"])

    assert result.status == fmt.STATUS_REJECTED
    assert result.codes == [fmt.CODE_UNSUPPORTED_MAJOR]
    assert result.committed_transactions == 0


def test_unknown_minor_version_is_read_normally() -> None:
    """A minor may only add kinds, so a minor alone never makes a spool unreadable."""
    known = fmt.scan_spool(VECTOR_BYTES["valid-complete"])
    future = fmt.scan_spool(VECTOR_BYTES["valid-unknown-minor"])

    assert future.status == fmt.STATUS_OK
    assert future.finalizable
    assert future.committed_transactions == known.committed_transactions
    assert (future.data_items, future.control_items) == (known.data_items, known.control_items)


def test_undefined_superblock_field_is_rejected() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["undefined-superblock-field"])

    assert result.status == fmt.STATUS_REJECTED
    assert result.codes == [fmt.CODE_SUPERBLOCK_FIELD]


def test_unknown_record_kind_blocks_finalization() -> None:
    """Skipping is safe for a cache. The spool is the only copy.

    The record was committed, so a promoted prefix that stepped over it would
    be a session missing a record nobody counted -- the silent loss the
    lifecycle contract refuses everywhere.
    """
    result = fmt.scan_spool(VECTOR_BYTES["unknown-record-kind"])

    assert result.status == fmt.STATUS_OK
    assert result.readable
    assert not result.finalizable
    assert result.codes == [fmt.CODE_UNKNOWN_RECORD_KIND]


# --- durability -----------------------------------------------------------


def test_durability_policies_report_distinct_extents() -> None:
    """Durability is an attribute of an extent, and the policy is what sets it."""
    transaction = fmt.scan_spool(VECTOR_BYTES["valid-complete"])
    checkpoint = fmt.scan_spool(VECTOR_BYTES["valid-checkpoint-sync"])
    buffered = fmt.scan_spool(VECTOR_BYTES["valid-buffered"])

    assert transaction.committed_prefix_end == checkpoint.committed_prefix_end
    assert checkpoint.committed_prefix_end == buffered.committed_prefix_end

    assert transaction.durable_extent_bytes == transaction.committed_prefix_end
    assert buffered.durable_extent_bytes < checkpoint.durable_extent_bytes
    assert checkpoint.durable_extent_bytes < transaction.durable_extent_bytes


def test_buffered_writes_claim_no_durable_record() -> None:
    """ "Guarantees nothing against power loss", expressed as an extent.

    The superblock region is synced under every policy -- a spool whose own
    identity did not survive could not be matched to a finalization target --
    so the floor is that region and not zero, and no record is covered.
    """
    buffered = fmt.scan_spool(VECTOR_BYTES["valid-buffered"])
    empty = fmt.scan_spool(VECTOR_BYTES["valid-empty"])

    assert buffered.durability_policy == "buffered"
    assert buffered.durable_extent_bytes == empty.committed_prefix_end


@pytest.mark.parametrize("name", sorted(VECTOR_BYTES))
def test_durable_extent_never_exceeds_committed_prefix(name: str) -> None:
    """The one durability claim that would be a lie in every case."""
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.durable_extent_bytes <= result.committed_prefix_end


def test_checkpoint_cannot_claim_future_bytes() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["checkpoint-extent-beyond-transaction"])

    assert result.codes == [fmt.CODE_CHECKPOINT_EXTENT]
    assert not result.finalizable


def test_policy_names_match_lifecycle_contract() -> None:
    contract = (
        Path(__file__).resolve().parents[2] / "docs" / "development" / "native_recording_replay.md"
    )
    if not contract.exists():  # pragma: no cover - docs are not always checked out
        pytest.skip("the lifecycle contract is not present in this tree")
    text = contract.read_text(encoding="utf-8")

    assert set(fmt.DURABILITY_NAMES.values()) == {"buffered", "checkpoint_sync", "transaction_sync"}
    for phrase in ("**buffered**", "**checkpoint sync**", "**transaction sync**"):
        assert phrase in text


# --- accounting -----------------------------------------------------------


def test_accounting_layout_has_no_verified_field() -> None:
    """Section 5.1 forbids storing it: it is a judgement, not a fact.

    Two values that can disagree, with no rule saying which one a caller sees,
    is precisely what the contract removed. The container therefore has no
    place to put one.
    """
    fields = fmt.DATA_COUNTERS + fmt.CONTROL_COUNTERS + fmt.CLOSE_COUNTERS + fmt.POSITION_FIELDS

    assert "accounting_verified" not in fields
    assert "accounting_verified" not in specification_text().replace(
        "no `accounting_verified` field", ""
    ).replace("`accounting_verified` is", "")


def test_spool_carries_no_finalization_counter() -> None:
    """The spool is written before finalization runs, so it cannot know them."""
    fields = fmt.DATA_COUNTERS + fmt.CONTROL_COUNTERS

    assert "nrf_committed" not in fields
    assert "lost_during_finalization" not in fields
    assert "control_nrf_committed" not in fields


def test_accounting_identity_violation_is_rejected() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["accounting-identity-violation"])

    assert fmt.CODE_ACCOUNTING_IDENTITY in result.codes
    assert not result.finalizable


def test_unsigned_counter_wraparound_cannot_satisfy_accounting() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["accounting-uint64-wraparound"])

    assert fmt.CODE_ACCOUNTING_IDENTITY in result.codes
    assert fmt.CODE_POSITION_CONSISTENCY in result.codes
    assert not result.finalizable


def test_counters_beyond_committed_prefix_are_rejected() -> None:
    """The failure layer 1 cannot see, and the reason layer 2 exists.

    Every identity holds; the summary claims ninety-nine committed data items
    over a prefix that holds two. A counter that exceeds the artifact is a
    failed verification, never reconciled by trusting the number.
    """
    result = fmt.scan_spool(VECTOR_BYTES["accounting-exceeds-prefix"])

    assert result.codes == [fmt.CODE_ACCOUNTING_EXCEEDS_PREFIX]
    assert not result.finalizable


def test_spool_accounting_rejects_recovery_origin() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["accounting-origin-not-recorder"])

    assert result.codes == [fmt.CODE_ACCOUNTING_ORIGIN]


def test_lossy_session_is_still_legal() -> None:
    """Losses are recorded, not hidden: an incomplete session is representable."""
    result = fmt.scan_spool(VECTOR_BYTES["valid-aborted-with-losses"])

    assert result.status == fmt.STATUS_OK
    assert result.finalizable
    assert result.capture_outcome == "aborted"


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("session-end-without-accounting", fmt.CODE_SESSION_END_WITHOUT_ACCOUNTING),
        ("accounting-with-items", fmt.CODE_ACCOUNTING_TRANSACTION_NOT_QUIET),
        ("records-after-session-end", fmt.CODE_RECORDS_AFTER_SESSION_END),
        ("container-payload-size", fmt.CODE_CONTAINER_PAYLOAD_SIZE),
    ],
)
def test_sealing_rule_violation_blocks_finalization(name: str, code: str) -> None:
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert code in result.codes
    assert not result.finalizable


def test_records_after_session_end_do_not_inflate_accounting() -> None:
    """Layer 2 is checked at the prefix the snapshot seals.

    A transaction committed after the session end adds items the summary never
    claimed; checking the summary against the inflated total would fire a
    spurious exceeds-prefix over a spool whose only real fault is the late
    transaction. The sealing prefix is what the counters must match.
    """
    result = fmt.scan_spool(VECTOR_BYTES["records-after-session-end"])

    assert result.codes == [fmt.CODE_RECORDS_AFTER_SESSION_END]
    assert not result.finalizable


# --- terminal provenance -------------------------------------------------


def test_complete_spool_reports_terminal_provenance() -> None:
    """The session-end record freezes the request and the outcome separately."""
    result = fmt.scan_spool(VECTOR_BYTES["valid-complete"])

    assert result.session_end_present
    assert result.requested_terminal_intent == "normal"
    assert result.capture_outcome == "normal"
    assert result.primary_fault_committed is False


def test_aborted_session_separates_request_from_outcome() -> None:
    """``requested=aborted`` and ``capture=aborted`` travel as two fields.

    A caller may abort and a recorder fault may then escalate the capture to
    ``faulted`` while the request stays ``aborted``; the contract forbids
    recovering either from the other, so the spool must carry both.
    """
    result = fmt.scan_spool(VECTOR_BYTES["valid-aborted-with-losses"])

    assert result.requested_terminal_intent == "aborted"
    assert result.capture_outcome == "aborted"
    assert result.primary_fault_committed is False
    assert result.terminal_reason == "operator abort"


def test_faulted_session_records_primary_fault() -> None:
    result = fmt.scan_spool(VECTOR_BYTES["valid-fault-record"])

    assert result.requested_terminal_intent == "fault"
    assert result.capture_outcome == "faulted"
    assert result.primary_fault_committed is True
    assert result.terminal_reason == "recorder fault"


def test_escalated_abort_keeps_request_and_outcome_separate() -> None:
    """The reviewer's escalation case, as a normative vector.

    The caller aborted first (``requested_terminal_intent = aborted``); a
    recorder fault during the drain then made the capture ``faulted`` and
    committed the fault row. The request and the outcome are different fields
    and both survive, which is the whole reason ``session_end`` carries them
    separately.
    """
    result = fmt.scan_spool(VECTOR_BYTES["valid-aborted-then-faulted"])

    assert result.status == fmt.STATUS_OK
    assert result.finalizable
    assert result.requested_terminal_intent == "aborted"
    assert result.capture_outcome == "faulted"
    assert result.primary_fault_committed is True
    assert result.terminal_reason == "abort then fault"


def test_session_end_payload_carries_reason() -> None:
    assert fmt.SESSION_END_PAYLOAD_BYTES == 80
    assert fmt.TERMINAL_REASON_BYTES == 64
    assert fmt.REQUESTED_TERMINAL_INTENT_NAMES == {
        0: "normal",
        1: "aborted",
        2: "fault",
    }


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("ordinal-block-mismatch", fmt.CODE_ORDINAL),
        ("ordinal-gap-mismatch", fmt.CODE_ORDINAL),
        ("ordinal-session-end-nonzero", fmt.CODE_ORDINAL),
        ("record-flags-nonzero", fmt.CODE_FLAGS_NONZERO),
        ("session-end-undefined-outcome", fmt.CODE_SESSION_END_FIELD),
        ("checkpoint-policy-mismatch", fmt.CODE_CHECKPOINT_POLICY),
        ("accounting-flag-invalid", fmt.CODE_ACCOUNTING_FLAG),
        ("accounting-position-tag-invalid", fmt.CODE_POSITION_TAG),
        ("accounting-position-consistency", fmt.CODE_POSITION_CONSISTENCY),
        ("multiple-accounting", fmt.CODE_MULTIPLE_ACCOUNTING),
        ("multiple-session-end", fmt.CODE_MULTIPLE_SESSION_END),
        ("accounting-without-session-end", fmt.CODE_ACCOUNTING_WITHOUT_SESSION_END),
    ],
)
def test_container_field_violation_blocks_finalization(name: str, code: str) -> None:
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.codes == [code]
    assert not result.finalizable


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("accounting-position-loss-missing", fmt.CODE_POSITION_CONSISTENCY),
        ("accounting-position-data-loss-tag", fmt.CODE_POSITION_TAG_FORM),
        ("accounting-position-data-rejection-tag", fmt.CODE_POSITION_TAG_FORM),
        ("accounting-position-control-loss-tag", fmt.CODE_POSITION_TAG_FORM),
        ("accounting-position-control-rejection-tag", fmt.CODE_POSITION_TAG_FORM),
    ],
)
def test_position_contract_violation_blocks_finalization(name: str, code: str) -> None:
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.codes == [code]
    assert not result.finalizable


def test_runtime_to_recorder_loss_requires_first_ordinal() -> None:
    """The reviewer's case: an accepted item that failed between the runtime
    and the recorder has a data-message ordinal, so ``data_first_loss`` must
    name it even when ``lost_between_recorder_and_spool`` is zero.

    Every accounting identity holds; the only fault is an accepted-item loss
    with no first-loss position latched.
    """
    result = fmt.scan_spool(VECTOR_BYTES["accounting-position-loss-missing"])

    assert result.codes == [fmt.CODE_POSITION_CONSISTENCY]
    assert not result.finalizable


def test_first_loss_position_covers_both_handoffs() -> None:
    """``data_first_loss`` is present for a runtime-to-recorder failure too.

    ``valid-aborted-with-losses`` records ``failed_between_runtime_and_recorder
    = 1`` and ``lost_between_recorder_and_spool = 2`` with a single ordinal
    first-loss position, so the position covers both handoffs and the spool is
    clean.
    """
    result = fmt.scan_spool(VECTOR_BYTES["valid-aborted-with-losses"])

    assert result.status == fmt.STATUS_OK
    assert result.finalizable
    assert result.codes == []


def test_loss_cannot_be_recorded_as_producer_identity() -> None:
    """Each position's tag form is fixed by the contract (section 4.4)."""
    for name in (
        "accounting-position-data-loss-tag",
        "accounting-position-control-loss-tag",
    ):
        result = fmt.scan_spool(VECTOR_BYTES[name])
        assert result.codes == [fmt.CODE_POSITION_TAG_FORM]
        assert not result.finalizable


def test_rejection_cannot_be_recorded_as_ordinal() -> None:
    for name in (
        "accounting-position-data-rejection-tag",
        "accounting-position-control-rejection-tag",
    ):
        result = fmt.scan_spool(VECTOR_BYTES[name])
        assert result.codes == [fmt.CODE_POSITION_TAG_FORM]
        assert not result.finalizable


@pytest.mark.parametrize(
    "name",
    [
        "accounting-position-identity-kind-unknown",
        "accounting-position-identity-kind-wrong-plane",
    ],
)
def test_unregistered_identity_kind_blocks_finalization(name: str) -> None:
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.codes == [fmt.CODE_IDENTITY_KIND]
    assert not result.finalizable


def test_identity_kind_registry_is_partitioned_by_plane() -> None:
    """The number is the only bridge to the NRF kind string, so the registry is
    frozen: a data rejection names a data message kind, a control rejection a
    control record-set kind, and the two planes are disjoint.
    """
    assert fmt.IDENTITY_KIND_REGISTRY == {
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
    assert fmt.DATA_IDENTITY_KINDS == frozenset({1, 2})
    assert fmt.CONTROL_IDENTITY_KINDS == frozenset({3, 4, 5, 6, 7, 8, 9, 10, 11})
    assert not (fmt.DATA_IDENTITY_KINDS & fmt.CONTROL_IDENTITY_KINDS)


def test_identity_kind_registry_is_documented() -> None:
    text = specification_text()
    for name in fmt.IDENTITY_KIND_REGISTRY.values():
        assert f"`{name}`" in text
    # The NRF conversion rule for the control identity is stated, not implied.
    assert "unsigned decimal ASCII" in text
    assert "no leading zeros" in text


def test_data_rejection_cannot_name_control_kind() -> None:
    """A control record-set kind in a data position would let the finalizer
    write it into the data first-rejected-message-kind field."""
    result = fmt.scan_spool(VECTOR_BYTES["accounting-position-identity-kind-wrong-plane"])

    assert result.codes == [fmt.CODE_IDENTITY_KIND]
    assert not result.finalizable


def test_lossy_session_uses_registered_identity_kinds() -> None:
    """``valid-aborted-with-losses`` names a rejected frame (data) and a
    rejected event (control), so both producer-identity positions carry kinds
    that are registered for their plane and the spool is clean.
    """
    result = fmt.scan_spool(VECTOR_BYTES["valid-aborted-with-losses"])

    assert result.status == fmt.STATUS_OK
    assert result.finalizable
    assert result.codes == []


def test_faulted_session_without_fault_row_is_representable() -> None:
    """``primary_fault_committed = 0`` with ``capture_outcome = faulted``.

    The contract allows a session to end faulted when the fault row could not
    be persisted; the session-end record says so, so the finalizer can declare
    the row missing rather than invent one.
    """
    builder = generate_vectors.SpoolBuilder(fmt.DURABILITY_TRANSACTION_SYNC)
    builder.commit(generate_vectors._data_transaction_records())
    builder.commit(
        [
            fmt.encode_record(fmt.KIND_ACCOUNTING_SNAPSHOT, generate_vectors._accounting_payload()),
            fmt.encode_record(
                fmt.KIND_SESSION_END,
                fmt.encode_session_end(
                    fmt.CAPTURE_OUTCOME_FAULTED,
                    generate_vectors.NANOS + 400,
                    requested_terminal_intent=fmt.REQUESTED_TERMINAL_INTENT_FAULT,
                    primary_fault_committed=False,
                    terminal_reason=b"recorder fault",
                ),
            ),
        ]
    )
    result = fmt.scan_spool(builder.bytes())

    assert result.status == fmt.STATUS_OK
    assert result.finalizable
    assert result.capture_outcome == "faulted"
    assert result.requested_terminal_intent == "fault"
    assert result.primary_fault_committed is False


# --- logical-ordinal ownership ------------------------------------------


def test_blocks_and_gaps_carry_owner_ordinal() -> None:
    """A valid spool's blocks match their frame and its gaps their discontinuity.

    The frame carries ordinal 1, its two blocks 1, the discontinuity 2, its gap
    2; the container does not need to parse the opaque payloads to attribute a
    block to a frame -- the ordinals say so, and a mismatch is a violation.
    """
    result = fmt.scan_spool(VECTOR_BYTES["valid-minimal"])

    assert result.status == fmt.STATUS_OK
    assert result.finalizable
    assert result.codes == []


# --- corrupt tails -------------------------------------------------------


@pytest.mark.parametrize("name", ["corrupt-tail-wrong-magic", "corrupt-tail-bad-header-bytes"])
def test_invalid_header_tail_is_corrupt_not_torn(name: str) -> None:
    """Present and wrong is corruption; only truncation is a torn tail."""
    result = fmt.scan_spool(VECTOR_BYTES[name])

    assert result.status == fmt.STATUS_CORRUPT_TAIL
    assert result.codes == [fmt.CODE_CORRUPT_TRANSACTION_HEADER]
    assert result.finalizable


def test_torn_and_corrupt_tail_bytes_are_distinct() -> None:
    torn = fmt.scan_spool(VECTOR_BYTES["torn-missing-trailer"])
    corrupt = fmt.scan_spool(VECTOR_BYTES["corrupt-tail-wrong-magic"])

    assert torn.status == fmt.STATUS_TORN_TAIL
    assert torn.codes == [fmt.CODE_TORN_TAIL]
    assert corrupt.status == fmt.STATUS_CORRUPT_TAIL
    assert corrupt.codes == [fmt.CODE_CORRUPT_TRANSACTION_HEADER]


# --- the document -------------------------------------------------------


def test_specification_marks_spool_private_and_rebuildable() -> None:
    """A required statement, because the retention rules depend on it.

    Both halves have to be there: private (nothing outside this repository may
    depend on these bytes) and rebuildable in one direction only (discardable
    after validated finalization, never reconstructible from NRF).
    """
    text = _flowed(specification_text())

    assert "**Private.**" in text
    assert "**Rebuildable, in one direction only.**" in text
    assert "cannot be reconstructed from an NRF session" in text
    assert "the only reconstruction input there is" in text


def test_specification_forbids_constrained_features() -> None:
    text = _flowed(specification_text())

    assert "fixed-width little-endian" in text
    assert "a memory image of a C++ struct" in text
    assert "there is no floating-point field anywhere" in text.lower()
    assert "sizeof" in text


@pytest.mark.parametrize(
    "code",
    sorted(
        value
        for name, value in vars(spool_format).items()
        if name.startswith("CODE_") and isinstance(value, str)
    ),
)
def test_every_diagnostic_code_is_documented(code: str) -> None:
    assert f"`{code}`" in specification_text()


@pytest.mark.parametrize(
    "code",
    sorted(
        value
        for name, value in vars(spool_format).items()
        if name.startswith("CODE_") and isinstance(value, str)
    ),
)
def test_every_diagnostic_code_has_vector(code: str) -> None:
    """A documented code nothing produces is a rule nobody has checked."""
    produced = {code for expect in VECTOR_EXPECTATIONS.values() for code in expect["codes"]}

    assert code in produced


def test_documented_code_table_matches_reference() -> None:
    documented = set(re.findall(r"`(SPOOL-\d{3})`", specification_text()))
    implemented = {
        value
        for name, value in vars(spool_format).items()
        if name.startswith("CODE_") and isinstance(value, str)
    }

    assert documented == implemented


def test_record_kind_registry_matches_document() -> None:
    text = specification_text()
    for value, name in fmt.RECORD_KIND_NAMES.items():
        assert re.search(rf"^\| {value} \| `{name}` \|", text, flags=re.MULTILINE), name


def test_documented_structure_sizes_match_reference() -> None:
    text = specification_text()
    sizes = {
        "fixed superblock": fmt.SUPERBLOCK_BYTES,
        "transaction header": fmt.TRANSACTION_HEADER_BYTES,
        "commit trailer": fmt.TRANSACTION_TRAILER_BYTES,
    }
    assert (
        f"offset 0                        fixed superblock            {sizes['fixed superblock']} bytes"
        in text
    )
    assert f"transaction header    {sizes['transaction header']} bytes" in text
    assert f"commit trailer        {sizes['commit trailer']} bytes" in text
    assert f"`accounting_snapshot` payload ({fmt.ACCOUNTING_PAYLOAD_BYTES} bytes)" in text
    assert f"`session_end` payload ({fmt.SESSION_END_PAYLOAD_BYTES} bytes)" in text
    assert f"`checkpoint` payload ({fmt.CHECKPOINT_PAYLOAD_BYTES} bytes)" in text


def test_specification_tree_is_not_package() -> None:
    """Specification material that happens to be executable, loaded by path."""
    assert not (SPOOL_TOOLS_DIR / "__init__.py").exists()
    assert not (SPOOL_SPEC_DIR / "__init__.py").exists()


def test_reference_tools_import_nothing_from_package() -> None:
    """A reference that used the implementation would be checking it against itself."""
    for path in sorted(SPOOL_TOOLS_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "import neurale" not in source
        assert "from neurale" not in source


def test_index_states_implemented_version() -> None:
    idx: dict[str, Any] = json.loads((SPOOL_VECTOR_DIR / "index.json").read_text(encoding="utf-8"))

    assert idx["version_major"] == fmt.VERSION_MAJOR
    assert idx["version_minor"] == fmt.VERSION_MINOR
    assert idx["checksum"] == "crc32c"
    assert SPOOL_README.exists()
