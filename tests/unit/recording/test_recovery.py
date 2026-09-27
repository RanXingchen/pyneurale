#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Crash recovery, resumable finalization, and the completeness verdict.

Three kinds of fixture appear here and they are not interchangeable:

* **Real spools**, written by a real native recorder, then damaged the way a
  crash damages one -- truncated at the sealing transaction, or given a tail of
  bytes that belong to no valid transaction. A hand-built spool would test this
  file's idea of the format rather than the bytes the other engine produces.
* **A real subprocess**, killed with ``SIGKILL`` while it finalizes. Interrupting
  a finalization in-process proves the error path; killing the process proves
  the *persisted* progress document, which is the only witness a crash leaves.
* **Real finalized sessions with one record set corrupted on the way out**, for
  the accounting cases. A session claiming more than it holds is exactly what
  the finalizer's own cross-check makes impossible to produce, and layer 2 exists
  for the sessions some *other* writer produces -- so the lie has to be injected.
  Everything outside the injected column is still what the finalizer wrote, which
  keeps the fixture a valid NRF v1.1 session rather than this file's idea of one.
  The legacy cases, which need no recorder at all, live in
  ``tests/unit/io/nrf/test_completeness.py``.

What is *not* evidenced here, stated rather than implied: a process crash during
a **native recording**. The only spool store that passes the readiness gate is
the bounded in-memory one, which provides no crash durability at all, so
there is no durable spool for a killed recording process to leave behind. The
crash-shaped spools below are snapshots of a real recorder's committed prefix,
which reproduces what recovery *finds* without reproducing how it got there.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import struct
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from neurale.io.nrf import LEDGERS, NrfReader, NrfWriter
from neurale.recording import (
    ACTION_DISCARD,
    ACTION_FINALIZE,
    ACTION_QUARANTINE,
    ACTION_REPAIR,
    FinalizationError,
    SpoolSourceError,
    abandon_finalization,
    diagnose_finalization,
    diagnose_spool,
    finalize_spool,
    read_repair_report,
    repair_spool,
    resume_finalization,
)
from neurale.recording import _spool_format as spool
from neurale.recording._finalizer import (
    NATIVE_REPLAY_NAMESPACE,
    PROGRESS_FILE,
    RECOVERY_NAMESPACE,
    STAGING_SESSION,
    STAGING_SUFFIX,
)

from .test_finalizer import _control_session, _record_session

# --- crash-shaped sources -----------------------------------------------------


def _unsealed(data: bytes) -> bytes:
    """A real spool truncated at the transaction that would have sealed it.

    The shape a process crash leaves: a committed prefix, no session-end record,
    and nothing beyond the last valid committed transaction.
    """
    scan = spool.scan_spool(data)
    sealing = next(
        record.transaction_offset
        for record in scan.records()
        if record.kind == spool.RECORD_SESSION_END
    )
    return data[:sealing]


def _with_torn_tail(data: bytes, tail: bytes = b"\xde\xad\xbe\xef" * 16) -> bytes:
    """A real spool followed by bytes that belong to no valid transaction."""
    return data + tail


def _spool_file(tmp_path: Path, data: bytes, name: str = "session.spool") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- the spool: read-only diagnosis -------------------------------------------


def test_clean_spool_diagnosis_recommends_finalizing(tmp_path: Path) -> None:
    """A spool with an accountable prefix and no tail is ready to convert."""
    data = _record_session(tmp_path, 16)
    diagnosis = diagnose_spool(data)
    assert diagnosis.readable
    assert diagnosis.finalizable
    assert diagnosis.session_end_present
    assert diagnosis.capture_outcome == "normal"
    assert diagnosis.invalid_tail_bytes == 0
    assert diagnosis.recommended_action == ACTION_FINALIZE
    assert diagnosis.accounting_present
    assert diagnosis.durability_policy in {"buffered", "checkpoint_sync", "transaction_sync"}


def test_diagnosis_never_mutates_spool(tmp_path: Path) -> None:
    """Ordinary diagnosis must be read-only (contract section 4.5).

    Checked over a file rather than over bytes, because the mutation this rule
    forbids is a write: a diagnosis that "helpfully" truncated a tail would be a
    repair nobody asked for, performed on the only reconstruction input there is.
    """
    path = _spool_file(tmp_path, _with_torn_tail(_record_session(tmp_path, 8)))
    before = path.read_bytes()
    stat_before = path.stat()
    time.sleep(0.01)
    diagnose_spool(path)
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stat_before.st_mtime_ns


def test_torn_tail_is_reported_not_promoted(tmp_path: Path) -> None:
    """Bytes past the committed prefix are named, never interpreted.

    A record inside an incomplete or checksum-invalid transaction must not be
    promoted, ever, by any code path -- so the committed prefix of a torn spool
    is byte-for-byte the committed prefix of the untorn one, and the difference
    shows up only as an invalid tail.
    """
    clean = _record_session(tmp_path, 12)
    torn = _with_torn_tail(clean)
    before = diagnose_spool(clean)
    after = diagnose_spool(torn)

    assert after.committed_prefix_end == before.committed_prefix_end
    assert after.committed_transactions == before.committed_transactions
    assert after.data_items == before.data_items
    assert after.invalid_tail_bytes == len(torn) - len(clean)
    assert after.recommended_action == ACTION_REPAIR
    assert after.finalizable


def test_broken_superblock_recommends_quarantine(
    tmp_path: Path,
) -> None:
    """Nothing is interpretable, so the evidence is preserved rather than repaired."""
    data = bytearray(_record_session(tmp_path, 8))
    data[4] ^= 0xFF
    diagnosis = diagnose_spool(bytes(data))
    assert not diagnosis.readable
    assert not diagnosis.finalizable
    assert diagnosis.recommended_action == ACTION_QUARANTINE
    assert diagnosis.session_id is None


def test_empty_spool_may_be_discarded(tmp_path: Path) -> None:
    """Superblock only: discarding it deletes no reconstruction input.

    The single exception contract section 7 allows, and it is not really an
    exception -- there is nothing to reconstruct.
    """
    data = _record_session(tmp_path, 8)
    scan = spool.scan_spool(data)
    assert scan.superblock is not None
    empty = data[: scan.superblock.first_transaction_offset]
    diagnosis = diagnose_spool(empty)
    assert diagnosis.readable
    assert diagnosis.empty
    assert diagnosis.recommended_action == ACTION_DISCARD


# --- the spool: explicit, audited repair --------------------------------------


def test_dry_run_repair_writes_nothing(tmp_path: Path) -> None:
    path = _spool_file(tmp_path, _with_torn_tail(_record_session(tmp_path, 8)))
    before = path.read_bytes()
    report = repair_spool(path, dry_run=True)
    assert report.dry_run
    assert not report.repaired
    assert report.removed_bytes == 64
    assert report.quarantine_path is None
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.quarantined*")) == []


def test_repair_quarantines_original_and_reports(tmp_path: Path) -> None:
    """Truncating an invalid tail preserves the evidence and says what it did.

    Contract section 7: a spool whose tail was truncated is **quarantined, not
    deleted**, and the repair emits a report naming what was moved and why. The
    quarantined copy is the original byte for byte -- a repair that overwrote it
    would destroy the only record of what the tail contained.
    """
    original = _with_torn_tail(_record_session(tmp_path, 12))
    path = _spool_file(tmp_path, original)

    report = repair_spool(path)
    assert report.repaired
    assert report.removed_bytes == 64
    assert report.quarantine_path is not None
    assert report.quarantine_path.read_bytes() == original
    assert path.read_bytes() == original[: report.committed_prefix_end]

    assert report.report_path is not None
    document = read_repair_report(report.report_path)
    assert document["removed_bytes"] == 64
    assert document["moved"] == str(report.quarantine_path)
    assert document["diagnosis"]["recommended_action"] == ACTION_REPAIR
    assert "belonged to no valid committed transaction" in document["reason"]


def test_repaired_spool_finalizes_like_untorn_spool(
    tmp_path: Path,
) -> None:
    """Repair removes bytes; it never promotes one.

    The strongest statement available about "nothing was promoted": the session
    built from the repaired spool holds the same rows as the session built from
    the spool before it was torn.
    """
    clean = _record_session(tmp_path, 16)
    path = _spool_file(tmp_path, _with_torn_tail(clean))
    repair_spool(path)

    from_repaired = finalize_spool(path, tmp_path / "repaired.nrf")
    from_clean = finalize_spool(clean, tmp_path / "clean.nrf")
    assert from_repaired.counts == from_clean.counts
    assert from_repaired.termination_kind == from_clean.termination_kind


def test_repair_without_tail_changes_nothing(tmp_path: Path) -> None:
    path = _spool_file(tmp_path, _record_session(tmp_path, 8))
    before = path.read_bytes()
    report = repair_spool(path)
    assert not report.repaired
    assert report.removed_bytes == 0
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.quarantined*")) == []


def test_repair_rejects_unaccountable_prefix(
    tmp_path: Path,
) -> None:
    """Repair means "what is left is usable", and here it is not.

    The spool below has a torn tail *and* a committed prefix whose own accounting
    does not balance, so truncating the tail would produce a spool no
    finalization accepts. Reporting that as ``repaired`` would leave a caller
    holding something refused by the next step with the tail -- possibly the only
    clue to why -- already moved aside; contract section 7 sends this spool to
    quarantine whole.
    """
    from .test_finalizer import _patch_payload

    lying = _patch_payload(
        _record_session(tmp_path, 8),
        spool.RECORD_ACCOUNTING_SNAPSHOT,
        lambda payload: _bump(payload, 0),
    )
    path = _spool_file(tmp_path, _with_torn_tail(lying))
    before = path.read_bytes()

    diagnosis = diagnose_spool(path)
    assert diagnosis.readable
    assert not diagnosis.finalizable
    assert diagnosis.recommended_action == ACTION_QUARANTINE

    with pytest.raises(SpoolSourceError, match="quarantine the spool whole"):
        repair_spool(path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.quarantined*")) == []


def _bump(payload: bytearray, idx: int) -> bytearray:
    """Add one to the accounting counter at *index*, breaking its identity."""
    value = struct.unpack_from("<Q", payload, idx * 8)[0]
    struct.pack_into("<Q", payload, idx * 8, value + 1)
    return payload


def test_repair_rejects_unreadable_spool(tmp_path: Path) -> None:
    """There is no committed prefix to truncate to, so truncating destroys evidence."""
    data = bytearray(_record_session(tmp_path, 8))
    data[4] ^= 0xFF
    path = _spool_file(tmp_path, bytes(data))
    with pytest.raises(SpoolSourceError, match="quarantine"):
        repair_spool(path)
    assert path.read_bytes() == bytes(data)


# --- finalization: dry run and state ------------------------------------------


def test_dry_run_finalization_predicts_outcome(
    tmp_path: Path,
) -> None:
    """It proposes: no session, no staging directory, no progress document."""
    data = _record_session(tmp_path, 16)
    target = tmp_path / "session.nrf"
    report = finalize_spool(data, target, dry_run=True)

    assert report.dry_run
    assert report.termination_kind == "normal"
    assert report.counts.frames == 16
    assert report.counts.data_items == 16
    assert not target.exists()
    assert not target.with_name(target.name + STAGING_SUFFIX).exists()

    # ... and the real attempt then produces exactly what was predicted.
    real = finalize_spool(data, target)
    assert real.counts.frames == report.counts.frames
    assert real.counts.data_items == report.counts.data_items
    assert real.termination_kind == report.termination_kind


def test_dry_run_predicts_recovery_origin(
    tmp_path: Path,
) -> None:
    report = finalize_spool(
        _unsealed(_record_session(tmp_path, 16)), tmp_path / "s.nrf", dry_run=True
    )
    assert report.dry_run
    assert report.termination_origin == "recovery"
    assert report.accounting_origin == "recovery_rebuilt"
    assert report.source_session_end_present is False


def test_untouched_target_reports_no_attempt(
    tmp_path: Path,
) -> None:
    state = diagnose_finalization(tmp_path / "session.nrf")
    assert state.status == "not_started"
    assert not state.published
    assert not state.resumable
    assert state.progress is None
    assert state.completeness_verdict is None
    assert state.complete is None


def test_published_target_reports_verdict(tmp_path: Path) -> None:
    data = _record_session(tmp_path, 16)
    finalize_spool(data, tmp_path / "session.nrf")
    state = diagnose_finalization(tmp_path / "session.nrf")
    assert state.status == "published"
    assert state.published
    assert not state.resumable
    assert state.completeness_verdict == "verified_complete"
    assert state.complete is True
    assert state.accounting_verified


# --- interruption at each phase, and resume -----------------------------------


class _NoPublish:
    """Refuse publication only, for both Windows rename and POSIX link."""

    def __init__(self, target: Path) -> None:
        self._target = str(target)

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def rename(self, source: Any, destination: Any) -> None:
        if str(destination) == self._target:
            raise OSError("interrupted")
        os.rename(source, destination)

    def link(self, source: Any, destination: Any) -> None:
        if str(destination) == self._target:
            raise OSError("interrupted")
        os.link(source, destination)


def _interrupt_at(monkeypatch: Any, phase: str, target: Path | None = None) -> None:
    """Make the phase named *phase* fail the way an interruption does."""
    from neurale.recording import _finalizer

    if phase == "publishing":
        assert target is not None
        monkeypatch.setattr(_finalizer, "os", _NoPublish(target))
        return

    def interrupted(*arguments: Any, **keywords: Any) -> Any:
        raise FinalizationError("interrupted", category="writer")

    monkeypatch.setattr(
        _finalizer, "_convert" if phase == "converting" else "_cross_check", interrupted
    )


def _interrupted_attempt(
    tmp_path: Path, monkeypatch: Any, phase: str, *, frames: int
) -> tuple[bytes, Path]:
    """Record a session and leave a first finalization attempt stopped at *phase*.

    Returns the spool and the target path, with the interruption already undone
    so the caller's resume runs against an unpatched finalizer -- which is the
    state every recovery test starts from.
    """
    data = _record_session(tmp_path, frames)
    target = tmp_path / "session.nrf"
    _interrupt_at(monkeypatch, phase, target)
    with pytest.raises(FinalizationError):
        finalize_spool(data, target)
    monkeypatch.undo()
    return data, target


@pytest.mark.parametrize("phase", ["converting", "cross_checking", "publishing"])
def test_interruption_records_stopped_phase(tmp_path: Path, monkeypatch: Any, phase: str) -> None:
    """The progress document names *where* an attempt stopped.

    Written before each phase begins rather than after it finishes: a document
    naming the last phase to complete would leave a crashed attempt
    indistinguishable from one that finished that phase and stopped, which is
    exactly the distinction recovery needs.
    """
    data = _record_session(tmp_path, 12)
    target = tmp_path / "session.nrf"
    _interrupt_at(monkeypatch, phase, target)
    with pytest.raises(FinalizationError):
        finalize_spool(data, target)

    state = diagnose_finalization(target)
    assert not state.published
    assert state.phase == phase
    assert state.status in {"failed", "failed_retryable"}
    assert state.resumable
    assert state.attempts == 1


@pytest.mark.parametrize("phase", ["converting", "cross_checking", "publishing"])
def test_resume_produces_one_of_every_row(tmp_path: Path, monkeypatch: Any, phase: str) -> None:
    """Interrupting and running again must not append the same logical record twice.

    The guarantee is structural rather than incremental: the previous attempt's
    staged session was never published, so no reader ever saw a row of it, and
    the resume rewrites it from the start. The check is against a session built
    from the same spool in one clean pass -- identical row counts everywhere, not
    merely "no obvious duplicates".
    """
    data = _record_session(tmp_path, 16)
    reference = finalize_spool(data, tmp_path / "reference.nrf")

    target = tmp_path / "session.nrf"
    _interrupt_at(monkeypatch, phase, target)
    with pytest.raises(FinalizationError):
        finalize_spool(data, target)
    monkeypatch.undo()

    resumed = resume_finalization(data, target)
    assert resumed.progress.attempts == 2
    assert resumed.counts == reference.counts

    with NrfReader.open(target) as reader, NrfReader.open(tmp_path / "reference.nrf") as expected:
        for ledger in LEDGERS:
            assert reader.record_extent(ledger.schema_id) == expected.record_extent(
                ledger.schema_id
            ), ledger.schema_id
        assert reader.stream_extent("neural") == expected.stream_extent("neural")
        assert reader.completeness_verdict == "verified_complete"


def test_resume_rejects_unattempted_target(tmp_path: Path) -> None:
    """Resuming is stated, not inferred: there must be an attempt to continue."""
    data = _record_session(tmp_path, 8)
    with pytest.raises(FinalizationError, match="records no finalization to resume"):
        resume_finalization(data, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_resume_rejects_published_target(tmp_path: Path) -> None:
    data = _record_session(tmp_path, 8)
    finalize_spool(data, tmp_path / "session.nrf")
    with pytest.raises(FinalizationError, match="already published"):
        resume_finalization(data, tmp_path / "session.nrf")


def test_resume_rejects_different_spool(tmp_path: Path, monkeypatch: Any) -> None:
    """Any mismatch rejects the resume (contract section 7).

    A target built from one spool must never absorb a second one, and the
    recorded progress is exactly what makes the two distinguishable. The two
    spools below record different sessions, which is the mismatch the rule names
    first -- not merely different contents of one.
    """
    first = _record_session(tmp_path, 12)
    second = _control_session(tmp_path)
    target = tmp_path / "session.nrf"

    _interrupt_at(monkeypatch, "cross_checking", target)
    with pytest.raises(FinalizationError):
        finalize_spool(first, target)
    monkeypatch.undo()

    with pytest.raises(FinalizationError, match="different source"):
        resume_finalization(second, target)
    assert not target.exists()


def test_resume_rejects_grown_source(tmp_path: Path, monkeypatch: Any) -> None:
    """Committed extents are part of the resume identity (contract section 7).

    The two sources below are the *same* session: same id, same plan
    fingerprint, same spool UUID, same output format version. Only the committed
    extents differ, because one is the crash-shaped prefix of the other -- and
    that is the mismatch the contract names last and the one a reader of the
    progress document cannot otherwise see. Reconciling it would silently
    finalize a session under a cursor produced from a prefix that is no longer
    the one being handed over.
    """
    full = _record_session(tmp_path, 16)
    prefix = _unsealed(full)
    assert (
        spool.scan_spool(prefix).committed_prefix_end < spool.scan_spool(full).committed_prefix_end
    )
    target = tmp_path / "session.nrf"

    _interrupt_at(monkeypatch, "converting", target)
    with pytest.raises(FinalizationError):
        finalize_spool(prefix, target)
    monkeypatch.undo()

    with pytest.raises(FinalizationError, match="committed_prefix_end") as raised:
        resume_finalization(full, target)
    assert "committed_transactions" in str(raised.value)
    assert not target.exists()


def test_only_completed_conversion_moves_cursor(tmp_path: Path, monkeypatch: Any) -> None:
    """``last_consumed_transaction_id`` is a cursor, not a success stamp.

    It says what is materialized in the staged session, so it moves the moment
    the conversion seals one -- not when the whole finalization succeeds, and
    not when the cross-check has passed -- and it stays at zero while nothing
    has been materialized. A value written only on success could not tell a
    resume anything, because a finalization that succeeded is not one that gets
    resumed.
    """
    data = _record_session(tmp_path, 12)
    last = spool.scan_spool(data).last_transaction_id
    assert last > 0

    stopped_early = tmp_path / "early.nrf"
    _interrupt_at(monkeypatch, "converting", stopped_early)
    with pytest.raises(FinalizationError):
        finalize_spool(data, stopped_early)
    monkeypatch.undo()
    early = diagnose_finalization(stopped_early)
    assert early.progress is not None
    assert early.progress.last_consumed_transaction_id == 0
    assert early.progress.staged is None

    stopped_late = tmp_path / "late.nrf"
    _interrupt_at(monkeypatch, "publishing", stopped_late)
    with pytest.raises(FinalizationError):
        finalize_spool(data, stopped_late)
    monkeypatch.undo()
    late = diagnose_finalization(stopped_late)
    assert late.progress is not None
    assert late.progress.last_consumed_transaction_id == last
    assert late.progress.staged is not None
    assert late.progress.staged.counts.frames == 12


def test_resume_at_publication_publishes_without_rewrite(tmp_path: Path, monkeypatch: Any) -> None:
    """Contract section 4.5: a resume MUST NOT rewrite already committed valid NRF data.

    The previous attempt sealed the staged session, validated it, and died on the
    rename. That session is committed NRF data -- every transaction in its
    journal is committed and it carries a termination record -- so the resume
    verifies and publishes it. Proven two ways, because either alone is weak:
    the conversion is made to fail if it is called at all, and the published
    session is compared byte for byte with the staged one the first attempt left.
    """
    from neurale.recording import _finalizer

    data, target = _interrupted_attempt(tmp_path, monkeypatch, "publishing", frames=16)

    staged = target.with_name(target.name + STAGING_SUFFIX) / STAGING_SESSION
    before = staged.read_bytes()
    assert before

    def refuse(*arguments: Any, **keywords: Any) -> Any:
        raise AssertionError("the resume rewrote a session that was already sealed and validated")

    monkeypatch.setattr(_finalizer, "_convert", refuse)
    report = resume_finalization(data, target)
    monkeypatch.undo()

    assert report.progress.attempts == 2
    after = target.read_bytes()
    assert after == before
    with NrfReader.open(target) as reader:
        assert reader.completeness_verdict == "verified_complete"


def test_resume_verifies_staged_session_before_publishing(tmp_path: Path, monkeypatch: Any) -> None:
    """Not rewriting is not the same as not checking.

    The staged session is verified again by the same cross-check the first
    attempt ran, against the counts that attempt recorded. Without that, "resume
    at publication" would degrade into publishing whatever happens to be sitting
    in the staging directory.
    """
    from neurale.recording import _finalizer

    data, target = _interrupted_attempt(tmp_path, monkeypatch, "publishing", frames=12)

    checked: list[Any] = []
    original = _finalizer._cross_check

    def record(session: Any, counts: Any, kind: str) -> Any:
        checked.append(counts)
        return original(session, counts, kind)

    monkeypatch.setattr(_finalizer, "_cross_check", record)
    resume_finalization(data, target)
    monkeypatch.undo()

    assert len(checked) == 1
    assert checked[0].frames == 12
    assert not target.with_name(target.name + STAGING_SUFFIX).exists()


def test_restarted_conversion_clears_cursor(tmp_path: Path, monkeypatch: Any) -> None:
    """A cursor may never outlive the session it points into.

    The staged session is removed when a conversion starts over, and the cursor
    naming it goes with it. Left behind, it would let a *later* resume publish a
    half-converted session as though the first attempt had sealed it -- the one
    way this design could promote something nothing verified.
    """
    from neurale.recording import _finalizer

    data, target = _interrupted_attempt(tmp_path, monkeypatch, "publishing", frames=12)
    assert diagnose_finalization(target).progress.staged is not None  # type: ignore[union-attr]

    # The staged session goes away between the attempts -- the shape a cleanup
    # script, or a half-finished manual intervention, leaves.
    (target.with_name(target.name + STAGING_SUFFIX) / STAGING_SESSION).unlink()

    # The rewrite is interrupted partway, which is the state the assertion is
    # about: the cursor was cleared when the conversion started over and the new
    # conversion has not finished, so nothing on disk is claimed by anything.
    def interrupted(*arguments: Any, **keywords: Any) -> Any:
        raise FinalizationError("interrupted", category="writer")

    monkeypatch.setattr(_finalizer, "_convert", interrupted)
    with pytest.raises(FinalizationError):
        resume_finalization(data, target)
    monkeypatch.undo()

    state = diagnose_finalization(target)
    assert state.progress is not None
    assert state.progress.last_consumed_transaction_id == 0
    assert state.progress.staged is None


def test_crash_during_cross_check_publishes_sealed_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The cursor records materialization, so the seal is never left unclaimed.

    The conversion seals the staged session and the cross-check runs *after*
    that. If the cursor moved only once the cross-check passed, a crash inside
    it would leave a sealed, terminated, committed NRF session that no progress
    document claimed -- and the next attempt would delete and rewrite it, which
    contract section 4.5 forbids. Recording the cursor between the two is what
    closes that window, and it costs no verification: the resume runs the same
    cross-check before publishing anything.
    """
    from neurale.recording import _finalizer

    data = _record_session(tmp_path, 16)
    target = tmp_path / "session.nrf"
    last = spool.scan_spool(data).last_transaction_id
    _interrupt_at(monkeypatch, "cross_checking", target)
    with pytest.raises(FinalizationError):
        finalize_spool(data, target)
    monkeypatch.undo()

    state = diagnose_finalization(target)
    assert state.progress is not None
    assert state.progress.phase == "cross_checking"
    assert state.progress.last_consumed_transaction_id == last
    assert state.progress.staged is not None

    staged = target.with_name(target.name + STAGING_SUFFIX) / STAGING_SESSION
    before = staged.read_bytes()
    assert before

    def refuse(*arguments: Any, **keywords: Any) -> Any:
        raise AssertionError("the resume rewrote a session that had already been sealed")

    monkeypatch.setattr(_finalizer, "_convert", refuse)
    resume_finalization(data, target)
    monkeypatch.undo()

    after = target.read_bytes()
    assert after == before
    with NrfReader.open(target) as reader:
        assert reader.completeness_verdict == "verified_complete"


def test_unaccounted_sealed_session_is_kept(tmp_path: Path, monkeypatch: Any) -> None:
    """The one window a single progress document cannot close, and what it costs.

    A conversion that sealed its session and died before the write recording it
    leaves committed NRF data nothing vouches for. Deleting it to convert again
    is the rewrite section 4.5 forbids; publishing it is worse, because layer 2
    compares the artifact's extents against a count the conversion produced
    independently and that count died with the process. So the finalizer stops
    and leaves the session exactly where it is, for a person to decide about.
    """
    from dataclasses import replace

    from neurale.recording._finalizer import _write_progress

    data, target = _interrupted_attempt(tmp_path, monkeypatch, "publishing", frames=12)

    staging = target.with_name(target.name + STAGING_SUFFIX)
    staged = staging / STAGING_SESSION
    before = staged.read_bytes()
    # Roll the progress document back to what a crash between the seal and the
    # write that records it would have left: a sealed session on disk, and a
    # document that knows nothing about it.
    progress = diagnose_finalization(target).progress
    assert progress is not None
    _write_progress(staging, replace(progress, last_consumed_transaction_id=0, staged=None))

    with pytest.raises(FinalizationError, match="no progress document accounts for") as raised:
        resume_finalization(data, target)
    assert raised.value.category == "resume_mismatch"
    assert not raised.value.retryable

    assert not target.exists()
    assert staged.read_bytes() == before


def test_unsealed_staged_session_is_rewritten(tmp_path: Path, monkeypatch: Any) -> None:
    """The control for the refusal above: a partial session is not committed data.

    A conversion interrupted partway leaves a directory whose targets were never
    sealed and whose rows no reader ever saw. Nothing in it is committed valid
    NRF data, so rewriting it appends no logical record twice -- and refusing
    here instead would make every ordinary interrupted conversion need a person.
    """
    data, target = _interrupted_attempt(tmp_path, monkeypatch, "converting", frames=12)

    report = resume_finalization(data, target)

    assert report.session_path == target
    assert report.counts.frames == 12
    with NrfReader.open(target) as reader:
        assert reader.completeness_verdict == "verified_complete"


def test_foreign_staged_session_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    """The cursor says a conversion finished; it does not say what is on disk now.

    The staging directory is an ordinary path, and between the crash and the
    resume anything with write access can change it. A different session with
    the same row counts satisfies the cross-check -- counts and extents agree
    *within* the intruder -- and would then be published under this source's
    identity. The binding the finalizer stamps into the manifest as it writes is
    what refuses it, and the refusal neither deletes nor publishes the artifact.
    """
    data, target = _interrupted_attempt(tmp_path, monkeypatch, "publishing", frames=12)

    # An entirely legitimate session of exactly the same shape, from a different
    # recording: same plan, same frame count, different identity. Every count
    # the cross-check compares matches.
    other = tmp_path / "other.nrf"
    assert finalize_spool(_record_session(tmp_path, 12), other).counts.frames == 12
    staged = target.with_name(target.name + STAGING_SUFFIX) / STAGING_SESSION
    staged.unlink()
    shutil.copyfile(other, staged)
    intruder = staged.read_bytes()

    with pytest.raises(FinalizationError, match="not the session this finalization staged") as bad:
        resume_finalization(data, target)
    assert bad.value.category == "resume_mismatch"
    assert not bad.value.retryable
    assert "session id" in str(bad.value)
    assert "spool session uuid" in str(bad.value)

    assert not target.exists()
    assert staged.read_bytes() == intruder


def test_unretryable_failure_is_not_resumed(tmp_path: Path, monkeypatch: Any) -> None:
    """``failed`` and ``failed_retryable`` stay distinguished in the recovery API.

    The finalizer separates them because a non-retryable fault is one a second
    pass over the same inputs reads exactly the same way. Reporting such an
    attempt as resumable would send a caller to spend the work and reach the same
    refusal, and would re-merge two states that were deliberately split.
    """
    from neurale.recording import _finalizer

    data = _record_session(tmp_path, 12)
    target = tmp_path / "session.nrf"

    def unretryable(*arguments: Any, **keywords: Any) -> Any:
        raise FinalizationError("the source cannot be accounted for", retryable=False)

    monkeypatch.setattr(_finalizer, "_cross_check", unretryable)
    with pytest.raises(FinalizationError):
        finalize_spool(data, target)
    monkeypatch.undo()

    state = diagnose_finalization(target)
    assert state.status == "failed"
    assert not state.resumable
    with pytest.raises(FinalizationError, match=r"retrying cannot fix|not a state a resume"):
        resume_finalization(data, target)
    assert not target.exists()


# --- the progress document's own durability -----------------------------------


class _NoReplace:
    """``os`` with the atomic replacement refused: a crash mid-persistence."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def replace(self, source: Any, destination: Any) -> None:
        if Path(destination).name == self._name:
            raise OSError("interrupted")
        os.replace(source, destination)


def test_crash_while_persisting_progress_keeps_previous(tmp_path: Path, monkeypatch: Any) -> None:
    """The write that records an attempt must not be the write that destroys it.

    Overwriting the document in place has a window in which the process can die
    with truncated JSON on disk -- and that is strictly worse than the document
    it replaced, because diagnosis then reports "no finalization to resume" over
    an attempt sitting right there. Durable replacement makes the outcome either
    the old valid document or the new one.
    """
    from neurale.recording import _finalizer

    _data, target = _interrupted_attempt(tmp_path, monkeypatch, "converting", frames=12)

    staging = target.with_name(target.name + STAGING_SUFFIX)
    before = (staging / PROGRESS_FILE).read_bytes()

    monkeypatch.setattr(_finalizer, "os", _NoReplace(PROGRESS_FILE))
    with pytest.raises(OSError, match="interrupted"):
        abandon_finalization(target, reason="dies while recording the decision")
    monkeypatch.undo()

    assert (staging / PROGRESS_FILE).read_bytes() == before
    state = diagnose_finalization(target)
    assert state.unreadable_reason is None
    assert state.progress is not None
    assert state.progress.status != "abandoned"
    assert state.resumable


def test_progress_write_leaves_no_scratch_file(tmp_path: Path, monkeypatch: Any) -> None:
    """The staging directory holds the document and the session, and nothing else.

    Stated because the durable write introduces a temporary file, and a
    temporary file that survives is one a later reader has to be taught to
    ignore.
    """
    _data, target = _interrupted_attempt(tmp_path, monkeypatch, "converting", frames=8)

    staging = target.with_name(target.name + STAGING_SUFFIX)
    assert [path.name for path in staging.iterdir()] == [PROGRESS_FILE]


def test_unreadable_progress_document_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    """Diagnosis describes a broken state rather than being stopped by it."""
    _data, target = _interrupted_attempt(tmp_path, monkeypatch, "converting", frames=8)

    staging = target.with_name(target.name + STAGING_SUFFIX)
    (staging / PROGRESS_FILE).write_text("{not json", encoding="utf-8")
    state = diagnose_finalization(target)
    assert state.progress is None
    assert state.unreadable_reason is not None
    assert not state.resumable


# --- a real process death -----------------------------------------------------

_CRASH_SCRIPT = """
import os, signal, sys
from pathlib import Path
from neurale.recording import finalize_spool
from neurale.recording import _finalizer

spool_path, target = sys.argv[1], sys.argv[2]
original = _finalizer._cross_check

def die(*arguments, **keywords):
    # The process dies with the staged session written and the progress document
    # naming the phase that was running -- which is exactly what a crash leaves.
    os.kill(os.getpid(), signal.SIGKILL)

_finalizer._cross_check = die
finalize_spool(Path(spool_path).read_bytes(), target)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="SIGKILL is POSIX-only")
def test_killed_finalization_leaves_resumable_attempt(
    tmp_path: Path,
) -> None:
    """The persisted progress document is the only witness a crash leaves.

    Interrupting a finalization in-process exercises the error path, which is a
    different thing: an exception unwinds and gets a chance to record what
    happened. ``SIGKILL`` gives it no chance, so what the next process finds is
    whatever was already on disk -- which is the whole reason the phase is
    written *before* the phase runs.
    """
    source = _spool_file(tmp_path, _record_session(tmp_path, 16))
    target = tmp_path / "session.nrf"
    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent(_CRASH_SCRIPT), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(source), str(target)],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()

    state = diagnose_finalization(target)
    assert not state.published
    assert state.status == "running"
    assert state.phase == "cross_checking"
    assert state.resumable
    assert state.completeness_verdict is None

    resumed = resume_finalization(source.read_bytes(), target)
    assert resumed.progress.attempts == 2
    with NrfReader.open(target) as reader:
        assert reader.completeness_verdict == "verified_complete"


# --- abandonment --------------------------------------------------------------


def test_abandon_records_decision_and_removes_staged_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Giving up is a recorded decision, not the absence of one.

    Contract section 5.1 gives an abandoned finalization no completeness verdict,
    ever, and ``finalization_status`` is what says why -- so the status has to
    survive in the progress document. The unpublished staged session goes,
    because no reader ever saw a row of it.
    """
    _, target = _interrupted_attempt(tmp_path, monkeypatch, "cross_checking", frames=12)
    staging = target.with_name(target.name + STAGING_SUFFIX)
    assert (staging / STAGING_SESSION).exists()

    progress = abandon_finalization(target, reason="operator gave up")
    assert progress.status == "abandoned"
    assert progress.detail == "operator gave up"
    assert (staging / PROGRESS_FILE).exists()
    assert not (staging / STAGING_SESSION).exists()

    state = diagnose_finalization(target)
    assert state.status == "abandoned"
    assert not state.resumable
    assert state.completeness_verdict is None
    assert state.complete is None


def test_abandon_never_touches_spool(tmp_path: Path, monkeypatch: Any) -> None:
    """The only reconstruction input there is survives the decision to give up."""
    data = _record_session(tmp_path, 12)
    source = _spool_file(tmp_path, data)
    target = tmp_path / "session.nrf"
    _interrupt_at(monkeypatch, "converting", target)
    with pytest.raises(FinalizationError):
        finalize_spool(source, target)
    monkeypatch.undo()

    abandon_finalization(target, reason="no longer needed")
    assert source.read_bytes() == data


def test_abandoned_finalization_is_not_resumed(tmp_path: Path, monkeypatch: Any) -> None:
    """Abandonment is terminal; restarting is a separate, explicit act."""
    data, target = _interrupted_attempt(tmp_path, monkeypatch, "cross_checking", frames=12)
    abandon_finalization(target, reason="operator gave up")

    with pytest.raises(FinalizationError, match="abandoned"):
        resume_finalization(data, target)
    with pytest.raises(FinalizationError, match="abandoned"):
        finalize_spool(data, target)
    assert not target.exists()


def test_published_session_cannot_be_abandoned(tmp_path: Path) -> None:
    """Deleting a recorded session is a user action, never finalization's."""
    data = _record_session(tmp_path, 8)
    finalize_spool(data, tmp_path / "session.nrf")
    with pytest.raises(FinalizationError, match="already published"):
        abandon_finalization(tmp_path / "session.nrf", reason="changed my mind")
    assert (tmp_path / "session.nrf").exists()


def test_abandon_requires_reason(tmp_path: Path, monkeypatch: Any) -> None:
    _, target = _interrupted_attempt(tmp_path, monkeypatch, "cross_checking", frames=12)
    with pytest.raises(FinalizationError, match="requires a reason"):
        abandon_finalization(target, reason="")


def test_abandon_rejects_unattempted_target(tmp_path: Path) -> None:
    with pytest.raises(FinalizationError, match="no finalization to abandon"):
        abandon_finalization(tmp_path / "session.nrf", reason="nothing here")


# --- the completeness verdict, one fixture per state --------------------------


def test_finalized_native_session_verifies_complete(tmp_path: Path) -> None:
    """Both layers checked, every loss term zero, terminated normal."""
    finalize_spool(_record_session(tmp_path, 24), tmp_path / "session.nrf")
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        assert reader.completeness_verdict == "verified_complete"
        assert reader.complete is True
        assert reader.accounting_verified
        assert not reader.legacy_termination_normal
        assert reader.completeness.accounting_origin == "recorder"
        assert reader.completeness.producer_acceptance_known is True


def test_aborted_native_session_verifies_incomplete(tmp_path: Path) -> None:
    """A sealed session whose termination is not ``normal`` is incomplete.

    Its accounting is still checked -- the summary is there and it is
    consistent -- so this is the case that shows why the verdict and
    ``accounting_verified`` are two fields.
    """
    finalize_spool(_control_session(tmp_path, abort=True), tmp_path / "session.nrf")
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        assert reader.completeness_verdict == "verified_incomplete"
        assert reader.complete is False
        assert reader.accounting_verified


def test_recovery_rebuilt_session_is_never_verified(tmp_path: Path) -> None:
    """A rebuilt summary cannot claim acceptance, so it is never "verified".

    The verdict rests on the abnormal termination and the missing session-end
    alone, and the acceptance columns are the surviving prefix rather than
    evidence of what a producer handed over.
    """
    finalize_spool(_unsealed(_record_session(tmp_path, 16)), tmp_path / "session.nrf")
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        completeness = reader.completeness
        assert completeness.verdict == "verified_incomplete"
        assert completeness.complete is False
        assert completeness.accounting_verified is False
        assert completeness.accounting_origin == "recovery_rebuilt"
        assert completeness.producer_acceptance_known is False

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        manifest = reader.manifest
    known = manifest["extensions"]["neurale.native_finalization"]["recovery_rebuilt_accounting"]
    assert known["known_spool_committed"] == 16
    assert known["known_surviving_prefix"] > 0


# --- accounting against the artifact (layer 2) --------------------------------
#
# These sessions are produced by the real finalizer with one record set corrupted
# on the way out. That is deliberate and is the only way to have the failure at
# all: a session claiming more than it holds is exactly what the finalizer's own
# cross-check makes impossible to produce, and layer 2 exists for the sessions
# some *other* writer produces. Corrupting one column of a real, valid NRF v1.1
# session is the closest available stand-in for that writer -- everything outside
# the injected lie is still what the finalizer wrote.


@contextlib.contextmanager
def _lying_writer(edits: dict[str, Any], *, commit_before_seal: bool = False) -> Iterator[None]:
    """Rewrite named record sets as the finalizer appends them."""
    original_append = NrfWriter.append_records
    original_finalize = NrfWriter.finalize

    def append_records(self: NrfWriter, schema_id: str, columns: Any) -> Any:
        edit = edits.get(schema_id)
        return original_append(self, schema_id, edit(columns) if edit else columns)

    def finalize(self: NrfWriter, **keywords: Any) -> Any:
        if commit_before_seal:
            # Push everything appended so far into its own transaction, so the
            # accounting row is committed by a transaction that is not the one
            # the termination record names.
            self.commit()
        return original_finalize(self, **keywords)

    with (
        mock.patch.object(NrfWriter, "append_records", append_records),
        mock.patch.object(NrfWriter, "finalize", finalize),
    ):
        yield


def _set(**values: Any) -> Any:
    """Return an edit that overwrites single-row columns."""

    def edit(columns: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {**columns, **{name: [value] for name, value in values.items()}}

    return edit


@contextlib.contextmanager
def _termination_extensions(extensions: dict[str, Any] | None) -> Iterator[None]:
    """Seal the session with exactly these journal termination extensions.

    The provenance a termination record declares is written by the finalizer and
    signed by that record's checksum, so the only way to produce a record
    declaring something else is to write it that way -- which is what some other
    writer, or a writer with a bug, would do.
    """
    original_finalize = NrfWriter.finalize

    def finalize(self: NrfWriter, **keywords: Any) -> Any:
        keywords["extensions"] = extensions
        return original_finalize(self, **keywords)

    with mock.patch.object(NrfWriter, "finalize", finalize):
        yield


def _tampered(
    tmp_path: Path, edits: dict[str, Any], *, source: bytes | None = None, **keywords: Any
) -> Path:
    data = _record_session(tmp_path, 8) if source is None else source
    target = tmp_path / "session.nrf"
    with _lying_writer(edits, **keywords):
        # The finalizer's own cross-check would catch the lie before publishing;
        # here it is stood down so the artifact reaches a reader, which is what
        # recovery has to judge.
        with mock.patch("neurale.recording._finalizer._check_accounting_row"):
            with mock.patch("neurale.recording._finalizer._cross_check"):
                finalize_spool(data, target)
    return target


def test_finalized_session_is_verification_control(
    tmp_path: Path,
) -> None:
    """Without it, every failure below could be the tampering harness itself."""
    target = _tampered(tmp_path, {})
    with NrfReader.open(target) as reader:
        assert reader.completeness.findings == ()
        assert reader.completeness_verdict == "verified_complete"


def test_overclaiming_summary_fails_verification(
    tmp_path: Path,
) -> None:
    """The failure gap 11 is about, and the one a generic writer product hits.

    Layer 1 alone is satisfied by a writer claiming a hundred items with every
    loss term zero while the session holds eight. A counter that exceeds what the
    artifact contains is a failed verification, never reconciled by trusting the
    number.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                runtime_accepted=100,
                recorder_accepted=100,
                spool_committed=100,
                nrf_committed=100,
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-DATA-EXTENT" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.complete is False
        assert reader.completeness_verdict == "verified_incomplete"


def test_unbalanced_summary_fails_layer_one(tmp_path: Path) -> None:
    """Adjacent stages only, per handoff, never one total."""
    target = _tampered(tmp_path, {"session-accounting-v1": _set(runtime_accepted=99)})
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-IDENTITY" in reader.completeness.codes
        assert not reader.accounting_verified


def test_rejection_cannot_repair_foreign_identity(
    tmp_path: Path,
) -> None:
    """``accepted = committed + rejected + lost`` is a contract violation.

    A rejected item was never inside the stage it was refused entry to, so
    counting it there would put items inside a count they never reached. The row
    below balances only for a reader that made exactly that mistake.
    """
    target = _tampered(
        tmp_path,
        {"session-accounting-v1": _set(runtime_accepted=10, rejected_before_runtime_acceptance=2)},
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-IDENTITY" in reader.completeness.codes


def test_unsealed_summary_fails_verification(
    tmp_path: Path,
) -> None:
    """A session may not be sealed with accounting committed somewhere else.

    Checked from the journal, where it is a fact rather than a promise: the
    sealing transaction is the one the termination record names, and the
    accounting record set must have grown inside it.
    """
    target = _tampered(tmp_path, {}, commit_before_seal=True)
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-NOT-SEALED" in reader.completeness.codes
        assert not reader.accounting_verified


def test_frame_referencing_missing_blocks_fails_verification(
    tmp_path: Path,
) -> None:
    """``nrf_committed`` must be a statement about readable data, not about rows."""

    def inflate(columns: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            **columns,
            "recorded_signal_block_count": [
                value + 1000 for value in columns["recorded_signal_block_count"]
            ],
        }

    target = _tampered(tmp_path, {"native-frames-v1": inflate})
    with NrfReader.open(target) as reader:
        assert "LEDGER-REFERENCE" in reader.completeness.codes
        assert not reader.accounting_verified


def test_childless_frame_cannot_anchor(tmp_path: Path) -> None:
    """The zero-child anchor rule, checked from the reader's side.

    Null means "none" and a count above zero requires a non-null ordinal; a zero
    standing in for "none" would make the two indistinguishable in a ledger whose
    whole purpose is to be unambiguous about children.
    """

    def orphan(columns: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            **columns,
            "recorded_signal_block_count": [0 for _ in columns["recorded_signal_block_count"]],
        }

    target = _tampered(tmp_path, {"native-frames-v1": orphan})
    with NrfReader.open(target) as reader:
        assert "LEDGER-ANCHOR" in reader.completeness.codes


def test_recorded_loss_makes_session_incomplete(
    tmp_path: Path,
) -> None:
    """Verified accounting and an incomplete session are not a contradiction.

    Every identity holds, the counters match the artifact, and one item was lost
    between the recorder and the spool. The accounting is verified *and* the
    session is incomplete, which is the pair of answers two separate fields exist
    to give.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                runtime_accepted=9,
                recorder_accepted=9,
                lost_between_recorder_and_spool=1,
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert reader.accounting_verified
        assert reader.complete is False
        assert reader.completeness_verdict == "verified_incomplete"


@pytest.mark.parametrize(
    "counters",
    [
        {"rejected_before_runtime_acceptance": 1},
        # ``control_offered`` moves with it: it is the one optional identity a
        # rejection *does* appear in, so leaving it behind would fail layer 1 and
        # the test would prove the wrong thing.
        {"control_rejected": 1, "control_offered": 1},
    ],
    ids=["data", "control"],
)
def test_recording_loss_rejection_is_still_checked(
    tmp_path: Path, counters: dict[str, int]
) -> None:
    """Completeness needs the rejections checked separately.

    These counters appear in no acceptance identity -- the items were never
    accepted by anything -- so without checking them the session would look
    complete while part of the recording was refused.
    """
    target = _tampered(tmp_path, {"session-accounting-v1": _set(**counters)})
    with NrfReader.open(target) as reader:
        assert reader.accounting_verified, reader.completeness.findings
        assert reader.complete is False
        assert reader.completeness_verdict == "verified_incomplete"


def test_late_submission_does_not_change_sealed_verdict(
    tmp_path: Path,
) -> None:
    """``rejected after close`` is caller misuse, not a recording loss.

    Contract section 1.3: it belongs to no identity and must not change the
    verdict, so an otherwise clean session stays complete with the counter
    nonzero on either plane.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                data_rejected_after_close=7, control_rejected_after_close=3
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert reader.complete is True
        assert reader.completeness_verdict == "verified_complete"


# --- the accounting's own facts: provenance and tagged positions --------------


def test_undefined_provenance_summary_is_not_verified(tmp_path: Path) -> None:
    """``accounting_origin`` and ``termination_origin`` are stored facts, not decoration.

    Both layers take the row's *meaning* for granted -- that ``recovery_rebuilt``
    means recovery wrote it, that a recorder origin means a recorder did. A row
    holding values the contract does not define has not established that meaning,
    so every number in it is a number about something unidentified, and calling
    the accounting "checked" would be false.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                accounting_origin="nonsense", termination_origin="nonsense"
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.completeness_verdict == "verified_incomplete"
        assert reader.complete is False


def test_rebuilt_summary_cannot_claim_acceptance(tmp_path: Path) -> None:
    """``recovery_rebuilt`` and ``producer_acceptance_known`` cannot both hold.

    An item the producer handed over and the worker had not committed when the
    process died leaves nothing on disk to count, so a rebuilt summary claiming
    to know acceptance describes a session that cannot exist. The verdict is
    already ``verified_incomplete`` for a rebuilt summary; what this pins is that
    the contradiction is *reported* rather than absorbed.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                accounting_origin="recovery_rebuilt", producer_acceptance_known=True
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified


def test_summary_claiming_undeclared_recovery_termination_fails(
    tmp_path: Path,
) -> None:
    """The termination record is where a recovery origin has to be declared.

    It is the record that says the session ended, and it is signed by its own
    checksum; a summary committed beside it is not a substitute. The session
    below was terminated by a real recorder-origin finalization, so the claim has
    nothing behind it.
    """
    target = _tampered(tmp_path, {"session-accounting-v1": _set(termination_origin="recovery")})
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified


def test_recovery_rebuilt_session_agrees_with_termination(
    tmp_path: Path,
) -> None:
    """The control for the two provenance tests above.

    A genuinely recovered session declares ``recovery`` in both places, so the
    cross-check has to pass there -- otherwise the checks above would be passing
    for the wrong reason.
    """
    finalize_spool(_unsealed(_record_session(tmp_path, 12)), tmp_path / "session.nrf")
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        assert "ACCOUNTING-PROVENANCE" not in reader.completeness.codes
        assert reader.completeness.accounting_origin == "recovery_rebuilt"


def test_recovery_termination_with_recorder_summary_fails(
    tmp_path: Path,
) -> None:
    """The provenance constraint read from the termination's end, not the summary's.

    A recovery wrote the termination exactly when the recorder did not reach the
    end of the session -- and a recorder that did not reach the end wrote no
    accounting summary for anything to find. So a recovery-written termination
    beside a summary claiming a recorder produced it, and claiming to know what
    a producer handed over, describes a session neither could have produced.
    Checking only the ``recovery_rebuilt`` direction leaves exactly this row
    verified, which is what makes the second direction a separate check rather
    than a restatement of the first.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                accounting_origin="recorder",
                termination_origin="recovery",
                producer_acceptance_known=True,
            )
        },
        source=_unsealed(_record_session(tmp_path, 8)),
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.completeness_verdict == "verified_incomplete"
        assert reader.complete is False


def test_termination_without_origin_leaves_summary_uncorroborated(
    tmp_path: Path,
) -> None:
    """Absence is not the ordinary case restated.

    The termination record is what says the session ended, and it is signed by
    its own checksum, so it is where provenance has to be declared. A reader
    that read ``"recorder"`` out of a missing field would derive provenance from
    nothing and could not tell an ordinary session from one whose provenance was
    never written -- which is the whole point of storing it.
    """
    data = _record_session(tmp_path, 8)
    target = tmp_path / "session.nrf"
    with _termination_extensions(None):
        finalize_spool(data, target)

    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.completeness_verdict == "verified_incomplete"


def test_recorder_origin_in_recovery_namespace_is_not_verified(tmp_path: Path) -> None:
    """The namespace is half the claim, so the two halves have to agree.

    ``neurale.recovery`` is written when no session-end record froze the outcome
    dimensions -- that is what makes it the recovery namespace -- so a record
    declaring a recorder-written termination inside it contradicts itself. The
    summary agrees with the *value*, which is exactly why reading only the value
    let this through: every number in the session is consistent and the record
    still describes a termination that could not have been written.
    """
    data = _record_session(tmp_path, 8)
    target = tmp_path / "session.nrf"
    with _termination_extensions({RECOVERY_NAMESPACE: {"termination_origin": "recorder"}}):
        finalize_spool(data, target)

    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.completeness_verdict == "verified_incomplete"
        assert reader.complete is False
        # The finding has to be about the namespace: the value agrees with the
        # summary, so a check that only compared values reports nothing here.
        assert any(RECOVERY_NAMESPACE in finding.detail for finding in reader.completeness.findings)


def test_recovery_origin_in_replay_namespace_is_not_verified(tmp_path: Path) -> None:
    """The same rule read from the other namespace.

    ``neurale.native_replay`` carries the dimensions a session-end record froze,
    and a recovered session has none of them latched, so a recovery origin
    declared there names a session whose recorder both did and did not reach the
    end. Checked from both sides because one side alone would leave the mirror
    image verified.
    """
    data = _unsealed(_record_session(tmp_path, 8))
    target = tmp_path / "session.nrf"
    with _termination_extensions({NATIVE_REPLAY_NAMESPACE: {"termination_origin": "recovery"}}):
        finalize_spool(data, target)

    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert any(
            NATIVE_REPLAY_NAMESPACE in finding.detail for finding in reader.completeness.findings
        )


def test_origin_in_both_namespaces_is_not_verified(
    tmp_path: Path,
) -> None:
    """One termination, one provenance.

    A session was ended either by a recorder that finished or by a recovery that
    stepped in for one that did not. A record declaring both leaves a reader
    choosing which to believe, and a reader that took the first one it found
    would let the contradiction through by the order it happened to look.
    """
    data = _record_session(tmp_path, 8)
    target = tmp_path / "session.nrf"
    with _termination_extensions(
        {
            NATIVE_REPLAY_NAMESPACE: {"termination_origin": "recorder"},
            RECOVERY_NAMESPACE: {"termination_origin": "recorder"},
        }
    ):
        finalize_spool(data, target)

    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.complete is False
        assert any(
            "more than one provenance namespace" in finding.detail
            for finding in reader.completeness.findings
        )


@pytest.mark.parametrize("second", [{}, {"source_session_end_present": False}, "recovery"])
def test_second_provenance_namespace_is_not_verified(tmp_path: Path, second: Any) -> None:
    """How many namespaces are present is settled before anything inside them is read.

    The contract's rule is about the namespaces a record *carries*, not about
    how many of them happen to parse. Counting only the well-formed declarations
    lets a record carrying both pass as an ordinary one whenever the second is
    empty, incomplete, or not even an object -- the reader would discard the
    evidence of the contradiction and then report no contradiction. The first
    namespace here is a perfectly ordinary recorder declaration that agrees with
    the summary, so nothing but the second namespace's presence can fail this.
    """
    data = _record_session(tmp_path, 8)
    target = tmp_path / "session.nrf"
    with _termination_extensions(
        {
            NATIVE_REPLAY_NAMESPACE: {"termination_origin": "recorder"},
            RECOVERY_NAMESPACE: second,
        }
    ):
        finalize_spool(data, target)

    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.complete is False
        assert any(
            "more than one provenance namespace" in finding.detail
            for finding in reader.completeness.findings
        )


@pytest.mark.parametrize("extension", [{}, {"source_session_end_present": True}, "recorder"])
def test_provenance_namespace_without_origin_is_not_verified(
    tmp_path: Path, extension: Any
) -> None:
    """Carrying the namespace is not declaring the origin.

    A record that names the provenance namespace and then says nothing inside it
    -- or holds something a ``termination_origin`` cannot be read from -- leaves
    the summary exactly as uncorroborated as a record that named no namespace at
    all. Reported separately from the empty case so the detail says which of the
    two a reader is looking at.
    """
    data = _record_session(tmp_path, 8)
    target = tmp_path / "session.nrf"
    with _termination_extensions({NATIVE_REPLAY_NAMESPACE: extension}):
        finalize_spool(data, target)

    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-PROVENANCE" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.complete is False
        assert any(
            NATIVE_REPLAY_NAMESPACE in finding.detail for finding in reader.completeness.findings
        )


def test_first_lost_position_without_loss_fails(
    tmp_path: Path,
) -> None:
    """The tagged position must agree with the counters (contract section 5.1).

    Every loss counter is zero and the summary still names item 7 as the first
    one lost. One of the two is wrong and the artifact does not say which, so
    ``accounting_verified`` cannot be true -- and without this check it would be,
    with the session reading ``verified_complete``.
    """
    target = _tampered(tmp_path, {"session-accounting-v1": _set(data_first_lost_ordinal=7)})
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-POSITION" in reader.completeness.codes
        assert not reader.accounting_verified
        assert reader.completeness_verdict == "verified_incomplete"


def test_partial_first_rejection_identity_is_not_position(tmp_path: Path) -> None:
    """The producer identity is a *pair*, and the pair carries the tag.

    An item refused before acceptance never received an ordinal, which is why the
    rejection form is ``(kind, identity)`` rather than a number. Half of it names
    nothing, so a row carrying one half has recorded a position that cannot be
    read.
    """
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                data_first_rejected_message_kind="frame",
                rejected_before_runtime_acceptance=1,
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-POSITION" in reader.completeness.codes
        assert not reader.accounting_verified


def test_first_rejection_identity_without_count_fails(
    tmp_path: Path,
) -> None:
    """A recorded refusal that no counter admits to is a summary contradicting itself."""
    target = _tampered(
        tmp_path,
        {
            "session-accounting-v1": _set(
                control_first_rejected_kind="events", control_first_rejected_identity="3"
            )
        },
    )
    with NrfReader.open(target) as reader:
        assert "ACCOUNTING-POSITION" in reader.completeness.codes
        assert not reader.accounting_verified


def test_unknown_acceptance_never_verifies_complete(
    tmp_path: Path,
) -> None:
    """``verified_complete`` requires acceptance evidence, not merely clean counters.

    With ``producer_acceptance_known`` false the acceptance columns describe what
    *survived*, not what a producer handed over -- so "every loss counter is
    zero" says the survivors were all kept, which is not a completeness claim.
    The accounting is still checked and still consistent, which is exactly why
    the two fields are separate.
    """
    target = _tampered(tmp_path, {"session-accounting-v1": _set(producer_acceptance_known=False)})
    with NrfReader.open(target) as reader:
        assert reader.accounting_verified
        assert reader.completeness.findings == ()
        assert reader.completeness_verdict == "verified_incomplete"
        assert reader.complete is False


def test_judging_never_mutates_session(tmp_path: Path) -> None:
    """Opening, reading, diagnosing, and judging are all read-only (section 7)."""
    finalize_spool(_record_session(tmp_path, 12), tmp_path / "session.nrf")
    root = tmp_path / "session.nrf"
    before = root.read_bytes()
    with NrfReader.open(root) as reader:
        assert reader.completeness_verdict == "verified_complete"
    diagnose_finalization(root)
    after = root.read_bytes()
    assert after == before
