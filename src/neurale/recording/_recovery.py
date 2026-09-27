#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Crash recovery and resumable finalization.

Normative source: ``docs/development/native_recording_replay.md`` sections 4.5
(process crash), 5.1 (accounting and the completeness verdict), and 7 (artifact
retention and resume).

Five operations, and the split between them is the contract's, not a
convenience:

``diagnose_spool`` / ``diagnose_finalization``
    Read-only. Ordinary diagnosis must never mutate a spool or a session, so
    these two take no destination and open nothing for writing. They are safe to
    run against a spool another process may still be appending to.
``finalize_spool(..., dry_run=True)``
    Proposes. Validates the source completely and reports what a real attempt
    would produce, having created no staging directory and no session.
``repair_spool``
    The one operation that changes a spool, and it is separately requested,
    audited, and non-destructive: truncating an invalid tail *quarantines the
    original* and emits a report naming what was moved and why. Nothing is
    deleted -- a spool is the only reconstruction input there is.
``resume_finalization``
    Continues an attempt that was interrupted, and refuses anything else. A
    resume is allowed only when the superblock validates and the session id and
    plan fingerprint match the recorded progress; any mismatch rejects rather
    than reconciles.
``abandon_finalization``
    Makes giving up explicit and terminal. It never touches the spool and never
    touches a published session -- the evidence outlives the decision.

The rule that shapes all five: **a record inside an incomplete or
checksum-invalid transaction must not be promoted, ever, by any code path.**
Everything here works from ``scan_spool``'s committed prefix, which is the only
thing that ever crosses out of the spool, and truncation removes bytes past that
prefix rather than promoting anything into it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurale.io.nrf import CompletenessVerdict, NrfError, NrfReader
from neurale.io.nrf._canonical import canonical_json_bytes

from . import _spool_format as spool
from ._errors import FinalizationError, SpoolSourceError
from ._finalizer import (
    PROGRESS_FILE,
    STAGING_SESSION,
    STAGING_SUFFIX,
    FinalizationProgress,
    FinalizationReport,
    FinalizerOptions,
    _write_progress,
    finalize_spool,
)

#: Format tag of the repair report. Versioned for the same reason the progress
#: document is: something later reads it, and a report whose shape changed
#: silently would be evidence nobody can interpret.
REPAIR_REPORT_FORMAT = "neurale-native-spool-repair"
REPAIR_REPORT_VERSION = 1

#: Suffix of the quarantine copy and of the report beside it.
QUARANTINE_SUFFIX = ".quarantined"
REPAIR_REPORT_SUFFIX = ".repair.json"

#: Progress statuses an interrupted finalization can be continued from. A
#: ``failed`` attempt is not among them: the finalizer records that status only
#: for a fault whose own ``retryable`` is false, meaning a second pass over the
#: same inputs reads them the same way.
RESUMABLE_STATUSES: frozenset[str] = frozenset({"running", "failed_retryable"})

#: What ``diagnose_spool`` recommends. Advice, not policy: the caller decides,
#: and the reason the advice rests on is reported beside it.
ACTION_FINALIZE = "finalize"
ACTION_REPAIR = "repair"
ACTION_QUARANTINE = "quarantine"
ACTION_DISCARD = "discard"


# --- the spool --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpoolDiagnosis:
    """What one read-only pass over a spool established.

    Read-only in the strict sense: producing this opens the source once for
    reading and writes nothing, so it is safe against a spool that is still
    being appended to. What it reports about the tail is what was true at the
    moment it read.
    """

    source: Path | None
    byte_length: int
    #: Whether the superblock validated. When false nothing else in the spool is
    #: interpretable and no finalization is possible from it.
    readable: bool
    #: Whether the committed prefix can be accounted for, which is the question
    #: finalization asks. Deliberately not the same as a clean tail: a torn tail
    #: is the ordinary crash path.
    finalizable: bool
    status: str
    codes: tuple[str, ...]
    committed_prefix_end: int
    committed_transactions: int
    last_transaction_id: int
    durable_extent_bytes: int
    data_items: int
    control_items: int
    session_end_present: bool
    capture_outcome: str | None
    session_id: str | None
    plan_fingerprint: str | None
    durability_policy: str | None
    accounting_present: bool
    recommended_action: str
    reason: str

    @property
    def invalid_tail_bytes(self) -> int:
        """Bytes past the committed prefix -- what an explicit repair removes.

        Never promoted and never interpreted: a record inside an incomplete or
        checksum-invalid transaction is not data, it is the shape of data.
        """
        return max(0, self.byte_length - self.committed_prefix_end)

    @property
    def empty(self) -> bool:
        """Whether the spool holds no committed record at all."""
        return self.readable and self.committed_transactions == 0

    def document(self) -> dict[str, Any]:
        """The diagnosis as a plain document, for a report or a log."""
        return {
            "source": None if self.source is None else str(self.source),
            "byte_length": self.byte_length,
            "readable": self.readable,
            "finalizable": self.finalizable,
            "status": self.status,
            "codes": list(self.codes),
            "committed_prefix_end": self.committed_prefix_end,
            "committed_transactions": self.committed_transactions,
            "last_transaction_id": self.last_transaction_id,
            "durable_extent_bytes": self.durable_extent_bytes,
            "invalid_tail_bytes": self.invalid_tail_bytes,
            "data_items": self.data_items,
            "control_items": self.control_items,
            "session_end_present": self.session_end_present,
            "capture_outcome": self.capture_outcome,
            "session_id": self.session_id,
            "plan_fingerprint": self.plan_fingerprint,
            "durability_policy": self.durability_policy,
            "accounting_present": self.accounting_present,
            "recommended_action": self.recommended_action,
            "reason": self.reason,
        }


def diagnose_spool(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> SpoolDiagnosis:
    """Inspect one spool and report what it holds. Never mutates it.

    *source* is the spool's bytes or a path to it. The scan promotes nothing:
    what it reports as committed is what the last valid committed transaction
    made visible, and every byte after that is reported as an invalid tail
    rather than read. A path is scanned in place and never read into memory.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        scan = spool.scan_spool(data)
        path = None
        byte_length = len(data)
    else:
        path = Path(source)
        scan = spool.scan_spool_path(str(path))
        byte_length = path.stat().st_size
    superblock = scan.superblock
    end = scan.session_end
    action, reason = _recommend(scan, byte_length)
    return SpoolDiagnosis(
        source=path,
        byte_length=byte_length,
        readable=superblock is not None,
        finalizable=scan.finalizable,
        status=scan.status,
        codes=scan.codes,
        committed_prefix_end=scan.committed_prefix_end,
        committed_transactions=scan.committed_transactions,
        last_transaction_id=scan.last_transaction_id,
        durable_extent_bytes=scan.durable_extent_bytes,
        data_items=scan.data_items,
        control_items=scan.control_items,
        session_end_present=scan.session_end_present,
        capture_outcome=None if end is None else end.capture_outcome_name,
        session_id=None if superblock is None else superblock.session_id,
        plan_fingerprint=None if superblock is None else superblock.plan_fingerprint,
        durability_policy=None if superblock is None else superblock.durability_policy_name,
        accounting_present=scan.accounting is not None,
        recommended_action=action,
        reason=reason,
    )


def _recommend(scan: spool.SpoolScan, byte_length: int) -> tuple[str, str]:
    """Advise on what to do with this spool, and say what the advice rests on."""
    if scan.superblock is None:
        return (
            ACTION_QUARANTINE,
            "the superblock did not validate, so nothing in the spool is interpretable and no "
            "finalization is possible; quarantine preserves the evidence",
        )
    if not scan.finalizable:
        return (
            ACTION_QUARANTINE,
            "the committed prefix cannot be accounted for "
            f"({', '.join(scan.codes) or 'no diagnostic'}); promoting a prefix nobody can verify "
            "would publish a session its own accounting contradicts",
        )
    if scan.empty:
        return (
            ACTION_DISCARD,
            "the spool holds no committed record, so discarding it deletes no reconstruction "
            "input (contract section 7)",
        )
    if byte_length > scan.committed_prefix_end:
        return (
            ACTION_REPAIR,
            f"{byte_length - scan.committed_prefix_end} bytes follow the committed prefix and "
            "belong to no valid transaction; finalization ignores them, and an explicit repair "
            "is what removes them with the original quarantined",
        )
    return (ACTION_FINALIZE, "the committed prefix is accountable and the spool has no tail")


@dataclass(frozen=True, slots=True)
class SpoolRepairReport:
    """What an explicit repair did, or would do.

    Emitted as a value *and* written beside the quarantined original, because
    contract section 7 requires the repair to name what was moved and why. A
    repair whose only record was a return value would leave a truncated spool on
    disk with nothing to explain it.
    """

    source: Path
    #: Where the untouched original was moved. ``None`` on a dry run and when
    #: there was nothing to repair.
    quarantine_path: Path | None
    report_path: Path | None
    removed_bytes: int
    committed_prefix_end: int
    codes: tuple[str, ...]
    repaired: bool
    dry_run: bool
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "format": REPAIR_REPORT_FORMAT,
            "version": REPAIR_REPORT_VERSION,
            "source": str(self.source),
            "quarantine_path": None if self.quarantine_path is None else str(self.quarantine_path),
            "removed_bytes": self.removed_bytes,
            "committed_prefix_end": self.committed_prefix_end,
            "codes": list(self.codes),
            "repaired": self.repaired,
            "dry_run": self.dry_run,
            "reason": self.reason,
        }


def repair_spool(
    source: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    quarantine_dir: str | os.PathLike[str] | None = None,
) -> SpoolRepairReport:
    """Truncate a spool's invalid tail, explicitly and auditably.

    This is the only operation in this module that changes a spool, and contract
    section 4.5 requires it to be exactly this: separately requested, never a
    side effect of reading, and reported. Section 7 adds that a spool whose tail
    was truncated is **quarantined, not deleted** -- so the original file is
    moved aside intact and the truncated committed prefix is written in its
    place, with a report beside the quarantined copy naming what was moved and
    why.

    Nothing is promoted. The truncation point is the committed prefix end, which
    is where the last valid committed transaction ended; every byte removed
    belonged to no valid transaction and was never readable as data.

    Raises :class:`~neurale.recording.SpoolSourceError` when the superblock does
    not validate, and when the committed prefix cannot be accounted for: in
    neither case is there a prefix worth keeping, so truncating would destroy
    evidence rather than repair anything, and both are spools contract section 7
    sends to quarantine.
    """
    path = Path(source)
    diagnosis = diagnose_spool(path)
    if not diagnosis.readable:
        raise SpoolSourceError(
            f"{path}'s superblock did not validate ({', '.join(diagnosis.codes)}); there is no "
            "committed prefix to truncate to, so quarantine it rather than repairing it"
        )
    if not diagnosis.finalizable:
        # Repair means "what is left is usable", and here it is not: the prefix
        # this would truncate to is one no finalization will accept. Calling that
        # a successful repair would leave a caller holding a spool reported as
        # repaired and refused by the next step, with the tail -- possibly the
        # only clue to why -- already moved aside.
        raise SpoolSourceError(
            f"{path}'s committed prefix cannot be accounted for "
            f"({', '.join(diagnosis.codes) or 'no diagnostic'}); truncating its tail would not "
            "make it finalizable, so quarantine the spool whole rather than repairing it"
        )
    if diagnosis.invalid_tail_bytes == 0:
        return SpoolRepairReport(
            source=path,
            quarantine_path=None,
            report_path=None,
            removed_bytes=0,
            committed_prefix_end=diagnosis.committed_prefix_end,
            codes=diagnosis.codes,
            repaired=False,
            dry_run=dry_run,
            reason="the spool has no bytes past its committed prefix; nothing to repair",
        )

    reason = (
        f"{diagnosis.invalid_tail_bytes} bytes followed the committed prefix at "
        f"{diagnosis.committed_prefix_end} and belonged to no valid committed transaction"
    )
    if dry_run:
        return SpoolRepairReport(
            source=path,
            quarantine_path=None,
            report_path=None,
            removed_bytes=diagnosis.invalid_tail_bytes,
            committed_prefix_end=diagnosis.committed_prefix_end,
            codes=diagnosis.codes,
            repaired=False,
            dry_run=True,
            reason=reason,
        )

    quarantine = _quarantine_path(path, quarantine_dir)
    prefix = _read_prefix(path, diagnosis.committed_prefix_end)
    # The original moves first and the truncated prefix is written second, so a
    # crash between the two leaves the evidence in the quarantine rather than a
    # half-written spool where the source used to be.
    os.replace(path, quarantine)
    path.write_bytes(prefix)
    report = SpoolRepairReport(
        source=path,
        quarantine_path=quarantine,
        report_path=quarantine.with_name(quarantine.name + REPAIR_REPORT_SUFFIX),
        removed_bytes=diagnosis.invalid_tail_bytes,
        committed_prefix_end=diagnosis.committed_prefix_end,
        codes=diagnosis.codes,
        repaired=True,
        dry_run=False,
        reason=reason,
    )
    assert report.report_path is not None
    report.report_path.write_bytes(
        canonical_json_bytes(
            {**report.document(), "diagnosis": diagnosis.document(), "moved": str(quarantine)}
        )
    )
    return report


def _quarantine_path(path: Path, quarantine_dir: str | os.PathLike[str] | None) -> Path:
    directory = path.parent if quarantine_dir is None else Path(quarantine_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    candidate = directory / f"{path.name}{QUARANTINE_SUFFIX}.{stamp}"
    idx = 0
    while candidate.exists():
        idx += 1
        candidate = directory / f"{path.name}{QUARANTINE_SUFFIX}.{stamp}.{idx}"
    return candidate


# --- the finalization -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinalizationState:
    """Where one finalization target stands. Derived by reading, never by writing.

    The target and its staging directory are two places, and which of them
    exists is most of the answer: a published session with no staging directory
    is a finished finalization, staging with no target is one that stopped, and
    both existing is a target somebody published by other means.
    """

    target: Path
    staging: Path
    published: bool
    staging_present: bool
    staged_session_present: bool
    progress: FinalizationProgress | None
    #: ``"not_started"`` when nothing has been attempted, otherwise the progress
    #: document's status, except that a published target reads ``"published"``
    #: whatever the leftover progress says -- the session on disk is the fact.
    status: str
    phase: str | None
    attempts: int
    resumable: bool
    #: The published session's verdict, derived by a reader now. ``None`` when
    #: there is no sealed session to judge, which includes every unpublished and
    #: abandoned finalization (contract section 5.1).
    completeness_verdict: CompletenessVerdict | None = None
    complete: bool | None = None
    accounting_verified: bool = False
    #: Why the session could not be read at all, when it could not be.
    unreadable_reason: str | None = None

    def document(self) -> dict[str, Any]:
        return {
            "target": str(self.target),
            "staging": str(self.staging),
            "published": self.published,
            "staging_present": self.staging_present,
            "staged_session_present": self.staged_session_present,
            "status": self.status,
            "phase": self.phase,
            "attempts": self.attempts,
            "resumable": self.resumable,
            "completeness_verdict": self.completeness_verdict,
            "complete": self.complete,
            "accounting_verified": self.accounting_verified,
            "unreadable_reason": self.unreadable_reason,
            "progress": None if self.progress is None else self.progress.document(),
        }


def diagnose_finalization(output: str | os.PathLike[str]) -> FinalizationState:
    """Report where a finalization target stands. Never mutates anything.

    Reads the published session if there is one -- opening, reading and judging a
    session is read-only (contract section 7) -- and the progress document if
    there is one. A progress document that cannot be parsed is reported rather
    than raised on: diagnosis exists to describe a broken state, not to be
    stopped by it.
    """
    target = Path(output)
    staging = target.with_name(target.name + STAGING_SUFFIX)
    staged_session = staging / STAGING_SESSION
    progress_path = staging / PROGRESS_FILE

    progress: FinalizationProgress | None = None
    unreadable: str | None = None
    if progress_path.exists():
        try:
            progress = FinalizationProgress.load(progress_path)
        except (FinalizationError, KeyError, ValueError, OSError) as error:
            unreadable = f"the progress document could not be read: {error}"

    published = target.exists()
    verdict: CompletenessVerdict | None = None
    complete: bool | None = None
    accounting_verified = False
    if published:
        try:
            with NrfReader.open(target) as reader:
                completeness = reader.completeness
            verdict = completeness.verdict
            complete = completeness.complete
            accounting_verified = completeness.accounting_verified
        except (NrfError, OSError) as error:
            unreadable = f"the published session could not be read: {error}"

    if published:
        status = "published"
    elif progress is None:
        status = "not_started"
    else:
        status = progress.status

    # ``failed`` is deliberately not resumable, and it is the one distinction a
    # recovery API must not blur: the attempt that wrote it had already decided
    # that another pass over the same inputs reads them the same way. Reporting
    # it as resumable would send a caller to spend the work and reach the same
    # refusal, and would make ``failed`` and ``failed_retryable`` -- which the
    # finalizer separates on purpose -- indistinguishable again here.
    resumable = not published and progress is not None and progress.status in RESUMABLE_STATUSES
    return FinalizationState(
        target=target,
        staging=staging,
        published=published,
        staging_present=staging.exists(),
        staged_session_present=staged_session.exists(),
        progress=progress,
        status=status,
        phase=None if progress is None else progress.phase,
        attempts=0 if progress is None else progress.attempts,
        resumable=resumable,
        completeness_verdict=verdict,
        complete=complete,
        accounting_verified=accounting_verified,
        unreadable_reason=unreadable,
    )


def resume_finalization(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    options: FinalizerOptions | None = None,
    dry_run: bool = False,
) -> FinalizationReport:
    """Continue an interrupted finalization, and refuse anything else.

    A resume is the same conversion over the same source, and what it does with
    the previous attempt's staged session depends on how far that attempt got.
    An unsealed, partial conversion may be written again from the start: it was
    never published and no reader ever saw a row of it, so it holds no committed
    valid NRF data and rewriting it appends no logical record twice. A
    conversion the cursor names as *sealed* is not rewritten at all -- contract
    section 4.5 forbids rewriting already committed valid NRF data -- it is
    bound to this source, verified by the same cross-check the first attempt
    ran, and published.

    What this adds over calling :func:`finalize_spool` again is that resuming is
    stated rather than inferred. It requires a recorded attempt to continue --
    calling it on an untouched target raises instead of quietly starting a first
    attempt -- and the mismatch rules of contract section 7 then apply
    unchanged: any disagreement of session id, plan fingerprint, spool identity,
    committed extents, or output format version rejects the resume rather than
    reconciling it.
    """
    state = diagnose_finalization(output)
    if state.published:
        raise FinalizationError(
            f"{state.target} is already published; a resume would produce a second session from "
            "one spool",
            category="resume_mismatch",
            retryable=False,
        )
    if state.progress is None:
        raise FinalizationError(
            f"{state.staging} records no finalization to resume; call finalize_spool to start one",
            category="resume_mismatch",
            retryable=False,
        )
    if state.progress.status == "abandoned":
        raise FinalizationError(
            f"{state.staging} records a finalization that was abandoned"
            + (f" ({state.progress.detail})" if state.progress.detail else "")
            + "; abandoning is terminal, so remove the staging directory to start again",
            category="abandoned",
            retryable=False,
        )
    if state.progress.status not in RESUMABLE_STATUSES:
        # ``failed`` reaches here; ``succeeded`` is caught by the published check
        # above unless the target was moved away, and either way a resume is not
        # what the caller wants. The refusal names the recorded cause rather than
        # the status alone, because "retrying cannot help" is only actionable
        # beside *what* could not be helped.
        raise FinalizationError(
            f"{state.staging} records a finalization whose status is {state.progress.status!r}, "
            "which is not a state a resume continues from"
            + (
                f" ({state.progress.category}: {state.progress.detail})"
                if state.progress.detail
                else ""
            ),
            category="resume_mismatch",
            retryable=False,
        )
    return finalize_spool(source, output, options=options, dry_run=dry_run)


def abandon_finalization(output: str | os.PathLike[str], *, reason: str) -> FinalizationProgress:
    """Give up on a finalization, explicitly and terminally.

    Contract section 5.1: an abandoned finalization never gets a completeness
    verdict -- ``finalization_status`` is what says why -- so abandonment has to
    be a recorded decision rather than the absence of one. This writes that
    decision into the progress document and removes the unpublished staged
    session, which no reader ever saw.

    What it does **not** do is destroy evidence. The spool is not touched: it is
    the only reconstruction input there is, and contract section 7 requires it to
    be retained when the finalizer was interrupted or gave up. A published
    session is not touched either -- deleting a recorded session is a user
    action, never something recording or finalization code does -- so abandoning
    a target that was already published is refused rather than performed.
    """
    if not reason:
        raise FinalizationError(
            "abandoning a finalization requires a reason; an abandoned attempt with no stated "
            "cause is indistinguishable from one nobody finished",
            category="abandoned",
            retryable=False,
        )
    state = diagnose_finalization(output)
    if state.published:
        raise FinalizationError(
            f"{state.target} is already published; a published session is not abandoned, and "
            "deleting one is a user action rather than something finalization code does",
            category="publication",
            retryable=False,
        )
    if state.progress is None:
        raise FinalizationError(
            f"{state.staging} records no finalization to abandon",
            category="resume_mismatch",
            retryable=False,
        )
    if state.progress.status == "succeeded":
        raise FinalizationError(
            f"{state.staging} records a finalization that succeeded; it is not abandoned",
            category="resume_mismatch",
            retryable=False,
        )

    from dataclasses import replace

    abandoned = replace(state.progress, status="abandoned", category="abandoned", detail=reason)
    # Through the finalizer's own durable replacement, not a plain overwrite:
    # abandonment is a terminal decision, and a power failure during the write
    # that records it must leave the previous document rather than a truncated
    # one that reads as "no finalization here at all".
    _write_progress(state.staging, abandoned)
    staged_session = state.staging / STAGING_SESSION
    if staged_session.exists():
        staged_session.unlink()
    return abandoned


# --- shared ------------------------------------------------------------------


def _read_prefix(path: Path, length: int) -> bytes:
    """The first *length* bytes of *path*, without reading the whole file."""
    with open(path, "rb") as handle:
        return handle.read(length)


def read_repair_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a repair report written beside a quarantined spool."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != REPAIR_REPORT_FORMAT:
        raise SpoolSourceError(f"{path} is not a native spool repair report")
    return document
