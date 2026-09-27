#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Reader-side completeness and accounting verification.

Normative source: ``docs/development/native_recording_replay.md`` section 5.1.
The distinction it draws is the whole point of this module: the persisted accounting
summary records **facts**, and every verdict about it is derived by the reader
answering the call. Nothing here is read back from the artifact as a judgement,
and ``accounting_verified`` is deliberately absent from the stored summary --
storing it would let an artifact assert a verdict the reader in front of it
disagrees with, with no rule saying which one a caller sees.

Before either layer runs, the summary's own facts are checked: the provenance
enums the contract fixes (``accounting_origin``, ``termination_origin``) and the
tagged first-failed positions. Both layers take the row's meaning for granted --
that ``recovery_rebuilt`` means recovery wrote it, that a position names a real
loss -- so a row whose provenance is undefined or whose position contradicts its
own counters has not established that meaning, and no arithmetic over it can.

Two verification layers, and a reader must perform both:

* **Layer 1 -- internal identities.** The summary must be arithmetically
  consistent with itself, per handoff, *adjacent stages only*. A rejection
  appears in no identity: an item refused entry to a stage was never inside it.
* **Layer 2 -- consistency with the artifact.** Layer 1 alone is satisfied by a
  writer that claims a hundred items and holds ninety-nine, so the committed
  counters are checked against the committed extents of the ledgers, the control
  record sets, and the payload each frame references.

What the two layers together establish is exactly three things -- the accounting
is internally consistent, its committed counters match the committed artifact,
and the accounting and the termination were sealed in one transaction. They are
**not** an independent proof that everything a producer handed over was counted:
a recorder that dropped one accepted frame and wrote every counter one lower
satisfies both layers, and no reader can see the difference, because the only
record of that acceptance was in the process that lost it. That failure is
caught by parity testing, not here, and this module must never be
described as auditing the writer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from ._errors import NrfError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._reader import NrfReader

#: The one public completeness verdict of contract section 5.1. There is exactly
#: one, ``complete`` is derived from it and never the reverse, and whether the
#: accounting was checked is a separate field -- naming two acceptable shapes for
#: this is what would let two tasks ship different answers to one question.
CompletenessVerdict = Literal[
    "verified_complete",
    "verified_incomplete",
    "unverified_legacy",
]

#: NRF v1 record kinds that carry control-plane items (manifest schema's ``kind``
#: enum). ``faults`` is one of them and is also where a *recorder's own* fault
#: rows land, which is why the control-plane check below is a bounded one.
CONTROL_RECORD_KINDS: frozenset[str] = frozenset(
    {
        "events",
        "trials",
        "experiment_state",
        "commands",
        "task_variables",
        "labels",
        "targets",
        "assistance",
        "drops",
        "faults",
    }
)

#: The control record kind whose rows have two provenances. A control-plane fault
#: submission and the recorder's own committed fault record are both ``faults``
#: rows, and the frozen v1 schema has no column that separates them.
FAULT_RECORD_KIND = "faults"

#: Ledger record kinds this module reads. Named here rather than imported from
#: ``neurale.recording`` because the direction of dependency is the other way:
#: the reader must not need the recorder to judge a session.
FRAMES_KIND = "native_frames"
SIGNAL_BLOCKS_KIND = "native_signal_blocks"
DISCONTINUITIES_KIND = "native_discontinuities"
SIGNAL_GAPS_KIND = "native_signal_gaps"
ACCOUNTING_KIND = "session_accounting"

#: Loss counters that must all be zero for a session to be complete. Every one
#: is a handoff that failed, and a finalized session with a nonzero
#: ``lost_during_finalization`` is incomplete by definition.
LOSS_COLUMNS: tuple[str, ...] = (
    "failed_between_runtime_and_recorder",
    "lost_between_recorder_and_spool",
    "lost_during_finalization",
    "lost_between_control_acceptance_and_spool",
    "control_lost_during_finalization",
)

#: Rejection counters that are recording losses and must also be zero. They
#: appear in no identity -- the items were never accepted by anything -- so
#: without checking them separately they would go unchecked entirely.
REJECTION_COLUMNS: tuple[str, ...] = (
    "rejected_before_runtime_acceptance",
    "control_rejected",
)

#: The refusal counters that are **not** part of completeness. Submitting to a
#: session that had already ended is caller misuse, not a recording loss
#: (contract section 1.3), and it must not change the session's verdict.
IGNORED_REJECTION_COLUMNS: tuple[str, ...] = (
    "data_rejected_after_close",
    "control_rejected_after_close",
)

#: The two provenance enums contract section 5.1 fixes. They are stored facts,
#: not decoration: ``accounting_origin`` says whether a producer or a recovery
#: wrote the summary, and ``termination_origin`` whether the recorder or a
#: recovery wrote the session's end. A reader that ignored them would accept a
#: summary whose own provenance is unreadable.
ACCOUNTING_ORIGINS: frozenset[str] = frozenset({"recorder", "recovery_rebuilt"})
TERMINATION_ORIGINS: frozenset[str] = frozenset({"recorder", "recovery"})

#: The two journal termination extensions that declare a termination's origin,
#: each mapped to the one origin it may declare. The namespace is not a container
#: the value happens to sit in: ``neurale.native_replay`` is written when the
#: recorder's session-end record froze the outcome dimensions and
#: ``neurale.recovery`` when none did, so the namespace *is* a provenance claim
#: and it has to agree with the value inside it. A termination carries one of
#: them, never both. Named as literals for the same reason the ledger kinds are:
#: the reader must be able to judge a session without importing the recorder.
TERMINATION_NAMESPACE_ORIGINS: Mapping[str, str] = MappingProxyType(
    {"neurale.recovery": "recovery", "neurale.native_replay": "recorder"}
)
TERMINATION_NAMESPACES: tuple[str, ...] = tuple(TERMINATION_NAMESPACE_ORIGINS)
RECOVERY_TERMINATION_NAMESPACE = "neurale.recovery"

#: Each first-loss ordinal against the loss counters that could have produced it.
#: The implication runs one way only. A non-null ordinal with every corresponding
#: loss counter zero is a summary contradicting itself; the converse is not a
#: defect, because the contract makes reporting a position a MAY -- a
#: finalization-side loss, in particular, has no capture-side ordinal to name.
LOSS_POSITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "data_first_lost_ordinal",
        (
            "failed_between_runtime_and_recorder",
            "lost_between_recorder_and_spool",
            "lost_during_finalization",
        ),
    ),
    (
        "control_first_lost_ordinal",
        (
            "lost_between_control_acceptance_and_spool",
            "control_lost_during_finalization",
        ),
    ),
)

#: Each first-rejection producer identity -- a *pair*, and it is a pair because
#: an item refused before acceptance never received an ordinal. Both halves carry
#: the tag, so half of one is not a position, and it may only be present when the
#: rejection counter it names is non-zero.
REJECTION_POSITIONS: tuple[tuple[tuple[str, str], str], ...] = (
    (
        ("data_first_rejected_message_kind", "data_first_rejected_frame_sequence"),
        "rejected_before_runtime_acceptance",
    ),
    (
        ("control_first_rejected_kind", "control_first_rejected_identity"),
        "control_rejected",
    ),
)


@dataclass(frozen=True, slots=True)
class AccountingFinding:
    """One reason the accounting did not verify.

    Kept as a code plus a rendered detail so a caller can branch on the code
    without parsing prose, and read the detail without re-deriving the numbers.
    """

    code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SessionCompleteness:
    """What the reader derived about one session, just now, from the artifact.

    Every field is computed on the read that produced this object. None of them
    is stored in the accounting summary and read back, so an older artifact never
    pins a newer reader's judgement.
    """

    #: ``None`` when there is no sealed session to judge -- not a defect report.
    #: Completeness is a property of a sealed session, so it does not exist for a
    #: session that was never terminated, and no verdict will ever appear for one
    #: whose finalization was abandoned.
    verdict: CompletenessVerdict | None = None
    #: Derived from :attr:`verdict`, never the reverse. ``None`` means the
    #: question cannot be answered from this artifact, which is not the same as
    #: ``False``: ``reader.complete is False`` distinguishes "verified
    #: incomplete" from "unverifiable", while ``if not reader.complete`` keeps
    #: refusing both.
    complete: bool | None = None
    #: Whether the persisted accounting was *checked* -- both layers -- not
    #: whether the session is complete and not a claim that nothing was lost.
    accounting_verified: bool = False
    #: A fact about a legacy session's termination record, exposed separately
    #: because it is not a completeness verdict. A legacy session terminated
    #: ``normal`` says only that; the completeness claim is unavailable.
    legacy_termination_normal: bool = False
    #: ``"recorder"`` or ``"recovery_rebuilt"``, straight from the summary, or
    #: ``None`` when there is no summary.
    accounting_origin: str | None = None
    #: Whether the acceptance counters are evidence of what a producer handed
    #: over. ``False`` on a rebuilt summary: an item the producer handed over and
    #: the worker had not committed when the process died leaves nothing on disk
    #: to count.
    producer_acceptance_known: bool | None = None
    findings: tuple[AccountingFinding, ...] = field(default_factory=tuple)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(finding.code for finding in self.findings)

    def has(self, code: str) -> bool:
        return any(finding.code == code for finding in self.findings)


class _Evaluation:
    """One evaluation pass. Holds the findings so the checks stay readable."""

    def __init__(self, reader: NrfReader) -> None:
        self._reader = reader
        self._findings: list[AccountingFinding] = []

    def _report(self, code: str, detail: str) -> None:
        self._findings.append(AccountingFinding(code, detail))

    # --- artifact shape ---------------------------------------------------

    def _schemas_of_kind(self, kind: str) -> list[Mapping[str, Any]]:
        return [
            schema
            for schema in self._reader.manifest["record_schemas"]
            if schema.get("kind") == kind
        ]

    def _extent_of_kind(self, kind: str) -> int:
        return sum(
            self._reader.committed_extent(schema["path"]) for schema in self._schemas_of_kind(kind)
        )

    # --- the accounting's own facts ---------------------------------------

    def _check_provenance(self, row: Mapping[str, Any]) -> None:
        """The summary's provenance, before any arithmetic over it.

        Layer 1 and layer 2 both take the row's *meaning* for granted -- that
        ``recovery_rebuilt`` means recovery wrote it and ``producer_acceptance_known``
        means the acceptance columns are evidence. A row whose provenance fields
        hold values the contract does not define, or that disagree with each other
        or with the termination record, has not established that meaning, so the
        arithmetic below is arithmetic over something unverified. Checked here
        rather than left to a caller because ``accounting_verified`` is documented
        as "the persisted accounting was checked", and a check that skipped the
        fields naming *what* was persisted would not be that.
        """
        origin = row["accounting_origin"]
        termination_origin = row["termination_origin"]
        acceptance_known = row["producer_acceptance_known"]

        if origin not in ACCOUNTING_ORIGINS:
            self._report(
                "ACCOUNTING-PROVENANCE",
                f"accounting_origin is {origin!r}, which is not one of "
                f"{sorted(ACCOUNTING_ORIGINS)}",
            )
        if termination_origin not in TERMINATION_ORIGINS:
            self._report(
                "ACCOUNTING-PROVENANCE",
                f"termination_origin is {termination_origin!r}, which is not one of "
                f"{sorted(TERMINATION_ORIGINS)}",
            )
        if origin == "recovery_rebuilt":
            # A rebuilt summary cannot claim acceptance and cannot have been
            # sealed by a recorder that was no longer running: the two facts are
            # what "rebuilt" means, and a row asserting either of them describes
            # a session that cannot exist.
            if acceptance_known:
                self._report(
                    "ACCOUNTING-PROVENANCE",
                    "a recovery_rebuilt summary claims producer_acceptance_known, but an item "
                    "the producer handed over and the worker had not committed leaves nothing "
                    "on disk to count",
                )
            if termination_origin == "recorder":
                self._report(
                    "ACCOUNTING-PROVENANCE",
                    "a recovery_rebuilt summary claims a termination the recorder wrote; "
                    "recovery rebuilt the accounting because the recorder did not finish",
                )
        if termination_origin == "recovery":
            # The same constraint read from the other end, and it has to be
            # checked from both: a recovery wrote the termination exactly when
            # the recorder did not reach the end of the session, and a recorder
            # that did not reach the end wrote no accounting summary either. So
            # a recovery-written termination beside a summary claiming a
            # recorder wrote it, or claiming to know what a producer handed
            # over, describes a session neither the recorder nor recovery could
            # have produced. Checking only the ``recovery_rebuilt`` direction
            # would leave exactly that row verified.
            if origin == "recorder":
                self._report(
                    "ACCOUNTING-PROVENANCE",
                    "the termination was written by recovery but the summary claims a recorder "
                    "wrote the accounting; a recorder that never sealed the session wrote no "
                    "summary for recovery to find",
                )
            if acceptance_known:
                self._report(
                    "ACCOUNTING-PROVENANCE",
                    "the termination was written by recovery but the summary claims "
                    "producer_acceptance_known; what a producer handed over and no worker "
                    "committed left nothing on disk for recovery to count",
                )

        self._check_declared_provenance(termination_origin)

    def _check_declared_provenance(self, termination_origin: Any) -> None:
        """The termination record's own provenance, namespace included.

        The namespace is half the claim, not packaging around it: the recorder
        writes ``neurale.native_replay`` exactly when its session-end record
        froze the outcome dimensions, and a recovery writes ``neurale.recovery``
        exactly when none did. So ``neurale.recovery`` holding
        ``termination_origin: "recorder"`` is a record contradicting itself, and
        reading only the value would take the contradiction at its word. Both
        namespaces at once is the same failure spread over two objects -- a
        session cannot have been ended by a recorder that finished and by a
        recovery that stepped in for one that did not -- and neither is a
        declaration a reader may pick from.

        Silence is a third failure and is already the reason this is checked at
        all: provenance a reader infers from an absent field is provenance
        nothing signed.

        How many namespaces are present is settled **before** anything inside
        them is read, and the order matters. The contract's rule is about the
        namespaces a record carries, not about how many of them happen to parse:
        counting only the well-formed ones would let a record carrying both
        namespaces pass as an ordinary one whenever the second is empty or
        malformed -- the reader would discard the evidence of the contradiction
        and then report no contradiction.
        """
        extensions = self._termination_extensions()
        present = [namespace for namespace in TERMINATION_NAMESPACES if namespace in extensions]
        if not present:
            self._report(
                "ACCOUNTING-PROVENANCE",
                "the termination record declares no termination_origin; the summary's "
                f"{termination_origin!r} is not corroborated by the record that ended the session",
            )
            return
        if len(present) > 1:
            self._report(
                "ACCOUNTING-PROVENANCE",
                "the termination record carries more than one provenance namespace "
                f"({', '.join(sorted(present))}); a termination was written either by a recorder "
                "that finished or by a recovery that stepped in for one that did not, never both",
            )
            return
        namespace = present[0]
        extension = extensions[namespace]
        if not isinstance(extension, Mapping):
            self._report(
                "ACCOUNTING-PROVENANCE",
                f"the termination record's {namespace!r} provenance is {type(extension).__name__}, "
                "not an object a termination_origin could be read from",
            )
            return
        if "termination_origin" not in extension:
            self._report(
                "ACCOUNTING-PROVENANCE",
                f"the termination record carries {namespace!r} but declares no "
                f"termination_origin in it; the summary's {termination_origin!r} is not "
                "corroborated by the record that ended the session",
            )
            return
        origin = str(extension["termination_origin"])
        expected = TERMINATION_NAMESPACE_ORIGINS[namespace]
        if origin != expected:
            self._report(
                "ACCOUNTING-PROVENANCE",
                f"the termination record declares termination_origin {origin!r} under "
                f"{namespace!r}, which is the namespace of a {expected!r}-written termination",
            )
        if origin != termination_origin:
            self._report(
                "ACCOUNTING-PROVENANCE",
                f"the summary states termination_origin {termination_origin!r} but the "
                f"termination record states {origin!r}",
            )

    def _termination_extensions(self) -> Mapping[str, Any]:
        """The termination record's ``extensions``, or an empty mapping.

        Returned whole rather than pre-filtered so the caller can ask which
        namespaces are *there* before asking what is in them. A record with no
        extensions, and one whose ``extensions`` is not an object at all, both
        answer "nothing is declared here", which the caller reports.
        """
        for record in self._reader.journal_records:
            if record.get("kind") != "termination":
                continue
            extensions = record.get("extensions")
            return extensions if isinstance(extensions, Mapping) else {}
        return {}

    def _check_positions(self, row: Mapping[str, Any]) -> None:
        """The tagged first-failed positions against the counters that allow them.

        The contract makes *reporting* a position a MAY, so a null position with
        a non-zero counter is legal -- an implementation may know that something
        was lost without knowing which item was first. What is not legal is the
        other direction: a position naming the first item a session lost, in a
        session whose every corresponding loss counter is zero, is a summary
        disagreeing with itself. Half a producer identity is not a position at
        all: the pair carries the tag, and one half of it names nothing.
        """
        for column, counters in LOSS_POSITIONS:
            if row[column] is None:
                continue
            if all(int(row[name] or 0) == 0 for name in counters):
                self._report(
                    "ACCOUNTING-POSITION",
                    f"{column} names item {row[column]} as the first one lost, but "
                    f"{' + '.join(counters)} is zero",
                )
        for columns, counter in REJECTION_POSITIONS:
            present = [name for name in columns if row[name] is not None]
            if not present:
                continue
            if len(present) != len(columns):
                missing = [name for name in columns if row[name] is None]
                self._report(
                    "ACCOUNTING-POSITION",
                    f"the first-rejection identity is a pair; {', '.join(present)} is set while "
                    f"{', '.join(missing)} is null",
                )
            if int(row[counter] or 0) == 0:
                self._report(
                    "ACCOUNTING-POSITION",
                    f"a first-rejection identity is recorded but {counter} is zero",
                )

    # --- layer 1 ----------------------------------------------------------

    def _check_identities(self, row: Mapping[str, Any]) -> None:
        """The internal identities, adjacent stages only.

        The spool-adjacent counters remain nullable for older sessions and
        other compatible writers that have no spool stage. They must not
        synthesize stage-3 numbers. When they are null the two spool-adjacent
        identities of a plane collapse into one; dropping a stage is legal,
        inventing one is not.
        """
        identities: list[tuple[str, int, int, str]] = []
        spool_committed = row["spool_committed"]
        control_spool_committed = row["control_spool_committed"]

        identities.append(
            (
                "runtime_accepted",
                row["runtime_accepted"],
                row["recorder_accepted"] + row["failed_between_runtime_and_recorder"],
                "recorder_accepted + failed_between_runtime_and_recorder",
            )
        )
        if spool_committed is None:
            identities.append(
                (
                    "recorder_accepted",
                    row["recorder_accepted"],
                    row["nrf_committed"] + row["lost_during_finalization"],
                    "nrf_committed + lost_during_finalization (no spool stage)",
                )
            )
        else:
            identities.append(
                (
                    "recorder_accepted",
                    row["recorder_accepted"],
                    spool_committed + (row["lost_between_recorder_and_spool"] or 0),
                    "spool_committed + lost_between_recorder_and_spool",
                )
            )
            identities.append(
                (
                    "spool_committed",
                    spool_committed,
                    row["nrf_committed"] + row["lost_during_finalization"],
                    "nrf_committed + lost_during_finalization",
                )
            )
        if control_spool_committed is None:
            identities.append(
                (
                    "control_accepted",
                    row["control_accepted"],
                    row["control_nrf_committed"] + row["control_lost_during_finalization"],
                    "control_nrf_committed + control_lost_during_finalization (no spool stage)",
                )
            )
        else:
            identities.append(
                (
                    "control_accepted",
                    row["control_accepted"],
                    control_spool_committed
                    + (row["lost_between_control_acceptance_and_spool"] or 0),
                    "control_spool_committed + lost_between_control_acceptance_and_spool",
                )
            )
            identities.append(
                (
                    "control_spool_committed",
                    control_spool_committed,
                    row["control_nrf_committed"] + row["control_lost_during_finalization"],
                    "control_nrf_committed + control_lost_during_finalization",
                )
            )
        if row["control_offered"] is not None:
            identities.append(
                (
                    "control_offered",
                    row["control_offered"],
                    row["control_accepted"] + row["control_rejected"],
                    "control_accepted + control_rejected",
                )
            )

        for name, left, right, spelled in identities:
            if left != right:
                self._report(
                    "ACCOUNTING-IDENTITY",
                    f"{name} is {left} but {spelled} is {right}",
                )

    # --- layer 2 ----------------------------------------------------------

    def _check_data_plane(self, row: Mapping[str, Any]) -> None:
        """``nrf_committed`` against the committed extent of the item ledgers.

        One accepted data message is one item, so the item ledgers are the frame
        ledger and the discontinuity ledger -- never the block or gap ledgers,
        whose rows are children. Substituting a child count here is the exact
        confusion contract section 1.1 forbids.
        """
        if not self._schemas_of_kind(FRAMES_KIND) or not self._schemas_of_kind(
            DISCONTINUITIES_KIND
        ):
            self._report(
                "ACCOUNTING-NO-LEDGERS",
                "the session carries an accounting summary but not the item ledgers its "
                "nrf_committed counter would be checked against",
            )
            return
        items = self._extent_of_kind(FRAMES_KIND) + self._extent_of_kind(DISCONTINUITIES_KIND)
        if row["nrf_committed"] != items:
            self._report(
                "ACCOUNTING-DATA-EXTENT",
                f"nrf_committed is {row['nrf_committed']} but the frame and discontinuity "
                f"ledgers committed {items} rows between them",
            )

    def _check_control_plane(self, row: Mapping[str, Any]) -> None:
        """``control_nrf_committed`` against the control record sets.

        Bounded rather than pinned, and deliberately so. Every control item lands
        in exactly one control record set, but the ``faults`` set also holds the
        recorder's own committed fault records and the row a finalizer writes for
        a faulted termination that no committed fault backs -- and the frozen v1
        fault schema has no column separating the two provenances. So the exact
        statement the artifact supports is that the counter is at least the rows
        in the sets that hold only control items, and at most those plus every
        fault row. The upper bound is the one layer 2 exists for: a session
        claiming more control records than it contains fails it.
        """
        exact = sum(
            self._extent_of_kind(kind)
            for kind in sorted(CONTROL_RECORD_KINDS - {FAULT_RECORD_KIND})
        )
        faults = self._extent_of_kind(FAULT_RECORD_KIND)
        committed = row["control_nrf_committed"]
        if committed > exact + faults:
            self._report(
                "ACCOUNTING-CONTROL-EXTENT",
                f"control_nrf_committed is {committed} but the control record sets hold at most "
                f"{exact + faults} rows ({exact} outside the fault set, {faults} fault rows)",
            )
        elif committed < exact:
            self._report(
                "ACCOUNTING-CONTROL-EXTENT",
                f"control_nrf_committed is {committed} but {exact} rows are committed to control "
                "record sets that hold nothing else",
            )

    def _check_sealed_with_termination(self) -> None:
        """The accounting and the termination were committed by one transaction.

        Checked from the journal, which is where it is a fact rather than a
        promise: the sealing transaction is the one the termination record names,
        and both record sets must have grown inside it. A session cannot be
        sealed with accounting that disagrees with it or with none at all.
        """
        termination = self._reader.committed_state.termination
        assert termination is not None
        accounting_paths = {schema["path"] for schema in self._schemas_of_kind(ACCOUNTING_KIND)}
        prepare = self._prepare_of(termination.last_transaction_id)
        if prepare is None:
            self._report(
                "ACCOUNTING-NOT-SEALED",
                f"the journal has no prepare record for sealing transaction "
                f"{termination.last_transaction_id}",
            )
            return
        grown = {
            entry["target_path"]
            for entry in prepare.get("extents", ())
            if entry["after"] > entry["before"]
        }
        if not (accounting_paths & grown):
            self._report(
                "ACCOUNTING-NOT-SEALED",
                "the accounting summary was not committed by the transaction that sealed the "
                f"session ({termination.last_transaction_id})",
            )

    def _prepare_of(self, transaction_id: str) -> Mapping[str, Any] | None:
        for record in self._reader.journal_records:
            if record.get("kind") == "prepare" and record.get("transaction_id") == transaction_id:
                return record
        return None

    def _check_references(self) -> None:
        """Every ledger reference resolves inside a committed extent.

        A frame that names blocks past the block ledger's extent, or a block
        whose rows land past its stream's extent, describes data the session does
        not contain. This is the part of layer 2 that reads rows rather than
        counters, and it is what makes ``nrf_committed`` a statement about
        readable data rather than about a number of rows.
        """
        if not self._schemas_of_kind(FRAMES_KIND):
            return
        blocks_extent = self._extent_of_kind(SIGNAL_BLOCKS_KIND)
        gaps_extent = self._extent_of_kind(SIGNAL_GAPS_KIND)

        for schema in self._schemas_of_kind(FRAMES_KIND):
            rows = self._reader.read_records(schema["id"])
            self._check_children(
                schema["id"],
                rows.get("first_signal_block_ordinal", []),
                rows.get("recorded_signal_block_count", []),
                blocks_extent,
                "signal block",
            )
        for schema in self._schemas_of_kind(DISCONTINUITIES_KIND):
            rows = self._reader.read_records(schema["id"])
            self._check_children(
                schema["id"],
                rows.get("first_signal_gap_ordinal", []),
                rows.get("signal_gap_count", []),
                gaps_extent,
                "signal gap",
            )
        for schema in self._schemas_of_kind(SIGNAL_BLOCKS_KIND):
            self._check_block_payload(schema["id"])

    def _check_children(
        self,
        schema_id: str,
        anchors: Sequence[Any],
        counts: Sequence[Any],
        extent: int,
        noun: str,
    ) -> None:
        """The zero-child anchor rule, plus the range it implies.

        A count of zero requires a null anchor and a count above zero requires a
        non-null one; a null standing in for a zero, or a zero standing in for
        "none", would make the two indistinguishable in a ledger whose whole
        purpose is to be unambiguous about children.
        """
        for i, (anchor, count) in enumerate(zip(anchors, counts, strict=True)):
            if count == 0:
                if anchor is not None:
                    self._report(
                        "LEDGER-ANCHOR",
                        f"{schema_id} row {i} has no {noun} children but anchors at {anchor}",
                    )
                continue
            if anchor is None:
                self._report(
                    "LEDGER-ANCHOR",
                    f"{schema_id} row {i} claims {count} {noun} children with no anchor",
                )
                continue
            if anchor + count > extent:
                self._report(
                    "LEDGER-REFERENCE",
                    f"{schema_id} row {i} references {noun} rows "
                    f"[{anchor}, {anchor + count}) but only {extent} are committed",
                )

    def _check_block_payload(self, schema_id: str) -> None:
        rows = self._reader.read_records(schema_id)
        stream_ids = rows.get("stream_id", [])
        offsets = rows.get("row_offset", [])
        counts = rows.get("n_samples", [])
        extents: dict[str, int] = {}
        for i, (stream_id, offset, count) in enumerate(
            zip(stream_ids, offsets, counts, strict=True)
        ):
            key = str(stream_id)
            if key not in extents:
                try:
                    extents[key] = self._reader.stream_extent(key)
                except (KeyError, NrfError):
                    self._report(
                        "LEDGER-REFERENCE",
                        f"{schema_id} row {i} names stream {key!r}, which the manifest "
                        "does not declare",
                    )
                    extents[key] = 0
                    continue
            if offset + count > extents[key]:
                self._report(
                    "LEDGER-REFERENCE",
                    f"{schema_id} row {i} places {count} samples of stream {key!r} at "
                    f"{offset}, past its committed extent {extents[key]}",
                )

    # --- the verdict ------------------------------------------------------

    def run(self) -> SessionCompleteness:
        reader = self._reader
        termination = reader.committed_state.termination
        if termination is None:
            # Not a defect report: completeness is a property of a sealed
            # session, and this one has not reached stage 5.
            return SessionCompleteness()

        schemas = self._schemas_of_kind(ACCOUNTING_KIND)
        if not schemas:
            return self._legacy(termination.kind)

        try:
            row = self._read_summary(schemas)
        except NrfError as error:
            self._report("ACCOUNTING-UNREADABLE", str(error))
            row = None
        if row is None:
            return SessionCompleteness(
                verdict="verified_incomplete",
                complete=False,
                accounting_verified=False,
                findings=tuple(self._findings),
            )

        origin = str(row["accounting_origin"])
        acceptance_known = bool(row["producer_acceptance_known"])
        # Provenance and positions first: they establish what the counters mean,
        # and the identities below are arithmetic over that meaning.
        self._check_provenance(row)
        self._check_positions(row)
        self._check_identities(row)
        try:
            self._check_data_plane(row)
            self._check_control_plane(row)
            self._check_sealed_with_termination()
            self._check_references()
        except NrfError as error:
            self._report("ACCOUNTING-UNREADABLE", str(error))

        checked = not self._findings
        # A rebuilt summary is never "verified": recovery states what it found,
        # and the acceptance columns are the surviving prefix rather than
        # evidence of what a producer handed over. The session reads
        # verified_incomplete on its abnormal termination and missing session-end
        # alone, without any accounting to check (contract section 5.1).
        if origin == "recovery_rebuilt":
            return SessionCompleteness(
                verdict="verified_incomplete",
                complete=False,
                accounting_verified=False,
                accounting_origin=origin,
                producer_acceptance_known=acceptance_known,
                findings=tuple(self._findings),
            )

        # ``verified_complete`` is the one verdict that asserts something
        # positive, so every leg of it has to be evidence the reader holds:
        # the accounting checked out, the session ended cleanly, nothing was
        # lost or refused -- and the acceptance columns those zeros are stated
        # in are evidence of what a producer handed over. Without
        # ``producer_acceptance_known`` the counters describe only what survived,
        # and a survivor count of zero losses is not a completeness claim.
        clean = (
            checked
            and acceptance_known
            and termination.kind == "normal"
            and all(int(row[name] or 0) == 0 for name in LOSS_COLUMNS)
            and all(int(row[name] or 0) == 0 for name in REJECTION_COLUMNS)
        )
        return SessionCompleteness(
            verdict="verified_complete" if clean else "verified_incomplete",
            complete=bool(clean),
            accounting_verified=checked,
            accounting_origin=origin,
            producer_acceptance_known=acceptance_known,
            findings=tuple(self._findings),
        )

    def _read_summary(self, schemas: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        if len(schemas) != 1:
            self._report(
                "ACCOUNTING-SHAPE",
                f"a session carries one accounting summary; this one declares {len(schemas)}",
            )
            return None
        schema = schemas[0]
        extent = self._reader.committed_extent(schema["path"])
        if extent == 0:
            # Declared but empty: this is a summary-capable session whose summary
            # is missing, which is a failed verification. It is *not*
            # `unverified_legacy` -- that verdict means "there is a sealed
            # session and it carries no evidence", and this one promised
            # evidence it did not commit.
            self._report(
                "ACCOUNTING-MISSING",
                f"{schema['id']} is declared but no row was committed to it",
            )
            return None
        if extent != 1:
            self._report(
                "ACCOUNTING-SHAPE",
                f"{schema['id']} holds {extent} rows; the summary is one row per session",
            )
            return None
        columns = self._reader.read_records(schema["id"], 0, 1)
        return {name: values[0] for name, values in columns.items()}

    def _legacy(self, termination_kind: str) -> SessionCompleteness:
        """A session written before the summary existed.

        The case that occurs: a legacy session terminated ``normal`` reads
        ``unverified_legacy`` with ``complete is None``. It must not read
        ``complete is True`` -- the session is perfectly readable, it is the
        completeness claim that is unavailable -- and it must not raise. What the
        artifact does say is exposed as ``legacy_termination_normal``: a fact
        about the termination record, not a completeness verdict.

        A legacy session terminated ``aborted`` or ``faulted`` is not legacy for
        this purpose at all: its own termination record proves it incomplete, so
        it reads ``verified_incomplete`` with ``accounting_verified: False``.
        Incompleteness has cheaper proofs than completeness.
        """
        if termination_kind == "normal":
            return SessionCompleteness(
                verdict="unverified_legacy",
                complete=None,
                accounting_verified=False,
                legacy_termination_normal=True,
            )
        return SessionCompleteness(
            verdict="verified_incomplete",
            complete=False,
            accounting_verified=False,
        )


def evaluate_completeness(reader: NrfReader) -> SessionCompleteness:
    """Derive one session's completeness verdict, now, from the artifact.

    Read-only in every branch: opening, reading, and judging a session never
    mutates it (contract section 7).
    """
    return _Evaluation(reader).run()
