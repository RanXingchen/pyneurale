#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Diagnose an NRF v1 session without opening it for reading.

Specification reference: ``README.md`` sections 4, 8, and 9.

:class:`~neurale.io.nrf.NrfReader` is deliberately unforgiving: anything that
would make it return data no commit record backs is an exception at open time,
because a reader that degrades gracefully is a reader that hands back plausible
wrong numbers. That leaves a question the reader cannot answer -- *what exactly
is wrong, and how much of the session survives?* -- which is what this module is
for.

Diagnosis never mutates the session and never raises for a damaged one. It
collects findings, replays the journal as far as it stays valid, and reports
where it stopped. Recovery in :mod:`neurale.io.nrf._recovery` is this same
analysis plus the explicitly requested repairs, so a dry run and a real run
cannot disagree about what is wrong.

Three severities separate outcomes that call for different responses:

``corrupt``
    Committed data is unreadable: a committed object is missing, truncated, or
    fails its checksum; the journal is damaged; the manifest cannot be trusted.
    Recovery cannot invent the bytes back.
``incomplete``
    The session never terminated, or terminated abnormally. Everything
    committed is intact -- the recording simply stops early, which must stay
    visible rather than look finished.
``stale``
    A rebuildable cache disagrees with committed state, or uncommitted material
    is lying around. Nothing committed is at risk and recovery can fix it.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from neurale.exceptions import DependencyError

from ._canonical import sha256_hex, verify_record_checksum
from ._errors import NrfError
from ._journal import CHECKPOINT, JournalTail, read_journal
from ._manifest import TargetContract, replay_contracts, target_contracts
from ._paths import (
    JOURNAL_HEAD,
    JOURNAL_TRANSACTIONS,
    MANIFEST,
    STAGING_ROOT,
    chunk_path,
)
from ._replay import CommittedState, ReplayCursor, checkpoint_object
from ._schemas import (
    CHECKPOINT_SCHEMA,
    HEAD_SCHEMA,
    MANIFEST_SCHEMA,
    require_supported_version,
    validate_document,
)
from ._semantics import validate_manifest
from ._zarr import ArraySpec, array_metadata

SEVERITY_CORRUPT = "corrupt"
SEVERITY_INCOMPLETE = "incomplete"
SEVERITY_STALE = "stale"

#: ``ignored_records`` reasons defined by ``recovery-report.schema.json``.
REASON_PARTIAL_LINE = "partial_line"
REASON_BAD_RECORD_CHECKSUM = "bad_record_checksum"
REASON_UNCOMMITTED_PREPARE = "uncommitted_prepare"
REASON_MALFORMED_ORDER = "malformed_order"

#: Top-level trees that hold caches or control records rather than committed
#: payload objects, and so can never be orphan payload.
_NON_PAYLOAD_TREES = frozenset({"journal", "recovery", "checksums", "feature_sets"})


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One finding about a session."""

    code: str
    severity: str
    message: str
    path: str | None = None

    def as_json(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.path is not None:
            document["path"] = self.path
        return document


def _report(
    diagnostics: list[Diagnostic],
    code: str,
    severity: str,
    message: str,
    path: str | None = None,
) -> None:
    """Record one finding on the running list."""
    diagnostics.append(Diagnostic(code, severity, message, path))


@dataclass(frozen=True, slots=True)
class IgnoredRecord:
    """A journal record replay did not use, and the schema-defined reason."""

    sequence: int
    reason: str

    def as_json(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "reason": self.reason}


@dataclass
class SessionDiagnosis:
    """Everything an inspection of one session found.

    ``state`` is the committed state journal replay reached before it stopped,
    which is exactly what a reader would be allowed to see and what recovery
    retains. It is never a guess: a record that failed validation contributes
    nothing to it.
    """

    root: Path
    diagnostics: tuple[Diagnostic, ...] = ()
    manifest: Mapping[str, Any] | None = None
    state: CommittedState = field(default_factory=CommittedState)
    journal_tail: JournalTail = field(default_factory=JournalTail)
    base_checkpoint_id: str | None = None
    ignored_records: tuple[IgnoredRecord, ...] = ()
    retained_transaction_ids: tuple[str, ...] = ()
    ignored_transaction_ids: tuple[str, ...] = ()
    missing_objects: tuple[str, ...] = ()
    corrupt_objects: tuple[str, ...] = ()
    orphan_objects: tuple[str, ...] = ()
    staged_paths: tuple[str, ...] = ()
    #: Files under ``indexes/`` that no commit record backs, and which
    #: quarantine may therefore move. A transaction may commit into
    #: ``indexes/`` with the ``index`` logical role; such an object is data and
    #: never appears here.
    index_cache_paths: tuple[str, ...] = ()
    #: Array paths whose active ``zarr.json`` cache is missing or disagrees
    #: with the frozen manifest, and which recovery therefore rebuilds.
    stale_metadata_arrays: tuple[str, ...] = ()
    #: Cache files that lag replay and which recovery therefore refreshes.
    lagging_caches: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.diagnostics)

    def find(self, code: str) -> tuple[Diagnostic, ...]:
        """Return every diagnostic with *code*."""
        return tuple(item for item in self.diagnostics if item.code == code)

    def has(self, code: str) -> bool:
        return any(item.code == code for item in self.diagnostics)

    def of_severity(self, severity: str) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == severity)

    @property
    def readable(self) -> bool:
        """Whether every committed byte this session claims is intact.

        ``True`` does not mean the session is finished -- see :attr:`complete` --
        only that nothing committed is missing, truncated, or unverifiable.
        """
        return not self.of_severity(SEVERITY_CORRUPT)

    @property
    def complete(self) -> bool:
        """Whether the session terminated normally."""
        termination = self.state.termination
        return termination is not None and termination.kind == "normal"

    @property
    def repairable(self) -> bool:
        """Whether recovery has anything to do that would change the session."""
        return bool(
            self.stale_metadata_arrays
            or self.lagging_caches
            or self.orphan_objects
            or self.staged_paths
            or self.index_cache_paths
        )


def diagnose_session(path: str | Path, *, verify_checksums: bool = True) -> SessionDiagnosis:
    """Inspect an immutable single-file NRF package without extracting it."""
    import zlib
    from zipfile import BadZipFile

    from ._package import Package

    try:
        package = Package(Path(path))
    except (NrfError, BadZipFile, zlib.error, EOFError) as error:
        return SessionDiagnosis(
            root=Path(path),
            diagnostics=(Diagnostic("invalid_package", SEVERITY_CORRUPT, str(error)),),
        )
    try:
        result = _diagnose_objects(package.root, verify_checksums=verify_checksums)
        result.root = Path(path)
        return result
    except (NrfError, BadZipFile, zlib.error, EOFError) as error:
        return SessionDiagnosis(
            root=Path(path),
            diagnostics=(Diagnostic("invalid_package", SEVERITY_CORRUPT, str(error)),),
        )
    finally:
        package.close()


def _diagnose_objects(path, *, verify_checksums: bool = True) -> SessionDiagnosis:
    """Inspect one ``<session>.nrf/`` directory and report what is wrong.

    Parameters
    ----------
    path
        The session directory. It is only read; nothing is created, moved, or
        rewritten, so this is safe to run against a session another process may
        still be writing.
    verify_checksums
        Re-hash every committed object. ``False`` keeps the cheap existence and
        byte-length checks, which catch a missing or truncated object but not
        one whose bytes changed in place.

    Returns
    -------
    SessionDiagnosis
        The findings. This function does not raise for a damaged session; a
        session so damaged that nothing can be established still comes back as
        a diagnosis whose :attr:`~SessionDiagnosis.readable` is ``False``.
    """
    root = path
    diagnostics: list[Diagnostic] = []

    manifest, contracts = _load_manifest(root, diagnostics)
    records, tail = read_journal(root / JOURNAL_TRANSACTIONS)
    ignored: list[IgnoredRecord] = []

    cursor = ReplayCursor(contracts=None if contracts is None else replay_contracts(contracts))
    checkpoints: list[tuple[Mapping[str, Any], CommittedState]] = []
    ignored_transactions: list[str] = []
    stopped_at: int | None = None

    for i, record in enumerate(records):
        snapshot = copy.deepcopy(cursor.state) if record.get("kind") == CHECKPOINT else None
        try:
            cursor.apply(record)
        except NrfError as error:
            stopped_at = i
            sequence = _sequence_of(record, cursor.state.journal_sequence + 1)
            ignored.append(IgnoredRecord(sequence, REASON_MALFORMED_ORDER))
            _report(
                diagnostics,
                "journal_replay_failed",
                SEVERITY_CORRUPT,
                f"journal replay stopped at sequence {sequence}: {error}",
                JOURNAL_TRANSACTIONS,
            )
            break
        if snapshot is not None:
            checkpoints.append((record, snapshot))

    if stopped_at is not None:
        for record in records[stopped_at:]:
            transaction_id = record.get("transaction_id")
            if isinstance(transaction_id, str) and transaction_id not in ignored_transactions:
                ignored_transactions.append(transaction_id)

    _diagnose_journal_tail(tail, cursor.state, diagnostics, ignored)

    for transaction_id, sequence in sorted(cursor.pending_prepares.items()):
        ignored.append(IgnoredRecord(sequence, REASON_UNCOMMITTED_PREPARE))
        if transaction_id not in ignored_transactions:
            ignored_transactions.append(transaction_id)
        _report(
            diagnostics,
            "uncommitted_prepare",
            SEVERITY_STALE,
            f"{transaction_id} prepared at sequence {sequence} but never committed; "
            "its staged objects are invisible and are not promoted by recovery",
            JOURNAL_TRANSACTIONS,
        )

    state = cursor.state
    base_checkpoint_id = _diagnose_checkpoints(root, checkpoints, state, diagnostics)
    lagging = _diagnose_caches(root, manifest, state, diagnostics)
    missing, corrupt = _diagnose_committed_objects(
        root, state, diagnostics, verify_checksums=verify_checksums
    )
    stale_metadata: tuple[str, ...] = ()
    if contracts is not None:
        _diagnose_extents(contracts, state, diagnostics)
        stale_metadata = _diagnose_array_metadata(root, contracts, diagnostics)
    orphans, staged, idxs = _survey_files(root, state, diagnostics)
    _diagnose_termination(state, diagnostics)

    return SessionDiagnosis(
        root=root,
        diagnostics=tuple(diagnostics),
        manifest=manifest,
        state=state,
        journal_tail=tail,
        base_checkpoint_id=base_checkpoint_id,
        ignored_records=tuple(sorted(ignored, key=lambda item: item.sequence)),
        retained_transaction_ids=tuple(cursor.committed_transaction_ids),
        ignored_transaction_ids=tuple(sorted(ignored_transactions)),
        missing_objects=missing,
        corrupt_objects=corrupt,
        orphan_objects=orphans,
        staged_paths=staged,
        index_cache_paths=idxs,
        stale_metadata_arrays=stale_metadata,
        lagging_caches=lagging,
    )


# --- manifest --------------------------------------------------------------


def _load_manifest(
    root: Path, diagnostics: list[Diagnostic]
) -> tuple[dict[str, Any] | None, dict[str, TargetContract] | None]:
    """Load and fully validate the manifest, or report why it cannot be used.

    Without a usable manifest nothing else can be checked against a contract:
    the frozen registries are the only description of what the session was
    supposed to contain. Replay still runs, on record-internal rules alone, so
    a report can at least say how far the journal was well formed.
    """
    path = root / MANIFEST
    if not path.exists():
        _report(
            diagnostics,
            "manifest_missing",
            SEVERITY_CORRUPT,
            "manifest.json is required and is absent; this is not a readable NRF session",
            MANIFEST,
        )
        return None, None
    try:
        manifest = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        _report(
            diagnostics,
            "manifest_unreadable",
            SEVERITY_CORRUPT,
            f"manifest.json is not valid UTF-8 JSON: {error}",
            MANIFEST,
        )
        return None, None
    if not isinstance(manifest, dict):
        _report(
            diagnostics,
            "manifest_unreadable",
            SEVERITY_CORRUPT,
            "manifest.json is not a JSON object",
            MANIFEST,
        )
        return None, None

    try:
        require_supported_version(manifest)
    except NrfError as error:
        _report(diagnostics, "unsupported_version", SEVERITY_CORRUPT, str(error), MANIFEST)
        return None, None

    try:
        validate_document(manifest, MANIFEST_SCHEMA)
    except NrfError as error:
        _report(diagnostics, "manifest_schema_invalid", SEVERITY_CORRUPT, str(error), MANIFEST)
        return manifest, None

    try:
        validate_manifest(manifest)
        contracts = target_contracts(manifest)
    except NrfError as error:
        _report(diagnostics, "manifest_semantics_invalid", SEVERITY_CORRUPT, str(error), MANIFEST)
        return manifest, None
    return manifest, contracts


# --- journal ---------------------------------------------------------------


def _sequence_of(record: Mapping[str, Any], fallback: int) -> int:
    sequence = record.get("sequence")
    return sequence if isinstance(sequence, int) and sequence > 0 else fallback


def _diagnose_journal_tail(
    tail: JournalTail,
    state: CommittedState,
    diagnostics: list[Diagnostic],
    ignored: list[IgnoredRecord],
) -> None:
    """Classify what lies beyond the last usable journal record.

    A partial trailing line is the ordinary outcome of a crash mid-append: the
    writer never finished the record, so nothing it described became visible. A
    *complete* line whose checksum fails is different -- the bytes were written
    and then changed -- and it makes every later record unverifiable. A line
    that is correctly signed but is not a well-formed record is the same class
    of damage: its checksum only proves nobody edited it since it was signed.
    """
    next_sequence = state.journal_sequence + 1
    if tail.invalid_schema_at is not None:
        # The closed ``ignored_records`` vocabulary has no reason for a
        # structurally malformed record; ``malformed_order`` is the entry for a
        # record replay could not use as written, which is what this is.
        ignored.append(IgnoredRecord(next_sequence, REASON_MALFORMED_ORDER))
        _report(
            diagnostics,
            "journal_record_schema_invalid",
            SEVERITY_CORRUPT,
            f"journal record {next_sequence} is signed but is not a valid journal "
            f"record ({tail.schema_error}); replay stops there and no later record is used",
            JOURNAL_TRANSACTIONS,
        )
        return
    if tail.invalid_checksum_at is not None:
        ignored.append(IgnoredRecord(next_sequence, REASON_BAD_RECORD_CHECKSUM))
        _report(
            diagnostics,
            "journal_record_checksum_invalid",
            SEVERITY_CORRUPT,
            f"journal record {next_sequence} fails its record checksum; no later record "
            "can be verified, so the session is capped at the last valid commit",
            JOURNAL_TRANSACTIONS,
        )
        return
    if tail.partial_line_bytes:
        ignored.append(IgnoredRecord(next_sequence, REASON_PARTIAL_LINE))
        _report(
            diagnostics,
            "journal_partial_line",
            SEVERITY_STALE,
            f"the journal ends with {tail.partial_line_bytes} bytes of an unterminated "
            "record; it was never complete and never became visible",
            JOURNAL_TRANSACTIONS,
        )


# --- checkpoints -----------------------------------------------------------


def _diagnose_checkpoints(
    root: Path,
    checkpoints: Sequence[tuple[Mapping[str, Any], CommittedState]],
    state: CommittedState,
    diagnostics: list[Diagnostic],
) -> str | None:
    """Return the newest valid checkpoint ID, reporting every rejected one.

    A checkpoint is valid only when its file checksum matches the journal
    record, its own ``record_checksum`` verifies, it validates against its
    schema, and its state equals replay through its journal sequence. Each
    snapshot passed here is the state *before* its checkpoint record was
    applied, which is precisely the sequence the checkpoint claims.
    """
    newest: str | None = None
    for record, snapshot in checkpoints:
        checkpoint_id = record["checkpoint_id"]
        relative = record["checkpoint_path"]
        absolute = root / relative
        if not absolute.exists():
            _report(
                diagnostics,
                "checkpoint_missing",
                SEVERITY_STALE,
                f"{checkpoint_id} is referenced by the journal but its file is absent; "
                "recovery falls back to an older checkpoint or to sequence 1",
                relative,
            )
            continue
        raw = absolute.read_bytes()
        if sha256_hex(raw) != record["checkpoint_sha256"]:
            _report(
                diagnostics,
                "checkpoint_checksum_mismatch",
                SEVERITY_STALE,
                f"{checkpoint_id} does not match the checksum its journal record records",
                relative,
            )
            continue
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            _report(
                diagnostics,
                "checkpoint_unreadable",
                SEVERITY_STALE,
                f"{checkpoint_id} is not valid UTF-8 JSON: {error}",
                relative,
            )
            continue
        try:
            validate_document(document, CHECKPOINT_SCHEMA)
        except NrfError as error:
            _report(diagnostics, "checkpoint_schema_invalid", SEVERITY_STALE, str(error), relative)
            continue
        if not verify_record_checksum(document):
            _report(
                diagnostics,
                "checkpoint_record_checksum_invalid",
                SEVERITY_STALE,
                f"{checkpoint_id} fails its own record checksum",
                relative,
            )
            continue
        disagreement = _checkpoint_disagreement(document, snapshot)
        if disagreement is not None:
            _report(
                diagnostics,
                "checkpoint_stale",
                SEVERITY_STALE,
                f"{checkpoint_id} does not equal replay through journal sequence "
                f"{snapshot.journal_sequence}: {disagreement}",
                relative,
            )
            continue
        newest = checkpoint_id

    if state.last_checkpoint_id is not None and newest != state.last_checkpoint_id:
        _report(
            diagnostics,
            "checkpoint_unusable",
            SEVERITY_STALE,
            f"the journal's last checkpoint {state.last_checkpoint_id!r} is not usable; "
            f"recovery starts from {newest or 'journal sequence 1'}",
            JOURNAL_TRANSACTIONS,
        )
    return newest


def _checkpoint_disagreement(document: Mapping[str, Any], snapshot: CommittedState) -> str | None:
    """Return why a checkpoint disagrees with replay, or ``None`` if it agrees.

    Every member of the checkpoint is compared, not merely the object *paths*.
    A checkpoint that names the right objects but the wrong digest or byte
    length for one of them is not a usable summary of committed state, and
    accepting it would let recovery report it as the newest valid checkpoint.
    """
    if document["journal_sequence"] != snapshot.journal_sequence:
        return (
            f"claims journal sequence {document['journal_sequence']}, replay reached "
            f"{snapshot.journal_sequence}"
        )
    if document["last_committed_transaction_id"] != snapshot.last_transaction_id:
        return "names a different last committed transaction"
    if dict(document["committed_extents"]) != dict(snapshot.committed_extents):
        return "records different committed extents"
    if sorted(document["sealed_targets"]) != sorted(snapshot.sealed_targets):
        return "records a different sealed-target set"
    # The v1 journal has no index-extent transition, so replay can never derive
    # a non-empty index position and a checkpoint must not claim one.
    if dict(document["index_extents"]):
        return "claims index extents that no journal record can produce"
    return _object_disagreement(document["committed_objects"], snapshot)


def _object_disagreement(
    recorded: Sequence[Mapping[str, Any]], snapshot: CommittedState
) -> str | None:
    """Return why a checkpoint's committed objects disagree with replay."""
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in recorded:
        path = entry["path"]
        if path in by_path:
            return f"lists {path} more than once"
        by_path[path] = entry
    expected = {
        path: checkpoint_object(entry) for path, entry in snapshot.committed_objects.items()
    }
    if set(by_path) != set(expected):
        return "records a different committed-object set"
    for path, want in sorted(expected.items()):
        found = by_path[path]
        differing = sorted(
            member for member in set(want) | set(found) if want.get(member) != found.get(member)
        )
        if differing:
            return (
                f"describes {path} with {', '.join(differing)} that the commit record "
                "creating it does not support"
            )
    return None


# --- caches ----------------------------------------------------------------


def _diagnose_caches(
    root: Path,
    manifest: Mapping[str, Any] | None,
    state: CommittedState,
    diagnostics: list[Diagnostic],
) -> tuple[str, ...]:
    """Check the two non-authoritative caches against replay.

    ``manifest.json`` and ``journal/head.json`` are replaced by independent
    atomic renames, so a crash between them legitimately leaves them at
    different sequences. Lagging is recoverable and recovery refreshes it;
    leading is a validation failure, because it would expose data no commit
    record backs.
    """
    lagging: list[str] = []
    if manifest is not None:
        _check_cache(MANIFEST, manifest.get("commit") or {}, state, diagnostics, lagging)

    head_path = root / JOURNAL_HEAD
    if not head_path.exists():
        _report(
            diagnostics,
            "head_cache_missing",
            SEVERITY_STALE,
            "journal/head.json is absent; it is a rebuildable pointer cache",
            JOURNAL_HEAD,
        )
        lagging.append(JOURNAL_HEAD)
        return tuple(lagging)
    try:
        head = json.loads(head_path.read_bytes().decode("utf-8"))
        validate_document(head, HEAD_SCHEMA)
    except (UnicodeDecodeError, ValueError, NrfError) as error:
        _report(
            diagnostics,
            "head_cache_invalid",
            SEVERITY_STALE,
            f"journal/head.json is not a usable head cache: {error}",
            JOURNAL_HEAD,
        )
        lagging.append(JOURNAL_HEAD)
        return tuple(lagging)
    _check_cache(JOURNAL_HEAD, head, state, diagnostics, lagging)
    return tuple(lagging)


def _check_cache(
    name: str,
    cache: Mapping[str, Any],
    state: CommittedState,
    diagnostics: list[Diagnostic],
    lagging: list[str],
) -> None:
    """Compare one cache's claimed sequence and extents against replay."""
    sequence = cache.get("journal_sequence", 0)
    extents: Mapping[str, int] = cache.get("committed_extents") or {}
    leads = [target for target, extent in extents.items() if extent > state.extent(target)]
    if sequence > state.journal_sequence or leads:
        _report(
            diagnostics,
            "cache_leads_replay",
            SEVERITY_CORRUPT,
            f"{name} claims journal sequence {sequence} and extents replay does not "
            f"support ({sorted(leads)}); a cache may lag replay but never lead it",
            name,
        )
        return
    if sequence < state.journal_sequence:
        _report(
            diagnostics,
            "cache_lags_replay",
            SEVERITY_STALE,
            f"{name} is at journal sequence {sequence} while replay reached "
            f"{state.journal_sequence}; the cache is behind but not wrong",
            name,
        )
        lagging.append(name)


# --- committed objects -----------------------------------------------------


def _diagnose_committed_objects(
    root: Path,
    state: CommittedState,
    diagnostics: list[Diagnostic],
    *,
    verify_checksums: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check every object a commit record claims against the bytes on disk.

    A missing or altered committed object is corruption, never an uncommitted
    tail: the commit line already made it visible, so its absence removes data
    a reader was promised rather than data that never arrived.
    """
    missing: list[str] = []
    corrupt: list[str] = []
    for path, entry in sorted(state.committed_objects.items()):
        absolute = root / path
        if not absolute.exists():
            missing.append(path)
            _report(
                diagnostics,
                "object_missing",
                SEVERITY_CORRUPT,
                f"committed object is absent: {path}",
                path,
            )
            continue
        size = absolute.stat().st_size
        expected = entry.get("byte_length")
        if isinstance(expected, int) and size != expected:
            corrupt.append(path)
            _report(
                diagnostics,
                "object_truncated",
                SEVERITY_CORRUPT,
                f"committed object is {size} bytes, the commit records {expected}: {path}",
                path,
            )
            continue
        if verify_checksums and sha256_hex(absolute.read_bytes()) != entry["sha256"]:
            corrupt.append(path)
            _report(
                diagnostics,
                "object_checksum_invalid",
                SEVERITY_CORRUPT,
                f"committed object fails the checksum its commit records: {path}",
                path,
            )
    return tuple(missing), tuple(corrupt)


def _diagnose_extents(
    contracts: Mapping[str, TargetContract],
    state: CommittedState,
    diagnostics: list[Diagnostic],
) -> None:
    """Check that each committed extent is backed by the objects it needs.

    An extent is a promise about how many items a reader may read. It is only
    meaningful if the frozen array is large enough to hold them and if a
    committed chunk object covers every one of them.
    """
    for target_path, contract in sorted(contracts.items()):
        extent = state.extent(target_path)
        if extent == 0:
            continue
        for spec in contract.arrays:
            capacity = spec.shape[0]
            if extent > capacity:
                _report(
                    diagnostics,
                    "extent_exceeds_capacity",
                    SEVERITY_CORRUPT,
                    f"{target_path} is committed to {extent} items but {spec.path} was "
                    f"frozen with room for {capacity}",
                    spec.path,
                )
                continue
            needed = [
                chunk_path(spec.path, coordinate)
                for coordinate in spec.chunk_coordinates(0, extent)
            ]
            absent = [path for path in needed if path not in state.committed_objects]
            if absent:
                _report(
                    diagnostics,
                    "extent_not_backed_by_objects",
                    SEVERITY_CORRUPT,
                    f"{target_path} is committed to {extent} items but no commit record "
                    f"backs {len(absent)} of the chunks that range needs, starting at "
                    f"{absent[0]}",
                    spec.path,
                )


def _diagnose_array_metadata(
    root: Path,
    contracts: Mapping[str, TargetContract],
    diagnostics: list[Diagnostic],
) -> tuple[str, ...]:
    """Check each active ``zarr.json`` against the frozen manifest.

    The active metadata is a cache, but it is the cache a Zarr reader decodes
    through: a wrong shape or chunk shape silently changes what a read returns.
    So a disagreement is reported and rebuilt rather than tolerated, while the
    reader refuses to decode through it in the meantime.
    """
    stale: list[str] = []
    for contract in contracts.values():
        for spec in contract.arrays:
            problem = array_metadata_problem(root, spec)
            if problem is None:
                continue
            stale.append(spec.path)
            _report(
                diagnostics,
                "zarr_metadata_stale",
                SEVERITY_STALE,
                f"{spec.path}: {problem}",
                f"{spec.path}/zarr.json",
            )
    return tuple(sorted(set(stale)))


#: Every ``zarr.json`` member that decides how committed bytes decode. All of
#: them are compared, because each one can change what a read returns without
#: changing a single stored byte: a swapped ``data_type`` reinterprets the
#: payload, a different ``chunk_key_encoding`` sends Zarr to paths that do not
#: exist and yields fill values, a dropped codec decodes the wrong bytes
#: entirely. ``attributes`` and ``dimension_names`` are annotations that no
#: decoder consults, so they are deliberately left free.
_DECODING_MEMBERS = (
    "zarr_format",
    "node_type",
    "data_type",
    "shape",
    "chunk_grid",
    "chunk_key_encoding",
    "codecs",
    "fill_value",
    "storage_transformers",
)

#: Members Zarr v3 lets a conforming document omit, with the value the omission
#: means. Treating absence as disagreement here would reject valid metadata.
_MEMBER_DEFAULTS: Mapping[str, Any] = {
    "node_type": "array",
    "storage_transformers": [],
}


@cache
def _expected_array_metadata(spec: ArraySpec) -> Mapping[str, Any]:
    """Return the Zarr metadata the frozen manifest implies for one array.

    Built through the same encoder that produced the committed chunks, so the
    comparison is against what this implementation would actually decode with
    rather than against a hand-written idea of the document.
    """
    return array_metadata(spec)


def array_metadata_problem(root: Path, spec: ArraySpec) -> str | None:
    """Return why one array's active metadata is unusable, or ``None``.

    The manifest freezes an array's shape, chunk shape, dtype, endianness,
    codecs, and fill value before the first append, so the whole decoding half
    of the document is derivable from committed state and every member of it is
    checked.
    """
    path = root / spec.path / "zarr.json"
    if not path.exists():
        return "the active zarr.json cache is absent"
    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        return f"the active zarr.json cache is not valid JSON: {error}"
    if not isinstance(document, Mapping):
        return "the active zarr.json cache is not a JSON object"
    try:
        expected = _expected_array_metadata(spec)
    except DependencyError:  # pragma: no cover - without zarr no read can occur
        return None
    for member in _DECODING_MEMBERS:
        found = document.get(member, _MEMBER_DEFAULTS.get(member))
        want = expected.get(member, _MEMBER_DEFAULTS.get(member))
        if _comparable(found) != _comparable(want):
            return (
                f"the active zarr.json declares {member} {_render(found)}, the manifest "
                f"freezes {_render(want)}"
            )
    return None


def _comparable(value: Any) -> Any:
    """Return a form of one metadata member that compares by meaning.

    Only one difference in spelling is tolerated, and only because zarr-python
    itself tolerates it: a named object may omit an empty ``configuration``.
    Anything looser would risk calling a document sound that the decoder then
    refuses, which is the opposite of what this check is for.
    """
    if isinstance(value, Mapping):
        members = {key: _comparable(item) for key, item in value.items()}
        if members.get("configuration") == {}:
            del members["configuration"]
        return members
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_comparable(item) for item in value]
    # NaN is a legal Zarr fill value and does not compare equal to itself.
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _render(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - JSON in, JSON out
        return repr(value)


# --- loose files -----------------------------------------------------------


def _survey_files(
    root: Path, state: CommittedState, diagnostics: list[Diagnostic]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Find staged, orphan, and index-cache files.

    Everything reported here is material *no commit record backs*, which is
    what makes it eligible for quarantine. A committed object is never loose
    material no matter where it lives -- including under ``indexes/``, which a
    transaction is allowed to write into with the ``index`` logical role. Those
    are judged by their commit records in
    :func:`_diagnose_committed_objects` instead, so a damaged one is reported as
    corruption rather than moved out of the session.

    An orphan is a final-path object that exists but that no commit record
    backs -- the residue of a crash between promotion and the commit line. It is
    invisible to a reader by construction, since visibility comes from the
    journal, and recovery never promotes it into meaning.
    """
    orphans: list[str] = []
    staged: list[str] = []
    idxs: list[str] = []
    if not root.exists():
        return (), (), ()

    for absolute in sorted(root.rglob("*")):
        if not absolute.is_file():
            continue
        relative = absolute.relative_to(root).as_posix()
        top = relative.split("/", maxsplit=1)[0]
        if top == STAGING_ROOT:
            staged.append(relative)
            continue
        if relative in state.committed_objects:
            continue
        if top == "indexes":
            idxs.append(relative)
            continue
        if top in _NON_PAYLOAD_TREES or relative == MANIFEST or absolute.name == "zarr.json":
            continue
        orphans.append(relative)

    if staged:
        _report(
            diagnostics,
            "staged_objects_present",
            SEVERITY_STALE,
            f"{len(staged)} staged or writer-tail objects are present; they are "
            "invisible to readers and are never promoted by recovery",
            STAGING_ROOT,
        )
    for relative in orphans:
        _report(
            diagnostics,
            "orphan_object",
            SEVERITY_STALE,
            f"no commit record backs this object, so it is invisible: {relative}",
            relative,
        )
    if idxs:
        _report(
            diagnostics,
            "index_cache_unrebuildable",
            SEVERITY_STALE,
            f"{len(idxs)} index files no commit record backs are present, but NRF v1 "
            "journals carry no index extent transitions, so committed state does not "
            "describe how to rebuild them; indexes never affect visibility",
            "indexes",
        )
    return tuple(orphans), tuple(staged), tuple(idxs)


def _diagnose_termination(state: CommittedState, diagnostics: list[Diagnostic]) -> None:
    termination = state.termination
    if termination is None:
        _report(
            diagnostics,
            "session_not_terminated",
            SEVERITY_INCOMPLETE,
            "the journal carries no termination record; the recording stopped without "
            "being finalized and everything committed so far is still valid",
            JOURNAL_TRANSACTIONS,
        )
        return
    if termination.kind != "normal":
        _report(
            diagnostics,
            "session_terminated_abnormally",
            SEVERITY_INCOMPLETE,
            f"the session terminated as {termination.kind!r}: {termination.reason}",
            JOURNAL_TRANSACTIONS,
        )


__all__ = [
    "REASON_BAD_RECORD_CHECKSUM",
    "REASON_MALFORMED_ORDER",
    "REASON_PARTIAL_LINE",
    "REASON_UNCOMMITTED_PREPARE",
    "SEVERITY_CORRUPT",
    "SEVERITY_INCOMPLETE",
    "SEVERITY_STALE",
    "Diagnostic",
    "IgnoredRecord",
    "SessionDiagnosis",
    "array_metadata_problem",
    "diagnose_session",
]
