#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Explicit recovery of an interrupted or damaged NRF v1 session.

Specification reference: ``README.md`` section 9.

Recovery is the deliberate counterpart to reading. Opening a session never
changes it, so a session that a reader refuses stays exactly as the crash left
it until someone asks for this. What that ask permits is narrow and worth
stating plainly:

* it **never** touches committed bytes -- not to truncate, rewrite, repair, or
  "fix" them. A missing or checksum-invalid committed object is corruption, and
  the report says so rather than papering over it;
* it **never** promotes an uncommitted tail, a staged object, or an orphan into
  visibility. Those became invisible by not having a commit line, and no amount
  of recovery gives them one;
* it rebuilds only caches -- the active ``zarr.json`` documents, ``manifest.json``,
  and ``journal/head.json`` -- from committed state, and only when they lag;
* it moves uncommitted material aside only when quarantine is explicitly
  requested, and moves it rather than deleting it, because that material is the
  evidence needed to diagnose what went wrong.

Every run produces a report conforming to ``recovery-report.schema.json``, and
:func:`recover` in dry-run mode produces the same report while writing nothing
at all -- including the report itself. The two share
:func:`~neurale.io.nrf._diagnostics.diagnose_session`, so a dry run cannot
promise something a real run would not do.

One deliberate strengthening of the specification's procedure is worth naming.
Section 9 selects the newest valid checkpoint and replays forward from it; this
implementation replays the journal from sequence 1 regardless and then checks
each checkpoint against the state replay reached at its own journal sequence.
The reported ``base_checkpoint_id`` is the newest checkpoint that survived that
check, so the outcome is the same when every checkpoint is sound -- and strictly
better when one is not, because a checkpoint that disagrees with the journal is
detected instead of being trusted as the starting state. The cost is time
proportional to the journal rather than to the tail after a checkpoint, which
is the right trade for an offline tool whose job is to be certain.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._canonical import canonical_json_bytes, sha256_hex, sign_record, verify_record_checksum
from ._diagnostics import SEVERITY_CORRUPT, SessionDiagnosis, _diagnose_objects
from ._errors import NrfCorruptionError, NrfStateError
from ._manifest import target_contracts
from ._paths import JOURNAL_HEAD, MANIFEST, require_canonical_relative_path
from ._schemas import (
    HEAD_SCHEMA,
    MANIFEST_SCHEMA,
    RECOVERY_REPORT_SCHEMA,
    validate_document,
)
from ._semantics import validate_manifest
from ._zarr import ArraySpec, array_metadata_bytes

#: Where a recovery report is written, per the required directory layout.
RECOVERY_ROOT = "recovery"

#: Where quarantined uncommitted material is moved, under the reserved
#: ``recovery/`` control namespace so it can never be mistaken for payload.
QUARANTINE_ROOT = "recovery/quarantine"

STATUS_RECOVERED = "recovered"
STATUS_PARTIALLY_RECOVERED = "partially_recovered"
STATUS_UNRECOVERABLE = "unrecoverable"


@dataclass(frozen=True)
class RecoveryResult:
    """What one recovery run found and what it did about it."""

    report: Mapping[str, Any]
    diagnosis: SessionDiagnosis
    status: str
    dry_run: bool
    #: Package containing the report. ``None`` for a dry run or an unrepairable
    #: package, where the report is returned only in memory.
    report_path: Path | None = None
    #: Caches rewritten from committed state, as session-relative paths.
    rebuilt_caches: tuple[str, ...] = ()

    @property
    def recovery_id(self) -> str:
        return str(self.report["recovery_id"])

    @property
    def committed_extents(self) -> Mapping[str, int]:
        return self.report["committed_extents"]

    @property
    def quarantined_paths(self) -> tuple[str, ...]:
        return tuple(self.report["quarantined_paths"])

    @property
    def complete(self) -> bool:
        """Whether the recovered session had terminated normally."""
        return self.diagnosis.complete


def recover(
    path: str | Path,
    *,
    dry_run: bool = False,
    quarantine: bool = False,
    rebuild_caches: bool = True,
    verify_checksums: bool = True,
    recovery_id: str | None = None,
    created_at: str | None = None,
) -> RecoveryResult:
    """Explicitly repair package caches, atomically replacing only a validated package.

    Ordinary reads never extract. Explicit repair uses a private workspace;
    committed payload corruption is reported without replacing the source.
    """
    from ._package import Package, publish_package

    root = Path(path)
    arguments = dict(
        quarantine=quarantine,
        rebuild_caches=rebuild_caches,
        verify_checksums=verify_checksums,
        recovery_id=recovery_id,
        created_at=created_at,
    )
    package = Package(root)
    original = root.stat()
    try:
        proposal = _recover_objects(package.root, dry_run=True, **arguments)
        proposal.diagnosis.root = root
        if dry_run:
            return proposal
        if not proposal.diagnosis.readable:
            report = _build_report(
                proposal.diagnosis,
                recovery_id=proposal.recovery_id,
                created_at=created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                status=proposal.status,
                quarantined=(),
                rebuilt_caches=(),
            )
            return RecoveryResult(report, proposal.diagnosis, proposal.status, False)
        with tempfile.TemporaryDirectory(prefix=f".{root.name}.repair-", dir=root.parent) as temp:
            workspace = Path(temp) / "objects"
            workspace.mkdir()
            for key in package.entries:
                destination = workspace / key
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (package.root / key).open("rb") as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
            result = _recover_objects(workspace, dry_run=False, **arguments)
            (workspace / RECOVERY_ROOT / "latest").write_text(
                f"{RECOVERY_ROOT}/{result.recovery_id}.json", encoding="utf-8"
            )
            repaired = Path(temp) / "repaired.nrf"
            publish_package(workspace, repaired, require_termination=False, include_staging=True)
            package.close()
            current = root.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (
                original.st_ino,
                original.st_size,
                original.st_mtime_ns,
            ):
                raise NrfStateError("NRF package changed during recovery")
            os.replace(repaired, root)
            result.diagnosis.root = root
            return replace(result, report_path=root)
    finally:
        package.close()


def _recover_objects(
    path: str | Path,
    *,
    dry_run: bool = False,
    quarantine: bool = False,
    rebuild_caches: bool = True,
    verify_checksums: bool = True,
    recovery_id: str | None = None,
    created_at: str | None = None,
) -> RecoveryResult:
    """Recover *path* and write a report, or describe what recovery would do.

    Parameters
    ----------
    dry_run
        Report the proposed actions and write nothing -- not the rebuilt
        caches, not the quarantine, not even the report file. The returned
        report is byte-for-value what a real run with the same arguments
        produces, apart from ``created_at`` and the checksum over it.
    quarantine
        Move staged objects, orphan final objects, and index caches into
        ``recovery/quarantine/<recovery-id>/``. Off by default: leaving them in
        place is harmless, because visibility comes from the journal, and they
        are evidence.
    rebuild_caches
        Restore active ``zarr.json`` documents and refresh ``manifest.json`` and
        ``journal/head.json`` when they lag replay.
    verify_checksums
        Re-hash every committed object. Turning this off keeps the existence and
        byte-length checks but can no longer prove a committed object's bytes
        are the ones that were committed.
    recovery_id
        Identifier for this run. Defaults to the next unused
        ``recovery-<n>``, which makes a dry run and the real run that follows it
        agree.
    created_at
        UTC timestamp for the report. Defaults to now.

    Returns
    -------
    RecoveryResult
        The report, the diagnosis behind it, and what was rebuilt.

    This helper operates on package entries for dry runs and on a private
    extraction workspace for explicit repair; it is not a public directory reader.
    """
    root = path
    diagnosis = _diagnose_objects(root, verify_checksums=verify_checksums)
    identifier = recovery_id or _next_recovery_id(root)
    require_canonical_relative_path(f"{RECOVERY_ROOT}/{identifier}.json", "recovery report path")

    status = _status(diagnosis)
    # A session whose manifest cannot be trusted has no frozen contract to
    # rebuild anything against, so recovery reports and stops rather than
    # guessing at a layout.
    planned_caches: tuple[str, ...] = ()
    if rebuild_caches and status != STATUS_UNRECOVERABLE:
        planned_caches = _planned_caches(diagnosis)
    planned_quarantine = _planned_quarantine(diagnosis) if quarantine else ()

    if not dry_run:
        specs = _array_specs(diagnosis)
        for relative in planned_caches:
            _rebuild_cache(root, relative, diagnosis, specs)
        if planned_quarantine:
            _quarantine(root, identifier, planned_quarantine)

    report = _build_report(
        diagnosis,
        recovery_id=identifier,
        created_at=created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        status=status,
        quarantined=planned_quarantine,
        rebuilt_caches=planned_caches,
    )

    report_path: Path | None = None
    if not dry_run:
        report_path = root / RECOVERY_ROOT / f"{identifier}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes(report))

    return RecoveryResult(
        report=report,
        diagnosis=diagnosis,
        status=status,
        dry_run=dry_run,
        report_path=report_path,
        rebuilt_caches=planned_caches,
    )


def _status(diagnosis: SessionDiagnosis) -> str:
    """Classify how much of the session survived.

    ``unrecoverable`` is reserved for a session whose manifest cannot be used:
    without the frozen registries there is no contract to validate anything
    against. Damaged committed objects are bad but bounded -- the rest of the
    session is still exactly what it always was -- so that is
    ``partially_recovered``, not a total loss.
    """
    if diagnosis.manifest is None or any(
        item.code
        in {
            "manifest_missing",
            "manifest_unreadable",
            "manifest_schema_invalid",
            "manifest_semantics_invalid",
            "unsupported_version",
        }
        for item in diagnosis.diagnostics
    ):
        return STATUS_UNRECOVERABLE
    if diagnosis.of_severity(SEVERITY_CORRUPT):
        return STATUS_PARTIALLY_RECOVERED
    return STATUS_RECOVERED


def _next_recovery_id(root: Path) -> str:
    directory = root / RECOVERY_ROOT
    existing = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
    return f"recovery-{existing + 1:04d}"


# --- cache rebuilding ------------------------------------------------------


def _planned_caches(diagnosis: SessionDiagnosis) -> tuple[str, ...]:
    """Return the cache files recovery would rewrite, in a stable order."""
    planned = [f"{arr}/zarr.json" for arr in diagnosis.stale_metadata_arrays]
    planned.extend(name for name in diagnosis.lagging_caches if name in {MANIFEST, JOURNAL_HEAD})
    return tuple(dict.fromkeys(planned))


def _array_specs(diagnosis: SessionDiagnosis) -> dict[str, ArraySpec]:
    """Return every physical array the frozen manifest declares, keyed by path."""
    if diagnosis.manifest is None:
        return {}
    return {
        spec.path: spec
        for contract in target_contracts(diagnosis.manifest).values()
        for spec in contract.arrays
    }


def _rebuild_cache(
    root: Path, relative: str, diagnosis: SessionDiagnosis, specs: Mapping[str, ArraySpec]
) -> None:
    if relative in {MANIFEST, JOURNAL_HEAD}:
        _rewrite_state_cache(root, relative, diagnosis)
        return
    spec = specs.get(relative.removesuffix("/zarr.json"))
    if spec is None:  # pragma: no cover - the plan only names manifest arrays
        return
    _write_atomically(root / relative, _metadata_bytes(root, diagnosis, spec))


def _metadata_bytes(root: Path, diagnosis: SessionDiagnosis, spec: ArraySpec) -> bytes:
    """Return the bytes to restore one array's active metadata cache from.

    The specification restores active metadata from the last *committed*
    metadata snapshot, so that a crash before a commit cannot make an
    uncommitted shape authoritative. When that snapshot is itself missing or
    damaged the manifest still froze the array's shape, chunk shape, and dtype
    before the first append, so rebuilding from the manifest restores the same
    contract without inventing anything.
    """
    snapshot = _last_metadata_snapshot(diagnosis, spec.path)
    if snapshot is not None:
        absolute = root / snapshot["path"]
        if absolute.exists():
            raw = absolute.read_bytes()
            if sha256_hex(raw) == snapshot["sha256"]:
                return raw
    return array_metadata_bytes(spec)


def _last_metadata_snapshot(
    diagnosis: SessionDiagnosis, array_path: str
) -> Mapping[str, Any] | None:
    """Return the newest committed metadata snapshot for one physical array.

    Snapshot paths embed their transaction ID, which is a zero-padded decimal
    sequence, so the lexicographically greatest path is the newest snapshot.
    """
    candidates = [
        entry
        for entry in diagnosis.state.committed_objects.values()
        if entry.get("logical_role") == "metadata_snapshot"
        and entry.get("array_path") == array_path
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry["path"])


def _rewrite_state_cache(root: Path, relative: str, diagnosis: SessionDiagnosis) -> None:
    """Refresh one non-authoritative committed-state cache from replay.

    Neither cache decides visibility, so this only makes a lagging file agree
    with the journal again. The registries are frozen and are copied through
    unchanged: recovery re-serializes the descriptors the session already had,
    it does not re-derive them.
    """
    manifest = diagnosis.manifest
    if manifest is None:  # pragma: no cover - unrecoverable sessions skip this
        return
    state = diagnosis.state
    if relative == JOURNAL_HEAD:
        # The rebuilt head names the checkpoint recovery could actually
        # validate, not the one the journal last mentioned: a reader must not
        # follow a ``last_checkpoint_id`` that resolves to nothing usable.
        head = {
            "format": "nrf-head",
            "version": {"major": 1, "minor": 0},
            "journal_sequence": state.journal_sequence,
            "last_transaction_id": state.last_transaction_id,
            "last_checkpoint_id": diagnosis.base_checkpoint_id,
            "committed_extents": dict(state.committed_extents),
            "sealed_targets": sorted(state.sealed_targets),
        }
        validate_document(head, HEAD_SCHEMA)
        _write_atomically(root / relative, canonical_json_bytes(head))
        return

    refreshed = dict(manifest)
    refreshed["streams"] = [
        dict(stream, committed_extent=state.extent(stream["data"]["path"]))
        for stream in manifest["streams"]
    ]
    refreshed["record_schemas"] = [
        dict(schema, committed_extent=state.extent(schema["path"]))
        for schema in manifest["record_schemas"]
    ]
    refreshed["commit"] = {
        "journal_sequence": state.journal_sequence,
        "last_transaction_id": state.last_transaction_id,
        "last_checkpoint_id": diagnosis.base_checkpoint_id,
        "committed_extents": dict(state.committed_extents),
        "sealed_targets": sorted(state.sealed_targets),
    }
    # Both mandatory layers, in the required order, before a manifest is
    # written: a refreshed cache that no longer validates would be worse than
    # the lagging one it replaced.
    validate_document(refreshed, MANIFEST_SCHEMA)
    validate_manifest(refreshed)
    _write_atomically(root / relative, canonical_json_bytes(refreshed))


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.recovery-tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# --- quarantine ------------------------------------------------------------


def _planned_quarantine(diagnosis: SessionDiagnosis) -> tuple[str, ...]:
    """Return the uncommitted material quarantine would move, in stable order.

    Only material no commit record backs is eligible. Committed objects are
    never listed here, damaged or not: quarantining one would remove evidence
    of the corruption and change what the session claims to hold.
    """
    return tuple(
        sorted({*diagnosis.staged_paths, *diagnosis.orphan_objects, *diagnosis.index_cache_paths})
    )


def _quarantine(root: Path, recovery_id: str, paths: tuple[str, ...]) -> None:
    """Move each path under ``recovery/quarantine/<recovery-id>/``.

    Moved, never deleted: the point of quarantine is to get uncommitted
    material out of the session tree while keeping the evidence needed to work
    out how it got there.
    """
    destination_root = root / QUARANTINE_ROOT / recovery_id
    emptied: set[Path] = set()
    for relative in paths:
        source = root / relative
        if not source.exists():  # pragma: no cover - raced with another process
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        emptied.add(source.parent)
    _prune_empty_directories(root, emptied)


def _prune_empty_directories(root: Path, candidates: set[Path]) -> None:
    """Remove only the directories this quarantine emptied, deepest first.

    Restricted to the parents of what was actually moved, and to directories
    that are empty, so a directory that was already empty for its own reasons
    is left alone and no file can ever be discarded.
    """
    quarantine = root / RECOVERY_ROOT
    pending = sorted(candidates, key=lambda item: len(item.parts), reverse=True)
    while pending:
        directory = pending.pop(0)
        if directory == root or not directory.is_relative_to(root):
            continue
        if directory.is_relative_to(quarantine) or not directory.is_dir():
            continue
        if any(directory.iterdir()):
            continue
        directory.rmdir()
        pending.append(directory.parent)
        pending.sort(key=lambda item: len(item.parts), reverse=True)


# --- report ----------------------------------------------------------------


def _build_report(
    diagnosis: SessionDiagnosis,
    *,
    recovery_id: str,
    created_at: str,
    status: str,
    quarantined: tuple[str, ...],
    rebuilt_caches: tuple[str, ...],
) -> dict[str, Any]:
    """Assemble and sign the report, then validate it against its schema.

    ``rebuilt_indexes`` stays empty by construction. An NRF v1 journal carries
    no index extent transitions, so committed state never describes how an
    index was built and recovery has nothing from which to rebuild one; saying
    so in the extensions is honest, listing an index it did not rebuild is not.
    """
    state = diagnosis.state
    report = sign_record(
        {
            "format": "nrf-recovery-report",
            "version": {"major": 1, "minor": 0},
            "recovery_id": recovery_id,
            "created_at": created_at,
            "base_checkpoint_id": diagnosis.base_checkpoint_id,
            "last_valid_journal_sequence": state.journal_sequence,
            "last_committed_transaction_id": state.last_transaction_id,
            "committed_extents": dict(state.committed_extents),
            "sealed_targets": sorted(state.sealed_targets),
            "retained_transaction_ids": list(diagnosis.retained_transaction_ids),
            "ignored_transaction_ids": list(diagnosis.ignored_transaction_ids),
            "ignored_records": [item.as_json() for item in diagnosis.ignored_records],
            "missing_objects": list(diagnosis.missing_objects),
            "corrupt_objects": list(diagnosis.corrupt_objects),
            "quarantined_paths": list(quarantined),
            "rebuilt_indexes": [],
            # Recovery is read-only with respect to committed logical data. This
            # is a constant in the schema, and it is a constant here for the
            # same reason: no path through this module writes a committed byte.
            "committed_data_modified": False,
            # A dry run and the real run that applies it produce byte-identical
            # reports, so nothing here records which one wrote it.
            "extensions": {
                "neurale": {
                    "status": status,
                    "complete": diagnosis.complete,
                    "readable": diagnosis.readable,
                    "rebuilt_caches": list(rebuilt_caches),
                    "diagnostics": [item.as_json() for item in diagnosis.diagnostics],
                }
            },
        }
    )
    validate_document(report, RECOVERY_REPORT_SCHEMA)
    return report


def read_report(path: str | Path) -> dict[str, Any]:
    """Load and validate the latest recovery report inside a package."""
    from ._package import Package

    package = Package(Path(path))
    try:
        pointer = package.root / RECOVERY_ROOT / "latest"
        if not pointer.is_file():
            raise NrfStateError("NRF package contains no recovery report")
        key = pointer.read_text()
        require_canonical_relative_path(key, "latest recovery report")
        if not key.startswith("recovery/") or key.count("/") != 1 or not key.endswith(".json"):
            raise NrfCorruptionError("invalid latest recovery report pointer")
        document = json.loads((package.root / key).read_bytes().decode("utf-8"))
    finally:
        package.close()
    validate_document(document, RECOVERY_REPORT_SCHEMA)
    if not verify_record_checksum(document):
        raise NrfCorruptionError(f"{path} fails its own record checksum")
    return document


__all__ = [
    "QUARANTINE_ROOT",
    "RECOVERY_ROOT",
    "STATUS_PARTIALLY_RECOVERED",
    "STATUS_RECOVERED",
    "STATUS_UNRECOVERABLE",
    "RecoveryResult",
    "read_report",
    "recover",
]
