#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The NRF v1 offline writer.

Specification reference: ``README.md`` section 8.

The append protocol is the whole point of this module, and its order is
normative:

1. stage every new immutable chunk and metadata snapshot under
   ``.staging/<transaction-id>/`` at a final path that has never existed;
2. flush and checksum the exact object bytes;
3. append a checksum-valid ``prepare`` record naming every staged object and
   the extent transition it backs;
4. promote each staged object with a create-if-absent rename, then append the
   ``commit`` record -- that complete line is the single visibility point;
5. refresh the ``zarr.json``, ``manifest.json``, and ``head.json`` caches.

Step 5 can fail without undoing the commit: the caches are allowed to lag.
Appends smaller than a chunk stay in a writer-owned tail, never reach the
journal, and are invisible to readers, so a crash can lose the tail but can
never alter committed bytes.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from ._canonical import canonical_json_bytes, sha256_hex, sign_record
from ._errors import NrfSemanticError, NrfStateError
from ._journal import (
    append_records,
    build_checkpoint,
    build_commit,
    build_prepare,
    build_termination,
    extent_transition,
    prepared_object,
)
from ._manifest import (
    ManifestBuilder,
    TargetContract,
    descriptor_by_id,
    replay_contracts,
    target_contracts,
)
from ._paths import (
    JOURNAL_CHECKPOINTS,
    JOURNAL_HEAD,
    JOURNAL_TRANSACTIONS,
    MANIFEST,
    STAGING_ROOT,
    chunk_path,
    metadata_snapshot_path,
)
from ._paths import (
    checkpoint_id as make_checkpoint_id,
)
from ._paths import (
    transaction_id as make_transaction_id,
)
from ._replay import CommittedState, checkpoint_object, replay
from ._schemas import CHECKPOINT_SCHEMA, HEAD_SCHEMA, validate_document
from ._zarr import (
    NUMPY_DTYPES,
    ArraySpec,
    array_metadata_bytes,
    encode_chunk,
    pad_to_chunk,
    write_array_metadata,
)

_SESSION_TERMINATION_KIND = "session_termination"


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace *path* with *data* through a single rename.

    Every cache update in the protocol is one atomic rename so a crash leaves
    the previous complete file rather than a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    # Written and flushed through one writable handle. Reopening read-only to
    # sync is what Windows rejects outright -- ``os.fsync`` there is
    # ``FlushFileBuffers``, which needs write access and returns EBADF without
    # it -- and even where it is tolerated it syncs a descriptor that never saw
    # the data.
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _promote(staged: Path, final: Path) -> None:
    """Rename a staged object to its final path, refusing to replace anything.

    NRF v1 has no replace or supersede disposition, so an existing final object
    is a bug or a repeated promotion, never something to overwrite.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise NrfStateError(f"final object already exists and must not be replaced: {final}")
    os.rename(staged, final)


class _Tail:
    """Writer-owned buffer for rows that do not yet fill a chunk."""

    __slots__ = ("columns", "rows")

    def __init__(self) -> None:
        self.columns: dict[str, list[Any]] = {}
        self.rows = 0

    def extend(self, values: Mapping[str, Sequence[Any]], count: int) -> None:
        for name, column in values.items():
            self.columns.setdefault(name, []).extend(column)
        self.rows += count

    def take(self, count: int) -> dict[str, list[Any]]:
        taken = {name: column[:count] for name, column in self.columns.items()}
        self.columns = {name: column[count:] for name, column in self.columns.items()}
        self.rows -= count
        return taken


class NrfWriter:
    """Build and publish one immutable ``<session>.nrf`` package.

    There is exactly one logical writer per live session. Use it as a context
    manager: leaving the block normally finalizes the session. An exception
    leaves the private workspace unpublished; it does not claim completion.
    Call ``abort()`` explicitly to publish an abnormal termination.
    """

    def __init__(self, root: Path, builder: ManifestBuilder) -> None:
        self._target = Path(root)
        self._root = Path(
            tempfile.mkdtemp(prefix=f".{root.name}.", suffix=".working", dir=root.parent)
        )
        self._published = False
        self.cleanup_paths: tuple[Path, ...] = ()
        self._progress_callback = None
        self._payload_archive = None
        self._builder = builder
        self._state = CommittedState()
        self._contracts: dict[str, TargetContract] = {}
        self._manifest: dict[str, Any] = {}
        self._tails: dict[str, _Tail] = {}
        self._primary_keys: dict[str, set[Any]] = {}
        self._sequence = 0
        self._transactions = 0
        self._checkpoints = 0
        self._closed = False
        self._frozen = False

    # --- construction -----------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        session_id: str,
        created_at: str,
        writer_name: str = "pyneurale",
        writer_version: str = "1",
        session_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extensions: Mapping[str, Any] | None = None,
        minor_version: int = 0,
    ) -> NrfWriter:
        """Create a private workspace; publish the target only at finalization.

        *extensions* and *minor_version* travel together: an extension is only
        legible against the minor version that defines it, and a session
        declaring one at minor 0 would be asking a v1.0 reader to interpret
        something v1.0 does not define.
        """
        root = Path(path)
        if root.exists():
            raise NrfStateError(f"{root} already exists")
        root.parent.mkdir(parents=True, exist_ok=True)
        builder = ManifestBuilder(
            session_id=session_id,
            created_at=created_at,
            writer_name=writer_name,
            writer_version=writer_version,
            session_metadata=dict(session_metadata or {}),
            metadata=dict(metadata or {}),
            extensions=dict(extensions or {}),
            minor_version=minor_version,
        )
        return cls(root, builder)

    @property
    def root(self) -> Path:
        return self._target

    @property
    def registry(self) -> ManifestBuilder:
        """The registry builder, usable only before :meth:`freeze`."""
        return self._builder

    @property
    def manifest(self) -> Mapping[str, Any]:
        self._require_frozen()
        return self._manifest

    @property
    def committed_state(self) -> CommittedState:
        return self._state

    @property
    def terminated(self) -> bool:
        return self._state.terminated

    # --- lifecycle --------------------------------------------------------

    def _require_frozen(self) -> None:
        if not self._frozen:
            raise NrfStateError("the session registries have not been frozen yet")

    def _require_open(self) -> None:
        if self._closed:
            raise NrfStateError("the session is closed")
        if self._state.terminated:
            raise NrfStateError("the session has terminated and cannot be appended to")

    def freeze(self) -> None:
        """Validate and write the manifest, then close the registries.

        Both mandatory validation layers run here. After this point no registry
        entry may be added, which is what guarantees a reader always has a
        descriptor for anything that was ever committed.
        """
        if self._frozen:
            raise NrfStateError("the session registries are already frozen")
        manifest = self._builder.build()
        self._builder.validate(manifest)
        self._manifest = manifest
        self._contracts = target_contracts(manifest)
        self._builder.frozen = True
        self._frozen = True

        (self._root / "zarr.json").write_bytes(
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode("utf-8")
        )
        for contract in self._contracts.values():
            for spec in contract.arrays:
                write_array_metadata(self._root, spec)
        self._write_caches()

    def _write_caches(self) -> None:
        manifest = self._builder.build(
            committed_extents=self._state.committed_extents,
            sealed_targets=sorted(self._state.sealed_targets),
            last_transaction_id=self._state.last_transaction_id,
            last_checkpoint_id=self._state.last_checkpoint_id,
            journal_sequence=self._state.journal_sequence,
        )
        self._manifest = manifest
        _atomic_write(self._root / MANIFEST, canonical_json_bytes(manifest))
        # head.json is a closed vocabulary: it carries no session ID, because
        # the manifest beside it is the only place session identity lives.
        # Validating before the rename keeps a malformed cache from ever
        # reaching disk, where a reader would have to decide what to do with it.
        head = {
            "format": "nrf-head",
            "version": {"major": 1, "minor": 0},
            "journal_sequence": self._state.journal_sequence,
            "last_transaction_id": self._state.last_transaction_id,
            "last_checkpoint_id": self._state.last_checkpoint_id,
            "committed_extents": dict(self._state.committed_extents),
            "sealed_targets": sorted(self._state.sealed_targets),
        }
        validate_document(head, HEAD_SCHEMA)
        _atomic_write(self._root / JOURNAL_HEAD, canonical_json_bytes(head))
        for descriptor in manifest["feature_sets"]:
            _atomic_write(
                self._root / "feature_sets" / f"{descriptor['id']}.json",
                canonical_json_bytes(descriptor),
            )

    # --- appending --------------------------------------------------------

    def append_stream(
        self,
        stream_id: str,
        values: np.ndarray,
        *,
        timestamps: np.ndarray | None = None,
    ) -> None:
        """Append sampled, behavioral, or feature rows to one stream.

        Every check runs before either tail is touched, so a rejected call
        leaves the writer exactly as it was. Appending to the data tail first
        and validating the timestamps afterwards would leave the two tails at
        different lengths, and a later successful retry would then commit
        duplicated data with mismatched timestamps.
        """
        self._require_frozen()
        self._require_open()
        stream = self._stream(stream_id)
        data_path = stream["data"]["path"]
        timing = stream["timing"]

        arr = np.asarray(values)
        if arr.ndim != len(stream["axes"]):
            raise NrfSemanticError(
                f"{stream_id} payload rank {arr.ndim} does not match its declared axes"
            )
        width = stream["data"]["shape"][-1]
        if arr.shape[-1] != width:
            raise NrfSemanticError(
                f"{stream_id} payload width {arr.shape[-1]} does not match the declared {width}"
            )
        _require_lossless_dtype(arr, stream["dtype"], f"{stream_id} payload")

        stamps: np.ndarray | None = None
        if timing["mode"] == "explicit":
            if timestamps is None:
                raise NrfSemanticError(f"{stream_id} uses explicit timing and requires timestamps")
            candidate = np.asarray(timestamps)
            _require_lossless_dtype(candidate, "int64", f"{stream_id} timestamps")
            stamps = candidate.astype("int64", copy=False).reshape(-1)
            if stamps.shape[0] != arr.shape[0]:
                raise NrfSemanticError(
                    f"{stream_id} needs one timestamp per item, got {stamps.shape[0]} "
                    f"for {arr.shape[0]} items"
                )
        elif timestamps is not None:
            raise NrfSemanticError(f"{stream_id} uses regular timing and takes no timestamps")

        self._require_capacity(data_path, arr.shape[0])
        if stamps is not None:
            self._require_capacity(timing["timestamps"]["path"], stamps.shape[0])

        # Every check has passed: now mutate.
        self._tail(data_path).extend({"values": list(arr)}, arr.shape[0])
        if stamps is not None:
            self._tail(timing["timestamps"]["path"]).extend(
                {"values": list(stamps)}, stamps.shape[0]
            )

    def _require_capacity(self, target_path: str, additional: int) -> None:
        """Reject an append that would exceed the frozen Zarr capacity.

        A Zarr shape is fixed when the registry freezes, so an extent beyond it
        would produce a manifest this package's own semantic validator rejects.
        Failing here keeps that state unreachable rather than detectable.
        """
        contract = self._contracts.get(target_path)
        if contract is None or contract.capacity <= 0:
            return
        committed = self._state.extent(target_path)
        pending = self._pending(target_path)
        total = committed + pending + additional
        if total > contract.capacity:
            raise NrfSemanticError(
                f"{target_path} would reach {total} items, exceeding its declared capacity "
                f"of {contract.capacity} (committed {committed}, pending {pending}, "
                f"appending {additional})"
            )

    def append_records(self, record_schema_id: str, rows: Mapping[str, Sequence[Any]]) -> None:
        """Append rows to one record set.

        Every declared field must be supplied. A nullable field accepts ``None``
        for a missing value; the validity array carries the distinction, never a
        dtype-specific sentinel.
        """
        self._require_frozen()
        self._require_open()
        schema = self._record_schema(record_schema_id)
        names = [item["name"] for item in schema["fields"]]
        missing = set(names) - set(rows)
        if missing:
            raise NrfSemanticError(f"{record_schema_id} append is missing fields {sorted(missing)}")
        unknown = set(rows) - set(names)
        if unknown:
            raise NrfSemanticError(
                f"{record_schema_id} append has unknown fields {sorted(unknown)}"
            )
        counts = {len(list(column)) for column in rows.values()}
        if len(counts) != 1:
            raise NrfSemanticError(f"{record_schema_id} columns have different lengths")
        count = counts.pop()
        if count == 0:
            return

        primary = list(rows[schema["primary_key"]])
        if any(value is None for value in primary):
            raise NrfSemanticError(f"{record_schema_id} primary key must not be null")
        if len(set(primary)) != len(primary):
            raise NrfSemanticError(f"{record_schema_id} primary key values must be unique")
        # A primary key identifies a row for the life of the session, not just
        # within one call, so uniqueness is tracked across every append this
        # writer has accepted -- committed rows and the pending tail alike.
        seen = self._primary_keys.setdefault(record_schema_id, set())
        repeated = sorted(seen.intersection(primary))
        if repeated:
            raise NrfSemanticError(
                f"{record_schema_id} primary key values were already used in this session: "
                f"{repeated}"
            )

        self._require_capacity(schema["path"], count)

        # Every check has passed: now mutate.
        seen.update(primary)
        self._tail(schema["path"]).extend({name: list(rows[name]) for name in names}, count)

    def _tail(self, target_path: str) -> _Tail:
        return self._tails.setdefault(target_path, _Tail())

    def _pending(self, target_path: str) -> int:
        """Return the rows buffered for *target_path* but not yet committed."""
        tail = self._tails.get(target_path)
        return tail.rows if tail is not None else 0

    def _replay_contracts(self) -> dict[str, dict[str, Any]]:
        return replay_contracts(self._contracts)

    def _stream(self, stream_id: str) -> Mapping[str, Any]:
        return descriptor_by_id(self._manifest["streams"], stream_id, "stream")

    def _record_schema(self, schema_id: str) -> Mapping[str, Any]:
        return descriptor_by_id(self._manifest["record_schemas"], schema_id, "record schema")

    # --- transactions -----------------------------------------------------

    def commit(self) -> str | None:
        """Commit every target whose tail holds at least one full chunk.

        Returns the transaction ID, or ``None`` when nothing was ready. A short
        tail stays invisible until :meth:`finalize` seals it.
        """
        self._require_frozen()
        self._require_open()
        return self._run_transaction(seal=False)

    def _plan_transaction(self, *, seal: bool) -> list[tuple[TargetContract, int, int, bool]]:
        """Decide what each target advances to in the next transaction.

        A stream's timestamp target is never planned independently: it advances
        by exactly the amount its data target does, and seals when the data
        seals. Two independent decisions could otherwise leave the timestamp
        extent behind the data extent, which the format forbids and this
        package's own semantic validator rejects.
        """
        plan: list[tuple[TargetContract, int, int, bool]] = []
        for target_path, contract in self._contracts.items():
            if contract.follows_linked_target:
                continue
            if target_path in self._state.sealed_targets:
                continue
            before = self._state.extent(target_path)
            pending = self._pending(target_path)
            if seal:
                after = before + pending
            else:
                whole = (pending // contract.chunk_length) * contract.chunk_length
                if not whole:
                    continue
                after = before + whole
            plan.append((contract, before, after, seal))

            linked = contract.linked_target_path
            if linked is None:
                continue
            follower = self._contracts[linked]
            follower_before = self._state.extent(linked)
            advance = after - before
            follower_pending = self._pending(linked)
            if follower_pending < advance:
                raise NrfStateError(
                    f"{linked} holds {follower_pending} pending items but {target_path} is "
                    f"advancing by {advance}; explicit timestamps must cover every item"
                )
            plan.append((follower, follower_before, follower_before + advance, seal))
        return plan

    def _run_transaction(self, *, seal: bool) -> str | None:
        plan = self._plan_transaction(seal=seal)
        if not plan:
            return None

        self._transactions += 1
        transaction_id = make_transaction_id(self._transactions)
        staging = self._root / STAGING_ROOT / transaction_id

        objects: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        staged_files: list[tuple[Path, Path]] = []

        for contract, before, after, seals in plan:
            object_paths: list[str] = []
            if after > before:
                taken = self._tails[contract.target_path].take(after - before)
                for spec in contract.arrays:
                    column = _column_for(contract, spec, taken)
                    for coordinate in spec.chunk_coordinates(before, after):
                        payload = _chunk_payload(spec, column, before, after, coordinate)
                        raw = encode_chunk(spec, payload)
                        final = chunk_path(spec.path, coordinate)
                        staged = staging / final
                        if self._payload_archive is None:
                            staged.parent.mkdir(parents=True, exist_ok=True)
                            staged.write_bytes(raw)
                            staged_files.append((staged, self._root / final))
                        else:
                            self._payload_archive.write(final, raw)
                        objects.append(
                            prepared_object(
                                path=final,
                                transaction_id=transaction_id,
                                target_path=contract.target_path,
                                array_path=spec.path,
                                logical_role=_chunk_role(contract, spec),
                                content_kind="zarr_chunk",
                                sha256=sha256_hex(raw),
                                byte_length=len(raw),
                                chunk_coordinate=coordinate,
                                record_schema_id=contract.record_schema_id,
                                column_name=_column_name(contract, spec),
                            )
                        )
                        object_paths.append(final)
                    snapshot = array_metadata_bytes(spec)
                    snapshot_path = metadata_snapshot_path(
                        transaction_id, spec.path.replace("/", "-")
                    )
                    staged_snapshot = staging / snapshot_path
                    if self._payload_archive is None:
                        staged_snapshot.parent.mkdir(parents=True, exist_ok=True)
                        staged_snapshot.write_bytes(snapshot)
                        staged_files.append((staged_snapshot, self._root / snapshot_path))
                    else:
                        self._payload_archive.write(snapshot_path, snapshot)
                    objects.append(
                        prepared_object(
                            path=snapshot_path,
                            transaction_id=transaction_id,
                            target_path=contract.target_path,
                            array_path=spec.path,
                            logical_role="metadata_snapshot",
                            content_kind="zarr_metadata",
                            sha256=sha256_hex(snapshot),
                            byte_length=len(snapshot),
                        )
                    )
                    object_paths.append(snapshot_path)

            transitions.append(
                extent_transition(
                    target_path=contract.target_path,
                    target_kind=contract.target_kind,
                    before=before,
                    after=after,
                    chunk_length=contract.chunk_length,
                    required_array_paths=contract.required_array_paths,
                    object_paths=object_paths,
                    seals_target=seals,
                    record_schema_id=contract.record_schema_id,
                )
            )

        self._sequence += 1
        prepare = build_prepare(
            sequence=self._sequence,
            transaction_id=transaction_id,
            previous_committed_transaction_id=self._state.last_transaction_id,
            extents=transitions,
            objects=objects,
        )
        journal = self._root / JOURNAL_TRANSACTIONS
        append_records(journal, [prepare])

        for staged, final in staged_files:
            _promote(staged, final)

        self._sequence += 1
        commit = build_commit(
            sequence=self._sequence,
            transaction_id=transaction_id,
            prepare_sequence=prepare["sequence"],
        )
        append_records(journal, [commit])

        self._state = replay(
            [prepare, commit], contracts=self._replay_contracts(), initial=self._state
        )
        for contract, _, _, _ in plan:
            for spec in contract.arrays:
                write_array_metadata(self._root, spec)
        self._write_caches()
        return transaction_id

    def checkpoint(self) -> str:
        """Write a checkpoint and record it in the journal.

        A checkpoint is where recovery starts instead of journal sequence 1, so
        it must preserve the *complete* extent/object ownership graph: each
        committed object keeps its byte length, content kind, chunk coordinate,
        and record-column association. A truncated descriptor would leave
        recovery unable to re-attribute a chunk to its target.
        """
        self._require_frozen()
        self._require_open()
        self._checkpoints += 1
        checkpoint = make_checkpoint_id(self._checkpoints)
        relative = f"{JOURNAL_CHECKPOINTS}/{checkpoint}.json"
        document = sign_record(
            {
                "format": "nrf-checkpoint",
                "version": {"major": 1, "minor": 0},
                "checkpoint_id": checkpoint,
                "journal_sequence": self._state.journal_sequence,
                "last_committed_transaction_id": self._state.last_transaction_id,
                "committed_extents": dict(self._state.committed_extents),
                "sealed_targets": sorted(self._state.sealed_targets),
                "committed_objects": [
                    checkpoint_object(entry) for entry in self._state.committed_objects.values()
                ],
                # The v1 journal carries no index-extent transitions, so replay
                # can never advance an index build position; a checkpoint must
                # agree with that empty replay state rather than invent one.
                "index_extents": {},
            }
        )
        validate_document(document, CHECKPOINT_SCHEMA)
        raw = canonical_json_bytes(document)
        _atomic_write(self._root / relative, raw)
        self._sequence += 1
        record = build_checkpoint(
            sequence=self._sequence,
            checkpoint_id=checkpoint,
            checkpoint_path=relative,
            checkpoint_sha256=sha256_hex(raw),
        )
        append_records(self._root / JOURNAL_TRANSACTIONS, [record])
        self._state.journal_sequence = record["sequence"]
        self._state.last_checkpoint_id = checkpoint
        self._write_caches()
        return checkpoint

    def finalize(
        self,
        *,
        kind: str = "normal",
        reason: str = "completed",
        time_ns: int = 0,
        fault_id: str | None = None,
        termination_id: str = "termination-0001",
        extensions: Mapping[str, Any] | None = None,
    ) -> None:
        """Seal every open target, append the termination row, and terminate.

        *extensions* is written into the journal ``termination`` record's
        optional ``extensions`` member and is covered by that record's checksum,
        so a minor extension's termination provenance is signed rather than
        appended unchecked. The standard ``session_termination`` typed row is
        unchanged: NRF v1's closed ``termination_kind`` enum carries the
        *effective* outcome, and the dimensions a closed enum cannot carry
        travel in the extension.

        A normal termination must seal *every* appendable target, including
        those that never received a row, so replay can prove the session ended
        deliberately rather than stopped.
        """
        self._require_frozen()
        if self._state.terminated:
            self._publish_package()
            return
        self._require_open()

        termination = next(
            schema
            for schema in self._manifest["record_schemas"]
            if schema["kind"] == _SESSION_TERMINATION_KIND
        )
        transaction_number = self._transactions + 1
        last_transaction = make_transaction_id(transaction_number)
        self.append_records(
            termination["id"],
            {
                "termination_id": [termination_id],
                "time_ns": [time_ns],
                "termination_kind": [kind],
                "reason": [reason],
                "fault_id": [fault_id],
                "last_transaction_id": [last_transaction],
            },
        )
        committed = self._run_transaction(seal=True)
        if committed != last_transaction:  # pragma: no cover - defensive
            raise NrfStateError("the final transaction did not receive the reserved identifier")

        self._sequence += 1
        record = build_termination(
            sequence=self._sequence,
            termination_record_id=termination_id,
            termination_kind=kind,
            time_ns=time_ns,
            reason=reason,
            fault_id=fault_id,
            last_transaction_id=last_transaction,
            extensions=extensions,
        )
        append_records(self._root / JOURNAL_TRANSACTIONS, [record])
        self._state = replay([record], contracts=self._replay_contracts(), initial=self._state)
        self._write_caches()
        self._publish_package()

    def abort(self, *, reason: str = "aborted", fault_id: str | None = None) -> None:
        """End the session abnormally, leaving unrelated targets open."""
        self.finalize(kind="aborted", reason=reason, fault_id=fault_id)

    def close(self) -> None:
        """Close without publishing or retrying failed finalization. Idempotent."""
        if self._payload_archive is not None:
            self._payload_archive.close()
        self._closed = True

    def _publish_package(self) -> None:
        if self._state.terminated and not self._published:
            from ._package import publish_package

            residual = publish_package(
                self._root,
                self._target,
                self._progress_callback,
                payload_archive=self._payload_archive,
            )
            self._published = True
            self.cleanup_paths = () if residual is None else (residual,)
            try:
                shutil.rmtree(self._root)
            except OSError:
                self.cleanup_paths += (self._root,)

    def __enter__(self) -> NrfWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_type is None and self._frozen and not self._state.terminated:
            self.finalize()
        self.close()
        return False


def _is_validity(spec: ArraySpec) -> bool:
    return "/validity/" in spec.path


def _chunk_role(contract: TargetContract, spec: ArraySpec) -> str:
    if contract.target_kind == "array":
        return "array_chunk"
    return "record_validity_chunk" if _is_validity(spec) else "record_column_chunk"


def _column_name(contract: TargetContract, spec: ArraySpec) -> str | None:
    if contract.target_kind == "array":
        return None
    return spec.path.rsplit("/", maxsplit=1)[-1]


def _column_for(
    contract: TargetContract, spec: ArraySpec, taken: Mapping[str, list[Any]]
) -> list[Any]:
    name = _column_name(contract, spec)
    if name is None:
        return taken["values"]
    values = taken[name]
    if _is_validity(spec):
        return [0 if value is None else 1 for value in values]
    return values


def _chunk_payload(
    spec: ArraySpec,
    column: Sequence[Any],
    before: int,
    after: int,
    coordinate: Sequence[int],
) -> np.ndarray:
    """Materialize one chunk's payload, padded to the full chunk shape."""
    start = coordinate[0] * spec.chunk_length
    stop = min(start + spec.chunk_length, after)
    lower = max(start, before)
    rows = column[lower - before : stop - before]
    if spec.is_text:
        values = np.array(["" if item is None else str(item) for item in rows], dtype=object)
    else:
        values = np.array(
            [_default_for(spec) if item is None else item for item in rows],
            dtype=spec.numpy_dtype,
        )
    # A flat column for a multi-dimensional array, or a stack whose leading axis
    # did not survive ``np.array``, has to be restored to the chunk's layout.
    flattened = values.ndim == 1 and len(spec.chunk_shape) > 1
    misaligned = values.ndim > 1 and values.shape[0] != len(rows)
    if flattened or misaligned:
        values = values.reshape(len(rows), *spec.chunk_shape[1:])
    return pad_to_chunk(spec, values)


def _default_for(spec: ArraySpec) -> Any:
    return False if spec.dtype == "bool" else 0


def _require_lossless_dtype(values: np.ndarray, declared: str, label: str) -> None:
    """Reject a payload that cannot become *declared* without losing data.

    A stream declares its dtype in the frozen manifest, and the format forbids
    implicit dtype conversion. Without this check ``np.array(values,
    dtype=...)`` would silently truncate float64 to int16 or wrap an
    out-of-range integer, and the loss would only be visible by comparing the
    recording against its source.

    A widening cast between compatible kinds is allowed -- it changes no value.
    Byte order is not a conversion at all, since NumPy compares by value.
    """
    if declared == "utf8":
        return
    target = np.dtype(NUMPY_DTYPES[declared])
    source = values.dtype
    if source == target or np.can_cast(source, target, casting="safe"):
        return
    raise NrfSemanticError(
        f"{label} has dtype {source.name!r} which cannot be stored as the declared "
        f"{declared!r} without a lossy conversion; cast it explicitly first"
    )
