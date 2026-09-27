#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The NRF v1 offline reader.

Specification reference: ``README.md`` sections 8 and 9.

The reader's defining rule is that visibility comes from journal replay, not
from any file that claims to summarize it. ``manifest.json`` and
``journal/head.json`` are caches: this reader replays the journal, refuses a
cache that leads replay, and caps every array at the replayed committed extent
even when more data physically exists on disk. Fill values beyond a committed
extent are not observations.

The second rule is that the reader never degrades gracefully. Zarr returns the
fill value for a chunk that is not on disk, which for NRF would mean handing a
caller zeros where committed samples used to be -- corrupt data wearing the
shape of valid data. So every read first proves that each committed chunk
backing the requested range exists and is the length its commit record states,
and that the array's active metadata still matches the frozen manifest. A
caller that wants to know *what* is wrong rather than be stopped by it uses
:func:`~neurale.io.nrf.diagnose_session`, which never raises and never mutates.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ._canonical import sha256_hex
from ._completeness import CompletenessVerdict, SessionCompleteness, evaluate_completeness
from ._diagnostics import SessionDiagnosis, array_metadata_problem, diagnose_session
from ._errors import NrfCorruptionError, NrfSchemaError, NrfSemanticError
from ._journal import JournalTail, read_journal
from ._manifest import (
    TargetContract,
    descriptor_by_id,
    replay_contracts,
    stream_data_spec,
    target_contracts,
    timestamps_spec,
)
from ._package import Package
from ._paths import JOURNAL_HEAD, JOURNAL_TRANSACTIONS, MANIFEST, chunk_path
from ._replay import CommittedState, replay, require_cache_does_not_lead
from ._schemas import (
    HEAD_SCHEMA,
    MANIFEST_SCHEMA,
    require_supported_version,
    validate_document,
)
from ._semantics import validate_manifest
from ._zarr import ArrayHandles, ArraySpec, read_range


class NrfReader:
    """Read committed data from one immutable ``<session>.nrf`` package."""

    def __init__(self, root: Path, *, verify_checksums: bool = False) -> None:
        self._package = Package(Path(root))
        try:
            self._initialize(self._package.root, verify_checksums=verify_checksums)
        except BaseException as error:
            self._package.close()
            if isinstance(error, (BadZipFile, zlib.error, EOFError)):
                raise NrfCorruptionError("damaged compressed NRF object") from error
            raise

    def _initialize(self, root, *, verify_checksums: bool) -> None:
        self._root = root
        self._verify = verify_checksums
        self._manifest = self._load_manifest()
        self._contracts: dict[str, TargetContract] = target_contracts(self._manifest)
        self._records, self._tail = read_journal(self._root / JOURNAL_TRANSACTIONS)
        self._state = replay(self._records, contracts=replay_contracts(self._contracts))
        self._completeness: SessionCompleteness | None = None
        self._handles = ArrayHandles(self._root)
        self._check_caches()
        if verify_checksums:
            self.verify_committed_objects()

    # --- construction -----------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, *, verify_checksums: bool = False) -> NrfReader:
        """Open a session for reading. Never mutates the session."""
        return cls(Path(path), verify_checksums=verify_checksums)

    def _load_manifest(self) -> dict[str, Any]:
        raw = (self._root / MANIFEST).read_bytes()
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise NrfCorruptionError("manifest.json is not valid JSON") from error
        require_supported_version(manifest)
        validate_document(manifest, MANIFEST_SCHEMA)
        validate_manifest(manifest)
        return manifest

    def _check_caches(self) -> None:
        commit = self._manifest["commit"]
        require_cache_does_not_lead(
            cache_name="manifest.json",
            cache_sequence=commit.get("journal_sequence", 0),
            cache_extents=commit["committed_extents"],
            state=self._state,
        )
        head_path = self._root / JOURNAL_HEAD
        if not head_path.exists():
            return
        try:
            head = json.loads(head_path.read_bytes().decode("utf-8"))
            validate_document(head, HEAD_SCHEMA)
        except (UnicodeDecodeError, ValueError, NrfSchemaError):
            # head.json is a rebuildable pointer cache whose integrity comes
            # from agreeing with replay, not from an intrinsic checksum. One
            # that cannot be parsed, or that is not a head document at all,
            # makes no claim this reader can compare against, so it cannot lead
            # replay and cannot make committed data unreadable. Diagnosis
            # reports it; recovery rebuilds it. Validating first is also what
            # keeps a member of the wrong type from reaching the comparison
            # below and surfacing as a raw ``TypeError``.
            return
        require_cache_does_not_lead(
            cache_name="head.json",
            cache_sequence=head.get("journal_sequence", 0),
            cache_extents=head.get("committed_extents", {}),
            state=self._state,
        )

    # --- session inspection ----------------------------------------------

    @property
    def root(self) -> Path:
        return self._package.path

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def session_id(self) -> str:
        return self._manifest["session"]["id"]

    @property
    def committed_state(self) -> CommittedState:
        return self._state

    @property
    def journal_tail(self) -> JournalTail:
        """Describe anything beyond the last usable journal record."""
        return self._tail

    @property
    def journal_records(self) -> tuple[Mapping[str, Any], ...]:
        """The replayed journal records, in order.

        Exposed because verification needs the *transaction* a record was
        committed by, which the committed state deliberately does not keep: it
        holds what a reader may see, not how it got there.
        """
        return tuple(self._records)

    @property
    def completeness(self) -> SessionCompleteness:
        """This session's completeness, derived from the artifact on first use.

        Cached per reader, not stored in the session: the verdict belongs to the
        reader answering the call, and an artifact never pins it. Computing it
        reads the accounting summary and the ledgers, so it is deferred until a
        caller asks rather than paid for on every open.
        """
        if self._completeness is None:
            self._completeness = evaluate_completeness(self)
        return self._completeness

    @property
    def completeness_verdict(self) -> CompletenessVerdict | None:
        """The one completeness verdict of contract section 5.1.

        ``None`` means there is no sealed session to judge -- a session that was
        never terminated, or one whose finalization was abandoned. It must not be
        read as ``unverified_legacy``, which means the opposite: there is a
        sealed session, and it carries no evidence.
        """
        return self.completeness.verdict

    @property
    def complete(self) -> bool | None:
        """Whether nothing was lost, derived from :attr:`completeness_verdict`.

        Three values, and the third is load-bearing. ``True`` requires the
        persisted accounting to have been checked against this artifact, because
        completeness is the claim that nothing was lost and only checked
        accounting can support it. ``False`` is a session proven incomplete --
        an abnormal termination, or a loss counter that is not zero. ``None`` is
        a session that carries no accounting to check: a legacy session
        terminated ``normal`` reads ``None``, never ``True``.

        Because ``None`` is falsy, ``if not reader.complete`` refuses legacy and
        incomplete sessions alike, unchanged; ``reader.complete is False`` is
        what distinguishes verified-incomplete from unverifiable.
        """
        return self.completeness.complete

    @property
    def accounting_verified(self) -> bool:
        """Whether the persisted accounting was checked against this artifact.

        A separate field from the verdict on purpose: incompleteness has cheaper
        proofs than completeness, so ``complete = False`` with unverified
        accounting is exactly the state every crash-recovered session is in, and
        binding the verdict to this flag would make it unrepresentable.

        It answers "was the accounting checked?", not "was nothing lost". Both
        layers together establish that the summary is internally consistent, that
        its committed counters match the committed artifact, and that it was
        sealed with the termination -- not that everything a producer handed over
        was counted.
        """
        return self.completeness.accounting_verified

    @property
    def legacy_termination_normal(self) -> bool:
        """Whether a session with no accounting nonetheless terminated ``normal``.

        A fact about the termination record, kept apart from the verdict because
        calling it ``complete`` would assert exactly what a legacy artifact
        cannot support.
        """
        return self.completeness.legacy_termination_normal

    @property
    def termination(self):
        return self._state.termination

    def stream_ids(self) -> list[str]:
        return [stream["id"] for stream in self._manifest["streams"]]

    def record_schema_ids(self) -> list[str]:
        return [schema["id"] for schema in self._manifest["record_schemas"]]

    def feature_set_ids(self) -> list[str]:
        return [descriptor["id"] for descriptor in self._manifest["feature_sets"]]

    def stream(self, stream_id: str) -> Mapping[str, Any]:
        return descriptor_by_id(self._manifest["streams"], stream_id, "stream")

    def record_schema(self, schema_id: str) -> Mapping[str, Any]:
        return descriptor_by_id(self._manifest["record_schemas"], schema_id, "record schema")

    def feature_set(self, feature_set_id: str) -> Mapping[str, Any]:
        return descriptor_by_id(self._manifest["feature_sets"], feature_set_id, "feature set")

    def committed_extent(self, target_path: str) -> int:
        """Return the replayed committed extent of one target."""
        return self._state.extent(target_path)

    def stream_extent(self, stream_id: str) -> int:
        return self._state.extent(self.stream(stream_id)["data"]["path"])

    def record_extent(self, schema_id: str) -> int:
        return self._state.extent(self.record_schema(schema_id)["path"])

    # --- stream reads -----------------------------------------------------

    def read_stream(self, stream_id: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Read a committed range of one stream's payload.

        ``stop`` is clamped to the committed extent, so a caller can ask for
        more than exists without ever seeing uncommitted data.
        """
        return self._read_committed(stream_data_spec(self.stream(stream_id)), start, stop)

    def read_timestamps(
        self, stream_id: str, start: int = 0, stop: int | None = None
    ) -> np.ndarray | None:
        """Read explicit timestamps, or ``None`` for a regular-timing stream."""
        timing = self.stream(stream_id)["timing"]
        if timing["mode"] != "explicit":
            return None
        return self._read_committed(timestamps_spec(timing), start, stop)

    def _read_committed(self, spec: ArraySpec, start: int, stop: int | None) -> np.ndarray:
        """Read ``[start, stop)`` with *stop* clamped to the committed extent.

        A caller may ask for more than exists without ever seeing uncommitted
        data, and a range that starts past the extent reads nothing at all.
        """
        extent = self._state.extent(spec.path)
        upper = extent if stop is None else min(stop, extent)
        if start > upper:
            return self._read(spec, 0, 0)
        return self._read(spec, start, upper)

    def iter_blocks(
        self, stream_id: str, block_size: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(start, block)`` pairs with bounded memory.

        Only one block is materialized at a time, so a session far larger than
        memory can be iterated. ``block_size`` defaults to the declared chunk
        length so a block never spans more chunks than necessary.
        """
        spec = stream_data_spec(self.stream(stream_id))
        extent = self._state.extent(spec.path)
        step = block_size or spec.chunk_length
        if step <= 0:
            raise NrfSemanticError("block size must be positive")
        start = 0
        while start < extent:
            stop = min(start + step, extent)
            yield start, self._read(spec, start, stop)
            start = stop

    # --- guarded array access ---------------------------------------------

    def _read(self, spec: ArraySpec, start: int, stop: int) -> np.ndarray:
        """Read ``[start, stop)`` only after proving the bytes are the committed ones.

        Zarr answers a missing chunk with the array's fill value, so without
        this check a deleted or half-written chunk would come back as a block of
        zeros indistinguishable from recorded data. Existence plus the commit
        record's byte length costs one ``stat`` per chunk and catches both
        deletion and truncation; a full re-hash is what ``verify_checksums``
        buys, because it is the only thing that catches bytes changed in place.
        """
        self._package.require_unchanged()
        if stop > start:
            self._require_array_metadata(spec)
            self._require_committed_chunks(spec, start, stop)
        try:
            result = read_range(self._root, spec, start, stop, handles=self._handles)
            self._package.require_unchanged()
            return result
        except (BadZipFile, zlib.error, EOFError) as error:
            raise NrfCorruptionError(f"damaged NRF array entry: {spec.path}") from error

    def _require_array_metadata(self, spec: ArraySpec) -> None:
        problem = array_metadata_problem(self._root, spec)
        if problem is not None:
            raise NrfCorruptionError(
                f"{spec.path} cannot be decoded: {problem}. Run recovery to rebuild the "
                "active metadata cache from committed state."
            )

    def _require_committed_chunks(self, spec: ArraySpec, start: int, stop: int) -> None:
        for coordinate in spec.chunk_coordinates(start, stop):
            path = chunk_path(spec.path, coordinate)
            entry = self._state.committed_objects.get(path)
            if entry is None:
                raise NrfCorruptionError(
                    f"no commit record backs the chunk {path} that "
                    f"[{start}, {stop}) of {spec.path} needs"
                )
            absolute = self._root / path
            if not absolute.exists():
                raise NrfCorruptionError(f"committed chunk is missing: {path}")
            expected = entry.get("byte_length")
            size = absolute.stat().st_size
            if isinstance(expected, int) and size != expected:
                raise NrfCorruptionError(
                    f"committed chunk is {size} bytes, the commit records {expected}: {path}"
                )

    # --- record reads -----------------------------------------------------

    def read_records(
        self, schema_id: str, start: int = 0, stop: int | None = None
    ) -> dict[str, list[Any]]:
        """Read a committed range of one record set as columns.

        A nullable field's missing values come back as ``None``, reconstructed
        from its validity array rather than from a sentinel in the data.
        """
        schema = self.record_schema(schema_id)
        contract = self._contracts[schema["path"]]
        extent = self._state.extent(schema["path"])
        upper = extent if stop is None else min(stop, extent)
        if start >= upper:
            return {item["name"]: [] for item in schema["fields"]}

        by_path = {spec.path: spec for spec in contract.arrays}
        columns: dict[str, list[Any]] = {}
        for item in schema["fields"]:
            values = self._read(by_path[item["array_path"]], start, upper)
            column = [_scalar(value) for value in values]
            if item["nullable"]:
                validity = self._read(by_path[item["validity_path"]], start, upper)
                column = [
                    None if int(flag) == 0 else value
                    for value, flag in zip(column, validity, strict=True)
                ]
            columns[item["name"]] = column
        return columns

    def iter_records(self, schema_id: str) -> Iterator[dict[str, Any]]:
        """Yield committed rows of one record set in stored order."""
        schema = self.record_schema(schema_id)
        extent = self._state.extent(schema["path"])
        step = max(int(schema["chunk_length"]), 1)
        start = 0
        while start < extent:
            stop = min(start + step, extent)
            columns = self.read_records(schema_id, start, stop)
            for i in range(stop - start):
                yield {name: column[i] for name, column in columns.items()}
            start = stop

    # --- typed reconstruction ---------------------------------------------

    def read_signal(self, stream_id: str):
        """Return one sampled stream as a :class:`~neurale.data.SignalArray`."""
        from ._typed import signal_array

        stream = self.stream(stream_id)
        return signal_array(
            self._manifest,
            stream,
            self.read_stream(stream_id),
            self.read_timestamps(stream_id),
        )

    def read_features(self, stream_id: str):
        """Return one feature stream as a :class:`~neurale.data.FeatureMatrix`."""
        from ._typed import feature_matrix

        stream = self.stream(stream_id)
        return feature_matrix(
            self._manifest,
            stream,
            self.read_stream(stream_id),
            self.read_timestamps(stream_id),
        )

    def unmapped_records(self) -> dict[str, list[dict[str, Any]]]:
        """Return committed rows of every record kind without a typed mapping.

        Experiment state, commands, task variables, targets, assistance, drops,
        faults, discontinuities, and termination have no single typed
        destination in :mod:`neurale.data`. They are surfaced here rather than
        dropped, so a caller can see that a session carried them.
        """
        from ._typed import MAPPED_RECORD_KINDS

        result: dict[str, list[dict[str, Any]]] = {}
        for schema in self._manifest["record_schemas"]:
            if schema["kind"] in MAPPED_RECORD_KINDS:
                continue
            rows = list(self.iter_records(schema["id"]))
            if rows:
                result[schema["id"]] = rows
        return result

    def to_recording(self):
        """Reconstruct a :class:`~neurale.data.Recording` from committed data.

        Streams map by kind; record kinds without an unambiguous typed home are
        exposed through the recording metadata instead of being discarded.
        """
        from ._typed import build_recording, event_series, optional_clock, trial_table

        signals = {}
        features = {}
        for stream in self._manifest["streams"]:
            stream_id = stream["id"]
            if stream["kind"] in {"neural", "behavioral"}:
                signals[stream_id] = self.read_signal(stream_id)
            elif stream["kind"] == "feature":
                features[stream_id] = self.read_features(stream_id)

        events = None
        trials = None
        clocks = {clock["id"]: clock for clock in self._manifest["clocks"]}
        for schema in self._manifest["record_schemas"]:
            if self.record_extent(schema["id"]) == 0:
                continue
            columns = self.read_records(schema["id"])
            if schema["kind"] == "events":
                events = event_series(columns, optional_clock(clocks.get(schema["clock_id"])))
            elif schema["kind"] == "trials":
                trials = trial_table(columns)

        return build_recording(
            self._manifest,
            signals=signals,
            features=features,
            events=events,
            trials=trials,
            unmapped_records=self.unmapped_records(),
        )

    # --- integrity --------------------------------------------------------

    def verify_committed_objects(self) -> None:
        """Re-check every committed object against its recorded checksum.

        A missing or checksum-invalid committed object is corruption, not an
        uncommitted tail, so it makes the affected committed range unreadable.
        """
        for path, entry in sorted(self._state.committed_objects.items()):
            absolute = self._root / path
            if not absolute.exists():
                raise NrfCorruptionError(f"committed object is missing: {path}")
            expected = entry.get("byte_length")
            size = absolute.stat().st_size
            if isinstance(expected, int) and size != expected:
                raise NrfCorruptionError(
                    f"committed object is {size} bytes, the commit records {expected}: {path}"
                )
            digest = sha256_hex(absolute.read_bytes())
            if digest != entry["sha256"]:
                raise NrfCorruptionError(f"committed object fails its checksum: {path}")

    def diagnose(self, *, verify_checksums: bool = True) -> SessionDiagnosis:
        """Report everything wrong with this session without raising.

        Opening the reader already established that the session is readable;
        this is for the parts that are not fatal -- lagging caches, orphan
        objects, an unusable checkpoint, an unfinished recording -- and for
        producing the same findings recovery would act on.
        """
        return diagnose_session(self.root, verify_checksums=verify_checksums)

    # --- context manager --------------------------------------------------

    def close(self) -> None:
        """Close the reader. Idempotent; a reader never mutates a session.

        Releases the Zarr handles this reader opened. That release is what lets
        finalization rename a staging tree straight after reading it, so a
        caller that reads a session about to be renamed or deleted should use
        the context manager rather than rely on garbage collection.
        """
        self._handles.close()
        self._package.close()

    def __enter__(self) -> NrfReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False


def _scalar(value: Any) -> Any:
    """Convert a NumPy scalar to a plain Python value."""
    if isinstance(value, np.generic):
        return value.item()
    return value
