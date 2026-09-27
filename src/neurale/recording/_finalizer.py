#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Turn a committed native spool into a canonical NRF session.

Normative sources: sections 3 to 7 of
``docs/development/native_recording_replay.md``, the ledger schemas in
:mod:`neurale.io.nrf._ledgers`, and the container specification in
``specifications/native-spool/v1/README.md``.

What this is
------------

A **noncritical, offline** converter. It runs on an ordinary thread, reads a
spool it never writes to, and produces a session through the existing
:class:`~neurale.io.nrf.NrfWriter`. There is no second Zarr writer and no second
journal here, and there must never be one: the append protocol, the checksum
rules, and the commit visibility point are the format's, and a finalizer with
its own copy of them would be a second implementation of NRF v1 that nothing
validates against the first.

What it produces beyond the standard session
--------------------------------------------

Streams, per-block provenance, per-stream discontinuity records, control records
and faults are the standard session, built by the single
:func:`~neurale.recording._registry.build_session` schema-mapping owner. Three
further outputs are not optional:

* **The five native replay ledgers**: the frame and block topology and the
  per-message discontinuity identity are not recoverable from the standard
  per-stream records (contract section 5.2).
* **The session accounting summary**, bound to the sealing transaction: a
  session that does not carry its own accounting cannot be checked by a reader
  (contract section 5.1).
* **Resumable progress metadata**, which is what lets recovery decide whether a
  retained spool may be resumed against a given target rather than guessing.

An accounting/artifact cross-check runs before anything is published -- the
identities alone are satisfied by a writer claiming a hundred items over a
session holding ninety-nine -- and publication is atomic: the work happens
beside the target and becomes the target through one rename, so a reader never
meets a half-converted session.

What it never does
------------------

It never deletes a spool that has not been finalized, validated, and published
(contract section 7). It never invents a ``normal`` termination for a session
with no clean end (section 5). It never moves the capture outcome: a
finalization failure raises :class:`~neurale.recording.FinalizationError`, which
is a statement about the conversion and about nothing else.

The retry model
---------------

A failed attempt leaves the staging directory and its progress document in
place and publishes nothing. A retry re-reads the progress document, refuses to
continue if the spool's session id, plan fingerprint, or output format version
does not match what the previous attempt recorded (section 7), and then
converts the committed prefix again into a fresh staging session.

That is a **restart**, not an append-resume: the previous attempt's output was
never visible to any reader and is discarded whole, so restarting appends no
logical record twice. What the progress document buys is the guarantee that a
retry is the same work over the same source. Append-resume is not implemented.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from neurale.io.nrf import NrfWriter, diagnose_session
from neurale.io.nrf._canonical import canonical_json_bytes
from neurale.io.nrf._ledgers import (
    LEDGERS,
    LEDGERS_BY_KIND,
    NATIVE_DISCONTINUITIES,
    NATIVE_FRAMES,
    NATIVE_REPLAY_EXTENSION_VERSION,
    NATIVE_REPLAY_NAMESPACE,
    NATIVE_SIGNAL_BLOCKS,
    NATIVE_SIGNAL_GAPS,
    RECOVERY_NAMESPACE,
    SESSION_ACCOUNTING,
    ledger_record_schema,
)
from neurale.io.nrf._plan_document import _DEFAULT_SPOOL_CAPACITY_BYTES

from . import _spool_format as spool
from ._errors import FinalizationError, RecorderConfigError, SpoolSourceError
from ._plan import (
    PreparedMetadata,
    PreparedStream,
    RecordingPlan,
    ResourceBounds,
    SessionIdentity,
    plan_from_manifest_extension,
    plan_from_spool_document,
)
from ._registry import (
    CONTROL_PRIMARY_KEYS,
    SessionPlan,
    block_index_row,
    build_session,
    segment_id,
)
from ._spec import (
    STREAM_KINDS,
    _ResolvedStreamRecording,
)

#: Manifest extension namespace for what the *finalizer* knows and the plan does
#: not: where the artifact's termination came from, whether the source carried a
#: session end, and which spool prefix it was built from. It is separate from
#: ``neurale.native_replay`` on purpose -- that one describes the recording, this
#: one describes the conversion, and folding conversion provenance into the
#: fingerprinted plan extension would make two sessions built from one spool
#: describe their plan differently.
FINALIZATION_NAMESPACE = "neurale.native_finalization"

FINALIZATION_EXTENSION_VERSION = 1

#: Format tag of the progress document. It is versioned because recovery reads it.
PROGRESS_FORMAT = "neurale-native-finalization-progress"
PROGRESS_VERSION = 3

#: The phases one attempt passes through, in order. Persisted so an interrupted
#: finalization says *where* it stopped rather than only that it did: "the
#: process died during publication" and "the process died before it had written
#: a row" call for different next actions, and after a crash the progress
#: document is the only witness to which happened.
PROGRESS_PHASES: tuple[str, ...] = (
    "validating",
    "converting",
    "packing",
    "verifying_package",
    "cross_checking",
    "publishing",
    "cleaning",
    "finalized",
)

_WINDOWS_PUBLICATION_RETRIES = 20
_WINDOWS_PUBLICATION_RETRY_SECONDS = 0.01

#: Statuses the progress document records. ``abandoned`` is terminal and is the
#: one a caller sets deliberately: contract section 5.1 gives an abandoned
#: finalization no completeness verdict, ever, and ``finalization_status`` is
#: what says why.
PROGRESS_STATUSES: tuple[str, ...] = (
    "running",
    "succeeded",
    "failed",
    "failed_retryable",
    "abandoned",
)

#: Name of the staging directory beside the output, and of what it holds.
STAGING_SUFFIX = ".finalizing"
STAGING_SESSION = "session.nrf"
PROGRESS_FILE = "progress.json"

#: Spool control kinds, by producer-identity number, to the NRF record-set kind
#: they become. The registry number is the only bridge between the two
#: (specification section 4.4), and ``experiment_states`` is the one place the
#: two vocabularies disagree in spelling: the accounting registry names the
#: plural, NRF's record kind is singular.
CONTROL_KIND_TO_NRF: Mapping[str, str] = {
    "events": "events",
    "experiment_states": "experiment_state",
    "commands": "commands",
    "targets": "targets",
    "labels": "labels",
    "assistance": "assistance",
    "task_variables": "task_variables",
}

#: Fault codes this module writes itself, rather than copying from a record.
FAULT_CODE_CAPTURE_FAULTED = "capture_faulted"

#: The body fields each control kind is allowed to carry. These are exactly the
#: fields the typed submitter builds; a body offered through the provisional
#: ``submit_control`` that carries more names fields the frozen NRF control
#: schemas have no column for. The finalizer either rejects it or would silently
#: drop it, and silent drop is the one failure this contract does not tolerate
#: (section 1.3): a typed control record must not lose a field it cannot map.
_CONTROL_BODY_FIELDS: Mapping[str, frozenset[str]] = {
    "trials": frozenset({"start_ns", "stop_ns", "label", "outcome"}),
    "faults": frozenset({"code", "stage", "frame_sequence", "signal_id", "text"}),
}
_CONTROL_BODY_FIELDS_DEFAULT = frozenset({"name", "value", "text"})

_PLAN_SIDECAR_SUFFIX = ".plan.json"
_PLAN_SIDECAR_FORMAT = "neurale-recording-plan-sidecar"


def _plan_sidecar_path(spool_path: Path) -> Path:
    return spool_path.with_name(spool_path.name + _PLAN_SIDECAR_SUFFIX)


def write_plan_sidecar(spool_path: Path, plan: RecordingPlan) -> Path:
    """Persist non-fingerprinted plan facts beside a crash-surviving spool.

    Created with the same mode the spool itself is opened with, and explicitly
    rather than through the process umask. This file carries the session
    metadata a caller handed the recorder -- subject and experiment identifiers
    among them -- so a default-umask ``0644`` beside a ``0600`` spool would make
    the companion document readable to accounts the recording it describes is
    not. ``O_EXCL`` is kept: a sidecar that already exists belongs to another
    spool with this name, and overwriting it would silently rebind it.
    """
    extension = plan.manifest_extension()
    encoded = canonical_json_bytes(extension)
    document = {
        "format": _PLAN_SIDECAR_FORMAT,
        "version": 1,
        "plan": extension,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    path = _plan_sidecar_path(spool_path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def discard_spool_bundle(spool_path: Path) -> OSError | None:
    """Remove a spool and its sidecar as the one artifact they are.

    The sidecar is part of the spool rather than a separate file: it exists to
    reconstruct the plan for that spool and describes nothing else, so wherever
    the spool goes it goes too, or the session metadata outlives the recording
    it belongs to. There is deliberately no way to remove one without the other.

    The order is the ownership protocol, not a preference. The spool's pathname
    is the claim on both files -- the store wins it with ``O_EXCL`` and the
    sidecar's name is derived from it -- so unlinking the spool first would free
    the pathname, letting the next creator win it and create *its* sidecar
    before this cleanup reaches its second unlink, which would then delete that
    recorder's file. Taking the sidecar out while the spool still holds the
    pathname keeps both unlinks pointed at the caller's own files.

    Returns the error that stopped it, or ``None`` once both files are gone. A
    returned error means something is still on disk under a pathname the caller
    claimed, and what to do about that is the caller's decision: retry while it
    still owns the path, or record the leftovers as an orphan for explicit
    cleanup. What is *not* on offer is removing one file and reporting the other
    gone -- failing to remove the sidecar leaves the spool alone as well,
    because unlinking it anyway would release the pathname with the sidecar
    still on it.

    This does not survive a crash between the two unlinks: a spool can be left
    without its sidecar. That is the accepted trade, bounded by what this is for
    -- a spool already decided to be discardable. Making the crash case atomic
    needs a tombstone or a rename protocol, not two bare unlinks.
    """
    try:
        _plan_sidecar_path(spool_path).unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
    except OSError as error:
        return error
    return None


def _read_plan_sidecar(spool_path: Path) -> RecordingPlan | None:
    path = _plan_sidecar_path(spool_path)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        extension = document["plan"]
        if document.get("format") != _PLAN_SIDECAR_FORMAT or document.get("version") != 1:
            raise ValueError("unsupported format or version")
        if document.get("sha256") != hashlib.sha256(canonical_json_bytes(extension)).hexdigest():
            raise ValueError("checksum mismatch")
        return plan_from_manifest_extension(extension)
    except (OSError, ValueError, TypeError, KeyError, RecorderConfigError) as error:
        raise SpoolSourceError(f"the retained plan sidecar {path} is invalid: {error}") from error


# --- options, progress, and the report -------------------------------------


@dataclass(frozen=True, slots=True)
class FinalizerOptions:
    """Output-shaping decisions the spool does not carry.

    Chunk lengths and capacities are deliberately not in the spool: they are not
    part of what a plan fingerprint identifies, and an offline finalizer knows
    something the live recorder never did -- exactly how many rows there are. It
    therefore sizes every target to what the committed prefix actually holds
    rather than to a capacity someone had to guess before the session started.
    """

    writer_name: str = "pyneurale-native-finalizer"
    writer_version: str = "1"
    #: Rows per committed chunk for the recorded data streams.
    stream_chunk_length: int = 1024
    #: Rows per committed chunk for the per-block provenance streams.
    block_index_chunk_length: int = 256
    #: Rows per chunk for the control-plane record sets. Raised automatically
    #: when a session holds more rows than ``chunk_length * 1024``, which is the
    #: capacity NRF v1 derives for a record set.
    control_chunk_length: int = 16
    #: Delete recovery input after validated publication unless explicitly retained.
    spool_retention: str = "delete_after_validated_finalization"
    #: Free-form metadata added to the produced session.
    metadata: Mapping[str, Any] = field(default_factory=dict)
    session_metadata: Mapping[str, Any] = field(default_factory=dict)
    #: The original compiled plan when finalization is owned by the public
    #: recorder.  Its fingerprinted document still has to match the spool; the
    #: non-fingerprinted identity, metadata, and resource bounds are retained
    #: instead of being silently replaced by offline defaults.
    recording_plan: RecordingPlan | None = None

    def __post_init__(self) -> None:
        # Imported here rather than at module scope: the retention vocabulary is
        # the native recorder's, and importing it at the top would make the
        # finalizer's import order depend on the facade's.
        from ._native_recorder import SPOOL_RETENTION_POLICIES

        if self.spool_retention not in SPOOL_RETENTION_POLICIES:
            raise RecorderConfigError(
                f"spool_retention must be one of {sorted(SPOOL_RETENTION_POLICIES)}, "
                f"got {self.spool_retention!r}"
            )
        for name in ("stream_chunk_length", "block_index_chunk_length", "control_chunk_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise RecorderConfigError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class StagedConversion:
    """What a finished conversion produced, recorded so a resume need not redo it.

    This is the other half of the resume cursor. ``last_consumed_transaction_id``
    says *how far* the source was consumed; this says what came out, and without
    it a resume that found a complete staged session would have nothing to verify
    it against and would have to rebuild it -- which is the rewrite contract
    section 4.5 forbids.

    The numbers here are the ones the conversion counted as it wrote, and they
    are never trusted on their own: a resume re-runs the same cross-check the
    first attempt did, which compares them against the sealed session's own
    committed extents. What is persisted is one side of a comparison, not a
    conclusion.
    """

    counts: FinalizationCounts
    termination_kind: str
    termination_origin: str
    source_session_end_present: bool
    accounting_origin: str

    def document(self) -> dict[str, Any]:
        return {
            "counts": {
                "frames": self.counts.frames,
                "signal_blocks": self.counts.signal_blocks,
                "discontinuities": self.counts.discontinuities,
                "signal_gaps": self.counts.signal_gaps,
                "control_records": self.counts.control_records,
                "fault_rows": self.counts.fault_rows,
                "stream_rows": dict(self.counts.stream_rows),
                "stream_discontinuity_rows": dict(self.counts.stream_discontinuity_rows),
                "record_rows": dict(self.counts.record_rows),
            },
            "termination_kind": self.termination_kind,
            "termination_origin": self.termination_origin,
            "source_session_end_present": self.source_session_end_present,
            "accounting_origin": self.accounting_origin,
        }

    @classmethod
    def load(cls, value: Mapping[str, Any]) -> StagedConversion:
        counts = value["counts"]
        return cls(
            counts=FinalizationCounts(
                frames=int(counts["frames"]),
                signal_blocks=int(counts["signal_blocks"]),
                discontinuities=int(counts["discontinuities"]),
                signal_gaps=int(counts["signal_gaps"]),
                control_records=int(counts["control_records"]),
                fault_rows=int(counts["fault_rows"]),
                stream_rows={str(k): int(v) for k, v in counts["stream_rows"].items()},
                stream_discontinuity_rows={
                    str(k): int(v) for k, v in counts["stream_discontinuity_rows"].items()
                },
                record_rows={str(k): int(v) for k, v in counts["record_rows"].items()},
            ),
            termination_kind=str(value["termination_kind"]),
            termination_origin=str(value["termination_origin"]),
            source_session_end_present=bool(value["source_session_end_present"]),
            accounting_origin=str(value["accounting_origin"]),
        )


@dataclass(frozen=True, slots=True)
class FinalizationProgress:
    """What one finalization target has recorded about the attempts on it.

    Contract section 7 allows a spool to be resumed only when its superblock
    validates, its session id and plan fingerprint match the finalization
    target, and the recorded progress identifies the last consumed transaction.
    This is that record; every field below exists because a mismatch in it must
    reject the resume rather than be reconciled.
    """

    session_id: str
    plan_fingerprint: str
    spool_session_uuid: str
    nrf_minor_version: int
    committed_prefix_end: int
    committed_transactions: int
    #: The resume cursor: the last source transaction whose records are
    #: materialized in the staged session. Zero until a conversion completes and
    #: the source's last committed transaction once one has, which is the
    #: granularity this finalizer materializes at -- it converts the committed
    #: prefix as one unit. A resume compares it against the source it was handed
    #: and, when the two agree and :attr:`staged` describes what came out,
    #: publishes the staged session instead of writing it a second time.
    last_consumed_transaction_id: int
    attempts: int
    status: str
    #: How far the *current* attempt got. Persisted before each phase begins, so
    #: a document found after a crash names the phase that was running when the
    #: process died rather than the last one that finished.
    phase: str = "validating"
    #: What the completed conversion produced, or ``None`` when none completed.
    #: Present exactly when the cursor above reached the source's end.
    staged: StagedConversion | None = None
    category: str | None = None
    detail: str | None = None
    processed_bytes: int = 0
    total_bytes: int = 0

    def document(self) -> dict[str, Any]:
        return {
            "format": PROGRESS_FORMAT,
            "version": PROGRESS_VERSION,
            "session_id": self.session_id,
            "plan_fingerprint": self.plan_fingerprint,
            "spool_session_uuid": self.spool_session_uuid,
            "nrf_minor_version": self.nrf_minor_version,
            "committed_prefix_end": self.committed_prefix_end,
            "committed_transactions": self.committed_transactions,
            "last_consumed_transaction_id": self.last_consumed_transaction_id,
            "attempts": self.attempts,
            "status": self.status,
            "phase": self.phase,
            "processed_bytes": self.processed_bytes,
            "total_bytes": self.total_bytes,
            "staged": None if self.staged is None else self.staged.document(),
            "category": self.category,
            "detail": self.detail,
        }

    @classmethod
    def load(cls, path: Path) -> FinalizationProgress:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != PROGRESS_FORMAT or value.get("version") != PROGRESS_VERSION:
            raise FinalizationError(
                f"{path} is not a finalization progress document this version understands",
                category="resume_mismatch",
                retryable=False,
            )
        return cls(
            session_id=str(value["session_id"]),
            plan_fingerprint=str(value["plan_fingerprint"]),
            spool_session_uuid=str(value["spool_session_uuid"]),
            nrf_minor_version=int(value["nrf_minor_version"]),
            committed_prefix_end=int(value["committed_prefix_end"]),
            committed_transactions=int(value["committed_transactions"]),
            last_consumed_transaction_id=int(value["last_consumed_transaction_id"]),
            attempts=int(value["attempts"]),
            status=str(value["status"]),
            phase=str(value.get("phase", "validating")),
            processed_bytes=int(value.get("processed_bytes", 0)),
            total_bytes=int(value.get("total_bytes", 0)),
            staged=(
                None if value.get("staged") is None else StagedConversion.load(value["staged"])
            ),
            category=value.get("category"),
            detail=value.get("detail"),
        )


@dataclass(frozen=True, slots=True)
class FinalizationCounts:
    """What the finalizer put into the artifact, counted as it wrote it.

    Counted here rather than read back from the manifest so the cross-check has
    two independently produced numbers to compare. A check that read the same
    value twice would pass for a writer that miscounted consistently.
    """

    frames: int = 0
    signal_blocks: int = 0
    discontinuities: int = 0
    signal_gaps: int = 0
    control_records: int = 0
    fault_rows: int = 0
    stream_rows: Mapping[str, int] = field(default_factory=dict)
    stream_discontinuity_rows: Mapping[str, int] = field(default_factory=dict)
    #: Rows written to each control-plane record set, by record schema id.
    #: Kept per set rather than as one total because that is what layer 2
    #: compares against: ``control_nrf_committed`` has to match the committed
    #: extent of the record sets, and a total could hide two sets that were each
    #: wrong by the same amount in opposite directions.
    record_rows: Mapping[str, int] = field(default_factory=dict)

    @property
    def data_items(self) -> int:
        """Data-plane items, in the sense of contract section 1.1.

        A frame carrying N blocks is one item and a discontinuity carrying N
        gaps is one item, which is why the block and gap counts are diagnostics
        beside this number and never substituted into it.
        """
        return self.frames + self.discontinuities


@dataclass(frozen=True, slots=True)
class FinalizationReport:
    """The outcome of one finalization attempt."""

    session_path: Path
    spool_retained: bool
    counts: FinalizationCounts
    progress: FinalizationProgress
    termination_kind: str
    termination_origin: str
    source_session_end_present: bool
    accounting_origin: str
    #: Spool diagnostic codes found while reading the source. A finalized
    #: session may still carry these -- a torn tail is the ordinary crash path.
    spool_codes: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    #: Whether this report describes a proposal rather than a session. On a dry
    #: run ``session_path`` names nothing on disk and the per-target row counts
    #: are empty, because those are produced by writing.
    dry_run: bool = False
    cleanup_paths: tuple[Path, ...] = ()


# --- adapting a rebuilt plan to the session builder -------------------------
#
# `build_session` reads a *prepared native schema* -- the live pybind11 objects
# the runtime was prepared with. The finalizer has no runtime and no live
# schema; it has the plan document the spool stored, which carries the same
# facts as frozen values. These views present those values in the shape the
# builder reads, so the native path and the Python recorder go through one
# session builder rather than two descriptions of the same session.


class _Named:
    """A value that answers ``.name``, the way a pybind11 enum member does."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return self.name == getattr(other, "name", other)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return self.name


@dataclass(frozen=True, slots=True)
class _RateView:
    numerator: int
    denominator: int


@dataclass(frozen=True, slots=True)
class _SignalView:
    id: int
    dtype: _Named
    layout: _Named
    kind: _Named
    n_channels: int
    nominal_block_samples: int
    max_block_samples: int
    fs: _RateView
    clock_domain: int
    feature_set_id: int
    physical_unit: str


@dataclass(frozen=True, slots=True)
class _FeatureSetView:
    id: int
    feature_names: tuple[str, ...]
    source_stream_id: int
    algorithm_name: str
    algorithm_version: str
    window_length_ns: int
    shift_ns: int


@dataclass(frozen=True, slots=True)
class _SchemaView:
    id: int
    signals: tuple[_SignalView, ...]
    feature_sets: tuple[_FeatureSetView, ...]


def _schema_view(plan: RecordingPlan) -> _SchemaView:
    return _SchemaView(
        id=plan.native_schema_id,
        signals=tuple(
            _SignalView(
                id=signal.id,
                dtype=_Named(signal.dtype),
                layout=_Named(signal.layout),
                kind=_Named(signal.kind),
                n_channels=signal.n_channels,
                nominal_block_samples=signal.nominal_block_samples,
                max_block_samples=signal.max_block_samples,
                fs=_RateView(*signal.fs),
                clock_domain=signal.clock_domain,
                feature_set_id=signal.feature_set_id,
                physical_unit=signal.physical_unit,
            )
            for signal in plan.native_schema.signals
        ),
        feature_sets=tuple(
            _FeatureSetView(
                id=descriptor.id,
                feature_names=descriptor.feature_names,
                source_stream_id=descriptor.source_stream_id,
                algorithm_name=descriptor.algorithm_name,
                algorithm_version=descriptor.algorithm_version,
                window_length_ns=descriptor.window_length_ns,
                shift_ns=descriptor.shift_ns,
            )
            for descriptor in plan.native_schema.feature_sets
        ),
    )


def _stream_spec(
    stream: PreparedStream,
    capacity: int,
    options: FinalizerOptions,
    resource_bounds: ResourceBounds,
) -> _ResolvedStreamRecording:
    """Restate one prepared stream as the declaration the builder consumes."""
    prepared_bound = next(
        (bound for bound in resource_bounds.streams if bound.stream_id == stream.stream_id),
        None,
    )
    if prepared_bound is None:
        chunk = min(options.stream_chunk_length, max(capacity, 1))
        block_idx_chunk = options.block_index_chunk_length
        target_capacity = max(capacity, chunk)
    else:
        chunk = prepared_bound.chunk_length
        block_idx_chunk = prepared_bound.block_index_chunk_length
        target_capacity = max(capacity, prepared_bound.capacity)
    return _ResolvedStreamRecording(
        signal_id=stream.native_signal_id,
        stream_id=stream.stream_id,
        capacity=target_capacity,
        kind=stream.kind,
        name=stream.name,
        unit=tuple(stream.unit),
        channel_names=tuple(stream.channel_names),
        feature_names=tuple(stream.feature_names),
        chunk_length=chunk,
        timing=stream.timing,
        segment_start_index=stream.segment_start_index,
        segment_start_time_ns=stream.segment_start_time_ns,
        block_index=stream.block_index,
        block_index_chunk_length=block_idx_chunk,
        algorithm_name=stream.algorithm_name,
        algorithm_version=stream.algorithm_version,
        source_stream_ids=tuple(stream.source_stream_ids),
    )


# --- the committed prefix, grouped into items -------------------------------


@dataclass(frozen=True, slots=True)
class _Frame:
    payload: spool.FramePayload
    blocks: tuple[spool.SignalBlockPayload, ...]


@dataclass(frozen=True, slots=True)
class _Discontinuity:
    payload: spool.DiscontinuityPayload
    gaps: tuple[spool.SignalGapPayload, ...]


@dataclass(frozen=True, slots=True)
class _Inventory:
    """What the committed prefix adds up to, without the prefix itself.

    Deliberately holds no frame, block, or gap: those are walked with
    :func:`_items` each time they are needed. What is here is either a total or
    a record kind bounded by the control plane rather than the sample rate.
    """

    controls: tuple[spool.ControlPayload, ...]
    faults: tuple[spool.FaultPayload, ...]
    samples_per_signal: Mapping[int, int]
    blocks_per_signal: Mapping[int, int]
    frames: int
    discontinuities: int
    signal_gaps: int
    control_rows_by_kind: Mapping[str, int]
    #: The latest host-monotonic instant any data item recorded. Folded during
    #: the counting walk because it is a scalar: re-deriving it would mean a
    #: third walk over the prefix to learn one number.
    last_data_host_time: int


def _items(scan: spool.SpoolScan) -> Iterator[Any]:
    """Walk the committed prefix as items, grouping children with their owner.

    A generator rather than a list, and that is the point: a frame arrives as a
    frame record followed by its block records, so an item is complete only when
    the next owner starts. Holding the completed ones costs the whole prefix
    resident. Yielding them costs one item.

    Each walk is independent: the record cursor underneath allocates nothing
    that outlives it, so a caller that needs two passes -- counting what the
    session will hold, then writing it -- makes two walks rather than one
    materialization.
    """
    pending_blocks: list[spool.SignalBlockPayload] = []
    pending_gaps: list[spool.SignalGapPayload] = []
    open_frame: spool.FramePayload | None = None
    open_discontinuity: spool.DiscontinuityPayload | None = None

    def close_open() -> Iterator[Any]:
        nonlocal open_frame, open_discontinuity, pending_blocks, pending_gaps
        if open_frame is not None:
            yield _Frame(open_frame, tuple(pending_blocks))
            open_frame = None
            pending_blocks = []
        if open_discontinuity is not None:
            yield _Discontinuity(open_discontinuity, tuple(pending_gaps))
            open_discontinuity = None
            pending_gaps = []

    for record in scan.records():
        if record.kind == spool.RECORD_FRAME:
            yield from close_open()
            open_frame = record.payload
        elif record.kind == spool.RECORD_SIGNAL_BLOCK:
            pending_blocks.append(record.payload)
        elif record.kind == spool.RECORD_DISCONTINUITY:
            yield from close_open()
            open_discontinuity = record.payload
        elif record.kind == spool.RECORD_SIGNAL_GAP:
            pending_gaps.append(record.payload)
        elif record.kind in (spool.RECORD_CONTROL, spool.RECORD_FAULT):
            yield from close_open()
            yield record.payload
    yield from close_open()


def _data_items(scan: spool.SpoolScan) -> Iterator[Any]:
    """Walk only the frames and discontinuities, in committed order.

    The data plane and the control plane are counted, validated, and written
    separately throughout, so a caller that means "every frame and
    discontinuity" says so rather than filtering a mixed walk by hand.
    """
    for item in _items(scan):
        if isinstance(item, (_Frame, _Discontinuity)):
            yield item


def _inventory(scan: spool.SpoolScan) -> _Inventory:
    """Count what the session will hold, without holding the prefix to do it.

    Only the totals survive this pass. The items themselves are walked again by
    the conversion, because a count and a conversion each need the prefix once
    and neither needs it twice: two bounded walks over the spool cost a fixed
    buffer, where one materialization costs the spool.

    The control and fault records are the exception and are retained. They are
    bounded by the session's control plane rather than by its sample rate -- a
    recording produces one control record per submitted event and at most a
    handful of faults, against one frame record per millisecond -- and both are
    needed as a whole rather than in order: the session's outcome is resolved
    from every fault at once, before any conversion starts.
    """
    controls: list[spool.ControlPayload] = []
    faults: list[spool.FaultPayload] = []
    samples: dict[int, int] = {}
    blocks: dict[int, int] = {}
    control_rows: dict[str, int] = {}
    frames = 0
    discontinuities = 0
    gaps_total = 0

    last_host_time = 0
    for item in _items(scan):
        if isinstance(item, _Frame):
            frames += 1
            last_host_time = max(last_host_time, item.payload.host_received_ns)
            for block in item.blocks:
                samples[block.native_signal_id] = (
                    samples.get(block.native_signal_id, 0) + block.n_samples
                )
                blocks[block.native_signal_id] = blocks.get(block.native_signal_id, 0) + 1
        elif isinstance(item, _Discontinuity):
            discontinuities += 1
            gaps_total += len(item.gaps)
            last_host_time = max(last_host_time, item.payload.runtime_accepted_host_time_ns)
        elif hasattr(item, "control_kind"):
            controls.append(item)
            name = spool.producer_identity_name(item.control_kind) or f"kind-{item.control_kind}"
            control_rows[name] = control_rows.get(name, 0) + 1
        else:
            faults.append(item)

    return _Inventory(
        controls=tuple(controls),
        faults=tuple(faults),
        samples_per_signal=samples,
        blocks_per_signal=blocks,
        frames=frames,
        discontinuities=discontinuities,
        signal_gaps=gaps_total,
        control_rows_by_kind=control_rows,
        last_data_host_time=last_host_time,
    )


# --- committed-prefix topology validation -----------------------------------


def _reject_unmaterializable_streams(plan: RecordingPlan) -> None:
    """Refuse a plan whose streams this finalizer cannot materialize.

    ``_write_frame`` materializes a signal block as a dense
    ``n_samples x n_channels`` array of the stream's dtype, and skips a
    block whose sample count is zero. Neither holds for a ``spike`` stream: its
    payload is a fixed-capacity sparse block whose byte count does not follow
    from a sample count, and a block carrying no spikes is a real committed
    block rather than an empty one. Finalizing one through the dense path would
    reject every block as mis-sized, or -- worse -- drop the empty ones while
    the parent frame's ledger still counted them, publishing a session whose
    ``recorded_signal_block_count`` no ``native-signal-blocks-v1`` row backs.

    The recorder cannot produce such a plan today, but a spool's plan document is
    *source data* reconstructed from bytes this process did not write, and the
    plan grammar admits the kind. The refusal is stated against
    :data:`STREAM_KINDS` rather than against ``spike`` by name so that opening a
    kind for recording forces this path to be revisited: the day ``spike``
    becomes recordable, this check stops rejecting it, and the dense
    materialization above must already handle it.
    """
    unsupported = sorted(
        {stream.kind for stream in plan.streams if stream.kind not in STREAM_KINDS}
    )
    if unsupported:
        raise SpoolSourceError(
            f"the plan the spool stores records stream kinds this finalizer cannot "
            f"materialize ({', '.join(unsupported)}); its blocks are not the dense "
            "sample-by-channel payload the conversion assumes, and finalizing them "
            "through it would publish a session whose blocks its ledgers contradict"
        )


def _validate_committed_prefix(
    scan: spool.SpoolScan, plan: RecordingPlan, inventory: _Inventory
) -> None:
    """Validate the committed prefix's logical topology before any NRF is written.

    The spool container validates framing, checksums, transaction/accounting and
    record-header owner ordinals, but treats record *payloads* as opaque
    (specification section 13). This pass decodes them -- which ``_inventory``
    already did -- and checks the interior topology the recorder guarantees: a
    frame's claimed recorded block count matches its children, children belong
    to their parent, block indices and payload offsets are consistent, block
    payloads stay within the frame, every recorded signal is in the plan, the
    frame's schema is the plan's schema, and the global ordinals are monotonic
    and contiguous the way the recorder assigns them.

    It also checks three cross-record invariants the container cannot, because
    they span record kinds the container treats independently: the whole
    committed prefix belongs to one native runtime session (one
    ``native_session_id`` on every frame, discontinuity, and recorder fault --
    two sessions cannot be folded into one canonical NRF session); every signal
    a gap names is a signal the native schema declares (a gap may name an
    unselected signal, but not one the schema never had, contract section 5.2);
    and every control record carries one clock domain, distinct from every data
    clock, so its ``time_ns`` has stated semantics rather than an ignored one
    (contract section 1.2: a control record carries the clock domain its time
    belongs to, and ignoring it loses the time semantics, not optional metadata).

    A violation is refused as a source problem, not sealed as a faulted session:
    a spool is the only copy, and promoting a prefix whose topology nobody can
    account for would publish a session whose accounting it cannot stand behind.
    A frame whose parts committed but whose claimed children did not is not a
    partial item -- it is not a committed item at all, and counting it would make
    ``nrf_committed`` a number the artifact contradicts.
    """
    recorded_signals = {stream.native_signal_id for stream in plan.streams}
    schema_signals = set(plan.planned_signal_ids)
    signal_clock_domains = {signal.clock_domain for signal in plan.native_schema.signals}
    expected_data_ordinal = 0
    expected_frame_ordinal = 0
    expected_block_ordinal = 0
    expected_gap_ordinal = 0
    expected_native_session_id: int | None = None

    def _check_session_id(value: int, where: str) -> None:
        nonlocal expected_native_session_id
        if expected_native_session_id is None:
            expected_native_session_id = value
        elif value != expected_native_session_id:
            raise SpoolSourceError(
                f"{where} carries native_session_id {value}, but the committed prefix's "
                f"native session identity is {expected_native_session_id}; a canonical NRF "
                "session may not fold two native runtime sessions into one"
            )

    for item in _data_items(scan):
        payload = item.payload
        _check_session_id(payload.native_session_id, f"data message {payload.data_message_ordinal}")
        if payload.data_message_ordinal != expected_data_ordinal:
            raise SpoolSourceError(
                f"data-message ordinal {payload.data_message_ordinal} is not the expected "
                f"{expected_data_ordinal}; the committed prefix is not contiguous"
            )
        expected_data_ordinal = payload.data_message_ordinal + 1
        if isinstance(item, _Frame):
            _validate_frame(
                item, plan, recorded_signals, expected_frame_ordinal, expected_block_ordinal
            )
            expected_frame_ordinal += 1
            expected_block_ordinal += len(item.blocks)
        else:
            _validate_discontinuity(item, schema_signals, expected_gap_ordinal)
            expected_gap_ordinal += len(item.gaps)

    expected_control_ordinal = 0
    control_clock_domain: int | None = None
    for control in inventory.controls:
        if control.submission_ordinal != expected_control_ordinal:
            raise SpoolSourceError(
                f"control submission ordinal {control.submission_ordinal} is not the expected "
                f"{expected_control_ordinal}; the control plane is not contiguous"
            )
        expected_control_ordinal += 1
        if control_clock_domain is None:
            control_clock_domain = control.clock_domain
        elif control.clock_domain != control_clock_domain:
            raise SpoolSourceError(
                f"control record {control.submission_ordinal} carries clock_domain "
                f"{control.clock_domain}, not {control_clock_domain}; a session has one "
                "control clock, and a time_ns whose clock domain is not stated consistently "
                "is a time whose semantics the finalizer cannot seal"
            )

    for fault in inventory.faults:
        _check_session_id(fault.native_session_id, "a recorder fault record")

    if control_clock_domain is not None and control_clock_domain in signal_clock_domains:
        raise SpoolSourceError(
            f"the control plane's clock_domain {control_clock_domain} is also a recorded "
            "signal's clock domain; the session clock is distinct from every data clock, "
            "and a spool that disagrees cannot be sealed with honest time semantics"
        )


def _validate_frame(
    frame: _Frame,
    plan: RecordingPlan,
    plan_signals: set[int],
    expected_frame_ordinal: int,
    expected_block_ordinal: int,
) -> None:
    payload = frame.payload
    if payload.native_schema_id != plan.native_schema_id:
        raise SpoolSourceError(
            f"frame {payload.frame_sequence} belongs to native schema {payload.native_schema_id}, "
            f"but the plan this spool stores is for schema {plan.native_schema_id}"
        )
    if payload.frame_ordinal != expected_frame_ordinal:
        raise SpoolSourceError(
            f"frame {payload.frame_sequence} carries frame_ordinal {payload.frame_ordinal}, "
            f"not the expected {expected_frame_ordinal}; frame ordinals are not contiguous"
        )
    if len(frame.blocks) != payload.recorded_signal_block_count:
        raise SpoolSourceError(
            f"frame {payload.frame_sequence} claims {payload.recorded_signal_block_count} "
            f"recorded signal block(s) and carries {len(frame.blocks)}; a frame whose claimed "
            "children did not commit is not a committed item"
        )
    if payload.recorded_signal_block_count > payload.signal_block_count:
        raise SpoolSourceError(
            f"frame {payload.frame_sequence} records {payload.recorded_signal_block_count} "
            f"blocks but the native frame carried {payload.signal_block_count}"
        )
    seen_indices: set[int] = set()
    occupied: list[tuple[int, int]] = []
    for block in frame.blocks:
        if block.data_message_ordinal != payload.data_message_ordinal:
            raise SpoolSourceError(
                f"signal block {block.signal_block_ordinal} belongs to data-message "
                f"{block.data_message_ordinal}, not its frame's {payload.data_message_ordinal}"
            )
        if block.frame_ordinal != payload.frame_ordinal:
            raise SpoolSourceError(
                f"signal block {block.signal_block_ordinal} belongs to frame_ordinal "
                f"{block.frame_ordinal}, not its frame's {payload.frame_ordinal}"
            )
        if block.signal_block_ordinal != expected_block_ordinal:
            raise SpoolSourceError(
                f"signal block ordinal {block.signal_block_ordinal} is not the expected "
                f"{expected_block_ordinal}; block ordinals are not contiguous"
            )
        expected_block_ordinal += 1
        if block.native_signal_id not in plan_signals:
            raise SpoolSourceError(
                f"signal block {block.signal_block_ordinal} is for signal "
                f"{block.native_signal_id}, which the plan this spool stores does not record"
            )
        if block.block_idx_in_frame in seen_indices:
            raise SpoolSourceError(
                f"frame {payload.frame_sequence} has two blocks at index {block.block_idx_in_frame}"
            )
        if block.block_idx_in_frame >= payload.signal_block_count:
            raise SpoolSourceError(
                f"frame {payload.frame_sequence} has a block at index "
                f"{block.block_idx_in_frame} but carried only {payload.signal_block_count} "
                "native block(s)"
            )
        seen_indices.add(block.block_idx_in_frame)
        if block.payload_byte_count > 0:
            if block.payload_offset + block.payload_byte_count > payload.total_payload_byte_count:
                raise SpoolSourceError(
                    f"signal block {block.signal_block_ordinal} ends at byte "
                    f"{block.payload_offset + block.payload_byte_count}, past the frame payload "
                    f"of {payload.total_payload_byte_count}"
                )
            occupied.append((block.payload_offset, block.payload_byte_count))
    occupied.sort()
    for (start_a, bytes_a), (start_b, _bytes_b) in itertools.pairwise(occupied):
        if start_b < start_a + bytes_a:
            raise SpoolSourceError(
                f"frame {payload.frame_sequence} has overlapping signal-block payloads at "
                f"byte {start_a}"
            )


def _validate_discontinuity(
    item: _Discontinuity, schema_signals: set[int], expected_gap_ordinal: int
) -> None:
    payload = item.payload
    if len(item.gaps) != payload.signal_gap_count:
        raise SpoolSourceError(
            f"discontinuity {payload.data_message_ordinal} claims "
            f"{payload.signal_gap_count} signal gap(s) and carries {len(item.gaps)}"
        )
    seen_indices: set[int] = set()
    for gap in item.gaps:
        if gap.data_message_ordinal != payload.data_message_ordinal:
            raise SpoolSourceError(
                f"signal gap {gap.signal_gap_ordinal} belongs to data-message "
                f"{gap.data_message_ordinal}, not its discontinuity's "
                f"{payload.data_message_ordinal}"
            )
        if gap.signal_gap_ordinal != expected_gap_ordinal:
            raise SpoolSourceError(
                f"signal gap ordinal {gap.signal_gap_ordinal} is not the expected "
                f"{expected_gap_ordinal}; gap ordinals are not contiguous"
            )
        expected_gap_ordinal += 1
        # A gap may name a signal the recording plan did not select -- "which
        # signals were affected" is part of what the discontinuity said, and the
        # gap ledger preserves gaps for unselected signals (contract section 5.2)
        # -- but it may not name a signal the native schema never declared: that
        # reference would persist a signal id nothing in the session recognises.
        if gap.native_signal_id not in schema_signals:
            raise SpoolSourceError(
                f"signal gap {gap.signal_gap_ordinal} references signal "
                f"{gap.native_signal_id}, which the native schema this spool stores does "
                "not declare; a gap may name an unselected signal, but not one the schema "
                "never had"
            )
        if gap.gap_idx_in_message in seen_indices:
            raise SpoolSourceError(
                f"discontinuity {payload.data_message_ordinal} has two gaps at index "
                f"{gap.gap_idx_in_message}"
            )
        if gap.gap_idx_in_message >= payload.signal_gap_count:
            raise SpoolSourceError(
                f"discontinuity {payload.data_message_ordinal} has a gap at index "
                f"{gap.gap_idx_in_message} but claims only {payload.signal_gap_count} gap(s)"
            )
        seen_indices.add(gap.gap_idx_in_message)


# --- the finalizer ----------------------------------------------------------


def _created_at(unix_nanos: int) -> str:
    """Return the superblock's creation time in the exact UTC spelling NRF uses."""
    seconds, remainder = divmod(int(unix_nanos), 1_000_000_000)
    moment = datetime.fromtimestamp(seconds, UTC).replace(microsecond=remainder // 1_000)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _chunk_for(rows: int, default: int) -> int:
    """Return a chunk length whose derived record-set capacity holds *rows*.

    NRF v1 derives a record set's capacity as ``chunk_length * 1024``, so the
    chunk length is the only lever a writer has over how many rows a set can
    ever hold. Raising it to fit is not a tuning decision; it is the difference
    between a session that can be written and one that cannot.
    """
    return max(default, math.ceil(rows / 1024) if rows else 1)


class _Finalization:
    """One attempt. Holds the mutable position the conversion walks through."""

    def __init__(
        self,
        scan: spool.SpoolScan,
        plan: RecordingPlan,
        options: FinalizerOptions,
        inventory: _Inventory,
    ) -> None:
        self._scan = scan
        self._plan = plan
        self._options = options
        self._inventory = inventory
        self._rows: dict[str, int] = {}
        self._segments: dict[str, int] = {}
        self._origin_checked: set[str] = set()
        self._counters: dict[str, int] = {}
        self._last_host_ns = 0
        self._session: SessionPlan | None = None
        self._frames = 0
        self._blocks = 0
        self._discontinuities = 0
        self._gaps = 0
        self._controls = 0
        self._fault_rows = 0
        self._stream_discontinuity_rows: dict[str, int] = {}
        self._record_rows: dict[str, int] = {}
        #: The row written from the recorder's own committed fault record, as
        #: distinct from a fault a caller submitted on the control plane. A
        #: faulted termination must name the fault that ended the capture, and a
        #: control-plane fault row submitted earlier is a different event.
        self._primary_fault_id: str | None = None

    # --- identifiers ------------------------------------------------------

    def _append_control_rows(self, writer: NrfWriter, schema_id: str, rows: dict[str, Any]) -> None:
        """Append to one control-plane record set and count what went in.

        Counted here, at the call, rather than derived afterwards: the number
        the cross-check compares has to come from the writing, or it is the same
        value read twice.
        """
        writer.append_records(schema_id, rows)
        added = len(next(iter(rows.values())))
        self._record_rows[schema_id] = self._record_rows.get(schema_id, 0) + added

    def _next_id(self, prefix: str) -> str:
        count = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = count
        return f"{prefix}-{count:08d}"

    # --- registration -----------------------------------------------------

    def _session_build_inputs(self) -> tuple[list[_ResolvedStreamRecording], int]:
        control_rows = sum(self._inventory.control_rows_by_kind.values()) + len(
            self._inventory.faults
        )
        specs = [
            _stream_spec(
                stream,
                self._inventory.samples_per_signal.get(stream.native_signal_id, 0),
                self._options,
                self._plan.resource_bounds,
            )
            for stream in self._plan.streams
        ]
        return (
            specs,
            (
                self._plan.resource_bounds.control_chunk_length
                if self._options.recording_plan is not None
                else _chunk_for(control_rows, self._plan.resource_bounds.control_chunk_length)
            ),
        )

    def _register_ledgers(self, writer: NrfWriter) -> None:
        """Declare the five native replay ledgers, sized for what will be written."""
        rows = {
            NATIVE_FRAMES.kind: self._inventory.frames,
            NATIVE_SIGNAL_BLOCKS.kind: sum(self._inventory.blocks_per_signal.values()),
            NATIVE_DISCONTINUITIES.kind: self._inventory.discontinuities,
            NATIVE_SIGNAL_GAPS.kind: self._inventory.signal_gaps,
            SESSION_ACCOUNTING.kind: 1,
        }
        for ledger in LEDGERS:
            descriptor = ledger_record_schema(
                ledger,
                clock_id="session.clock",
                chunk_length=_chunk_for(rows[ledger.kind], ledger.default_chunk_length),
            )
            writer.registry.register_record_schema(
                descriptor["id"],
                kind=descriptor["kind"],
                primary_key=descriptor["primary_key"],
                clock_id=descriptor["clock_id"],
                chunk_length=descriptor["chunk_length"],
                fields=descriptor["fields"],
            )

    # --- the data plane ---------------------------------------------------

    def _write_frame(self, writer: NrfWriter, frame: _Frame) -> None:
        assert self._session is not None
        payload = frame.payload
        if payload.native_schema_id != self._session.schema_id:
            raise FinalizationError(
                f"spool frame {payload.frame_sequence} belongs to native schema "
                f"{payload.native_schema_id}, but the plan this spool stores is for schema "
                f"{self._session.schema_id}",
                category="source",
                retryable=False,
            )
        self._last_host_ns = payload.host_received_ns
        first_block_ordinal: int | None = None
        block_rows: list[dict[str, Any]] = []

        for block in frame.blocks:
            stream = self._session.plan_for(block.native_signal_id)
            if stream is None:
                # A block for a signal the plan does not record cannot exist:
                # the recorder writes a block record only for a recorded signal.
                raise FinalizationError(
                    f"the spool holds a signal block for signal {block.native_signal_id}, which "
                    "the plan it stores does not record",
                    category="source",
                    retryable=False,
                )
            expected = block.n_samples * stream.n_channels * stream.numpy_dtype.itemsize
            if len(block.samples) != expected or block.payload_byte_count != expected:
                raise FinalizationError(
                    f"the spool's block for signal {block.native_signal_id} carries "
                    f"{len(block.samples)} sample bytes and claims {block.payload_byte_count}, but "
                    f"{block.n_samples} samples of {stream.n_channels} channels need "
                    f"{expected}",
                    category="source",
                    retryable=False,
                )
            if block.n_samples == 0:
                continue
            if not stream.explicit_timing:
                self._check_regular_origin(stream, block)
            values = stream.decode(
                np.frombuffer(block.samples, dtype=np.uint8), 0, expected, block.n_samples
            )
            stamps = (
                stream.timestamps(block.observation_time_start_ns, block.n_samples)
                if stream.explicit_timing
                else None
            )
            writer.append_stream(stream.stream_id, values, timestamps=stamps)
            row_offset = self._rows.get(stream.stream_id, 0)
            self._rows[stream.stream_id] = row_offset + block.n_samples

            if stream.block_index_stream_id is not None:
                writer.append_stream(
                    stream.block_index_stream_id,
                    np.asarray(
                        [
                            block_index_row(
                                frame_sequence=payload.frame_sequence,
                                sample_idx_start=block.sample_idx_start,
                                last_sample_idx=block.last_sample_idx,
                                n_samples=block.n_samples,
                                row_offset=row_offset,
                                device_tick_start=block.device_tick_start,
                                host_received_ns=payload.host_received_ns,
                            )
                        ],
                        dtype="int64",
                    ),
                    timestamps=np.asarray([block.observation_time_start_ns], dtype="int64"),
                )
                self._rows[stream.block_index_stream_id] = (
                    self._rows.get(stream.block_index_stream_id, 0) + 1
                )

            if first_block_ordinal is None:
                first_block_ordinal = block.signal_block_ordinal
            block_rows.append(
                {
                    "signal_block_ordinal": block.signal_block_ordinal,
                    "data_message_ordinal": block.data_message_ordinal,
                    "frame_ordinal": block.frame_ordinal,
                    "block_index_in_frame": block.block_idx_in_frame,
                    "native_signal_id": block.native_signal_id,
                    "stream_id": stream.stream_id,
                    "sample_idx_start": block.sample_idx_start,
                    "last_sample_idx": block.last_sample_idx,
                    "n_samples": block.n_samples,
                    "device_tick_start": block.device_tick_start,
                    "observation_time_start_ns": block.observation_time_start_ns,
                    "row_offset": row_offset,
                    "payload_byte_count": block.payload_byte_count,
                    "payload_offset": block.payload_offset,
                    "clock_sync_device_tick_reference": block.clock_sync_device_tick_reference,
                    "clock_sync_host_time_reference_ns": block.clock_sync_host_time_reference_ns,
                    "clock_sync_rate_numerator": block.clock_sync_rate_numerator,
                    "clock_sync_rate_denominator": block.clock_sync_rate_denominator,
                    "clock_sync_uncertainty_ns": block.clock_sync_uncertainty_ns,
                    "clock_sync_clock_domain": block.clock_sync_clock_domain,
                    "clock_sync_generation": block.clock_sync_generation,
                    "clock_sync_flags": block.clock_sync_flags,
                }
            )
            self._blocks += 1

        if block_rows:
            writer.append_records(
                NATIVE_SIGNAL_BLOCKS.schema_id,
                {
                    name: [row[name] for row in block_rows]
                    for name in NATIVE_SIGNAL_BLOCKS.field_names
                },
            )
        writer.append_records(
            NATIVE_FRAMES.schema_id,
            {
                "data_message_ordinal": [payload.data_message_ordinal],
                "native_session_id": [payload.native_session_id],
                "frame_sequence": [payload.frame_sequence],
                "frame_ordinal": [payload.frame_ordinal],
                "host_received_ns": [payload.host_received_ns],
                # Null is "the frame carried none", which the header flags say
                # and a zero could not: zero is a legal tick and a legal deadline.
                "source_tick": [
                    payload.source_tick
                    if payload.frame_flags & spool.FRAME_FLAG_SOURCE_TICK_VALID
                    else None
                ],
                "valid_until_ns": [
                    payload.valid_until_ns
                    if payload.frame_flags & spool.FRAME_FLAG_DEADLINE_VALID
                    else None
                ],
                "native_schema_id": [payload.native_schema_id],
                "source_clock_domain": [payload.source_clock_domain],
                "frame_flags": [payload.frame_flags],
                "signal_block_count": [payload.signal_block_count],
                "recorded_signal_block_count": [payload.recorded_signal_block_count],
                "total_payload_byte_count": [payload.total_payload_byte_count],
                "first_signal_block_ordinal": [first_block_ordinal],
            },
        )
        self._frames += 1

    def _check_regular_origin(self, stream: Any, block: spool.SignalBlockPayload) -> None:
        """Refuse a regular stream whose first recorded item is not where declared.

        A regular stream stores no timestamp, so its whole time axis is the
        declared origin plus the rate. Writing one that actually starts
        elsewhere produces a session that reads back at the wrong index and the
        wrong time with nothing on disk saying so.
        """
        if stream.stream_id in self._origin_checked:
            return
        self._origin_checked.add(stream.stream_id)
        if (
            block.sample_idx_start == stream.segment_start_index
            and block.observation_time_start_ns == stream.segment_start_time_ns
        ):
            return
        raise FinalizationError(
            f"stream {stream.stream_id!r} records regular timing from sample index "
            f"{stream.segment_start_index} at {stream.segment_start_time_ns} ns, but the spool's "
            f"first block starts at sample index {block.sample_idx_start} at "
            f"{block.observation_time_start_ns} ns",
            category="source",
            retryable=False,
        )

    def _write_discontinuity(self, writer: NrfWriter, item: _Discontinuity) -> None:
        assert self._session is not None
        payload = item.payload
        if payload.signal_gap_count != len(item.gaps):
            raise FinalizationError(
                f"the spool's discontinuity {payload.data_message_ordinal} declares "
                f"{payload.signal_gap_count} signal gaps and carries {len(item.gaps)}",
                category="source",
                retryable=False,
            )

        # The standard per-stream records, written exactly as the Python
        # recorder writes them: one native message fans out into one record per
        # affected stream, and a frame-level gap affects every recorded stream.
        # This is the projection contract section 5.2 calls lossy, which is why
        # the ledgers below exist beside it rather than instead of it.
        targets: list[tuple[Any, spool.SignalGapPayload | None]] = []
        if item.gaps:
            for gap in item.gaps:
                stream = self._session.plan_for(gap.native_signal_id)
                if stream is not None:
                    targets.append((stream, gap))
        else:
            targets = [(stream, None) for stream in self._session.streams.values()]

        for stream, gap in targets:
            closing = self._segments.get(stream.stream_id, 1)
            opening = closing + 1
            self._segments[stream.stream_id] = opening
            if gap is None:
                reason = spool.gap_reason_name(payload.reason)
                expected_idx = stream.segment_start_index + self._rows.get(stream.stream_id, 0)
                actual_idx = None
                missing = None
                missing_ticks = None
            else:
                reason = spool.gap_reason_name(gap.reason)
                expected_idx = gap.expected_sample_idx
                actual_idx = gap.actual_sample_idx
                missing = gap.missing_samples if _missing_samples_known(gap) else None
                missing_ticks = (
                    gap.actual_device_tick - gap.expected_device_tick
                    if gap.gap_flags & spool.GAP_FLAG_DEVICE_TICKS_AVAILABLE
                    else None
                )
            writer.append_records(
                self._session.discontinuity_schema_ids[stream.stream_id],
                {
                    "discontinuity_id": [self._next_id("discontinuity")],
                    # The native message carries a time of its own -- stamped at
                    # runtime acceptance, where its ordinal was assigned -- so
                    # this is a measured value rather than the placement time
                    # the Python path has to fall back to.
                    "time_ns": [payload.runtime_accepted_host_time_ns],
                    "reason": [reason],
                    "stream_id": [stream.stream_id],
                    "previous_segment_id": [segment_id(closing)],
                    "next_segment_id": [segment_id(opening)],
                    "previous_frame_sequence": [payload.previous_frame_sequence],
                    "actual_frame_sequence": [payload.actual_frame_sequence],
                    "expected_sample_index": [expected_idx],
                    "actual_sample_index": [actual_idx],
                    "missing_samples": [missing],
                    "missing_device_ticks": [missing_ticks],
                },
            )
            self._stream_discontinuity_rows[stream.stream_id] = (
                self._stream_discontinuity_rows.get(stream.stream_id, 0) + 1
            )

        if item.gaps:
            writer.append_records(
                NATIVE_SIGNAL_GAPS.schema_id,
                {
                    "signal_gap_ordinal": [gap.signal_gap_ordinal for gap in item.gaps],
                    "data_message_ordinal": [gap.data_message_ordinal for gap in item.gaps],
                    "gap_index_in_message": [gap.gap_idx_in_message for gap in item.gaps],
                    "native_signal_id": [gap.native_signal_id for gap in item.gaps],
                    "expected_sample_index": [gap.expected_sample_idx for gap in item.gaps],
                    "actual_sample_index": [gap.actual_sample_idx for gap in item.gaps],
                    "missing_samples": [
                        gap.missing_samples if _missing_samples_known(gap) else None
                        for gap in item.gaps
                    ],
                    "expected_device_tick": [gap.expected_device_tick for gap in item.gaps],
                    "actual_device_tick": [gap.actual_device_tick for gap in item.gaps],
                    "reason": [spool.gap_reason_name(gap.reason) for gap in item.gaps],
                    "gap_flags": [gap.gap_flags for gap in item.gaps],
                },
            )
            self._gaps += len(item.gaps)

        writer.append_records(
            NATIVE_DISCONTINUITIES.schema_id,
            {
                "data_message_ordinal": [payload.data_message_ordinal],
                "native_session_id": [payload.native_session_id],
                "previous_frame_sequence": [payload.previous_frame_sequence],
                "actual_frame_sequence": [payload.actual_frame_sequence],
                "reason": [spool.gap_reason_name(payload.reason)],
                # Null under the zero-child anchor rule: a frame-level gap has
                # no first child, and zero would name gap ordinal zero.
                "first_signal_gap_ordinal": [
                    item.gaps[0].signal_gap_ordinal if item.gaps else None
                ],
                "signal_gap_count": [payload.signal_gap_count],
                "runtime_accepted_host_time_ns": [payload.runtime_accepted_host_time_ns],
            },
        )
        self._discontinuities += 1

    # --- the control plane ------------------------------------------------

    def _write_control(self, writer: NrfWriter, control: spool.ControlPayload) -> None:
        assert self._session is not None
        kind = spool.producer_identity_name(control.control_kind)
        if kind is None:
            raise FinalizationError(
                f"the spool holds a control record of unregistered kind {control.control_kind}",
                category="source",
                retryable=False,
            )
        try:
            # The body is a view into the spool, so it is materialized here.
            # Control bodies are small JSON documents, one per control record;
            # the copy the scanner avoids is the per-signal-block one.
            body = json.loads(bytes(control.body).decode("utf-8")) if control.body else {}
        except (UnicodeDecodeError, ValueError) as error:
            raise FinalizationError(
                f"the body of control record {control.submission_ordinal} is not JSON: {error}",
                category="source",
                retryable=False,
            ) from error
        if not isinstance(body, Mapping):
            raise FinalizationError(
                f"the body of control record {control.submission_ordinal} is not an object",
                category="source",
                retryable=False,
            )
        allowed = _CONTROL_BODY_FIELDS.get(kind, _CONTROL_BODY_FIELDS_DEFAULT)
        unexpected = set(body) - allowed
        if unexpected:
            raise FinalizationError(
                f"control record {control.submission_ordinal} of kind {kind} carries "
                f"field(s) {sorted(unexpected)} the NRF control schema has no column for; "
                "the finalizer does not silently drop fields it cannot map",
                category="source",
                retryable=False,
            )

        if kind == "trials":
            self._append_control_rows(
                writer,
                self._session.trials_schema_id,
                {
                    "trial_id": [self._next_id("trial")],
                    "start_ns": [int(body.get("start_ns", control.time_ns))],
                    "stop_ns": [int(body.get("stop_ns", control.time_ns))],
                    "label": [_optional_text(body.get("label"))],
                    "outcome": [_optional_text(body.get("outcome"))],
                },
            )
        elif kind == "faults":
            self._append_fault_row(
                writer,
                time_ns=control.time_ns,
                code=str(body.get("code", "")),
                stage=str(body.get("stage", "")),
                frame_sequence=_optional_int(body.get("frame_sequence")),
                signal_id=_optional_int(body.get("signal_id")),
                text=_optional_text(body.get("text")),
            )
        else:
            nrf_kind = CONTROL_KIND_TO_NRF[kind]
            schema_id = self._session.control_schema_ids[nrf_kind]
            primary_key = CONTROL_PRIMARY_KEYS[nrf_kind]
            self._append_control_rows(
                writer,
                schema_id,
                {
                    primary_key: [self._next_id(primary_key.removesuffix("_id"))],
                    "time_ns": [control.time_ns],
                    "name": [str(body.get("name", ""))],
                    "value": [_optional_float(body.get("value"))],
                    "text": [_optional_text(body.get("text"))],
                },
            )
        self._controls += 1

    def _append_fault_row(
        self,
        writer: NrfWriter,
        *,
        time_ns: int,
        code: str,
        stage: str,
        frame_sequence: int | None,
        signal_id: int | None,
        text: str | None,
    ) -> str:
        assert self._session is not None
        fault_id = self._next_id("fault")
        self._append_control_rows(
            writer,
            self._session.faults_schema_id,
            {
                "fault_id": [fault_id],
                "time_ns": [time_ns],
                "code": [code],
                "stage": [stage],
                "frame_sequence": [frame_sequence],
                "signal_id": [signal_id],
                "text": [text],
            },
        )
        self._fault_rows += 1
        return fault_id

    def _write_recorder_fault(self, writer: NrfWriter, fault: spool.FaultPayload) -> None:
        """Write the recorder's own committed fault record as a fault row.

        The standard ``faults-v1`` columns carry the fields a reader queries by
        -- when, what code, what stage, which frame and signal -- and the rest of
        the native fault payload travels as a deterministic structured JSON object
        in the row's ``text`` column. The frozen v1 fault schema has no column
        for ``runtime_generation``, ``sample_index``, ``device_tick``,
        ``component_id``, ``detail``, ``schema_id``, or ``clock_domain``, and
        dropping them would be the silent field loss the contract forbids
        (section 5: typed fault records must not silently lose provenance, and
        the clock a fault's time belongs to is time semantics, not metadata).
        """
        origin = "recorder" if fault.origin == 1 else "runtime"
        reason = _RECORDER_FAULT_REASONS.get(fault.recorder_reason, str(fault.recorder_reason))
        native_provenance = {
            "native_session_id": fault.native_session_id,
            "runtime_generation": fault.runtime_generation,
            "frame_sequence": fault.frame_sequence,
            "sample_index": fault.sample_idx,
            "device_tick": fault.device_tick,
            "detected_at_ns": fault.detected_at_ns,
            "data_ordinal_at_fault": fault.data_ordinal_at_fault,
            "control_ordinal_at_fault": fault.control_ordinal_at_fault,
            "component_id": fault.component_id,
            "detail": fault.detail,
            "schema_id": fault.schema_id,
            "clock_domain": fault.clock_domain,
            "signal_id": fault.signal_id,
            "fault_code": fault.fault_code,
            "stream_status": fault.stream_status,
            "fault_stage": fault.fault_stage,
            "origin": fault.origin,
            "recorder_reason": fault.recorder_reason,
        }
        fault_id = self._append_fault_row(
            writer,
            time_ns=fault.detected_at_ns,
            code=f"{origin}.{reason}"
            if fault.origin == 1
            else f"{origin}.fault-{fault.fault_code}",
            stage=_FAULT_STAGES.get(fault.fault_stage, str(fault.fault_stage)),
            frame_sequence=fault.frame_sequence or None,
            signal_id=fault.signal_id or None,
            text=json.dumps(native_provenance, sort_keys=True, separators=(",", ":")),
        )
        if self._primary_fault_id is None:
            self._primary_fault_id = fault_id


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


#: ``RecorderFaultReason``, by value (``record_payloads.h``).
_RECORDER_FAULT_REASONS: Mapping[int, str] = {
    0: "none",
    1: "data_queue_saturated",
    2: "control_queue_saturated",
    3: "plan_violation",
    4: "spool_writer_failed",
    5: "drain_timeout",
    6: "edge_rejection",
    7: "readiness_gate_failed",
}

#: ``FaultStage``, by value (``cpp/include/neurale/streaming/fault.h``).
_FAULT_STAGES: Mapping[int, str] = {
    0: "source",
    1: "continuity",
    2: "processor",
    3: "output",
    4: "consumer",
    5: "actuator",
    6: "observer",
    7: "runtime",
}


# --- the accounting summary -------------------------------------------------


#: The ``ordinal`` form of a first-failed position (section 4.4). A loss after
#: acceptance names the accepted item's ordinal; a rejection before acceptance
#: never received one and names the producer identity instead, so the two forms
#: are read through different fields and never compared with each other.
_POSITION_TAG_ORDINAL: Final = 1


def _missing_samples_known(gap: spool.SignalGapPayload) -> bool:
    """Whether the gap measured how many samples were missing.

    A gap that could not measure it writes null rather than a zero, because zero
    missing samples is a different statement from an unmeasured count.
    """
    return bool(gap.gap_flags & spool.GAP_FLAG_MISSING_SAMPLES_KNOWN)


def _position_ordinal(pos: spool.SpoolPosition) -> int | None:
    """Return a loss position's ordinal, or null when there was no loss."""
    return pos.ordinal if pos.tag == _POSITION_TAG_ORDINAL else None


def _accounting_row(
    scan: spool.SpoolScan,
    counts: FinalizationCounts,
    *,
    termination_origin: str,
) -> dict[str, Any]:
    """Build the one accounting row, from the spool's snapshot plus what was written.

    The four finalization-side counters are supplied here and nowhere else: the
    spool is written before finalization runs, so a snapshot carrying them would
    carry numbers nothing had measured (spool specification section 4.4).

    ``accounting_verified`` is deliberately absent, and adding it would be a
    contract violation rather than an omission: it is a judgement the reader
    answering the call derives from both verification layers, and a stored copy
    would let an artifact assert a verdict its reader disagrees with.
    """
    snapshot = scan.accounting
    if snapshot is None:
        # Nothing sealed this session, so there is no acceptance accounting to
        # carry forward. What survived is stated, and `producer_acceptance_known`
        # is what says the acceptance columns are the surviving prefix rather
        # than evidence of what a producer handed over (contract section 5.1).
        return {
            "accounting_id": ["accounting-0001"],
            "accounting_origin": ["recovery_rebuilt"],
            "termination_origin": [termination_origin],
            "producer_acceptance_known": [False],
            "runtime_accepted": [scan.data_items],
            "recorder_accepted": [scan.data_items],
            "spool_committed": [scan.data_items],
            "nrf_committed": [counts.data_items],
            "rejected_before_runtime_acceptance": [0],
            "failed_between_runtime_and_recorder": [0],
            "lost_between_recorder_and_spool": [0],
            "lost_during_finalization": [scan.data_items - counts.data_items],
            "control_offered": [None],
            "control_accepted": [scan.control_items],
            "control_spool_committed": [scan.control_items],
            "control_nrf_committed": [counts.control_records],
            "control_rejected": [0],
            "lost_between_control_acceptance_and_spool": [0],
            "control_lost_during_finalization": [scan.control_items - counts.control_records],
            "data_first_lost_ordinal": [None],
            "data_first_rejected_message_kind": [None],
            "data_first_rejected_frame_sequence": [None],
            "control_first_lost_ordinal": [None],
            "control_first_rejected_kind": [None],
            "control_first_rejected_identity": [None],
            "data_rejected_after_close": [0],
            "control_rejected_after_close": [0],
        }

    return {
        "accounting_id": ["accounting-0001"],
        "accounting_origin": ["recorder"],
        "termination_origin": [termination_origin],
        "producer_acceptance_known": [snapshot.producer_acceptance_known],
        "runtime_accepted": [snapshot.runtime_accepted],
        "recorder_accepted": [snapshot.recorder_accepted],
        "spool_committed": [snapshot.spool_committed],
        "nrf_committed": [counts.data_items],
        "rejected_before_runtime_acceptance": [snapshot.rejected_before_runtime_acceptance],
        "failed_between_runtime_and_recorder": [snapshot.failed_between_runtime_and_recorder],
        "lost_between_recorder_and_spool": [snapshot.lost_between_recorder_and_spool],
        "lost_during_finalization": [snapshot.spool_committed - counts.data_items],
        "control_offered": [snapshot.control_offered if snapshot.control_offered_present else None],
        "control_accepted": [snapshot.control_accepted],
        "control_spool_committed": [snapshot.control_spool_committed],
        "control_nrf_committed": [counts.control_records],
        "control_rejected": [snapshot.control_rejected],
        "lost_between_control_acceptance_and_spool": [
            snapshot.lost_between_control_acceptance_and_spool
        ],
        "control_lost_during_finalization": [
            snapshot.control_spool_committed - counts.control_records
        ],
        "data_first_lost_ordinal": [_position_ordinal(snapshot.data_first_loss)],
        # The registry number is what crosses from the spool to the NRF kind
        # string; rendering it here is the only conversion the contract fixes.
        "data_first_rejected_message_kind": [
            spool.producer_identity_name(snapshot.data_first_rejection.identity_kind)
        ],
        "data_first_rejected_frame_sequence": [
            snapshot.data_first_rejection.identity_value
            if snapshot.data_first_rejection.present
            else None
        ],
        "control_first_lost_ordinal": [_position_ordinal(snapshot.control_first_loss)],
        "control_first_rejected_kind": [
            spool.producer_identity_name(snapshot.control_first_rejection.identity_kind)
        ],
        # Unsigned decimal ASCII with no leading zeros, as section 4.4 fixes it.
        "control_first_rejected_identity": [
            str(snapshot.control_first_rejection.identity_value)
            if snapshot.control_first_rejection.present
            else None
        ],
        "data_rejected_after_close": [snapshot.rejected_after_close_data],
        "control_rejected_after_close": [snapshot.rejected_after_close_control],
    }


def _check_accounting_row(row: Mapping[str, Any], counts: FinalizationCounts) -> None:
    """Layer 1 over the row about to be written, before it is written.

    The identities are adjacent stages only, and a rejection appears in none of
    them: an item that was refused was never inside the stage it was refused
    entry to. Checking here rather than only at read time is what stops the
    finalizer from sealing a session with accounting that contradicts itself.
    """

    def value(name: str) -> Any:
        return row[name][0]

    identities = (
        (
            "runtime_accepted",
            value("runtime_accepted"),
            value("recorder_accepted") + value("failed_between_runtime_and_recorder"),
        ),
        (
            "recorder_accepted",
            value("recorder_accepted"),
            value("spool_committed") + value("lost_between_recorder_and_spool"),
        ),
        (
            "spool_committed",
            value("spool_committed"),
            value("nrf_committed") + value("lost_during_finalization"),
        ),
        (
            "control_accepted",
            value("control_accepted"),
            value("control_spool_committed") + value("lost_between_control_acceptance_and_spool"),
        ),
        (
            "control_spool_committed",
            value("control_spool_committed"),
            value("control_nrf_committed") + value("control_lost_during_finalization"),
        ),
    )
    if value("control_offered") is not None:
        identities += (
            (
                "control_offered",
                value("control_offered"),
                value("control_accepted") + value("control_rejected"),
            ),
        )
    for name, left, right in identities:
        if left != right:
            raise FinalizationError(
                f"the accounting summary this session would carry is not internally consistent: "
                f"{name} is {left} but its next stage plus its loss term is {right}",
                category="accounting",
                retryable=False,
            )
    if value("nrf_committed") != counts.data_items:
        raise FinalizationError(
            f"the accounting summary claims {value('nrf_committed')} data items in NRF but "
            f"{counts.data_items} were written",
            category="accounting",
            retryable=False,
        )
    if value("control_nrf_committed") != counts.control_records:
        raise FinalizationError(
            f"the accounting summary claims {value('control_nrf_committed')} control records in "
            f"NRF but {counts.control_records} were written",
            category="accounting",
            retryable=False,
        )
    for name in ("lost_during_finalization", "control_lost_during_finalization"):
        if value(name) < 0:
            raise FinalizationError(
                f"the finalizer wrote more items than the spool committed: {name} is {value(name)}",
                category="accounting",
                retryable=False,
            )


# --- the conversion ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SessionOutcomes:
    """The three outcome dimensions of contract section 3.2, kept separate.

    ``capture_outcome`` is frozen at the spool's session-end record and never
    moves; ``effective_session_outcome`` is what the NRF ``termination_kind``
    states and may escalate to ``faulted`` when finalization proves required data
    lost or a committed fault row the capture did not record; and
    ``requested_terminal_intent`` is what the first exit from ``recording``
    latched, or ``None`` when no session-end record froze it (recovery). They
    are separate fields because escalation after the capture is over has to be
    expressible, and collapsing them would publish a session known to be short
    as normal or erase that the recording ran correctly.
    """

    capture_outcome: str
    requested_terminal_intent: str | None
    effective_session_outcome: str
    primary_fault_row_committed: bool
    termination_origin: str
    source_session_end_present: bool
    reason: str

    @property
    def termination_kind(self) -> str:
        return self.effective_session_outcome


def _resolve_outcomes(scan: spool.SpoolScan, faults_committed: bool) -> _SessionOutcomes:
    """Derive the base outcome dimensions from the spool, before finalization loss.

    A committed prefix with no session-end record is finalized as abnormal and
    incomplete, never as normal (contract section 4.5): ``capture_outcome`` is
    ``unknown`` because nothing froze it, and the effective outcome is derived
    from what is on disk -- ``faulted`` when a committed fault row exists,
    ``aborted`` otherwise. With a session-end record, ``capture_outcome`` is the
    frozen value and the effective outcome starts equal to it, then escalates to
    ``faulted`` when a committed fault row the capture did not already record
    proves the session faulted: a fault is a fact about the session that outranks
    the capture's claim, and this is the case the contract must be able to state.
    Finalization-time required-data loss escalates the effective outcome further,
    in :func:`_convert`, once the counts that measure it exist.
    """
    end = scan.session_end
    if end is not None and end.capture_outcome_name is not None:
        capture = end.capture_outcome_name
        effective = "faulted" if (faults_committed and capture != "faulted") else capture
        return _SessionOutcomes(
            capture_outcome=capture,
            requested_terminal_intent=end.requested_terminal_intent_name,
            effective_session_outcome=effective,
            primary_fault_row_committed=bool(end.primary_fault_committed),
            termination_origin="recorder",
            source_session_end_present=True,
            reason=end.terminal_reason or f"capture ended {capture}",
        )
    return _SessionOutcomes(
        capture_outcome="unknown",
        requested_terminal_intent=None,
        effective_session_outcome="faulted" if faults_committed else "aborted",
        primary_fault_row_committed=faults_committed,
        termination_origin="recovery",
        source_session_end_present=scan.session_end_present,
        reason="the spool carries no session end; finalized as abnormal and incomplete",
    )


def _native_replay_termination_extension(
    scan: spool.SpoolScan, outcomes: _SessionOutcomes
) -> dict[str, Any]:
    """Build the ``neurale.native_replay`` journal termination extension.

    NRF v1's ``termination_kind`` is a closed enum, so the dimensions a closed
    enum cannot carry -- the frozen capture outcome, the latched request, the
    effective outcome that may have escalated past it, and whether the primary
    fault row reached a committed extent -- travel in the journal record's
    ``extensions`` member, signed by the record checksum. ``finalization_status``
    is ``succeeded`` because this is written at seal time; a failed attempt
    publishes nothing and leaves its status in the progress document instead.

    This is written only when the spool's session-end record froze the outcome
    dimensions -- ``requested_terminal_intent`` is latched and non-null, and
    ``capture_outcome`` is a known value, both of which the frozen schema
    requires. A recovered spool (no session-end record) has neither latched, so
    the frozen ``neurale.native_replay`` schema cannot carry it; the recovery
    termination record instead carries a ``neurale.recovery`` extension (see
    :func:`_recovery_termination_extension`) that declares the recovery origin in
    the record itself, as contract section 4.5 requires.

    ``termination_origin`` is required on **both** kinds of termination
    (contract section 4.5), ``"recorder"`` in the ordinary case. Letting a reader
    infer the origin from the absence of a recovery extension would put
    provenance in a missing field, which is the one place a checksum cannot sign
    it.
    """
    superblock = scan.superblock
    assert superblock is not None
    return {
        "extension_version": NATIVE_REPLAY_EXTENSION_VERSION,
        "plan_fingerprint": superblock.plan_fingerprint,
        "termination_origin": outcomes.termination_origin,
        "requested_terminal_intent": outcomes.requested_terminal_intent,
        "capture_outcome": outcomes.capture_outcome,
        "effective_session_outcome": outcomes.effective_session_outcome,
        "finalization_status": "succeeded",
        "primary_fault_row_committed": outcomes.primary_fault_row_committed,
    }


def _last_host_time(inventory: _Inventory) -> int:
    """The latest host-monotonic instant the committed prefix recorded.

    The NRF termination is timestamped on the session's host-monotonic clock,
    and the spool's ``session_end`` carries wall-clock nanoseconds, so the one
    cannot be copied into the other -- that is the timestamp-domain error, not a
    naming difference. What is available is the last host-monotonic instant the
    session actually recorded, which is a real measurement and an honest lower
    bound on when the session ended. The wall-clock value is not discarded: it
    is carried in the finalization extension, under a name that says what it is.
    """
    latest = inventory.last_data_host_time
    for control in inventory.controls:
        latest = max(latest, control.time_ns)
    for fault in inventory.faults:
        latest = max(latest, fault.detected_at_ns)
    return latest


def _finalization_extension(
    scan: spool.SpoolScan,
    options: FinalizerOptions,
    outcomes: _SessionOutcomes,
) -> dict[str, Any]:
    superblock = scan.superblock
    assert superblock is not None
    end = scan.session_end
    recovery: dict[str, Any] | None = None
    if scan.accounting is None:
        # What recovery *found*, stated as what it found (contract section 5.1).
        # These are deliberately not accounting columns: an item the producer
        # handed over and the worker had not committed when the process died
        # leaves nothing on disk to count, so a rebuilt summary that printed a
        # precise `runtime_accepted` would be inventing it. Every value here is a
        # measurement of the surviving prefix, named so it can never be read as
        # acceptance. The third value the contract names, `known_nrf_committed`,
        # is not repeated here: it is the summary's own `nrf_committed` column,
        # which on a rebuilt summary is exactly that measurement, and a second
        # copy would be a second place for it to disagree.
        recovery = {
            "known_spool_committed": scan.data_items,
            "known_control_spool_committed": scan.control_items,
            "known_surviving_prefix": scan.committed_prefix_end,
        }
    return {
        **({"recovery_rebuilt_accounting": recovery} if recovery is not None else {}),
        "extension_version": FINALIZATION_EXTENSION_VERSION,
        "finalizer": {"name": options.writer_name, "version": options.writer_version},
        "termination_origin": outcomes.termination_origin,
        # Required by contract section 5: an NRF termination whose origin is
        # recovery must say so in the record itself, and a faulted session whose
        # primary fault row never committed must declare the row missing. This
        # mirrors the journal extension's ``primary_fault_row_committed`` so the
        # two namespaces agree about whether the recorder's own fault row reached
        # a committed extent.
        "source_session_end_present": outcomes.source_session_end_present,
        "primary_fault_record_committed": outcomes.primary_fault_row_committed,
        # A decimal string, not a number: wall-clock nanoseconds are past
        # I-JSON's exact integer range, and a JSON number there is a value two
        # parsers can disagree about. The unit is in the name so the string is
        # not mistaken for an opaque token.
        "source_end_unix_nanos": str(end.end_unix_nanos) if end is not None else None,
        "spool": {
            "version": {"major": superblock.version_major, "minor": superblock.version_minor},
            "session_uuid": superblock.session_uuid.hex(),
            "plan_fingerprint": superblock.plan_fingerprint,
            "durability_policy": superblock.durability_policy_name,
            "committed_prefix_end": scan.committed_prefix_end,
            "committed_transactions": scan.committed_transactions,
            "durable_extent_bytes": scan.durable_extent_bytes,
            "tail_status": scan.status,
            "diagnostics": list(scan.codes),
        },
    }


#: Accounting columns whose non-zero value proves required data was lost at an
#: acceptance stage. None of them is a "rejected after close" counter (API
#: misuse, not a recording loss -- contract section 1.3), and none may be
#: sealed as a merely normal or aborted session: a session that lost required
#: data is not complete (contract section 3.2), and finalization proving it
#: escalates the effective outcome to ``faulted`` while the frozen capture
#: outcome is left exactly as the session-end record latched it (rule 4).
_REQUIRED_DATA_LOSS_COLUMNS = (
    "rejected_before_runtime_acceptance",
    "failed_between_runtime_and_recorder",
    "lost_between_recorder_and_spool",
    "control_rejected",
    "lost_between_control_acceptance_and_spool",
    "lost_during_finalization",
    "control_lost_during_finalization",
)


def _required_data_lost(row: Mapping[str, Any]) -> bool:
    """True when the accounting row proves required data was lost."""
    return any(int(row[name][0]) > 0 for name in _REQUIRED_DATA_LOSS_COLUMNS)


def _escalate_outcomes(outcomes: _SessionOutcomes, row: Mapping[str, Any]) -> _SessionOutcomes:
    """Escalate the effective outcome to ``faulted`` on proven required-data loss.

    Finalization is the point that proves a clean capture turned out short: the
    accounting row's loss counters are measured here, and any non-zero one is
    required data the session cannot stand behind. The capture outcome stays
    frozen (it records what the run itself ended as), and only the effective
    outcome -- what the NRF ``termination_kind`` states -- escalates, exactly the
    separation section 3.2 rule 4 requires. Escalation is one-way: a session
    already ``faulted`` is unchanged, and a session that proved no loss is
    unchanged.
    """
    if outcomes.effective_session_outcome == "faulted":
        return outcomes
    if not _required_data_lost(row):
        return outcomes
    return _SessionOutcomes(
        capture_outcome=outcomes.capture_outcome,
        requested_terminal_intent=outcomes.requested_terminal_intent,
        effective_session_outcome="faulted",
        primary_fault_row_committed=outcomes.primary_fault_row_committed,
        termination_origin=outcomes.termination_origin,
        source_session_end_present=outcomes.source_session_end_present,
        reason=(
            "finalization proved required data was lost -- the accounting summary the "
            "session carries records a loss at an acceptance stage, so the session is "
            "sealed as faulted though the capture outcome is left as the session-end "
            "record froze it"
        ),
    )


def _recovery_termination_extension(outcomes: _SessionOutcomes) -> dict[str, Any]:
    """Build the journal ``neurale.recovery`` termination extension.

    Contract section 4.5 requires a termination record the finalizer writes for
    a spool with no session-end record to declare its own origin in the record's
    ``extensions``: the recovery origin cannot live only in the manifest, because
    the termination record is the thing that says the session ended, and a
    reader must be able to tell a termination the recorder wrote from one
    recovery reconstructed without consulting a separate object. NRF v1's
    ``termination_kind`` carries the effective outcome (``faulted`` when a
    committed fault caused the end, ``aborted`` otherwise, never ``normal``),
    and the dimensions a closed enum cannot carry travel under
    ``neurale.recovery``, signed by the termination record's checksum. The key
    spelling is what the contract's section 4.5 example fixes.
    """
    return {
        "termination_origin": "recovery",
        "source_session_end_present": outcomes.source_session_end_present,
        "recovery_reason": "process_crash",
        "capture_outcome": outcomes.capture_outcome,
    }


def _convert(
    scan: spool.SpoolScan,
    plan: RecordingPlan,
    inventory: _Inventory,
    options: FinalizerOptions,
    target: Path,
    progress_callback=None,
) -> tuple[FinalizationCounts, _SessionOutcomes]:
    """Write the whole session into *target* and seal it.

    The termination's ``kind`` is the *effective* session outcome, not the
    frozen capture outcome: :func:`_resolve_outcomes` separates the two, and
    :func:`_escalate_outcomes` then escalates the effective outcome to
    ``faulted`` once the accounting row proves required data was lost -- the one
    case the capture dimension cannot state on its own, because the loss is
    measured at finalization. The frozen capture outcome, the latched request,
    and whether the primary fault row committed travel in the journal's
    ``neurale.native_replay`` extension (signed by the termination record's
    checksum) when a session-end record froze them; a recovered spool has none of
    those latched, and its termination record carries a ``neurale.recovery``
    extension declaring the recovery origin in the record itself, as contract
    section 4.5 requires -- not only in the manifest.
    """
    state = _Finalization(scan, plan, options, inventory)
    superblock = scan.superblock
    assert superblock is not None
    outcomes = _resolve_outcomes(scan, bool(inventory.faults))
    kind = outcomes.termination_kind
    streams, control_chunk_length = state._session_build_inputs()

    writer = NrfWriter.create(
        target,
        session_id=plan.session.session_id,
        created_at=plan.session.created_at,
        writer_name=plan.session.writer_name,
        writer_version=plan.session.writer_version,
        session_metadata=plan.session_metadata.document(),
        metadata=plan.metadata.document(),
        extensions={
            **plan.manifest_extensions(),
            FINALIZATION_NAMESPACE: _finalization_extension(scan, options, outcomes),
        },
        minor_version=plan.nrf_minor_version,
    )
    writer._progress_callback = progress_callback
    try:
        from neurale.io.nrf._package import PayloadArchive

        writer._payload_archive = PayloadArchive(writer._root / "payloads.packing")
        state._session = build_session(
            writer.registry,
            _schema_view(plan),
            streams,
            control_chunk_length=control_chunk_length,
        )
        state._register_ledgers(writer)
        writer.freeze()

        for item in _data_items(scan):
            if isinstance(item, _Frame):
                state._write_frame(writer, item)
            else:
                state._write_discontinuity(writer, item)
        for control in inventory.controls:
            state._write_control(writer, control)
        for fault in inventory.faults:
            state._write_recorder_fault(writer, fault)

        # The accounting row measures finalization loss and so must be built
        # before the effective outcome is decided: a clean capture whose
        # artifact turns out short is the case rule 4 separates the fields for,
        # and the loss counters exist only here. It uses only the data and
        # control item counts, which the synthetic fault row below does not
        # change (a fault is neither a data nor a control item), so it is safe
        # to build and check before that row is written.
        provisional = FinalizationCounts(
            frames=state._frames,
            signal_blocks=state._blocks,
            discontinuities=state._discontinuities,
            signal_gaps=state._gaps,
            control_records=state._controls,
            fault_rows=state._fault_rows,
            stream_rows=dict(state._rows),
            stream_discontinuity_rows=dict(state._stream_discontinuity_rows),
            record_rows=dict(state._record_rows),
        )
        row = _accounting_row(scan, provisional, termination_origin=outcomes.termination_origin)
        _check_accounting_row(row, provisional)
        outcomes = _escalate_outcomes(outcomes, row)
        kind = outcomes.termination_kind

        if kind == "faulted" and state._primary_fault_id is None:
            # NRF requires a faulted termination to name a fault row. The
            # effective outcome is faulted -- the session-end record says so, a
            # committed fault row proves it, or the accounting proved required
            # data lost -- but no committed primary fault record backs it. The
            # row is the finalizer's, written from that evidence and labelled as
            # such, and the manifest extension records that the recorder's own
            # fault row is missing.
            #
            # The termination names *this* row, not the session's first fault
            # row: a caller may have submitted a control-plane fault long before
            # the capture went wrong, and pointing the termination at it would
            # make the artifact claim an unrelated event ended the session.
            state._primary_fault_id = state._append_fault_row(
                writer,
                time_ns=_last_host_time(inventory),
                code=FAULT_CODE_CAPTURE_FAULTED,
                stage="runtime",
                frame_sequence=None,
                signal_id=None,
                text=(
                    "the session's effective outcome is faulted and no committed primary "
                    f"fault record backs it; terminal reason: {outcomes.reason}"
                ),
            )

        # Rebuild the counts after the synthetic fault row so the cross-check
        # compares against the rows actually written, including that row; the
        # data and control item counts it shares with ``provisional`` are
        # unchanged, so the accounting row already built stays consistent.
        counts = FinalizationCounts(
            frames=state._frames,
            signal_blocks=state._blocks,
            discontinuities=state._discontinuities,
            signal_gaps=state._gaps,
            control_records=state._controls,
            fault_rows=state._fault_rows,
            stream_rows=dict(state._rows),
            stream_discontinuity_rows=dict(state._stream_discontinuity_rows),
            record_rows=dict(state._record_rows),
        )
        # The summary is appended here and the termination seals every open
        # target in one transaction, which is contract section 5.1's rule that
        # the accounting is written in the same transaction that seals the
        # session -- a session cannot be sealed with accounting that disagrees
        # with it, or with none at all.
        writer.append_records(SESSION_ACCOUNTING.schema_id, row)

        # The outcome dimensions a closed ``termination_kind`` enum cannot carry
        # travel in the journal termination record's signed ``extensions``. When
        # the session-end record froze them -- ``requested_terminal_intent``
        # latched and non-null, ``capture_outcome`` a known value -- they ride in
        # the ``neurale.native_replay`` extension. A recovered spool has none of
        # those latched, and the frozen schema cannot carry it, so the finalizer
        # writes the ``neurale.recovery`` keys the contract's section 4.5 example
        # fixes -- ``termination_origin`` present in the record, not only the
        # manifest.
        # The namespace is chosen by the origin, not by a condition that merely
        # implies it. The two agree in every spool that reaches here -- a
        # session-end record whose outcome is undefined is not finalizable -- but
        # keying on the origin makes the namespace and the value it carries one
        # decision rather than two that have to be kept in step, which is the
        # invariant a reader checks: neurale.native_replay is a recorder-written
        # termination and neurale.recovery is a recovery-written one.
        termination_extensions = (
            {NATIVE_REPLAY_NAMESPACE: _native_replay_termination_extension(scan, outcomes)}
            if outcomes.termination_origin == "recorder"
            else {RECOVERY_NAMESPACE: _recovery_termination_extension(outcomes)}
        )
        writer.finalize(
            kind=kind,
            reason=outcomes.reason,
            time_ns=_last_host_time(inventory),
            fault_id=state._primary_fault_id if kind == "faulted" else None,
            extensions=termination_extensions,
        )
    finally:
        writer.close()
    return counts, outcomes


def _cross_check(
    target: Path, counts: FinalizationCounts, termination_kind: str = "normal"
) -> None:
    """Compare what was written against what the artifact says it holds.

    This is contract section 5.1's layer 2, run by the writer before it
    publishes rather than left for a reader to discover: the identities alone
    are satisfied by a session claiming a hundred items while holding
    ninety-nine, and the committed extents are the only independent statement of
    what is actually there.

    The diagnosis that follows is validation of the *artifact*, not of the
    recording. ``session_terminated_abnormally`` over a session the finalizer
    deliberately terminated as ``aborted`` or ``faulted`` is the diagnosis
    agreeing with the artifact, and refusing to publish there would make an
    abnormal session impossible to finalize -- which is the one case the spool
    exists for.
    """
    from neurale.io.nrf import NrfReader

    with NrfReader.open(target, verify_checksums=False) as reader:
        manifest = reader.manifest
    extents = manifest["commit"]["committed_extents"]
    expected = {
        LEDGERS_BY_KIND["native_frames"].path: counts.frames,
        LEDGERS_BY_KIND["native_signal_blocks"].path: counts.signal_blocks,
        LEDGERS_BY_KIND["native_discontinuities"].path: counts.discontinuities,
        LEDGERS_BY_KIND["native_signal_gaps"].path: counts.signal_gaps,
        LEDGERS_BY_KIND["session_accounting"].path: 1,
    }
    streams = {stream["id"]: stream for stream in manifest["streams"]}
    for stream_id, rows in counts.stream_rows.items():
        expected[streams[stream_id]["data"]["path"]] = rows
    record_schemas = {schema["id"]: schema for schema in manifest["record_schemas"]}
    for schema_id, rows in counts.record_rows.items():
        expected[record_schemas[schema_id]["path"]] = rows
    for name, rows in expected.items():
        actual = int(extents.get(name, 0))
        if actual != rows:
            raise FinalizationError(
                f"the finalizer wrote {rows} rows to {name} but the sealed session reports "
                f"{actual}; refusing to publish a session whose accounting it cannot stand behind",
                category="accounting",
                retryable=False,
            )

    diagnosis = diagnose_session(target)
    expected_codes = {"session_terminated_abnormally"} if termination_kind != "normal" else set()
    unexpected = sorted({item.code for item in diagnosis.diagnostics} - expected_codes)
    if unexpected:
        raise FinalizationError(
            "the session this finalizer just wrote does not validate: " + ", ".join(unexpected),
            category="validation",
            retryable=False,
        )


# --- the public entry point -------------------------------------------------


def _scan_source(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> tuple[spool.SpoolScan, Path | None]:
    """Scan the spool, without ever holding all of it.

    A path is scanned in place: the reader opens it read-only and walks it with
    a fixed buffer, so the peak cost of a finalization does not include a copy
    of its own input. Reading the file into memory first would make that copy
    the floor, on top of the decoded prefix the conversion then builds, for a
    spool the reader is about to walk sequentially anyway.

    Bytes a caller already holds are scanned where they are.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return spool.scan_spool(bytes(source)), None
    path = Path(source)
    return spool.scan_spool_path(str(path)), path


def _resume_or_start(staging: Path, expected: FinalizationProgress) -> FinalizationProgress:
    """Return the progress to record for this attempt, refusing a mismatch.

    Contract section 7 allows a resume only when the superblock validates, the
    session id and plan fingerprint match the finalization target, and the
    recorded progress identifies the last consumed transaction. **Any** mismatch
    of session id, plan fingerprint, output format version, or committed extents
    rejects the resume rather than reconciling it -- extents included, and that
    is not a formality: a source that grew between the two attempts is a
    different source in the only sense that matters, because the staged session
    and the cursor into it were produced from a prefix that is no longer the one
    being handed over. Reconciling would silently finalize a session under a
    progress record describing something else.
    """
    progress_path = staging / PROGRESS_FILE
    if not progress_path.exists():
        return expected
    previous = FinalizationProgress.load(progress_path)
    mismatches = [
        name
        for name in (
            "session_id",
            "plan_fingerprint",
            "spool_session_uuid",
            "nrf_minor_version",
            "committed_prefix_end",
            "committed_transactions",
        )
        if getattr(previous, name) != getattr(expected, name)
    ]
    if mismatches:
        raise FinalizationError(
            f"{staging} holds the progress of a finalization of a different source "
            f"({', '.join(mismatches)} differ); resolve or move it rather than resuming into it",
            category="resume_mismatch",
            retryable=False,
        )
    if previous.status == "failed":
        # ``failed`` and ``failed_retryable`` are already distinguished by the
        # attempt that wrote them, and re-merging them here would undo that: a
        # non-retryable failure is one a second attempt reads exactly the same
        # way -- a source whose committed prefix cannot be accounted for, an
        # accounting mismatch -- so resuming would spend the work to reach the
        # same refusal and leave a caller believing recovery was tried.
        raise FinalizationError(
            f"{staging} records a finalization that failed in a way retrying cannot fix"
            + (f" ({previous.category}: {previous.detail})" if previous.detail else "")
            + "; diagnose the source rather than resuming",
            category="resume_mismatch",
            retryable=False,
        )
    if previous.status == "succeeded":
        raise FinalizationError(
            f"{staging} records a finalization that already succeeded; a second attempt would "
            "produce a second session from one spool",
            category="resume_mismatch",
            retryable=False,
        )
    if previous.status == "abandoned":
        # Abandonment is a deliberate terminal decision, and contract section 5.1
        # gives an abandoned finalization no completeness verdict, ever. Silently
        # resuming into it would erase the record of that decision, so restarting
        # is made an explicit act: remove the staging directory, or finalize into
        # a different output.
        raise FinalizationError(
            f"{staging} records a finalization that was abandoned"
            + (f" ({previous.detail})" if previous.detail else "")
            + "; remove the staging directory or choose another output to start again",
            category="abandoned",
            retryable=False,
        )
    from dataclasses import replace

    # The cursor and what it points at carry forward. Every identity above has
    # matched, so the staged session the previous attempt produced was produced
    # from *this* source, and discarding the cursor here is what would turn a
    # resume back into a restart.
    return replace(
        expected,
        attempts=previous.attempts + 1,
        last_consumed_transaction_id=previous.last_consumed_transaction_id,
        staged=previous.staged,
    )


def _write_progress(staging: Path, progress: FinalizationProgress) -> None:
    """Replace the progress document durably, never in place.

    The progress document is the only witness a crash leaves, so the write that
    produces it must not be the write that destroys it. Overwriting the file in
    place has a window in which the process can die with a truncated document on
    disk -- and a document that will not parse is worse than the one it replaced,
    because recovery then reports "no finalization to resume" over an attempt
    that is sitting right there. The temporary file is fsynced before the rename
    and the directory after it, so a crash leaves either the previous valid
    document or the new one, never half of either.
    """
    staging.mkdir(parents=True, exist_ok=True)
    _write_durably(staging / PROGRESS_FILE, canonical_json_bytes(progress.document()))


def _write_durably(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable where the platform supports it.

    Best effort on purpose: opening a directory for reading is POSIX, and
    Windows refuses it. The rename is atomic on both, so a platform without a
    directory fsync still never sees a partial document -- it can only lose the
    most recent complete one to a power failure, which is the weaker guarantee
    that platform offers for every file it stores.
    """
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(handle)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(handle)


def _publish_session(staged: Path, target: Path) -> None:
    """Atomically publish *staged*, tolerating transient Windows handle release."""
    attempts = _WINDOWS_PUBLICATION_RETRIES if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            if os.name == "nt":
                os.rename(staged, target)
            else:
                os.link(staged, target)
                # Publication succeeded; a stale staging link is cleanup, not failure.
                try:
                    staged.unlink()
                except OSError:
                    pass
            return
        except OSError as error:
            transient_windows_error = os.name == "nt" and getattr(error, "winerror", None) in {
                5,  # ERROR_ACCESS_DENIED
                32,  # ERROR_SHARING_VIOLATION
            }
            if not transient_windows_error or attempt + 1 == attempts:
                raise
            time.sleep(_WINDOWS_PUBLICATION_RETRY_SECONDS)


def finalize_spool(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    options: FinalizerOptions | None = None,
    dry_run: bool = False,
) -> FinalizationReport:
    """Convert one committed spool into a sealed NRF session at *output*.

    *source* is the spool's bytes or the path to a spool file. Passing bytes is
    what makes the memory store finalizable at all -- it has no path -- and a
    path is additionally what a retention policy can act on, because deleting
    something requires knowing where it is.

    With ``dry_run`` the call is read-only in every branch: it scans the source,
    rebuilds the plan, validates the committed prefix's topology, and reports
    what a real attempt would produce -- without creating the staging directory,
    the progress document, or the session. It proposes; it writes nothing.

    Raises :class:`~neurale.recording.SpoolSourceError` when the spool cannot be
    finalized at all and :class:`~neurale.recording.FinalizationError` when this
    attempt did not finish. Neither says anything about how the capture ended.
    """
    started = time.monotonic()
    settings = options or FinalizerOptions()
    target = Path(output)
    staging = target.with_name(target.name + STAGING_SUFFIX)
    staged_session = staging / STAGING_SESSION

    scan, source_path = _scan_source(source)
    sidecar_plan = (
        _read_plan_sidecar(source_path)
        if source_path is not None and settings.recording_plan is None
        else None
    )
    if sidecar_plan is not None:
        settings = FinalizerOptions(
            writer_name=settings.writer_name,
            writer_version=settings.writer_version,
            stream_chunk_length=settings.stream_chunk_length,
            block_index_chunk_length=settings.block_index_chunk_length,
            control_chunk_length=settings.control_chunk_length,
            spool_retention=settings.spool_retention,
            metadata=settings.metadata,
            session_metadata=settings.session_metadata,
            recording_plan=sidecar_plan,
        )
    if scan.superblock is None:
        raise SpoolSourceError(
            f"the spool's superblock did not validate ({', '.join(scan.codes)}); nothing in it "
            "is readable and no finalization is possible"
        )
    if not scan.finalizable:
        raise SpoolSourceError(
            f"the spool's committed prefix cannot be accounted for ({', '.join(scan.codes)}); "
            "diagnose or quarantine it rather than promoting a prefix nobody can verify"
        )

    superblock = scan.superblock
    try:
        if settings.recording_plan is None:
            plan = plan_from_spool_document(
                superblock.plan_document,
                session=SessionIdentity(
                    session_id=superblock.session_id,
                    created_at=_created_at(superblock.created_unix_nanos),
                    writer_name=settings.writer_name,
                    writer_version=settings.writer_version,
                ),
                resource_bounds=_bounds_for(scan, settings),
                session_metadata=PreparedMetadata.of(dict(settings.session_metadata)),
                metadata=PreparedMetadata.of(dict(settings.metadata)),
            )
        else:
            plan = settings.recording_plan
            if plan.session.session_id != superblock.session_id:
                raise RecorderConfigError(
                    "the supplied recording plan belongs to a different session"
                )
            if canonical_json_bytes(plan.document()) != superblock.plan_document:
                raise RecorderConfigError(
                    "the supplied recording plan does not match the plan stored in the spool"
                )
            if plan.fingerprint != superblock.plan_fingerprint:
                raise RecorderConfigError(
                    "the supplied recording plan fingerprint does not match the spool"
                )
    except RecorderConfigError as error:
        raise SpoolSourceError(
            f"the plan the spool stores is not a recording plan: {error}"
        ) from error

    if target.exists():
        raise FinalizationError(
            f"{target} already exists; a finalizer never replaces a recorded session",
            category="publication",
            retryable=False,
        )

    inventory = _inventory(scan)
    _reject_unmaterializable_streams(plan)
    _validate_committed_prefix(scan, plan, inventory)
    progress = _resume_or_start(
        staging,
        FinalizationProgress(
            session_id=superblock.session_id,
            plan_fingerprint=superblock.plan_fingerprint,
            spool_session_uuid=superblock.session_uuid.hex(),
            nrf_minor_version=plan.nrf_minor_version,
            committed_prefix_end=scan.committed_prefix_end,
            committed_transactions=scan.committed_transactions,
            last_consumed_transaction_id=0,
            attempts=1,
            status="running",
        ),
    )
    if dry_run:
        # Nothing above this line wrote a byte: the source was read, the plan was
        # rebuilt, and the committed prefix's topology was validated. A dry run
        # stops here and reports what a real attempt would produce, which is what
        # makes it safe to run against a spool a caller is not ready to convert.
        return _dry_run_report(scan, inventory, progress, started)

    _write_progress(staging, progress)
    resumable = _staged_to_publish(progress, scan, staged_session)

    def package_progress(phase, done, total):
        nonlocal progress
        progress = replace(progress, phase=phase, processed_bytes=done, total_bytes=total)
        _write_progress(staging, progress)

    try:
        if resumable is None:
            # No conversion to continue from. Whatever is in the staging
            # workspace is a partial session -- :func:`_refuse_unrecorded_session`
            # has just proved it was never sealed -- so it is not committed valid
            # NRF data, and rewriting it from the start is what guarantees no
            # logical record is appended twice.
            _refuse_unrecorded_session(staged_session)
            if staged_session.exists():
                staged_session.unlink()
            # The cursor is cleared with the staged artifact it pointed into. Leaving
            # it set over a session that is being rewritten is what would let a
            # later resume publish a half-converted one.
            progress = _advance(staging, _without_cursor(progress), "converting")
            counts, outcomes = _convert(
                scan, plan, inventory, settings, staged_session, package_progress
            )
            staged = StagedConversion(
                counts=counts,
                termination_kind=outcomes.termination_kind,
                termination_origin=outcomes.termination_origin,
                source_session_end_present=outcomes.source_session_end_present,
                accounting_origin=(
                    "recorder" if scan.accounting is not None else "recovery_rebuilt"
                ),
            )
            # The cursor moves the instant the conversion returns, *before* the
            # session is cross-checked. It states what it says it states -- the
            # source has been materialized into the staged session -- and by the
            # time control reaches here that session is sealed: a crash between
            # the seal and this write would leave committed NRF data the next
            # attempt could only delete, which section 4.5 forbids. Verification
            # is not what the cursor records, and it is not skipped by recording
            # it early: the resume path below runs the same cross-check before
            # publishing anything.
            progress = _advance(
                staging,
                progress,
                "cross_checking",
                last_consumed_transaction_id=scan.last_transaction_id,
                staged=staged,
            )
            _cross_check(staged_session, counts, outcomes.termination_kind)
            progress = _advance(staging, progress, "publishing")
        else:
            # The previous attempt sealed this session and died before publishing
            # it. Contract section 4.5 forbids rewriting already committed valid
            # NRF data, so it is bound to this source, verified -- the same
            # cross-check, against the numbers that attempt recorded -- and
            # published, not rebuilt.
            staged = resumable
            counts = staged.counts
            _verify_staged_binding(staged_session, progress)
            progress = _advance(staging, progress, "cross_checking")
            _cross_check(staged_session, counts, staged.termination_kind)
            progress = _advance(staging, progress, "publishing")
        # Publication: from a package that has already been
        # written, sealed, and validated. Until this instant the target does not
        # exist, so there is no moment at which a reader can open a half-converted
        # session. Publication is inside the retryable state machine, not after
        # it: a rename failure is a retryable finalization failure rather than a
        # raw OSError, so the spool and the staged session are retained and the
        # progress document records the attempt instead of stalling at "running".
        try:
            _publish_session(staged_session, target)
            _fsync_directory(target.parent)
        except OSError as error:
            raise FinalizationError(
                f"publishing the finalized session to {target} failed: {error}",
                category="publication",
            ) from error
    except FinalizationError as error:
        _record_attempt_outcome(staging, progress, error)
        raise
    except Exception as error:
        wrapped = FinalizationError(
            f"finalizing the spool into {target} failed: {error}", category="writer"
        )
        _record_attempt_outcome(staging, progress, wrapped)
        raise wrapped from error

    finished = _with_status(
        progress,
        "succeeded",
        None,
        None,
        last_consumed_transaction_id=scan.last_transaction_id,
        phase="finalized",
    )
    spool_codes = scan.codes
    try:
        _write_progress(staging, replace(finished, phase="cleaning"))
    except OSError:
        # Publication has succeeded. A progress-file failure cannot undo it.
        pass
    retained = True
    if (
        settings.spool_retention == "delete_after_validated_finalization"
        and source_path is not None
    ):
        # Every condition of contract section 7 has been met by this point:
        # finalization produced a session, the session was validated, the
        # publication step succeeded, and the policy asks for deletion. The
        # order is the rule -- deleting the only reconstruction input because
        # finalization *appeared* to work is what it exists to prevent.
        #
        # Publication, spool deletion, and sidecar deletion are three separate
        # facts, and only the first is irreversible. The session is on disk and
        # sealed and the progress document already says so, so *no* failure
        # below may raise: an exception here would be latched by the caller as
        # a failed finalization over a session that succeeded, and the retry it
        # invites would then fail non-retryably on a target that already exists
        # -- a recorder reporting `failed` about a published NRF session. The
        # bundle is removed as one artifact, sidecar first (see
        # `discard_spool_bundle`), and whether that worked is reported as
        # `spool_retained` rather than allowed to mask the success.
        scan.close()
        retained = discard_spool_bundle(source_path) is not None
    shutil.rmtree(staging, ignore_errors=True)

    cleanup_paths = (staging,) if staging.exists() else ()
    if retained and source_path is not None and settings.spool_retention != "retain":
        cleanup_paths += tuple(
            p for p in (source_path, _plan_sidecar_path(source_path)) if p.exists()
        )

    return FinalizationReport(
        session_path=target,
        spool_retained=retained,
        cleanup_paths=cleanup_paths,
        counts=counts,
        progress=finished,
        termination_kind=staged.termination_kind,
        termination_origin=staged.termination_origin,
        source_session_end_present=staged.source_session_end_present,
        accounting_origin=staged.accounting_origin,
        spool_codes=spool_codes,
        duration_seconds=time.monotonic() - started,
    )


def _without_cursor(progress: FinalizationProgress) -> FinalizationProgress:
    """The same attempt with nothing materialized, for a conversion starting over."""
    from dataclasses import replace

    return replace(progress, last_consumed_transaction_id=0, staged=None)


def _staged_to_publish(
    progress: FinalizationProgress, scan: spool.SpoolScan, staged_session: Path
) -> StagedConversion | None:
    """The staged session a resume must publish rather than rewrite, if there is one.

    Three things have to hold together, and each rules out a different way of
    publishing something nobody verified: the cursor has to name this source's
    last committed transaction, the previous attempt has to have recorded what
    its conversion produced, and that session has to still be on disk. Any one
    of them missing means there is no completed conversion here, and the caller
    converts from the start.

    What this does *not* establish is that the package on disk is the session
    that attempt produced -- only that one was produced and something is there.
    :func:`_verify_staged_binding` establishes the rest, from the source
    identity the artifact carries in its own manifest.
    """
    if progress.staged is None:
        return None
    if progress.last_consumed_transaction_id != scan.last_transaction_id:
        return None
    if not staged_session.exists():
        return None
    return progress.staged


def _sealed_session_manifest(staged_session: Path) -> Mapping[str, Any] | None:
    """The manifest of a *sealed* staged session, or ``None`` if there is none.

    Sealing is what makes a directory committed NRF data rather than a
    half-written one: the finalizer seals every open target in the single
    transaction that writes the termination, so a non-empty ``sealed_targets``
    is exactly the statement "this conversion finished". A directory with no
    manifest, an unreadable one, or one that seals nothing is a partial session
    and is treated as absent -- the caller may delete it.
    """
    if not staged_session.is_file():
        return None
    from neurale.io.nrf._errors import NrfCorruptionError
    from neurale.io.nrf._package import Package

    try:
        package = Package(staged_session)
        try:
            manifest = json.loads((package.root / "manifest.json").read_text())
        finally:
            package.close()
    except (OSError, ValueError, NrfCorruptionError):
        return None
    if not isinstance(manifest, Mapping):
        return None
    commit = manifest.get("commit")
    if not isinstance(commit, Mapping) or not commit.get("sealed_targets"):
        return None
    return manifest


def _refuse_unrecorded_session(staged_session: Path) -> None:
    """Refuse to convert over a sealed session the progress document never claimed.

    There is one window a single progress document cannot close: a conversion
    that sealed its session and died before the write that records it. What is
    left is committed NRF data -- sealed, terminated, readable -- that nothing
    vouches for, and both ways of proceeding are wrong. Deleting it to convert
    again is the rewrite contract section 4.5 forbids. Publishing it is worse:
    layer 2 compares the artifact's committed extents against a count the
    conversion produced independently, that count died with the process, and
    reading the extents back to compare them with themselves would be a check
    that passes for any artifact.

    So the finalizer stops and leaves the session exactly where it is. The
    refusal is not retryable because retrying reaches this same state; a person
    decides whether that session is worth keeping, and moving it away or
    removing it makes the next attempt an ordinary conversion.
    """
    manifest = _sealed_session_manifest(staged_session)
    if manifest is None:
        return
    raise FinalizationError(
        f"{staged_session} holds a sealed session no progress document accounts for; it was "
        "converted by an attempt that died before recording what it produced. Refusing to "
        "rewrite committed NRF data and refusing to publish a session whose row counts nothing "
        "independent can confirm -- move or remove it, then finalize again",
        category="resume_mismatch",
        retryable=False,
    )


def _verify_staged_binding(staged_session: Path, progress: FinalizationProgress) -> None:
    """Prove the staged session is the one this progress document describes.

    The cursor says a conversion of *this source* finished. It does not say that
    the directory on disk is what that conversion produced -- and the two are
    separate facts, because the staging directory is an ordinary path that
    anything with write access can change between the crash and the resume. A
    swapped-in session with the same row counts would satisfy the cross-check
    (it compares counts against extents, and those agree within the intruder)
    and would then be published under this source's identity.

    What closes it is already in the artifact: the finalizer stamps the source's
    identity into the manifest as it writes, so the staged session states which
    spool it came from and how much of it it covers. Comparing that against the
    progress document is the binding, and any disagreement rejects the resume
    without deleting or publishing the session -- it is evidence now, not
    garbage.
    """
    manifest = _sealed_session_manifest(staged_session)
    if manifest is None:
        raise FinalizationError(
            f"{staged_session} is not the sealed session the progress document records as ready "
            "to publish; it is missing or was never sealed",
            category="resume_mismatch",
            retryable=False,
        )
    extensions = manifest.get("extensions")
    extensions = extensions if isinstance(extensions, Mapping) else {}
    replay = extensions.get(NATIVE_REPLAY_NAMESPACE)
    replay = replay if isinstance(replay, Mapping) else {}
    finalization = extensions.get(FINALIZATION_NAMESPACE)
    finalization = finalization if isinstance(finalization, Mapping) else {}
    spool_extension = finalization.get("spool")
    spool_extension = spool_extension if isinstance(spool_extension, Mapping) else {}
    session = manifest.get("session")
    session = session if isinstance(session, Mapping) else {}
    version = manifest.get("version")
    version = version if isinstance(version, Mapping) else {}

    bindings: tuple[tuple[str, Any, Any], ...] = (
        ("session id", session.get("id"), progress.session_id),
        ("plan fingerprint", replay.get("plan_fingerprint"), progress.plan_fingerprint),
        ("spool session uuid", spool_extension.get("session_uuid"), progress.spool_session_uuid),
        ("NRF minor version", version.get("minor"), progress.nrf_minor_version),
        (
            "committed prefix end",
            spool_extension.get("committed_prefix_end"),
            progress.committed_prefix_end,
        ),
        (
            "committed transactions",
            spool_extension.get("committed_transactions"),
            progress.committed_transactions,
        ),
    )
    mismatches = [
        f"{name} is {found!r}, expected {expected!r}"
        for name, found, expected in bindings
        if found != expected
    ]
    if mismatches:
        raise FinalizationError(
            f"{staged_session} is not the session this finalization staged "
            f"({'; '.join(mismatches)}); it is left untouched -- resolve or move it rather than "
            "publishing it under another source's identity",
            category="resume_mismatch",
            retryable=False,
        )


def _advance(
    staging: Path,
    progress: FinalizationProgress,
    phase: str,
    *,
    last_consumed_transaction_id: int | None = None,
    staged: StagedConversion | None = None,
) -> FinalizationProgress:
    """Record that *phase* is starting, before it starts.

    Written ahead of the work rather than after it: a document that named the
    last phase to *finish* would leave a crashed attempt indistinguishable from
    one that completed that phase and stopped, which is exactly the distinction
    a crash needs. The cursor travels with the phase for the same reason -- the
    document that says publication is starting is the one that has to say what
    is waiting to be published.
    """
    assert phase in PROGRESS_PHASES
    advanced = _with_status(
        progress,
        progress.status,
        None,
        None,
        phase=phase,
        last_consumed_transaction_id=last_consumed_transaction_id,
        staged=staged,
    )
    _write_progress(staging, advanced)
    return advanced


def _dry_run_report(
    scan: spool.SpoolScan,
    inventory: _Inventory,
    progress: FinalizationProgress,
    started: float,
) -> FinalizationReport:
    """What a real attempt over this source would produce, having written nothing.

    The counts come from the committed prefix, which is where they are already
    exact: the finalizer writes one row per committed record, so a dry run does
    not have to guess. The per-target row counts are deliberately left empty --
    those are produced *by writing*, and reporting a predicted value for them
    would be the writer's promise rather than a measurement.
    """
    counts = FinalizationCounts(
        frames=inventory.frames,
        signal_blocks=sum(inventory.blocks_per_signal.values()),
        discontinuities=inventory.discontinuities,
        signal_gaps=inventory.signal_gaps,
        control_records=len(inventory.controls),
        fault_rows=len(inventory.faults),
    )
    outcomes = _resolve_outcomes(scan, bool(inventory.faults))
    row = _accounting_row(scan, counts, termination_origin=outcomes.termination_origin)
    outcomes = _escalate_outcomes(outcomes, row)
    return FinalizationReport(
        session_path=Path(),
        spool_retained=True,
        counts=counts,
        progress=progress,
        termination_kind=outcomes.termination_kind,
        termination_origin=outcomes.termination_origin,
        source_session_end_present=outcomes.source_session_end_present,
        accounting_origin="recorder" if scan.accounting is not None else "recovery_rebuilt",
        spool_codes=scan.codes,
        duration_seconds=time.monotonic() - started,
        dry_run=True,
    )


def _record_attempt_outcome(
    staging: Path, progress: FinalizationProgress, error: FinalizationError
) -> None:
    """Record a failed attempt's outcome, honouring whether it can be retried.

    A retryable failure (a full disk, a permission blip, a publication that can
    be attempted again) leaves ``failed_retryable``; a non-retryable one (a
    source whose committed prefix cannot be accounted for, an accounting
    mismatch a retry reads the same way) leaves ``failed``. Stamping
    ``failed_retryable`` for an error whose own ``retryable`` is ``False`` would
    hand recovery a contradiction -- a progress document that says "try again" over
    a fault that says "trying again cannot help".
    """
    status = "failed_retryable" if error.retryable else "failed"
    _write_progress(staging, _with_status(progress, status, error.category, str(error)))


def _with_status(
    progress: FinalizationProgress,
    status: str,
    category: str | None,
    detail: str | None,
    *,
    last_consumed_transaction_id: int | None = None,
    phase: str | None = None,
    staged: StagedConversion | None = None,
) -> FinalizationProgress:
    from dataclasses import replace

    return replace(
        progress,
        status=status,
        category=category,
        detail=detail,
        phase=progress.phase if phase is None else phase,
        last_consumed_transaction_id=(
            progress.last_consumed_transaction_id
            if last_consumed_transaction_id is None
            else last_consumed_transaction_id
        ),
        staged=progress.staged if staged is None else staged,
    )


def _bounds_for(scan: spool.SpoolScan, options: FinalizerOptions) -> ResourceBounds:
    """Resource bounds for the rebuilt plan, sized to what will be written.

    The spool does not carry these -- they are storage sizing, not content, and
    the fingerprint deliberately excludes them. The finalizer supplies them from
    the committed prefix, which is the one place they can be exact.
    """
    return ResourceBounds(
        frame_queue_capacity=1,
        control_queue_capacity=1,
        spool_capacity_bytes=_DEFAULT_SPOOL_CAPACITY_BYTES,
        control_chunk_length=options.control_chunk_length,
        checkpoint_interval=0,
        overflow_policy="fault",
        streams=(),
    )
