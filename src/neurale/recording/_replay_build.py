#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Compile a committed NRF session into a replay image (section 8).

This is the whole of the replay *decision*: which items a run emits, in which
order, under which sequence numbers, with which provenance, and what it could
not emit. Section 8 of ``docs/development/native_recording_replay.md`` is the
only source for every one of those, and it is frozen there precisely because
two implementations choosing differently would deliver different frames from
one session under one request.

The three modes are three builders, and they are deliberately not layered on
one another:

* :func:`_build_exact_frames` reconstructs the recorded topology from the
  native ledgers. It renumbers nothing, omits nothing, and refuses a session
  whose plan did not cover every block -- naming ``recorded_projection``
  instead of quietly becoming it.
* :func:`_build_recorded_projection` emits the recorded blocks of the selected
  streams under a contiguous, zero-based, run-local frame sequence, and reports
  every source message it dropped and why.
* :func:`_build_stream_frames` synthesizes one frame per committed block from
  the per-stream block index, under the frozen k-way merge, the frozen ID
  registry, and a documented fidelity report.

What is **not** here: the run. Pacing, step permits, cancellation, reset, and
the firing of injected faults are the replay source's. This module
resolves fault positions and rejects the illegal ones, because section 8.11
requires that before a run starts, and because the position of a fault is a
fact about the compiled item sequence rather than about the run.

Everything the builder reads, it reads once, and the payload never lands in
memory whole: blocks are read from the source one at a time and appended to
the image's payload scratch file. The per-item index does scale with the
emitted item count, which is what an index is.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from neurale.io.nrf import NrfReader
from neurale.io.nrf._canonical import canonical_json_bytes
from neurale.io.nrf._ledgers import (
    NATIVE_DISCONTINUITIES,
    NATIVE_FRAMES,
    NATIVE_REPLAY_NAMESPACE,
    NATIVE_SIGNAL_BLOCKS,
    NATIVE_SIGNAL_GAPS,
    declares_native_replay,
)

from . import _replay_image as image_format
from ._errors import ReplayConfigError, ReplayImageError
from ._plan import RecordingPlan, plan_from_manifest_extension
from ._replay_config import ReplayConfig, StreamReplayRange
from ._replay_image import ReplayImage, ReplayImageWriter

#: The block-index columns section 8.3 requires before a stream may be
#: synthesized. The rest of the index is provenance: its absence lowers
#: the reported fidelity without making the index incompatible.
REQUIRED_BLOCK_IDX_COLUMNS: Final[tuple[str, ...]] = (
    "frame_sequence",
    "sample_idx_start",
    "n_samples",
    "row_offset",
    "host_received_ns",
)

#: Frozen v1 values of the synthesized-fidelity report (section 8.3).
FRAME_CONSTRUCTION: Final = "one_frame_per_block"
ORDERING_KEY: Final = "original_host_received_ns"

#: The one union ``SchemaId`` a synthesized run declares (section 8.3).
SYNTHESIZED_SCHEMA_ID: Final = 1

_NATIVE_DTYPES: Final[Mapping[str, str]] = {
    "INT16": "<i2",
    "INT32": "<i4",
    "FLOAT32": "<f4",
    "FLOAT64": "<f8",
}

_NUMPY_TO_NATIVE: Final[Mapping[str, str]] = {
    "int16": "INT16",
    "int32": "INT32",
    "float32": "FLOAT32",
    "float64": "FLOAT64",
}

_NRF_KIND_TO_NATIVE: Final[Mapping[str, str]] = {
    "neural": "SAMPLED",
    "behavioral": "SAMPLED",
    "feature": "FEATURE",
    "spike": "SPIKE",
}


# --- results -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedFault:
    """One injected fault, positioned against the compiled item sequence.

    ``fired`` is decided here rather than at run time because section 8.11
    makes it a property of the range: a fault whose position lies outside the
    requested range never fires, and must be reported as unfired rather than
    silently dropped. ``item_index`` is the position in the image's item
    section, which is what the run counts down to.
    """

    effect: str
    fired: bool
    item_index: int | None
    stall_ns: int | None
    target: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayImageBuild:
    """One image, and whether this call had to build it."""

    image: ReplayImage
    path: Path
    reused: bool
    rebuild_reason: str | None


# --- public entry points -----------------------------------------------------


def _require_target_outside_source(session: str | Path, target: Path) -> None:
    """Refuse to write a replay image into the session it is built from.

    Building a replay image from an NRF session is read-only over the session
    (contract section 8), and the image is a separate artifact a run reads. A
    target inside the source session would let a build create or overwrite files
    in the session -- up to replacing its manifest -- so it is rejected before
    any scratch file is created. Paths are canonicalized first, so a relative
    alias, a ``..`` segment, or a symlink cannot reach inside the source under a
    different spelling.
    """
    source_root = Path(session).resolve()
    target_root = Path(target).resolve()
    if target_root == source_root or source_root in target_root.parents:
        raise ReplayConfigError(
            f"the replay image target {target} resolves inside the source session "
            f"{session}, which replay MUST NOT modify; write the image outside the "
            "session directory"
        )


def build_replay_image(
    session: str | Path,
    config: ReplayConfig,
    *,
    path: str | Path,
    verify_source: bool = True,
) -> ReplayImage:
    """Compile *session* under *config* into a replay image at *path*.

    The image is written durably and atomically: a temporary file, fsynced,
    then renamed over any previous image. A caller either sees the whole new
    image or the whole previous one, and the session it was built from is left
    untouched: the target must resolve outside the source session.

    ``verify_source`` re-hashes the committed objects the build reads. It
    defaults to on because section 8.5 makes rejecting a corrupt session the
    default rather than an option a caller has to remember.
    """
    target = Path(path)
    _require_target_outside_source(session, target)
    with NrfReader.open(session) as reader:
        if verify_source:
            reader.verify_committed_objects()
        resolution = _resolve(reader, config)
        writer = ReplayImageWriter(directory=target.parent, prefix=target.name)
        try:
            _compile(reader, config, resolution, writer)
            writer.write(target, bytes.fromhex(resolution.image_fingerprint))
        finally:
            writer.close()
    opened = ReplayImage.open(target)
    try:
        _validate_faults(opened, config)
    except Exception:
        opened.close()
        raise
    return opened


def load_replay_image(
    session: str | Path,
    config: ReplayConfig,
    *,
    path: str | Path,
    verify_source: bool = True,
) -> ReplayImageBuild:
    """Return the image for this request, reusing a valid cached one.

    A cached image is reused only when it opens, passes every container check,
    and carries the fingerprint this request computes -- which covers the
    source session's committed state, the mode, the selection, and the range.
    Anything else is a rebuild with a stated reason: a stale, corrupt, or
    foreign image is never partially trusted, because the session is the source
    of truth and rebuilding from it is always available.

    A cache hit is held to the same correctness bar as a fresh build. When
    ``verify_source`` is set the session is re-hashed on the hit, not only on a
    rebuild: the image is self-contained, but section 8.5 makes rejecting a
    corrupt session the default, and a fingerprint over identity, journal
    position, and extents does not prove the committed object content is still
    intact. The configured faults are re-validated against the cached image
    too, so a request whose faults are illegal under this image raises rather
    than being served from a cache a fresh build would have refused.
    """
    target = Path(path)
    _require_target_outside_source(session, target)
    with NrfReader.open(session) as reader:
        if verify_source:
            reader.verify_committed_objects()
        resolution = _resolve(reader, config)
    expected = resolution.image_fingerprint

    reason: str | None = None
    if not target.exists():
        reason = "no image is cached at this path"
    else:
        try:
            cached = ReplayImage.open(target)
        except ReplayImageError as error:
            reason = str(error)
        else:
            if cached.fingerprint == expected:
                _validate_faults(cached, config)
                return ReplayImageBuild(cached, target, reused=True, rebuild_reason=None)
            cached.close()
            reason = "the cached image was built for another source, mode, selection, or range"
    built = build_replay_image(session, config, path=target, verify_source=verify_source)
    return ReplayImageBuild(built, target, reused=False, rebuild_reason=reason)


def open_replay_image(path: str | Path) -> ReplayImage:
    """Open and validate an existing replay image."""
    return ReplayImage.open(path)


def resolve_replay_faults(image: ReplayImage, config: ReplayConfig) -> tuple[ResolvedFault, ...]:
    """Position every configured fault against *image*, and reject the illegal.

    Returned rather than stored: the image is a pure function of the source and
    the request's selection, and a fault set is a property of the run.
    """
    return _validate_faults(image, config)


# --- resolution --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Resolution:
    """Everything the request means once the session has been consulted."""

    mode: str
    selected_streams: tuple[str, ...]
    message_range: tuple[int, int] | None
    stream_ranges: Mapping[str, StreamReplayRange]
    plan: RecordingPlan | None
    source_fingerprint: str
    image_fingerprint: str
    completeness: str | None
    incomplete: bool
    committed_ordinals: tuple[int, ...]
    block_idx_columns: Mapping[str, tuple[str, ...]]
    committed_blocks: Mapping[str, int]


def _resolve(reader: NrfReader, config: ReplayConfig) -> _Resolution:
    _check_completeness(reader, config)
    plan = _plan_of(reader) if config.ledger_based else None
    if config.ledger_based:
        selected = _resolve_ledger_selection(config, plan)
        ordinals = _committed_ordinals(reader)
        message_range = _resolve_message_range(config, ordinals)
        columns: dict[str, tuple[str, ...]] = {}
        blocks: dict[str, int] = {}
        stream_ranges: dict[str, StreamReplayRange] = {}
    else:
        columns = {}
        blocks = {}
        selected = tuple(sorted(config.stream_ranges))
        for stream_id in selected:
            columns[stream_id] = _block_index_columns(reader, stream_id)
            blocks[stream_id] = reader.stream_extent(f"{stream_id}.blocks")
        stream_ranges = _resolve_stream_ranges(config, blocks)
        ordinals = ()
        message_range = None

    source_fingerprint = _source_fingerprint(reader, plan)
    key = {
        "format_version": image_format.FORMAT_VERSION,
        "source_fingerprint": source_fingerprint,
        "mode": config.mode,
        "selected_streams": list(selected),
        "message_range": None if message_range is None else list(message_range),
        "stream_ranges": {
            stream_id: value.document() for stream_id, value in sorted(stream_ranges.items())
        },
    }
    return _Resolution(
        mode=config.mode,
        selected_streams=selected,
        message_range=message_range,
        stream_ranges=stream_ranges,
        plan=plan,
        source_fingerprint=source_fingerprint,
        image_fingerprint=hashlib.sha256(canonical_json_bytes(key)).hexdigest(),
        completeness=reader.completeness_verdict,
        incomplete=reader.complete is False,
        committed_ordinals=ordinals,
        block_idx_columns=columns,
        committed_blocks=blocks,
    )


def _check_completeness(reader: NrfReader, config: ReplayConfig) -> None:
    """Reject an incomplete session unless the caller asked for the prefix.

    ``complete is None`` is a third answer and is deliberately not treated as
    incomplete: it means the artifact carries no accounting to check, which is
    every session written before the native ledgers existed. That session is
    unverified, not known-truncated, and the distinction is recorded in the
    image metadata rather than resolved by guessing.
    """
    if reader.complete is False and not config.allow_incomplete:
        raise ReplayConfigError(
            "this session's persisted accounting says it is incomplete; replay refuses it by "
            "default. Set allow_incomplete to replay the valid committed prefix, which ends "
            "with an abnormal terminal result rather than end_of_data"
        )


def _plan_of(reader: NrfReader) -> RecordingPlan:
    extensions = reader.manifest.get("extensions") or {}
    extension = extensions.get(NATIVE_REPLAY_NAMESPACE)
    if not isinstance(extension, Mapping):
        raise ReplayConfigError(
            "this session declares no native replay plan; the ledger-based modes reconstruct "
            "the recorded topology from it and cannot be answered from the per-stream records. "
            "Replay it as stream_frames, which is a documented synthesis rather than the original"
        )
    if not declares_native_replay(reader.manifest.get("record_schemas") or ()):
        raise ReplayConfigError(
            "this session does not declare all five native replay ledgers; exact_frames and "
            "recorded_projection require them and MUST NOT degrade to a synthesis. Replay it "
            "as stream_frames if a reconstruction is acceptable"
        )
    return plan_from_manifest_extension(extension)


def _resolve_ledger_selection(config: ReplayConfig, plan: RecordingPlan) -> tuple[str, ...]:
    recorded = tuple(sorted(stream.stream_id for stream in plan.streams))
    if config.mode == "exact_frames" and not plan.covers_every_signal:
        raise ReplayConfigError(
            f"this session was recorded under a partial plan ({len(plan.streams)} of "
            f"{len(plan.planned_signal_ids)} signals); exact_frames reconstructs the original "
            "frames and cannot deliver blocks that are not on disk. Replay it as "
            "recorded_projection"
        )
    if not config.selected_streams:
        return recorded
    unknown = [stream for stream in config.selected_streams if stream not in recorded]
    if unknown:
        raise ReplayConfigError(
            f"this session does not record the selected stream(s) {unknown}; it records "
            f"{list(recorded)}"
        )
    selected = tuple(sorted(config.selected_streams))
    if config.mode == "exact_frames" and selected != recorded:
        raise ReplayConfigError(
            "exact_frames requires selected_streams to be empty or exactly the recorded set "
            f"{list(recorded)}; {list(selected)} is a projection of it. Replay it as "
            "recorded_projection"
        )
    return selected


def _committed_ordinals(reader: NrfReader) -> tuple[int, ...]:
    """Return every committed data-message ordinal, ascending.

    Frames and discontinuities share one ordinal space (section 1.1), so the
    range boundaries are checked against their union and never against either
    ledger alone.
    """
    ordinals: set[int] = set()
    for schema_id in (NATIVE_FRAMES.schema_id, NATIVE_DISCONTINUITIES.schema_id):
        # Chunk at a time: an unbounded read_records would materialize every
        # column of the whole ledger to answer a question about one of them.
        for row in reader.iter_records(schema_id):
            ordinals.add(int(row["data_message_ordinal"]))
    return tuple(sorted(ordinals))


def _resolve_message_range(config: ReplayConfig, ordinals: Sequence[int]) -> tuple[int, int]:
    """Return the explicit ``[start, stop)`` this request means.

    A request with no bounds covers the whole session, which is resolved to the
    explicit range here so that the image fingerprint of "everything" and of
    the same range spelled out are one value rather than two.
    """
    end = (ordinals[-1] + 1) if ordinals else 0
    start = ordinals[0] if ordinals else 0
    if config.message_range is None:
        return (start, end)
    requested = (config.message_range.start, config.message_range.stop)
    legal = set(ordinals)
    legal.add(end)
    for label, value in (("start", requested[0]), ("stop", requested[1])):
        if value in legal:
            continue
        if not ordinals:
            raise ReplayConfigError(
                f"this session committed no data message, so the only legal range is [0, 0); "
                f"message_range {label} {value} is beyond its committed extent"
            )
        if value < start or value > end:
            raise ReplayConfigError(
                f"message_range {label} {value} is beyond the committed extent "
                f"[{start}, {end}) of this session"
            )
        raise ReplayConfigError(
            f"message_range {label} {value} does not name a committed data message, so it does "
            "not fall between two of them. Boundaries are never widened to the nearest legal "
            "position"
        )
    return requested


def _block_index_columns(reader: NrfReader, stream_id: str) -> tuple[str, ...]:
    """Return the block-index columns *stream_id* carries, or raise.

    ``stream_frames`` requires a *compatible* committed index, and section 8.3
    names the five columns that make one compatible. A missing index is an
    error rather than a stream this run quietly skips.
    """
    if stream_id not in reader.stream_ids():
        raise ReplayConfigError(f"this session has no stream {stream_id!r}")
    idx_id = f"{stream_id}.blocks"
    if idx_id not in reader.stream_ids():
        raise ReplayConfigError(
            f"stream {stream_id!r} declares no committed block index; stream_frames builds "
            "frames from one and MUST NOT synthesize them from anything else"
        )
    descriptor = reader.stream(idx_id)
    channels = {channel["id"]: channel for channel in reader.manifest["channels"]}
    present = tuple(
        str(channels[channel_id]["name"])
        for channel_id in descriptor["channel_ids"]
        if channel_id in channels
    )
    missing = [column for column in REQUIRED_BLOCK_IDX_COLUMNS if column not in present]
    if missing:
        raise ReplayConfigError(
            f"the block index of stream {stream_id!r} is missing {missing}, which "
            "stream_frames needs to order blocks and to place discontinuities"
        )
    return present


def _resolve_stream_ranges(
    config: ReplayConfig, committed: Mapping[str, int]
) -> dict[str, StreamReplayRange]:
    resolved: dict[str, StreamReplayRange] = {}
    for stream_id, value in sorted(config.stream_ranges.items()):
        extent = committed[stream_id]
        if value.stop > extent or value.start > extent:
            raise ReplayConfigError(
                f"the range [{value.start}, {value.stop}) of stream {stream_id!r} is beyond its "
                f"committed extent of {extent} blocks; a range past the committed data is an "
                "error, not an empty result"
            )
        resolved[stream_id] = value
    return resolved


def _source_fingerprint(reader: NrfReader, plan: RecordingPlan | None) -> str:
    """Fingerprint the committed state a replay image would be built from.

    Everything that could change what a rebuild produces is in it: the session
    identity, the format version, the recording plan, the journal position the
    reader replayed to, and the committed extent of every target. A session
    that grew, was recovered, or was re-finalized fingerprints differently, so
    a cached image of it is a miss rather than stale data served as fresh.
    """
    manifest = reader.manifest
    extents: dict[str, int] = {}
    for stream in manifest["streams"]:
        extents[stream["data"]["path"]] = reader.stream_extent(stream["id"])
    for schema in manifest["record_schemas"]:
        extents[schema["path"]] = reader.record_extent(schema["id"])
    state = reader.committed_state
    document = {
        "format_version": image_format.FORMAT_VERSION,
        "session_id": reader.session_id,
        "nrf_version": manifest.get("version"),
        "plan_fingerprint": None if plan is None else plan.fingerprint,
        "journal_sequence": state.journal_sequence,
        "last_transaction_id": state.last_transaction_id,
        "completeness": reader.completeness_verdict,
        "extents": dict(sorted(extents.items())),
    }
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


# --- compiled item plan ------------------------------------------------------


@dataclass(slots=True)
class _Compiled:
    """The emitted item sequence, before it is encoded."""

    items: list[bytes] = field(default_factory=list)
    blocks: list[bytes] = field(default_factory=list)
    gaps: list[bytes] = field(default_factory=list)
    omissions: list[bytes] = field(default_factory=list)
    frame_count: int = 0
    discontinuity_count: int = 0
    #: ``data_message_ordinal -> item index``, for ledger-based fault positioning.
    ordinal_positions: dict[int, int] = field(default_factory=dict)
    #: ``(stream_id, block_ordinal) -> item index`` for synthesized frames.
    block_positions: dict[tuple[str, int], int] = field(default_factory=dict)
    #: ``(stream_id, record ordinal) -> item index`` for synthesized discontinuities.
    record_positions: dict[tuple[str, int], int] = field(default_factory=dict)
    first_timeline_ns: int | None = None
    last_timeline_ns: int | None = None
    per_stream: dict[str, list[int]] = field(default_factory=dict)
    #: ``stream_id -> native signal id`` for a synthesized run, so the image's
    #: fidelity section can name the signal a stream was assigned.
    stream_signal_ids: dict[str, int] = field(default_factory=dict)
    #: Synthesized streams whose declared rate came from the recording plan.
    plan_rated_streams: set[str] = field(default_factory=set)

    def note_timeline(self, value: int) -> None:
        if self.first_timeline_ns is None:
            self.first_timeline_ns = value
        self.last_timeline_ns = value

    def counts_for(self, stream_id: str) -> list[int]:
        return self.per_stream.setdefault(stream_id, [0, 0, 0])


def _compile(
    reader: NrfReader,
    config: ReplayConfig,
    resolution: _Resolution,
    writer: ReplayImageWriter,
) -> _Compiled:
    registry: _Registry | None = None
    if config.mode == "stream_frames":
        idx = {
            stream_id: _index_rows(reader, stream_id, resolution.block_idx_columns[stream_id])
            for stream_id in resolution.selected_streams
        }
        registry = _synthesized_registry(reader, resolution, idx)
        compiled = _build_stream_frames(reader, resolution, writer, idx, registry)
        schema = _synthesized_schema(reader, resolution, writer, registry)
    else:
        compiled = _build_ledger_modes(reader, config, resolution, writer)
        schema = _plan_schema(resolution, writer)
    _encode_sections(reader, config, resolution, writer, compiled, schema, registry)
    return compiled


# --- ledger-based modes ------------------------------------------------------


@dataclass(slots=True)
class _FrameRow:
    ordinal: int
    frame_sequence: int
    frame_ordinal: int
    host_received_ns: int
    source_tick: int | None
    valid_until_ns: int | None
    native_schema_id: int
    source_clock_domain: int
    frame_flags: int
    total_payload_byte_count: int
    first_block: int | None
    recorded_blocks: int
    native_session_id: int


def _frame_rows(reader: NrfReader, start: int, stop: int) -> dict[int, _FrameRow]:
    rows: dict[int, _FrameRow] = {}
    for row in reader.iter_records(NATIVE_FRAMES.schema_id):
        ordinal = int(row["data_message_ordinal"])
        if not start <= ordinal < stop:
            continue
        first = row["first_signal_block_ordinal"]
        rows[ordinal] = _FrameRow(
            ordinal=ordinal,
            frame_sequence=int(row["frame_sequence"]),
            frame_ordinal=int(row["frame_ordinal"]),
            host_received_ns=int(row["host_received_ns"]),
            source_tick=None if row["source_tick"] is None else int(row["source_tick"]),
            valid_until_ns=None if row["valid_until_ns"] is None else int(row["valid_until_ns"]),
            native_schema_id=int(row["native_schema_id"]),
            source_clock_domain=int(row["source_clock_domain"]),
            frame_flags=int(row["frame_flags"]),
            total_payload_byte_count=int(row["total_payload_byte_count"]),
            first_block=None if first is None else int(first),
            recorded_blocks=int(row["recorded_signal_block_count"]),
            native_session_id=int(row["native_session_id"]),
        )
    return rows


def _discontinuity_rows(reader: NrfReader, start: int, stop: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in reader.iter_records(NATIVE_DISCONTINUITIES.schema_id):
        ordinal = int(row["data_message_ordinal"])
        if start <= ordinal < stop:
            rows[ordinal] = row
    return rows


def _build_ledger_modes(
    reader: NrfReader,
    config: ReplayConfig,
    resolution: _Resolution,
    writer: ReplayImageWriter,
) -> _Compiled:
    assert resolution.message_range is not None
    start, stop = resolution.message_range
    frames = _frame_rows(reader, start, stop)
    discontinuities = _discontinuity_rows(reader, start, stop)
    plan = resolution.plan
    assert plan is not None
    exact = config.mode == "exact_frames"
    selected_signals = {
        stream.native_signal_id
        for stream in plan.streams
        if stream.stream_id in resolution.selected_streams
    }
    signals = {signal.id: signal for signal in plan.native_schema.signals}

    # Pass 1 decides which frames this run emits, because a discontinuity's
    # representability depends on a frame that comes *after* it.
    emitted_blocks: dict[int, list[dict[str, Any]]] = {}
    replay_sequence_of: dict[int, int] = {}
    original_to_replay: dict[int, int] = {}
    next_sequence = 0
    for ordinal in sorted(frames):
        row = frames[ordinal]
        blocks = _blocks_of(reader, row)
        projected = (
            blocks
            if exact
            else [block for block in blocks if int(block["native_signal_id"]) in selected_signals]
        )
        if not projected:
            if exact:
                raise ReplayImageError(
                    f"frame {ordinal} records no signal block; exact_frames omits nothing, so "
                    "there is no honest result for a frame with nothing to emit"
                )
            continue
        emitted_blocks[ordinal] = projected
        sequence = row.frame_sequence if exact else next_sequence
        replay_sequence_of[ordinal] = sequence
        original_to_replay[row.frame_sequence] = sequence
        next_sequence += 1

    compiled = _Compiled()
    previous_replay: int | None = None
    for ordinal in sorted(set(frames) | set(discontinuities)):
        if ordinal in frames:
            row = frames[ordinal]
            if ordinal not in emitted_blocks:
                _omit_message(
                    compiled,
                    kind=image_format.ITEM_FRAME,
                    reason=image_format.OMISSION_NO_PROJECTED_BLOCKS,
                    ordinal=ordinal,
                    original_frame_sequence=row.frame_sequence,
                )
                continue
            previous_replay = _emit_frame(
                reader,
                writer,
                compiled,
                resolution,
                row,
                emitted_blocks[ordinal],
                replay_sequence_of[ordinal],
                signals,
                exact=exact,
            )
        else:
            _handle_discontinuity(
                reader,
                writer,
                compiled,
                resolution,
                discontinuities[ordinal],
                selected_signals,
                original_to_replay,
                previous_replay,
                exact=exact,
            )
    return compiled


def _blocks_of(reader: NrfReader, row: _FrameRow) -> list[dict[str, Any]]:
    """Return one frame's recorded block rows, in ``block_index_in_frame`` order."""
    if row.first_block is None or row.recorded_blocks == 0:
        return []
    columns = reader.read_records(
        NATIVE_SIGNAL_BLOCKS.schema_id, row.first_block, row.first_block + row.recorded_blocks
    )
    count = len(columns["signal_block_ordinal"])
    if count != row.recorded_blocks:
        raise ReplayImageError(
            f"frame {row.ordinal} claims {row.recorded_blocks} block rows and the ledger "
            f"commits {count}"
        )
    rows = [{name: column[idx] for name, column in columns.items()} for idx in range(count)]
    for block in rows:
        if int(block["data_message_ordinal"]) != row.ordinal:
            raise ReplayImageError(
                f"block ordinal {block['signal_block_ordinal']} belongs to data message "
                f"{block['data_message_ordinal']}, not to frame {row.ordinal}"
            )
    rows.sort(key=lambda block: int(block["block_index_in_frame"]))
    return rows


def _emit_frame(
    reader: NrfReader,
    writer: ReplayImageWriter,
    compiled: _Compiled,
    resolution: _Resolution,
    row: _FrameRow,
    blocks: Sequence[Mapping[str, Any]],
    replay_sequence: int,
    signals: Mapping[int, Any],
    *,
    exact: bool,
) -> int:
    """Append one emitted frame, its payload, and its block records."""
    encoded: list[tuple[Mapping[str, Any], bytes, int]] = []
    cursor = 0
    for block in blocks:
        payload = _encode_recorded_block(reader, block, signals)
        declared = int(block["payload_byte_count"])
        if len(payload) != declared:
            raise ReplayImageError(
                f"block {block['signal_block_ordinal']} of frame {row.ordinal} decodes to "
                f"{len(payload)} bytes and the ledger records {declared}"
            )
        offset = int(block["payload_offset"]) if exact else cursor
        encoded.append((block, payload, offset))
        cursor += declared

    if exact:
        _require_full_payload_coverage(row, encoded)
        region = bytearray(row.total_payload_byte_count)
        for _, payload, offset in encoded:
            region[offset : offset + len(payload)] = payload
        body = bytes(region)
    else:
        body = b"".join(payload for _, payload, _ in encoded)

    payload_offset = writer.append_payload(body)
    child_first = len(compiled.blocks)
    for block, payload, offset in encoded:
        stream_id = str(block["stream_id"])
        counts = compiled.counts_for(stream_id)
        counts[1] += 1
        compiled.blocks.append(
            image_format.BLOCK.pack(
                int(block["native_signal_id"]),
                int(block["block_index_in_frame"]),
                writer.strings.intern(stream_id),
                int(block["n_samples"]),
                int(block["sample_idx_start"]),
                int(block["last_sample_idx"]),
                int(block["device_tick_start"]),
                int(block["observation_time_start_ns"]),
                offset,
                len(payload),
                int(block["payload_offset"]),
                int(block["clock_sync_device_tick_reference"]),
                int(block["clock_sync_host_time_reference_ns"]),
                int(block["clock_sync_rate_numerator"]),
                int(block["clock_sync_rate_denominator"]),
                int(block["clock_sync_uncertainty_ns"]),
                int(block["signal_block_ordinal"]),
                image_format.UINT64_ABSENT,
                int(block["clock_sync_clock_domain"]),
                int(block["clock_sync_generation"]),
                int(block["clock_sync_flags"]),
                image_format.BLOCK_FLAG_CLOCK_SYNC,
            )
        )
    for stream_id in {str(block["stream_id"]) for block, _, _ in encoded}:
        compiled.counts_for(stream_id)[0] += 1

    flags = 0
    if row.source_tick is not None:
        flags |= image_format.ITEM_FLAG_SOURCE_TICK
    if row.valid_until_ns is not None:
        flags |= image_format.ITEM_FLAG_VALID_UNTIL
    compiled.ordinal_positions[row.ordinal] = len(compiled.items)
    compiled.items.append(
        image_format.ITEM.pack(
            image_format.ITEM_FRAME,
            flags,
            0,
            len(encoded),
            child_first,
            row.ordinal,
            replay_sequence,
            row.frame_sequence,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            row.host_received_ns,
            row.host_received_ns,
            payload_offset,
            len(body),
            row.total_payload_byte_count,
            image_format.UINT64_ABSENT if row.source_tick is None else row.source_tick,
            image_format.UINT64_ABSENT if row.valid_until_ns is None else row.valid_until_ns,
            row.native_schema_id,
            row.source_clock_domain,
            row.frame_flags,
            image_format.UINT32_ABSENT,
        )
    )
    compiled.frame_count += 1
    compiled.note_timeline(row.host_received_ns)
    return replay_sequence


def _require_full_payload_coverage(
    row: _FrameRow, encoded: Sequence[tuple[Mapping[str, Any], bytes, int]]
) -> None:
    """Prove the recorded blocks tile the original frame payload exactly.

    Section 1.1 keeps ``payload_offset`` and ``total_payload_byte_count`` so
    exact replay can verify coverage against the *original* recording rather
    than against its own reconstruction. A hole or an overlap means the image
    would carry a frame payload that never existed, so it is refused here.
    """
    spans = sorted((offset, offset + len(payload)) for _, payload, offset in encoded)
    cursor = 0
    for begin, end in spans:
        if begin != cursor:
            raise ReplayImageError(
                f"the recorded blocks of frame {row.ordinal} do not cover its payload: "
                f"bytes [{cursor}, {begin}) are named by no block"
                if begin > cursor
                else f"the recorded blocks of frame {row.ordinal} overlap at byte {begin}"
            )
        cursor = end
    if cursor != row.total_payload_byte_count:
        raise ReplayImageError(
            f"the recorded blocks of frame {row.ordinal} cover {cursor} bytes and the frame "
            f"records {row.total_payload_byte_count}"
        )


def _encode_recorded_block(
    reader: NrfReader, block: Mapping[str, Any], signals: Mapping[int, Any]
) -> bytes:
    """Read one recorded block back and re-encode it in its native layout.

    NRF stores every stream sample-major; the native block carried whatever the
    schema declared, so a channel-major signal is transposed back rather than
    handed to the runtime in the storage layout.
    """
    signal = signals.get(int(block["native_signal_id"]))
    if signal is None:
        raise ReplayImageError(
            f"the recording plan declares no signal {block['native_signal_id']}, which "
            f"block {block['signal_block_ordinal']} records"
        )
    start = int(block["row_offset"])
    count = int(block["n_samples"])
    values = reader.read_stream(str(block["stream_id"]), start, start + count)
    if values.shape[0] != count:
        raise ReplayImageError(
            f"block {block['signal_block_ordinal']} needs rows [{start}, {start + count}) of "
            f"{block['stream_id']!r} and only {values.shape[0]} are committed"
        )
    arr = np.ascontiguousarray(values, dtype=_NATIVE_DTYPES[signal.dtype])
    if signal.layout == "CHANNEL_MAJOR":
        arr = np.ascontiguousarray(arr.T)
    return arr.tobytes()


def _handle_discontinuity(
    reader: NrfReader,
    writer: ReplayImageWriter,
    compiled: _Compiled,
    resolution: _Resolution,
    row: Mapping[str, Any],
    selected_signals: set[int],
    original_to_replay: Mapping[int, int],
    previous_replay: int | None,
    *,
    exact: bool,
) -> None:
    """Emit or omit one recorded discontinuity (section 8.2)."""
    ordinal = int(row["data_message_ordinal"])
    n_gaps = int(row["signal_gap_count"])
    first_gap = row["first_signal_gap_ordinal"]
    gaps = _gaps_of(reader, ordinal, first_gap, n_gaps)
    actual = int(row["actual_frame_sequence"])

    if exact:
        _emit_discontinuity(
            writer,
            compiled,
            row,
            gaps,
            replay_sequence=actual,
            previous_replay=int(row["previous_frame_sequence"]),
        )
        return

    relevant = n_gaps == 0 or any(int(gap["native_signal_id"]) in selected_signals for gap in gaps)
    target_replay = original_to_replay.get(actual)
    no_preceding = previous_replay is None
    no_target = target_replay is None
    if relevant and not no_preceding and not no_target:
        projected = [gap for gap in gaps if int(gap["native_signal_id"]) in selected_signals]
        _emit_discontinuity(
            writer,
            compiled,
            row,
            projected,
            replay_sequence=target_replay,
            previous_replay=previous_replay,
        )
        return

    # The tests decide *whether*; the frozen precedence decides what is
    # reported, over every condition that holds rather than the first that
    # failed.
    if no_preceding:
        reason = image_format.OMISSION_NO_PRECEDING_EMITTED_FRAME
    elif no_target:
        reason = image_format.OMISSION_TARGET_FRAME_NOT_EMITTED
    else:
        reason = image_format.OMISSION_NO_PROJECTED_SIGNALS
    _omit_message(
        compiled,
        kind=image_format.ITEM_DISCONTINUITY,
        reason=reason,
        ordinal=ordinal,
        original_frame_sequence=None,
    )


def _gaps_of(reader: NrfReader, ordinal: int, first_gap: Any, n_gaps: int) -> list[dict[str, Any]]:
    if n_gaps == 0:
        return []
    if first_gap is None:
        raise ReplayImageError(
            f"discontinuity {ordinal} declares {n_gaps} signal gaps and no first gap ordinal"
        )
    first = int(first_gap)
    columns = reader.read_records(NATIVE_SIGNAL_GAPS.schema_id, first, first + n_gaps)
    count = len(columns["signal_gap_ordinal"])
    if count != n_gaps:
        raise ReplayImageError(
            f"discontinuity {ordinal} claims {n_gaps} gap rows and the ledger commits {count}"
        )
    rows = [{name: column[idx] for name, column in columns.items()} for idx in range(count)]
    rows.sort(key=lambda gap: int(gap["gap_index_in_message"]))
    return rows


def _emit_discontinuity(
    writer: ReplayImageWriter,
    compiled: _Compiled,
    row: Mapping[str, Any],
    gaps: Sequence[Mapping[str, Any]],
    *,
    replay_sequence: int,
    previous_replay: int,
) -> None:
    child_first = len(compiled.gaps)
    for gap in gaps:
        compiled.gaps.append(
            image_format.GAP.pack(
                int(gap["native_signal_id"]),
                int(gap["gap_index_in_message"]),
                writer.strings.intern(str(gap["reason"])),
                int(gap["gap_flags"]),
                int(gap["expected_sample_index"]),
                int(gap["actual_sample_index"]),
                image_format.UINT64_ABSENT
                if gap["missing_samples"] is None
                else int(gap["missing_samples"]),
                int(gap["expected_device_tick"]),
                int(gap["actual_device_tick"]),
                int(gap["signal_gap_ordinal"]),
            )
        )
    ordinal = int(row["data_message_ordinal"])
    timeline = int(row["runtime_accepted_host_time_ns"])
    flags = image_format.ITEM_FLAG_FRAME_LEVEL_GAP if int(row["signal_gap_count"]) == 0 else 0
    compiled.ordinal_positions[ordinal] = len(compiled.items)
    compiled.items.append(
        image_format.ITEM.pack(
            image_format.ITEM_DISCONTINUITY,
            flags,
            0,
            len(gaps),
            child_first,
            ordinal,
            replay_sequence,
            int(row["actual_frame_sequence"]),
            previous_replay,
            int(row["previous_frame_sequence"]),
            timeline,
            image_format.UINT64_ABSENT,
            0,
            0,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            0,
            0,
            0,
            writer.strings.intern(str(row["reason"])),
        )
    )
    compiled.discontinuity_count += 1
    compiled.note_timeline(timeline)


def _omit_message(
    compiled: _Compiled,
    *,
    kind: int,
    reason: int,
    ordinal: int | None,
    original_frame_sequence: int | None,
    stream: int = image_format.UINT32_ABSENT,
    source_ordinal: int | None = None,
) -> None:
    compiled.omissions.append(
        image_format.OMISSION.pack(
            kind,
            reason,
            0,
            stream,
            image_format.UINT64_ABSENT if ordinal is None else ordinal,
            image_format.UINT64_ABSENT
            if original_frame_sequence is None
            else original_frame_sequence,
            image_format.UINT64_ABSENT if source_ordinal is None else source_ordinal,
        )
    )


# --- synthesized stream replay -----------------------------------------------


@dataclass(slots=True)
class _IndexRow:
    stream_id: str
    block_ordinal: int
    frame_sequence: int
    sample_idx_start: int
    last_sample_idx: int
    n_samples: int
    row_offset: int
    device_tick_start: int
    host_received_ns: int


def _index_rows(reader: NrfReader, stream_id: str, columns: Sequence[str]) -> list[_IndexRow]:
    """Read one stream's whole committed block index, in committed order.

    The whole index rather than the requested window, because three separate
    decisions need the parts outside it: which frame sequence a discontinuity
    names, whether that block exists at all, and the largest committed block --
    which fixes the declared schema and must not change with the range.
    """
    extent = reader.stream_extent(f"{stream_id}.blocks")
    order = {name: pos for pos, name in enumerate(columns)}
    rows: list[_IndexRow] = []
    for start in range(0, extent, 4096):
        values = reader.read_stream(f"{stream_id}.blocks", start, min(start + 4096, extent))
        for offset in range(values.shape[0]):
            row = values[offset]
            get = lambda name, row=row: int(row[order[name]]) if name in order else 0  # noqa: E731
            rows.append(
                _IndexRow(
                    stream_id=stream_id,
                    block_ordinal=start + offset,
                    frame_sequence=get("frame_sequence"),
                    sample_idx_start=get("sample_idx_start"),
                    last_sample_idx=get("last_sample_idx"),
                    n_samples=get("n_samples"),
                    row_offset=get("row_offset"),
                    device_tick_start=get("device_tick_start"),
                    host_received_ns=get("host_received_ns"),
                )
            )
    return rows


def _build_stream_frames(
    reader: NrfReader,
    resolution: _Resolution,
    writer: ReplayImageWriter,
    idx: Mapping[str, Sequence[_IndexRow]],
    registry: _Registry,
) -> _Compiled:
    """Synthesize one frame per committed block, under the frozen merge."""
    windows: dict[str, list[_IndexRow]] = {}
    for stream_id in resolution.selected_streams:
        window = resolution.stream_ranges[stream_id]
        windows[stream_id] = list(idx[stream_id][window.start : window.stop])

    pending, unresolvable = _synthesized_discontinuities(reader, resolution, idx)
    compiled = _Compiled()
    compiled.stream_signal_ids = {
        stream_id: registry.signals[stream_id].native_id
        for stream_id in resolution.selected_streams
    }
    compiled.plan_rated_streams = {
        stream_id
        for stream_id in resolution.selected_streams
        if registry.signals[stream_id].rate_from_plan
    }
    cursors = dict.fromkeys(resolution.selected_streams, 0)
    sequence = 0
    previous_replay: int | None = None

    while True:
        head: tuple[int, bytes, int] | None = None
        chosen: str | None = None
        for stream_id in resolution.selected_streams:
            rows = windows[stream_id]
            pos = cursors[stream_id]
            if pos >= len(rows):
                continue
            row = rows[pos]
            key = (row.host_received_ns, stream_id.encode("utf-8"), row.block_ordinal)
            if head is None or key < head:
                head = key
                chosen = stream_id
        if chosen is None:
            break
        row = windows[chosen][cursors[chosen]]
        cursors[chosen] += 1

        # A discontinuity occupies the same head position as the block it
        # precedes and is emitted first; two at one position keep their
        # committed record order.
        for record in pending.pop((chosen, row.block_ordinal), ()):
            _synthesized_discontinuity(
                writer, compiled, registry, record, sequence, previous_replay
            )
        _emit_synthesized_frame(reader, writer, compiled, registry, row, sequence)
        previous_replay = sequence
        sequence += 1

    # Whatever is left named a block this run did not emit. Records whose
    # target block exists but was not requested were dropped before they got
    # here: section 8.2 scopes the omission list to the request, so "not
    # requested" never appears beside "requested and dropped".
    for key in sorted(pending):
        for record in pending[key]:
            _omit_stream_message(compiled, writer, record)
    for record in unresolvable:
        _omit_stream_message(compiled, writer, record)
    return compiled


def _omit_stream_message(
    compiled: _Compiled, writer: ReplayImageWriter, record: Mapping[str, Any]
) -> None:
    _omit_message(
        compiled,
        kind=image_format.ITEM_DISCONTINUITY,
        reason=image_format.OMISSION_TARGET_BLOCK_NOT_EMITTED,
        ordinal=None,
        original_frame_sequence=int(record["actual_frame_sequence"]),
        stream=writer.strings.intern(str(record["stream_id"])),
        source_ordinal=int(record["record_ordinal"]),
    )


def _synthesized_discontinuities(
    reader: NrfReader,
    resolution: _Resolution,
    idx: Mapping[str, Sequence[_IndexRow]],
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[dict[str, Any]]]:
    """Group per-stream discontinuity records by the block they precede.

    Records from different streams are never merged back into one message: the
    evidence that they came from one recorded message lives in the native
    discontinuity ledger, which this mode is defined not to read, so merging
    would be a guess. One recorded message that fanned out over three streams
    therefore reappears here as three.
    """
    pending: dict[tuple[str, int], list[dict[str, Any]]] = {}
    unresolvable: list[dict[str, Any]] = []
    for stream_id in resolution.selected_streams:
        window = resolution.stream_ranges[stream_id]
        by_sequence: dict[int, int] = {}
        for row in idx[stream_id]:
            by_sequence.setdefault(row.frame_sequence, row.block_ordinal)
        descriptor = reader.stream(stream_id)
        schema_id = descriptor["segment_policy"]["discontinuity_record_schema_id"]
        for ordinal, record in enumerate(reader.iter_records(schema_id)):
            target = int(record["actual_frame_sequence"])
            entry = {
                "stream_id": stream_id,
                "record_ordinal": ordinal,
                "actual_frame_sequence": target,
                "previous_frame_sequence": int(record["previous_frame_sequence"]),
                "reason": str(record["reason"]),
                "time_ns": int(record["time_ns"]),
                "expected_sample_index": record["expected_sample_index"],
                "actual_sample_index": record["actual_sample_index"],
                "missing_samples": record["missing_samples"],
            }
            block_ordinal = by_sequence.get(target)
            if block_ordinal is None:
                unresolvable.append(entry)
            elif window.start <= block_ordinal < window.stop:
                pending.setdefault((stream_id, block_ordinal), []).append(entry)
    return pending, unresolvable


def _synthesized_discontinuity(
    writer: ReplayImageWriter,
    compiled: _Compiled,
    registry: _Registry,
    record: Mapping[str, Any],
    target_sequence: int,
    previous_replay: int | None,
) -> None:
    """Emit one synthesized discontinuity, or omit an unrepresentable one.

    Section 8.2's boundary rule applies unchanged: with no frame emitted yet
    there is nothing for ``previous_frame_sequence`` to name, and a
    discontinuity emitted before any frame would reach a consumer whose
    continuity tracker has no expectation to compare it against.
    """
    stream_id = str(record["stream_id"])
    if previous_replay is None:
        _omit_message(
            compiled,
            kind=image_format.ITEM_DISCONTINUITY,
            reason=image_format.OMISSION_NO_PRECEDING_EMITTED_FRAME,
            ordinal=None,
            original_frame_sequence=int(record["actual_frame_sequence"]),
            stream=writer.strings.intern(stream_id),
            source_ordinal=int(record["record_ordinal"]),
        )
        return

    signal = registry.signals[stream_id]
    child_first = len(compiled.gaps)
    compiled.gaps.append(
        image_format.GAP.pack(
            signal.native_id,
            0,
            writer.strings.intern(str(record["reason"])),
            0,
            0 if record["expected_sample_index"] is None else int(record["expected_sample_index"]),
            0 if record["actual_sample_index"] is None else int(record["actual_sample_index"]),
            image_format.UINT64_ABSENT
            if record["missing_samples"] is None
            else int(record["missing_samples"]),
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            # This mode has no gap ledger, so the identity a fault positions
            # against is the committed record's own ordinal within its stream.
            int(record["record_ordinal"]),
        )
    )
    compiled.record_positions[(stream_id, int(record["record_ordinal"]))] = len(compiled.items)
    timeline = int(record["time_ns"])
    compiled.items.append(
        image_format.ITEM.pack(
            image_format.ITEM_DISCONTINUITY,
            0,
            0,
            1,
            child_first,
            image_format.UINT64_ABSENT,
            target_sequence,
            int(record["actual_frame_sequence"]),
            previous_replay,
            int(record["previous_frame_sequence"]),
            timeline,
            image_format.UINT64_ABSENT,
            0,
            0,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            SYNTHESIZED_SCHEMA_ID,
            0,
            0,
            writer.strings.intern(str(record["reason"])),
        )
    )
    compiled.discontinuity_count += 1
    compiled.counts_for(stream_id)[2] += 1
    compiled.note_timeline(timeline)


def _observation_time_start_ns(reader: NrfReader, row: _IndexRow) -> int:
    """Recover the recorded observation time of a synthesized block's first sample.

    Section 8.6 makes observation times recorded data replayed unchanged in
    every mode, and a per-stream block index does not store them, so they are
    recovered from the source rather than fabricated. Explicit-timing streams
    read their stored timestamps; regular-timing streams derive the grid from
    the declared rate and the segment origin the recording froze. The value is
    never zero-by-default -- a stream whose timestamp cannot be recovered is
    refused, not given a made-up one.
    """
    timing = reader.stream(row.stream_id)["timing"]
    if timing["mode"] == "explicit":
        stamps = reader.read_timestamps(row.stream_id, row.row_offset, row.row_offset + 1)
        if stamps is None or stamps.shape[0] == 0:
            raise ReplayImageError(
                f"stream {row.stream_id!r} declares explicit timing but block "
                f"{row.block_ordinal} has no committed observation timestamp"
            )
        return int(stamps[0])
    rate = timing["rate"]
    num = int(rate["numerator"])
    den = int(rate["denominator"])
    if num <= 0:
        raise ReplayImageError(
            f"stream {row.stream_id!r} declares a non-positive sample rate, so its "
            "observation grid cannot be recovered"
        )
    offset = row.sample_idx_start - int(timing["segment_start_index"])
    return int(timing["segment_start_time_ns"]) + (offset * 1_000_000_000 * den) // num


def _emit_synthesized_frame(
    reader: NrfReader,
    writer: ReplayImageWriter,
    compiled: _Compiled,
    registry: _Registry,
    row: _IndexRow,
    sequence: int,
) -> None:
    signal = registry.signals[row.stream_id]
    values = reader.read_stream(row.stream_id, row.row_offset, row.row_offset + row.n_samples)
    if values.shape[0] != row.n_samples:
        raise ReplayImageError(
            f"block {row.block_ordinal} of {row.stream_id!r} needs rows "
            f"[{row.row_offset}, {row.row_offset + row.n_samples}) and only "
            f"{values.shape[0]} are committed"
        )
    payload = np.ascontiguousarray(values, dtype=_NATIVE_DTYPES[signal.dtype]).tobytes()
    payload_offset = writer.append_payload(payload)

    child_first = len(compiled.blocks)
    compiled.blocks.append(
        image_format.BLOCK.pack(
            signal.native_id,
            0,
            writer.strings.intern(row.stream_id),
            row.n_samples,
            row.sample_idx_start,
            row.last_sample_idx,
            row.device_tick_start,
            _observation_time_start_ns(reader, row),
            0,
            len(payload),
            image_format.UINT64_ABSENT,
            0,
            0,
            0,
            0,
            0,
            row.block_ordinal,
            row.frame_sequence,
            0,
            0,
            0,
            # No clock-sync snapshot is invented: a block index of this shape
            # carries none, and the fidelity report says so.
            0,
        )
    )
    compiled.block_positions[(row.stream_id, row.block_ordinal)] = len(compiled.items)
    compiled.items.append(
        image_format.ITEM.pack(
            image_format.ITEM_FRAME,
            0,
            0,
            1,
            child_first,
            image_format.UINT64_ABSENT,
            sequence,
            row.frame_sequence,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            row.host_received_ns,
            row.host_received_ns,
            payload_offset,
            len(payload),
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            image_format.UINT64_ABSENT,
            SYNTHESIZED_SCHEMA_ID,
            signal.clock_domain,
            0,
            image_format.UINT32_ABSENT,
        )
    )
    compiled.frame_count += 1
    counts = compiled.counts_for(row.stream_id)
    counts[0] += 1
    counts[1] += 1
    compiled.note_timeline(row.host_received_ns)


# --- the synthesized ID registry (section 8.3) -------------------------------


@dataclass(frozen=True, slots=True)
class _SynthesizedSignal:
    stream_id: str
    native_id: int
    clock_domain: int
    n_channels: int
    dtype: str
    kind: str
    feature_set_id: int
    observation_timing: str
    device_tick_tracking: str
    rate: tuple[int, int]
    rate_from_plan: bool
    max_block_samples: int


@dataclass(frozen=True, slots=True)
class _Registry:
    signals: Mapping[str, _SynthesizedSignal]
    #: The full SignalId namespace: every selected stream plus every source
    #: stream a selected feature stream references, even one this run does not
    #: emit. A feature descriptor's ``source_stream_id`` must name a mapped
    #: nonzero SignalId, so a feature-only selection still cross-references its
    #: source, and this records that mapping (contract section 8.3).
    signal_ids: Mapping[str, int]
    clock_domains: Mapping[str, int]
    feature_sets: Mapping[str, int]
    units: Mapping[str, int]


def _synthesized_registry(
    reader: NrfReader, resolution: _Resolution, idx: Mapping[str, Sequence[_IndexRow]]
) -> _Registry:
    """Assign the native IDs a synthesized run declares.

    Every namespace is filled independently from its NRF descriptor's string
    ids, sorted by UTF-8 byte order and numbered from 1, so two runs over the
    same selected set produce the same IDs and the same schema regardless of
    insertion order -- and regardless of any ``native_id`` a recorder happened
    to store, which is provenance here and never an input.
    """
    selected = resolution.selected_streams
    clocks = _numbered(str(reader.stream(stream_id)["clock_id"]) for stream_id in selected)
    referenced_sources: set[str] = set()
    feature_set_keys: list[str] = []
    for stream_id in selected:
        feature_set_id = reader.stream(stream_id).get("feature_set_id")
        if not feature_set_id:
            continue
        feature_set_keys.append(str(feature_set_id))
        source = reader.feature_set(str(feature_set_id)).get("source_stream_id")
        if source:
            referenced_sources.add(str(source))
    feature_ids = _numbered(feature_set_keys)
    unit_ids = _numbered(
        str(unit)
        for feature_set_id in feature_ids
        for unit in reader.feature_set(feature_set_id)["unit_ids"]
    )
    # The SignalId namespace is the selected (emitted) streams plus every source
    # stream a selected feature stream references, so a feature-only selection
    # can still name its source with a nonzero id. Section 8.3 makes every
    # cross-reference use a mapped native id, never 0, and a feature descriptor
    # whose source was not selected is the case that needs the source in the
    # namespace without emitting it.
    signal_ids = _numbered(set(selected) | referenced_sources)

    signals: dict[str, _SynthesizedSignal] = {}
    for stream_id in selected:
        descriptor = reader.stream(stream_id)
        rate, rate_from_plan = _synthesized_rate(reader, stream_id, descriptor)
        dtype = _NUMPY_TO_NATIVE.get(str(descriptor["dtype"]))
        if dtype is None:
            raise ReplayConfigError(
                f"stream {stream_id!r} stores {descriptor['dtype']!r}, which no native signal "
                "dtype represents"
            )
        kind = _NRF_KIND_TO_NATIVE.get(str(descriptor["kind"]))
        if kind is None:
            raise ReplayConfigError(
                f"stream {stream_id!r} is of kind {descriptor['kind']!r}, which no native "
                "signal kind represents"
            )
        if kind == "SPIKE":
            raise ReplayConfigError(
                f"stream {stream_id!r} is a spike stream; stream_frames v1 synthesizes dense "
                "fixed-rate blocks from a per-stream block index, but a native spike signal "
                "requires a zero-rate, fixed-size sparse block the index does not describe. "
                "Spike replay synthesis is not defined in v1"
            )
        rows = idx[stream_id]
        feature_set_id = descriptor.get("feature_set_id")
        signals[stream_id] = _SynthesizedSignal(
            stream_id=stream_id,
            native_id=signal_ids[stream_id],
            clock_domain=clocks[str(descriptor["clock_id"])],
            n_channels=len(descriptor["unit_ids"]),
            dtype=dtype,
            kind=kind,
            feature_set_id=feature_ids[str(feature_set_id)] if feature_set_id else 0,
            observation_timing="REGULAR" if kind == "FEATURE" else "NOT_APPLICABLE",
            device_tick_tracking=(
                "SAMPLE_COUNTER"
                if "device_tick_start" in resolution.block_idx_columns[stream_id]
                else "UNAVAILABLE"
            ),
            rate=rate,
            rate_from_plan=rate_from_plan,
            max_block_samples=max((row.n_samples for row in rows), default=1) or 1,
        )
    return _Registry(signals, signal_ids, clocks, feature_ids, unit_ids)


def _synthesized_rate(
    reader: NrfReader, stream_id: str, descriptor: Mapping[str, Any]
) -> tuple[tuple[int, int], bool]:
    """Return the rate a synthesized signal declares, and where it came from.

    A native ``SignalSchema`` must carry a rate; an NRF stream with explicit
    timing carries none, only per-observation timestamps. Rather than invent
    one, the rate is taken in order from two *recorded* facts -- the stream's
    own regular-timing declaration, then the recording plan when the session
    declares one -- and the second is reported as such in the fidelity record.
    Deriving a rate from stored timestamps would be a guess about the grid, and
    section 8.6 does not admit an unlabelled derivation in place of a fact.
    """
    timing = descriptor["timing"]
    if timing["mode"] == "regular":
        return (int(timing["rate"]["numerator"]), int(timing["rate"]["denominator"])), False
    extension = (reader.manifest.get("extensions") or {}).get(NATIVE_REPLAY_NAMESPACE)
    if isinstance(extension, Mapping):
        plan = plan_from_manifest_extension(extension)
        for stream in plan.streams:
            if stream.stream_id != stream_id:
                continue
            for signal in plan.native_schema.signals:
                if signal.id == stream.native_signal_id:
                    return signal.fs, True
    raise ReplayConfigError(
        f"stream {stream_id!r} declares explicit timing and this session carries no recording "
        "plan, so no recorded sample rate exists for it; a synthesized native signal must "
        "declare one, and v1 does not derive a rate from stored timestamps"
    )


def _numbered(values: Any) -> dict[str, int]:
    """Number distinct string ids from 1, in UTF-8 byte order."""
    distinct = sorted({str(value) for value in values}, key=lambda text: text.encode("utf-8"))
    return {value: idx + 1 for idx, value in enumerate(distinct)}


# --- schema sections ---------------------------------------------------------


def _plan_schema(resolution: _Resolution, writer: ReplayImageWriter) -> bytes:
    """Encode the original ``StreamSchema`` the recording plan froze."""
    plan = resolution.plan
    assert plan is not None
    schema = plan.native_schema
    body = bytearray(
        image_format.SCHEMA_HEADER.pack(
            schema.schema_id, len(schema.signals), len(schema.feature_sets), len(schema.units)
        )
    )
    for signal in schema.signals:
        body += image_format.SIGNAL.pack(
            signal.id,
            signal.clock_domain,
            signal.n_channels,
            signal.nominal_block_samples,
            signal.max_block_samples,
            writer.strings.intern(signal.dtype),
            writer.strings.intern(signal.layout),
            writer.strings.intern(signal.device_tick_tracking),
            writer.strings.intern(signal.kind),
            writer.strings.intern(signal.physical_unit),
            signal.channel_set_id,
            signal.calibration_id,
            signal.reference_id,
            signal.feature_set_id,
            writer.strings.intern(signal.observation_timing),
            0,
            signal.fs[0],
            signal.fs[1],
            signal.fixed_block_bytes,
            signal.max_block_bytes,
        )
    for feature_set in schema.feature_sets:
        names = writer.lists.add(
            [writer.strings.intern(name) for name in feature_set.feature_names]
        )
        units = writer.lists.add(list(feature_set.unit_ids))
        body += image_format.FEATURE_SET.pack(
            feature_set.id,
            feature_set.source_stream_id,
            writer.strings.intern(feature_set.source_stream),
            writer.strings.intern(feature_set.algorithm_name),
            writer.strings.intern(feature_set.algorithm_version),
            writer.strings.intern(feature_set.timestamp_reference),
            names[0],
            names[1],
            units[0],
            units[1],
            feature_set.window_length_ns,
            feature_set.shift_ns,
            0,
        )
    for unit in schema.units:
        body += image_format.UNIT.pack(
            unit.id,
            writer.strings.intern(unit.symbol),
            writer.strings.intern(unit.description),
            0,
        )
    return bytes(body)


def _synthesized_schema(
    reader: NrfReader,
    resolution: _Resolution,
    writer: ReplayImageWriter,
    registry: _Registry,
) -> bytes:
    """Encode the one union ``StreamSchema`` a synthesized run declares."""
    feature_sets = sorted(registry.feature_sets.items(), key=lambda item: item[1])
    units = sorted(registry.units.items(), key=lambda item: item[1])
    body = bytearray(
        image_format.SCHEMA_HEADER.pack(
            SYNTHESIZED_SCHEMA_ID,
            len(registry.signals),
            len(feature_sets),
            len(units),
        )
    )
    for stream_id in resolution.selected_streams:
        signal = registry.signals[stream_id]
        item_bytes = np.dtype(_NATIVE_DTYPES[signal.dtype]).itemsize
        block_bytes = item_bytes * signal.n_channels * signal.max_block_samples
        body += image_format.SIGNAL.pack(
            signal.native_id,
            signal.clock_domain,
            signal.n_channels,
            signal.max_block_samples,
            signal.max_block_samples,
            writer.strings.intern(signal.dtype),
            writer.strings.intern("SAMPLE_MAJOR"),
            writer.strings.intern(signal.device_tick_tracking),
            writer.strings.intern(signal.kind),
            writer.strings.intern("UNSPECIFIED"),
            # ChannelSetId, CalibrationId and ReferenceId are not promised by
            # this mode: they are synthesized 0 and reported unavailable.
            0,
            0,
            0,
            signal.feature_set_id,
            writer.strings.intern(signal.observation_timing),
            0,
            signal.rate[0],
            signal.rate[1],
            block_bytes,
            block_bytes,
        )
    for feature_set_id, native_id in feature_sets:
        descriptor = reader.feature_set(feature_set_id)
        source = str(descriptor["source_stream_id"])
        names = writer.lists.add(
            [writer.strings.intern(str(name)) for name in descriptor["feature_names"]]
        )
        unit_ids = writer.lists.add([registry.units[str(unit)] for unit in descriptor["unit_ids"]])
        source_id = registry.signal_ids.get(source, 0)
        body += image_format.FEATURE_SET.pack(
            native_id,
            source_id,
            writer.strings.intern(source),
            writer.strings.intern(str(descriptor["algorithm"]["name"])),
            writer.strings.intern(str(descriptor["algorithm"]["version"])),
            writer.strings.intern(str(descriptor["timestamp_reference"])),
            names[0],
            names[1],
            unit_ids[0],
            unit_ids[1],
            int(descriptor["window_length_ns"]),
            int(descriptor["shift_ns"]),
            0,
        )
    for unit_id, native_id in units:
        body += image_format.UNIT.pack(
            native_id,
            writer.strings.intern(unit_id),
            writer.strings.intern(unit_id),
            0,
        )
    return bytes(body)


# --- section encoding --------------------------------------------------------


def _encode_id_registry(writer: ReplayImageWriter, registry: _Registry) -> bytes:
    """Encode the full string -> native ID mapping for every synthesized namespace.

    Section 8.3 requires the mapping for every namespace to be recorded in the
    replay metadata, so a reader can resolve any cross-reference without
    re-deriving it. Each namespace is sorted by UTF-8 byte order, the order the
    IDs were assigned in, and each entry is a string-table index paired with its
    native id.
    """
    namespaces = (
        registry.signal_ids,
        registry.clock_domains,
        registry.feature_sets,
        registry.units,
    )
    body = bytearray(struct.pack("<IIII", *(len(namespace) for namespace in namespaces)))
    for namespace in namespaces:
        for key, native_id in sorted(namespace.items(), key=lambda item: item[0].encode("utf-8")):
            body += struct.pack("<II", writer.strings.intern(key), int(native_id))
    return bytes(body)


def _encode_sections(
    reader: NrfReader,
    config: ReplayConfig,
    resolution: _Resolution,
    writer: ReplayImageWriter,
    compiled: _Compiled,
    schema: bytes,
    registry: _Registry | None = None,
) -> None:
    ledger_based = config.ledger_based
    plan = resolution.plan

    fidelity = bytearray()
    for stream_id in resolution.selected_streams:
        counts = compiled.per_stream.get(stream_id, [0, 0, 0])
        columns = resolution.block_idx_columns.get(stream_id, ())
        column_list = writer.lists.add([writer.strings.intern(name) for name in columns])
        flags = 0
        if ledger_based:
            flags |= image_format.FLAG_CLOCK_SYNC_AVAILABLE
            flags |= image_format.FLAG_DESCRIPTOR_METADATA_AVAILABLE
        elif compiled.plan_rated_streams and stream_id in compiled.plan_rated_streams:
            flags |= image_format.FLAG_RATE_FROM_RECORDING_PLAN
        native_id = compiled.stream_signal_ids.get(stream_id, 0)
        if plan is not None:
            for stream in plan.streams:
                if stream.stream_id == stream_id:
                    native_id = stream.native_signal_id
        fidelity += image_format.FIDELITY.pack(
            writer.strings.intern(stream_id),
            flags,
            column_list[0],
            column_list[1],
            native_id,
            counts[0],
            counts[1],
            counts[2],
            resolution.committed_blocks.get(stream_id, 0),
            0,
        )

    stream_ranges = bytearray()
    for stream_id, window in sorted(resolution.stream_ranges.items()):
        stream_ranges += image_format.STREAM_RANGE.pack(
            writer.strings.intern(stream_id),
            writer.strings.intern(window.unit),
            window.start,
            window.stop,
            resolution.committed_blocks.get(stream_id, 0),
        )

    flags = image_format.FLAG_LEDGER_BASED if ledger_based else 0
    # The image records facts about the source, not the request that built it:
    # ``allow_incomplete`` is a request bit that only decides whether an
    # incomplete session is built at all, and on a complete session it changes
    # neither the emitted data nor the terminal result. Encoding it would make
    # one fingerprint cover two byte-different images, so both flags state the
    # source fact -- it is incomplete -- and a complete source produces one
    # image whether or not the caller set allow_incomplete (section 8.5).
    if resolution.incomplete:
        flags |= image_format.FLAG_ALLOW_INCOMPLETE
        flags |= image_format.FLAG_ABNORMAL_END_REQUIRED
    if ledger_based:
        flags |= image_format.FLAG_CLOCK_SYNC_AVAILABLE
        flags |= image_format.FLAG_DESCRIPTOR_METADATA_AVAILABLE

    selected = writer.lists.add(
        [writer.strings.intern(stream_id) for stream_id in resolution.selected_streams]
    )
    planned = writer.lists.add(list(plan.planned_signal_ids) if plan else [])
    recorded = writer.lists.add(list(plan.recorded_signal_ids) if plan else [])
    native_session_id = _native_session_id(reader) if ledger_based else None
    source_messages = (
        len(resolution.committed_ordinals)
        if ledger_based
        else sum(resolution.committed_blocks.values())
    )
    start, stop = resolution.message_range or (0, 0)

    summary = image_format.SUMMARY.pack(
        writer.strings.intern(config.mode),
        writer.strings.intern(plan.coverage if plan else None),
        writer.strings.intern(resolution.completeness),
        writer.strings.intern(reader.session_id),
        writer.strings.intern(plan.fingerprint if plan else None),
        writer.strings.intern(resolution.source_fingerprint),
        flags,
        selected[0],
        selected[1],
        planned[0],
        planned[1],
        recorded[0],
        recorded[1],
        writer.strings.intern(None if ledger_based else FRAME_CONSTRUCTION),
        writer.strings.intern(None if ledger_based else ORDERING_KEY),
        0,
        start,
        stop,
        len(compiled.items),
        compiled.frame_count,
        compiled.discontinuity_count,
        len(compiled.blocks),
        len(compiled.gaps),
        len(compiled.omissions),
        writer.payload_byte_count,
        image_format.UINT64_ABSENT if native_session_id is None else native_session_id,
        source_messages,
        image_format.UINT64_ABSENT
        if compiled.first_timeline_ns is None
        else compiled.first_timeline_ns,
        image_format.UINT64_ABSENT
        if compiled.last_timeline_ns is None
        else compiled.last_timeline_ns,
    )

    writer.add_section(image_format.SECTION_SUMMARY, summary)
    writer.add_section(image_format.SECTION_SCHEMA, schema)
    writer.add_section(image_format.SECTION_STREAM_RANGES, bytes(stream_ranges))
    writer.add_section(image_format.SECTION_ITEMS, b"".join(compiled.items))
    writer.add_section(image_format.SECTION_BLOCKS, b"".join(compiled.blocks))
    writer.add_section(image_format.SECTION_GAPS, b"".join(compiled.gaps))
    writer.add_section(image_format.SECTION_OMISSIONS, b"".join(compiled.omissions))
    writer.add_section(image_format.SECTION_FIDELITY, bytes(fidelity))
    if registry is not None:
        # The id-registry section is encoded before the string table is frozen,
        # because interning its keys is what fills part of that table.
        writer.add_section(image_format.SECTION_ID_REGISTRY, _encode_id_registry(writer, registry))
    # The string and list tables are appended last because encoding every other
    # section is what fills them.
    writer.add_section(image_format.SECTION_STRINGS, writer.strings.encode())
    writer.add_section(image_format.SECTION_LISTS, writer.lists.encode())


def _native_session_id(reader: NrfReader) -> int | None:
    columns = reader.read_records(NATIVE_FRAMES.schema_id, 0, 1)
    values = columns.get("native_session_id") or []
    return int(values[0]) if values else None


# --- fault positioning (section 8.11) ----------------------------------------


def _validate_faults(image: ReplayImage, config: ReplayConfig) -> tuple[ResolvedFault, ...]:
    """Position every configured fault and reject the ones section 8.11 forbids.

    A position outside the requested range is legal: the fault never fires and
    is reported as unfired rather than silently dropped. What is not legal is a
    fault that could not have its declared effect -- an effect aimed at the
    wrong kind of item, or a ``sequence_gap`` with no emitted frame on one
    side, which would produce no observable jump at all.
    """
    if not config.faults:
        return ()
    positions = _positions_of(image)
    frames = [item.index for item in image.items() if item.kind == "frame"]
    resolved: list[ResolvedFault] = []
    for fault in config.faults:
        target = fault.target
        if config.ledger_based:
            key: tuple[Any, ...] = ("ordinal", target.data_message_ordinal)
        elif target.kind == "frame":
            key = ("block", target.stream_id, target.block_ordinal)
        else:
            key = ("record", target.stream_id, target.discontinuity_record_ordinal)
        idx = positions.get(key)
        if idx is not None:
            found = image.item(idx)
            if found.kind != target.kind:
                raise ReplayConfigError(
                    f"the fault target {target.document()} names a {found.kind}, not a "
                    f"{target.kind}"
                )
            if fault.effect == "sequence_gap" and idx in (frames[0], frames[-1]):
                pos = "first" if idx == frames[0] else "last"
                raise ReplayConfigError(
                    "a sequence_gap target must have an emitted frame on both sides so the "
                    f"consumer sees a jump; {target.document()} is this run's {pos} "
                    "emitted frame"
                )
        resolved.append(
            ResolvedFault(
                effect=fault.effect,
                fired=idx is not None,
                item_index=idx,
                stall_ns=fault.stall_ns,
                target=target.document(),
            )
        )
    return tuple(resolved)


def _positions_of(image: ReplayImage) -> dict[tuple[Any, ...], int]:
    """Map every recorded identity a fault may name onto its item index.

    A run-local sequence is never a key here: section 8.11 positions a fault by
    a recorded identity precisely so that the same configuration means the same
    thing across ranges, and the replay-run sequence is request-local.
    """
    streams = {entry.native_signal_id: entry.stream_id for entry in image.fidelity()}
    positions: dict[tuple[Any, ...], int] = {}
    for item in image.items():
        if item.data_message_ordinal is not None:
            positions[("ordinal", item.data_message_ordinal)] = item.index
        elif item.kind == "frame":
            for block in image.blocks_of(item):
                positions[("block", block.stream_id, block.source_block_ordinal)] = item.index
        else:
            for gap in image.gaps_of(item):
                stream_id = streams.get(gap.native_signal_id)
                if stream_id is not None and gap.signal_gap_ordinal is not None:
                    positions[("record", stream_id, gap.signal_gap_ordinal)] = item.index
    return positions


__all__ = [
    "ReplayImageBuild",
    "ResolvedFault",
    "build_replay_image",
    "load_replay_image",
    "open_replay_image",
    "resolve_replay_faults",
]
