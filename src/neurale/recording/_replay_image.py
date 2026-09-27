#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The private replay-image container, version 1.

A replay image is the compiled form of one replay request: the exact items a
run emits, in emission order, with every field the emitted frames and
discontinuities carry, and the payload bytes themselves. It exists for one
reason: **the native replay source must not
parse JSON or Zarr in its read loop.** Every decision below follows from that.

* **Fixed-size records in flat sections.** Item, block, gap, omission, and
  fidelity records are fixed-width little-endian structures at a known offset,
  so a native reader indexes them; nothing is length-prefixed, variable, or
  compressed on the item path. Text lives in one string table and is referenced
  by index -- a run's read loop never touches it.
* **Payload bytes are embedded.** The alternative -- pointing at the source
  Zarr arrays -- would put chunk decoding, and therefore JSON metadata, in the
  read loop. The cost is that an image duplicates the bytes of the range it
  covers, which is also what makes a sub-range image small, self-contained, and
  independently reproducible.
* **No configuration in the image.** Section 8.3 makes the emitted sequence and
  the reported metadata a pure function of ``(session, mode, selected streams,
  range)``. Pacing, ``speed_factor``, injected faults, and the replay-run
  session id are run attributes and are not here, so two runs that differ only
  in pacing share one image, and an image is byte-identical whenever that
  four-part key is.

The format is **private**: it has no specification directory and no
compatibility promise. It is a cache of a committed NRF session, and the
session -- never the image -- is the source of truth. A version this build does
not understand, a failed checksum, or a fingerprint that does not match the
request is answered by rebuilding, never by reading what can be salvaged.

It has exactly one reader, in this repository and versioned with it:
``cpp/src/recording/replay/image.h``, which is what a native replay
source consumes and what :class:`ReplayImage` below reads through. This module
writes the format; it no longer also decodes it.

That is deliberate. A private container with two decoders can be read two ways,
and the parity test that used to hold this module's decoder and the native one
together skipped wherever the C++ tests were not built -- so on a wheel-only
checkout nothing pinned them at all. With one reader, a layout change the
native side did not follow fails every test that reads back what it wrote,
which is all of them, rather than a test that might not run.

What remains stated twice is the *layout*: the ``struct`` formats below, which
the writer packs with, and the offsets the native reader loads from. That is
inherent to writing in Python and reading in C++, and it is now checked by
construction on every round trip.

Layout::

    0    header            64 bytes, ending in a CRC over itself
    64   section table     section_count x 24 bytes
    ...  sections          in the order the writer appended them

Nothing here builds an image; :mod:`neurale.recording._replay_build` does.
"""

from __future__ import annotations

import os
import struct
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ._errors import ReplayImageError
from ._spool_format import crc32c

MAGIC: Final = b"NRLRPIMG"
FORMAT_VERSION: Final = 1
HEADER_BYTES: Final = 64
SECTION_ENTRY_BYTES: Final = 24
FINGERPRINT_BYTES: Final = 32

UINT32_ABSENT: Final = (1 << 32) - 1
UINT64_ABSENT: Final = (1 << 64) - 1

#: Section identifiers. A reader looks a section up by identifier rather than
#: by position, so a later version may append one without moving the others.
SECTION_STRINGS: Final = 1
SECTION_LISTS: Final = 2
SECTION_SUMMARY: Final = 3
SECTION_SCHEMA: Final = 4
SECTION_STREAM_RANGES: Final = 5
SECTION_ITEMS: Final = 6
SECTION_BLOCKS: Final = 7
SECTION_GAPS: Final = 8
SECTION_OMISSIONS: Final = 9
SECTION_FIDELITY: Final = 10
SECTION_PAYLOAD: Final = 11
SECTION_ID_REGISTRY: Final = 12

#: Item kinds, matching the two data-plane message kinds of section 1.1.
ITEM_FRAME: Final = 1
ITEM_DISCONTINUITY: Final = 2

#: ``omission_reason`` values, in the precedence order section 8.2 freezes for
#: a discontinuity. The numeric value is the on-disk encoding; the order is not
#: incidental, and the builder chooses by it.
OMISSION_NO_PRECEDING_EMITTED_FRAME: Final = 1
OMISSION_TARGET_FRAME_NOT_EMITTED: Final = 2
OMISSION_NO_PROJECTED_SIGNALS: Final = 3
OMISSION_NO_PROJECTED_BLOCKS: Final = 4
OMISSION_TARGET_BLOCK_NOT_EMITTED: Final = 5

OMISSION_REASON_NAMES: Final[Mapping[int, str]] = {
    OMISSION_NO_PRECEDING_EMITTED_FRAME: "no_preceding_emitted_frame",
    OMISSION_TARGET_FRAME_NOT_EMITTED: "target_frame_not_emitted",
    OMISSION_NO_PROJECTED_SIGNALS: "no_projected_signals",
    OMISSION_NO_PROJECTED_BLOCKS: "no_projected_blocks",
    OMISSION_TARGET_BLOCK_NOT_EMITTED: "target_block_not_emitted",
}

#: Summary flag bits.
FLAG_ALLOW_INCOMPLETE: Final = 1 << 0
FLAG_ABNORMAL_END_REQUIRED: Final = 1 << 1
FLAG_LEDGER_BASED: Final = 1 << 2
FLAG_CLOCK_SYNC_AVAILABLE: Final = 1 << 3
FLAG_DESCRIPTOR_METADATA_AVAILABLE: Final = 1 << 4
#: Set on a synthesized stream whose sample rate came from the recording plan
#: rather than from the stream's own regular-timing declaration. An explicit-
#: timing NRF stream carries no rate at all, and a native signal must declare
#: one, so the provenance of that value is reported rather than assumed.
FLAG_RATE_FROM_RECORDING_PLAN: Final = 1 << 5

#: Item flag bits.
ITEM_FLAG_SOURCE_TICK: Final = 1 << 0
ITEM_FLAG_VALID_UNTIL: Final = 1 << 1
ITEM_FLAG_FRAME_LEVEL_GAP: Final = 1 << 2

#: Block flag bits.
BLOCK_FLAG_CLOCK_SYNC: Final = 1 << 0

HEADER = struct.Struct(f"<8sHHI{FINGERPRINT_BYTES}sQII")
SECTION_ENTRY = struct.Struct("<IIQQ")
STRING_HEADER = struct.Struct("<II")
STRING_ENTRY = struct.Struct("<II")
LIST_HEADER = struct.Struct("<II")
SUMMARY = struct.Struct("<" + "I" * 16 + "Q" * 13 + "88x")
SCHEMA_HEADER = struct.Struct("<IIII")
SIGNAL = struct.Struct("<" + "I" * 16 + "Q" * 4)
FEATURE_SET = struct.Struct("<" + "I" * 10 + "Q" * 3)
UNIT = struct.Struct("<IIII")
STREAM_RANGE = struct.Struct("<IIQQQ")
ITEM = struct.Struct("<BBHI" + "Q" * 13 + "IIII")
BLOCK = struct.Struct("<IIII" + "Q" * 14 + "IIII")
GAP = struct.Struct("<IIII" + "Q" * 6)
OMISSION = struct.Struct("<BBHI" + "Q" * 3)
FIDELITY = struct.Struct("<IIII" + "Q" * 6)

assert SUMMARY.size == 256
assert SIGNAL.size == 96
assert FEATURE_SET.size == 64
assert STREAM_RANGE.size == 32
assert ITEM.size == 128
assert BLOCK.size == 144
assert GAP.size == 64
assert OMISSION.size == 32
assert FIDELITY.size == 64

#: Fixed record width per section, or 0 where the section is opaque bytes.
SECTION_ENTRY_WIDTHS: Final[Mapping[int, int]] = {
    SECTION_STRINGS: 0,
    SECTION_LISTS: 0,
    SECTION_SUMMARY: SUMMARY.size,
    SECTION_SCHEMA: 0,
    SECTION_STREAM_RANGES: STREAM_RANGE.size,
    SECTION_ITEMS: ITEM.size,
    SECTION_BLOCKS: BLOCK.size,
    SECTION_GAPS: GAP.size,
    SECTION_OMISSIONS: OMISSION.size,
    SECTION_FIDELITY: FIDELITY.size,
    SECTION_PAYLOAD: 0,
    SECTION_ID_REGISTRY: 0,
}


# --- writing -----------------------------------------------------------------


class StringTable:
    """Interning table for every text value an image carries.

    Deduplicating is not a size optimization here: a stream id repeats once per
    emitted block, and the whole point of the format is that the read loop
    compares small integers instead of bytes.
    """

    def __init__(self) -> None:
        self._values: list[str] = []
        self._idx: dict[str, int] = {}
        self.intern("")

    def intern(self, value: str | None) -> int:
        """Return the index of *value*; ``None`` is the absent marker."""
        if value is None:
            return UINT32_ABSENT
        found = self._idx.get(value)
        if found is not None:
            return found
        idx = len(self._values)
        self._values.append(value)
        self._idx[value] = idx
        return idx

    def encode(self) -> bytes:
        blob = bytearray()
        entries = bytearray()
        for value in self._values:
            encoded = value.encode("utf-8")
            entries += STRING_ENTRY.pack(len(blob), len(encoded))
            blob += encoded
        return STRING_HEADER.pack(len(self._values), 0) + bytes(entries) + bytes(blob)


class ListTable:
    """Flat ``uint32`` arrays referenced by ``(offset, count)`` pairs."""

    def __init__(self) -> None:
        self._values: list[int] = []

    def add(self, values: Sequence[int]) -> tuple[int, int]:
        offset = len(self._values)
        self._values.extend(int(value) for value in values)
        return offset, len(values)

    def encode(self) -> bytes:
        return LIST_HEADER.pack(len(self._values), 0) + struct.pack(
            f"<{len(self._values)}I", *self._values
        )


class ReplayImageWriter:
    """Assemble one image, appending sections in a fixed order.

    Payload bytes are streamed into a scratch file rather than accumulated, so
    building an image of a long session does not need the session in memory --
    the index records are the only thing that scales with the item count, and
    they are what an index is.
    """

    def __init__(self, *, directory: Path, prefix: str) -> None:
        self.strings = StringTable()
        self.lists = ListTable()
        self._sections: list[tuple[int, bytes]] = []
        # Exclusive creation: a pre-existing symlink or ordinary file at a
        # predictable name (``<prefix>.payload``) is never followed or
        # truncated. mkstemp returns a unique name with O_CREAT | O_EXCL, so
        # only a brand-new file is opened, and the fd is what is written to --
        # never a pathname re-opened later (contract section 8: building a
        # replay image from an NRF session is read-only over the session).
        fd, scratch_path = tempfile.mkstemp(
            prefix=f".{prefix}.", suffix=".payload", dir=str(directory)
        )
        self._scratch_path = Path(scratch_path)
        self._scratch = os.fdopen(fd, "w+b")
        self._payload_bytes = 0

    # --- payload ---------------------------------------------------------

    def append_payload(self, data: bytes) -> int:
        """Append payload bytes and return the offset they start at."""
        offset = self._payload_bytes
        self._scratch.write(data)
        self._payload_bytes += len(data)
        return offset

    @property
    def payload_byte_count(self) -> int:
        return self._payload_bytes

    # --- sections --------------------------------------------------------

    def add_section(self, kind: int, payload: bytes) -> None:
        self._sections.append((kind, payload))

    def close(self) -> None:
        self._scratch.close()
        self._scratch_path.unlink(missing_ok=True)

    def write(self, path: Path, fingerprint: bytes) -> None:
        """Write the image to *path* durably, and never in place.

        The temporary file is fsynced before the rename and the directory
        after it, so a crash leaves either the previous image or the new one.
        A half-written image that still had a valid header would be the one
        outcome a cache must never produce.
        """
        if len(fingerprint) != FINGERPRINT_BYTES:
            raise ReplayImageError("an image fingerprint is 32 bytes")
        sections = [*self._sections, (SECTION_PAYLOAD, b"")]
        table_bytes = SECTION_ENTRY_BYTES * len(sections)
        offset = HEADER_BYTES + table_bytes
        entries = bytearray()
        bodies: list[bytes] = []
        for kind, payload in sections:
            length = self._payload_bytes if kind == SECTION_PAYLOAD else len(payload)
            entries += SECTION_ENTRY.pack(kind, SECTION_ENTRY_WIDTHS.get(kind, 0), offset, length)
            offset += length
            if kind != SECTION_PAYLOAD:
                bodies.append(payload)
        total = offset

        content = crc32c_continue(CRC_SEED, entries)
        for body in bodies:
            content = crc32c_continue(content, body)
        self._scratch.flush()
        self._scratch.seek(0)
        while True:
            chunk = self._scratch.read(1 << 20)
            if not chunk:
                break
            content = crc32c_continue(content, chunk)

        header = bytearray(
            HEADER.pack(
                MAGIC,
                FORMAT_VERSION,
                len(sections),
                HEADER_BYTES,
                fingerprint,
                total,
                content,
                0,
            )
        )
        struct.pack_into("<I", header, HEADER_BYTES - 4, crc32c(bytes(header[: HEADER_BYTES - 4])))

        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(bytes(header))
                handle.write(bytes(entries))
                for body in bodies:
                    handle.write(body)
                self._scratch.seek(0)
                while True:
                    chunk = self._scratch.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            _fsync_directory(path.parent)
        except BaseException:
            # A half-written or stranded temporary is never the previous image
            # and must not be left for a later call to mistake for one; the
            # rename is the only step that publishes, so a failure before it
            # leaves the previous image (if any) untouched and the scratch
            # cleaned up.
            tmp.unlink(missing_ok=True)
            raise


#: Seed of an empty CRC-32C, so a chained checksum starts somewhere named.
CRC_SEED: Final = 0


def crc32c_continue(seed: int, data: bytes | memoryview) -> int:
    """Fold *data* into a running CRC-32C and return the new value.

    An image is written in chunks precisely so that it is never all in memory at
    once, so its checksum has to be chainable. It is the container's kernel with
    the state threaded through, which is the point: an image and a spool are
    checked by the same code, so neither can drift into a checksum the other
    would reject.
    """
    return crc32c(data, seed)


# --- reading -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayBlock:
    """One emitted signal block, with its recorded provenance."""

    native_signal_id: int
    block_index_in_frame: int
    stream_id: str
    n_samples: int
    sample_idx_start: int
    last_sample_idx: int
    device_tick_start: int
    observation_time_start_ns: int
    payload_offset: int
    payload_byte_count: int
    original_payload_offset: int | None
    clock_sync: Mapping[str, int] | None
    source_block_ordinal: int
    recorded_frame_sequence: int | None


@dataclass(frozen=True, slots=True)
class ReplayItem:
    """One emitted data message: a frame or a discontinuity (section 1.1)."""

    index: int
    kind: str
    data_message_ordinal: int | None
    replay_frame_sequence: int
    original_frame_sequence: int | None
    previous_replay_sequence: int | None
    original_previous_frame_sequence: int | None
    timeline_ns: int
    original_host_received_ns: int | None
    source_tick: int | None
    valid_until_ns: int | None
    native_schema_id: int
    source_clock_domain: int
    frame_flags: int
    reason: str | None
    frame_level_gap: bool
    child_first: int
    n_children: int
    payload_offset: int
    payload_byte_count: int
    original_payload_byte_count: int | None


@dataclass(frozen=True, slots=True)
class ReplayGap:
    """One per-signal gap carried by an emitted discontinuity."""

    native_signal_id: int
    gap_index_in_message: int
    reason: str | None
    gap_flags: int
    expected_sample_index: int
    actual_sample_index: int
    missing_samples: int | None
    expected_device_tick: int | None
    actual_device_tick: int | None
    signal_gap_ordinal: int | None


@dataclass(frozen=True, slots=True)
class OmittedMessage:
    """One source message the projection could not emit (sections 8.2, 8.3)."""

    kind: str
    reason: str
    data_message_ordinal: int | None
    original_frame_sequence: int | None
    stream_id: str | None
    source_ordinal: int | None


@dataclass(frozen=True, slots=True)
class StreamFidelity:
    """What one selected stream was reconstructed from (section 8.3)."""

    stream_id: str
    native_signal_id: int
    block_index_columns_present: tuple[str, ...]
    clock_sync_available: bool
    descriptor_metadata_available: bool
    rate_from_recording_plan: bool
    frames_emitted: int
    blocks_emitted: int
    discontinuities_emitted: int
    committed_blocks: int
    range_start: int | None
    range_stop: int | None


@dataclass(frozen=True, slots=True)
class ReplayImageMetadata:
    """Everything the image reports about the run it compiled."""

    mode: str
    image_fingerprint: str
    source_fingerprint: str
    source_session_id: str
    plan_fingerprint: str | None
    native_session_id: int | None
    selected_streams: tuple[str, ...]
    planned_signal_ids: tuple[int, ...]
    recorded_signal_ids: tuple[int, ...]
    plan_coverage: str | None
    completeness: str | None
    allow_incomplete: bool
    abnormal_end_required: bool
    clock_sync_available: bool
    descriptor_metadata_available: bool
    frame_construction: str | None
    ordering_key: str | None
    message_range: tuple[int, int] | None
    stream_ranges: Mapping[str, tuple[int, int]]
    n_items: int
    n_frames: int
    n_discontinuities: int
    n_blocks: int
    n_gaps: int
    n_omissions: int
    payload_byte_count: int
    source_message_count: int
    first_timeline_ns: int | None
    last_timeline_ns: int | None
    #: The full string -> native ID mapping for every synthesized namespace
    #: (signals, clocks, feature_sets, units), or empty namespaces for a
    #: ledger-based mode whose ids come from the recording plan (section 8.3).
    id_registry: Mapping[str, Mapping[str, int]]

    def document(self) -> dict[str, Any]:
        """Return the metadata as a plain document, for reporting and tests."""
        return {
            "mode": self.mode,
            "image_fingerprint": self.image_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "source_session_id": self.source_session_id,
            "plan_fingerprint": self.plan_fingerprint,
            "native_session_id": self.native_session_id,
            "selected_streams": list(self.selected_streams),
            "planned_signal_ids": list(self.planned_signal_ids),
            "recorded_signal_ids": list(self.recorded_signal_ids),
            "plan_coverage": self.plan_coverage,
            "completeness": self.completeness,
            "allow_incomplete": self.allow_incomplete,
            "abnormal_end_required": self.abnormal_end_required,
            "clock_sync_available": self.clock_sync_available,
            "descriptor_metadata_available": self.descriptor_metadata_available,
            "frame_construction": self.frame_construction,
            "ordering_key": self.ordering_key,
            "message_range": None if self.message_range is None else list(self.message_range),
            "stream_ranges": {
                stream_id: list(value) for stream_id, value in sorted(self.stream_ranges.items())
            },
            "n_items": self.n_items,
            "frame_count": self.n_frames,
            "discontinuity_count": self.n_discontinuities,
            "n_blocks": self.n_blocks,
            "n_gaps": self.n_gaps,
            "n_omissions": self.n_omissions,
            "payload_byte_count": self.payload_byte_count,
            "source_message_count": self.source_message_count,
            "first_timeline_ns": self.first_timeline_ns,
            "last_timeline_ns": self.last_timeline_ns,
            "id_registry": {
                name: dict(sorted(mapping.items())) for name, mapping in self.id_registry.items()
            },
        }


class ReplayImage:
    """A validated, read-only view of one replay image.

    The container is read by the native reader in
    ``cpp/src/recording/replay/image.h`` and by nothing else. This class
    opens it, keeps the handle, and turns each decoded record into the value
    objects above: absent markers become ``None``, string-table indices become
    strings, and flag words become the booleans and groupings a caller reads.

    That split is the point. A private container with two decoders is a
    container that can be read two ways, and the parity test that used to hold
    them together skipped wherever the C++ tests were not built. Now the writer
    in this module and the reader in the native library are exercised against
    each other by every test that reads back what it wrote.

    Opening validates the container and nothing else: the magic, the format
    version, the header CRC, the section table's bounds, the content CRC, and
    that every fixed-width section holds a whole number of records. A file that
    fails any of those is :class:`ReplayImageError`, never a partial read --
    the cache path answers that by rebuilding.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        try:
            self._native = _native_image().NativeReplayImage(str(self._path))
        except Exception as error:
            # Every refusal the native reader can produce is terminal for this
            # file. The status is a machine word; what a caller needs is which
            # rule the file broke, so it is spelled out here rather than passed
            # through -- the cache path reads these to decide what to say when
            # it rebuilds.
            raise ReplayImageError(
                f"{_REFUSAL_REASONS.get(str(error), 'the replay image is not readable')}: "
                f"{self._path}"
            ) from error
        self._closed = False
        try:
            self._metadata = self._read_metadata()
        except Exception:
            self.close()
            raise

    # --- lifecycle -------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> ReplayImage:
        return cls(Path(path))

    def close(self) -> None:
        native = getattr(self, "_native", None)
        if native is not None and not getattr(self, "_closed", True):
            native.close()
            self._closed = True

    def __enter__(self) -> ReplayImage:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    # --- table lookups ----------------------------------------------------

    def _string(self, idx: int) -> str | None:
        """One interned string, or ``None`` for the absent marker."""
        return self._native.string(idx)

    def _list(self, offset: int, count: int) -> tuple[int, ...]:
        return tuple(self._native.list_at(offset + step) for step in range(count))

    def _strings(self, offset: int, count: int) -> tuple[str, ...]:
        return tuple(self._string(value) or "" for value in self._list(offset, count))

    # --- metadata ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        return self._native.fingerprint.hex()

    @property
    def metadata(self) -> ReplayImageMetadata:
        return self._metadata

    @property
    def mode(self) -> str:
        return self._metadata.mode

    def _read_id_registry(self) -> Mapping[str, Mapping[str, int]]:
        image = _native_image()
        spaces = {
            "signals": image.ReplayIdNamespace.signals,
            "clocks": image.ReplayIdNamespace.clocks,
            "feature_sets": image.ReplayIdNamespace.feature_sets,
            "units": image.ReplayIdNamespace.units,
        }
        registry: dict[str, dict[str, int]] = {}
        for name, space in spaces.items():
            mapping: dict[str, int] = {}
            for i in range(self._native.id_registry_count(space)):
                entry = self._native.id_registry_entry(space, i)
                key = self._string(entry.name)
                if key is None:
                    raise ReplayImageError("a replay image id-registry entry names no string")
                mapping[key] = int(entry.native_id)
            registry[name] = mapping
        return registry

    def _read_metadata(self) -> ReplayImageMetadata:
        summary = self._native.summary
        ledger_based = bool(summary.flags & FLAG_LEDGER_BASED)
        stream_ranges = {
            entry.stream_id: (entry.range_start, entry.range_stop)
            for entry in self.fidelity()
            if entry.range_start is not None and entry.range_stop is not None
        }
        return ReplayImageMetadata(
            mode=self._string(summary.mode) or "",
            image_fingerprint=self.fingerprint,
            source_fingerprint=self._string(summary.source_fingerprint) or "",
            source_session_id=self._string(summary.source_session_id) or "",
            plan_fingerprint=self._string(summary.plan_fingerprint),
            native_session_id=_optional(summary.native_session_id),
            selected_streams=self._strings(summary.selected_list, summary.n_selected),
            planned_signal_ids=self._list(summary.planned_list, summary.n_planned),
            recorded_signal_ids=self._list(summary.recorded_list, summary.n_recorded),
            plan_coverage=self._string(summary.plan_coverage),
            completeness=self._string(summary.completeness),
            allow_incomplete=bool(summary.flags & FLAG_ALLOW_INCOMPLETE),
            abnormal_end_required=bool(summary.flags & FLAG_ABNORMAL_END_REQUIRED),
            clock_sync_available=bool(summary.flags & FLAG_CLOCK_SYNC_AVAILABLE),
            descriptor_metadata_available=bool(summary.flags & FLAG_DESCRIPTOR_METADATA_AVAILABLE),
            frame_construction=self._string(summary.frame_construction),
            ordering_key=self._string(summary.ordering_key),
            message_range=(summary.range_start, summary.range_stop) if ledger_based else None,
            stream_ranges={} if ledger_based else stream_ranges,
            n_items=summary.n_items,
            n_frames=summary.n_frames,
            n_discontinuities=summary.n_discontinuities,
            n_blocks=summary.n_blocks,
            n_gaps=summary.n_gaps,
            n_omissions=summary.n_omissions,
            payload_byte_count=summary.payload_byte_count,
            source_message_count=summary.source_message_count,
            first_timeline_ns=_optional(summary.first_timeline_ns),
            last_timeline_ns=_optional(summary.last_timeline_ns),
            id_registry=self._read_id_registry(),
        )

    # --- item access -----------------------------------------------------

    @property
    def n_items(self) -> int:
        return self._native.n_items

    def item(self, idx: int) -> ReplayItem:
        if idx < 0 or idx >= self.n_items:
            raise IndexError(f"replay image item {idx} is out of range")
        record = self._native.item(idx)
        flags = record.flags
        return ReplayItem(
            index=idx,
            kind=record.kind.name,
            data_message_ordinal=_optional(record.data_message_ordinal),
            replay_frame_sequence=record.replay_sequence,
            original_frame_sequence=_optional(record.original_sequence),
            previous_replay_sequence=_optional(record.previous_replay_sequence),
            original_previous_frame_sequence=_optional(record.original_previous_sequence),
            timeline_ns=record.timeline_ns,
            original_host_received_ns=_optional(record.original_host_received_ns),
            source_tick=record.source_tick if flags & ITEM_FLAG_SOURCE_TICK else None,
            valid_until_ns=record.valid_until_ns if flags & ITEM_FLAG_VALID_UNTIL else None,
            native_schema_id=record.native_schema_id,
            source_clock_domain=record.source_clock_domain,
            frame_flags=record.frame_flags,
            reason=self._string(record.reason),
            frame_level_gap=bool(flags & ITEM_FLAG_FRAME_LEVEL_GAP),
            child_first=record.child_first,
            n_children=record.n_children,
            payload_offset=record.payload_offset,
            payload_byte_count=record.payload_byte_count,
            original_payload_byte_count=_optional(record.original_payload_byte_count),
        )

    def items(self) -> Iterator[ReplayItem]:
        for i in range(self.n_items):
            yield self.item(i)

    def blocks_of(self, item: ReplayItem) -> tuple[ReplayBlock, ...]:
        if item.kind != "frame":
            return ()
        blocks: list[ReplayBlock] = []
        for offset in range(item.child_first, item.child_first + item.n_children):
            if offset >= self._native.n_blocks:
                raise ReplayImageError("a replay image frame names a block it does not carry")
            record = self._native.block(offset)
            clock_sync = (
                {
                    "device_tick_reference": record.clock_sync_device_tick_reference,
                    "host_time_reference_ns": record.clock_sync_host_time_reference_ns,
                    "rate_numerator": record.clock_sync_rate_numerator,
                    "rate_denominator": record.clock_sync_rate_denominator,
                    "uncertainty_ns": record.clock_sync_uncertainty_ns,
                    "clock_domain": record.clock_sync_clock_domain,
                    "generation": record.clock_sync_generation,
                    "flags": record.clock_sync_flags,
                }
                if record.flags & BLOCK_FLAG_CLOCK_SYNC
                else None
            )
            blocks.append(
                ReplayBlock(
                    native_signal_id=record.native_signal_id,
                    block_index_in_frame=record.block_idx_in_frame,
                    stream_id=self._string(record.stream) or "",
                    n_samples=record.n_samples,
                    sample_idx_start=record.sample_idx_start,
                    last_sample_idx=record.last_sample_idx,
                    device_tick_start=record.device_tick_start,
                    observation_time_start_ns=record.observation_time_start_ns,
                    payload_offset=record.payload_offset,
                    payload_byte_count=record.payload_byte_count,
                    original_payload_offset=_optional(record.original_payload_offset),
                    clock_sync=clock_sync,
                    source_block_ordinal=record.source_block_ordinal,
                    recorded_frame_sequence=_optional(record.recorded_frame_sequence),
                )
            )
        return tuple(blocks)

    def gaps_of(self, item: ReplayItem) -> tuple[ReplayGap, ...]:
        if item.kind != "discontinuity":
            return ()
        gaps: list[ReplayGap] = []
        for offset in range(item.child_first, item.child_first + item.n_children):
            if offset >= self._native.n_gaps:
                raise ReplayImageError("a replay image discontinuity names a gap it does not carry")
            record = self._native.gap(offset)
            gaps.append(
                ReplayGap(
                    native_signal_id=record.native_signal_id,
                    gap_index_in_message=record.gap_idx_in_message,
                    reason=self._string(record.reason),
                    gap_flags=record.gap_flags,
                    expected_sample_index=record.expected_sample_idx,
                    actual_sample_index=record.actual_sample_idx,
                    missing_samples=_optional(record.missing_samples),
                    expected_device_tick=_optional(record.expected_device_tick),
                    actual_device_tick=_optional(record.actual_device_tick),
                    signal_gap_ordinal=_optional(record.signal_gap_ordinal),
                )
            )
        return tuple(gaps)

    def payload_of(self, item: ReplayItem) -> bytes:
        """Return the frame payload the run emits, exactly as it is stored."""
        if item.payload_byte_count == 0:
            return b""
        try:
            return self._native.payload(item.payload_offset, item.payload_byte_count)
        except Exception as error:
            raise ReplayImageError(
                "a replay image frame names payload it does not carry"
            ) from error

    def omissions(self) -> tuple[OmittedMessage, ...]:
        entries: list[OmittedMessage] = []
        for i in range(self._native.n_omissions):
            record = self._native.omission(i)
            entries.append(
                OmittedMessage(
                    kind=record.kind.name,
                    reason=OMISSION_REASON_NAMES.get(record.reason, "unknown"),
                    data_message_ordinal=_optional(record.data_message_ordinal),
                    original_frame_sequence=_optional(record.original_frame_sequence),
                    stream_id=self._string(record.stream),
                    source_ordinal=_optional(record.source_ordinal),
                )
            )
        return tuple(entries)

    def fidelity(self) -> tuple[StreamFidelity, ...]:
        """Return the per-stream fidelity entries, one per selected stream."""
        ranges = self._stream_range_bounds()
        entries: list[StreamFidelity] = []
        for i in range(self._native.n_fidelities):
            record = self._native.fidelity(i)
            stream_id = self._string(record.stream) or ""
            start, stop = ranges.get(stream_id, (None, None))
            entries.append(
                StreamFidelity(
                    stream_id=stream_id,
                    native_signal_id=record.native_signal_id,
                    block_index_columns_present=self._strings(
                        record.block_idx_column_first, record.block_idx_column_count
                    ),
                    clock_sync_available=bool(record.flags & FLAG_CLOCK_SYNC_AVAILABLE),
                    descriptor_metadata_available=bool(
                        record.flags & FLAG_DESCRIPTOR_METADATA_AVAILABLE
                    ),
                    rate_from_recording_plan=bool(record.flags & FLAG_RATE_FROM_RECORDING_PLAN),
                    frames_emitted=record.frames_emitted,
                    blocks_emitted=record.blocks_emitted,
                    discontinuities_emitted=record.discontinuities_emitted,
                    committed_blocks=record.committed_blocks,
                    range_start=start,
                    range_stop=stop,
                )
            )
        return tuple(entries)

    def _stream_range_bounds(self) -> dict[str, tuple[int, int]]:
        bounds: dict[str, tuple[int, int]] = {}
        for i in range(self._native.stream_range_count):
            record = self._native.stream_range(i)
            bounds[self._string(record.stream) or ""] = (record.range_start, record.range_stop)
        return bounds

    def schema_document(self) -> dict[str, Any]:
        """Return the run's declared ``StreamSchema`` as a plain document."""
        if not self._native.n_signals and not self._native.schema_id:
            return {}
        signals = []
        for i in range(self._native.n_signals):
            record = self._native.signal(i)
            signals.append(
                {
                    "id": record.id,
                    "clock_domain": record.clock_domain,
                    "n_channels": record.n_channels,
                    "nominal_block_samples": record.nominal_block_samples,
                    "max_block_samples": record.max_block_samples,
                    "dtype": self._string(record.dtype),
                    "layout": self._string(record.layout),
                    "device_tick_tracking": self._string(record.device_tick_tracking),
                    "kind": self._string(record.kind),
                    "physical_unit": self._string(record.physical_unit),
                    "channel_set_id": record.channel_set_id,
                    "calibration_id": record.calibration_id,
                    "reference_id": record.reference_id,
                    "feature_set_id": record.feature_set_id,
                    "observation_timing": self._string(record.observation_timing),
                    "fs": {
                        "numerator": record.rate_num,
                        "denominator": record.rate_den,
                    },
                    "fixed_block_bytes": record.fixed_block_bytes,
                    "max_block_bytes": record.max_block_bytes,
                }
            )
        feature_sets = []
        for i in range(self._native.feature_set_count):
            record = self._native.feature_set(i)
            feature_sets.append(
                {
                    "id": record.id,
                    "source_stream_id": record.source_stream_id,
                    "source_stream": self._string(record.source_stream),
                    "algorithm_name": self._string(record.algorithm_name),
                    "algorithm_version": self._string(record.algorithm_version),
                    "timestamp_reference": self._string(record.timestamp_reference),
                    "feature_names": [
                        self._string(value)
                        for value in self._list(
                            record.feature_name_first, record.feature_name_count
                        )
                    ],
                    "unit_ids": list(self._list(record.unit_id_first, record.unit_id_count)),
                    "window_length_ns": record.window_length_ns,
                    "shift_ns": record.shift_ns,
                }
            )
        units = []
        for i in range(self._native.n_units):
            record = self._native.unit(i)
            units.append(
                {
                    "id": record.id,
                    "symbol": self._string(record.symbol),
                    "description": self._string(record.description),
                }
            )
        return {
            "schema_id": self._native.schema_id,
            "signals": signals,
            "feature_sets": feature_sets,
            "units": units,
        }


#: What each native refusal means, in the words the cache path and its tests
#: read. The native reader answers with a status; naming the broken rule is the
#: reader-facing half of that answer and belongs on this side of the boundary.
_REFUSAL_REASONS: Final[Mapping[str, str]] = {
    "unreadable": "the replay image cannot be opened or mapped",
    "not_an_image": "the file does not start with a replay-image magic",
    "checksum_mismatch": "the replay image does not match its checksum",
    "unsupported_version": (
        f"the replay image is not the format version {FORMAT_VERSION} this build writes; "
        "rebuild the image"
    ),
    "truncated": "the replay image is shorter than the length it declares",
    "malformed": "the replay image container is self-inconsistent",
}


def _native_image() -> Any:
    """Return the native replay-image bindings, loading them on first use.

    Loaded here rather than at module scope for the reason
    :mod:`neurale.recording._native` states: ``import neurale.recording`` must
    keep working on a build with no native extension. Writing an image needs no
    native code; only reading one does.
    """
    global _NATIVE_IMAGE
    if _NATIVE_IMAGE is None:
        from neurale._native_loader import load_native_namespace

        _NATIVE_IMAGE = load_native_namespace("recording").replay_image
    return _NATIVE_IMAGE


_NATIVE_IMAGE: Any = None


def _optional(value: int) -> int | None:
    return None if value == UINT64_ABSENT else value


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable where the platform supports it."""
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


__all__ = [
    "FORMAT_VERSION",
    "OmittedMessage",
    "ReplayBlock",
    "ReplayGap",
    "ReplayImage",
    "ReplayImageMetadata",
    "ReplayItem",
    "StreamFidelity",
]
