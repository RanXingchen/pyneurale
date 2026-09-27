#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""End-to-end NRF v1 sessions: writer, reader, and committed visibility.

A writer-to-reader round trip alone would only prove the two agree with each
other, so these tests also check the session against things the implementation
does not control: zarr-python must be able to open the arrays, the manifest must
pass both normative validation layers, and the chunk bytes must match the
specification's payload vectors.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest import mock

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

import numpy as np
import pytest

from neurale.io.nrf import (
    ManifestBuilder,
    NrfCorruptionError,
    NrfReader,
    NrfSemanticError,
    NrfStateError,
    NrfWriter,
)
from neurale.io.nrf._paths import JOURNAL_TRANSACTIONS, MANIFEST
from neurale.io.nrf._schemas import MANIFEST_SCHEMA, validate_document
from neurale.io.nrf._semantics import validate_manifest

from .nrf_support import (
    BANDPOWER,
    CREATED_AT,
    CURSOR,
    CURSOR_TIMES,
    EVENT_ROWS,
    NEURAL,
    SESSION_ID,
    VECTORS,
    build_writer,
    first_file,
    register_discontinuities,
    register_records,
)
from .package_objects import Objects, snapshot

#: Where every committed neural chunk of a :func:`build_writer` session lives.
NEURAL_CHUNKS = "streams/neural/data/c"


@pytest.fixture
def session(tmp_path: Path) -> Path:
    """A complete, normally terminated session."""
    root = Objects(tmp_path / "complete.nrf")
    writer = build_writer(root)
    writer.append_stream("neural", NEURAL)
    writer.append_stream("cursor", CURSOR, timestamps=CURSOR_TIMES)
    writer.append_stream("bandpower", BANDPOWER)
    writer.commit()
    writer.append_records("events-v1", EVENT_ROWS)
    writer.append_records(
        "trials-v1",
        {
            "trial_id": ["trial-0001", "trial-0002"],
            "start_ns": [0, 30_000_000],
            "stop_ns": [25_000_000, 60_000_000],
            "label": ["center-out", None],
        },
    )
    writer.append_records(
        "experiment-state-v1",
        {
            "state_transition_id": ["state-0001", "state-0002"],
            "time_ns": [0, 5_000_000],
            "new_state": ["hold", "reach"],
        },
    )
    writer.commit()
    writer.checkpoint()
    writer.finalize()
    writer.close()
    return root


# --- round trip -----------------------------------------------------------


def test_streams_round_trip_exactly(session: Path) -> None:
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)
        assert np.array_equal(reader.read_stream("cursor"), CURSOR)
        assert np.array_equal(reader.read_stream("bandpower"), BANDPOWER)


def test_dtypes_and_axis_order_are_preserved(session: Path) -> None:
    with NrfReader.open(session) as reader:
        assert reader.read_stream("neural").dtype == np.dtype("int16")
        assert reader.read_stream("cursor").dtype == np.dtype("float32")
        assert reader.read_stream("bandpower").dtype == np.dtype("float64")
        assert reader.read_stream("neural").shape == NEURAL.shape


def test_explicit_timestamps_round_trip(session: Path) -> None:
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_timestamps("cursor"), CURSOR_TIMES)
        # A regular-timing stream has no timestamp array at all.
        assert reader.read_timestamps("neural") is None


def test_records_round_trip_with_null_preservation(session: Path) -> None:
    with NrfReader.open(session) as reader:
        events = reader.read_records("events-v1")
        assert events["event_id"] == ["event-0001", "event-0002"]
        assert events["time_ns"] == [10_000_000, 20_000_000]
        # The missing label comes back as None, from the validity array rather
        # than from a sentinel value in the data.
        assert events["label"] == ["go", None]


def test_random_range_reads_match_whole_stream(session: Path) -> None:
    with NrfReader.open(session) as reader:
        whole = reader.read_stream("neural")
        for start, stop in [(0, 1), (2, 6), (5, 10), (9, 10), (0, 10)]:
            assert np.array_equal(reader.read_stream("neural", start, stop), whole[start:stop])


def test_range_beyond_committed_extent_is_clamped(session: Path) -> None:
    with NrfReader.open(session) as reader:
        extent = reader.stream_extent("neural")
        assert np.array_equal(reader.read_stream("neural", 0, extent + 100), NEURAL)
        assert reader.read_stream("neural", extent + 5, extent + 9).shape[0] == 0


def test_iter_blocks_is_bounded_and_complete(session: Path) -> None:
    with NrfReader.open(session) as reader:
        blocks = list(reader.iter_blocks("neural", block_size=3))
        assert [start for start, _ in blocks] == [0, 3, 6, 9]
        assert all(block.shape[0] <= 3 for _, block in blocks)
        assert np.array_equal(np.concatenate([block for _, block in blocks]), NEURAL)


def test_iter_records_yields_stored_order(session: Path) -> None:
    with NrfReader.open(session) as reader:
        rows = list(reader.iter_records("events-v1"))
        assert [row["event_id"] for row in rows] == ["event-0001", "event-0002"]
        assert rows[1]["label"] is None


def test_reading_one_stream_ignores_others(session: Path) -> None:
    # Damage the immutable package before opening it.
    for chunk in (session / "streams" / "cursor" / "data" / "c").rglob("*"):
        if chunk.is_file():
            chunk.unlink()
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)


# --- session state --------------------------------------------------------


def test_finalized_session_is_complete(session: Path) -> None:
    with NrfReader.open(session) as reader:
        assert reader.legacy_termination_normal
        assert reader.termination is not None
        assert reader.termination.kind == "normal"
        assert reader.termination.fault_id is None


def test_descriptors_are_listed(session: Path) -> None:
    with NrfReader.open(session) as reader:
        assert reader.stream_ids() == ["neural", "cursor", "bandpower"]
        assert "events-v1" in reader.record_schema_ids()
        assert reader.feature_set_ids() == ["bandpower-70-200"]
        assert reader.session_id == SESSION_ID
        assert reader.feature_set("bandpower-70-200")["window_length_ns"] == 100_000_000


def test_manifest_passes_normative_layers(session: Path) -> None:
    manifest = json.loads((session / MANIFEST).read_text(encoding="utf-8"))
    validate_document(manifest, MANIFEST_SCHEMA)
    validate_manifest(manifest)


def test_committed_objects_match_checksums(session: Path) -> None:
    NrfReader.open(session, verify_checksums=True).close()


def test_corrupt_committed_object_is_detected(session: Path) -> None:
    chunk = first_file(session, NEURAL_CHUNKS)
    chunk.write_bytes(b"\x00" * chunk.stat().st_size)
    with pytest.raises(NrfCorruptionError, match="checksum"):
        NrfReader.open(session, verify_checksums=True)


def test_missing_committed_object_is_detected(session: Path) -> None:
    first_file(session, NEURAL_CHUNKS).unlink()
    with pytest.raises(NrfCorruptionError, match="missing"):
        NrfReader.open(session, verify_checksums=True)


# --- committed visibility -------------------------------------------------


def test_uncommitted_tail_is_invisible_and_incomplete(
    tmp_path: Path,
) -> None:
    root = Objects(tmp_path / "interrupted.nrf")
    writer = build_writer(root)
    writer.append_stream("neural", NEURAL)
    writer.commit()  # 8 rows fill two chunks; 2 remain in the writer tail
    writer.close()  # no finalize: simulate an interrupted recording
    assert not root.exists()  # unsealed workspace is not a published package
    snapshot(writer)  # explicit damaged-package fixture

    with NrfReader.open(root, verify_checksums=True) as reader:
        assert reader.stream_extent("neural") == 8
        assert np.array_equal(reader.read_stream("neural"), NEURAL[:8])
        # An interrupted session must be detectably incomplete.
        assert not reader.complete
        assert reader.termination is None


def test_appends_without_full_chunk_commit_nothing(tmp_path: Path) -> None:
    root = Objects(tmp_path / "short.nrf")
    writer = build_writer(root)
    writer.append_stream("neural", NEURAL[:3])  # chunk length is 4
    assert writer.commit() is None
    writer.close()
    snapshot(writer)
    with NrfReader.open(root) as reader:
        assert reader.stream_extent("neural") == 0
        assert reader.read_stream("neural").shape[0] == 0


def test_empty_session_round_trips(tmp_path: Path) -> None:
    root = Objects(tmp_path / "empty.nrf")
    writer = build_writer(root)
    writer.finalize()
    writer.close()
    with NrfReader.open(root, verify_checksums=True) as reader:
        assert reader.legacy_termination_normal
        assert reader.stream_extent("neural") == 0
        assert reader.read_stream("neural").shape[0] == 0
        assert list(reader.iter_blocks("neural")) == []


def test_context_manager_finalizes_and_close_is_idempotent(tmp_path: Path) -> None:
    root = Objects(tmp_path / "managed.nrf")
    writer = build_writer(root)
    with writer:
        writer.append_stream("neural", NEURAL[:4])
        writer.commit()
    assert writer.terminated
    writer.close()
    writer.close()
    with NrfReader.open(root) as reader:
        assert reader.legacy_termination_normal


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    root = Objects(tmp_path / "twice.nrf")
    writer = build_writer(root)
    writer.finalize()
    writer.finalize()
    writer.close()
    with NrfReader.open(root) as reader:
        assert reader.legacy_termination_normal


# --- writer contracts -----------------------------------------------------


def test_registries_freeze_after_manifest_write(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "frozen.nrf")
    with pytest.raises(NrfStateError, match="frozen"):
        writer.registry.register_unit("ampere", symbol="A")
    writer.close()


def test_appending_after_termination_is_rejected(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "closed.nrf")
    writer.finalize()
    with pytest.raises(NrfStateError, match="terminated"):
        writer.append_stream("neural", NEURAL[:4])
    writer.close()


def test_payload_width_must_match_declared_stream(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "width.nrf")
    with pytest.raises(NrfSemanticError, match="width"):
        writer.append_stream("neural", np.zeros((4, 3), dtype="int16"))
    writer.close()


def test_explicit_timing_requires_timestamp_per_item(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "stamps.nrf")
    with pytest.raises(NrfSemanticError, match="one timestamp per item"):
        writer.append_stream("cursor", CURSOR, timestamps=CURSOR_TIMES[:1])
    with pytest.raises(NrfSemanticError, match="requires timestamps"):
        writer.append_stream("cursor", CURSOR)
    writer.close()


def test_regular_timing_rejects_timestamps(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "regular.nrf")
    with pytest.raises(NrfSemanticError, match="takes no timestamps"):
        writer.append_stream("neural", NEURAL[:4], timestamps=CURSOR_TIMES)
    writer.close()


def test_record_appends_validate_columns(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "records.nrf")
    with pytest.raises(NrfSemanticError, match="missing fields"):
        writer.append_records("events-v1", {"event_id": ["a"]})
    with pytest.raises(NrfSemanticError, match="unknown fields"):
        writer.append_records(
            "events-v1",
            {
                "event_id": ["a"],
                "time_ns": [0],
                "duration_ns": [0],
                "label": [None],
                "surprise": [1],
            },
        )
    with pytest.raises(NrfSemanticError, match="different lengths"):
        writer.append_records(
            "events-v1",
            {"event_id": ["a", "b"], "time_ns": [0], "duration_ns": [0], "label": [None]},
        )
    writer.close()


def test_duplicate_primary_keys_are_rejected(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "duplicate.nrf")
    with pytest.raises(NrfSemanticError, match="unique"):
        writer.append_records(
            "events-v1",
            {
                "event_id": ["same", "same"],
                "time_ns": [0, 1],
                "duration_ns": [0, 0],
                "label": [None, None],
            },
        )
    writer.close()


def test_null_primary_key_is_rejected(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "nullkey.nrf")
    with pytest.raises(NrfSemanticError, match="must not be null"):
        writer.append_records(
            "events-v1",
            {"event_id": [None], "time_ns": [0], "duration_ns": [0], "label": [None]},
        )
    writer.close()


def test_session_over_non_empty_directory_is_rejected(tmp_path: Path) -> None:
    root = Objects(tmp_path / "occupied.nrf")
    root.mkdir()
    (root / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(NrfStateError, match="already exists"):
        NrfWriter.create(root, session_id=SESSION_ID, created_at=CREATED_AT)


def test_unknown_stream_and_schema_are_rejected(session: Path) -> None:
    with NrfReader.open(session) as reader:
        with pytest.raises(NrfSemanticError, match="unknown stream"):
            reader.read_stream("absent")
        with pytest.raises(NrfSemanticError, match="unknown record schema"):
            reader.read_records("absent-v1")


# --- external conformance -------------------------------------------------


def test_arrays_are_readable_by_zarr_python(session: Path) -> None:
    zarr = pytest.importorskip("zarr")
    from zarr.storage import ZipStore

    with ZipStore(str(session), mode="r") as store:
        arr = zarr.open_array(store=store, path="streams/neural/data", mode="r")
        assert arr.shape == (64, 2)
        assert np.array_equal(np.asarray(arr[:10]), NEURAL)


def test_chunk_bytes_match_normative_payload_vector(tmp_path: Path) -> None:
    """The stored chunk must equal the specification's own payload vector."""
    expected = next(item for item in VECTORS["payload_vectors"] if item["name"] == "neural")

    root = Objects(tmp_path / "vector.nrf")
    writer = build_writer(root)
    writer.append_stream("neural", np.arange(1, 9, dtype="int16").reshape(4, 2))
    writer.commit()
    writer.close()

    chunk = root / "streams" / "neural" / "data" / "c" / "0" / "0"
    assert chunk.read_bytes().hex() == expected["payload_hex"]


def test_session_contains_no_pickle(session: Path) -> None:
    for path in session.rglob("*"):
        if path.is_file():
            assert path.suffix not in {".pkl", ".pickle"}
            assert b"pickle" not in path.read_bytes()[:512].lower()


def test_journal_lines_each_end_with_one_newline(session: Path) -> None:
    raw = (session / JOURNAL_TRANSACTIONS).read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.split(b"\n")[:-1]
    assert lines
    for line in lines:
        assert line
        assert json.loads(line.decode("utf-8"))["kind"]


# --- typed reconstruction -------------------------------------------------


def test_typed_signal_preserves_metadata(session: Path) -> None:
    with NrfReader.open(session) as reader:
        signal = reader.read_signal("neural")
        assert np.array_equal(signal.data, NEURAL)
        assert signal.fs == 4000.0
        assert [channel.name for channel in signal.channels] == ["channel-0000", "channel-0001"]
        assert signal.clock is not None
        assert signal.attrs["nrf_stream_id"] == "neural"


def test_typed_features_preserve_descriptor(session: Path) -> None:
    with NrfReader.open(session) as reader:
        features = reader.read_features("bandpower")
        assert np.array_equal(features.data, BANDPOWER)
        assert features.feature_names == ["70-200:E1", "70-200:E2"]
        assert features.source_signal == "neural"
        assert features.window_size == pytest.approx(0.1)
        assert features.shift == pytest.approx(0.01)


def test_recording_reconstruction_exposes_unmapped_records(session: Path) -> None:
    with NrfReader.open(session) as reader:
        recording = reader.to_recording()
        assert set(recording.signals) == {"neural", "cursor"}
        assert set(recording.features) == {"bandpower"}
        assert recording.events is not None
        assert len(recording.events) == 2
        assert recording.trials is not None
        assert len(recording.trials) == 2
        # Experiment state has no typed home; it must remain visible.
        unmapped = recording.metadata["nrf_unmapped_records"]
        assert "experiment-state-v1" in unmapped
        assert [row["new_state"] for row in unmapped["experiment-state-v1"]] == ["hold", "reach"]


def test_unmapped_records_include_termination_row(session: Path) -> None:
    with NrfReader.open(session) as reader:
        unmapped = reader.unmapped_records()
        assert "termination-v1" in unmapped
        assert unmapped["termination-v1"][0]["termination_kind"] == "normal"


# --- hand-built fixtures --------------------------------------------------


def test_reader_rejects_directory_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "bare.nrf").mkdir()
    with pytest.raises(NrfCorruptionError, match="not a single-file NRF package"):
        NrfReader.open(tmp_path / "bare.nrf")


def test_reader_rejects_unknown_major_version(session: Path) -> None:
    manifest = json.loads((session / MANIFEST).read_text(encoding="utf-8"))
    manifest["version"]["major"] = 2
    (session / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    from neurale.io.nrf import NrfSchemaError

    with pytest.raises(NrfSchemaError, match="unsupported major version"):
        NrfReader.open(session)


def test_reader_rejects_manifest_cache_leading_replay(session: Path) -> None:
    manifest = json.loads((session / MANIFEST).read_text(encoding="utf-8"))
    manifest["commit"]["committed_extents"]["streams/neural/data"] = 999
    (session / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((NrfSemanticError, NrfCorruptionError)):
        NrfReader.open(session)


def test_truncated_journal_line_hides_incomplete_record(session: Path) -> None:
    journal = session / JOURNAL_TRANSACTIONS
    raw = journal.read_bytes()
    journal.write_bytes(raw + b'{"kind":"commit","sequence":99')

    with NrfReader.open(session) as reader:
        # The partial line is reported but changes nothing a reader can see.
        assert reader.journal_tail.partial_line_bytes > 0
        assert np.array_equal(reader.read_stream("neural"), NEURAL)


def test_manifest_without_journal_reads_as_empty(tmp_path: Path) -> None:
    """A reader must not need the writer to have produced the session."""
    root = Objects(tmp_path / "handbuilt.nrf")
    builder = ManifestBuilder(
        session_id=SESSION_ID,
        created_at=CREATED_AT,
        writer_name="hand",
        writer_version="1",
    )
    builder.register_clock("host-clock", clock_type="host_monotonic")
    builder.register_unit("volt", symbol="V")
    builder.register_channel("channel-0000", idx=0, unit_id="volt")
    builder.register_schema("schema-source", stream_ids=["neural"])
    register_records(builder)
    register_discontinuities(builder, "neural")
    builder.register_stream(
        "neural",
        kind="neural",
        dtype="int16",
        channel_ids=["channel-0000"],
        unit_ids=["volt"],
        clock_id="host-clock",
        schema_id="schema-source",
        chunk_length=4,
        capacity=16,
        discontinuity_record_schema_id="discontinuities-neural-v1",
        rate={"numerator": 4000, "denominator": 1},
    )
    manifest = builder.build()
    builder.validate(manifest)
    from zipfile import ZIP_DEFLATED, ZipFile

    from neurale.io.nrf._package import PACKAGE_MARKER

    with ZipFile(root, "w", compression=ZIP_DEFLATED) as archive:
        archive.comment = PACKAGE_MARKER
        archive.writestr(MANIFEST, json.dumps(manifest))

    with NrfReader.open(root) as reader:
        assert reader.stream_extent("neural") == 0
        assert not reader.complete
        assert reader.read_stream("neural").shape[0] == 0


def test_orphan_object_beyond_extent_is_ignored(session: Path) -> None:
    """A stray final object that no commit backs must not become visible."""
    orphan = session / "streams" / "neural" / "data" / "c" / "9" / "0"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(first_file(session, NEURAL_CHUNKS).read_bytes())

    with NrfReader.open(session) as reader:
        assert reader.stream_extent("neural") == 10
        assert np.array_equal(reader.read_stream("neural"), NEURAL)


def test_placeholder_channels_are_marked_as_derived(session: Path) -> None:
    """A stream that declared no channels must not look like it named them."""
    with NrfReader.open(session) as reader:
        neural = reader.read_signal("neural")
        cursor = reader.read_signal("cursor")
        assert neural.attrs["nrf_channels_declared"] is True
        assert cursor.attrs["nrf_channels_declared"] is False
        assert [channel.name for channel in cursor.channels] == ["column-0000", "column-0001"]


@pytest.mark.skipif(fcntl is None, reason="descriptor access mode is readable through fcntl only")
def test_synced_descriptors_were_opened_for_writing(tmp_path: Path) -> None:
    """Syncing a read-only descriptor is a no-op at best and an error at worst.

    ``os.fsync`` on Windows is ``FlushFileBuffers``, which requires write
    access and fails with ``EBADF`` without it; the writer once reopened its
    temporary file read-only to sync it, which made every session unwritable
    there while passing on Linux. The platform difference is what hid it, so
    the invariant is asserted directly rather than through an outcome: a
    descriptor that never saw the data cannot be the one that guarantees it
    reached the disk.
    """
    real_fsync = os.fsync
    modes: list[int] = []

    def recording_fsync(fd: int) -> None:
        modes.append(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE)
        real_fsync(fd)

    with mock.patch.object(os, "fsync", recording_fsync):
        writer = build_writer(tmp_path / "synced.nrf")
        writer.append_stream("neural", NEURAL)
        writer.commit()
        writer.close()

    assert modes, "the writer synced nothing at all"
    assert set(modes) <= {os.O_WRONLY, os.O_RDWR}


def test_zarr_encoding_reuses_event_loop(tmp_path: Path) -> None:
    """NRF encoding reuses Zarr's synchronous loop instead of ``asyncio.run``.

    A Windows Proactor loop creates a socket pair.  Creating one for every
    metadata snapshot and chunk eventually blocked the recorder worker in
    CPython's fallback ``accept`` and made the recording suite hang
    nondeterministically.
    """
    with mock.patch.object(
        asyncio,
        "run",
        side_effect=AssertionError("NRF Zarr encoding must reuse Zarr's sync loop"),
    ):
        writer = build_writer(tmp_path / "shared-zarr-loop.nrf")
        writer.append_stream("neural", NEURAL)
        writer.commit()
        writer.close()
