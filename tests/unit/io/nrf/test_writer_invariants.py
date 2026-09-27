#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Contracts the first NRF writer/reader implementation violated.

Each test here corresponds to a defect the review found: a writer that could
produce a session its own reader rejects, or one that silently changed the data
it was given. They are grouped in one module so the invariants stay visible as
a set rather than scattered among the round-trip tests that missed them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurale.data import (
    ChannelInfo,
    ChannelTable,
    Clock,
    Event,
    EventSeries,
    FeatureMatrix,
    Recording,
    SignalArray,
    Trial,
    TrialTable,
)
from neurale.io.nrf import (
    NrfReader,
    NrfSemanticError,
    NrfWriter,
    write_recording,
)
from neurale.io.nrf._canonical import verify_record_checksum
from neurale.io.nrf._schemas import CHECKPOINT_SCHEMA, validate_document
from neurale.io.nrf._semantics import validate_manifest

from .nrf_support import (
    CREATED_AT,
    NEURAL,
    SESSION_ID,
    SPEC_DIR,
    build_writer,
    register_discontinuities,
    register_records,
)

# --- 1. checkpoints must conform to checkpoint.schema.json ----------------


@pytest.fixture
def checkpoint(tmp_path: Path) -> dict:
    """The checkpoint document a session takes after its first commit."""
    root = tmp_path / "checkpoint.nrf"
    writer = build_writer(root)
    writer.append_stream("neural", NEURAL[:8])
    writer.commit()
    checkpoint_id = writer.checkpoint()
    writer.close()
    path = writer._root / "journal" / "checkpoints" / f"{checkpoint_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_writer_checkpoint_validates_bundled_schema(checkpoint: dict) -> None:
    validate_document(checkpoint, CHECKPOINT_SCHEMA)


def test_writer_checkpoint_validates_normative_schema(checkpoint: dict) -> None:
    """The bundled copy is vendored; check the specification's own file too."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SPEC_DIR / "checkpoint.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        checkpoint
    )


def test_checkpoint_preserves_object_ownership_graph(checkpoint: dict) -> None:
    """Recovery starts from a checkpoint, so truncated descriptors break it."""
    assert verify_record_checksum(checkpoint)
    assert checkpoint["last_committed_transaction_id"] == "tx-0000000000000001"
    assert checkpoint["index_extents"] == {}
    assert checkpoint["committed_objects"]

    chunk = next(
        entry for entry in checkpoint["committed_objects"] if entry["logical_role"] == "array_chunk"
    )
    assert chunk["byte_length"] > 0
    assert chunk["content_kind"] == "zarr_chunk"
    assert "chunk_coordinate" in chunk

    snapshot = next(
        entry
        for entry in checkpoint["committed_objects"]
        if entry["logical_role"] == "metadata_snapshot"
    )
    # A metadata snapshot has no chunk coordinate; the optional member must be
    # absent rather than present-and-meaningless.
    assert "chunk_coordinate" not in snapshot


# --- 2. committed extent may never exceed the frozen Zarr capacity --------


def test_append_beyond_capacity_is_rejected_before_staging(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capacity.nrf"
    writer = build_writer(root)
    oversized = np.zeros((72, 2), dtype="int16")  # the stream declares capacity 64
    with pytest.raises(NrfSemanticError, match="capacity"):
        writer.append_stream("neural", oversized)
    # Nothing staged, nothing journalled, nothing buffered.
    assert not (root / ".staging").exists()
    assert writer.committed_state.extent("streams/neural/data") == 0
    writer.close()


def test_capacity_is_enforced_across_accumulated_appends(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "accumulate.nrf")
    block = np.zeros((32, 2), dtype="int16")
    writer.append_stream("neural", block)
    writer.append_stream("neural", block)  # exactly at capacity 64
    with pytest.raises(NrfSemanticError, match="capacity"):
        writer.append_stream("neural", np.zeros((1, 2), dtype="int16"))
    writer.close()


def test_record_capacity_is_enforced(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "recordcap.nrf")
    schema = writer.manifest["record_schemas"][0]
    capacity = writer._contracts[schema["path"]].capacity
    count = capacity + 1
    with pytest.raises(NrfSemanticError, match="capacity"):
        writer.append_records(
            "events-v1",
            {
                "event_id": [f"e{idx}" for idx in range(count)],
                "time_ns": list(range(count)),
                "duration_ns": [0] * count,
                "label": [None] * count,
            },
        )
    writer.close()


def test_session_filled_to_capacity_still_validates(tmp_path: Path) -> None:
    """The boundary case must remain a session the reader accepts."""
    root = tmp_path / "full.nrf"
    writer = build_writer(root)
    writer.append_stream("neural", np.zeros((64, 2), dtype="int16"))
    writer.commit()
    writer.finalize()
    writer.close()
    with NrfReader.open(root, verify_checksums=True) as reader:
        assert reader.stream_extent("neural") == 64
        validate_manifest(reader.manifest)


# --- 3. explicit timestamps advance with their data ----------------------


def _explicit_stream_writer(root: Path, *, timestamp_chunk_length: int | None) -> NrfWriter:
    writer = NrfWriter.create(root, session_id=SESSION_ID, created_at=CREATED_AT)
    registry = writer.registry
    registry.register_clock(
        "host-clock",
        clock_type="host_monotonic",
        rate={"numerator": 1_000_000_000, "denominator": 1},
    )
    registry.register_unit("volt", symbol="V")
    registry.register_channel("channel-0000", idx=0, unit_id="volt")
    registry.register_schema("schema-source", stream_ids=["cursor"])
    register_records(registry)
    register_discontinuities(registry, "cursor")
    registry.register_stream(
        "cursor",
        kind="behavioral",
        dtype="float32",
        channel_ids=[],
        unit_ids=["volt", "volt"],
        clock_id="host-clock",
        schema_id="schema-source",
        chunk_length=2,
        capacity=32,
        timestamp_chunk_length=timestamp_chunk_length,
        discontinuity_record_schema_id="discontinuities-cursor-v1",
        timing_mode="explicit",
    )
    return writer


def test_mismatched_timestamp_chunk_length_is_rejected_at_freeze(tmp_path: Path) -> None:
    writer = _explicit_stream_writer(tmp_path / "mismatch.nrf", timestamp_chunk_length=4)
    with pytest.raises(NrfSemanticError, match="timestamp chunk length"):
        writer.freeze()
    writer.close()


def test_timestamps_and_data_reach_same_extent(tmp_path: Path) -> None:
    root = tmp_path / "linked.nrf"
    writer = _explicit_stream_writer(root, timestamp_chunk_length=None)
    writer.freeze()
    for i in range(3):
        writer.append_stream(
            "cursor",
            np.full((2, 2), i, dtype="float32"),
            timestamps=np.array([i * 2, i * 2 + 1], dtype="int64"),
        )
        writer.commit()
        data = writer.committed_state.extent("streams/cursor/data")
        stamps = writer.committed_state.extent("streams/cursor/timestamps")
        assert data == stamps, f"data {data} and timestamps {stamps} diverged"
    writer.finalize()
    writer.close()

    with NrfReader.open(root, verify_checksums=True) as reader:
        assert reader.stream_extent("cursor") == 6
        assert len(reader.read_timestamps("cursor")) == 6
        validate_manifest(reader.manifest)


def test_short_final_chunk_seals_data_and_timestamps_together(tmp_path: Path) -> None:
    root = tmp_path / "seal.nrf"
    writer = _explicit_stream_writer(root, timestamp_chunk_length=None)
    writer.freeze()
    # Three rows against a chunk length of two: one full chunk plus a tail.
    writer.append_stream(
        "cursor",
        np.arange(6, dtype="float32").reshape(3, 2),
        timestamps=np.array([0, 1, 2], dtype="int64"),
    )
    writer.finalize()
    writer.close()

    with NrfReader.open(root, verify_checksums=True) as reader:
        assert reader.stream_extent("cursor") == 3
        assert np.array_equal(reader.read_timestamps("cursor"), np.array([0, 1, 2]))
        assert "streams/cursor/timestamps" in reader.committed_state.sealed_targets


# --- 4. a failed append leaves the writer unchanged ----------------------


def _tail_rows(writer: NrfWriter, target_path: str) -> int:
    tail = writer._tails.get(target_path)
    return tail.rows if tail else 0


def test_failed_append_does_not_touch_either_tail(tmp_path: Path) -> None:
    root = tmp_path / "guard.nrf"
    writer = _explicit_stream_writer(root, timestamp_chunk_length=None)
    writer.freeze()

    with pytest.raises(NrfSemanticError, match="one timestamp per item"):
        writer.append_stream(
            "cursor",
            np.zeros((2, 2), dtype="float32"),
            timestamps=np.array([0], dtype="int64"),
        )
    assert _tail_rows(writer, "streams/cursor/data") == 0
    assert _tail_rows(writer, "streams/cursor/timestamps") == 0

    # The corrected retry must not double-write.
    writer.append_stream(
        "cursor",
        np.zeros((2, 2), dtype="float32"),
        timestamps=np.array([0, 1], dtype="int64"),
    )
    assert _tail_rows(writer, "streams/cursor/data") == 2
    assert _tail_rows(writer, "streams/cursor/timestamps") == 2
    writer.close()


def test_regular_stream_rejects_timestamps_without_buffering(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "regular.nrf")
    with pytest.raises(NrfSemanticError, match="takes no timestamps"):
        writer.append_stream("neural", NEURAL[:4], timestamps=np.array([0, 1, 2, 3], dtype="int64"))
    assert _tail_rows(writer, "streams/neural/data") == 0
    writer.close()


def test_failed_record_append_does_not_buffer_rows(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "recguard.nrf")
    with pytest.raises(NrfSemanticError):
        writer.append_records("events-v1", {"event_id": ["a"]})  # missing fields
    assert _tail_rows(writer, "records/events/events-v1") == 0
    writer.close()


# --- 5. typed Recording round trip ---------------------------------------


@pytest.fixture
def source_recording() -> Recording:
    """One recording that exercises every typed member the writer maps."""
    channels = ChannelTable(
        [
            ChannelInfo(name="E1", index=0, type="ecog", unit="uV", contact=0),
            ChannelInfo(name="E2", index=1, type="ecog", unit="uV", contact=1, bad=True),
        ]
    )
    signal = SignalArray(
        data=np.arange(20, dtype="int16").reshape(10, 2),
        fs=4000.0,
        time=None,
        t0=0.5,
        clock=Clock("acquisition", "device", synchronization_domain="rig"),
        channels=channels,
        unit="uV",
        name="wideband",
    )
    features = FeatureMatrix(
        data=np.arange(6, dtype="float64").reshape(3, 2),
        fs=100.0,
        feature_names=["70-200:E1", "70-200:E2"],
        source_signal="wideband",
        window_size=0.1,
        shift=0.01,
        unit="V^2",
    )
    events = EventSeries(
        [
            Event(onset=0.25, duration=0.05, label="go", code=7, source="task"),
            Event(onset=0.75, duration=0.0, label=None, code=None),
        ]
    )
    trials = TrialTable(
        [
            Trial(trial_id=1, start=0.0, stop=0.5, label="center-out", outcome="success"),
            Trial(trial_id=2, start=0.5, stop=1.0, label=None, outcome="failure", block=3),
        ]
    )
    return Recording(
        signals={"wideband": signal},
        features={"bandpower": features},
        events=events,
        trials=trials,
        subject={"id": "subject-01"},
        session={"id": SESSION_ID, "subject": {"id": "subject-01"}},
    )


@pytest.fixture
def written_session(tmp_path: Path, source_recording: Recording) -> Path:
    return write_recording(
        tmp_path / "recording.nrf",
        source_recording,
        session_id=SESSION_ID,
        created_at=CREATED_AT,
    )


@pytest.fixture
def restored(written_session: Path) -> Recording:
    """What :data:`source_recording` becomes after a write and a read."""
    with NrfReader.open(written_session, verify_checksums=True) as reader:
        assert reader.legacy_termination_normal
        return reader.to_recording()


def test_recording_round_trip_preserves_signals_and_metadata(
    source_recording: Recording, restored: Recording
) -> None:
    original = source_recording.signals["wideband"]
    signal = restored.signals["wideband"]
    assert np.array_equal(signal.data, original.data)
    assert signal.data.dtype == original.data.dtype
    assert signal.fs == original.fs
    assert signal.t0 == pytest.approx(original.t0)
    assert [channel.name for channel in signal.channels] == ["E1", "E2"]
    assert [channel.type for channel in signal.channels] == ["ecog", "ecog"]
    assert [channel.bad for channel in signal.channels] == [False, True]
    assert signal.clock is not None
    assert signal.clock.synchronization_domain == "rig"


def test_recording_round_trip_preserves_features(
    source_recording: Recording, restored: Recording
) -> None:
    original = source_recording.features["bandpower"]
    features = restored.features["bandpower"]
    assert np.array_equal(features.data, original.data)
    assert features.data.dtype == original.data.dtype
    assert features.feature_names == original.feature_names
    assert features.window_size == pytest.approx(original.window_size)
    assert features.shift == pytest.approx(original.shift)
    assert features.source_signal == "wideband"


def test_recording_round_trip_preserves_events_and_trials(restored: Recording) -> None:
    assert restored.events is not None
    events = list(restored.events)
    assert [event.onset for event in events] == pytest.approx([0.25, 0.75])
    assert [event.duration for event in events] == pytest.approx([0.05, 0.0])
    assert [event.label for event in events] == ["go", None]
    assert [event.code for event in events] == [7, None]

    assert restored.trials is not None
    trials = list(restored.trials)
    assert [trial.start for trial in trials] == pytest.approx([0.0, 0.5])
    assert [trial.stop for trial in trials] == pytest.approx([0.5, 1.0])
    assert [trial.label for trial in trials] == ["center-out", None]
    assert [trial.outcome for trial in trials] == ["success", "failure"]
    assert [trial.block for trial in trials] == [None, 3]


def test_round_trip_keeps_metadata_and_unmapped_records(
    restored: Recording,
) -> None:
    assert restored.session is not None
    assert restored.session["id"] == SESSION_ID
    assert restored.subject == {"id": "subject-01"}
    # The termination row has no typed home and must stay visible.
    unmapped = restored.metadata["nrf_unmapped_records"]
    assert unmapped["termination-v1"][0]["termination_kind"] == "normal"


def test_written_recording_session_is_valid_nrf(written_session: Path) -> None:
    with NrfReader.open(written_session, verify_checksums=True) as reader:
        validate_manifest(reader.manifest)
        assert reader.legacy_termination_normal


def test_recording_with_unsupported_dtype_is_rejected(tmp_path: Path) -> None:
    signal = SignalArray(
        data=np.zeros((4, 1), dtype="complex128"),
        fs=100.0,
        time=None,
        t0=0.0,
        clock=None,
        channels=ChannelTable([ChannelInfo(name="E1", index=0, type="ecog", unit="uV")]),
        unit="uV",
        name="complex",
    )
    with pytest.raises(NrfSemanticError, match="does not define"):
        write_recording(
            tmp_path / "complex.nrf",
            Recording(signals={"complex": signal}),
            session_id=SESSION_ID,
            created_at=CREATED_AT,
        )


# --- 6. primary keys are unique for the life of the session ---------------


def test_duplicate_primary_key_across_appends_is_rejected(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "keys.nrf")
    row = {"event_id": ["same"], "time_ns": [0], "duration_ns": [0], "label": [None]}
    writer.append_records("events-v1", dict(row))
    with pytest.raises(NrfSemanticError, match="already used in this session"):
        writer.append_records("events-v1", dict(row))
    assert _tail_rows(writer, "records/events/events-v1") == 1
    writer.close()


def test_duplicate_primary_key_is_rejected_after_commit(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "committedkeys.nrf")
    writer.append_records(
        "events-v1",
        {
            "event_id": ["a", "b"],
            "time_ns": [0, 1],
            "duration_ns": [0, 0],
            "label": [None, None],
        },
    )
    writer.commit()
    with pytest.raises(NrfSemanticError, match="already used in this session"):
        writer.append_records(
            "events-v1", {"event_id": ["b"], "time_ns": [2], "duration_ns": [0], "label": [None]}
        )
    writer.close()


def test_distinct_schemas_have_independent_key_spaces(tmp_path: Path) -> None:
    writer = build_writer(tmp_path / "spaces.nrf")
    writer.append_records(
        "events-v1", {"event_id": ["x"], "time_ns": [0], "duration_ns": [0], "label": [None]}
    )
    # The same string in a different record set is a different identity.
    writer.append_records(
        "trials-v1",
        {"trial_id": ["x"], "start_ns": [0], "stop_ns": [1], "label": [None]},
    )
    writer.close()


# --- 7. no silent dtype conversion ---------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        np.array([[1.7, 2.9], [3.1, 4.8]], dtype="float64"),
        np.array([[1, 2], [3, 4]], dtype="int64"),
        np.array([[1, 2], [3, 4]], dtype="uint64"),
    ],
)
def test_lossy_stream_dtype_is_rejected(tmp_path: Path, payload: np.ndarray) -> None:
    writer = build_writer(tmp_path / f"dtype-{payload.dtype.name}.nrf")
    with pytest.raises(NrfSemanticError, match="lossy conversion"):
        writer.append_stream("neural", payload)
    assert _tail_rows(writer, "streams/neural/data") == 0
    writer.close()


def test_widening_stream_dtype_is_accepted(tmp_path: Path) -> None:
    """A safe cast changes no value, so it is not a silent conversion."""
    root = tmp_path / "widen.nrf"
    writer = build_writer(root)
    payload = np.array([[1, 2], [3, 4]], dtype="int8")  # int8 -> int16 is safe
    writer.append_stream("neural", np.vstack([payload, payload]))
    writer.commit()
    writer.finalize()
    writer.close()
    with NrfReader.open(root) as reader:
        assert np.array_equal(reader.read_stream("neural"), np.vstack([payload, payload]))


def test_byte_order_change_is_not_lossy(tmp_path: Path) -> None:
    root = tmp_path / "endian.nrf"
    writer = build_writer(root)
    big_endian = np.arange(1, 9, dtype=">i2").reshape(4, 2)
    writer.append_stream("neural", big_endian)
    writer.commit()
    writer.finalize()
    writer.close()
    with NrfReader.open(root) as reader:
        stored = reader.read_stream("neural")
        assert np.array_equal(stored, big_endian)
        # Stored little-endian as the manifest declares, with equal values.
        assert stored.dtype.byteorder in {"<", "="}


def test_lossy_timestamp_dtype_is_rejected(tmp_path: Path) -> None:
    writer = _explicit_stream_writer(tmp_path / "stamps.nrf", timestamp_chunk_length=None)
    writer.freeze()
    with pytest.raises(NrfSemanticError, match="lossy conversion"):
        writer.append_stream(
            "cursor",
            np.zeros((2, 2), dtype="float32"),
            timestamps=np.array([0.5, 1.5], dtype="float64"),
        )
    assert _tail_rows(writer, "streams/cursor/data") == 0
    writer.close()
