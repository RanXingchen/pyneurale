#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The replay image builder.

Every session here is a **real** one: a native recorder writes a spool, the
finalizer converts it, and the builder reads the committed NRF the same
way any other reader would. A hand-built manifest would be testing this module
against this file's idea of a session, and the whole claim of a replay image is
that it is the compiled form of what a recording actually produced.

The contract these tests pin is
``docs/development/native_recording_replay.md`` section 8, and its three modes
are tested as three different promises rather than as one code path with
options: ``exact_frames`` promises the original topology, ``recorded_projection``
promises exact block payloads under an explicit projection, and
``stream_frames`` promises a documented synthesis. Where the contract forbids
an answer -- a degraded mode, a widened range, a silently skipped stream -- the
test asserts the error, because "it raised something" and "it refused for the
right reason" are different results.
"""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import neurale.streaming as streaming
from neurale.io.nrf import NrfCorruptionError, NrfReader, NrfWriter
from neurale.recording import (
    ReplayConfig,
    ReplayConfigError,
    ReplayImageError,
    StreamReplayRange,
    build_replay_image,
    compile_recording_plan,
    finalize_spool,
    load_replay_image,
    open_replay_image,
    resolve_replay_faults,
)
from neurale.recording._native_recorder import NativeRecorderOptions, NativeSessionRecorder
from neurale.recording._registry import block_index_row, build_session
from neurale.recording._replay_config import (
    InjectedFault,
    MessageFaultTarget,
    MessageRange,
    StreamDiscontinuityFaultTarget,
    StreamFrameFaultTarget,
)

from .conftest import (
    CURSOR_CHANNELS,
    NEURAL_CHANNELS,
    native_schema,
    observation_ns,
    recorder_config,
    recording_runner,
    sample_values,
    stream_specs,
)
from .test_finalizer import MULTI_MAX_FRAME_PAYLOAD_BYTES, _realtime_config


def _create_symlink_or_skip(link: Path, target: Path) -> None:
    """Create a test symlink, or skip when the host does not permit one."""
    try:
        link.symlink_to(target)
    except NotImplementedError as exc:
        pytest.skip(f"symlink creation is unavailable on this host: {exc}")
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM} and getattr(exc, "winerror", None) != 1314:
            raise
        pytest.skip(f"symlink creation is not permitted on this host: {exc}")


NEURAL = 1
CURSOR = 2
BANDPOWER = 3


# --- real sessions -------------------------------------------------------------


def _record(
    tmp_path: Path,
    frames: int,
    *,
    streams: list[Any] | None = None,
    sequence_gap_at: int | None = None,
    block_idx: bool = True,
) -> bytes:
    """Run one real native session over the three-signal schema."""
    schema = native_schema()
    config = _realtime_config(frames, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES)
    specs = stream_specs(block_index=block_idx) if streams is None else streams
    native_config = recorder_config(tmp_path / "unused.nrf", streams=specs, checkpoint_interval=0)

    source = streaming.SyntheticNativeSource(schema, frames, 1, None, sequence_gap_at)
    runner = recording_runner(schema, config, source)
    recorder = NativeSessionRecorder.create(
        native_config,
        schema,
        options=NativeRecorderOptions(spool="memory", edge_capacity=max(8, frames * 2)),
    )
    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    runner.join()
    recorder.stop("done")
    snapshot = recorder._recorder.spool_snapshot()
    recorder.close()
    return snapshot


def _session(tmp_path: Path, frames: int = 8, name: str = "session.nrf", **keywords: Any) -> Path:
    """Record and finalize one session, returning the NRF directory."""
    path = tmp_path / name
    finalize_spool(_record(tmp_path, frames, **keywords), path)
    return path


def _partial_session(tmp_path: Path, frames: int = 8) -> Path:
    """A session whose plan records two of the schema's three signals."""
    specs = [spec for spec in stream_specs() if spec.stream_id != "bandpower"]
    return _session(tmp_path, frames, streams=specs)


def _legacy_session(tmp_path: Path, frames: int = 6) -> Path:
    """A pre-native-replay session: block indexes, no native ledgers."""
    path = tmp_path / "legacy.nrf"
    config = recorder_config(path, streams=stream_specs(timing="regular"))
    recording_plan = compile_recording_plan(config, native_schema())
    writer = NrfWriter.create(
        path,
        session_id=recording_plan.session.session_id,
        created_at=recording_plan.session.created_at,
        writer_name=recording_plan.session.writer_name,
        writer_version=recording_plan.session.writer_version,
    )
    plan = build_session(
        writer.registry,
        native_schema(),
        config.streams,
        control_chunk_length=recording_plan.resource_bounds.control_chunk_length,
    )
    writer.freeze()
    neural_plan = plan.plan_for(NEURAL)
    assert neural_plan is not None
    for sequence in range(1, frames + 1):
        start = (sequence - 1) * 8
        values = sample_values(NEURAL, start, 8)
        writer.append_stream("neural", values)
        writer.append_stream(
            "neural.blocks",
            np.asarray(
                [
                    block_index_row(
                        frame_sequence=sequence,
                        sample_idx_start=start,
                        last_sample_idx=start + 7,
                        n_samples=8,
                        row_offset=start,
                        device_tick_start=start,
                        host_received_ns=observation_ns(NEURAL, start),
                    )
                ],
                dtype="int64",
            ),
            timestamps=np.asarray([observation_ns(NEURAL, start)], dtype="int64"),
        )
    writer.finalize(kind="normal", reason="legacy fixture")
    writer.close()
    return path


@pytest.fixture
def session(tmp_path: Path) -> Path:
    return _session(tmp_path)


# --- 1. configuration validation, before any session is opened -----------------


def test_mode_is_explicit() -> None:
    with pytest.raises(ReplayConfigError, match="unknown replay mode"):
        ReplayConfig(mode="exact")


def test_ledger_mode_rejects_synthesized_range() -> None:
    """A message ordinal is not derivable from a per-stream sample position."""
    with pytest.raises(ReplayConfigError, match="Use message_range"):
        ReplayConfig(
            mode="recorded_projection",
            stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 1)},
        )


def test_synthesized_mode_rejects_ledger_range() -> None:
    with pytest.raises(ReplayConfigError, match="Use stream_ranges"):
        ReplayConfig(
            mode="stream_frames",
            message_range=MessageRange(0, 4),
            stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 1)},
        )


@pytest.mark.parametrize("unit", ["sample_index", "observation_index"])
def test_reserved_range_unit_is_rejected(unit: str) -> None:
    """Never silently reinterpreted as block ordinals under another spelling."""
    with pytest.raises(ReplayConfigError, match="reserved and not defined in v1"):
        StreamReplayRange(unit, 0, 4)


def test_stream_ranges_share_one_unit() -> None:
    """The request-level rule is independent of the per-range one.

    v1 admits exactly one unit, so a mixed request cannot be spelled with two
    legal ranges; the second unit is forced past the per-range check here on
    purpose. The day a second unit is defined, this is the rule that stops a
    request that mixes them, and it should already be under test.
    """
    forced = StreamReplayRange("block_ordinal", 0, 1)
    object.__setattr__(forced, "unit", "sample_index")
    with pytest.raises(ReplayConfigError, match="share a unit"):
        ReplayConfig(
            mode="stream_frames",
            stream_ranges={
                "neural": StreamReplayRange("block_ordinal", 0, 1),
                "cursor": forced,
            },
        )


def test_stream_range_requires_replay_range_type() -> None:
    with pytest.raises(ReplayConfigError, match="must be a StreamReplayRange"):
        ReplayConfig(
            mode="stream_frames",
            stream_ranges={"neural": (0, 1)},  # type: ignore[dict-item]
        )


def test_synthesized_run_requires_one_stream() -> None:
    """The empty set has no legal union schema; an empty range is how you say it."""
    with pytest.raises(ReplayConfigError, match="at least one stream"):
        ReplayConfig(mode="stream_frames")


def test_selected_streams_match_range_keys() -> None:
    with pytest.raises(ReplayConfigError, match="disagrees"):
        ReplayConfig(
            mode="stream_frames",
            selected_streams=("cursor",),
            stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 1)},
        )


@pytest.mark.parametrize("pacing", ["as_fast_as_possible", "step"])
def test_unsupported_speed_factor_is_rejected(pacing: str) -> None:
    with pytest.raises(ReplayConfigError, match="rather than ignored"):
        ReplayConfig(mode="exact_frames", pacing=pacing, speed_factor=2.0)


@pytest.mark.parametrize("factor", [0.0, -1.0, 1000.5, float("inf"), float("nan")])
def test_speed_factor_bounds_are_enforced(factor: float) -> None:
    with pytest.raises(ReplayConfigError, match="finite double"):
        ReplayConfig(mode="exact_frames", pacing="recorded", speed_factor=factor)


def test_recorded_pacing_defaults_speed_factor_to_one() -> None:
    assert ReplayConfig(mode="exact_frames", pacing="recorded").speed_factor == 1.0


def test_sequence_gap_on_discontinuity_fails_configuration() -> None:
    """Dropping a discontinuity produces no frame-sequence hole at all."""
    with pytest.raises(ReplayConfigError, match="targets a frame and only a frame"):
        InjectedFault(MessageFaultTarget("discontinuity", 3), "sequence_gap")


def test_only_stall_declares_bound() -> None:
    with pytest.raises(ReplayConfigError, match="not bounded"):
        InjectedFault(MessageFaultTarget("frame", 1), "stall")
    with pytest.raises(ReplayConfigError, match="does not wait"):
        InjectedFault(MessageFaultTarget("frame", 1), "abnormal_end", stall_ns=5)


def test_faults_reject_duplicate_position() -> None:
    with pytest.raises(ReplayConfigError, match="same position"):
        ReplayConfig(
            mode="exact_frames",
            faults=(
                InjectedFault(MessageFaultTarget("frame", 2), "stall", stall_ns=1),
                InjectedFault(MessageFaultTarget("frame", 2), "abnormal_end"),
            ),
        )


def test_fault_target_must_match_mode() -> None:
    with pytest.raises(ReplayConfigError, match="has no data-message ordinals"):
        ReplayConfig(
            mode="stream_frames",
            stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 1)},
            faults=(InjectedFault(MessageFaultTarget("frame", 1), "abnormal_end"),),
        )
    with pytest.raises(ReplayConfigError, match="positions a fault in 'stream_frames'"):
        ReplayConfig(
            mode="exact_frames",
            faults=(InjectedFault(StreamFrameFaultTarget("neural", 1), "abnormal_end"),),
        )


def test_replay_run_session_id_is_nonzero() -> None:
    with pytest.raises(ReplayConfigError, match="must be non-zero"):
        ReplayConfig(mode="exact_frames", replay_run_session_id=0)


def test_total_injected_delay_is_known_before_run() -> None:
    config = ReplayConfig(
        mode="exact_frames",
        faults=(
            InjectedFault(MessageFaultTarget("frame", 1), "stall", stall_ns=1_000),
            InjectedFault(MessageFaultTarget("frame", 2), "stall", stall_ns=2_000),
        ),
    )
    assert config.total_injected_delay_ns == 3_000


# --- 2. mode legality against a session ----------------------------------------


def test_partial_plan_rejects_exact_frames(tmp_path: Path) -> None:
    """Never silently answered with a projection, and the error names the mode."""
    session = _partial_session(tmp_path)
    with pytest.raises(ReplayConfigError, match="Replay it as recorded_projection"):
        build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image")


def test_stream_subset_is_projection_under_complete_plan(session: Path, tmp_path: Path) -> None:
    """A full-coverage session does not stay exact merely because its plan was."""
    with pytest.raises(ReplayConfigError, match="recorded_projection"):
        build_replay_image(
            session,
            ReplayConfig(mode="exact_frames", selected_streams=("neural",)),
            path=tmp_path / "image",
        )


def test_exact_frames_accepts_explicit_recorded_set(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(mode="exact_frames", selected_streams=("bandpower", "cursor", "neural"))
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        assert image.metadata.mode == "exact_frames"


def test_ledger_mode_rejects_session_without_ledgers(tmp_path: Path) -> None:
    """A missing ledger is never answered with a synthesis."""
    session = _legacy_session(tmp_path)
    with pytest.raises(ReplayConfigError, match="stream_frames"):
        build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image")


def test_session_without_ledgers_replays_stream_frames(tmp_path: Path) -> None:
    session = _legacy_session(tmp_path)
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 6)},
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        assert image.metadata.n_frames == 6
        assert image.metadata.plan_fingerprint is None
        assert image.metadata.completeness in (None, "unverified_legacy")


def test_stream_without_rate_rejects_synthesis(tmp_path: Path) -> None:
    """A known v1 limit, refused rather than guessed.

    An explicit-timing NRF stream stores per-observation timestamps and no
    rate; a native ``SignalSchema`` must declare one. Where the session also
    carries no recording plan there is no recorded rate anywhere, and v1 does
    not derive one from the stored timestamps.
    """
    path = tmp_path / "explicit.nrf"
    config = recorder_config(path)
    recording_plan = compile_recording_plan(config, native_schema())
    writer = NrfWriter.create(
        path,
        session_id=recording_plan.session.session_id,
        created_at=recording_plan.session.created_at,
        writer_name=recording_plan.session.writer_name,
        writer_version=recording_plan.session.writer_version,
    )
    build_session(
        writer.registry,
        native_schema(),
        config.streams,
        control_chunk_length=recording_plan.resource_bounds.control_chunk_length,
    )
    writer.freeze()
    writer.append_stream(
        "neural",
        sample_values(NEURAL, 0, 8),
        timestamps=np.asarray([observation_ns(NEURAL, i) for i in range(8)], dtype="int64"),
    )
    writer.append_stream(
        "neural.blocks",
        np.asarray(
            [
                block_index_row(
                    frame_sequence=1,
                    sample_idx_start=0,
                    last_sample_idx=7,
                    n_samples=8,
                    row_offset=0,
                    device_tick_start=0,
                    host_received_ns=observation_ns(NEURAL, 0),
                )
            ],
            dtype="int64",
        ),
        timestamps=np.asarray([observation_ns(NEURAL, 0)], dtype="int64"),
    )
    writer.finalize(kind="normal", reason="legacy fixture")
    writer.close()

    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 1)},
    )
    with pytest.raises(ReplayConfigError, match="does not derive a rate"):
        build_replay_image(path, config, path=tmp_path / "image")


def test_stream_without_block_index_rejects_stream_frames(tmp_path: Path) -> None:
    session = _session(tmp_path, 6, block_idx=False)
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 1)},
    )
    with pytest.raises(ReplayConfigError, match="declares no committed block index"):
        build_replay_image(session, config, path=tmp_path / "image")


def test_unrecorded_stream_is_reported(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(mode="recorded_projection", selected_streams=("absent",))
    with pytest.raises(ReplayConfigError, match="does not record the selected stream"):
        build_replay_image(session, config, path=tmp_path / "image")


# --- 3. range validation --------------------------------------------------------


def test_message_range_boundary_requires_committed_message(session: Path, tmp_path: Path) -> None:
    with pytest.raises(ReplayConfigError, match="beyond the committed extent"):
        build_replay_image(
            session,
            ReplayConfig(mode="exact_frames", message_range=MessageRange(0, 99)),
            path=tmp_path / "image",
        )


def test_empty_message_range_yields_nothing(session: Path, tmp_path: Path) -> None:
    """Legal, and both lists are empty: nothing was requested, so nothing was dropped."""
    config = ReplayConfig(mode="recorded_projection", message_range=MessageRange(3, 3))
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        assert image.metadata.n_items == 0
        assert image.metadata.n_omissions == 0
        assert image.metadata.payload_byte_count == 0


def test_stream_range_past_committed_extent_is_rejected(session: Path, tmp_path: Path) -> None:
    """Different facts: a range beyond the data is not an empty result."""
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 99)},
    )
    with pytest.raises(ReplayConfigError, match="beyond its committed extent"):
        build_replay_image(session, config, path=tmp_path / "image")


def test_empty_replay_uses_one_empty_range(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 5, 5)},
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        assert image.metadata.n_items == 0
        assert image.metadata.selected_streams == ("neural",)


# --- 4. exact frame replay ------------------------------------------------------


def test_exact_replay_emits_recorded_frames_and_blocks(session: Path, tmp_path: Path) -> None:
    with build_replay_image(
        session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image"
    ) as image:
        frames = [item for item in image.items() if item.kind == "frame"]
        assert len(frames) == 8
        for item in frames:
            assert item.replay_frame_sequence == item.original_frame_sequence
            assert len(image.blocks_of(item)) == 3
        assert image.omissions() == ()
        assert image.metadata.n_omissions == 0


def test_exact_frame_payload_is_byte_identical(session: Path, tmp_path: Path) -> None:
    """The blocks tile the recorded payload at their recorded offsets.

    Section 1.1 keeps ``payload_offset`` and ``total_payload_byte_count`` so
    coverage is verified against the original recording rather than against the
    reconstruction; this asserts the image reproduces that original layout.
    """
    with build_replay_image(
        session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image"
    ) as image:
        item = next(item for item in image.items() if item.kind == "frame")
        payload = image.payload_of(item)
        assert len(payload) == item.original_payload_byte_count
        for block in image.blocks_of(item):
            assert block.payload_offset == block.original_payload_offset
            assert block.clock_sync is not None
        covered = sum(block.payload_byte_count for block in image.blocks_of(item))
        assert covered == item.original_payload_byte_count


def test_replayed_block_carries_recorded_samples(session: Path, tmp_path: Path) -> None:
    with (
        build_replay_image(
            session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image"
        ) as image,
        NrfReader.open(session) as reader,
    ):
        item = next(item for item in image.items() if item.kind == "frame")
        payload = image.payload_of(item)
        block = next(block for block in image.blocks_of(item) if block.stream_id == "neural")
        raw = payload[block.payload_offset : block.payload_offset + block.payload_byte_count]
        values = np.frombuffer(raw, dtype="<i2").reshape(-1, NEURAL_CHANNELS)
        expected = reader.read_stream("neural", 0, block.n_samples)
        assert np.array_equal(values, expected)


def test_channel_major_signal_keeps_native_layout(session: Path, tmp_path: Path) -> None:
    """NRF stores sample-major; the runtime is handed what the schema declared."""
    with (
        build_replay_image(
            session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image"
        ) as image,
        NrfReader.open(session) as reader,
    ):
        item = next(item for item in image.items() if item.kind == "frame")
        payload = image.payload_of(item)
        block = next(block for block in image.blocks_of(item) if block.stream_id == "cursor")
        raw = payload[block.payload_offset : block.payload_offset + block.payload_byte_count]
        values = np.frombuffer(raw, dtype="<f4").reshape(CURSOR_CHANNELS, -1)
        expected = reader.read_stream("cursor", 0, block.n_samples)
        assert np.array_equal(values.T, expected)


def test_exact_replay_emits_discontinuity_references(
    tmp_path: Path,
) -> None:
    """It renumbers nothing, so a boundary reference is the recorded truth."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    with (
        build_replay_image(
            session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image"
        ) as image,
        NrfReader.open(session) as reader,
    ):
        rows = reader.read_records("native-discontinuities-v1")
        emitted = [item for item in image.items() if item.kind == "discontinuity"]
        assert len(emitted) == len(rows["data_message_ordinal"])
        item = emitted[0]
        assert item.replay_frame_sequence == rows["actual_frame_sequence"][0]
        assert item.previous_replay_sequence == rows["previous_frame_sequence"][0]
        assert item.timeline_ns == rows["runtime_accepted_host_time_ns"][0]


# --- 5. recorded projection ------------------------------------------------------


def test_full_coverage_projection_omits_nothing(session: Path, tmp_path: Path) -> None:
    """What full coverage promises, and the only thing it promises."""
    with build_replay_image(
        session, ReplayConfig(mode="recorded_projection"), path=tmp_path / "image"
    ) as image:
        assert image.metadata.n_frames == 8
        assert image.metadata.n_blocks == 24
        assert [entry for entry in image.omissions() if entry.reason == "no_projected_blocks"] == []
        assert image.metadata.mode == "recorded_projection"


def test_projection_renumbers_frames_from_zero(session: Path, tmp_path: Path) -> None:
    """The ordinals stay original; the frame sequence is the run's own.

    Read over a sub-range, because a whole-session projection of a
    full-coverage session renumbers 0..n-1 onto itself and would prove nothing.
    """
    config = ReplayConfig(mode="recorded_projection", message_range=MessageRange(3, 7))
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        frames = [item for item in image.items() if item.kind == "frame"]
        assert [item.replay_frame_sequence for item in frames] == list(range(len(frames)))
        assert [item.original_frame_sequence for item in frames] == [3, 4, 5, 6]
        assert [item.data_message_ordinal for item in frames] == [3, 4, 5, 6]


def test_empty_projected_frame_is_dropped_and_reported(tmp_path: Path) -> None:
    """The one field that can hold a frame and a discontinuity, tagged."""
    session = _partial_session(tmp_path)
    config = ReplayConfig(mode="recorded_projection", selected_streams=("cursor",))
    with (
        build_replay_image(session, config, path=tmp_path / "image") as image,
        NrfReader.open(session) as reader,
    ):
        rows = reader.read_records("native-signal-blocks-v1")
        cursor_frames = {
            ordinal
            for ordinal, stream in zip(rows["data_message_ordinal"], rows["stream_id"], strict=True)
            if stream == "cursor"
        }
        every = set(reader.read_records("native-frames-v1")["data_message_ordinal"])
        dropped = every - cursor_frames
        omitted = {
            entry.data_message_ordinal for entry in image.omissions() if entry.kind == "frame"
        }
        assert omitted == dropped
        for entry in image.omissions():
            if entry.kind == "frame":
                assert entry.reason == "no_projected_blocks"
                assert entry.original_frame_sequence is not None


def test_projection_keeps_ordinal_holes(tmp_path: Path) -> None:
    """Ordinals are not renumbered to close the gaps a projection left."""
    session = _partial_session(tmp_path)
    config = ReplayConfig(mode="recorded_projection", selected_streams=("neural",))
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        emitted = [item.data_message_ordinal for item in image.items() if item.kind == "frame"]
        assert emitted == sorted(emitted)
        assert all(ordinal is not None for ordinal in emitted)


def test_range_starting_at_discontinuity_omits_it(tmp_path: Path) -> None:
    """No frame has been emitted yet, so previous_frame_sequence names nothing."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    with NrfReader.open(session) as reader:
        rows = reader.read_records("native-discontinuities-v1")
        ordinal = int(rows["data_message_ordinal"][0])
        end = max(
            int(value) for value in reader.read_records("native-frames-v1")["data_message_ordinal"]
        )
    config = ReplayConfig(mode="recorded_projection", message_range=MessageRange(ordinal, end + 1))
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        omitted = [entry for entry in image.omissions() if entry.kind == "discontinuity"]
        assert [entry.reason for entry in omitted] == ["no_preceding_emitted_frame"]
        assert omitted[0].data_message_ordinal == ordinal


def test_discontinuity_without_emitted_frame_is_omitted(
    tmp_path: Path,
) -> None:
    """A dangling discontinuity is never emitted, and no boundary frame is invented."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    with NrfReader.open(session) as reader:
        rows = reader.read_records("native-discontinuities-v1")
        ordinal = int(rows["data_message_ordinal"][0])
    config = ReplayConfig(mode="recorded_projection", message_range=MessageRange(0, ordinal + 1))
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        omitted = [entry for entry in image.omissions() if entry.kind == "discontinuity"]
        assert [entry.reason for entry in omitted] == ["target_frame_not_emitted"]


def test_omission_list_is_scoped_to_request(tmp_path: Path) -> None:
    """A message outside the range was never requested, so it is not an omission."""
    session = _partial_session(tmp_path, 12)
    config = ReplayConfig(
        mode="recorded_projection",
        selected_streams=("cursor",),
        message_range=MessageRange(0, 4),
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        assert all(
            entry.data_message_ordinal is not None and entry.data_message_ordinal < 4
            for entry in image.omissions()
        )


def test_projection_reports_covered_streams(tmp_path: Path) -> None:
    """So that 'absent' is distinguishable from 'lost'."""
    session = _partial_session(tmp_path)
    with build_replay_image(
        session, ReplayConfig(mode="recorded_projection"), path=tmp_path / "image"
    ) as image:
        assert image.metadata.plan_coverage == "partial"
        assert image.metadata.planned_signal_ids == (NEURAL, CURSOR, BANDPOWER)
        assert image.metadata.recorded_signal_ids == (NEURAL, CURSOR)
        assert image.metadata.selected_streams == ("cursor", "neural")


def test_full_coverage_projection_matches_exact_replay(session: Path, tmp_path: Path) -> None:
    """Parity holds for frames, order, payloads and provenance -- not for headers.

    Over a sub-range, because a whole-session run of a full-coverage session
    numbers both modes 0..n-1 and would hide the one difference being asserted.
    """
    window = MessageRange(3, 7)
    with (
        build_replay_image(
            session,
            ReplayConfig(mode="exact_frames", message_range=window),
            path=tmp_path / "exact",
        ) as exact,
        build_replay_image(
            session,
            ReplayConfig(mode="recorded_projection", message_range=window),
            path=tmp_path / "projected",
        ) as projected,
    ):
        exact_frames = [item for item in exact.items() if item.kind == "frame"]
        projected_frames = [item for item in projected.items() if item.kind == "frame"]
        assert len(exact_frames) == len(projected_frames)
        for left, right in zip(exact_frames, projected_frames, strict=True):
            assert left.data_message_ordinal == right.data_message_ordinal
            assert left.original_frame_sequence == right.original_frame_sequence
            assert exact.payload_of(left) == projected.payload_of(right)
        assert [item.replay_frame_sequence for item in projected_frames] != [
            item.replay_frame_sequence for item in exact_frames
        ]


# --- 6. synthesized stream replay ------------------------------------------------


def _synthesized(session: Path, path: Path, **windows: tuple[int, int]) -> Any:
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={
            stream_id: StreamReplayRange("block_ordinal", start, stop)
            for stream_id, (start, stop) in windows.items()
        },
    )
    return build_replay_image(session, config, path=path)


def test_committed_block_becomes_one_synthesized_frame(session: Path, tmp_path: Path) -> None:
    with _synthesized(session, tmp_path / "image", neural=(0, 8), cursor=(0, 8)) as image:
        assert image.metadata.n_frames == 16
        assert image.metadata.n_blocks == 16
        assert all(
            len(image.blocks_of(item)) == 1 for item in image.items() if item.kind == "frame"
        )
        assert image.metadata.frame_construction == "one_frame_per_block"


def test_synthesized_frames_share_zero_based_sequence(session: Path, tmp_path: Path) -> None:
    """Per-stream numbering would be read as a duplicate or a regression."""
    with _synthesized(session, tmp_path / "image", neural=(0, 8), cursor=(0, 8)) as image:
        frames = [item for item in image.items() if item.kind == "frame"]
        assert [item.replay_frame_sequence for item in frames] == list(range(16))


def test_merge_orders_by_host_time_stream_block(session: Path, tmp_path: Path) -> None:
    """A deterministic k-way merge over per-stream cursors, not a global sort."""
    with _synthesized(session, tmp_path / "image", neural=(0, 8), cursor=(0, 8)) as image:
        keys = []
        for item in image.items():
            if item.kind != "frame":
                continue
            block = image.blocks_of(item)[0]
            keys.append(
                (item.original_host_received_ns, block.stream_id, block.source_block_ordinal)
            )
        assert keys == sorted(keys, key=lambda key: (key[0], key[1].encode("utf-8"), key[2]))
        for stream_id in ("neural", "cursor"):
            ordinals = [key[2] for key in keys if key[1] == stream_id]
            assert ordinals == sorted(ordinals)


def test_synthesized_id_registry_derives_from_string_ids(session: Path, tmp_path: Path) -> None:
    """Byte order and numbering from 1, so the mapping never depends on insertion."""
    with _synthesized(
        session, tmp_path / "image", neural=(0, 8), cursor=(0, 8), bandpower=(0, 8)
    ) as image:
        schema = image.schema_document()
        assert schema["schema_id"] == 1
        by_stream = {entry.stream_id: entry.native_signal_id for entry in image.fidelity()}
        assert by_stream == {"bandpower": 1, "cursor": 2, "neural": 3}
        assert [signal["id"] for signal in schema["signals"]] == [1, 2, 3]
        assert [unit["id"] for unit in schema["units"]] == [1]
        feature = next(signal for signal in schema["signals"] if signal["feature_set_id"])
        descriptor = schema["feature_sets"][0]
        assert feature["feature_set_id"] == descriptor["id"]
        # One unit id per feature column, all naming the one distinct unit
        # symbol this session declares, which is numbered 1.
        assert descriptor["unit_ids"] == [1, 1, 1]


def test_synthesized_signal_declares_no_channel_metadata(session: Path, tmp_path: Path) -> None:
    with _synthesized(session, tmp_path / "image", neural=(0, 8)) as image:
        signal = image.schema_document()["signals"][0]
        assert (signal["channel_set_id"], signal["calibration_id"], signal["reference_id"]) == (
            0,
            0,
            0,
        )
        assert image.metadata.descriptor_metadata_available is False


def test_synthesized_block_reports_clock_sync_unavailable(session: Path, tmp_path: Path) -> None:
    """A block index of this shape carries none, and none is invented."""
    with _synthesized(session, tmp_path / "image", neural=(0, 8)) as image:
        item = next(item for item in image.items() if item.kind == "frame")
        assert image.blocks_of(item)[0].clock_sync is None
        assert image.metadata.clock_sync_available is False
        assert all(entry.clock_sync_available is False for entry in image.fidelity())


def test_fidelity_report_names_indexed_columns(session: Path, tmp_path: Path) -> None:
    with _synthesized(session, tmp_path / "image", neural=(0, 8)) as image:
        entry = image.fidelity()[0]
        assert set(entry.block_index_columns_present) >= {
            "frame_sequence",
            "sample_idx_start",
            "n_samples",
            "row_offset",
            "host_received_ns",
        }
        assert entry.frames_emitted == 8
        assert entry.blocks_emitted == 8
        assert image.metadata.ordering_key == "original_host_received_ns"


def test_synthesized_discontinuity_is_frame_barrier(
    tmp_path: Path,
) -> None:
    """One record per stream, never merged, and referencing the run-wide numbering."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    with _synthesized(session, tmp_path / "image", neural=(0, 12), cursor=(0, 12)) as image:
        emitted = [item for item in image.items() if item.kind == "discontinuity"]
        assert emitted
        for item in emitted:
            assert item.previous_replay_sequence is not None
            assert item.replay_frame_sequence == item.previous_replay_sequence + 1
            assert item.data_message_ordinal is None
            assert len(image.gaps_of(item)) == 1


def test_synthesized_run_reports_own_omissions(tmp_path: Path) -> None:
    """Keyed by stream and record position, because this mode has no ordinals."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    with NrfReader.open(session) as reader:
        rows = reader.read_records("discontinuities.neural")
        target = int(rows["actual_frame_sequence"][0])
        blocks = reader.read_stream("neural.blocks")
        block_ordinal = int(np.flatnonzero(blocks[:, 0] == target)[0])
    with _synthesized(
        session, tmp_path / "image", neural=(block_ordinal, block_ordinal + 2)
    ) as image:
        omitted = image.omissions()
        assert [entry.reason for entry in omitted] == ["no_preceding_emitted_frame"]
        assert omitted[0].stream_id == "neural"
        assert omitted[0].source_ordinal == 0
        assert omitted[0].data_message_ordinal is None


# --- 7. the image container, determinism, and the cache --------------------------


def test_repeated_build_is_byte_identical(session: Path, tmp_path: Path) -> None:
    """Deterministic bytes are what make a sub-range result independently reproducible."""
    config = ReplayConfig(mode="recorded_projection")
    build_replay_image(session, config, path=tmp_path / "first").close()
    build_replay_image(session, config, path=tmp_path / "second").close()
    assert (tmp_path / "first").read_bytes() == (tmp_path / "second").read_bytes()


def test_pacing_and_faults_do_not_change_image(session: Path, tmp_path: Path) -> None:
    """The image is a function of the source, the mode, the selection and the range."""
    plain = ReplayConfig(mode="recorded_projection")
    paced = ReplayConfig(
        mode="recorded_projection",
        pacing="recorded",
        speed_factor=4.0,
        replay_run_session_id=99,
        faults=(InjectedFault(MessageFaultTarget("frame", 2), "stall", stall_ns=10),),
    )
    build_replay_image(session, plain, path=tmp_path / "plain").close()
    build_replay_image(session, paced, path=tmp_path / "paced").close()
    assert (tmp_path / "plain").read_bytes() == (tmp_path / "paced").read_bytes()


def test_different_range_gets_own_numbering(session: Path, tmp_path: Path) -> None:
    """Per-request local numbering and a range-invariant sequence cannot both hold."""
    whole = ReplayConfig(mode="recorded_projection")
    tail = ReplayConfig(mode="recorded_projection", message_range=MessageRange(2, 5))
    with (
        build_replay_image(session, whole, path=tmp_path / "whole") as first,
        build_replay_image(session, tail, path=tmp_path / "tail") as second,
    ):
        assert first.fingerprint != second.fingerprint
        third = next(item for item in first.items() if item.data_message_ordinal == 2)
        head = next(item for item in second.items() if item.data_message_ordinal == 2)
        assert third.replay_frame_sequence == 2
        assert head.replay_frame_sequence == 0
        assert third.original_frame_sequence == head.original_frame_sequence


def test_cached_image_is_reused_foreign_is_rebuilt(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(mode="recorded_projection")
    first = load_replay_image(session, config, path=tmp_path / "image")
    assert first.reused is False
    assert first.rebuild_reason == "no image is cached at this path"
    first.image.close()

    second = load_replay_image(session, config, path=tmp_path / "image")
    assert second.reused is True
    second.image.close()

    other = ReplayConfig(mode="recorded_projection", selected_streams=("neural",))
    third = load_replay_image(session, other, path=tmp_path / "image")
    assert third.reused is False
    assert "another source, mode, selection, or range" in (third.rebuild_reason or "")
    third.image.close()


def test_corrupt_cached_image_is_rebuilt(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(mode="recorded_projection")
    build_replay_image(session, config, path=tmp_path / "image").close()
    data = bytearray((tmp_path / "image").read_bytes())
    data[-1] ^= 0xFF
    (tmp_path / "image").write_bytes(bytes(data))

    with pytest.raises(ReplayImageError, match="does not match its checksum"):
        open_replay_image(tmp_path / "image")

    rebuilt = load_replay_image(session, config, path=tmp_path / "image")
    assert rebuilt.reused is False
    assert "checksum" in (rebuilt.rebuild_reason or "")
    rebuilt.image.close()


def test_truncated_image_is_rejected(session: Path, tmp_path: Path) -> None:
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image").close()
    data = (tmp_path / "image").read_bytes()
    (tmp_path / "image").write_bytes(data[: len(data) // 2])
    with pytest.raises(ReplayImageError):
        open_replay_image(tmp_path / "image")


def test_foreign_format_version_is_rejected(session: Path, tmp_path: Path) -> None:
    from neurale.recording import _replay_image as container

    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image").close()
    data = bytearray((tmp_path / "image").read_bytes())
    data[8:10] = (container.FORMAT_VERSION + 1).to_bytes(2, "little")
    container.struct.pack_into(
        "<I",
        data,
        container.HEADER_BYTES - 4,
        container.crc32c(bytes(data[: container.HEADER_BYTES - 4])),
    )
    (tmp_path / "image").write_bytes(bytes(data))
    with pytest.raises(ReplayImageError, match="format version"):
        open_replay_image(tmp_path / "image")


def test_publication_leaves_no_temporary(session: Path, tmp_path: Path) -> None:
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image").close()
    assert sorted(entry.name for entry in tmp_path.iterdir() if entry.name.startswith("image")) == [
        "image"
    ]


def test_changed_source_invalidates_cache(session: Path, tmp_path: Path) -> None:
    """The fingerprint covers the committed state, not just the path."""
    config = ReplayConfig(mode="recorded_projection")
    build_replay_image(session, config, path=tmp_path / "image").close()
    other = _session(tmp_path, 6, name="other.nrf")
    rebuilt = load_replay_image(other, config, path=tmp_path / "image")
    assert rebuilt.reused is False
    rebuilt.image.close()


def test_builder_never_reads_whole_stream(
    session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded memory: the payload is streamed block by block into the image.

    Asserted on the reads rather than on process memory, because the property
    that matters is that no read is proportional to the session.
    """
    reads: list[int] = []
    original = NrfReader._read

    def record(self: NrfReader, spec: Any, start: int, stop: int) -> Any:
        if spec.path.startswith("streams/") and spec.path.endswith("/data"):
            reads.append(stop - start)
        return original(self, spec, start, stop)

    monkeypatch.setattr(NrfReader, "_read", record)
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image").close()
    assert reads
    assert max(reads) <= 8


# --- 8. incomplete sessions -------------------------------------------------------


def test_incomplete_session_is_rejected_by_default(session: Path, tmp_path: Path) -> None:
    _mark_incomplete(session)
    with pytest.raises(ReplayConfigError, match="allow_incomplete"):
        build_replay_image(
            session,
            ReplayConfig(mode="recorded_projection"),
            path=tmp_path / "image",
            verify_source=False,
        )


def test_allow_incomplete_replays_prefix_with_abnormal_end(session: Path, tmp_path: Path) -> None:
    _mark_incomplete(session)
    config = ReplayConfig(mode="recorded_projection", allow_incomplete=True)
    with build_replay_image(session, config, path=tmp_path / "image", verify_source=False) as image:
        assert image.metadata.allow_incomplete is True
        assert image.metadata.abnormal_end_required is True
        assert image.metadata.completeness == "verified_incomplete"
        assert image.metadata.n_frames == 8


def _mark_incomplete(session: Path) -> None:
    """Make the persisted accounting say a data item was lost.

    The summary is the artifact's own statement about its capture; editing it
    is how a truncated recording is simulated without corrupting the data the
    replay then reads.
    """
    column = "records/session_accounting/session-accounting-v1/columns/lost_during_finalization"
    chunk = f"{column}/c/0"
    # The same eight bytes, so the chunk's committed byte length still holds
    # and the only thing that changed is what the accounting says. The tests
    # that use this build with ``verify_source=False``, because re-hashing the
    # committed objects would -- correctly -- refuse the edited chunk.
    _rewrite_object(session, chunk, (1).to_bytes(8, "little"))


def _read_object(session: Path, key: str) -> bytes:
    from zipfile import ZipFile

    with ZipFile(session) as source:
        return source.read(key)


def _rewrite_object(session: Path, key: str, replacement: bytes) -> None:
    from zipfile import ZIP_DEFLATED, ZipFile

    modified = session.with_suffix(".modified")
    with ZipFile(session) as source, ZipFile(modified, "w", compression=ZIP_DEFLATED) as target:
        target.comment = source.comment
        assert key in source.namelist()
        for entry in source.infolist():
            data = replacement if entry.filename == key else source.read(entry)
            target.writestr(entry.filename, data)
    modified.replace(session)


# --- 9. fault positioning ----------------------------------------------------------


def test_fault_outside_range_is_reported_unfired(session: Path, tmp_path: Path) -> None:
    """Never silently dropped: the run states that the position was not reached."""
    config = ReplayConfig(
        mode="recorded_projection",
        message_range=MessageRange(0, 3),
        faults=(InjectedFault(MessageFaultTarget("frame", 6), "abnormal_end"),),
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        resolved = resolve_replay_faults(image, config)
        assert [fault.fired for fault in resolved] == [False]
        assert resolved[0].item_index is None


def test_fault_with_wrong_item_kind_is_rejected(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(
        mode="recorded_projection",
        faults=(InjectedFault(MessageFaultTarget("discontinuity", 0), "abnormal_end"),),
    )
    with pytest.raises(ReplayConfigError, match="names a frame, not a discontinuity"):
        build_replay_image(session, config, path=tmp_path / "image")


@pytest.mark.parametrize("pos", ["first", "last"])
def test_sequence_gap_needs_frames_on_both_sides(session: Path, tmp_path: Path, pos: str) -> None:
    """Without one there is no observable jump, so the effect would not happen."""
    with build_replay_image(
        session, ReplayConfig(mode="recorded_projection"), path=tmp_path / "probe"
    ) as image:
        frames = [item for item in image.items() if item.kind == "frame"]
        item = frames[0] if pos == "first" else frames[-1]
        ordinal = item.data_message_ordinal
    assert ordinal is not None
    config = ReplayConfig(
        mode="recorded_projection",
        faults=(InjectedFault(MessageFaultTarget("frame", ordinal), "sequence_gap"),),
    )
    with pytest.raises(ReplayConfigError, match=f"{pos} emitted frame"):
        build_replay_image(session, config, path=tmp_path / "image")


def test_interior_sequence_gap_is_positioned_on_item(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(
        mode="recorded_projection",
        faults=(InjectedFault(MessageFaultTarget("frame", 3), "sequence_gap"),),
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        resolved = resolve_replay_faults(image, config)
        assert resolved[0].fired is True
        assert image.item(resolved[0].item_index or 0).data_message_ordinal == 3


def test_synthesized_fault_is_positioned_by_stream_and_block(session: Path, tmp_path: Path) -> None:
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 8)},
        faults=(
            InjectedFault(StreamFrameFaultTarget("neural", 3), "read_failure"),
            InjectedFault(StreamDiscontinuityFaultTarget("neural", 0), "abnormal_end"),
        ),
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        resolved = resolve_replay_faults(image, config)
        assert resolved[0].fired is True
        item = image.item(resolved[0].item_index or 0)
        assert image.blocks_of(item)[0].source_block_ordinal == 3
        # This session has no discontinuity, so the second fault never fires.
        assert resolved[1].fired is False


def test_synthesized_discontinuity_fault_uses_record_ordinal(
    tmp_path: Path,
) -> None:
    """One block can have several discontinuities ahead of it, so the record wins."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 12)},
        faults=(InjectedFault(StreamDiscontinuityFaultTarget("neural", 0), "stall", stall_ns=7),),
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        resolved = resolve_replay_faults(image, config)
        assert resolved[0].fired is True
        assert image.item(resolved[0].item_index or 0).kind == "discontinuity"
        assert resolved[0].stall_ns == 7


def test_emitted_discontinuity_drops_unprojected_gaps(
    tmp_path: Path,
) -> None:
    """The gaps it does not carry stay in the ledger, where the message is whole."""
    session = _session(tmp_path, 12, sequence_gap_at=4)
    with (
        build_replay_image(
            session,
            ReplayConfig(mode="recorded_projection", selected_streams=("neural",)),
            path=tmp_path / "image",
        ) as image,
        NrfReader.open(session) as reader,
    ):
        recorded = reader.read_records("native-discontinuities-v1")["signal_gap_count"][0]
        assert recorded == 3
        item = next(item for item in image.items() if item.kind == "discontinuity")
        gaps = image.gaps_of(item)
        assert [gap.native_signal_id for gap in gaps] == [NEURAL]


def test_irrelevant_discontinuity_reports_no_projected_signal(
    tmp_path: Path,
) -> None:
    """The reason is chosen over every condition that holds, by the frozen precedence.

    The synthetic source emits one gap per signal, so no stream selection can
    make a recorded discontinuity irrelevant on its own. One committed column
    is rewritten instead -- the gaps are re-pointed at a signal the run does
    not project -- which leaves a real session whose one changed fact is the
    one under test, and keeps the other two conditions false: a frame is
    emitted before the message, and the frame it precedes is emitted too.
    """
    session = _session(tmp_path, 12, sequence_gap_at=4)
    chunk = "records/native_signal_gaps/native-signal-gaps-v1/columns/native_signal_id/c/0"
    with NrfReader.open(session) as reader:
        committed = reader.record_extent("native-signal-gaps-v1")
    values = np.frombuffer(_read_object(session, chunk), dtype="<u4").copy()
    assert set(values[:committed].tolist()) == {NEURAL, CURSOR, BANDPOWER}
    values[:committed] = BANDPOWER
    _rewrite_object(session, chunk, values.tobytes())

    config = ReplayConfig(mode="recorded_projection", selected_streams=("neural",))
    with build_replay_image(session, config, path=tmp_path / "image", verify_source=False) as image:
        omitted = [entry for entry in image.omissions() if entry.kind == "discontinuity"]
        assert [entry.reason for entry in omitted] == ["no_projected_signals"]
        assert image.metadata.n_discontinuities == 0


# --- cache equivalence, source isolation, synthesis correctness ---


def _corrupt_a_source_chunk(session: Path) -> None:
    """Flip one byte of a committed data chunk, leaving journal and extents intact.

    This is the reviewer's case for section 8.5: a source fingerprint covers
    identity, journal position, and extents, not the committed object content,
    so a bit-flipped chunk keeps the same fingerprint and only a re-hash catches
    it.
    """
    chunk = "streams/neural/data/c/0/0"
    data = bytearray(_read_object(session, chunk))
    data[0] ^= 0xFF
    _rewrite_object(session, chunk, bytes(data))


def test_image_target_resolves_outside_source(session: Path, tmp_path: Path) -> None:
    """Building a replay image is read-only over the session (section 8).

    A target inside the source session would let a build create or overwrite
    files in it, so it is refused before any scratch file is made -- including a
    relative alias that resolves back inside.
    """
    config = ReplayConfig(mode="exact_frames")
    with pytest.raises(ReplayConfigError, match="resolves inside the source session"):
        build_replay_image(session, config, path=session / "image")
    with pytest.raises(ReplayConfigError, match="resolves inside the source session"):
        load_replay_image(session, config, path=session / "image")
    alias = session.parent / session.name / "image"
    with pytest.raises(ReplayConfigError, match="resolves inside the source session"):
        build_replay_image(session, config, path=alias)
    assert not (session / "image").exists()


def test_allow_incomplete_keeps_complete_session_image(session: Path, tmp_path: Path) -> None:
    """A request bit that changes nothing on a complete source changes no byte.

    ``allow_incomplete`` only decides whether an incomplete session is built at
    all; on a complete session it changes neither the emitted data nor the
    terminal result, so the image records the source fact (it is complete) and
    both requests produce one image (section 8.5).
    """
    plain = ReplayConfig(mode="recorded_projection")
    allowed = ReplayConfig(mode="recorded_projection", allow_incomplete=True)
    build_replay_image(session, plain, path=tmp_path / "plain", verify_source=False).close()
    build_replay_image(session, allowed, path=tmp_path / "allowed", verify_source=False).close()
    assert (tmp_path / "plain").read_bytes() == (tmp_path / "allowed").read_bytes()
    with build_replay_image(
        session, allowed, path=tmp_path / "check", verify_source=False
    ) as image:
        assert image.metadata.allow_incomplete is False
        assert image.metadata.abnormal_end_required is False


def test_cache_hit_revalidates_faults(session: Path, tmp_path: Path) -> None:
    """A cache hit is held to the same fault bar as a fresh build.

    Faults are not part of the image fingerprint, so a request with illegal
    faults hashes the same as the cached one. The cache hit must re-position the
    faults and reject them, rather than serving an image a fresh build would
    have refused (section 8.11).
    """
    config = ReplayConfig(mode="recorded_projection")
    load_replay_image(session, config, path=tmp_path / "image").image.close()
    bad = ReplayConfig(
        mode="recorded_projection",
        faults=(InjectedFault(MessageFaultTarget("frame", 0), "sequence_gap"),),
    )
    with pytest.raises(ReplayConfigError, match="emitted frame on both sides"):
        load_replay_image(session, bad, path=tmp_path / "image")


def test_cache_hit_rejects_corrupt_source_under_verify(session: Path, tmp_path: Path) -> None:
    """Section 8.5 makes rejecting a corrupt session the default, even on a hit.

    The fingerprint does not re-hash the committed object content, so a
    bit-flipped chunk keeps the same fingerprint; ``verify_source`` re-hashes the
    session on the hit and rejects it, while a hit without verification still
    serves the intact cached image.
    """
    config = ReplayConfig(mode="recorded_projection")
    load_replay_image(session, config, path=tmp_path / "image").image.close()
    _corrupt_a_source_chunk(session)
    with pytest.raises(NrfCorruptionError, match="fails its checksum"):
        load_replay_image(session, config, path=tmp_path / "image")
    load_replay_image(session, config, path=tmp_path / "image", verify_source=False).image.close()


def test_feature_only_selection_names_nonzero_source_id(session: Path, tmp_path: Path) -> None:
    """A feature stream selected without its source still cross-references it.

    Section 8.3 makes every cross-reference use a mapped native id, never 0, and
    a feature descriptor's ``source_stream_id`` must be nonzero. The source is
    not emitted, but it is in the SignalId namespace and the id registry, so a
    feature-only selection produces a valid schema rather than one that writes 0
    for an unselected source.
    """
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"bandpower": StreamReplayRange("block_ordinal", 0, 8)},
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        schema = image.schema_document()
        descriptor = schema["feature_sets"][0]
        assert descriptor["source_stream_id"] != 0
        assert image.metadata.id_registry["signals"]["neural"] == descriptor["source_stream_id"]
        assert [signal["id"] for signal in schema["signals"]] == [1]


def test_id_registry_records_every_namespace(session: Path, tmp_path: Path) -> None:
    """The full string -> native id mapping for every namespace is in metadata.

    Section 8.3 requires the mapping to be recorded so a reader can resolve any
    cross-reference without re-deriving it; each namespace is sorted by UTF-8
    order and numbered from 1.
    """
    with _synthesized(
        session, tmp_path / "image", neural=(0, 8), cursor=(0, 8), bandpower=(0, 8)
    ) as image:
        registry = image.metadata.id_registry
        assert registry["signals"] == {"bandpower": 1, "cursor": 2, "neural": 3}
        assert registry["clocks"] == {"clock.domain-7": 1}
        assert registry["feature_sets"] == {"bandpower.set": 1}
        assert registry["units"] == {"v-2": 1}


def test_synthesized_block_carries_observation_time(session: Path, tmp_path: Path) -> None:
    """Observation times are recorded data replayed unchanged (section 8.6).

    A per-stream block index does not store them, so the builder recovers them
    from the source -- explicit-timing streams read their stored timestamps --
    rather than fabricating zero. The bandpower stream is 50 Hz with one sample
    per block, so block N starts at N * 20_000_000 ns; a block past the first is
    nonzero, which a fabricated zero would not be.
    """
    with _synthesized(session, tmp_path / "image", bandpower=(0, 8)) as image:
        blocks = [image.blocks_of(item)[0] for item in image.items() if item.kind == "frame"]
        for block in blocks:
            assert block.observation_time_start_ns == block.source_block_ordinal * 20_000_000
        assert any(block.observation_time_start_ns > 0 for block in blocks)


def test_regular_timing_block_recovers_observation_time(tmp_path: Path) -> None:
    """A regular-timing stream derives the grid from the rate and segment origin.

    The legacy session stores no timestamps; the builder recovers the grid from
    the declared rate and the segment start the recording froze, not a made-up
    zero (section 8.6).
    """
    session = _legacy_session(tmp_path)
    config = ReplayConfig(
        mode="stream_frames",
        stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 6)},
    )
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        blocks = [image.blocks_of(item)[0] for item in image.items() if item.kind == "frame"]
        for block in blocks:
            assert block.observation_time_start_ns == block.sample_idx_start * 1_000_000
        assert any(block.observation_time_start_ns > 0 for block in blocks)


def test_allow_incomplete_must_be_bool() -> None:
    """A truthy non-bool such as the string 'false' would silently enable the mode."""
    with pytest.raises(ReplayConfigError, match="allow_incomplete must be a bool"):
        ReplayConfig(mode="exact_frames", allow_incomplete="false")


def test_failed_publication_leaves_no_temporary(
    session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename failure cleans up its temporary, so neither a half image nor a
    stranded temp is left for a later call to mistake for a real image."""
    import os

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="injected rename failure"):
        build_replay_image(session, ReplayConfig(mode="exact_frames"), path=tmp_path / "image")
    assert not (tmp_path / "image.tmp").exists()
    assert not (tmp_path / "image").exists()
    assert not (tmp_path / "image.payload").exists()
    # With mkstemp the scratch and temp names are unpredictable, so also
    # verify no stray temp files of any name are left behind.
    assert not list(tmp_path.glob(".*.payload"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_payload_scratch_symlink_is_not_followed(session: Path, tmp_path: Path) -> None:
    """A pre-existing symlink at the old scratch name cannot truncate the source.

    Before mkstemp, the payload scratch was opened at a predictable name
    (``<target>.payload``) with a plain ``open(..., "w+b")``, which follows an
    existing symlink. An attacker who pre-creates ``image.payload`` as a symlink
    into the source session could truncate a source file. mkstemp creates a
    unique name with O_CREAT | O_EXCL, so the symlink is never opened and the
    source is left read-only (contract section 8).
    """
    manifest = session
    original = manifest.read_bytes()
    # Pre-create the old predictable scratch name as a symlink into the source.
    _create_symlink_or_skip(tmp_path / "image.payload", manifest)
    config = ReplayConfig(mode="exact_frames")
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        image.close()
    assert manifest.read_bytes() == original
    assert (tmp_path / "image.payload").is_symlink()
    assert (tmp_path / "image").exists()


def test_publication_temp_symlink_is_not_followed(session: Path, tmp_path: Path) -> None:
    """A pre-existing symlink at the old publication-temp name cannot overwrite the source.

    Before mkstemp, the publication temp was opened at a predictable name
    (``<target>.tmp``) with a plain ``open(..., "wb")``, which follows an
    existing symlink. An attacker who pre-creates ``image.tmp`` as a symlink
    into the source session could overwrite a source file with the replay-image
    header. mkstemp creates a unique name with O_CREAT | O_EXCL, so the symlink
    is never opened and the source is left read-only (contract section 8).
    """
    manifest = session
    original = manifest.read_bytes()
    # Pre-create the old predictable publication-temp name as a symlink into
    # the source.
    _create_symlink_or_skip(tmp_path / "image.tmp", manifest)
    config = ReplayConfig(mode="exact_frames")
    with build_replay_image(session, config, path=tmp_path / "image") as image:
        image.close()
    assert manifest.read_bytes() == original
    assert (tmp_path / "image.tmp").is_symlink()
    assert (tmp_path / "image").exists()
