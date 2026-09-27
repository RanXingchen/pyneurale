#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Public simulated-device -> NRF -> native replay composition."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _native_harness import find_native_test_binary

import neurale.streaming as streaming
from neurale.devices.simulation import (
    AcquisitionEvent,
    KnownSampleLoss,
    ManualHostClock,
    SimulationFaultPlan,
    SimulationTimingConfig,
    SourceFault,
    _finite_simulated_neural_device,
)
from neurale.io.nrf import NrfReader
from neurale.recording import (
    RecorderConfig,
    RecorderLimits,
    ReplayConfig,
    SessionRecorder,
    StreamRecording,
    block_index_columns,
    build_replay_image,
)
from neurale.signal.simulation import SignalGenerator

_SPOOL_CAPACITY = 1 << 20
_BINARY_NAME = "neurale_recording_end_to_end_test"


def _stable_backend_available() -> bool:
    if sys.platform == "win32":
        return True
    if sys.platform != "linux" or not Path("/dev/shm").is_dir():
        return False
    import resource

    soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return soft == resource.RLIM_INFINITY or soft >= _SPOOL_CAPACITY


pytestmark = [
    pytest.mark.native,
    pytest.mark.stable_backend,
    pytest.mark.skipif(
        not _stable_backend_available(),
        reason="record/replay needs the shipping Linux tmpfs or Windows mapped spool",
    ),
]


def _replay_harness() -> Path | None:
    # Return the built end-to-end harness, or None when there is no C++ build.

    return find_native_test_binary(
        _BINARY_NAME,
        explicit_env="NEURALE_END_TO_END_TEST_BINARY",
        repo_root=Path(__file__).resolve().parents[3],
    )


@pytest.fixture(scope="module")
def replay_harness() -> Path:
    binary = _replay_harness()
    if binary is None:
        pytest.skip(f"{_BINARY_NAME} is not built")
    return binary


def _runner_config() -> streaming.RealtimeConfig:
    return streaming.RealtimeConfig(platform=streaming.RealtimePlatformConfig(prefault_pools=True))


def _record(
    tmp_path: Path,
    *,
    events: tuple[AcquisitionEvent, ...] = (KnownSampleLoss(3, 4),),
    output_name: str = "m7-simulated.nrf",
    expected_join: streaming.StreamStatus = streaming.StreamStatus.OK,
    expected_complete: bool = True,
    paced: bool = False,
) -> tuple[Path, SignalGenerator, int]:
    generator = SignalGenerator.tones(
        2,
        1_000.0,
        [7.0, 83.0],
        amps=[[1.0, 0.75], [0.2, 0.3]],
        phases=[[0.0, 0.25], [0.5, -0.25]],
    )
    device = _finite_simulated_neural_device(
        generator,
        samples_per_frame=16,
        total_sample_count=96,
        channel_names=("left", "right"),
        paced=paced,
        timing=SimulationTimingConfig(
            initial_sample_idx=100,
            initial_device_tick=5_000,
            clock_offset_ns=250,
            clock_sync_uncertainty_ns=50,
            clock=None if paced else ManualHostClock(1_000_000_000),
        ),
        faults=SimulationFaultPlan(events),
    )
    output = tmp_path / output_name
    config = RecorderConfig(
        path=output,
        streams=(StreamRecording(),),
        limits=RecorderLimits(
            frame_queue_capacity=32,
            control_queue_capacity=32,
            spool_capacity_bytes=_SPOOL_CAPACITY,
            max_control_records=4 * 1024,
            checkpoint_interval=0,
        ),
    )
    recorder = SessionRecorder.create(config, device)
    planned = recorder.plan.stream_for(1)
    assert planned is not None
    assert planned.unit == ("V", "V")
    assert planned.channel_names == ("left", "right")
    assert recorder.plan.resource_bounds.streams[0].capacity == 1024
    runner = streaming.StreamRunner(
        device.schema,
        _runner_config(),
        device.source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    assert runner.join() == expected_join
    assert runner.outstanding_frames == 0
    assert runner.outstanding_discontinuities == 0
    if expected_join == streaming.StreamStatus.OK:
        assert runner.primary_fault is None
    else:
        assert runner.primary_fault is not None
        assert runner.primary_fault.status == expected_join
    recorder.stop("m7-simulated-terminal")
    status = recorder.finalize()
    assert status.complete is expected_complete
    assert status.accounting_verified is True
    recorder.close()
    return output, generator, device._native_session_id


def _run_harness(binary: Path, *arguments: str) -> list[dict[str, object]]:
    result = subprocess.run(
        [str(binary), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert result.returncode == 0, (rows, result.stderr)
    assert rows, result.stderr
    return rows


def test_simulated_device_records_canonical_nrf_and_replays_natively(
    tmp_path: Path, replay_harness: Path
) -> None:
    session, generator, _native_session_id = _record(tmp_path)
    with NrfReader.open(session) as reader:
        assert reader.complete is True
        assert reader.accounting_verified is True
        names = block_index_columns()
        columns = {name: idx for idx, name in enumerate(names)}
        blocks = reader.read_stream("neural.blocks")
        payload = reader.read_stream("neural")
        expected_blocks = []
        for row in blocks:
            start = int(row[columns["sample_idx_start"]])
            count = int(row[columns["n_samples"]])
            expected_blocks.append(generator.generate(start, count))
            assert int(row[columns["device_tick_start"]]) == 5_000 + start - 100
        np.testing.assert_array_equal(payload, np.concatenate(expected_blocks, axis=0))
        assert blocks[:, columns["frame_sequence"]].tolist() == list(range(len(blocks)))
        assert blocks[:, columns["sample_idx_start"]].tolist() == [100, 116, 132, 152, 168, 184]

        discontinuities = [
            schema
            for schema in reader.manifest["record_schemas"]
            if schema["kind"] == "discontinuities" and reader.record_extent(schema["id"]) > 0
        ]
        assert len(discontinuities) == 1
        assert reader.record_extent(discontinuities[0]["id"]) == 1

    image = tmp_path / "m7-simulated.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    runtime = _run_harness(replay_harness, "--runner", str(image))[0]
    assert runtime["status"] == "ok"
    assert runtime["runtime_state"] == "stopped"
    assert runtime["primary_fault_present"] is False
    assert runtime["image_frame_count"] == 6
    assert runtime["image_discontinuity_count"] == 1
    assert runtime["consumed_frames"] == 6
    assert runtime["consumed_discontinuities"] == 1
    assert runtime["outstanding_frames"] == 0
    assert runtime["outstanding_discontinuities"] == 0
    assert [block["sample_idx_start"] for block in runtime["consumed"]] == [
        100,
        116,
        132,
        152,
        168,
        184,
    ]
    assert [block["device_tick_start"] for block in runtime["consumed"]] == [
        5_000,
        5_016,
        5_032,
        5_052,
        5_068,
        5_084,
    ]

    direct = _run_harness(replay_harness, "--replay", str(image))
    messages = direct[1:-1]
    frames = [message for message in messages if message["kind"] == "frame"]
    gaps = [message for message in messages if message["kind"] == "discontinuity"]
    assert len(frames) == 6
    assert len(gaps) == 1

    # The point of chain D is that one simulated gap keeps its identity all the
    # way through recording and replay, so the gap has to be checked for what it
    # says, not only that it exists. `KnownSampleLoss(3, 4)` drops four samples
    # before the fourth frame: frame 2 ends at sample 147 and tick 5047, so the
    # stream expects 148/5048 and the device delivers 152/5052.
    gap_message = gaps[0]
    assert gap_message["reason"] == "sample_gap"
    assert gap_message["previous_sequence"] == 2
    assert gap_message["sequence"] == 3
    assert len(gap_message["gaps"]) == 1
    gap = gap_message["gaps"][0]
    assert gap["signal_id"] == 1
    assert gap["reason"] == "sample_gap"
    assert gap["expected_sample_idx"] == 148
    assert gap["actual_sample_idx"] == 152
    assert gap["missing_samples"] == 4
    assert gap["expected_device_tick"] == 5_048
    assert gap["actual_device_tick"] == 5_052

    for frame in frames:
        block = frame["blocks"][0]
        assert block["clock_sync_clock_domain"] == 1
        assert block["clock_sync_rate_numerator"] == 1_000
        assert block["clock_sync_rate_denominator"] == 1
        assert block["clock_sync_uncertainty_ns"] == 50
        assert block["clock_sync_generation"] == 1
        # The device's tick -> host-time mapping is what a later analysis reads
        # its absolute times from, so the configured 250 ns offset has to survive
        # the round trip rather than be recomputed at replay. The manual clock
        # starts at 1 s and the tick rate is 1 kHz, which fixes the reference
        # pair for every block.
        assert block["clock_sync_device_tick_reference"] == block["device_tick_start"]
        assert block["clock_sync_host_time_reference_ns"] == (
            1_000_000_000 + (block["device_tick_start"] - 5_000) * 1_000_000 + 250
        )


def test_simulated_source_fault_is_accounted_and_prefix_replayable(
    tmp_path: Path, replay_harness: Path
) -> None:
    session, _generator, _native_session_id = _record(
        tmp_path,
        events=(SourceFault(3),),
        output_name="m7-simulated-faulted.nrf",
        expected_join=streaming.StreamStatus.SOURCE_FAILURE,
        expected_complete=False,
        paced=True,
    )
    with NrfReader.open(session) as reader:
        assert reader.complete is False
        assert reader.accounting_verified is True
        committed_samples = reader.stream_extent("neural")
        assert 0 < committed_samples <= 48
        assert committed_samples % 16 == 0
        assert reader.record_extent("faults-v1") >= 1

    image = tmp_path / "m7-simulated-faulted.nrimg"
    build_replay_image(
        session,
        ReplayConfig(mode="exact_frames", allow_incomplete=True),
        path=image,
    ).close()
    rows = _run_harness(replay_harness, "--replay", str(image))
    assert rows[0]["abnormal_end_required"] is True
    assert len([row for row in rows[1:-1] if row["kind"] == "frame"]) == (committed_samples // 16)
    assert rows[-1]["terminal"] == "abnormal_end"


def test_fault_before_first_frame_keeps_session_identity(tmp_path: Path) -> None:
    """A source that fails before delivering a frame still records one session.

    The runtime has no frame header to attribute such a fault to, so it reports
    a zero session ID meaning "no context". The recorder knows the identity from
    its own plan, and every frame, discontinuity, and fault in an NRF session
    must share it -- so the unknown value must not overwrite what the plan froze.
    """
    session, _generator, native_session_id = _record(
        tmp_path,
        events=(KnownSampleLoss(0, 4), SourceFault(0)),
        output_name="m7-simulated-early-fault.nrf",
        expected_join=streaming.StreamStatus.SOURCE_FAILURE,
        expected_complete=False,
    )
    with NrfReader.open(session) as reader:
        assert reader.complete is False
        assert reader.accounting_verified is True
        # The fault aborts before any data message is accepted, so the session
        # is empty apart from the fault itself; the point here is whose session
        # the fault says it belongs to.
        assert reader.stream_extent("neural") == 0
        faults = reader.read_records("faults-v1")
        provenance = [json.loads(text) for text in faults["text"]]
        assert provenance, "the runtime fault must be recorded"
        runtime_faults = [row for row in provenance if row["frame_sequence"] == 0]
        assert runtime_faults
        for row in runtime_faults:
            assert row["native_session_id"] == native_session_id
