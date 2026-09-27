# SPDX-License-Identifier: MIT
"""Atomic multi-signal ingress and existing recording/replay compatibility."""

from __future__ import annotations

import numpy as np
import pytest

from neurale import devices, streaming
from neurale._native_loader import load_native_namespace
from neurale.devices.provider import _native_signals, host_time_ns


def source_at(path, capacity=4):
    return load_native_namespace("devices").GenericDeviceSource(
        _native_signals(
            [
                devices.Signal("data", 2, 1000, 4, dtype="float32"),
                devices.Signal("timestamps", 2, 1000, 4, dtype="float64"),
            ]
        ),
        str(path),
        capacity,
    )


@pytest.mark.parametrize("bad", ["duplicate", "dtype", "shape", "empty"])
def test_invalid_frame_never_partially_publishes(tmp_path, bad):
    source = source_at(tmp_path / "queue")
    data = np.ones((4, 2), dtype=np.float32)
    times = np.ones((4, 2), dtype=np.float64)
    blocks = [(0, data, {}), (1, times, {})]
    if bad == "duplicate":
        blocks[1] = (0, data, {})
    if bad == "dtype":
        blocks[1] = (1, data, {})
    if bad == "shape":
        blocks[1] = (1, np.ones((5, 2)), {})
    if bad == "empty":
        blocks = []
    with pytest.raises((ValueError, RuntimeError)):
        source.ingress.publish_frame(blocks)
    assert source.ingress.stats["published"] == 0
    source.close()


def test_native_clock():
    assert 0 < host_time_ns() <= host_time_ns()


def test_paired_frame_record_replay(tmp_path):
    from neurale.io.nrf import NrfReader
    from neurale.recording import (
        RecorderConfig,
        RecorderLimits,
        ReplayConfig,
        SessionRecorder,
        StreamMetadata,
        StreamRecording,
        build_replay_image,
    )

    source = source_at(tmp_path / "queue")
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    times = np.column_stack((100 + np.arange(4) * 0.001, 200 + np.arange(4) * 0.001))
    metadata = {
        "sample_index": 0,
        "device_tick": 0,
        "tick_reference": 0,
        "host_reference_ns": 200_000_000_000,
        "tick_rate_numerator": 1_000_000_000,
        "tick_rate_denominator": 1,
        "uncertainty_ns": 100,
        "clock_generation": 1,
        "synchronized": 1,
    }
    source.ingress.publish_frame([(0, values, metadata), (1, times, metadata)])
    source.ingress.finish()
    schema = source.schema
    path = tmp_path / "paired.nrf"
    recorder = SessionRecorder.create(
        RecorderConfig(
            path=path,
            streams=(
                StreamRecording(
                    "data", signal=schema.signals[0], metadata=StreamMetadata(unit="V")
                ),
                StreamRecording(
                    "timestamps", signal=schema.signals[1], metadata=StreamMetadata(unit="s")
                ),
            ),
            limits=RecorderLimits(
                frame_queue_capacity=16,
                control_queue_capacity=32,
                spool_capacity_bytes=1 << 20,
                max_control_records=4096,
                checkpoint_interval=0,
            ),
        ),
        source,
    )
    consumer = streaming.CountingNativeConsumer()
    runner = streaming.StreamRunner(
        schema,
        streaming.RealtimeConfig(),
        source,
        streaming.IdentityNativeProcessor(),
        consumer,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    try:
        recorder.prepare()
        recorder.attach(runner)
        assert runner.prepare() == streaming.StreamStatus.OK
        assert runner.arm() == streaming.StreamStatus.OK
        status = runner.run()
        fault = runner.primary_fault
        assert status == streaming.StreamStatus.OK, (fault.code, fault.stage, fault.detail)
        recorder.stop("complete")
        assert recorder.finalize().complete
    finally:
        runner.stop()
        recorder.close()
        source.close()
    with NrfReader.open(path) as reader:
        np.testing.assert_array_equal(reader.read_stream("data"), values)
        np.testing.assert_array_equal(reader.read_stream("timestamps"), times)
    with build_replay_image(
        path, ReplayConfig(mode="exact_frames"), path=tmp_path / "paired.nrimg"
    ) as image:
        frames = [item for item in image.items() if image.blocks_of(item)]
        assert len(frames) == 1
        blocks = image.blocks_of(frames[0])
        assert len(blocks) == 2
        assert blocks[0].clock_sync == blocks[1].clock_sync
        assert blocks[0].clock_sync is not None
        payload = image.payload_of(frames[0])
        assert payload == values.tobytes() + times.tobytes()
