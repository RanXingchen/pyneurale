#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise independently installed providers using only public wheel APIs."""

from __future__ import annotations

import json
import time

import numpy as np

from neurale import devices, streaming


class Capture:
    def __init__(self):
        self.frames = []
        self.gaps = []

    def write(self, frame):
        self.frames.append(frame)

    def handle_discontinuity(self, gap):
        self.gaps.append(gap)


def validate(name):
    with devices.open(name) as device:
        consumer = streaming.CountingNativeConsumer()
        runner = streaming.StreamRunner(
            device.schema,
            streaming.RealtimeConfig(),
            device.source,
            streaming.IdentityNativeProcessor(),
            consumer,
            safety_controller=streaming.RecordingNativeSafetyController(),
        )
        durations = []
        try:
            for repetition in range(2):
                if repetition:
                    assert runner.reset() == streaming.StreamStatus.OK
                assert runner.prepare() == streaming.StreamStatus.OK
                assert runner.arm() == streaming.StreamStatus.OK
                started = time.perf_counter_ns()
                device.start()
                assert runner.run() == streaming.StreamStatus.OK, runner.primary_fault
                durations.append(time.perf_counter_ns() - started)
                assert consumer.frame_count == 4
                assert consumer.discontinuity_count == 2
        finally:
            runner.stop()

    with devices.open(name) as device:
        capture = Capture()
        runner = streaming.StreamRunner(
            device.schema,
            streaming.RealtimeConfig(),
            device.source,
            streaming.IdentityNativeProcessor(),
            streaming.PythonSinkAdapter(capture),
            profile="research",
        )
        try:
            assert runner.prepare() == streaming.StreamStatus.OK
            assert runner.arm() == streaming.StreamStatus.OK
            device.start()
            assert runner.run() == streaming.StreamStatus.OK, runner.primary_fault
            assert [f.blocks[0].signal_id for f in capture.frames] == [1, 1, 2, 3]
            assert [f.blocks[0].sample_idx_start for f in capture.frames] == [0, 6, 0, 0]
            expected = [np.arange(8), np.arange(12, 16), np.array([20, 21]), np.array([7])]
            for frame, values in zip(capture.frames, expected, strict=True):
                dtype = np.int32 if frame.blocks[0].signal_id == 3 else np.float64
                np.testing.assert_array_equal(frame.payload.view(dtype), values)
            assert [g.signal_gaps[0].missing_samples for g in capture.gaps] == [2, None]
            assert capture.gaps[1].reason == streaming.GapReason.DEVICE_RESTART
            stats = device.stats
        finally:
            runner.stop()
    return {"provider": name, "realtime_run_ns": durations, "stats": stats}


def main():
    names = ["example.pull", "example.callback", "example.python"]
    missing = set(names) - set(devices.list())
    if missing:
        raise RuntimeError(f"install all standalone example packages first: {sorted(missing)}")
    print(json.dumps([validate(name) for name in names], indent=2))


if __name__ == "__main__":
    main()
