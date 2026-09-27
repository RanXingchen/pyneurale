# SPDX-License-Identifier: MIT
"""Run --publish in one terminal, then run without flags in another terminal."""

from __future__ import annotations

import argparse
import time

import numpy as np

from neurale import devices, streaming


def publish(seconds):
    import pylsl

    info = pylsl.StreamInfo("PyNeurale demo", "EEG", 2, 1000, "float32", "pyneurale-demo")
    channels = info.desc().append_child("channels")
    for name in ("left", "right"):
        channel = channels.append_child("channel")
        channel.append_child_value("label", name)
        channel.append_child_value("unit", "V")
    outlet = pylsl.StreamOutlet(info)
    print("Publishing source_id=pyneurale-demo; Ctrl+C stops.", flush=True)
    started = pylsl.local_clock()
    while pylsl.local_clock() - started < seconds:
        # Timestamp each block on the sender clock, including the sample spacing.
        stamps = pylsl.local_clock() - 0.009 + np.arange(10) / 1000
        values = np.column_stack(
            (np.sin(stamps * 2 * np.pi * 10), np.cos(stamps * 2 * np.pi * 10))
        ).astype(np.float32)
        outlet.push_chunk(values, stamps.tolist())
        time.sleep(0.01)


def receive(source_id, seconds):
    with devices.open("lsl", source_id=source_id) as device:
        consumer = streaming.CountingNativeConsumer()
        runner = streaming.StreamRunner(
            device.schema,
            streaming.RealtimeConfig(),
            device.source,
            streaming.IdentityNativeProcessor(),
            consumer,
            safety_controller=streaming.RecordingNativeSafetyController(),
        )
        try:
            assert runner.prepare() == streaming.StreamStatus.OK
            assert runner.arm() == streaming.StreamStatus.OK
            device.start()
            assert runner.start() == streaming.StreamStatus.OK
            time.sleep(seconds)
        finally:
            runner.stop()
            runner.join()
        if runner.primary_fault is not None:
            raise RuntimeError(str(runner.primary_fault))
        print({"frames": consumer.frame_count, "stats": device.stats})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--source-id", default="pyneurale-demo")
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    if args.publish:
        publish(args.seconds)
    else:
        receive(args.source_id, args.seconds)
