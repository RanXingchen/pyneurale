# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Sustained public recording with a fixed queue and a growing disk spool."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from neurale import streaming
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.recording import RecorderConfig, SessionRecorder
from neurale.signal.simulation import SignalGenerator
from neurale.streaming import _native as native_streaming

pytestmark = pytest.mark.slow


def test_continuous_recording_uses_fixed_queues_and_growing_spool(tmp_path: Path) -> None:
    output = tmp_path / "continuous.nrf"
    device = SimulatedNeuralDevice(
        SignalGenerator.noise(64, 30_000.0, seed=17), samples_per_frame=300, paced=True
    )
    recorder = SessionRecorder.create(RecorderConfig(path=output), device)
    budget = native_streaming.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = 8
    budget.processor_owned = 2
    budget.critical_edge_capacity = 9
    budget.actuator_owned = 1
    budget.observer_edge_capacity = 8
    config = native_streaming.RealtimeConfig()
    config.pool_capacity = budget
    config.buffer_size = 64 * 300 * 8
    config.max_signal_blocks = 1
    config.discontinuity_capacity = 2
    config.gaps_per_discontinuity = 1
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 4
    config.validate()
    runner = streaming.StreamRunner(
        device.schema,
        config,
        device.source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    try:
        recorder.prepare()
        assert recorder.spool_path.stat().st_size == 0
        recorder.attach(runner, capacity=8)
        assert runner.prepare() == streaming.StreamStatus.OK
        assert runner.arm() == streaming.StreamStatus.OK
        assert runner.start() == streaming.StreamStatus.OK
        queue_capacity = recorder.status.data_queue_capacity

        first_extent = None
        last_extent = 0
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            status = recorder.status
            assert status.data_queue_capacity == queue_capacity
            assert not status.primary_fault.present
            extent = recorder.spool_path.stat().st_size
            assert extent >= last_extent
            if first_extent is None:
                first_extent = extent
            last_extent = extent

        stopped = recorder.stop("continuous-recording-validation")
        assert stopped.spool_committed == stopped.recorder_accepted
        assert stopped.recorder_accepted > 0
        assert first_extent is not None and last_extent > first_extent
        finalized = recorder.finalize()
        assert finalized.complete
        assert finalized.queue_storage_released
        assert not finalized.worker_running
        assert output.stat().st_size > 0
    finally:
        if recorder.runtime_is_live():
            recorder.abort("validation-cleanup")
        recorder.close()
        device.close()
