#!/usr/bin/env python3

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import neurale.streaming as streaming
from neurale._native_loader import load_native_namespace

native = load_native_namespace("streaming")


def _schema():
    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT32,
        2,
        4,
        4,
        streaming.RationalRate(1_000, 1),
        7,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
    )
    return streaming.StreamSchema(11, [signal])


def _config(n_frames: int):
    capacity = max(4, n_frames)
    budget = native.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = capacity
    budget.processor_owned = 2
    budget.critical_edge_capacity = capacity + 1
    budget.actuator_owned = 1
    budget.observer_edge_capacity = capacity

    config = native.RealtimeConfig()
    config.pool_capacity = budget
    config.buffer_size = 32
    config.max_signal_blocks = 1
    config.discontinuity_capacity = 2
    config.gaps_per_discontinuity = 1
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 4
    config.platform.mode = native.RealtimeConfigMode.STRICT
    config.validate()
    return config


def _pipeline(
    n_frames: int,
    callback,
    *,
    bridge_capacity: int = 8,
    bridge_policy=None,
):
    schema = _schema()
    config = _config(n_frames)
    source = streaming.SyntheticNativeSource(schema, n_frames)
    processor = streaming.IdentityNativeProcessor()
    actuator = streaming.CountingNativeConsumer()
    bridge = streaming.PythonObserverBridge(
        callback,
        schema,
        capacity=bridge_capacity,
        drop_policy=(
            streaming.ObserverDropPolicy.DROP_OLDEST if bridge_policy is None else bridge_policy
        ),
        drop_history_capacity=max(8, n_frames),
    )
    edge = streaming.ObserverEdgeConfig()
    edge.id = 41
    edge.capacity = max(4, n_frames)
    edge.drop_history_capacity = max(8, n_frames)
    edge.drop_policy = streaming.ObserverDropPolicy.DROP_NEWEST

    runner = streaming.StreamRunner(
        schema,
        config,
        source,
        processor,
        actuator,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    assert runner.add_observer(bridge, edge) == streaming.StreamStatus.OK
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    return runner, actuator, bridge, edge


def test_blocked_callback_does_not_block_critical_path():
    callback_entered = threading.Event()
    callback_release = threading.Event()
    callback_done = threading.Event()
    callback_threads: list[int] = []

    def callback(frame) -> None:
        callback_threads.append(threading.get_ident())
        callback_entered.set()
        time.sleep(0.02)
        callback_release.wait(timeout=5)
        callback_done.set()

    runner, actuator, bridge, _ = _pipeline(1, callback, bridge_capacity=1)
    main_thread = threading.get_ident()

    assert runner.run() == streaming.StreamStatus.OK
    assert callback_entered.wait(timeout=2)
    assert not callback_done.is_set()
    assert actuator.frame_count == 1
    assert runner.primary_fault is None
    assert runner.outstanding_frames == 0
    assert bridge.stats.callback_active
    assert bridge.stats.native_frames_outstanding == 0
    assert len(callback_threads) == 1
    assert callback_threads[0] != main_thread

    callback_release.set()
    bridge.close()
    assert callback_done.is_set()
    assert bridge.stats.worker_done
    bridge.close()


def test_callback_exception_is_local_to_bridge():
    def callback(_frame) -> None:
        raise RuntimeError("observer callback failed")

    runner, actuator, bridge, edge = _pipeline(8, callback)
    assert runner.run() == streaming.StreamStatus.OK
    bridge.close()

    assert actuator.frame_count == 8
    assert runner.primary_fault is None
    assert runner.observer_stats(edge.id).failures == 0
    assert bridge.stats.callback_errors == 1
    assert "observer callback failed" in bridge.callback_error
    assert bridge.stats.closed


def test_retained_frame_and_delayed_gc_use_python_owned_numpy_buffer():
    retained = []
    gc.disable()
    try:
        runner, _, bridge, _ = _pipeline(1, retained.append)
        assert runner.run() == streaming.StreamStatus.OK
        bridge.close()
        assert len(retained) == 1

        frame = retained[0]
        assert isinstance(frame, streaming.PythonObserverFrame)
        assert isinstance(frame.payload, np.ndarray)
        assert frame.payload.dtype == np.uint8
        assert frame.payload.flags.owndata
        assert frame.sequence == 0
        assert frame.blocks[0].n_samples == 4
        assert bridge.stats.native_frames_outstanding == 0

        del runner, bridge
        assert frame.payload.shape == (32,)
        assert np.count_nonzero(frame.payload) == 0
    finally:
        gc.enable()
        gc.collect()


def test_discontinuity_is_delivered_in_order():
    events = []
    schema = _schema()
    config = _config(2)
    source = streaming.SyntheticNativeSource(schema, 2, sequence_gap_at=1)
    processor = streaming.IdentityNativeProcessor()
    actuator = streaming.CountingNativeConsumer()
    bridge = streaming.PythonObserverBridge(events.append, schema)
    edge = streaming.ObserverEdgeConfig()
    edge.id = 13
    edge.capacity = 4
    edge.drop_history_capacity = 4
    runner = streaming.StreamRunner(
        schema,
        config,
        source,
        processor,
        actuator,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    assert runner.add_observer(bridge, edge) == streaming.StreamStatus.OK
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    bridge.close()

    assert [type(event) for event in events] == [
        streaming.PythonObserverFrame,
        streaming.PythonObserverDiscontinuity,
        streaming.PythonObserverFrame,
    ]
    assert events[1].reason == streaming.GapReason.FRAME_SEQUENCE_GAP
    assert events[1].previous_frame_sequence == 0
    assert events[1].actual_frame_sequence == 2
    assert actuator.discontinuity_count == 1
    assert bridge.stats.frames_delivered == 2
    assert bridge.stats.discontinuities_delivered == 1


def test_bridge_queue_full_drops_without_failing_runtime():
    callback_entered = threading.Event()
    callback_release = threading.Event()

    def callback(_frame) -> None:
        callback_entered.set()
        callback_release.wait(timeout=5)

    runner, actuator, bridge, edge = _pipeline(
        64,
        callback,
        bridge_capacity=1,
        bridge_policy=streaming.ObserverDropPolicy.DROP_NEWEST,
    )
    assert runner.run() == streaming.StreamStatus.OK
    assert callback_entered.wait(timeout=2)
    assert actuator.frame_count == 64
    assert runner.primary_fault is None
    assert runner.observer_stats(edge.id).failures == 0
    assert bridge.stats.dropped > 0
    assert bridge.stats.high_water_mark == 1
    assert bridge.drop_ranges

    callback_release.set()
    bridge.close()
    assert bridge.stats.native_frames_outstanding == 0


def test_python_bridge_cannot_be_critical_recorder():
    schema = _schema()
    config = _config(1)
    source = streaming.SyntheticNativeSource(schema, 1)
    processor = streaming.IdentityNativeProcessor()
    actuator = streaming.CountingNativeConsumer()
    bridge = streaming.PythonObserverBridge(lambda _frame: None, schema)
    edge = streaming.ObserverEdgeConfig()
    edge.id = 7
    edge.capacity = 1
    edge.drop_history_capacity = 1
    edge.critical_recorder = True
    runner = streaming.StreamRunner(
        schema,
        config,
        source,
        processor,
        actuator,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    with pytest.raises(ValueError, match="cannot be critical"):
        runner.add_observer(bridge, edge)
    bridge.close()

    with pytest.raises(TypeError):

        class PythonBridge(streaming.PythonObserverBridge):
            pass


def test_bridge_rejects_invalid_control_plane_configuration():
    schema = _schema()

    with pytest.raises(ValueError, match="callback must be callable"):
        streaming.PythonObserverBridge(object(), schema)
    with pytest.raises(ValueError, match="must be positive"):
        streaming.PythonObserverBridge(lambda _frame: None, schema, capacity=0)


def test_bridge_resets_with_native_runner():
    received = []
    runner, actuator, bridge, _ = _pipeline(1, received.append)

    assert runner.run() == streaming.StreamStatus.OK
    assert runner.reset() == streaming.StreamStatus.OK
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    bridge.close()

    assert actuator.frame_count == 1
    assert [frame.sequence for frame in received] == [0, 0]
    assert bridge.stats.delivered == 1
    assert bridge.stats.native_frames_outstanding == 0


def test_bridge_destruction_during_interpreter_shutdown():
    script = textwrap.dedent(
        """
        import neurale.streaming as s

        signal = s.SignalSchema(
            1, s.SignalDType.FLOAT32, 2, 4, 4,
            s.RationalRate(1000, 1), 7,
            device_tick_tracking=s.DeviceTickTracking.SAMPLE_COUNTER,
        )
        schema = s.StreamSchema(11, [signal])
        config = s.RealtimeConfig()
        source = s.SyntheticNativeSource(schema, 4)
        processor = s.IdentityNativeProcessor()
        actuator = s.CountingNativeConsumer()
        bridge = s.PythonObserverBridge(lambda frame: frame.sequence, schema)
        edge = s.ObserverEdgeConfig()
        edge.id = 1
        edge.capacity = 4
        edge.drop_history_capacity = 4
        runner = s.StreamRunner(
            schema, config, source, processor, actuator,
            safety_controller=s.RecordingNativeSafetyController(),
        )
        assert runner.add_observer(bridge, edge) == s.StreamStatus.OK
        assert runner.prepare() == s.StreamStatus.OK
        assert runner.arm() == s.StreamStatus.OK
        assert runner.run() == s.StreamStatus.OK
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
