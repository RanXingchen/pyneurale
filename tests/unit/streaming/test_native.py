#!/usr/bin/env python3

from __future__ import annotations

import gc
import sys
import threading
import time

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


def _config(ingress_capacity: int = 4):
    budget = native.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = ingress_capacity
    budget.processor_owned = 2
    budget.critical_edge_capacity = ingress_capacity + 1
    budget.actuator_owned = 1

    config = native.RealtimeConfig()
    config.pool_capacity = budget
    config.buffer_size = 32
    config.max_signal_blocks = 1
    config.discontinuity_capacity = 2
    config.gaps_per_discontinuity = 1
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 2
    config.platform.mode = native.RealtimeConfigMode.STRICT
    config.validate()
    return config


def _pipeline(n_frames: int, *, fail_at: int | None = None):
    schema = _schema()
    source = streaming.SyntheticNativeSource(
        schema,
        n_frames,
        fail_at=fail_at,
    )
    processor = streaming.IdentityNativeProcessor()
    consumer = streaming.CountingNativeConsumer()
    runner = streaming.StreamRunner(
        schema,
        _config(max(4, n_frames)),
        source,
        processor,
        consumer,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    return runner, source, processor, consumer


def test_native_facade_lifecycle_and_stats() -> None:
    assert not hasattr(streaming, "NativeRuntime")
    assert not hasattr(streaming, "NativeStreamRunner")

    runner, source, processor, consumer = _pipeline(3)
    assert runner.state == streaming.RuntimeState.CREATED
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.heartbeat.safety_inhibited
    assert runner.arm() == streaming.StreamStatus.OK
    assert not runner.heartbeat.safety_inhibited
    assert runner.run() == streaming.StreamStatus.OK
    assert runner.state == streaming.RuntimeState.STOPPED
    assert runner.heartbeat.safety_inhibited
    assert runner.heartbeat.runtime_generation == 1
    assert runner.heartbeat.output_valid
    assert runner.heartbeat.last_valid_output > 0
    assert runner.stats.safety_inhibitions == 2

    stats = runner.stats
    assert stats.frames_acquired == 3
    assert stats.frames_read == 3
    assert stats.frames_processed == 3
    assert stats.frames_published == 3
    assert stats.frames_consumed == 3
    assert stats.end_of_streams == 1
    assert stats.queue_overruns == 0
    assert stats.pool_exhaustions == 0
    assert source.frames_emitted == 3
    assert processor.process_count == 3
    assert processor.flush_count == 1
    assert consumer.frame_count == 3
    assert consumer.last_sequence == 2
    assert runner.primary_fault is None
    assert runner.fault_history == []
    assert runner.outstanding_frames == 0
    assert runner.outstanding_discontinuities == 0
    assert runner.stop() == streaming.StreamStatus.OK

    assert runner.reset() == streaming.StreamStatus.OK
    assert runner.state == streaming.RuntimeState.CREATED
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    assert runner.join() == streaming.StreamStatus.OK
    assert runner.state == streaming.RuntimeState.STOPPED
    assert runner.heartbeat.runtime_generation == 2


def test_native_schema_exposes_bci_semantics_and_clock_sync() -> None:
    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT32,
        8,
        4,
        8,
        streaming.RationalRate(1_000, 1),
        3,
        physical_unit=streaming.PhysicalUnit.VOLTS,
        channel_set_id=11,
        calibration_id=12,
        reference_id=13,
        channel_names=tuple(f"C{index + 1}" for index in range(8)),
        channel_impedances_ohm=(1_000.0,) * 8,
    )
    assert signal.physical_unit == streaming.PhysicalUnit.VOLTS
    assert signal.channel_set_id == 11
    assert signal.calibration_id == 12
    assert signal.reference_id == 13
    assert signal.channel_impedances_ohm == [1_000.0] * 8

    event = streaming.SignalSchema(
        2,
        streaming.SignalDType.INT32,
        1,
        1,
        1,
        streaming.RationalRate(0, 1),
        3,
        kind=streaming.SignalKind.EVENT,
    )
    assert event.kind == streaming.SignalKind.EVENT

    sync = streaming.ClockSyncSnapshot()
    sync.clock_domain = 3
    sync.generation = 2
    sync.device_tick_rate = streaming.RationalRate(1_000, 1)
    sync.flags = streaming.ClockSyncFlags.SYNCHRONIZED
    assert sync.clock_domain == 3
    assert sync.generation == 2


def test_fault_is_copied_to_stable_python_values() -> None:
    runner, source, processor, consumer = _pipeline(3, fail_at=1)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.SOURCE_FAILURE
    assert runner.state == streaming.RuntimeState.FAILED

    fault = runner.primary_fault
    stats = runner.stats
    assert fault is not None
    assert fault.code == streaming.FaultCode.SOURCE_READ
    assert fault.status == streaming.StreamStatus.SOURCE_FAILURE
    assert fault.stage == streaming.FaultStage.SOURCE
    assert fault.runtime_generation > 0
    assert stats.max_ingress_dwell_ns >= 0
    assert stats.max_processor_execution_ns >= 0
    assert stats.max_source_to_actuator_ns >= 0
    assert stats.faults == 1
    assert runner.outstanding_frames == 0

    del runner, source, processor, consumer
    gc.collect()
    assert fault.code == streaming.FaultCode.SOURCE_READ
    assert stats.faults == 1


def test_run_releases_gil() -> None:
    runner, _, _, consumer = _pipeline(20_000)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK

    ready = threading.Event()
    stop = threading.Event()
    progress = [0]

    def worker() -> None:
        ready.set()
        while not stop.is_set():
            progress[0] += 1
            time.sleep(0)

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(timeout=1)

    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(10.0)
        before = progress[0]
        assert runner.run() == streaming.StreamStatus.OK
        after = progress[0]
    finally:
        sys.setswitchinterval(previous_interval)
        stop.set()
        thread.join(timeout=1)

    assert not thread.is_alive()
    assert after > before
    assert consumer.frame_count == 20_000


def test_realtime_components_are_native_only() -> None:
    with pytest.raises(TypeError):

        class PythonSource(streaming.SyntheticNativeSource):
            pass

    schema = _schema()
    processor = streaming.IdentityNativeProcessor()
    consumer = streaming.CountingNativeConsumer()
    with pytest.raises(TypeError):
        streaming.StreamRunner(
            schema,
            _config(),
            object(),
            processor,
            consumer,
        )

    runner, source, processor, consumer = _pipeline(1)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    del runner
    gc.collect()
    assert source.frames_emitted <= 1

    assert streaming.StreamRunner.__module__ == "neurale.streaming.runner"


def test_native_safety_controller_is_explicit_and_final() -> None:
    schema = _schema()
    source = streaming.SyntheticNativeSource(schema, 1)
    processor = streaming.IdentityNativeProcessor()
    consumer = streaming.CountingNativeConsumer()
    safety = streaming.RecordingNativeSafetyController()
    runner = streaming.StreamRunner(
        schema,
        _config(),
        source,
        processor,
        consumer,
        safety_controller=safety,
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert safety.inhibit_count == 1
    assert safety.last_reason == streaming.SafetyReason.STARTUP
    assert runner.start() == streaming.StreamStatus.INVALID_STATE
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    assert safety.release_count == 1
    assert runner.join() == streaming.StreamStatus.OK
    assert safety.inhibit_count == 2
    assert runner.heartbeat.safety_inhibited

    with pytest.raises(TypeError):

        class PythonSafety(streaming.RecordingNativeSafetyController):
            pass


def test_native_observer_edge_control_plane() -> None:
    schema = _schema()
    source = streaming.SyntheticNativeSource(schema, 4)
    processor = streaming.IdentityNativeProcessor()
    actuator = streaming.CountingNativeConsumer()
    observer = streaming.CountingNativeObserver()
    config = _config(4)
    config.pool_capacity.observer_edge_capacity = 4
    edge = streaming.ObserverEdgeConfig()
    edge.id = 9
    edge.capacity = 4
    edge.drop_history_capacity = 4
    edge.drop_policy = streaming.ObserverDropPolicy.DROP_NEWEST

    runner = streaming.StreamRunner(
        schema,
        config,
        source,
        processor,
        actuator,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    assert runner.add_observer(observer, edge) == streaming.StreamStatus.OK
    with pytest.raises(ValueError, match="unique"):
        runner.add_observer(streaming.CountingNativeObserver(), edge)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert (
        runner.add_observer(streaming.CountingNativeObserver(), edge)
        == streaming.StreamStatus.INVALID_STATE
    )
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert observer.frame_count == 4
    assert runner.observer_stats(9).delivered == 4
    assert runner.observer_stats(9).last_enqueued_sequence == 3
    assert runner.observer_stats(9).last_delivered_sequence == 3
    assert runner.observer_drop_ranges(9) == []

    with pytest.raises(TypeError):

        class PythonObserver(streaming.CountingNativeObserver):
            pass


def test_realtime_platform_configuration_and_capability_facade() -> None:
    capabilities = streaming.realtime_platform_capabilities()
    assert capabilities.logical_cpu_count >= 0
    assert not capabilities.hard_realtime_guaranteed
    assert capabilities.windows is sys.platform.startswith("win")
    assert capabilities.linux is sys.platform.startswith("linux")

    thread = streaming.RealtimeThreadConfig(
        cpu_affinity_mask=1,
        stack_size=4096,
        prefault_stack_bytes=4096,
    )
    config = streaming.RealtimeConfig(platform=streaming.RealtimePlatformConfig(processing=thread))
    runner = streaming.StreamRunner(
        _schema(),
        config,
        streaming.SyntheticNativeSource(_schema(), 1),
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        profile=streaming.ExecutionProfile.RESEARCH,
    )
    assert runner._native_config.platform.mode == native.RealtimeConfigMode.BEST_EFFORT
    assert runner._native_config.platform.processing.cpu_affinity_mask == 1
    assert runner._native_config.platform.processing.stack_size == 4096


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (streaming.ExecutionProfile.RESEARCH, streaming.StreamStatus.OK),
        (
            streaming.ExecutionProfile.REALTIME,
            streaming.StreamStatus.REALTIME_CONFIGURATION_FAILED,
        ),
    ],
)
def test_realtime_stack_size_is_strict_or_warning(profile, expected) -> None:
    capabilities = streaming.realtime_platform_capabilities()
    if capabilities.supports(streaming.RealtimeFeature.STACK_SIZE):
        pytest.skip("platform backend supports custom stack size")

    schema = _schema()
    config = streaming.RealtimeConfig(
        platform=streaming.RealtimePlatformConfig(
            processing=streaming.RealtimeThreadConfig(stack_size=64 * 1024)
        )
    )
    source = streaming.SyntheticNativeSource(schema, 1)
    processor = streaming.IdentityNativeProcessor()
    consumer = streaming.CountingNativeConsumer()
    runner = streaming.StreamRunner(
        schema,
        config,
        source,
        processor,
        consumer,
        profile=profile,
        safety_controller=(
            streaming.RecordingNativeSafetyController()
            if profile == streaming.ExecutionProfile.REALTIME
            else None
        ),
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == expected
    status = runner.realtime_configuration_status
    stack_size = int(streaming.RealtimeFeature.STACK_SIZE.value)
    assert status.processing.unsupported & stack_size
    assert status.warning_count == 1
    if profile == streaming.ExecutionProfile.REALTIME:
        assert runner.state == streaming.RuntimeState.FAILED
        assert runner.primary_fault.code == streaming.FaultCode.REALTIME_CONFIGURATION
    else:
        assert runner.state == streaming.RuntimeState.STOPPED
        assert runner.primary_fault is None
