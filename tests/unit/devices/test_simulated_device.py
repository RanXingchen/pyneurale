#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import neurale.streaming as streaming
from neurale.devices.simulation import (
    AppliedNeuralControl,
    BoundedStall,
    DeviceClockRestart,
    DeviceTickJump,
    Disconnect,
    IntentDrivenNeuralDevice,
    KnownSampleLoss,
    ManualHostClock,
    NeuralDriftSchedule,
    SimulatedNeuralDevice,
    SimulationFaultPlan,
    SimulationTimingConfig,
    SourceFault,
    TransientWouldBlock,
    _finite_simulated_neural_device,
)
from neurale.exceptions import StreamStateError, ValidationError
from neurale.signal.simulation import NeuralSignalConfig, NeuralSignalGenerator, SignalGenerator
from neurale.streaming import _native as streaming_native


def _config(buffer_size: int) -> object:
    budget = streaming_native.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = 4
    budget.processor_owned = 2
    budget.critical_edge_capacity = 5
    budget.actuator_owned = 1

    config = streaming_native.RealtimeConfig()
    config.pool_capacity = budget
    config.buffer_size = buffer_size
    config.max_signal_blocks = 1
    config.discontinuity_capacity = 2
    config.gaps_per_discontinuity = 1
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 2
    config.validate()
    return config


def test_device_freezes_sample_major_schema_and_channel_metadata() -> None:
    generator = SignalGenerator.sine(2, 4_000.0, [70.0, 120.0])
    device = SimulatedNeuralDevice(
        generator,
        samples_per_frame=4,
        channel_names=["left", "right"],
        channel_impedances_ohm=[5_000, 20_000],
        physical_unit="volts",
        paced=False,
        timing=SimulationTimingConfig(initial_sample_idx=100, initial_device_tick=9_000),
    )

    assert device.generator is generator
    assert device.channel_names == ("left", "right")
    assert device.schema.id == 1
    assert len(device.schema.signals) == 1
    signal = device.schema.signals[0]
    assert signal.id == 1
    assert signal.n_channels == 2
    assert signal.nominal_block_samples == 4
    assert signal.max_block_samples == 4
    assert signal.fs.numerator == 4_000
    assert signal.fs.denominator == 1
    assert signal.dtype == streaming.SignalDType.FLOAT64
    assert signal.layout == streaming.SignalLayout.SAMPLE_MAJOR
    assert signal.kind == streaming.SignalKind.SAMPLED
    assert signal.channel_names == ["left", "right"]
    assert signal.channel_impedances_ohm == [5_000.0, 20_000.0]
    assert signal.device_tick_tracking == streaming.DeviceTickTracking.SAMPLE_COUNTER
    assert signal.physical_unit == streaming.PhysicalUnit.VOLTS
    assert signal.clock_domain == 1
    assert signal.channel_set_id == 1
    assert signal.calibration_id == 0
    assert signal.reference_id == 0


def test_public_constructor_exposes_only_researcher_facing_parameters() -> None:
    parameters = inspect.signature(SimulatedNeuralDevice).parameters
    assert tuple(parameters) == (
        "generator",
        "samples_per_frame",
        "channel_names",
        "channel_impedances_ohm",
        "sample_dtype",
        "physical_unit",
        "paced",
        "timing",
        "faults",
    )
    assert parameters["timing"].default is None
    assert parameters["faults"].default is None
    with pytest.raises(TypeError, match="total_sample_count"):
        SimulatedNeuralDevice(
            SignalGenerator.zeros(1, 1_000.0),
            samples_per_frame=1,
            total_sample_count=1,  # type: ignore[call-arg]
        )


def test_intent_driven_device_is_opt_in_and_accepts_manual_control() -> None:
    config = NeuralSignalConfig(n_channels=2, fs=1_000.0, seed=17)
    drift_fingerprint = NeuralSignalGenerator(config).drift_fingerprint
    device = IntentDrivenNeuralDevice(
        config,
        samples_per_frame=4,
        paced=True,
        drift_schedule=NeuralDriftSchedule(2, 4),
    )

    assert tuple(inspect.signature(IntentDrivenNeuralDevice).parameters) == (
        "config",
        "samples_per_frame",
        "paced",
        "drift_schedule",
    )
    assert device.config is config
    assert not hasattr(device, "generator")
    assert device.schema.signals[0].n_channels == 2
    assert device.channel_names == ("channel_0", "channel_1")
    assert device.drift_fingerprint == drift_fingerprint
    assert device.pop_applied_control() is None
    with pytest.raises(ValidationError, match="config must be a NeuralSignalConfig"):
        IntentDrivenNeuralDevice(
            NeuralSignalGenerator(config),
            samples_per_frame=4,
        )
    device.publish_control((0.25, -0.5), drift_progress=0.3, context_ordinal=7)
    with pytest.raises(ValidationError, match="exactly two"):
        device.publish_control((0.25,), drift_progress=0.3, context_ordinal=7)
    consumer = streaming.CountingNativeConsumer()
    runner = streaming.StreamRunner(
        device.schema,
        _config(64),
        device.source,
        streaming.IdentityNativeProcessor(),
        consumer,
        profile=streaming.ExecutionProfile.RESEARCH,
    )
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    deadline = time.monotonic() + 2.0
    while device.frames_emitted < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    runner.stop()
    assert runner.join() == streaming.StreamStatus.OK
    control = device.pop_applied_control()
    assert control is not None
    assert (control.intent_x, control.intent_y, control.drift_progress) == (0.25, -0.5, 0.3)
    assert control.first_sample_index == 0
    assert device.dropped_applied_control_count == 0
    device.close()

    assert AppliedNeuralControl.__slots__


def test_recording_intent_derives_simulated_device_schema_metadata(tmp_path: Path) -> None:
    from neurale.recording import (
        RecorderConfig,
        RecorderConfigError,
        StreamMetadata,
        StreamRecording,
        compile_recording_plan,
    )

    device = SimulatedNeuralDevice(
        SignalGenerator.sine(2, 1_000.0, [10.0, 20.0]),
        samples_per_frame=8,
        channel_names=("left", "right"),
        physical_unit="volts",
        paced=False,
    )

    plan = compile_recording_plan(
        RecorderConfig(path=tmp_path / "session.nrf"),
        device,
    )

    stream = plan.stream_for(1)
    assert stream is not None
    assert stream.stream_id == "neural"
    assert stream.unit == ("V", "V")
    assert stream.channel_names == ("left", "right")

    with pytest.raises(RecorderConfigError, match="native physical unit"):
        compile_recording_plan(
            RecorderConfig(
                path=tmp_path / "mislabeled.nrf",
                streams=(StreamRecording(metadata=StreamMetadata(unit="1")),),
            ),
            device,
        )


def test_advanced_simulation_configs_are_immutable() -> None:
    timing = SimulationTimingConfig(initial_sample_idx=3)
    faults = SimulationFaultPlan([KnownSampleLoss(1, 2)])

    assert timing.initial_sample_idx == 3
    assert faults.events == (KnownSampleLoss(1, 2),)
    with pytest.raises(FrozenInstanceError):
        timing.initial_sample_idx = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        faults.events = ()  # type: ignore[misc]


def test_sample_dtype_selects_declared_payload_type() -> None:
    generator = SignalGenerator.sine(4, 30_000.0, [70.0] * 4)
    device = SimulatedNeuralDevice(
        generator,
        samples_per_frame=30,
        sample_dtype="int16",
        physical_unit="dimensionless",
        paced=False,
    )

    signal = device.schema.signals[0]
    assert signal.dtype == streaming.SignalDType.INT16
    # An int16 device ships half the bytes of the float64 default for the same
    # geometry, which is what frame-pool and spool sizing must follow.
    assert signal.max_block_bytes == 4 * 30 * 2
    assert signal.physical_unit == streaming.PhysicalUnit.DIMENSIONLESS


@pytest.mark.parametrize("dtype", ["float32", "int32", "INT8", 16])
def test_unsupported_sample_dtype_is_rejected(dtype: object) -> None:
    with pytest.raises(ValidationError, match="sample_dtype must be"):
        SimulatedNeuralDevice(
            SignalGenerator.zeros(1, 1_000.0),
            samples_per_frame=1,
            sample_dtype=dtype,  # type: ignore[arg-type]
        )


def test_native_source_runs_without_python_callback_and_resets() -> None:
    generator = SignalGenerator.noise(2, 1_000.0, seed=42)
    device = _finite_simulated_neural_device(
        generator,
        samples_per_frame=4,
        total_sample_count=12,
        paced=False,
    )
    processor = streaming.IdentityNativeProcessor()
    consumer = streaming.CountingNativeConsumer()
    runner = streaming.StreamRunner(
        device.schema,
        _config(64),
        device.source,
        processor,
        consumer,
        profile=streaming.ExecutionProfile.RESEARCH,
    )
    assert type(device.source).__module__ == "neurale._native.devices.simulation"
    assert type(device.source).__name__ == "SimulatedNeuralSource"

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert consumer.frame_count == 3
    assert consumer.last_sequence == 2
    assert device.frames_emitted == 3
    assert device.samples_emitted == 12
    assert runner.outstanding_frames == 0
    assert runner.outstanding_discontinuities == 0

    assert runner.reset() == streaming.StreamStatus.OK
    assert device.frames_emitted == 0
    assert device.samples_emitted == 0
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert consumer.frame_count == 3
    assert device.frames_emitted == 3


def test_cancel_reset_and_close_are_distinct() -> None:
    device = SimulatedNeuralDevice(
        SignalGenerator.zeros(1, 1_000.0),
        samples_per_frame=1,
        paced=False,
    )

    assert not device.closed
    device.cancel()
    device.cancel()
    device.reset()
    device.reset()
    assert not device.closed

    device.close()
    device.close()
    device.cancel()
    assert device.closed
    assert device.source.closed
    with pytest.raises(StreamStateError, match="cannot be reset"):
        device.reset()


def test_reset_is_rejected_during_read() -> None:
    """A reset concurrent with an acquisition read is refused, not raced.

    ``reset()`` rewrites the sample offset, device tick, pacing epoch, and
    schedule cursor that the acquisition thread is using inside ``read``. The
    public facade can be called from any thread, so the refusal has to come
    from the native source rather than from a convention nobody enforces.
    """
    clock = ManualHostClock()
    device = SimulatedNeuralDevice(
        SignalGenerator.zeros(1, 1_000.0),
        samples_per_frame=1,
        paced=True,
        timing=SimulationTimingConfig(clock=clock),
    )
    runner = streaming.StreamRunner(
        device.schema,
        _config(8),
        device.source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        profile=streaming.ExecutionProfile.RESEARCH,
    )
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    try:
        # The clock never advances, so the acquisition thread parks inside the
        # paced read and keeps the gate for as long as this test needs it.
        deadline = time.monotonic() + 5.0
        while device.frames_emitted == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert device.frames_emitted == 1, "the source must be parked on its second frame"
        with pytest.raises(StreamStateError, match="cannot be reset"):
            device.reset()
    finally:
        runner.stop()
        runner.join()

    # Once the read has returned, the same call is accepted.
    device.reset()
    assert device.frames_emitted == 0


def test_native_paced_runner_releases_gil() -> None:
    import subprocess
    import sys

    script = r"""
import threading
import time

import neurale.streaming as streaming
from neurale.devices.simulation import (
    ManualHostClock,
    SimulationTimingConfig,
    _finite_simulated_neural_device,
)
from neurale.signal.simulation import SignalGenerator

config = streaming.RealtimeConfig()

clock = ManualHostClock()
device = _finite_simulated_neural_device(
    SignalGenerator.zeros(1, 1_000.0),
    samples_per_frame=1,
    total_sample_count=2,
    paced=True,
    timing=SimulationTimingConfig(clock=clock),
)
consumer = streaming.CountingNativeConsumer()
runner = streaming.StreamRunner(
    device.schema,
    config,
    device.source,
    streaming.IdentityNativeProcessor(),
    consumer,
    profile=streaming.ExecutionProfile.RESEARCH,
)
assert runner.prepare() == streaming.StreamStatus.OK
assert runner.arm() == streaming.StreamStatus.OK

def advance_clock():
    time.sleep(0.05)
    clock.advance(1_000_000)

control = threading.Thread(target=advance_clock)
control.start()
assert runner.run() == streaming.StreamStatus.OK
control.join()
assert consumer.frame_count == 2
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"generator": object()}, "generator"),
        ({"samples_per_frame": 0}, "samples_per_frame"),
        ({"channel_names": ["only-one"]}, "channel_names"),
        ({"channel_names": ["same", "same"]}, "unique"),
        ({"channel_names": "ab"}, "tuple or list"),
        ({"channel_impedances_ohm": [1_000]}, "one value per generator channel"),
        ({"channel_impedances_ohm": [1_000, -1]}, "channel_impedances_ohm"),
        ({"channel_impedances_ohm": "ab"}, "tuple or list"),
        ({"physical_unit": "tesla"}, "physical_unit"),
        ({"paced": 1}, "bool"),
        ({"timing": object()}, "SimulationTimingConfig"),
        ({"faults": object()}, "SimulationFaultPlan"),
    ],
)
def test_invalid_device_configuration_is_rejected(updates: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "generator": SignalGenerator.zeros(2, 1_000.0),
        "samples_per_frame": 4,
        "paced": False,
    }
    arguments.update(updates)
    with pytest.raises(ValidationError, match=message):
        SimulatedNeuralDevice(**arguments)


def test_finite_supplied_samples_bound_total_count() -> None:
    import numpy as np

    generator = SignalGenerator.samples(1, 100.0, np.arange(5.0)[:, None])
    device = SimulatedNeuralDevice(
        generator,
        samples_per_frame=2,
        paced=False,
        timing=SimulationTimingConfig(initial_sample_idx=2),
    )
    assert device.schema.signals[0].nominal_block_samples == 2

    with pytest.raises(ValidationError, match="exceeds supplied samples"):
        _finite_simulated_neural_device(
            generator,
            samples_per_frame=2,
            total_sample_count=4,
            paced=False,
            timing=SimulationTimingConfig(initial_sample_idx=2),
        )


def test_typed_schedule_is_immutable_and_reaches_streaming() -> None:
    generator = SignalGenerator.noise(1, 1_000.0, seed=17)
    events = (
        TransientWouldBlock(0),
        KnownSampleLoss(1, 2),
        DeviceTickJump(2, 5),
        DeviceClockRestart(3, 100),
    )
    device = _finite_simulated_neural_device(
        generator,
        samples_per_frame=4,
        total_sample_count=18,
        paced=True,
        faults=SimulationFaultPlan(events),
    )
    assert device.events == events

    processor = streaming.IdentityNativeProcessor()
    consumer = streaming.CountingNativeConsumer()
    runner_config = _config(32)
    runner_config.pool_capacity.ingress_capacity = 16
    runner_config.pool_capacity.critical_edge_capacity = 17
    runner_config.discontinuity_capacity = 8
    runner = streaming.StreamRunner(
        device.schema,
        runner_config,
        device.source,
        processor,
        consumer,
        profile=streaming.ExecutionProfile.RESEARCH,
    )
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert consumer.frame_count == 4
    assert consumer.discontinuity_count == 3
    assert processor.discontinuity_count == 3
    assert device.samples_emitted == 16
    assert runner.outstanding_frames == 0
    assert runner.outstanding_discontinuities == 0

    assert runner.reset() == streaming.StreamStatus.OK


def test_manual_host_clock_and_device_clock_mapping_are_distinct() -> None:
    clock = ManualHostClock(1_000)
    assert clock.now_ns == 1_000
    clock.advance(250)
    assert clock.now_ns == 1_250
    clock.set(5_000)
    assert clock.now_ns == 5_000
    with pytest.raises(ValidationError, match="backwards"):
        clock.set(4_999)

    device = SimulatedNeuralDevice(
        SignalGenerator.zeros(1, 1_000.0),
        samples_per_frame=1,
        paced=False,
        timing=SimulationTimingConfig(
            initial_device_tick=40,
            clock_offset_ns=-100,
            clock_drift_ppm=1_000,
            clock_sync_uncertainty_ns=25,
            clock=clock,
        ),
    )
    assert device.schema.signals[0].device_tick_tracking == (
        streaming.DeviceTickTracking.SAMPLE_COUNTER
    )


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([KnownSampleLoss(2, 1), KnownSampleLoss(1, 1)], "ordered"),
        ([Disconnect(1), TransientWouldBlock(2)], "terminal"),
        ([SourceFault(0), DeviceClockRestart(0)], "terminal"),
        ([object()], "typed acquisition"),
    ],
)
def test_invalid_fault_schedules_are_rejected(events: list[object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SimulatedNeuralDevice(
            SignalGenerator.zeros(1, 1_000.0),
            samples_per_frame=4,
            paced=False,
            faults=SimulationFaultPlan(events),  # type: ignore[arg-type]
        )


def test_invalid_clock_and_event_parameters_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duration_ns"):
        BoundedStall(0, 0)
    with pytest.raises(ValidationError, match="nonzero"):
        DeviceTickJump(0, 0)
    with pytest.raises(ValidationError, match="requires sample-counter"):
        SimulationTimingConfig(
            device_ticks=False,
            clock_drift_ppm=1,
        )
    with pytest.raises(ValidationError, match="requires sample-counter"):
        SimulationTimingConfig(
            device_ticks=False,
            clock_sync_uncertainty_ns=1,
        )
