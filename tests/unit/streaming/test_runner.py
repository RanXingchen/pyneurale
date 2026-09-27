#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

import neurale.streaming as streaming


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


def _frame(sequence: int):
    block = streaming.PythonObserverSignalBlock(
        sequence * 4,
        sequence * 4,
        0,
        32,
        1,
        4,
    )
    return streaming.PythonObserverFrame(
        1,
        sequence,
        sequence,
        None,
        None,
        11,
        0,
        [block],
        np.arange(32, dtype=np.uint8),
    )


class _Source:
    def __init__(self, count: int) -> None:
        self._count = count
        self._sequence = 0

    def read(self):
        if self._sequence == self._count:
            return None
        frame = _frame(self._sequence)
        self._sequence += 1
        return frame


class _Processor:
    def process(self, frame):
        return frame


class _Sink:
    def __init__(self) -> None:
        self.frames = []

    def write(self, frame) -> None:
        self.frames.append(frame)


def _adapted_pipeline(profile: streaming.ExecutionProfile | str):
    source = streaming.PythonSourceAdapter(_Source(2))
    processor = streaming.PythonProcessorAdapter(_Processor())
    sink_component = _Sink()
    sink = streaming.PythonSinkAdapter(sink_component)
    runner = streaming.StreamRunner(
        _schema(),
        streaming.RealtimeConfig(),
        source,
        processor,
        sink,
        profile=profile,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    return runner, source, processor, sink, sink_component


def test_realtime_profile_rejects_python_critical_adapters_at_prepare() -> None:
    runner, *_ = _adapted_pipeline(streaming.ExecutionProfile.REALTIME)

    with pytest.raises(ValueError, match="requires native"):
        runner.prepare()

    assert runner.state == streaming.RuntimeState.CREATED


def test_realtime_profile_requires_explicit_safety_and_selects_strict_platform() -> None:
    schema = _schema()
    source = streaming.SyntheticNativeSource(schema, 1)
    processor = streaming.IdentityNativeProcessor()
    actuator = streaming.CountingNativeConsumer()

    without_safety = streaming.StreamRunner(
        schema, streaming.RealtimeConfig(), source, processor, actuator
    )
    with pytest.raises(ValueError, match="safety controller"):
        without_safety.prepare()

    with_safety = streaming.StreamRunner(
        schema,
        streaming.RealtimeConfig(),
        source,
        processor,
        actuator,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    assert with_safety.prepare() == streaming.StreamStatus.OK


@pytest.mark.parametrize(
    "profile", [streaming.ExecutionProfile.RESEARCH, streaming.ExecutionProfile.OFFLINE]
)
def test_non_realtime_profiles_run_explicit_python_adapters(profile) -> None:
    runner, source, processor, sink, sink_component = _adapted_pipeline(profile)

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert runner.state == streaming.RuntimeState.STOPPED
    assert runner.stats.frames_consumed == 2
    assert [frame.sequence for frame in sink_component.frames] == [0, 1]
    assert source.callback_errors == 0
    assert processor.callback_errors == 0
    assert sink.callback_errors == 0


def test_python_adapter_fault_is_reported_by_native_fault_state_machine() -> None:
    class FaultingSource:
        def read(self):
            raise RuntimeError("source failed")

    runner = streaming.StreamRunner(
        _schema(),
        streaming.RealtimeConfig(),
        streaming.PythonSourceAdapter(FaultingSource()),
        streaming.PythonProcessorAdapter(_Processor()),
        streaming.PythonSinkAdapter(_Sink()),
        profile=streaming.ExecutionProfile.RESEARCH,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.SOURCE_FAILURE
    assert runner.state == streaming.RuntimeState.FAILED
    assert runner.primary_fault.code == streaming.FaultCode.SOURCE_READ
    assert runner.primary_fault.stage == streaming.FaultStage.SOURCE
    assert runner.outstanding_frames == 0


def test_invalid_processor_output_is_rejected_without_leak() -> None:
    class MalformedProcessor:
        def process(self, frame):
            return streaming.PythonObserverFrame(
                frame.session_id,
                frame.sequence,
                frame.host_received_ns,
                frame.source_tick,
                frame.valid_until_ns,
                frame.schema_id,
                frame.source_clock_domain,
                frame.blocks,
                np.zeros((2, 16), dtype=np.uint8),
            )

    runner = streaming.StreamRunner(
        _schema(),
        streaming.RealtimeConfig(),
        streaming.PythonSourceAdapter(_Source(1)),
        streaming.PythonProcessorAdapter(MalformedProcessor()),
        streaming.PythonSinkAdapter(_Sink()),
        profile=streaming.ExecutionProfile.RESEARCH,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.INVALID_FRAME
    assert runner.primary_fault.code == streaming.FaultCode.OUTPUT_VALIDATION
    assert runner.primary_fault.stage == streaming.FaultStage.OUTPUT
    assert runner.outstanding_frames == 0


def test_processor_iteration_exception_reclaims_frames() -> None:
    class FaultingProcessor:
        def process(self, frame):
            yield frame
            raise RuntimeError("processor iteration failed")

    processor = streaming.PythonProcessorAdapter(FaultingProcessor())
    runner = streaming.StreamRunner(
        _schema(),
        streaming.RealtimeConfig(),
        streaming.PythonSourceAdapter(_Source(1)),
        processor,
        streaming.PythonSinkAdapter(_Sink()),
        profile=streaming.ExecutionProfile.RESEARCH,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.PROCESSOR_FAILURE
    assert runner.state == streaming.RuntimeState.FAILED
    assert runner.primary_fault.code == streaming.FaultCode.PROCESSOR_PROCESS
    assert runner.primary_fault.stage == streaming.FaultStage.PROCESSOR
    assert processor.callback_errors == 1
    assert runner.outstanding_frames == 0


def test_context_manager_uses_native_lifecycle() -> None:
    runner = streaming.StreamRunner(
        _schema(),
        streaming.RealtimeConfig(),
        streaming.SyntheticNativeSource(_schema(), 2),
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    with runner as active:
        assert active is runner
        assert runner.state in {
            streaming.RuntimeState.RUNNING,
            streaming.RuntimeState.STOPPING,
            streaming.RuntimeState.STOPPED,
        }

    assert runner.state == streaming.RuntimeState.STOPPED


class _ShutdownStatusRunner:
    def __init__(self, status) -> None:
        self.status = status

    def stop(self):
        return self.status

    def abort(self):
        return self.status


def _runner_with_shutdown_status(status):
    runner = object.__new__(streaming.StreamRunner)
    runner._native = streaming
    runner._runner = _ShutdownStatusRunner(status)
    return runner


def test_context_manager_raises_when_clean_stop_fails() -> None:
    runner = _runner_with_shutdown_status(streaming.StreamStatus.DEADLINE_EXCEEDED)

    with pytest.raises(RuntimeError, match="failed to stop"):
        runner.__exit__(None, None, None)


def test_context_preserves_body_error_and_notes_abort() -> None:
    runner = _runner_with_shutdown_status(streaming.StreamStatus.DEADLINE_EXCEEDED)
    error = ValueError("body failed")

    assert runner.__exit__(ValueError, error, None) is False
    assert "failed to abort" in error.__notes__[-1]
