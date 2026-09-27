#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The provisional native recorder surface.

What this module pins is the *boundary*, not the recorder core: the core's own
behaviour is covered by the C++ suites, and repeating it here would only prove
that pybind11 copies integers. What is new at this boundary, and therefore what
is tested, is the orchestration -- that a compiled ``RecordingPlan`` reaches the
native core intact, that the runtime drives the lifecycle in the order contract
section 3.1 requires, that every native failure surfaces as an exception instead
of a fallback, and that the status a caller reads is the native one rather than
a translation into the Python recorder's vocabulary.

Two paths appear below and they are not interchangeable:

* **Runtime-driven** -- the real one. ``prepare()``, ``attach()``, then
  ``runner.prepare()/arm()/start()/join()`` move the recorder through
  ``ready``, ``recording``, ``draining`` and ``stopped``.
* **Standalone** -- ``_recorder.standalone_*`` on the private binding, used only
  where a runtime would make the test a race. The control plane is the case:
  with a runtime driving a synthetic source, the window between ``start()`` and
  the source running out is scheduling, and a test that submits into it is
  testing the scheduler. The facade has no standalone path on purpose.
"""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import neurale.streaming as streaming
from neurale.recording import (
    RecorderConfig,
    RecorderLimits,
    StreamMetadata,
    StreamRecording,
    StreamStorageConfig,
    StreamTimingConfig,
    StreamTimingMode,
    compile_recording_plan,
)
from neurale.recording._errors import (
    RecorderConfigError,
    RecorderRuntimeShutdownError,
    RecorderStateError,
)
from neurale.recording._native_recorder import (
    CONTROL_KINDS,
    HOST_MONOTONIC_CLOCK_DOMAIN,
    SPOOL_RETENTION_POLICIES,
    NativeRecorderOptions,
    NativeSessionRecorder,
)
from neurale.streaming import _native as streaming_native

from .conftest import native_schema as multi_schema
from .conftest import recorder_config as multi_recorder_config

SIGNAL_ID = 1
SCHEMA_ID = 11
CLOCK_DOMAIN = 7
CHANNELS = 4
BLOCK_SAMPLES = 8

#: Signals, blocks per frame, and payload bytes of the shared three-stream
#: fixture in ``conftest`` -- one neural (int16, sample-major), one behavioral
#: (float32, channel-major), one feature (float64, with a descriptor). Stated
#: here rather than recomputed so a test asserting "every stream was recorded"
#: is asserting against the fixture, not against its own arithmetic.
MULTI_BLOCKS_PER_FRAME = 3
#: What the synthetic source actually emits per frame: nominal block samples.
MULTI_FRAME_PAYLOAD_BYTES = 4 * 8 * 2 + 2 * 2 * 4 + 3 * 1 * 8
#: What the runtime must be able to hold: the *maximum*-size frame the schema
#: allows. The two differ because every signal's maximum block is larger than
#: its nominal one, and the buffer is sized for the worst case.
MULTI_MAX_FRAME_PAYLOAD_BYTES = 4 * 16 * 2 + 2 * 4 * 4 + 3 * 2 * 8


# --- fixtures ---------------------------------------------------------------


def _schema() -> Any:
    signal = streaming.SignalSchema(
        SIGNAL_ID,
        streaming.SignalDType.INT16,
        CHANNELS,
        BLOCK_SAMPLES,
        BLOCK_SAMPLES,
        streaming.RationalRate(1_000, 1),
        CLOCK_DOMAIN,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
    )
    return streaming.StreamSchema(SCHEMA_ID, [signal])


def _realtime_config(frames: int, *, blocks: int = 1, payload_bytes: int | None = None) -> Any:
    # Generous on purpose. A critical edge is lossless until fault, so an edge
    # that had to drop would not be a statistic here -- it would fault the
    # recorder and abort the runtime, and this fixture is not the place to
    # provoke that.
    capacity = max(8, frames * 2)
    budget = streaming_native.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = capacity
    budget.processor_owned = 2
    budget.critical_edge_capacity = capacity + 1
    budget.actuator_owned = 1
    budget.observer_edge_capacity = capacity

    config = streaming_native.RealtimeConfig()
    config.pool_capacity = budget
    config.buffer_size = BLOCK_SAMPLES * CHANNELS * 2 if payload_bytes is None else payload_bytes
    config.max_signal_blocks = blocks
    config.discontinuity_capacity = 4
    config.gaps_per_discontinuity = 2
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 4
    config.platform.mode = streaming_native.RealtimeConfigMode.STRICT
    config.validate()
    return config


def _recorder_config(path: Path, **overrides: Any) -> RecorderConfig:
    limit_values: dict[str, Any] = {"checkpoint_interval": 0}
    for name in (
        "frame_queue_capacity",
        "control_queue_capacity",
        "spool_capacity_bytes",
        "checkpoint_interval",
        "max_control_records",
    ):
        if name in overrides:
            limit_values[name] = overrides.pop(name)
    if "control_chunk_length" in overrides:
        limit_values["max_control_records"] = overrides.pop("control_chunk_length") * 1024
    arguments: dict[str, Any] = {
        "path": path,
        "streams": [
            StreamRecording(
                metadata=StreamMetadata(unit="V"),
                storage=StreamStorageConfig(
                    capacity=8192, chunk_length=8, block_index_chunk_length=8
                ),
                timing=StreamTimingConfig(mode=StreamTimingMode.REGULAR),
            )
        ],
        "limits": RecorderLimits(**limit_values),
    }
    arguments.update(overrides)
    return RecorderConfig(**arguments)


class _NativePipeline:
    """A runtime with a native critical recorder attached, in contract order."""

    #: Overrides that belong to the plan rather than to the native-only
    #: options. See the note on the ``standalone_recorder`` fixture.
    CONFIG_FIELDS = frozenset({"control_queue_capacity", "frame_queue_capacity"})

    def __init__(
        self, path: Path, frames: int, *, multi_stream: bool = False, **overrides: Any
    ) -> None:
        config_overrides = {k: v for k, v in overrides.items() if k in self.CONFIG_FIELDS}
        option_overrides = {k: v for k, v in overrides.items() if k not in self.CONFIG_FIELDS}
        if multi_stream:
            self.schema = multi_schema()
            self.config = _realtime_config(
                frames,
                blocks=MULTI_BLOCKS_PER_FRAME,
                payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES,
            )
            recorder_config = multi_recorder_config(path, checkpoint_interval=0, **config_overrides)
        else:
            self.schema = _schema()
            self.config = _realtime_config(frames)
            recorder_config = _recorder_config(path, **config_overrides)
        self.source = streaming.SyntheticNativeSource(self.schema, frames)
        self.actuator = streaming.CountingNativeConsumer()
        self.runner = streaming.StreamRunner(
            self.schema,
            self.config,
            self.source,
            streaming.IdentityNativeProcessor(),
            self.actuator,
            safety_controller=streaming.RecordingNativeSafetyController(),
        )
        options = NativeRecorderOptions(
            spool="memory", edge_capacity=max(8, frames * 2), **option_overrides
        )
        self.recorder = NativeSessionRecorder.create(recorder_config, self.schema, options=options)
        self.recorder.prepare()
        self.recorder.attach(self.runner)

    def start(self) -> Any:
        """Reach ``recording`` and return, leaving the runtime live."""
        assert self.runner.prepare() == streaming.StreamStatus.OK
        assert self.runner.arm() == streaming.StreamStatus.OK
        return self.runner.start()

    def run(self) -> Any:
        assert self.start() == streaming.StreamStatus.OK
        return self.runner.join()

    def close(self) -> Any:
        return self.recorder.close()


@pytest.fixture
def pipeline_factory(tmp_path: Path):
    built: list[_NativePipeline] = []

    def build(frames: int, **kwargs: Any) -> _NativePipeline:
        pipeline = _NativePipeline(tmp_path / f"session-{len(built)}.nrf", frames, **kwargs)
        built.append(pipeline)
        return pipeline

    try:
        yield build
    finally:
        # Closing is the fixture's job: a test that ends early on a failed
        # assertion would otherwise leave a recorder worker thread and a spool
        # behind it, and `close()` is idempotent so doing it twice is free.
        #
        # The abort comes first because `close()` refuses a live runtime by
        # design, and a teardown that raised there would replace the failing
        # test's error with its own. Aborting says how the session ended, which
        # is exactly what an abandoned test did not say.
        for pipeline in built:
            if pipeline.recorder.runtime_is_live():
                pipeline.recorder.abort("test-teardown")
            pipeline.close()


@pytest.fixture
def standalone_recorder(tmp_path: Path):
    """A recorder driven with no runtime, for the control-plane tests.

    See the module docstring for why the control plane cannot be tested through
    a runtime without testing the scheduler instead.
    """
    created: list[NativeSessionRecorder] = []

    # `RecorderConfig` owns the queue capacities (they are part of the plan) and
    # `NativeRecorderOptions` owns everything native-only, so a test that wants
    # a shallow control queue has to reach the first and a test that wants a
    # sleepy worker the second. Split by name rather than making the caller say
    # which, because the split is an implementation fact and not a choice.
    config_fields = {"control_queue_capacity", "frame_queue_capacity", "checkpoint_interval"}

    def build(**overrides: Any) -> NativeSessionRecorder:
        config_overrides = {k: v for k, v in overrides.items() if k in config_fields}
        options = NativeRecorderOptions(
            spool="memory", **{k: v for k, v in overrides.items() if k not in config_fields}
        )
        recorder = NativeSessionRecorder.create(
            _recorder_config(tmp_path / f"standalone-{len(created)}.nrf", **config_overrides),
            _schema(),
            options=options,
        )
        created.append(recorder)
        recorder.prepare()
        native = recorder._native
        assert (
            recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
        )
        assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
        return recorder

    try:
        yield build
    finally:
        for recorder in created:
            # A test that wedged a stuck runtime in front of the recorder leaves
            # `close()` refusing it by design. Dropping the double is teardown's
            # job: raising here would replace the failing test's error with one
            # about the fixture.
            recorder._runner = None
            recorder.close()


# --- lazy loading -----------------------------------------------------------


def test_package_import_does_not_load_native_extension() -> None:
    """``import neurale.recording`` must keep working with no native extension.

    The Python recorder does not need one, so the native bindings are imported
    inside the first call that needs a native object, not at module scope. This
    is checked by construction rather than by mocking the loader: the module
    that pulls the extension in is a distinct one, and the package must not name
    it at import time.
    """
    import neurale.recording as recording

    source = Path(recording.__file__).read_text(encoding="utf-8")
    assert "from ._native import" not in source
    assert "import _native" not in source
    # The facade names it, but only inside function bodies.
    facade = Path(recording._native_recorder.__file__).read_text(encoding="utf-8")
    for line in facade.splitlines():
        if line.startswith(("import ", "from ")) and "_native" in line:
            raise AssertionError(f"the facade imports the extension at module scope: {line!r}")


def test_package_import_leaves_extension_unloaded() -> None:
    """The same claim, checked by running it rather than by reading the source.

    It has to be a subprocess: by the time this module runs, the suite has
    already imported ``neurale.streaming``, so ``sys.modules`` in *this*
    interpreter says nothing about what ``neurale.recording`` pulled in.
    """
    program = (
        "import sys; import neurale.recording; "
        "print('LOADED' if 'neurale._native' in sys.modules else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "CLEAN"


def test_stable_recorder_exposes_no_backend_selector() -> None:
    """The stable recorder keeps the native engine and removes engine selection."""
    from neurale.recording import SessionRecorder

    for forbidden in ("native", "backend", "engine", "use_native"):
        assert not hasattr(SessionRecorder, forbidden)


# --- the plan reaches the core intact ---------------------------------------


def test_native_core_prepares_against_compiled_plan(tmp_path: Path) -> None:
    """One prepared statement, shared by both engines.

    The plan compiled here is the same object the Python recorder would compile
    from the same config and schema, which is what makes the parity module a
    comparison rather than a coincidence.
    """
    config = _recorder_config(tmp_path / "session.nrf")
    schema = _schema()
    recorder = NativeSessionRecorder.create(config, schema)
    try:
        expected = compile_recording_plan(config, schema)
        assert recorder.plan.fingerprint == expected.fingerprint
        assert recorder.plan.recorded_signal_ids == (SIGNAL_ID,)
        assert recorder.plan.coverage == "full"
    finally:
        recorder.close()


def test_recorder_generates_valid_session_identity(tmp_path: Path) -> None:
    recorder = NativeSessionRecorder.create(_recorder_config(tmp_path / "session.nrf"), _schema())
    try:
        uuid.UUID(recorder.plan.session.session_id)
        assert recorder.plan.session.created_at.endswith("Z")
    finally:
        recorder.close()


# --- lifecycle --------------------------------------------------------------


def test_runtime_drives_whole_lifecycle(pipeline_factory) -> None:
    """created -> prepared -> ready -> recording -> draining -> stopped -> closed.

    Each transition is asserted where the contract puts it: the readiness gate
    inside ``arm()``, acceptance inside ``start()``, and the spool session-end
    at the end of the drain. Nothing here calls a lifecycle method on the
    recorder except ``prepare()`` and ``close()``.
    """
    pipeline = pipeline_factory(6)
    native = pipeline.recorder._native
    states = native.RecorderLifecycleState

    assert pipeline.recorder.state == states.PREPARED
    assert pipeline.runner.prepare() == streaming.StreamStatus.OK
    assert pipeline.recorder.state == states.PREPARED

    assert pipeline.runner.arm() == streaming.StreamStatus.OK
    assert pipeline.recorder.state == states.READY

    assert pipeline.runner.start() == streaming.StreamStatus.OK
    assert pipeline.runner.join() == streaming.StreamStatus.OK
    assert pipeline.recorder.state == states.STOPPED

    status = pipeline.recorder.close()
    assert pipeline.recorder.state == states.CLOSED
    assert status.state == states.CLOSED


def test_acceptance_ladder_stages_are_reported_separately(pipeline_factory) -> None:
    """A native recorder has all five stages and must report all five.

    A single "recorded" number spanning stages is a contract violation
    (section 1), so the assertion is per stage, per plane -- and
    ``nrf_committed`` is asserted to be zero rather than omitted, because
    finalization is a separate offline step and a fabricated success would be
    worse than the gap.
    """
    pipeline = pipeline_factory(6)
    assert pipeline.run() == streaming.StreamStatus.OK
    status = pipeline.recorder.status

    assert status.runtime_accepted == 6
    assert status.recorder_accepted == 6
    assert status.spool_committed == 6
    assert status.nrf_committed == 0

    assert status.rejected_before_runtime_acceptance == 0
    assert status.failed_between_runtime_and_recorder == 0
    assert status.lost_between_recorder_and_spool == 0

    # Diagnostics are named distinctly and never substituted into the ladder:
    # one frame carrying one block is one item, not one plus one.
    assert status.frames_accepted == 6
    assert status.signal_blocks_recorded == 6
    assert status.discontinuities_accepted == 0


def test_closed_recorder_reports_no_artifact(pipeline_factory) -> None:
    """``closed`` is a statement about the object (contract section 3.1).

    Whether a session exists, is sealed, or is complete is reported by separate
    fields, and the honest answer here is a spool that ended cleanly and still
    needs finalizing.
    """
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK
    status = pipeline.recorder.close()

    assert status.session_created is True
    assert status.spool_ended_cleanly is True
    # Finalization is a separate offline step, so the NRF session does not
    # exist yet and nothing pretends otherwise.
    assert status.sealed is False
    assert status.finalization_required is True
    assert status.recovery_required is False


def test_completeness_is_absent_before_sealing(
    pipeline_factory,
) -> None:
    """Contract section 3.2's fourth case, and the reason ``complete`` is tri-state.

    Completeness is a property of a *sealed* session. ``False`` here would say
    the session was judged and found wanting; the truth is that there is no
    sealed session to judge, and the contract is explicit that the two are
    different answers and that ``None`` must not be reported as
    ``unverified_legacy``.
    """
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK
    status = pipeline.recorder.close()

    assert status.complete is None
    assert status.completeness_verdict is None
    # Paired exactly as the contract's table pairs them.
    assert status.accounting_verified is False
    # What the artifact *says*, which is nothing: no termination record exists.
    assert status.termination_kind is None
    # The finalizer surface is present and honestly empty, not omitted.
    assert status.finalization_attempts == []
    assert status.finalization_status == pipeline.recorder._native.FinalizationStatus.NOT_STARTED


def test_termination_kind_differs_from_effective_outcome(
    standalone_recorder,
) -> None:
    """ "What we concluded" and "what the artifact says" are separate fields.

    An aborted session concludes ``aborted`` immediately, and still has no
    termination kind, because nothing has written a termination record. Folding
    the two together would make an unfinalized session indistinguishable from a
    sealed one.
    """
    recorder = standalone_recorder()
    native = recorder._native

    status = recorder.abort("operator-abort")
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.ABORTED
    assert status.capture_outcome == native.CaptureOutcome.ABORTED
    assert status.termination_kind is None


def test_capture_outcome_and_recorder_state_are_distinct(
    pipeline_factory,
) -> None:
    """``faulted`` belongs to the session, ``failed`` to the recorder.

    A clean run ends with the recorder in a terminal object state and the
    session's capture outcome normal, and the two are read from different
    fields -- the state alone cannot carry both.
    """
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK
    status = pipeline.recorder.status
    native = pipeline.recorder._native

    assert status.state == native.RecorderLifecycleState.STOPPED
    # One answer, not two fields to combine: `None` would be section 3.2's
    # `unknown`, and this session-end record froze a real outcome.
    assert status.capture_outcome == native.CaptureOutcome.NORMAL
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.NORMAL
    assert status.requested_terminal_intent == native.RequestedTerminalIntent.NORMAL
    assert status.primary_fault.present is False


def test_close_is_idempotent(pipeline_factory) -> None:
    """Repeated ``close()`` returns the same verdict and moves nothing."""
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK

    first = pipeline.recorder.close()
    second = pipeline.recorder.close()
    third = pipeline.recorder.close()

    for status in (second, third):
        assert status.state == first.state
        assert status.session_created == first.session_created
        assert status.spool_ended_cleanly == first.spool_ended_cleanly
        assert status.sealed == first.sealed
        assert status.spool_committed == first.spool_committed
        assert status.effective_session_outcome == first.effective_session_outcome


def test_stop_after_runtime_stop_returns_latched_shutdown(
    pipeline_factory,
) -> None:
    """The first exit from ``recording`` latches the terminal intent.

    A later ``stop()`` only waits for the shutdown already in progress and
    returns its status; it does not replace what was decided.
    """
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK
    native = pipeline.recorder._native

    before = pipeline.recorder.status
    after = pipeline.recorder.stop("second-stop")
    assert after.requested_terminal_intent == native.RequestedTerminalIntent.NORMAL
    assert after.spool_committed == before.spool_committed
    assert after.capture_outcome == before.capture_outcome


# --- ending a session with the runtime still live ---------------------------
#
# The three tests below are the ones that would pass on a facade that reached
# past the runtime, which is why they drive a *running* pipeline rather than a
# joined one: the ordering only matters while there is still a producer.


def test_stop_stops_runtime_first(pipeline_factory) -> None:
    """Runtime first, recorder second -- the order contract section 3.1 fixes.

    Telling the recorder to stop accepting while the runtime is still producing
    does not end the session, it damages it: the next frame meets a recorder
    that is no longer accepting, which on a critical edge is a fault. A caller
    who asked for a graceful stop must not get a faulted session.
    """
    pipeline = pipeline_factory(4096)
    assert pipeline.start() == streaming.StreamStatus.OK
    assert pipeline.recorder.runtime_is_live() is True
    native = pipeline.recorder._native

    status = pipeline.recorder.stop("operator-stop")

    assert pipeline.recorder.runtime_is_live() is False
    assert pipeline.runner.state == streaming.RuntimeState.STOPPED
    assert status.state == native.RecorderLifecycleState.STOPPED
    assert status.requested_terminal_intent == native.RequestedTerminalIntent.NORMAL
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.NORMAL
    assert status.capture_outcome == native.CaptureOutcome.NORMAL
    # The whole point: no fault was manufactured by the shutdown itself.
    assert status.primary_fault.present is False
    assert status.spool_ended_cleanly is True
    # And nothing the runtime accepted was left behind.
    assert status.spool_committed == status.runtime_accepted


def test_abort_aborts_runtime_first(pipeline_factory) -> None:
    """An abort must actually stop the producer, not just close the recorder's door.

    Without aborting the runtime, the runtime keeps producing until some later
    critical-edge failure faults it -- turning an operator abort into a fault,
    or, on a short source, into a run that finishes normally after the caller
    asked it to stop.
    """
    pipeline = pipeline_factory(4096)
    assert pipeline.start() == streaming.StreamStatus.OK
    native = pipeline.recorder._native

    status = pipeline.recorder.abort("operator-abort")

    assert pipeline.recorder.runtime_is_live() is False
    assert status.requested_terminal_intent == native.RequestedTerminalIntent.ABORTED
    assert status.capture_outcome == native.CaptureOutcome.ABORTED
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.ABORTED
    # An abort is not a fault: it stops recording new data rather than
    # destroying data already handed over.
    assert status.primary_fault.present is False
    assert status.spool_ended_cleanly is True
    assert status.lost_between_recorder_and_spool == 0


def test_closing_live_session_is_rejected(
    pipeline_factory,
) -> None:
    """``close()`` releases the storage the critical edge is writing into.

    Refused instead of coordinated, because ending a session says *how* it
    ended, and that choice is latched into the terminal intent and cannot be
    revised. ``close()`` picking ``normal`` or ``aborted`` on the caller's
    behalf would decide something only the caller knows.
    """
    pipeline = pipeline_factory(4096)
    assert pipeline.start() == streaming.StreamStatus.OK

    with pytest.raises(RecorderStateError, match="attached runtime"):
        pipeline.recorder.close()

    # Refused, not half-done: the session is untouched. The recorder stays in a
    # live, non-terminal state -- ``recording`` while the synthetic source is
    # still producing, or ``draining`` if the source already ran to
    # end-of-stream on its own. The acquisition loop paces nothing, so 4096
    # frames can exhaust before this assertion runs; that transition is the
    # runtime's terminal notice driving the recorder's own drain, not anything
    # close() did. Either way close() neither closed the recorder nor stopped
    # the runtime it is bound to.
    assert pipeline.recorder.state in (
        pipeline.recorder._native.RecorderLifecycleState.RECORDING,
        pipeline.recorder._native.RecorderLifecycleState.DRAINING,
    )
    assert pipeline.recorder.runtime_is_live() is True
    # And the documented way out works.
    pipeline.recorder.abort("after-refusal")
    assert pipeline.recorder.close().state == (
        pipeline.recorder._native.RecorderLifecycleState.CLOSED
    )


def test_context_exit_on_exception_stops_runtime(pipeline_factory) -> None:
    """The context manager's exception path, with a producer still running.

    This is the case the earlier context-manager tests could not reach: one
    exits after ``join()`` and the other has no runtime at all, so neither would
    notice a facade that left the runtime running.
    """
    pipeline = pipeline_factory(4096)
    assert pipeline.start() == streaming.StreamStatus.OK
    native = pipeline.recorder._native

    with pytest.raises(ValueError, match="boom"):
        with pipeline.recorder:
            raise ValueError("boom")

    assert pipeline.recorder.runtime_is_live() is False
    assert pipeline.runner.state == streaming.RuntimeState.STOPPED
    status = pipeline.recorder.status
    assert status.state == native.RecorderLifecycleState.CLOSED
    assert status.requested_terminal_intent == native.RequestedTerminalIntent.ABORTED
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.ABORTED


# --- a runtime that will not quiesce ----------------------------------------
#
# Quiescing the runtime is a gate, not a courtesy call: `stop()`, `abort()` and
# `join()` may all answer `deadline_exceeded`, which means the bound elapsed and
# the workers may still be there. Ignoring that answer and stopping the recorder
# anyway is the ordering the contract forbids -- and it converts a runtime
# shutdown timeout into a critical-edge fault charged to the recording.


class _StubbornRunner:
    """A runtime that answers its shutdown without ever quiescing.

    A real runtime cannot be made to do this on demand -- refusing to stop is
    precisely the failure mode that has no API -- so the runner is the test
    double here, and it is the *only* one: the recorder underneath is the real
    native recorder, in ``recording``, and every assertion is about what
    happened to it.

    It is used with the standalone recorder rather than a live pipeline on
    purpose. The synthetic source runs to completion in milliseconds, so a
    runtime-driven version of these tests would race the source: the runtime
    would end on its own, drain the recorder legitimately, and the assertion
    that the recorder was untouched would be measuring the scheduler. The
    contract being tested belongs to the facade's orchestration, and the three
    tests above already pin the same ordering against a genuinely live runtime.
    """

    def __init__(self, state: Any, status: Any) -> None:
        self.state = state
        self._status = status
        self.calls: list[str] = []

    def stop(self) -> Any:
        self.calls.append("stop")
        return self._status

    def abort(self) -> Any:
        self.calls.append("abort")
        return self._status

    def join(self) -> Any:
        self.calls.append("join")
        return self._status


def _wedge_runtime(
    recorder: NativeSessionRecorder,
    status: Any,
    *,
    state: Any = None,
) -> _StubbornRunner:
    """Put a runner that never quiesces in front of *recorder*."""
    stubborn = _StubbornRunner(streaming.RuntimeState.RUNNING if state is None else state, status)
    recorder._runner = stubborn
    return stubborn


def _assert_recorder_untouched(recorder: NativeSessionRecorder) -> None:
    """The recorder never entered its own shutdown stage.

    Checked by what a caller can see rather than by spying on the call: a
    ``stop()`` or ``abort()`` that had run would have latched a terminal intent
    -- which is irreversible -- and left the recorder no longer accepting. Both
    are still true here, so nothing was decided.
    """
    native = recorder._native
    status = recorder.status
    assert status.state == native.RecorderLifecycleState.RECORDING
    assert status.capture_outcome is None
    assert status.spool_ended_cleanly is False
    # Still accepting, which is the observable form of "shutdown did not run".
    assert recorder.record_event("still-accepting") is True


def test_unstoppable_runtime_blocks_recorder_stop(
    standalone_recorder,
) -> None:
    """``stop()`` refuses rather than stopping the recorder under a live producer.

    ``deadline_exceeded`` is not a slower success: it means the bound elapsed
    and the workers may still be there. Continuing into the recorder's own
    shutdown would let a worker that is still running meet a recorder that no
    longer accepts, which on a critical edge is a fault -- so a runtime
    shutdown timeout would be charged to the recording as data loss.
    """
    recorder = standalone_recorder()
    stubborn = _wedge_runtime(recorder, streaming.StreamStatus.DEADLINE_EXCEEDED)

    with pytest.raises(RecorderRuntimeShutdownError, match="did not quiesce"):
        recorder.stop("operator-stop")

    # The runtime was asked, and asked in the right order, before being refused.
    assert stubborn.calls == ["stop", "join"]
    _assert_recorder_untouched(recorder)

    # Detaching the stuck runtime is enough to end the session: the refusal
    # keeps the recorder available rather than stranding it.
    recorder._runner = None
    assert recorder.stop("after-resolving").state == (
        recorder._native.RecorderLifecycleState.STOPPED
    )


def test_unabortable_runtime_blocks_recorder_abort(
    standalone_recorder,
) -> None:
    """The same gate on the abort path, which has the same hazard.

    An abort that stopped the recorder while the runtime kept producing would
    turn the operator's abort into the next frame's critical-edge fault.
    """
    recorder = standalone_recorder()
    stubborn = _wedge_runtime(recorder, streaming.StreamStatus.DEADLINE_EXCEEDED)

    with pytest.raises(RecorderRuntimeShutdownError, match="did not quiesce"):
        recorder.abort("operator-abort")

    assert stubborn.calls == ["abort", "join"]
    _assert_recorder_untouched(recorder)


def test_still_running_runtime_is_rejected(
    standalone_recorder,
) -> None:
    """``OK`` is not the question; whether anything can still produce is.

    The gate is deliberately not ``status == OK``: a runtime that already went
    terminal through its own fault path answers with that fault, and refusing
    there would make a faulted runtime the one case whose ending cannot be
    recorded. What it checks instead is that the runtime is *terminal*, which
    an ``OK`` from a runtime still in ``RUNNING`` is not.
    """
    recorder = standalone_recorder()
    _wedge_runtime(recorder, streaming.StreamStatus.OK)

    with pytest.raises(RecorderRuntimeShutdownError, match=r"RuntimeState\.RUNNING"):
        recorder.stop("operator-stop")

    _assert_recorder_untouched(recorder)


def test_faulted_runtime_can_still_end_session(
    standalone_recorder,
) -> None:
    """A terminal runtime is quiesced even when it answers with its fault.

    This is the case a ``status == OK`` gate would have broken: a runtime that
    already failed reports that failure from ``stop()``, and treating it as "did
    not quiesce" would leave the one session that most needs an ending unable to
    get one.
    """
    recorder = standalone_recorder()
    _wedge_runtime(
        recorder,
        streaming.StreamStatus.SOURCE_FAILURE,
        state=streaming.RuntimeState.FAILED,
    )

    status = recorder.stop("after-a-runtime-fault")

    assert status.state == recorder._native.RecorderLifecycleState.STOPPED


def test_stuck_runtime_does_not_swallow_caller_exception(standalone_recorder) -> None:
    """On the context manager's exception path the caller's error wins.

    Raising the shutdown failure instead would replace the exception that ended
    the block -- usually the more informative one -- with a consequence of it.
    The failure is attached as a note, and ``close()`` is skipped, because
    closing releases storage a runtime that never stopped may still write into.
    """
    recorder = standalone_recorder()
    _wedge_runtime(recorder, streaming.StreamStatus.DEADLINE_EXCEEDED)

    with pytest.raises(ValueError, match="boom") as raised:
        with recorder:
            raise ValueError("boom")

    assert any("did not quiesce" in note for note in getattr(raised.value, "__notes__", ()))
    _assert_recorder_untouched(recorder)


# --- attach ordering --------------------------------------------------------


def test_attach_before_prepare_is_rejected(tmp_path: Path) -> None:
    recorder = NativeSessionRecorder.create(_recorder_config(tmp_path / "session.nrf"), _schema())
    runner = streaming.StreamRunner(
        _schema(),
        _realtime_config(4),
        streaming.SyntheticNativeSource(_schema(), 4),
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    try:
        with pytest.raises(RecorderStateError, match="prepared recorder"):
            recorder.attach(runner)
    finally:
        recorder.close()


def test_attach_after_runner_prepare_is_rejected(tmp_path: Path) -> None:
    """The readiness gate runs inside ``arm()`` and cannot gate a later edge.

    Refused here rather than discovered later: a runtime armed against a
    recorder it never gated is exactly what contract section 3.1 forbids.
    """
    schema = _schema()
    recorder = NativeSessionRecorder.create(_recorder_config(tmp_path / "session.nrf"), schema)
    runner = streaming.StreamRunner(
        schema,
        _realtime_config(4),
        streaming.SyntheticNativeSource(schema, 4),
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    try:
        recorder.prepare()
        assert runner.prepare() == streaming.StreamStatus.OK
        with pytest.raises(RecorderStateError, match="before the runner is prepared"):
            recorder.attach(runner)
    finally:
        recorder.close()


def test_recorder_observes_exactly_one_runtime(pipeline_factory) -> None:
    """A session is single-use; a second attach is a state error, not a move."""
    pipeline = pipeline_factory(4)
    with pytest.raises(RecorderStateError, match="already attached"):
        pipeline.recorder.attach(pipeline.runner)


# --- the store, and the gap it makes visible --------------------------------


def test_file_backend_records_without_claiming_bounded_cancel(tmp_path: Path) -> None:
    """Disk I/O is accepted while its cancellation capability remains false."""
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "session.nrf"),
        _schema(),
        options=NativeRecorderOptions(
            spool="file",
            spool_path=tmp_path / "session.spool",
            durability_policy="checkpoint_sync",
        ),
    )
    try:
        recorder.prepare()
        code = recorder._recorder.standalone_pass_readiness_gate(True)
        assert code == recorder._native.RecorderStatusCode.OK
        assert recorder._recorder.standalone_start() == recorder._native.RecorderStatusCode.OK
        assert recorder.record_event("file-recording")
        recorder.stop()
        assert recorder.status.session_created is True
        assert recorder.status.spool_backend_cancellable is False
        assert recorder.status.control_spool_committed == 1
    finally:
        recorder.close()


@pytest.mark.parametrize("disk_limit", [32_768, 4 << 20])
def test_file_spool_grows_with_fixed_queues_and_enforces_disk_limit(tmp_path, disk_limit):
    path = tmp_path / "continuous.spool"
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "session.nrf", control_queue_capacity=8),
        _schema(),
        options=NativeRecorderOptions(
            spool="file",
            spool_path=path,
            spool_capacity_bytes=disk_limit,
            max_transaction_bytes=16 << 10,
        ),
    )
    try:
        recorder.prepare()
        assert path.stat().st_size == 0
        native = recorder._native
        assert (
            recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
        )
        assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
        deadline = time.monotonic() + 20
        for _ in range(2048):
            while (
                recorder.status.control_queue_pending >= 4
                and not recorder.status.primary_fault.present
            ):
                assert time.monotonic() < deadline
                time.sleep(0.001)
            if recorder.status.primary_fault.present:
                break
            if not recorder.record_event("sample", text="x" * 900):
                # The disk writer may hit its limit after the preceding check
                # but before admission. The small-limit case expects that fault.
                assert disk_limit == 32_768
                assert recorder.status.primary_fault.present
                break
            assert recorder.status.control_queue_capacity == 8
        code = recorder._recorder.stop("file-limit-test")
        if disk_limit == 32_768:
            assert code == native.RecorderStatusCode.WRITER_FAILED
            assert recorder.status.primary_fault.present
            assert path.stat().st_size <= disk_limit
        else:
            assert code == native.RecorderStatusCode.OK
            assert recorder.status.control_spool_committed == 2048
            assert path.stat().st_size > 1 << 20  # far beyond queue + staging size
    finally:
        recorder.close()


def test_memory_store_survives_nothing(pipeline_factory) -> None:
    """The store's durability is reported, not implied by a committed extent."""
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK

    assert pipeline.recorder.store_provides_crash_durability is False
    status = pipeline.recorder.status
    # The gate checked this, and the status says what it checked.
    assert status.spool_backend_cancellable is True
    assert status.spool_holds_committed_record is True
    assert len(pipeline.recorder.spool_snapshot()) == status.spool_committed_extent


def test_memory_store_rejects_syncing_policy() -> None:
    """A writer that synced a store surviving nothing would publish a lie."""
    with pytest.raises(RecorderConfigError, match="no crash durability"):
        NativeRecorderOptions(spool="memory", durability_policy="transaction_sync")


def test_file_backend_requires_path() -> None:
    with pytest.raises(RecorderConfigError, match="spool_path"):
        NativeRecorderOptions(spool="file", durability_policy="checkpoint_sync")


@pytest.mark.skipif(
    sys.platform not in {"linux", "win32"}
    or (sys.platform == "linux" and not Path("/dev/shm").is_dir()),
    reason="the bounded mapped backend requires Linux tmpfs or Windows local storage",
)
def test_mapped_backend_passes_readiness_without_fallback(tmp_path: Path) -> None:
    """Locked mapped pages give the buffered policy a bounded critical path."""
    spool = (
        Path("/dev/shm") / f"pyneurale-mapped-{uuid.uuid4().hex}.spool"
        if sys.platform == "linux"
        else tmp_path / "mapped.spool"
    )
    recorder: NativeSessionRecorder | None = None
    try:
        recorder = NativeSessionRecorder.create(
            _recorder_config(tmp_path / "mapped.nrf", checkpoint_interval=0),
            _schema(),
            options=NativeRecorderOptions(
                spool="mapped",
                spool_path=spool,
                spool_capacity_bytes=256 << 10,
                durability_policy="buffered",
            ),
        )
        recorder.prepare()
        native = recorder._native
        assert recorder._recorder.standalone_pass_readiness_gate(True) == (
            native.RecorderStatusCode.OK
        )
        assert recorder.store_provides_crash_durability is True
        assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
        recorder.stop("mapped-store-test")
        logical_bytes = len(recorder.spool_snapshot())
        recorder.close()
        recorder = None

        assert logical_bytes > 0
        assert spool.stat().st_size == logical_bytes
    finally:
        if recorder is not None:
            recorder.close()
        spool.unlink(missing_ok=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux filesystem contract")
def test_mapped_backend_rejects_non_tmpfs_path(tmp_path: Path) -> None:
    """The store is created by ``prepare()``, so that is where the path is judged.

    Constructing a recorder allocates nothing and writes nothing: ``created``
    is defined as holding no resources and having no session on disk (contract
    section 3.1), and the store is a resource. The refusal is therefore raised
    by the call that would have created it, and the path it refused is left
    with no file on it.
    """
    spool = tmp_path / "not-tmpfs.spool"
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "mapped.nrf", checkpoint_interval=0),
        _schema(),
        options=NativeRecorderOptions(
            spool="mapped",
            spool_path=spool,
            spool_capacity_bytes=256 << 10,
        ),
    )
    assert not spool.exists()
    try:
        with pytest.raises(RuntimeError, match="bounded mapped spool"):
            recorder.prepare()
        assert not spool.exists()
    finally:
        recorder.close()


def test_mapped_backend_is_rejected_on_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import neurale.recording._native_recorder as native_recorder

    monkeypatch.setattr(native_recorder.sys, "platform", "darwin")
    with pytest.raises(RecorderConfigError, match="Linux tmpfs and Windows"):
        NativeRecorderOptions(spool="mapped", spool_path=tmp_path / "mapped.spool")


def test_mapped_backend_rejects_syncing_policy_and_missing_path() -> None:
    with pytest.raises(RecorderConfigError, match=r"only be used.*buffered"):
        NativeRecorderOptions(spool="mapped", durability_policy="checkpoint_sync")
    with pytest.raises(RecorderConfigError, match="spool_path"):
        NativeRecorderOptions(spool="mapped")


def test_spool_retention_defaults_to_validated_cleanup() -> None:
    """Default cleanup is conditional on validation; debug retention is explicit."""
    assert NativeRecorderOptions().spool_retention == "delete_after_validated_finalization"
    assert NativeRecorderOptions(spool_retention="retain").spool_retention == "retain"
    assert SPOOL_RETENTION_POLICIES == {"retain", "delete_after_validated_finalization"}
    assert (
        NativeRecorderOptions(spool_retention="delete_after_validated_finalization").spool_retention
        == "delete_after_validated_finalization"
    )
    with pytest.raises(RecorderConfigError, match="spool_retention"):
        NativeRecorderOptions(spool_retention="delete")


def test_unknown_spool_backend_is_rejected() -> None:
    with pytest.raises(RecorderConfigError, match="spool must be one of"):
        NativeRecorderOptions(spool="s3")


# --- the control plane ------------------------------------------------------


def test_control_record_returns_control_accepted(
    standalone_recorder,
) -> None:
    """``True`` means exactly stage 1 and 2, which coincide on this plane.

    It is not a statement about the spool, the NRF session, or finalization, so
    the assertion checks that ``control_accepted`` moved while
    ``control_nrf_committed`` did not.
    """
    recorder = standalone_recorder()
    assert recorder.record_event("trial-start", value=0.0) is True

    status = recorder.status
    assert status.control_offered == 1
    assert status.control_accepted == 1
    assert status.control_rejected == 0
    assert status.control_nrf_committed == 0


def test_typed_control_api_matches_native_facade() -> None:
    """The stable facade preserves the typed control contract exactly."""
    import inspect

    from neurale.recording import SessionRecorder

    for name in (
        "record_event",
        "record_state",
        "record_command",
        "record_task_variable",
        "record_label",
        "record_target",
        "record_assistance",
        "record_trial",
        "record_fault",
        "checkpoint",
    ):
        python = inspect.signature(getattr(SessionRecorder, name))
        native = inspect.signature(getattr(NativeSessionRecorder, name))
        assert list(python.parameters) == list(native.parameters), name
        for parameter in python.parameters.values():
            mirror = native.parameters[parameter.name]
            assert mirror.kind == parameter.kind, f"{name}.{parameter.name}"
            assert mirror.default == parameter.default, f"{name}.{parameter.name}"


def test_control_kinds_have_typed_methods(standalone_recorder) -> None:
    """The nine control kinds of contract section 1.1, each counted.

    ``task_variables`` is the ninth and is why the spool container went to
    version 1.1: the registry number is the only bridge from a spool position to
    an NRF kind string, so a kind with no number is a kind the native path
    cannot record.
    """
    recorder = standalone_recorder()
    calls = (
        lambda: recorder.record_event("reach-onset"),
        lambda: recorder.record_trial(start_ns=10, stop_ns=20, outcome="hit"),
        lambda: recorder.record_state("inter-trial"),
        lambda: recorder.record_command("cursor-reset"),
        lambda: recorder.record_target("target-3", value=1.0),
        lambda: recorder.record_label("artifact"),
        lambda: recorder.record_assistance("assist-gain", value=0.25),
        lambda: recorder.record_fault("source_failure", stage="source"),
        lambda: recorder.record_task_variable("difficulty", value=2.0),
    )
    assert len(calls) == len(CONTROL_KINDS)
    for call in calls:
        assert call() is True
    assert recorder.status.control_accepted == len(CONTROL_KINDS)


def test_task_variable_records_registry_number(standalone_recorder) -> None:
    """Version 1.1's whole content, exercised end to end.

    Asserted through the first-rejection position rather than by decoding the
    spool: the position is where the registry number is *used*, and a kind the
    registry could not name would have no expressible position at all.
    """
    recorder = standalone_recorder(max_control_payload_bytes=64, max_control_string_bytes=64)
    native = recorder._native

    assert recorder.record_task_variable("x" * 60, text="y" * 60) is False
    status = recorder.status
    assert status.control_first_rejection.present is True
    assert status.control_first_rejection.identity_kind == (
        native.ProducerIdentityKind.TASK_VARIABLES
    )


def test_checkpoint_is_rejected_after_session_end(
    standalone_recorder,
) -> None:
    """Mirrors ``SessionRecorder.checkpoint()``: only an open session may be checkpointed.

    Requesting one on a session that is no longer accepting would return quietly
    having written nothing, which reads like a checkpoint that succeeded.
    """
    recorder = standalone_recorder()
    assert recorder.record_event("before-checkpoint") is True
    assert recorder.checkpoint() is None

    recorder.stop("done")
    with pytest.raises(RecorderStateError, match="checkpoint"):
        recorder.checkpoint()


def test_control_ordering_is_recoverable(standalone_recorder) -> None:
    """Order within a kind is carried on the record, never inferred from a file position.

    The facade supplies a per-kind counter when the caller does not, which is
    what makes an ordering recoverable at all; an explicit identity overrides it.
    """
    recorder = standalone_recorder()
    for i in range(3):
        assert recorder.record_event(f"event-{i}") is True
    assert recorder._control_identities["events"] == 3
    assert recorder._control_identities["trials"] == 0

    assert recorder.submit_control("events", {"name": "explicit"}, identity=99) is True
    # An explicit identity does not disturb the counter the facade keeps.
    assert recorder._control_identities["events"] == 3


def test_data_plane_kind_is_rejected_on_control_plane(
    standalone_recorder,
) -> None:
    recorder = standalone_recorder()
    with pytest.raises(RecorderConfigError, match="not a control-plane kind"):
        recorder.submit_control("frame", {})


def test_oversized_control_body_faults_recorder(standalone_recorder) -> None:
    """A record the plan cannot hold is a plan violation, not a resize.

    For a critical recorder that is a fault rather than a countable event: the
    submission is refused, counted with its reason, and the primary fault names
    it.
    """
    # Each string stays inside `max_control_string_bytes`; it is their sum plus
    # the JSON framing that overflows the record. That is the case worth
    # pinning: the facade's per-string bound cannot subsume the core's
    # whole-body bound, so the core still has to own it.
    recorder = standalone_recorder(max_control_payload_bytes=64, max_control_string_bytes=64)
    native = recorder._native

    assert recorder.record_event("x" * 60, text="y" * 60) is False
    status = recorder.status
    assert status.control_rejected == 1
    assert status.control_first_rejection.present is True
    assert status.control_first_rejection.identity_kind == native.ProducerIdentityKind.EVENTS
    assert status.primary_fault.present is True
    assert status.primary_fault.reason == native.RecorderFaultReason.PLAN_VIOLATION


def test_late_control_record_is_api_misuse(
    pipeline_factory,
) -> None:
    """Contract section 1.3, and the native half of gap 9.

    An item submitted after the recorder stopped accepting was never accepted by
    anything. The native recorder counts it under a *separate*
    rejected-after-close counter and must not fold it into either plane's loss
    numbers -- which is exactly where the Python recorder diverges, reporting a
    rejected control record as a lost frame.
    """
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK

    assert pipeline.recorder.record_event("too-late") is False
    status = pipeline.recorder.status

    assert status.rejected_after_close_control == 1
    # Not a recording loss, and not counted against the other plane.
    assert status.control_rejected == 0
    assert status.lost_between_control_acceptance_and_spool == 0
    assert status.failed_between_runtime_and_recorder == 0
    assert status.lost_between_recorder_and_spool == 0
    # And it did not retroactively change what the capture ended as.
    assert status.spool_ended_cleanly is True


def test_control_record_declares_clock_domain(standalone_recorder) -> None:
    """Every control record carries the domain its ``time_ns`` belongs to.

    An unlabelled nanosecond value is not a time anyone downstream can place,
    so the facade declares a default domain rather than leaving it implied.
    """
    recorder = standalone_recorder()
    assert HOST_MONOTONIC_CLOCK_DOMAIN == 0
    assert recorder.record_event("stamped", time_ns=1_700_000_000_000_000_000) is True
    assert (
        recorder.submit_control(
            "events", {"name": "device-timed"}, clock_domain=CLOCK_DOMAIN, time_ns=42
        )
        is True
    )
    assert recorder.status.control_accepted == 2


def test_control_timestamp_defaults_to_session_clock(
    standalone_recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host **monotonic**, not the wall clock, and the same clock the Python recorder uses.

    The NRF session registers ``session.clock`` as ``host_monotonic`` and
    ``SessionRecorder`` stamps control rows with :func:`time.monotonic_ns`, so a
    native record carrying Unix wall-clock nanoseconds under the same declared
    domain would be a timestamp-domain error rather than a naming difference.

    Checked by watching which clock is read, because the two values are only
    distinguishable by magnitude and that is not something to assert on.
    """
    import neurale.recording._native_recorder as facade

    recorder = standalone_recorder()
    read: list[str] = []
    monkeypatch.setattr(facade.time, "monotonic_ns", lambda: (read.append("monotonic"), 12_345)[1])
    monkeypatch.setattr(facade.time, "time_ns", lambda: (read.append("wall"), 0)[1])

    assert recorder.record_event("unstamped") is True
    assert read == ["monotonic"]


def test_control_clock_domain_cannot_collide_with_signal(tmp_path: Path) -> None:
    """One number cannot name both a host clock and a device clock.

    A control record and a signal block would then declare the same
    ``clock_domain`` while carrying times from different clocks, and nothing
    downstream could detect it -- so it is refused where it is still cheap.
    """
    with pytest.raises(RecorderConfigError, match="control_clock_domain"):
        NativeSessionRecorder.create(
            _recorder_config(tmp_path / "session.nrf"),
            _schema(),
            options=NativeRecorderOptions(control_clock_domain=CLOCK_DOMAIN),
        )


def test_oversized_string_is_rejected_before_submit(
    standalone_recorder,
) -> None:
    """The string bound is a contract of its own, and breaking it is caller error.

    Without it one caller-supplied string may fill an entire record, which is
    how an unbounded free-text field reaches a container that promised bounded
    records. It raises rather than returning ``False`` because the record never
    entered the recording path: nothing was offered, nothing was lost, and the
    session's accounting must not gain an entry describing API misuse.
    """
    recorder = standalone_recorder(max_control_string_bytes=32)

    with pytest.raises(RecorderConfigError, match="max_control_string_bytes"):
        recorder.record_event("fine", text="y" * 33)

    status = recorder.status
    assert status.control_offered == 0
    assert status.control_rejected == 0
    assert status.primary_fault.present is False
    # And the session is still recording: caller error did not end it.
    assert recorder.record_event("still-accepting") is True


def test_string_bound_is_measured_in_utf8_bytes(standalone_recorder) -> None:
    """A character count would let an astral-plane string be four times its bound."""
    recorder = standalone_recorder(max_control_string_bytes=8)
    # Two characters, eight UTF-8 bytes: inside the bound.
    assert recorder.record_event("\U0001f9e0\U0001f9e0") is True
    with pytest.raises(RecorderConfigError, match="UTF-8 bytes"):
        recorder.record_event("\U0001f9e0" * 3)


def test_string_bound_reaches_nested_strings(standalone_recorder) -> None:
    """A top-level-only check is not a bound.

    ``submit_control`` takes a body, not a flat row, so ``{"metadata": {...}}``
    and a list of strings route the same unbounded free text into the same
    record. Checking only the outermost values would let every one of them past
    the maximum the container was promised, leaving the whole-body limit in the
    native core to catch it later as a *fault* -- turning caller error into a
    recording outcome, which is exactly what this bound exists to prevent.
    """
    recorder = standalone_recorder(max_control_string_bytes=32)
    over = "y" * 33

    with pytest.raises(RecorderConfigError, match="max_control_string_bytes"):
        recorder.submit_control("events", {"name": "fine", "metadata": {"note": over}})
    with pytest.raises(RecorderConfigError, match="max_control_string_bytes"):
        recorder.submit_control("events", {"name": "fine", "tags": ["short", over]})
    # A key is caller-supplied text too, and reaches the record just the same.
    with pytest.raises(RecorderConfigError, match="max_control_string_bytes"):
        recorder.submit_control("events", {"name": "fine", over: "short"})

    status = recorder.status
    assert status.control_offered == 0
    assert status.control_rejected == 0
    assert status.primary_fault.present is False
    assert recorder.submit_control("events", {"name": "fine", "metadata": {"note": "y"}}) is True


def test_self_referencing_body_does_not_wedge_bound_check(
    standalone_recorder,
) -> None:
    """The walk terminates on a cycle instead of spinning inside the facade.

    A body like this cannot be encoded at all, and the encoder is what should
    say so; the bound check must not be where the caller's mistake becomes a
    hang.
    """
    recorder = standalone_recorder(max_control_string_bytes=32)
    body: dict[str, Any] = {"name": "fine"}
    body["self"] = body

    with pytest.raises(Exception):  # noqa: B017 - the encoder names it, not us
        recorder.submit_control("events", body)


def test_string_bound_wider_than_record_is_rejected() -> None:
    """A string that fits its own bound must be able to fit the record carrying it."""
    with pytest.raises(RecorderConfigError, match="max_control_string_bytes"):
        NativeRecorderOptions(max_control_payload_bytes=64, max_control_string_bytes=128)


def test_saturated_control_queue_faults(
    standalone_recorder,
) -> None:
    """A critical recorder is lossless until fault, on both planes.

    The worker is given a long idle poll so it sleeps for the run, but thread
    creation can lose the race with ``start()`` -- the worker enters
    ``running`` instead of ``parked`` and drains before it sleeps. That frees
    slots and lets more than ``depth`` be accepted in total, but the queue
    never holds more than ``depth`` at once. The high-water mark is the witness
    that the queue was genuinely full, which is the condition the fault tests;
    ``accepted <= depth`` would only hold when the worker does not drain,
    which is a scheduling race rather than a property of the recorder.
    """
    depth = 4
    recorder = standalone_recorder(
        control_queue_capacity=depth,
        # Long enough that once the worker sleeps it stays asleep for the run.
        worker_idle_poll_nanos=2_000_000_000,
    )
    native = recorder._native

    accepted = 0
    for i in range(depth * 4):
        if not recorder.record_event(f"event-{i}"):
            break
        accepted += 1

    status = recorder.status
    assert status.control_queue_high_water_mark == depth
    assert status.control_accepted == accepted
    assert status.control_rejected == 1
    assert status.primary_fault.present is True
    assert status.primary_fault.reason == native.RecorderFaultReason.CONTROL_QUEUE_SATURATED
    # Refused, not silently dropped: the first refusal has a position.
    assert status.control_first_rejection.present is True
    assert status.control_first_rejection.identity_kind == native.ProducerIdentityKind.EVENTS


# --- failure surfaces, never a fallback -------------------------------------


def test_missing_finalization_is_reported(pipeline_factory) -> None:
    """The native core implements neither, and the surface says so."""
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK

    with pytest.raises(NotImplementedError, match=r"finalize.*not implemented"):
        pipeline.recorder.finalize()
    with pytest.raises(NotImplementedError, match=r"abandon_finalization.*not implemented"):
        pipeline.recorder.abandon_finalization()

    status = pipeline.recorder.status
    native = pipeline.recorder._native
    assert status.finalization_status == native.FinalizationStatus.NOT_STARTED
    assert status.finalization_required is True


def test_preparing_twice_is_state_error(tmp_path: Path) -> None:
    """``prepare()`` is not idempotent; a second call is a state error."""
    recorder = NativeSessionRecorder.create(_recorder_config(tmp_path / "session.nrf"), _schema())
    try:
        recorder.prepare()
        with pytest.raises(RecorderStateError, match="not legal"):
            recorder.prepare()
    finally:
        recorder.close()


# --- context manager --------------------------------------------------------


def test_normal_context_exit_stops_gracefully(pipeline_factory) -> None:
    pipeline = pipeline_factory(4)
    assert pipeline.run() == streaming.StreamStatus.OK

    native = pipeline.recorder._native
    with pipeline.recorder as recorder:
        assert recorder is pipeline.recorder
    status = pipeline.recorder.status
    assert status.state == native.RecorderLifecycleState.CLOSED
    assert status.requested_terminal_intent == native.RequestedTerminalIntent.NORMAL
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.NORMAL


def test_context_exit_on_exception_never_ends_normally(
    standalone_recorder,
) -> None:
    """What is normative is the outcome, not the number of calls.

    The session must not end as ``normal`` because an exception left the block,
    and the exception itself must propagate unchanged.
    """
    recorder = standalone_recorder()
    native = recorder._native

    with pytest.raises(ValueError, match="boom"):
        with recorder:
            raise ValueError("boom")

    status = recorder.status
    assert status.state == native.RecorderLifecycleState.CLOSED
    assert status.requested_terminal_intent == native.RequestedTerminalIntent.ABORTED
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.ABORTED
    assert status.capture_outcome == native.CaptureOutcome.ABORTED
    # An abort is not a fault: no fault record is invented.
    assert status.primary_fault.present is False


def test_abort_drains_handed_over_records(standalone_recorder) -> None:
    """An abort stops recording new data; it does not destroy data already accepted."""
    recorder = standalone_recorder()
    for i in range(4):
        assert recorder.record_event(f"event-{i}") is True

    status = recorder.abort("operator-abort")
    assert status.control_accepted == 4
    assert status.control_spool_committed == 4
    assert status.lost_between_control_acceptance_and_spool == 0


# --- more than one stream ---------------------------------------------------


def test_declared_streams_reach_spool(pipeline_factory) -> None:
    """Neural, behavioral, and feature, in one session.

    A single-signal fixture cannot tell "the recorder records what the plan
    says" from "the recorder records the only thing there was", and it cannot
    exercise a frame carrying several blocks at all. The three-stream fixture
    covers int16 sample-major, float32 channel-major, and float64 feature data
    with a descriptor.
    """
    frames = 6
    pipeline = pipeline_factory(frames, multi_stream=True)
    assert pipeline.run() == streaming.StreamStatus.OK
    status = pipeline.recorder.status

    assert pipeline.recorder.plan.recorded_signal_ids == (1, 2, 3)

    # One frame carrying three blocks is **one** item, not three (section 1.1).
    assert status.runtime_accepted == frames
    assert status.recorder_accepted == frames
    assert status.spool_committed == frames
    # The blocks are a diagnostic, named distinctly, never folded into the
    # ladder.
    assert status.signal_blocks_recorded == frames * MULTI_BLOCKS_PER_FRAME
    assert status.payload_bytes_copied == frames * MULTI_FRAME_PAYLOAD_BYTES
    assert status.lost_between_recorder_and_spool == 0


def test_multi_stream_session_carries_whole_plan(
    pipeline_factory,
) -> None:
    """The plan fingerprint the spool stores covers all three streams.

    Checked against the plan the *Python* compiler produced, so the assertion
    fails if the native mirror drops a stream rather than merely if it drops a
    byte.
    """
    pipeline = pipeline_factory(2, multi_stream=True)
    assert pipeline.run() == streaming.StreamStatus.OK

    plan = pipeline.recorder.plan
    assert {stream.stream_id for stream in plan.streams} == {"neural", "cursor", "bandpower"}
    snapshot = pipeline.recorder.spool_snapshot()
    assert bytes.fromhex(plan.fingerprint) in snapshot


# --- a writer that fails under a running runtime ----------------------------


def test_spool_writer_failure_faults_recorder_and_runtime(
    pipeline_factory,
) -> None:
    """A store that runs out of space is a capture fault, and it must reach the runtime.

    Provoked with a memory spool too small for the session, which is the honest
    Python-drivable writer failure: the store reports out of space exactly as a
    full disk does. What is asserted is the whole chain -- the recorder latches
    a spool-writer fault, the session escalates to ``faulted``, and the runtime
    that was producing into it stops rather than running to completion unaware.
    """
    # The queue is deep enough to hold the whole run and the store is not, so
    # the writer is what fails. Sizing it the other way round would latch queue
    # saturation as the primary fault -- a real outcome, but a different one,
    # and the primary fault is written exactly once.
    frames = 512
    pipeline = pipeline_factory(frames, spool_capacity_bytes=2048, frame_queue_capacity=frames * 2)
    native = pipeline.recorder._native

    assert pipeline.start() == streaming.StreamStatus.OK
    # The runtime ends on its own here: the critical recorder faulted, and a
    # critical edge failure is not something the runtime carries on through.
    terminal = pipeline.runner.join()
    assert terminal != streaming.StreamStatus.OK

    status = pipeline.recorder.status
    assert status.primary_fault.present is True
    assert status.primary_fault.origin == native.FaultOrigin.RECORDER
    assert status.primary_fault.reason == native.RecorderFaultReason.SPOOL_WRITER_FAILED
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.FAULTED

    # And the capture outcome is `None` -- section 3.2's `unknown` -- which is
    # the whole reason it is a separate field from the outcome above. The store
    # that failed is the store the session-end record would have been written
    # to, so nothing froze the answer, and reporting `faulted` here would state
    # as recorded fact something no artifact says. What this recorder
    # *concluded* is `effective_session_outcome`; what the artifact *says* is
    # nothing at all.
    assert status.capture_outcome is None
    assert status.spool_ended_cleanly is False
    # Still no completeness verdict: there is no sealed session either way.
    assert status.complete is None
    # And the failure surfaced as a failure -- nothing fell back to anything.
    assert pipeline.runner.state == streaming.RuntimeState.FAILED


# --- session identity -------------------------------------------------------


def test_generated_session_id_reaches_superblock(tmp_path: Path) -> None:
    """The plan's session identity is what the spool stores, unchanged."""
    recorder = NativeSessionRecorder.create(_recorder_config(tmp_path / "session.nrf"), _schema())
    session_id = recorder.plan.session.session_id
    try:
        recorder.prepare()
        assert recorder._recorder.standalone_pass_readiness_gate(True) == (
            recorder._native.RecorderStatusCode.OK
        )
        snapshot = recorder.spool_snapshot()
        assert uuid.UUID(session_id).bytes in snapshot
        assert session_id.encode("ascii") in snapshot
    finally:
        recorder.close()
