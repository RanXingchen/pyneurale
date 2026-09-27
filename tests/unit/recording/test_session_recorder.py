#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Public recorder convergence tests."""

from __future__ import annotations

import inspect
import os
import stat
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import neurale.streaming as streaming
from neurale import __version__
from neurale.io.nrf import NrfReader
from neurale.recording import (
    FinalizationError,
    RecorderConfig,
    RecorderConfigError,
    RecorderSessionMetadata,
    RecorderState,
    RecorderStateError,
    SessionRecorder,
)
from neurale.recording._finalizer import finalize_spool, write_plan_sidecar
from neurale.recording._native_recorder import NativeRecorderOptions

from .conftest import native_schema, recorder_config, stream_specs
from .test_finalizer import MULTI_MAX_FRAME_PAYLOAD_BYTES, _realtime_config


@pytest.mark.parametrize("explicit", [False, True])
def test_experiment_budget_preserves_identity_and_respects_explicit_limits(tmp_path, explicit):
    from neurale.devices.simulation import SimulatedNeuralDevice
    from neurale.recording import RecorderLimits
    from neurale.signal.simulation import SignalGenerator

    device = SimulatedNeuralDevice(SignalGenerator.zeros(64, 30_000.0), samples_per_frame=30)
    config = RecorderConfig(
        path=tmp_path / "budget.nrf", limits=RecorderLimits() if explicit else None
    )
    recorder = SessionRecorder.create(config, device)
    try:
        before = recorder.plan
        spool = recorder._spool_path
        recorder._budget_experiment(duration_seconds=600, control_records=65_536)
        after = recorder.plan
        assert after.session == before.session
        assert after.fingerprint == before.fingerprint
        assert recorder._spool_path == spool
        assert recorder.state == RecorderState.CREATED
        if explicit:
            assert after is before
        else:
            assert after.resource_bounds.spool_capacity_bytes == (1 << 53) - 1
            assert after.resource_bounds.frame_queue_capacity == 6250
            assert (
                recorder._impl.options.edge_capacity == after.resource_bounds.frame_queue_capacity
            )
            recorder._budget_experiment(duration_seconds=600, control_records=65_536)
            assert recorder.plan == after
        assert not config.path.exists()
    finally:
        recorder.close()
        device.close()


def test_automatic_budget_scales_and_rejects_unknown_cadence(tmp_path):
    from neurale.recording._budget import budget_recording
    from neurale.recording._plan import compile_recording_plan

    plan = compile_recording_plan(
        recorder_config(tmp_path / "budget.nrf", streams=stream_specs()), native_schema()
    )

    def budget(duration, controls=65_536, source_plan=plan):
        return budget_recording(
            source_plan,
            duration_seconds=duration,
            control_records=controls,
        ).resource_bounds

    baseline = budget(600)
    assert budget(1200, 131_072) == baseline
    assert budget(600, 262_144).control_queue_capacity > baseline.control_queue_capacity
    assert baseline.spool_capacity_bytes == plan.resource_bounds.spool_capacity_bytes
    assert baseline.frame_queue_capacity >= plan.resource_bounds.frame_queue_capacity
    assert baseline.checkpoint_interval == plan.resource_bounds.checkpoint_interval
    for duration in (0, -1, float("inf"), float("nan")):
        with pytest.raises(RecorderConfigError, match="finite and positive"):
            budget(duration)
    signals = plan.native_schema.signals
    unknown = replace(
        plan,
        native_schema=replace(
            plan.native_schema, signals=(replace(signals[0], fs=(0, 1)), *signals[1:])
        ),
    )
    with pytest.raises(RecorderConfigError, match="nominal frame cadence"):
        budget(600, source_plan=unknown)


def test_timed_out_close_retains_live_storage_and_can_retry(tmp_path, monkeypatch):
    recorder = SessionRecorder.create(recorder_config(tmp_path / "pending.nrf"), native_schema())
    recorder._impl.close()
    calls = []
    status = SimpleNamespace(worker_running=True, spool_holds_committed_record=True)

    def close():
        calls.append("close")
        if status.worker_running:
            raise RuntimeError("disk I/O still pending")

    recorder._impl = SimpleNamespace(
        state="failed", status=status, runtime_is_live=lambda: False, close=close
    )
    monkeypatch.setattr(
        SessionRecorder, "_discard_spool_artifacts", lambda _self: calls.append("delete")
    )
    assert recorder.state is RecorderState.FAILED
    assert calls == []  # polling a failed recorder must not wait on live I/O
    recorder.close()
    assert not recorder._released
    assert calls == ["close"]
    status.worker_running = False
    recorder.close()
    assert recorder._released
    assert calls == ["close", "close"]


_REAL_STABLE_NATIVE_OPTIONS = __import__(
    "neurale.recording._session_recorder", fromlist=["_stable_native_options"]
)._stable_native_options


def _run(tmp_path: Path, *, frames: int = 8, config: object | None = None):
    schema = native_schema()
    if config is None:
        config = recorder_config(
            tmp_path / "session.nrf", streams=stream_specs(), checkpoint_interval=0
        )
    output = Path(config.path)
    recorder = SessionRecorder.create(config, schema)
    source = streaming.SyntheticNativeSource(schema, frames, 1, None, None, payload_pattern=True)
    runner = streaming.StreamRunner(
        schema,
        _realtime_config(frames, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES),
        source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    assert runner.join() == streaming.StreamStatus.OK
    return recorder, output


@pytest.fixture(autouse=True)
def _memory_spool_for_facade_lifecycle_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise facade orchestration without changing its stable backend policy.

    Substituting the backend is what lets the orchestration tests run where the
    approved store does not exist -- and it is also what makes them say nothing
    about the store the product ships. Tests marked ``stable_backend`` therefore
    opt out and run against the real file spool and its sidecar, which is the
    only way the path selection and artifact lifecycle are exercised at all.
    """
    import neurale.recording._session_recorder as facade

    if request.node.get_closest_marker("stable_backend") is not None:
        return
    monkeypatch.setattr(
        facade,
        "_stable_native_options",
        lambda plan, _output, **_kwargs: NativeRecorderOptions(
            spool="memory",
            spool_capacity_bytes=min(plan.resource_bounds.spool_capacity_bytes, 64 << 20),
            edge_capacity=plan.resource_bounds.frame_queue_capacity,
        ),
    )


#: Linux-specific permissions and cleanup checks; no tmpfs or memlock required.
stable_backend = pytest.mark.stable_backend
requires_linux_files = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux file permissions and cleanup contract"
)
_STABLE_SPOOL_CAPACITY_BYTES = 1 << 20


def _stable_config(path: Path, **overrides: object):
    """A recorder configuration with a small file-spool capacity."""
    return recorder_config(
        path,
        streams=stream_specs(),
        spool_capacity_bytes=_STABLE_SPOOL_CAPACITY_BYTES,
        **overrides,
    )


def _sidecar(spool: Path) -> Path:
    return spool.with_name(spool.name + ".plan.json")


def _descriptors_for(path: Path) -> list[str]:
    """Every open descriptor in this process that still refers to *path*.

    A deleted file keeps its target readable as ``<path> (deleted)``, which is
    exactly the case that matters: the spool is unlinked on the discard paths,
    and a descriptor still holding it would be invisible to any check that only
    looks at the filesystem.
    """
    found = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:  # the descriptor closed while the directory was walked
            continue
        if target.split(" (deleted)")[0] == str(path):
            found.append(target)
    return found


def test_public_recorder_finalizes_canonical_nrf(tmp_path: Path) -> None:
    recorder, output = _run(tmp_path)

    stopped = recorder.stop("done")
    assert stopped.state is RecorderState.STOPPED
    assert stopped.finalization_required
    assert stopped.complete is None

    closed = recorder.finalize()
    assert closed.state is RecorderState.CLOSED
    assert closed.finalization_status == "succeeded"
    assert closed.sealed
    assert closed.complete is True
    assert closed.accounting_verified
    assert closed.nrf_committed == closed.spool_committed
    assert closed.control_nrf_committed == closed.control_spool_committed
    assert output.is_file()

    with NrfReader.open(output) as reader:
        assert reader.complete is True
        assert reader.accounting_verified
        assert {"neural", "cursor", "bandpower"} <= set(reader.stream_ids())


def test_finalization_preserves_plan_metadata_and_bounds(tmp_path: Path) -> None:
    base = recorder_config(
        tmp_path / "session.nrf",
        streams=stream_specs(),
        checkpoint_interval=0,
    )
    assert base.limits is not None
    config = replace(
        base,
        session=RecorderSessionMetadata(subject={"id": "test-subject"}),
        metadata={"experiment": {"name": "m6-14"}},
        limits=replace(base.limits, max_control_records=7 * 1024),
    )
    recorder = SessionRecorder.create(config, native_schema())
    source = streaming.SyntheticNativeSource(native_schema(), 2, 1, None, None)
    runner = streaming.StreamRunner(
        native_schema(),
        _realtime_config(2, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES),
        source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    assert runner.join() == streaming.StreamStatus.OK
    recorder.stop("done")
    recorder.finalize()

    with NrfReader.open(config.path) as reader:
        extension = reader.manifest["extensions"]["neurale.native_replay"]
        assert reader.manifest["writer"] == {"name": "pyneurale", "version": __version__}
        assert reader.manifest["session"]["subject"]["id"] == "test-subject"
        assert reader.manifest["metadata"]["experiment"]["name"] == "m6-14"
        assert extension["resource_bounds"] == recorder.plan.resource_bounds.document()


def test_retained_spool_sidecar_reconstructs_public_plan_offline(tmp_path: Path) -> None:
    base = recorder_config(tmp_path / "live.nrf", streams=stream_specs(), checkpoint_interval=0)
    assert base.limits is not None
    config = replace(
        base,
        session=RecorderSessionMetadata(subject={"id": "offline-subject"}),
        metadata={"experiment": {"name": "offline-recovery"}},
        limits=replace(base.limits, max_control_records=7 * 1024),
    )
    recorder = SessionRecorder.create(config, native_schema())
    source = streaming.SyntheticNativeSource(native_schema(), 2, 1, None, None)
    runner = streaming.StreamRunner(
        native_schema(),
        _realtime_config(2, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES),
        source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    assert runner.join() == streaming.StreamStatus.OK
    recorder.stop("done")

    spool = tmp_path / "retained.spool"
    spool.write_bytes(recorder._impl.spool_snapshot())
    write_plan_sidecar(spool, recorder.plan)
    output = tmp_path / "offline.nrf"
    finalize_spool(spool, output)
    recorder._impl.close()

    with NrfReader.open(output) as reader:
        extension = reader.manifest["extensions"]["neurale.native_replay"]
        assert reader.manifest["writer"] == {"name": "pyneurale", "version": __version__}
        assert reader.manifest["session"]["subject"]["id"] == "offline-subject"
        assert extension["resource_bounds"] == recorder.plan.resource_bounds.document()


def test_close_performs_graceful_stop_and_finalization(tmp_path: Path) -> None:
    recorder, output = _run(tmp_path)

    status = recorder.close()

    assert status.state is RecorderState.CLOSED
    assert status.finalization_status == "succeeded"
    assert output.is_file()
    assert recorder.close().state is RecorderState.CLOSED


def test_retryable_finalization_failure_remains_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, output = _run(tmp_path)
    recorder.stop("done")

    import neurale.recording._session_recorder as facade

    real = facade.finalize_spool
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FinalizationError("injected publication failure", category="publication")
        return real(*args, **kwargs)

    monkeypatch.setattr(facade, "finalize_spool", fail_once)
    with pytest.raises(FinalizationError, match="injected publication failure"):
        recorder.finalize()

    failed = recorder.status
    assert failed.state is RecorderState.FINALIZATION_FAILED
    assert failed.finalization_status == "failed_retryable"
    assert failed.finalization_required
    assert len(failed.finalization_attempts) == 1

    recovered = recorder.finalize()
    assert recovered.state is RecorderState.CLOSED
    assert recovered.complete is True
    assert len(recovered.finalization_attempts) == 2
    assert output.is_file()


def test_nonretryable_finalization_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``retryable=False`` is the finalizer's answer, and the public status must not invert it.

    The layers below already act on the distinction: the finalizer stamps the
    progress document ``failed`` rather than ``failed_retryable``, and recovery
    treats only ``running`` and ``failed_retryable`` as resumable. A public
    recorder reporting ``failed_retryable`` here would tell the caller to try
    again about the same fault the progress document tells recovery cannot be
    retried -- one session, two contradictory answers.

    So it is terminal, and it is not ``abandoned``: nobody decided to stop, the
    finalizer reported that continuing over the same bytes is pointless. What
    is left is diagnosis, repair, or quarantine of the retained spool.
    """
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "finalize_spool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FinalizationError("bad source", category="source", retryable=False)
        ),
    )
    with pytest.raises(FinalizationError, match="bad source"):
        recorder.finalize()

    status = recorder.status
    assert status.state is RecorderState.FAILED
    assert status.finalization_status == "failed"
    assert status.finalization_attempts[-1].outcome == "failed"
    assert status.finalization_attempts[-1].category == "source"
    # The items are still in the spool, and a spool nobody converted has lost
    # nothing -- the fault is that no retry will convert it.
    assert status.lost_during_finalization == 0
    assert status.awaiting_finalization == status.spool_committed

    monkeypatch.undo()
    with pytest.raises(RecorderStateError, match="non-retryably"):
        recorder.finalize()
    with pytest.raises(RecorderStateError, match="requires a retryable finalization failure"):
        recorder.abandon_finalization("operator gave up")
    assert recorder.status.finalization_status == "failed"


def test_close_waits_for_running_finalizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    recorder._finalization_wait_seconds = 0.01
    entered = threading.Event()
    release = threading.Event()
    import neurale.recording._session_recorder as facade

    real = facade.finalize_spool

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(facade, "finalize_spool", blocked)
    thread = threading.Thread(target=recorder.finalize)
    thread.start()
    assert entered.wait(1.0)
    assert recorder.close().state is RecorderState.FINALIZING
    assert thread.is_alive()
    release.set()
    thread.join(5.0)
    assert not thread.is_alive()
    assert recorder.status.state is RecorderState.CLOSED


def test_context_keeps_caller_error_on_shutdown_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SessionRecorder.create(recorder_config(tmp_path / "session.nrf"), native_schema())

    def fail_abort(_reason: str):
        from neurale.recording import RecorderRuntimeShutdownError

        raise RecorderRuntimeShutdownError("shutdown-failed")

    monkeypatch.setattr(type(recorder._impl), "abort", lambda _self, reason: fail_abort(reason))
    with pytest.raises(ValueError, match="caller-error") as caught:
        with recorder:
            raise ValueError("caller-error")
    assert any("left open" in note for note in caught.value.__notes__)


def test_context_keeps_caller_error_on_finalization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing finalizes, and a finalization that fails is still cleanup.

    The exception that unwound the block is the diagnosis; what the recorder
    was doing about it is a footnote. Letting a ``FinalizationError`` out of
    ``__exit__`` replaces the application's error with the recorder's -- and
    the asymmetry decides it: the finalization can be retried from the retained
    spool afterwards, while an exception that was thrown away is gone.
    """
    recorder, _ = _run(tmp_path)
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "finalize_spool",
        lambda *_a, **_k: (_ for _ in ()).throw(
            FinalizationError("injected publication failure", category="publication")
        ),
    )
    with pytest.raises(ValueError, match="application failure") as caught:
        with recorder:
            raise ValueError("application failure")

    assert any("finalizing the recording failed" in note for note in caught.value.__notes__)
    assert any("injected publication failure" in note for note in caught.value.__notes__)
    # The recording is not lost by being demoted to a note: it is exactly the
    # state a retry acts on.
    assert recorder.status.finalization_status == "failed_retryable"
    assert recorder.status.state is RecorderState.FINALIZATION_FAILED


def test_clean_exit_propagates_finalization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no caller exception to protect, the finalization failure *is* the result."""
    recorder, _ = _run(tmp_path)
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "finalize_spool",
        lambda *_a, **_k: (_ for _ in ()).throw(
            FinalizationError("injected publication failure", category="publication")
        ),
    )
    with pytest.raises(FinalizationError, match="injected publication failure"):
        with recorder:
            pass


def test_public_config_does_not_expose_lossy_policy(tmp_path: Path) -> None:
    assert "overflow_policy" not in inspect.signature(RecorderConfig).parameters
    plan = SessionRecorder.create(recorder_config(tmp_path / "session.nrf"), native_schema()).plan
    assert plan.resource_bounds.overflow_policy == "fault"


def test_unsupported_platform_refuses_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(facade, "_stable_native_options", _REAL_STABLE_NATIVE_OPTIONS)
    monkeypatch.setattr(facade.sys, "platform", "darwin")
    with pytest.raises(RecorderConfigError, match="requires Linux or Windows"):
        SessionRecorder.create(recorder_config(tmp_path / "session.nrf"), native_schema())


def test_public_surface_omits_in_memory_recorder() -> None:
    assert not hasattr(SessionRecorder, "create_in_memory")


def test_public_surface_has_no_provisional_backend_selection() -> None:
    import neurale.recording as recording

    assert not hasattr(recording, "NativeSessionRecorder")
    assert not hasattr(recording, "NativeRecorderOptions")
    assert not hasattr(recording, "SPOOL_BACKENDS")
    assert not hasattr(recording.SessionRecorder, "options")


def test_status_uses_finalized_artifact_for_outcome(
    tmp_path: Path,
) -> None:
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    status = recorder.finalize()
    assert status.effective_session_outcome == status.termination_kind == "normal"
    assert status.legacy_termination_normal is False


def test_import_does_not_load_native_extension() -> None:
    code = "import sys, neurale.recording; assert 'neurale._neurale' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_public_recorder_contains_no_python_live_engine_surface() -> None:
    forbidden = {
        "observer",
        "submit_frame",
        "drain",
        "queue_depth",
        "_on_observation",
        "_run",
        "_loop",
        "_write_batch",
    }
    assert forbidden.isdisjoint(vars(SessionRecorder))


# --- what the accounting says before finalization has an outcome ------------


def test_stopped_recording_awaits_finalization(tmp_path: Path) -> None:
    """Committed and not yet converted is pending, not lost.

    The distinction is the whole point of the field: the items are bytes in a
    retained spool that a finalizer has not been asked to read yet, and a status
    reporting them as lost during a finalization that never ran describes a data
    loss that did not happen.
    """
    recorder, _ = _run(tmp_path)

    stopped = recorder.stop("done")

    assert stopped.spool_committed > 0
    assert stopped.nrf_committed == 0
    assert stopped.lost_during_finalization == 0
    assert stopped.control_lost_during_finalization == 0
    assert stopped.awaiting_finalization == stopped.spool_committed
    assert stopped.control_awaiting_finalization == stopped.control_spool_committed
    recorder.finalize()


def test_retryable_finalization_failure_is_not_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract section 3.2: the capture outcome is unchanged by a failed finalization.

    A retry over the same spool is what this state exists for, so a status
    reporting every committed item as lost would contradict the retry that is
    about to recover them -- and does, in the same test.
    """
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    committed = recorder.status.spool_committed
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "finalize_spool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FinalizationError("injected source failure", category="source")
        ),
    )
    with pytest.raises(FinalizationError):
        recorder.finalize()

    failed = recorder.status
    assert failed.finalization_status == "failed_retryable"
    assert failed.lost_during_finalization == 0
    assert failed.control_lost_during_finalization == 0
    assert failed.nrf_committed == 0
    assert failed.awaiting_finalization == committed

    monkeypatch.undo()
    recovered = recorder.finalize()
    assert recovered.nrf_committed == committed
    assert recovered.lost_during_finalization == 0
    assert recovered.awaiting_finalization == 0


def test_publication_failure_reports_committed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 4 is *committed to NRF*, and publishing is not what commits it.

    A conversion that sealed its session and then failed to move it into place
    has written those rows; reporting zero would under-report a commit that
    happened, and reporting the difference as loss would invent one that did
    not. The retry then publishes exactly what was already converted.
    """
    recorder, output = _run(tmp_path)
    recorder.stop("done")
    committed = recorder.status.spool_committed
    import neurale.recording._finalizer as finalizer

    def refuse_publication(*_args, **_kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(finalizer, "_publish_session", refuse_publication)
    with pytest.raises(FinalizationError, match="publish"):
        recorder.finalize()

    staged = recorder.status
    assert staged.finalization_status == "failed_retryable"
    assert staged.finalization_attempts[-1].category == "publication"
    assert staged.nrf_committed == committed
    assert staged.control_nrf_committed == staged.control_spool_committed
    assert staged.lost_during_finalization == 0
    assert staged.awaiting_finalization == 0
    assert not output.exists()

    monkeypatch.undo()
    published = recorder.finalize()
    assert published.nrf_committed == committed
    assert published.complete is True


def test_abandoning_reports_no_commit_and_no_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abandoned finalization publishes nothing and loses nothing.

    The staged rows are removed with the decision, so claiming them as NRF
    commits would point at a session that no longer exists -- and the spool is
    retained, so the items are still recoverable offline rather than lost.
    """
    recorder, output = _run(tmp_path)
    recorder.stop("done")
    import neurale.recording._finalizer as finalizer

    monkeypatch.setattr(
        finalizer,
        "_publish_session",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("injected publication failure")),
    )
    with pytest.raises(FinalizationError):
        recorder.finalize()
    assert recorder.status.nrf_committed > 0

    abandoned = recorder.abandon_finalization("operator gave up")

    assert abandoned.finalization_status == "abandoned"
    assert abandoned.state is RecorderState.FAILED
    assert abandoned.nrf_committed == 0
    assert abandoned.control_nrf_committed == 0
    assert abandoned.lost_during_finalization == 0
    assert abandoned.sealed is False
    assert abandoned.complete is None
    assert not output.exists()


# --- publication is irreversible; releasing the recorder is not -------------


def test_release_failure_after_publication_does_not_strand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session is sealed, so the recorder must not claim it is still finalizing.

    Releasing the spool is four platform calls after the NRF is published, and
    any of them can fail. The failure is the recorder's own lifecycle -- it
    becomes ``failed`` and says why -- while the finalization outcome stays
    latched at what actually happened, the ownership flag is cleared, and every
    waiter is woken.
    """
    recorder, output = _run(tmp_path)
    recorder.stop("done")
    monkeypatch.setattr(
        type(recorder._impl),
        "close",
        lambda _self: (_ for _ in ()).throw(RuntimeError("could not release the mapped spool")),
    )

    status = recorder.finalize()

    assert status.finalization_status == "succeeded"
    assert status.sealed is True
    assert status.complete is True
    assert status.state is RecorderState.FAILED
    assert "could not release the mapped spool" in status.resource_release_error
    assert output.is_file()
    assert recorder._finalization_owner is None
    # Neither call may block on an owner that will never notify again, and
    # neither may re-decide the finalization.
    assert recorder.close().state is RecorderState.FAILED
    assert recorder.finalize().finalization_status == "succeeded"


def test_release_failure_after_abandoning_keeps_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abandoning is terminal on disk, so it is latched before anything else can fail."""
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    import neurale.recording._finalizer as finalizer

    monkeypatch.setattr(
        finalizer,
        "_publish_session",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("injected publication failure")),
    )
    with pytest.raises(FinalizationError):
        recorder.finalize()
    monkeypatch.setattr(
        type(recorder._impl),
        "close",
        lambda _self: (_ for _ in ()).throw(RuntimeError("could not release the mapped spool")),
    )

    status = recorder.abandon_finalization("operator gave up")

    assert status.finalization_status == "abandoned"
    assert status.state is RecorderState.FAILED
    assert "could not release the mapped spool" in status.resource_release_error
    assert recorder._finalization_owner is None


def test_waiter_wakes_when_finalizer_fails_to_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked ``close()`` must not wait on an owner that died mid-cleanup."""
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    recorder._finalization_wait_seconds = 0.01
    entered = threading.Event()
    release = threading.Event()
    import neurale.recording._session_recorder as facade

    real = facade.finalize_spool

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(facade, "finalize_spool", blocked)
    monkeypatch.setattr(
        type(recorder._impl),
        "close",
        lambda _self: (_ for _ in ()).throw(RuntimeError("could not release the mapped spool")),
    )
    thread = threading.Thread(target=recorder.finalize)
    thread.start()
    assert entered.wait(2.0)
    assert recorder.close().state is RecorderState.FINALIZING
    release.set()
    thread.join(10.0)
    assert not thread.is_alive()

    assert recorder.status.state is RecorderState.FAILED
    assert recorder.status.finalization_status == "succeeded"
    assert recorder.close().state is RecorderState.FAILED


# --- one owner of the staged finalization, whichever operation it is --------
#
# Retrying a finalization and abandoning one rewrite the same staged session
# and the same progress document. Abandoning removes the staged session tree,
# so a retry that checked the state before that decision was taken is a
# finalizer converting into a directory being deleted underneath it. Checking
# the state and then acting on it is not enough: they have to be one exclusive
# operation.


def _fails_then_blocks(
    monkeypatch: pytest.MonkeyPatch, entered: threading.Event, release: threading.Event
) -> None:
    """First finalization fails retryably; the next one blocks until released."""
    import neurale.recording._session_recorder as facade

    real = facade.finalize_spool
    calls = 0

    def staged(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FinalizationError("injected publication failure", category="publication")
        entered.set()
        assert release.wait(10.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(facade, "finalize_spool", staged)


def test_abandoning_waits_for_staging_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state says ``finalization_failed`` right up until a retry starts.

    A caller that read it, decided to abandon, and then acted would be acting
    on a past the retry has already left -- and abandoning deletes the staged
    session the retry is writing into. The refusal here is the serialization
    working: the decision is taken while holding the same ownership the
    finalization holds, so it sees the retry rather than the state it left.
    """
    recorder, output = _run(tmp_path)
    recorder.stop("done")
    entered, release = threading.Event(), threading.Event()
    _fails_then_blocks(monkeypatch, entered, release)
    with pytest.raises(FinalizationError):
        recorder.finalize()
    assert recorder.status.finalization_status == "failed_retryable"
    recorder._finalization_wait_seconds = 0.05

    thread = threading.Thread(target=recorder.finalize)
    thread.start()
    try:
        assert entered.wait(5.0)
        with pytest.raises(RecorderStateError, match="already running"):
            recorder.abandon_finalization("operator gave up")
    finally:
        release.set()
        thread.join(20.0)

    assert not thread.is_alive()
    # The retry the abandonment would have deleted out from under finished.
    assert recorder.status.finalization_status == "succeeded"
    assert output.is_file()


def _block_inside_abandonment(
    monkeypatch: pytest.MonkeyPatch, entered: threading.Event, release: threading.Event
) -> list[str]:
    """Hold the offline abandonment open, and record every reason it is given."""
    import neurale.recording._recovery as recovery

    real = recovery.abandon_finalization
    reasons: list[str] = []

    def blocked(output, *, reason):
        reasons.append(reason)
        entered.set()
        assert release.wait(10.0)
        return real(output, reason=reason)

    monkeypatch.setattr(recovery, "abandon_finalization", blocked)
    return reasons


def _refuse_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    import neurale.recording._finalizer as finalizer

    monkeypatch.setattr(
        finalizer,
        "_publish_session",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("injected publication failure")),
    )


def test_second_abandonment_cannot_rewrite_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abandonment in flight publishes no state a second caller could see.

    Its recorder still reads ``finalization_failed`` until the decision is
    latched, so a second caller checking the state finds the same answer the
    first one did -- and would write a second reason over a terminal decision
    that is already being committed. Waiting for the operation is what makes
    that impossible, not the state check.
    """
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    _refuse_publication(monkeypatch)
    with pytest.raises(FinalizationError):
        recorder.finalize()
    entered, release = threading.Event(), threading.Event()
    reasons = _block_inside_abandonment(monkeypatch, entered, release)

    thread = threading.Thread(target=recorder.abandon_finalization, args=("first reason",))
    thread.start()
    try:
        assert entered.wait(5.0)
        assert recorder.state is RecorderState.FINALIZATION_FAILED
        recorder._finalization_wait_seconds = 0.05
        with pytest.raises(RecorderStateError, match="already running"):
            recorder.abandon_finalization("second reason")
    finally:
        release.set()
        thread.join(20.0)

    assert not thread.is_alive()
    assert reasons == ["first reason"]
    assert recorder.status.finalization_status == "abandoned"
    assert recorder.status.finalization_attempts[-1].category == "publication"


def test_close_waits_for_abandonment_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing must not answer with the state an abandonment is in the middle of leaving.

    ``finalization_failed`` says a retry is still possible, and while an
    abandonment is committing that is exactly what is no longer true. The
    finalization operation is what ``close()`` waits on -- not the
    ``finalizing`` state, which an abandonment never passes through.
    """
    recorder, _ = _run(tmp_path)
    recorder.stop("done")
    _refuse_publication(monkeypatch)
    with pytest.raises(FinalizationError):
        recorder.finalize()
    entered, release = threading.Event(), threading.Event()
    _block_inside_abandonment(monkeypatch, entered, release)
    recorder._finalization_wait_seconds = 20.0

    thread = threading.Thread(target=recorder.abandon_finalization, args=("operator gave up",))
    thread.start()
    try:
        assert entered.wait(5.0)
        threading.Timer(0.1, release.set).start()
        status = recorder.close()
    finally:
        release.set()
        thread.join(20.0)

    assert status.finalization_status == "abandoned"
    assert status.state is RecorderState.FAILED


# --- the shipping path: the tmpfs mapped spool and its sidecar --------------
#
# Everything below runs with `_stable_native_options` left alone, so it is the
# only place the path selection, the sidecar, the artifact lifecycle, and the
# offline handoff are exercised at all. The tests above substitute a private
# memory store to check orchestration where the approved store may not exist,
# and that substitution is exactly why they cannot answer for any of this.


@stable_backend
@pytest.mark.skipif(sys.platform != "win32", reason="Windows persistent-spool contract")
def test_windows_public_create_uses_continuously_written_spool(tmp_path: Path) -> None:
    config = _stable_config(tmp_path / "session.nrf", checkpoint_interval=0)
    config = replace(config, spool_retention="retain")
    recorder, output = _run(tmp_path, config=config)
    spool = recorder.spool_path
    sidecar = _sidecar(spool)
    try:
        assert recorder._impl.options.spool == "file"
        assert spool.parent == tmp_path.absolute()
        assert spool.name.startswith(f"{output.name}.")
        assert spool.name.endswith(".spool")
        assert spool.exists() and spool.stat().st_size < _STABLE_SPOOL_CAPACITY_BYTES
        assert sidecar.exists()

        stopped = recorder.stop("windows-persistent-spool")
        assert stopped.spool_holds_committed_record is True
        closed = recorder.finalize()
        assert closed.state is RecorderState.CLOSED
        assert closed.complete is True
        assert output.exists()
        assert spool.exists() and sidecar.exists()

        offline = tmp_path / "offline.nrf"
        finalize_spool(spool, offline)
        with NrfReader(offline) as reader:
            assert reader.session_id == recorder.session_id
    finally:
        recorder.close()
        sidecar.unlink(missing_ok=True)
        spool.unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_created_recorder_writes_nothing_to_disk(tmp_path: Path) -> None:
    """``created`` means no resources and no session artifact (contract section 3.1).

    The spool path is decided when the recorder is created; the file is not,
    and neither is the sidecar that carries the session metadata.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    try:
        assert recorder.state is RecorderState.CREATED
        assert recorder.spool_path.parent == tmp_path.absolute()
        assert not recorder.spool_path.exists()
        assert not _sidecar(recorder.spool_path).exists()
        assert recorder.status.session_created is False
    finally:
        recorder.close()


@stable_backend
@requires_linux_files
def test_prepare_creates_owner_private_artifacts(tmp_path: Path) -> None:
    """The sidecar is as private as the spool it describes.

    It carries the session and free-form metadata the caller configured --
    subject and experiment identifiers in this test, as in a real session -- so
    a default-umask ``0644`` beside a ``0600`` spool would publish to every
    local account what the recording itself is closed to.
    """
    config = replace(
        _stable_config(tmp_path / "session.nrf"),
        session=RecorderSessionMetadata(subject={"id": "private-subject"}),
        metadata={"experiment": {"name": "private-experiment"}},
    )
    recorder = SessionRecorder.create(config, native_schema())
    spool = recorder.spool_path
    try:
        recorder.prepare()
        assert spool.exists()
        assert _sidecar(spool).exists()
        assert stat.S_IMODE(spool.stat().st_mode) == 0o600
        assert stat.S_IMODE(_sidecar(spool).stat().st_mode) == 0o600
        assert "private-subject" in _sidecar(spool).read_text(encoding="utf-8")
    finally:
        recorder.close()


@stable_backend
@requires_linux_files
def test_closing_created_recorder_leaves_nothing(tmp_path: Path) -> None:
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path

    status = recorder.close()

    assert status.state is RecorderState.CLOSED
    assert status.session_created is False
    assert not spool.exists()
    assert not _sidecar(spool).exists()


@stable_backend
@requires_linux_files
def test_closing_uncommitted_prepared_recorder_discards_both(
    tmp_path: Path,
) -> None:
    """``prepared`` to ``closed`` discards a spool holding no committed record.

    Nothing was recorded, so nothing here is a reconstruction input -- and the
    sidecar describing a session that never happened must not outlive it on a
    tmpfs every local account can read the directory of.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    recorder.prepare()
    assert spool.exists() and _sidecar(spool).exists()

    status = recorder.close()

    assert status.state is RecorderState.CLOSED
    assert status.spool_holds_committed_record is False
    assert not spool.exists()
    assert not _sidecar(spool).exists()
    assert recorder.close().state is RecorderState.CLOSED


@stable_backend
@requires_linux_files
def test_refused_readiness_gate_leaves_no_artifact(tmp_path: Path) -> None:
    """Contract section 3.1: the gate refusing releases everything and discards the spool.

    The gate is driven here without a runtime, which is the only way to refuse
    it deterministically -- its argument is what the runtime knows and the
    recorder does not. What is being checked is the recorder's side of that
    answer, and ``failed`` is defined as holding no recorder-owned resource --
    so it has to be true *when the recorder says ``failed``*, not after a
    further call. Nothing calls the facade when the gate refuses, so the three
    things a caller cannot see from Python are checked directly: the descriptor
    and the artifacts.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    recorder.prepare()
    assert spool.exists()
    assert _descriptors_for(spool)

    refused = recorder._impl._recorder.standalone_pass_readiness_gate(False)

    assert refused == recorder._impl._native.RecorderStatusCode.NOT_READY
    assert recorder.state is RecorderState.FAILED
    # Every one of these holds already, with no close() in between.
    assert _descriptors_for(spool) == []
    assert not spool.exists()
    assert not _sidecar(spool).exists()

    status = recorder.close()
    assert status.state is RecorderState.FAILED
    assert status.session_created is False


@stable_backend
@requires_linux_files
def test_failed_sidecar_write_rolls_back_retryably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ``prepare()`` failed, so section 3.1's answer applies whole.

    Allocating the native resources and writing the sidecar are two halves of
    one public operation, and a caller whose ``prepare()`` raised has no way to
    tell -- or care -- which half it was. The state table gives failure one
    meaning: nothing retained, no artifact, still ``created``, may be prepared
    again. Reporting a terminal ``failed`` here would answer a different
    question, and would take the retry away.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "write_plan_sidecar",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("injected sidecar failure")),
    )
    with pytest.raises(OSError, match="injected sidecar failure"):
        recorder.prepare()

    assert recorder.state is RecorderState.CREATED
    assert _descriptors_for(spool) == []
    assert not spool.exists()
    assert not _sidecar(spool).exists()

    # The part a terminal state would have cost: the same object prepares again.
    monkeypatch.undo()
    try:
        recorder.prepare()
        assert recorder.state is RecorderState.PREPARED
        assert spool.exists()
        assert _sidecar(spool).exists()
    finally:
        recorder.close()
    assert not spool.exists()
    assert not _sidecar(spool).exists()


@stable_backend
@requires_linux_files
def test_failed_prepare_leaves_nothing_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract section 3.1: a failed ``prepare()`` retains no partial resource.

    The failure is real rather than injected into the facade: the plan carries a
    bound the native core refuses, so the binding creates the store, the core
    rejects the plan and rolls itself back, and the store is released without
    being unlinked -- which paths a failed attempt may remove is decided in the
    facade, not in C++. What must not survive is that file, the sidecar, or a
    recorder claiming to be prepared.
    """
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "_stable_native_options",
        lambda plan, output, **kwargs: replace(
            _REAL_STABLE_NATIVE_OPTIONS(plan, output, **kwargs),
            # Refused by `NativeRecordingPlan::validate` as `invalid_bound`, and
            # refused there rather than here, which is what puts the failure
            # after the store has been created.
            max_records_per_transaction=0,
        ),
    )
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path

    with pytest.raises(RecorderConfigError, match="refused the recording plan"):
        recorder.prepare()

    assert recorder.state is RecorderState.CREATED
    assert _descriptors_for(spool) == []
    assert not spool.exists()
    assert not _sidecar(spool).exists()


def _refuse_to_unlink(spool: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this spool's bundle un-removable, and nothing else's."""
    real_unlink = Path.unlink

    def refuse(self: Path, *arguments: object, **keywords: object):
        if self.name.startswith(spool.name):
            raise PermissionError("injected cleanup failure")
        return real_unlink(self, *arguments, **keywords)

    monkeypatch.setattr(Path, "unlink", refuse)


@stable_backend
@requires_linux_files
def test_sidecar_failure_with_failed_cleanup_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback that could not finish may not promise the retry it broke.

    ``created`` after a failed ``prepare()`` is one claim in three parts:
    nothing retained, nothing on disk, and free to be prepared again -- and the
    third holds *because* of the second. A spool this rollback could not unlink
    is sitting at the pathname the next ``prepare()`` must win with ``O_EXCL``,
    so reporting ``created`` would hand the caller a retry that is guaranteed
    to fail on ``EEXIST``. The recorder could not complete its own lifecycle,
    which is terminal, and the caller's own exception still propagates.
    """
    import neurale.recording._session_recorder as facade

    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    try:
        monkeypatch.setattr(
            facade,
            "write_plan_sidecar",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("injected sidecar failure")),
        )
        _refuse_to_unlink(spool, monkeypatch)

        with pytest.raises(OSError, match="injected sidecar failure"):
            recorder.prepare()

        assert recorder.state is RecorderState.FAILED
        status = recorder.status
        assert status.resource_release_error is not None
        assert str(spool) in status.resource_release_error
        assert spool.exists()

        # The retry a `created` recorder is entitled to is refused here, and
        # refusing it changes nothing on disk.
        monkeypatch.undo()
        with pytest.raises(RecorderStateError, match="prepare\\(\\) is legal only"):
            recorder.prepare()
        assert spool.exists()

        first = recorder.status
        assert recorder.stop().state is RecorderState.FAILED
        assert recorder.abort().state is RecorderState.FAILED
        assert recorder.close().state is RecorderState.FAILED
        assert recorder.status.resource_release_error == first.resource_release_error
        assert spool.exists()
    finally:
        monkeypatch.undo()
        spool.unlink(missing_ok=True)
        _sidecar(spool).unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_prepare_failure_with_failed_cleanup_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other entrance to the same cleanup, and it needs the same answer.

    A native ``prepare()`` that creates the store and then refuses the plan
    cleans up through the narrower per-attempt path rather than through the
    rollback, so a fix applied only to the rollback would leave this one
    reporting ``created`` over its own orphan.
    """
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "_stable_native_options",
        lambda plan, output, **kwargs: replace(
            _REAL_STABLE_NATIVE_OPTIONS(plan, output, **kwargs),
            # Refused by the core *after* the store exists, which is what makes
            # this a cleanup path rather than a refusal before anything is made.
            max_records_per_transaction=0,
        ),
    )
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    try:
        _refuse_to_unlink(spool, monkeypatch)

        with pytest.raises(RecorderConfigError, match="refused the recording plan"):
            recorder.prepare()

        assert recorder.state is RecorderState.FAILED
        status = recorder.status
        assert status.resource_release_error is not None
        assert str(spool) in status.resource_release_error
        assert spool.exists()

        monkeypatch.undo()
        with pytest.raises(RecorderStateError, match="prepare\\(\\) is legal only"):
            recorder.prepare()
        assert spool.exists()

        # `failed` is terminal for every operation, not only for the one that
        # reached it: these are legal, idempotent, and decide nothing new. The
        # obstacle is gone by now, so a cleanup re-attempted from any of them
        # would succeed and show up as a missing spool.
        first = recorder.status
        assert recorder.stop().state is RecorderState.FAILED
        assert recorder.abort().state is RecorderState.FAILED
        assert recorder.close().state is RecorderState.FAILED
        assert recorder.status.resource_release_error == first.resource_release_error
        assert spool.exists()
    finally:
        monkeypatch.undo()
        spool.unlink(missing_ok=True)
        _sidecar(spool).unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_disk_capture_does_not_require_memlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import resource

    monkeypatch.setattr(resource, "getrlimit", lambda _which: (0, 0))
    recorder = SessionRecorder.create(
        recorder_config(tmp_path / "session.nrf", streams=stream_specs()), native_schema()
    )
    spool = recorder.spool_path
    try:
        recorder.prepare()
        assert recorder.state is RecorderState.PREPARED
        assert spool.exists()
        assert spool.stat().st_size == 0
    finally:
        recorder.close()


class _LosesTheRaceToAnotherCreator:
    """A native boundary that finds the path taken, staged deterministically.

    It reproduces the one interleaving a real recorder cannot be made to
    perform on demand: the path is free when Python looks, another creator wins
    it, and this recorder's own exclusive create then fails. What the binding
    reports is the fact that matters -- *this* create did not happen -- and the
    facade has nothing else it is allowed to conclude ownership from.
    """

    def __init__(self, spool: Path, options: object) -> None:
        self._spool = spool
        self.options = options

    def prepare(self) -> None:
        self._spool.write_bytes(b"the other creator's spool")
        raise RuntimeError(
            f"could not create the spool file at '{self._spool}' (platform error 17)"
        )

    @property
    def state(self) -> str:
        return "created"

    @property
    def spool_created_by_this_recorder(self) -> bool:
        return False

    @property
    def spool_created_by_last_prepare(self) -> bool:
        return False


@stable_backend
@requires_linux_files
def test_spool_created_during_race_is_not_removed(
    tmp_path: Path,
) -> None:
    """Ownership comes from the exclusive create, not from looking first.

    ``not path.exists()`` and ``therefore it is mine`` are two observations with
    a scheduling window between them, and what can appear inside that window is
    another run's spool -- in the case that produces it, a crashed run's only
    reconstruction input. A recorder that inferred ownership from the check
    would delete exactly that file while cleaning up after the ``prepare()`` the
    file caused to fail.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    recorder._impl = _LosesTheRaceToAnotherCreator(spool, recorder._impl.options)
    try:
        with pytest.raises(RuntimeError, match="platform error 17"):
            recorder.prepare()

        assert spool.read_bytes() == b"the other creator's spool"
    finally:
        spool.unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_foreign_spool_is_never_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leftover spool is the only reconstruction input a crashed run left.

    The store is opened with ``O_EXCL``, so a path that already exists belongs
    to a previous incarnation of this same session. Refusing to start over it is
    correct; tidying it away on the way out would destroy the evidence contract
    section 7 requires be retained.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    spool.write_bytes(b"a crashed run left this here")
    try:
        with pytest.raises(RuntimeError, match="could not create the spool file"):
            recorder.prepare()
        assert spool.read_bytes() == b"a crashed run left this here"
        recorder.close()
        assert spool.read_bytes() == b"a crashed run left this here"
    finally:
        spool.unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_second_prepare_is_refused_and_keeps_spool(
    tmp_path: Path,
) -> None:
    """A refused call is not a failed attempt, and must not be cleaned up after.

    The second ``prepare()`` creates nothing: the spool at the path is the one
    the first call is still holding, mapped, locked, and open. Rolling *that*
    back -- unlinking the spool and its sidecar because a redundant call
    raised -- would destroy a live session's only spool over a caller mistake
    that changed nothing.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    try:
        recorder.prepare()
        prepared = spool.stat().st_ino

        with pytest.raises(RecorderStateError, match=r"prepare\(\) is legal only"):
            recorder.prepare()

        assert recorder.state is RecorderState.PREPARED
        assert spool.exists()
        assert spool.stat().st_ino == prepared
        assert _sidecar(spool).exists()
        # Still the same open, mapped file -- not a path that happens to exist.
        assert _descriptors_for(spool)
    finally:
        recorder.close()
    assert not spool.exists()
    assert not _sidecar(spool).exists()


@stable_backend
@requires_linux_files
def test_retry_after_rollback_keeps_race_winner_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ownership describes one file, and this recorder has already given that file up.

    The rollback path is where a lifetime answer goes stale: the first
    ``prepare()`` did create the spool, and the rollback then removed it. What
    is at the path on the retry is whatever won it next -- here another
    creator -- and a recorder still carrying ``I created this`` from the file it
    deleted would unlink that one too, which is the crashed-run spool contract
    section 7 requires be retained.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "write_plan_sidecar",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("injected sidecar failure")),
    )
    with pytest.raises(OSError, match="injected sidecar failure"):
        recorder.prepare()
    assert recorder.state is RecorderState.CREATED
    assert not spool.exists()
    monkeypatch.undo()

    spool.write_bytes(b"another creator won the path")
    try:
        with pytest.raises(RuntimeError, match="could not create the spool file"):
            recorder.prepare()
        assert spool.read_bytes() == b"another creator won the path"

        # And not on the way out either: closing applies the same retention
        # rule to a path that is no longer this recorder's.
        assert recorder.close().state in (RecorderState.CLOSED, RecorderState.FAILED)
        assert spool.read_bytes() == b"another creator won the path"
    finally:
        spool.unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_failed_discard_is_terminal_and_close_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One terminal close decides one cleanup, and a refused one ends there.

    A recorder that could not remove its own artifacts did not complete its
    lifecycle, so it is ``failed`` and says why -- not ``closed`` with work it
    still intends to do. The second ``close()`` is the point: repeating a call
    may not change what the first one decided, so it must not unlink what the
    first attempt could not, even once the obstacle is gone. What was left
    behind is an orphan for explicit cleanup, and the next creator's ``O_EXCL``
    refuses that path rather than adopting it.
    """
    recorder = SessionRecorder.create(_stable_config(tmp_path / "session.nrf"), native_schema())
    spool = recorder.spool_path
    real_unlink = Path.unlink

    def refuse(self: Path, *arguments: object, **keywords: object):
        if self.name.startswith(spool.name):
            raise PermissionError("injected cleanup failure")
        return real_unlink(self, *arguments, **keywords)

    try:
        recorder.prepare()
        monkeypatch.setattr(Path, "unlink", refuse)

        first = recorder.close()
        assert first.state is RecorderState.FAILED
        assert first.resource_release_error is not None
        assert str(spool) in first.resource_release_error
        assert spool.exists()
        assert _sidecar(spool).exists()
        assert recorder._spool_is_ours is False

        # With the obstacle removed, a retry would succeed -- which is exactly
        # what a repeated call is not allowed to do.
        monkeypatch.undo()
        second = recorder.close()

        assert second.state is first.state
        assert second.resource_release_error == first.resource_release_error
        assert spool.exists()
        assert _sidecar(spool).exists()

        # The leftovers are not silently adoptable: they hold the pathname.
        with pytest.raises(FileExistsError):
            os.close(os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    finally:
        monkeypatch.undo()
        spool.unlink(missing_ok=True)
        _sidecar(spool).unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_nonretryable_failure_retains_refused_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal here means released, not tidied away.

    A finalizer that read this spool and refused it non-retryably has produced
    the one thing offline diagnosis, repair, and quarantine work from, so the
    spool is retained whatever its own committed-record flag says -- forced to
    say ``nothing worth keeping`` here, which is the case the ordinary
    retention rule would discard. The recorder's *own* resources go, because
    ``failed`` is defined as holding none.
    """
    recorder, _ = _run(
        tmp_path, config=_stable_config(tmp_path / "session.nrf", checkpoint_interval=0)
    )
    spool = recorder.spool_path
    try:
        recorder.stop("done")
        assert _descriptors_for(spool)
        import neurale.recording._session_recorder as facade

        monkeypatch.setattr(type(recorder), "_holds_committed_record", lambda _self: False)
        monkeypatch.setattr(
            facade,
            "finalize_spool",
            lambda *_a, **_k: (_ for _ in ()).throw(
                FinalizationError("bad source", category="source", retryable=False)
            ),
        )
        with pytest.raises(FinalizationError, match="bad source"):
            recorder.finalize()

        status = recorder.status
        assert status.state is RecorderState.FAILED
        assert status.finalization_status == "failed"
        assert spool.exists()
        assert _sidecar(spool).exists()
        assert _descriptors_for(spool) == []
    finally:
        recorder.close()
        spool.unlink(missing_ok=True)
        _sidecar(spool).unlink(missing_ok=True)


@stable_backend
@requires_linux_files
def test_shipping_path_hands_spool_to_recovery(
    tmp_path: Path,
) -> None:
    """One pass over the whole public wiring, with nothing substituted.

    Path selection, sidecar creation, capture, public finalization, and the
    offline handoff in one run: the retained spool and the sidecar the facade
    wrote are handed to the offline finalizer exactly as an operator would after
    a crash, and the session it rebuilds carries the plan facts the spool does
    not fingerprint.
    """
    config = replace(
        _stable_config(tmp_path / "session.nrf", checkpoint_interval=0),
        session=RecorderSessionMetadata(subject={"id": "shipping-subject"}),
        spool_retention="retain",
    )
    recorder = SessionRecorder.create(config, native_schema())
    spool = recorder.spool_path
    frames = 8
    source = streaming.SyntheticNativeSource(
        native_schema(), frames, 1, None, None, payload_pattern=True
    )
    runner = streaming.StreamRunner(
        native_schema(),
        _realtime_config(frames, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES),
        source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    try:
        recorder.prepare()
        recorder.attach(runner)
        assert runner.prepare() == streaming.StreamStatus.OK
        assert runner.arm() == streaming.StreamStatus.OK
        assert runner.start() == streaming.StreamStatus.OK
        assert runner.join() == streaming.StreamStatus.OK
        recorder.record_event("shipping-marker")

        stopped = recorder.stop("done")
        assert stopped.spool_holds_committed_record is True
        assert stopped.awaiting_finalization == stopped.spool_committed

        closed = recorder.finalize()
        assert closed.state is RecorderState.CLOSED
        assert closed.complete is True
        assert closed.nrf_committed == closed.spool_committed
        assert closed.lost_during_finalization == 0
        # Retained under contract section 7: the spool a finalized session came
        # from is the only thing an operator can re-run the finalizer over.
        assert spool.exists()
        assert _sidecar(spool).exists()

        offline = tmp_path / "offline.nrf"
        finalize_spool(spool, offline)
        with NrfReader.open(offline) as reader:
            assert reader.manifest["writer"] == {"name": "pyneurale", "version": __version__}
            assert reader.manifest["session"]["subject"]["id"] == "shipping-subject"
            assert reader.complete is True
        with NrfReader.open(config.path) as reader:
            assert {"neural", "cursor", "bandpower"} <= set(reader.stream_ids())
    finally:
        recorder.close()
        spool.unlink(missing_ok=True)
        _sidecar(spool).unlink(missing_ok=True)
