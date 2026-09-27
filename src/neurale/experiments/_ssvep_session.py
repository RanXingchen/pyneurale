# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Calibrate and evaluate SSVEP decoding using the native streaming runtime."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurale.decoding import ClassificationDecoder
    from neurale.pipeline import PipelinePlan
    from neurale.recording import SessionRecorder
    from neurale.streaming import RealtimeConfig

    from .presentation import SSVEPDisplayConfig
    from .ssvep import SSVEPTask, SSVEPTrial


@dataclass(frozen=True, slots=True)
class SSVEPProtocol:
    """Balanced labelled calibration followed by fixed-model evaluation."""

    task: SSVEPTask
    calibration_trials_per_target: int = 3
    evaluation_trials: int = 8

    def __post_init__(self) -> None:
        from .ssvep import SSVEPTask

        if not isinstance(self.task, SSVEPTask):
            raise TypeError("task must be an SSVEPTask")
        for name, minimum in (("calibration_trials_per_target", 2), ("evaluation_trials", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")

    @property
    def calibration_trials(self) -> int:
        return len(self.task.targets) * self.calibration_trials_per_target

    @property
    def trials(self) -> int:
        return self.calibration_trials + self.evaluation_trials


@dataclass(frozen=True, slots=True)
class SSVEPOutcome:
    """Calibration trials are labelled observations, not decoded successes."""

    trials: tuple[SSVEPTrial, ...]
    calibration_trials: tuple[SSVEPTrial, ...]
    evaluation_trials: tuple[SSVEPTrial, ...]
    correct_count: int
    decoder_version: int
    decoder_fingerprint: str | None
    frames: int
    max_frame_interval: float
    expired_presentation_frames: int
    stopped: bool
    experiment_trace_complete: bool
    trace_drops: int
    runtime_status: object
    environment: object | None
    recording_status: object | None


class SSVEPSession:
    """Native acquisition, features, trial aggregation, decoding and task feedback.

    Calibration is fitted once outside realtime threads. Evaluation uses a
    frozen native classifier. ``run()`` owns recording and window lifecycles.
    """

    def __init__(
        self,
        protocol: SSVEPProtocol,
        feature_plan: PipelinePlan,
        device: object,
        decoder: ClassificationDecoder,
        *,
        runtime_config: RealtimeConfig | None = None,
        presentation_config: SSVEPDisplayConfig | None = None,
        recording: SessionRecorder | None = None,
    ) -> None:
        from neurale._native_loader import load_native_namespace
        from neurale.decoding import ClassificationDecoder
        from neurale.pipeline import PipelinePlan, compile_pipeline
        from neurale.pipeline._deployment import require_native_decoder
        from neurale.recording import SessionRecorder
        from neurale.streaming import (
            ExecutionProfile,
            RealtimeConfig,
            RecordingNativeSafetyController,
            StreamStatus,
        )

        from .presentation import SSVEPDisplayConfig

        if not isinstance(protocol, SSVEPProtocol):
            raise TypeError("protocol must be an SSVEPProtocol")
        if not isinstance(feature_plan, PipelinePlan):
            raise TypeError("feature_plan must be a PipelinePlan")
        if not isinstance(decoder, ClassificationDecoder) or decoder.is_fitted:
            raise ValueError("decoder must be an unfitted ClassificationDecoder")
        require_native_decoder(decoder)
        if runtime_config is not None and not isinstance(runtime_config, RealtimeConfig):
            raise TypeError("runtime_config must be a RealtimeConfig or None")
        if presentation_config is not None and not isinstance(
            presentation_config, SSVEPDisplayConfig
        ):
            raise TypeError("presentation_config must be an SSVEPDisplayConfig or None")
        if recording is not None and not isinstance(recording, SessionRecorder):
            raise TypeError("recording must be a SessionRecorder or None")
        pipeline = compile_pipeline(feature_plan)
        schema = pipeline.output_schema(device.schema)
        native = load_native_namespace("experiments.ssvep")
        self._native = native._SSVEPSession(
            schema, protocol.task, protocol.calibration_trials, protocol.trials
        )
        self._safety = RecordingNativeSafetyController()
        self._runner = pipeline.create_runner(
            device.schema,
            runtime_config or RealtimeConfig(),
            device.source,
            self._native.actuator,
            profile=ExecutionProfile.REALTIME,
            safety_controller=self._safety,
        )
        task = protocol.task
        duration = protocol.trials * (
            task.cue_duration
            + task.stimulation_duration
            + task.decision_timeout
            + task.feedback_duration
            + task.inter_trial
        )
        self._deadline = duration * 2 + 60
        if recording is not None:
            recording._budget_experiment(
                duration_seconds=self._deadline,
                control_records=(
                    int(self._deadline * 1e9 / schema.feature_sets[0].shift_ns + protocol.trials)
                    * schema.signals[0].n_channels
                    + protocol.trials * 100
                    + 1000
                    + (int(self._deadline * 1000) if presentation_config is not None else 0)
                ),
            )
            attachment = recording._prepare_experiment_attachment()
            try:
                status = self._native._enable_recording(attachment, recording)
                if status != StreamStatus.OK:
                    raise RuntimeError(f"failed to attach SSVEP recording: {status}")
                recording._attach_experiment(self._runner)
            except BaseException:
                recording._rollback_experiment_attachment()
                raise
        self._protocol, self._decoder, self._device = protocol, decoder, device
        self._pipeline, self._schema = pipeline, schema
        self._presentation_config, self._recording = presentation_config, recording
        self._stop = Event()
        self._started = self._published = False
        self._fingerprint = None

    def stop(self) -> None:
        """Request orderly termination; callable from another thread."""
        self._stop.set()

    def _fit(self):
        import numpy as np

        from neurale.data import FeatureMatrix
        from neurale.decoding import ClassificationTarget
        from neurale.pipeline import PipelinePlan, compile_pipeline, decoder_stage

        rows = np.asarray(self._native.features, dtype=np.float64)
        trials = self._native.trials
        if len(rows) != self._protocol.calibration_trials or len(trials) != len(rows):
            raise RuntimeError("SSVEP calibration did not produce every trial feature vector")
        times = np.array([value - self._epoch for value in self._native.feature_times_ns]) / 1e9
        features = FeatureMatrix(
            data=rows,
            fs=None,
            time=times,
            feature_names=self._schema.feature_sets[0].feature_names,
        )
        labels = np.array([trial.record.trial.target_id for trial in trials])
        self._decoder.fit(features, ClassificationTarget(labels=labels, time=times))
        plan = PipelinePlan((decoder_stage(self._decoder, self._native.feature_schema),))
        return compile_pipeline(plan)

    def run(self) -> SSVEPOutcome:
        """Run once, calibrate once, then return fixed-model evaluation results."""
        from neurale.streaming import RuntimeState, StreamStatus

        from .presentation import SSVEPDisplay
        from .ssvep import SSVEPMachine

        if self._started:
            raise RuntimeError("SSVEPSession can only run once")
        self._started = True
        display = None
        frames, max_gap = 0, 0
        expired_frames = 0
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ssvep-calibration")
        fit = None
        last_swap = environment = recording_status = None
        epoch = time.perf_counter_ns()
        snapshot = SSVEPMachine().snapshot()
        try:
            if self._presentation_config is not None:
                display = SSVEPDisplay(self._protocol.task, self._presentation_config)
                environment = display.environment
            self._runner._validate_profile_contract()
            status = self._native.attach(self._runner._runner)
            if status != StreamStatus.OK:
                raise RuntimeError(f"failed to attach SSVEP runtime: {status}")
            epoch = time.perf_counter_ns()
            self._epoch = epoch
            status = self._native.start(epoch)
            if status != StreamStatus.OK:
                raise RuntimeError(f"failed to start SSVEP runtime: {status}")
            while self._runner.state == RuntimeState.RUNNING:
                while (next_snapshot := self._native.pop_snapshot()) is not None:
                    snapshot = next_snapshot
                if self._native.training and fit is None:
                    fit = executor.submit(self._fit)
                if fit is not None and fit.done() and not self._published:
                    decoder_pipeline = fit.result()
                    if not self._stop.is_set():
                        plan = decoder_pipeline.plan
                        self._native.publish(
                            decoder_pipeline._native,
                            json.dumps(
                                plan.to_document(), ensure_ascii=True, separators=(",", ":")
                            ),
                            plan.fingerprint,
                            time.perf_counter_ns() - epoch,
                        )
                        self._fingerprint = plan.fingerprint
                        self._published = True
                if self._native.complete or self._stop.is_set():
                    break
                if display is not None:
                    display.poll()
                    if display.should_close:
                        self._stop.set()
                        break
                    now = time.perf_counter_ns() - epoch
                    timing, expired = display._render_session(snapshot, now, epoch)
                    expired_frames += expired
                    if not self._native.report_display(
                        snapshot, now, timing.submitted_ns, timing.presented_ns, expired
                    ):
                        raise RuntimeError("SSVEP presentation evidence queue overflow")
                    frames += 1
                    if last_swap is not None:
                        max_gap = max(max_gap, timing.presented_ns - last_swap)
                    last_swap = timing.presented_ns
                else:
                    time.sleep(0.001)
                if (time.perf_counter_ns() - epoch) / 1e9 > self._deadline:
                    raise RuntimeError("SSVEP exceeded its internally derived lifecycle deadline")
            self._runner.stop()
            status = self._runner.join()
            if status != StreamStatus.OK:
                raise RuntimeError(f"SSVEP runtime failed: {status}; {self._runner.primary_fault}")
            if not self._native.complete and not self._stop.is_set():
                raise RuntimeError("SSVEP acquisition ended before all trials completed")
            status = self._native.stop(time.perf_counter_ns() - epoch)
            if status != StreamStatus.OK:
                raise RuntimeError(f"failed to stop SSVEP session: {status}")
            if display is not None:
                display.close()
                display = None
            if self._recording is not None:
                recording_status = self._recording.finalize()
        except BaseException:
            self._native.abort(time.perf_counter_ns() - epoch)
            if display is not None:
                display.close()
                display = None
            if self._recording is not None:
                self._recording.finalize()
            raise
        finally:
            self._native.close()
            executor.shutdown(wait=True, cancel_futures=True)
            if display is not None:
                display.close()
        outcome = self._native.outcome
        if self._native.drops or not outcome["experiment_trace_complete"]:
            raise RuntimeError(f"SSVEP trace is incomplete: {outcome}")
        from . import TrialOutcome

        trials = tuple(self._native.trials)
        split = self._protocol.calibration_trials
        evaluation = trials[split:]
        return SSVEPOutcome(
            trials,
            trials[:split],
            evaluation,
            sum(trial.record.outcome == TrialOutcome.SUCCESS for trial in evaluation),
            2 if self._published else 0,
            self._fingerprint,
            frames,
            max_gap / 1e9,
            expired_frames,
            self._stop.is_set(),
            outcome["experiment_trace_complete"],
            self._native.drops,
            outcome["runtime_status"],
            environment,
            recording_status,
        )
