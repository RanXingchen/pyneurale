#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Formal native Center-Out acquisition, decoding, training, and return loop."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from neurale.exceptions import ValidationError
from neurale.pipeline import PipelinePlan, compile_pipeline
from neurale.streaming import RealtimeConfig

from .center_out import CenterOutGuidanceConfig, _with_trial_limit
from .center_out_session import (
    CenterOutProtocol,
    CenterOutTrainingObservation,
    _native_controller_config,
    _native_session_config,
    _resolve_guidance,
    _trace_capacity,
    _training_capture_capacity,
)
from .center_out_training import (
    OnlineDecoderTrainingConfig,
    RollingDecoderTrainer,
    _feature_contract,
    _resolve_decoder_deployment,
    supervision_values,
)

if TYPE_CHECKING:
    from neurale.decoding import KalmanDecoder, LinearDecoder
    from neurale.experiments.presentation import CenterOutPresentationConfig
    from neurale.recording import SessionRecorder

_BRIDGE_POLL_INTERVAL_SECONDS = 0.001
_LIFECYCLE_GRACE_SECONDS = 5.0
_NANOSECONDS_PER_SECOND = 1_000_000_000.0


def _resolve_training_config(
    protocol: CenterOutProtocol,
    training: OnlineDecoderTrainingConfig | None,
) -> OnlineDecoderTrainingConfig:
    if not isinstance(protocol, CenterOutProtocol):
        raise TypeError("protocol must be a CenterOutProtocol")
    if training is None:
        training = OnlineDecoderTrainingConfig()
    elif not isinstance(training, OnlineDecoderTrainingConfig):
        raise TypeError("training must be an OnlineDecoderTrainingConfig or None")
    if protocol.assistance_blocks[0].assistance != 1.0:
        raise ValidationError("the initial online-training block must use 100% assistance.")
    initial_trials = protocol.assistance_blocks[0].trials
    update_interval = training.update_interval or initial_trials
    if update_interval > initial_trials:
        raise ValidationError(
            "update_interval must not exceed the trial count of the initial 100%-assisted block."
        )
    training_window = training.training_window or protocol.trials
    if training_window < update_interval:
        raise ValidationError("training_window must be at least update_interval.")
    return replace(
        training,
        update_interval=update_interval,
        training_window=training_window,
    )


@dataclass(frozen=True, slots=True)
class DecoderPublication:
    """One fitted model published for activation at the next trial boundary."""

    version: int
    plan_fingerprint: str
    training_blocks: int
    training_trials: int
    completed_trials_at_publication: int
    fit_duration_ns: int = 0


@dataclass(frozen=True, slots=True)
class CenterOutIntentSnapshot:
    """Latest target-directed intent published by the native controller."""

    sequence: int
    intent_x: float
    intent_y: float
    context_ordinal: int
    source_time_ns: int
    source_frame_sequence: int
    source_sample_index: int
    trial_key: int
    target_id: int
    valid: bool


@dataclass(frozen=True, slots=True)
class CenterOutOutcome:
    """Terminal experiment outcome plus decoder publication evidence."""

    state: int
    experiment_trace_complete: bool
    trace_losses: int
    producer_trace_drops: int
    abnormal_conditions: int
    abnormal_trials_affected: int
    abnormal_session_aborted: bool
    terminal_abort: bool
    runtime_status: Any
    publications: tuple[DecoderPublication, ...]
    active_decoder_version: int
    completed_trials: int
    training_capture_drops: tuple[int, int]
    presented_frames: int
    presentation_state_drops: int
    recording: Any | None


class CenterOutSession:
    """Run one native realtime Center-Out adaptive-decoder session."""

    __slots__ = (
        "_current_trial",
        "_deployment",
        "_feature_schema",
        "_initial_plan_fingerprint",
        "_lifecycle_deadline_seconds",
        "_native",
        "_presentation_bridge",
        "_presentation_config",
        "_presentation_task",
        "_presented_frames",
        "_presenter",
        "_protocol",
        "_publications",
        "_recording",
        "_resolved_guidance",
        "_runner",
        "_safety_controller",
        "_trainer",
        "_training",
    )

    def __init__(
        self,
        protocol: CenterOutProtocol,
        feature_plan: PipelinePlan,
        device: Any,
        decoder: LinearDecoder | KalmanDecoder,
        *,
        runtime_config: RealtimeConfig | None = None,
        training: OnlineDecoderTrainingConfig | None = None,
        presentation_config: CenterOutPresentationConfig | None = None,
        recording: SessionRecorder | None = None,
    ) -> None:
        resolved_training = _resolve_training_config(protocol, training)
        from neurale._native_loader import load_native_namespace
        from neurale.decoding import KalmanDecoder, LinearDecoder
        from neurale.streaming import (
            ExecutionProfile,
            RecordingNativeSafetyController,
            StreamStatus,
        )

        if not isinstance(feature_plan, PipelinePlan):
            raise TypeError("feature_plan must be a PipelinePlan")
        if not hasattr(device, "schema") or not hasattr(device, "source"):
            raise TypeError("device must provide schema and source")
        if not isinstance(decoder, LinearDecoder | KalmanDecoder):
            raise TypeError("decoder must be a LinearDecoder or KalmanDecoder")
        if runtime_config is not None and not isinstance(runtime_config, RealtimeConfig):
            raise TypeError("runtime_config must be a RealtimeConfig or None")
        if recording is not None:
            from neurale.recording import SessionRecorder

            if not isinstance(recording, SessionRecorder):
                raise TypeError("recording must be a SessionRecorder or None")
        input_schema = device.schema
        source = device.source
        resolved_runtime = RealtimeConfig() if runtime_config is None else runtime_config
        presentation_task = _with_trial_limit(protocol.task, protocol.trials)
        resolved_presentation_config = None
        if presentation_config is not None:
            from neurale.experiments.presentation import CenterOutPresentationConfig
            from neurale.experiments.presentation._center_out import (
                _resolve_center_out_presentation,
            )

            if not isinstance(presentation_config, CenterOutPresentationConfig):
                raise TypeError("presentation_config must be a CenterOutPresentationConfig or None")
            resolved_presentation_config = _resolve_center_out_presentation(
                presentation_task, presentation_config
            )
        feature_pipeline = compile_pipeline(feature_plan)
        feature_schema = feature_pipeline.output_schema(input_schema)
        feature_signal, feature_descriptor, _ = _feature_contract(feature_schema)
        _label_lag_samples(
            resolved_training.label_lag_seconds,
            feature_signal.fs.numerator / feature_signal.fs.denominator,
        )
        deployment = _resolve_decoder_deployment(input_schema, feature_schema, feature_plan.stages)
        initial_plan = PipelinePlan(stages=(deployment.initial_stage(feature_schema),))
        initial_pipeline = compile_pipeline(initial_plan)
        native = load_native_namespace("experiments.center_out")
        guidance, resolver_version = _resolve_guidance(protocol, feature_schema)
        controller = _native_controller_config(
            native,
            protocol,
            feature_schema,
            decoded_signal_id=deployment.output_signal_id,
            guidance=guidance,
            guidance_resolver_version=resolver_version,
            training_capture_capacity=_training_capture_capacity(
                protocol.task,
                feature_schema,
                protocol.trials,
            ),
            presentation_state_capacity=(
                _trace_capacity(
                    protocol.task,
                    feature_schema,
                    protocol.trials,
                )
                if presentation_config is not None
                else 0
            ),
        )
        session_config = _native_session_config(native, feature_schema)
        lifecycle_deadline = _lifecycle_deadline_seconds(
            protocol, feature_signal, feature_descriptor, resolved_runtime
        )
        if recording is not None:
            recording._budget_experiment(
                duration_seconds=lifecycle_deadline,
                control_records=_recording_control_budget(
                    protocol,
                    feature_signal,
                    lifecycle_deadline,
                    presentation=presentation_config is not None,
                ),
            )
            session_config.max_control_body_bytes = recording._experiment_max_control_body_bytes
        session = native._AdaptiveCenterOutSession()
        status = session.prepare(
            feature_schema,
            initial_pipeline._native,
            1,
            controller,
            session_config,
        )
        if status != StreamStatus.OK:
            raise RuntimeError(f"failed to prepare adaptive Center-Out session: {status}")
        safety_controller = RecordingNativeSafetyController()
        runner = feature_pipeline.create_runner(
            input_schema,
            resolved_runtime,
            source,
            session.actuator,
            profile=ExecutionProfile.REALTIME,
            safety_controller=safety_controller,
        )
        self._protocol = protocol
        self._current_trial: list[CenterOutTrainingObservation] = []
        self._deployment = deployment
        self._feature_schema = feature_schema
        self._initial_plan_fingerprint = initial_plan.fingerprint
        self._lifecycle_deadline_seconds = lifecycle_deadline
        self._native = session
        self._presentation_bridge: Any | None = None
        self._presentation_config = resolved_presentation_config
        self._presentation_task = presentation_task
        self._presented_frames = 0
        self._presenter: Any | None = None
        if resolved_presentation_config is not None:
            from neurale.experiments.presentation._native import load_presentation_extension

            presentation = load_presentation_extension()
            self._presenter = presentation.CenterOutPresenter()
            self._presentation_bridge = presentation.CenterOutSessionPresentationBridge(session)
        self._publications: list[DecoderPublication] = []
        self._recording = recording
        self._resolved_guidance = guidance
        self._runner = runner
        self._safety_controller = safety_controller
        self._training = resolved_training
        self._trainer = RollingDecoderTrainer(decoder, resolved_training, initial_version=1)
        if recording is not None:
            attachment = recording._prepare_experiment_attachment()
            status = session._enable_recording(attachment, recording)
            if status != StreamStatus.OK:
                recording._rollback_experiment_attachment()
                self._trainer.close()
                raise RuntimeError(f"failed to enable Center-Out recording: {status}")
            try:
                recording._attach_experiment(runner)
            except BaseException:
                recording._rollback_experiment_attachment()
                self._trainer.close()
                raise

    @property
    def snapshot(self) -> Any:
        return self._native.snapshot

    @property
    def position(self) -> Any:
        return self._native.position

    def get_intent(self) -> CenterOutIntentSnapshot:
        """Return the latest native task-intent snapshot for diagnostics."""

        return CenterOutIntentSnapshot(**self._native.get_intent())

    @property
    def intent_source(self) -> object:
        """Native read capability accepted by intent-driven simulation devices."""

        return self._native.intent_source

    @property
    def active_decoder_version(self) -> int:
        return int(self._native.active_decoder_version)

    @property
    def publications(self) -> tuple[DecoderPublication, ...]:
        return tuple(self._publications)

    @property
    def resolved_guidance(self) -> CenterOutGuidanceConfig:
        """Concrete guidance used by the native controller."""

        return self._resolved_guidance

    def run(self) -> CenterOutOutcome:
        """Run acquisition and control while fitting models off the data plane."""

        from neurale.streaming import RuntimeState, StreamStatus

        start_ns = 0
        runner = self._runner
        self._open_presentation(start_ns)
        runner._validate_profile_contract()
        status = self._native.attach_runtime(runner._runner)
        if status != StreamStatus.OK:
            self._trainer.close()
            raise RuntimeError(f"failed to attach runtime to Center-Out session: {status}")
        if not self._native._record_decoder_publication(
            1, self._initial_plan_fingerprint, 0, 0, 0, start_ns
        ):
            self._native.close()
            self._trainer.close()
            raise RuntimeError("failed to record initial decoder publication")
        host_epoch_ns = time.monotonic_ns()
        if not self._native._set_host_epoch(host_epoch_ns):
            self._native.close()
            self._trainer.close()
            raise RuntimeError("failed to set the Center-Out host epoch before start")
        status = self._native.start(start_ns)
        if status != StreamStatus.OK:
            self._native.close()
            self._trainer.close()
            raise RuntimeError(f"failed to start Center-Out session: {status}")

        ended = StreamStatus.OK
        recording_status = None
        wall_start = time.monotonic()
        task_complete = False
        abort_reason = "closed-loop-failure"
        presentation_closed = False
        try:
            while runner.state == RuntimeState.RUNNING:
                try:
                    self._drain_training()
                    if not self._native.complete:
                        self._publish_ready_candidate(
                            start_ns + max(0, time.monotonic_ns() - host_epoch_ns)
                        )
                except BaseException:
                    abort_reason = "online-training-failure"
                    raise
                try:
                    self._pump_presentation()
                except BaseException:
                    abort_reason = "presentation-failure"
                    raise
                if self._native.complete:
                    task_complete = True
                    runner.stop()
                    break
                if time.monotonic() - wall_start > self._lifecycle_deadline_seconds:
                    abort_reason = "lifecycle-timeout"
                    raise RuntimeError(
                        "Center-Out closed loop exceeded its internally derived lifecycle "
                        "deadline: "
                        f"state={runner.state}, frames={runner.stats.frames_consumed}, "
                        f"completed_trials={self._native.completed_trials}"
                    )
                time.sleep(_BRIDGE_POLL_INTERVAL_SECONDS)
            joined = runner.join()
            if joined != StreamStatus.OK:
                # A failed runtime cannot accept a final decoder. Preserve its
                # original status even when abort itself completes successfully.
                ended = joined
                self._native.abort(
                    start_ns + max(0, time.monotonic_ns() - host_epoch_ns),
                    "runtime-failure",
                )
            else:
                self._drain_training()
                self._publish_ready_candidate(
                    start_ns + max(0, time.monotonic_ns() - host_epoch_ns),
                    wait=True,
                    latest=True,
                )
                self._pump_presentation()
                task_complete = task_complete or bool(self._native.complete)
                end_ns = start_ns + max(0, time.monotonic_ns() - host_epoch_ns)
                ended = self._native.stop(
                    end_ns, "task-complete" if task_complete else "source-eos"
                )
            # Finish presentation evidence before closing the window. Offline NRF
            # finalization may take minutes and does not pump window events.
            if self._presenter is not None:
                self._presenter.close()
                presentation_closed = True
            if self._recording is not None:
                recording_status = self._recording.finalize()
        except BaseException:
            end_ns = start_ns + max(0, time.monotonic_ns() - host_epoch_ns)
            self._native.abort(end_ns, abort_reason)
            if self._presenter is not None and not presentation_closed:
                self._presenter.close()
                presentation_closed = True
            if self._recording is not None:
                self._recording.finalize()
            raise
        finally:
            self._trainer.close()
            self._native.close()
            if self._presenter is not None and not presentation_closed:
                self._presenter.close()
        if ended != StreamStatus.OK:
            fault = runner.primary_fault
            runtime_fault_detail = (
                None
                if fault is None
                else {
                    "code": fault.code,
                    "status": fault.status,
                    "stage": fault.stage,
                    "detail": fault.detail,
                }
            )
            recorder_detail = None
            if self._recording is not None:
                native_status = self._recording._impl.status
                recorder_detail = {
                    "state": native_status.state,
                    "runtime_accepted": native_status.runtime_accepted,
                    "recorder_accepted": native_status.recorder_accepted,
                    "control_offered": native_status.control_offered,
                    "control_accepted": native_status.control_accepted,
                    "fault_present": native_status.primary_fault.present,
                    "fault_origin": native_status.primary_fault.origin,
                    "fault_reason": native_status.primary_fault.reason,
                    "fault_stream_status": native_status.primary_fault.stream_status,
                }
            raise RuntimeError(
                f"failed to end Center-Out session: {ended}; outcome={self._native.outcome}; "
                f"runtime_fault={runtime_fault_detail}; recording={recorder_detail}"
            )
        drops = tuple(int(value) for value in self._native.training_capture_drops)
        if drops != (0, 0):
            raise RuntimeError(f"closed-loop training capture was incomplete: drops={drops}")
        if self._current_trial:
            raise RuntimeError(
                "the source ended with an incomplete Center-Out training trial: "
                f"observations={len(self._current_trial)}, "
                f"last_sample_idx={self._current_trial[-1].sample_idx}, "
                f"completed_trials={self._native.completed_trials}"
            )
        presentation_drops = int(self._native.presentation_state_drops)
        if presentation_drops:
            raise RuntimeError(
                f"closed-loop presentation state handoff was incomplete: drops={presentation_drops}"
            )
        native_outcome = self._native.outcome
        return CenterOutOutcome(
            state=int(native_outcome["state"]),
            experiment_trace_complete=bool(native_outcome["experiment_trace_complete"]),
            trace_losses=int(native_outcome["trace_losses"]),
            producer_trace_drops=int(native_outcome["producer_trace_drops"]),
            abnormal_conditions=int(native_outcome["abnormal_conditions"]),
            abnormal_trials_affected=int(native_outcome["abnormal_trials_affected"]),
            abnormal_session_aborted=bool(native_outcome["abnormal_session_aborted"]),
            terminal_abort=bool(native_outcome["terminal_abort"]),
            runtime_status=native_outcome["runtime_status"],
            publications=tuple(self._publications),
            active_decoder_version=int(self._native.active_decoder_version),
            completed_trials=int(self._native.completed_trials),
            training_capture_drops=drops,
            presented_frames=self._presented_frames,
            presentation_state_drops=presentation_drops,
            recording=recording_status,
        )

    def _open_presentation(self, start_ns: int) -> None:
        if self._presenter is None:
            return
        from neurale.experiments.presentation import (
            PresentationRuntimeStatus,
            renderer_monotonic_now_ns,
        )

        status = self._presenter.open(
            self._presentation_task,
            self._presentation_config,
            renderer_monotonic_now_ns(),
            start_ns,
        )
        if status != PresentationRuntimeStatus.OK:
            self._presenter.close()
            raise RuntimeError(f"failed to open Center-Out presentation: {status}")
        if not self._presentation_bridge.record_config(self._presenter):
            self._presenter.close()
            raise RuntimeError("failed to record Center-Out presentation configuration")

    def _pump_presentation(self) -> None:
        if self._presenter is None:
            return
        from neurale.experiments.presentation import PresentationRuntimeStatus

        status = self._presenter.pump_events()
        if status != PresentationRuntimeStatus.OK:
            self._report_presentation_failure(status)
            raise RuntimeError(f"Center-Out presentation event pump failed: {status}")
        while True:
            status, control = self._presenter.poll_control()
            if status != PresentationRuntimeStatus.OK:
                self._report_presentation_failure(status)
                raise RuntimeError(f"Center-Out presentation input failed: {status}")
            if control is None:
                break
            raise RuntimeError(f"Center-Out presentation requested stop: {control.kind}")

        latest = None
        while True:
            value = self._native.pop_presentation_state()
            if value is None:
                break
            latest = value
        if latest is None:
            return
        status = self._presenter.update(latest["snapshot"], latest["cursor"])
        if status != PresentationRuntimeStatus.OK:
            self._presentation_bridge.report_failure(
                self._presenter, int(latest["time_ns"]), status, latest["snapshot"].trial
            )
            raise RuntimeError(f"failed to update Center-Out presentation: {status}")
        times = self._presenter.render(int(latest["time_ns"]), int(latest["time_ns"]))
        if not self._presentation_bridge.report(
            self._presenter,
            times,
            int(latest["source_ordinal"]),
            latest["snapshot"].trial,
        ):
            raise RuntimeError("failed to record Center-Out presentation evidence")
        if times.status != PresentationRuntimeStatus.OK:
            raise RuntimeError(f"failed to render Center-Out presentation: {times.status}")
        self._presented_frames += 1

    def _report_presentation_failure(self, status: Any) -> None:
        latest = self._native.pop_presentation_state()
        if latest is None:
            return
        self._presentation_bridge.report_failure(
            self._presenter, int(latest["time_ns"]), status, latest["snapshot"].trial
        )

    def _drain_training(self) -> None:
        while True:
            value = self._native.pop_training_observation()
            if value is None:
                return
            observation = CenterOutTrainingObservation(
                sample_idx=int(value["sample_idx"]),
                features=tuple(float(item) for item in value["features"]),
                time_ns=int(value["time_ns"]),
                trial=value["trial"],
                target_position=tuple(float(item) for item in value["target_position"]),
                cursor_position=tuple(float(item) for item in value["cursor_position"]),
                cursor_velocity=tuple(float(item) for item in value["cursor_velocity"]),
                guidance_velocity=tuple(float(item) for item in value["guidance_velocity"]),
                trial_stop=bool(value["trial_stop"]),
            )
            if (
                not observation.trial_stop
                and int(self._native.completed_trials) >= self._protocol.trials
            ):
                continue
            self._current_trial.append(observation)
            if observation.trial_stop:
                self._submit_trial()

    def _submit_trial(self) -> None:
        X, y = _training_segment(tuple(self._current_trial), self._feature_schema, self._training)
        self._current_trial.clear()
        self._trainer.submit_trial(X, y)

    def _publish_ready_candidate(
        self,
        publication_time_ns: int,
        *,
        wait: bool = False,
        latest: bool = False,
    ) -> None:
        from neurale.streaming import StreamStatus

        if wait and not self._trainer.candidate_pending:
            return
        active = self._trainer.active
        if (
            not latest
            and active is not None
            and int(self._native.active_decoder_version) < active.version
        ):
            return
        if latest:
            candidate = self._trainer.wait_latest_candidate()
        elif wait:
            candidate = self._trainer.wait_candidate()
        else:
            candidate = self._trainer.poll_candidate()
        if candidate is None:
            return
        stage = self._deployment.candidate_stage(candidate.decoder, self._feature_schema)
        plan = PipelinePlan(stages=(stage,))
        pipeline = compile_pipeline(plan)
        prepare = (
            self._native._prepare_final_candidate if latest else self._native.prepare_candidate
        )
        status = prepare(pipeline._native, candidate.version)
        if status != StreamStatus.OK:
            self._trainer.reject_candidate()
            raise RuntimeError(f"failed to publish decoder candidate: {status}")
        completed_trials = int(self._native.completed_trials)
        if not self._native._record_decoder_publication(
            candidate.version,
            plan.fingerprint,
            candidate.training_blocks,
            candidate.training_trials,
            completed_trials,
            publication_time_ns,
        ):
            self._trainer.reject_candidate()
            raise RuntimeError("failed to record decoder candidate publication")
        self._trainer.activate_candidate()
        self._publications.append(
            DecoderPublication(
                version=candidate.version,
                plan_fingerprint=plan.fingerprint,
                training_blocks=candidate.training_blocks,
                training_trials=candidate.training_trials,
                completed_trials_at_publication=completed_trials,
                fit_duration_ns=candidate.fit_duration_ns,
            )
        )


def _training_segment(
    observations: tuple[CenterOutTrainingObservation, ...],
    feature_schema: Any,
    training: OnlineDecoderTrainingConfig,
) -> tuple[Any, Any]:
    import numpy as np

    from neurale.data import FeatureMatrix, SignalArray

    if not observations:
        raise RuntimeError("cannot fit an empty Center-Out trial")
    signal, descriptor, contract = _feature_contract(feature_schema)
    rate = signal.fs.numerator / signal.fs.denominator
    lag_samples = _label_lag_samples(training.label_lag_seconds, rate)
    if lag_samples >= len(observations):
        raise ValidationError(
            "label_lag_seconds must leave at least one aligned observation in each trial."
        )
    feature_observations = observations if lag_samples == 0 else observations[:-lag_samples]
    label_observations = observations[lag_samples:]
    time_values = np.arange(len(feature_observations), dtype=np.float64) / rate
    time_values += observations[0].time_ns / 1_000_000_000.0
    X = FeatureMatrix(
        data=np.asarray([value.features for value in feature_observations], dtype=np.float64),
        fs=rate,
        time=time_values,
        feature_names=list(descriptor.feature_names),
        source_signal=descriptor.source_stream,
        window_size=contract.window_length_ns / 1_000_000_000.0,
        shift=contract.shift_ns / 1_000_000_000.0,
        unit=list(contract.feature_unit_symbols),
    )
    labels = [
        supervision_values(
            training.target,
            target_position=value.target_position,
            guidance_velocity=value.guidance_velocity,
        )
        for value in label_observations
    ]
    y = SignalArray.from_array(
        np.asarray(labels, dtype=np.float64),
        fs=rate,
        time=time_values.copy(),
        channel_names=("x_velocity", "y_velocity"),
        channel_types="behavior",
        units="1",
        name="center_out_intended_velocity",
    )
    return X, y


def _label_lag_samples(label_lag_seconds: float, rate: float) -> int:
    exact = label_lag_seconds * rate
    samples = round(exact)
    if not math.isclose(exact, samples, rel_tol=0.0, abs_tol=1e-9):
        raise ValidationError(
            f"label_lag_seconds must be an integer number of feature observations at {rate:g} Hz."
        )
    return samples


def _recording_control_budget(
    protocol: CenterOutProtocol, feature_signal: Any, duration_seconds: float, *, presentation: bool
) -> int:
    from .center_out import MAX_STEP_EVENTS, MAX_STEP_TRANSITIONS

    observations = (
        math.ceil(duration_seconds * feature_signal.fs.numerator / feature_signal.fs.denominator)
        + 1
    )
    # Six observation records, an active target and trial record, plus the
    # machine's bounded events/transitions. Allow two steps per observation
    # (segment start and normal step) and presentation request/outcome evidence.
    per_observation = 8 + 2 * (MAX_STEP_EVENTS + MAX_STEP_TRANSITIONS)
    if presentation:
        per_observation += 4
    return observations * per_observation + 128 * protocol.trials + 256


def _lifecycle_deadline_seconds(
    protocol: CenterOutProtocol,
    feature_signal: Any,
    feature_descriptor: Any,
    runtime_config: RealtimeConfig,
) -> float:
    movement_to_center, movement_to_out = map(float, protocol.task.movement_timeout_seconds)
    reward_to_center, reward_to_out = map(float, protocol.task.reward_dwell_seconds)
    punish_to_center, punish_to_out = map(float, protocol.task.punish_dwell_seconds)
    trial_seconds = movement_to_center + max(
        punish_to_center,
        reward_to_center + movement_to_out + max(reward_to_out, punish_to_out),
    )
    observation_period = feature_signal.fs.denominator / feature_signal.fs.numerator
    feature_warmup = feature_descriptor.window_length_ns / _NANOSECONDS_PER_SECOND
    observation_slack = (2 * protocol.trials + 1) * observation_period
    shutdown_slack = max(
        _LIFECYCLE_GRACE_SECONDS,
        2.0 * runtime_config.source_timeout_seconds,
    )
    return feature_warmup + protocol.trials * trial_seconds + observation_slack + shutdown_slack


__all__ = [
    "CenterOutIntentSnapshot",
    "CenterOutOutcome",
    "CenterOutSession",
    "DecoderPublication",
]
