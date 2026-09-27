#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

import neurale.experiments.closed_loop as closed_loop
import neurale.streaming as streaming
from neurale.data import FeatureMatrix, SignalArray
from neurale.decoding import KalmanDecoder, LinearDecoder
from neurale.exceptions import ValidationError
from neurale.experiments import center_out as co
from neurale.experiments.center_out_session import (
    CenterOutTrainingObservation,
)
from neurale.experiments.center_out_training import (
    AssistanceBlock,
    OnlineDecoderTrainingConfig,
    RollingDecoderTrainer,
    _resolve_decoder_deployment,
)
from neurale.experiments.closed_loop import (
    _label_lag_samples,
    _lifecycle_deadline_seconds,
    _resolve_training_config,
    _training_segment,
)
from neurale.pipeline import KalmanDecoderStage, LinearDecoderStage


def _input_schema() -> object:
    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT64,
        2,
        1,
        1,
        streaming.RationalRate(100, 1),
        1,
        channel_set_id=1,
    )
    return streaming.StreamSchema(
        1,
        [signal],
        [],
        [streaming.UnitDescriptor(1, "1", "dimensionless")],
    )


def _schema() -> object:
    feature = streaming.SignalSchema(
        2,
        streaming.SignalDType.FLOAT64,
        2,
        1,
        1,
        streaming.RationalRate(100, 1),
        1,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=2,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    descriptor = streaming.FeatureSetDescriptor(
        2,
        ["x", "y"],
        [2, 2],
        1,
        "simulated",
        20_000_000,
        10_000_000,
        algorithm_name="lmp",
        algorithm_version="1",
    )
    return streaming.StreamSchema(
        2,
        [feature],
        [descriptor],
        [streaming.UnitDescriptor(2, "1", "dimensionless")],
    )


def _segment() -> tuple[FeatureMatrix, SignalArray]:
    time = np.arange(8, dtype=np.float64) / 100.0
    features = np.column_stack((np.linspace(-1.0, 1.0, 8), np.linspace(1.0, -1.0, 8)))
    target = np.column_stack((features[:, 0] + 0.25, features[:, 1] - 0.5))
    return (
        FeatureMatrix(
            features,
            fs=100.0,
            time=time,
            feature_names=["x", "y"],
            source_signal="simulated",
            window_size=0.02,
            shift=0.01,
            unit=["1", "1"],
        ),
        SignalArray.from_array(
            target,
            fs=100.0,
            time=time.copy(),
            channel_names=("x_velocity", "y_velocity"),
            units="1",
        ),
    )


def _deployment() -> object:
    return _resolve_decoder_deployment(_input_schema(), _schema(), ())


def _protocol() -> co.CenterOutProtocol:
    task = co.CenterOutTask(
        geometry_unit=co.GeometryUnit.NORMALIZED,
        layout=co.build_radial_layout(co.RadialLayoutRequest(radius=0.5)),
        acceptance=0.1,
        movement_timeout_seconds=1.0,
        selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
        seed=17,
    )
    return co.CenterOutProtocol(
        task=task,
        assistance_blocks=(AssistanceBlock(1.0, 4), AssistanceBlock(0.5, 6)),
    )


def test_initial_decoder_is_neutral_and_uses_the_feature_contract() -> None:
    stage = _deployment().initial_stage(_schema())

    assert isinstance(stage, LinearDecoderStage)
    assert stage.selected_feature_names == ("x", "y")
    assert stage.coefficients == (0.0, 0.0, 0.0, 0.0)
    assert stage.intercept == (0.0, 0.0)


def test_decoder_deployment_identifiers_avoid_the_prepared_pipeline() -> None:
    deployment = _resolve_decoder_deployment(
        _input_schema(),
        _schema(),
        (
            SimpleNamespace(
                output_schema_id=3,
                output_signal_id=3,
                output_channel_set_id=2,
            ),
        ),
    )

    assert deployment.output_schema_id == 4
    assert deployment.output_signal_id == 4
    assert deployment.output_channel_set_id == 3


@pytest.mark.parametrize(
    ("decoder", "expected"),
    [
        (LinearDecoder(), LinearDecoderStage),
        (KalmanDecoder(jitter=1e-6, innovation_jitter=1e-6), KalmanDecoderStage),
    ],
)
def test_fitted_decoder_snapshots_to_native_stage(decoder, expected) -> None:
    X, y = _segment()
    fitted = decoder.fit_segments(((X, y),))

    stage = _deployment().candidate_stage(fitted, _schema())

    assert isinstance(stage, expected)
    assert stage.selected_feature_names == ("x", "y")
    assert stage.output_signal_id == 3


def test_trainer_versions_follow_the_deployed_initial_model() -> None:
    X, y = _segment()
    config = OnlineDecoderTrainingConfig(
        update_interval=1,
        training_window=1,
    )
    with RollingDecoderTrainer(LinearDecoder(), config, initial_version=7) as trainer:
        assert trainer.submit_trial(X, y)
        assert trainer.wait_candidate().version == 8


def test_decoder_publication_retains_the_original_positional_signature() -> None:
    publication = co.DecoderPublication(1, "plan", 2, 3, 4)

    assert publication.fit_duration_ns == 0


def test_final_candidate_publication_waits_for_an_inflight_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neurale.experiments import center_out_training

    original_fit = center_out_training._fit_decoder
    fit_started = threading.Event()
    release_fit = threading.Event()

    def delayed_fit(decoder, segments):
        fit_started.set()
        assert release_fit.wait(timeout=5.0)
        return original_fit(decoder, segments)

    monkeypatch.setattr(center_out_training, "_fit_decoder", delayed_fit)
    trainer = RollingDecoderTrainer(
        LinearDecoder(),
        OnlineDecoderTrainingConfig(update_interval=1, training_window=1),
    )
    session = object.__new__(co.CenterOutSession)
    session._trainer = trainer
    session._deployment = _deployment()
    session._feature_schema = _schema()
    session._publications = []

    class NativeSession:
        completed_trials = 1
        active_decoder_version = 1

        @staticmethod
        def prepare_candidate(_pipeline, _version):
            return streaming.StreamStatus.OK

        @staticmethod
        def _record_decoder_publication(*_args):
            return True

    session._native = NativeSession()
    try:
        assert trainer.submit_trial(*_segment())
        assert fit_started.wait(timeout=5.0)
        errors: list[BaseException] = []

        def publish() -> None:
            try:
                session._publish_ready_candidate(123, wait=True)
            except BaseException as exc:
                errors.append(exc)

        publisher = threading.Thread(target=publish)
        publisher.start()
        publisher.join(timeout=0.05)
        assert publisher.is_alive()
        release_fit.set()
        publisher.join(timeout=5.0)

        assert not publisher.is_alive()
        assert errors == []
        assert len(session.publications) == 1
        assert session.publications[0].fit_duration_ns >= 0
        assert trainer.active is not None
    finally:
        release_fit.set()
        trainer.close()


def test_final_publication_replaces_unactivated_candidate_and_drains_backlog() -> None:
    with RollingDecoderTrainer(
        LinearDecoder(),
        OnlineDecoderTrainingConfig(update_interval=1, training_window=3),
        initial_version=1,
    ) as trainer:
        trainer.submit_trial(*_segment())
        trainer.wait_candidate()
        trainer.submit_trial(*_segment())
        trainer.submit_trial(*_segment())
        trainer.activate_candidate()
        session = object.__new__(co.CenterOutSession)
        session._trainer = trainer
        session._deployment = _deployment()
        session._feature_schema = _schema()
        session._publications = []
        recorded = []

        class NativeSession:
            active_decoder_version = 1
            completed_trials = 3
            pending_version = 2

            def _prepare_final_candidate(self, _pipeline, version):
                self.pending_version = version
                return streaming.StreamStatus.OK

            def _record_decoder_publication(self, version, *_args):
                recorded.append(version)
                return True

        session._native = NativeSession()
        session._publish_ready_candidate(123, wait=True, latest=True)
        assert recorded == [4]
        assert session._native.pending_version == 4
        assert session._native.active_decoder_version == 1
        assert session.publications[0].training_trials == 3
        assert not trainer.candidate_pending
        assert trainer.trials_in_block == 0


@pytest.mark.parametrize(
    "failure", [None, "presentation", "finalize", "runtime", "runtime_incomplete"]
)
@pytest.mark.parametrize("show", [False, True])
def test_task_complete_publication_precedes_native_trace_stop(
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
    show: bool,
) -> None:
    events: list[object] = []
    runtime_failed = failure in ("runtime", "runtime_incomplete")

    class Runner:
        state = (
            streaming.RuntimeState.FAILED
            if failure == "runtime_incomplete"
            else streaming.RuntimeState.RUNNING
        )
        _runner = object()
        primary_fault = (
            SimpleNamespace(
                code="queue_overrun",
                status=streaming.StreamStatus.QUEUE_OVERFLOW,
                stage="runtime",
                detail=0,
            )
            if runtime_failed
            else None
        )

        @staticmethod
        def _validate_profile_contract() -> None:
            return None

        def stop(self):
            events.append("runner-stop")
            self.state = streaming.RuntimeState.STOPPED
            return streaming.StreamStatus.OK

        @staticmethod
        def join():
            events.append("runner-join")
            return (
                streaming.StreamStatus.QUEUE_OVERFLOW
                if runtime_failed
                else streaming.StreamStatus.OK
            )

    class Trainer:
        @staticmethod
        def close() -> None:
            events.append("trainer-close")

    class NativeSession:
        complete = failure != "runtime_incomplete"
        completed_trials = 1
        active_decoder_version = 1
        training_capture_drops = (0, 0)
        presentation_state_drops = 0

        @property
        def outcome(self):
            return {
                "state": 1,
                "experiment_trace_complete": True,
                "trace_losses": 0,
                "producer_trace_drops": 0,
                "abnormal_conditions": 0,
                "abnormal_trials_affected": 0,
                "abnormal_session_aborted": False,
                "terminal_abort": False,
                "runtime_status": streaming.StreamStatus.OK,
            }

        @staticmethod
        def attach_runtime(_runner):
            return streaming.StreamStatus.OK

        @staticmethod
        def _record_decoder_publication(*_args):
            return True

        @staticmethod
        def _set_host_epoch(_host_epoch_ns):
            return True

        @staticmethod
        def start(_start_ns):
            return streaming.StreamStatus.OK

        @staticmethod
        def stop(_end_ns, _reason):
            events.append("native-stop")
            return streaming.StreamStatus.OK

        @staticmethod
        def close() -> None:
            events.append("native-close")

        @staticmethod
        def abort(_end_ns, _reason):
            if runtime_failed:
                assert _reason == "runtime-failure"
            events.append("native-abort")
            return streaming.StreamStatus.OK

    class Presenter:
        def close(self):
            events.append("presenter-close")

    class Recording:
        _impl = SimpleNamespace(
            status=SimpleNamespace(
                state="failed",
                runtime_accepted=1,
                recorder_accepted=1,
                control_offered=1,
                control_accepted=1,
                primary_fault=SimpleNamespace(
                    present=True,
                    origin=0,
                    reason=0,
                    stream_status=streaming.StreamStatus.QUEUE_OVERFLOW,
                ),
            )
        )

        def finalize(self):
            if show:
                assert events.count("presenter-close") == 1
            events.append("finalize")
            if failure == "finalize":
                raise RuntimeError("finalize failed")

    def pump(_self):
        events.append("pump")
        if failure == "presentation":
            raise RuntimeError("presentation failed")

    def drain(_self) -> None:
        events.append("drain-training")

    def publish(_self, _time_ns, *, wait=False, latest=False) -> None:
        events.append(("publish", wait, latest))

    monkeypatch.setattr(co.CenterOutSession, "_open_presentation", lambda *_args: None)
    monkeypatch.setattr(co.CenterOutSession, "_pump_presentation", pump)
    monkeypatch.setattr(co.CenterOutSession, "_drain_training", drain)
    monkeypatch.setattr(co.CenterOutSession, "_publish_ready_candidate", publish)

    session = object.__new__(co.CenterOutSession)
    session._runner = Runner()
    session._native = NativeSession()
    session._trainer = Trainer()
    session._presenter = Presenter() if show else None
    session._recording = Recording()
    session._current_trial = []
    session._initial_plan_fingerprint = "initial"
    session._lifecycle_deadline_seconds = 10.0
    session._presented_frames = 0
    session._publications = []

    if runtime_failed:
        with pytest.raises(RuntimeError, match=r"QUEUE_OVERFLOW.*queue_overrun"):
            session.run()
    elif failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            session.run()
    else:
        session.run()

    if show:
        assert events.count("presenter-close") == 1
        assert events.index("presenter-close") < events.index("finalize")
    if runtime_failed:
        assert not any(isinstance(event, tuple) and event[0] == "publish" for event in events)
        assert "native-stop" not in events
        assert events.count("native-abort") == 1
        assert events.count("finalize") == 1
        assert events.index("runner-join") < events.index("native-abort")
        assert events.index("native-abort") < events.index("finalize")
        assert events.index("finalize") < events.index("trainer-close")
        assert events.index("trainer-close") < events.index("native-close")
        assert "drain-training" not in events[events.index("runner-join") + 1 :]
        return
    if failure == "presentation":
        assert events.index("native-abort") < events.index("finalize")
        if show:
            assert events.index("native-abort") < events.index("presenter-close")
        return

    final_publication = ("publish", True, True)
    assert events.index("runner-stop") < events.index("runner-join")
    assert events.index("runner-join") < events.index(final_publication)
    assert events.index(final_publication) < events.index("native-stop")
    assert events.index("native-stop") < events.index("finalize")
    if show:
        assert events.index("native-stop") < events.index("presenter-close")


def test_closed_loop_resolves_automatic_training_bounds_from_the_protocol() -> None:
    training = _resolve_training_config(_protocol(), None)
    parameters = signature(co.CenterOutSession).parameters

    assert not hasattr(closed_loop, "CenterOutClosedLoopConfig")
    assert not hasattr(closed_loop, "CenterOutClosedLoopSession")
    assert not hasattr(co, "CenterOutClosedLoopSession")
    assert not hasattr(co, "CenterOutSessionConfig")
    assert not hasattr(co, "CenterOut2DConfig")
    assert not hasattr(co, "HeadlessCenterOutSession")
    assert parameters["protocol"].annotation == "CenterOutProtocol"
    assert parameters["feature_plan"].annotation == "PipelinePlan"
    assert parameters["device"].annotation == "Any"
    assert parameters["decoder"].annotation == "LinearDecoder | KalmanDecoder"
    assert parameters["recording"].annotation == "SessionRecorder | None"
    assert parameters["runtime_config"].default is None
    assert tuple(parameters) == (
        "protocol",
        "feature_plan",
        "device",
        "decoder",
        "runtime_config",
        "training",
        "presentation_config",
        "recording",
    )
    assert "feature_pipeline" not in parameters
    assert "input_schema" not in parameters
    assert "source" not in parameters
    assert "profile" not in parameters
    assert "safety_controller" not in parameters
    assert "host_epoch_ns" not in parameters
    assert parameters["training"].default is None
    assert "poll_interval_seconds" not in parameters
    assert "max_wall_time_seconds" not in parameters
    assert tuple(signature(co.CenterOutSession.run).parameters) == ("self",)
    assert training.update_interval == 4
    assert training.training_window == 10


def test_outcome_exposes_terminal_fields_without_a_nested_session() -> None:
    parameters = signature(co.CenterOutOutcome).parameters

    assert "session" not in parameters
    assert tuple(parameters) == (
        "state",
        "experiment_trace_complete",
        "trace_losses",
        "producer_trace_drops",
        "abnormal_conditions",
        "abnormal_trials_affected",
        "abnormal_session_aborted",
        "terminal_abort",
        "runtime_status",
        "publications",
        "active_decoder_version",
        "completed_trials",
        "training_capture_drops",
        "presented_frames",
        "presentation_state_drops",
        "recording",
    )


def test_closed_loop_requires_a_declarative_feature_plan() -> None:
    with pytest.raises(TypeError, match="feature_plan must be a PipelinePlan"):
        co.CenterOutSession(
            _protocol(),
            object(),
            object(),
            LinearDecoder(),
        )


def test_closed_loop_rejects_decoder_and_optional_config_types_before_compilation() -> None:
    plan = closed_loop.PipelinePlan(stages=(object(),))
    device = SimpleNamespace(schema=object(), source=object())

    with pytest.raises(TypeError, match="decoder must be a LinearDecoder or KalmanDecoder"):
        co.CenterOutSession(_protocol(), plan, device, object())
    with pytest.raises(TypeError, match="runtime_config must be a RealtimeConfig or None"):
        co.CenterOutSession(_protocol(), plan, device, LinearDecoder(), runtime_config=object())
    with pytest.raises(TypeError, match="recording must be a SessionRecorder or None"):
        co.CenterOutSession(_protocol(), plan, device, LinearDecoder(), recording=object())


def test_update_interval_must_train_before_assistance_decreases() -> None:
    with pytest.raises(ValidationError, match="initial 100%-assisted block"):
        _resolve_training_config(
            _protocol(),
            OnlineDecoderTrainingConfig(update_interval=5, training_window=10),
        )


def test_lifecycle_deadline_is_derived_from_prepared_session_state() -> None:
    schema = _schema()

    seconds = _lifecycle_deadline_seconds(
        _protocol(),
        schema.signals[0],
        schema.feature_sets[0],
        streaming.RealtimeConfig(),
    )

    assert seconds == pytest.approx(25.23)


def test_positive_label_lag_pairs_earlier_features_with_later_intent() -> None:
    observations = tuple(
        CenterOutTrainingObservation(
            sample_idx=index,
            features=(float(index), float(-index)),
            time_ns=index * 10_000_000,
            trial=None,
            target_position=(0.0, 0.0),
            cursor_position=(0.0, 0.0),
            cursor_velocity=(0.0, 0.0),
            guidance_velocity=(float(10 + index), float(20 + index)),
            trial_stop=index == 5,
        )
        for index in range(6)
    )
    training = OnlineDecoderTrainingConfig(
        update_interval=1,
        training_window=1,
        label_lag_seconds=0.02,
    )

    X, y = _training_segment(observations, _schema(), training)

    np.testing.assert_array_equal(X.data[:, 0], (0.0, 1.0, 2.0, 3.0))
    np.testing.assert_array_equal(y.data[:, 0], (12.0, 13.0, 14.0, 15.0))


def test_label_lag_must_match_the_prepared_feature_period() -> None:
    with pytest.raises(ValidationError, match="integer number of feature observations"):
        _label_lag_samples(0.015, 100.0)
