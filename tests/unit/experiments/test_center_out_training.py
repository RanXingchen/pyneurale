#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
from inspect import signature

import numpy as np
import pytest

from neurale.data import FeatureMatrix, SignalArray
from neurale.decoding import KalmanDecoder, LinearDecoder
from neurale.exceptions import ValidationError
from neurale.experiments import center_out_training
from neurale.experiments.center_out_training import (
    AssistanceBlock,
    OnlineDecoderTrainingConfig,
    OnlineDecoderUpdatePolicy,
    RollingDecoderTrainer,
    SupervisionTarget,
    project_decoder_velocity,
    supervision_values,
)


def _segment(offset: float) -> tuple[FeatureMatrix, SignalArray]:
    time = np.arange(12, dtype=np.float64) / 50.0
    target = np.column_stack((time + offset, 2.0 * time - offset))
    features = np.column_stack((target, target[:, 0] - target[:, 1]))
    X = FeatureMatrix(
        data=features,
        fs=50.0,
        time=time,
        feature_names=["f0", "f1", "f2"],
        unit="uV",
        shift=0.02,
    )
    y = SignalArray.from_array(
        target,
        fs=50.0,
        time=X.time.copy(),
        channel_names=["vx", "vy"],
        channel_types="behavior",
        units="normalized",
        name="guidance_velocity",
    )
    return X, y


def test_assistance_block_validates_exact_endpoint() -> None:
    assert AssistanceBlock(1.0, 50).assistance == 1.0
    with pytest.raises(ValidationError):
        AssistanceBlock(1.01, 50)


def test_training_config_is_keyword_only_and_defaults_to_automatic_bounds() -> None:
    config = OnlineDecoderTrainingConfig()

    assert tuple(signature(OnlineDecoderTrainingConfig).parameters) == (
        "target",
        "update_interval",
        "training_window",
        "label_lag_seconds",
        "update_policy",
    )
    assert config.target == SupervisionTarget.VELOCITY
    assert config.update_interval is None
    assert config.training_window is None
    assert config.label_lag_seconds == 0.0
    assert config.update_policy == OnlineDecoderUpdatePolicy.PERIODIC
    assert not hasattr(center_out_training, "OnlineDecoderDeploymentConfig")


def test_training_config_currently_supports_only_velocity_targets() -> None:
    with pytest.raises(ValidationError, match="currently supports only 'velocity'"):
        OnlineDecoderTrainingConfig(target="position")


def test_supervision_never_uses_assisted_cursor_velocity() -> None:
    assert supervision_values(
        SupervisionTarget.POSITION_VELOCITY,
        target_position=(0.5, -0.5),
        guidance_velocity=(0.25, 0.75),
    ) == (0.5, -0.5, 0.25, 0.75)


def test_position_projection_is_explicit_error_control() -> None:
    assert project_decoder_velocity(
        "position",
        (0.75, -0.25),
        cursor_position=(0.25, 0.25),
        position_error_time_constant_ns=500_000_000,
    ) == (1.0, -1.0)
    assert project_decoder_velocity(
        "position_velocity",
        (9.0, 8.0, 0.4, -0.2),
        cursor_position=(0.0, 0.0),
    ) == (0.4, -0.2)


def test_rolling_trainer_fits_only_at_block_boundary_and_activates_explicitly() -> None:
    config = OnlineDecoderTrainingConfig(
        update_interval=2,
        training_window=2,
    )
    with RollingDecoderTrainer(LinearDecoder(), config) as trainer:
        assert trainer.submit_trial(*_segment(0.0)) is False
        assert trainer.submit_trial(*_segment(0.1)) is True
        candidate = trainer.wait_candidate(timeout=10.0)
        assert candidate.version == 1
        assert candidate.training_blocks == 1
        assert candidate.training_trials == 2
        assert trainer.poll_candidate() is candidate
        assert candidate.fit_duration_ns >= 0
        assert trainer.active is None
        active = trainer.activate_candidate()
        assert active is candidate
        assert active.decoder.is_fitted


def test_initial_only_policy_never_submits_a_second_fit() -> None:
    config = OnlineDecoderTrainingConfig(
        update_interval=2,
        training_window=2,
        update_policy="initial_only",
    )
    with RollingDecoderTrainer(LinearDecoder(), config) as trainer:
        assert trainer.poll_candidate() is None
        assert not trainer.submit_trial(*_segment(0.0))
        assert trainer.submit_trial(*_segment(0.1))
        trainer.wait_candidate(timeout=10.0)
        trainer.activate_candidate()
        assert not trainer.submit_trial(*_segment(0.2))
        assert not trainer.submit_trial(*_segment(0.3))
        assert trainer.poll_candidate() is None
        assert trainer.trials_in_block == 0


def test_rolling_window_retains_the_declared_number_of_trials() -> None:
    config = OnlineDecoderTrainingConfig(update_interval=2, training_window=3)
    with RollingDecoderTrainer(LinearDecoder(), config) as trainer:
        for trial in range(1, 7):
            submitted = trainer.submit_trial(*_segment(float(trial)))
            if trial % 2:
                assert not submitted
                continue
            assert submitted
            candidate = trainer.wait_candidate(timeout=10.0)
            assert candidate.version == trial // 2
            assert candidate.training_blocks == min(trial // 2, 2)
            assert candidate.training_trials == min(trial, 3)
            trainer.activate_candidate()


def test_rolling_trainer_preserves_trials_completed_while_fit_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = center_out_training._fit_decoder
    fit_started = threading.Event()
    release_fit = threading.Event()
    fitted_offsets: list[tuple[float, ...]] = []

    def delayed_fit(decoder, segments):
        fitted_offsets.append(tuple(float(X.data[0, 0]) for X, _ in segments))
        if len(fitted_offsets) == 1:
            fit_started.set()
            assert release_fit.wait(timeout=5.0)
        return original_fit(decoder, segments)

    monkeypatch.setattr(center_out_training, "_fit_decoder", delayed_fit)
    config = OnlineDecoderTrainingConfig(update_interval=2, training_window=6)
    with RollingDecoderTrainer(LinearDecoder(), config) as trainer:
        assert not trainer.submit_trial(*_segment(1.0))
        assert trainer.submit_trial(*_segment(2.0))
        assert fit_started.wait(timeout=5.0)
        assert not trainer.submit_trial(*_segment(3.0))
        assert not trainer.submit_trial(*_segment(4.0))
        assert trainer.trials_in_block == 2

        release_fit.set()
        trainer.wait_candidate(timeout=10.0)
        trainer.activate_candidate()
        assert trainer.candidate_pending
        candidate = trainer.wait_candidate(timeout=10.0)
        assert not trainer.submit_trial(*_segment(5.0))

        assert fitted_offsets == [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0)]
        assert candidate.training_trials == 4
        assert trainer.trials_in_block == 1


def test_wait_latest_candidate_drains_every_complete_queued_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = center_out_training._fit_decoder
    fit_started = threading.Event()
    release_fit = threading.Event()
    fitted_offsets: list[tuple[float, ...]] = []

    def delayed_fit(decoder, segments):
        fitted_offsets.append(tuple(float(X.data[0, 0]) for X, _ in segments))
        if len(fitted_offsets) == 1:
            fit_started.set()
            assert release_fit.wait(timeout=5.0)
        return original_fit(decoder, segments)

    monkeypatch.setattr(center_out_training, "_fit_decoder", delayed_fit)
    config = OnlineDecoderTrainingConfig(update_interval=2, training_window=6)
    with RollingDecoderTrainer(LinearDecoder(), config) as trainer:
        assert not trainer.submit_trial(*_segment(1.0))
        assert trainer.submit_trial(*_segment(2.0))
        assert fit_started.wait(timeout=5.0)
        for trial in range(3, 7):
            assert not trainer.submit_trial(*_segment(float(trial)))

        release_fit.set()
        candidate = trainer.wait_latest_candidate(timeout=10.0)

        assert candidate.version == 3
        assert candidate.training_trials == 6
        assert trainer.trials_in_block == 0
        assert fitted_offsets == [
            (1.0, 2.0),
            (1.0, 2.0, 3.0, 4.0),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        ]


def test_rolling_trainer_supports_kalman_segments() -> None:
    config = OnlineDecoderTrainingConfig(update_interval=2, training_window=2)
    decoder = KalmanDecoder(jitter=1e-6, innovation_jitter=1e-6)
    with RollingDecoderTrainer(decoder, config) as trainer:
        assert not trainer.submit_trial(*_segment(0.0))
        assert trainer.submit_trial(*_segment(0.2))
        candidate = trainer.wait_candidate(timeout=10.0)
        assert candidate.decoder.is_fitted
        assert candidate.decoder.n_outputs_ == 2
