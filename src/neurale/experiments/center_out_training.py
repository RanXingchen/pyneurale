#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Bounded block-wise decoder training for Center-Out sessions."""

from __future__ import annotations

import copy
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from neurale.exceptions import ValidationError
from neurale.pipeline._deployment import _feature_contract


class OnlineDecoderKind(StrEnum):
    """Decoder family fitted by one session."""

    LINEAR = "linear"
    KALMAN = "kalman"


class SupervisionTarget(StrEnum):
    """Center-Out intent values paired with one feature observation."""

    VELOCITY = "velocity"
    POSITION = "position"
    POSITION_VELOCITY = "position_velocity"


class OnlineDecoderUpdatePolicy(StrEnum):
    """Whether online fitting stops after initialization or continues periodically."""

    INITIAL_ONLY = "initial_only"
    PERIODIC = "periodic"


@dataclass(frozen=True, slots=True)
class AssistanceBlock:
    """One trial-aligned assistance interval."""

    assistance: float
    trials: int

    def __post_init__(self) -> None:
        if isinstance(self.assistance, bool) or not isinstance(self.assistance, int | float):
            raise ValidationError("assistance must be a finite number in [0, 1].")
        value = float(self.assistance)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValidationError("assistance must be a finite number in [0, 1].")
        if isinstance(self.trials, bool) or not isinstance(self.trials, int) or self.trials <= 0:
            raise ValidationError("trials must be a positive integer.")
        object.__setattr__(self, "assistance", value)


@dataclass(frozen=True, slots=True, kw_only=True)
class OnlineDecoderTrainingConfig:
    """User-facing cadence, history, target, and label-alignment contract."""

    target: SupervisionTarget | str = SupervisionTarget.VELOCITY
    update_interval: int | None = None
    training_window: int | None = None
    label_lag_seconds: float = 0.0
    update_policy: OnlineDecoderUpdatePolicy | str = OnlineDecoderUpdatePolicy.PERIODIC

    def __post_init__(self) -> None:
        try:
            target = SupervisionTarget(self.target)
        except ValueError as exc:
            raise ValidationError(
                "target must be 'velocity', 'position', or 'position_velocity'."
            ) from exc
        if target != SupervisionTarget.VELOCITY:
            raise ValidationError("target currently supports only 'velocity'.")
        try:
            update_policy = OnlineDecoderUpdatePolicy(self.update_policy)
        except ValueError as exc:
            raise ValidationError("update_policy must be 'initial_only' or 'periodic'.") from exc
        for name in ("update_interval", "training_window"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValidationError(f"{name} must be a positive integer or None.")
        if (
            self.update_interval is not None
            and self.training_window is not None
            and self.training_window < self.update_interval
        ):
            raise ValidationError("training_window must be at least update_interval.")
        lag = self.label_lag_seconds
        if isinstance(lag, bool) or not isinstance(lag, int | float):
            raise ValidationError("label_lag_seconds must be a finite non-negative number.")
        lag = float(lag)
        if not math.isfinite(lag) or lag < 0.0:
            raise ValidationError("label_lag_seconds must be a finite non-negative number.")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "update_policy", update_policy)
        object.__setattr__(self, "label_lag_seconds", lag)


@dataclass(frozen=True, slots=True)
class OnlineDecoderCandidate:
    """A completed fit waiting for explicit trial-boundary activation."""

    version: int
    decoder: Any
    training_blocks: int
    training_trials: int
    fit_started_ns: int
    fit_completed_ns: int

    @property
    def fit_duration_ns(self) -> int:
        return self.fit_completed_ns - self.fit_started_ns


def _fit_decoder(decoder: Any, segments: tuple[tuple[Any, Any], ...]) -> tuple[Any, int, int]:
    started = time.perf_counter_ns()
    fitted = decoder.fit_segments(segments)
    return fitted, started, time.perf_counter_ns()


@dataclass(frozen=True, slots=True)
class _OnlineDecoderDeployment:
    """Resolved identities and native-stage construction for one session."""

    output_schema_id: int
    output_signal_id: int
    output_channel_set_id: int

    def initial_stage(self, feature_schema: Any) -> Any:
        """Build the version-zero neutral decoder used during full assistance."""

        from neurale.pipeline import LinearDecoderStage

        signal, descriptor, contract = _feature_contract(feature_schema)
        n_features = len(descriptor.feature_names)
        return LinearDecoderStage(
            output_schema_id=self.output_schema_id,
            output_signal_id=self.output_signal_id,
            output_channel_set_id=self.output_channel_set_id,
            output_physical_unit="dimensionless",
            feature_set_id=signal.feature_set_id,
            selection=tuple(range(n_features)),
            selected_feature_names=tuple(descriptor.feature_names),
            fitted_feature_contract=contract,
            n_features=n_features,
            n_outputs=2,
            coefficients=(0.0,) * (2 * n_features),
            intercept=(0.0, 0.0),
        )

    def candidate_stage(self, decoder: Any, feature_schema: Any) -> Any:
        """Export a decoder, then enforce the task's two-dimensional output."""
        from neurale.pipeline import decoder_stage

        stage = decoder_stage(
            decoder,
            feature_schema,
            output_schema_id=self.output_schema_id,
            output_signal_id=self.output_signal_id,
            output_channel_set_id=self.output_channel_set_id,
        )
        if getattr(stage, "n_outputs", getattr(stage, "state_dim", None)) != 2:
            raise ValidationError("Center-out deployment requires two velocity outputs.")
        return stage


_MAX_STREAM_IDENTIFIER = (1 << 32) - 1
_MAX_CHANNEL_SET_IDENTIFIER = (1 << 64) - 1


def _resolve_decoder_deployment(
    input_schema: Any, feature_schema: Any, feature_stages: tuple[Any, ...]
) -> _OnlineDecoderDeployment:
    schemas = {int(input_schema.id), int(feature_schema.id)}
    signals = {
        int(signal.id) for schema in (input_schema, feature_schema) for signal in schema.signals
    }
    channel_sets = {
        int(signal.channel_set_id)
        for schema in (input_schema, feature_schema)
        for signal in schema.signals
    }
    for stage in feature_stages:
        if hasattr(stage, "output_schema_id"):
            schemas.add(int(stage.output_schema_id))
        if hasattr(stage, "output_signal_id"):
            signals.add(int(stage.output_signal_id))
        if hasattr(stage, "output_channel_set_id"):
            channel_sets.add(int(stage.output_channel_set_id))
    return _OnlineDecoderDeployment(
        output_schema_id=_unused_identifier(schemas, _MAX_STREAM_IDENTIFIER, "decoder schema"),
        output_signal_id=_unused_identifier(signals, _MAX_STREAM_IDENTIFIER, "decoder signal"),
        output_channel_set_id=_unused_identifier(
            channel_sets, _MAX_CHANNEL_SET_IDENTIFIER, "decoder channel set"
        ),
    )


def _unused_identifier(values: set[int], maximum: int, name: str) -> int:
    candidate = 1
    while candidate in values:
        candidate += 1
    if candidate > maximum:
        raise ValidationError(f"no {name} identifier is available.")
    return candidate


def supervision_values(
    target: SupervisionTarget | str,
    *,
    target_position: tuple[float, float],
    guidance_velocity: tuple[float, float],
) -> tuple[float, ...]:
    """Build the declared intent label without using assisted cursor motion."""

    selected = SupervisionTarget(target)
    position = _finite_pair(target_position, "target_position")
    velocity = _finite_pair(guidance_velocity, "guidance_velocity")
    if selected == SupervisionTarget.POSITION:
        return position
    if selected == SupervisionTarget.VELOCITY:
        return velocity
    return (*position, *velocity)


def project_decoder_velocity(
    target: SupervisionTarget | str,
    decoded: tuple[float, ...],
    *,
    cursor_position: tuple[float, float],
    position_error_time_constant_ns: int | None = None,
) -> tuple[float, float]:
    """Project a fitted target onto the controller's two velocity axes."""

    selected = SupervisionTarget(target)
    values = tuple(float(value) for value in decoded)
    if not all(math.isfinite(value) for value in values):
        raise ValidationError("decoded values must be finite.")
    if selected == SupervisionTarget.VELOCITY:
        if len(values) != 2:
            raise ValidationError("velocity decoding requires exactly two outputs.")
        return values
    if selected == SupervisionTarget.POSITION_VELOCITY:
        if len(values) != 4:
            raise ValidationError("position_velocity decoding requires exactly four outputs.")
        return values[2], values[3]
    if len(values) != 2:
        raise ValidationError("position decoding requires exactly two outputs.")
    if (
        isinstance(position_error_time_constant_ns, bool)
        or not isinstance(position_error_time_constant_ns, int)
        or position_error_time_constant_ns <= 0
    ):
        raise ValidationError("position_error_time_constant_ns must be positive.")
    cursor = _finite_pair(cursor_position, "cursor_position")
    tau_seconds = position_error_time_constant_ns / 1_000_000_000.0
    return (values[0] - cursor[0]) / tau_seconds, (values[1] - cursor[1]) / tau_seconds


class RollingDecoderTrainer:
    """Fit a trial-bounded rolling window on one non-realtime worker."""

    __slots__ = (
        "_active",
        "_candidate",
        "_config",
        "_current",
        "_executor",
        "_future",
        "_history",
        "_next_block",
        "_next_version",
        "_template",
        "_updates_submitted",
    )

    def __init__(
        self,
        decoder: Any,
        config: OnlineDecoderTrainingConfig,
        *,
        initial_version: int = 0,
    ) -> None:
        from neurale.decoding import KalmanDecoder, LinearDecoder

        if not isinstance(config, OnlineDecoderTrainingConfig):
            raise TypeError("config must be an OnlineDecoderTrainingConfig")
        if not isinstance(decoder, LinearDecoder | KalmanDecoder):
            raise TypeError("decoder must be a LinearDecoder or KalmanDecoder")
        if config.update_interval is None or config.training_window is None:
            raise ValidationError(
                "update_interval and training_window must be resolved against a session."
            )
        if decoder.is_fitted:
            raise ValidationError("the trainer requires an unfitted decoder configuration.")
        if (
            isinstance(initial_version, bool)
            or not isinstance(initial_version, int)
            or initial_version < 0
        ):
            raise ValidationError("initial_version must be a non-negative integer.")
        self._config = config
        self._template = copy.deepcopy(decoder)
        self._history: deque[tuple[int, tuple[Any, Any]]] = deque(maxlen=config.training_window)
        self._current: deque[tuple[Any, Any]] = deque()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neurale-decoder-fit")
        self._future: Future[Any] | None = None
        self._candidate: OnlineDecoderCandidate | None = None
        self._active: OnlineDecoderCandidate | None = None
        self._next_block = 1
        self._next_version = initial_version + 1
        self._updates_submitted = 0

    @property
    def active(self) -> OnlineDecoderCandidate | None:
        return self._active

    @property
    def candidate(self) -> OnlineDecoderCandidate | None:
        return self._candidate

    @property
    def candidate_pending(self) -> bool:
        """Whether a submitted or completed fit still needs publication."""

        return self._future is not None or self._candidate is not None

    @property
    def trials_in_block(self) -> int:
        return len(self._current)

    def submit_trial(self, X: Any, y: Any) -> bool:
        """Retain one completed trial and submit a fit exactly at the N boundary."""

        if self._config.update_policy == OnlineDecoderUpdatePolicy.INITIAL_ONLY and (
            self._updates_submitted != 0
        ):
            return False
        self._current.append(copy.deepcopy((X, y)))
        return self._submit_ready_fit()

    def _submit_ready_fit(self) -> bool:
        if self._future is not None or self._candidate is not None:
            return False
        assert self._config.update_interval is not None
        if len(self._current) < self._config.update_interval:
            return False
        block = tuple(self._current.popleft() for _ in range(self._config.update_interval))
        self._history.extend((self._next_block, segment) for segment in block)
        self._next_block += 1
        segments = tuple(segment for _, segment in self._history)
        decoder = copy.deepcopy(self._template)
        self._future = self._executor.submit(_fit_decoder, decoder, segments)
        self._updates_submitted += 1
        return True

    def poll_candidate(self) -> OnlineDecoderCandidate | None:
        """Return a completed candidate without waiting for the fit worker."""

        if self._candidate is not None:
            return self._candidate
        if self._future is None or not self._future.done():
            return None
        return self._collect_candidate()

    def wait_candidate(self, timeout: float | None = None) -> OnlineDecoderCandidate:
        """Wait outside the realtime path for the submitted fit to finish."""

        if self._future is None:
            if self._candidate is None:
                raise RuntimeError("no decoder fit is pending")
            return self._candidate
        result = self._future.result(timeout=timeout)
        return self._collect_candidate(result)

    def wait_latest_candidate(self, timeout: float | None = None) -> OnlineDecoderCandidate:
        """Wait for all complete queued update blocks and return the latest fit."""

        candidate = self.wait_candidate(timeout=timeout)
        assert self._config.update_interval is not None
        while len(self._current) >= self._config.update_interval:
            self.activate_candidate()
            candidate = self.wait_candidate(timeout=timeout)
        return candidate

    def _collect_candidate(
        self, result: tuple[Any, int, int] | None = None
    ) -> OnlineDecoderCandidate:
        if self._future is None:
            assert self._candidate is not None
            return self._candidate
        if result is None:
            result = self._future.result()
        decoder, fit_started_ns, fit_completed_ns = result
        self._future = None
        candidate = OnlineDecoderCandidate(
            version=self._next_version,
            decoder=decoder,
            training_blocks=len({block for block, _ in self._history}),
            training_trials=len(self._history),
            fit_started_ns=fit_started_ns,
            fit_completed_ns=fit_completed_ns,
        )
        self._next_version += 1
        self._candidate = candidate
        return candidate

    def activate_candidate(self) -> OnlineDecoderCandidate:
        """Activate the completed candidate at a caller-owned trial boundary."""

        if self._future is not None:
            raise RuntimeError("the decoder fit has not completed")
        if self._candidate is None:
            raise RuntimeError("no decoder candidate is ready")
        self._active = self._candidate
        self._candidate = None
        self._submit_ready_fit()
        return self._active

    def reject_candidate(self) -> None:
        if self._future is not None:
            raise RuntimeError("the decoder fit has not completed")
        if self._candidate is None:
            raise RuntimeError("no decoder candidate is ready")
        self._candidate = None

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> RollingDecoderTrainer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


def _finite_pair(values: tuple[float, float], name: str) -> tuple[float, float]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValidationError(f"{name} must be a two-value tuple.")
    pair = float(values[0]), float(values[1])
    if not all(math.isfinite(value) for value in pair):
        raise ValidationError(f"{name} values must be finite.")
    return pair


__all__ = [
    "AssistanceBlock",
    "OnlineDecoderCandidate",
    "OnlineDecoderKind",
    "OnlineDecoderTrainingConfig",
    "OnlineDecoderUpdatePolicy",
    "RollingDecoderTrainer",
    "SupervisionTarget",
    "project_decoder_velocity",
    "supervision_values",
]
