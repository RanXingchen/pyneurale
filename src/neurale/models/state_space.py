#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Linear-Gaussian state-space estimation and Kalman filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from neurale._validation import validate_choice, validate_number, validate_real_array
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_models_operation, load_restored_models_operation
from ._inference import make_readonly

_EPS = float(np.finfo(np.float64).eps)

# Both covariance tolerances are relative to the largest magnitude in the matrix
# and neither has an absolute floor. A floor turns the check into an absolute
# one for small covariances, which accepts a matrix whose negative eigenvalue is
# orders of magnitude larger than its positive ones.
#
# Symmetry is forgiving: the caller may have built the matrix with a product
# whose two triangles rounded differently, and the matrix is symmetrized before
# use anyway. Definiteness is not. A symmetric eigensolver resolves eigenvalues
# to about ``eps * n * scale``, so anything more negative than a small multiple
# of that is a property of the matrix rather than of the solver.
_SYMMETRY_RELATIVE_TOLERANCE = 1e-8
_PSD_TOLERANCE_FACTOR = 64.0

_MISSING_POLICIES = ("error", "predict")


def _validate_covariance(value: np.ndarray, name: str, size: int) -> np.ndarray:
    """Validate a finite, symmetric, positive-semidefinite covariance matrix."""

    matrix = validate_real_array(value, name, ndim=2)
    if matrix.shape != (size, size):
        raise ValidationError(f"{name} must have shape ({size}, {size}).")
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_RELATIVE_TOLERANCE * scale):
        raise ValidationError(f"{name} must be symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    tol = _PSD_TOLERANCE_FACTOR * float(size) * _EPS * scale
    if float(np.min(np.linalg.eigvalsh(symmetric))) < -tol:
        raise ValidationError(f"{name} must be positive semidefinite.")
    return np.ascontiguousarray(symmetric)


def _validate_jitter(value: float) -> float:
    return float(
        validate_number(
            value,
            "jitter",
            kind="real",
            minimum=0,
            coerce=True,
        )
    )


def _validate_segment_lengths(segment_lengths, n_samples: int) -> np.ndarray:
    """Validate positive segment lengths that partition ``n_samples`` rows."""

    if segment_lengths is None:
        return np.array([n_samples], dtype=np.uintp)
    if not isinstance(segment_lengths, np.ndarray):
        raise ValidationError("segment_lengths must be a numpy.ndarray or None.")
    if segment_lengths.ndim != 1 or segment_lengths.size == 0:
        raise ValidationError("segment_lengths must be a non-empty 1D array.")
    if not np.issubdtype(segment_lengths.dtype, np.integer):
        raise ValidationError("segment_lengths must contain integers.")
    if np.any(segment_lengths <= 0):
        raise ValidationError("segment_lengths must be positive.")
    if int(np.sum(segment_lengths)) != n_samples:
        raise ValidationError("segment_lengths must sum to the number of samples.")
    return np.ascontiguousarray(segment_lengths, dtype=np.uintp)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Posterior states and covariances of one batch filtering pass."""

    states: np.ndarray
    covariances: np.ndarray


class LinearGaussianParameters:
    """Immutable snapshot of one fitted linear-Gaussian parameter set.

    The model is affine in both equations::

        x[t] = transition_ @ x[t - 1] + transition_offset_ + w,  w ~ N(0, process_covariance_)
        z[t] = observation_ @ x[t] + observation_offset_ + v,    v ~ N(0, observation_covariance_)

    :meth:`LinearGaussianStateSpace.fit` produces one of these and never touches
    it again: a later fit on the same estimator builds a *new* snapshot and
    leaves this one alone. A :class:`KalmanFilter` binds to the snapshot, so it
    keeps running against exactly the parameters, dimensions, and device it was
    constructed with no matter what happens to the estimator afterwards.

    Every array is a read-only view of the native parameters, and filtering
    never writes to them.
    """

    __slots__ = ("_native",)

    def __init__(self, native: object) -> None:
        object.__setattr__(self, "_native", native)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("LinearGaussianParameters is immutable.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LinearGaussianParameters is immutable.")

    @classmethod
    def _restore_fitted(
        cls,
        *,
        transition: np.ndarray,
        transition_offset: np.ndarray,
        observation: np.ndarray,
        observation_offset: np.ndarray,
        process_covariance: np.ndarray,
        observation_covariance: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> LinearGaussianParameters:
        """Rebuild a snapshot from parameters a fit produced.

        Private, and the only supported way to reconstitute a snapshot: decoder
        persistence stores the eight parameter arrays and has to restore a
        filter that runs the same native recursion, not a second implementation
        of it. The native state constructor cross-checks every array against
        the two dimensions the first two imply, so an inconsistent set is
        refused rather than entering the recursion half-shaped.
        """

        native = load_restored_models_operation("state_space")
        try:
            model = native.linear_gaussian_from_state(
                validate_real_array(transition, "transition", ndim=2),
                validate_real_array(transition_offset, "transition_offset", ndim=1),
                validate_real_array(observation, "observation", ndim=2),
                validate_real_array(observation_offset, "observation_offset", ndim=1),
                validate_real_array(process_covariance, "process_covariance", ndim=2),
                validate_real_array(observation_covariance, "observation_covariance", ndim=2),
                validate_real_array(initial_state, "initial_state", ndim=1),
                validate_real_array(initial_covariance, "initial_covariance", ndim=2),
            )
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc
        return cls(model)

    @property
    def device_(self) -> str:
        """Device that holds the parameters; always ``"cpu"``."""

        return "cpu"

    @property
    def n_states_(self) -> int:
        """Dimension of the latent state."""

        return int(self._native.state_dim)

    @property
    def n_observations_(self) -> int:
        """Dimension of one observation."""

        return int(self._native.observation_dim)

    @property
    def transition_(self) -> np.ndarray:
        """``(n_states_, n_states_)`` state transition matrix."""

        return self._native.transition

    @property
    def transition_offset_(self) -> np.ndarray:
        """``(n_states_,)`` transition offset; zero when the fit had no offsets."""

        return self._native.transition_offset

    @property
    def observation_(self) -> np.ndarray:
        """``(n_observations_, n_states_)`` observation matrix."""

        return self._native.observation

    @property
    def observation_offset_(self) -> np.ndarray:
        """``(n_observations_,)`` observation offset; zero when the fit had no offsets."""

        return self._native.observation_offset

    @property
    def process_covariance_(self) -> np.ndarray:
        """Maximum-likelihood covariance of the transition residuals."""

        return self._native.process_covariance

    @property
    def observation_covariance_(self) -> np.ndarray:
        """Maximum-likelihood covariance of the observation residuals."""

        return self._native.observation_covariance

    @property
    def initial_state_(self) -> np.ndarray:
        """Mean first state across the fitted segments.

        It estimates the state at the *first* sample of a segment, so replaying
        that segment's observations from it is the ``predict_first=False`` case
        of :meth:`KalmanFilter.filter`.
        """

        return self._native.initial_state

    @property
    def initial_covariance_(self) -> np.ndarray:
        """Maximum-likelihood covariance of the per-segment first states."""

        return self._native.initial_covariance


class LinearGaussianStateSpace:
    """Estimator of the parameters of a linear-Gaussian state-space model.

    See :class:`LinearGaussianParameters` for the affine model the fitted
    parameters describe.

    :meth:`fit` estimates the parameters from aligned state and observation
    sequences: the two maps are least-squares fits, the covariances are the
    maximum-likelihood residual covariances, and the initial state and covariance
    are the mean and maximum-likelihood covariance of the first state of every
    segment. ``segment_lengths`` splits the rows into independent segments, and no
    transition pair is ever formed across a segment boundary.

    ``jitter`` is the only regularization: it is added to the diagonal of every
    estimated covariance and nothing is added when it is zero. A single segment
    therefore yields an exactly zero ``initial_covariance_``, which starts a
    filter fully confident in ``initial_state_``; pass a positive ``jitter``, or
    an explicit initial covariance to :class:`KalmanFilter`, to avoid that.

    Each fit produces an immutable :class:`LinearGaussianParameters` snapshot,
    published as ``parameters_``; the fitted attributes below read from the
    current one. Refitting replaces the snapshot rather than modifying it, so
    filters already built from the previous one are unaffected.
    """

    def __init__(self, *, fit_offsets: bool = True, jitter: float = 0.0) -> None:
        if not isinstance(fit_offsets, bool):
            raise ValidationError("fit_offsets must be a bool.")
        self.fit_offsets = fit_offsets
        self.jitter = _validate_jitter(jitter)

    def fit(
        self,
        states: np.ndarray,
        observations: np.ndarray,
        *,
        segment_lengths: np.ndarray | None = None,
    ) -> Self:
        """Estimate the model from aligned state and observation sequences."""

        state_values = validate_real_array(states, "states", ndim=2)
        if not np.all(state_values.shape):
            raise ValidationError("states must not be empty.")
        observation_values = validate_real_array(observations, "observations", ndim=2)
        if not np.all(observation_values.shape):
            raise ValidationError("observations must not be empty.")
        if observation_values.shape[0] != state_values.shape[0]:
            raise ValidationError("observations must have the same number of samples as states.")
        lengths = _validate_segment_lengths(segment_lengths, state_values.shape[0])

        native = load_cpu_models_operation("state_space")
        try:
            result = native.fit_linear_gaussian(
                state_values,
                observation_values,
                lengths,
                self.fit_offsets,
                self.jitter,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

        self._parameters = LinearGaussianParameters(result["model"])
        self._n_samples = int(result["n_samples"])
        self._n_segments = int(result["n_segments"])
        self._n_transitions = int(result["n_transitions"])
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_parameters"):
            raise ValidationError("LinearGaussianStateSpace instance is not fitted.")

    @property
    def parameters_(self) -> LinearGaussianParameters:
        """Immutable snapshot of the parameters this fit produced."""

        self._check_is_fitted()
        return self._parameters

    @property
    def device_(self) -> str:
        """Device that holds the fitted state; always ``"cpu"``."""

        return self.parameters_.device_

    @property
    def n_samples_(self) -> int:
        """Number of samples seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._n_samples

    @property
    def n_segments_(self) -> int:
        """Number of independent segments seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._n_segments

    @property
    def n_transitions_(self) -> int:
        """Number of within-segment transition pairs used for the transition fit."""

        self._check_is_fitted()
        return self._n_transitions

    @property
    def n_states_(self) -> int:
        """Dimension of the latent state."""

        return self.parameters_.n_states_

    @property
    def n_observations_(self) -> int:
        """Dimension of one observation."""

        return self.parameters_.n_observations_

    @property
    def transition_(self) -> np.ndarray:
        """Fitted ``(n_states_, n_states_)`` state transition matrix."""

        return self.parameters_.transition_

    @property
    def transition_offset_(self) -> np.ndarray:
        """Fitted ``(n_states_,)`` transition offset; zero without ``fit_offsets``."""

        return self.parameters_.transition_offset_

    @property
    def observation_(self) -> np.ndarray:
        """Fitted ``(n_observations_, n_states_)`` observation matrix."""

        return self.parameters_.observation_

    @property
    def observation_offset_(self) -> np.ndarray:
        """Fitted ``(n_observations_,)`` observation offset; zero without ``fit_offsets``."""

        return self.parameters_.observation_offset_

    @property
    def process_covariance_(self) -> np.ndarray:
        """Maximum-likelihood covariance of the transition residuals."""

        return self.parameters_.process_covariance_

    @property
    def observation_covariance_(self) -> np.ndarray:
        """Maximum-likelihood covariance of the observation residuals."""

        return self.parameters_.observation_covariance_

    @property
    def initial_state_(self) -> np.ndarray:
        """Mean first state across the fitted segments."""

        return self.parameters_.initial_state_

    @property
    def initial_covariance_(self) -> np.ndarray:
        """Maximum-likelihood covariance of the per-segment first states."""

        return self.parameters_.initial_covariance_


class KalmanFilter:
    """Mutable filter state and covariance driven by one parameter snapshot.

    The filter binds to the immutable :class:`LinearGaussianParameters` snapshot
    the estimator held when it was constructed -- parameters, dimensions, and
    device all come from that snapshot, never from the estimator. Refitting the
    estimator therefore has no effect on filters already built from it: they
    keep reporting and using the parameters they actually run. This object owns
    only the evolving state and covariance, so several filters can run
    independently against one snapshot.

    Time convention, frozen: the filter state is the posterior estimate of
    ``x[t-1]``, the last step it completed. Each row handed to :meth:`filter` is
    the measurement ``z[t]`` of the next step, so a row costs one predict and
    one update and the returned row is the posterior ``x[t|t]``. Pass
    ``predict_first=False`` when the current state is instead the prior for the
    first row's own step -- as it is right after :meth:`reset`, where
    ``initial_state_`` estimates the state at the first sample of a segment
    rather than the one before it. Under either setting result row ``t`` is the
    posterior at the step that observation row ``t`` measures; the flag decides
    the prior of the *first* row alone. No row is ever dropped or shifted, so
    replaying a segment from ``initial_state_`` passes the whole of its
    ``observations`` rather than ``observations[1:]``.

    The gain comes from a symmetric linear solve of the innovation covariance
    rather than an explicit inverse, the posterior covariance uses the Joseph
    form, and every covariance is re-symmetrized after each step. ``jitter`` is
    the only regularization: it is added to the diagonal of the innovation
    covariance before the solve, and nothing is added when it is zero.

    Missing observations are explicit and are spelled ``nan``. :meth:`update`
    requires a finite observation, and :meth:`filter` rejects non-finite rows
    unless ``missing="predict"``, which runs the predict step alone for rows
    that are *entirely* ``nan``. Partially missing rows are rejected, and so is
    any infinity: an ``inf`` means an overflow or a division upstream, not an
    absent measurement.
    """

    def __init__(
        self,
        model: LinearGaussianStateSpace | LinearGaussianParameters,
        *,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        jitter: float = 0.0,
    ) -> None:
        if isinstance(model, LinearGaussianStateSpace):
            parameters = model.parameters_
        elif isinstance(model, LinearGaussianParameters):
            parameters = model
        else:
            raise ValidationError(
                "model must be a LinearGaussianStateSpace or LinearGaussianParameters instance."
            )
        self._parameters = parameters
        self._native = parameters._native
        self._n_state = parameters.n_states_
        self._n_observation = parameters.n_observations_
        self._device = parameters.device_
        self.jitter = _validate_jitter(jitter)

        n_states = self._n_state
        if initial_state is None:
            start_state = np.array(parameters.initial_state_, dtype=np.float64)
        else:
            start_state = validate_real_array(initial_state, "initial_state", ndim=1)
            if start_state.shape != (n_states,):
                raise ValidationError(f"initial_state must have shape ({n_states},).")
            start_state = np.array(start_state, dtype=np.float64)
        if initial_covariance is None:
            start_covariance = np.array(parameters.initial_covariance_, dtype=np.float64)
        else:
            start_covariance = _validate_covariance(
                initial_covariance,
                "initial_covariance",
                n_states,
            )

        self._initial_state = make_readonly(np.array(start_state))
        self._initial_covariance = make_readonly(np.array(start_covariance))
        self._state = np.ascontiguousarray(start_state, dtype=np.float64)
        self._covariance = np.ascontiguousarray(start_covariance, dtype=np.float64)

    def predict(self) -> np.ndarray:
        """Advance one step through the transition equation.

        The state goes from the posterior at ``t-1`` to the prior at ``t``.
        """

        self._call(self._native.predict, self._state, self._covariance)
        return self.state

    def update(self, observation: np.ndarray) -> np.ndarray:
        """Correct the current state with one fully observed measurement.

        The observation is ``z[t]`` of the step the state currently sits at, so
        the state goes from the prior at ``t`` to the posterior at ``t``.
        """

        values = validate_real_array(observation, "observation", ndim=1)
        if values.shape != (self.n_observations,):
            raise ValidationError(f"observation must have shape ({self.n_observations},).")
        self._call(
            self._native.update,
            self._state,
            self._covariance,
            np.ascontiguousarray(values, dtype=np.float64),
            self.jitter,
        )
        return self.state

    def filter(
        self,
        observations: np.ndarray,
        *,
        missing: str = "error",
        predict_first: bool = True,
    ) -> FilterResult:
        """Run predict/update over a batch and return every posterior.

        Row ``t`` of ``observations`` is the measurement of one step, and row
        ``t`` of the result is that step's posterior. With the default
        ``predict_first``, the state entering the call is the posterior of the
        step *before* the first row, so every row costs a predict and an update;
        this is the alignment for continuing a stream. With
        ``predict_first=False`` the state is instead the prior of the first
        row's own step, so that row is only corrected and every later row is
        unchanged; this is the alignment for replaying a segment from the fitted
        ``initial_state_``, or for the first frame of a trial after
        :meth:`reset`.
        """

        policy = validate_choice(missing, _MISSING_POLICIES, "missing")
        if not isinstance(predict_first, bool):
            raise ValidationError("predict_first must be a bool.")
        values = validate_real_array(observations, "observations", ndim=2, finite=False)
        if not np.all(values.shape):
            raise ValidationError("observations must not be empty.")
        if values.shape[1] != self.n_observations:
            raise ValidationError(
                "observations must have the same number of columns as the model observation "
                "dimension."
            )
        observed = self._observed_mask(values, policy)

        n_samples = values.shape[0]
        n_states = self.n_states
        states = np.empty((n_samples, n_states), dtype=np.float64)
        covariances = np.empty((n_samples, n_states, n_states), dtype=np.float64)
        start = 0
        if not predict_first:
            # The state already sits at the first row's step, so that row only
            # corrects it; a missing first row leaves it exactly as it is.
            if observed[0]:
                self._call(
                    self._native.update,
                    self._state,
                    self._covariance,
                    np.ascontiguousarray(values[0], dtype=np.float64),
                    self.jitter,
                )
            states[0] = self._state
            covariances[0] = self._covariance
            start = 1
        if start < n_samples:
            self._call(
                self._native.filter,
                values[start:],
                observed[start:],
                self._state,
                self._covariance,
                states[start:],
                covariances[start:],
                self.jitter,
            )
        return FilterResult(states=make_readonly(states), covariances=make_readonly(covariances))

    def reset(
        self,
        state: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Reset to the configured initial state, or to caller-supplied values."""

        if state is None:
            restored_state = self._initial_state
        else:
            values = validate_real_array(state, "state", ndim=1)
            if values.shape != (self.n_states,):
                raise ValidationError(f"state must have shape ({self.n_states},).")
            restored_state = values
        if covariance is None:
            restored_covariance = self._initial_covariance
        else:
            restored_covariance = _validate_covariance(covariance, "covariance", self.n_states)

        self._state[...] = restored_state
        self._covariance[...] = restored_covariance

    @staticmethod
    def _observed_mask(values: np.ndarray, policy: str) -> np.ndarray:
        if policy == "error":
            if not np.all(np.isfinite(values)):
                raise ValidationError(
                    "observations must contain finite values; pass missing='predict' to run the "
                    "predict step alone for rows that are entirely nan."
                )
            return np.ones(values.shape[0], dtype=np.uint8)
        if np.any(np.isinf(values)):
            # An infinity is an upstream overflow or division, not a measurement
            # that was never taken; only nan marks a row as missing.
            raise ValidationError(
                "observations must not contain infinite values; a missing row must be entirely nan."
            )
        missing = np.isnan(values)
        absent = missing.all(axis=1)
        if np.any(missing.any(axis=1) & ~absent):
            raise ValidationError(
                "partially missing observation rows are not supported; a missing row must be "
                "entirely nan."
            )
        return (~absent).astype(np.uint8)

    @staticmethod
    def _call(native_method, *arguments) -> None:
        try:
            native_method(*arguments)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

    @property
    def parameters(self) -> LinearGaussianParameters:
        """Immutable parameter snapshot that drives this filter."""

        return self._parameters

    @property
    def device_(self) -> str:
        """Device that holds the filter state; always ``"cpu"``."""

        return self._device

    @property
    def n_states(self) -> int:
        """Dimension of the latent state."""

        return self._n_state

    @property
    def n_observations(self) -> int:
        """Dimension of one observation."""

        return self._n_observation

    @property
    def state(self) -> np.ndarray:
        """Read-only copy of the current state estimate."""

        return make_readonly(np.array(self._state))

    @property
    def covariance(self) -> np.ndarray:
        """Read-only copy of the current state covariance."""

        return make_readonly(np.array(self._covariance))

    @property
    def initial_state(self) -> np.ndarray:
        """State that :meth:`reset` restores when no state is supplied."""

        return self._initial_state

    @property
    def initial_covariance(self) -> np.ndarray:
        """Covariance that :meth:`reset` restores when none is supplied."""

        return self._initial_covariance


__all__ = [
    "FilterResult",
    "KalmanFilter",
    "LinearGaussianParameters",
    "LinearGaussianStateSpace",
]
