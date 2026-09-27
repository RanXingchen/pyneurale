#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Signal smoothing algorithms."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.sparse.linalg import MatrixRankWarning
from scipy.special import comb

from neurale._validation import (
    validate_integer,
    validate_number,
    validate_numeric_array,
    validate_positive_float,
)
from neurale.data import SignalArray
from neurale.exceptions import ValidationError

from ._input import (
    normalize_signal_input,
    restore_signal_output,
    uniform_sampling_rate,
)


def moving_average(
    x: np.ndarray | SignalArray,
    span: int = 5,
    *,
    axis: int = 0,
) -> np.ndarray | SignalArray:
    """Smooth a signal with centered odd-width moving averages.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.arrays.SignalArray
        One- or two-dimensional signal.
    span : int, default=5
        Positive odd window width not exceeding the sample count.
    axis : int, default=0
        Sample axis for ndarray input.

    Returns
    -------
    numpy.ndarray or neurale.data.arrays.SignalArray
        Smoothed signal with shrinking windows at the boundaries.

    Raises
    ------
    neurale.exceptions.ValidationError
        If x, axis, or span is invalid.
    """
    span = validate_integer(span, "span", minimum=1)
    if span % 2 == 0:
        raise ValidationError("span must be odd.")
    data, context = normalize_signal_input(x, axis=axis)
    n_samples = data.shape[0]
    if n_samples == 0:
        raise ValidationError("signal must contain at least one sample.")
    if span > n_samples:
        raise ValidationError("span must not exceed the number of samples.")

    result_dtype = np.result_type(data.dtype, np.float64)
    result = np.empty(data.shape, dtype=result_dtype)
    half_span = span // 2
    for sample in range(n_samples):
        radius = min(half_span, sample, n_samples - sample - 1)
        result[sample] = np.mean(
            data[sample - radius : sample + radius + 1],
            axis=0,
            dtype=result_dtype,
        )
    return restore_signal_output(result, context)


def gaussian_smooth(
    x: np.ndarray | SignalArray,
    width: float = 3.0,
    *,
    axis: int = 0,
) -> np.ndarray | SignalArray:
    """Smooth a signal with PyNeurale's Gaussian width convention.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.arrays.SignalArray
        One- or two-dimensional signal.
    width : float, default=3.0
        Positive Gaussian half-height width in samples.
    axis : int, default=0
        Sample axis for ndarray input.

    Returns
    -------
    numpy.ndarray or neurale.data.arrays.SignalArray
        Boundary-normalized smoothed signal.

    Raises
    ------
    neurale.exceptions.ValidationError
        If x, axis, or width is invalid.
    """
    width = validate_positive_float(width, "width")
    data, context = normalize_signal_input(x, axis=axis)
    n_samples = data.shape[0]
    if n_samples == 0:
        raise ValidationError("signal must contain at least one sample.")

    radius = int(width) + 1
    offsets = np.arange(-radius, radius + 1, dtype=float)
    sigma = -(width * width) * 0.25 / np.log(0.5)
    kernel = np.exp(-(offsets * offsets) / sigma)
    kernel /= np.sum(kernel)

    result_dtype = np.result_type(data.dtype, np.float64)
    result = np.empty(data.shape, dtype=result_dtype)
    normalization = _centered_convolution(np.ones(n_samples), kernel)
    for channel in range(data.shape[1]):
        result[:, channel] = _centered_convolution(data[:, channel], kernel) / normalization
    return restore_signal_output(result, context)


def _centered_convolution(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full = np.convolve(x, kernel, mode="full")
    start = kernel.size // 2
    return full[start : start + x.size]


class WhittakerSmoother:
    """Smooth signals by penalized least squares.

    Parameters
    ----------
    order : int, default=2
        Positive finite-difference penalty order.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``order`` is not a positive integer.
    """

    def __init__(self, order: int = 2) -> None:
        self.order = validate_integer(order, "order", minimum=1)

    def apply(
        self,
        x: np.ndarray | SignalArray,
        weight: np.ndarray | None = None,
        lam: float = 1.0,
        *,
        axis: int = 0,
    ) -> np.ndarray | SignalArray:
        """Smooth one or more uniformly sampled channels.

        Parameters
        ----------
        x : numpy.ndarray or neurale.data.arrays.SignalArray
            Signal to smooth.
        weight : numpy.ndarray or None, optional
            Non-negative sample weights matching or broadcasting across the signal.
        lam : float, default=1.0
            Non-negative smoothing penalty.
        axis : int, default=0
            Sample axis for ndarray input.

        Returns
        -------
        numpy.ndarray or neurale.data.arrays.SignalArray
            Smoothed signal.

        Raises
        ------
        neurale.exceptions.ValidationError
            If x, weights, sampling, order, or penalty are invalid.
        """
        lam = float(validate_number(lam, "lam", kind="real", minimum=0, coerce=True))
        data, context = normalize_signal_input(x, axis=axis)
        uniform_sampling_rate(x)
        if data.shape[0] <= self.order:
            raise ValidationError("signal length must be greater than order.")
        weights = _normalize_weights(weight, data, axis=axis)
        penalty = _difference_penalty(data.shape[0], self.order, sparse)

        result_dtype = np.result_type(data.dtype, np.float64)
        result = np.empty(data.shape, dtype=result_dtype)
        for channel in range(data.shape[1]):
            channel_weight = weights[:, channel]
            channel_data = data[:, channel]
            if np.any(~np.isfinite(channel_data[channel_weight > 0])):
                raise ValidationError("signal values with positive weights must be finite.")
            safe_data = np.where(channel_weight > 0, channel_data, 0)
            weight_matrix = sparse.diags(channel_weight, format="csc")
            system = weight_matrix + lam * penalty
            result[:, channel] = _solve_sparse_system(
                system,
                channel_weight * safe_data,
                "Whittaker smoothing system is singular.",
            )
        return restore_signal_output(result, context)

    def get_optimal_lambda(
        self,
        x: np.ndarray | SignalArray,
        weight: np.ndarray | None = None,
        *,
        min_exp: float = -2.0,
        max_exp: float = 8.0,
        exp_step: float = 0.2,
        axis: int = 0,
    ) -> float:
        """Select a smoothing penalty by leave-one-out cross-validation.

        Parameters
        ----------
        x : numpy.ndarray or neurale.data.arrays.SignalArray
            Single-channel uniformly sampled signal.
        weight : numpy.ndarray or None, optional
            Non-negative sample weights.
        min_exp : float, default=-2.0
            Minimum base-10 exponent searched.
        max_exp : float, default=8.0
            Maximum base-10 exponent searched.
        exp_step : float, default=0.2
            Positive exponent step.
        axis : int, default=0
            Sample axis for ndarray input.

        Returns
        -------
        float
            Candidate lambda with minimum cross-validation error.

        Raises
        ------
        neurale.exceptions.ValidationError
            If input, weights, sampling, or search bounds are invalid.
        """
        data, _ = normalize_signal_input(x, axis=axis)
        uniform_sampling_rate(x)
        if data.shape[1] != 1:
            raise ValidationError("get_optimal_lambda requires exactly one signal channel.")
        if data.shape[0] <= self.order:
            raise ValidationError("signal length must be greater than order.")
        weights = _normalize_weights(weight, data, axis=axis)[:, 0]
        values = data[:, 0]
        if np.any(~np.isfinite(values[weights > 0])):
            raise ValidationError("signal values with positive weights must be finite.")
        values = np.where(weights > 0, values, 0)

        min_exp = float(validate_number(min_exp, "min_exp", kind="real", coerce=True))
        max_exp = float(validate_number(max_exp, "max_exp", kind="real", coerce=True))
        exp_step = validate_positive_float(exp_step, "exp_step")
        if min_exp > max_exp:
            raise ValidationError("min_exp must not exceed max_exp.")
        count = int(np.floor((max_exp - min_exp) / exp_step)) + 1
        exponents = min_exp + np.arange(count, dtype=float) * exp_step
        if exponents[-1] < max_exp and not np.isclose(exponents[-1], max_exp):
            exponents = np.append(exponents, max_exp)
        lambdas = np.power(10.0, exponents)

        n_samples = values.size
        penalty = _difference_penalty(n_samples, self.order, sparse)
        weight_matrix = sparse.diags(weights, format="csc")
        errors = np.empty(lambdas.size, dtype=float)

        if n_samples <= 100:
            for i, lam in enumerate(lambdas):
                system = weight_matrix + lam * penalty
                fitted = _solve_sparse_system(
                    system,
                    weights * values,
                    "Whittaker cross-validation system is singular.",
                )
                inverse = _invert_sparse_system(system)
                leverage = np.asarray((weight_matrix @ inverse).diagonal()).reshape(-1)
                errors[i] = _cross_validation_error(values, fitted, weights, leverage)
        else:
            errors[:] = self._large_sample_cv(
                values,
                weights,
                lambdas,
                penalty,
                weight_matrix,
            )
        return float(lambdas[int(np.nanargmin(errors))])

    def _large_sample_cv(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        lambdas: np.ndarray,
        penalty: Any,
        weight_matrix: Any,
    ) -> np.ndarray:
        # Approximate leverage with bounded 100-point subsampling.
        n_samples = x.size
        reduced_size = 100
        ratio = (n_samples - 1) / (reduced_size - 1)
        reduced_indices = np.round(np.arange(reduced_size, dtype=float) * ratio).astype(int)
        reduced_weights = weights[reduced_indices]
        reduced_weight_matrix = sparse.diags(reduced_weights, format="csc")
        reduced_penalty = _difference_penalty(reduced_size, self.order, sparse)
        center = n_samples // 2 - 1
        reduced_center = reduced_size // 2 - 1
        pulse = np.zeros(n_samples)
        pulse[center] = 1.0
        mapped_indices = np.minimum(
            np.round(np.arange(n_samples) / ratio).astype(int),
            reduced_size - 1,
        )

        errors = np.empty(lambdas.size, dtype=float)
        for i, lam in enumerate(lambdas):
            system = weight_matrix + lam * penalty
            fitted = _solve_sparse_system(
                system,
                weights * x,
                "Whittaker cross-validation system is singular.",
            )
            scaled_lam = lam * (reduced_size / n_samples) ** (2 * self.order)
            reduced_inverse = _invert_sparse_system(
                reduced_weight_matrix + scaled_lam * reduced_penalty
            )
            reduced_leverage = np.asarray(reduced_inverse.diagonal()).reshape(-1)
            impulse_response = _solve_sparse_system(
                system,
                pulse,
                "Whittaker leverage system is singular.",
            )
            den = reduced_leverage[reduced_center]
            if den == 0:
                errors[i] = np.inf
                continue
            leverage = weights * reduced_leverage[mapped_indices] * impulse_response[center] / den
            errors[i] = _cross_validation_error(x, fitted, weights, leverage)
        return errors


def _normalize_weights(
    weight: np.ndarray | None,
    x: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    if weight is None:
        return np.ones(x.shape, dtype=float)
    weight = validate_numeric_array(weight, "weight")
    if weight.ndim == 1:
        if weight.size != x.shape[0]:
            raise ValidationError("1D weight length must match the sample count.")
        normalized = np.broadcast_to(weight[:, None], x.shape)
    elif weight.ndim == 2:
        normalized, _ = normalize_signal_input(weight, axis=axis)
        if normalized.shape != x.shape:
            raise ValidationError("weight shape must match signal shape.")
    else:
        raise ValidationError("weight must be one- or two-dimensional.")
    normalized = np.asarray(normalized, dtype=float)
    if np.any(~np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValidationError("weight values must be finite and non-negative.")
    if np.any(np.sum(normalized, axis=0) == 0):
        raise ValidationError("each channel must have at least one positive weight.")
    return normalized


def _difference_penalty(length: int, order: int, sparse: Any) -> Any:
    coefs = np.asarray(
        [(-1.0) ** (order - idx) * comb(order, idx, exact=False) for idx in range(order + 1)]
    )
    difference = sparse.diags(
        coefs,
        offsets=np.arange(order + 1),
        shape=(length - order, length),
        format="csc",
    )
    return difference.T @ difference


def _solve_sparse_system(
    system: Any,
    rhs: np.ndarray,
    message: str,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            result = sparse_linalg.spsolve(system, rhs)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise ValidationError(message) from exc
    result = np.asarray(result)
    if np.any(~np.isfinite(result)):
        raise ValidationError(message)
    return result


def _invert_sparse_system(system: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            inverse = sparse_linalg.inv(system)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise ValidationError("Whittaker cross-validation system is singular.") from exc
    if np.any(~np.isfinite(inverse.data)):
        raise ValidationError("Whittaker cross-validation system is singular.")
    return inverse


def _cross_validation_error(
    x: np.ndarray,
    fitted: np.ndarray,
    weights: np.ndarray,
    leverage: np.ndarray,
) -> float:
    den = 1.0 - leverage
    valid = (weights > 0) & (np.abs(den) > np.finfo(float).eps)
    if not np.any(valid):
        return np.inf
    residual = (x[valid] - fitted[valid]) / den[valid]
    return float(np.sqrt(np.sum(weights[valid] * residual * residual) / np.sum(weights[valid])))
