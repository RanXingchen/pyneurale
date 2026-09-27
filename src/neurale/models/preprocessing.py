#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Feature preparation transforms shared by classical decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from neurale._validation import validate_integer, validate_number, validate_real_array
from neurale.exceptions import ValidationError

from ._dispatch import require_cpu_models_operation
from ._inference import make_readonly, validate_2d_array

# Scale used for features whose fitted spread is zero. Transforming such a
# feature therefore only removes its offset, and inverse_transform() restores
# the original constant exactly instead of dividing by zero.
_CONSTANT_SCALE = 1.0


def _safe_scale(spread: np.ndarray) -> np.ndarray:
    # Replace a zero spread by the documented constant-feature scale.
    scale = np.array(spread, dtype=np.float64, copy=True)
    scale[scale == 0.0] = _CONSTANT_SCALE
    return scale


def _require_representable(**statistics: np.ndarray) -> None:
    # A finite matrix can still have statistics that overflow float64. Such a
    # fit is rejected rather than stored, because an infinite statistic silently
    # turns transform() into a constant map and inverse_transform() into NaN.
    for name, value in statistics.items():
        if not np.all(np.isfinite(value)):
            raise ValidationError(
                f"fitted {name} is not representable in float64; X is finite but its {name} "
                "overflows. Rescale X before fitting."
            )


def _mean_variance_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-feature mean, biased variance, and standard deviation.

    Every accumulation runs on features divided by their largest magnitude, so
    the sums stay inside ``[-1, 1]`` and cannot overflow for finite input. Only
    the final rescaling can leave the float64 range, and the variance is derived
    from the standard deviation so that a representable spread never turns into
    ``0 * inf``. The caller rejects whatever is still not representable.
    """

    magnitude = np.max(np.abs(X), axis=0)
    unit = np.where(magnitude == 0.0, 1.0, magnitude)
    scaled = X / unit
    scaled_mean = np.mean(scaled, axis=0)
    scaled_variance = np.mean(np.square(scaled - scaled_mean), axis=0)

    mean = scaled_mean * unit
    std = np.sqrt(scaled_variance) * unit
    with np.errstate(over="ignore"):
        variance = np.square(std)
    return mean, variance, std


def _restored_statistics(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    # One place for the checks every restored scaler needs: each statistic is a
    # finite 1D vector of the same length, published read-only.
    # A restore that got this wrong would not fail until the first transform,
    # by which point the caller is holding a scaler that looks fitted.
    restored: dict[str, np.ndarray] = {}
    length: int | None = None
    for name, values in arrays.items():
        vector = validate_real_array(values, name, ndim=1)
        if vector.shape[0] == 0:
            raise ValidationError(f"{name} must contain at least one feature.")
        if not np.all(np.isfinite(vector)):
            raise ValidationError(f"{name} must contain finite statistics.")
        if length is None:
            length = int(vector.shape[0])
        elif int(vector.shape[0]) != length:
            raise ValidationError("the restored statistics must describe the same features.")
        restored[name] = make_readonly(np.asarray(vector, dtype=np.float64))
    return restored


class _FittedScaler:
    """Shared fit/transform plumbing for the CPU-only feature scalers."""

    def fit(self, X: np.ndarray) -> Self:
        """Fit the scaler on a sample-by-feature matrix."""

        self._check_not_frozen()
        require_cpu_models_operation(f"preprocessing.{type(self).__name__}.fit")
        return self._fit_validated(validate_2d_array(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler and return the transformed fit data."""

        self._check_not_frozen()
        require_cpu_models_operation(f"preprocessing.{type(self).__name__}.fit")
        X = validate_2d_array(X)
        self._fit_validated(X)
        return self._forward(X)

    def freeze(self) -> Self:
        """Fix the fitted statistics permanently and return the scaler.

        A frozen scaler still transforms, inverts, and reports what it
        measured; what it can no longer do is *become a different scaler*. It
        cannot be refitted and its fitted state cannot be rebound, so a caller
        that receives one cannot change what its owner computes.

        This is what an estimator that owns a scaler publishes. The statistic
        arrays are read-only views already; freezing closes the remaining way
        to replace them wholesale. Freezing is one-way and idempotent: there is
        no thaw, because the point is that a published scaler stays the one
        that was published. Fit a new scaler instead.

        Returns
        -------
        Self
            The scaler itself, so a fit can be published in one expression.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the scaler is not fitted; there is nothing to fix in place.
        """

        self._check_is_fitted()
        object.__setattr__(self, "_frozen", True)
        return self

    @property
    def is_frozen(self) -> bool:
        """Whether the fitted statistics have been fixed by :meth:`freeze`."""

        return getattr(self, "_frozen", False)

    def __setattr__(self, name: str, value: object) -> None:
        if self.is_frozen:
            raise AttributeError(f"{type(self).__name__} instance is frozen.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.is_frozen:
            raise AttributeError(f"{type(self).__name__} instance is frozen.")
        object.__delattr__(self, name)

    def _check_not_frozen(self) -> None:
        if self.is_frozen:
            raise ValidationError(
                f"{type(self).__name__} instance is frozen; its fitted statistics cannot "
                "change. Fit a new scaler instead."
            )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale a sample-by-feature matrix with the fitted statistics."""

        return self._forward(self._validate_transform_input(X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo :meth:`transform` for a scaled sample-by-feature matrix."""

        return self._backward(self._validate_transform_input(X))

    def _validate_transform_input(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = validate_2d_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValidationError("X must have the same number of features as fit data.")
        return X

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_device"):
            raise ValidationError(f"{type(self).__name__} instance is not fitted.")

    def _fit_validated(self, X: np.ndarray) -> Self:
        raise NotImplementedError

    def _forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _backward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def device_(self) -> str:
        """Device that holds the fitted state; always ``"cpu"``."""

        self._check_is_fitted()
        return self._device

    @property
    def n_samples_(self) -> int:
        """Number of samples seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._n_samples

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._n_features_in


class StandardScaler(_FittedScaler):
    """Standardize features by removing the mean and scaling to unit variance.

    ``fit()`` records per-feature statistics on the CPU; ``transform()`` never
    updates them, so repeated inference is deterministic. Features whose fitted
    variance is zero use a scale of ``1.0``, which keeps ``inverse_transform()``
    an exact round trip instead of dividing by zero.

    :attr:`mean_` and :attr:`var_` always describe the fit data. ``with_mean``
    and ``with_std`` select what :meth:`transform` applies -- a disabled option
    neutralizes the corresponding term of the affine map, leaving :attr:`mean_`
    at zero or :attr:`scale_` at one, but never changes what was measured.

    A matrix of finite values can still have statistics that overflow float64.
    The accumulations are scaled so that finite statistics stay finite, and a
    fit whose mean or variance is still unrepresentable raises
    :class:`~neurale.exceptions.ValidationError` instead of recording it.
    """

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
        if not isinstance(with_mean, bool):
            raise ValidationError("with_mean must be a bool.")
        if not isinstance(with_std, bool):
            raise ValidationError("with_std must be a bool.")
        self.with_mean = with_mean
        self.with_std = with_std

    def _fit_validated(self, X: np.ndarray) -> Self:
        n_features = X.shape[1]
        # The statistics always describe the fit data: ``with_mean`` and
        # ``with_std`` select what transform() applies, not what fit() measures.
        data_mean, variance, std = _mean_variance_std(X)
        _require_representable(mean=data_mean, variance=variance)
        mean = data_mean if self.with_mean else np.zeros(n_features, dtype=np.float64)
        scale = _safe_scale(std) if self.with_std else np.ones(n_features, dtype=np.float64)

        self._n_samples = int(X.shape[0])
        self._n_features_in = int(n_features)
        self._mean = make_readonly(mean)
        self._var = make_readonly(variance)
        self._scale = make_readonly(scale)
        self._device = "cpu"
        return self

    @classmethod
    def _restore_fitted(
        cls,
        *,
        with_mean: bool,
        with_std: bool,
        mean: np.ndarray,
        var: np.ndarray,
        scale: np.ndarray,
        n_samples: int,
    ) -> Self:
        """Rebuild a fitted scaler from statistics a fit already produced.

        Private, and the only supported way to reconstitute a fit: decoder
        persistence stores the measured statistics and has to restore *this*
        transform rather than re-derive one from data it no longer has. The
        arrays are taken as the fit's own output and are not re-measured; they
        are checked only for the shape agreement the transform depends on.
        """

        statistics = _restored_statistics(mean=mean, var=var, scale=scale)
        scaler = cls(with_mean=with_mean, with_std=with_std)
        scaler._n_samples = validate_integer(n_samples, "n_samples", minimum=1)
        scaler._n_features_in = int(statistics["mean"].shape[0])
        scaler._mean = statistics["mean"]
        scaler._var = statistics["var"]
        scaler._scale = statistics["scale"]
        scaler._device = "cpu"
        return scaler

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._scale

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return X * self._scale + self._mean

    @property
    def scale_(self) -> np.ndarray:
        """Divisor applied by :meth:`transform`.

        The per-feature standard deviation of the fit data, with ``1.0`` for
        constant features, or all ones when ``with_std`` is disabled.
        """

        self._check_is_fitted()
        return self._scale

    @property
    def mean_(self) -> np.ndarray:
        """Offset removed by :meth:`transform`.

        The per-feature mean of the fit data, or all zeros when ``with_mean``
        is disabled.
        """

        self._check_is_fitted()
        return self._mean

    @property
    def var_(self) -> np.ndarray:
        """Per-feature biased variance of the fit data, about its own mean.

        Measured whether or not ``with_std`` is enabled, so this attribute is
        the variance of the fit data rather than a description of
        :attr:`scale_`.
        """

        self._check_is_fitted()
        return self._var


class MinMaxScaler(_FittedScaler):
    """Scale features onto an explicit finite range.

    Each feature is mapped from its fitted ``[data_min_, data_max_]`` interval
    onto ``feature_range``. Features that are constant during :meth:`fit` have a
    zero data range and use a scale of ``1.0``; they map to the lower bound of
    ``feature_range`` and ``inverse_transform()`` restores the fitted constant.

    A matrix of finite values can still span more than float64 can represent. A
    fit whose data range or affine coefficients are unrepresentable raises
    :class:`~neurale.exceptions.ValidationError` instead of recording a map that
    collapses every sample onto the lower bound.
    """

    def __init__(self, *, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if not isinstance(feature_range, tuple) or len(feature_range) != 2:
            raise ValidationError("feature_range must be a tuple of two numbers.")
        low = float(validate_number(feature_range[0], "feature_range[0]", kind="real", coerce=True))
        high = float(
            validate_number(feature_range[1], "feature_range[1]", kind="real", coerce=True)
        )
        if not low < high:
            raise ValidationError("feature_range must satisfy feature_range[0] < feature_range[1].")
        self.feature_range = (low, high)

    def _fit_validated(self, X: np.ndarray) -> Self:
        low, high = self.feature_range
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        # A finite feature can still span more than float64 can represent, and a
        # subnormal span can still blow the affine coefficients up. Compute both
        # quietly and reject the fit rather than storing an unusable map.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            data_range = data_max - data_min
            scale = (high - low) / _safe_scale(data_range)
            minimum = low - data_min * scale
        _require_representable(
            data_min=data_min,
            data_max=data_max,
            data_range=data_range,
            scale=scale,
            min=minimum,
        )

        self._n_samples = int(X.shape[0])
        self._n_features_in = int(X.shape[1])
        self._data_min = make_readonly(data_min)
        self._data_max = make_readonly(data_max)
        self._data_range = make_readonly(data_range)
        self._scale = make_readonly(scale)
        self._min = make_readonly(minimum)
        self._device = "cpu"
        return self

    @classmethod
    def _restore_fitted(
        cls,
        *,
        feature_range: tuple[float, float],
        data_min: np.ndarray,
        data_max: np.ndarray,
        data_range: np.ndarray,
        scale: np.ndarray,
        min: np.ndarray,
        n_samples: int,
    ) -> Self:
        """Rebuild a fitted scaler from statistics a fit already produced.

        See :meth:`StandardScaler._restore_fitted`.
        """

        statistics = _restored_statistics(
            data_min=data_min,
            data_max=data_max,
            data_range=data_range,
            scale=scale,
            min=min,
        )
        scaler = cls(feature_range=feature_range)
        scaler._n_samples = validate_integer(n_samples, "n_samples", minimum=1)
        scaler._n_features_in = int(statistics["data_min"].shape[0])
        scaler._data_min = statistics["data_min"]
        scaler._data_max = statistics["data_max"]
        scaler._data_range = statistics["data_range"]
        scaler._scale = statistics["scale"]
        scaler._min = statistics["min"]
        scaler._device = "cpu"
        return scaler

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X * self._scale + self._min

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return (X - self._min) / self._scale

    @property
    def scale_(self) -> np.ndarray:
        """Per-feature multiplier applied by :meth:`transform`."""

        self._check_is_fitted()
        return self._scale

    @property
    def min_(self) -> np.ndarray:
        """Per-feature offset added by :meth:`transform` after scaling."""

        self._check_is_fitted()
        return self._min

    @property
    def data_min_(self) -> np.ndarray:
        """Per-feature minimum seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._data_min

    @property
    def data_max_(self) -> np.ndarray:
        """Per-feature maximum seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._data_max

    @property
    def data_range_(self) -> np.ndarray:
        """Per-feature ``data_max_ - data_min_`` seen during :meth:`fit`."""

        self._check_is_fitted()
        return self._data_range


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """Stack neighboring frames into a fixed temporal context window.

    The transform is immutable and deterministic: it has no fitted state and no
    device state. Output row ``i`` concatenates source rows ``i`` through
    ``i + left + right`` — oldest frame first, newest last — keeping features in
    their original order inside each frame. ``transform()`` therefore drops the
    first ``left`` and last ``right`` rows, and :meth:`trim` applies the same
    trimming to an aligned target array.

    ``right > 0`` looks into the future and is offline-only. Causal online use
    requires ``right = 0``, reported by :attr:`is_causal`.
    """

    left: int = 0
    right: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", validate_integer(self.left, "left", minimum=0))
        object.__setattr__(self, "right", validate_integer(self.right, "right", minimum=0))

    @property
    def n_frames(self) -> int:
        """Number of source frames stacked into one output row."""

        return self.left + self.right + 1

    @property
    def is_causal(self) -> bool:
        """Whether the context only uses past and current frames."""

        return self.right == 0

    def output_length(self, n_samples: int) -> int:
        """Return the number of output rows produced for ``n_samples`` rows."""

        count = validate_integer(n_samples, "n_samples", minimum=0)
        return max(count - self.left - self.right, 0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Stack the temporal context of a sample-by-feature matrix."""

        X = validate_2d_array(X)
        n_out = self._require_length(X.shape[0], "X")
        blocks = [X[offset : offset + n_out] for offset in range(self.n_frames)]
        return np.concatenate(blocks, axis=1)

    def trim(self, y: np.ndarray) -> np.ndarray:
        """Trim a target array to the rows kept by :meth:`transform`."""

        if not isinstance(y, np.ndarray):
            raise ValidationError("y must be a numpy.ndarray.")
        if y.ndim not in (1, 2):
            raise ValidationError("y must be a 1D or 2D array.")
        if not np.all(y.shape):
            raise ValidationError("y must not be empty.")
        n_out = self._require_length(y.shape[0], "y")
        return y[self.left : self.left + n_out].copy()

    def _require_length(self, n_samples: int, name: str) -> int:
        n_out = self.output_length(n_samples)
        if n_out == 0:
            raise ValidationError(
                f"{name} must have at least left + right + 1 = {self.n_frames} samples; "
                f"got {n_samples}."
            )
        return n_out


__all__ = ["MinMaxScaler", "StandardScaler", "TemporalContext"]
