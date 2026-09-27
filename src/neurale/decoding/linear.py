#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Continuous decoding by least-squares and penalized least-squares regression."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neurale._validation import validate_number
from neurale.data import FeatureMatrix, SignalArray
from neurale.exceptions import ValidationError
from neurale.models.linear_model import LinearRegression, Ridge
from neurale.models.preprocessing import TemporalContext

from ._stages import FeatureScaler, FeatureStages, validate_stages
from .base import ContinuousDecoder, FitSegment


class _RegressionDecoder(ContinuousDecoder):
    """Shared lifecycle of the two multi-output regression decoders.

    Both own the same two explicit stages and are equally stateless; they
    differ only in the estimator a fit builds, which is what
    :meth:`_build_model` returns.
    """

    def __init__(
        self,
        *,
        scaler: FeatureScaler | None = None,
        context: TemporalContext | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.fit_intercept = _validate_fit_intercept(fit_intercept)
        self.scaler, self.context = validate_stages(scaler, context)

    def _build_model(self) -> LinearRegression | Ridge:
        raise NotImplementedError

    def _validate_configuration(self) -> None:
        """Validate the current configuration before a fit consumes it.

        Every constructor argument stays a plain public attribute, so what
        ``__init__`` accepted is not necessarily what a later fit will use. The
        estimator would reject a rebound one too, but only from inside
        :meth:`_fit_decoder`, by which point the previous fit is already gone;
        checking here keeps an invalid configuration an argument error and
        leaves a fitted decoder untouched.
        """

        validate_stages(self.scaler, self.context)
        _validate_fit_intercept(self.fit_intercept)

    # ----------------------------------------------------------------- fitting

    def _prepare_fit_segment(self, X: FeatureMatrix, y: SignalArray) -> FitSegment:
        # The base checks alignment; the configuration is checked here too, so
        # that every argument error is raised before _fit_prepared discards
        # anything and a rejected refit leaves a fitted decoder untouched.
        prepared = super()._prepare_fit_segment(X, y)
        self._validate_configuration()
        FeatureStages(None, self.context).require_frames(prepared.X.shape[0], "X")
        return prepared

    def _fit_decoder(self, segments: Sequence[FitSegment]) -> None:
        stages, blocks = FeatureStages.fitted(
            self.scaler,
            self.context,
            [segment.X for segment in segments],
        )
        X = np.concatenate(blocks, axis=0, dtype=np.float64)
        y = np.concatenate(
            [stages.trim(np.asarray(segment.y.data, dtype=np.float64)) for segment in segments],
            axis=0,
        )

        model = self._build_model()
        model.fit(X, y)

        self._stages = stages
        self._model = model
        self._n_samples = int(X.shape[0])

    def _clear_fitted(self) -> None:
        super()._clear_fitted()
        self._stages = None
        self._model = None
        self._n_samples = None

    # -------------------------------------------------------------- prediction

    def _decode(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._stages.transform(X))

    def _output_time(self, time: np.ndarray) -> np.ndarray:
        # The same transform that trimmed the fit target trims the timestamps,
        # so a result can never carry a time for a row it did not predict.
        return self._stages.trim(np.asarray(time, dtype=np.float64), name="X")

    # ---------------------------------------------------------- fitted results

    @property
    def scaler_(self) -> FeatureScaler | None:
        """Fitted feature scaler, or ``None`` when the features are unscaled.

        The decoder's own copy of the configured :attr:`scaler`; the instance
        handed to the constructor is never fitted or mutated. The copy is
        frozen, so it can be read from and applied but not refitted behind the
        decoder that runs it.
        """

        self._check_is_fitted()
        return self._stages.scaler

    @property
    def context_(self) -> TemporalContext | None:
        """Temporal context the fit used, or ``None`` for one frame per prediction."""

        self._check_is_fitted()
        return self._stages.context

    @property
    def n_model_features_(self) -> int:
        """Number of columns the regression consumes.

        :attr:`~BaseDecoder.n_features_in_` times the number of frames a
        context stacks, which is the same number when there is no context.
        """

        self._check_is_fitted()
        return self._stages.n_model_features(self.n_features_in_)

    @property
    def n_samples_(self) -> int:
        """Number of rows the regression was fitted on, after any trimming."""

        self._check_is_fitted()
        return self._n_samples

    @property
    def coef_(self) -> np.ndarray:
        """``(n_outputs_, n_model_features_)`` fitted coefficients."""

        self._check_is_fitted()
        return self._model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """``(n_outputs_,)`` fitted intercepts; zero when they are not fitted."""

        self._check_is_fitted()
        return self._model.intercept_

    @property
    def rank_(self) -> int:
        """Numerical rank of the design the fit factorized."""

        self._check_is_fitted()
        return self._model.rank_

    @property
    def singular_values_(self) -> np.ndarray:
        """Singular values of the design the fit factorized."""

        self._check_is_fitted()
        return self._model.singular_values_


class LinearDecoder(_RegressionDecoder):
    """Decode a continuous target by multi-output least squares.

    The decoder composes owned pieces and adds no arithmetic of its own: an
    optional feature :attr:`scaler`, an optional temporal :attr:`context`, and
    a :class:`~neurale.models.linear_model.LinearRegression` over the result.
    Every output channel is regressed against the same design, so the outputs
    come from one factorization but do not constrain each other.

    Prediction is stateless. The same frames decode to the same values in any
    order, and :meth:`~BaseDecoder.reset` has nothing to return to -- it is a
    no-op by inheritance, not by an override that does nothing.

    Parameters
    ----------
    scaler : neurale.models.preprocessing.StandardScaler or \
neurale.models.preprocessing.MinMaxScaler or None, optional
        Feature scaler to own. The decoder fits a *copy*, so the instance
        passed here stays a configuration and is never mutated; the fitted copy
        is :attr:`scaler_`, published frozen. A scaler that is already frozen
        is a published fit rather than a configuration, and is rejected.
        ``None`` feeds the features to the model unscaled.
    context : neurale.models.preprocessing.TemporalContext or None, optional
        Temporal context to stack before regressing. It drops the first
        ``left`` and last ``right`` frames, and the fit target and every
        predicted timestamp lose exactly those same frames. ``None`` decodes
        one output row per feature frame.
    fit_intercept : bool, optional
        Whether the regression estimates an intercept. An unfitted intercept is
        reported as zero rather than omitted.

    Raises
    ------
    neurale.exceptions.ValidationError
        If any argument is invalid.

    Notes
    -----
    Scaling runs before context stacking, so the scaler measures one statistic
    per real feature rather than one per (feature, lag) pair. Nothing else
    happens inside: no feature-subset search, no dimensionality reduction, no
    resampling, no trial slicing. A caller who wants a fitted
    :class:`~neurale.models.decomposition.PCA` or a fixed feature subset
    applies it to the :class:`~neurale.data.arrays.FeatureMatrix` first and
    fits the decoder on the result, where the transform stays visible and the
    fitted schema describes what the decoder actually consumed.

    Examples
    --------
    >>> import numpy as np
    >>> from neurale.data import FeatureMatrix, SignalArray
    >>> from neurale.decoding import LinearDecoder
    >>> rng = np.random.default_rng(0)
    >>> rate = 50.0
    >>> position = np.cumsum(rng.normal(scale=0.05, size=(200, 2)), axis=0)
    >>> X = FeatureMatrix(
    ...     data=position @ rng.normal(size=(2, 6)) + rng.normal(scale=0.1, size=(200, 6)),
    ...     fs=rate,
    ...     feature_names=[f"rate_{i}" for i in range(6)],
    ...     unit="Hz",
    ...     shift=1.0 / rate,
    ... )
    >>> y = SignalArray.from_array(
    ...     position,
    ...     fs=rate,
    ...     time=X.time.copy(),
    ...     channel_names=["x", "y"],
    ...     channel_types="behavior",
    ...     units="m",
    ...     name="cursor",
    ... )
    >>> decoder = LinearDecoder().fit(X, y)
    >>> predicted = decoder.predict(X)
    >>> predicted.channels.names, predicted.unit, predicted.n_samples
    (['x', 'y'], 'm', 200)
    """

    def _build_model(self) -> LinearRegression:
        return LinearRegression(fit_intercept=self.fit_intercept)


class RidgeDecoder(_RegressionDecoder):
    """Decode a continuous target by penalized multi-output least squares.

    :class:`LinearDecoder` with an L2 penalty on the coefficients. The penalty
    keeps every direction the design spans instead of truncating the smallest
    ones, which is what makes it the useful choice when the features are
    collinear -- neighboring lags of a temporal context, or channels sharing a
    common reference.

    Parameters
    ----------
    alpha : float, optional
        Penalty weight on the coefficients. The intercept is estimated by
        centering and is never penalized. ``0.0`` reduces exactly to
        :class:`LinearDecoder`, including its minimum-norm handling of a
        rank-deficient design.
    scaler : neurale.models.preprocessing.StandardScaler or \
neurale.models.preprocessing.MinMaxScaler or None, optional
        Feature scaler to own; see :class:`LinearDecoder`. A penalty is a
        statement about the size of the coefficients, so it means something
        different for features that were never brought onto a common scale.
    context : neurale.models.preprocessing.TemporalContext or None, optional
        Temporal context to stack before regressing; see :class:`LinearDecoder`.
    fit_intercept : bool, optional
        Whether the regression estimates an intercept.

    Raises
    ------
    neurale.exceptions.ValidationError
        If any argument is invalid.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        scaler: FeatureScaler | None = None,
        context: TemporalContext | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = _validate_alpha(alpha)
        super().__init__(scaler=scaler, context=context, fit_intercept=fit_intercept)

    def _validate_configuration(self) -> None:
        super()._validate_configuration()
        _validate_alpha(self.alpha)

    def _build_model(self) -> Ridge:
        return Ridge(self.alpha, fit_intercept=self.fit_intercept)


def _validate_fit_intercept(value: object) -> bool:
    """Validate a configured intercept flag and return it."""

    if not isinstance(value, bool):
        raise ValidationError("fit_intercept must be a bool.")
    return value


def _validate_alpha(value: object) -> float:
    """Validate a configured penalty weight and return it as a float."""

    return float(validate_number(value, "alpha", kind="real", minimum=0, coerce=True))


__all__ = ["LinearDecoder", "RidgeDecoder"]
