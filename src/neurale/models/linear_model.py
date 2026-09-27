#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Linear and ridge regression estimators backed by native model kernels."""

from __future__ import annotations

from typing import Self

import numpy as np

from neurale._validation import validate_integer, validate_number, validate_real_array
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_models_operation, load_restored_models_operation
from ._inference import (
    call_native_with_out,
    make_readonly,
    validate_2d_array,
    validate_inference_input,
)


class _LinearModel:
    """Shared fit and predict plumbing for the native multi-output linear models.

    Targets may be given as ``(n_samples,)`` or ``(n_samples, n_outputs)``, but the
    fitted state is always 2D: ``coef_`` has shape
    ``(n_outputs, n_features)``, ``intercept_`` has shape ``(n_outputs,)``, and
    :meth:`predict` returns ``(n_samples, n_outputs)`` in every case.

    The design is factorized once with a thin SVD that resolves small singular
    values, and the coefficients are read off the projection of the targets onto
    the left singular vectors. Nothing forms a cross product, which would square
    the condition number and cancel away the near-singular directions the
    factorization just resolved. ``rank_`` counts the singular values above
    ``max(n_samples, n_features) * eps * s_max`` and describes the design alone.
    The intercept is estimated by centering and is never penalized.

    Operands are rescaled only as far as the factorization's working range
    requires, so a design or target that is finite but extreme still fits and
    keeps the entries far below its largest one. Where no rescaling can hold
    both ends -- or where the coefficients themselves are unrepresentable -- the
    fit raises :class:`~neurale.exceptions.ValidationError` rather than
    recording a solution with a lost direction or a ``nan``.
    """

    def __init__(self, *, alpha: float, fit_intercept: bool) -> None:
        if not isinstance(fit_intercept, bool):
            raise ValidationError("fit_intercept must be a bool.")
        self.fit_intercept = fit_intercept
        self._alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model on a sample-by-feature design and aligned targets."""

        X = validate_2d_array(X)
        y = self._validate_targets(y, X.shape[0])

        native = load_cpu_models_operation("linear_model")
        try:
            result = native.fit_linear_model(
                X,
                y,
                float(self._alpha),
                self.fit_intercept,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

        self._model = result["model"]
        self._n_samples = int(result["n_samples"])
        self._rank = int(result["rank"])
        self._coef = result["coef"]
        self._intercept = result["intercept"]
        self._singular_values = make_readonly(result["singular_values"])
        self._device = "cpu"
        return self

    def _restore_fitted(
        self,
        *,
        coef: np.ndarray,
        intercept: np.ndarray,
        singular_values: np.ndarray,
        rank: int,
        n_samples: int,
    ) -> Self:
        """Rebuild this model's fitted state from parameters a fit produced.

        Private, and the only supported way to reconstitute a fit: decoder
        persistence stores the coefficients and has to restore *this* predictor
        rather than a second implementation of the affine map, which would
        agree with the native one only to within rounding. The native model is
        rebuilt through its own state constructor, so a restored model and the
        one it was saved from run the same kernel on the same numbers.
        """

        native = load_restored_models_operation("linear_model")
        try:
            model = native.linear_model_from_state(
                validate_real_array(coef, "coef", ndim=2),
                validate_real_array(intercept, "intercept", ndim=1),
            )
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

        self._model = model
        self._n_samples = validate_integer(n_samples, "n_samples", minimum=1)
        self._rank = validate_integer(rank, "rank", minimum=0)
        self._coef = model.coef
        self._intercept = model.intercept
        self._singular_values = make_readonly(
            validate_real_array(singular_values, "singular_values", ndim=1)
        )
        self._device = "cpu"
        return self

    def predict(self, X: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        """Predict ``(n_samples, n_outputs)`` targets for a design matrix."""

        self._check_is_fitted()
        X = validate_inference_input(X, self.n_features_in_)
        return call_native_with_out(self._model.predict, X, out)

    @staticmethod
    def _validate_targets(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = validate_real_array(y, "y")
        if y.ndim not in (1, 2):
            raise ValidationError("y must be a 1D or 2D array.")
        if not np.all(y.shape):
            raise ValidationError("y must not be empty.")
        if y.shape[0] != n_samples:
            raise ValidationError("y must have the same number of samples as X.")
        if y.ndim == 1:
            return y.reshape(n_samples, 1)
        return y

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise ValidationError(f"{type(self).__name__} instance is not fitted.")

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
        return int(self._model.n_features)

    @property
    def n_outputs_(self) -> int:
        """Number of target columns seen during :meth:`fit`."""

        self._check_is_fitted()
        return int(self._model.n_outputs)

    @property
    def rank_(self) -> int:
        """Numerical rank of the design the solver factorized."""

        self._check_is_fitted()
        return self._rank

    @property
    def singular_values_(self) -> np.ndarray:
        """Descending singular values of the design the solver factorized."""

        self._check_is_fitted()
        return self._singular_values

    @property
    def coef_(self) -> np.ndarray:
        """Fitted ``(n_outputs, n_features)`` coefficients."""

        self._check_is_fitted()
        return self._coef

    @property
    def intercept_(self) -> np.ndarray:
        """Fitted ``(n_outputs,)`` intercept; all zero when ``fit_intercept`` is false."""

        self._check_is_fitted()
        return self._intercept


class LinearRegression(_LinearModel):
    """Ordinary least-squares regression with a sklearn-style estimator API.

    Rank-deficient designs return the minimum-norm least-squares solution:
    singular values at or below ``max(n_samples, n_features) * eps * s_max`` are
    dropped, and the solution has no component along the discarded directions.
    :class:`Ridge` deliberately does not truncate; see its documentation.
    """

    def __init__(self, *, fit_intercept: bool = True) -> None:
        super().__init__(alpha=0.0, fit_intercept=fit_intercept)


class Ridge(_LinearModel):
    """L2-penalized least-squares regression with a sklearn-style estimator API.

    ``alpha`` penalizes the coefficients only; the intercept is estimated by
    centering and is never penalized. ``alpha = 0`` reduces exactly to the
    :class:`LinearRegression` solver, including its minimum-norm handling of
    rank-deficient designs.

    A positive ``alpha`` keeps every direction the design spans, including those
    below the rank tolerance that :class:`LinearRegression` drops: the penalty
    already bounds the filter ``1 / (s ** 2 + alpha)``, and a direction with a
    small but nonzero singular value dominates the solution whenever ``alpha`` is
    smaller than ``s ** 2``. ``rank_`` therefore reports the numerical rank of
    the design and not the number of directions this estimator used.
    """

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        penalty = float(
            validate_number(
                alpha,
                "alpha",
                kind="real",
                minimum=0,
                coerce=True,
            )
        )
        super().__init__(alpha=penalty, fit_intercept=fit_intercept)
        self.alpha = penalty


__all__ = ["LinearRegression", "Ridge"]
