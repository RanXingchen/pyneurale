#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Classification estimators backed by native model kernels."""

from __future__ import annotations

import numpy as np

from neurale._validation import (
    validate_choice,
    validate_labels,
    validate_number,
    validate_real_array,
)
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_models_operation, load_restored_models_operation
from ._inference import (
    call_native_with_out,
    make_readonly,
    validate_2d_array,
    validate_inference_input,
)


class LDA:
    """Linear discriminant analysis with the native lsqr solver."""

    def __init__(
        self,
        *,
        solver: str = "lsqr",
        shrinkage: str | float | None = None,
    ) -> None:
        self.solver = validate_choice(solver, ("lsqr",), "solver")
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDA:
        X = validate_2d_array(X)

        labels = validate_labels(y, "y", ndim=1, n_samples=X.shape[0])

        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValidationError("y must contain at least two classes.")
        if X.shape[0] <= classes.size:
            raise ValidationError("X must contain more samples than classes.")
        shrinkage_kind, shrinkage_value = _normalize_shrinkage(self.shrinkage)

        native = load_cpu_models_operation("classification")
        try:
            result = native.fit_lda(
                X,
                np.asarray(encoded, dtype=np.uintp, order="C"),
                int(classes.size),
                shrinkage_kind,
                shrinkage_value,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

        self._classes = make_readonly(classes)
        self._model = result["model"]
        self._priors = make_readonly(result["priors"])
        self._means = make_readonly(result["means"])
        self._covariance = make_readonly(result["covariance"])
        self._coef = result["coef"]
        self._intercept = result["intercept"]
        self._device = "cpu"
        return self

    def _restore_fitted(
        self,
        *,
        classes: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        priors: np.ndarray,
        means: np.ndarray,
        covariance: np.ndarray,
    ) -> LDA:
        """Rebuild this discriminant's fitted state from parameters a fit produced.

        Private, and the only supported way to reconstitute a fit; see
        :meth:`neurale.models.linear_model._LinearModel._restore_fitted`.
        ``classes`` is the model's own sorted order, as :meth:`fit` recorded it.
        """

        # No n_samples here: the only length a class order answers to is its
        # own, and reading classes.shape before the type check would turn a
        # non-array into an AttributeError instead of a ValidationError.
        order = validate_labels(classes, "classes", ndim=1)
        n_classes = int(order.shape[0])
        if n_classes < 2:
            raise ValidationError("classes must contain at least two classes.")

        native = load_restored_models_operation("classification")
        try:
            model = native.lda_model_from_state(
                validate_real_array(coef, "coef", ndim=2),
                validate_real_array(intercept, "intercept", ndim=1),
                n_classes,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

        n_features = int(model.n_features)
        _require_shape(priors, (n_classes,), "priors")
        _require_shape(means, (n_classes, n_features), "means")
        _require_shape(covariance, (n_features, n_features), "covariance")

        self._classes = make_readonly(order)
        self._model = model
        self._priors = make_readonly(np.asarray(priors, dtype=np.float64))
        self._means = make_readonly(np.asarray(means, dtype=np.float64))
        self._covariance = make_readonly(np.asarray(covariance, dtype=np.float64))
        self._coef = model.coef
        self._intercept = model.intercept
        self._device = "cpu"
        return self

    def decision_function(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        X = validate_inference_input(X, self.n_features_in_)
        return call_native_with_out(self._model.decision_function, X, out)

    def predict_proba(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        X = validate_inference_input(X, self.n_features_in_)
        return call_native_with_out(self._model.predict_proba, X, out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self.n_outputs_ == 1:
            indices = (scores > 0.0).astype(np.intp, copy=False)
        else:
            indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise ValidationError("LDA instance is not fitted.")

    @property
    def device_(self) -> str:
        self._check_is_fitted()
        return self._device

    @property
    def n_features_in_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_features)

    @property
    def n_outputs_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_outputs)

    @property
    def classes_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._classes

    @property
    def priors_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._priors

    @property
    def means_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._means

    @property
    def covariance_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._covariance

    @property
    def coef_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._coef

    @property
    def intercept_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._intercept


def _require_shape(values: object, shape: tuple[int, ...], name: str) -> None:
    # A restored statistic that disagrees with the model it accompanies would
    # not fail until something read it, so the disagreement is caught here.
    arr = validate_real_array(values, name, ndim=len(shape))
    if arr.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}; got {arr.shape}.")


def _normalize_shrinkage(value: str | float | None) -> tuple[str, float]:
    if value is None:
        return "none", 0.0
    if value == "auto":
        return "auto", 0.0
    if isinstance(value, str):
        raise ValidationError("shrinkage must be None, 'auto', or a float in [0, 1].")
    shrinkage = float(
        validate_number(
            value,
            "shrinkage",
            kind="real",
            minimum=0,
            maximum=1,
            coerce=True,
        )
    )
    return "fixed", shrinkage


__all__ = ["LDA"]
