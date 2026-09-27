#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Decomposition estimators backed by native model kernels."""

from __future__ import annotations

import numpy as np

from neurale._validation import validate_integer
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_models_operation
from ._inference import call_native_with_out, make_readonly, validate_2d_array


class PCA:
    """Principal component analysis with a sklearn-style estimator API."""

    def __init__(self, n_components: int, *, center: bool = True) -> None:
        self.n_components = validate_integer(
            n_components,
            "n_components",
            minimum=1,
        )
        if not isinstance(center, bool):
            raise ValidationError("center must be a bool.")
        self.center = center

    def fit(self, X: np.ndarray) -> PCA:
        return self._fit_validated(validate_2d_array(X))

    def _fit_validated(self, X: np.ndarray) -> PCA:
        if self.n_components > X.shape[1]:
            raise ValidationError("n_components must be at most n_features.")
        native = load_cpu_models_operation("decomposition")
        try:
            result = native.fit_pca(X, self.n_components, self.center)
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc

        self._n_samples = int(result["n_samples"])
        self._model = result["model"]
        self._mean = result["mean"]
        self._components = result["components"]
        self._offset = result["offset"]
        self._singular_values = make_readonly(result["singular_values"])
        self._explained_variance = make_readonly(result["explained_variance"])
        self._explained_variance_ratio = make_readonly(result["explained_variance_ratio"])
        self._device = "cpu"
        return self

    def transform(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        X = validate_2d_array(X, finite=False)
        return self._transform_validated(X, out)

    def _transform_validated(
        self,
        X: np.ndarray,
        out: np.ndarray | None,
    ) -> np.ndarray:
        if X.shape[1] != self.n_features_in_:
            raise ValidationError("X must have the same number of features as fit data.")
        return call_native_with_out(self._model.transform, X, out)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = validate_2d_array(X)
        self._fit_validated(X)
        return self._transform_validated(X, None)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise ValidationError("PCA instance is not fitted.")

    @property
    def device_(self) -> str:
        self._check_is_fitted()
        return self._device

    @property
    def n_samples_(self) -> int:
        self._check_is_fitted()
        return self._n_samples

    @property
    def n_features_in_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_features)

    @property
    def n_components_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_components)

    @property
    def mean_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._mean

    @property
    def components_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._components

    @property
    def offset_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._offset

    @property
    def singular_values_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._singular_values

    @property
    def explained_variance_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._explained_variance

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._explained_variance_ratio


__all__ = ["PCA"]
