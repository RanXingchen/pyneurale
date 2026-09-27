#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Native-backed locality-preserving projection estimators."""

from __future__ import annotations

from typing import Literal

import numpy as np

from neurale._validation import validate_integer, validate_real_array
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.runtime import runtime_context

from ._dispatch import load_models_operation
from ._inference import (
    call_native_with_out,
    make_readonly,
    validate_inference_input,
)


class LPP:
    """Locality-preserving linear projection with an sklearn-style API.

    ``proj_method="LPP"`` implements the locality-preserving projection
    objective. ``proj_method="OPP"`` selects the regularized orthogonal
    projection objective. CUDA may accelerate neighborhood search
    during fitting, but the fitted projection and ``transform()`` are CPU
    state, exposed by ``device_ == "cpu"``. Copying or serialization stores
    the fit input and reconstructs an independent CPU projection.
    """

    def __init__(
        self,
        n_components: int,
        *,
        n_neighbors: int = 5,
        metric: Literal["sqeuclidean", "euclidean"] = "sqeuclidean",
        proj_method: Literal["OPP", "LPP"] = "OPP",
    ) -> None:
        self.n_components = validate_integer(
            n_components,
            "n_components",
            minimum=1,
        )
        self.n_neighbors = validate_integer(
            n_neighbors,
            "n_neighbors",
            minimum=1,
        )
        if metric not in ("sqeuclidean", "euclidean"):
            raise ValidationError("metric must be 'sqeuclidean' or 'euclidean'.")
        if proj_method not in ("OPP", "LPP"):
            raise ValidationError("proj_method must be 'OPP' or 'LPP'.")
        self.metric = metric
        self.proj_method = proj_method

    def fit(self, X: np.ndarray) -> LPP:
        X = self._validate_fit_input(X)
        return self._fit_validated(X)

    def _fit_validated(self, X: np.ndarray) -> LPP:
        native, device_id, unavailable_error = load_models_operation("manifold")
        try:
            arguments = (
                X,
                self.n_components,
                self.n_neighbors,
                self.metric,
                self.proj_method,
            )
            if device_id is None:
                result = native.fit_lpp(*arguments)
            else:
                result = native.fit_lpp(*arguments, device_id)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        except RuntimeError as exc:
            if unavailable_error is not None and isinstance(exc, unavailable_error):
                raise DeviceUnavailableError(str(exc)) from exc
            raise ValidationError(str(exc)) from exc

        self._n_samples = int(result["n_samples"])
        self._model = result["model"]
        self._components = result["components"]
        self._device = "cpu"
        self._fit_values = make_readonly(np.array(X, copy=True, order="C"))
        return self

    def transform(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        X = validate_inference_input(X, self.n_features_in_)
        return self._transform_validated(X, out)

    def _transform_validated(
        self,
        X: np.ndarray,
        out: np.ndarray | None,
    ) -> np.ndarray:
        return call_native_with_out(self._model.transform, X, out)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_fit_input(X)
        self._fit_validated(X)
        return self._transform_validated(X, None)

    def _validate_fit_input(self, X: np.ndarray) -> np.ndarray:
        X = validate_real_array(X, "X", ndim=2)
        if X.shape[0] < 2 or X.shape[1] == 0:
            raise ValidationError("X must contain at least two non-empty samples.")
        if self.n_components > X.shape[1]:
            raise ValidationError("n_components must be at most n_features.")
        if self.n_neighbors > X.shape[0]:
            raise ValidationError("n_neighbors must be at most n_samples.")
        return X

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise ValidationError("LPP instance is not fitted.")

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
    def components_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._components

    @property
    def device_(self) -> str:
        """Actual device used by the fitted projection model."""

        self._check_is_fitted()
        return self._device

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = {
            "version": 1,
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "proj_method": self.proj_method,
            "fitted": hasattr(self, "_model"),
        }
        if hasattr(self, "_model"):
            state.update(
                {
                    "device": self._device,
                    "values": np.array(self._fit_values, copy=True, order="C"),
                }
            )
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported LPP serialized state version.")
        self.__init__(
            state["n_components"],
            n_neighbors=state["n_neighbors"],
            metric=state["metric"],
            proj_method=state["proj_method"],
        )
        if not state["fitted"]:
            return
        if state["device"] != "cpu":
            raise ValueError("invalid LPP serialized device.")
        with runtime_context(device="cpu"):
            self.fit(np.array(state["values"], copy=True, order="C"))

    def __copy__(self) -> LPP:
        restored = type(self).__new__(type(self))
        restored.__setstate__(self.__getstate__())
        return restored

    def __deepcopy__(self, memo: dict[int, object]) -> LPP:
        restored = self.__copy__()
        memo[id(self)] = restored
        return restored


__all__ = ["LPP"]
