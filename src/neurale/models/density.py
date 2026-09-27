#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Density estimators backed by native kernels."""

from __future__ import annotations

from typing import Literal

import numpy as np

from neurale._validation import validate_choice, validate_number
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.runtime import runtime_context

from ._dispatch import load_models_operation
from ._inference import make_readonly

DensityMethod = Literal["pdf", "logpdf"]


class GaussianKDE:
    """Gaussian kernel density estimator with sample-major input.

    ``bw_method`` may be ``"scott"``, ``"silverman"``, or a positive scalar
    covariance factor.  The kernel covariance is ``factor**2 * Cov(samples)``;
    scalar values are not absolute kernel standard deviations.

    The estimator uses an unregularized full covariance matrix, so fitting
    requires ``n_samples > n_features`` and data that are numerically positive
    definite in double precision.  Strongly correlated features or features
    with very different scales should be standardized before fitting.

    ``pdf()`` returns density values, while ``logpdf()`` and
    ``score_samples()`` return log-density values.  Query arrays and ``out``
    arrays must be aligned, C-contiguous ``float64`` arrays for the no-copy
    native inference path.  ``out`` buffers must also be writable.

    The runtime device active during ``fit()`` fixes the fitted model's
    execution device, exposed by the read-only ``device_`` attribute. CUDA
    models retain fitted samples and reusable query workspaces on the selected
    device and must not be evaluated concurrently. Copying or serializing a
    fitted estimator records the actual device and fit samples; restoration
    reconstructs an independent native model on that device and requires CUDA
    to remain available for a CUDA model.
    """

    def __init__(self, bw_method: str | float = "scott") -> None:
        self.bw_method = bw_method

    def fit(self, X: np.ndarray) -> GaussianKDE:
        bw_method = self.bw_method
        kind, scalar = _normalize_bw_method(bw_method)
        native, device_id, unavailable_error = load_models_operation("density")
        try:
            if device_id is None:
                self._model = native.fit_gaussian_kde(X, kind, scalar)
            else:
                self._model = native.fit_gaussian_kde(
                    X,
                    kind,
                    scalar,
                    device_id,
                )
            self._cuda_unavailable_error = unavailable_error
            self.bw_method_ = kind if kind != "scalar" else scalar
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        except RuntimeError as exc:
            if unavailable_error is not None and isinstance(
                exc,
                unavailable_error,
            ):
                raise DeviceUnavailableError(str(exc)) from exc
            if unavailable_error is None:
                raise ValidationError(str(exc)) from exc
            raise
        self._device = "cpu" if device_id is None else "cuda"
        self._cuda_device = None if device_id in (None, -1) else int(device_id)
        self._fit_samples = make_readonly(
            np.array(X, copy=True, order="C"),
        )
        return self

    def pdf(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._evaluate("pdf", X, out)

    def logpdf(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._evaluate("logpdf", X, out)

    def score_samples(
        self,
        X: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.logpdf(X, out=out)

    def _evaluate(
        self,
        method: DensityMethod,
        X: np.ndarray,
        out: np.ndarray | None,
    ) -> np.ndarray:
        self._check_is_fitted()
        native_method = getattr(self._model, method)
        try:
            if out is None:
                return native_method(X)
            return native_method(X, out)
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        except RuntimeError as exc:
            unavailable_error = self._cuda_unavailable_error
            if unavailable_error is not None and isinstance(
                exc,
                unavailable_error,
            ):
                raise DeviceUnavailableError(str(exc)) from exc
            if unavailable_error is None:
                raise ValidationError(str(exc)) from exc
            raise

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise ValidationError("GaussianKDE instance is not fitted.")

    @property
    def n_samples_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_samples)

    @property
    def n_features_in_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_features)

    @property
    def bandwidth_factor_(self) -> float:
        self._check_is_fitted()
        return float(self._model.bandwidth_factor)

    @property
    def device_(self) -> str:
        """Actual device that owns and executes the fitted model."""

        self._check_is_fitted()
        return self._device

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = {
            "version": 1,
            "bw_method": self.bw_method,
            "fitted": hasattr(self, "_model"),
        }
        if hasattr(self, "_model"):
            state.update(
                {
                    "bw_method_fitted": self.bw_method_,
                    "device": self._device,
                    "cuda_device": self._cuda_device,
                    "samples": np.array(self._fit_samples, copy=True, order="C"),
                }
            )
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported GaussianKDE serialized state version.")
        current_bw_method = state["bw_method"]
        fitted_bw_method = state.get("bw_method_fitted", current_bw_method)
        self.__init__(fitted_bw_method)
        if not state["fitted"]:
            return

        device = state["device"]
        if device not in ("cpu", "cuda"):
            raise ValueError("invalid GaussianKDE serialized device.")
        overrides: dict[str, object] = {"device": device}
        if device == "cuda":
            overrides["cuda_device"] = state["cuda_device"]
        with runtime_context(**overrides):
            self.fit(np.array(state["samples"], copy=True, order="C"))
        self.bw_method = current_bw_method

    def __copy__(self) -> GaussianKDE:
        restored = type(self).__new__(type(self))
        restored.__setstate__(self.__getstate__())
        return restored

    def __deepcopy__(self, memo: dict[int, object]) -> GaussianKDE:
        restored = self.__copy__()
        memo[id(self)] = restored
        return restored


def _normalize_bw_method(value: str | float) -> tuple[str, float]:
    if isinstance(value, str):
        return validate_choice(
            value,
            ("scott", "silverman"),
            "bw_method",
        ), 0.0
    factor = validate_number(
        value,
        "bw_method",
        kind="real",
        minimum=0.0,
        minimum_inclusive=False,
        coerce=True,
    )
    return "scalar", float(factor)


__all__ = ["GaussianKDE"]
