# SPDX-FileCopyrightText: 2026 PyNeurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from neurale._validation import validate_real_array
from neurale.exceptions import ValidationError


def make_readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def validate_2d_array(X, *, finite: bool = True) -> np.ndarray:
    """Validate a C-contiguous, aligned, non-empty 2D real inference array."""

    X = validate_real_array(X, "X", ndim=2, finite=finite)
    if not np.all(X.shape):
        raise ValidationError("X must not be empty.")
    return X


def validate_inference_input(X, n_features: int) -> np.ndarray:
    """Validate an inference array against the feature count a model was fit on.

    Inference inputs are not checked for finiteness: a caller is entitled to
    score a matrix holding NaN and read NaN back, and the native kernels
    propagate it. What is never allowed is a different number of features from
    the fit, because the result would be arithmetic on the wrong columns rather
    than an error.
    """

    X = validate_2d_array(X, finite=False)
    if X.shape[1] != n_features:
        raise ValidationError("X must have the same number of features as fit data.")
    return X


def call_native_with_out(
    native_method,
    X: np.ndarray,
    out: np.ndarray | None,
) -> np.ndarray:
    """Invoke a native method with an optional output buffer."""

    try:
        if out is None:
            return native_method(X)
        return native_method(X, out)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError(str(exc)) from exc
