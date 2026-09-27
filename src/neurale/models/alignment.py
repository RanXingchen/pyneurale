#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Sequence alignment models backed by native kernels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from neurale._validation import validate_choice, validate_real_array
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_models_operation
from ._inference import make_readonly


def _normalize_radius(r: int | None, max_steps: int) -> int | None:
    if r is None:
        return None

    if isinstance(r, bool) or not isinstance(r, Integral):
        raise ValidationError("radius must be None or a non-negative integer.")

    radius = min(int(r), max_steps - 1)
    if radius < 0:
        raise ValidationError("radius must be None or a non-negative integer.")
    return radius


def _as_sequence(value: np.ndarray, name: str) -> np.ndarray:
    values = validate_real_array(value, name)

    if not np.all(values.shape):
        raise ValidationError(f"{name} must not be empty.")
    if values.ndim not in [1, 2]:
        raise ValidationError(f"{name} must be a 1D or 2D array.")

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    return values


@dataclass(frozen=True)
class DTWResult:
    """Dynamic time warping result."""

    cost: float
    path_x: np.ndarray
    path_y: np.ndarray

    def __iter__(self):
        yield self.cost
        yield self.path_x
        yield self.path_y


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    radius: int | None = None,
    metric: str = "euclidean",
) -> DTWResult:
    """Compute dynamic time warping alignment between two sequences.

    One-dimensional inputs are treated as ``(n_steps, 1)``. Two-dimensional
    inputs must use sample-major layout ``(n_steps, n_features)``. The
    ``"euclidean"`` metric accumulates Euclidean local distances along the
    path; ``"sqeuclidean"`` accumulates squared Euclidean local distances.

    ``radius`` limits the path to a scaled band around the line connecting the
    two endpoints. The band is measured on the longer sequence axis, so it is
    not the same as a plain ``abs(i - j) <= radius`` window when sequence
    lengths differ. Too small a radius can make a complete path impossible.

    The returned cost is the raw accumulated path cost and is not normalized by
    path length. ``path_x`` and ``path_y`` include both start and end indices.
    """
    x_values = _as_sequence(x, "x")
    y_values = _as_sequence(y, "y")
    if x_values.shape[1] != y_values.shape[1]:
        raise ValidationError("x and y must have the same number of dimensions.")
    window_radius = _normalize_radius(
        radius,
        max(x_values.shape[0], y_values.shape[0]),
    )
    distance_metric = validate_choice(
        metric,
        ("euclidean", "sqeuclidean"),
        "metric",
    )

    native = load_cpu_models_operation("alignment")
    try:
        result = native.dtw(
            x_values,
            y_values,
            window_radius,
            distance_metric,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return DTWResult(
        cost=float(result["cost"]),
        path_x=make_readonly(result["path_x"]),
        path_y=make_readonly(result["path_y"]),
    )


__all__ = ["DTWResult", "dtw"]
