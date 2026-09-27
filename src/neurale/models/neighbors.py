#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Nearest-neighbor models backed by native kernels."""

from __future__ import annotations

import numpy as np

from neurale._validation import validate_choice, validate_integer
from neurale.exceptions import DeviceUnavailableError, ValidationError

from ._dispatch import load_models_operation
from ._inference import validate_2d_array


def knn(
    X: np.ndarray,
    k: int,
    metric: str = "sqeuclidean",
    include_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact k-nearest neighbors for a sample-major matrix.

    Parameters
    ----------
    X:
        Two-dimensional input array with shape ``(n_samples, n_features)``.
    k:
        Number of neighbors to return for each sample.
    metric:
        ``"sqeuclidean"`` returns squared Euclidean distances.
        ``"euclidean"`` returns Euclidean distances.
    include_self:
        If true, guarantees that each row's own index is present in its result
        set. If false, excludes only the same row index. Duplicate rows are
        still valid neighbors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(indices, distances)``, each with shape ``(n_samples, k)``.
        Neighbors are sorted by an extended-range squared-distance key, with
        row index used as the deterministic tie-breaker for equal computed
        keys. For extreme finite inputs, returned distances may saturate to 0
        or infinity when the requested representation is outside the range of
        ``float64``.

    Notes
    -----
    The default ``device="auto"`` runtime policy uses the CPU implementation.
    An explicit CUDA runtime context selects the exact CUDA implementation,
    which currently supports at most 64 neighbors. Both implementations use
    the same deterministic distance and tie-breaking contract.
    """
    X = validate_2d_array(X, finite=False)

    neighbors = validate_integer(k, "k", minimum=1)
    distance_metric = validate_choice(
        metric,
        ("sqeuclidean", "euclidean"),
        "metric",
    )
    if not isinstance(include_self, bool):
        raise ValidationError("include_self must be a bool.")
    max_neighbors = X.shape[0] if include_self else X.shape[0] - 1
    if neighbors > max_neighbors:
        raise ValidationError("k exceeds the available neighbor count.")

    native, device_id, unavailable_error = load_models_operation("neighbors")
    try:
        if device_id is None:
            return native.knn(X, neighbors, distance_metric, include_self)
        return native.knn(
            X,
            neighbors,
            distance_metric,
            include_self,
            device_id,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    except RuntimeError as exc:
        if unavailable_error is not None and isinstance(exc, unavailable_error):
            raise DeviceUnavailableError(str(exc)) from exc
        raise


__all__ = ["knn"]
