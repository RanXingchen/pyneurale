#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Private numerical helpers for filter representation conversions."""

from __future__ import annotations

import numpy as np

from neurale.exceptions import ValidationError


def finite_vector(value, name: str, *, allow_empty: bool = True) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1D.")
    if not allow_empty and arr.size == 0:
        raise ValidationError(f"{name} must not be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValidationError(f"{name} must contain numeric values.")
    result = np.asarray(arr, dtype=np.result_type(arr.dtype, np.float64))
    if np.any(~np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values.")
    return result


def sort_conjugate_pairs(
    x,
    name: str,
    *,
    tol: float = 100.0 * np.finfo(float).eps,
) -> np.ndarray:
    """Return real roots followed by deterministic complex-conjugate pairs."""
    roots = finite_vector(x, name)
    real_roots: list[complex] = []
    positive: list[complex] = []
    negative: list[complex] = []
    for value in np.asarray(roots, dtype=complex):
        scale = max(1.0, abs(value))
        if abs(value.imag) <= tol * scale:
            real_roots.append(complex(value.real, 0.0))
        elif value.imag > 0:
            positive.append(value)
        else:
            negative.append(value)

    positive.sort(key=lambda value: (value.real, abs(value.imag)))
    negative.sort(key=lambda value: (value.real, abs(value.imag)))
    pairs: list[complex] = []
    unused = negative.copy()
    for value in positive:
        if not unused:
            raise ValidationError(f"{name} must contain conjugate pairs.")
        distances = np.abs(np.asarray(unused) - np.conjugate(value))
        idx = int(np.argmin(distances))
        match = unused[idx]
        if distances[idx] > tol * max(1.0, abs(value), abs(match)):
            raise ValidationError(f"{name} must contain conjugate pairs.")
        unused.pop(idx)
        canonical = complex(
            0.5 * (value.real + match.real),
            0.5 * (abs(value.imag) + abs(match.imag)),
        )
        pairs.extend((np.conjugate(canonical), canonical))
    if unused:
        raise ValidationError(f"{name} must contain conjugate pairs.")
    real_roots.sort(key=lambda value: value.real)
    return np.asarray(real_roots + pairs, dtype=complex)


def polynomial_from_roots(roots, name: str = "roots") -> np.ndarray:
    ordered = sort_conjugate_pairs(roots, name)
    coefs = np.atleast_1d(np.poly(ordered))
    return _real_if_close(coefs, f"{name} polynomial")


def polynomial_roots(coefs, name: str = "coefficients") -> np.ndarray:
    values = finite_vector(coefs, name, allow_empty=False)
    nonzero = np.flatnonzero(values)
    if nonzero.size == 0:
        raise ValidationError(f"{name} must contain a nonzero coefficient.")
    normalized = values[nonzero[0] :]
    if normalized.size == 1:
        return np.empty(0, dtype=complex)
    return np.asarray(np.roots(normalized), dtype=complex)


def _real_if_close(value: np.ndarray, name: str) -> np.ndarray:
    result = np.real_if_close(np.asarray(value), tol=1000)
    if np.iscomplexobj(result):
        raise ValidationError(f"{name} is complex; roots must define a real-coefficient system.")
    return np.asarray(result, dtype=float)
