#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Validation helpers shared by the decoding contract."""

from __future__ import annotations

import numpy as np

from neurale._validation import validate_number
from neurale.exceptions import ValidationError


def readonly(values: np.ndarray) -> np.ndarray:
    """Return a private, non-writable copy of an array."""

    copy = np.array(values, copy=True)
    copy.setflags(write=False)
    return copy


def optional_positive_float(value: object, name: str) -> float | None:
    """Validate an optional positive finite scalar."""

    if value is None:
        return None
    return float(
        validate_number(
            value,
            name,
            kind="real",
            minimum=0.0,
            minimum_inclusive=False,
        )
    )


def optional_text(value: object, name: str) -> str | None:
    """Validate an optional non-empty string."""

    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} must be a non-empty string or None.")
    return value


def validate_time_vector(time: object, name: str, *, n_samples: int) -> np.ndarray:
    """Validate a 1D, finite, non-decreasing timestamp vector."""

    if not isinstance(time, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray.")
    if time.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array; got {time.ndim} dimensions.")
    if not np.issubdtype(time.dtype, np.number) or np.issubdtype(time.dtype, np.complexfloating):
        raise ValidationError(f"{name} must contain real-valued timestamps.")
    values = np.asarray(time, dtype=np.float64)
    if values.shape[0] != n_samples:
        raise ValidationError(f"{name} length must be {n_samples}; got {values.shape[0]}.")
    if not np.all(np.isfinite(values)):
        raise ValidationError(f"{name} must contain finite timestamps.")
    if values.size > 1 and np.any(np.diff(values) < 0.0):
        raise ValidationError(f"{name} must be non-decreasing.")
    return values


def validate_class_order(classes: object, name: str) -> np.ndarray:
    """Validate an explicit class order: one dimension, non-empty, unique."""

    values = np.asarray(classes)
    if values.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array; got {values.ndim} dimensions.")
    if values.shape[0] == 0:
        raise ValidationError(f"{name} must contain at least one class.")
    if np.issubdtype(values.dtype, np.complexfloating):
        raise ValidationError(f"{name} must contain real-valued classes.")
    if np.issubdtype(values.dtype, np.inexact) and not np.all(np.isfinite(values)):
        raise ValidationError(f"{name} must contain only finite classes.")
    if len({_hashable(entry) for entry in values.tolist()}) != values.shape[0]:
        raise ValidationError(f"{name} must be unique.")
    return values


def encode_labels(labels: np.ndarray, classes: np.ndarray, name: str) -> np.ndarray:
    """Map every label onto its position in ``classes``.

    Labels are treated as opaque hashable scalars, so the mapping is by value
    and never by an ordering assumption. A label the class order does not
    declare is an error rather than a silently dropped sample.
    """

    positions = {_hashable(entry): idx for idx, entry in enumerate(classes.tolist())}
    try:
        encoded = [positions[_hashable(entry)] for entry in labels.tolist()]
    except KeyError as exc:
        raise ValidationError(
            f"{name} contains a label the class order does not declare: {exc.args[0]!r}."
        ) from exc
    return np.asarray(encoded, dtype=np.intp)


def _hashable(value: object) -> object:
    # NumPy's tolist() already yields hashable Python scalars for every dtype
    # the label validators admit; a list here means a nested object array.
    if isinstance(value, list):
        raise ValidationError("labels and classes must be scalars, not nested sequences.")
    return value
