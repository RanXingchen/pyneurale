#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed decoding targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurale._validation import validate_labels
from neurale.data import Clock
from neurale.data._helpers import freeze_metadata
from neurale.exceptions import ValidationError

from ._validation import encode_labels, readonly, validate_class_order, validate_time_vector


@dataclass(frozen=True, slots=True)
class ClassificationTarget:
    """A time-stamped classification target aligned to feature frames.

    One label per feature frame, with the timestamp of that frame. The target
    is already aligned when it reaches a decoder: it declares the times it
    belongs to so that a decoder can *check* the alignment, not repair it.

    Parameters
    ----------
    labels : numpy.ndarray
        One-dimensional array of labels, one per frame. Labels are opaque
        hashable scalars and are compared by value only.
    time : numpy.ndarray
        One-dimensional array of frame timestamps in seconds, one per label,
        finite and non-decreasing.
    classes : sequence or numpy.ndarray or None, optional
        Explicit class order. This is the column order of every score and
        probability a decoder fitted on this target produces, so it is part of
        the contract rather than an implementation detail. When omitted the
        order is ``numpy.unique(labels)``, which is deterministic and
        independent of row order. An explicit order may declare classes that
        the labels do not contain; it may not omit one that they do.
    clock : neurale.data.time.Clock or None, optional
        Clock the timestamps belong to. Supplying it lets a decoder reject a
        target measured on a different clock; a decoder never converts between
        clocks itself.
    attrs : mapping, optional
        Application metadata, deep-frozen on construction using the closed
        value domain documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If labels, timestamps, or the class order are invalid or inconsistent.

    Notes
    -----
    ``labels``, ``time``, ``classes``, and ``class_indices`` are private
    non-writable copies, so a target cannot change under a fitted decoder.
    """

    labels: np.ndarray
    time: np.ndarray
    classes: Sequence[object] | np.ndarray | None = None
    clock: Clock | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    class_indices: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        labels = validate_labels(self.labels, "labels", ndim=1)
        if labels.shape[0] == 0:
            raise ValidationError("labels must contain at least one sample.")
        time = validate_time_vector(self.time, "time", n_samples=labels.shape[0])

        if self.classes is None:
            classes = np.unique(labels)
        else:
            classes = validate_class_order(self.classes, "classes")
            if classes.dtype != labels.dtype and not _comparable(classes, labels):
                raise ValidationError("classes must have the same kind of values as labels.")
        indices = encode_labels(labels, classes, "labels")

        if self.clock is not None and not isinstance(self.clock, Clock):
            raise ValidationError("clock must be a Clock or None.")

        object.__setattr__(self, "labels", readonly(labels))
        object.__setattr__(self, "time", readonly(time))
        object.__setattr__(self, "classes", readonly(classes))
        object.__setattr__(self, "class_indices", readonly(indices))
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @property
    def n_samples(self) -> int:
        """Number of labelled frames."""

        return int(self.labels.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of declared classes."""

        return int(self.classes.shape[0])


def _comparable(classes: np.ndarray, labels: np.ndarray) -> bool:
    # Distinct dtypes are fine as long as the values still compare by value:
    # int64 classes for int32 labels, or object classes for strings. Two
    # unrelated kinds -- numbers against strings -- are a caller error.
    kinds = {_kind(classes.dtype), _kind(labels.dtype)}
    return len(kinds) == 1 or "object" in kinds


def _kind(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_):
        return "number"
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
        return "text"
    return "object"
