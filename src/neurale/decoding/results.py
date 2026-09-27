#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed decoding prediction results."""

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

# Probability rows are compared against one, not required to be exactly one: a
# normalization in float64 leaves a few ULPs of slack, and a decoder that
# renormalized would be hiding the arithmetic rather than reporting it.
_PROBABILITY_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    """The result of decoding classes from feature frames.

    The result is self-describing: it carries the input frame times it was
    predicted from and the class order its columns are in, so a caller never
    has to reconstruct either from the decoder that produced it.

    Parameters
    ----------
    labels : numpy.ndarray
        One-dimensional array of predicted labels, one per input frame. Every
        predicted label must be one of ``classes``.
    classes : sequence or numpy.ndarray
        Class order of the ``scores`` and ``probabilities`` columns.
    time : numpy.ndarray
        Timestamps of the input frames the prediction was made from, one per
        label, finite and non-decreasing.
    scores : numpy.ndarray or None, optional
        ``(n_samples, n_classes)`` decision values on the decoder's own scale.
    probabilities : numpy.ndarray or None, optional
        ``(n_samples, n_classes)`` posterior probabilities. Every entry lies in
        ``[0, 1]`` and every row sums to one.
    clock : neurale.data.time.Clock or None, optional
        Clock the timestamps belong to. A decoder fills this from the frames
        the prediction was made from, so it always describes the ``time`` in
        this same result.
    attrs : mapping, optional
        Application metadata, deep-frozen on construction using the closed
        value domain documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If labels, class order, timestamps, scores, or probabilities are
        invalid or inconsistent.

    Notes
    -----
    ``labels`` is not required to be the column-wise argmax of ``scores``: a
    decoder may abstain or apply a threshold, and forcing agreement here would
    make that impossible to report honestly.
    """

    labels: np.ndarray
    classes: Sequence[object] | np.ndarray
    time: np.ndarray
    scores: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    clock: Clock | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    class_indices: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        labels = validate_labels(self.labels, "labels", ndim=1)
        if labels.shape[0] == 0:
            raise ValidationError("labels must contain at least one sample.")
        classes = validate_class_order(self.classes, "classes")
        time = validate_time_vector(self.time, "time", n_samples=labels.shape[0])
        indices = encode_labels(labels, classes, "labels")

        shape = (labels.shape[0], classes.shape[0])
        scores = _validate_matrix(self.scores, "scores", shape)
        probabilities = _validate_matrix(self.probabilities, "probabilities", shape)
        if probabilities is not None:
            _validate_probabilities(probabilities)

        if self.clock is not None and not isinstance(self.clock, Clock):
            raise ValidationError("clock must be a Clock or None.")

        object.__setattr__(self, "labels", readonly(labels))
        object.__setattr__(self, "classes", readonly(classes))
        object.__setattr__(self, "time", readonly(time))
        object.__setattr__(self, "scores", None if scores is None else readonly(scores))
        object.__setattr__(
            self,
            "probabilities",
            None if probabilities is None else readonly(probabilities),
        )
        object.__setattr__(self, "class_indices", readonly(indices))
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @property
    def n_samples(self) -> int:
        """Number of predicted frames."""

        return int(self.labels.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of declared classes."""

        return int(self.classes.shape[0])


def _validate_matrix(
    values: object,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if values is None:
        return None
    if not isinstance(values, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray or None.")
    if values.ndim != 2:
        raise ValidationError(f"{name} must be a 2D array; got {values.ndim} dimensions.")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValidationError(f"{name} must contain floating-point values.")
    if values.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}; got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValidationError(f"{name} must contain finite values.")
    return values


def _validate_probabilities(values: np.ndarray) -> None:
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValidationError("probabilities must lie in [0, 1].")
    sums = np.sum(values, axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=_PROBABILITY_SUM_TOLERANCE):
        raise ValidationError("probabilities rows must sum to one.")
