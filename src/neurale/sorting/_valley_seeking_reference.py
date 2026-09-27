#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Deterministic small-input reference for Valley Seeking clustering.

This module is deliberately private. It fixes the numerical and termination
contract used to validate the native implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real

import numpy as np

from neurale.data._helpers import immutable_array_copy
from neurale.exceptions import ValidationError

if hasattr(math, "fma"):
    _fused_multiply_add = math.fma
else:

    def _fused_multiply_add(left: float, right: float, addend: float) -> float:
        # Python < 3.13 has no ``math.fma``. Fraction preserves the exact
        # binary64 operands through the multiply and add; converting the exact
        # rational result back to float performs the one final binary64 round.
        return float(
            Fraction.from_float(left) * Fraction.from_float(right) + Fraction.from_float(addend)
        )


@dataclass(frozen=True, slots=True)
class ValleySeekingReferenceResult:
    """Immutable result returned by :func:`valley_seeking_reference`."""

    labels: np.ndarray
    neighbor_counts: np.ndarray
    label_order: tuple[int, ...]
    radius: float
    n_iterations: int
    converged: bool
    termination: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", immutable_array_copy(self.labels))
        object.__setattr__(
            self,
            "neighbor_counts",
            immutable_array_copy(self.neighbor_counts),
        )


def valley_seeking_reference(
    features: np.ndarray,
    initial_labels: np.ndarray,
    *,
    radius: float,
    max_iterations: int = 1000,
) -> ValleySeekingReferenceResult:
    """Refine an initial classification using fixed-radius neighbor votes.

    The input is exact ``float64`` data in ``(observation, feature)`` order.
    Neighborhoods are symmetric, exclude the observation itself, and include
    pairs whose Euclidean distance is less than or equal to ``radius``.
    Updates are synchronous. A tied observation retains its current label when
    that label is among the winners; otherwise the numerically smallest label
    wins.

    The routine is a deterministic oracle for small inputs, not an optimized
    clustering entry point. It stops when one complete update changes no label
    or after ``max_iterations`` updates; the latter result has
    ``converged=False`` because Valley Seeking is not guaranteed to converge.
    """
    points = _validate_features(features)
    labels = _validate_labels(initial_labels, points.shape[0])
    resolved_radius = _validate_radius(radius)
    iteration_limit = _validate_max_iterations(max_iterations)

    if points.shape[0] == 0:
        return ValleySeekingReferenceResult(
            labels=labels,
            neighbor_counts=np.empty(0, dtype=np.int64),
            label_order=(),
            radius=resolved_radius,
            n_iterations=0,
            converged=True,
            termination="empty",
        )

    label_order_array = np.unique(labels)
    label_order = tuple(int(label) for label in label_order_array)
    label_positions = {label: idx for idx, label in enumerate(label_order)}
    neighborhoods = _fixed_radius_neighborhoods(points, resolved_radius)
    neighbor_counts = np.fromiter(
        (len(neighbors) for neighbors in neighborhoods),
        dtype=np.int64,
        count=points.shape[0],
    )

    current = labels.copy()
    following = np.empty_like(current)
    votes = np.zeros(len(label_order), dtype=np.int64)
    for iteration in range(1, iteration_limit + 1):
        for observation, neighbors in enumerate(neighborhoods):
            votes.fill(0)
            for neighbor in neighbors:
                votes[label_positions[int(current[neighbor])]] += 1

            incumbent = label_positions[int(current[observation])]
            most_votes = int(votes.max(initial=0))
            if votes[incumbent] == most_votes:
                following[observation] = current[observation]
            else:
                # ``label_order`` is ascending, so the first winner is the
                # deterministic resolution of a tie not involving incumbent.
                winner = int(np.flatnonzero(votes == most_votes)[0])
                following[observation] = label_order_array[winner]

        if np.array_equal(following, current):
            return ValleySeekingReferenceResult(
                labels=following,
                neighbor_counts=neighbor_counts,
                label_order=label_order,
                radius=resolved_radius,
                n_iterations=iteration,
                converged=True,
                termination="converged",
            )
        current, following = following, current

    return ValleySeekingReferenceResult(
        labels=current,
        neighbor_counts=neighbor_counts,
        label_order=label_order,
        radius=resolved_radius,
        n_iterations=iteration_limit,
        converged=False,
        termination="max_iterations",
    )


def _validate_features(features: np.ndarray) -> np.ndarray:
    points = np.asarray(features)
    if points.ndim != 2:
        raise ValidationError("features must have shape (n_observations, n_features).")
    if points.shape[1] == 0:
        raise ValidationError("features must contain at least one feature column.")
    if points.dtype != np.dtype(np.float64):
        raise ValidationError(
            "features must have dtype float64; no implicit conversion is performed."
        )
    if not np.all(np.isfinite(points)):
        raise ValidationError("features must contain only finite values.")
    return points


def _validate_labels(initial_labels: np.ndarray, n_observations: int) -> np.ndarray:
    labels = np.asarray(initial_labels)
    if labels.ndim != 1 or labels.shape[0] != n_observations:
        raise ValidationError("initial_labels must be 1D with one label per observation.")
    if labels.dtype != np.dtype(np.int64):
        raise ValidationError(
            "initial_labels must have dtype int64; no implicit conversion is performed."
        )
    return labels


def _validate_radius(radius: float) -> float:
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real):
        raise ValidationError("radius must be a finite positive real number.")
    resolved = float(radius)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValidationError("radius must be a finite positive real number.")
    return resolved


def _validate_max_iterations(max_iterations: int) -> int:
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral):
        raise ValidationError("max_iterations must be a positive integer.")
    resolved = int(max_iterations)
    if resolved <= 0:
        raise ValidationError("max_iterations must be a positive integer.")
    return resolved


def _fixed_radius_neighborhoods(
    points: np.ndarray,
    radius: float,
) -> list[list[int]]:
    neighborhoods = [[] for _ in range(points.shape[0])]
    for left in range(points.shape[0]):
        for right in range(left + 1, points.shape[0]):
            with np.errstate(over="ignore", invalid="ignore"):
                delta = np.abs(points[left] - points[right])
            scale = float(delta.max(initial=0.0))
            if not np.isfinite(scale) or scale > radius:
                continue
            if scale == 0.0:
                within_radius = True
            else:
                squared_scaled_distance = 0.0
                for difference in delta:
                    normalized = float(difference) / radius
                    squared_scaled_distance = _fused_multiply_add(
                        normalized,
                        normalized,
                        squared_scaled_distance,
                    )
                    if squared_scaled_distance > 1.0:
                        break
                within_radius = squared_scaled_distance <= 1.0
            if within_radius:
                neighborhoods[left].append(right)
                neighborhoods[right].append(left)
    return neighborhoods
