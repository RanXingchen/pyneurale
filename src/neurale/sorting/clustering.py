#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed public clustering operations for offline spike features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

from neurale.data import FeatureMatrix
from neurale.data._helpers import immutable_array_copy
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_sorting_operation

ValleySeekingTermination = Literal["empty", "converged", "max_iterations"]
_TERMINATIONS = frozenset({"empty", "converged", "max_iterations"})


@dataclass(frozen=True, slots=True, eq=False)
class ValleySeekingResult:
    """Immutable diagnostics and event-aligned labels from Valley Seeking."""

    labels: np.ndarray
    neighbor_counts: np.ndarray
    label_order: tuple[int, ...]
    radius: float
    n_iterations: int
    converged: bool
    termination: ValleySeekingTermination
    neighbor_pairs: int
    workspace_bytes: int

    def __post_init__(self) -> None:
        labels = _int64_vector(self.labels, "labels")
        neighbor_counts = _int64_vector(
            self.neighbor_counts,
            "neighbor_counts",
            length=labels.size,
        )
        if np.any(neighbor_counts < 0):
            raise ValidationError("neighbor_counts must be non-negative.")
        label_order = _label_order(self.label_order)
        if not {int(value) for value in np.unique(labels)}.issubset(label_order):
            raise ValidationError("every output label must occur in label_order.")
        radius = _positive_real(self.radius, "radius")
        iterations = _non_negative_integer(self.n_iterations, "n_iterations")
        if not isinstance(self.converged, (bool, np.bool_)):
            raise ValidationError("converged must be a bool.")
        if self.termination not in _TERMINATIONS:
            raise ValidationError("termination must be 'empty', 'converged', or 'max_iterations'.")
        neighbor_pairs = _non_negative_integer(self.neighbor_pairs, "neighbor_pairs")
        workspace_bytes = _non_negative_integer(self.workspace_bytes, "workspace_bytes")

        object.__setattr__(self, "labels", immutable_array_copy(labels))
        object.__setattr__(self, "neighbor_counts", immutable_array_copy(neighbor_counts))
        object.__setattr__(self, "label_order", label_order)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "n_iterations", iterations)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "neighbor_pairs", neighbor_pairs)
        object.__setattr__(self, "workspace_bytes", workspace_bytes)

    def __copy__(self) -> ValleySeekingResult:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ValleySeekingResult:
        rebuilt = _rebuild_valley_seeking_result(
            self.labels,
            self.neighbor_counts,
            self.label_order,
            self.radius,
            self.n_iterations,
            self.converged,
            self.termination,
            self.neighbor_pairs,
            self.workspace_bytes,
        )
        memo[id(self)] = rebuilt
        return rebuilt

    def __reduce__(self):
        return (
            _rebuild_valley_seeking_result,
            (
                self.labels,
                self.neighbor_counts,
                self.label_order,
                self.radius,
                self.n_iterations,
                self.converged,
                self.termination,
                self.neighbor_pairs,
                self.workspace_bytes,
            ),
        )


def valley_seeking(
    features: FeatureMatrix,
    initial_labels: np.ndarray | None = None,
    *,
    radius: float,
    max_iterations: int = 1000,
) -> ValleySeekingResult:
    """Cluster spike-major features with the native CPU Valley Seeking kernel.

    ``features.data`` must be exact C-contiguous ``float64`` with
    ``(observation, feature)`` axis order. When ``initial_labels`` is omitted,
    observation ``i`` receives the explicit deterministic seed label ``i``.
    Supplied labels remain arbitrary signed, non-contiguous ``int64`` values.
    Neither path estimates a bandwidth or silently converts dtype/layout.
    """

    if not isinstance(features, FeatureMatrix):
        raise ValidationError("features must be a FeatureMatrix.")
    points = features.data
    if points.dtype != np.dtype(np.float64):
        raise ValidationError("features.data must have dtype float64; no conversion is performed.")
    if not points.flags.c_contiguous:
        raise ValidationError("features.data must be C-contiguous; no copy is performed.")
    if points.shape[1] == 0:
        raise ValidationError("features must contain at least one feature column.")
    if not np.all(np.isfinite(points)):
        raise ValidationError("features.data must contain only finite values.")

    if initial_labels is None:
        labels = np.arange(points.shape[0], dtype=np.int64)
    else:
        labels = _int64_vector(initial_labels, "initial_labels", length=points.shape[0])
        if not labels.flags.c_contiguous:
            raise ValidationError("initial_labels must be C-contiguous; no copy is performed.")
    resolved_radius = _positive_real(radius, "radius")
    iteration_limit = _positive_integer(max_iterations, "max_iterations")

    try:
        raw = load_cpu_sorting_operation("valley_seeking")._valley_seeking(
            points,
            labels,
            resolved_radius,
            iteration_limit,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(str(exc)) from exc
    return ValleySeekingResult(
        labels=np.asarray(raw["labels"]),
        neighbor_counts=np.asarray(raw["neighbor_counts"]),
        label_order=tuple(int(value) for value in np.asarray(raw["label_order"])),
        radius=raw["radius"],
        n_iterations=raw["n_iterations"],
        converged=raw["converged"],
        termination=raw["termination"],
        neighbor_pairs=raw["neighbor_pairs"],
        workspace_bytes=raw["workspace_bytes"],
    )


def _int64_vector(value: object, name: str, *, length: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray with dtype int64.")
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.dtype != np.dtype(np.int64):
        raise ValidationError(f"{name} must be 1D int64.")
    if length is not None and arr.size != length:
        raise ValidationError(f"{name} length must match the observation count.")
    return arr


def _label_order(value: object) -> tuple[int, ...]:
    try:
        labels = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("label_order must be an ordered integer sequence.") from exc
    normalized: list[int] = []
    for label in labels:
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral):
            raise ValidationError("label_order must contain only integers.")
        normalized.append(int(label))
    if normalized != sorted(set(normalized)):
        raise ValidationError("label_order must be unique and ascending.")
    return tuple(normalized)


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite positive real number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValidationError(f"{name} must be a finite positive real number.")
    return resolved


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be a non-negative integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValidationError(f"{name} must be a non-negative integer.")
    return resolved


def _positive_integer(value: object, name: str) -> int:
    resolved = _non_negative_integer(value, name)
    if resolved == 0:
        raise ValidationError(f"{name} must be a positive integer.")
    return resolved


def _rebuild_valley_seeking_result(
    labels: np.ndarray,
    neighbor_counts: np.ndarray,
    label_order: tuple[int, ...],
    radius: float,
    n_iterations: int,
    converged: bool,
    termination: ValleySeekingTermination,
    neighbor_pairs: int,
    workspace_bytes: int,
) -> ValleySeekingResult:
    return ValleySeekingResult(
        labels=labels,
        neighbor_counts=neighbor_counts,
        label_order=label_order,
        radius=radius,
        n_iterations=n_iterations,
        converged=converged,
        termination=termination,
        neighbor_pairs=neighbor_pairs,
        workspace_bytes=workspace_bytes,
    )
