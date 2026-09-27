#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Foundational metrics for ordered spike events and cluster spike trains."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from neurale.data._helpers import immutable_array_copy
from neurale.exceptions import ValidationError

from ._helpers import _int64_scalar, _nonnegative_float


@dataclass(frozen=True, slots=True)
class MatchingEventsResult:
    """Immutable one-to-one correspondence returned by :func:`matching_events`.

    Indices address positions in the supplied source and target arrays, not
    absolute sample indices. ``sample_errors`` is signed and is defined as
    ``target_sample - source_sample`` for each matched pair.
    """

    matched_source_indices: np.ndarray
    matched_target_indices: np.ndarray
    unmatched_source_indices: np.ndarray
    unmatched_target_indices: np.ndarray
    sample_errors: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "matched_source_indices": self.matched_source_indices,
            "matched_target_indices": self.matched_target_indices,
            "unmatched_source_indices": self.unmatched_source_indices,
            "unmatched_target_indices": self.unmatched_target_indices,
            "sample_errors": self.sample_errors,
        }
        normalized: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            arr = np.asarray(value)
            if arr.ndim != 1 or arr.dtype != np.dtype(np.int64):
                raise ValidationError(f"MatchingEventsResult.{name} must be 1D int64.")
            normalized[name] = immutable_array_copy(arr)

        n_matches = normalized["matched_source_indices"].size
        if (
            normalized["matched_target_indices"].size != n_matches
            or normalized["sample_errors"].size != n_matches
        ):
            raise ValidationError("Matched indices and sample_errors must have equal length.")
        for name in (
            "matched_source_indices",
            "matched_target_indices",
            "unmatched_source_indices",
            "unmatched_target_indices",
        ):
            if np.any(normalized[name] < 0):
                raise ValidationError(f"MatchingEventsResult.{name} must be non-negative.")

        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def n_matches(self) -> int:
        """Number of one-to-one event pairs."""

        return int(self.matched_source_indices.size)

    def __copy__(self) -> MatchingEventsResult:
        # The result is deeply immutable, so a shallow copy shares the instance
        # instead of using the default ``__reduce_ex__`` reconstruction, which
        # bypasses ``__post_init__`` and restores writable arrays.
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> MatchingEventsResult:
        rebuilt = _rebuild_matching_events_result(
            self.matched_source_indices,
            self.matched_target_indices,
            self.unmatched_source_indices,
            self.unmatched_target_indices,
            self.sample_errors,
        )
        memo[id(self)] = rebuilt
        return rebuilt

    def __reduce__(self):
        # Route pickle through the constructor so ``__post_init__`` re-runs and
        # re-establishes strong immutability instead of restoring the arrays as
        # ordinary writable buffers.
        return (
            _rebuild_matching_events_result,
            (
                self.matched_source_indices,
                self.matched_target_indices,
                self.unmatched_source_indices,
                self.unmatched_target_indices,
                self.sample_errors,
            ),
        )


def _rebuild_matching_events_result(
    matched_source_indices: np.ndarray,
    matched_target_indices: np.ndarray,
    unmatched_source_indices: np.ndarray,
    unmatched_target_indices: np.ndarray,
    sample_errors: np.ndarray,
) -> MatchingEventsResult:
    """Reconstruct a :class:`MatchingEventsResult` through its constructor.

    Pickle and ``copy.deepcopy`` route here so ``__post_init__`` re-validates
    and re-establishes strong array immutability rather than restoring the
    arrays as ordinary writable buffers.
    """

    return MatchingEventsResult(
        matched_source_indices=matched_source_indices,
        matched_target_indices=matched_target_indices,
        unmatched_source_indices=unmatched_source_indices,
        unmatched_target_indices=unmatched_target_indices,
        sample_errors=sample_errors,
    )


@dataclass(frozen=True, slots=True)
class ClusterIsiMetrics:
    """Immutable ISI and refractory summary for one cluster.

    ``isi_samples`` concatenates only within-segment differences in input
    order. A cluster with no interval has ``None`` summaries and a violation
    fraction of ``0.0``.
    """

    label: int
    n_spikes: int
    n_segments: int
    isi_samples: np.ndarray
    refractory_samples: int
    fs: float
    n_isis: int = field(init=False)
    n_violations: int = field(init=False)
    violation_fraction: float = field(init=False)
    minimum_isi_samples: int | None = field(init=False)
    maximum_isi_samples: int | None = field(init=False)
    mean_isi_samples: float | None = field(init=False)
    median_isi_samples: float | None = field(init=False)

    def __post_init__(self) -> None:
        label = _int64_scalar(self.label, "ClusterIsiMetrics.label")
        n_spikes = _nonnegative_integer(self.n_spikes, "ClusterIsiMetrics.n_spikes")
        n_segments = _nonnegative_integer(self.n_segments, "ClusterIsiMetrics.n_segments")
        refractory_samples = _nonnegative_integer(
            self.refractory_samples, "ClusterIsiMetrics.refractory_samples"
        )
        fs = _positive_float(self.fs, "ClusterIsiMetrics.fs")
        intervals = np.asarray(self.isi_samples)
        if intervals.ndim != 1 or intervals.dtype != np.dtype(np.int64):
            raise ValidationError("ClusterIsiMetrics.isi_samples must be 1D int64.")
        if np.any(intervals < 0):
            raise ValidationError("ClusterIsiMetrics.isi_samples must be non-negative.")
        if n_segments > n_spikes:
            raise ValidationError("ClusterIsiMetrics.n_segments cannot exceed n_spikes.")
        expected_intervals = n_spikes - n_segments
        if intervals.size != expected_intervals:
            raise ValidationError(
                "ClusterIsiMetrics.isi_samples length must equal n_spikes - n_segments."
            )

        intervals = immutable_array_copy(intervals)
        n_isis = int(intervals.size)
        n_violations = int(np.count_nonzero(intervals < refractory_samples))
        if n_isis:
            minimum: int | None = int(np.min(intervals))
            maximum: int | None = int(np.max(intervals))
            mean: float | None = float(np.mean(intervals, dtype=np.float64))
            median: float | None = float(np.median(intervals))
        else:
            minimum = maximum = None
            mean = median = None

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "n_spikes", n_spikes)
        object.__setattr__(self, "n_segments", n_segments)
        object.__setattr__(self, "isi_samples", intervals)
        object.__setattr__(self, "refractory_samples", refractory_samples)
        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "n_isis", n_isis)
        object.__setattr__(self, "n_violations", n_violations)
        object.__setattr__(
            self,
            "violation_fraction",
            0.0 if n_isis == 0 else n_violations / n_isis,
        )
        object.__setattr__(self, "minimum_isi_samples", minimum)
        object.__setattr__(self, "maximum_isi_samples", maximum)
        object.__setattr__(self, "mean_isi_samples", mean)
        object.__setattr__(self, "median_isi_samples", median)

    def __copy__(self) -> ClusterIsiMetrics:
        # Deeply immutable, so a shallow copy shares the instance instead of
        # bypassing ``__post_init__`` (which would restore writable arrays).
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ClusterIsiMetrics:
        rebuilt = _rebuild_cluster_isi_metrics(
            self.label,
            self.n_spikes,
            self.n_segments,
            self.isi_samples,
            self.refractory_samples,
            self.fs,
        )
        memo[id(self)] = rebuilt
        return rebuilt

    def __reduce__(self):
        # Only the constructor fields are serialized; the derived ISI summaries
        # are recomputed by ``__post_init__`` so a stale value can never survive
        # a round-trip.
        return (
            _rebuild_cluster_isi_metrics,
            (
                self.label,
                self.n_spikes,
                self.n_segments,
                self.isi_samples,
                self.refractory_samples,
                self.fs,
            ),
        )


def _rebuild_cluster_isi_metrics(
    label: int,
    n_spikes: int,
    n_segments: int,
    isi_samples: np.ndarray,
    refractory_samples: int,
    fs: float,
) -> ClusterIsiMetrics:
    """Reconstruct a :class:`ClusterIsiMetrics` through its constructor.

    Only the constructor fields are accepted: ``__post_init__`` recomputes
    ``n_isis``, the violation summary, and the ISI statistics so they can
    never deserialize out of sync with ``isi_samples``.
    """

    return ClusterIsiMetrics(
        label=label,
        n_spikes=n_spikes,
        n_segments=n_segments,
        isi_samples=isi_samples,
        refractory_samples=refractory_samples,
        fs=fs,
    )


def matching_events(
    source_sample_indices: np.ndarray,
    target_sample_indices: np.ndarray,
    *,
    tol: int = 0,
    source_segment_ids: np.ndarray | None = None,
    target_segment_ids: np.ndarray | None = None,
) -> MatchingEventsResult:
    """Match ordered event samples one-to-one within an inclusive tolerance.

    Matching is greedy in input order: within each segment the earliest
    unmatched source is paired with the earliest eligible target. This fixes
    duplicate and equidistant tie behavior and maximizes the number of pairs
    for ordered inputs. The algorithm is linear in the total event count.

    Sample arrays must be 1D ``int64``. Without segment IDs they
    must be globally nondecreasing. When segment IDs are supplied, both must be
    1D ``int64`` arrays, every segment ID must occupy one
    contiguous run, and samples must be nondecreasing inside each run. Segment
    run order may differ between source and target; events are never paired
    across unequal IDs.

    ``sample_errors`` is ``target_sample - source_sample``. ``tol`` is an
    integer number of samples and its boundary is inclusive.
    """
    source = _ordered_samples(source_sample_indices, "source_sample_indices")
    target = _ordered_samples(target_sample_indices, "target_sample_indices")
    resolved_tol = _nonnegative_integer(tol, "tol")
    source_runs, target_runs = _paired_segment_runs(
        source,
        target,
        source_segment_ids,
        target_segment_ids,
    )

    matched_source: list[int] = []
    matched_target: list[int] = []
    sample_errors: list[int] = []
    source_matched = np.zeros(source.size, dtype=np.bool_)
    target_matched = np.zeros(target.size, dtype=np.bool_)

    for segment_id, (source_start, source_stop) in source_runs.items():
        target_run = target_runs.get(segment_id)
        if target_run is None:
            continue
        target_start, target_stop = target_run
        source_position = source_start
        target_position = target_start
        while source_position < source_stop and target_position < target_stop:
            source_sample = int(source[source_position])
            target_sample = int(target[target_position])
            error = target_sample - source_sample
            if error < -resolved_tol:
                target_position += 1
            elif error > resolved_tol:
                source_position += 1
            else:
                matched_source.append(source_position)
                matched_target.append(target_position)
                sample_errors.append(error)
                source_matched[source_position] = True
                target_matched[target_position] = True
                source_position += 1
                target_position += 1

    return MatchingEventsResult(
        matched_source_indices=np.asarray(matched_source, dtype=np.int64),
        matched_target_indices=np.asarray(matched_target, dtype=np.int64),
        unmatched_source_indices=np.flatnonzero(~source_matched).astype(np.int64, copy=False),
        unmatched_target_indices=np.flatnonzero(~target_matched).astype(np.int64, copy=False),
        sample_errors=np.asarray(sample_errors, dtype=np.int64),
    )


def isi_metrics(
    sample_indices: np.ndarray,
    labels: np.ndarray,
    *,
    fs: float,
    refractory_interval: float,
    segment_ids: np.ndarray | None = None,
    noise_label: int | None = None,
) -> tuple[ClusterIsiMetrics, ...]:
    """Compute per-cluster ISI and refractory summaries without crossing gaps.

    Inputs are ordered spike samples and arbitrary signed, non-contiguous
    ``int64`` cluster labels. Results are ordered by increasing numeric label.
    A label is excluded only when it equals an explicitly supplied
    ``noise_label``; negative labels otherwise remain ordinary clusters.

    The refractory boundary follows threshold detection:
    ``refractory_samples = ceil(refractory_interval * fs)`` and an ISI
    is a violation only when it is strictly less than that count. Therefore an
    ISI exactly on the rounded boundary is allowed, while repeated samples
    produce a zero ISI and violate every positive refractory interval.

    ``segment_ids`` follows the same contiguous-run ordering contract as
    :func:`matching_events`. Differences are formed only between consecutive
    spikes of the same label in the same segment. Empty input returns an empty
    tuple.
    """
    samples = _ordered_samples(sample_indices, "sample_indices")
    cluster_labels = _int64_array(labels, "labels", length=samples.size)
    rate = _positive_float(fs, "fs")
    refractory_seconds = _nonnegative_float(refractory_interval, "refractory_interval")
    refractory_value = refractory_seconds * rate
    if not math.isfinite(refractory_value):
        raise ValidationError("refractory_interval times fs must be finite.")
    refractory_samples = math.ceil(refractory_value)
    if refractory_samples > np.iinfo(np.int64).max:
        raise ValidationError("refractory_interval times fs must round to an int64 sample count.")
    excluded_label = None if noise_label is None else _int64_scalar(noise_label, "noise_label")

    if segment_ids is None:
        segment_values = np.zeros(samples.size, dtype=np.int64)
    else:
        segment_values = _int64_array(segment_ids, "segment_ids", length=samples.size)
    segment_runs = _segment_runs(samples, segment_values, "sample_indices")

    intervals: dict[int, list[int]] = {}
    spike_counts: dict[int, int] = {}
    segment_counts: dict[int, int] = {}
    for start, stop in segment_runs.values():
        previous_sample: dict[int, int] = {}
        labels_in_segment: set[int] = set()
        for pos in range(start, stop):
            label = int(cluster_labels[pos])
            if excluded_label is not None and label == excluded_label:
                continue
            sample = int(samples[pos])
            spike_counts[label] = spike_counts.get(label, 0) + 1
            labels_in_segment.add(label)
            previous = previous_sample.get(label)
            if previous is not None:
                interval = sample - previous
                if interval > np.iinfo(np.int64).max:
                    raise ValidationError("Within-segment ISIs must fit in int64.")
                intervals.setdefault(label, []).append(interval)
            previous_sample[label] = sample
        for label in labels_in_segment:
            segment_counts[label] = segment_counts.get(label, 0) + 1

    return tuple(
        ClusterIsiMetrics(
            label=label,
            n_spikes=spike_counts[label],
            n_segments=segment_counts[label],
            isi_samples=np.asarray(intervals.get(label, ()), dtype=np.int64),
            refractory_samples=refractory_samples,
            fs=rate,
        )
        for label in sorted(spike_counts)
    )


def _paired_segment_runs(
    source: np.ndarray,
    target: np.ndarray,
    source_segment_ids: np.ndarray | None,
    target_segment_ids: np.ndarray | None,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    if (source_segment_ids is None) != (target_segment_ids is None):
        raise ValidationError(
            "source_segment_ids and target_segment_ids must be supplied together."
        )
    if source_segment_ids is None:
        source_segments = np.zeros(source.size, dtype=np.int64)
        target_segments = np.zeros(target.size, dtype=np.int64)
    else:
        source_segments = _int64_array(source_segment_ids, "source_segment_ids", length=source.size)
        target_segments = _int64_array(target_segment_ids, "target_segment_ids", length=target.size)
    return (
        _segment_runs(source, source_segments, "source_sample_indices"),
        _segment_runs(target, target_segments, "target_sample_indices"),
    )


def _segment_runs(
    samples: np.ndarray,
    segment_ids: np.ndarray,
    sample_name: str,
) -> dict[int, tuple[int, int]]:
    runs: dict[int, tuple[int, int]] = {}
    start = 0
    while start < samples.size:
        segment_id = int(segment_ids[start])
        if segment_id in runs:
            raise ValidationError("Each segment ID must occupy one contiguous run.")
        stop = start + 1
        while stop < samples.size and int(segment_ids[stop]) == segment_id:
            stop += 1
        if stop - start >= 2 and np.any(samples[start + 1 : stop] < samples[start : stop - 1]):
            raise ValidationError(f"{sample_name} must be nondecreasing within each segment.")
        runs[segment_id] = (start, stop)
        start = stop
    return runs


def _ordered_samples(value: np.ndarray, name: str) -> np.ndarray:
    return _int64_array(value, name)


def _int64_array(value: np.ndarray, name: str, *, length: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray.")
    if value.ndim != 1 or value.dtype != np.dtype(np.int64):
        raise ValidationError(f"{name} must be 1D int64.")
    if length is not None and value.size != length:
        raise ValidationError(f"{name} must have length {length}.")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    integer = _int64_scalar(value, name)
    if integer < 0:
        raise ValidationError(f"{name} must be non-negative.")
    return integer


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{name} must be finite and greater than zero.")
    return result


__all__ = [
    "ClusterIsiMetrics",
    "MatchingEventsResult",
    "isi_metrics",
    "matching_events",
]
