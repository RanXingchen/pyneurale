#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Explicit, auditable offline spike-sorting curation operations."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from neurale.data import SpikeWaveformBatch
from neurale.data._helpers import freeze_metadata, immutable_array_copy
from neurale.exceptions import ValidationError

from ._helpers import _int64_scalar, _nonnegative_float

CurationOperation = Literal[
    "remove_tiny_clusters",
    "reject_waveform_outliers",
    "merge_clusters",
    "split_cluster",
]
_OPERATIONS = frozenset(
    {
        "remove_tiny_clusters",
        "reject_waveform_outliers",
        "merge_clusters",
        "split_cluster",
    }
)


@dataclass(frozen=True, slots=True, eq=False)
class CurationOperationRecord:
    """Immutable audit record for one explicit label-changing operation.

    ``source_counts`` and ``target_counts`` count changed events and align with
    ``source_labels`` and ``target_labels`` respectively. The affected indices
    address the original input event axis and are strictly increasing.
    ``target_event_indices`` aligns with ``target_labels`` and partitions
    ``affected_event_indices``: element ``i`` is the strictly-increasing
    original-event indices relabeled to ``target_labels[i]``. Single-target
    operations (remove, outlier, merge) carry one element equal to the whole
    affected set; a split carries one element per child. This makes the record
    a complete, replayable description of where every changed event landed.
    """

    operation: CurationOperation
    source_labels: tuple[int, ...]
    target_labels: tuple[int, ...]
    source_counts: tuple[int, ...]
    target_counts: tuple[int, ...]
    affected_event_indices: np.ndarray
    target_event_indices: tuple[np.ndarray, ...]
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or self.operation not in _OPERATIONS:
            raise ValidationError(f"Unsupported curation operation: {self.operation!r}.")
        source_labels = _label_tuple(self.source_labels, "source_labels", allow_empty=True)
        target_labels = _label_tuple(self.target_labels, "target_labels", allow_empty=False)
        if set(source_labels) & set(target_labels):
            raise ValidationError("Curation source_labels and target_labels must be disjoint.")
        source_counts = _count_tuple(
            self.source_counts,
            "source_counts",
            length=len(source_labels),
        )
        target_counts = _count_tuple(
            self.target_counts,
            "target_counts",
            length=len(target_labels),
        )
        affected = _event_indices(self.affected_event_indices, "affected_event_indices")
        if sum(source_counts) != affected.size or sum(target_counts) != affected.size:
            raise ValidationError(
                "Curation source_counts and target_counts must each sum to the affected count."
            )
        target_event_indices = _target_event_indices(
            self.target_event_indices,
            target_labels=target_labels,
            target_counts=target_counts,
            affected=affected,
        )
        if not isinstance(self.parameters, Mapping):
            raise ValidationError("parameters must be a mapping.")

        object.__setattr__(self, "source_labels", source_labels)
        object.__setattr__(self, "target_labels", target_labels)
        object.__setattr__(self, "source_counts", source_counts)
        object.__setattr__(self, "target_counts", target_counts)
        object.__setattr__(self, "affected_event_indices", immutable_array_copy(affected))
        object.__setattr__(
            self,
            "target_event_indices",
            tuple(immutable_array_copy(indices) for indices in target_event_indices),
        )
        object.__setattr__(self, "parameters", freeze_metadata(self.parameters))

    @property
    def affected_count(self) -> int:
        """Number of events whose label changed."""

        return int(self.affected_event_indices.size)

    def __copy__(self) -> CurationOperationRecord:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> CurationOperationRecord:
        rebuilt = _rebuild_operation_record(
            self.operation,
            self.source_labels,
            self.target_labels,
            self.source_counts,
            self.target_counts,
            self.affected_event_indices,
            self.target_event_indices,
            copy.deepcopy(dict(self.parameters), memo),
        )
        memo[id(self)] = rebuilt
        return rebuilt

    def __reduce__(self):
        return (
            _rebuild_operation_record,
            (
                self.operation,
                self.source_labels,
                self.target_labels,
                self.source_counts,
                self.target_counts,
                self.affected_event_indices,
                self.target_event_indices,
                dict(self.parameters),
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CurationResult:
    """Immutable labels and audit records for an explicit curation call."""

    updated_labels: np.ndarray
    operations: tuple[CurationOperationRecord, ...]

    def __post_init__(self) -> None:
        labels = _labels(self.updated_labels)
        operations = _operations(self.operations, labels.size)
        object.__setattr__(self, "updated_labels", immutable_array_copy(labels))
        object.__setattr__(self, "operations", operations)

    def __copy__(self) -> CurationResult:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> CurationResult:
        rebuilt = _rebuild_curation_result(
            self.updated_labels,
            tuple(copy.deepcopy(operation, memo) for operation in self.operations),
        )
        memo[id(self)] = rebuilt
        return rebuilt

    def __reduce__(self):
        return _rebuild_curation_result, (self.updated_labels, self.operations)


@dataclass(frozen=True, slots=True, eq=False)
class WaveformOutlierResult:
    """Immutable event-aligned output from :func:`reject_waveform_outliers`.

    ``scores``, ``thresholds``, ``rejection_mask``, ``updated_labels``, and
    ``source_sample_indices`` all use the input batch's spike axis. Scores and
    thresholds are ``NaN`` for the explicit noise label; thresholds are also
    ``NaN`` for clusters below ``minimum_cluster_size``.
    """

    scores: np.ndarray
    thresholds: np.ndarray
    rejection_mask: np.ndarray
    updated_labels: np.ndarray
    source_sample_indices: np.ndarray
    operations: tuple[CurationOperationRecord, ...]

    def __post_init__(self) -> None:
        labels = _labels(self.updated_labels)
        n_events = labels.size
        scores = _float_vector(self.scores, "scores", n_events)
        thresholds = _float_vector(self.thresholds, "thresholds", n_events)
        rejection_mask = np.asarray(self.rejection_mask)
        if rejection_mask.ndim != 1 or rejection_mask.dtype != np.dtype(np.bool_):
            raise ValidationError("rejection_mask must be 1D bool.")
        if rejection_mask.size != n_events:
            raise ValidationError("rejection_mask length must match updated_labels.")
        source_samples = _int64_vector(
            self.source_sample_indices,
            "source_sample_indices",
            length=n_events,
        )
        operations = _operations(self.operations, n_events)
        affected = np.concatenate([operation.affected_event_indices for operation in operations])
        expected_mask = np.zeros(n_events, dtype=np.bool_)
        expected_mask[affected] = True
        if not np.array_equal(rejection_mask, expected_mask):
            raise ValidationError("rejection_mask must match operation affected_event_indices.")

        object.__setattr__(self, "scores", immutable_array_copy(scores))
        object.__setattr__(self, "thresholds", immutable_array_copy(thresholds))
        object.__setattr__(self, "rejection_mask", immutable_array_copy(rejection_mask))
        object.__setattr__(self, "updated_labels", immutable_array_copy(labels))
        object.__setattr__(self, "source_sample_indices", immutable_array_copy(source_samples))
        object.__setattr__(self, "operations", operations)

    def __copy__(self) -> WaveformOutlierResult:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> WaveformOutlierResult:
        rebuilt = _rebuild_waveform_outlier_result(
            self.scores,
            self.thresholds,
            self.rejection_mask,
            self.updated_labels,
            self.source_sample_indices,
            tuple(copy.deepcopy(operation, memo) for operation in self.operations),
        )
        memo[id(self)] = rebuilt
        return rebuilt

    def __reduce__(self):
        return (
            _rebuild_waveform_outlier_result,
            (
                self.scores,
                self.thresholds,
                self.rejection_mask,
                self.updated_labels,
                self.source_sample_indices,
                self.operations,
            ),
        )


def remove_tiny_clusters(
    labels: np.ndarray,
    *,
    min_cluster_size: int,
    noise_label: int,
) -> CurationResult:
    """Relabel clusters smaller than ``min_cluster_size`` as explicit noise.

    The noise label itself is never size-filtered. Clusters exactly at the
    threshold are retained. A no-op still returns one audit record with empty
    source labels and affected indices.
    """
    values = _labels(labels)
    minimum = _positive_integer(min_cluster_size, "min_cluster_size")
    noise = _int64_scalar(noise_label, "noise_label")
    unique, counts = np.unique(values, return_counts=True)
    removed = tuple(
        (int(label), int(count))
        for label, count in zip(unique, counts, strict=True)
        if int(label) != noise and int(count) < minimum
    )
    removed_labels = tuple(label for label, _ in removed)
    removed_counts = tuple(count for _, count in removed)
    mask = (
        np.isin(values, np.asarray(removed_labels, dtype=np.int64))
        if removed
        else np.zeros(values.size, dtype=np.bool_)
    )
    affected = np.flatnonzero(mask).astype(np.int64, copy=False)
    updated = values.copy()
    updated[mask] = noise
    operation = CurationOperationRecord(
        operation="remove_tiny_clusters",
        source_labels=removed_labels,
        target_labels=(noise,),
        source_counts=removed_counts,
        target_counts=(int(affected.size),),
        affected_event_indices=affected,
        target_event_indices=(affected,),
        parameters={"min_cluster_size": minimum, "noise_label": noise},
    )
    return CurationResult(updated_labels=updated, operations=(operation,))


#: Ceiling on one gathered waveform block, in bytes. The per-cluster work below
#: is blocked so its temporary memory depends on this constant rather than on
#: the cluster size: a single-cluster 100k-spike batch would otherwise allocate
#: several times the waveform payload.
_OUTLIER_BLOCK_BYTES = 4 * 1024 * 1024


def _block_length(total: int, values_per_entry: int) -> int:
    """Return how many entries of *values_per_entry* float64 values one block holds.

    Never zero, so an entry wider than the ceiling still makes progress one at a
    time, and never more than *total*.
    """
    return min(total, max(1, _OUTLIER_BLOCK_BYTES // (values_per_entry * 8)))


def _residual_distances(flat_waveforms: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return each spike's Euclidean residual norm from its cluster median.

    ``flat_waveforms`` is the ``(n_spikes, n_samples * n_channels)`` view of the
    batch. The two passes block along different axes, because each needs the
    whole of the other:

    * the elementwise median needs every spike, so it blocks over columns;
    * a spike's residual norm needs every column, so it blocks over spikes.

    Blocking the norm over spikes rather than columns is what keeps the result
    bit-identical to reducing the full cluster at once: each spike's sum is
    still one uninterrupted reduction over its complete residual row, so no
    partial sums are combined in a new order. Blocking over columns instead
    would perturb the result by a few units in the last place, which can flip a
    rejection for a spike sitting exactly on the threshold.

    The largest temporary is therefore one gathered block bounded by
    :data:`_OUTLIER_BLOCK_BYTES`, not several copies of the whole cluster.
    """
    count = int(indices.size)
    positions = int(flat_waveforms.shape[1])
    if count == 0 or positions == 0:
        return np.zeros(count, dtype=np.float64)

    block_rows = _block_length(count, positions)
    block_columns = _block_length(positions, count)

    median = np.empty(positions, dtype=np.float64)
    for start in range(0, positions, block_columns):
        stop = min(start + block_columns, positions)
        block = np.asarray(flat_waveforms[indices, start:stop], dtype=np.float64)
        median[start:stop] = np.median(block, axis=0)

    distances = np.empty(count, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, count, block_rows):
            stop = min(start + block_rows, count)
            block = np.asarray(flat_waveforms[indices[start:stop]], dtype=np.float64)
            # In place: the gathered block is already a private copy, so the
            # residual and its square need no further allocation.
            block -= median
            block *= block
            distances[start:stop] = np.sqrt(block.sum(axis=1, dtype=np.float64))
    return distances


def reject_waveform_outliers(
    batch: SpikeWaveformBatch,
    labels: np.ndarray,
    *,
    threshold_multiplier: float,
    minimum_cluster_size: int,
    noise_label: int,
) -> WaveformOutlierResult:
    """Reject per-cluster waveform outliers with a centered-MAD rule.

    For each non-noise cluster, the elementwise median waveform is computed.
    An event score is the Euclidean norm of its flattened residual from that
    median. With ``d_med = median(score)`` and
    ``mad = median(abs(score - d_med))``, the rejection threshold is
    ``d_med + threshold_multiplier * mad``. Only scores strictly above the
    threshold are rejected.

    A zero MAD is not replaced or scaled: the threshold remains ``d_med``.
    Clusters smaller than ``minimum_cluster_size`` receive scores but no
    threshold (``NaN``) and are not rejected. The explicit noise label receives
    neither scores nor thresholds.
    """
    if not isinstance(batch, SpikeWaveformBatch):
        raise ValidationError("batch must be a SpikeWaveformBatch.")
    values = _labels(labels, length=batch.n_spikes)
    multiplier = _nonnegative_float(threshold_multiplier, "threshold_multiplier")
    minimum = _positive_integer(minimum_cluster_size, "minimum_cluster_size")
    noise = _int64_scalar(noise_label, "noise_label")
    # The Euclidean residual norm mixes every channel's samples into one sum, so
    # the score is only unit-independent when all channels share one amplitude
    # unit. A per-channel unit (e.g. uV alongside mV) would let the representation
    # alone swing the residual and the rejection decision.
    waveform_unit = _shared_waveform_unit(batch)
    # (n_spikes, n_samples * n_channels). A C-contiguous waveform array reshapes
    # to a view, so this costs nothing and lets the per-cluster work run in
    # column blocks instead of on whole-cluster temporaries.
    flat_waveforms = batch.waveforms.reshape(batch.n_spikes, -1)

    scores = np.full(batch.n_spikes, np.nan, dtype=np.float64)
    thresholds = np.full(batch.n_spikes, np.nan, dtype=np.float64)
    rejected = np.zeros(batch.n_spikes, dtype=np.bool_)
    eligible_labels: list[int] = []
    median_distances: list[float] = []
    mads: list[float] = []
    cluster_thresholds: list[float] = []

    for raw_label in np.unique(values):
        label = int(raw_label)
        if label == noise:
            continue
        indices = np.flatnonzero(values == label)
        distances = _residual_distances(flat_waveforms, indices)
        if not np.all(np.isfinite(distances)):
            raise ValidationError("Waveform residual distances must remain finite.")
        scores[indices] = distances
        if indices.size < minimum:
            continue
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        threshold = median_distance + multiplier * mad
        if not math.isfinite(threshold):
            raise ValidationError("Waveform outlier thresholds must remain finite.")
        thresholds[indices] = threshold
        rejected[indices] = distances > threshold
        eligible_labels.append(label)
        median_distances.append(median_distance)
        mads.append(mad)
        cluster_thresholds.append(threshold)

    updated = values.copy()
    updated[rejected] = noise
    affected = np.flatnonzero(rejected).astype(np.int64, copy=False)
    rejected_labels, rejected_counts = _affected_label_counts(values, affected)
    operation = CurationOperationRecord(
        operation="reject_waveform_outliers",
        source_labels=rejected_labels,
        target_labels=(noise,),
        source_counts=rejected_counts,
        target_counts=(int(affected.size),),
        affected_event_indices=affected,
        target_event_indices=(affected,),
        parameters={
            "threshold_multiplier": multiplier,
            "minimum_cluster_size": minimum,
            "noise_label": noise,
            "score": "euclidean_residual_from_elementwise_median_waveform",
            "mad_scale": 1.0,
            "comparison": "score > threshold",
            "waveform_unit": waveform_unit,
            "eligible_labels": tuple(eligible_labels),
            "median_distances": tuple(median_distances),
            "centered_mads": tuple(mads),
            "cluster_thresholds": tuple(cluster_thresholds),
        },
    )
    return WaveformOutlierResult(
        scores=scores,
        thresholds=thresholds,
        rejection_mask=rejected,
        updated_labels=updated,
        source_sample_indices=batch.sample_indices,
        operations=(operation,),
    )


def merge_clusters(
    labels: np.ndarray,
    *,
    source_labels: Sequence[int] | np.ndarray,
    target_label: int,
) -> CurationResult:
    """Merge explicit existing source labels into a caller-supplied target.

    Source labels must be non-empty, unique, present in the input, and must not
    contain the target. The target may be either an existing label or a new
    caller-supplied label. No proposal or label allocation is performed.
    """
    values = _labels(labels)
    sources = _label_sequence(source_labels, "source_labels", minimum_length=1)
    target = _int64_scalar(target_label, "target_label")
    if target in sources:
        raise ValidationError("target_label must not also appear in source_labels.")
    present = {int(label) for label in np.unique(values)}
    missing = tuple(label for label in sources if label not in present)
    if missing:
        raise ValidationError(f"source_labels contain labels absent from input: {missing!r}.")
    canonical_sources = tuple(sorted(sources))
    mask = np.isin(values, np.asarray(canonical_sources, dtype=np.int64))
    affected = np.flatnonzero(mask).astype(np.int64, copy=False)
    counts = tuple(int(np.count_nonzero(values == label)) for label in canonical_sources)
    updated = values.copy()
    updated[mask] = target
    operation = CurationOperationRecord(
        operation="merge_clusters",
        source_labels=canonical_sources,
        target_labels=(target,),
        source_counts=counts,
        target_counts=(int(affected.size),),
        affected_event_indices=affected,
        target_event_indices=(affected,),
        parameters={},
    )
    return CurationResult(updated_labels=updated, operations=(operation,))


def split_cluster(
    labels: np.ndarray,
    *,
    source_label: int,
    child_assignments: Mapping[int, np.ndarray],
) -> CurationResult:
    """Split one cluster using complete caller-supplied child assignments.

    ``child_assignments`` maps each explicit child label to original event
    indices. At least two non-empty children must cover every source event
    exactly once. Indices outside the source cluster, overlap, missing source
    events, the source label itself, and child labels already belonging to
    another cluster are rejected. Child labels are never invented.
    """
    values = _labels(labels)
    source = _int64_scalar(source_label, "source_label")
    source_indices = np.flatnonzero(values == source).astype(np.int64, copy=False)
    if source_indices.size == 0:
        raise ValidationError("source_label must exist in labels.")
    if not isinstance(child_assignments, Mapping):
        raise ValidationError("child_assignments must be a mapping of child labels to indices.")
    if len(child_assignments) < 2:
        raise ValidationError("child_assignments must contain at least two child labels.")

    assignments: dict[int, np.ndarray] = {}
    for raw_child, raw_indices in child_assignments.items():
        child = _int64_scalar(raw_child, "child label")
        if child == source:
            raise ValidationError("A child label must not equal source_label.")
        if child in assignments:
            raise ValidationError("child_assignments contains duplicate child labels.")
        indices = _event_indices(raw_indices, f"child_assignments[{child}]")
        if indices.size == 0:
            raise ValidationError("Every child assignment must contain at least one event.")
        if indices[-1] >= values.size:
            raise ValidationError("child assignment indices are outside the event axis.")
        if np.any(values[indices] != source):
            raise ValidationError("child assignments contain events foreign to source_label.")
        assignments[child] = indices

    other_labels = {int(label) for label in np.unique(values[values != source])}
    collisions = tuple(sorted(set(assignments) & other_labels))
    if collisions:
        raise ValidationError(
            f"child labels already belong to other clusters: {collisions!r}; merge separately."
        )
    combined = np.concatenate(tuple(assignments.values()))
    if np.unique(combined).size != combined.size:
        raise ValidationError("child assignments must not overlap.")
    if not np.array_equal(np.sort(combined), source_indices):
        raise ValidationError("child assignments must completely cover the source cluster.")

    child_labels = tuple(sorted(assignments))
    updated = values.copy()
    for child in child_labels:
        updated[assignments[child]] = child
    child_counts = tuple(int(assignments[child].size) for child in child_labels)
    operation = CurationOperationRecord(
        operation="split_cluster",
        source_labels=(source,),
        target_labels=child_labels,
        source_counts=(int(source_indices.size),),
        target_counts=child_counts,
        affected_event_indices=source_indices,
        target_event_indices=tuple(assignments[child] for child in child_labels),
        parameters={},
    )
    return CurationResult(updated_labels=updated, operations=(operation,))


def _rebuild_operation_record(
    operation: CurationOperation,
    source_labels: tuple[int, ...],
    target_labels: tuple[int, ...],
    source_counts: tuple[int, ...],
    target_counts: tuple[int, ...],
    affected_event_indices: np.ndarray,
    target_event_indices: tuple[np.ndarray, ...],
    parameters: Mapping[str, object],
) -> CurationOperationRecord:
    return CurationOperationRecord(
        operation=operation,
        source_labels=source_labels,
        target_labels=target_labels,
        source_counts=source_counts,
        target_counts=target_counts,
        affected_event_indices=affected_event_indices,
        target_event_indices=target_event_indices,
        parameters=parameters,
    )


def _rebuild_curation_result(
    updated_labels: np.ndarray,
    operations: tuple[CurationOperationRecord, ...],
) -> CurationResult:
    return CurationResult(updated_labels=updated_labels, operations=operations)


def _rebuild_waveform_outlier_result(
    scores: np.ndarray,
    thresholds: np.ndarray,
    rejection_mask: np.ndarray,
    updated_labels: np.ndarray,
    source_sample_indices: np.ndarray,
    operations: tuple[CurationOperationRecord, ...],
) -> WaveformOutlierResult:
    return WaveformOutlierResult(
        scores=scores,
        thresholds=thresholds,
        rejection_mask=rejection_mask,
        updated_labels=updated_labels,
        source_sample_indices=source_sample_indices,
        operations=operations,
    )


def _labels(value: np.ndarray, *, length: int | None = None) -> np.ndarray:
    return _int64_vector(value, "labels", length=length)


def _int64_vector(
    value: np.ndarray,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray.")
    if value.ndim != 1 or value.dtype != np.dtype(np.int64):
        raise ValidationError(f"{name} must be 1D int64.")
    if length is not None and value.size != length:
        raise ValidationError(f"{name} must have length {length}.")
    return value


def _float_vector(value: np.ndarray, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.dtype != np.dtype(np.float64) or arr.size != length:
        raise ValidationError(f"{name} must be 1D float64 with length {length}.")
    if np.any(np.isinf(arr)):
        raise ValidationError(f"{name} must not contain infinite values.")
    return arr


def _event_indices(value: np.ndarray, name: str) -> np.ndarray:
    indices = _int64_vector(value, name)
    if np.any(indices < 0):
        raise ValidationError(f"{name} must be non-negative.")
    if indices.size >= 2 and np.any(indices[1:] <= indices[:-1]):
        raise ValidationError(f"{name} must be strictly increasing without duplicates.")
    return indices


def _target_event_indices(
    value: object,
    *,
    target_labels: tuple[int, ...],
    target_counts: tuple[int, ...],
    affected: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if not isinstance(value, tuple) or len(value) != len(target_labels):
        raise ValidationError("target_event_indices must be a tuple aligned with target_labels.")
    indices_list: list[np.ndarray] = [
        _event_indices(item, "target_event_indices") for item in value
    ]
    for indices, count in zip(indices_list, target_counts, strict=True):
        if indices.size != count:
            raise ValidationError("target_event_indices sizes must align with target_counts.")
    combined = np.concatenate(indices_list)
    if np.unique(combined).size != combined.size:
        raise ValidationError("target_event_indices must not overlap.")
    if not np.array_equal(np.sort(combined), affected):
        raise ValidationError("target_event_indices must partition affected_event_indices.")
    return tuple(indices_list)


def _positive_integer(value: object, name: str) -> int:
    integer = _int64_scalar(value, name)
    if integer <= 0:
        raise ValidationError(f"{name} must be greater than zero.")
    return integer


def _label_sequence(
    value: Sequence[int] | np.ndarray,
    name: str,
    *,
    minimum_length: int,
) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        arr = _int64_vector(value, name)
        labels = tuple(int(item) for item in arr)
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{name} must be a sequence of integer labels.")
    else:
        labels = tuple(_int64_scalar(item, name) for item in value)
    if len(labels) < minimum_length:
        raise ValidationError(f"{name} must contain at least {minimum_length} label(s).")
    if len(set(labels)) != len(labels):
        raise ValidationError(f"{name} must not contain duplicate labels.")
    return labels


def _label_tuple(value: object, name: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValidationError(f"{name} must be a tuple.")
    labels = tuple(_int64_scalar(item, name) for item in value)
    if not allow_empty and not labels:
        raise ValidationError(f"{name} must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValidationError(f"{name} must not contain duplicate labels.")
    return labels


def _count_tuple(value: object, name: str, *, length: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValidationError(f"{name} must be a tuple of length {length}.")
    counts = tuple(_int64_scalar(item, name) for item in value)
    if any(count < 0 for count in counts):
        raise ValidationError(f"{name} must contain non-negative counts.")
    return counts


def _operations(
    value: object,
    n_events: int,
) -> tuple[CurationOperationRecord, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValidationError("operations must be a non-empty tuple.")
    for operation in value:
        if not isinstance(operation, CurationOperationRecord):
            raise ValidationError("operations must contain CurationOperationRecord values.")
        if (
            operation.affected_event_indices.size
            and operation.affected_event_indices[-1] >= n_events
        ):
            raise ValidationError("operation affected_event_indices exceed the event axis.")
    return value


def _affected_label_counts(
    labels: np.ndarray,
    affected: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if affected.size == 0:
        return (), ()
    source_labels, counts = np.unique(labels[affected], return_counts=True)
    return (
        tuple(int(label) for label in source_labels),
        tuple(int(count) for count in counts),
    )


def _shared_waveform_unit(batch: SpikeWaveformBatch) -> str:
    units = tuple(batch.channels.units)
    if not units or any(unit != units[0] for unit in units[1:]):
        raise ValidationError("waveform outlier rejection requires one shared channel unit.")
    return units[0]


__all__ = [
    "CurationOperationRecord",
    "CurationResult",
    "WaveformOutlierResult",
    "merge_clusters",
    "reject_waveform_outliers",
    "remove_tiny_clusters",
    "split_cluster",
]
