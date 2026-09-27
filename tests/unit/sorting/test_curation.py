#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import pickle
from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, SpikeWaveformBatch
from neurale.exceptions import ValidationError
from neurale.sorting import curation
from neurale.sorting.curation import (
    CurationOperationRecord,
    CurationResult,
    WaveformOutlierResult,
    merge_clusters,
    reject_waveform_outliers,
    remove_tiny_clusters,
    split_cluster,
)


def _labels(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _batch(waveforms: np.ndarray, *, units: str | Sequence[str] = "uV") -> SpikeWaveformBatch:
    n_spikes, n_samples, n_channels = waveforms.shape
    unit_values = [units] * n_channels if isinstance(units, str) else list(units)
    if len(unit_values) != n_channels:
        raise ValueError("units must provide one entry per channel.")
    sample_indices = 100 + np.arange(n_spikes, dtype=np.int64)
    return SpikeWaveformBatch(
        waveforms=waveforms,
        sample_indices=sample_indices,
        times=1.0 + np.arange(n_spikes, dtype=np.float64) / 1_000.0,
        peak_channel_indices=np.zeros(n_spikes, dtype=np.int64),
        electrode_group_ids=np.zeros(n_spikes, dtype=np.int64),
        amps=np.asarray(waveforms[:, n_samples // 2, 0], dtype=np.float64),
        channels=ChannelTable(
            tuple(
                ChannelInfo(f"ch{idx}", idx, "spike", unit_values[idx]) for idx in range(n_channels)
            )
        ),
        fs=1_000.0,
        clock=None,
        source_stream="wideband",
        segment_id=4,
        pre_samples=n_samples // 2,
        post_samples=n_samples - n_samples // 2 - 1,
        polarity="negative",
    )


def test_remove_tiny_clusters_keeps_boundary_and_audits() -> None:
    labels = _labels([-7, -7, 42, 42, 42, 100, -99])
    original = labels.copy()

    result = remove_tiny_clusters(labels, min_cluster_size=2, noise_label=-99)

    assert isinstance(result, CurationResult)
    np.testing.assert_array_equal(result.updated_labels, [-7, -7, 42, 42, 42, -99, -99])
    np.testing.assert_array_equal(labels, original)
    (record,) = result.operations
    assert record.operation == "remove_tiny_clusters"
    assert record.source_labels == (100,)
    assert record.source_counts == (1,)
    assert record.target_labels == (-99,)
    assert record.target_counts == (1,)
    np.testing.assert_array_equal(record.affected_event_indices, [5])
    assert record.parameters["min_cluster_size"] == 2


def test_remove_tiny_clusters_noop_never_filters_noise() -> None:
    labels = _labels([-5, -5, 11, 11, -1])
    first = remove_tiny_clusters(labels, min_cluster_size=2, noise_label=-1)
    second = remove_tiny_clusters(first.updated_labels, min_cluster_size=2, noise_label=-1)

    np.testing.assert_array_equal(first.updated_labels, labels)
    np.testing.assert_array_equal(second.updated_labels, labels)
    assert first.operations[0].source_labels == ()
    assert first.operations[0].affected_count == 0
    assert second.operations[0].affected_count == 0


def test_waveform_outliers_use_median_residual_per_cluster() -> None:
    waveforms = np.zeros((6, 3, 1), dtype=np.float64)
    waveforms[3, :, 0] = 10.0
    waveforms[4, :, 0] = 2.0
    waveforms[5, :, 0] = 100.0
    batch = _batch(waveforms)
    labels = _labels([7, 7, 7, 7, 50, -1])
    labels_before = labels.copy()
    waveforms_before = batch.waveforms.copy()

    result = reject_waveform_outliers(
        batch,
        labels,
        threshold_multiplier=3.0,
        minimum_cluster_size=4,
        noise_label=-1,
    )

    assert isinstance(result, WaveformOutlierResult)
    np.testing.assert_allclose(result.scores[:4], [0.0, 0.0, 0.0, np.sqrt(300.0)])
    np.testing.assert_array_equal(result.thresholds[:4], 0.0)
    assert result.scores[4] == 0.0
    assert np.isnan(result.thresholds[4])
    assert np.isnan(result.scores[5])
    assert np.isnan(result.thresholds[5])
    np.testing.assert_array_equal(result.rejection_mask, [False, False, False, True, False, False])
    np.testing.assert_array_equal(result.updated_labels, [7, 7, 7, -1, 50, -1])
    np.testing.assert_array_equal(result.source_sample_indices, batch.sample_indices)
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(batch.waveforms, waveforms_before)

    (record,) = result.operations
    assert record.source_labels == (7,)
    assert record.source_counts == (1,)
    assert record.target_labels == (-1,)
    np.testing.assert_array_equal(record.affected_event_indices, [3])
    assert record.parameters["mad_scale"] == 1.0
    assert record.parameters["centered_mads"] == (0.0,)
    assert record.parameters["waveform_unit"] == "uV"


def test_waveform_outlier_threshold_is_strict_and_stable() -> None:
    batch = _batch(np.asarray([[[0.0]], [[2.0]]], dtype=np.float64))
    labels = _labels([8, 8])

    first = reject_waveform_outliers(
        batch,
        labels,
        threshold_multiplier=0.0,
        minimum_cluster_size=2,
        noise_label=-1,
    )
    second = reject_waveform_outliers(
        batch,
        first.updated_labels,
        threshold_multiplier=0.0,
        minimum_cluster_size=2,
        noise_label=-1,
    )

    np.testing.assert_allclose(first.scores, [1.0, 1.0])
    np.testing.assert_allclose(first.thresholds, [1.0, 1.0])
    assert not np.any(first.rejection_mask)
    np.testing.assert_array_equal(second.updated_labels, first.updated_labels)
    assert second.operations[0].affected_count == 0


def test_waveform_outliers_handle_noncontiguous_labels() -> None:
    waveforms = np.asarray([[[0.0]], [[0.0]], [[9.0]], [[4.0]], [[4.0]], [[4.0]]])
    result = reject_waveform_outliers(
        _batch(waveforms),
        _labels([-10, -10, -10, 250, 250, 250]),
        threshold_multiplier=2.0,
        minimum_cluster_size=3,
        noise_label=999,
    )

    np.testing.assert_array_equal(result.updated_labels, [-10, -10, 999, 250, 250, 250])
    assert result.operations[0].source_labels == (-10,)


def test_waveform_outliers_reject_mixed_channel_units() -> None:
    # The Euclidean residual norm folds every channel into one sum, so mixing
    # amplitude units (uV alongside mV) would let the unit representation alone
    # swing the residual and the rejection decision. The batch itself accepts
    # per-channel units; the outlier operation must refuse to score it.
    waveforms = np.zeros((4, 2, 2), dtype=np.float64)
    batch = _batch(waveforms, units=("uV", "mV"))

    with pytest.raises(ValidationError, match="shared channel unit"):
        reject_waveform_outliers(
            batch,
            _labels([7, 7, 7, 7]),
            threshold_multiplier=3.0,
            minimum_cluster_size=2,
            noise_label=-1,
        )


def test_merge_clusters_preserves_order_without_renumbering() -> None:
    labels = _labels([-10, 4, 99, -10, 4, 7])
    original = labels.copy()

    result = merge_clusters(labels, source_labels=_labels([99, -10]), target_label=4)

    np.testing.assert_array_equal(result.updated_labels, [4, 4, 4, 4, 4, 7])
    np.testing.assert_array_equal(labels, original)
    (record,) = result.operations
    assert record.source_labels == (-10, 99)
    assert record.source_counts == (2, 1)
    assert record.target_labels == (4,)
    assert record.target_counts == (3,)
    np.testing.assert_array_equal(record.affected_event_indices, [0, 2, 3])


@pytest.mark.parametrize(
    ("sources", "target", "message"),
    [
        ((), 8, "at least 1"),
        ((2, 2), 8, "duplicate"),
        ((2,), 2, "target_label"),
        ((100,), 8, "absent"),
    ],
)
def test_merge_clusters_rejects_ambiguous_or_foreign_sources(
    sources: tuple[int, ...],
    target: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        merge_clusters(_labels([2, 8]), source_labels=sources, target_label=target)


def test_split_cluster_requires_complete_children() -> None:
    labels = _labels([-10, 5, -10, 8, -10])
    original = labels.copy()

    result = split_cluster(
        labels,
        source_label=-10,
        child_assignments={100: _labels([0, 4]), -7: _labels([2])},
    )

    np.testing.assert_array_equal(result.updated_labels, [100, 5, -7, 8, 100])
    np.testing.assert_array_equal(labels, original)
    (record,) = result.operations
    assert record.source_labels == (-10,)
    assert record.source_counts == (3,)
    assert record.target_labels == (-7, 100)
    assert record.target_counts == (1, 2)
    np.testing.assert_array_equal(record.affected_event_indices, [0, 2, 4])
    # target_event_indices aligns with target_labels and partitions the affected
    # set, naming exactly which events landed on each child.
    assert len(record.target_event_indices) == 2
    np.testing.assert_array_equal(record.target_event_indices[0], [2])
    np.testing.assert_array_equal(record.target_event_indices[1], [0, 4])


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        ({10: _labels([0, 2])}, "at least two"),
        ({10: _labels([0]), 11: _labels([2])}, "completely cover"),
        ({10: _labels([0, 2]), 11: _labels([2, 4])}, "overlap"),
        ({10: _labels([0, 1]), 11: _labels([2, 4])}, "foreign"),
        ({-3: _labels([0, 2]), 11: _labels([4])}, "must not equal"),
        ({5: _labels([0, 2]), 11: _labels([4])}, "other clusters"),
    ],
)
def test_split_cluster_rejects_invalid_assignments(
    assignments: dict[int, np.ndarray],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        split_cluster(
            _labels([-3, 5, -3, 8, -3]),
            source_label=-3,
            child_assignments=assignments,
        )


def test_split_record_replays_from_target_indices() -> None:
    # The record alone — without the original child_assignments mapping — must
    # carry enough information to reproduce the relabeling. target_event_indices
    # names which original events each target label received, so replaying it
    # against a fresh copy of the input reconstructs the updated labels.
    labels = _labels([-10, 5, -10, 8, -10])
    result = split_cluster(
        labels,
        source_label=-10,
        child_assignments={100: _labels([0, 4]), -7: _labels([2])},
    )
    (record,) = result.operations

    replayed = labels.copy()
    for child, indices in zip(record.target_labels, record.target_event_indices, strict=True):
        replayed[np.asarray(indices, dtype=np.int64)] = child

    np.testing.assert_array_equal(replayed, result.updated_labels)


def test_split_records_distinguish_different_child_assignments() -> None:
    # Two splits of the same source cluster that send different events to the
    # same child labels produce identical source/target labels, counts, and
    # affected indices. Before target_event_indices their audit records were
    # indistinguishable; the per-target indices now make them distinct and
    # each record replays to its own outcome.
    labels = _labels([5, 5, 5, 9])

    result_a = split_cluster(
        labels, source_label=5, child_assignments={10: _labels([0, 1]), 11: _labels([2])}
    )
    result_b = split_cluster(
        labels, source_label=5, child_assignments={10: _labels([0, 2]), 11: _labels([1])}
    )
    (record_a,) = result_a.operations
    (record_b,) = result_b.operations

    np.testing.assert_array_equal(result_a.updated_labels, [10, 10, 11, 9])
    np.testing.assert_array_equal(result_b.updated_labels, [10, 11, 10, 9])

    # Identical on every pre-existing field...
    assert record_a.target_labels == record_b.target_labels
    assert record_a.target_counts == record_b.target_counts
    np.testing.assert_array_equal(record_a.affected_event_indices, record_b.affected_event_indices)
    # ...but the per-target index tuples differ and name the true assignment.
    np.testing.assert_array_equal(record_a.target_event_indices[0], [0, 1])
    np.testing.assert_array_equal(record_b.target_event_indices[0], [0, 2])


@pytest.mark.parametrize(
    ("target_event_indices", "message"),
    [
        ((_labels([0, 1, 2]),), "aligned with target_labels"),  # wrong arity for 2 targets
        ((_labels([0, 1]), _labels([2, 3])), "align with target_counts"),  # size mismatch
        ((_labels([0, 2]), _labels([2])), "must not overlap"),  # overlapping
        (
            (_labels([0, 3]), _labels([1])),
            "must partition affected_event_indices",
        ),  # not a partition
    ],
)
def test_operation_record_requires_partitioning_indices(
    target_event_indices: tuple[np.ndarray, ...],
    message: str,
) -> None:
    # affected = [0, 1, 2]; target_labels=(10, 11), target_counts=(2, 1).
    with pytest.raises(ValidationError, match=message):
        CurationOperationRecord(
            operation="split_cluster",
            source_labels=(5,),
            target_labels=(10, 11),
            source_counts=(3,),
            target_counts=(2, 1),
            affected_event_indices=_labels([0, 1, 2]),
            target_event_indices=target_event_indices,
            parameters={},
        )


@pytest.mark.parametrize(
    "clone", [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_curation_records_and_results_remain_strongly_immutable(clone) -> None:
    result = remove_tiny_clusters(_labels([1, 2, 2]), min_cluster_size=2, noise_label=-1)
    cloned = clone(result)
    record = cloned.operations[0]

    assert isinstance(record, CurationOperationRecord)
    assert not cloned.updated_labels.flags.writeable
    assert not record.affected_event_indices.flags.writeable
    for indices in record.target_event_indices:
        assert not indices.flags.writeable
    with pytest.raises(ValueError):
        cloned.updated_labels.flags.writeable = True
    with pytest.raises(ValueError):
        record.affected_event_indices.flags.writeable = True
    for indices in record.target_event_indices:
        with pytest.raises(ValueError):
            indices.flags.writeable = True
    with pytest.raises(TypeError):
        record.parameters["new"] = 1
    with pytest.raises(FrozenInstanceError):
        record.operation = "merge_clusters"


@pytest.mark.parametrize(
    "clone", [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_waveform_outlier_result_remains_strongly_immutable(clone) -> None:
    result = reject_waveform_outliers(
        _batch(np.zeros((2, 1, 1), dtype=np.float64)),
        _labels([4, 4]),
        threshold_multiplier=3.0,
        minimum_cluster_size=2,
        noise_label=-1,
    )
    cloned = clone(result)

    for value in (
        cloned.scores,
        cloned.thresholds,
        cloned.rejection_mask,
        cloned.updated_labels,
        cloned.source_sample_indices,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True


def test_curation_inputs_require_int64_and_explicit_noise() -> None:
    with pytest.raises(ValidationError, match="int64"):
        remove_tiny_clusters(
            np.asarray([1, 1], dtype=np.int32),
            min_cluster_size=2,
            noise_label=-1,
        )
    with pytest.raises(TypeError):
        remove_tiny_clusters(_labels([1]), min_cluster_size=2)  # type: ignore[call-arg]


def test_waveform_outlier_scores_ignore_block_size() -> None:
    """Blocking must not change a single bit of the scores.

    ``_residual_distances`` blocks over spikes for the norm precisely so each
    spike's reduction stays one uninterrupted pass over its complete residual
    row. Forcing a tiny block exercises the multi-block path; the result must
    be bit-identical to computing the whole cluster at once, because a few
    units in the last place can flip a spike sitting exactly on the threshold.
    """
    rng = np.random.default_rng(11)
    waveforms = rng.standard_normal((64, 12, 3))
    batch = _batch(waveforms)
    labels = np.zeros(64, dtype=np.int64)

    reference = reject_waveform_outliers(
        batch, labels, threshold_multiplier=3.0, minimum_cluster_size=2, noise_label=-1
    )

    original = curation._OUTLIER_BLOCK_BYTES
    try:
        # 64 bytes forces both passes into many blocks.
        curation._OUTLIER_BLOCK_BYTES = 64
        blocked = reject_waveform_outliers(
            batch, labels, threshold_multiplier=3.0, minimum_cluster_size=2, noise_label=-1
        )
    finally:
        curation._OUTLIER_BLOCK_BYTES = original

    assert np.array_equal(reference.scores.view(np.uint64), blocked.scores.view(np.uint64)), (
        "blocked residual norms are not bit-identical"
    )
    assert np.array_equal(reference.thresholds, blocked.thresholds, equal_nan=True)
    assert np.array_equal(reference.rejection_mask, blocked.rejection_mask)
    assert np.array_equal(reference.updated_labels, blocked.updated_labels)


def test_waveform_outlier_temporaries_stay_bounded() -> None:
    """Peak temporary memory must not scale with the cluster size.

    The whole-cluster implementation this replaced held three cluster-sized
    float64 temporaries at once, so a single-cluster batch peaked at roughly
    three times the waveform payload.
    """
    import tracemalloc

    # The payload must exceed the block ceiling for blocking to engage at all:
    # 20000 x 32 x 4 float64 is ~19.5 MiB against a ~4 MiB block.
    rng = np.random.default_rng(5)
    waveforms = rng.standard_normal((20_000, 32, 4))
    batch = _batch(waveforms)
    labels = np.zeros(20_000, dtype=np.int64)
    payload = waveforms.nbytes
    assert payload > 4 * curation._OUTLIER_BLOCK_BYTES

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    reject_waveform_outliers(
        batch, labels, threshold_multiplier=3.0, minimum_cluster_size=2, noise_label=-1
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Allow block overhead without allocating another full payload.
    assert peak - before < payload // 2, (
        f"peak temporary {(peak - before) / 2**20:.1f} MiB is not bounded well "
        f"below the {payload / 2**20:.1f} MiB payload"
    )
