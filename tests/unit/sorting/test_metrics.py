#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

import copy
import pickle
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.sorting.metrics import (
    ClusterIsiMetrics,
    MatchingEventsResult,
    isi_metrics,
    matching_events,
)


def _int64(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def test_event_matching_returns_ordered_pairs_and_errors() -> None:
    result = matching_events(
        _int64([10, 10, 20, 30]),
        _int64([9, 10, 21, 50]),
        tol=1,
    )

    assert isinstance(result, MatchingEventsResult)
    assert result.n_matches == 3
    np.testing.assert_array_equal(result.matched_source_indices, [0, 1, 2])
    np.testing.assert_array_equal(result.matched_target_indices, [0, 1, 2])
    np.testing.assert_array_equal(result.unmatched_source_indices, [3])
    np.testing.assert_array_equal(result.unmatched_target_indices, [3])
    np.testing.assert_array_equal(result.sample_errors, [-1, 0, 1])


def test_event_matching_tie_uses_earliest_input() -> None:
    source = _int64([10, 10, 10])
    target = _int64([10, 10])

    first = matching_events(source, target)
    second = matching_events(source, target)

    np.testing.assert_array_equal(first.matched_source_indices, [0, 1])
    np.testing.assert_array_equal(first.matched_target_indices, [0, 1])
    np.testing.assert_array_equal(first.unmatched_source_indices, [2])
    np.testing.assert_array_equal(first.matched_source_indices, second.matched_source_indices)
    np.testing.assert_array_equal(first.matched_target_indices, second.matched_target_indices)


def test_matching_events_tol_is_inclusive() -> None:
    result = matching_events(_int64([10]), _int64([13]), tol=3)

    assert result.n_matches == 1
    np.testing.assert_array_equal(result.sample_errors, [3])


def test_event_matching_respects_segment_order() -> None:
    result = matching_events(
        _int64([10, 100, 10]),
        _int64([10, 10, 100]),
        source_segment_ids=_int64([0, 0, 1]),
        target_segment_ids=_int64([1, 0, 0]),
    )

    np.testing.assert_array_equal(result.matched_source_indices, [0, 1, 2])
    np.testing.assert_array_equal(result.matched_target_indices, [1, 2, 0])
    np.testing.assert_array_equal(result.sample_errors, [0, 0, 0])

    no_cross_segment_match = matching_events(
        _int64([10]),
        _int64([10]),
        source_segment_ids=_int64([0]),
        target_segment_ids=_int64([1]),
    )
    assert no_cross_segment_match.n_matches == 0
    np.testing.assert_array_equal(no_cross_segment_match.unmatched_source_indices, [0])
    np.testing.assert_array_equal(no_cross_segment_match.unmatched_target_indices, [0])


def test_event_matching_results_are_immutable() -> None:
    result = matching_events(_int64([]), _int64([]), tol=2)

    assert result.n_matches == 0
    for value in (
        result.matched_source_indices,
        result.matched_target_indices,
        result.unmatched_source_indices,
        result.unmatched_target_indices,
        result.sample_errors,
    ):
        assert value.dtype == np.int64
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True

    with pytest.raises(FrozenInstanceError):
        result.sample_errors = _int64([0])


@pytest.mark.parametrize(
    "clone",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda obj: pickle.loads(pickle.dumps(obj)), id="pickle"),
    ],
)
def test_event_matching_result_stays_immutable(
    clone: Callable[[MatchingEventsResult], MatchingEventsResult],
) -> None:
    # A frozen slots dataclass restored by the default ``__reduce_ex__`` would
    # bypass ``__post_init__`` and come back with writable arrays. Every clone
    # path must re-enter the constructor so strong immutability survives.
    result = matching_events(
        _int64([10, 10, 20, 30]),
        _int64([9, 10, 21, 50]),
        tol=1,
    )

    restored = clone(result)

    assert restored.n_matches == 3
    np.testing.assert_array_equal(restored.matched_source_indices, [0, 1, 2])
    np.testing.assert_array_equal(restored.sample_errors, [-1, 0, 1])
    for arr in (
        restored.matched_source_indices,
        restored.matched_target_indices,
        restored.unmatched_source_indices,
        restored.unmatched_target_indices,
        restored.sample_errors,
    ):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.flags.writeable = True


@pytest.mark.parametrize(
    ("source", "target", "source_segments", "target_segments", "message"),
    [
        (_int64([2, 1]), _int64([]), None, None, "nondecreasing"),
        (
            _int64([0, 1, 2]),
            _int64([]),
            _int64([0, 1, 0]),
            _int64([]),
            "contiguous run",
        ),
        (_int64([0]), _int64([0]), _int64([0]), None, "supplied together"),
    ],
)
def test_event_matching_rejects_invalid_order_or_segments(
    source: np.ndarray,
    target: np.ndarray,
    source_segments: np.ndarray | None,
    target_segments: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        matching_events(
            source,
            target,
            source_segment_ids=source_segments,
            target_segment_ids=target_segments,
        )


def test_event_matching_requires_int64_and_nonnegative_tol() -> None:
    with pytest.raises(ValidationError, match=r"numpy\.ndarray"):
        matching_events([1], _int64([1]))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="int64"):
        matching_events(np.asarray([1], dtype=np.int32), _int64([1]))
    with pytest.raises(ValidationError, match="non-negative"):
        matching_events(_int64([1]), _int64([1]), tol=-1)


def test_isi_metrics_are_per_cluster_in_label_order() -> None:
    results = isi_metrics(
        _int64([0, 2, 4, 0, 1, 4]),
        _int64([-9, -9, 3, -9, -9, 3]),
        fs=1_000.0,
        refractory_interval=0.0021,
        segment_ids=_int64([7, 7, 7, -2, -2, -2]),
    )

    assert tuple(result.label for result in results) == (-9, 3)
    negative, positive = results
    assert isinstance(negative, ClusterIsiMetrics)
    assert negative.n_spikes == 4
    assert negative.n_segments == 2
    assert negative.n_isis == 2
    np.testing.assert_array_equal(negative.isi_samples, [2, 1])
    assert negative.refractory_samples == 3
    assert negative.n_violations == 2
    assert negative.violation_fraction == 1.0
    assert negative.minimum_isi_samples == 1
    assert negative.maximum_isi_samples == 2
    assert negative.mean_isi_samples == 1.5
    assert negative.median_isi_samples == 1.5

    assert positive.n_spikes == 2
    assert positive.n_segments == 2
    assert positive.n_isis == 0
    assert positive.n_violations == 0
    assert positive.violation_fraction == 0.0
    assert positive.minimum_isi_samples is None
    assert positive.maximum_isi_samples is None
    assert positive.mean_isi_samples is None
    assert positive.median_isi_samples is None


def test_isi_metrics_handle_repeats_and_refractory_boundary() -> None:
    (result,) = isi_metrics(
        _int64([0, 0, 4, 0, 4]),
        _int64([5, 5, 5, 5, 5]),
        fs=1_000.0,
        refractory_interval=0.0031,
        segment_ids=_int64([0, 0, 0, 1, 1]),
    )

    assert result.refractory_samples == 4
    np.testing.assert_array_equal(result.isi_samples, [0, 4, 4])
    assert result.n_violations == 1
    assert result.violation_fraction == pytest.approx(1.0 / 3.0)


def test_isi_metrics_noise_label_is_excluded_only_when_explicit() -> None:
    samples = _int64([0, 1, 2, 3])
    labels = _int64([-2, -1, -2, -1])

    all_clusters = isi_metrics(
        samples,
        labels,
        fs=1_000.0,
        refractory_interval=0.001,
    )
    without_noise = isi_metrics(
        samples,
        labels,
        fs=1_000.0,
        refractory_interval=0.001,
        noise_label=-1,
    )

    assert tuple(result.label for result in all_clusters) == (-2, -1)
    assert tuple(result.label for result in without_noise) == (-2,)


def test_isi_metrics_empty_and_single_spike_degenerate_inputs() -> None:
    assert (
        isi_metrics(
            _int64([]),
            _int64([]),
            fs=1_000.0,
            refractory_interval=0.001,
        )
        == ()
    )

    (single,) = isi_metrics(
        _int64([7]),
        _int64([101]),
        fs=1_000.0,
        refractory_interval=0.001,
    )
    assert single.n_spikes == 1
    assert single.n_segments == 1
    assert single.n_isis == 0


def test_isi_metrics_are_strongly_immutable() -> None:
    (result,) = isi_metrics(
        _int64([0, 2]),
        _int64([4, 4]),
        fs=1_000.0,
        refractory_interval=0.001,
    )

    assert not result.isi_samples.flags.writeable
    with pytest.raises(ValueError):
        result.isi_samples.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        result.n_spikes = 0


@pytest.mark.parametrize(
    "clone",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda obj: pickle.loads(pickle.dumps(obj)), id="pickle"),
    ],
)
def test_cluster_isi_metrics_stay_immutable(
    clone: Callable[[ClusterIsiMetrics], ClusterIsiMetrics],
) -> None:
    # The derived ISI summaries must be recomputed by ``__post_init__`` on
    # rebuild rather than deserialized from a stale value, so a round-trip can
    # never leave the object internally inconsistent. ``isi_samples=[0, 4, 4]``
    # with ``refractory_samples=4`` yields exactly one violation.
    (result,) = isi_metrics(
        _int64([0, 0, 4, 0, 4]),
        _int64([5, 5, 5, 5, 5]),
        fs=1_000.0,
        refractory_interval=0.0031,
        segment_ids=_int64([0, 0, 0, 1, 1]),
    )

    restored = clone(result)

    assert restored.label == 5
    assert restored.n_spikes == 5
    assert restored.n_segments == 2
    assert restored.refractory_samples == 4
    np.testing.assert_array_equal(restored.isi_samples, [0, 4, 4])
    assert not restored.isi_samples.flags.writeable
    with pytest.raises(ValueError):
        restored.isi_samples.flags.writeable = True

    # Derived summaries are recomputed and consistent with ``isi_samples``.
    assert restored.n_isis == 3
    assert restored.n_violations == 1
    assert restored.violation_fraction == pytest.approx(1.0 / 3.0)
    assert restored.minimum_isi_samples == 0
    assert restored.maximum_isi_samples == 4
    assert restored.mean_isi_samples == pytest.approx(8.0 / 3.0)
    assert restored.median_isi_samples == 4.0


@pytest.mark.parametrize(
    ("samples", "labels", "segments", "message"),
    [
        (_int64([1, 0]), _int64([0, 0]), None, "nondecreasing"),
        (_int64([0, 1]), _int64([0]), None, "length 2"),
        (_int64([0, 1, 2]), _int64([0, 0, 0]), _int64([0, 1, 0]), "contiguous run"),
    ],
)
def test_isi_metrics_rejects_invalid_order_and_shapes(
    samples: np.ndarray,
    labels: np.ndarray,
    segments: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        isi_metrics(
            samples,
            labels,
            fs=1_000.0,
            refractory_interval=0.001,
            segment_ids=segments,
        )


@pytest.mark.parametrize(
    ("fs", "refractory_interval", "message"),
    [
        (0.0, 0.001, "greater than zero"),
        (1_000.0, -0.001, "non-negative"),
        (float("inf"), 0.001, "finite"),
    ],
)
def test_isi_metrics_rejects_invalid_refractory_configuration(
    fs: float,
    refractory_interval: float,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        isi_metrics(
            _int64([0]),
            _int64([0]),
            fs=fs,
            refractory_interval=refractory_interval,
        )


def test_isi_metrics_reject_unrepresentable_interval() -> None:
    with pytest.raises(ValidationError, match="ISIs must fit in int64"):
        isi_metrics(
            _int64([np.iinfo(np.int64).min, np.iinfo(np.int64).max]),
            _int64([0, 0]),
            fs=1_000.0,
            refractory_interval=0.001,
        )
