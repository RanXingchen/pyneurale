#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.sorting._valley_seeking_reference import valley_seeking_reference


def test_radius_boundary_is_inclusive_and_self_is_excluded() -> None:
    result = valley_seeking_reference(
        np.array([[0.0], [1.0], [3.0]], dtype=np.float64),
        np.array([4, 4, 9], dtype=np.int64),
        radius=1.0,
    )

    np.testing.assert_array_equal(result.neighbor_counts, [1, 1, 0])
    np.testing.assert_array_equal(result.labels, [4, 4, 9])
    assert result.label_order == (4, 9)
    assert result.converged
    assert result.n_iterations == 1


def test_radius_boundary_is_euclidean_in_multiple_dimensions() -> None:
    # Multi-dimensional geometric baseline for the native kernel: the
    # fixed-radius ball is Euclidean, not per-axis (Chebyshev). Point 1 sits
    # exactly on the unit sphere around point 0 (inclusive boundary), so
    # 0 <-> 1. Point 2 is within 1 on every axis yet outside the Euclidean
    # ball, so 0 <-/-> 2. Point 2 remains within radius of point 1
    # (distance 0.2), giving the full neighbor_counts = [1, 2, 1].
    result = valley_seeking_reference(
        np.array(
            [
                [0.0, 0.0],
                [0.6, 0.8],  # distance to 0 == 1.0 exactly -> neighbor
                [0.8, 0.8],  # per-axis <= 1 but distance > 1 -> not a neighbor of 0
            ],
            dtype=np.float64,
        ),
        np.array([4, 4, 9], dtype=np.int64),
        radius=1.0,
    )

    np.testing.assert_array_equal(result.neighbor_counts, [1, 2, 1])
    np.testing.assert_array_equal(result.labels, [4, 4, 4])
    assert result.converged
    assert result.n_iterations == 2


def test_radius_boundary_uses_frozen_fused_multiply_add_order() -> None:
    result = valley_seeking_reference(
        np.array(
            [
                [0.0, 0.0],
                [0.68418360801139777, 0.72930980421800595],
            ],
            dtype=np.float64,
        ),
        np.array([0, 1], dtype=np.int64),
        radius=1.0,
        max_iterations=1,
    )

    np.testing.assert_array_equal(result.neighbor_counts, [0, 0])
    np.testing.assert_array_equal(result.labels, [0, 1])


def test_incumbent_wins_vote_tie() -> None:
    result = valley_seeking_reference(
        np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([7, 3, 7], dtype=np.int64),
        radius=1.0,
        max_iterations=1,
    )

    assert result.labels[0] == 7


def test_lowest_label_wins_tie_not_involving_incumbent() -> None:
    result = valley_seeking_reference(
        np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([8, 5, 2], dtype=np.int64),
        radius=1.0,
        max_iterations=1,
    )

    np.testing.assert_array_equal(result.labels, [2, 8, 8])
    assert result.label_order == (2, 5, 8)


def test_updates_are_synchronous_and_iteration_limit_is_explicit() -> None:
    features = np.array([[0.0], [0.5]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)

    first = valley_seeking_reference(features, labels, radius=1.0, max_iterations=1)
    second = valley_seeking_reference(features, labels, radius=1.0, max_iterations=2)

    np.testing.assert_array_equal(first.labels, [1, 0])
    np.testing.assert_array_equal(second.labels, labels)
    assert first.termination == second.termination == "max_iterations"
    assert not first.converged and not second.converged


def test_majority_change_converges_next_pass() -> None:
    result = valley_seeking_reference(
        np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.25]], dtype=np.float64),
        np.array([1, 1, 2], dtype=np.int64),
        radius=1.0,
    )

    np.testing.assert_array_equal(result.labels, [1, 1, 1])
    assert result.converged
    assert result.termination == "converged"
    assert result.n_iterations == 2


def test_empty_isolated_and_duplicate_observations_are_defined() -> None:
    empty = valley_seeking_reference(
        np.empty((0, 2), dtype=np.float64),
        np.empty(0, dtype=np.int64),
        radius=1.0,
    )
    assert empty.termination == "empty"
    assert empty.n_iterations == 0

    degenerate = valley_seeking_reference(
        np.array([[0.0], [0.0], [10.0]], dtype=np.float64),
        np.array([-1, -1, 4], dtype=np.int64),
        radius=0.5,
    )
    np.testing.assert_array_equal(degenerate.neighbor_counts, [1, 1, 0])
    np.testing.assert_array_equal(degenerate.labels, [-1, -1, 4])


@pytest.mark.parametrize(
    ("features", "labels", "radius", "max_iterations", "message"),
    [
        (np.ones(3), np.array([0, 0, 0], dtype=np.int64), 1.0, 2, "shape"),
        (np.empty((2, 0)), np.array([0, 0], dtype=np.int64), 1.0, 2, "feature column"),
        (np.ones((2, 1), dtype=np.float32), np.array([0, 0], dtype=np.int64), 1.0, 2, "float64"),
        (np.array([[0.0], [np.nan]]), np.array([0, 0], dtype=np.int64), 1.0, 2, "finite"),
        (np.ones((2, 1)), np.array([0], dtype=np.int64), 1.0, 2, "one label"),
        (np.ones((2, 1)), np.array([0, 1], dtype=np.int32), 1.0, 2, "int64"),
        (np.ones((2, 1)), np.array([0, 1], dtype=np.int64), 0.0, 2, "positive real"),
        (np.ones((2, 1)), np.array([0, 1], dtype=np.int64), 1.0, 0, "positive integer"),
    ],
)
def test_invalid_contract_inputs_are_rejected(
    features: np.ndarray,
    labels: np.ndarray,
    radius: float,
    max_iterations: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        valley_seeking_reference(
            features,
            labels,
            radius=radius,
            max_iterations=max_iterations,
        )


def test_reference_is_deterministic_and_result_arrays_are_immutable() -> None:
    features = np.array([[0.0], [0.2], [3.0]], dtype=np.float64)
    labels = np.array([11, 4, 9], dtype=np.int64)

    left = valley_seeking_reference(features, labels, radius=0.5, max_iterations=5)
    right = valley_seeking_reference(features, labels, radius=0.5, max_iterations=5)

    np.testing.assert_array_equal(left.labels, right.labels)
    np.testing.assert_array_equal(left.neighbor_counts, right.neighbor_counts)
    with pytest.raises(ValueError, match="read-only"):
        left.labels[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        left.neighbor_counts[0] = 99


def test_reference_is_not_public_export() -> None:
    import neurale.sorting as sorting

    assert not hasattr(sorting, "valley_seeking_reference")
    assert "valley_seeking_reference" not in sorting.__all__
