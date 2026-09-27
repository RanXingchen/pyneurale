#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale._native_loader import load_native_namespace
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.models import DTWResult, dtw


def _native_models():
    try:
        return load_native_namespace("models")
    except NativeUnavailableError:
        return None


def test_dtw_returns_typed_result_for_1d_sequences() -> None:
    result = dtw(
        np.array([0.0, 1.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
    )

    assert isinstance(result, DTWResult)
    assert result.cost == 0.0
    assert np.array_equal(result.path_x, np.array([0, 1, 2, 3]))
    assert np.array_equal(result.path_y, np.array([0, 1, 1, 2]))


def test_dtw_finite_radius_uses_non_negative_window() -> None:
    result = dtw(
        np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        radius=1,
    )

    assert result.cost == 0.0
    assert np.array_equal(result.path_x, np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(result.path_y, np.array([0, 0, 1, 2, 2]))


def test_dtw_supports_2d_sequences_and_tuple_unpacking() -> None:
    result = dtw(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )

    cost, path_x, path_y = result
    assert cost == result.cost
    assert np.array_equal(path_x, result.path_x)
    assert np.array_equal(path_y, result.path_y)
    assert result.cost == 2.0
    assert np.array_equal(result.path_x, np.array([0, 1, 2]))
    assert np.array_equal(result.path_y, np.array([0, 0, 1]))


def test_dtw_sqeuclidean_cost_accumulates_squared_local_distances() -> None:
    result = dtw(
        np.array([0.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        metric="sqeuclidean",
    )

    assert result.cost == 1.0


def test_dtw_full_radius_matches_unconstrained_result() -> None:
    x = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])

    unconstrained = dtw(x, y)
    full_window = dtw(x, y, radius=max(len(x), len(y)))

    assert full_window.cost == unconstrained.cost
    assert np.array_equal(full_window.path_x, unconstrained.path_x)
    assert np.array_equal(full_window.path_y, unconstrained.path_y)


def test_dtw_is_symmetric_when_inputs_are_swapped() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 1.0]])

    forward = dtw(x, y, radius=1)
    reverse = dtw(y, x, radius=1)

    assert reverse.cost == forward.cost
    assert np.array_equal(reverse.path_x, forward.path_y)
    assert np.array_equal(reverse.path_y, forward.path_x)


def test_dtw_uses_double_accumulation_for_long_paths() -> None:
    x = np.full(100_000, 0.1, dtype=np.float64)
    y = np.array([0.0], dtype=np.float64)

    unconstrained = dtw(x, y, metric="sqeuclidean")
    full_window = dtw(x, y, radius=100_000, metric="sqeuclidean")

    assert full_window.cost == unconstrained.cost
    assert np.isclose(unconstrained.cost, 1000.0)


def test_dtw_handles_large_squared_distance() -> None:
    result = dtw(
        np.array([1e20], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        metric="sqeuclidean",
    )

    assert result.cost == 1e40


def test_dtw_rejects_unreachable_radius() -> None:
    with pytest.raises(ValidationError, match="radius"):
        dtw(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0]), radius=0)


def test_dtw_radius_window_uses_exact_integer_bounds() -> None:
    result = dtw(
        np.array([0, -5, 1, -4, 3, 2, 5, 3, -1, 5], dtype=float),
        np.array([-1, -2, 4, -1, -5, 0, 3, -3], dtype=float),
        radius=1,
    )

    assert result.cost == 36.0
    assert np.array_equal(
        result.path_y,
        np.array([0, 1, 2, 3, 3, 4, 5, 6, 7, 7]),
    )


def test_dtw_accepts_huge_radius_as_full_window() -> None:
    result = dtw(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0]),
        radius=10**100,
    )

    assert result.cost == 1.0


def test_dtw_propagates_numeric_overflow() -> None:
    finfo = np.finfo(np.float64)

    with pytest.raises(OverflowError):
        dtw(np.array([-finfo.max]), np.array([finfo.max]))


def test_dtw_euclidean_keeps_tiny_nonzero_distance() -> None:
    value = 1e-200

    result = dtw(
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[value, -value]], dtype=np.float64),
    )

    assert result.cost > 0.0
    assert result.cost == pytest.approx(np.hypot(value, value), rel=1e-15)


@pytest.mark.parametrize("value", [1e200, np.finfo(np.float64).max])
def test_dtw_euclidean_preserves_large_representable_distance(
    value: float,
) -> None:
    result = dtw(
        np.array([value], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )

    assert result.cost == pytest.approx(value)


def test_dtw_accepts_non_contiguous_and_integer_inputs() -> None:
    x = np.arange(10, dtype=np.int32)[::2]
    y = np.asfortranarray(np.array([[0.0], [4.0], [8.0]], dtype=np.float32))

    result = dtw(x, y, metric="sqeuclidean")

    assert result.path_x[0] == 0
    assert result.path_y[0] == 0
    assert result.path_x[-1] == len(x) - 1
    assert result.path_y[-1] == len(y) - 1


def test_dtw_rejects_negative_radius_and_dimension_mismatch() -> None:
    with pytest.raises(ValidationError):
        dtw(np.array([0.0, 1.0]), np.array([0.0, 1.0]), radius=-1)
    with pytest.raises(ValidationError):
        dtw(np.ones((2, 2)), np.ones((2, 3)))
