#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from neurale.runtime import runtime_context
from neurale.sorting import _dispatch as dispatch_module
from neurale.sorting._dispatch import load_cpu_sorting_operation
from neurale.sorting._valley_seeking_reference import valley_seeking_reference

pytestmark = pytest.mark.native


def _native(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    radius: float,
    max_iterations: int,
) -> dict[str, object]:
    return load_cpu_sorting_operation("valley_seeking")._valley_seeking(
        features,
        labels,
        radius,
        max_iterations,
    )


@pytest.mark.parametrize("seed", range(24))
def test_native_randomized_parity_with_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_observations = int(rng.integers(0, 33))
    n_features = int(rng.integers(1, 7))
    features = np.ascontiguousarray(
        rng.normal(size=(n_observations, n_features)),
        dtype=np.float64,
    )
    if n_observations >= 2 and seed % 4 == 0:
        features[1] = features[0]
    labels = np.ascontiguousarray(
        rng.choice(np.array([-5, 2, 11, 40], dtype=np.int64), size=n_observations),
        dtype=np.int64,
    )
    radius = float(rng.uniform(0.2, 2.5))
    max_iterations = int(rng.integers(1, 21))

    expected = valley_seeking_reference(
        features,
        labels,
        radius=radius,
        max_iterations=max_iterations,
    )
    actual = _native(
        features,
        labels,
        radius=radius,
        max_iterations=max_iterations,
    )

    np.testing.assert_array_equal(actual["labels"], expected.labels)
    np.testing.assert_array_equal(actual["neighbor_counts"], expected.neighbor_counts)
    np.testing.assert_array_equal(actual["label_order"], expected.label_order)
    assert actual["radius"] == expected.radius
    assert actual["n_iterations"] == expected.n_iterations
    assert actual["converged"] is expected.converged
    assert actual["termination"] == expected.termination
    assert actual["neighbor_pairs"] == int(np.sum(expected.neighbor_counts) // 2)
    assert actual["workspace_bytes"] >= 0


def test_native_matches_reference_on_radius_tie_and_cycle_boundaries() -> None:
    cases = (
        (
            np.array([[0.0], [1.0], [3.0]], dtype=np.float64),
            np.array([4, 4, 9], dtype=np.int64),
            1.0,
            10,
        ),
        (
            np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            np.array([8, 5, 2], dtype=np.int64),
            1.0,
            1,
        ),
        (
            np.array([[0.0], [0.5]], dtype=np.float64),
            np.array([0, 1], dtype=np.int64),
            1.0,
            6,
        ),
    )
    for features, labels, radius, max_iterations in cases:
        expected = valley_seeking_reference(
            features,
            labels,
            radius=radius,
            max_iterations=max_iterations,
        )
        actual = _native(
            features,
            labels,
            radius=radius,
            max_iterations=max_iterations,
        )
        np.testing.assert_array_equal(actual["labels"], expected.labels)
        np.testing.assert_array_equal(actual["neighbor_counts"], expected.neighbor_counts)
        assert actual["termination"] == expected.termination


def test_native_matches_reference_near_radius_rounding_boundary() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [0.68418360801139777, 0.72930980421800595],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1], dtype=np.int64)

    expected = valley_seeking_reference(
        features,
        labels,
        radius=1.0,
        max_iterations=1,
    )
    actual = _native(
        features,
        labels,
        radius=1.0,
        max_iterations=1,
    )

    np.testing.assert_array_equal(actual["neighbor_counts"], expected.neighbor_counts)
    np.testing.assert_array_equal(actual["labels"], expected.labels)


def test_native_binding_rejects_conversion_and_maximum_inputs() -> None:
    namespace = load_cpu_sorting_operation("valley_seeking")
    native = namespace._valley_seeking
    labels = np.array([0, 1], dtype=np.int64)
    with pytest.raises(TypeError, match="float64"):
        native(np.zeros((2, 1), dtype=np.float32), labels, 1.0, 2)
    with pytest.raises(ValueError, match="C-contiguous"):
        native(np.zeros((2, 2), dtype=np.float64)[:, ::2], labels, 1.0, 2)
    with pytest.raises(TypeError, match="int64"):
        native(np.zeros((2, 1), dtype=np.float64), labels.astype(np.int32), 1.0, 2)

    maximum_observations = namespace._valley_seeking_max_observations
    too_many_observations = np.empty((maximum_observations + 1, 1), dtype=np.float64)
    too_many_labels = np.zeros(maximum_observations + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="observation limit"):
        native(too_many_observations, too_many_labels, 1.0, 2)

    maximum_features = namespace._valley_seeking_max_features
    too_many_features = np.empty((0, maximum_features + 1), dtype=np.float64)
    with pytest.raises(ValueError, match="feature count"):
        native(too_many_features, np.empty(0, dtype=np.int64), 1.0, 2)


def test_native_execution_is_deterministic_and_unexported() -> None:
    features = np.array([[0.0], [0.1], [0.2], [4.0]], dtype=np.float64)
    labels = np.array([9, 2, 9, -1], dtype=np.int64)
    first = _native(features, labels, radius=0.25, max_iterations=20)
    second = _native(features, labels, radius=0.25, max_iterations=20)

    for name in first:
        if isinstance(first[name], np.ndarray):
            np.testing.assert_array_equal(first[name], second[name])
        else:
            assert first[name] == second[name]

    import neurale.sorting as sorting

    assert hasattr(sorting, "valley_seeking")
    assert "valley_seeking" in sorting.__all__
    assert not hasattr(sorting, "_valley_seeking")


def test_native_sorting_namespace_is_partitioned_by_operation() -> None:
    from neurale._native_loader import load_native_namespace

    sorting = load_native_namespace("sorting")
    assert hasattr(sorting, "detection")
    assert hasattr(sorting, "valley_seeking")
    assert not hasattr(sorting, "_detect_threshold")
    assert not hasattr(sorting, "_detect_threshold_waveforms")
    assert not hasattr(sorting, "_valley_seeking")


def test_dispatch_loads_exact_operation_subnamespace(monkeypatch) -> None:
    sentinel = object()
    requested: list[str] = []

    def record(namespace: str) -> object:
        requested.append(namespace)
        return sentinel

    monkeypatch.setattr(dispatch_module, "load_native_namespace", record)
    with runtime_context(device="cpu"):
        assert dispatch_module.load_cpu_sorting_operation("detection") is sentinel
        assert dispatch_module.load_cpu_sorting_operation("valley_seeking") is sentinel
    assert requested == ["sorting.detection", "sorting.valley_seeking"]
