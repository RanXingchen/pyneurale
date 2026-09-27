#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale._native_loader import load_native_namespace
from neurale.exceptions import (
    DeviceUnavailableError,
    NativeUnavailableError,
    ValidationError,
)
from neurale.models import knn
from neurale.runtime import runtime_context, runtime_info


def _native_models():
    try:
        return load_native_namespace("models")
    except NativeUnavailableError:
        return None


def _require_cuda_models() -> None:
    native = _native_models()
    if native is None or not hasattr(native, "cuda"):
        pytest.skip("CUDA model kernels are not built")
    cuda = runtime_info(probe_cuda=True).cuda
    if cuda is None or not cuda.available:
        pytest.skip("no usable CUDA device is available")


def test_knn_returns_fixed_shape_sorted_neighbors() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ],
    )

    indices, distances = knn(X, 2, metric="sqeuclidean")

    assert indices.shape == (4, 2)
    assert distances.shape == (4, 2)
    assert np.array_equal(
        indices,
        np.array(
            [
                [2, 1],
                [0, 3],
                [0, 1],
                [1, 2],
            ]
        ),
    )
    assert np.allclose(
        distances,
        np.array(
            [
                [1.0, 4.0],
                [4.0, 4.0],
                [1.0, 5.0],
                [4.0, 5.0],
            ]
        ),
    )


def test_knn_include_self_and_duplicates_are_explicit() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )

    indices, distances = knn(X, 2, include_self=True)
    single_idx, single_distance = knn(X, 1, include_self=True)

    assert np.array_equal(indices[0], np.array([0, 1]))
    assert np.array_equal(distances[0], np.array([0.0, 0.0]))
    assert np.array_equal(single_idx.ravel(), np.array([0, 1, 2]))
    assert np.array_equal(single_distance.ravel(), np.zeros(3))


def test_knn_orders_by_squared_distance_without_sqrt_rounding() -> None:
    magnitude = np.ldexp(1.0, 26)
    X = np.array(
        [
            [0.0, 0.0],
            [magnitude, 1.0],
            [magnitude, 0.0],
        ],
        dtype=np.float64,
    )

    indices, distances = knn(X, 1, include_self=False)

    assert indices[0, 0] == 2
    assert distances[0, 0] == magnitude * magnitude


def test_knn_handles_finite_difference_overflow_when_ordering() -> None:
    maximum = np.finfo(np.float64).max
    X = np.array(
        [
            [-maximum],
            [maximum],
            [maximum / 2.0],
        ],
        dtype=np.float64,
    )

    indices, distances = knn(X, 1, include_self=False)

    assert indices[0, 0] == 2
    assert np.isinf(distances[0, 0])


def test_knn_uses_stable_squared_distance_at_large_offsets() -> None:
    X = np.array(
        [
            [1e16],
            [1e16 + 2.0],
            [1e16 + 4.0],
        ],
        dtype=np.float64,
    )

    indices, distances = knn(X, 1, include_self=False)

    assert np.array_equal(indices.ravel(), np.array([1, 0, 1]))
    assert np.array_equal(distances.ravel(), np.array([4.0, 4.0, 4.0]))


def test_knn_orders_overflowing_squared_distances_with_scaled_key() -> None:
    large = np.array([[0.0], [2e200], [1e200]], dtype=np.float64)

    large_indices, large_distances = knn(large, 1, include_self=False)

    assert np.array_equal(large_indices.ravel(), np.array([2, 2, 0]))
    assert np.all(np.isinf(large_distances))


def test_knn_orders_underflowing_squared_distances_with_scaled_key() -> None:
    tiny = np.array([[0.0], [2e-200], [1e-200]], dtype=np.float64)

    tiny_indices, tiny_distances = knn(tiny, 1, include_self=False)

    assert np.array_equal(tiny_indices.ravel(), np.array([2, 2, 0]))
    assert np.array_equal(tiny_distances.ravel(), np.zeros(3))


def test_knn_orders_subnormal_squared_distances_with_scaled_key() -> None:
    X = np.array(
        [
            [np.ldexp(1.0, -536)],
            [-np.ldexp(1.0, -566)],
            [np.ldexp(1.0, -859)],
        ],
        dtype=np.float64,
    )

    indices, distances = knn(X, 2, include_self=False)

    assert np.array_equal(indices[0], np.array([2, 1]))
    assert distances[0, 0] == distances[0, 1]


def test_knn_formats_euclidean_distance_from_scaled_key() -> None:
    large = np.array([[0.0], [2e200], [1e200]], dtype=np.float64)

    euclidean_indices, euclidean_distances = knn(
        large,
        1,
        metric="euclidean",
        include_self=False,
    )

    assert np.array_equal(euclidean_indices.ravel(), np.array([2, 2, 0]))
    assert np.array_equal(
        euclidean_distances.ravel(),
        np.array([1e200, 1e200, 1e200]),
    )


def test_knn_orders_equal_distance_neighbors_by_index() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )

    indices, distances = knn(X, 4, include_self=False)

    assert np.array_equal(indices[0], np.array([1, 2, 3, 4]))
    assert np.array_equal(distances[0], np.ones(4))


def test_native_knn_rejects_non_finite_input() -> None:
    native = _native_models()
    assert native is not None

    with pytest.raises(ValueError):
        native.neighbors.knn(
            np.array([[0.0], [np.inf]], dtype=np.float64),
            1,
            "sqeuclidean",
            False,
        )


def test_knn_rejects_non_finite_input() -> None:
    with pytest.raises(ValidationError):
        knn(np.array([[0.0], [np.nan]], dtype=np.float64), 1)


def test_native_knn_rejects_invalid_k_before_output_allocation() -> None:
    native = _native_models()
    assert native is not None

    with pytest.raises(ValueError):
        native.neighbors.knn(
            np.eye(2, dtype=np.float64),
            10_000_000,
            "sqeuclidean",
            False,
        )


def test_knn_rejects_unavailable_neighbor_count_and_device_argument() -> None:
    X = np.eye(3)

    with pytest.raises(ValidationError):
        knn(X, 3, include_self=False)
    with pytest.raises(TypeError):
        knn(X, 1, device="CUDA")


@pytest.mark.gpu
def test_knn_cuda_matches_cpu_distance_and_ordering() -> None:
    _require_cuda_models()
    magnitude = np.ldexp(1.0, 26)
    X = np.array(
        [
            [0.0, 0.0],
            [magnitude, 1.0],
            [magnitude, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    with runtime_context(device="cpu"):
        expected = knn(X, 3, metric="euclidean", include_self=True)
    with runtime_context(device="cuda", cuda_device=0):
        actual = knn(X, 3, metric="euclidean", include_self=True)

    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1], rtol=1e-14, atol=0.0)

    with runtime_context(device="cuda", cuda_device=0):
        required = knn(X, 3, metric="euclidean", include_self=True)
    assert np.array_equal(required[0], expected[0])
    assert np.allclose(required[1], expected[1], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("k", [1, 8, 9, 16, 17, 32, 33, 48, 49, 64])
@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.gpu
def test_knn_cuda_matches_cpu_across_dispatch_boundaries(
    k: int,
    include_self: bool,
) -> None:
    _require_cuda_models()
    X = np.random.default_rng(2026).normal(size=(129, 5)).astype(np.float64)

    with runtime_context(device="cpu"):
        expected = knn(X, k, include_self=include_self)
    with runtime_context(device="cuda", cuda_device=0):
        actual = knn(X, k, include_self=include_self)

    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1], rtol=1e-14, atol=0.0)


@pytest.mark.gpu
def test_knn_cuda_preserves_extended_distance_ordering() -> None:
    _require_cuda_models()
    cases = (
        (np.array([[0.0], [2e200], [1e200]], dtype=np.float64), 1),
        (np.array([[0.0], [2e-200], [1e-200]], dtype=np.float64), 1),
        (
            np.array(
                [
                    [np.ldexp(1.0, -536)],
                    [-np.ldexp(1.0, -566)],
                    [np.ldexp(1.0, -859)],
                ],
                dtype=np.float64,
            ),
            2,
        ),
        (np.array([[1e16], [1e16 + 2.0], [1e16 + 4.0]]), 1),
    )

    for X, k in cases:
        with runtime_context(device="cpu"):
            expected = knn(X, k)
        with runtime_context(device="cuda", cuda_device=0):
            actual = knn(X, k)
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])

    duplicates = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    with runtime_context(device="cuda", cuda_device=0):
        indices, distances = knn(duplicates, 1, include_self=True)
    assert np.array_equal(indices.ravel(), np.arange(3))
    assert np.array_equal(distances.ravel(), np.zeros(3))


@pytest.mark.gpu
def test_knn_cuda_matches_cpu_for_large_feature_tile() -> None:
    _require_cuda_models()
    X = np.random.default_rng(2027).normal(size=(257, 61)).astype(np.float64)

    with runtime_context(device="cpu"):
        expected = knn(X, 5, include_self=True)
    with runtime_context(device="cuda", cuda_device=0):
        actual = knn(X, 5, include_self=True)

    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1], rtol=1e-14, atol=0.0)


def test_knn_explicit_cuda_does_not_fall_back_to_cpu(monkeypatch) -> None:
    from neurale.models import _dispatch

    def unavailable(namespace: str) -> object:
        raise NativeUnavailableError(namespace)

    monkeypatch.setattr(_dispatch, "load_native_namespace", unavailable)
    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match=r"models\.neighbors"):
            knn(np.eye(3), 1)


@pytest.mark.gpu
def test_knn_cuda_invalid_device_is_device_unavailable() -> None:
    _require_cuda_models()
    n_devices = len(runtime_info(probe_cuda=True).cuda.devices)
    native_cuda = load_native_namespace("cuda")
    previous_device = native_cuda.current_device()
    with runtime_context(device="cuda", cuda_device=n_devices):
        with pytest.raises(DeviceUnavailableError, match="available devices"):
            knn(np.eye(3), 1)
    assert native_cuda.current_device() == previous_device


@pytest.mark.gpu
def test_knn_cuda_restores_previous_device() -> None:
    _require_cuda_models()
    cuda_info = runtime_info(probe_cuda=True).cuda
    if len(cuda_info.devices) < 2:
        pytest.skip("multiple CUDA devices are required")
    native_cuda = load_native_namespace("cuda")
    native_cuda.set_device(0)
    with runtime_context(device="cuda", cuda_device=1):
        knn(np.eye(3), 1)
    assert native_cuda.current_device() == 0


def test_knn_cuda_execution_error_is_not_device_unavailable(monkeypatch) -> None:
    from neurale.models import neighbors

    class NativeDeviceUnavailableError(RuntimeError):
        pass

    class Native:
        @staticmethod
        def knn(*args):
            raise RuntimeError("CUDA execution failed")

    monkeypatch.setattr(
        neighbors,
        "load_models_operation",
        lambda operation: (Native(), 0, NativeDeviceUnavailableError),
    )
    with runtime_context(device="cuda"):
        with pytest.raises(RuntimeError, match="execution failed") as error:
            knn(np.eye(3), 1)
    assert not isinstance(error.value, DeviceUnavailableError)
