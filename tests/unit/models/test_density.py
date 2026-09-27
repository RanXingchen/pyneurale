#!/usr/bin/env python3

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from neurale._native_loader import load_native_namespace
from neurale.exceptions import (
    DeviceUnavailableError,
    NativeUnavailableError,
    ValidationError,
)
from neurale.models import GaussianKDE
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


def _unaligned_float64(shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    buffer = bytearray(1 + size * np.dtype(np.float64).itemsize)
    values = np.frombuffer(
        buffer,
        dtype=np.float64,
        count=size,
        offset=1,
    ).reshape(shape)
    assert values.flags.c_contiguous
    assert not values.flags.aligned
    return values


def test_gaussian_kde_matches_scipy_sample_major_reference() -> None:
    samples = np.array(
        [
            [-1.0, 0.2],
            [-0.4, 0.5],
            [0.1, -0.3],
            [0.8, 0.9],
            [1.3, -0.7],
            [1.7, 0.4],
        ],
        dtype=float,
    )
    points = np.array(
        [
            [-0.2, 0.1],
            [0.5, 0.5],
            [1.0, -0.4],
        ],
        dtype=float,
    )

    kde = GaussianKDE(bw_method="scott").fit(samples)
    expected = gaussian_kde(samples.T, bw_method="scott")(points.T)

    assert kde.n_samples_ == samples.shape[0]
    assert kde.n_features_in_ == samples.shape[1]
    assert kde.bw_method_ == "scott"
    assert kde.device_ == "cpu"
    assert np.allclose(kde.pdf(points), expected)
    assert np.allclose(kde.logpdf(points), np.log(expected))
    assert np.allclose(kde.score_samples(points), np.log(expected))


def test_gaussian_kde_fit_returns_self_and_unfitted_calls_raise() -> None:
    samples = np.array(
        [[-1.0], [-0.2], [0.5], [1.4]],
        dtype=float,
    )
    kde = GaussianKDE()

    with pytest.raises(ValidationError):
        kde.pdf(samples)

    assert kde.fit(samples) is kde
    assert kde.pdf(samples).shape == (samples.shape[0],)


@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
)
def test_gaussian_kde_copy_and_pickle_reconstruct_cpu_state(
    clone,
    monkeypatch,
) -> None:
    from neurale.models import density

    original_load = density.load_models_operation
    selected_devices = []

    def recording_load(operation: str):
        loaded = original_load(operation)
        selected_devices.append(loaded[1])
        return loaded

    monkeypatch.setattr(density, "load_models_operation", recording_load)
    samples = np.array([[-1.0], [-0.2], [0.5], [1.4]], dtype=np.float64)
    points = np.array([[-0.5], [0.75]], dtype=np.float64)
    with runtime_context(device="cpu"):
        kde = GaussianKDE(0.5).fit(samples)

    with runtime_context(device="cuda"):
        restored = clone(kde)
        actual = restored.logpdf(points)

    assert kde.device_ == "cpu"
    assert restored.device_ == "cpu"
    assert restored._model is not kde._model
    assert selected_devices == [None, None]
    np.testing.assert_array_equal(actual, kde.logpdf(points))


def test_gaussian_kde_serialized_cuda_requires_cuda(monkeypatch) -> None:
    from neurale.models import _dispatch

    state = {
        "version": 1,
        "bw_method": "scott",
        "bw_method_fitted": "scott",
        "fitted": True,
        "device": "cuda",
        "cuda_device": 0,
        "samples": np.array([[-1.0], [0.0], [1.0]], dtype=np.float64),
    }

    def unavailable(namespace: str) -> object:
        raise NativeUnavailableError(namespace)

    monkeypatch.setattr(_dispatch, "load_native_namespace", unavailable)
    restored = GaussianKDE.__new__(GaussianKDE)
    with pytest.raises(DeviceUnavailableError, match=r"models\.density"):
        restored.__setstate__(state)


def test_gaussian_kde_records_normalized_bw_method() -> None:
    samples = np.array(
        [[-1.0], [-0.2], [0.5], [1.4]],
        dtype=float,
    )
    kde = GaussianKDE(bw_method=0.5).fit(samples)

    assert kde.bw_method_ == 0.5


def test_gaussian_kde_out_reuses_output_buffers() -> None:
    samples = np.array(
        [[-1.0], [-0.2], [0.5], [1.4]],
        dtype=np.float64,
    )
    kde = GaussianKDE().fit(samples)
    pdf_out = np.empty(samples.shape[0], dtype=np.float64)
    logpdf_out = np.empty(samples.shape[0], dtype=np.float64)

    assert kde.pdf(samples, out=pdf_out) is pdf_out
    assert kde.logpdf(samples, out=logpdf_out) is logpdf_out
    assert np.allclose(pdf_out, kde.pdf(samples))
    assert np.allclose(logpdf_out, kde.logpdf(samples))


def test_gaussian_kde_accepts_empty_query_batches() -> None:
    samples = np.array(
        [[-1.0], [-0.2], [0.5], [1.4]],
        dtype=float,
    )
    kde = GaussianKDE().fit(samples)

    assert kde.pdf(np.empty((0, 1))).shape == (0,)
    assert kde.logpdf(np.empty((0, 1))).shape == (0,)


def test_gaussian_kde_rejects_singular_full_covariance() -> None:
    with pytest.raises(ValidationError):
        GaussianKDE().fit(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                ]
            )
        )
    with pytest.raises(ValidationError):
        GaussianKDE().fit(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                ]
            )
        )


def test_gaussian_kde_far_finite_points_underflow_cleanly() -> None:
    kde = GaussianKDE().fit(
        np.array(
            [[0.0], [1.0], [2.0]],
            dtype=float,
        )
    )
    points = np.array([[1e308]], dtype=float)

    assert np.array_equal(kde.pdf(points), np.array([0.0]))
    assert np.array_equal(kde.logpdf(points), np.array([-np.inf]))


def test_gaussian_kde_rejects_non_finite_query_points() -> None:
    kde = GaussianKDE().fit(
        np.array(
            [[0.0], [1.0], [2.0]],
            dtype=np.float64,
        )
    )

    with pytest.raises(ValidationError):
        kde.pdf(np.array([[np.nan]], dtype=np.float64))
    with pytest.raises(ValidationError):
        kde.logpdf(np.array([[np.inf]], dtype=np.float64))


def test_native_kde_rejects_infinite_query_before_write() -> None:
    native = _native_models()
    assert native is not None
    samples = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    model = native.density.fit_gaussian_kde(samples, "scott", 0.0)
    out = np.array([123.0], dtype=np.float64)

    with pytest.raises(ValueError):
        model.logpdf(np.array([[np.inf]], dtype=np.float64), out)

    assert np.array_equal(out, np.array([123.0]))


def test_kde_rejects_bandwidth_on_covariance_underflow() -> None:
    tiny_positive = float(np.sqrt(np.nextafter(0.0, 1.0)))

    with pytest.raises(ValidationError, match="bw_method"):
        GaussianKDE(bw_method=tiny_positive).fit(
            np.array(
                [[0.0], [1.0], [2.0]],
                dtype=np.float64,
            )
        )


def test_gaussian_kde_requires_strict_float64_query_points() -> None:
    kde = GaussianKDE().fit(
        np.array(
            [[0.0], [1.0], [2.0]],
            dtype=np.float64,
        )
    )

    with pytest.raises(ValidationError):
        kde.pdf(np.array([[1.0]], dtype=np.float32))


def test_gaussian_kde_rejects_unaligned_samples() -> None:
    samples = _unaligned_float64((3, 1))
    samples[:, 0] = [0.0, 1.0, 2.0]

    with pytest.raises(ValidationError, match="aligned"):
        GaussianKDE().fit(samples)


def test_gaussian_kde_rejects_unaligned_query_and_output() -> None:
    samples = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    kde = GaussianKDE().fit(samples)
    points = _unaligned_float64((1, 1))
    output = _unaligned_float64((1,))

    with pytest.raises(ValidationError, match="aligned"):
        kde.pdf(points)
    with pytest.raises(ValidationError, match="aligned"):
        kde.pdf(np.array([[1.0]], dtype=np.float64), out=output)


def test_native_gaussian_kde_rejects_unaligned_arrays() -> None:
    native = _native_models()
    assert native is not None
    samples = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    model = native.density.fit_gaussian_kde(samples, "scott", 0.0)
    points = _unaligned_float64((1, 1))
    native_samples = _unaligned_float64((3, 1))

    with pytest.raises(ValueError, match="aligned"):
        model.logpdf(points)
    with pytest.raises(ValueError, match="aligned"):
        native.density.fit_gaussian_kde(native_samples, "scott", 0.0)


def test_kde_rejects_obsolete_arguments_and_mismatch() -> None:
    samples = np.array(
        [[-1.0], [-0.2], [0.5], [1.4]],
        dtype=float,
    )
    kde = GaussianKDE().fit(samples)

    with pytest.raises(TypeError):
        GaussianKDE(device="CUDA")
    with pytest.raises(TypeError):
        GaussianKDE(bandwidth="scott")
    with pytest.raises(ValidationError):
        kde.pdf(np.ones((2, 2)))
    with pytest.raises(ValidationError):
        kde.pdf(samples, out=np.empty((samples.shape[0], 1), dtype=np.float64))


@pytest.mark.gpu
def test_gaussian_kde_cuda_matches_cpu_pdf_and_logpdf() -> None:
    _require_cuda_models()
    rng = np.random.default_rng(1234)
    samples = rng.normal(size=(96, 4)).astype(np.float64)
    points = rng.normal(size=(31, 4)).astype(np.float64)

    with runtime_context(device="cpu"):
        expected = GaussianKDE("silverman").fit(samples)
    with runtime_context(device="cuda", cuda_device=0):
        actual = GaussianKDE("silverman").fit(samples)

    output = np.empty(points.shape[0], dtype=np.float64)
    returned = actual.logpdf(points, out=output)
    assert returned is output
    assert np.allclose(output, expected.logpdf(points), rtol=1e-13, atol=1e-13)
    assert np.allclose(actual.pdf(points), expected.pdf(points), rtol=1e-13, atol=1e-15)
    assert expected.device_ == "cpu"
    assert actual.device_ == "cuda"

    payload = pickle.dumps(actual)
    with runtime_context(device="cpu"):
        restored = pickle.loads(payload)
        restored_values = restored.logpdf(points)
    assert restored.device_ == "cuda"
    np.testing.assert_allclose(restored_values, expected.logpdf(points), rtol=1e-13)


@pytest.mark.gpu
def test_gaussian_kde_cuda_handles_empty_and_far_queries() -> None:
    _require_cuda_models()
    samples = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)
    with runtime_context(device="cuda", cuda_device=0):
        kde = GaussianKDE().fit(samples)

    assert kde.pdf(np.empty((0, 1), dtype=np.float64)).shape == (0,)
    assert np.isfinite(kde.logpdf(np.array([[1e100]], dtype=np.float64)))[0]
    assert kde.pdf(np.array([[1e100]], dtype=np.float64))[0] == 0.0


@pytest.mark.gpu
def test_gaussian_kde_cuda_reduction_matches_cpu() -> None:
    _require_cuda_models()
    rng = np.random.default_rng(2026)
    samples = rng.normal(size=(513, 5)).astype(np.float64)
    points = rng.normal(size=(37, 5)).astype(np.float64)

    with runtime_context(device="cpu"):
        expected = GaussianKDE(0.7).fit(samples)
    with runtime_context(device="cuda", cuda_device=0):
        actual = GaussianKDE(0.7).fit(samples)

    assert np.allclose(
        actual.logpdf(points),
        expected.logpdf(points),
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.allclose(
        actual.pdf(points),
        expected.pdf(points),
        rtol=1e-13,
        atol=1e-15,
    )


def test_kde_cuda_error_is_not_validation_error(
    monkeypatch,
) -> None:
    from neurale.models import density

    class NativeDeviceUnavailableError(RuntimeError):
        pass

    class Model:
        @staticmethod
        def pdf(*args):
            raise RuntimeError("CUDA execution failed")

    class Native:
        @staticmethod
        def fit_gaussian_kde(*args):
            return Model()

    monkeypatch.setattr(
        density,
        "load_models_operation",
        lambda operation: (Native(), 0, NativeDeviceUnavailableError),
    )
    with runtime_context(device="cuda"):
        kde = GaussianKDE().fit(np.array([[0.0], [1.0]]))
    with pytest.raises(RuntimeError, match="execution failed") as error:
        kde.pdf(np.array([[0.5]]))
    assert type(error.value) is RuntimeError
