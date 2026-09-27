#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.signal.filtering import (
    FirCoefficients,
    FirFilter,
    fir_filter,
    firls,
    firwin,
)


def _native() -> object:
    native = importlib.import_module("neurale._native").signal
    native.filtering  # noqa: B018
    return native


@pytest.mark.parametrize("order", [10, 20])
def test_native_firls_matches_scipy(order: int) -> None:
    _native()
    bands = [0.0, 0.2, 0.3, 1.0]
    desired = [1.0, 1.0, 0.0, 0.0]
    coefs = firls(order, bands, desired)
    expected = scipy_signal.firls(order + 1, bands, desired, fs=2.0)
    np.testing.assert_allclose(coefs.taps, expected, atol=1e-11)


@pytest.mark.parametrize(
    ("band", "cutoff"),
    [
        ("lowpass", 0.2),
        ("highpass", 0.2),
        ("bandpass", [0.2, 0.4]),
        ("bandstop", [0.2, 0.4]),
    ],
)
@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "flattop"])
def test_native_firwin_matches_scipy(band, cutoff, window) -> None:
    _native()
    order = 32 if band in ("highpass", "bandstop") else 31
    coefs = firwin(order, cutoff, band=band, window=window)
    expected = scipy_signal.firwin(
        coefs.order + 1,
        cutoff,
        window=window,
        pass_zero=band in ("lowpass", "bandstop"),
        fs=2.0,
    )
    np.testing.assert_allclose(coefs.taps, expected, atol=2e-14)


@pytest.mark.parametrize(
    ("n_samples", "n_channels", "n_taps"),
    [(0, 1, 1), (1, 1, 3), (17, 2, 5), (128, 16, 21)],
)
def test_fir_filter_matches_scipy(
    n_samples: int,
    n_channels: int,
    n_taps: int,
) -> None:
    _native()
    rng = np.random.default_rng(123)
    values = rng.standard_normal((n_samples, n_channels))
    coefs = FirCoefficients(rng.standard_normal(n_taps))
    actual, actual_state = fir_filter(values, coefs, return_state=True)
    expected = (
        np.empty_like(values)
        if n_samples == 0
        else scipy_signal.lfilter(coefs.taps, [1.0], values, axis=0)
    )
    if n_samples == 0:
        expected_state = np.zeros((coefs.order, n_channels))
    elif coefs.order == 0:
        expected_state = np.zeros((0, n_channels))
    elif n_samples >= coefs.order:
        expected_state = values[-coefs.order :]
    else:
        expected_state = np.vstack(
            (
                np.zeros((coefs.order - n_samples, n_channels)),
                values,
            )
        )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual_state, expected_state, atol=1e-12)


def test_all_native_fir_kernels_match_builtin_and_state() -> None:
    native = _native().filtering
    rng = np.random.default_rng(91)
    values = rng.standard_normal((257, 7))
    taps = rng.standard_normal(129)
    state = rng.standard_normal((128, 7))
    reference, reference_state = native.fir_filter(values, taps, state, "builtin-direct")[:2]
    kernels = native.fir_available_kernels()
    assert "builtin-direct" in kernels
    for kernel in kernels:
        result, final_state, selected = native.fir_filter(values, taps, state, kernel)
        assert selected == kernel
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-11)
        np.testing.assert_array_equal(final_state, reference_state)


def test_native_fir_kernel_selection_uses_channels_and_taps() -> None:
    native = _native().filtering
    assert native.fir_kernel_name(1, 1, 5) == "builtin-direct"
    if "mkl-vsl-direct" in native.fir_available_kernels():
        assert native.fir_kernel_name(1, 32, 17) != "builtin-direct"
    if "mkl-vsl-fft" in native.fir_available_kernels():
        assert native.fir_kernel_name(36001, 1, 361) == "mkl-vsl-fft"


def test_fir_realtime_processor_processes_in_place() -> None:
    _native()
    rng = np.random.default_rng(8)
    values = np.ascontiguousarray(rng.standard_normal((97, 8)), dtype=np.float64)
    source = values.copy()
    coefs = FirCoefficients(rng.standard_normal(17))
    expected = fir_filter(source, coefs)
    processor = FirFilter(coefs, n_channels=8)
    assert processor.kernel == "builtin-ring"

    for chunk in (values[:10], values[10:51], values[51:]):
        assert processor.process(chunk) is chunk

    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_fir_realtime_processor_state_matches_offline_continuation() -> None:
    _native()
    rng = np.random.default_rng(82)
    values = np.ascontiguousarray(rng.standard_normal((311, 3)), dtype=np.float64)
    coefs = FirCoefficients(rng.standard_normal(37))
    expected = fir_filter(values, coefs)

    first = values[:127].copy()
    second = values[127:].copy()
    processor = FirFilter(coefs, n_channels=3)
    assert processor.kernel in {
        "builtin-ring",
        "mkl-channel-dot-ring",
        "mkl-blas-ring",
    }
    processor.process(first)
    state = processor.get_state()

    resumed = FirFilter(coefs, n_channels=3)
    resumed.set_state(state)
    resumed.process(second)

    np.testing.assert_allclose(
        np.vstack((first, second)),
        expected,
        atol=1e-12,
    )


def test_fir_realtime_uses_mkl_for_wide_frames() -> None:
    native = _native().filtering
    rng = np.random.default_rng(83)
    values = np.ascontiguousarray(rng.standard_normal((79, 32)), dtype=np.float64)
    coefs = FirCoefficients(rng.standard_normal(41))
    expected = fir_filter(values, coefs)

    processor = FirFilter(coefs, n_channels=32)
    if "mkl-vsl-direct" in native.fir_available_kernels():
        assert processor.kernel == "mkl-blas-ring"
    processor.process(values)

    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_fir_realtime_processor_long_chunk_matches_offline() -> None:
    _native()
    rng = np.random.default_rng(85)
    values = np.ascontiguousarray(rng.standard_normal((257, 2)), dtype=np.float64)
    coefs = FirCoefficients(rng.standard_normal(257))
    expected = fir_filter(values.copy(), coefs)

    processor = FirFilter(coefs, n_channels=2)
    assert processor.process(values) is values

    np.testing.assert_allclose(values, expected, atol=1e-11)


def test_fir_realtime_processor_single_channel_advances_state() -> None:
    native = _native().filtering
    rng = np.random.default_rng(84)
    values = np.ascontiguousarray(rng.standard_normal((257, 1)), dtype=np.float64)
    coefs = FirCoefficients(rng.standard_normal(65))
    expected = fir_filter(values.copy(), coefs)

    processor = FirFilter(coefs, n_channels=1)
    if "mkl-vsl-direct" in native.fir_available_kernels():
        assert processor.kernel == "mkl-channel-dot-ring"
    for i in range(values.shape[0]):
        processor.process(values[i : i + 1])

    np.testing.assert_allclose(values, expected, atol=1e-12)
