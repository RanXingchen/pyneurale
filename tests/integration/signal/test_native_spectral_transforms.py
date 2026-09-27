#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.signal import analytic_signal, czt


def _native() -> object:
    return importlib.import_module("neurale._native").signal.spectral


@pytest.mark.parametrize("length", [1, 7, 8, 31, 64, 127, 128, 4097])
def test_all_native_fft_kernels_match_numpy(length: int) -> None:
    native = _native()
    rng = np.random.default_rng(13)
    values = rng.standard_normal((length, 3)) + 1j * rng.standard_normal((length, 3))
    expected = np.fft.fft(values, axis=0)
    for kernel in native.fft_available_kernels():
        result, selected = native.fft(values, False, kernel)
        assert selected == kernel
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
        restored, inverse_selected = native.fft(result, True, kernel)
        assert inverse_selected == kernel
        np.testing.assert_allclose(restored, values, rtol=1e-10, atol=1e-10)


def test_native_fft_auto_selection_is_observable() -> None:
    native = _native()
    values = np.ones((128, 2), dtype=complex)
    _, selected = native.fft(values)
    assert selected == native.fft_kernel_name(128)
    assert selected in native.fft_available_kernels()


@pytest.mark.parametrize(
    ("length", "channels", "m"),
    [(7, 1, 7), (16, 3, 16), (48, 2, 31)],
)
def test_native_default_czt_matches_numpy(
    length: int,
    channels: int,
    m: int,
) -> None:
    _native()
    rng = np.random.default_rng(14)
    values = rng.standard_normal((length, channels)) + 1j * rng.standard_normal((length, channels))

    native = czt(values, m)
    reference = scipy_signal.czt(values, m=m, axis=0)

    np.testing.assert_allclose(native, reference, rtol=1e-11, atol=1e-11)


def test_native_arbitrary_czt_matches_numpy() -> None:
    _native()
    rng = np.random.default_rng(15)
    values = rng.standard_normal((9, 2))
    w = 0.99 * np.exp(-0.3j)
    a = 0.95 * np.exp(0.1j)

    native = czt(values, 13, w=w, a=a)
    reference = scipy_signal.czt(values, m=13, w=w, a=a, axis=0)

    np.testing.assert_allclose(native, reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(("length", "nfft"), [(7, 7), (7, 8), (16, 32)])
def test_native_analytic_signal_matches_numpy(
    length: int,
    nfft: int,
) -> None:
    _native()
    rng = np.random.default_rng(16)
    values = rng.standard_normal((length, 3))

    native = analytic_signal(values, nfft=nfft)
    reference = scipy_signal.hilbert(values, N=nfft, axis=0)

    np.testing.assert_allclose(native, reference, rtol=1e-11, atol=1e-11)
