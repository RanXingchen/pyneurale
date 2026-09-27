#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest

from neurale.signal import fft_integrate


def _native() -> object:
    return importlib.import_module("neurale._native").signal.transforms


def _reference_fft_integrate(values: np.ndarray, dt: float, order: int) -> np.ndarray:
    freqs = np.fft.fftfreq(values.shape[0], d=dt)
    omega = 2.0 * np.pi * freqs
    multiplier = np.zeros(omega.shape, dtype=complex)
    nonzero = omega != 0
    multiplier[nonzero] = np.power(-1j / omega[nonzero], order)
    integrated = np.fft.ifft(
        np.fft.fft(values, axis=0) * multiplier[:, None],
        axis=0,
    )
    return integrated.real if np.isrealobj(values) else integrated


@pytest.mark.parametrize(("length", "channels", "order"), [(31, 1, 1), (32, 3, 2)])
def test_native_fft_integration_matches_numpy(
    length: int,
    channels: int,
    order: int,
) -> None:
    _native()
    rng = np.random.default_rng(88)
    values = rng.standard_normal((length, channels))

    native = fft_integrate(values, dt=0.01, order=order)
    reference = _reference_fft_integrate(values, dt=0.01, order=order)

    np.testing.assert_allclose(native, reference, rtol=1e-10, atol=1e-10)


def test_native_fft_integration_supports_complex_input() -> None:
    _native()
    rng = np.random.default_rng(89)
    values = rng.standard_normal((17, 2)) + 1j * rng.standard_normal((17, 2))

    native = fft_integrate(values, dt=0.01)
    reference = _reference_fft_integrate(values, dt=0.01, order=1)

    np.testing.assert_allclose(native, reference, rtol=1e-10, atol=1e-10)
