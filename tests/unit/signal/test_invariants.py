#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.signal import (
    FirCoefficients,
    FirFilter,
    fir_filter,
    resample,
    welch_psd,
)


def test_welch_psd_integrates_to_white_noise_variance() -> None:
    rng = np.random.default_rng(2026)
    values = rng.standard_normal(32768)
    psd, freqs = welch_psd(
        values,
        1000.0,
        nfft=1024,
        window_length=1024,
        overlap=0.5,
        detrend="mean",
    )
    power = np.trapezoid(psd, freqs)
    assert power == pytest.approx(np.var(values), rel=0.05)


def test_welch_psd_finds_sinusoid_frequency() -> None:
    fs = 500.0
    time = np.arange(5000) / fs
    values = np.sin(2.0 * np.pi * 37.0 * time)
    psd, freqs = welch_psd(
        values,
        fs,
        nfft=2000,
        window_length=1000,
    )
    assert freqs[int(np.argmax(psd))] == 37.0


def test_streaming_fir_matches_offline_across_irregular_chunks() -> None:
    rng = np.random.default_rng(17)
    values = rng.standard_normal((257, 2))
    coefs = FirCoefficients(scipy_signal.firwin(31, 0.2))
    expected = fir_filter(values, coefs)
    actual = np.ascontiguousarray(values.copy(), dtype=np.float64)
    processor = FirFilter(coefs, n_channels=2)
    for chunk in (
        actual[:3],
        actual[3:129],
        actual[129:130],
        actual[130:],
    ):
        processor.process(chunk)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_resampling_preserves_passband_and_suppresses_alias() -> None:
    fs = 1000.0
    time = np.arange(10000) / fs
    values = np.sin(2.0 * np.pi * 40.0 * time) + np.sin(2.0 * np.pi * 420.0 * time)
    downsampled = resample(values, 1, 4)
    freqs, spectrum = scipy_signal.periodogram(downsampled, fs=250.0)
    passband = spectrum[np.argmin(np.abs(freqs - 40.0))]
    alias = spectrum[np.argmin(np.abs(freqs - 80.0))]
    assert passband > 1000.0 * alias
