#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal

from neurale.signal import MultitaperPsdProcessor, multitaper_psd
from neurale.signal.spectral._psd import (
    postprocess_psd,
    preprocess_segment,
    resolve_smoothing_width,
)
from neurale.signal.spectral.frequency import freq_vector
from neurale.signal.spectral.scaling import scale_spectrum


def _native() -> object:
    native = importlib.import_module("neurale._native").signal.spectral
    native._MultitaperPsdProcessor  # noqa: B018

    return native


def _reference_multitaper_psd(
    values: np.ndarray,
    *,
    fs: float | None = None,
    nw: float,
    n_tapers: int,
    nfft: int,
    weighting: str,
    sides: str = "auto",
    detrend: str = "none",
    smoothing_width: float | None = None,
    db: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    data = preprocess_segment(values, detrend)
    tapers, ratios = scipy_signal.windows.dpss(
        data.shape[0],
        nw,
        Kmax=n_tapers,
        sym=True,
        norm=2,
        return_ratios=True,
    )
    whole = _reference_multitaper_spectrum(
        data,
        np.asarray(tapers, dtype=float).T,
        np.asarray(ratios, dtype=float),
        nfft,
        weighting,
    )
    output_sides = (
        "one-sided"
        if sides == "auto" and not np.iscomplexobj(data)
        else "two-sided"
        if sides == "auto"
        else sides
    )
    freqs = freq_vector(nfft, fs)
    scaled, freqs, _ = scale_spectrum(
        whole,
        freqs,
        sides=output_sides,
        nfft=nfft,
        fs=fs,
        estimate="psd",
        axis=0,
    )
    width = resolve_smoothing_width(smoothing_width, None, freqs)
    return postprocess_psd(scaled, db, width), freqs


def _reference_multitaper_spectrum(
    data: np.ndarray,
    tapers: np.ndarray,
    ratios: np.ndarray,
    nfft: int,
    weighting: str,
) -> np.ndarray:
    output = np.empty((nfft, data.shape[1]), dtype=float)
    for channel in range(data.shape[1]):
        tapered = tapers.T * data[:, channel]
        transforms = (
            scipy_fft.fft(tapered, n=nfft, axis=1)
            if data.shape[0] <= nfft
            else scipy_signal.czt(tapered, m=nfft, axis=1)
        )
        spectra = np.abs(transforms) ** 2
        output[:, channel] = _combine_reference_spectra(
            spectra,
            ratios,
            data[:, channel],
            weighting,
        )
    return output


def _combine_reference_spectra(
    spectra: np.ndarray,
    ratios: np.ndarray,
    data: np.ndarray,
    weighting: str,
) -> np.ndarray:
    if weighting == "unity":
        return np.mean(spectra, axis=0)
    if weighting == "eigen":
        return np.sum(ratios[:, None] * spectra, axis=0) / ratios.size

    signal_power = float(np.vdot(data, data).real / data.size)
    if signal_power == 0.0:
        return np.zeros(spectra.shape[1], dtype=float)
    estimate = np.mean(spectra[:2], axis=0)
    leakage = signal_power * (1.0 - ratios[:, None])
    tol = 0.0005 * signal_power
    for _ in range(100):
        den = ratios[:, None] * estimate[None, :] + leakage
        weights = np.divide(
            estimate[None, :],
            den,
            out=np.zeros_like(den),
            where=den != 0,
        )
        adaptive_weights = weights * weights * ratios[:, None]
        weight_sum = np.sum(adaptive_weights, axis=0)
        next_estimate = np.divide(
            np.sum(adaptive_weights * spectra, axis=0),
            weight_sum,
            out=np.zeros_like(estimate),
            where=weight_sum != 0,
        )
        if np.sum(np.abs(next_estimate - estimate)) <= tol:
            return next_estimate
        estimate = next_estimate
    return estimate


def _padded_segments(values: np.ndarray, length: int) -> list[np.ndarray]:
    complete = values.shape[0] // length
    segments = [values[idx * length : (idx + 1) * length] for idx in range(complete)]
    remainder = values.shape[0] % length
    if remainder:
        padded = np.zeros((length, values.shape[1]))
        padded[:remainder] = values[complete * length :]
        segments.append(padded)
    return segments


@pytest.mark.parametrize("weighting", ["unity", "eigen", "adaptive"])
@pytest.mark.parametrize(("n_samples", "nfft"), [(48, 64), (48, 31)])
def test_native_real_multitaper_matches_reference(
    weighting: str,
    n_samples: int,
    nfft: int,
) -> None:
    _native()
    rng = np.random.default_rng(17)
    values = rng.standard_normal((n_samples, 3))
    kwargs = {
        "fs": 200.0,
        "nw": 3.0,
        "n_tapers": 4,
        "nfft": nfft,
        "weighting": weighting,
    }

    native, native_freqs = multitaper_psd(values, **kwargs)
    reference, reference_freqs = _reference_multitaper_psd(values, **kwargs)

    np.testing.assert_allclose(native_freqs, reference_freqs)
    np.testing.assert_allclose(native, reference, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("weighting", ["unity", "eigen", "adaptive"])
def test_native_complex_multitaper_matches_reference(weighting: str) -> None:
    _native()
    rng = np.random.default_rng(18)
    values = rng.standard_normal((37, 2)) + 1j * rng.standard_normal((37, 2))
    kwargs = {
        "nw": 2.5,
        "n_tapers": 3,
        "nfft": 64,
        "weighting": weighting,
    }

    native, native_freqs = multitaper_psd(values, **kwargs)
    reference, reference_freqs = _reference_multitaper_psd(values, **kwargs)

    np.testing.assert_allclose(native_freqs, reference_freqs)
    np.testing.assert_allclose(native, reference, rtol=2e-10, atol=2e-10)


def test_native_windowed_multitaper_matches_reference() -> None:
    _native()
    rng = np.random.default_rng(19)
    values = rng.standard_normal((190, 3))
    kwargs = {
        "fs": 250.0,
        "nw": 3.0,
        "n_tapers": 5,
        "nfft": 128,
        "window_length": 64,
        "incomplete": "pad",
        "detrend": "mean",
        "weighting": "adaptive",
        "smoothing_width": 1.5,
    }

    native, native_freqs = multitaper_psd(values, **kwargs)
    reference_values = [
        _reference_multitaper_psd(
            segment,
            fs=kwargs["fs"],
            nw=kwargs["nw"],
            n_tapers=kwargs["n_tapers"],
            nfft=kwargs["nfft"],
            weighting=kwargs["weighting"],
            detrend=kwargs["detrend"],
            smoothing_width=kwargs["smoothing_width"],
        )[0]
        for segment in _padded_segments(values, kwargs["window_length"])
    ]
    reference = np.mean(reference_values, axis=0)
    _, reference_freqs = _reference_multitaper_psd(
        values[: kwargs["window_length"]],
        fs=kwargs["fs"],
        nw=kwargs["nw"],
        n_tapers=kwargs["n_tapers"],
        nfft=kwargs["nfft"],
        weighting=kwargs["weighting"],
        detrend=kwargs["detrend"],
        smoothing_width=kwargs["smoothing_width"],
    )

    np.testing.assert_allclose(native_freqs, reference_freqs)
    np.testing.assert_allclose(native, reference, rtol=2e-10, atol=2e-10)


def test_native_real_two_sided_multitaper_matches_reference() -> None:
    _native()
    rng = np.random.default_rng(23)
    values = rng.standard_normal((80, 4))
    kwargs = {
        "fs": 500.0,
        "nw": 3.0,
        "n_tapers": 5,
        "nfft": 128,
        "weighting": "adaptive",
        "sides": "two-sided",
    }

    native, native_freqs = multitaper_psd(values, **kwargs)
    reference, reference_freqs = _reference_multitaper_psd(values, **kwargs)

    assert native.shape == (128, 4)
    np.testing.assert_allclose(native_freqs, reference_freqs)
    np.testing.assert_allclose(native, reference, rtol=2e-10, atol=2e-10)


def test_native_multitaper_zero_padding_matches_reference() -> None:
    _native()
    rng = np.random.default_rng(24)
    values = rng.standard_normal((17, 3))
    kwargs = {
        "fs": 250.0,
        "nw": 2.5,
        "n_tapers": 3,
        "nfft": 128,
        "weighting": "adaptive",
    }

    native, native_freqs = multitaper_psd(values, **kwargs)
    reference, reference_freqs = _reference_multitaper_psd(values, **kwargs)

    np.testing.assert_allclose(native_freqs, reference_freqs)
    np.testing.assert_allclose(native, reference, rtol=2e-10, atol=2e-10)


def test_multitaper_processor_reset_reproduces_cold_start() -> None:
    _native()
    rng = np.random.default_rng(20)
    values = rng.standard_normal((96, 4))
    processor = MultitaperPsdProcessor(
        96,
        4,
        500.0,
        nw=3.0,
        n_tapers=5,
        nfft=128,
        weighting="adaptive",
    )

    cold, freqs = processor.process(values)
    processor.process(values * 0.7)
    processor.reset_adaptive_state()
    reset, reset_freqs = processor.process(values)

    np.testing.assert_allclose(reset_freqs, freqs)
    np.testing.assert_allclose(reset, cold, rtol=1e-12, atol=1e-12)


def test_multitaper_processor_warm_start_uses_current_frame() -> None:
    _native()
    rng = np.random.default_rng(21)
    first = rng.standard_normal((96, 3))
    second = rng.standard_normal((96, 3)) + 0.5
    processor = MultitaperPsdProcessor(
        96,
        3,
        500.0,
        nw=3.0,
        n_tapers=5,
        nfft=128,
        weighting="adaptive",
    )

    first_psd, _ = processor.process(first)
    warm_second, _ = processor.process(second)
    processor.reset_adaptive_state()
    cold_second, _ = processor.process(second)

    assert not np.allclose(warm_second, first_psd)
    assert np.linalg.norm(warm_second - cold_second) < 1e-3 * np.linalg.norm(
        first_psd - cold_second
    )
    np.testing.assert_allclose(warm_second, cold_second, rtol=2e-4, atol=5e-10)


def test_multitaper_processor_zero_power_channel_clears_warm_state() -> None:
    _native()
    rng = np.random.default_rng(22)
    first = rng.standard_normal((96, 2))
    zeroed = rng.standard_normal((96, 2))
    zeroed[:, 0] = 0.0
    next_frame = rng.standard_normal((96, 2))
    processor = MultitaperPsdProcessor(
        96,
        2,
        500.0,
        nw=3.0,
        n_tapers=5,
        nfft=128,
        weighting="adaptive",
    )

    processor.process(first)
    zero_psd, _ = processor.process(zeroed)
    warm_next, _ = processor.process(next_frame)
    processor.reset_adaptive_state()
    cold_next, _ = processor.process(next_frame)

    np.testing.assert_array_equal(zero_psd[:, 0], np.zeros(65))
    np.testing.assert_allclose(warm_next[:, 0], cold_next[:, 0], rtol=1e-12, atol=1e-12)
