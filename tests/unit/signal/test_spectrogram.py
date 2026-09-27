#!/usr/bin/env python3

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import (
    multitaper_psd,
    multitaper_spectrogram,
    stft_spectrogram,
)
from neurale.signal.spectral import multitaper as multitaper_module


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "flattop", "boxcar"])
def test_stft_spectrogram_matches_individual_scipy_frames(
    window: str,
) -> None:
    rng = np.random.default_rng(51)
    values = rng.standard_normal((256, 2))
    result, freqs, times = stft_spectrogram(
        values,
        100.0,
        window_size=0.64,
        shift=0.32,
        n_windows=4,
        nfft=128,
        window=window,
        detrend="mean",
    )
    analysis_window = scipy_signal.get_window(window, 64, fftbins=False)
    expected = []
    for start in (0, 32, 64, 96):
        frame = values[start : start + 64]
        frame = frame - frame.mean(axis=0, keepdims=True)
        expected_freqs, _, frame_psd = scipy_signal.spectrogram(
            frame,
            fs=100.0,
            window=analysis_window,
            nperseg=64,
            noverlap=0,
            nfft=128,
            detrend=False,
            axis=0,
            scaling="density",
            mode="psd",
        )
        expected.append(frame_psd[..., 0])

    np.testing.assert_allclose(result, np.stack(expected, axis=-1), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(freqs, expected_freqs)
    np.testing.assert_allclose(times, [0.32, 0.64, 0.96, 1.28])


def test_stft_spectrogram_axis_db_and_smoothing() -> None:
    rng = np.random.default_rng(52)
    values = rng.standard_normal((2, 200))

    result, freqs, times = stft_spectrogram(
        values,
        100.0,
        window_size=0.5,
        shift=0.25,
        n_windows=5,
        nfft=64,
        axis=1,
        db=True,
        smoothing_width=1.5,
    )

    assert result.shape == (33, 2, 5)
    assert freqs.shape == (33,)
    assert times.shape == (5,)
    assert np.all(np.isfinite(result))


def test_stft_spectrogram_returns_absolute_window_centers() -> None:
    source = SignalArray(
        data=np.arange(400.0).reshape(200, 2),
        fs=100.0,
        time=5.0 + np.arange(200) / 100.0,
        t0=5.0,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="source",
    )

    result, freqs, times = stft_spectrogram(
        source,
        window_size=0.5,
        shift=0.25,
        n_windows=3,
        start_time=0.5,
    )

    assert result.shape == (26, 2, 3)
    assert freqs[-1] == pytest.approx(50.0)
    np.testing.assert_allclose(times, [5.75, 6.0, 6.25])


def test_stft_spectrogram_without_explicit_time_uses_t0() -> None:
    source = SignalArray(
        data=np.arange(400.0).reshape(200, 2),
        fs=100.0,
        time=None,
        t0=3.5,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="source",
    )

    _, _, times = stft_spectrogram(
        source,
        window_size=0.5,
        shift=0.25,
        n_windows=2,
        start_time=0.5,
    )

    np.testing.assert_allclose(times, [4.25, 4.5])


def test_spectrogram_time_reference_and_hz_smoothing() -> None:
    values = np.arange(200.0)
    _, freqs, starts = stft_spectrogram(
        values,
        100.0,
        window_size=0.5,
        shift=0.25,
        n_windows=2,
        time_reference="start",
        smoothing_width_hz=4.0,
    )
    _, _, ends = stft_spectrogram(
        values,
        100.0,
        window_size=0.5,
        shift=0.25,
        n_windows=2,
        time_reference="end",
    )
    np.testing.assert_allclose(starts, [0.0, 0.25])
    np.testing.assert_allclose(ends, [0.5, 0.75])
    assert freqs.size == 26
    with pytest.raises(ValidationError, match="mutually exclusive"):
        stft_spectrogram(
            values,
            100.0,
            window_size=0.5,
            shift=0.25,
            smoothing_width=1.0,
            smoothing_width_hz=2.0,
        )


def test_stft_spectrogram_validates_frame_settings() -> None:
    values = np.ones(100)
    with pytest.raises(ValidationError, match="shift"):
        stft_spectrogram(values, 100.0, window_size=0.5, shift=0.6)
    with pytest.raises(ValidationError, match="available"):
        stft_spectrogram(
            values,
            100.0,
            window_size=0.5,
            shift=0.25,
            n_windows=4,
            start_time=0.5,
        )
    with pytest.raises(ValidationError, match="sampling rate"):
        stft_spectrogram(values, window_size=10.0, shift=5.0)
    with pytest.raises(ValidationError, match="nfft"):
        stft_spectrogram(
            values,
            100.0,
            window_size=0.5,
            shift=0.25,
            nfft=32,
        )


def test_stft_spectrogram_does_not_expose_fake_backend() -> None:
    assert "backend" not in inspect.signature(stft_spectrogram).parameters


def test_multitaper_spectrogram_matches_per_frame_psd() -> None:
    rng = np.random.default_rng(53)
    values = rng.standard_normal((160, 2))
    result, freqs, times = multitaper_spectrogram(
        values,
        200.0,
        window_size=0.32,
        shift=0.16,
        n_windows=4,
        nfft=64,
        nw=2.5,
        n_tapers=3,
        weighting="unity",
        detrend="linear",
    )
    expected = []
    for start in (0, 32, 64, 96):
        frame = scipy_signal.detrend(values[start : start + 64], axis=0, type="linear")
        frame_psd, expected_freqs = multitaper_psd(
            frame,
            200.0,
            nw=2.5,
            n_tapers=3,
            nfft=64,
            weighting="unity",
        )
        expected.append(frame_psd)

    np.testing.assert_allclose(result, np.stack(expected, axis=-1), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(freqs, expected_freqs)
    np.testing.assert_allclose(times, [0.16, 0.32, 0.48, 0.64])


def test_multitaper_spectrogram_without_time_uses_t0() -> None:
    source = SignalArray(
        data=np.arange(256.0).reshape(128, 2),
        fs=200.0,
        time=None,
        t0=10.0,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="source",
    )

    _, _, times = multitaper_spectrogram(
        source,
        window_size=0.32,
        shift=0.16,
        n_windows=2,
        nfft=64,
        nw=2.5,
        n_tapers=3,
        weighting="unity",
    )

    np.testing.assert_allclose(times, [10.16, 10.32])


def test_multitaper_spectrogram_does_not_expose_backend() -> None:
    assert "backend" not in inspect.signature(multitaper_spectrogram).parameters


def test_multitaper_spectrogram_native_backend_error(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(multitaper_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        multitaper_spectrogram(
            np.ones(128),
            100.0,
            window_size=0.64,
            shift=0.32,
            nw=2.5,
        )
