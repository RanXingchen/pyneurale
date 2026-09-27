#!/usr/bin/env python3

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import ValidationError
from neurale.signal import multitaper_psd, welch_psd


def test_welch_psd_matches_scipy_reference() -> None:
    rng = np.random.default_rng(41)
    values = rng.standard_normal((1024, 3))
    window = scipy_signal.get_window("hann", 256, fftbins=False)
    expected_freqs, expected = scipy_signal.welch(
        values,
        fs=1000.0,
        window=window,
        nperseg=256,
        noverlap=128,
        nfft=512,
        detrend="constant",
        axis=0,
    )

    result, freqs = welch_psd(
        values,
        1000.0,
        nfft=512,
        window_length=256,
        overlap=0.5,
        detrend="mean",
    )

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(freqs, expected_freqs)


def test_welch_psd_supports_time_range_axis_db_and_smoothing() -> None:
    rng = np.random.default_rng(42)
    values = rng.standard_normal((2, 1000))

    linear, freqs = welch_psd(
        values,
        100.0,
        nfft=100,
        window_length=100,
        time_range=(2.0, 8.0),
        axis=1,
    )
    db, db_freqs = welch_psd(
        values,
        100.0,
        nfft=100,
        window_length=100,
        time_range=(2.0, 8.0),
        db=True,
        smoothing_width=2.0,
        axis=1,
    )

    assert linear.shape == (2, 51)
    assert db.shape == linear.shape
    np.testing.assert_array_equal(freqs, db_freqs)
    assert np.all(np.isfinite(db))


def test_welch_psd_uses_signal_array_sampling_metadata() -> None:
    source = SignalArray(
        data=np.arange(512.0).reshape(256, 2),
        fs=128.0,
        time=2.0 + np.arange(256) / 128.0,
        t0=2.0,
        clock=None,
        channels=ChannelTable.from_names(["ch000", "ch001"], unit=["uV", "mV"]),
        unit=["uV", "mV"],
        name="source",
    )

    result, freqs = welch_psd(
        source,
        nfft=64,
        window_length=64,
        time_range=(0.5, 1.5),
    )

    assert result.shape == (33, 2)
    assert freqs[-1] == pytest.approx(64.0)
    with pytest.raises(ValidationError, match="inconsistent"):
        welch_psd(source, 100.0, nfft=64)


def test_welch_psd_validates_windowing_contract() -> None:
    values = np.ones(128)
    with pytest.raises(ValidationError, match="window_length"):
        welch_psd(values, nfft=64, window_length=65)
    with pytest.raises(ValidationError, match="selected sample"):
        welch_psd(values, nfft=256, window_length=256)
    with pytest.raises(ValidationError, match="overlap"):
        welch_psd(values, nfft=64, overlap=1.0)
    with pytest.raises(ValidationError, match="time_range requires"):
        welch_psd(values, nfft=64, time_range=(0.0, 1.0))


def test_welch_psd_does_not_expose_fake_backend() -> None:
    assert "backend" not in inspect.signature(welch_psd).parameters


@pytest.mark.parametrize("incomplete", ["drop", "pad"])
def test_high_level_multitaper_matches_manual_window_average(
    incomplete: str,
) -> None:
    rng = np.random.default_rng(43)
    values = rng.standard_normal((150, 2))
    kwargs = {
        "fs": 200.0,
        "nw": 2.5,
        "n_tapers": 3,
        "nfft": 64,
        "weighting": "unity",
    }
    result, freqs = multitaper_psd(
        values,
        window_length=64,
        incomplete=incomplete,
        **kwargs,
    )

    segments = [values[:64], values[64:128]]
    if incomplete == "pad":
        padded = np.zeros((64, 2))
        padded[:22] = values[128:]
        segments.append(padded)
    expected = np.mean(
        [multitaper_psd(segment, **kwargs)[0] for segment in segments],
        axis=0,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
    assert freqs.shape == (33,)


def test_high_level_multitaper_validates_incomplete_windows() -> None:
    values = np.ones(150)
    with pytest.raises(ValidationError, match="integer number"):
        multitaper_psd(
            values,
            nw=2.5,
            window_length=64,
            incomplete="error",
        )
    with pytest.raises(ValidationError, match="exceeds"):
        multitaper_psd(
            values,
            nw=2.5,
            window_length=256,
            incomplete="drop",
        )


def test_multitaper_preprocessing_is_applied_per_window() -> None:
    constant_offsets = np.concatenate((np.ones(64) * 10.0, np.ones(64) * -7.0))

    result, _ = multitaper_psd(
        constant_offsets,
        100.0,
        nw=2.5,
        n_tapers=3,
        nfft=64,
        window_length=64,
        detrend="mean",
        weighting="unity",
    )

    np.testing.assert_allclose(result, 0.0, atol=1e-28)


def test_multitaper_time_range_db_and_smoothing() -> None:
    rng = np.random.default_rng(44)
    values = rng.standard_normal((400, 2))
    selected = values[100:300]
    kwargs = {
        "fs": 100.0,
        "nw": 2.5,
        "n_tapers": 3,
        "nfft": 64,
        "window_length": 50,
        "weighting": "unity",
    }

    ranged, freqs = multitaper_psd(
        values,
        time_range=(1.0, 3.0),
        db=True,
        smoothing_width=1.5,
        **kwargs,
    )
    expected, expected_freqs = multitaper_psd(
        selected,
        db=True,
        smoothing_width=1.5,
        **kwargs,
    )

    np.testing.assert_allclose(ranged, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(freqs, expected_freqs)
