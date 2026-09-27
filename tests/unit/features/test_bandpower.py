#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, FeatureMatrix, SignalArray
from neurale.exceptions import ValidationError
from neurale.features import (
    bandpower_features,
    hilbert_envelope_features,
    lmp_features,
)


def _signal(data: np.ndarray, rate: float, *, t0: float = 2.0) -> SignalArray:
    channels = ChannelTable(
        [
            ChannelInfo("motor-a", 10, "ecog", "uV"),
            ChannelInfo("motor-b", 20, "ecog", "uV"),
        ]
    )
    return SignalArray(
        data=data,
        fs=rate,
        time=None,
        t0=t0,
        clock=None,
        channels=channels,
        unit="uV",
        name="motor-cortex",
    )


def _tones(rate: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(round(rate * duration)) / rate
    data = np.column_stack(
        (
            2.0 * np.sin(2.0 * np.pi * 10.0 * time) + 0.1 * np.sin(2.0 * np.pi * 40.0 * time),
            0.1 * np.sin(2.0 * np.pi * 10.0 * time) + 3.0 * np.sin(2.0 * np.pi * 40.0 * time),
        )
    )
    return time, data


def test_bandpower_separates_tones_and_keeps_metadata() -> None:
    rate = 256.0
    _, data = _tones(rate, 3.0)
    source = _signal(data, rate, t0=7.5)
    window_size = 0.5
    shift = 0.25

    result = bandpower_features(
        source,
        {"alpha": (8.0, 13.0), "gamma": (35.0, 45.0)},
        window_size,
        shift,
        weighting="unity",
    )

    assert isinstance(result, FeatureMatrix)
    assert result.feature_names == [
        "alpha:motor-a",
        "alpha:motor-b",
        "gamma:motor-a",
        "gamma:motor-b",
    ]
    assert np.mean(result.data[:, 0]) > 20.0 * np.mean(result.data[:, 1])
    assert np.mean(result.data[:, 3]) > 20.0 * np.mean(result.data[:, 2])
    n_expected = 1 + (source.n_samples - 128) // 64
    expected_time = 7.5 + 0.25 + np.arange(n_expected) * shift
    np.testing.assert_allclose(result.time, expected_time)
    assert result.window_size == window_size
    assert result.shift == shift
    assert result.fs == 1.0 / shift
    assert result.source_signal == "motor-cortex"


def test_hilbert_envelope_keeps_band_channel_order() -> None:
    rate = 256.0
    _, data = _tones(rate, 4.0)
    source = _signal(data, rate)

    result = hilbert_envelope_features(
        source,
        {"alpha": (8.0, 13.0), "gamma": (35.0, 45.0)},
        0.25,
        0.125,
    )

    assert result.feature_names == [
        "alpha:motor-a",
        "alpha:motor-b",
        "gamma:motor-a",
        "gamma:motor-b",
    ]
    interior = result.data[2:-2]
    assert np.mean(interior[:, 0]) > 10.0 * np.mean(interior[:, 1])
    assert np.mean(interior[:, 3]) > 10.0 * np.mean(interior[:, 2])
    np.testing.assert_allclose(
        result.time,
        2.0 + 0.125 + np.arange(result.n_frames) * 0.125,
    )


def test_lmp_features_follow_low_frequency_component() -> None:
    rate = 500.0
    time = np.arange(round(rate * 4.0)) / rate
    low = np.sin(2.0 * np.pi * 3.0 * time)
    data = np.column_stack(
        (
            low + np.sin(2.0 * np.pi * 100.0 * time),
            0.5 * low + 2.0 * np.sin(2.0 * np.pi * 100.0 * time),
        )
    )

    result = lmp_features(_signal(data, rate), cutoff=20.0, window_size=0.1, shift=0.05)

    assert result.feature_names == ["lmp:motor-a", "lmp:motor-b"]
    expected = np.array([np.mean(low[start : start + 50]) for start in range(0, low.size - 49, 25)])
    assert np.corrcoef(result.data[:, 0], expected)[0, 1] > 0.99
    assert np.corrcoef(result.data[:, 1], 0.5 * expected)[0, 1] > 0.99


def test_band_features_support_channel_major_ndarray() -> None:
    rate = 200.0
    time = np.arange(400) / rate
    channel_major = np.vstack((np.sin(2.0 * np.pi * 3.0 * time), np.cos(2.0 * np.pi * 3.0 * time)))

    result = lmp_features(
        channel_major,
        cutoff=15.0,
        window_size=0.2,
        shift=0.1,
        fs=rate,
        axis=1,
    )

    assert result.feature_names == ["lmp:ch000", "lmp:ch001"]
    assert result.source_signal is None
    np.testing.assert_allclose(result.time, 0.1 + np.arange(result.n_frames) * 0.1)


def test_band_features_reject_invalid_bands_and_complex_input() -> None:
    values = np.ones((256, 2))

    with pytest.raises(ValidationError, match="Nyquist"):
        hilbert_envelope_features(
            values,
            {"invalid": (20.0, 60.0)},
            0.5,
            0.25,
            fs=100.0,
        )
    with pytest.raises(ValidationError, match="real-valued"):
        bandpower_features(
            values.astype(complex),
            {"alpha": (8.0, 13.0)},
            0.5,
            0.25,
            fs=100.0,
        )
