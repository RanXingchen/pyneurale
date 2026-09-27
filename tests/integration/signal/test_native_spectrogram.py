#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest

from neurale.signal import multitaper_psd, multitaper_spectrogram


def _native() -> object:
    native = importlib.import_module("neurale._native").signal.spectral
    return native._MultitaperPsdProcessor


@pytest.mark.parametrize("weighting", ["unity", "eigen", "adaptive"])
def test_native_multitaper_spectrogram_matches_per_frame_psd(
    weighting: str,
) -> None:
    _native()
    rng = np.random.default_rng(54)
    values = rng.standard_normal((220, 3))
    kwargs = {
        "fs": 250.0,
        "window_size": 0.256,
        "shift": 0.128,
        "n_windows": 5,
        "nfft": 128,
        "nw": 3.0,
        "n_tapers": 5,
        "weighting": weighting,
        "detrend": "mean",
        "db": True,
        "smoothing_width": 1.5,
    }

    result, freqs, times = multitaper_spectrogram(values, **kwargs)
    reference = []
    for start in (0, 32, 64, 96, 128):
        frame = values[start : start + 64]
        psd, reference_freqs = multitaper_psd(
            frame,
            kwargs["fs"],
            nw=kwargs["nw"],
            n_tapers=kwargs["n_tapers"],
            nfft=kwargs["nfft"],
            weighting=weighting,
            detrend=kwargs["detrend"],
            db=kwargs["db"],
            smoothing_width=kwargs["smoothing_width"],
        )
        reference.append(psd)

    np.testing.assert_allclose(result, np.stack(reference, axis=-1), rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(freqs, reference_freqs)
    np.testing.assert_allclose(times, [0.128, 0.256, 0.384, 0.512, 0.64])
