#!/usr/bin/env python3

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.signal.spectral import freq_vector, scale_spectrum


def test_scale_spectrum_does_not_expose_fake_backend() -> None:
    assert "backend" not in inspect.signature(scale_spectrum).parameters


def test_freq_vector_even_odd_whole_and_half_ranges() -> None:
    np.testing.assert_allclose(
        freq_vector(4),
        [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0],
    )
    np.testing.assert_allclose(
        freq_vector(5, freq_range="half"),
        [0.0, 2.0 * math.pi / 5.0, 4.0 * math.pi / 5.0],
    )
    np.testing.assert_allclose(
        freq_vector(4, centered=True),
        [-math.pi / 2.0, 0.0, math.pi / 2.0, math.pi],
    )
    np.testing.assert_allclose(
        freq_vector(5, centered=True),
        [-4.0 * math.pi / 5.0, -2.0 * math.pi / 5.0, 0.0, 2.0 * math.pi / 5.0, 4.0 * math.pi / 5.0],
    )


def test_freq_vector_uses_hertz_and_centered_half_legacy_grid() -> None:
    np.testing.assert_allclose(
        freq_vector(8, 80.0, freq_range="half"),
        [0.0, 10.0, 20.0, 30.0, 40.0],
    )
    np.testing.assert_allclose(
        freq_vector(8, 80.0, centered=True, freq_range="half"),
        [-20.0, -10.0, 0.0, 10.0, 20.0],
    )
    np.testing.assert_allclose(
        freq_vector(6, 60.0, centered=True, freq_range="half"),
        [-10.0, 0.0, 10.0],
    )


def test_freq_vector_validates_parameters() -> None:
    with pytest.raises(ValidationError):
        freq_vector(0)
    with pytest.raises(ValidationError):
        freq_vector(8, 0)
    with pytest.raises(ValidationError):
        freq_vector(8, centered=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        freq_vector(8, freq_range="quarter")  # type: ignore[arg-type]


@pytest.mark.parametrize("nfft", [5, 8])
def test_scale_spectrum_one_sided_power_scaling(nfft: int) -> None:
    spectrum = np.ones((nfft, 2))
    freqs = freq_vector(nfft, 100.0)

    scaled, output_freqs, units = scale_spectrum(
        spectrum,
        freqs,
        sides="one-sided",
        nfft=nfft,
        fs=100.0,
        estimate="power",
    )

    expected_length = nfft // 2 + 1
    expected = np.full((expected_length, 2), 2.0)
    expected[0] = 1.0
    if nfft % 2 == 0:
        expected[-1] = 1.0
    np.testing.assert_array_equal(scaled, expected)
    np.testing.assert_array_equal(output_freqs, freqs[:expected_length])
    assert units == "Hz"


def test_scale_spectrum_psd_units_and_nonzero_axis() -> None:
    spectrum = np.arange(16.0).reshape(2, 8) + 1.0
    freqs = freq_vector(8)

    scaled, output_freqs, units = scale_spectrum(
        spectrum,
        freqs,
        sides="two-sided",
        estimate="psd",
        axis=1,
    )

    np.testing.assert_allclose(scaled, spectrum / (2.0 * math.pi))
    np.testing.assert_array_equal(output_freqs, freqs)
    assert units == "rad/sample"
    np.testing.assert_array_equal(spectrum, np.arange(16.0).reshape(2, 8) + 1.0)


def test_scale_spectrum_validates_shape_and_backend() -> None:
    with pytest.raises(ValidationError, match="frequencies length"):
        scale_spectrum(np.ones(8), np.ones(7))
    with pytest.raises(ValidationError, match="expected nfft"):
        scale_spectrum(np.ones(8), np.ones(8), nfft=16)
