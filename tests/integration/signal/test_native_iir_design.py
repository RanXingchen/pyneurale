#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.signal import (
    IirCoefficients,
    SosCoefficients,
    StateSpaceCoefficients,
    ZpkCoefficients,
    bessel,
    butter,
    ellip,
    notch,
)


def _response(coefs):
    if isinstance(coefs, ZpkCoefficients):
        return scipy_signal.freqz_zpk(coefs.z, coefs.p, coefs.k, worN=2048)[1]
    if isinstance(coefs, IirCoefficients):
        return scipy_signal.freqz(coefs.b, coefs.a, worN=2048)[1]
    if isinstance(coefs, SosCoefficients):
        return scipy_signal.sosfreqz(coefs.sos, worN=2048)[1]
    assert isinstance(coefs, StateSpaceCoefficients)
    freqs = np.linspace(0.0, np.pi, 2048, endpoint=False)
    identity = np.eye(coefs.a.shape[0])
    return np.asarray(
        [
            coefs.c
            @ np.linalg.solve(
                np.exp(1j * freq) * identity - coefs.a,
                coefs.b,
            )
            + coefs.d
            for freq in freqs
        ]
    )


def _scipy_response(kind, order, cutoff, band, fs, **keywords):
    if kind == "butterworth":
        z, p, k = scipy_signal.butter(order, cutoff, btype=band, fs=fs, output="zpk")
    elif kind == "bessel":
        z, p, k = scipy_signal.bessel(
            order,
            cutoff,
            btype=band,
            fs=fs,
            output="zpk",
            norm="phase",
        )
    else:
        z, p, k = scipy_signal.ellip(
            order,
            keywords["rp"],
            keywords["rs"],
            cutoff,
            btype=band,
            fs=fs,
            output="zpk",
        )
    return scipy_signal.freqz_zpk(z, p, k, worN=2048)[1]


@pytest.mark.parametrize(
    ("designer", "kind", "order", "keywords", "tol"),
    [
        (butter, "butterworth", 1, {}, 2e-12),
        (butter, "butterworth", 12, {}, 2e-11),
        (bessel, "bessel", 1, {}, 2e-12),
        (bessel, "bessel", 8, {}, 2e-10),
        (bessel, "bessel", 16, {}, 5e-9),
        (ellip, "elliptic", 1, {"rp": 0.5, "rs": 40}, 2e-12),
        (ellip, "elliptic", 7, {"rp": 0.5, "rs": 60}, 2e-9),
        (ellip, "elliptic", 10, {"rp": 1.0, "rs": 80}, 2e-8),
    ],
)
@pytest.mark.parametrize(
    ("band", "cutoff"),
    [
        ("lowpass", 100),
        ("highpass", 100),
        ("bandpass", [100, 220]),
        ("bandstop", [100, 220]),
    ],
)
def test_native_iir_design_matches_scipy_response(
    designer, kind, order, keywords, tol, band, cutoff
) -> None:
    native = designer(
        order,
        cutoff,
        band=band,
        fs=1000,
        output="zpk",
        **keywords,
    )
    expected = _scipy_response(kind, order, cutoff, band, 1000, **keywords)
    np.testing.assert_allclose(_response(native), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    ("designer", "kind", "keywords"),
    [
        (butter, "butterworth", {}),
        (bessel, "bessel", {}),
        (ellip, "elliptic", {"rp": 0.5, "rs": 40}),
    ],
)
@pytest.mark.parametrize(
    ("band", "cutoff"),
    [
        ("lowpass", 100),
        ("highpass", 100),
        ("bandpass", [100, 220]),
        ("bandstop", [100, 220]),
    ],
)
@pytest.mark.parametrize("output", ["tf", "sos", "zpk", "ss"])
def test_native_design_covers_every_output_conversion(
    designer, kind, keywords, band, cutoff, output
) -> None:
    native = designer(
        4,
        cutoff,
        band=band,
        fs=1000,
        output=output,
        **keywords,
    )
    expected = _scipy_response(kind, 4, cutoff, band, 1000, **keywords)
    np.testing.assert_allclose(_response(native), expected, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize(
    ("band", "cutoff"),
    [
        ("lowpass", 300),
        ("highpass", 300),
        ("bandpass", [300, 1200]),
        ("bandstop", [300, 1200]),
    ],
)
def test_native_high_order_butter_state_space_is_stable(band, cutoff) -> None:
    state = butter(10, cutoff, band=band, fs=30000, output="ss")
    zpk = butter(10, cutoff, band=band, fs=30000, output="zpk")
    np.testing.assert_allclose(_response(state), _response(zpk), rtol=2e-11, atol=2e-11)
    assert np.max(np.abs(state.a)) < 2.0


def test_native_state_space_uses_normalized_state_coordinates() -> None:
    physical = butter(10, 300, fs=30000, output="ss")
    normalized = butter(10, 0.02, fs=2, output="ss")
    np.testing.assert_allclose(physical.a, normalized.a, atol=2e-15)
    np.testing.assert_allclose(physical.b, normalized.b, atol=2e-15)
    np.testing.assert_allclose(physical.c, normalized.c, atol=2e-15)
    assert physical.d == pytest.approx(normalized.d, abs=2e-15)


@pytest.mark.parametrize("output", ["tf", "sos", "zpk", "ss"])
def test_native_notch_matches_scipy_for_all_outputs(output) -> None:
    native = notch(50, q=30, fs=1000, output=output)
    reference_b, reference_a = scipy_signal.iirnotch(50, 30, fs=1000)
    expected = scipy_signal.freqz(reference_b, reference_a, worN=2048)[1]
    np.testing.assert_allclose(_response(native), expected, atol=2e-12)
