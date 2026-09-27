#!/usr/bin/env python3

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.exceptions import ValidationError
from neurale.signal import (
    IirCoefficients,
    IirFilter,
    SosCoefficients,
    SosFilter,
    StateSpaceCoefficients,
    ZpkCoefficients,
    bessel,
    butter,
    design_iir,
    ellip,
    iir_filter,
    notch,
    sos_filter,
)


def _response(coefs):
    if isinstance(coefs, IirCoefficients):
        return scipy_signal.freqz(coefs.b, coefs.a, worN=1024)[1]
    if isinstance(coefs, SosCoefficients):
        return scipy_signal.sosfreqz(coefs.sos, worN=1024)[1]
    if isinstance(coefs, ZpkCoefficients):
        return scipy_signal.freqz_zpk(coefs.z, coefs.p, coefs.k, worN=1024)[1]
    if isinstance(coefs, StateSpaceCoefficients):
        return _state_response(coefs, 1024)
    raise AssertionError(type(coefs))


def _state_response(coefs, size):
    freqs = np.linspace(0.0, np.pi, size, endpoint=False)
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
    return scipy_signal.freqz_zpk(z, p, k, worN=1024)[1]


@pytest.mark.parametrize("output", ["tf", "sos", "zpk", "ss"])
def test_notch_all_outputs_match_scipy_and_are_typed(output) -> None:
    result = notch(50, q=30, fs=1000, output=output)
    reference_b, reference_a = scipy_signal.iirnotch(50, 30, fs=1000)
    expected = scipy_signal.freqz(reference_b, reference_a, worN=1024)[1]
    np.testing.assert_allclose(_response(result), expected, atol=2e-12)
    assert result.fs == 1000


@pytest.mark.parametrize(
    ("designer", "kind", "keywords", "tol"),
    [
        (butter, "butterworth", {}, 2e-10),
        (bessel, "bessel", {}, 2e-8),
        (ellip, "elliptic", {"rp": 0.5, "rs": 40}, 2e-8),
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
def test_classic_iir_designers_cover_every_band_and_output(
    designer, kind, keywords, tol, band, cutoff, output
) -> None:
    result = designer(4, cutoff, band=band, fs=1000, output=output, **keywords)
    expected = _scipy_response(kind, 4, cutoff, band, 1000, **keywords)
    np.testing.assert_allclose(_response(result), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    ("band", "cutoff"),
    [
        ("lowpass", 300),
        ("highpass", 300),
        ("bandpass", [300, 1200]),
        ("bandstop", [300, 1200]),
    ],
)
def test_high_order_butter_state_space_avoids_polynomial_instability(band, cutoff) -> None:
    state = butter(10, cutoff, band=band, fs=30000, output="ss")
    zpk = butter(10, cutoff, band=band, fs=30000, output="zpk")
    np.testing.assert_allclose(_response(state), _response(zpk), rtol=2e-11, atol=2e-11)
    assert np.max(np.abs(state.a)) < 2.0


def test_state_space_design_uses_normalized_state_coordinates() -> None:
    physical = butter(10, 300, fs=30000, output="ss")
    normalized = butter(10, 0.02, fs=2, output="ss")
    np.testing.assert_allclose(physical.a, normalized.a, atol=2e-15)
    np.testing.assert_allclose(physical.b, normalized.b, atol=2e-15)
    np.testing.assert_allclose(physical.c, normalized.c, atol=2e-15)
    assert physical.d == pytest.approx(normalized.d, abs=2e-15)


@pytest.mark.parametrize(
    ("kind", "keywords"),
    [
        ("butterworth", {}),
        ("bessel", {}),
        ("elliptic", {"rp": 0.5, "rs": 40}),
    ],
)
def test_design_iir_facade_matches_dedicated_designers(kind, keywords) -> None:
    result = design_iir(kind, 5, [0.2, 0.4], band="bandpass", output="sos", **keywords)
    dedicated = {
        "butterworth": butter,
        "bessel": bessel,
        "elliptic": ellip,
    }[kind](5, [0.2, 0.4], band="bandpass", output="sos", **keywords)
    assert result == dedicated


def test_design_iir_notch_facade_and_bandwidth_path() -> None:
    by_quality = design_iir("notch", cutoff=0.32, q=30, output="tf")
    by_bandwidth = notch(0.32, bandwidth=0.32 / 30, output="tf")
    np.testing.assert_allclose(by_quality.b, by_bandwidth.b)
    np.testing.assert_allclose(by_quality.a, by_bandwidth.a)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: notch(0.2), "exactly one"),
        (lambda: notch(0.2, q=30, bandwidth=0.01), "exactly one"),
        (lambda: butter(4, [0.2, 0.3], band="lowpass"), "cutoff"),
        (lambda: butter(4, [0.4, 0.2], band="bandpass"), "increasing"),
        (lambda: ellip(4, 0.2, rp=40, rs=20), "less than"),
        (lambda: design_iir("elliptic", 4, 0.2), "requires rp"),
    ],
)
def test_iir_design_validation_paths(call, message) -> None:
    with pytest.raises(ValidationError, match=message):
        call()


def test_iir_designers_do_not_expose_backend() -> None:
    functions = (notch, butter, bessel, ellip, design_iir)
    assert all("backend" not in inspect.signature(function).parameters for function in functions)


def test_bessel_rejects_orders_above_native_limit() -> None:
    with pytest.raises(ValidationError, match="16"):
        bessel(17, 0.2, output="zpk")


@pytest.mark.parametrize(
    ("designer", "keywords"),
    [
        (butter, {}),
        (bessel, {}),
        (ellip, {"rp": 0.5, "rs": 40}),
    ],
)
def test_designed_coefficients_work_offline_and_streaming(designer, keywords) -> None:
    rng = np.random.default_rng(110)
    values = rng.standard_normal((257, 3))
    for output, function, processor_type in (
        ("tf", iir_filter, IirFilter),
        ("sos", sos_filter, SosFilter),
    ):
        coefs = designer(5, 0.2, output=output, **keywords)
        expected = function(values, coefs)
        actual = values.copy()
        processor = processor_type(coefs, n_channels=3)
        for chunk in (actual[:17], actual[17:91], actual[91:]):
            assert processor.process(chunk) is chunk
        np.testing.assert_allclose(actual, expected, atol=2e-11)


def test_typed_representations_are_immutable_and_validate() -> None:
    zpk = ZpkCoefficients([], [-0.5], 1, 1000)
    state = StateSpaceCoefficients([[-0.5]], [1], [1], 0, 1000)
    assert not zpk.p.flags.writeable
    assert not state.a.flags.writeable
    with pytest.raises(ValueError):
        zpk.p[0] = 0
    with pytest.raises(ValidationError, match="shapes"):
        StateSpaceCoefficients(np.eye(2), [1], [1, 2], 0)


def test_zpk_coefficients_use_standard_z_p_k_names() -> None:
    zpk = ZpkCoefficients([], [-0.5], 1.0)
    assert not hasattr(zpk, "zeros")
    assert not hasattr(zpk, "poles")
    assert not hasattr(zpk, "gain")
