#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.exceptions import ValidationError
from neurale.signal import (
    bilinear_ss,
    bilinear_tf,
    bilinear_zpk,
    lp2bp_ss,
    lp2bs_ss,
    lp2hp_ss,
    lp2lp_ss,
    ss2tf,
    ss2zpk,
    tf2ss,
    tf2zpk,
    zpk2sos,
    zpk2ss,
    zpk2tf,
)


def _native() -> object:
    native = importlib.import_module("neurale._native").signal
    return native.representations


def _prototype(order=7):
    return scipy_signal.butter(order, 1.0, analog=True, output="zpk")


def _assert_transfer_equivalent(left, right, tol=2e-9) -> None:
    points = np.asarray([0.17 + 0.23j, -0.31 + 0.47j, 0.73 - 0.19j, 1.21 + 0.11j])
    left_response = np.polyval(left[0], points) / np.polyval(left[1], points)
    right_response = np.polyval(right[0], points) / np.polyval(right[1], points)
    np.testing.assert_allclose(left_response, right_response, rtol=tol, atol=tol)


def _assert_roots_close(actual, expected, tol=2e-8) -> None:
    from scipy.optimize import linear_sum_assignment

    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    assert actual.size == expected.size
    distances = np.abs(actual[:, None] - expected[None, :])
    rows, columns = linear_sum_assignment(distances)
    assert np.max(distances[rows, columns], initial=0.0) <= tol


def test_native_basic_representation_conversions_match_scipy() -> None:
    _native()
    z, p, k = _prototype()
    num, den = scipy_signal.zpk2tf(z, p, k)

    _assert_transfer_equivalent(zpk2tf(z, p, k), (num, den))
    _assert_transfer_equivalent(ss2tf(*zpk2ss(z, p, k)), (num, den))
    _assert_transfer_equivalent(ss2tf(*tf2ss(num, den)), (num, den))
    _assert_transfer_equivalent(scipy_signal.sos2tf(zpk2sos(z, p, k)), (num, den))


def test_native_root_based_conversions_match_scipy() -> None:
    _native()
    z, p, k = _prototype()
    num, den = scipy_signal.zpk2tf(z, p, k)
    a, b, c, d = scipy_signal.tf2ss(num, den)

    for actual_z, actual_p, actual_k in (
        tf2zpk(num, den),
        ss2zpk(a, b[:, 0], c[0], d[0, 0]),
    ):
        _assert_roots_close(actual_z, z)
        _assert_roots_close(actual_p, p)
        np.testing.assert_allclose(actual_k, k, rtol=2e-10)


@pytest.mark.parametrize(
    ("function", "scipy_function", "arguments"),
    [
        (lp2lp_ss, scipy_signal.lp2lp, (3.0,)),
        (lp2hp_ss, scipy_signal.lp2hp, (3.0,)),
        (lp2bp_ss, scipy_signal.lp2bp, (3.0, 1.5)),
        (lp2bs_ss, scipy_signal.lp2bs, (3.0, 1.5)),
    ],
)
def test_native_lp_state_transforms_match_scipy(function, scipy_function, arguments) -> None:
    _native()
    z, p, k = _prototype(5)
    num, den = scipy_signal.zpk2tf(z, p, k)
    state = tf2ss(num, den)
    native = function(*state, *arguments)
    expected = scipy_function(num, den, *arguments)
    _assert_transfer_equivalent(ss2tf(*native), expected)


def test_native_bilinear_forms_match_scipy() -> None:
    _native()
    z, p, k = _prototype(5)
    num, den = scipy_signal.zpk2tf(z, p, k)
    state = tf2ss(num, den)
    expected = scipy_signal.zpk2tf(*scipy_signal.bilinear_zpk(z, p, k, fs=1000.0))

    _assert_transfer_equivalent(bilinear_tf(num, den, 1000.0), expected, tol=2e-8)
    _assert_transfer_equivalent(zpk2tf(*bilinear_zpk(z, p, k, 1000.0)), expected, tol=2e-8)
    _assert_transfer_equivalent(ss2tf(*bilinear_ss(*state, 1000.0)), expected, tol=2e-8)


def test_native_bilinear_raw_binding_owns_outputs_and_validates() -> None:
    _native()
    native = importlib.import_module("neurale._native").signal.representations
    z, p, k = _prototype(4)
    num, den = scipy_signal.zpk2tf(z, p, k)
    a, b, c, d = scipy_signal.tf2ss(num, den)

    zd, pd, _ = native.bilinear_zpk(z.tolist(), p, k, 1000.0, None)
    ad, bd, cd, _ = native.bilinear_ss(a, b[:, 0], c[0], d[0, 0], 1000.0, None)
    numd, dend = native.bilinear_tf(num, den, 1000.0, None)

    for output in (zd, pd, ad, bd, cd, numd, dend):
        assert output.flags.owndata

    with pytest.raises(ValueError, match="1D"):
        native.bilinear_zpk(np.ones((1, 1), complex), p, k, 1000.0, None)
    with pytest.raises(ValueError, match="finite"):
        native.bilinear_zpk([np.nan + 0j], p, k, 1000.0, None)
    with pytest.raises(ValueError, match="state-space shapes"):
        native.bilinear_ss(a, b[:-1, 0], c[0], d[0, 0], 1000.0, None)
    with pytest.raises(ValueError, match="positive"):
        native.bilinear_tf(num, den, 0.0, None)


def test_native_bilinear_public_api_preserves_validation_errors() -> None:
    _native()
    with pytest.raises(ValidationError, match="1D"):
        bilinear_zpk(np.ones((1, 1), complex), [-1], 1, 100)
    with pytest.raises(ValidationError, match="state-space shapes"):
        bilinear_ss(np.eye(2), [1], [1, 2], 0, 100)
    with pytest.raises(ValidationError, match="positive"):
        bilinear_tf([1], [1, 1], 0)
