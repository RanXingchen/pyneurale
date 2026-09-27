#!/usr/bin/env python3

from __future__ import annotations

import inspect

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
from neurale.signal.filtering._representation_numerics import (
    polynomial_from_roots,
    polynomial_roots,
    sort_conjugate_pairs,
)


def _prototype(order=5):
    return scipy_signal.butter(order, 1.0, analog=True, output="zpk")


def _assert_same_transfer(left, right, *, tol=1e-9) -> None:
    points = np.asarray([0.17 + 0.23j, -0.31 + 0.47j, 0.73 - 0.19j, 1.21 + 0.11j])
    np.testing.assert_allclose(
        np.polyval(left[0], points) / np.polyval(left[1], points),
        np.polyval(right[0], points) / np.polyval(right[1], points),
        rtol=tol,
        atol=tol,
    )


def _assert_roots_close(actual, expected, *, tol=1e-8) -> None:
    from scipy.optimize import linear_sum_assignment

    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    assert actual.size == expected.size
    distances = np.abs(actual[:, None] - expected[None, :])
    rows, columns = linear_sum_assignment(distances)
    assert np.max(distances[rows, columns], initial=0.0) <= tol


def test_private_root_numerics_are_deterministic() -> None:
    roots = np.asarray([2.0, -1 + 3j, -4.0, -1 - 3j])
    ordered = sort_conjugate_pairs(roots[::-1], "roots")
    np.testing.assert_array_equal(ordered, [-4.0, 2.0, -1 - 3j, -1 + 3j])
    coefs = polynomial_from_roots(roots)
    recovered = polynomial_roots(coefs)
    np.testing.assert_allclose(
        np.sort_complex(recovered),
        np.sort_complex(roots),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zpk_transfer_function_round_trip() -> None:
    z, p, k = _prototype()
    num, den = zpk2tf(z, p, k)
    expected_num, expected_den = scipy_signal.zpk2tf(z, p, k)
    np.testing.assert_allclose(num[-expected_num.size :], expected_num, atol=1e-13)
    np.testing.assert_allclose(den, expected_den, atol=1e-13)

    actual_z, actual_p, actual_k = tf2zpk(num, den)
    _assert_roots_close(actual_z, z)
    _assert_roots_close(actual_p, p)
    assert actual_k == pytest.approx(k)


def test_state_space_conversions_preserve_transfer_function() -> None:
    z, p, k = _prototype()
    expected = scipy_signal.zpk2tf(z, p, k)
    a, b, c, d = zpk2ss(z, p, k)
    _assert_same_transfer(ss2tf(a, b, c, d), expected)

    actual_z, actual_p, actual_k = ss2zpk(a, b, c, d)
    _assert_same_transfer(zpk2tf(actual_z, actual_p, actual_k), expected)

    round_a, round_b, round_c, round_d = tf2ss(*expected)
    _assert_same_transfer(ss2tf(round_a, round_b, round_c, round_d), expected)


def test_zpk_to_sos_preserves_transfer_function() -> None:
    z, p, k = _prototype()
    sos = zpk2sos(z, p, k)
    _assert_same_transfer(scipy_signal.sos2tf(sos), scipy_signal.zpk2tf(z, p, k))
    np.testing.assert_array_equal(
        zpk2sos([], [], 2.0),
        [[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    )


@pytest.mark.parametrize(
    ("transform", "scipy_transform", "arguments"),
    [
        (lp2lp_ss, scipy_signal.lp2lp, (3.0,)),
        (lp2hp_ss, scipy_signal.lp2hp, (3.0,)),
        (lp2bp_ss, scipy_signal.lp2bp, (3.0, 1.5)),
        (lp2bs_ss, scipy_signal.lp2bs, (3.0, 1.5)),
    ],
)
def test_analog_lowpass_transforms_preserve_expected_transfer(
    transform, scipy_transform, arguments
) -> None:
    z, p, k = _prototype()
    num, den = scipy_signal.zpk2tf(z, p, k)
    state = tf2ss(num, den)
    transformed = transform(*state, *arguments)
    _assert_same_transfer(ss2tf(*transformed), scipy_transform(num, den, *arguments))


def test_bilinear_forms_match_scipy_transfer() -> None:
    z, p, k = _prototype()
    num, den = scipy_signal.zpk2tf(z, p, k)
    expected = scipy_signal.zpk2tf(*scipy_signal.bilinear_zpk(z, p, k, fs=100.0))

    _assert_same_transfer(bilinear_tf(num, den, 100.0), expected)

    zd, pd, kd = bilinear_zpk(z, p, k, 100.0)
    _assert_same_transfer(zpk2tf(zd, pd, kd), expected)

    digital_ss = bilinear_ss(*tf2ss(num, den), 100.0)
    _assert_same_transfer(ss2tf(*digital_ss), expected)


def test_bilinear_prewarp_matches_requested_frequency() -> None:
    num = np.asarray([1.0])
    den = np.asarray([1.0, 1.0])
    digital_num, digital_den = bilinear_tf(num, den, 100.0, prewarp_freq=10.0)
    _, analog_response = scipy_signal.freqs(num, den, [2.0 * np.pi * 10.0])
    _, digital_response = scipy_signal.freqz(
        digital_num, digital_den, worN=[2.0 * np.pi * 10.0 / 100.0]
    )
    np.testing.assert_allclose(digital_response, analog_response, rtol=1e-13, atol=1e-13)


def test_representation_validation_rejects_invalid_systems() -> None:
    with pytest.raises(ValidationError, match="conjugate"):
        zpk2tf([1 + 2j], [-1], 1.0)
    with pytest.raises(ValidationError, match="proper"):
        zpk2sos([0, 1], [-1], 1.0)
    with pytest.raises(ValidationError, match="state-space"):
        ss2zpk(np.ones((2, 3)), [1, 1], [1, 1], 0)
    with pytest.raises(ValidationError, match="nonzero"):
        tf2ss([1], [0, 0])
    with pytest.raises(ValidationError, match="singular"):
        lp2hp_ss(np.zeros((1, 1)), [1], [1], 0, 1)
    with pytest.raises(ValidationError, match="Nyquist"):
        bilinear_zpk([], [-1], 1, 100, prewarp_freq=50)


def test_representation_functions_do_not_expose_backend() -> None:
    functions = (
        zpk2tf,
        ss2tf,
        zpk2ss,
        tf2ss,
        tf2zpk,
        ss2zpk,
        zpk2sos,
        lp2lp_ss,
        lp2hp_ss,
        lp2bp_ss,
        lp2bs_ss,
        bilinear_zpk,
        bilinear_ss,
        bilinear_tf,
    )
    assert all("backend" not in inspect.signature(function).parameters for function in functions)


@pytest.mark.parametrize("order", [2, 5, 10, 16])
def test_random_stable_system_round_trip_on_frequency_grid(order: int) -> None:
    rng = np.random.default_rng(order)
    poles = []
    for _ in range(order // 2):
        value = -rng.uniform(0.1, 2.0) + 1j * rng.uniform(0.1, 2.0)
        poles.extend((value, np.conjugate(value)))
    if order % 2:
        poles.append(-rng.uniform(0.1, 2.0))
    n_zeros = max(0, order - 2)
    zeros = poles[: n_zeros - (n_zeros % 2)]
    if n_zeros % 2:
        zeros.append(poles[-1])
    num, den = zpk2tf(zeros, poles, 0.75)
    recovered_tf = zpk2tf(*tf2zpk(num, den))
    _assert_same_transfer(recovered_tf, (num, den), tol=2e-8)


def test_repeated_and_near_conjugate_roots() -> None:
    roots = np.asarray(
        [
            -1 + 1e-10j,
            -1 - 1e-10j,
            -2 + 0.5j,
            -2 - 0.5j,
            -2 + 0.5j,
            -2 - 0.5j,
        ]
    )
    coefs = polynomial_from_roots(roots)
    assert np.all(np.isfinite(coefs))
    recovered = polynomial_roots(coefs)
    from scipy.optimize import linear_sum_assignment

    distances = np.abs(recovered[:, None] - roots[None, :])
    rows, columns = linear_sum_assignment(distances)
    assert np.max(distances[rows, columns]) < 2e-5


def test_zero_order_and_zero_transfer_function_contracts() -> None:
    a, b, c, d = tf2ss([2.0], [4.0])
    assert a.shape == (0, 0)
    assert b.shape == (0,)
    assert c.shape == (0,)
    assert d == pytest.approx(0.5)
    zeros, poles, gain = tf2zpk([0.0], [1.0, 1.0])
    assert zeros.size == 0
    assert poles.size == 1
    assert gain == 0.0
