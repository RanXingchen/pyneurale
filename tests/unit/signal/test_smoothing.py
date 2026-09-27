#!/usr/bin/env python3

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import ValidationError
from neurale.signal.smoothing import (
    WhittakerSmoother,
    gaussian_smooth,
    moving_average,
)


def test_reference_only_smoothing_does_not_expose_fake_backend() -> None:
    assert "backend" not in inspect.signature(moving_average).parameters
    assert "backend" not in inspect.signature(gaussian_smooth).parameters


def _signal(data: np.ndarray) -> SignalArray:
    return SignalArray(
        data=data,
        fs=1000.0,
        time=None,
        t0=2.0,
        clock=None,
        channels=ChannelTable.default(data.shape[1], unit="uV"),
        unit="uV",
        name="neural",
        attrs={"source": "fixture"},
    )


def test_moving_average_matches_legacy_edge_windows() -> None:
    values = np.array(
        [
            0.0799,
            0.6769,
            0.4493,
            0.5361,
            -0.1472,
            -0.5045,
            0.6486,
            -0.4710,
            0.3018,
            -0.1768,
            -0.62909,
            -1.4286,
        ]
    )
    expected = np.array(
        [
            0.0799,
            0.402033333333333,
            0.3190,
            0.20212,
            0.19646,
            0.0124,
            -0.03446,
            -0.04038,
            -0.065298,
            -0.480738,
            -0.74483,
            -1.4286,
        ]
    )

    result = moving_average(values, span=5)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(
        values,
        [
            0.0799,
            0.6769,
            0.4493,
            0.5361,
            -0.1472,
            -0.5045,
            0.6486,
            -0.4710,
            0.3018,
            -0.1768,
            -0.62909,
            -1.4286,
        ],
    )


def test_moving_average_handles_sample_axis_and_signal_metadata() -> None:
    values = np.arange(18.0).reshape(3, 6)
    by_row = moving_average(values, span=3, axis=1)
    expected = np.vstack([moving_average(row, span=3) for row in values])
    np.testing.assert_allclose(by_row, expected)

    source = _signal(np.arange(12.0).reshape(6, 2))
    result = moving_average(source, span=3)
    assert isinstance(result, SignalArray)
    assert result is not source
    assert result.fs == source.fs
    assert result.channel_names == source.channel_names
    assert result.attrs == source.attrs
    np.testing.assert_array_equal(source.data, np.arange(12.0).reshape(6, 2))


@pytest.mark.parametrize("span", [0, 2, 7])
def test_moving_average_rejects_invalid_span(span: int) -> None:
    with pytest.raises(ValidationError):
        moving_average(np.arange(5.0), span=span)


def test_gaussian_smooth_normalizes_boundaries_and_multichannel_data() -> None:
    constant = np.full((4, 2), [2.0, -3.0])
    result = gaussian_smooth(constant, width=3.0)
    np.testing.assert_allclose(result, constant)

    impulse = np.zeros(9)
    impulse[4] = 1.0
    smoothed = gaussian_smooth(impulse, width=3.0)
    np.testing.assert_allclose(smoothed, smoothed[::-1])
    assert smoothed[4] == np.max(smoothed)
    np.testing.assert_array_equal(impulse, np.eye(1, 9, 4).reshape(-1))


def test_gaussian_smooth_supports_nonzero_axis_and_short_inputs() -> None:
    values = np.array([[1.0, 2.0], [10.0, 20.0]])
    result = gaussian_smooth(values, width=5.0, axis=1)
    assert result.shape == values.shape
    assert np.all(result[:, 0] > values[:, 0])
    assert np.all(result[:, 1] < values[:, 1])
    np.testing.assert_allclose(result[1], 10.0 * result[0])


def test_whittaker_apply_matches_dense_penalized_least_squares() -> None:
    values = np.array([0.0, 1.0, 9.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    lam = 2.0
    difference = np.diff(np.eye(values.size), n=2, axis=0)
    expected = np.linalg.solve(
        np.diag(weights) + lam * difference.T @ difference,
        weights * np.where(weights > 0, values, 0),
    )

    result = WhittakerSmoother(2).apply(values, weights, lam)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_whittaker_rejects_singular_system() -> None:
    values = np.arange(5.0)
    weights = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValidationError, match="singular"):
        WhittakerSmoother(2).apply(values, weights, lam=0.0)


def test_whittaker_penalty_does_not_construct_dense_identity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        np,
        "eye",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense identity must not be constructed")
        ),
    )
    result = WhittakerSmoother(2).apply(
        np.linspace(0.0, 1.0, 1000),
        np.ones(1000),
        lam=1.0,
    )
    assert np.all(np.isfinite(result))


def test_sample_domain_smoothing_accepts_irregular_timing() -> None:
    source = SignalArray(
        data=np.ones((4, 1)),
        fs=1000.0,
        time=np.array([0.0, 0.001, 0.0021, 0.003]),
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1, unit="uV"),
        unit="uV",
        name="irregular",
    )
    moving_result = moving_average(source, span=3)
    gaussian_result = gaussian_smooth(source, width=3.0)

    assert isinstance(moving_result, SignalArray)
    assert isinstance(gaussian_result, SignalArray)
    np.testing.assert_array_equal(moving_result.time, source.time)
    np.testing.assert_array_equal(gaussian_result.time, source.time)


def test_whittaker_requires_uniform_signal_array_timing() -> None:
    source = SignalArray(
        data=np.ones((4, 1)),
        fs=1000.0,
        time=np.array([0.0, 0.001, 0.0021, 0.003]),
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1, unit="uV"),
        unit="uV",
        name="irregular",
    )
    with pytest.raises(ValidationError, match="uniformly sampled"):
        WhittakerSmoother(2).apply(source)


@pytest.mark.parametrize("length", [40, 101])
def test_whittaker_lambda_search_exercises_small_and_large_paths(
    length: int,
) -> None:
    phase = np.linspace(0.0, 4.0 * np.pi, length)
    values = np.sin(phase) + 0.05 * np.cos(7.0 * phase)
    weights = np.ones(length)
    weights[length // 3] = 0.0

    lam = WhittakerSmoother(2).get_optimal_lambda(
        values,
        weights,
        min_exp=-1.0,
        max_exp=0.0,
        exp_step=1.0,
    )

    assert lam in (0.1, 1.0)
