#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.signal.simulation import SignalGenerator


def test_zeros_and_per_channel_constant_are_exact() -> None:
    zeros = SignalGenerator.zeros(3, 4_000.0)
    constant = SignalGenerator.constant(3, 4_000.0, [1.5, -2.0, 0.25])

    assert zeros.n_channels == 3
    assert zeros.sample_rate == 4_000.0
    assert zeros.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(zeros.generate(19, 4), np.zeros((4, 3)))
    np.testing.assert_array_equal(
        constant.generate(100, 2),
        np.array([[1.5, -2.0, 0.25], [1.5, -2.0, 0.25]]),
    )


def test_sine_matches_absolute_position_per_channel() -> None:
    generator = SignalGenerator.sine(
        2,
        1_000.0,
        [25.0, 40.0],
        amp=[2.0, 0.5],
        phase=[0.25, -0.5],
    )
    start = 71
    positions = np.arange(start, start + 8, dtype=np.float64)[:, None]
    expected = np.array([2.0, 0.5])[None, :] * np.sin(
        np.array([0.25, -0.5])[None, :]
        + 2.0 * np.pi * positions * np.array([25.0, 40.0])[None, :] / 1_000.0
    )

    np.testing.assert_allclose(generator.generate(start, 8), expected, rtol=0.0, atol=2e-14)


def test_multiple_tones_match_direct_sum_and_are_chunk_invariant() -> None:
    freqs = np.array([[10.0, 15.0], [50.0, 75.0]])
    amps = np.array([[1.0, 2.0], [0.25, 0.5]])
    phases = np.array([[0.0, 0.1], [0.3, -0.2]])
    generator = SignalGenerator.tones(
        2,
        500.0,
        freqs,
        amps=amps,
        phases=phases,
    )
    start = 23
    positions = np.arange(start, start + 17, dtype=np.float64)
    expected = np.empty((17, 2))
    for channel in range(2):
        expected[:, channel] = sum(
            amps[tone, channel]
            * np.sin(phases[tone, channel] + 2.0 * np.pi * freqs[tone, channel] * positions / 500.0)
            for tone in range(2)
        )

    whole = generator.generate(start, 17)
    chunked = np.concatenate(
        [generator.generate(start, 6), generator.generate(start + 6, 11)], axis=0
    )
    np.testing.assert_array_equal(chunked, whole)
    np.testing.assert_allclose(whole, expected, rtol=0.0, atol=2e-14)


def test_seeded_noise_is_reproducible_and_chunk_invariant() -> None:
    first = SignalGenerator.noise(3, 1_000.0, seed=987, low=-3.0, high=2.0)
    reconstructed = SignalGenerator.noise(3, 1_000.0, seed=987, low=-3.0, high=2.0)
    different_seed = SignalGenerator.noise(3, 1_000.0, seed=988, low=-3.0, high=2.0)

    whole = first.generate(123, 19)
    chunked = np.concatenate([first.generate(123, 7), first.generate(130, 12)])
    np.testing.assert_array_equal(whole, chunked)
    np.testing.assert_array_equal(whole, reconstructed.generate(123, 19))
    assert not np.array_equal(whole[:, 0], whole[:, 1])
    assert not np.array_equal(whole, different_seed.generate(123, 19))
    assert np.all(whole >= -3.0)
    assert np.all(whole < 2.0)


def test_every_generator_kind_obeys_absolute_chunk_invariance() -> None:
    supplied = np.arange(24, dtype=np.float64).reshape(12, 2)
    generators = [
        SignalGenerator.zeros(2, 200.0),
        SignalGenerator.constant(2, 200.0, [1.0, -1.0]),
        SignalGenerator.sine(2, 200.0, [10.0, 20.0]),
        SignalGenerator.tones(2, 200.0, [10.0, 30.0]),
        SignalGenerator.noise(2, 200.0, seed=4),
        SignalGenerator.samples(2, 200.0, supplied),
        SignalGenerator.samples(2, 200.0, supplied, repeat=True),
    ]

    for generator in generators:
        whole = generator.generate(2, 8)
        chunked = np.concatenate([generator.generate(2, 3), generator.generate(5, 5)])
        np.testing.assert_array_equal(chunked, whole)


def test_supplied_samples_support_repeating_ranges() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    finite = SignalGenerator.samples(2, 100.0, values)
    repeating = SignalGenerator.samples(2, 100.0, values, repeat=True)

    np.testing.assert_array_equal(finite.generate(1, 2), values[1:])
    np.testing.assert_array_equal(
        repeating.generate(2, 4),
        np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    with pytest.raises(ValidationError, match="exceeds finite"):
        finite.generate(2, 2)


def test_generate_into_fills_caller_owned_buffer() -> None:
    generator = SignalGenerator.constant(2, 250.0, [7.0, -4.0])
    output = np.empty((5, 2), dtype=np.float64)
    returned = generator.generate_into(output, start=99)

    assert returned is None
    np.testing.assert_array_equal(output, np.tile([7.0, -4.0], (5, 1)))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SignalGenerator.zeros(0, 1_000.0), "n_channels"),
        (lambda: SignalGenerator.zeros(1, 0.0), "fs"),
        (lambda: SignalGenerator.zeros(1, 1_000.0, dtype=np.float32), "float64"),
        (lambda: SignalGenerator.sine(1, 1_000.0, 501.0), "frequencies"),
        (lambda: SignalGenerator.sine(2, 1_000.0, [10.0]), "frequency"),
        (lambda: SignalGenerator.sine(1, 1_000.0, 10.0 + 1.0j), "finite real"),
        (lambda: SignalGenerator.tones(2, 1_000.0, []), "frequencies"),
        (lambda: SignalGenerator.noise(1, 1_000.0, seed=-1), "seed"),
        (lambda: SignalGenerator.noise(1, 1_000.0, seed=1, low=1.0, high=1.0), "low"),
        (
            lambda: SignalGenerator.noise(1, 1_000.0, seed=1, low=-1e308, high=1e308),
            "finite positive span",
        ),
        (
            lambda: SignalGenerator.samples(2, 1_000.0, np.ones((2, 2), dtype=np.float32)),
            "float64",
        ),
        (
            lambda: SignalGenerator.samples(2, 1_000.0, np.ones((0, 2), dtype=np.float64)),
            "non-empty",
        ),
    ],
)
def test_invalid_generator_configuration_is_rejected(factory: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory()  # type: ignore[operator]


def test_invalid_caller_buffers_and_ranges_are_rejected() -> None:
    generator = SignalGenerator.zeros(2, 1_000.0)
    read_only = np.empty((2, 2), dtype=np.float64)
    read_only.flags.writeable = False

    invalid = [
        np.empty((2, 2), dtype=np.float32),
        np.empty((2,), dtype=np.float64),
        np.empty((2, 3), dtype=np.float64),
        np.empty((4, 2), dtype=np.float64)[::2],
        read_only,
    ]
    for output in invalid:
        with pytest.raises(ValidationError):
            generator.generate_into(output, start=0)

    with pytest.raises(ValidationError, match="start"):
        generator.generate(-1, 1)
    with pytest.raises(ValidationError, match="count"):
        generator.generate(0, -1)
    with pytest.raises(ValidationError, match="overflows"):
        generator.generate(2**64 - 1, 2)
