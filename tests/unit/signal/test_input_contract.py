#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._input import normalize_signal_input, restore_signal_output
from neurale.signal._numerics import next_power_of_two
from neurale.signal._validation import (
    validate_fs,
    validate_signal_matrix,
)


def _signal_array() -> SignalArray:
    return SignalArray(
        data=np.arange(12.0).reshape(4, 3),
        fs=1000.0,
        time=None,
        t0=1.5,
        clock=None,
        channels=ChannelTable(
            [
                ChannelInfo("A", 0, "ecog", "uV"),
                ChannelInfo("B", 1, "ecog", "uV"),
                ChannelInfo("C", 2, "ecog", "uV"),
            ]
        ),
        unit="uV",
        name="neural",
        attrs={"subject": "example"},
    )


def test_validate_signal_matrix_requires_numeric_2d_ndarray() -> None:
    data = np.zeros((4, 2))
    assert validate_signal_matrix(data) is data

    with pytest.raises(ValidationError):
        validate_signal_matrix(np.zeros(4))
    with pytest.raises(ValidationError):
        validate_signal_matrix(np.array([["x"]], dtype=object))
    with pytest.raises(ValidationError):
        validate_signal_matrix([[1.0]])  # type: ignore[arg-type]


def test_normalize_and_restore_1d_ndarray_as_single_channel() -> None:
    source = np.arange(5.0)

    normalized, context = normalize_signal_input(source, axis=-1)
    output = restore_signal_output(normalized + 1.0, context)

    assert not hasattr(context, "kind")
    assert not hasattr(context, "original_ndim")
    assert normalized.shape == (5, 1)
    assert isinstance(output, np.ndarray)
    assert output.shape == source.shape
    np.testing.assert_array_equal(output, source + 1.0)
    np.testing.assert_array_equal(source, np.arange(5.0))


def test_normalize_and_restore_nonzero_sample_axis() -> None:
    source = np.arange(12.0).reshape(3, 4)

    normalized, context = normalize_signal_input(source, axis=1)
    output = restore_signal_output(normalized * 2.0, context)

    assert normalized.shape == (4, 3)
    assert isinstance(output, np.ndarray)
    assert output.shape == source.shape
    np.testing.assert_array_equal(output, source * 2.0)


def test_round_trip_preserves_metadata_without_mutation() -> None:
    source = _signal_array()
    normalized, context = normalize_signal_input(source)

    output = restore_signal_output(normalized + 10.0, context)

    assert isinstance(output, SignalArray)
    assert output is not source
    np.testing.assert_array_equal(output.data, source.data + 10.0)
    np.testing.assert_array_equal(source.data, np.arange(12.0).reshape(4, 3))
    assert output.fs == source.fs
    np.testing.assert_array_equal(output.time, source.time)
    assert output.channel_names == source.channel_names
    assert output.channels is not source.channels
    assert output.unit == source.unit
    assert output.name == source.name
    assert output.attrs == source.attrs
    assert output.attrs is not source.attrs


def test_signal_array_rejects_non_sample_axis() -> None:
    with pytest.raises(ValidationError, match="fixed sample axis"):
        normalize_signal_input(_signal_array(), axis=1)


def test_restore_rejects_shape_changes_needing_metadata() -> None:
    normalized, context = normalize_signal_input(np.arange(4.0))

    with pytest.raises(ValidationError, match="shape-preserving"):
        restore_signal_output(normalized[:-1], context)


def test_backend_and_fs_contracts() -> None:
    assert validate_fs(1000) == 1000.0
    assert validate_fs(None, optional=True) is None

    for value in (None, 0, -1, np.inf, np.nan, True):
        with pytest.raises(ValidationError):
            validate_fs(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 1), (1, 1), (2, 2), (3, 4), (16, 16), (17, 32)],
)
def test_next_power_of_two_returns_fft_length(value: int, expected: int) -> None:
    assert next_power_of_two(value) == expected


def test_next_power_of_two_rejects_negative_and_non_integer_values() -> None:
    with pytest.raises(ValidationError):
        next_power_of_two(-1)
    with pytest.raises(ValidationError):
        next_power_of_two(2.5)  # type: ignore[arg-type]
