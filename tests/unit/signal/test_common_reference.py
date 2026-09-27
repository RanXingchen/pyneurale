#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import common_reference
from neurale.signal.filtering import spatial as spatial_module


def _reference(values: np.ndarray, positions, method: str):
    selected = values[:, positions]
    reference = getattr(np, method)(selected, axis=1)
    return values - reference[:, None], reference


@pytest.mark.parametrize("method", ["mean", "median"])
@pytest.mark.parametrize("channels", [1, 2, 3, 8, 67])
def test_common_reference_matches_numpy_for_all_channel_paths(method, channels) -> None:
    rng = np.random.default_rng(channels)
    values = rng.standard_normal((101, channels))
    result, reference = common_reference(values, method, return_reference=True)
    expected, expected_reference = _reference(values, np.arange(channels), method)
    np.testing.assert_allclose(reference, expected_reference)
    np.testing.assert_allclose(result, expected)


def test_reference_channels_and_bad_channels_only_change_estimator() -> None:
    values = np.asarray(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
        ]
    )
    result, reference = common_reference(
        values,
        reference_channels=[0, 1, 2],
        bad_channels=[1],
        return_reference=True,
    )
    np.testing.assert_array_equal(reference, [50.5, 101.0])
    np.testing.assert_array_equal(result, values - reference[:, None])


def test_scalar_channel_selectors_and_python_boolean_mask() -> None:
    values = np.arange(12.0).reshape(3, 4)
    _, by_position = common_reference(
        values,
        reference_channels=2,
        return_reference=True,
    )
    _, by_mask = common_reference(
        values,
        reference_channels=[False, False, True, False],
        return_reference=True,
    )
    np.testing.assert_array_equal(by_position, values[:, 2])
    np.testing.assert_array_equal(by_mask, values[:, 2])


def test_nonzero_axis_boolean_mask_negative_index_and_float32() -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    mask = np.asarray([True, False, True])
    result = common_reference(
        values,
        "median",
        axis=1,
        reference_channels=mask,
        bad_channels=[-1],
    )
    expected = values - values[[0], :]
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_integer_inputs_are_promoted_without_truncation(method) -> None:
    values = np.asarray(
        [
            [-2, -1, 0, 2],
            [1, 2, 3, 5],
        ],
        dtype=np.int16,
    )
    result, reference = common_reference(values, method, return_reference=True)
    expected, expected_reference = _reference(
        values.astype(float), np.arange(values.shape[1]), method
    )
    assert result.dtype == np.float64
    assert reference.dtype == np.float64
    np.testing.assert_array_equal(reference, expected_reference)
    np.testing.assert_allclose(result, expected)

    with pytest.raises(ValidationError, match="floating-point"):
        common_reference(values, method, inplace=True)


def _signal(values):
    channels = ChannelTable(
        [
            ChannelInfo("good", 10, "eeg", "uV"),
            ChannelInfo("bad", 11, "eeg", "uV", bad=True),
            ChannelInfo("invalid", 12, "eeg", "uV", valid=False),
            ChannelInfo("good2", 13, "eeg", "uV"),
        ]
    )
    return SignalArray(
        np.asarray(values, dtype=float),
        fs=1000,
        time=None,
        t0=3.0,
        clock=None,
        channels=channels,
        unit="uV",
        name="source",
        attrs={"stage": 12},
    )


def test_signal_array_uses_good_mask_and_preserves_metadata() -> None:
    source = _signal(
        [
            [1, 100, 200, 3],
            [2, 110, 210, 4],
        ]
    )
    result, reference = common_reference(source, return_reference=True)
    np.testing.assert_array_equal(reference, [2, 3])
    np.testing.assert_array_equal(result.data, source.data - reference[:, None])
    assert result.channel_names == source.channel_names
    assert result.channels.good_mask.tolist() == [True, False, False, True]
    assert result.fs == source.fs
    assert result.t0 == source.t0
    assert result.attrs == source.attrs


def test_explicit_names_and_exclude_bad_override() -> None:
    source = _signal([[1, 10, 100, 1000]])
    _, selected = common_reference(
        source,
        reference_channels=["good", 13],
        return_reference=True,
    )
    _, including_bad = common_reference(
        source,
        reference_channels=["bad", "invalid"],
        exclude_bad=False,
        return_reference=True,
    )
    np.testing.assert_array_equal(selected, [500.5])
    np.testing.assert_array_equal(including_bad, [55])


@pytest.mark.parametrize("signal_array", [False, True])
def test_inplace_returns_original_object_and_mutates_data(signal_array) -> None:
    values = np.arange(12.0).reshape(3, 4)
    source = _signal(values) if signal_array else values.copy()
    original = source.data.copy() if signal_array else source.copy()
    result, reference = common_reference(
        source,
        inplace=True,
        return_reference=True,
        exclude_bad=False,
    )
    assert result is source
    target = source.data if signal_array else source
    expected_reference = original.mean(axis=1)
    np.testing.assert_allclose(reference, expected_reference)
    np.testing.assert_allclose(target, original - expected_reference[:, None])


def test_empty_samples_and_readonly_inplace() -> None:
    values = np.empty((0, 3))
    result, reference = common_reference(values, return_reference=True)
    assert result.shape == (0, 3)
    assert reference.shape == (0,)
    readonly = np.ones((2, 3))
    readonly.flags.writeable = False
    with pytest.raises(ValidationError, match="writable"):
        common_reference(readonly, inplace=True)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"method": "mode"}, "method"),
        ({"reference_channels": []}, "at least one"),
        (
            {"reference_channels": np.asarray([True, False])},
            "wrong shape",
        ),
        ({"reference_channels": [0, 0]}, "duplicates"),
        ({"reference_channels": [4]}, "out-of-range"),
        ({"bad_channels": [0, 1, 2]}, "at least one"),
    ],
)
def test_common_reference_validation_paths(arguments, message) -> None:
    with pytest.raises(ValidationError, match=message):
        common_reference(np.ones((5, 3)), **arguments)


@pytest.mark.parametrize(
    "values",
    [
        np.ones((4, 2), dtype=complex),
        np.ones((4, 2), dtype=bool),
        np.asarray([[1.0, np.nan]]),
    ],
)
def test_common_reference_rejects_unsupported_values(values) -> None:
    with pytest.raises(ValidationError):
        common_reference(values)


def test_common_reference_requires_native_extension(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(spatial_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        common_reference(np.ones((4, 2)))
