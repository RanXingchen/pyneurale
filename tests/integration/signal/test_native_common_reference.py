#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest

from neurale.signal import common_reference


def _native() -> object:
    native = importlib.import_module("neurale._native").signal.filtering
    native.common_reference  # noqa: B018

    return native


def _reference(values: np.ndarray, positions: np.ndarray, method: str):
    reference = getattr(np, method)(values[:, positions], axis=1)
    return values - reference[:, None], reference


@pytest.mark.parametrize("method", ["mean", "median"])
@pytest.mark.parametrize("samples", [0, 1, 127])
@pytest.mark.parametrize("channels", [1, 2, 3, 8, 67, 256])
def test_native_common_reference_matches_numpy(method, samples, channels) -> None:
    _native()
    rng = np.random.default_rng(samples * 1000 + channels)
    values = rng.standard_normal((samples, channels))
    selected = np.arange(0, channels, 2)
    actual = common_reference(
        values,
        method,
        reference_channels=selected,
        return_reference=True,
    )
    expected = _reference(values, selected, method)
    for actual_values, expected_values in zip(actual, expected, strict=False):
        np.testing.assert_allclose(actual_values, expected_values, atol=2e-12)


def test_native_raw_binding_validates_contract() -> None:
    native = _native()
    with pytest.raises(ValueError):
        native.common_reference(np.ones(5), np.asarray([0], dtype=np.uintp), "mean")
    with pytest.raises(ValueError, match="at least one"):
        native.common_reference(np.ones((5, 2)), np.asarray([], dtype=np.uintp), "mean")
    with pytest.raises(ValueError, match="out of bounds"):
        native.common_reference(np.ones((5, 2)), np.asarray([2], dtype=np.uintp), "mean")
    # A repeated position used to be accepted and silently weighted that
    # channel twice. The Python wrapper has always refused it, so this asserts
    # the native boundary refuses it too rather than trusting the caller.
    with pytest.raises(ValueError, match="unique"):
        native.common_reference(np.ones((5, 2)), np.asarray([0, 0], dtype=np.uintp), "mean")
    with pytest.raises(ValueError, match="method"):
        native.common_reference(np.ones((5, 2)), np.asarray([0], dtype=np.uintp), "mode")
