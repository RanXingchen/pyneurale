#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared native filtering helpers."""

from __future__ import annotations

import numpy as np

from neurale._validation import validate_finite, validate_numeric_array, validate_real_array
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array
from neurale.signal._input import (
    normalize_signal_input,
    restore_signal_output,
    uniform_sampling_rate,
)


def validate_coefficient_rate(
    x: np.ndarray | SignalArray,
    rate: float | None,
    label: str,
) -> None:
    if not isinstance(x, SignalArray) or rate is None:
        return
    actual = uniform_sampling_rate(x, required=True)
    if not np.isclose(actual, rate):
        raise ValidationError(f"SignalArray fs does not match {label} coefficients.")


def _check_state(
    zi: np.ndarray | None,
    expected: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if zi is None:
        return np.zeros(expected, dtype=float)
    result = validate_numeric_array(np.asarray(zi), name)
    if result.shape != expected:
        raise ValidationError(f"{name} must have shape {expected}; got {result.shape}.")
    if np.any(~np.isfinite(result)):
        raise ValidationError(f"{name} must contain finite values.")
    return native_array(result, float)


def _filter(
    x: np.ndarray | SignalArray,
    order: int,
    *,
    axis: int,
    fs: float | None,
    label: str,
    coef_arrays: tuple[np.ndarray, ...],
    zi: np.ndarray | None,
    check_finite: bool,
    operation,
) -> np.ndarray | SignalArray | tuple[np.ndarray | SignalArray, np.ndarray]:
    if not isinstance(check_finite, bool):
        raise ValidationError("check_finite must be a bool.")

    data, context = normalize_signal_input(x, axis=axis)
    validate_coefficient_rate(x, fs, label)
    validate_real_array(data, name="Signal")
    if check_finite:
        validate_finite(data, "Signal")

    for c in coef_arrays:
        validate_real_array(c, name=f"{label} filtering")

    expected_state_shape = (order, 2, data.shape[1]) if label == "SOS" else (order, data.shape[1])

    state = _check_state(zi, expected_state_shape, "zi")
    output, final_state = operation(native_array(data, float), state, check_finite)
    result = restore_signal_output(output, context)

    return result, final_state


class _FilterBase:
    """Shared lifecycle for native realtime filter processors."""

    @property
    def kernel(self) -> str:
        return self._kernel

    def process(self, x: np.ndarray) -> np.ndarray:
        self._process(x)
        return x

    def reset(self) -> None:
        self._processor.reset()

    def get_state(self) -> np.ndarray:
        return self._processor.state()

    def set_state(self, state: np.ndarray) -> None:
        self._processor.set_state(self._state(state, "state"))

    def _attach_processor(self, processor) -> None:
        self._processor = processor
        self._process = processor.process
        self._kernel = processor.kernel
