#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Frequency-domain signal transforms."""

from __future__ import annotations

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import validate_integer, validate_positive_float
from neurale.data import SignalArray
from neurale.exceptions import ValidationError

from ._arrays import native_array
from ._input import (
    multiply_unit_by_seconds,
    normalize_signal_input,
    restore_signal_output,
    transform_signal_units,
    uniform_sampling_rate,
)


def fft_integrate(
    x: np.ndarray | SignalArray,
    dt: float | None = None,
    *,
    axis: int = 0,
    order: int = 1,
) -> np.ndarray | SignalArray:
    """Integrate a signal in the frequency domain.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.arrays.SignalArray
        One- or two-dimensional signal.
    dt : float or None, optional
        Positive sample interval. Inferred from ``SignalArray`` metadata when
        omitted.
    axis : int, default=0
        Sample axis for ndarray input.
    order : int, default=1
        Positive integration order.
    Returns
    -------
    numpy.ndarray or neurale.data.arrays.SignalArray
        Integrated signal. ``SignalArray`` units gain one seconds factor per order.

    Raises
    ------
    neurale.exceptions.ValidationError
        If signal, timing metadata, sample interval, axis, or order are invalid.

    Notes
    -----
    The DC component is set to zero, fixing the arbitrary integration constant.
    """
    order = validate_integer(order, "order", minimum=1)
    data, context = normalize_signal_input(x, axis=axis)
    if data.shape[0] == 0:
        raise ValidationError("signal must contain at least one sample.")

    if dt is not None:
        dt = validate_positive_float(dt, "dt")
    if isinstance(x, SignalArray):
        fs = uniform_sampling_rate(x, required=True)
        if fs is None:
            raise ValidationError("SignalArray integration requires sampling metadata.")
        metadata_dt = 1.0 / fs
        if dt is None:
            dt = metadata_dt
        elif not np.isclose(dt, metadata_dt):
            raise ValidationError("dt is inconsistent with SignalArray.fs.")
    elif dt is None:
        raise ValidationError("dt is required for ndarray input.")

    native = load_native_namespace("signal.transforms")
    if np.isrealobj(data):
        integrated = native.fft_integrate_real(native_array(data, float), dt, order).real
    else:
        integrated = native.fft_integrate(native_array(data, complex), dt, order)
    restored = restore_signal_output(integrated, context)
    if isinstance(restored, SignalArray):
        return transform_signal_units(
            restored,
            lambda unit: multiply_unit_by_seconds(unit, order),
        )
    return restored
