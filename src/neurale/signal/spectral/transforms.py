#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Chirp-Z and analytic-signal transforms."""

from __future__ import annotations

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_axis,
    validate_integer,
    validate_numeric_array,
)
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array
from neurale.signal._input import normalize_signal_input, uniform_sampling_rate


def czt(
    x: np.ndarray,
    m: int | None = None,
    *,
    w: complex | None = None,
    a: complex = 1.0 + 0.0j,
    axis: int = 0,
) -> np.ndarray:
    """Compute a chirp Z-transform along one array axis.

    Parameters
    ----------
    x : numpy.ndarray
        One- or two-dimensional numeric array.
    m : int or None, optional
        Positive output length; defaults to input length.
    w : complex or None, optional
        Non-zero ratio between consecutive contour points.
    a : complex, default=1+0j
        Non-zero first contour point.
    axis : int, default=0
        Transform axis.
    Returns
    -------
    numpy.ndarray
        Complex transform with ``m`` points along ``axis``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If x, axis, output length, or contour parameters are invalid.
    """
    if not isinstance(x, np.ndarray) or x.ndim not in (1, 2):
        raise ValidationError("x must be a one- or two-dimensional ndarray.")
    validate_numeric_array(x, "x")
    axis = validate_axis(axis, x.ndim)
    n_in = x.shape[axis]
    if n_in == 0:
        raise ValidationError("x must contain at least one sample.")
    n_out = n_in if m is None else validate_integer(m, "m", minimum=1)
    start = _validate_complex_scalar(a, "a", nonzero=True)
    ratio = (
        np.exp(-2j * np.pi / n_out) if w is None else _validate_complex_scalar(w, "w", nonzero=True)
    )
    normalized = x.reshape(-1, 1) if x.ndim == 1 else np.moveaxis(x, axis, 0)
    native = load_native_namespace("signal.spectral")
    result = native.czt(
        native_array(normalized, complex),
        n_out,
        ratio,
        start,
    )
    if x.ndim == 1:
        return result[:, 0]
    return np.moveaxis(result, 0, axis)


def analytic_signal(
    x: np.ndarray | SignalArray,
    *,
    axis: int = 0,
    nfft: int | None = None,
) -> np.ndarray | SignalArray:
    """Construct the analytic representation of a real signal.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.arrays.SignalArray
        Real one- or two-dimensional signal.
    axis : int, default=0
        Sample axis for ndarray input.
    nfft : int or None, optional
        Transform length, at least the input sample count.
    Returns
    -------
    numpy.ndarray or neurale.data.arrays.SignalArray
        Complex analytic signal, optionally zero-padded to ``nfft``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If input is complex, empty, malformed, or ``nfft`` is too short.
    """
    data, context = normalize_signal_input(x, axis=axis)
    if np.iscomplexobj(data):
        raise ValidationError("analytic_signal requires real-valued input.")
    if data.shape[0] == 0:
        raise ValidationError("signal must contain at least one sample.")
    fft_length = data.shape[0] if nfft is None else validate_integer(nfft, "nfft", minimum=1)
    if fft_length < data.shape[0]:
        raise ValidationError("nfft must be greater than or equal to the input length.")

    native = load_native_namespace("signal.spectral")
    result = native.analytic_signal(
        native_array(data, float),
        fft_length,
    )

    if isinstance(context.source, np.ndarray):
        if context.source.ndim == 1:
            return result[:, 0]
        return np.moveaxis(result, 0, context.sample_axis)

    source = context.source
    if not isinstance(source, SignalArray):
        raise TypeError("analytic_signal metadata restoration requires SignalArray input.")
    if fft_length == source.n_samples:
        explicit_time = source.time.copy()
    else:
        rate = uniform_sampling_rate(source, required=True)
        if rate is None:
            raise ValidationError("analytic_signal requires sampling metadata.")
        explicit_time = float(source.time[0]) + np.arange(fft_length) / rate
    return SignalArray(
        data=result,
        fs=source.fs,
        time=explicit_time,
        t0=source.t0,
        clock=source.clock,
        channels=source.channels.copy(),
        unit=source.unit,
        name=source.name,
        attrs=dict(source.attrs),
    )


def _validate_complex_scalar(
    value: complex,
    name: str,
    *,
    nonzero: bool,
) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a complex scalar.") from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValidationError(f"{name} must be finite.")
    if nonzero and result == 0:
        raise ValidationError(f"{name} must be nonzero.")
    return result
