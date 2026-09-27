#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Spatial common-reference filtering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import validate_choice
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array
from neurale.signal._input import normalize_signal_input, restore_signal_output


def common_reference(
    x: np.ndarray | SignalArray,
    method: str = "mean",
    *,
    axis: int = 0,
    reference_channels: Sequence[int | str] | np.ndarray | None = None,
    bad_channels: Sequence[int | str] | np.ndarray | None = None,
    exclude_bad: bool = True,
    inplace: bool = False,
    return_reference: bool = False,
):
    """Subtract a sample-wise common mean or median across selected channels.

    ``reference_channels`` selects channels used to estimate the reference.
    The resulting reference is subtracted from every channel. For a
    ``SignalArray``, invalid or bad channels are excluded by default.
    Integer inputs are promoted to ``float64`` to avoid truncation and
    overflow; consequently, in-place operation requires floating-point input.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional signal. Arrays are interpreted along
        ``axis``; ``SignalArray`` input always uses its sample-major layout.
    method : {"mean", "median"}, default="mean"
        Statistic used to estimate the common reference at each sample.
    axis : int, default=0
        Sample axis for ndarray input.
    reference_channels : sequence of int or str, numpy.ndarray, or None, optional
        Channels used to estimate the reference. When omitted, all usable
        channels are used.
    bad_channels : sequence of int or str, numpy.ndarray, or None, optional
        Channels excluded from the reference estimate.
    exclude_bad : bool, default=True
        Exclude invalid or bad ``SignalArray`` channels from the reference.
    inplace : bool, default=False
        Write the result back to the input buffer when possible.
    return_reference : bool, default=False
        Return the estimated 1D reference signal alongside the
        rereferenced signal.

    Returns
    -------
    result : numpy.ndarray or neurale.data.SignalArray
        Rereferenced signal with the input layout and metadata preserved.
    reference : numpy.ndarray, optional
        Sample-wise reference. Returned only when ``return_reference=True``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If input data, channel selectors, or flags are invalid.
    """
    method = validate_choice(method, ("mean", "median"), "method")
    if not isinstance(exclude_bad, bool):
        raise ValidationError("exclude_bad must be a bool.")
    if not isinstance(inplace, bool):
        raise ValidationError("inplace must be a bool.")
    if not isinstance(return_reference, bool):
        raise ValidationError("return_reference must be a bool.")

    data, context = normalize_signal_input(x, axis=axis)
    if np.iscomplexobj(data) or np.issubdtype(data.dtype, np.bool_):
        raise ValidationError("common reference requires real non-boolean numeric data.")
    if np.any(~np.isfinite(data)):
        raise ValidationError("signal must contain finite values.")
    if inplace and np.issubdtype(data.dtype, np.integer):
        raise ValidationError(
            "inplace common reference requires floating-point input; "
            "integer input is promoted to floating point."
        )
    positions = _reference_positions(
        x,
        data.shape[1],
        reference_channels,
        bad_channels,
        exclude_bad,
    )
    implementation = load_native_namespace("signal.filtering")
    output, reference = implementation.common_reference(
        native_array(data, float),
        np.asarray(positions, dtype=np.uintp),
        method,
    )
    if np.issubdtype(data.dtype, np.floating):
        if output.dtype != data.dtype:
            output = output.astype(data.dtype, copy=False)
        if reference.dtype != data.dtype:
            reference = reference.astype(data.dtype, copy=False)

    if inplace:
        _write_inplace(data, output)
        result = x
    else:
        result = restore_signal_output(output, context)
    return (result, reference) if return_reference else result


def _reference_positions(
    x,
    n_channels,
    reference_channels,
    bad_channels,
    exclude_bad,
):
    if reference_channels is None:
        selected = np.ones(n_channels, dtype=bool)
    else:
        selected = np.zeros(n_channels, dtype=bool)
        selected[_resolve_channels(x, reference_channels, n_channels, "reference_channels")] = True

    if isinstance(x, SignalArray) and exclude_bad:
        selected &= x.channels.good_mask
    if bad_channels is not None:
        selected[_resolve_channels(x, bad_channels, n_channels, "bad_channels")] = False
    positions = np.flatnonzero(selected)
    if positions.size == 0:
        raise ValidationError("at least one usable reference channel is required.")
    return positions


def _resolve_channels(x, channels, n_channels, name):
    if isinstance(channels, (str, int, np.integer)):
        channels = [channels]
    values = np.asarray(channels)
    if values.dtype == bool:
        if values.ndim != 1 or values.size != n_channels:
            raise ValidationError(f"{name} boolean mask has the wrong shape.")
        return np.flatnonzero(values)
    if isinstance(x, SignalArray):
        try:
            positions = x.channels.positions(channels)
        except KeyError as exc:
            raise ValidationError(str(exc)) from exc
    else:
        try:
            positions = [int(value) for value in channels]
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must contain integer channel positions.") from exc
        positions = [pos + n_channels if pos < 0 else pos for pos in positions]
    if len(positions) != len(set(positions)):
        raise ValidationError(f"{name} must not contain duplicates.")
    if any(pos < 0 or pos >= n_channels for pos in positions):
        raise ValidationError(f"{name} contains an out-of-range channel.")
    return np.asarray(positions, dtype=int)


def _write_inplace(x, output):
    if not x.flags.writeable:
        raise ValidationError("inplace signal data must be writable.")
    try:
        x[...] = output
    except (TypeError, ValueError) as exc:
        raise ValidationError("inplace common reference cannot write to the input array.") from exc
