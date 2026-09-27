#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Normalize public signal inputs for internal sample-major kernels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from neurale._validation import validate_axis
from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import ValidationError

from ._validation import validate_signal_matrix


@dataclass(frozen=True, slots=True)
class SignalInputContext:
    source: np.ndarray | SignalArray
    sample_axis: int
    normalized_shape: tuple[int, int]


def normalize_signal_input(
    x: np.ndarray | SignalArray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, SignalInputContext]:
    if isinstance(x, SignalArray):
        normalized_axis = validate_axis(axis, 2)
        if normalized_axis != 0:
            raise ValidationError("SignalArray uses axis 0 as its fixed sample axis.")
        data = validate_signal_matrix(x.data, "signal.data")
        context = SignalInputContext(
            source=x,
            sample_axis=0,
            normalized_shape=data.shape,
        )
        return data, context

    if not isinstance(x, np.ndarray):
        raise ValidationError("signal must be a numpy.ndarray or SignalArray.")
    if x.ndim not in (1, 2):
        raise ValidationError("signal ndarray must be one- or two-dimensional.")

    normalized_axis = validate_axis(axis, x.ndim)
    if x.ndim == 1:
        data = x.reshape(-1, 1)
    else:
        data = np.moveaxis(x, normalized_axis, 0)
    validate_signal_matrix(data)

    context = SignalInputContext(
        source=x,
        sample_axis=normalized_axis,
        normalized_shape=data.shape,
    )
    return data, context


def restore_signal_output(
    data: np.ndarray,
    context: SignalInputContext,
) -> np.ndarray | SignalArray:
    data = np.asarray(data)
    if data.shape != context.normalized_shape:
        raise ValidationError(
            "restore_signal_output only supports shape-preserving results; "
            f"expected {context.normalized_shape}, got {data.shape}."
        )

    source = context.source
    if isinstance(source, SignalArray):
        return source.copy(data=data)

    if source.ndim == 1:
        return data[:, 0]
    return np.moveaxis(data, 0, context.sample_axis)


def uniform_sampling_rate(
    signal: np.ndarray | SignalArray,
    *,
    required: bool = False,
) -> float | None:
    if not isinstance(signal, SignalArray):
        if required:
            raise ValidationError("sampling metadata is required for this operation.")
        return None

    declared = signal.fs

    time = np.asarray(signal.time, dtype=float)
    if time.size < 2:
        return declared

    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValidationError(
            "uniformly sampled SignalArray timestamps must be strictly increasing."
        )
    reference = float(np.median(intervals))
    absolute_tol = max(
        np.finfo(float).eps * max(1.0, abs(reference)) * 16.0,
        1e-12,
    )
    if not np.allclose(
        intervals,
        reference,
        rtol=1e-7,
        atol=absolute_tol,
    ):
        raise ValidationError("signal processing operation requires uniformly sampled timestamps.")
    inferred = 1.0 / reference
    if not np.isclose(
        declared,
        inferred,
        rtol=1e-7,
        atol=max(1e-12, abs(inferred) * 1e-12),
    ):
        raise ValidationError("SignalArray fs is inconsistent with explicit timestamps.")
    return declared


def transform_signal_units(
    signal: SignalArray,
    transform: Callable[[str], str],
) -> SignalArray:
    source_units = (
        list(signal.unit) if not isinstance(signal.unit, str) else [signal.unit] * signal.n_channels
    )
    transformed = [transform(unit) for unit in source_units]
    channels = ChannelTable(
        replace(channel, unit=unit, attrs=dict(channel.attrs))
        for channel, unit in zip(signal.channels, transformed, strict=False)
    )
    return SignalArray(
        data=signal.data,
        fs=signal.fs,
        time=signal.time.copy(),
        t0=signal.t0,
        clock=signal.clock,
        channels=channels,
        unit=(transformed[0] if isinstance(signal.unit, str) else transformed),
        name=signal.name,
        attrs=dict(signal.attrs),
    )


def multiply_unit_by_seconds(unit: str, order: int = 1) -> str:
    result = unit
    for _ in range(order):
        result = f"({result})*s" if result.endswith("*s") else f"{result}*s"
    return result
