#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared implementation details for high-level PSD estimators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

from neurale._validation import validate_choice, validate_number
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._input import uniform_sampling_rate
from neurale.signal._validation import validate_fs

PsdDetrend = Literal["none", "mean", "linear"]


def resolve_psd_sampling_rate(
    x: np.ndarray | SignalArray,
    fs: float | None,
) -> float | None:
    if not isinstance(x, SignalArray):
        return validate_fs(fs, optional=True)
    metadata_rate = uniform_sampling_rate(x, required=True)
    if metadata_rate is None:
        raise ValidationError("SignalArray PSD requires sampling metadata.")
    if fs is None:
        return metadata_rate
    requested = validate_fs(fs)
    if requested is None:
        raise ValidationError("fs must be positive when provided.")
    if not np.isclose(requested, metadata_rate):
        raise ValidationError("fs is inconsistent with SignalArray metadata.")
    return requested


def select_time_range(
    x: np.ndarray,
    fs: float | None,
    time_range: Sequence[float] | None,
) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValidationError("signal must contain at least one sample.")
    if time_range is None:
        return x
    if fs is None:
        raise ValidationError("time_range requires a sampling rate.")
    if isinstance(time_range, (str, bytes)) or len(time_range) != 2:
        raise ValidationError("time_range must contain exactly two values.")
    start = float(
        validate_number(
            time_range[0],
            "time_range start",
            kind="real",
            minimum=0,
            coerce=True,
        )
    )
    stop = float(
        validate_number(
            time_range[1],
            "time_range stop",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    if start >= stop:
        raise ValidationError("time_range start must be less than stop.")
    start_sample = math.floor(start * fs)
    stop_sample = math.ceil(stop * fs)
    if start_sample >= x.shape[0] or stop_sample > x.shape[0]:
        raise ValidationError("time_range lies outside the signal duration.")
    return x[start_sample:stop_sample]


def preprocess_segment(
    x: np.ndarray,
    detrend: PsdDetrend,
) -> np.ndarray:
    if detrend == "none":
        return x
    if detrend == "mean":
        return x - np.mean(x, axis=0, keepdims=True)
    return scipy_signal.detrend(x, axis=0, type="linear")


def postprocess_psd(
    x: np.ndarray,
    db: bool,
    smoothing_width: float | None,
) -> np.ndarray:
    result = np.asarray(x, dtype=float)
    if db:
        with np.errstate(divide="ignore"):
            result = 10.0 * np.log10(result)
    if smoothing_width is None:
        return result
    width = float(
        validate_number(
            smoothing_width,
            "smoothing_width",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    radius = int(width) + 1
    offsets = np.arange(-radius, radius + 1, dtype=float)
    variance = -(width * width) * 0.25 / np.log(0.5)
    kernel = np.exp(-(offsets * offsets) / variance)
    kernel /= np.sum(kernel)
    normalization = np.convolve(np.ones(result.shape[0]), kernel, mode="full")[
        radius : radius + result.shape[0]
    ]
    smoothed = np.empty_like(result)
    for channel in range(result.shape[1]):
        full = np.convolve(result[:, channel], kernel, mode="full")
        smoothed[:, channel] = full[radius : radius + result.shape[0]] / normalization
    return smoothed


def resolve_smoothing_width(
    smoothing_width: float | None,
    smoothing_width_hz: float | None,
    freqs: np.ndarray,
) -> float | None:
    """Resolve mutually exclusive bin- and hertz-based smoothing widths."""
    if smoothing_width is not None and smoothing_width_hz is not None:
        raise ValidationError("smoothing_width and smoothing_width_hz are mutually exclusive.")
    if smoothing_width_hz is None:
        return smoothing_width
    width_hz = float(
        validate_number(
            smoothing_width_hz,
            "smoothing_width_hz",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    if freqs.size < 2:
        raise ValidationError("smoothing_width_hz requires at least two frequency bins.")
    spacing = np.diff(np.asarray(freqs, dtype=float))
    resolution = float(np.median(np.abs(spacing)))
    if resolution <= 0 or not np.allclose(np.abs(spacing), resolution, rtol=1e-7, atol=1e-12):
        raise ValidationError("smoothing_width_hz requires a uniformly spaced frequency grid.")
    return width_hz / resolution


def restore_psd(values, freqs, context):
    source = context.source
    if isinstance(source, np.ndarray):
        if source.ndim == 1:
            return values[:, 0], freqs
        return np.moveaxis(values, 0, context.sample_axis), freqs
    return values, freqs


def validate_detrend(value: str) -> PsdDetrend:
    if value == "detrend":
        value = "linear"
    return validate_choice(value, ("none", "mean", "linear"), "detrend")


def validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a bool.")
    return value
