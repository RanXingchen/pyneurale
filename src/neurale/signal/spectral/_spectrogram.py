#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared frame construction for time-frequency estimators."""

from __future__ import annotations

import numpy as np

from neurale._validation import validate_choice, validate_integer, validate_number
from neurale.data import SignalArray
from neurale.exceptions import ValidationError


def spectrogram_frames(
    x: np.ndarray,
    fs: float,
    *,
    window_size: float,
    shift: float,
    start_time: float,
    n_windows: int | None,
    time_reference: str = "center",
) -> tuple[list[np.ndarray], np.ndarray, int, int]:
    starts, times, window_length, shift_length = spectrogram_frame_geometry(
        x.shape[0],
        fs,
        window_size=window_size,
        shift=shift,
        start_time=start_time,
        n_windows=n_windows,
        time_reference=time_reference,
    )
    frames = [x[start : start + window_length] for start in starts]
    return frames, times, window_length, shift_length


def spectrogram_frame_geometry(
    n_samples: int,
    fs: float,
    *,
    window_size: float,
    shift: float,
    start_time: float,
    n_windows: int | None,
    time_reference: str = "center",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    window_size = float(
        validate_number(
            window_size,
            "window_size",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    shift = float(
        validate_number(
            shift,
            "shift",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    start_time = float(
        validate_number(
            start_time,
            "start_time",
            kind="real",
            minimum=0,
            coerce=True,
        )
    )
    time_reference = validate_choice(time_reference, ("start", "center", "end"), "time_reference")
    if shift > window_size:
        raise ValidationError("shift must not exceed window_size.")
    window_length = round(window_size * fs)
    shift_length = round(shift * fs)
    start_sample = round(start_time * fs)
    if window_length < 1:
        raise ValidationError("window_size is shorter than one sample.")
    if shift_length < 1:
        raise ValidationError("shift is shorter than one sample.")
    if start_sample >= n_samples:
        raise ValidationError("start_time lies outside the signal.")

    available = n_samples - start_sample
    maximum_windows = (
        0 if available < window_length else 1 + (available - window_length) // shift_length
    )
    if n_windows is None:
        count = maximum_windows
    else:
        count = validate_integer(n_windows, "n_windows", minimum=1)
        if count > maximum_windows:
            raise ValidationError("requested windows exceed the available signal duration.")
    if count == 0:
        raise ValidationError("the selected signal does not contain one complete window.")
    starts = start_sample + np.arange(count) * shift_length
    offsets = {
        "start": 0.0,
        "center": window_length / (2.0 * fs),
        "end": window_length / fs,
    }
    times = starts.astype(float) / fs + offsets[time_reference]
    return starts, times, window_length, shift_length


def absolute_frame_times(
    source: np.ndarray | SignalArray,
    relative_times: np.ndarray,
) -> np.ndarray:
    if not isinstance(source, SignalArray):
        return relative_times
    if source.time is not None and source.time.size:
        return relative_times + float(source.time[0])
    return relative_times + float(source.t0)


def restore_spectrogram(
    values: np.ndarray,
    context,
) -> np.ndarray:
    source = context.source
    if isinstance(source, np.ndarray) and source.ndim == 1:
        return values[:, 0, :]
    return values
