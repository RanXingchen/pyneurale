#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Welch power spectral density estimation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_number,
)
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._input import normalize_signal_input

from ._psd import (
    PsdDetrend,
    postprocess_psd,
    resolve_psd_sampling_rate,
    resolve_smoothing_width,
    restore_psd,
    select_time_range,
    validate_bool,
    validate_detrend,
)

PsdWindow = Literal["hann", "hamming", "blackman", "flattop", "boxcar"]


def welch_psd(
    x: np.ndarray | SignalArray,
    fs: float | None = None,
    *,
    nfft: int = 256,
    window_length: int | None = None,
    window: PsdWindow = "hann",
    overlap: float = 0.5,
    detrend: PsdDetrend = "none",
    time_range: Sequence[float] | None = None,
    db: bool = False,
    smoothing_width: float | None = None,
    smoothing_width_hz: float | None = None,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PSD by Welch averaging.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional real or complex signal.
    fs : float or None, optional
        Sampling rate in hertz. Omit for radians per sample. For
        ``SignalArray`` this is inferred from metadata.
    nfft : int, default=256
        Number of FFT points.
    window_length : int or None, optional
        Samples per segment. Defaults to ``nfft``.
    window : {"hann", "hamming", "blackman", "flattop", "boxcar"}
        Symmetric analysis window.
    overlap : float, default=0.5
        Fractional overlap in ``[0, 1)``.
    detrend : {"none", "mean", "linear"}, default="none"
        Segment preprocessing.
    time_range : sequence of two float or None, optional
        Half-open interval in seconds relative to the first input sample.
    db : bool, default=False
        Convert the averaged linear PSD to decibels.
    smoothing_width : float or None, optional
        Gaussian half-height width in frequency bins.
    smoothing_width_hz : float or None, optional
        Gaussian half-height width in hertz. Mutually exclusive with
        ``smoothing_width``.
    axis : int, default=0
        Sample axis for ndarray input.

    Returns
    -------
    psd : numpy.ndarray
        PSD with the frequency axis replacing the sample axis.
    frequencies : numpy.ndarray
        Frequencies in hertz or radians per sample.
    """
    data, context = normalize_signal_input(x, axis=axis)
    rate = resolve_psd_sampling_rate(x, fs)
    selected = select_time_range(data, rate, time_range)
    if not np.all(np.isfinite(selected)):
        raise ValidationError("signal must contain only finite values.")

    nfft = validate_integer(nfft, "nfft", minimum=1)
    segment_length = (
        nfft
        if window_length is None
        else validate_integer(window_length, "window_length", minimum=1)
    )
    if segment_length > selected.shape[0]:
        raise ValidationError("window_length must not exceed the selected sample count.")
    if nfft < segment_length:
        raise ValidationError("nfft must not be shorter than window_length.")
    window = validate_choice(
        window,
        ("hann", "hamming", "blackman", "flattop", "boxcar"),
        "window",
    )
    overlap = float(
        validate_number(
            overlap,
            "overlap",
            kind="real",
            minimum=0,
            maximum=1,
            maximum_inclusive=False,
            coerce=True,
        )
    )
    detrend = validate_detrend(detrend)
    db = validate_bool(db, "db")
    noverlap = math.floor(overlap * segment_length)
    analysis_window = scipy_signal.get_window(window, segment_length, fftbins=False)
    scipy_detrend: str | bool = {
        "none": False,
        "mean": "constant",
        "linear": "linear",
    }[detrend]
    scipy_rate = 2.0 * math.pi if rate is None else rate
    freqs, values = scipy_signal.welch(
        selected,
        fs=scipy_rate,
        window=analysis_window,
        nperseg=segment_length,
        noverlap=noverlap,
        nfft=nfft,
        detrend=scipy_detrend,
        return_onesided=not np.iscomplexobj(selected),
        scaling="density",
        axis=0,
    )
    smoothing_width = resolve_smoothing_width(smoothing_width, smoothing_width_hz, freqs)
    values = postprocess_psd(values, db, smoothing_width)
    return restore_psd(values, freqs, context)
