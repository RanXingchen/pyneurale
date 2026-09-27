#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Frequency-grid construction."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from neurale._validation import validate_choice, validate_integer
from neurale.exceptions import ValidationError
from neurale.signal._validation import validate_fs

FrequencyRange = Literal["whole", "half"]


def freq_vector(
    n_points: int = 1024,
    fs: float | None = None,
    *,
    centered: bool = False,
    freq_range: FrequencyRange = "whole",
) -> np.ndarray:
    """Construct the frequency grid used by PyNeurale spectral estimators.

    Parameters
    ----------
    n_points : int, default=1024
        Positive number of frequency points.
    fs : float or None, optional
        Sampling rate in hertz. Omit for radians per sample.
    centered : bool, default=False
        Center the grid around zero.
    freq_range : {"whole", "half"}, default="whole"
        Return a full-period or half-period grid.

    Returns
    -------
    numpy.ndarray
        Frequency vector in hertz or radians per sample.

    Raises
    ------
    neurale.exceptions.ValidationError
        If point count, sampling rate, centering, or range is invalid.
    """
    n_points = validate_integer(n_points, "n_points", minimum=1)
    if not isinstance(centered, bool):
        raise ValidationError("centered must be a bool.")
    freq_range = validate_choice(freq_range, ("whole", "half"), "freq_range")
    period = 2.0 * math.pi if fs is None else validate_fs(fs)
    if period is None:
        raise ValidationError("fs must be positive when provided.")

    freqs = np.arange(n_points, dtype=float) * (period / n_points)
    if n_points == 1:
        return freqs

    nyquist = period / 2.0
    half_points = n_points // 2
    is_odd = bool(n_points % 2)
    if is_odd:
        half_resolution = period / n_points / 2.0
        freqs[half_points] = nyquist - half_resolution
        freqs[half_points + 1] = nyquist + half_resolution
    else:
        freqs[half_points] = nyquist
    freqs[-1] = period - period / n_points

    if freq_range == "whole":
        if not centered:
            return freqs
        negative_end = half_points + 1 if is_odd else half_points
        return np.concatenate((-freqs[1:negative_end][::-1], freqs[: half_points + 1]))

    positive_half = freqs[: half_points + 1]
    if not centered:
        return positive_half

    half_is_odd = bool(half_points % 2)
    quarter_points = (half_points + 1) // 2 if half_is_odd else half_points // 2
    negative_end = quarter_points if half_is_odd else quarter_points + 1
    centered_half = np.concatenate(
        (-positive_half[1:negative_end][::-1], positive_half[:negative_end])
    )
    if n_points % 4 == 0 and centered_half.size:
        centered_half[-1] = nyquist / 2.0
    return centered_half
