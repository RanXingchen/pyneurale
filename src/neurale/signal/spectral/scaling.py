#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Power-spectrum range and density scaling."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from neurale._validation import (
    validate_axis,
    validate_choice,
    validate_integer,
    validate_numeric_array,
)
from neurale.exceptions import ValidationError
from neurale.signal._validation import (
    validate_fs,
)

SpectrumSides = Literal["one-sided", "two-sided"]
SpectrumEstimate = Literal["psd", "power"]


def scale_spectrum(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    sides: SpectrumSides = "two-sided",
    nfft: int | None = None,
    fs: float | None = None,
    estimate: SpectrumEstimate = "psd",
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Scale a whole power spectrum as density or power.

    Parameters
    ----------
    spectrum : numpy.ndarray
        One- or two-dimensional whole spectrum.
    frequencies : numpy.ndarray
        Frequency value for every FFT bin.
    sides : {"one-sided", "two-sided"}, default="two-sided"
        Output frequency range.
    nfft : int or None, optional
        FFT length; inferred from ``spectrum`` when omitted.
    fs : float or None, optional
        Sampling rate in hertz. Omit for normalized angular frequency.
    estimate : {"psd", "power"}, default="psd"
        Density or power scaling.
    axis : int, default=0
        Frequency-bin axis.

    Returns
    -------
    scaled : numpy.ndarray
        Scaled spectrum.
    frequencies : numpy.ndarray
        Output frequency vector.
    units : str
        Frequency units, ``"Hz"`` or ``"rad/sample"``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If dimensions, lengths, options, or sampling rate are invalid.
    """
    spectrum = validate_numeric_array(spectrum, "spectrum")
    if spectrum.ndim not in (1, 2):
        raise ValidationError("spectrum must be one- or two-dimensional.")
    axis = validate_axis(axis, spectrum.ndim)
    freqs = validate_numeric_array(freqs, "frequencies", ndim=1)
    sides = validate_choice(sides, ("one-sided", "two-sided"), "sides")
    estimate = validate_choice(estimate, ("psd", "power"), "estimate")

    n_bins = spectrum.shape[axis]
    if nfft is None:
        nfft = n_bins
    else:
        nfft = validate_integer(nfft, "nfft", minimum=1)
    if n_bins != nfft:
        raise ValidationError(
            f"spectrum has {n_bins} bins along axis {axis}, expected nfft={nfft}."
        )
    if freqs.size != nfft:
        raise ValidationError("frequencies length must equal nfft.")

    sample_major = np.moveaxis(spectrum, axis, 0)
    scaled = np.array(
        sample_major,
        dtype=np.result_type(sample_major.dtype, np.float64),
        copy=True,
    )
    output_freqs = freqs.copy()
    if sides == "one-sided":
        end = (nfft // 2) + 1
        scaled = scaled[:end].copy()
        if nfft % 2:
            scaled[1:] *= 2
        elif end > 2:
            scaled[1:-1] *= 2
        output_freqs = output_freqs[:end]

    if fs is None:
        density_scale = 2.0 * math.pi
        units = "rad/sample"
    else:
        density_scale = validate_fs(fs)
        if density_scale is None:
            raise ValidationError("fs must be positive when provided.")
        units = "Hz"
    if estimate == "psd":
        scaled = scaled / density_scale

    return np.moveaxis(scaled, 0, axis), output_freqs, units
