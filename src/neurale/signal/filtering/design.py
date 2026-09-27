#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""FIR filter-design functions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_numeric_array,
    validate_positive_float,
)
from neurale.exceptions import ValidationError
from neurale.signal._validation import validate_fs

from .coefficients import FirCoefficients

FirBand = Literal["lowpass", "highpass", "bandpass", "bandstop"]
FirWindow = Literal["hann", "hamming", "blackman", "flattop"]

_ORDER_FACTORS: dict[FirWindow, float] = {
    "hann": 6.20,
    "hamming": 6.60,
    "blackman": 11.0,
    "flattop": 19.6,
}


def firls(
    order: int,
    bands: Sequence[float] | np.ndarray,
    desired: Sequence[float] | np.ndarray,
    *,
    weight: Sequence[float] | np.ndarray | None = None,
    fs: float | None = None,
) -> FirCoefficients:
    """Design a linear-phase FIR filter by weighted least squares.

    Parameters
    ----------
    order : int
        Positive filter order.
    bands : sequence of float or numpy.ndarray
        Ordered, non-overlapping band-edge pairs from DC to Nyquist.
    desired : sequence of float or numpy.ndarray
        Desired gain at every band edge.
    weight : sequence of float or numpy.ndarray or None, optional
        Positive weight for each band.
    fs : float or None, optional
        Sampling rate in hertz. Frequencies are normalized when omitted.
    Returns
    -------
    neurale.signal.filtering.coefficients.FirCoefficients
        Designed coefficients.

    Raises
    ------
    neurale.exceptions.ValidationError
        If order, bands, gains, weights, or sampling rate are invalid.
    """
    order = validate_integer(order, "order", minimum=1)
    normalized_bands, desired_values, weights = _validate_firls_inputs(bands, desired, weight, fs)
    if desired_values[-1] != 0 and normalized_bands[-1] == 1 and order % 2:
        warnings.warn(
            "Increasing FIRLS order by one because a Type-II symmetric FIR "
            "filter must have zero gain at Nyquist.",
            RuntimeWarning,
            stacklevel=2,
        )
        order += 1

    taps = load_native_namespace("signal.filtering").firls(
        order, normalized_bands, desired_values, weights
    )
    return FirCoefficients._from_trusted_array(taps, fs=fs)


def firwin(
    order: int,
    cutoff: float | Sequence[float] | np.ndarray,
    *,
    band: FirBand = "lowpass",
    window: FirWindow = "hamming",
    fs: float | None = None,
    scale: bool = True,
) -> FirCoefficients:
    """Design a windowed linear-phase FIR filter.

    Parameters
    ----------
    order : int
        Positive filter order.
    cutoff : float or sequence of float
        One cutoff for low/high-pass or two for band-pass/stop.
    band : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
        Filter response type.
    window : {"hann", "hamming", "blackman", "flattop"}, default="hamming"
        Design window.
    fs : float or None, optional
        Sampling rate in hertz. Nyquist-normalized frequencies are used when omitted.
    scale : bool, default=True
        Normalize the response at an appropriate reference frequency.

    Returns
    -------
    neurale.signal.filtering.coefficients.FirCoefficients
        Designed coefficients.

    Raises
    ------
    neurale.exceptions.ValidationError
        If order, cutoffs, response type, window, or sampling rate are invalid.
    """
    if type(order) is not int or order < 1:
        raise ValidationError("order must be an integer greater than or equal to 1.")
    if band not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValidationError("band must be one of lowpass, highpass, bandpass, bandstop.")
    if window not in ("hann", "hamming", "blackman", "flattop"):
        raise ValidationError("window must be one of hann, hamming, blackman, flattop.")
    fs = 2.0 if fs is None else validate_fs(fs)
    expected = 1 if band in ("lowpass", "highpass") else 2
    numtaps = order + 1
    if band in ("highpass", "bandstop") and numtaps % 2 == 0:
        warnings.warn(
            "Increasing FIR order by one because an even-length symmetric FIR "
            "filter cannot have nonzero gain at Nyquist.",
            RuntimeWarning,
            stacklevel=2,
        )
        numtaps += 1
    native = load_native_namespace("signal.filtering")
    if expected == 1:
        if np.isscalar(cutoff):
            cutoff_value = _validate_scalar_cutoff(float(cutoff), fs)
        else:
            cutoff_values = _validate_cutoff_values(cutoff, expected, band, fs)
            cutoff_value = float(cutoff_values[0])
        taps = native.firwin_scalar(
            numtaps - 1,
            cutoff_value / (fs / 2.0),
            band,
            window,
            bool(scale),
        )
    else:
        cutoff_values = _validate_cutoff_values(cutoff, expected, band, fs)
        normalized_cutoff = cutoff_values / (fs / 2.0)
        taps = native.firwin_pair(
            numtaps - 1,
            float(normalized_cutoff[0]),
            float(normalized_cutoff[1]),
            band,
            window,
            bool(scale),
        )
    return FirCoefficients._from_trusted_array(taps, fs=fs)


def fir_order(
    transition_width: float,
    fs: float,
    *,
    window: FirWindow = "hamming",
) -> int:
    """Estimate FIR order from transition bandwidth.

    Parameters
    ----------
    transition_width : float
        Positive transition width in hertz.
    fs : float
        Positive sampling rate in hertz.
    window : {"hann", "hamming", "blackman", "flattop"}, default="hamming"
        Window used by the empirical order rule.

    Returns
    -------
    int
        Estimated positive filter order.

    Raises
    ------
    neurale.exceptions.ValidationError
        If numeric values or ``window`` are invalid.
    """
    transition_width = validate_positive_float(transition_width, "transition_width")
    fs = validate_positive_float(fs, "fs")
    window = validate_choice(window, _ORDER_FACTORS, "window")
    return int(np.ceil(_ORDER_FACTORS[window] * fs / (2.0 * transition_width)))


def fir_transition_width(
    order: int,
    fs: float = 2.0,
    *,
    window: FirWindow = "hamming",
) -> float:
    """Return transition bandwidth implied by an FIR order.

    Parameters
    ----------
    order : int
        Positive filter order.
    fs : float, default=2.0
        Positive sampling rate in hertz.
    window : {"hann", "hamming", "blackman", "flattop"}, default="hamming"
        Window used by the empirical order rule.

    Returns
    -------
    float
        Estimated transition bandwidth in hertz.
    """
    order = validate_integer(order, "order", minimum=1)
    fs = validate_positive_float(fs, "fs")
    window = validate_choice(window, _ORDER_FACTORS, "window")
    return _ORDER_FACTORS[window] * fs / (2.0 * order)


def phase_comp_fir(
    phase: np.ndarray,
    fs: float,
    *,
    num_taps: int = 25,
    freqs: np.ndarray | None = None,
) -> FirCoefficients:
    """Design an FIR filter that approximately cancels a phase response.

    Parameters
    ----------
    phase : numpy.ndarray
        One-dimensional finite phase response from DC through Nyquist.
    fs : float
        Positive sampling rate in hertz.
    num_taps : int, default=25
        Number of compensator coefficients.
    frequencies : numpy.ndarray or None, optional
        Strictly increasing frequencies spanning DC through Nyquist.

    Returns
    -------
    neurale.signal.filtering.coefficients.FirCoefficients
        Phase-compensation coefficients.

    Raises
    ------
    neurale.exceptions.ValidationError
        If phase, frequencies, sampling rate, or tap count are invalid.
    """
    phase = validate_numeric_array(np.asarray(phase), "phase", ndim=1)
    if phase.size < 2 or np.iscomplexobj(phase):
        raise ValidationError("phase must be a real vector with at least two values.")
    if np.any(~np.isfinite(phase)):
        raise ValidationError("phase must contain finite values.")
    fs = validate_positive_float(fs, "fs")
    num_taps = validate_integer(num_taps, "num_taps", minimum=2)
    if freqs is None:
        freqs = np.linspace(0.0, fs / 2.0, phase.size)
    else:
        freqs = validate_numeric_array(np.asarray(freqs), "frequencies", ndim=1).astype(
            float, copy=False
        )
        if freqs.size != phase.size:
            raise ValidationError("frequencies and phase must have equal length.")
    if np.any(~np.isfinite(freqs)):
        raise ValidationError("frequencies must contain finite values.")
    if freqs[0] != 0 or freqs[-1] != fs / 2:
        raise ValidationError("frequencies must span DC through Nyquist.")
    if np.any(np.diff(freqs) <= 0):
        raise ValidationError("frequencies must be strictly increasing.")

    response = np.exp(-1j * phase)
    taps = _least_squares_fir_response(freqs, response, fs, num_taps)
    return FirCoefficients(taps, fs=fs)


def _least_squares_fir_response(
    freqs: np.ndarray,
    response: np.ndarray,
    fs: float,
    num_taps: int,
) -> np.ndarray:
    omega = 2.0 * np.pi * freqs / fs
    basis = np.exp(-1j * np.outer(omega, np.arange(num_taps)))
    system = np.vstack((basis.real, basis.imag))
    target = np.concatenate((response.real, response.imag))
    taps, *_ = np.linalg.lstsq(system, target, rcond=None)
    return np.asarray(taps, dtype=float)


def _cutoff_values(cutoff) -> np.ndarray:
    if np.isscalar(cutoff):
        return np.asarray([float(cutoff)], dtype=float)
    return np.asarray(cutoff, dtype=float).reshape(-1)


def _validate_cutoff_values(cutoff, expected: int, band: str, fs: float) -> np.ndarray:
    cutoff_values = _cutoff_values(cutoff)
    if cutoff_values.size != expected:
        raise ValidationError(f"{band} requires {expected} cutoff frequency value(s).")
    if np.any(~np.isfinite(cutoff_values)):
        raise ValidationError("cutoff frequencies must be finite.")
    if np.any(cutoff_values <= 0) or np.any(cutoff_values >= fs / 2):
        raise ValidationError("cutoff frequencies must lie strictly between 0 and Nyquist.")
    if np.any(np.diff(cutoff_values) <= 0):
        raise ValidationError("cutoff frequencies must be strictly increasing.")
    return cutoff_values


def _validate_scalar_cutoff(cutoff: float, fs: float) -> float:
    if not np.isfinite(cutoff):
        raise ValidationError("cutoff frequencies must be finite.")
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValidationError("cutoff frequencies must lie strictly between 0 and Nyquist.")
    return cutoff


def _validate_firls_inputs(
    bands: Sequence[float] | np.ndarray,
    desired: Sequence[float] | np.ndarray,
    weight: Sequence[float] | np.ndarray | None,
    fs: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_values = np.asarray(bands, dtype=float).reshape(-1)
    desired_values = np.asarray(desired, dtype=float).reshape(-1)
    if band_values.size < 2 or band_values.size % 2:
        raise ValidationError("bands must contain one or more frequency pairs.")
    if desired_values.size != band_values.size:
        raise ValidationError("desired must contain one value per band edge.")
    if np.any(~np.isfinite(band_values)) or np.any(~np.isfinite(desired_values)):
        raise ValidationError("bands and desired must contain finite values.")
    nyquist = 1.0 if fs is None else validate_fs(fs) / 2.0
    normalized = band_values / nyquist
    pairs = normalized.reshape(-1, 2)
    if np.any(normalized < 0) or np.any(normalized > 1):
        raise ValidationError("bands must lie between DC and Nyquist.")
    if np.any(pairs[:, 1] <= pairs[:, 0]):
        raise ValidationError("each band must have positive width.")
    if np.any(pairs[1:, 0] < pairs[:-1, 1]):
        raise ValidationError("bands must be ordered and non-overlapping.")
    if weight is None:
        weights = np.ones(pairs.shape[0], dtype=float)
    else:
        weights = np.asarray(weight, dtype=float).reshape(-1)
        if weights.size != pairs.shape[0]:
            raise ValidationError("weight must contain one value per band.")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValidationError("weight values must be finite and positive.")
    return normalized, desired_values, weights
