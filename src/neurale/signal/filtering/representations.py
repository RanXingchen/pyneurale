#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Native conversions between SISO filter representations."""

from __future__ import annotations

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array


def zpk2tf(z, p, k):
    """Convert zeros, poles, and gain to transfer-function coefficients.

    Parameters
    ----------
    z : array_like
        System zeros.
    p : array_like
        System poles.
    k : float or complex
        System gain.

    Returns
    -------
    num : numpy.ndarray
        Numerator coefficients.
    den : numpy.ndarray
        Denominator coefficients.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    return _call_representation(
        "zpk2tf",
        native_array(z, complex),
        native_array(p, complex),
        k,
        tuple_result=True,
    )


def ss2tf(a, b, c, d):
    """Convert SISO state space to transfer-function coefficients.

    Parameters
    ----------
    a : array_like
        Square state-transition matrix.
    b : array_like
        Input vector.
    c : array_like
        Output vector.
    d : float
        Feedthrough scalar.

    Returns
    -------
    num : numpy.ndarray
        Numerator coefficients.
    den : numpy.ndarray
        Denominator coefficients.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    return _call_representation(
        "ss2tf",
        *_check_state_space_params(a, b, c, d),
        tuple_result=True,
    )


def zpk2ss(z, p, k):
    """Convert zeros, poles, and gain to SISO state space.

    Parameters
    ----------
    z : array_like
        System zeros.
    p : array_like
        System poles.
    k : float or complex
        System gain.

    Returns
    -------
    a : numpy.ndarray
        State-transition matrix.
    b : numpy.ndarray
        Input vector.
    c : numpy.ndarray
        Output vector.
    d : float
        Feedthrough scalar.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    return _call_representation(
        "zpk2ss",
        native_array(z, complex),
        native_array(p, complex),
        k,
        tuple_result=True,
    )


def tf2ss(num, den):
    """Convert transfer-function coefficients to SISO state space.

    Parameters
    ----------
    num : array_like
        Numerator coefficients.
    den : array_like
        Denominator coefficients.

    Returns
    -------
    a : numpy.ndarray
        State-transition matrix.
    b : numpy.ndarray
        Input vector.
    c : numpy.ndarray
        Output vector.
    d : float
        Feedthrough scalar.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    return _call_representation(
        "tf2ss",
        native_array(num, float),
        native_array(den, float),
        tuple_result=True,
    )


def tf2zpk(num, den):
    """Convert transfer-function coefficients to zeros, poles, and gain.

    Parameters
    ----------
    num : array_like
        Numerator coefficients.
    den : array_like
        Denominator coefficients.

    Returns
    -------
    z : numpy.ndarray
        System zeros.
    p : numpy.ndarray
        System poles.
    k : float
        System gain.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    return _call_representation(
        "tf2zpk",
        native_array(num, float),
        native_array(den, float),
        tuple_result=True,
    )


def ss2zpk(a, b, c, d):
    """Convert SISO state space to zeros, poles, and gain.

    Parameters
    ----------
    a : array_like
        Square state-transition matrix.
    b : array_like
        Input vector.
    c : array_like
        Output vector.
    d : float
        Feedthrough scalar.

    Returns
    -------
    z : numpy.ndarray
        System zeros.
    p : numpy.ndarray
        System poles.
    k : float
        System gain.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    return _call_representation(
        "ss2zpk",
        *_check_state_space_params(a, b, c, d),
        tuple_result=True,
    )


def zpk2sos(z, p, k):
    """Convert zeros, poles, and gain to second-order sections.

    Parameters
    ----------
    z : array_like
        System zeros.
    p : array_like
        System poles.
    k : float
        Real-valued system gain.

    Returns
    -------
    numpy.ndarray
        Second-order sections with shape ``(n_sections, 6)``.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or conversion fails.
    """
    try:
        if isinstance(k, complex):
            if k.imag != 0:
                raise ValueError("gain must be real.")
            gain = k.real
        else:
            gain = float(k)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(str(exc)) from exc

    return _call_representation(
        "zpk2sos",
        native_array(z, complex),
        native_array(p, complex),
        gain,
        tuple_result=False,
    )


def lp2lp_ss(a, b, c, d, cutoff):
    """Scale an analog lowpass state-space prototype.

    Parameters
    ----------
    a, b, c, d : array_like
        SISO state-space prototype.
    cutoff : float
        Target lowpass cutoff frequency.

    Returns
    -------
    a, b, c, d : tuple
        Transformed SISO state-space model.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "lp2lp_ss",
        *_check_state_space_params(a, b, c, d),
        cutoff,
        tuple_result=True,
    )


def lp2hp_ss(a, b, c, d, cutoff):
    """Transform an analog lowpass state-space prototype to highpass.

    Parameters
    ----------
    a, b, c, d : array_like
        SISO state-space prototype.
    cutoff : float
        Target highpass cutoff frequency.

    Returns
    -------
    a, b, c, d : tuple
        Transformed SISO state-space model.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "lp2hp_ss",
        *_check_state_space_params(a, b, c, d),
        cutoff,
        tuple_result=True,
    )


def lp2bp_ss(a, b, c, d, center_freq, bandwidth):
    """Transform an analog lowpass state-space prototype to bandpass.

    Parameters
    ----------
    a, b, c, d : array_like
        SISO state-space prototype.
    center_freq : float
        Bandpass center frequency.
    bandwidth : float
        Positive bandpass bandwidth.

    Returns
    -------
    a, b, c, d : tuple
        Transformed SISO state-space model.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "lp2bp_ss",
        *_check_state_space_params(a, b, c, d),
        center_freq,
        bandwidth,
        tuple_result=True,
    )


def lp2bs_ss(a, b, c, d, center_freq, bandwidth):
    """Transform an analog lowpass state-space prototype to bandstop.

    Parameters
    ----------
    a, b, c, d : array_like
        SISO state-space prototype.
    center_freq : float
        Bandstop center frequency.
    bandwidth : float
        Positive bandstop bandwidth.

    Returns
    -------
    a, b, c, d : tuple
        Transformed SISO state-space model.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "lp2bs_ss",
        *_check_state_space_params(a, b, c, d),
        center_freq,
        bandwidth,
        tuple_result=True,
    )


def bilinear_zpk(z, p, k, fs, *, prewarp_freq=None):
    """Map an analog zero-pole-gain model to a digital model.

    Parameters
    ----------
    z : array_like
        Analog zeros.
    p : array_like
        Analog poles.
    k : float or complex
        Analog gain.
    fs : float
        Digital sampling rate.
    prewarp_frequency : float or None, optional
        Frequency to preserve exactly during the bilinear transform.

    Returns
    -------
    z : numpy.ndarray
        Digital zeros.
    p : numpy.ndarray
        Digital poles.
    k : float
        Digital gain.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "bilinear_zpk",
        native_array(z, complex),
        native_array(p, complex),
        k,
        fs,
        prewarp_freq,
        tuple_result=True,
    )


def bilinear_ss(a, b, c, d, fs, *, prewarp_freq=None):
    """Map a continuous state-space model to discrete time.

    Parameters
    ----------
    a, b, c, d : array_like
        Continuous-time SISO state-space model.
    fs : float
        Digital sampling rate.
    prewarp_frequency : float or None, optional
        Frequency to preserve exactly during the bilinear transform.

    Returns
    -------
    a, b, c, d : tuple
        Discrete-time SISO state-space model.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "bilinear_ss",
        *_check_state_space_params(a, b, c, d),
        fs,
        prewarp_freq,
        tuple_result=True,
    )


def bilinear_tf(num, den, fs, *, prewarp_freq=None):
    """Map an analog transfer function to a digital transfer function.

    Parameters
    ----------
    num : array_like
        Analog numerator coefficients.
    den : array_like
        Analog denominator coefficients.
    fs : float
        Digital sampling rate.
    prewarp_frequency : float or None, optional
        Frequency to preserve exactly during the bilinear transform.

    Returns
    -------
    num : numpy.ndarray
        Digital numerator coefficients.
    den : numpy.ndarray
        Digital denominator coefficients.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native representation extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs are invalid or transformation fails.
    """
    return _call_representation(
        "bilinear_tf",
        native_array(num, float),
        native_array(den, float),
        fs,
        prewarp_freq,
        tuple_result=True,
    )


def _call_representation(name: str, *arguments, tuple_result: bool = False):
    try:
        result = getattr(load_native_namespace("signal.representations"), name)(*arguments)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValidationError(str(exc)) from exc
    return tuple(result) if tuple_result else result


def _check_state_space_params(a, b, c, d):
    try:
        a = native_array(a, float)
        b = np.asarray(b)
        c = np.asarray(c)
        if b.ndim == 2 and b.shape[1] == 1:
            b = b[:, 0]
        if c.ndim == 2 and c.shape[0] == 1:
            c = c[0]
        return (
            a,
            native_array(b, float),
            native_array(c, float),
            float(np.asarray(d).reshape(-1)[0]),
        )
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValidationError(str(exc)) from exc
