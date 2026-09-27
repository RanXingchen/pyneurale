#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed IIR filter designers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_positive_float,
)
from neurale.exceptions import ValidationError
from neurale.signal._validation import validate_fs

from .coefficients import (
    IirCoefficients,
    SosCoefficients,
    StateSpaceCoefficients,
    ZpkCoefficients,
)
from .representations import (
    zpk2sos,
    zpk2ss,
    zpk2tf,
)

IirBand = Literal["lowpass", "highpass", "bandpass", "bandstop"]
IirOutput = Literal["sos", "tf", "zpk", "ss"]
IirKind = Literal["notch", "butterworth", "bessel", "elliptic"]


def notch(
    freq: float,
    *,
    q: float | None = None,
    bandwidth: float | None = None,
    fs: float | None = None,
    output: IirOutput = "sos",
):
    """Design a second-order digital notch filter.

    Parameters
    ----------
    frequency : float
        Positive notch center frequency below Nyquist.
    q : float or None, optional
        Positive quality factor. Exactly one of ``q`` and ``bandwidth`` must
        be provided.
    bandwidth : float or None, optional
        Positive notch bandwidth below Nyquist. Exactly one of ``bandwidth``
        and ``q`` must be provided.
    fs : float or None, optional
        Positive sampling rate in hertz. When omitted, frequencies are
        normalized so that Nyquist is 1.
    output : {"sos", "tf", "zpk", "ss"}, default="sos"
        Coefficient representation returned by the designer.
    Returns
    -------
    SosCoefficients or IirCoefficients or ZpkCoefficients or StateSpaceCoefficients
        Designed notch filter in the representation selected by ``output``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If frequency, quality factor, bandwidth, sampling rate, or output
        representation is invalid.
    """
    rate = _design_rate(fs)
    freq = validate_positive_float(freq, "frequency")
    if freq >= rate / 2:
        raise ValidationError("frequency must lie below Nyquist.")
    if (q is None) == (bandwidth is None):
        raise ValidationError("provide exactly one of q or bandwidth.")
    if q is not None:
        q = validate_positive_float(q, "q")
        bandwidth = freq / q
    else:
        bandwidth = validate_positive_float(bandwidth, "bandwidth")
    if bandwidth >= rate / 2:
        raise ValidationError("bandwidth must lie below Nyquist.")
    normalized_freq = freq / (rate / 2)
    normalized_bandwidth = bandwidth / (rate / 2)
    implementation = load_native_namespace("signal.filtering")
    try:
        z, p, k = implementation.notch_zpk(normalized_freq, normalized_bandwidth)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValidationError(str(exc)) from exc
    metadata_rate = None if fs is None else rate
    return _format_design_trusted_native(z, p, k, output, metadata_rate)


def butter(
    order: int,
    cutoff,
    *,
    band: IirBand = "lowpass",
    fs: float | None = None,
    output: IirOutput = "sos",
):
    """Design a digital Butterworth filter.

    Parameters
    ----------
    order : int
        Positive filter order.
    cutoff : float or array_like
        One cutoff frequency for lowpass or highpass designs, or two strictly
        increasing frequencies for bandpass or bandstop designs.
    band : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
        Filter response type.
    fs : float or None, optional
        Positive sampling rate in hertz. When omitted, frequencies are
        normalized so that Nyquist is 1.
    output : {"sos", "tf", "zpk", "ss"}, default="sos"
        Coefficient representation returned by the designer.
    Returns
    -------
    SosCoefficients or IirCoefficients or ZpkCoefficients or StateSpaceCoefficients
        Designed Butterworth filter in the requested representation.

    Raises
    ------
    neurale.exceptions.ValidationError
        If order, cutoff frequencies, band type, sampling rate, or output
        representation is invalid.
    """
    return _design_classic("butterworth", order, cutoff, band, fs, output)


def bessel(
    order: int,
    cutoff,
    *,
    band: IirBand = "lowpass",
    fs: float | None = None,
    output: IirOutput = "sos",
):
    """Design a phase-normalized digital Bessel filter.

    Parameters
    ----------
    order : int
        Positive filter order not greater than 16.
    cutoff : float or array_like
        One cutoff frequency for lowpass or highpass designs, or two strictly
        increasing frequencies for bandpass or bandstop designs.
    band : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
        Filter response type.
    fs : float or None, optional
        Positive sampling rate in hertz. When omitted, frequencies are
        normalized so that Nyquist is 1.
    output : {"sos", "tf", "zpk", "ss"}, default="sos"
        Coefficient representation returned by the designer.
    Returns
    -------
    SosCoefficients or IirCoefficients or ZpkCoefficients or StateSpaceCoefficients
        Designed Bessel filter in the requested representation.

    Raises
    ------
    neurale.exceptions.ValidationError
        If order, cutoff frequencies, band type, sampling rate, or output
        representation is invalid.
    """
    if validate_integer(order, "order", minimum=1) > 16:
        raise ValidationError("Bessel order must not exceed 16.")
    return _design_classic("bessel", order, cutoff, band, fs, output)


def ellip(
    order: int,
    cutoff,
    *,
    rp: float,
    rs: float,
    band: IirBand = "lowpass",
    fs: float | None = None,
    output: IirOutput = "sos",
):
    """Design a digital elliptic filter.

    Parameters
    ----------
    order : int
        Positive filter order.
    cutoff : float or array_like
        One cutoff frequency for lowpass or highpass designs, or two strictly
        increasing frequencies for bandpass or bandstop designs.
    rp : float
        Positive maximum passband ripple in decibels.
    rs : float
        Positive minimum stopband attenuation in decibels. Must exceed
        ``rp``.
    band : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
        Filter response type.
    fs : float or None, optional
        Positive sampling rate in hertz. When omitted, frequencies are
        normalized so that Nyquist is 1.
    output : {"sos", "tf", "zpk", "ss"}, default="sos"
        Coefficient representation returned by the designer.
    Returns
    -------
    SosCoefficients or IirCoefficients or ZpkCoefficients or StateSpaceCoefficients
        Designed elliptic filter in the requested representation.

    Raises
    ------
    neurale.exceptions.ValidationError
        If order, cutoff frequencies, ripple, attenuation, band type,
        sampling rate, or output representation is invalid.
    """
    rp = validate_positive_float(rp, "rp")
    rs = validate_positive_float(rs, "rs")
    if rp >= rs:
        raise ValidationError("rp must be less than rs.")
    return _design_classic("elliptic", order, cutoff, band, fs, output, rp, rs)


def design_iir(
    kind: IirKind,
    order: int | None = None,
    cutoff=None,
    *,
    band: IirBand = "lowpass",
    fs: float | None = None,
    output: IirOutput = "sos",
    rp: float | None = None,
    rs: float | None = None,
    q: float | None = None,
    bandwidth: float | None = None,
):
    """Design an IIR filter using a selected prototype family.

    Parameters
    ----------
    kind : {"notch", "butterworth", "bessel", "elliptic"}
        IIR design family.
    order : int or None, optional
        Positive filter order required by all designs except ``"notch"``.
    cutoff : float or array_like or None, optional
        Notch frequency or classical filter cutoff frequencies.
    band : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
        Response type for Butterworth, Bessel, and elliptic designs.
    fs : float or None, optional
        Positive sampling rate in hertz. When omitted, frequencies are
        normalized so that Nyquist is 1.
    output : {"sos", "tf", "zpk", "ss"}, default="sos"
        Coefficient representation returned by the designer.
    rp : float or None, optional
        Passband ripple in decibels required by elliptic designs.
    rs : float or None, optional
        Stopband attenuation in decibels required by elliptic designs.
    q : float or None, optional
        Quality factor used by notch designs.
    bandwidth : float or None, optional
        Bandwidth used by notch designs.
    Returns
    -------
    SosCoefficients or IirCoefficients or ZpkCoefficients or StateSpaceCoefficients
        Designed filter in the representation selected by ``output``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the selected design is missing required arguments or any parameter
        is invalid.
    """
    kind = validate_choice(kind, ("notch", "butterworth", "bessel", "elliptic"), "kind")
    if kind == "notch":
        if cutoff is None:
            raise ValidationError("notch design requires cutoff frequency.")
        return notch(
            cutoff,
            q=q,
            bandwidth=bandwidth,
            fs=fs,
            output=output,
        )
    if order is None or cutoff is None:
        raise ValidationError(f"{kind} design requires order and cutoff.")
    if kind == "butterworth":
        return butter(order, cutoff, band=band, fs=fs, output=output)
    if kind == "bessel":
        return bessel(order, cutoff, band=band, fs=fs, output=output)
    if rp is None or rs is None:
        raise ValidationError("elliptic design requires rp and rs.")
    return ellip(
        order,
        cutoff,
        rp=rp,
        rs=rs,
        band=band,
        fs=fs,
        output=output,
    )


_IIR_OUTPUTS = ("sos", "tf", "zpk", "ss")


def _design_classic(kind, order, cutoff, band, fs, output, ripple=0.5, attenuation=30.0):
    output = validate_choice(output, _IIR_OUTPUTS, "output")
    if kind == "bessel" and order > 16:
        raise ValidationError("Bessel order must not exceed 16.")
    implementation = load_native_namespace("signal.filtering")
    cutoff_values = np.asarray(cutoff, dtype=float).reshape(-1)
    metadata_rate = None if fs is None else float(fs)
    rate = 2.0 if fs is None else metadata_rate
    try:
        if output == "ss":
            a, b, c, d = implementation.iir_ss(
                kind,
                order,
                cutoff_values,
                band,
                rate,
                ripple,
                attenuation,
            )
            return StateSpaceCoefficients._from_trusted_arrays(a, b, c, d, metadata_rate)
        if output == "tf":
            b, a = implementation.iir_tf(
                kind, order, cutoff_values, band, rate, ripple, attenuation
            )
            return IirCoefficients._from_trusted_arrays(b, a, metadata_rate)
        if output == "sos":
            sos = implementation.iir_sos(
                kind, order, cutoff_values, band, rate, ripple, attenuation
            )
            return SosCoefficients._from_trusted_array(sos, metadata_rate)
        z, p, k = implementation.iir_zpk(
            kind, order, cutoff_values, band, rate, ripple, attenuation
        )
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValidationError(str(exc)) from exc
    return _format_design_trusted_native(z, p, k, output, metadata_rate)


def _format_design_trusted_native(z, p, k, output, fs):
    output = validate_choice(output, _IIR_OUTPUTS, "output")
    zpk = ZpkCoefficients._from_trusted_arrays(z, p, k, fs)
    if output == "zpk":
        return zpk
    if output == "tf":
        b, a = zpk2tf(zpk.z, zpk.p, zpk.k)
        return IirCoefficients._from_trusted_arrays(b, a, fs)
    if output == "sos":
        sos = zpk2sos(zpk.z, zpk.p, zpk.k)
        return SosCoefficients._from_trusted_array(sos, fs)
    a, b, c, d = zpk2ss(zpk.z, zpk.p, zpk.k)
    return StateSpaceCoefficients._from_trusted_arrays(a, b, c, d, fs)


def _design_rate(fs):
    if fs is None:
        return 2.0
    return validate_fs(fs)
