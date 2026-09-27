#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Windowed short-time power spectral density."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

from neurale._validation import validate_choice, validate_integer
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._input import normalize_signal_input

from ._psd import (
    PsdDetrend,
    postprocess_psd,
    preprocess_segment,
    resolve_psd_sampling_rate,
    resolve_smoothing_width,
    validate_bool,
    validate_detrend,
)
from ._spectrogram import (
    absolute_frame_times,
    restore_spectrogram,
    spectrogram_frames,
)

SpectrogramWindow = Literal["hann", "hamming", "blackman", "flattop", "boxcar"]


def stft_spectrogram(
    x: np.ndarray | SignalArray,
    fs: float | None = None,
    *,
    window_size: float,
    shift: float,
    n_windows: int | None = None,
    start_time: float = 0.0,
    nfft: int | None = None,
    window: SpectrogramWindow = "hann",
    detrend: PsdDetrend = "none",
    db: bool = False,
    smoothing_width: float | None = None,
    smoothing_width_hz: float | None = None,
    time_reference: str = "center",
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a short-time Fourier transform power spectrogram.

    The result layout is ``(frequency, channel, frame)``. One-dimensional
    ndarray input omits the channel axis. Frame times refer to the window
    center by default and may instead refer to the start or end.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional signal.
    fs : float or None, optional
        Sampling rate in hertz. For ``SignalArray``, the rate is inferred from
        metadata when omitted and checked against metadata when supplied.
    window_size : float
        Window duration in seconds.
    shift : float
        Step between adjacent windows in seconds.
    n_windows : int or None, optional
        Number of windows to compute. When omitted, all complete windows are
        used.
    start_time : float, default=0.0
        Windowing start time in seconds relative to the first sample.
    nfft : int or None, optional
        FFT length. Defaults to the window length.
    window : {"hann", "hamming", "blackman", "flattop", "boxcar"}, default="hann"
        Analysis window.
    detrend : {"none", "mean", "linear"}, default="none"
        Detrending applied independently to each window.
    db : bool, default=False
        Convert power values to decibels.
    smoothing_width : float or None, optional
        Gaussian half-height smoothing width in frequency bins.
    smoothing_width_hz : float or None, optional
        Gaussian half-height smoothing width in hertz.
    time_reference : {"center", "start", "end"}, default="center"
        Time reference used for returned frame times.
    axis : int, default=0
        Sample axis for ndarray input.

    Returns
    -------
    spectrogram : numpy.ndarray
        Power spectrogram. The layout is ``(frequency, channel, frame)`` for
        2D input and ``(frequency, frame)`` for 1D
        ndarray input.
    frequencies : numpy.ndarray
        Frequency coordinate in hertz.
    times : numpy.ndarray
        Frame times in seconds.

    Raises
    ------
    neurale.exceptions.ValidationError
        If input data or spectrogram parameters are invalid.
    """
    data, context = normalize_signal_input(x, axis=axis)
    rate = resolve_psd_sampling_rate(x, fs)
    if rate is None:
        raise ValidationError("stft_spectrogram requires a sampling rate.")
    if not np.all(np.isfinite(data)):
        raise ValidationError("signal must contain only finite values.")
    frames, times, window_length, _ = spectrogram_frames(
        data,
        rate,
        window_size=window_size,
        shift=shift,
        start_time=start_time,
        n_windows=n_windows,
        time_reference=time_reference,
    )
    fft_length = window_length if nfft is None else validate_integer(nfft, "nfft", minimum=1)
    if fft_length < window_length:
        raise ValidationError("nfft must not be shorter than the window.")
    window = validate_choice(
        window,
        ("hann", "hamming", "blackman", "flattop", "boxcar"),
        "window",
    )
    detrend = validate_detrend(detrend)
    db = validate_bool(db, "db")
    analysis_window = scipy_signal.get_window(window, window_length, fftbins=False)
    spectra = []
    freqs = None
    for frame in frames:
        prepared = preprocess_segment(frame, detrend)
        freqs, _, spectrum = scipy_signal.spectrogram(
            prepared,
            fs=rate,
            window=analysis_window,
            nperseg=window_length,
            noverlap=0,
            nfft=fft_length,
            detrend=False,
            return_onesided=not np.iscomplexobj(prepared),
            scaling="density",
            mode="psd",
            axis=0,
        )
        spectra.append(np.asarray(spectrum[..., 0], dtype=float))
    values = np.stack(spectra, axis=-1)
    smoothing_width = resolve_smoothing_width(smoothing_width, smoothing_width_hz, freqs)
    for frame_idx in range(values.shape[-1]):
        values[..., frame_idx] = postprocess_psd(values[..., frame_idx], db, smoothing_width)
    return (
        restore_spectrogram(values, context),
        np.asarray(freqs),
        absolute_frame_times(context.source, times),
    )
