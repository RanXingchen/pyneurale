#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Discrete prolate spheroidal sequences and multitaper spectra."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_choice,
    validate_integer,
)
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array
from neurale.signal._input import normalize_signal_input
from neurale.signal._numerics import next_power_of_two
from neurale.signal._validation import validate_fs
from neurale.signal.windows import _validate_nw, dpss

from ._psd import (
    PsdDetrend,
    postprocess_psd,
    preprocess_segment,
    resolve_psd_sampling_rate,
    resolve_smoothing_width,
    restore_psd,
    select_time_range,
    validate_bool,
    validate_detrend,
)
from ._spectrogram import (
    absolute_frame_times,
    restore_spectrogram,
    spectrogram_frames,
)
from .frequency import freq_vector

MultitaperWeighting = Literal["unity", "eigen", "adaptive"]
MultitaperSides = Literal["auto", "one-sided", "two-sided"]
IncompleteWindow = Literal["drop", "pad", "error"]


@dataclass(frozen=True, slots=True)
class MultitaperPlanData:
    n_samples: int
    n_channels: int
    fft_length: int
    density_scale: float
    n_tapers: int
    tapers: np.ndarray
    ratios: np.ndarray
    weighting: MultitaperWeighting
    output_sides: Literal["one-sided", "two-sided"]
    complex_input: bool
    freqs: np.ndarray


class MultitaperPsdProcessor:
    """Reusable native multitaper PSD processor for repeated low-latency calls.

    The processor handles one complete sample-major block per call. High-level
    offline functions use it as the native compute core and apply windowing,
    detrending, smoothing, and decibel conversion around it.

    A processor instance is not thread-safe. Do not call ``process`` or reset
    its adaptive state concurrently on the same instance.

    Parameters
    ----------
    n_samples : int
        Samples per processed block.
    n_channels : int, default=1
        Channels per processed block.
    fs : float or None, optional
        Sampling rate in hertz. Frequencies are radians per sample when
        omitted.
    nw : float, default=4.0
        DPSS time-half-bandwidth product.
    n_tapers : int or None, optional
        Number of tapers. Defaults to ``floor(2*nw - 1)``.
    nfft : int or None, optional
        FFT length. Defaults to ``max(256, next_power_of_two(n_samples))``.
    weighting : {"unity", "eigen", "adaptive"}, default="adaptive"
        Rule used to combine individual tapered spectra.
    sides : {"auto", "one-sided", "two-sided"}, default="auto"
        Spectrum range. ``"auto"`` selects one-sided for real input.
    complex_input : bool, default=False
        Process complex-valued input blocks.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native spectral extension is unavailable.
    neurale.exceptions.ValidationError
        If dimensions or estimator parameters are invalid.
    """

    def __init__(
        self,
        n_samples: int,
        n_channels: int = 1,
        fs: float | None = None,
        *,
        nw: float = 4.0,
        n_tapers: int | None = None,
        nfft: int | None = None,
        weighting: MultitaperWeighting = "adaptive",
        sides: MultitaperSides = "auto",
        complex_input: bool = False,
    ) -> None:
        plan = _multitaper_plan_data(
            n_samples,
            n_channels,
            fs,
            nw,
            n_tapers,
            nfft,
            weighting,
            sides,
            bool(complex_input),
        )
        native = _native_spectral()

        self._n_samples = plan.n_samples
        self._n_channels = plan.n_channels
        self._complex_input = plan.complex_input
        self._freqs = plan.freqs
        self._processor = native._MultitaperPsdProcessor(
            plan.n_samples,
            plan.n_channels,
            plan.fft_length,
            plan.density_scale,
            native_array(plan.tapers, float),
            plan.n_tapers,
            native_array(plan.ratios, float),
            plan.weighting,
            plan.output_sides == "one-sided",
            not plan.complex_input,
        )

    @property
    def frequencies(self) -> np.ndarray:
        return self._freqs.copy()

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the PSD for one fixed-shape input block.

        Parameters
        ----------
        x : numpy.ndarray
            Input block with shape ``(n_samples,)`` for a single channel or
            ``(n_samples, n_channels)`` for multiple channels.

        Returns
        -------
        psd : numpy.ndarray
            Power spectral density for the block.
        frequencies : numpy.ndarray
            Frequency coordinate for the PSD.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``x`` does not match the processor dimensions.
        """
        psd = self._process_matrix(x)
        if self._n_channels == 1:
            psd = psd[:, 0]
        return psd, self.frequencies

    def _process_matrix(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape != (self._n_samples, self._n_channels):
            raise ValidationError("x shape must match processor dimensions.")
        if self._complex_input:
            values = native_array(arr, np.complex128)
            psd = self._processor.process_complex(values)
        else:
            values = native_array(arr, float)
            psd = self._processor.process_real(values)
        return psd

    def reset_adaptive_state(self) -> None:
        """Clear adaptive weighting warm-start state."""
        self._processor.reset_adaptive_state()


def multitaper_psd(
    x: np.ndarray | SignalArray,
    fs: float | None = None,
    *,
    nw: float = 4.0,
    n_tapers: int | None = None,
    nfft: int | None = None,
    weighting: MultitaperWeighting = "adaptive",
    sides: MultitaperSides = "auto",
    window_length: int | None = None,
    incomplete: IncompleteWindow = "drop",
    detrend: PsdDetrend = "none",
    time_range: tuple[float, float] | None = None,
    db: bool = False,
    smoothing_width: float | None = None,
    smoothing_width_hz: float | None = None,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a power spectral density using Thomson multitapers.

    Real input defaults to a one-sided PSD and complex input to a two-sided
    PSD. Frequencies are in hertz when ``fs`` is supplied and in
    radians per sample otherwise.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional signal.
    fs : float or None, optional
        Sampling rate in hertz. For ``SignalArray``, the rate is inferred from
        metadata when omitted and checked against metadata when supplied.
    nw : float, default=4
        DPSS time-half-bandwidth product.
    n_tapers : int or None, optional
        Number of tapers. Defaults to ``floor(2*nw - 1)``.
    nfft : int or None, optional
        Number of spectrum points. Defaults to
        ``max(256, next_power_of_two(n_samples))``.
    weighting : {"unity", "eigen", "adaptive"}, default="adaptive"
        Rule used to combine individual tapered spectra.
    sides : {"auto", "one-sided", "two-sided"}, default="auto"
        Spectrum range. ``"auto"`` selects one-sided for real input.
    window_length : int or None, optional
        Samples per averaged PMTM segment. Omit to use the selected signal as
        one segment.
    incomplete : {"drop", "pad", "error"}, default="drop"
        Treatment of a final incomplete segment.
    detrend : {"none", "mean", "linear"}, default="none"
        Preprocessing applied independently to every segment.
    time_range : tuple of float or None, optional
        Half-open interval in seconds relative to the first input sample.
    db : bool, default=False
        Convert the averaged linear PSD to decibels.
    smoothing_width : float or None, optional
        Gaussian half-height width in frequency bins.
    smoothing_width_hz : float or None, optional
        Gaussian half-height width in hertz.
    axis : int, default=0
        Sample axis for ndarray input. ``SignalArray`` always uses axis zero.

    Returns
    -------
    psd : numpy.ndarray
        Power spectral density. The frequency axis replaces the sample axis.
    frequencies : numpy.ndarray
        Frequency coordinate for the PSD.

    Raises
    ------
    neurale.exceptions.ValidationError
        If input dimensions or estimator parameters are invalid.
    neurale.exceptions.NativeUnavailableError
        If the native spectral extension is unavailable.
    """
    normalized, context = normalize_signal_input(x, axis=axis)
    fs = resolve_psd_sampling_rate(x, fs)
    selected = select_time_range(normalized, fs, time_range)
    if not np.all(np.isfinite(selected)):
        raise ValidationError("x must contain only finite values.")

    weighting = _validate_weighting(weighting)
    sides = validate_choice(sides, ("auto", "one-sided", "two-sided"), "sides")
    incomplete = validate_choice(incomplete, ("drop", "pad", "error"), "incomplete")
    detrend = validate_detrend(detrend)
    db = validate_bool(db, "db")
    segment_length = (
        selected.shape[0]
        if window_length is None
        else validate_integer(window_length, "window_length", minimum=1)
    )
    segments = _multitaper_segments(selected, segment_length, incomplete)
    fft_length = (
        max(256, next_power_of_two(segment_length))
        if nfft is None
        else validate_integer(nfft, "nfft", minimum=1)
    )
    validated_nw = _validate_nw(segment_length, nw)
    if sides == "auto":
        output_sides = "two-sided" if np.iscomplexobj(selected) else "one-sided"
    else:
        output_sides = sides
    if output_sides == "one-sided" and np.iscomplexobj(selected):
        raise ValidationError("one-sided PSD requires real-valued input.")

    processor = MultitaperPsdProcessor(
        segment_length,
        selected.shape[1],
        fs,
        nw=validated_nw,
        n_tapers=n_tapers,
        nfft=fft_length,
        weighting=weighting,
        sides=output_sides,
        complex_input=bool(np.iscomplexobj(selected)),
    )
    scaled = _average_native_multitaper_segments(segments, detrend, processor)
    freqs = processor.frequencies
    smoothing_width = resolve_smoothing_width(smoothing_width, smoothing_width_hz, freqs)
    scaled = postprocess_psd(scaled, db, smoothing_width)
    return restore_psd(scaled, freqs, context)


def multitaper_spectrogram(
    x: np.ndarray | SignalArray,
    fs: float | None = None,
    *,
    window_size: float,
    shift: float,
    n_windows: int | None = None,
    start_time: float = 0.0,
    nw: float = 4.0,
    n_tapers: int | None = None,
    nfft: int | None = None,
    weighting: MultitaperWeighting = "adaptive",
    detrend: PsdDetrend = "none",
    db: bool = False,
    smoothing_width: float | None = None,
    smoothing_width_hz: float | None = None,
    time_reference: str = "center",
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a multitaper spectrogram.

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
    nw : float, default=4.0
        DPSS time-half-bandwidth product.
    n_tapers : int or None, optional
        Number of tapers. Defaults to ``floor(2*nw - 1)``.
    nfft : int or None, optional
        FFT length. Defaults to the window length.
    weighting : {"unity", "eigen", "adaptive"}, default="adaptive"
        Rule used to combine individual tapered spectra.
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
    neurale.exceptions.NativeUnavailableError
        If the native spectral extension is unavailable.
    neurale.exceptions.ValidationError
        If input data or spectrogram parameters are invalid.
    """
    normalized, context = normalize_signal_input(x, axis=axis)
    rate = resolve_psd_sampling_rate(x, fs)
    if rate is None:
        raise ValidationError("multitaper_spectrogram requires a sampling rate.")
    if not np.all(np.isfinite(normalized)):
        raise ValidationError("x must contain only finite values.")
    frames, times, window_length, _ = spectrogram_frames(
        normalized,
        rate,
        window_size=window_size,
        shift=shift,
        start_time=start_time,
        n_windows=n_windows,
        time_reference=time_reference,
    )
    weighting = _validate_weighting(weighting)
    detrend = validate_detrend(detrend)
    db = validate_bool(db, "db")
    fft_length = window_length if nfft is None else validate_integer(nfft, "nfft", minimum=1)
    if fft_length < window_length:
        raise ValidationError("nfft must not be shorter than the window.")
    output_sides: MultitaperSides = "two-sided" if np.iscomplexobj(normalized) else "one-sided"
    processor = MultitaperPsdProcessor(
        window_length,
        normalized.shape[1],
        rate,
        nw=_validate_nw(window_length, nw),
        n_tapers=n_tapers,
        nfft=fft_length,
        weighting=weighting,
        sides=output_sides,
        complex_input=bool(np.iscomplexobj(normalized)),
    )
    spectra = []
    freqs = processor.frequencies
    resolved_width = resolve_smoothing_width(smoothing_width, smoothing_width_hz, freqs)
    for frame in frames:
        prepared = preprocess_segment(frame, detrend)
        processor.reset_adaptive_state()
        scaled = processor._process_matrix(prepared)
        spectra.append(postprocess_psd(scaled, db, resolved_width))
    values = np.stack(spectra, axis=-1)
    return (
        restore_spectrogram(values, context),
        freqs,
        absolute_frame_times(context.source, times),
    )


def _validate_weighting(value: str) -> MultitaperWeighting:
    return validate_choice(value, ("unity", "eigen", "adaptive"), "weighting")


def _select_frequencies(
    freqs: np.ndarray,
    sides: MultitaperSides,
    nfft: int,
) -> np.ndarray:
    if sides == "one-sided":
        return freqs[: nfft // 2 + 1].copy()
    return freqs.copy()


def _native_spectral():
    return load_native_namespace("signal.spectral")


def _multitaper_plan_data(
    n_samples: int,
    n_channels: int,
    fs: float | None,
    nw: float,
    n_tapers: int | None,
    nfft: int | None,
    weighting: MultitaperWeighting,
    sides: MultitaperSides,
    complex_input: bool,
) -> MultitaperPlanData:
    n_samples = validate_integer(n_samples, "n_samples", minimum=1)
    n_channels = validate_integer(n_channels, "n_channels", minimum=1)
    weighting = _validate_weighting(weighting)
    sides = validate_choice(sides, ("auto", "one-sided", "two-sided"), "sides")
    if complex_input:
        output_sides = "two-sided"
    elif sides == "auto":
        output_sides = "one-sided"
    else:
        output_sides = sides
    if complex_input and output_sides == "one-sided":
        raise ValidationError("one-sided PSD requires real-valued input.")
    fft_length = (
        max(256, next_power_of_two(n_samples))
        if nfft is None
        else validate_integer(nfft, "nfft", minimum=1)
    )
    density_scale = 2.0 * np.pi if fs is None else validate_fs(fs)
    if density_scale is None:
        raise ValidationError("fs must be positive when provided.")
    return _cached_multitaper_plan_data(
        n_samples,
        n_channels,
        None if fs is None else float(fs),
        _validate_nw(n_samples, nw),
        n_tapers,
        fft_length,
        weighting,
        output_sides,
        complex_input,
        float(density_scale),
    )


@lru_cache(maxsize=64)
def _cached_multitaper_plan_data(
    n_samples: int,
    n_channels: int,
    fs: float | None,
    nw: float,
    n_tapers: int | None,
    fft_length: int,
    weighting: MultitaperWeighting,
    output_sides: Literal["one-sided", "two-sided"],
    complex_input: bool,
    density_scale: float,
) -> MultitaperPlanData:
    n_tapers, tapers, ratios = _build_native_multitaper_tapers(
        n_samples,
        nw=nw,
        n_tapers=n_tapers,
    )
    freqs = _select_frequencies(
        freq_vector(fft_length, fs),
        output_sides,
        fft_length,
    )
    return MultitaperPlanData(
        n_samples=n_samples,
        n_channels=n_channels,
        fft_length=fft_length,
        density_scale=density_scale,
        n_tapers=n_tapers,
        tapers=tapers,
        ratios=ratios,
        weighting=weighting,
        output_sides=output_sides,
        complex_input=complex_input,
        freqs=freqs,
    )


def _build_native_multitaper_tapers(
    length: int,
    *,
    nw: float,
    n_tapers: int | None,
) -> tuple[int, np.ndarray, np.ndarray]:
    validated_nw = _validate_nw(length, nw)
    maximum = min(length, int(np.floor(2.0 * validated_nw - 1.0)))
    if maximum < 2:
        raise ValidationError("nw must allow at least two PMTM tapers.")
    n_tapers = maximum if n_tapers is None else validate_integer(n_tapers, "n_tapers", minimum=2)
    if n_tapers > maximum:
        raise ValidationError(f"n_tapers must not exceed {maximum} for PMTM.")
    tapers, ratios = dpss(
        length,
        validated_nw,
        n_tapers,
    )
    return n_tapers, tapers, ratios


def _average_native_multitaper_segments(
    segments: list[np.ndarray],
    detrend: PsdDetrend,
    processor: MultitaperPsdProcessor,
) -> np.ndarray:
    iterator = iter(segments)
    first = next(iterator)
    processor.reset_adaptive_state()
    output = np.array(
        processor._process_matrix(preprocess_segment(first, detrend)),
        copy=True,
    )
    for segment in iterator:
        processor.reset_adaptive_state()
        output += processor._process_matrix(preprocess_segment(segment, detrend))
    return output / len(segments)


def _multitaper_segments(
    x: np.ndarray,
    window_length: int,
    incomplete: IncompleteWindow,
) -> list[np.ndarray]:
    if window_length > x.shape[0]:
        if incomplete != "pad":
            raise ValidationError("window_length exceeds the selected sample count.")
        padded = np.zeros((window_length, x.shape[1]), dtype=x.dtype)
        padded[: x.shape[0]] = x
        return [padded]

    complete = x.shape[0] // window_length
    remainder = x.shape[0] % window_length
    if incomplete == "error" and remainder:
        raise ValidationError("selected samples do not contain an integer number of windows.")
    segments = [x[idx * window_length : (idx + 1) * window_length] for idx in range(complete)]
    if incomplete == "pad" and remainder:
        padded = np.zeros((window_length, x.shape[1]), dtype=x.dtype)
        padded[:remainder] = x[complete * window_length :]
        segments.append(padded)
    if not segments:
        raise ValidationError("no complete multitaper windows are available.")
    return segments
