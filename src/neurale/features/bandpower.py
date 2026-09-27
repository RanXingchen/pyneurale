#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline band-power, Hilbert-envelope, and LMP features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_positive_float,
    validate_real_array,
)
from neurale.data import FeatureMatrix, SignalArray
from neurale.exceptions import ValidationError
from neurale.signal import (
    analytic_signal,
    butter,
    multitaper_spectrogram,
    sos_filtfilt,
)
from neurale.signal._input import normalize_signal_input
from neurale.signal.spectral._psd import resolve_psd_sampling_rate
from neurale.signal.spectral._spectrogram import (
    absolute_frame_times,
    spectrogram_frame_geometry,
)

Bands = Mapping[str, Sequence[float]]
MultitaperWeighting = Literal["unity", "eigen", "adaptive"]
PsdDetrend = Literal["none", "mean", "linear"]


def bandpower_features(
    signal: np.ndarray | SignalArray,
    bands: Bands,
    window_size: float,
    shift: float,
    *,
    fs: float | None = None,
    method: Literal["multitaper"] = "multitaper",
    nw: float = 3.5,
    n_tapers: int | None = None,
    nfft: int | None = None,
    weighting: MultitaperWeighting = "adaptive",
    detrend: PsdDetrend = "none",
    axis: int = 0,
) -> FeatureMatrix:
    """Extract multitaper band-power features from complete windows.

    ``bands`` is an ordered mapping from feature labels to ``(low, high)``
    frequency bounds in hertz. Band upper bounds are exclusive.
    """
    validate_choice(method, ("multitaper",), "method")
    data, source, rate, channel_names = _prepare_signal(signal, fs, axis)
    validated_bands = _validate_bands(bands, rate)
    spectra, freqs, relative_times = multitaper_spectrogram(
        data,
        rate,
        window_size=window_size,
        shift=shift,
        nw=nw,
        n_tapers=n_tapers,
        nfft=nfft,
        weighting=weighting,
        detrend=detrend,
    )

    freq_step = _frequency_step(freqs)
    n_channels = data.shape[1]
    values = np.empty(
        (spectra.shape[2], len(validated_bands) * n_channels),
        dtype=float,
    )
    for band_idx, (_, low, high) in enumerate(validated_bands):
        selected = (freqs >= low) & (freqs < high)
        if not np.any(selected):
            raise ValidationError(f"band [{low}, {high}) contains no frequency bins.")
        columns = slice(band_idx * n_channels, (band_idx + 1) * n_channels)
        values[:, columns] = np.sum(spectra[selected], axis=0).T * freq_step

    return _feature_matrix(
        values,
        validated_bands,
        channel_names,
        source,
        rate,
        window_size,
        shift,
        absolute_frame_times(source, relative_times),
    )


def hilbert_envelope_features(
    signal: np.ndarray | SignalArray,
    bands: Bands,
    window_size: float,
    shift: float,
    *,
    fs: float | None = None,
    order: int = 4,
    axis: int = 0,
) -> FeatureMatrix:
    """Extract windowed band-limited Hilbert-envelope features."""
    data, source, rate, channel_names = _prepare_signal(signal, fs, axis)
    validated_bands = _validate_bands(bands, rate)
    filter_order = validate_integer(order, "order", minimum=1)
    frame_spec = _frame_spec(data, source, rate, window_size, shift)
    n_channels = data.shape[1]
    values = np.empty(
        (frame_spec.times.size, len(validated_bands) * n_channels),
        dtype=float,
    )

    for band_idx, (_, low, high) in enumerate(validated_bands):
        coefs = butter(
            filter_order,
            (low, high),
            band="bandpass",
            fs=rate,
            output="sos",
        )
        filtered = sos_filtfilt(data, coefs)
        envelope = np.abs(analytic_signal(filtered))
        columns = slice(band_idx * n_channels, (band_idx + 1) * n_channels)
        values[:, columns] = _window_means(envelope, frame_spec)

    return _feature_matrix(
        values,
        validated_bands,
        channel_names,
        source,
        rate,
        window_size,
        shift,
        frame_spec.times,
    )


def lmp_features(
    signal: np.ndarray | SignalArray,
    cutoff: float = 200.0,
    window_size: float = 0.05,
    shift: float = 0.01,
    *,
    fs: float | None = None,
    order: int = 4,
    axis: int = 0,
) -> FeatureMatrix:
    """Extract low-pass local motor potential window means."""
    data, source, rate, channel_names = _prepare_signal(signal, fs, axis)
    cutoff = validate_positive_float(cutoff, "cutoff")
    filter_order = validate_integer(order, "order", minimum=1)
    frame_spec = _frame_spec(data, source, rate, window_size, shift)
    coefs = butter(
        filter_order,
        cutoff,
        band="lowpass",
        fs=rate,
        output="sos",
    )
    filtered = sos_filtfilt(data, coefs)
    values = _window_means(filtered, frame_spec)
    bands = (("lmp", 0.0, cutoff),)
    return _feature_matrix(
        values,
        bands,
        channel_names,
        source,
        rate,
        window_size,
        shift,
        frame_spec.times,
    )


@dataclass(frozen=True, slots=True)
class _FrameSpec:
    times: np.ndarray
    window_length: int
    shift_length: int


def _prepare_signal(
    signal: np.ndarray | SignalArray,
    fs: float | None,
    axis: int,
) -> tuple[np.ndarray, np.ndarray | SignalArray, float, list[str]]:
    data, context = normalize_signal_input(signal, axis=axis)
    data = validate_real_array(data, "signal", ndim=2)
    rate = resolve_psd_sampling_rate(signal, fs)
    if rate is None:
        raise ValidationError("ndarray input requires fs.")
    source = context.source
    channel_names = (
        source.channel_names
        if isinstance(source, SignalArray)
        else _default_channel_names(data.shape[1])
    )
    return data, source, rate, channel_names


def _validate_bands(bands: Bands, fs: float) -> tuple[tuple[str, float, float], ...]:
    if not isinstance(bands, Mapping) or not bands:
        raise ValidationError("bands must be a non-empty mapping.")
    nyquist = fs / 2.0
    validated = []
    for label, bounds in bands.items():
        if not isinstance(label, str) or not label.strip():
            raise ValidationError("band labels must be non-empty strings.")
        values = np.asarray(bounds)
        if values.ndim != 1 or values.size != 2:
            raise ValidationError(f"band {label!r} must contain two bounds.")
        low = float(values[0])
        high = float(values[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValidationError(f"band {label!r} bounds must be finite.")
        if low < 0.0 or high <= low or high > nyquist:
            raise ValidationError(f"band {label!r} must satisfy 0 <= low < high <= Nyquist.")
        validated.append((label, low, high))
    return tuple(validated)


def _frame_spec(
    data: np.ndarray,
    source: np.ndarray | SignalArray,
    fs: float,
    window_size: float,
    shift: float,
) -> _FrameSpec:
    _, relative_times, window_length, shift_length = spectrogram_frame_geometry(
        data.shape[0],
        fs,
        window_size=window_size,
        shift=shift,
        start_time=0.0,
        n_windows=None,
    )
    return _FrameSpec(
        absolute_frame_times(source, relative_times),
        window_length,
        shift_length,
    )


def _window_means(data: np.ndarray, spec: _FrameSpec) -> np.ndarray:
    starts = np.arange(spec.times.size) * spec.shift_length
    stops = starts + spec.window_length
    cumulative = np.empty((data.shape[0] + 1, data.shape[1]), dtype=float)
    cumulative[0] = 0.0
    np.cumsum(data, axis=0, out=cumulative[1:])
    return (cumulative[stops] - cumulative[starts]) / spec.window_length


def _feature_matrix(
    values: np.ndarray,
    bands: Sequence[tuple[str, float, float]],
    channel_names: Sequence[str],
    source: np.ndarray | SignalArray,
    fs: float,
    window_size: float,
    shift: float,
    times: np.ndarray,
) -> FeatureMatrix:
    window_length = round(float(window_size) * fs)
    shift_length = round(float(shift) * fs)
    return FeatureMatrix(
        data=values,
        fs=fs / shift_length,
        time=times,
        feature_names=[f"{label}:{channel}" for label, _, _ in bands for channel in channel_names],
        source_signal=source.name if isinstance(source, SignalArray) else None,
        window_size=window_length / fs,
        shift=shift_length / fs,
    )


def _frequency_step(freqs: np.ndarray) -> float:
    if freqs.size < 2:
        raise ValidationError("band power requires at least two frequency bins.")
    return float(freqs[1] - freqs[0])


def _default_channel_names(n_channels: int) -> list[str]:
    width = max(3, len(str(max(n_channels - 1, 0))))
    return [f"ch{idx:0{width}d}" for idx in range(n_channels)]
