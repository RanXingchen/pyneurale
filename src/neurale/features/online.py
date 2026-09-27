#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Causal online band-feature processors.

Frames are emitted only when a complete half-open window is available and are
timestamped at that window's geometric center. Consequently, a frame becomes
available ``window_size / 2`` after its feature timestamp. ``flush()`` drops an
incomplete trailing window, and ``reset()`` reanchors the next stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_positive_float,
)
from neurale.data import ChannelTable, FeatureMatrix, SignalArray
from neurale.exceptions import ValidationError
from neurale.signal import butter
from neurale.signal._arrays import native_array
from neurale.signal._numerics import next_power_of_two
from neurale.signal.spectral.multitaper import _multitaper_plan_data

from ._streaming_input import normalize_stream_block
from .bandpower import (
    Bands,
    MultitaperWeighting,
    PsdDetrend,
    _default_channel_names,
    _frequency_step,
    _validate_bands,
)

SpectralBackend = Literal["auto", "builtin"]


class _OnlineFeatureProcessor:
    def __init__(
        self,
        fs: float,
        channels: int | ChannelTable,
        window_size: float,
        shift: float,
    ) -> None:
        self.fs = validate_positive_float(fs, "fs")
        self.window_samples = _duration_samples(window_size, self.fs, "window_size")
        self.hop_samples = _duration_samples(shift, self.fs, "shift")
        if self.hop_samples > self.window_samples:
            raise ValidationError("shift must not exceed window_size.")
        self.window_size = self.window_samples / self.fs
        self.shift = self.hop_samples / self.fs
        (
            self.n_channels,
            self.channel_names,
            self._channel_signature,
        ) = _channel_layout(channels)
        self.feature_names: list[str] = []
        self._native = None
        self._stream_kind: type[np.ndarray] | type[SignalArray] | None = None
        self._stream_start: float | None = None
        self._next_input_time: float | None = None
        self._emitted_frames = 0
        self._source_signal: str | None = None
        self._failed = False

    def process(self, block: np.ndarray | SignalArray) -> FeatureMatrix:
        """Consume one sample-major block and emit all completed windows."""
        if self._failed:
            raise RuntimeError("processor failed; call reset() before continuing")
        candidate = normalize_stream_block(
            block,
            n_channels=self.n_channels,
            fs=self.fs,
            channel_signature=self._channel_signature,
            kind=self._stream_kind,
            stream_start=self._stream_start,
            next_input_time=self._next_input_time,
            source_signal=self._source_signal,
        )
        if self._native is None:
            raise RuntimeError("online feature processor is not initialized")
        try:
            values = self._native.process(native_array(candidate.data, float))
        except (ValueError, OverflowError, RuntimeError) as exc:
            self._failed = True
            raise ValidationError(str(exc)) from exc
        self._stream_kind = candidate.kind
        self._stream_start = candidate.stream_start
        self._next_input_time = candidate.next_input_time
        self._source_signal = candidate.source_signal
        result = self._restore_features(values)
        self._emitted_frames += values.shape[0]
        return result

    def flush(self) -> FeatureMatrix:
        """End the stream, discard causal state, and emit no frames."""
        if self._failed:
            raise RuntimeError("processor failed; call reset() before continuing")
        result = self._restore_features(np.empty((0, len(self.feature_names)), dtype=float))
        self.reset()
        return result

    def reset(self) -> None:
        """Clear causal state and re-anchor the next non-empty block."""
        if self._native is not None:
            self._native.reset()
        self._stream_kind = None
        self._stream_start = None
        self._next_input_time = None
        self._emitted_frames = 0
        self._source_signal = None
        self._failed = False

    def _restore_features(self, values: np.ndarray) -> FeatureMatrix:
        start = 0.0 if self._stream_start is None else self._stream_start
        frame_indices = self._emitted_frames + np.arange(values.shape[0], dtype=float)
        times = start + (self.window_samples / 2.0 + frame_indices * self.hop_samples) / self.fs
        return FeatureMatrix(
            data=values,
            fs=self.fs / self.hop_samples,
            time=times,
            feature_names=self.feature_names,
            source_signal=self._source_signal,
            window_size=self.window_size,
            shift=self.shift,
        )


class BandpowerProcessor(_OnlineFeatureProcessor):
    """Online multitaper band power over complete causal windows.

    Adaptive weights are reset for every window to match the offline estimator.
    ``nfft`` defaults to the exact window length, matching the offline
    spectrogram contract. Select the builtin backend with a power-of-two
    ``nfft`` when steady-state allocation must be excluded.
    """

    def __init__(
        self,
        fs: float,
        channels: int | ChannelTable,
        bands: Bands,
        window_size: float,
        shift: float,
        *,
        nw: float = 3.5,
        n_tapers: int | None = None,
        nfft: int | None = None,
        weighting: MultitaperWeighting = "adaptive",
        detrend: PsdDetrend = "none",
        backend: SpectralBackend = "auto",
    ) -> None:
        rate = validate_positive_float(fs, "fs")
        validated_bands = _validate_bands(bands, rate)
        super().__init__(rate, channels, window_size, shift)
        self.feature_names = _band_feature_names(validated_bands, self.channel_names)
        backend = validate_choice(backend, ("auto", "builtin"), "backend")
        detrend = validate_choice(detrend, ("none", "mean", "linear"), "detrend")
        fft_length = (
            self.window_samples
            if nfft is None
            else validate_integer(nfft, "nfft", minimum=self.window_samples)
        )
        plan = _multitaper_plan_data(
            self.window_samples,
            self.n_channels,
            rate,
            nw,
            n_tapers,
            fft_length,
            weighting,
            "one-sided",
            False,
        )
        bin_ranges = _band_bin_ranges(validated_bands, plan.freqs)
        try:
            self._native = load_native_namespace("features.online").BandpowerProcessor(
                self.window_samples,
                self.hop_samples,
                self.n_channels,
                plan.fft_length,
                plan.density_scale,
                native_array(plan.tapers, float),
                plan.n_tapers,
                native_array(plan.ratios, float),
                plan.weighting,
                bin_ranges,
                _frequency_step(plan.freqs),
                detrend,
                backend,
            )
        except (ValueError, OverflowError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc


class HilbertEnvelopeProcessor(_OnlineFeatureProcessor):
    """Online mean envelope from causal bandpass, then local FFT Hilbert.

    The bandpass state is continuous across chunks. Each complete feature
    window receives an independent analytic transform, so this causal API is
    intentionally distinct from offline zero-phase Hilbert features.
    """

    def __init__(
        self,
        fs: float,
        channels: int | ChannelTable,
        bands: Bands,
        window_size: float,
        shift: float,
        *,
        order: int = 4,
        nfft: int | None = None,
        backend: SpectralBackend = "builtin",
    ) -> None:
        rate = validate_positive_float(fs, "fs")
        validated_bands = _validate_bands(bands, rate)
        super().__init__(rate, channels, window_size, shift)
        self.feature_names = _band_feature_names(validated_bands, self.channel_names)
        filter_order = validate_integer(order, "order", minimum=1)
        backend = validate_choice(backend, ("auto", "builtin"), "backend")
        fft_length = (
            next_power_of_two(self.window_samples)
            if nfft is None
            else validate_integer(nfft, "nfft", minimum=self.window_samples)
        )
        filters = [
            butter(
                filter_order,
                (low, high),
                band="bandpass",
                fs=rate,
                output="sos",
            ).sos
            for _, low, high in validated_bands
        ]
        section_counts = {values.shape[0] for values in filters}
        if len(section_counts) != 1:
            raise ValidationError("Hilbert band filters must share one SOS shape.")
        try:
            self._native = load_native_namespace("features.online").HilbertEnvelopeProcessor(
                self.window_samples,
                self.hop_samples,
                self.n_channels,
                native_array(np.stack(filters), float),
                fft_length,
                backend,
            )
        except (ValueError, OverflowError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc


class LmpProcessor(_OnlineFeatureProcessor):
    """Online causal low-pass local motor potential with rolling means."""

    def __init__(
        self,
        fs: float,
        channels: int | ChannelTable,
        cutoff: float = 200.0,
        window_size: float = 0.05,
        shift: float = 0.01,
        *,
        order: int = 4,
    ) -> None:
        rate = validate_positive_float(fs, "fs")
        cutoff = validate_positive_float(cutoff, "cutoff")
        filter_order = validate_integer(order, "order", minimum=1)
        super().__init__(rate, channels, window_size, shift)
        self.feature_names = [f"lmp:{channel}" for channel in self.channel_names]
        coefs = butter(
            filter_order,
            cutoff,
            band="lowpass",
            fs=rate,
            output="sos",
        )
        try:
            self._native = load_native_namespace("features.online").LmpProcessor(
                self.window_samples,
                self.hop_samples,
                self.n_channels,
                native_array(coefs.sos, float),
            )
        except (ValueError, OverflowError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc


def _band_feature_names(
    bands: Sequence[tuple[str, float, float]], channel_names: Sequence[str]
) -> list[str]:
    return [f"{label}:{channel}" for label, _, _ in bands for channel in channel_names]


def _duration_samples(value: float, rate: float, name: str) -> int:
    duration = validate_positive_float(value, name)
    samples = round(duration * rate)
    if samples < 1:
        raise ValidationError(f"{name} is shorter than one sample.")
    return samples


def _channel_layout(
    channels: int | ChannelTable,
) -> tuple[int, list[str], tuple[tuple[str, int], ...] | None]:
    if isinstance(channels, ChannelTable):
        if len(channels) == 0:
            raise ValidationError("channels must not be empty.")
        return (
            len(channels),
            channels.names,
            tuple(zip(channels.names, channels.indices, strict=False)),
        )
    count = validate_integer(channels, "channels", minimum=1)
    return count, _default_channel_names(count), None


def _band_bin_ranges(bands: Sequence[tuple[str, float, float]], freqs: np.ndarray) -> np.ndarray:
    ranges = np.empty((len(bands), 2), dtype=np.uintp)
    for i, (label, low, high) in enumerate(bands):
        selected = np.flatnonzero((freqs >= low) & (freqs < high))
        if selected.size == 0:
            raise ValidationError(f"band {label!r} [{low}, {high}) contains no frequency bins.")
        ranges[i] = (selected[0], selected[-1] + 1)
    return ranges
