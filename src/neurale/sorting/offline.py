#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline typed threshold spike detection."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from neurale.data import EventSeries, SignalArray, SpikeWaveformBatch, convert_time
from neurale.exceptions import ValidationError

from ._dispatch import load_cpu_sorting_operation
from ._helpers import _resolved_groups, _sample_index_offset, _validate_regular_time
from .detection import DetectionConfig


def detect_threshold_spikes(
    signal: SignalArray,
    config: DetectionConfig,
    *,
    chunk_size: int | None = None,
    discontinuities: EventSeries | None = None,
    sample_idx_offset: int = 0,
) -> tuple[SpikeWaveformBatch, ...]:
    """Detect spikes and extract typed waveforms from one offline signal.

    The result always contains one :class:`SpikeWaveformBatch` per non-empty
    continuous segment, ordered by source position. With no discontinuities an
    empty input still returns one typed empty batch. A batch never crosses a
    discontinuity. Segment identifiers are zero-based positions in this
    returned tuple.

    ``chunk_size=None`` processes each continuous segment as one native chunk.
    A positive integer processes the same segment through consecutive logical
    chunks. Noise geometry is estimated over the complete continuous segment;
    crossing state, refractory state, peak alignment, and waveform samples
    continue across logical chunk boundaries. Therefore chunk boundaries are
    never treated as signal boundaries and do not affect ``boundary_behavior``.

    Instantaneous discontinuities are state barriers that do not discard a
    sample. Duration discontinuities remove source samples whose timestamps lie
    in their half-open ``[onset, stop)`` interval. The existing clock conversion
    contract maps discontinuities into ``signal.clock`` before segmentation.

    The production CPU path accepts exact C-contiguous sample-major ``float64``
    data. It never converts dtype or layout. Returned ``sample_indices`` are
    absolute source positions plus ``sample_index_offset``; ``times`` remain in
    the signal clock, and waveform layout is ``(spike, sample, channel)``.

    Parameters
    ----------
    signal : neurale.data.SignalArray
        Regularly sampled continuous data with shape ``(sample, channel)``.
    config : neurale.sorting.DetectionConfig
        Frozen detection and waveform contract.
    chunk_size : int or None, default None
        Positive number of source samples per logical chunk, or ``None`` for a
        single chunk per continuous segment.
    discontinuities : neurale.data.EventSeries or None, default None
        Ordered barriers/gaps. Its clock must be convertible to ``signal.clock``.
    sample_index_offset : int, default 0
        Non-negative absolute index of ``signal.data[0]``.

    Returns
    -------
    tuple of neurale.data.SpikeWaveformBatch
        One batch per retained continuous segment.

    Raises
    ------
    neurale.exceptions.ValidationError
        If input, timing, grouping, chunking, discontinuity, boundary, or int64
        index requirements are violated.
    """

    data, resolved_chunk_size, offset, groups = _validate_inputs(
        signal,
        config,
        chunk_size,
        discontinuities,
        sample_idx_offset,
    )
    native = load_cpu_sorting_operation("detection")
    bounds = _continuous_segment_bounds(signal, discontinuities)
    if not bounds:
        return ()

    refractory_value = config.refractory_interval * signal.fs
    if not math.isfinite(refractory_value):
        raise ValidationError("refractory_interval times fs must be finite.")
    refractory_samples = math.ceil(refractory_value)
    batches: list[SpikeWaveformBatch] = []
    for segment_id, (start, stop) in enumerate(bounds):
        segment = data[start:stop]
        chunks = _logical_chunks(segment, resolved_chunk_size)
        try:
            result = native._detect_threshold_waveforms(
                chunks,
                config.threshold_multiplier,
                refractory_samples,
                config.alignment_search_radius,
                config.pre_samples,
                config.post_samples,
                config.polarity,
                config.boundary_behavior,
                None if config.electrode_groups is None else [list(group) for group in groups],
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc

        positions = np.asarray(result["sample_indices"], dtype=np.int64) + np.int64(start)
        absolute_indices = positions + np.int64(offset)
        spike_polarities = (
            np.asarray(result["polarities"], dtype=np.int8) if config.polarity == "both" else None
        )
        batches.append(
            SpikeWaveformBatch(
                waveforms=np.asarray(result["waveforms"]),
                sample_indices=absolute_indices,
                times=np.asarray(signal.time[positions], dtype=np.float64),
                peak_channel_indices=np.asarray(result["peak_channel_indices"], dtype=np.int64),
                electrode_group_ids=np.asarray(result["electrode_group_ids"], dtype=np.int64),
                amps=np.asarray(result["amps"]),
                channels=signal.channels,
                fs=signal.fs,
                clock=signal.clock,
                source_stream=signal.name,
                segment_id=segment_id,
                pre_samples=config.pre_samples,
                post_samples=config.post_samples,
                polarity=config.polarity,
                spike_polarities=spike_polarities,
                attrs={
                    "detection_method": "threshold_native_cpu",
                    "electrode_groups": groups,
                    "noise_estimator": config.noise_estimator,
                    "threshold_multiplier": config.threshold_multiplier,
                    "channel_centers": np.asarray(result["channel_centers"]),
                    "channel_noise": np.asarray(result["channel_noise"]),
                    "channel_thresholds": np.asarray(result["channel_thresholds"]),
                    "refractory_interval": config.refractory_interval,
                    "refractory_samples": refractory_samples,
                    "alignment_search_radius": config.alignment_search_radius,
                    "boundary_behavior": config.boundary_behavior,
                    "sample_index_offset": offset + start,
                    "segment_start_sample": start,
                    "segment_stop_sample": stop,
                },
            )
        )
    return tuple(batches)


def _validate_inputs(
    signal: SignalArray,
    config: DetectionConfig,
    chunk_size: int | None,
    discontinuities: EventSeries | None,
    sample_idx_offset: int,
) -> tuple[np.ndarray, int | None, int, tuple[tuple[int, ...], ...]]:
    if not isinstance(signal, SignalArray):
        raise ValidationError("signal must be a SignalArray.")
    if not isinstance(config, DetectionConfig):
        raise ValidationError("config must be a DetectionConfig.")
    if discontinuities is not None and not isinstance(discontinuities, EventSeries):
        raise ValidationError("discontinuities must be an EventSeries or None.")
    signal.validate()
    data = np.asarray(signal.data)
    if signal.n_channels == 0:
        raise ValidationError("signal must contain at least one channel.")
    if data.dtype != np.dtype(np.float64):
        raise ValidationError("signal.data dtype must be float64 for native threshold detection.")
    if not data.flags.c_contiguous:
        raise ValidationError(
            "signal.data must be C-contiguous sample-major data for native threshold detection."
        )
    if not np.all(np.isfinite(data)):
        raise ValidationError("signal.data must contain only finite values.")
    _validate_regular_time(signal)
    resolved_chunk_size = _chunk_size(chunk_size)
    offset = _sample_index_offset(sample_idx_offset, signal.n_samples)
    groups = _resolved_groups(config, signal.n_channels)
    return data, resolved_chunk_size, offset, groups


def _logical_chunks(data: np.ndarray, chunk_size: int | None) -> tuple[np.ndarray, ...]:
    if chunk_size is None or data.shape[0] == 0:
        return (data,)
    return tuple(data[start : start + chunk_size] for start in range(0, data.shape[0], chunk_size))


def _continuous_segment_bounds(
    signal: SignalArray,
    discontinuities: EventSeries | None,
) -> tuple[tuple[int, int], ...]:
    if discontinuities is None or len(discontinuities) == 0:
        return ((0, signal.n_samples),)
    times = np.asarray(signal.time, dtype=np.float64)
    cursor = 0
    bounds: list[tuple[int, int]] = []
    for event in discontinuities:
        onset = convert_time(event.onset, source=discontinuities.clock, target=signal.clock)
        stop = convert_time(event.stop, source=discontinuities.clock, target=signal.clock)
        left = int(np.searchsorted(times, onset, side="left"))
        if event.duration == 0.0:
            if left > cursor:
                bounds.append((cursor, left))
            cursor = max(cursor, left)
            continue
        right = int(np.searchsorted(times, stop, side="left"))
        if left > cursor:
            bounds.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < signal.n_samples:
        bounds.append((cursor, signal.n_samples))
    return tuple(bounds)


def _chunk_size(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValidationError("chunk_size must be a positive integer or None.")
    return int(value)


__all__ = ["detect_threshold_spikes"]
