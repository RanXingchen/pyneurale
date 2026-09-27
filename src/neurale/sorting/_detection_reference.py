#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Small-input Python oracle for threshold spike detection.

This module is deliberately private. It defines numerical behavior for future
optimized implementations; it is not an optimized detector or an online
processor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurale.data import SignalArray, SpikeWaveformBatch
from neurale.exceptions import ValidationError

from ._helpers import _resolved_groups, _sample_index_offset, _validate_regular_time
from .detection import DetectionConfig

_NORMAL_75TH_PERCENTILE = 0.6744897501960817


@dataclass(frozen=True, slots=True)
class _Candidate:
    sample_index: int
    crossing_index: int
    channel_index: int
    group_id: int
    score: float
    polarity: int


def threshold_detection_reference(
    signal: SignalArray,
    config: DetectionConfig,
    *,
    segment_id: int | str = 0,
    sample_idx_offset: int = 0,
) -> SpikeWaveformBatch:
    """Detect threshold crossings and return aligned typed waveforms.

    This oracle is intended for small synthetic inputs and parity tests. It
    performs no filtering, rereferencing, dtype conversion of waveform data,
    or discontinuity inference. The supplied ``SignalArray`` must already be
    the selected, regularly sampled, continuous segment to detect.

    A candidate begins at the first sample of an inclusive threshold
    excursion. Crossings are sorted within each electrode group; simultaneous
    crossings are resolved by the largest normalized excursion, then the
    lowest channel position. The earliest remaining crossing wins the
    refractory interval. Alignment then selects the earliest signed extremum
    in the clipped interval ``[crossing - radius, crossing + radius]``.

    Returned ``sample_indices`` equal local aligned positions plus
    ``sample_index_offset``. ``times`` are copied from ``signal.time`` and
    therefore remain in ``signal.clock``. No event may cross the single
    ``segment_id`` supplied for this call.
    """
    if not isinstance(signal, SignalArray):
        raise ValidationError("signal must be a SignalArray.")
    if not isinstance(config, DetectionConfig):
        raise ValidationError("config must be a DetectionConfig.")
    signal.validate()
    data = np.asarray(signal.data)
    if signal.n_channels == 0:
        raise ValidationError("signal must contain at least one channel.")
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
        raise ValidationError("signal.data must contain real numeric values.")
    if not np.all(np.isfinite(data)):
        raise ValidationError("signal.data must contain only finite values.")
    _validate_regular_time(signal)
    offset = _sample_index_offset(sample_idx_offset, signal.n_samples)
    groups = _resolved_groups(config, signal.n_channels)

    centered, centers, noise, thresholds = _noise_geometry(data, config)
    refractory_value = config.refractory_interval * signal.fs
    if not math.isfinite(refractory_value):
        raise ValidationError("refractory_interval times fs must be finite.")
    refractory_samples = math.ceil(refractory_value)
    candidates: list[_Candidate] = []
    for group_id, channels in enumerate(groups):
        group_crossings: list[_Candidate] = []
        for channel_idx in channels:
            group_crossings.extend(
                _channel_crossings(
                    centered[:, channel_idx],
                    threshold=float(thresholds[channel_idx]),
                    channel_idx=channel_idx,
                    group_id=group_id,
                    config=config,
                )
            )
        for crossing in _apply_group_refractory(group_crossings, refractory_samples):
            aligned = _align_candidate(
                crossing,
                centered[:, crossing.channel_index],
                threshold=float(thresholds[crossing.channel_index]),
                config=config,
                n_samples=signal.n_samples,
            )
            if aligned is not None:
                candidates.append(aligned)
    candidates.sort(key=lambda item: (item.sample_index, item.group_id, item.channel_index))

    positions = np.asarray([item.sample_index for item in candidates], dtype=np.int64)
    absolute_indices = positions + np.int64(offset)
    peak_channels = np.asarray([item.channel_index for item in candidates], dtype=np.int64)
    group_ids = np.asarray([item.group_id for item in candidates], dtype=np.int64)
    amps = np.asarray(
        [centered[item.sample_index, item.channel_index] for item in candidates],
        dtype=centered.dtype,
    )
    waveform_length = config.pre_samples + 1 + config.post_samples
    if candidates:
        waveforms = np.stack(
            [
                data[
                    item.sample_index - config.pre_samples : item.sample_index
                    + config.post_samples
                    + 1
                ]
                for item in candidates
            ],
            axis=0,
        )
    else:
        waveforms = np.empty((0, waveform_length, signal.n_channels), dtype=data.dtype)
    spike_polarities = (
        np.asarray([item.polarity for item in candidates], dtype=np.int8)
        if config.polarity == "both"
        else None
    )

    return SpikeWaveformBatch(
        waveforms=waveforms,
        sample_indices=absolute_indices,
        times=np.asarray(signal.time[positions], dtype=np.float64),
        peak_channel_indices=peak_channels,
        electrode_group_ids=group_ids,
        amps=amps,
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
            "detection_method": "threshold_reference",
            "electrode_groups": groups,
            "noise_estimator": config.noise_estimator,
            "threshold_multiplier": config.threshold_multiplier,
            "channel_centers": centers,
            "channel_noise": noise,
            "channel_thresholds": thresholds,
            "refractory_interval": config.refractory_interval,
            "refractory_samples": refractory_samples,
            "alignment_search_radius": config.alignment_search_radius,
            "boundary_behavior": config.boundary_behavior,
            "sample_index_offset": offset,
        },
    )


def _noise_geometry(
    data: np.ndarray,
    config: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data.shape[0] == 0:
        centers = np.zeros(data.shape[1], dtype=np.float64)
        noise = np.zeros(data.shape[1], dtype=np.float64)
        return data.astype(np.float64), centers, noise, noise.copy()
    numeric = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        # The centered-MAD geometry is location invariant only while the
        # integer waveform is exactly representable as float64. Past the 2**53
        # precision limit, small oscillations collapse onto the DC offset and
        # the same waveform flips from "detected" to an empty batch, violating
        # the reference oracle contract. Reject such inputs instead of
        # silently returning a wrong result; the same constraint keeps the
        # Python oracle aligned with a future native ``double`` kernel.
        if not np.array_equal(numeric.astype(data.dtype), data):
            raise ValidationError(
                "integer signal values must be exactly representable as float64 "
                "for threshold detection."
            )
    centers = np.median(numeric, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = numeric - centers
        noise = np.median(np.abs(centered), axis=0) / _NORMAL_75TH_PERCENTILE
        thresholds = config.threshold_multiplier * noise
    if not (
        np.all(np.isfinite(centered))
        and np.all(np.isfinite(noise))
        and np.all(np.isfinite(thresholds))
    ):
        raise ValidationError("signal magnitude and threshold geometry must remain finite.")
    return centered, centers, noise, thresholds


def _channel_crossings(
    values: np.ndarray,
    *,
    threshold: float,
    channel_idx: int,
    group_id: int,
    config: DetectionConfig,
) -> list[_Candidate]:
    if threshold <= 0.0 or values.size == 0:
        return []
    if config.polarity == "negative":
        active = values <= -threshold
    elif config.polarity == "positive":
        active = values >= threshold
    else:
        active = np.abs(values) >= threshold
    previous = np.empty_like(active)
    previous[0] = False
    previous[1:] = active[:-1]
    crossings = np.flatnonzero(active & ~previous)

    return [
        _Candidate(
            sample_index=int(crossing),
            crossing_index=int(crossing),
            channel_index=channel_idx,
            group_id=group_id,
            score=float(abs(values[crossing]) / threshold),
            polarity=-1 if values[crossing] < 0.0 else 1,
        )
        for crossing in crossings
    ]


def _align_candidate(
    crossing: _Candidate,
    values: np.ndarray,
    *,
    threshold: float,
    config: DetectionConfig,
    n_samples: int,
) -> _Candidate | None:
    start = max(0, crossing.crossing_index - config.alignment_search_radius)
    stop = min(n_samples, crossing.crossing_index + config.alignment_search_radius + 1)
    window = values[start:stop]
    if config.polarity == "negative":
        local_peak = int(np.argmin(window))
    elif config.polarity == "positive":
        local_peak = int(np.argmax(window))
    else:
        local_peak = int(np.argmax(np.abs(window)))
    peak = start + local_peak
    waveform_start = peak - config.pre_samples
    waveform_stop = peak + config.post_samples + 1
    if waveform_start < 0 or waveform_stop > n_samples:
        if config.boundary_behavior == "raise":
            raise ValidationError(
                "detected spike waveform crosses the input boundary at "
                f"channel position {crossing.channel_index}, sample {peak}."
            )
        return None
    return _Candidate(
        sample_index=peak,
        crossing_index=crossing.crossing_index,
        channel_index=crossing.channel_index,
        group_id=crossing.group_id,
        score=float(abs(values[peak]) / threshold),
        polarity=-1 if values[peak] < 0.0 else 1,
    )


def _apply_group_refractory(
    candidates: list[_Candidate],
    refractory_samples: int,
) -> list[_Candidate]:
    if not candidates:
        return []
    by_sample: dict[int, _Candidate] = {}
    for candidate in candidates:
        current = by_sample.get(candidate.sample_index)
        if current is None or (-candidate.score, candidate.channel_index) < (
            -current.score,
            current.channel_index,
        ):
            by_sample[candidate.sample_index] = candidate
    ordered = [by_sample[idx] for idx in sorted(by_sample)]
    accepted = [ordered[0]]
    for candidate in ordered[1:]:
        if candidate.sample_index - accepted[-1].sample_index >= refractory_samples:
            accepted.append(candidate)
    return accepted
