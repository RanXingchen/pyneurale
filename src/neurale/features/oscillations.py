#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline typed oscillation detection from windowed band-power ratios.

The detector operates on complete, non-overlapping windows. Target and
reference bands use half-open ``[low, high)`` frequency intervals, and an
active window satisfies ``target_mean_power / reference_mean_power >=
threshold``. Exact ``0 / 0`` scores zero; positive target power over an exact
zero reference power and finite positive ratios exceeding float64 range use
the largest finite float64 score. Detection
intervals use half-open ``[start, end)`` semantics in
the selected signal's absolute time coordinate. It is offline and stateless:
every call analyzes the supplied signal independently and retains no history
between calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from neurale._validation import validate_number, validate_real_array
from neurale.data import (
    ChannelTable,
    Clock,
    Event,
    EventSeries,
    Recording,
    SignalArray,
    convert_time,
)
from neurale.data._helpers import immutable_array_copy
from neurale.exceptions import ValidationError
from neurale.signal import stft_spectrogram
from neurale.signal._input import uniform_sampling_rate
from neurale.signal.spectral._spectrogram import spectrogram_frame_geometry

OscillationBands = Mapping[str, Sequence[float]]
OscillationMethod = Literal["windowed_power_ratio"]

_METHOD: OscillationMethod = "windowed_power_ratio"
_WINDOW: Literal["hann"] = "hann"
_DETREND: Literal["mean"] = "mean"
_TIME_REFERENCE: Literal["start"] = "start"
_PSD_SCALING: Literal["density"] = "density"
_MAX_FINITE_SCORE = np.finfo(float).max


@dataclass(frozen=True, slots=True)
class OscillationScore:
    """Typed window-level detection scores.

    ``data`` has shape ``(window, band, channel)`` and holds non-negative
    power ratios. ``time`` contains the absolute start time of each complete,
    non-overlapping window, so consecutive spacings equal ``window_size``.
    ``bands`` is an ordered axis of uniquely named target bands. ``valid`` is
    false for windows intersecting a discontinuity; invalid windows never
    contribute to detections or event merging. An empty score (zero windows)
    is exempt from the spacing requirement.
    """

    data: np.ndarray
    time: np.ndarray
    valid: np.ndarray
    bands: tuple[tuple[str, float, float], ...]
    reference_band: tuple[float, float]
    threshold: float
    window_size: float
    source_fs: float
    channels: ChannelTable
    clock: Clock | None
    source_signal: str
    method: OscillationMethod = _METHOD
    window: Literal["hann"] = _WINDOW
    detrend: Literal["mean"] = _DETREND
    shift: float | None = None
    time_reference: Literal["start"] = _TIME_REFERENCE
    psd_scaling: Literal["density"] = _PSD_SCALING

    def __post_init__(self) -> None:
        data = validate_real_array(self.data, "OscillationScore.data", ndim=3, finite=True)
        if np.any(data < 0.0):
            raise ValidationError("OscillationScore.data must be non-negative power ratios.")
        time = validate_real_array(self.time, "OscillationScore.time", ndim=1, finite=True)
        if not isinstance(self.valid, np.ndarray):
            raise ValidationError("OscillationScore.valid must be a numpy.ndarray.")
        valid = self.valid
        if valid.dtype != np.bool_ or valid.ndim != 1:
            raise ValidationError("OscillationScore.valid must be a 1D boolean array.")
        if data.shape[0] != time.size or valid.size != time.size:
            raise ValidationError("OscillationScore window axes must have equal lengths.")
        source_rate = _positive_float(self.source_fs, "source_fs")
        bands = _validate_stored_bands(self.bands, source_rate)
        if data.shape[1] != len(bands):
            raise ValidationError("OscillationScore band axis must match bands.")
        if not isinstance(self.channels, ChannelTable) or data.shape[2] != len(self.channels):
            raise ValidationError("OscillationScore channel axis must match channels.")
        if self.clock is not None and not isinstance(self.clock, Clock):
            raise ValidationError("OscillationScore.clock must be a Clock or None.")
        if not isinstance(self.source_signal, str) or not self.source_signal:
            raise ValidationError("OscillationScore.source_signal must be a non-empty string.")
        if self.method != _METHOD:
            raise ValidationError(f"method must be {_METHOD!r}.")
        threshold = _positive_float(self.threshold, "threshold")
        window_size = _positive_float(self.window_size, "window_size")
        reference_band = _validate_stored_reference_band(self.reference_band, source_rate)
        window_length, freq_grid = _score_window_geometry(window_size, source_rate)
        for _, low, high in bands:
            _require_band_bins(freq_grid, low, high)
        _require_band_bins(freq_grid, reference_band[0], reference_band[1])
        if self.window != _WINDOW:
            raise ValidationError(f"window must be {_WINDOW!r}.")
        if self.detrend != _DETREND:
            raise ValidationError(f"detrend must be {_DETREND!r}.")
        shift = window_size if self.shift is None else _positive_float(self.shift, "shift")
        if not np.isclose(shift, window_size, rtol=0.0, atol=_ulp_tol(window_size)):
            raise ValidationError("shift must equal window_size for non-overlapping windows.")
        if self.time_reference != _TIME_REFERENCE:
            raise ValidationError(f"time_reference must be {_TIME_REFERENCE!r}.")
        if self.psd_scaling != _PSD_SCALING:
            raise ValidationError(f"psd_scaling must be {_PSD_SCALING!r}.")
        if time.size > 1:
            if np.any(np.diff(time) <= 0.0):
                raise ValidationError("OscillationScore.time must be strictly increasing.")
            # Validate the grid in sample coordinates relative to time[0].
            # Absolute timestamps lose low-order precision at large epochs, so an
            # allclose in seconds relaxes with the absolute scale and can accept a
            # 0.75 s / 1.25 s spacing next to a 1 s window. In samples, each step
            # must advance exactly ``window_length`` samples; a sub-sample
            # tolerance absorbs timestamp rounding, but a whole-sample offset --
            # real overlap or an unanalyzed gap -- stays outside it regardless of
            # how large time[0] is.
            #
            # The tolerance must cover two independent error sources. First, the
            # absolute-timestamp representation error: ``(time - time[0]) *
            # source_rate`` inherits the ULP of the largest timestamp. Second,
            # the relative-offset roundoff: a grid written as
            # ``arange(n) * window_size`` or ``t0 + arange(n) * window_size``
            # accumulates O(eps * scale) error as the index grows, and that error
            # is unrelated to the absolute time scale -- it appears even at t0=0
            # for a non-dyadic ``window_size``. A tolerance covering only the
            # absolute ULP rejects mathematically equivalent grids that merely
            # differ in arithmetic order. ``0.25`` samples stays the hard ceiling:
            # a full-sample overlap or gap (e.g. the 0.75 s / 1.25 s spacings at
            # 1e15) is always outside it.
            actual_offsets = (time - time[0]) * source_rate
            expected_offsets = np.arange(time.size, dtype=float) * float(window_length)
            offset_scale = max(
                1.0,
                float(np.max(np.abs(actual_offsets))),
                float(np.max(np.abs(expected_offsets))),
            )
            roundoff_tol = np.finfo(float).eps * offset_scale * 16.0
            timestamp_tol = _absolute_time_ulp(float(np.max(np.abs(time)))) * source_rate
            sample_tol = min(
                0.25,
                max(1e-12, roundoff_tol, timestamp_tol),
            )
            if not np.allclose(actual_offsets, expected_offsets, rtol=0.0, atol=sample_tol):
                raise ValidationError(
                    "OscillationScore.time spacing must match window_size "
                    "(complete, non-overlapping windows)."
                )
        object.__setattr__(self, "data", immutable_array_copy(data))
        object.__setattr__(self, "time", immutable_array_copy(time))
        object.__setattr__(self, "valid", immutable_array_copy(valid))
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "reference_band", reference_band)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "source_fs", source_rate)
        object.__setattr__(self, "shift", shift)


@dataclass(frozen=True, slots=True)
class OscillationDetectionResult:
    """Detected intervals together with their typed window-level scores.

    ``events`` and ``score`` must describe one detection pass: their
    clocks, method, source signal, sampling rate, bands, reference band,
    threshold, and window size must agree, every event must lie within the
    score's time coverage, and no event may cover a window the score marks
    invalid. The event collection is reconstructed from the score and its
    recorded duration/merge configuration during validation, so event labels,
    channel and band metadata, peak values, and interval bounds must also
    match. Comparing only the clock would let an unrelated ``EventSeries`` be
    paired with an unrelated score.
    """

    events: EventSeries
    score: OscillationScore

    def __post_init__(self) -> None:
        if not isinstance(self.events, EventSeries):
            raise ValidationError("events must be an EventSeries.")
        if not isinstance(self.score, OscillationScore):
            raise ValidationError("score must be an OscillationScore.")
        if self.events.clock != self.score.clock:
            raise ValidationError("events and score must use the same clock.")
        _validate_result_consistency(self.events, self.score)


# Fields that ``EventSeries.attrs`` must carry with values matching the score,
# so an unrelated event collection cannot be paired with an unrelated score.
_RESULT_ATTR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("method", "method", "method"),
    ("source_signal", "source_signal", "source_signal"),
    ("source_fs", "source_fs", "source_fs"),
    ("bands", "bands", "bands"),
    ("reference_band", "reference_band", "reference_band"),
    ("threshold", "threshold", "threshold"),
    ("window_size", "window_size", "window_size"),
    ("source_channels", "source_channels", "channels"),
    ("window", "window", "window"),
    ("detrend", "detrend", "detrend"),
    ("shift", "shift", "shift"),
    ("time_reference", "time_reference", "time_reference"),
    ("psd_scaling", "psd_scaling", "psd_scaling"),
)


def _validate_result_consistency(events: EventSeries, score: OscillationScore) -> None:
    attrs = events.attrs or {}
    for label, events_key, score_attr in _RESULT_ATTR_PAIRS:
        events_value = attrs.get(events_key)
        score_value = getattr(score, score_attr)
        if events_value != score_value:
            raise ValidationError(
                f"OscillationDetectionResult {label} mismatch: "
                f"events has {events_value!r}, score has {score_value!r}."
            )

    min_duration = _positive_float(attrs.get("min_duration"), "events.attrs['min_duration']")
    merge_gap = _non_negative_float(attrs.get("merge_gap"), "events.attrs['merge_gap']")
    if attrs.get("interval_semantics") != "[start, end)":
        raise ValidationError(
            "OscillationDetectionResult interval_semantics mismatch: "
            "events must use '[start, end)'."
        )

    if score.time.size == 0:
        if len(events) > 0:
            raise ValidationError(
                "OscillationDetectionResult events present but score has no windows."
            )
        return

    coverage_start = float(score.time[0])
    coverage_stop = float(score.time[-1]) + score.window_size
    coverage_tol = np.finfo(float).eps * max(16.0, abs(coverage_stop))
    window_starts = np.asarray(score.time, dtype=float)
    window_stops = window_starts + score.window_size
    invalid = ~np.asarray(score.valid, dtype=bool)
    for event in events:
        if event.onset < coverage_start - coverage_tol or (
            event.stop > coverage_stop + coverage_tol
        ):
            raise ValidationError(
                "OscillationDetectionResult events must lie within score time coverage."
            )
        # Half-open overlap [window_start, window_stop) vs [event.onset, event.stop).
        overlaps = (window_starts < event.stop) & (window_stops > event.onset)
        if np.any(invalid & overlaps):
            raise ValidationError(
                "OscillationDetectionResult events must not cover score-invalid windows."
            )

    expected = _detected_events(
        source_signal=score.source_signal,
        source_start=float(score.time[0]),
        source_fs=score.source_fs,
        channels=score.channels,
        bands=score.bands,
        reference_band=score.reference_band,
        threshold=score.threshold,
        window_size=score.window_size,
        min_duration=min_duration,
        merge_gap=merge_gap,
        starts=score.time,
        scores=score.data,
        valid=score.valid,
    )
    if events.events != expected:
        raise ValidationError(
            "OscillationDetectionResult events must match the intervals and metadata "
            "reconstructed from score."
        )


@dataclass(frozen=True, slots=True)
class _Episode:
    start: float
    stop: float
    peak_score: float
    start_window: int
    stop_window: int


def detect_oscillations(
    source: SignalArray | Recording,
    bands: OscillationBands,
    reference_band: Sequence[float],
    threshold: float,
    window_size: float,
    min_duration: float,
    *,
    signal_name: str = "neural",
    merge_gap: float = 0.0,
    discontinuities: EventSeries | None = None,
    return_score: bool = False,
) -> EventSeries | OscillationDetectionResult:
    """Detect offline oscillation intervals using windowed power ratios.

    Every complete non-overlapping window is scored independently. Each window
    is mean-detrended before spectral estimation, so a constant (DC) signal has
    zero power in every band and produces score zero, preventing a DC offset
    from becoming a detection. A target band uses frequency bins
    ``low <= f < high`` and is active when its mean power divided by the
    reference-band mean power is greater than or equal to ``threshold``. A
    window whose reference-band power is negligible but whose target band
    carries genuine power scores high, which is the intended ratio-detector
    behavior. No positive reference-power floor is imposed. Per-window mean
    detrending removes DC offsets; it does not clamp the denominator.

    Exact zero power has explicit finite semantics: ``0 / 0`` scores zero,
    while positive target power over exact zero reference power scores the
    largest finite ``float64`` value. Positive numerator and denominator
    powers use ordinary division, with ratios exceeding the finite float64
    range saturated to that same largest finite value.

    Consecutive active windows form a candidate. Candidates shorter than
    ``min_duration`` are discarded before accepted candidates separated by at
    most ``merge_gap`` are merged. Merging never crosses a discontinuity.
    Only complete windows are analyzed; a trailing partial window is ignored,
    and a signal too short to contain one complete window raises
    ``ValidationError`` rather than returning a silent empty result.

    ``window_size`` is quantized to whole source samples with Python ``round``
    (round-half-to-even), and the quantized duration ``window_length /
    fs`` -- not the requested value -- is what every window covers
    and what is stored as the result ``window_size``. ``threshold`` compares
    against per-window ratios and ``min_duration`` against candidate durations
    measured in this quantized window, so both operate on the actual geometry.
    A trailing partial window is judged against the quantized length: only
    ``n_samples // window_length`` complete windows contribute.

    Parameters
    ----------
    source : SignalArray or Recording
        Uniformly sampled source signal, or a recording containing it.
    bands : mapping of str to two floats
        Ordered target bands expressed as half-open ``[low, high)`` intervals.
    reference_band : sequence of two floats
        Half-open denominator band ``[low, high)``.
    threshold : float
        Finite positive inclusive power-ratio threshold.
    window_size : float
        Positive window duration in seconds. Windows do not overlap. The
        duration is quantized to whole source samples with Python ``round``
        (round-half-to-even), and the fixed Hann analysis window needs at least
        3 samples, so a duration quantizing to fewer than 3 samples is rejected
        rather than producing a degenerate spectrum. The quantized duration is
        stored as the result ``window_size``.
    min_duration : float
        Positive minimum duration in seconds for each thresholded candidate,
        measured against the quantized window geometry.
    signal_name : str, default="neural"
        Signal selected when ``source`` is a Recording.
    merge_gap : float, default=0
        Maximum non-negative gap between already-qualified candidates.
    discontinuities : EventSeries or None, optional
        Segment barriers. Windows intersecting a barrier are invalid.
    return_score : bool, default=False
        Return :class:`OscillationDetectionResult` with typed scores when true;
        otherwise return only the :class:`EventSeries`.
    """
    signal = _select_signal(source, signal_name)
    if signal.clock is not None and not isinstance(signal.clock, Clock):
        raise ValidationError("signal.clock must be a Clock or None.")
    if discontinuities is not None and not isinstance(discontinuities, EventSeries):
        raise ValidationError("discontinuities must be an EventSeries or None.")
    if (
        discontinuities is not None
        and discontinuities.clock is not None
        and not isinstance(discontinuities.clock, Clock)
    ):
        raise ValidationError("discontinuities.clock must be a Clock or None.")
    signal.validate()
    if signal.n_channels == 0:
        raise ValidationError("oscillation detection requires at least one channel.")
    rate = uniform_sampling_rate(signal, required=True)
    if rate is None:  # pragma: no cover - guaranteed by required=True
        raise ValidationError("oscillation detection requires a sampling rate.")
    data = validate_real_array(signal.data, "signal.data", ndim=2, finite=True)
    if signal.time.size and float(signal.time[0]) < 0.0:
        raise ValidationError("oscillation events require non-negative absolute signal time.")

    validated_bands = _validate_bands(bands, rate)
    validated_reference = _validate_band(reference_band, "reference_band", rate)
    threshold = _positive_float(threshold, "threshold")
    requested_window = _positive_float(window_size, "window_size")
    min_duration = _positive_float(min_duration, "min_duration")
    merge_gap = _non_negative_float(merge_gap, "merge_gap")
    if not isinstance(return_score, bool):
        raise ValidationError("return_score must be a bool.")

    # Reuse the shared spectrogram frame-geometry quantization instead of
    # duplicating round(window_size * rate) here. spectrogram_frame_geometry
    # raises on a signal too short for one complete window, so a too-short
    # signal surfaces a ValidationError rather than a silent empty result.
    sample_starts, _, window_length, _ = spectrogram_frame_geometry(
        data.shape[0],
        rate,
        window_size=requested_window,
        shift=requested_window,
        start_time=0.0,
        n_windows=None,
        time_reference="start",
    )
    if window_length < 3:
        raise ValidationError(
            f"window_size of {requested_window} s quantizes to {window_length} "
            f"samples at {rate} Hz; the fixed Hann analysis window needs at least "
            f"3 samples to be non-degenerate."
        )
    actual_window = window_length / rate
    freq_grid = np.fft.rfftfreq(window_length, d=1.0 / rate)
    for _, low, high in validated_bands:
        _require_band_bins(freq_grid, low, high)
    _require_band_bins(freq_grid, validated_reference[0], validated_reference[1])
    n_windows = len(sample_starts)

    spectra, freqs, starts = stft_spectrogram(
        signal,
        window_size=actual_window,
        shift=actual_window,
        window=_WINDOW,
        detrend=_DETREND,
        time_reference=_TIME_REFERENCE,
    )
    spectra = np.asarray(spectra, dtype=float)
    if not np.all(np.isfinite(spectra)) or np.any(spectra < 0.0):
        raise ValidationError(
            "oscillation spectrogram must be finite and non-negative; "
            "a degenerate analysis window or invalid spectral configuration "
            "must not be masked as a zero-score result."
        )
    target_power = np.empty((n_windows, len(validated_bands), signal.n_channels), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for band_idx, (_, low, high) in enumerate(validated_bands):
            target_power[:, band_idx, :] = _mean_band_power(spectra, freqs, low, high).T
        reference_power = _mean_band_power(
            spectra, freqs, validated_reference[0], validated_reference[1]
        ).T
    if (
        not np.all(np.isfinite(target_power))
        or not np.all(np.isfinite(reference_power))
        or np.any(target_power < 0.0)
        or np.any(reference_power < 0.0)
    ):
        raise ValidationError("oscillation band powers must be finite and non-negative.")
    scores = _finite_power_ratios(target_power, reference_power)

    converted_gaps = _converted_discontinuities(discontinuities, signal.clock)
    valid = _valid_windows(starts, actual_window, converted_gaps)
    events = _detected_events(
        source_signal=signal.name,
        source_start=float(signal.time[0]),
        source_fs=signal.fs,
        channels=signal.channels,
        bands=validated_bands,
        reference_band=validated_reference,
        threshold=threshold,
        window_size=actual_window,
        min_duration=min_duration,
        merge_gap=merge_gap,
        starts=starts,
        scores=scores,
        valid=valid,
    )
    series = _event_series(
        signal,
        validated_bands,
        validated_reference,
        threshold,
        actual_window,
        min_duration,
        merge_gap,
        events,
    )
    if not return_score:
        return series
    score = OscillationScore(
        data=scores,
        time=starts,
        valid=valid,
        bands=validated_bands,
        reference_band=validated_reference,
        threshold=threshold,
        window_size=actual_window,
        source_fs=rate,
        channels=signal.channels,
        clock=signal.clock,
        source_signal=signal.name,
        window=_WINDOW,
        detrend=_DETREND,
        shift=actual_window,
        time_reference=_TIME_REFERENCE,
        psd_scaling=_PSD_SCALING,
    )
    return OscillationDetectionResult(series, score)


def _select_signal(source: SignalArray | Recording, signal_name: str) -> SignalArray:
    if not isinstance(signal_name, str) or not signal_name:
        raise ValidationError("signal_name must be a non-empty string.")
    if isinstance(source, SignalArray):
        return source
    if isinstance(source, Recording):
        try:
            return source.signal(signal_name)
        except KeyError as exc:
            raise ValidationError(f"recording does not contain signal {signal_name!r}.") from exc
    raise ValidationError("source must be a SignalArray or Recording.")


def _validate_bands(bands: OscillationBands, fs: float) -> tuple[tuple[str, float, float], ...]:
    if not isinstance(bands, Mapping) or not bands:
        raise ValidationError("bands must be a non-empty mapping.")
    result = []
    for name, bounds in bands.items():
        if not isinstance(name, str) or not name:
            raise ValidationError("band names must be non-empty strings.")
        low, high = _validate_band(bounds, f"band {name!r}", fs)
        result.append((name, low, high))
    return tuple(result)


def _validate_band(bounds: Sequence[float], name: str, fs: float) -> tuple[float, float]:
    try:
        values = np.asarray(bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain two numeric bounds.") from exc
    values = validate_real_array(values, name, ndim=1, finite=True)
    if values.size != 2:
        raise ValidationError(f"{name} must contain two bounds.")
    low, high = float(values[0]), float(values[1])
    if low < 0.0 or high <= low or high > fs / 2.0:
        raise ValidationError(f"{name} must satisfy 0 <= low < high <= Nyquist.")
    return low, high


def _validate_stored_bands(
    bands: tuple[tuple[str, float, float], ...],
    fs: float,
) -> tuple[tuple[str, float, float], ...]:
    if not isinstance(bands, tuple) or not bands:
        raise ValidationError("OscillationScore.bands must be a non-empty tuple.")
    result = []
    seen_names: set[str] = set()
    for item in bands:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValidationError("OscillationScore bands must contain (name, low, high).")
        name, low, high = item
        if not isinstance(name, str) or not name:
            raise ValidationError("OscillationScore band names must be non-empty strings.")
        if name in seen_names:
            raise ValidationError(
                f"OscillationScore band names must be unique; {name!r} is repeated."
            )
        seen_names.add(name)
        low = _non_negative_float(low, "band low")
        high = _positive_float(high, "band high")
        if high <= low or high > fs / 2.0:
            raise ValidationError("OscillationScore bands require low < high <= Nyquist.")
        result.append((name, low, high))
    return tuple(result)


def _validate_stored_reference_band(value: tuple[float, float], fs: float) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValidationError("reference_band must be a (low, high) tuple.")
    low = _non_negative_float(value[0], "reference band low")
    high = _positive_float(value[1], "reference band high")
    if high <= low or high > fs / 2.0:
        raise ValidationError("reference_band requires low < high <= Nyquist.")
    return low, high


def _positive_float(value: object, name: str) -> float:
    return float(
        validate_number(
            value,
            name,
            kind="real",
            minimum=0.0,
            minimum_inclusive=False,
            coerce=True,
        )
    )


def _non_negative_float(value: object, name: str) -> float:
    return float(validate_number(value, name, kind="real", minimum=0.0, coerce=True))


def _mean_band_power(spectra: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    selected = _require_band_bins(freqs, low, high)
    return np.mean(spectra[selected], axis=0)


def _finite_power_ratios(target_power: np.ndarray, reference_power: np.ndarray) -> np.ndarray:
    den = reference_power[:, np.newaxis, :]
    scores = np.zeros_like(target_power)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        np.divide(
            target_power,
            den,
            out=scores,
            where=den > 0.0,
        )
    zero_reference = den == 0.0
    scores[zero_reference & (target_power > 0.0)] = _MAX_FINITE_SCORE
    scores[np.isposinf(scores)] = _MAX_FINITE_SCORE
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValidationError("oscillation power ratios must be finite and non-negative.")
    return scores


def _require_band_bins(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    selected = (freqs >= low) & (freqs < high)
    if not np.any(selected):
        raise ValidationError(f"band [{low}, {high}) contains no frequency bins.")
    return selected


def _score_window_geometry(window_size: float, fs: float) -> tuple[int, np.ndarray]:
    window_length = round(window_size * fs)
    if window_length < 3:
        raise ValidationError(
            "OscillationScore.window_size must describe at least 3 source samples."
        )
    actual_window = window_length / fs
    if not np.isclose(
        window_size,
        actual_window,
        rtol=0.0,
        atol=_ulp_tol(actual_window),
    ):
        raise ValidationError(
            "OscillationScore.window_size must describe a whole number of source samples."
        )
    return window_length, np.fft.rfftfreq(window_length, d=1.0 / fs)


def _converted_discontinuities(
    discontinuities: EventSeries | None, target_clock: Clock | None
) -> tuple[tuple[float, float], ...]:
    if discontinuities is None:
        return ()
    converted = []
    for event in discontinuities:
        start = convert_time(event.onset, source=discontinuities.clock, target=target_clock)
        stop = (
            start
            if event.duration == 0.0
            else convert_time(event.stop, source=discontinuities.clock, target=target_clock)
        )
        converted.append((start, stop))
    return tuple(converted)


def _overlaps_gap(start: float, stop: float, gaps: Sequence[tuple[float, float]]) -> bool:
    for gap_start, gap_stop in gaps:
        if gap_stop == gap_start:
            if start <= gap_start < stop:
                return True
        elif gap_start < stop and gap_stop > start:
            return True
    return False


def _valid_windows(
    starts: np.ndarray, window_size: float, gaps: Sequence[tuple[float, float]]
) -> np.ndarray:
    return np.asarray(
        [not _overlaps_gap(float(start), float(start + window_size), gaps) for start in starts],
        dtype=bool,
    )


def _detected_events(
    source_signal: str,
    source_start: float,
    source_fs: float,
    channels: ChannelTable,
    bands: tuple[tuple[str, float, float], ...],
    reference_band: tuple[float, float],
    threshold: float,
    window_size: float,
    min_duration: float,
    merge_gap: float,
    starts: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
) -> tuple[Event, ...]:
    detected: list[Event] = []
    for band_idx, (band_name, low, high) in enumerate(bands):
        for channel_idx, channel in enumerate(channels):
            active = valid & (scores[:, band_idx, channel_idx] >= threshold)
            episodes = _qualifying_episodes(
                active,
                starts,
                scores[:, band_idx, channel_idx],
                window_size,
                min_duration,
            )
            episodes = _merge_episodes(episodes, window_size, merge_gap, valid)
            for episode in episodes:
                sample_idx = round((episode.start - source_start) * source_fs)
                detected.append(
                    Event(
                        onset=episode.start,
                        duration=episode.stop - episode.start,
                        label=f"oscillation:{band_name}:{channel.name}",
                        source=source_signal,
                        value=episode.peak_score,
                        sample_index=sample_idx,
                        attrs={
                            "channel": channel.name,
                            "channel_index": channel_idx,
                            "source_channel_index": channel.index,
                            "band": (low, high),
                            "band_name": band_name,
                            "reference_band": reference_band,
                            "threshold": threshold,
                            "peak_score": episode.peak_score,
                            "method": _METHOD,
                        },
                    )
                )
    detected.sort(key=lambda event: (event.onset, event.stop, event.label or ""))
    return tuple(detected)


def _qualifying_episodes(
    active: np.ndarray,
    starts: np.ndarray,
    scores: np.ndarray,
    window_size: float,
    min_duration: float,
) -> tuple[_Episode, ...]:
    episodes = []
    idx = 0
    while idx < active.size:
        if not active[idx]:
            idx += 1
            continue
        run_start = idx
        while idx + 1 < active.size and active[idx + 1]:
            idx += 1
        run_stop = idx + 1
        start = float(starts[run_start])
        stop = float(starts[idx] + window_size)
        candidate_duration = (run_stop - run_start) * window_size
        if candidate_duration + _duration_tol(candidate_duration, min_duration) >= min_duration:
            episodes.append(
                _Episode(
                    start,
                    stop,
                    float(np.max(scores[run_start:run_stop])),
                    run_start,
                    run_stop,
                )
            )
        idx += 1
    return tuple(episodes)


def _merge_episodes(
    episodes: Sequence[_Episode],
    window_size: float,
    merge_gap: float,
    valid: np.ndarray,
) -> tuple[_Episode, ...]:
    if not episodes:
        return ()
    merged = [episodes[0]]
    for episode in episodes[1:]:
        previous = merged[-1]
        gap_windows = episode.start_window - previous.stop_window
        separation = gap_windows * window_size
        crosses_barrier = bool(np.any(~valid[previous.stop_window : episode.start_window]))
        if separation <= merge_gap + _duration_tol(separation, merge_gap) and not crosses_barrier:
            merged[-1] = _Episode(
                previous.start,
                episode.stop,
                max(previous.peak_score, episode.peak_score),
                previous.start_window,
                episode.stop_window,
            )
        else:
            merged.append(episode)
    return tuple(merged)


def _duration_tol(*values: float) -> float:
    scale = max(1.0, *(abs(value) for value in values))
    return max(np.finfo(float).eps * scale * 16.0, 1e-12)


def _ulp_tol(value: float) -> float:
    return np.finfo(float).eps * max(16.0, abs(value))


def _absolute_time_ulp(value: float) -> float:
    # One float64 unit in the last place at the absolute time scale, in seconds.
    if value == 0.0:
        return np.finfo(float).eps
    _, exponent = np.frexp(abs(value))
    return float(np.ldexp(1.0, exponent - 53))


def _event_series(
    signal: SignalArray,
    bands: tuple[tuple[str, float, float], ...],
    reference_band: tuple[float, float],
    threshold: float,
    window_size: float,
    min_duration: float,
    merge_gap: float,
    events: Sequence[Event],
) -> EventSeries:
    return EventSeries(
        events,
        clock=signal.clock,
        attrs={
            "method": _METHOD,
            "source_signal": signal.name,
            "source_fs": signal.fs,
            "source_channels": signal.channels,
            "bands": bands,
            "reference_band": reference_band,
            "threshold": threshold,
            "window_size": window_size,
            "min_duration": min_duration,
            "merge_gap": merge_gap,
            "interval_semantics": "[start, end)",
            "window": _WINDOW,
            "detrend": _DETREND,
            "shift": window_size,
            "time_reference": _TIME_REFERENCE,
            "psd_scaling": _PSD_SCALING,
        },
    )


__all__ = [
    "OscillationDetectionResult",
    "OscillationScore",
    "detect_oscillations",
]
