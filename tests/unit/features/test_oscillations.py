#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from _subprocess_probe import probe_json
from scipy import signal as scipy_signal

from neurale.data import Clock, Event, EventSeries, Recording, SignalArray
from neurale.exceptions import ValidationError
from neurale.features import (
    OscillationDetectionResult,
    OscillationScore,
    detect_oscillations,
)


def _windowed_signal(
    target_amps: np.ndarray,
    *,
    target_freq: float = 10.0,
    reference_freq: float = 30.0,
    fs: float = 100.0,
    t0: float = 10.0,
    clock: Clock | None = None,
    name: str = "lfp",
) -> SignalArray:
    amps = np.asarray(target_amps, dtype=float)
    if amps.ndim == 1:
        amps = amps[:, np.newaxis]
    samples_per_window = round(fs)
    local_time = np.arange(samples_per_window, dtype=float) / fs
    windows = []
    for amp in amps:
        target = np.sin(2.0 * np.pi * target_freq * local_time)[:, np.newaxis]
        reference = np.sin(2.0 * np.pi * reference_freq * local_time)[:, np.newaxis]
        windows.append(reference + target * amp[np.newaxis, :])
    data = np.concatenate(windows, axis=0) if windows else np.empty((0, amps.shape[1]))
    return SignalArray.from_array(
        data,
        fs=fs,
        t0=t0,
        channel_names=[f"c{idx}" for idx in range(amps.shape[1])],
        channel_types="aux",
        units="V",
        name=name,
        clock=clock,
        attrs={"session": "synthetic"},
    )


def _detect(source, **overrides):
    options = {
        "bands": {"alpha": (8.0, 12.0)},
        "reference_band": (28.0, 32.0),
        "threshold": 4.0,
        "window_size": 1.0,
        "min_duration": 1.0,
    }
    options.update(overrides)
    return detect_oscillations(source, **options)


def test_known_bursts_preserve_interval_and_metadata() -> None:
    clock = Clock("acquisition", "device", synchronization_domain="session-a")
    source = _windowed_signal(np.array([0.0, 3.0, 3.0, 0.0, 3.0, 3.0]), clock=clock)

    events = _detect(source)

    assert isinstance(events, EventSeries)
    assert events.clock is clock
    assert [(event.onset, event.stop) for event in events] == [(11.0, 13.0), (14.0, 16.0)]
    assert [event.duration for event in events] == [2.0, 2.0]
    assert all(event.attrs["channel"] == "c0" for event in events)
    assert all(event.attrs["band"] == (8.0, 12.0) for event in events)
    assert all(event.attrs["threshold"] == 4.0 for event in events)
    assert all(event.attrs["method"] == "windowed_power_ratio" for event in events)
    assert events.attrs["source_fs"] == 100.0
    assert events.attrs["source_channels"].names == ["c0"]
    assert events.attrs["interval_semantics"] == "[start, end)"


def test_scores_match_direct_scipy_hann_psd_ratio() -> None:
    source = _windowed_signal(np.array([[0.0, 2.0], [3.0, 1.0], [1.0, 4.0]]))

    result = _detect(source, threshold=100.0, return_score=True)

    assert isinstance(result, OscillationDetectionResult)
    samples_per_window = 100
    expected = []
    window = scipy_signal.get_window("hann", samples_per_window, fftbins=False)
    for i in range(3):
        frame = source.data[i * 100 : (i + 1) * 100]
        # detect_oscillations mean-detrends each window before the PSD, so the
        # reference computation must subtract the per-window mean too.
        detrended = frame - frame.mean(axis=0)
        freqs, density = scipy_signal.periodogram(
            detrended,
            fs=100.0,
            window=window,
            detrend=False,
            scaling="density",
            axis=0,
        )
        target = density[(freqs >= 8.0) & (freqs < 12.0)].mean(axis=0)
        reference = density[(freqs >= 28.0) & (freqs < 32.0)].mean(axis=0)
        expected.append(target / reference)
    np.testing.assert_allclose(result.score.data[:, 0, :], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.score.time, [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(result.score.valid, [True, True, True])
    assert result.score.data.shape == (3, 1, 2)
    assert result.score.channels.names == ["c0", "c1"]
    assert result.score.source_fs == source.fs


def test_channels_and_overlapping_bands_are_independent() -> None:
    rate = 100.0
    local = np.arange(100) / rate
    windows = []
    for i in range(4):
        reference = np.sin(2 * np.pi * 30 * local)
        c0 = reference + (6.0 if i in (0, 1) else 0.0) * np.sin(2 * np.pi * 10 * local)
        c1 = reference + (6.0 if i in (1, 2) else 0.0) * np.sin(2 * np.pi * 20 * local)
        windows.append(np.column_stack((c0, c1)))
    source = SignalArray.from_array(
        np.concatenate(windows),
        fs=rate,
        t0=5.0,
        channel_names=["left", "right"],
        channel_types="aux",
        units="uV",
        name="wideband",
    )

    events = _detect(
        source,
        bands={"alpha": (8.0, 12.0), "broad": (8.0, 22.0), "beta": (18.0, 22.0)},
    )

    summaries = {
        (event.attrs["band_name"], event.attrs["channel"]): (event.onset, event.stop)
        for event in events
    }
    assert summaries[("alpha", "left")] == (5.0, 7.0)
    assert summaries[("beta", "right")] == (6.0, 8.0)
    assert summaries[("broad", "left")] == (5.0, 7.0)
    assert summaries[("broad", "right")] == (6.0, 8.0)


def test_minimum_duration_applies_before_merging() -> None:
    source = _windowed_signal(np.array([3.0, 0.0, 3.0, 3.0]))

    events = _detect(source, min_duration=2.0, merge_gap=5.0)

    assert [(event.onset, event.stop) for event in events] == [(12.0, 14.0)]


@pytest.mark.parametrize("return_score", [False, True])
def test_large_absolute_time_does_not_relax_minimum_duration(return_score) -> None:
    source = _windowed_signal(
        np.array([3.0]),
        target_freq=1.0,
        reference_freq=3.0,
        fs=8.0,
        t0=1e15,
    )

    detected = _detect(
        source,
        bands={"target": (0.5, 1.5)},
        reference_band=(2.5, 3.5),
        threshold=4.0,
        window_size=1.0,
        min_duration=2.0,
        return_score=return_score,
    )

    events = detected.events if isinstance(detected, OscillationDetectionResult) else detected
    assert len(events) == 0


def test_nearby_events_merge_with_half_open_bounds() -> None:
    source = _windowed_signal(np.array([3.0, 3.0, 0.0, 3.0, 3.0]))

    separate = _detect(source, min_duration=2.0, merge_gap=0.5)
    merged = _detect(source, min_duration=2.0, merge_gap=1.0)

    assert [(event.onset, event.stop) for event in separate] == [(10.0, 12.0), (13.0, 15.0)]
    assert [(event.onset, event.stop) for event in merged] == [(10.0, 15.0)]
    assert merged[0].duration == 5.0


@pytest.mark.parametrize("return_score", [False, True])
def test_large_absolute_time_does_not_relax_merge_gap(return_score) -> None:
    source = _windowed_signal(
        np.array([3.0, 0.0, 3.0]),
        target_freq=1.0,
        reference_freq=3.0,
        fs=8.0,
        t0=1e15,
    )

    detected = _detect(
        source,
        bands={"target": (0.5, 1.5)},
        reference_band=(2.5, 3.5),
        threshold=4.0,
        window_size=1.0,
        min_duration=1.0,
        merge_gap=0.0,
        return_score=return_score,
    )

    events = detected.events if isinstance(detected, OscillationDetectionResult) else detected
    assert [(event.onset, event.stop) for event in events] == [
        (1e15, 1e15 + 1.0),
        (1e15 + 2.0, 1e15 + 3.0),
    ]


def test_discontinuity_invalidates_window_and_blocks_merge() -> None:
    clock = Clock("source", "device", synchronization_domain="s")
    source = _windowed_signal(np.full(6, 3.0), clock=clock)
    gap_clock = Clock("gap", "task", offset=1.0, synchronization_domain="s")
    gaps = EventSeries([Event(11.0, label="gap")], clock=gap_clock)

    result = _detect(source, merge_gap=10.0, discontinuities=gaps, return_score=True)

    # Gap-local 11 s maps to reference 12 s and then source-local 12 s.
    assert [(event.onset, event.stop) for event in result.events] == [(10.0, 12.0), (13.0, 16.0)]
    np.testing.assert_array_equal(result.score.valid, [True, True, False, True, True, True])


def test_discontinuity_at_event_stop_does_not_clip_event() -> None:
    source = _windowed_signal(np.array([3.0, 3.0, 0.0]))
    gaps = EventSeries([Event(12.0, label="gap")])

    events = _detect(source, discontinuities=gaps)

    assert [(event.onset, event.stop) for event in events] == [(10.0, 12.0)]


def test_short_input_raises_and_constant_detects_nothing() -> None:
    short = SignalArray.from_array(
        np.zeros((20, 1)),
        fs=100.0,
        t0=2.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="short",
    )
    # A signal shorter than one analysis window surfaces an explicit error
    # rather than a silent empty result.
    with pytest.raises(ValidationError, match="does not contain one complete window"):
        _detect(short, return_score=True)

    long_enough = _windowed_signal(np.array([3.0]))
    with pytest.raises(ValidationError, match="no frequency bins"):
        _detect(long_enough, bands={"too_narrow": (8.1, 8.2)})

    constant = _windowed_signal(np.zeros(3))
    assert len(_detect(constant)) == 0


def test_nonzero_constant_signal_detects_nothing() -> None:
    # A constant DC signal has no oscillation. Without per-window mean
    # detrending, Hann leakage left a tiny but nonzero reference power and the
    # ratio spiked; mean detrending drives every band's power to zero.
    constant = SignalArray.from_array(
        np.full((300, 1), 1.0),
        fs=100.0,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="dc",
    )

    result = _detect(constant, return_score=True)

    assert len(result.events) == 0
    np.testing.assert_array_equal(result.score.data, np.zeros((3, 1, 1)))


def test_large_dc_offset_matches_clean_oscillation() -> None:
    # Mean detrending makes the ratio invariant to a DC offset: the same
    # oscillation with and without a large offset must produce equal scores.
    clean = _windowed_signal(np.array([3.0, 0.0, 3.0]))
    offset_data = clean.data + 1000.0
    offset = SignalArray(
        data=offset_data,
        fs=clean.fs,
        time=clean.time,
        t0=clean.t0,
        clock=clean.clock,
        channels=clean.channels,
        unit=clean.unit,
        name=clean.name,
    )

    clean_score = _detect(clean, return_score=True).score.data
    offset_score = _detect(offset, return_score=True).score.data

    np.testing.assert_allclose(offset_score, clean_score, rtol=1e-12, atol=1e-12)
    assert np.all(offset_score >= 4.0) == np.all(clean_score >= 4.0)


def test_negligible_reference_power_yields_finite_high_score() -> None:
    # A pure target-band tone with no reference activity is a genuine
    # oscillation: the score is finite and large, not infinite or NaN.
    rate = 100.0
    local = np.arange(100) / rate
    tone = np.sin(2.0 * np.pi * 10.0 * local)[:, np.newaxis]
    source = SignalArray.from_array(
        np.tile(tone, (3, 1)),
        fs=rate,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="pure_alpha",
    )

    result = _detect(source, return_score=True)

    assert np.all(np.isfinite(result.score.data))
    assert np.all(result.score.data > 4.0)
    assert len(result.events) == 1


def test_zero_reference_power_uses_finite_saturation(monkeypatch) -> None:
    source = _windowed_signal(np.array([3.0]))

    def _zero_reference_spectrogram(*args, **kwargs):
        freqs = np.arange(51, dtype=float)
        spectra = np.zeros((51, 1, 1), dtype=float)
        spectra[(freqs >= 8.0) & (freqs < 12.0), 0, 0] = 1.0
        return spectra, freqs, np.array([10.0])

    monkeypatch.setattr(
        "neurale.features.oscillations.stft_spectrogram",
        _zero_reference_spectrogram,
    )

    result = _detect(source, return_score=True)

    assert result.score.data[0, 0, 0] == np.finfo(float).max
    assert np.isfinite(result.score.data[0, 0, 0])
    assert len(result.events) == 1


def test_ratio_overflow_saturates_in_both_return_modes(
    monkeypatch,
) -> None:
    source = _windowed_signal(np.array([3.0]))

    def _extreme_ratio_spectrogram(*args, **kwargs):
        freqs = np.arange(51, dtype=float)
        spectra = np.zeros((51, 1, 1), dtype=float)
        spectra[(freqs >= 8.0) & (freqs < 12.0), 0, 0] = 1e307
        spectra[(freqs >= 28.0) & (freqs < 32.0), 0, 0] = 1e-308
        return spectra, freqs, np.array([10.0])

    monkeypatch.setattr(
        "neurale.features.oscillations.stft_spectrogram",
        _extreme_ratio_spectrogram,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        events = _detect(source)
        result = _detect(source, return_score=True)

    assert events[0].value == np.finfo(float).max
    assert events[0].attrs["peak_score"] == np.finfo(float).max
    assert result.score.data[0, 0, 0] == np.finfo(float).max
    assert result.events == events


@pytest.mark.parametrize("return_score", [False, True])
def test_band_power_overflow_fails_before_event_creation(
    monkeypatch,
    return_score,
) -> None:
    source = _windowed_signal(np.array([3.0]))

    def _overflowing_mean_spectrogram(*args, **kwargs):
        freqs = np.arange(51, dtype=float)
        spectra = np.zeros((51, 1, 1), dtype=float)
        spectra[(freqs >= 8.0) & (freqs < 12.0), 0, 0] = 1e308
        spectra[(freqs >= 28.0) & (freqs < 32.0), 0, 0] = 1.0
        return spectra, freqs, np.array([10.0])

    monkeypatch.setattr(
        "neurale.features.oscillations.stft_spectrogram",
        _overflowing_mean_spectrogram,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError, match="band powers must be finite"):
            _detect(source, return_score=return_score)


def test_uniform_rescaling_preserves_ratio() -> None:
    source = _windowed_signal(np.array([3.0, 0.0, 3.0]))
    scaled = source.copy(data=source.data * 1e-150)

    original = _detect(source, return_score=True)
    reduced = _detect(scaled, return_score=True)

    np.testing.assert_allclose(reduced.score.data, original.score.data, rtol=1e-10, atol=0.0)
    assert [(event.onset, event.stop) for event in reduced.events] == [
        (event.onset, event.stop) for event in original.events
    ]


def test_pure_zero_signal_scores_zero_and_detects_nothing() -> None:
    zero = SignalArray.from_array(
        np.zeros((300, 1)),
        fs=100.0,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="zero",
    )

    result = _detect(zero, return_score=True)

    assert len(result.events) == 0
    np.testing.assert_array_equal(result.score.data, np.zeros((3, 1, 1)))


def test_two_sample_hann_window_is_rejected_as_degenerate() -> None:
    # window_size=0.015 s at 100 Hz quantizes to 2 samples; the symmetric Hann
    # window is then identically zero, the PSD divides by zero, and the result
    # must not be masked as an all-zero score. Bands are valid within Nyquist
    # so the degeneracy check is what fires.
    source = SignalArray.from_array(
        np.ones((300, 1)),
        fs=100.0,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="dc",
    )

    with pytest.raises(ValidationError, match="non-degenerate"):
        _detect(source, window_size=0.015)


def test_shortest_legal_hann_window_is_accepted() -> None:
    # 3 samples is the smallest non-degenerate symmetric Hann window
    # ([0, 1, 0]). A 3-sample real FFT has only two bins (0 Hz and fs/3), so
    # the bands must match those bins; a zero signal then yields a legitimate
    # all-zero score rather than a degenerate-spectrum error.
    source = SignalArray.from_array(
        np.zeros((3, 1)),
        fs=100.0,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="short",
    )

    result = _detect(
        source,
        window_size=0.03,
        bands={"lo": (30.0, 40.0)},
        reference_band=(0.0, 5.0),
        return_score=True,
    )

    assert result.score.data.shape == (1, 1, 1)
    assert np.all(np.isfinite(result.score.data))


def test_non_finite_spectrogram_is_rejected_not_masked(monkeypatch) -> None:
    # A degenerate or invalid spectral configuration must surface as an error
    # rather than being silently converted to a zero score.
    source = _windowed_signal(np.array([3.0]))

    def _nan_spectrogram(*args, **kwargs):
        spectra = np.full((1, 1, 1), np.nan, dtype=float)
        freqs = np.array([0.0, 10.0, 30.0])
        starts = np.array([10.0])
        return spectra, freqs, starts

    monkeypatch.setattr("neurale.features.oscillations.stft_spectrogram", _nan_spectrogram)

    with pytest.raises(ValidationError, match="finite and non-negative"):
        _detect(source, return_score=True)


def test_window_size_is_quantized_to_actual_sample_length() -> None:
    # A requested duration that does not divide the sampling rate must surface
    # the quantized length, not the requested one, in the typed score.
    rate = 100.0
    local = np.arange(100) / rate
    tone = np.sin(2.0 * np.pi * 10.0 * local)[:, np.newaxis]
    source = SignalArray.from_array(
        np.tile(tone, (3, 1)),
        fs=rate,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="alpha",
    )

    result = _detect(source, window_size=0.105, return_score=True)

    # 0.105 s * 100 Hz = 10.5 samples -> Python round() uses round-half-to-even
    # and quantizes to 10 -> 0.1 s, not the requested 0.105 s.
    assert result.score.window_size == 0.1
    np.testing.assert_allclose(np.diff(result.score.time), 0.1)


def test_window_size_rounds_half_to_even() -> None:
    # round(11.5) == 12 (round-half-to-even rounds to the even neighbor), so a
    # 0.115 s request at 100 Hz quantizes to 12 samples -> 0.12 s. Bands are
    # chosen to match the 12-sample frequency grid (bins at multiples of 100/12).
    rate = 100.0
    local = np.arange(120) / rate
    tone = np.sin(2.0 * np.pi * 10.0 * local)[:, np.newaxis]
    source = SignalArray.from_array(
        np.tile(tone, (3, 1)),
        fs=rate,
        t0=0.0,
        channel_names=["c"],
        channel_types="aux",
        units="V",
        name="alpha",
    )

    result = _detect(
        source,
        window_size=0.115,
        bands={"lo": (8.0, 12.0)},
        reference_band=(24.0, 26.0),
        return_score=True,
    )

    assert result.score.window_size == 0.12
    np.testing.assert_allclose(np.diff(result.score.time), 0.12)


def test_recording_selection_uses_last_complete_window() -> None:
    source = _windowed_signal(np.array([0.0, 0.0, 3.0]), t0=20.0, name="selected")
    recording = Recording(signals={"selected": source})

    events = _detect(recording, signal_name="selected")

    assert len(events) == 1
    assert events[0].onset == 22.0
    assert events[0].stop == 23.0
    assert events[0].sample_index == 200
    assert events[0].source == "selected"


def test_threshold_is_inclusive() -> None:
    source = _windowed_signal(np.array([2.0]))
    scored = _detect(source, threshold=1.0, return_score=True)
    exact = float(scored.score.data[0, 0, 0])

    assert len(_detect(source, threshold=exact)) == 1
    assert len(_detect(source, threshold=np.nextafter(exact, np.inf))) == 0


def test_score_arrays_are_immutable_and_typed() -> None:
    source = _windowed_signal(np.array([3.0]))
    result = _detect(source, return_score=True)

    assert isinstance(result.score, OscillationScore)
    with pytest.raises(ValueError, match="read-only"):
        result.score.data[0, 0, 0] = 0.0
    with pytest.raises(ValidationError, match="same clock"):
        OscillationDetectionResult(
            EventSeries(clock=Clock("other", "device", synchronization_domain="x")),
            result.score,
        )


def _valid_score() -> OscillationScore:
    source = _windowed_signal(np.array([3.0, 0.0, 3.0]))
    return _detect(source, return_score=True).score


def test_public_score_rejects_negative_data() -> None:
    score = _valid_score()

    with pytest.raises(ValidationError, match="non-negative"):
        dataclasses.replace(score, data=np.full(score.data.shape, -1.0))


def test_public_score_rejects_inconsistent_time_spacing() -> None:
    # time=[10, 10.5, 11] with window_size=1.0 describes overlapping windows,
    # contradicting the class's complete, non-overlapping claim.
    score = _valid_score()

    with pytest.raises(ValidationError, match="spacing must match window_size"):
        dataclasses.replace(score, time=np.array([10.0, 10.5, 11.0]))


def test_public_score_rejects_missing_window_spacing() -> None:
    score = _valid_score()

    with pytest.raises(ValidationError, match="spacing must match window_size"):
        dataclasses.replace(score, time=np.array([10.0, 12.0, 14.0]))


def _large_absolute_time_score() -> OscillationScore:
    source = _windowed_signal(
        np.array([3.0, 0.0, 3.0]),
        target_freq=1.0,
        reference_freq=3.0,
        fs=8.0,
        t0=1e15,
    )
    return _detect(
        source,
        bands={"target": (0.5, 1.5)},
        reference_band=(2.5, 3.5),
        threshold=4.0,
        window_size=1.0,
        min_duration=1.0,
        return_score=True,
    ).score


def test_public_score_rejects_overlap_at_large_absolute_time() -> None:
    # At 1e15 s the float64 ULP is 0.125 s, so an absolute-second allclose would
    # tolerate a 0.75 s spacing that overlaps 25% of a 1 s window. Sample
    # coordinate validation sees 6 samples vs the expected 8 and rejects it.
    score = _large_absolute_time_score()

    with pytest.raises(ValidationError, match="spacing must match window_size"):
        dataclasses.replace(score, time=np.array([1e15, 1e15 + 0.75, 1e15 + 1.75]))


def test_public_score_rejects_gap_at_large_absolute_time() -> None:
    # A 1.25 s spacing leaves a 0.25 s unanalyzed gap; sample coordinate
    # validation sees 10 samples vs the expected 8 and rejects it regardless of
    # the large epoch.
    score = _large_absolute_time_score()

    with pytest.raises(ValidationError, match="spacing must match window_size"):
        dataclasses.replace(score, time=np.array([1e15, 1e15 + 1.25, 1e15 + 2.25]))


def _non_dyadic_grid_score(n_windows: int) -> OscillationScore:
    # 0.1 s is non-dyadic in float64; round(0.1 * 100) == 10 samples per window.
    # ``_windowed_signal`` emits one second of samples per amplitude entry at
    # rate 100, so n_windows // 10 entries yield n_windows 0.1 s windows.
    n_entries = n_windows // 10
    source = _windowed_signal(np.full(n_entries, 3.0))
    return _detect(
        source,
        window_size=0.1,
        min_duration=0.1,
        return_score=True,
    ).score


@pytest.mark.parametrize("n_windows", [100, 1000])
def test_public_score_accepts_equivalent_sample_grid_arithmetic(n_windows) -> None:
    # The same complete, non-overlapping grid can be written several ways that
    # differ only in floating-point arithmetic order: ``(arange(n) * samples) /
    # rate`` and ``arange(n) * (samples / rate)`` accumulate different O(eps *
    # scale) roundoff, and a non-zero t0 shift adds another. A tolerance
    # covering only the absolute-timestamp ULP rejects the duration-first form
    # even though it describes the exact same windows; the tolerance must also
    # cover relative-offset roundoff that is independent of the absolute scale.
    score = _non_dyadic_grid_score(n_windows)
    n = n_windows

    sample_first = (np.arange(n, dtype=float) * 10) / 100.0
    duration_first = np.arange(n, dtype=float) * 0.1
    shifted = 1.0 + np.arange(n, dtype=float) * 0.1

    for _label, time in [
        ("sample-first", sample_first),
        ("duration-first", duration_first),
        ("shifted", shifted),
    ]:
        # Must construct successfully; no exception means the grid was accepted.
        dataclasses.replace(score, time=time)


def test_public_score_rejects_overlap_on_non_dyadic_grid() -> None:
    # A whole-sample offset (real overlap or gap) stays outside the tolerance
    # even when the window size is non-dyadic and admits arithmetic-order
    # roundoff: a 0.09 s step is 9 samples at rate 100, one sample short of the
    # 10-sample window, so every window after the first overlaps its
    # predecessor by a full sample -- far beyond the sub-sample ceiling.
    score = _non_dyadic_grid_score(100)

    with pytest.raises(ValidationError, match="spacing must match window_size"):
        dataclasses.replace(
            score,
            time=np.arange(100, dtype=float) * 0.09,
        )


@pytest.mark.parametrize("return_score", [False, True])
def test_large_absolute_time_is_accepted_in_all_modes(return_score) -> None:
    source = _windowed_signal(np.array([3.0]), t0=1_000_000.0)

    detected = _detect(source, window_size=0.1, return_score=return_score)

    events = detected.events if isinstance(detected, OscillationDetectionResult) else detected
    assert isinstance(events, EventSeries)
    if isinstance(detected, OscillationDetectionResult):
        np.testing.assert_allclose(
            detected.score.time,
            1_000_000.0 + np.arange(10) * 0.1,
            rtol=0.0,
            atol=np.spacing(1_000_000.0) * 2.0,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"window_size": 0.015}, "at least 3 source samples"),
        ({"window_size": 0.105}, "whole number of source samples"),
        ({"bands": (("alpha", 8.1, 8.2),)}, "no frequency bins"),
        ({"reference_band": (28.1, 28.2)}, "no frequency bins"),
    ],
)
def test_public_score_enforces_window_and_frequency_geometry(changes, message) -> None:
    score = _valid_score()

    with pytest.raises(ValidationError, match=message):
        dataclasses.replace(score, **changes)


def test_public_score_rejects_duplicate_band_names() -> None:
    score = _valid_score()

    with pytest.raises(ValidationError, match="unique"):
        dataclasses.replace(
            score,
            bands=(("alpha", 8.0, 12.0), ("alpha", 18.0, 22.0)),
        )


def test_public_score_accepts_empty_window_axis() -> None:
    # An empty score (zero windows) is exempt from the spacing requirement. The
    # detector no longer produces empty scores (a too-short signal raises), so
    # exercise the exemption by constructing an empty OscillationScore directly.
    score = _valid_score()
    empty = dataclasses.replace(
        score,
        data=np.empty((0, score.data.shape[1], score.data.shape[2])),
        time=np.empty(0, dtype=float),
        valid=np.empty(0, dtype=bool),
    )

    assert empty.data.shape == (0, score.data.shape[1], score.data.shape[2])
    assert empty.time.size == 0


def _matching_pair(
    discontinuities: EventSeries | None = None,
) -> tuple[EventSeries, OscillationScore]:
    source = _windowed_signal(np.full(6, 3.0))
    result = _detect(source, discontinuities=discontinuities, return_score=True)
    return result.events, result.score


def test_result_rejects_events_without_detection_attrs() -> None:
    events, score = _matching_pair()
    unrelated = EventSeries(
        [Event(onset=100.0, duration=1.0, label="unrelated")],
        clock=events.clock,
    )

    with pytest.raises(ValidationError, match="method mismatch"):
        OscillationDetectionResult(unrelated, score)


@pytest.mark.parametrize(
    ("attr", "value", "match"),
    [
        ("method", "other", "method mismatch"),
        ("source_signal", "other", "source_signal mismatch"),
        ("source_fs", 50.0, "source_fs mismatch"),
        ("threshold", 999.0, "threshold mismatch"),
        ("window_size", 2.0, "window_size mismatch"),
        ("bands", (("other", 8.0, 12.0),), "bands mismatch"),
        ("reference_band", (40.0, 44.0), "reference_band mismatch"),
        ("source_channels", "other", "source_channels mismatch"),
        ("window", "boxcar", "window mismatch"),
        ("detrend", "none", "detrend mismatch"),
        ("shift", 0.5, "shift mismatch"),
        ("time_reference", "center", "time_reference mismatch"),
        ("psd_scaling", "spectrum", "psd_scaling mismatch"),
    ],
)
def test_result_rejects_mismatched_detection_attr(attr, value, match) -> None:
    events, score = _matching_pair()
    bad = EventSeries(
        events.events,
        clock=events.clock,
        attrs={**events.attrs, attr: value},
    )

    with pytest.raises(ValidationError, match=match):
        OscillationDetectionResult(bad, score)


def test_result_rejects_events_outside_score_time_coverage() -> None:
    events, score = _matching_pair()
    # Score coverage is [10, 16] s; an event at 100 s has matching attrs but
    # lies outside the scored windows.
    far = EventSeries(
        [Event(onset=100.0, duration=1.0, label="far")],
        clock=events.clock,
        attrs=dict(events.attrs),
    )

    with pytest.raises(ValidationError, match="time coverage"):
        OscillationDetectionResult(far, score)


def test_result_rejects_events_over_invalid_window() -> None:
    clock = Clock("source", "device", synchronization_domain="s")
    source = _windowed_signal(np.full(6, 3.0), clock=clock)
    gap_clock = Clock("gap", "task", offset=1.0, synchronization_domain="s")
    gaps = EventSeries([Event(11.0, label="gap")], clock=gap_clock)
    result = _detect(source, discontinuities=gaps, return_score=True)
    events, score = result.events, result.score
    # valid is [T, T, F, T, T, T]; an event [11, 13) covers the invalid window
    # [12, 13) even though its attrs match the score.
    bad = EventSeries(
        [Event(onset=11.0, duration=2.0, label="covers-invalid")],
        clock=events.clock,
        attrs=dict(events.attrs),
    )

    with pytest.raises(ValidationError, match="score-invalid windows"):
        OscillationDetectionResult(bad, score)


def test_result_accepts_consistent_pair() -> None:
    events, score = _matching_pair()
    rebuilt = EventSeries(events.events, clock=events.clock, attrs=dict(events.attrs))

    OscillationDetectionResult(rebuilt, score)


def test_result_rejects_events_not_from_score() -> None:
    events, score = _matching_pair()
    zero_score = dataclasses.replace(score, data=np.zeros_like(score.data))

    with pytest.raises(ValidationError, match="reconstructed from score"):
        OscillationDetectionResult(events, zero_score)

    forged_event = dataclasses.replace(
        events[0],
        attrs={**events[0].attrs, "channel": "forged"},
    )
    forged = EventSeries(
        (forged_event, *events.events[1:]),
        clock=events.clock,
        attrs=dict(events.attrs),
    )
    with pytest.raises(ValidationError, match="reconstructed from score"):
        OscillationDetectionResult(forged, score)


def test_fixed_analysis_configuration_is_typed_and_recorded() -> None:
    result = _detect(_windowed_signal(np.array([3.0])), return_score=True)

    assert result.score.window == "hann"
    assert result.score.detrend == "mean"
    assert result.score.shift == result.score.window_size
    assert result.score.time_reference == "start"
    assert result.score.psd_scaling == "density"
    for name, value in {
        "window": "hann",
        "detrend": "mean",
        "shift": result.score.window_size,
        "time_reference": "start",
        "psd_scaling": "density",
    }.items():
        assert result.events.attrs[name] == value


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"window": "boxcar"}, "window must be"),
        ({"detrend": "none"}, "detrend must be"),
        ({"shift": 0.5}, "shift must equal"),
        ({"time_reference": "center"}, "time_reference must be"),
        ({"psd_scaling": "spectrum"}, "psd_scaling must be"),
    ],
)
def test_public_score_rejects_other_analysis_configuration(changes, message) -> None:
    with pytest.raises(ValidationError, match=message):
        dataclasses.replace(_valid_score(), **changes)


@pytest.mark.parametrize("return_score", [False, True])
def test_invalid_signal_clock_is_rejected_in_all_return_modes(return_score) -> None:
    source = _windowed_signal(np.array([3.0]), clock="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match=r"signal\.clock"):
        _detect(source, return_score=return_score)


@pytest.mark.parametrize("with_event", [False, True])
def test_invalid_discontinuity_clock_is_rejected(with_event) -> None:
    source = _windowed_signal(np.array([3.0]))
    contents = [Event(10.5, label="gap")] if with_event else []
    gaps = EventSeries(contents, clock="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match=r"discontinuities\.clock"):
        _detect(source, discontinuities=gaps)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bands": {}}, "non-empty mapping"),
        ({"bands": {"bad": (12.0, 8.0)}}, "low < high"),
        ({"bands": {"bad": (8.0, 51.0)}}, "Nyquist"),
        ({"bands": {"bad": (8.1, 8.2)}}, "no frequency bins"),
        ({"reference_band": (30.0, 30.0)}, "low < high"),
        ({"threshold": 0.0}, "greater than"),
        ({"threshold": np.nan}, "finite"),
        ({"window_size": 0.001}, "shorter than one sample"),
        ({"min_duration": 0.0}, "greater than"),
        ({"merge_gap": -0.1}, "at least"),
        ({"return_score": 1}, "bool"),
        ({"discontinuities": "gap"}, "EventSeries"),
    ],
)
def test_invalid_detection_contracts_raise_validation_error(overrides, message) -> None:
    source = _windowed_signal(np.array([3.0]))

    with pytest.raises(ValidationError, match=message):
        _detect(source, **overrides)


def test_nan_irregular_time_and_cross_domain_gap_rejected() -> None:
    source = _windowed_signal(np.array([3.0]))
    with_nan = source.copy(data=source.data.copy())
    with_nan.data[0, 0] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        _detect(with_nan)

    irregular_time = source.time.copy()
    irregular_time[20] += 0.001
    irregular = SignalArray(
        data=source.data,
        fs=source.fs,
        time=irregular_time,
        t0=source.t0,
        clock=source.clock,
        channels=source.channels,
        unit=source.unit,
        name=source.name,
    )
    with pytest.raises(ValidationError, match="uniformly sampled"):
        _detect(irregular)

    clocked = _windowed_signal(
        np.array([3.0]), clock=Clock("source", "device", synchronization_domain="a")
    )
    gaps = EventSeries([Event(10.5)], clock=Clock("gap", "device", synchronization_domain="b"))
    with pytest.raises(ValidationError, match="sync_domain"):
        _detect(clocked, discontinuities=gaps)


def test_invalid_source_and_negative_time_are_explicit() -> None:
    source = _windowed_signal(np.array([3.0]), t0=-1.0)
    with pytest.raises(ValidationError, match="non-negative absolute"):
        _detect(source)
    with pytest.raises(ValidationError, match="does not contain"):
        _detect(Recording(signals={"other": _windowed_signal(np.array([3.0]))}))
    with pytest.raises(ValidationError, match="SignalArray or Recording"):
        _detect(np.zeros(100))


def test_zero_channel_signal_is_rejected_early() -> None:
    # A zero-channel SignalArray is constructible but must surface a stable
    # ValidationError before reaching spectral estimation, not leak an
    # IndexError from an empty channel axis.
    zero_channel = SignalArray.from_array(
        np.zeros((100, 0)),
        fs=100.0,
        t0=0.0,
        channel_names=[],
        channel_types="aux",
        units="V",
        name="empty",
    )

    with pytest.raises(ValidationError, match="at least one channel"):
        _detect(zero_channel)


def test_module_import_does_not_load_native_extensions() -> None:
    code = """
import importlib
import json
import sys

module = importlib.import_module("neurale.features.oscillations")
print(json.dumps({
    "has_api": hasattr(module, "detect_oscillations"),
    "native": "neurale._native" in sys.modules,
    "cuda": "neurale._native_cuda" in sys.modules,
}))
"""
    result = probe_json(code)
    assert result == {"has_api": True, "native": False, "cuda": False}
