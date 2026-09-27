#!/usr/bin/env python3

from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pytest
from _subprocess_probe import probe_json

from neurale.data import (
    Clock,
    Event,
    EventSeries,
    Recording,
    SignalArray,
    Trial,
    TrialTable,
    split_recording_trials,
)
from neurale.data._time_validation import _require_unified_sync_domain
from neurale.exceptions import ValidationError
from neurale.features import ERPEpochs, ERPWaveform, average_erp, erp_epochs


def _signal(
    data: np.ndarray,
    *,
    fs: float = 10.0,
    t0: float = 0.0,
    clock: Clock | None = None,
) -> SignalArray:
    return SignalArray.from_array(
        data,
        fs=fs,
        t0=t0,
        channel_names=["left", "right"],
        channel_types="ecog",
        units=["uV", "uV"],
        name="neural",
        clock=clock,
        attrs={"reference": "synthetic"},
    )


def test_event_erp_matches_direct_numpy() -> None:
    clock = Clock("acquisition", "device", synchronization_domain="session")
    data = np.column_stack(
        (
            np.arange(60, dtype=float),
            100.0 + 2.0 * np.arange(60, dtype=float),
        )
    )
    signal = _signal(data, clock=clock)
    events = EventSeries(
        [
            Event(2.0, label="target"),
            Event(3.0, label="other"),
            Event(4.0, label="target"),
        ],
        clock=clock,
        attrs={"source": "task"},
    )
    recording = Recording(
        signals={"neural": signal},
        events=events,
        metadata={"session": "known-response"},
    )

    result = erp_epochs(
        recording,
        "events",
        tmin=-0.2,
        tmax=0.4,
        label="target",
        baseline=(-0.2, 0.0),
    )

    raw = np.stack((data[18:24], data[38:44]))
    expected = raw - np.mean(raw[:, :2, :], axis=1, keepdims=True)
    assert isinstance(result, ERPEpochs)
    assert result.data.shape == (2, 6, 2)
    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_allclose(result.time, [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    assert result.fs == 10.0
    assert result.source_clock is clock
    assert result.channels == signal.channels
    assert result.unit == signal.unit
    assert result.signal_name == "neural"
    assert dict(result.signal_attrs[0]) == {"reference": "synthetic"}
    assert dict(result.recording_metadata[0]) == {"session": "known-response"}
    assert [event.onset for event in result.selections] == [2.0, 4.0]
    assert not result.data.flags.writeable
    assert not result.time.flags.writeable

    averaged = result.average()
    assert isinstance(averaged, ERPWaveform)
    assert averaged.data.shape == (6, 2)
    np.testing.assert_allclose(averaged.data, np.mean(expected, axis=0))
    assert averaged.n_epochs == 2
    assert averaged.channels == signal.channels
    assert averaged.source_clock is clock


def test_event_clock_aligns_to_signal_clock() -> None:
    signal_clock = Clock(
        "signal",
        "device",
        offset=10.0,
        synchronization_domain="session",
    )
    event_clock = Clock(
        "events",
        "task",
        offset=9.0,
        synchronization_domain="session",
    )
    data = np.column_stack((np.arange(10, dtype=float), np.arange(10, dtype=float) + 20.0))
    signal = _signal(data, clock=signal_clock)
    events = EventSeries([Event(1.0, label="go")], clock=event_clock)

    result = erp_epochs(signal, events, tmin=0.0, tmax=0.2)

    np.testing.assert_array_equal(result.data[0], data[:2])
    np.testing.assert_allclose(result.time, [0.0, 0.1])
    assert result.source_clock is signal_clock
    # The event clock that interprets the raw onsets is preserved losslessly,
    # together with each anchor's reference timestamp, so a reader can recover
    # the full event coordinate without the original EventSeries.
    assert result.anchor_clock is event_clock
    assert [event.onset for event in result.selections] == [1.0]
    np.testing.assert_allclose(result.anchor_reference_time, [10.0])


def test_event_bounds_use_reference_duration() -> None:
    # tmin/tmax are recording-reference durations for both anchor kinds. With an
    # event clock drift of 0.1, onset 1.0 maps to 1.1 reference seconds; a 0.5 s
    # window is [1.1, 1.6) reference -> 5 samples, not the 6 that event-clock-
    # local duration semantics would yield (0.5 * 1.1 = 0.55 reference seconds).
    data = np.arange(40, dtype=float).reshape(20, 2)
    signal = _signal(data)  # no clock -> signal-local == reference, 10 Hz
    event_clock = Clock("events", "task", drift=0.1, synchronization_domain="s")
    events = EventSeries([Event(1.0, label="go")], clock=event_clock)

    event_result = erp_epochs(signal, events, tmin=0.0, tmax=0.5)

    np.testing.assert_allclose(event_result.time, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert event_result.data.shape == (1, 5, 2)
    np.testing.assert_array_equal(event_result.data[0], data[11:16])

    # A trial anchored at the same reference time (1.1 s) selects the identical
    # window, confirming event- and trial-locked durations now match.
    recording = Recording(
        signals={"neural": signal},
        trials=TrialTable([Trial(0, 1.1, 2.0)]),
    )
    trial_result = erp_epochs(recording, "trials", tmin=0.0, tmax=0.5)

    np.testing.assert_allclose(trial_result.time, event_result.time)
    np.testing.assert_array_equal(trial_result.data, event_result.data)


def test_trial_table_and_typed_epochs_agree() -> None:
    data = np.column_stack((np.arange(50, dtype=float), -np.arange(50, dtype=float)))
    signal = _signal(data)
    trials = TrialTable(
        [
            Trial(0, 1.0, 2.0, label="keep", attrs={"condition": 1}),
            Trial(1, 3.0, 4.0, label="drop", attrs={"condition": 2}),
        ]
    )
    recording = Recording(
        signals={"neural": signal},
        trials=trials,
        metadata={"subject": "S1"},
    )

    from_recording = erp_epochs(
        recording,
        "trials",
        tmin=0.0,
        tmax=0.4,
        label="keep",
    )
    typed = split_recording_trials(recording, reference_clock=None)
    from_typed = erp_epochs(typed, tmin=0.0, tmax=0.4, label="keep")

    np.testing.assert_array_equal(from_recording.data, data[10:14][None, :, :])
    np.testing.assert_array_equal(from_typed.data, from_recording.data)
    np.testing.assert_allclose(from_typed.time, [0.0, 0.1, 0.2, 0.3])
    assert from_typed.selections == (trials[0],)
    assert dict(from_typed.recording_metadata[0]) == {"subject": "S1"}


def test_single_typed_trial_epoch_is_accepted() -> None:
    signal = _signal(np.zeros((20, 2)))
    trial = Trial(0, 0.0, 1.0, label="one")
    recording = Recording(signals={"neural": signal}, trials=TrialTable([trial]))
    typed = split_recording_trials(recording, reference_clock=None)[0]

    result = erp_epochs(typed, tmin=0.0, tmax=0.2)

    assert result.data.shape == (1, 2, 2)
    assert result.selections == (trial,)


def test_erp_delegates_slicing_to_public_functions(monkeypatch) -> None:
    module = importlib.import_module("neurale.features.erp")
    signal = _signal(np.zeros((30, 2)))
    events = EventSeries([Event(1.0), Event(2.0)])
    trials = TrialTable([Trial(0, 0.0, 1.0)])
    event_calls = 0
    trial_calls = 0
    original_signal = module._alignment.extract_signal_epoch
    original_trial = module._alignment.extract_trial_epoch

    def signal_spy(*args, **kwargs):
        nonlocal event_calls
        event_calls += 1
        return original_signal(*args, **kwargs)

    def trial_spy(*args, **kwargs):
        nonlocal trial_calls
        trial_calls += 1
        return original_trial(*args, **kwargs)

    monkeypatch.setattr(module._alignment, "extract_signal_epoch", signal_spy)
    monkeypatch.setattr(module._alignment, "extract_trial_epoch", trial_spy)

    erp_epochs(signal, events, tmin=0.0, tmax=0.2)
    erp_epochs(signal, trials, tmin=0.0, tmax=0.2)

    # Two event epochs call extract_signal_epoch directly; extract_trial_epoch
    # delegates once more to that same shared primitive.
    assert event_calls == 3
    assert trial_calls == 1


def test_out_of_coverage_epoch_is_rejected() -> None:
    signal = _signal(np.arange(20, dtype=float).reshape(10, 2))
    events = EventSeries([Event(0.1)])

    with pytest.raises(ValidationError, match="outside"):
        erp_epochs(signal, events, tmin=-0.2, tmax=0.2)


def test_epochs_with_different_relative_grids_are_rejected() -> None:
    signal = _signal(np.zeros((20, 2)))
    events = EventSeries([Event(0.05), Event(0.1)])

    with pytest.raises(ValidationError, match="relative sample grids"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.2)


def test_nan_values_propagate_through_baseline_and_average() -> None:
    data = np.zeros((30, 2), dtype=float)
    data[10, 0] = np.nan
    data[20, 1] = np.nan
    signal = _signal(data)
    events = EventSeries([Event(1.0), Event(2.0)])

    raw = erp_epochs(signal, events, tmin=0.0, tmax=0.2)
    averaged = average_erp(raw)
    corrected = erp_epochs(signal, events, tmin=0.0, tmax=0.2, baseline=(0.0, 0.1))

    assert np.isnan(raw.data[0, 0, 0])
    assert np.isnan(averaged.data[0, 0])
    assert np.isnan(averaged.data[0, 1])
    assert np.all(np.isnan(corrected.data[0, :, 0]))
    assert np.all(np.isnan(corrected.data[1, :, 1]))


@pytest.mark.parametrize(
    "baseline",
    [
        (0.1, 0.1),
        (0.2, 0.1),
        (-0.1, 0.1),
        (0.0, 0.3),
        (0.01, 0.09),
        (0.0,),
    ],
)
def test_invalid_baseline_intervals_are_rejected(baseline) -> None:
    signal = _signal(np.zeros((20, 2)))
    events = EventSeries([Event(1.0)])

    with pytest.raises(ValidationError, match=r"baseline|tmax"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.2, baseline=baseline)


def test_empty_missing_and_unmatched_selections_are_rejected() -> None:
    signal = _signal(np.zeros((20, 2)))
    recording = Recording(signals={"neural": signal})

    with pytest.raises(ValidationError, match="empty"):
        erp_epochs(signal, EventSeries(), tmin=0.0, tmax=0.2)
    with pytest.raises(ValidationError, match="no events"):
        erp_epochs(recording, "events", tmin=0.0, tmax=0.2)
    with pytest.raises(ValidationError, match="missing"):
        erp_epochs(
            signal,
            EventSeries([Event(1.0, label="present")]),
            tmin=0.0,
            tmax=0.2,
            label="absent",
        )


def test_invalid_epoch_and_source_contracts_are_rejected() -> None:
    signal = _signal(np.zeros((20, 2)))
    events = EventSeries([Event(1.0)])
    trial = Trial(0, 0.0, 0.5)

    with pytest.raises(ValidationError, match="greater"):
        erp_epochs(signal, events, tmin=0.2, tmax=0.2)
    with pytest.raises(ValidationError, match="requires"):
        erp_epochs(signal, None, tmin=0.0, tmax=0.2)
    with pytest.raises(ValidationError, match=r"trial\.stop"):
        erp_epochs(signal, TrialTable([trial]), tmin=0.0, tmax=0.6)
    with pytest.raises(ValidationError, match="source"):
        erp_epochs(42, tmin=0.0, tmax=0.2)


def test_clocked_trial_erp_requires_explicit_reference_clock() -> None:
    # The ERP trial path delegates to extract_trial_epoch, which must enforce
    # the same contract as split_recording_trials: a clocked trial source
    # rejects reference_clock=None instead of silently succeeding.
    clock = Clock("acquisition", "device", synchronization_domain="session-a")
    signal = _signal(np.zeros((20, 2)), clock=clock)
    recording = Recording(
        signals={"neural": signal},
        trials=TrialTable([Trial(0, 0.0, 0.4)]),
    )

    with pytest.raises(ValidationError, match="reference clock"):
        erp_epochs(recording, "trials", tmin=0.0, tmax=0.2, reference_clock=None)
    # split_recording_trials enforces the identical rule.
    with pytest.raises(ValidationError, match="reference clock"):
        split_recording_trials(recording, reference_clock=None)

    reference = Clock("reference", "reference", synchronization_domain="session-a")
    epochs = erp_epochs(recording, "trials", tmin=0.0, tmax=0.2, reference_clock=reference)
    assert epochs.data.shape == (1, 2, 2)


def test_event_anchor_clock_and_collection_attrs_are_preserved() -> None:
    # Distinct event-clock offset so the raw onset (1.0) is unambiguously
    # event-local and must not be confused with signal-clock or reference time.
    signal_clock = Clock("signal", "device", offset=10.0, synchronization_domain="s")
    event_clock = Clock("events", "task", offset=9.0, synchronization_domain="s")
    data = np.zeros((20, 2))
    signal = _signal(data, clock=signal_clock)
    events = EventSeries(
        [Event(1.0, label="go")],
        clock=event_clock,
        attrs={"source": "task-log"},
    )

    result = erp_epochs(signal, events, tmin=0.0, tmax=0.2)

    assert result.source_clock is signal_clock
    assert result.anchor_clock is event_clock
    assert result.anchor_clock is not result.source_clock
    # Raw onset stays in event-clock local seconds; anchor_reference_time is the
    # same onset mapped to recording-reference seconds (9.0 + 1.0 = 10.0).
    assert [event.onset for event in result.selections] == [1.0]
    np.testing.assert_allclose(result.anchor_reference_time, [10.0])
    assert dict(result.anchor_collection_attrs) == {"source": "task-log"}
    assert not isinstance(result.anchor_collection_attrs, dict)
    # The averaged waveform forwards the same anchor metadata.
    averaged = result.average()
    assert averaged.anchor_clock is event_clock
    np.testing.assert_allclose(averaged.anchor_reference_time, [10.0])
    assert dict(averaged.anchor_collection_attrs) == {"source": "task-log"}


def test_trial_anchor_has_no_clock_and_starts_at_trial() -> None:
    signal = _signal(np.zeros((20, 2)))
    trials = TrialTable([Trial(0, 0.5, 0.9), Trial(1, 1.0, 1.5)], attrs={"protocol": "A"})
    recording = Recording(signals={"neural": signal}, trials=trials)

    result = erp_epochs(recording, "trials", tmin=0.0, tmax=0.2)

    # Trial anchors are already in recording-reference coordinates, signalled
    # by anchor_clock=None; anchor_reference_time is each trial.start.
    assert result.anchor_clock is None
    assert result.source_clock is None
    np.testing.assert_allclose(result.anchor_reference_time, [0.5, 1.0])
    assert dict(result.anchor_collection_attrs) == {"protocol": "A"}
    # Lossless recovery: selections[i].start matches anchor_reference_time[i].
    assert [trial.start for trial in result.selections] == [0.5, 1.0]


def test_anchor_reference_time_is_immutable_and_matched() -> None:
    signal = _signal(np.zeros((20, 2)))
    events = EventSeries([Event(0.1), Event(0.3)])

    result = erp_epochs(signal, events, tmin=0.0, tmax=0.2)

    assert result.anchor_reference_time.shape == (2,)
    np.testing.assert_allclose(result.anchor_reference_time, [0.1, 0.3])
    assert not result.anchor_reference_time.flags.writeable


def test_public_average_requires_typed_epochs() -> None:
    with pytest.raises(ValidationError, match="ERPEpochs"):
        average_erp(np.zeros((1, 2, 3)))


def _valid_two_sample_epochs() -> ERPEpochs:
    # tmin=0, tmax=0.2 at 10 Hz -> half-open [0.0, 0.2) yields 2 samples with a
    # regular 0.1 s grid. A clean base instance whose public constructor already
    # passed every invariant; ``dataclasses.replace`` re-runs ``__post_init__`` so
    # single-field mutations exercise the dataclass-level validators directly.
    signal = _signal(np.arange(40, dtype=float).reshape(20, 2))
    events = EventSeries([Event(1.0, label="go")])
    return erp_epochs(signal, events, tmin=0.0, tmax=0.2)


def test_public_epochs_reject_time_grid_inconsistent_with_fs() -> None:
    base = _valid_two_sample_epochs()
    np.testing.assert_allclose(base.time, [0.0, 0.1])
    assert base.fs == 10.0

    # fs=10 implies a 0.1 s period; a 0.15 s spacing contradicts it.
    with pytest.raises(ValidationError, match="regular with period"):
        replace(base, time=np.array([0.0, 0.15]))


@pytest.mark.parametrize(
    ("time", "message"),
    [
        (np.array(["0.0", "0.1"]), "numeric"),
        (np.array([0.0, "0.1"], dtype=object), "numeric"),
        (np.array([0.0 + 0.0j, 0.1 + 0.2j]), "real-valued"),
    ],
)
def test_public_results_reject_non_real_time_arrays(
    time: np.ndarray,
    message: str,
) -> None:
    epochs = _valid_two_sample_epochs()

    with pytest.raises(ValidationError, match=message):
        replace(epochs, time=time)
    with pytest.raises(ValidationError, match=message):
        replace(epochs.average(), time=time)


def test_public_epochs_reject_baseline_outside_time_coverage() -> None:
    base = _valid_two_sample_epochs()
    # coverage is [0.0, 0.2); a baseline at [10.0, 11.0) is far outside it.
    with pytest.raises(ValidationError, match="outside"):
        replace(base, baseline=(10.0, 11.0))
    # A baseline that selects no samples is rejected even when in range.
    with pytest.raises(ValidationError, match="selects no samples"):
        replace(base, baseline=(0.05, 0.09))
    # A baseline that covers at least one sample is accepted.
    in_range = replace(base, baseline=(0.0, 0.05))
    assert in_range.baseline == (0.0, 0.05)


def test_public_waveform_rejects_empty_selections() -> None:
    waveform = _valid_two_sample_epochs().average()
    assert waveform.n_epochs == 1

    # Averaging zero epochs is nonsensical; the public constructor must reject an
    # empty selection even when the 2D data axis is non-empty.
    with pytest.raises(ValidationError, match="empty"):
        replace(
            waveform,
            selections=(),
            signal_attrs=(),
            recording_metadata=(),
            anchor_reference_time=np.array([]),
        )


def _gap_signal(*, clock: Clock | None = None) -> SignalArray:
    # 30 samples at 10 Hz -> coverage [0.0, 3.0) reference (an unclocked signal's
    # local axis == reference). Wide enough to host event/trial windows with gaps
    # both inside and outside, exercising the ERP wrapper's discontinuity gate.
    return _signal(np.arange(60, dtype=float).reshape(30, 2), clock=clock)


def test_event_epoch_rejects_instant_gap_in_window() -> None:
    signal = _gap_signal()
    events = EventSeries([Event(1.0, label="go")])
    # window [1.0, 1.5) reference -> samples 10..14; an instant gap at 1.2 spans it.
    gap = EventSeries([Event(1.2, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=gap)

    # A gap outside the window is accepted and the epoch is unaffected.
    clean = erp_epochs(
        signal, events, tmin=0.0, tmax=0.5, discontinuities=EventSeries([Event(1.6)])
    )
    np.testing.assert_array_equal(clean.data[0], signal.data[10:15])


def test_event_epoch_rejects_sustained_gap_overlapping_window() -> None:
    signal = _gap_signal()
    events = EventSeries([Event(1.0, label="go")])
    # window [1.0, 1.5); a sustained gap [1.1, 1.2) overlaps it.
    sustained = EventSeries([Event(1.1, duration=0.1, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=sustained)

    # A sustained gap ending exactly at the window start is half-open safe.
    ends_at_start = EventSeries([Event(0.9, duration=0.1, label="gap")])
    clean = erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=ends_at_start)
    np.testing.assert_array_equal(clean.data[0], signal.data[10:15])


def test_trial_epoch_rejects_gap_inside_trial_window() -> None:
    signal = _gap_signal()
    recording = Recording(signals={"neural": signal}, trials=TrialTable([Trial(0, 1.0, 2.0)]))
    # trial-relative window [1.0, 1.5); instant gap at 1.2 inside.
    gap = EventSeries([Event(1.2, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        erp_epochs(recording, "trials", tmin=0.0, tmax=0.5, discontinuities=gap)

    clean = erp_epochs(
        recording, "trials", tmin=0.0, tmax=0.5, discontinuities=EventSeries([Event(1.6)])
    )
    np.testing.assert_array_equal(clean.data[0], signal.data[10:15])


def test_typed_trial_epoch_rejects_gap_inside_trial_window() -> None:
    # The typed TrialEpoch source path (pre-split via split_recording_trials) is a
    # distinct entry point from Recording+"trials"; it must gate discontinuities
    # identically. This locks the typed TrialEpoch path against the strict path.
    signal = _gap_signal()
    recording = Recording(signals={"neural": signal}, trials=TrialTable([Trial(0, 1.0, 2.0)]))
    typed = split_recording_trials(recording, reference_clock=None)
    gap = EventSeries([Event(1.2, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        erp_epochs(typed, tmin=0.0, tmax=0.5, discontinuities=gap)

    clean = erp_epochs(typed, tmin=0.0, tmax=0.5, discontinuities=EventSeries([Event(1.6)]))
    np.testing.assert_array_equal(clean.data[0], signal.data[10:15])


def test_gap_exactly_at_epoch_stop_is_half_open_safe() -> None:
    signal = _gap_signal()
    gap_at_stop = EventSeries([Event(1.5, label="gap")])

    # Event path: window [1.0, 1.5); an instant gap exactly at the stop boundary
    # is excluded by the half-open interval and must not reject the epoch.
    events = EventSeries([Event(1.0, label="go")])
    event_result = erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=gap_at_stop)
    np.testing.assert_array_equal(event_result.data[0], signal.data[10:15])

    # Trial path: same boundary, same half-open semantics.
    recording = Recording(signals={"neural": signal}, trials=TrialTable([Trial(0, 1.0, 2.0)]))
    trial_result = erp_epochs(recording, "trials", tmin=0.0, tmax=0.5, discontinuities=gap_at_stop)
    np.testing.assert_array_equal(trial_result.data[0], signal.data[10:15])

    # The complementary instant gap exactly at the window start IS a barrier.
    gap_at_start = EventSeries([Event(1.0, label="gap")])
    with pytest.raises(ValidationError, match="discontinuity"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=gap_at_start)


def test_gap_clock_with_mismatched_sync_domain_is_rejected() -> None:
    # Trial path: a clocked signal + reference clock (domain "s") with a gap clock
    # in a different domain ("other") cannot be verified and is rejected.
    signal = _gap_signal(clock=Clock("acq", "device", synchronization_domain="s"))
    reference = Clock("reference", "reference", synchronization_domain="s")
    recording = Recording(signals={"neural": signal}, trials=TrialTable([Trial(0, 1.0, 2.0)]))
    gap = EventSeries(
        [Event(1.2, label="gap")], clock=Clock("gap", "device", synchronization_domain="other")
    )

    with pytest.raises(ValidationError, match="sync_domain"):
        erp_epochs(
            recording,
            "trials",
            tmin=0.0,
            tmax=0.5,
            reference_clock=reference,
            discontinuities=gap,
        )

    # A domain-matching gap clock outside the window is accepted.
    compatible_gap = EventSeries(
        [Event(1.6, label="gap")], clock=Clock("gap", "device", synchronization_domain="s")
    )
    clean = erp_epochs(
        recording,
        "trials",
        tmin=0.0,
        tmax=0.5,
        reference_clock=reference,
        discontinuities=compatible_gap,
    )
    np.testing.assert_array_equal(clean.data[0], signal.data[10:15])

    # Event path: the gap clock is checked against the clocked signal's domain
    # inside extract_signal_epoch, so a mismatched gap clock is rejected there too.
    event_clock = Clock("events", "task", synchronization_domain="s")
    events = EventSeries([Event(1.0, label="go")], clock=event_clock)
    with pytest.raises(ValidationError, match="sync_domain"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=gap)


def test_event_path_rejects_cross_domain_event_and_signal_clocks() -> None:
    # Event clock in domain A, signal clock in domain B: the event path must not
    # convert the event onset and the signal coverage into the same reference
    # coordinate without verifying the two clocks share a domain.
    signal = _gap_signal(clock=Clock("sig", "device", synchronization_domain="A"))
    events = EventSeries(
        [Event(1.0, label="go")], clock=Clock("events", "task", synchronization_domain="B")
    )

    with pytest.raises(ValidationError, match="sync_domain"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5)


def test_event_path_rejects_event_clock_vs_reference() -> None:
    # An explicit reference_clock in a different domain than the event clock must
    # be rejected; previously reference_clock was silently ignored on the event path.
    signal = _gap_signal()  # unclocked -> imposes no domain constraint of its own
    events = EventSeries(
        [Event(1.0, label="go")], clock=Clock("events", "task", synchronization_domain="A")
    )
    reference = Clock("reference", "reference", synchronization_domain="B")

    with pytest.raises(ValidationError, match="sync_domain"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5, reference_clock=reference)


def test_event_path_rejects_cross_domain_gap_clock() -> None:
    # Unclocked signal, event clock in domain A, discontinuity clock in domain B:
    # extract_signal_epoch would convert the discontinuity from domain B into an
    # unstructured reference coordinate without knowing the event window came from
    # domain A. The unified gate rejects the cross-domain pair before extraction.
    signal = _gap_signal()
    events = EventSeries(
        [Event(1.0, label="go")], clock=Clock("events", "task", synchronization_domain="A")
    )
    gap = EventSeries(
        [Event(1.2, label="gap")], clock=Clock("gap", "device", synchronization_domain="B")
    )

    with pytest.raises(ValidationError, match="sync_domain"):
        erp_epochs(signal, events, tmin=0.0, tmax=0.5, discontinuities=gap)


def test_event_path_accepts_one_shared_domain() -> None:
    # reference, signal, event, and discontinuity clocks all in domain "s"; the
    # gap sits outside the window so it does not reject on overlap.
    signal = _gap_signal(clock=Clock("sig", "device", synchronization_domain="s"))
    events = EventSeries(
        [Event(1.0, label="go")], clock=Clock("events", "task", synchronization_domain="s")
    )
    reference = Clock("reference", "reference", synchronization_domain="s")
    gap = EventSeries(
        [Event(1.6, label="gap")], clock=Clock("gap", "device", synchronization_domain="s")
    )

    result = erp_epochs(
        signal,
        events,
        tmin=0.0,
        tmax=0.5,
        reference_clock=reference,
        discontinuities=gap,
    )
    np.testing.assert_array_equal(result.data[0], signal.data[10:15])


def test_public_epochs_reject_inconsistent_anchor_time() -> None:
    base = _valid_two_sample_epochs()  # event at 1.0, no clock -> reference time 1.0
    # anchor_reference_time must equal to_reference_time(event.onset, anchor_clock);
    # 99.0 contradicts the 1.0 s onset the selection claims.
    with pytest.raises(ValidationError, match="to_reference_time"):
        replace(base, anchor_reference_time=np.array([99.0]))


def test_public_epochs_reject_complex_anchor_reference_time() -> None:
    base = _valid_two_sample_epochs()
    # A complex dtype must be rejected outright, not silently cast to float with
    # its imaginary part dropped (which would also emit a ComplexWarning).
    with pytest.raises(ValidationError, match="real-valued"):
        replace(base, anchor_reference_time=np.array([1 + 2j]))


def test_public_epochs_reject_large_event_timestamp_drift() -> None:
    # At 1e9 s, a 0.5 s drift between the stored anchor timestamp and the event
    # onset must be rejected. A relative tolerance would admit ~1 s here; the
    # absolute ULP-scale tolerance stays sub-microsecond.
    base = _valid_two_sample_epochs()  # anchor_clock=None, onset 1.0
    with pytest.raises(ValidationError, match="to_reference_time"):
        replace(
            base,
            selections=(Event(1e9),),
            anchor_reference_time=np.array([1e9 + 0.5]),
        )
    # The exact large timestamp is accepted, so the tolerance is not too tight.
    exact = replace(base, selections=(Event(1e9),), anchor_reference_time=np.array([1e9]))
    np.testing.assert_allclose(exact.anchor_reference_time, [1e9])


def test_public_epochs_reject_large_trial_timestamp_drift() -> None:
    signal = _gap_signal()
    recording = Recording(signals={"neural": signal}, trials=TrialTable([Trial(0, 1.0, 2.0)]))
    base = erp_epochs(recording, "trials", tmin=0.0, tmax=0.2)
    with pytest.raises(ValidationError, match=r"trial\.start"):
        replace(
            base,
            selections=(Trial(0, 1e9, 1e9 + 1.0),),
            anchor_reference_time=np.array([1e9 + 0.5]),
        )
    exact = replace(
        base,
        selections=(Trial(0, 1e9, 1e9 + 1.0),),
        anchor_reference_time=np.array([1e9]),
    )
    np.testing.assert_allclose(exact.anchor_reference_time, [1e9])


def test_public_epochs_reject_anchor_clock_on_trial_selections() -> None:
    signal = _gap_signal()
    recording = Recording(signals={"neural": signal}, trials=TrialTable([Trial(0, 1.0, 2.0)]))
    base = erp_epochs(recording, "trials", tmin=0.0, tmax=0.2)
    assert base.anchor_clock is None
    # Trial selections are already in recording-reference coordinates, so a
    # non-None anchor_clock is self-contradictory.
    with pytest.raises(ValidationError, match="anchor_clock must be None"):
        replace(base, anchor_clock=Clock("events", "task", synchronization_domain="s"))


def test_public_epochs_reject_mixed_event_and_trial_selections() -> None:
    signal = _gap_signal()
    events = EventSeries([Event(1.0), Event(2.0)])
    base = erp_epochs(signal, events, tmin=0.0, tmax=0.2)
    assert base.data.shape[0] == 2
    # Replace one event selection with a Trial of the same count; the data and
    # anchor_reference_time axes still match, but Event/Trial mixing is rejected.
    mixed = (base.selections[0], Trial(0, 0.0, 0.5))
    with pytest.raises(ValidationError, match="must not mix"):
        replace(base, selections=mixed)


def test_public_epochs_reject_cross_domain_source_and_anchor_clock() -> None:
    base = _valid_two_sample_epochs()  # both clocks None -> unconstrained
    # The signal time axis and the anchor onsets must share a synchronization
    # domain; a hand-assembled result with two different-session clocks would
    # otherwise look like a valid typed result with no recoverable reference.
    with pytest.raises(ValidationError, match="sync_domain"):
        replace(
            base,
            source_clock=Clock("signal", "device", synchronization_domain="session-A"),
            anchor_clock=Clock("events", "task", synchronization_domain="session-B"),
        )


def test_public_waveform_rejects_cross_domain_anchor() -> None:
    waveform = _valid_two_sample_epochs().average()  # both clocks None
    with pytest.raises(ValidationError, match="sync_domain"):
        replace(
            waveform,
            source_clock=Clock("signal", "device", synchronization_domain="session-A"),
            anchor_clock=Clock("events", "task", synchronization_domain="session-B"),
        )


def test_unified_sync_domain_rejects_cross_domain() -> None:
    same_a = Clock("a", "device", synchronization_domain="session-a")
    same_a2 = Clock("a2", "device", synchronization_domain="session-a")
    other = Clock("b", "device", synchronization_domain="session-b")

    # No clocks, a single clock, or several None inputs are trivially consistent.
    _require_unified_sync_domain()
    _require_unified_sync_domain(None)
    _require_unified_sync_domain(None, None, same_a, None)

    # Multiple clocks in the same domain are accepted.
    _require_unified_sync_domain(same_a, same_a2, None)
    _require_unified_sync_domain(same_a, same_a2, same_a)

    # Any cross-domain pair is rejected, regardless of position.
    with pytest.raises(ValidationError, match="sync_domain"):
        _require_unified_sync_domain(same_a, other)
    with pytest.raises(ValidationError, match="sync_domain"):
        _require_unified_sync_domain(other, None, same_a)
    with pytest.raises(ValidationError, match="sync_domain"):
        _require_unified_sync_domain(None, same_a, None, other, None)


def test_event_unified_clock_gate_runs_once_before_anchor_loop(monkeypatch) -> None:
    module = importlib.import_module("neurale.features.erp")
    signal = _signal(np.zeros((40, 2)))
    events = EventSeries([Event(1.0), Event(2.0), Event(3.0)])
    calls: list[int] = []
    original = module._time_validation._require_unified_sync_domain

    def spy(*clocks):
        calls.append(len(clocks))
        return original(*clocks)

    monkeypatch.setattr(module._time_validation, "_require_unified_sync_domain", spy)

    result = erp_epochs(signal, events, tmin=0.0, tmax=0.2)

    assert result.n_epochs == 3
    # The four-clock event gate is invariant across anchors and runs once. A
    # separate two-clock call validates the constructed public result.
    assert calls.count(4) == 1


def test_module_import_does_not_load_native_extensions() -> None:
    code = """
import importlib
import json
import sys

module = importlib.import_module("neurale.features.erp")
print(json.dumps({
    "has_api": hasattr(module, "erp_epochs"),
    "native": "neurale._native" in sys.modules,
    "cuda": "neurale._native_cuda" in sys.modules,
}))
"""
    assert probe_json(code) == {
        "has_api": True,
        "native": False,
        "cuda": False,
    }
