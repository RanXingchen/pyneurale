#!/usr/bin/env python3

import numpy as np
import pytest

from neurale.data import (
    AlignmentIndex,
    Clock,
    Event,
    EventSeries,
    Recording,
    SignalArray,
    Trial,
    TrialEpoch,
    TrialTable,
    align_neural_behavior_nearest,
    convert_time,
    events_in_interval,
    extract_signal_epoch,
    extract_trial_epoch,
    from_reference_time,
    split_recording_trials,
    to_reference_time,
)
from neurale.exceptions import ValidationError


def _clock(name, *, offset=0.0, drift=0.0, domain="session-a"):
    return Clock(
        name=name,
        type="device" if name != "reference" else "reference",
        offset=offset,
        drift=drift,
        synchronization_domain=domain,
    )


def _signal(*, fs=10.0, t0=0.0, clock=None, name="neural"):
    data = np.arange(20.0).reshape(10, 2)
    return SignalArray.from_array(
        data,
        fs=fs,
        t0=t0,
        channel_names=["C3", "C4"],
        channel_types="eeg",
        units="uV",
        name=name,
        clock=clock,
        attrs={"source": "fixture"},
    )


def test_epoch_interval_is_half_open_and_keeps_metadata():
    signal = _signal(t0=2.0)

    epoch = extract_signal_epoch(signal, 2.2, 2.6)

    np.testing.assert_array_equal(epoch.data, signal.data[2:6])
    np.testing.assert_allclose(epoch.time, [2.2, 2.3, 2.4, 2.5])
    assert epoch.t0 == pytest.approx(2.2)
    assert epoch.channel_names == ["C3", "C4"]
    assert epoch.unit == "uV"
    assert epoch.name == "neural"
    assert epoch.attrs == signal.attrs


def test_epoch_out_of_range_is_strict():
    signal = _signal(t0=2.0)

    with pytest.raises(ValidationError, match="outside"):
        extract_signal_epoch(signal, 1.9, 2.2)
    with pytest.raises(ValidationError, match="outside"):
        extract_signal_epoch(signal, 2.8, 3.1)


def test_clock_conversion_requires_matching_domain_and_drift():
    source = _clock("behavior", offset=0.25, drift=0.001)
    target = _clock("acq", offset=0.05)

    assert convert_time(10.0, source=source, target=target) == pytest.approx(10.21)
    with pytest.raises(ValidationError, match="sync_domain"):
        convert_time(1.0, source=source, target=_clock("other", domain="session-b"))
    with pytest.raises(ValidationError, match="sync_domain"):
        convert_time(1.0, source=source, target=Clock(name="unsynced", type="device"))
    with pytest.raises(ValidationError, match="sync_domain"):
        convert_time(
            1.0,
            source=Clock(name="attrs-only", type="device", attrs={"sync_domain": "session-a"}),
            target=source,
        )
    with pytest.raises(ValidationError, match="sync_domain"):
        convert_time(
            1.0,
            source=Clock(name="same", type="device"),
            target=Clock(name="same", type="device"),
        )
    with pytest.raises(ValidationError, match="synchronization_domain"):
        Clock(name="empty-domain", type="device", synchronization_domain="")
    with pytest.raises(ValidationError, match="drift"):
        Clock(name="broken", type="device", drift=-1.0)


def test_reference_clock_epoch_converts_to_local_time():
    reference = _clock("reference", offset=0.5)
    signal_clock = _clock("acq", offset=0.25)
    signal = _signal(t0=0.0, clock=signal_clock)
    recording = Recording(
        signals={"neural": signal},
        trials=TrialTable([Trial(trial_id=0, start=0.45, stop=0.65)]),
    )

    epochs = split_recording_trials(recording, reference_clock=reference)

    np.testing.assert_array_equal(epochs[0].signals["neural"].data, signal.data[2:4])
    np.testing.assert_allclose(epochs[0].signals["neural"].time, [0.2, 0.3])


def test_reference_clock_validates_splitting_clock_identity():
    reference = _clock("reference", domain="session-a")
    recording = Recording(
        signals={
            "neural": _signal(
                t0=0.0,
                clock=_clock("acq", offset=0.25, domain="session-b"),
            )
        },
        trials=TrialTable([Trial(trial_id=0, start=0.45, stop=0.65)]),
    )

    with pytest.raises(ValidationError, match="sync_domain"):
        split_recording_trials(recording, reference_clock=reference)


def test_trial_relative_epoch_uses_tmin_tmax_in_reference_time():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.8)

    epoch = extract_trial_epoch(signal, trial, tmin=0.1, tmax=0.3)

    np.testing.assert_array_equal(epoch.data, signal.data[3:5])
    np.testing.assert_allclose(epoch.time, [0.3, 0.4])


def test_trial_relative_epoch_tmax_none_uses_trial_stop():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)

    epoch = extract_trial_epoch(signal, trial, tmin=0.1, tmax=None)

    np.testing.assert_array_equal(epoch.data, signal.data[3:5])
    np.testing.assert_allclose(epoch.time, [0.3, 0.4])


def test_trial_relative_epoch_rejects_invalid_stop_bounds():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)

    with pytest.raises(ValidationError, match=r"trial\.stop"):
        extract_trial_epoch(signal, trial, tmin=0.1, tmax=0.4)
    with pytest.raises(ValidationError, match="greater than tmin"):
        extract_trial_epoch(signal, trial, tmin=0.1, tmax=0.1)


def test_trial_relative_epoch_allows_negative_tmin_for_baseline():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)

    epoch = extract_trial_epoch(signal, trial, tmin=-0.1, tmax=0.2)

    np.testing.assert_array_equal(epoch.data, signal.data[1:4])
    np.testing.assert_allclose(epoch.time, [0.1, 0.2, 0.3])


def test_trial_relative_epoch_converts_bounds_to_signal_clock():
    reference = _clock("reference", offset=0.5)
    signal_clock = _clock("acq", offset=0.25, drift=0.25)
    signal = _signal(t0=0.0, clock=signal_clock)
    trial = Trial(trial_id=0, start=0.5, stop=0.875)

    epoch = extract_trial_epoch(signal, trial, tmin=0.0, tmax=0.25, reference_clock=reference)

    np.testing.assert_array_equal(epoch.data, signal.data[2:4])
    np.testing.assert_allclose(epoch.time, [0.2, 0.3])


def test_trial_relative_epoch_requires_reference_clock():
    # A clocked signal cannot be sliced on trial-relative reference seconds
    # without an explicit reference clock to validate its synchronization
    # domain -- the same contract split_recording_trials enforces.
    signal = _signal(t0=0.0, clock=_clock("acq", domain="session-a"))
    trial = Trial(trial_id=0, start=0.2, stop=0.5)

    with pytest.raises(ValidationError, match="reference clock"):
        extract_trial_epoch(signal, trial, tmin=0.0, tmax=0.2)

    # An explicit, domain-matching reference clock is accepted.
    reference = _clock("reference", domain="session-a")
    epoch = extract_trial_epoch(signal, trial, tmin=0.0, tmax=0.2, reference_clock=reference)
    np.testing.assert_array_equal(epoch.data, signal.data[2:4])


def test_trial_relative_epoch_rejects_discontinuity_in_baseline():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)
    # epoch is [0.1, 0.4); an instantaneous gap at 0.15 falls inside the
    # pre-trial baseline window and must reject the epoch.
    gap = EventSeries([Event(0.15, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        extract_trial_epoch(signal, trial, tmin=-0.1, tmax=0.2, discontinuities=gap)


def test_trial_relative_epoch_gap_at_stop_is_safe():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)
    # epoch is [0.1, 0.4); an instantaneous gap exactly at the stop boundary
    # is excluded by the half-open interval and must not reject the epoch.
    gap_at_stop = EventSeries([Event(0.4, label="gap")])

    epoch = extract_trial_epoch(signal, trial, tmin=-0.1, tmax=0.2, discontinuities=gap_at_stop)

    np.testing.assert_array_equal(epoch.data, signal.data[1:4])
    np.testing.assert_allclose(epoch.time, [0.1, 0.2, 0.3])


def test_trial_relative_epoch_rejects_overlapping_gap():
    signal = _signal(t0=0.0, clock=_clock("acq", domain="session-a"))
    reference = _clock("reference", domain="session-a")
    trial = Trial(trial_id=0, start=0.2, stop=0.5)
    gap = EventSeries([Event(0.15, label="gap")], clock=_clock("gap", domain="session-a"))

    with pytest.raises(ValidationError, match="discontinuity"):
        extract_trial_epoch(
            signal,
            trial,
            tmin=-0.1,
            tmax=0.2,
            reference_clock=reference,
            discontinuities=gap,
        )


def test_trial_relative_epoch_rejects_foreign_sync_domain():
    signal = _signal(t0=0.0, clock=_clock("acq", domain="session-a"))
    reference = _clock("reference", domain="session-a")
    trial = Trial(trial_id=0, start=0.2, stop=0.5)
    gap = EventSeries([Event(0.15, label="gap")], clock=_clock("gap", domain="session-b"))

    with pytest.raises(ValidationError, match="sync_domain"):
        extract_trial_epoch(
            signal,
            trial,
            tmin=-0.1,
            tmax=0.2,
            reference_clock=reference,
            discontinuities=gap,
        )


def test_trial_relative_epoch_requires_reference_domain():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)
    gap = EventSeries([Event(0.15, label="gap")], clock=_clock("gap", domain="session-a"))

    with pytest.raises(ValidationError, match="reference clock"):
        extract_trial_epoch(signal, trial, tmin=-0.1, tmax=0.2, discontinuities=gap)


def test_discontinuity_boundaries_follow_half_open_barrier_rules():
    signal = _signal(t0=0.0)

    sustained_ends_at_start = EventSeries([Event(0.1, duration=0.2, label="gap")])
    sustained_starts_at_stop = EventSeries([Event(0.5, duration=0.2, label="gap")])
    instant_at_start = EventSeries([Event(0.3, label="gap")])
    instant_at_stop = EventSeries([Event(0.5, label="gap")])

    extract_signal_epoch(signal, 0.3, 0.5, discontinuities=sustained_ends_at_start)
    extract_signal_epoch(signal, 0.3, 0.5, discontinuities=sustained_starts_at_stop)
    extract_signal_epoch(signal, 0.3, 0.5, discontinuities=instant_at_stop)
    with pytest.raises(ValidationError, match="discontinuity"):
        extract_signal_epoch(signal, 0.3, 0.5, discontinuities=instant_at_start)


def test_events_in_interval_uses_half_open_bounds_and_labels():
    events = EventSeries(
        [
            Event(1.0, label="trial_start"),
            Event(1.5, label="cue"),
            Event(2.0, label="trial_stop"),
        ],
        attrs={"source": "behavior"},
    )

    selected = events_in_interval(events, 1.0, 2.0, required_labels=["trial_start", "cue"])

    assert [event.label for event in selected] == ["trial_start", "cue"]
    assert selected.attrs == events.attrs
    with pytest.raises(ValidationError, match="missing required events"):
        events_in_interval(events, 1.0, 2.0, required_labels=["go"])
    with pytest.raises(ValidationError, match="required_labels"):
        events_in_interval(events, 1.0, 2.0, required_labels="cue")


def test_trial_table_splitting_returns_typed_epochs_and_metadata():
    recording = Recording(
        signals={
            "neural": _signal(t0=0.0, name="neural"),
            "behavior": _signal(fs=5.0, t0=0.0, name="behavior"),
        },
        events=EventSeries([Event(0.1, label="cue"), Event(0.5, label="go")]),
        trials=TrialTable(
            [
                Trial(trial_id=0, start=0.0, stop=0.4, attrs={"target": 1}),
                Trial(trial_id=1, start=0.4, stop=0.8, attrs={"target": 2}),
            ]
        ),
        metadata={"session": "S01"},
    )

    epochs = split_recording_trials(recording, reference_clock=None)

    assert len(epochs) == 2
    assert set(epochs[0].signals) == {"neural", "behavior"}
    assert epochs[0].signals["neural"].data.shape == (4, 2)
    assert epochs[0].signals["behavior"].data.shape == (2, 2)
    assert [event.label for event in epochs[0].events] == ["cue"]
    assert epochs[0].recording_metadata == {"session": "S01"}
    assert epochs[0].trial_attrs == {"target": 1}


def test_trial_epoch_metadata_is_frozen_and_does_not_leak():
    trial = Trial(trial_id=0, start=0.0, stop=0.2, attrs={"nested": {"x": 1}})
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([trial]),
        metadata={"nested": {"x": 1}},
    )

    epoch = split_recording_trials(recording, reference_clock=None)[0]

    # Mutating nested metadata on the result must not reach the source objects.
    with pytest.raises(TypeError):
        epoch.recording_metadata["nested"]["x"] = 9
    with pytest.raises(TypeError):
        epoch.trial_attrs["nested"]["x"] = 8
    assert recording.metadata["nested"]["x"] == 1
    assert trial.attrs["nested"]["x"] == 1
    # The result still compares equal to a plain dict of the same contents.
    assert epoch.recording_metadata == {"nested": {"x": 1}}
    assert epoch.trial_attrs == {"nested": {"x": 1}}


def test_trial_is_immutable():
    trial = Trial(trial_id=0, start=0.0, stop=0.2, attrs={"x": 1}, target=np.array([1, 2]))

    # attrs is a deep-frozen mapping; target is an immutable array.
    with pytest.raises(TypeError):
        trial.attrs["x"] = 9
    with pytest.raises(ValueError):
        trial.target[0] = 99
    assert trial.attrs["x"] == 1
    assert trial.target[0] == 1
    assert not trial.target.flags.writeable


def test_trial_epoch_trial_field_cannot_leak_to_source_trial():
    trial = Trial(trial_id=0, start=0.0, stop=0.2, attrs={"x": 1}, target=np.array([1, 2]))
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([trial]),
    )

    epoch = split_recording_trials(recording, reference_clock=None)[0]

    # TrialEpoch.trial references the source Trial, but Trial is immutable, so
    # mutating through the result raises and does not leak back.
    with pytest.raises(TypeError):
        epoch.trial.attrs["x"] = 9
    with pytest.raises(ValueError):
        epoch.trial.target[0] = 99
    assert trial.attrs["x"] == 1
    assert trial.target[0] == 1


def test_trial_epoch_signals_mapping_is_immutable():
    recording = Recording(
        signals={"neural": _signal(t0=0.0, name="neural")},
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )

    epoch = split_recording_trials(recording, reference_clock=None)[0]

    with pytest.raises(TypeError):
        epoch.signals["neural"] = _signal(t0=0.0, name="neural")
    with pytest.raises(AttributeError):
        epoch.signals.clear()
    assert set(epoch.signals) == {"neural"}


def test_trial_epoch_rejects_invalid_construction():
    neural = _signal(t0=0.0, name="neural")
    trial = Trial(trial_id=0, start=0.0, stop=0.2)

    with pytest.raises(ValidationError, match="start < stop"):
        TrialEpoch(
            trial=trial,
            start=0.2,
            stop=0.2,
            signals={"neural": neural},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="non-empty strings"):
        TrialEpoch(
            trial=trial,
            start=0.0,
            stop=0.2,
            signals={"": neural},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="SignalArray"):
        TrialEpoch(
            trial=trial,
            start=0.0,
            stop=0.2,
            signals={"neural": object()},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="EventSeries or None"):
        TrialEpoch(
            trial=trial,
            start=0.0,
            stop=0.2,
            signals={"neural": neural},
            events=object(),
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="Trial"):
        TrialEpoch(
            trial=object(),  # type: ignore[arg-type]
            start=0.0,
            stop=0.2,
            signals={"neural": neural},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="within the source trial"):
        TrialEpoch(
            trial=trial,
            start=0.1,
            stop=0.3,  # stop exceeds trial.stop (0.2)
            signals={"neural": neural},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="within the source trial"):
        TrialEpoch(
            trial=trial,
            start=-0.1,  # start precedes trial.start (0.0)
            stop=0.2,
            signals={"neural": neural},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="signals must not be empty"):
        TrialEpoch(
            trial=trial,
            start=0.0,
            stop=0.2,
            signals={},
            events=None,
            recording_metadata={},
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="recording_metadata must be a mapping"):
        TrialEpoch(
            trial=trial,
            start=0.0,
            stop=0.2,
            signals={"neural": neural},
            events=None,
            recording_metadata=None,
            trial_attrs={},
        )
    with pytest.raises(ValidationError, match="trial_attrs must be a mapping"):
        TrialEpoch(
            trial=trial,
            start=0.0,
            stop=0.2,
            signals={"neural": neural},
            events=None,
            recording_metadata={},
            trial_attrs=None,
        )


def test_trial_splitting_requires_events_for_required_labels():
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )

    epochs = split_recording_trials(recording, reference_clock=None)

    assert epochs[0].events is None
    with pytest.raises(ValidationError, match="recording has no events"):
        split_recording_trials(
            recording,
            reference_clock=None,
            required_event_labels=["cue"],
        )


def test_trial_splitting_rejects_interval_outside_coverage():
    signal_a = SignalArray.from_array(np.zeros((10, 1)), fs=10.0, t0=0.0)
    signal_b = SignalArray.from_array(np.zeros((6, 1)), fs=10.0, t0=0.2)
    recording = Recording(
        signals={"a": signal_a, "b": signal_b},
        events=EventSeries(
            [
                Event(0.1, label="before_common"),
                Event(0.3, label="inside_common"),
                Event(0.9, label="after_common"),
            ]
        ),
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=1.0)]),
    )
    # Strict mode slices each trial at its own [start, stop); signal b only
    # covers [0.2, 0.8), so the trial interval [0.0, 1.0) is out of range and
    # splitting raises rather than clipping to common coverage.
    with pytest.raises(ValidationError, match="outside"):
        split_recording_trials(recording, reference_clock=None)


def test_trial_splitting_validates_selected_signal_names():
    recording = Recording(
        signals={
            "neural": _signal(t0=0.0, name="neural"),
            "behavior": _signal(t0=0.0, name="behavior"),
        },
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )

    epochs = split_recording_trials(
        recording,
        reference_clock=None,
        signal_names=["behavior", "neural"],
    )

    assert tuple(epochs[0].signals) == ("behavior", "neural")
    with pytest.raises(ValidationError, match="not a string"):
        split_recording_trials(recording, reference_clock=None, signal_names="neural")
    with pytest.raises(ValidationError, match="duplicate"):
        split_recording_trials(
            recording,
            reference_clock=None,
            signal_names=["neural", "neural"],
        )
    with pytest.raises(ValidationError, match="unknown signal names"):
        split_recording_trials(recording, reference_clock=None, signal_names=["missing"])


def test_trial_table_splitting_requires_ordered_positive_trials():
    with pytest.raises(ValidationError, match="greater than trial start"):
        Trial(trial_id=1, start=1.0, stop=1.0)
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable(
            [
                Trial(trial_id=0, start=0.2, stop=0.6),
                Trial(trial_id=1, start=0.1, stop=0.3),
            ]
        ),
    )

    with pytest.raises(ValidationError, match="non-overlapping"):
        split_recording_trials(recording, reference_clock=None)


def test_resampled_signals_are_sliced_by_their_current_time_grid():
    resampled = SignalArray.from_array(
        np.arange(10.0).reshape(5, 2),
        fs=5.0,
        t0=0.0,
        channel_names=["C3", "C4"],
        channel_types="eeg",
        units="uV",
        name="neural",
        attrs={"resampled_from_hz": 10.0},
    )

    epoch = extract_signal_epoch(resampled, 0.2, 0.8)

    np.testing.assert_array_equal(epoch.data, resampled.data[1:4])
    np.testing.assert_allclose(epoch.time, [0.2, 0.4, 0.6])
    assert epoch.fs == 5.0


def test_irregular_signal_epoch_extraction_is_rejected():
    signal = SignalArray.from_array(
        np.zeros((4, 1)),
        fs=10.0,
        time=np.array([0.0, 0.1, 0.25, 0.4]),
    )

    with pytest.raises(ValidationError, match="regular"):
        extract_signal_epoch(signal, 0.0, 0.3)


def test_alignment_uses_one_to_one_nearest_matches():
    neural = SignalArray.from_array(np.zeros((4, 1)), fs=10.0, t0=0.0)
    behavior = SignalArray.from_array(
        np.zeros((3, 1)),
        fs=10.0,
        time=np.array([0.01, 0.1, 0.29]),
    )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.01,
    )

    # neural 0.0 precedes behavior coverage (starts at 0.01), so it lies outside
    # common coverage and stays unmatched; rows 0.1 and 0.3 demonstrate one-to-one
    # nearest matching with inclusive tolerance (distances 0.0 and 0.01).
    np.testing.assert_array_equal(alignment.neural_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(alignment.behavior_indices, [-1, 1, -1, 2])
    np.testing.assert_array_equal(alignment.matched, [False, True, False, True])
    np.testing.assert_allclose(alignment.reference_time, neural.time)


def test_alignment_target_indices_survive_unmatched_rows():
    neural = SignalArray.from_array(
        np.zeros((3, 1)),
        fs=10.0,
        time=np.array([0.0, 0.1, 0.2]),
    )
    behavior = SignalArray.from_array(
        np.zeros((4, 1)),
        fs=10.0,
        time=np.array([0.0, 0.1, 0.2, 0.3]),
    )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="behavior",
        tol=0.0,
    )

    np.testing.assert_array_equal(alignment.neural_indices, [0, 1, 2, -1])
    np.testing.assert_array_equal(alignment.behavior_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(alignment.matched, [True, True, True, False])


def test_one_to_one_alignment_is_sparse_on_dense_target():
    neural = SignalArray.from_array(np.zeros((20, 1)), fs=1000.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=100.0, t0=0.0)

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.0005,
    )

    expected_behavior_indices = np.full(20, -1, dtype=int)
    expected_behavior_indices[[0, 10]] = [0, 1]
    np.testing.assert_array_equal(alignment.behavior_indices, expected_behavior_indices)
    assert np.count_nonzero(alignment.matched) == behavior.n_samples


def test_nearest_alignment_direct_tie_chooses_earlier_source():
    # neural 0.5 ties behavior 0.4 and 0.6, both inside common coverage [0.4,
    # 0.7); the earlier source wins. neural 0.3 is in the target-only tail and
    # stays unmatched.
    neural = SignalArray.from_array(
        np.zeros((2, 1)),
        fs=5.0,
        time=np.array([0.3, 0.5]),
    )
    behavior = SignalArray.from_array(
        np.zeros((2, 1)),
        fs=5.0,
        time=np.array([0.4, 0.6]),
    )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.1,
    )

    np.testing.assert_array_equal(alignment.neural_indices, [0, 1])
    np.testing.assert_array_equal(alignment.behavior_indices, [-1, 0])
    np.testing.assert_array_equal(alignment.matched, [False, True])


def test_nearest_alignment_ties_choose_earlier_unused_source():
    # Integer grid so the tie at neural 1.0 is exact (distance 1.0 to both
    # behavior 0.0 and behavior 2.0), not an artifact of float rounding. Common
    # coverage is [0, 3); neural 3.0 lies on the half-open stop and stays
    # unmatched. neural 1.0 ties the two sources and picks the earlier unused
    # one (behavior 2.0, since behavior 0.0 is already bound to neural 0.0);
    # neural 2.0 cannot reuse behavior 2.0 and behavior 0.0 is out of tolerance.
    neural = SignalArray.from_array(
        np.zeros((4, 1)),
        fs=1.0,
        time=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    behavior = SignalArray.from_array(
        np.zeros((2, 1)),
        fs=1.0,
        time=np.array([0.0, 2.0]),
    )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=1.0,
    )

    np.testing.assert_array_equal(alignment.behavior_indices, [0, 1, -1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, True, False, False])


def test_nearest_alignment_forbids_crossing_source_matches():
    neural = SignalArray.from_array(
        np.zeros((2, 1)),
        fs=100.0,
        time=np.array([0.49, 0.51]),
    )
    behavior = SignalArray.from_array(
        np.zeros((2, 1)),
        fs=100.0,
        time=np.array([0.0, 0.5]),
    )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.6,
    )

    np.testing.assert_array_equal(alignment.behavior_indices, [1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, False])


def test_nearest_alignment_indices_strictly_increase():
    neural = SignalArray.from_array(
        np.zeros((5, 1)),
        fs=10.0,
        time=np.array([0.0, 0.11, 0.21, 0.32, 0.43]),
    )
    behavior = SignalArray.from_array(
        np.zeros((4, 1)),
        fs=10.0,
        time=np.array([0.01, 0.2, 0.31, 0.44]),
    )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.05,
    )

    matched_neural = alignment.neural_indices[alignment.matched]
    matched_behavior = alignment.behavior_indices[alignment.matched]
    assert np.all(np.diff(matched_neural) > 0)
    assert np.all(np.diff(matched_behavior) > 0)


def test_alignment_rejects_discontinuity_in_any_input_stream():
    neural = SignalArray.from_array(np.zeros((4, 1)), fs=10.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((4, 1)), fs=10.0, t0=0.0)
    gap = EventSeries([Event(0.2, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        align_neural_behavior_nearest(
            neural,
            behavior,
            target="neural",
            tol=0.01,
            discontinuities=[gap],
        )


def test_alignment_gate_uses_common_coverage_not_target_tail():
    neural = SignalArray.from_array(np.zeros((5, 1)), fs=10.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, t0=0.0)
    inside_common_coverage = EventSeries([Event(0.15, label="gap")])
    target_only_tail = EventSeries([Event(0.3, label="gap")])

    with pytest.raises(ValidationError, match="discontinuity"):
        align_neural_behavior_nearest(
            neural,
            behavior,
            target="neural",
            tol=0.0,
            discontinuities=[inside_common_coverage],
        )

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.0,
        discontinuities=[target_only_tail],
    )

    np.testing.assert_array_equal(alignment.behavior_indices, [0, 1, -1, -1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, True, False, False, False])


def test_alignment_excludes_source_tail_from_matching():
    # neural coverage is [0.05, 0.25); behavior coverage is [0.00, 0.20); common
    # coverage is [0.05, 0.20). behavior 0.00 lies in the source-only tail and
    # must not match even though it is within tolerance of neural 0.05.
    neural = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.05, 0.15]))
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.0, 0.1]))

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.051,
    )

    np.testing.assert_array_equal(alignment.neural_indices, [0, 1])
    np.testing.assert_array_equal(alignment.behavior_indices, [1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, False])


def test_alignment_excludes_source_tail_when_target_is_behavior():
    # Symmetric to the neural-target case: neural is the source here and its
    # 0.00 sample is in the source-only tail of common coverage [0.05, 0.20).
    neural = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.0, 0.1]))
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.05, 0.15]))

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="behavior",
        tol=0.051,
    )

    np.testing.assert_array_equal(alignment.behavior_indices, [0, 1])
    np.testing.assert_array_equal(alignment.neural_indices, [1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, False])


def test_alignment_ignores_source_tail_discontinuity():
    # A discontinuity in behavior's source-only tail is outside common coverage
    # so the gate ignores it; the source-only-tail sample 0.00 is excluded from
    # matching too, so no cross-gap match is produced into the ungated tail.
    neural = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.05, 0.15]))
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.0, 0.1]))
    gap = EventSeries([Event(0.025, label="gap")])

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.051,
        discontinuities=[gap],
    )

    np.testing.assert_array_equal(alignment.behavior_indices, [1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, False])


def test_alignment_leaves_target_rows_outside_coverage_unmatched():
    # neural coverage is [0, 2); behavior coverage is [0, 0.2); common coverage
    # is [0, 0.2). neural 1.0 lies outside common coverage, so even though it is
    # within tolerance of behavior 0.1, it must stay unmatched.
    neural = SignalArray.from_array(np.zeros((2, 1)), fs=1.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, t0=0.0)

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="neural",
        tol=0.9,
    )

    np.testing.assert_array_equal(alignment.neural_indices, [0, 1])
    np.testing.assert_array_equal(alignment.behavior_indices, [0, -1])
    np.testing.assert_array_equal(alignment.matched, [True, False])


def test_alignment_leaves_source_rows_outside_coverage():
    # When behavior is the target, its rows outside common coverage stay
    # unmatched too: behavior coverage is [0, 0.4), neural coverage is
    # [0, 0.2); behavior 0.3 is outside common coverage and must not match.
    neural = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((4, 1)), fs=10.0, t0=0.0)

    alignment = align_neural_behavior_nearest(
        neural,
        behavior,
        target="behavior",
        tol=0.9,
    )

    np.testing.assert_array_equal(alignment.behavior_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(alignment.neural_indices, [0, 1, -1, -1])
    np.testing.assert_array_equal(alignment.matched, [True, True, False, False])


def test_alignment_rejects_signals_with_no_common_time_coverage():
    neural = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, t0=10.0)
    behavior = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, t0=0.0)
    long_gap = EventSeries([Event(0.0, duration=20.0, label="gap")])

    with pytest.raises(ValidationError, match="no common time coverage"):
        align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.0)
    with pytest.raises(ValidationError, match="no common time coverage"):
        align_neural_behavior_nearest(
            neural,
            behavior,
            target="neural",
            tol=0.0,
            discontinuities=[long_gap],
        )


def test_alignment_rejects_empty_neural_or_behavior_signal():
    neural = SignalArray.from_array(np.empty((0, 1)), fs=10.0)
    behavior = SignalArray.from_array(np.zeros((3, 1)), fs=10.0)

    with pytest.raises(ValidationError, match="non-empty neural and behavior"):
        align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.01)
    with pytest.raises(ValidationError, match="non-empty neural and behavior"):
        align_neural_behavior_nearest(behavior, neural, target="neural", tol=0.01)


def test_alignment_rejects_duplicate_behavior_timestamps():
    # SignalArray permits non-decreasing time, so duplicate timestamps are a
    # legal input that the strictly-increasing matcher must reject rather than
    # mis-matching by skipping the duplicate.
    neural = SignalArray.from_array(np.zeros((1, 1)), fs=10.0, time=np.array([0.5]))
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, time=np.array([0.0, 0.0]))

    with pytest.raises(ValidationError, match="strictly increasing"):
        align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.5)


def test_alignment_rejects_duplicate_neural_timestamps():
    neural = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, time=np.array([0.0, 0.1, 0.1]))
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, t0=0.0)

    with pytest.raises(ValidationError, match="strictly increasing"):
        align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.2)


def test_alignment_accepts_single_sample_signals():
    # A single sample has no diff to check; alignment must still succeed.
    neural = SignalArray.from_array(np.zeros((1, 1)), fs=10.0, time=np.array([0.5]))
    behavior = SignalArray.from_array(np.zeros((1, 1)), fs=10.0, time=np.array([0.5]))

    alignment = align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.0)

    np.testing.assert_array_equal(alignment.neural_indices, [0])
    np.testing.assert_array_equal(alignment.behavior_indices, [0])
    np.testing.assert_array_equal(alignment.matched, [True])


def test_alignment_index_arrays_are_truly_immutable():
    neural = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((2, 1)), fs=10.0, t0=0.0)
    alignment = align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.01)

    for arr in (
        alignment.reference_time,
        alignment.neural_indices,
        alignment.behavior_indices,
        alignment.matched,
    ):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 0
        # flags.writeable = False can be re-enabled; the frombuffer-backed copy
        # must resist re-enabling so a frozen AlignmentIndex stays consistent.
        with pytest.raises(ValueError):
            arr.flags.writeable = True


def _valid_alignment_index_kwargs() -> dict:
    return dict(
        mode="nearest",
        target="neural",
        tol=0.1,
        reference_time=np.array([0.0, 1.0]),
        neural_indices=np.array([0, 1]),
        behavior_indices=np.array([0, -1]),
        matched=np.array([True, False]),
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"mode": "other"}, "mode"),
        ({"target": "other"}, "target"),
        ({"tol": -1.0}, "tol"),
        ({"reference_time": np.array([[0.0, 1.0]])}, "1D"),
        ({"neural_indices": np.array([0])}, "same length"),
        ({"matched": np.array([1, 0])}, "boolean"),
        ({"neural_indices": np.array([0, 0])}, "arange"),
        ({"matched": np.array([True, True])}, "validity"),
        ({"behavior_indices": np.array([0, -2])}, "must be -1"),
        (
            {
                "reference_time": np.array([0.0, 1.0, 2.0]),
                "neural_indices": np.array([0, 1, 2]),
                "behavior_indices": np.array([1, 0, -1]),
                "matched": np.array([True, True, False]),
            },
            "source indices must be strictly increasing",
        ),
        ({"reference_time": np.array([0.0, np.nan])}, "finite"),
        ({"reference_time": np.array([np.inf, 0.0])}, "finite"),
        ({"reference_time": np.array([1.0, 0.0])}, "strictly increasing"),
        ({"reference_time": np.array([0.0, 0.0])}, "strictly increasing"),
        ({"reference_time": np.array([0 + 1j, 1 + 0j])}, "real"),
        ({"reference_time": np.array([0.0, 1.0], dtype=object)}, "numeric"),
    ],
)
def test_alignment_index_rejects_invalid_construction(overrides, match):
    kwargs = _valid_alignment_index_kwargs()
    kwargs.update(overrides)

    with pytest.raises(ValidationError, match=match):
        AlignmentIndex(**kwargs)


def test_alignment_index_rejects_overflowed_indices():
    # An int8 target index overflows past 127; with dtype normalization the
    # expected arange is built in intp and the overflowed values no longer match.
    n = 200
    with pytest.raises(ValidationError, match="arange"):
        AlignmentIndex(
            mode="nearest",
            target="neural",
            tol=0.0,
            reference_time=np.arange(n, dtype=np.float64),
            neural_indices=np.arange(n, dtype=np.int8),
            behavior_indices=np.full(n, -1, dtype=np.int8),
            matched=np.zeros(n, dtype=bool),
        )


def test_alignment_index_normalizes_array_dtypes():
    # Constructed with float32 reference_time and default-int indices; the
    # stored arrays are normalized to float64 / intp / bool and remain immutable.
    idx = AlignmentIndex(
        mode="nearest",
        target="neural",
        tol=0.0,
        reference_time=np.array([0.0, 1.0], dtype=np.float32),
        neural_indices=np.array([0, 1]),
        behavior_indices=np.array([0, -1]),
        matched=np.array([True, False]),
    )

    assert idx.reference_time.dtype == np.float64
    assert np.issubdtype(idx.neural_indices.dtype, np.integer)
    assert idx.neural_indices.dtype == np.intp
    assert idx.behavior_indices.dtype == np.intp
    assert idx.matched.dtype == np.bool_


def test_split_recording_trials_rejects_non_clock_reference():
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )

    with pytest.raises(ValidationError, match="reference_clock must be a Clock"):
        split_recording_trials(recording, reference_clock="bad")
    with pytest.raises(ValidationError, match="discontinuities must be"):
        split_recording_trials(recording, reference_clock=None, discontinuities="bad")


def test_extract_trial_epoch_rejects_non_clock_reference():
    signal = _signal(t0=0.0)
    trial = Trial(trial_id=0, start=0.2, stop=0.5)

    # signal has no clock, so the old short-circuit silently accepted bad clocks.
    with pytest.raises(ValidationError, match="reference_clock must be a Clock"):
        extract_trial_epoch(signal, trial, reference_clock="bad")
    with pytest.raises(ValidationError, match="discontinuities must be"):
        extract_trial_epoch(signal, trial, discontinuities="bad")


def test_time_conversion_rejects_non_clock_arguments():
    clock = _clock("acq")
    with pytest.raises(ValidationError, match="source must be a Clock"):
        convert_time(1.0, source="bad", target=clock)
    with pytest.raises(ValidationError, match="target must be a Clock"):
        convert_time(1.0, source=clock, target="bad")
    with pytest.raises(ValidationError, match="source must be a Clock"):
        to_reference_time(1.0, "bad")
    with pytest.raises(ValidationError, match="target must be a Clock"):
        from_reference_time(1.0, "bad")


def test_time_conversion_rejects_non_finite_result_from_overflow():
    # Finite inputs/offset/drift can still overflow the affine result to inf.
    big_to = _clock("big-to", offset=1e308, drift=0.0)
    with pytest.raises(ValidationError, match="non-finite"):
        to_reference_time(1e308, big_to)
    with pytest.raises(ValidationError, match="non-finite"):
        convert_time(1e308, source=big_to, target=None)

    big_from = _clock("big-from", offset=-1e308, drift=0.0)
    with pytest.raises(ValidationError, match="non-finite"):
        from_reference_time(1e308, big_from)
    with pytest.raises(ValidationError, match="non-finite"):
        convert_time(1e308, source=None, target=big_from)


def test_extract_signal_epoch_rejects_non_clock_and_non_events():
    signal = _signal(t0=0.0)
    with pytest.raises(ValidationError, match="clock must be a Clock"):
        extract_signal_epoch(signal, 0.0, 0.2, clock="bad")
    with pytest.raises(ValidationError, match="discontinuities must be"):
        extract_signal_epoch(signal, 0.0, 0.2, discontinuities="bad")

    events = EventSeries([Event(0.1, label="cue")])
    with pytest.raises(ValidationError, match="clock must be a Clock"):
        events_in_interval(events, 0.0, 0.4, clock="bad")


def test_iterable_params_reject_non_iterable_without_typeerror():
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )
    with pytest.raises(ValidationError, match="signal_names must be an iterable"):
        split_recording_trials(recording, reference_clock=None, signal_names=123)
    with pytest.raises(ValidationError, match="required_labels must be an iterable"):
        split_recording_trials(recording, reference_clock=None, required_event_labels=123)

    events = EventSeries([Event(0.1, label="cue")])
    with pytest.raises(ValidationError, match="required_labels must be an iterable"):
        events_in_interval(events, 0.0, 0.4, required_labels=123)


def test_required_labels_none_is_accepted_as_empty():
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )
    # required_event_labels=None means "no required labels", like the default ().
    epochs = split_recording_trials(recording, reference_clock=None, required_event_labels=None)
    assert epochs[0].events is None

    events = EventSeries([Event(0.1, label="cue")])
    selected = events_in_interval(events, 0.0, 0.4, required_labels=None)
    assert [event.label for event in selected] == ["cue"]


def test_align_rejects_non_signal_and_normalizes_gaps():
    behavior = _signal(t0=0.0, name="behavior")
    with pytest.raises(ValidationError, match="neural must be a SignalArray"):
        align_neural_behavior_nearest("not-a-signal", behavior, tol=0.1)
    with pytest.raises(ValidationError, match="behavior must be a SignalArray"):
        align_neural_behavior_nearest(behavior, "not-a-signal", tol=0.1)

    neural = _signal(t0=0.0, name="neural")
    gap = EventSeries([Event(10.0, label="gap")])

    # discontinuities=None is accepted (no TypeError leaked).
    alignment_none = align_neural_behavior_nearest(
        neural, behavior, target="neural", tol=0.0, discontinuities=None
    )
    assert alignment_none.matched.all()

    # A bare EventSeries is accepted and normalized (not iterated as events).
    alignment_bare = align_neural_behavior_nearest(
        neural, behavior, target="neural", tol=0.0, discontinuities=gap
    )
    assert alignment_bare.matched.all()

    with pytest.raises(ValidationError, match="discontinuities must be"):
        align_neural_behavior_nearest(
            neural, behavior, target="neural", tol=0.0, discontinuities="bad"
        )
    with pytest.raises(ValidationError, match="discontinuities must contain only"):
        align_neural_behavior_nearest(
            neural, behavior, target="neural", tol=0.0, discontinuities=[gap, "bad"]
        )


def test_alignment_converts_gap_clock_to_reference_time():
    neural_clock = _clock("neural")
    behavior_clock = _clock("behavior")
    neural = SignalArray.from_array(
        np.zeros((7, 1)),
        fs=10.0,
        t0=0.0,
        clock=neural_clock,
    )
    behavior = SignalArray.from_array(
        np.zeros((7, 1)),
        fs=10.0,
        t0=0.0,
        clock=behavior_clock,
    )
    gap = EventSeries([Event(0.0, label="gap")], clock=_clock("gap", offset=0.5))

    with pytest.raises(ValidationError, match="discontinuity"):
        align_neural_behavior_nearest(
            neural,
            behavior,
            target="neural",
            tol=0.0,
            discontinuities=[gap],
        )


def test_alignment_rejects_foreign_discontinuity_sync_domain():
    neural = SignalArray.from_array(
        np.zeros((4, 1)),
        fs=10.0,
        t0=0.0,
        clock=_clock("neural", domain="session-a"),
    )
    behavior = SignalArray.from_array(
        np.zeros((4, 1)),
        fs=10.0,
        t0=0.0,
        clock=_clock("behavior", domain="session-a"),
    )
    gap = EventSeries([Event(0.2, label="gap")], clock=_clock("gap", domain="session-b"))

    with pytest.raises(ValidationError, match="sync_domain"):
        align_neural_behavior_nearest(
            neural,
            behavior,
            target="neural",
            tol=0.0,
            discontinuities=[gap],
        )


def test_alignment_requires_clock_for_clocked_discontinuity():
    neural = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, t0=0.0)
    behavior = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, t0=0.0)
    gap = EventSeries([Event(10.0, label="gap")], clock=_clock("gap"))

    with pytest.raises(ValidationError, match="alignment reference clock"):
        align_neural_behavior_nearest(
            neural,
            behavior,
            target="neural",
            tol=0.0,
            discontinuities=[gap],
        )


def test_alignment_rejects_mismatched_sync_domains():
    neural = SignalArray.from_array(
        np.zeros((3, 1)),
        fs=10.0,
        t0=0.0,
        clock=_clock("neural", domain="session-a"),
    )
    behavior = SignalArray.from_array(
        np.zeros((3, 1)),
        fs=10.0,
        t0=0.0,
        clock=_clock("behavior", domain="session-b"),
    )

    with pytest.raises(ValidationError, match="sync_domain"):
        align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.01)


def test_alignment_treats_missing_clock_as_reference_coordinate():
    neural = SignalArray.from_array(
        np.zeros((3, 1)),
        fs=10.0,
        t0=0.0,
        clock=_clock("neural", offset=0.1),
    )
    behavior = SignalArray.from_array(np.zeros((3, 1)), fs=10.0, t0=0.1)

    alignment = align_neural_behavior_nearest(neural, behavior, target="neural", tol=0.0)

    np.testing.assert_array_equal(alignment.behavior_indices, [0, 1, 2])
    np.testing.assert_array_equal(alignment.matched, [True, True, True])


def test_trial_splitting_requires_reference_clock_for_signals():
    recording = Recording(
        signals={
            "a": _signal(t0=0.0, name="a", clock=_clock("acq-a", domain="session-a")),
            "b": _signal(t0=0.0, name="b", clock=_clock("acq-b", domain="session-b")),
        },
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )

    with pytest.raises(ValidationError, match="reference clock"):
        split_recording_trials(recording, reference_clock=None)


def test_trial_splitting_requires_reference_clock_for_events():
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        events=EventSeries(
            [Event(0.1, label="cue")],
            clock=_clock("events", domain="session-a"),
        ),
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )

    with pytest.raises(ValidationError, match="reference clock"):
        split_recording_trials(recording, reference_clock=None)


def test_trial_splitting_requires_reference_clock_for_gaps():
    recording = Recording(
        signals={"neural": _signal(t0=0.0)},
        trials=TrialTable([Trial(trial_id=0, start=0.0, stop=0.4)]),
    )
    gap = EventSeries(
        [Event(0.2, label="gap")],
        clock=_clock("gap", domain="session-b"),
    )

    with pytest.raises(ValidationError, match="reference clock"):
        split_recording_trials(recording, reference_clock=None, discontinuities=gap)


def test_alignment_api_is_public_and_helpers_private():
    import neurale.data as data

    assert data.AlignmentIndex is AlignmentIndex
    assert data.TrialEpoch is TrialEpoch
    assert data.extract_signal_epoch is extract_signal_epoch
    assert data.extract_trial_epoch is extract_trial_epoch
    assert data.align_neural_behavior_nearest is align_neural_behavior_nearest
    assert not hasattr(data, "_nearest_one_to_one")


def _reference_nearest_one_to_one(target_time, source_time, tol, in_coverage=None):
    # Brute-force restatement of the original scan-and-lexsort semantics, kept
    # as an oracle for the monotonic two-pointer implementation.
    indices = np.full(target_time.shape, -1, dtype=int)
    matched = np.zeros(target_time.shape, dtype=bool)
    used: set[int] = set()
    last_source_idx = -1
    for target_idx, value in enumerate(target_time):
        if in_coverage is not None and not in_coverage[target_idx]:
            continue
        distances = np.abs(source_time - value)
        order = np.lexsort((np.arange(source_time.size), distances))
        for source_idx in order:
            source_idx = int(source_idx)
            if source_idx in used or source_idx <= last_source_idx:
                continue
            if distances[source_idx] <= tol or np.isclose(
                distances[source_idx], tol, rtol=1e-12, atol=1e-15
            ):
                indices[target_idx] = source_idx
                matched[target_idx] = True
                used.add(source_idx)
                last_source_idx = source_idx
            break
    return indices, matched


def _strictly_increasing_grid(rng, size, scale):
    increments = rng.exponential(scale=scale, size=size)
    return np.cumsum(increments)


@pytest.mark.parametrize("seed", range(200))
def test_nearest_one_to_one_matches_brute_force(seed):
    from neurale.data._alignment import _nearest_one_to_one

    rng = np.random.default_rng(seed)
    n_targets = int(rng.integers(1, 40))
    n_sources = int(rng.integers(1, 40))
    target_time = _strictly_increasing_grid(rng, n_targets, 0.05)
    source_time = _strictly_increasing_grid(rng, n_sources, 0.05)
    tol = float(rng.uniform(0.0, 0.2))
    if rng.random() < 0.5:
        in_coverage = rng.random(n_targets) < 0.8
    else:
        in_coverage = None

    expected_indices, expected_matched = _reference_nearest_one_to_one(
        target_time, source_time, tol, in_coverage
    )
    indices, matched = _nearest_one_to_one(target_time, source_time, tol, in_coverage)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(matched, expected_matched)


def _regular_signal(rng, max_start, min_span):
    rate = float(rng.uniform(10.0, 100.0))
    period = 1.0 / rate
    n = max(2, int(rng.integers(2, 30)))
    while n * period < min_span:
        n += 1
    t0 = float(rng.uniform(0.0, max_start))
    time = t0 + np.arange(n) * period
    signal = SignalArray.from_array(np.zeros((n, 1)), fs=rate, t0=t0)
    return signal, time, period


def _reference_align_matching(target_ref, source_ref, tol, common_start, common_stop):
    from neurale.data._alignment import _within_common_coverage

    target_in = _within_common_coverage(target_ref, common_start, common_stop)
    source_in = _within_common_coverage(source_ref, common_start, common_stop)
    indices = np.full(target_ref.shape, -1, dtype=int)
    matched = np.zeros(target_ref.shape, dtype=bool)
    last = -1
    for i in range(target_ref.shape[0]):
        if not target_in[i]:
            continue
        value = target_ref[i]
        best = -1
        best_distance = 0.0
        for j in range(source_ref.shape[0]):
            if not source_in[j] or j <= last:
                continue
            distance = abs(source_ref[j] - value)
            if best == -1 or distance < best_distance:
                best = j
                best_distance = distance
        if best == -1:
            continue
        if best_distance <= tol or np.isclose(best_distance, tol, rtol=1e-12, atol=1e-15):
            indices[i] = best
            matched[i] = True
            last = best
    return indices, matched


@pytest.mark.parametrize("seed", range(150))
def test_restricted_matching_matches_brute_force(seed):
    from neurale.data._alignment import _common_reference_coverage

    rng = np.random.default_rng(seed + 1000)
    # Both signals start within [0, 0.1] and span at least 1.0 s, so their
    # coverage always overlaps and common coverage is non-empty.
    neural, neural_time, neural_period = _regular_signal(rng, 0.1, 1.0)
    behavior, behavior_time, behavior_period = _regular_signal(rng, 0.1, 1.0)
    tol = float(rng.uniform(0.0, 4.0 * max(neural_period, behavior_period)))

    # Share the implementation's common-coverage computation so the randomized
    # comparison isolates the matching + two-sided coverage restriction.
    common_start, common_stop = _common_reference_coverage(neural, behavior)

    alignment = align_neural_behavior_nearest(neural, behavior, target="neural", tol=tol)

    expected_indices, expected_matched = _reference_align_matching(
        neural_time, behavior_time, tol, common_start, common_stop
    )

    np.testing.assert_array_equal(alignment.neural_indices, np.arange(neural_time.size))
    np.testing.assert_array_equal(alignment.behavior_indices, expected_indices)
    np.testing.assert_array_equal(alignment.matched, expected_matched)
