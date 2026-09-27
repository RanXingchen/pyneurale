#!/usr/bin/env python3

import numpy as np
import pytest

from neurale.data import Event, EventSeries, Trial, TrialTable
from neurale.exceptions import ValidationError


def test_event_series_from_ttl_detects_edges_in_time_order():
    ttl = np.array([0, 0, 1, 1, 0, 1], dtype=float)

    events = EventSeries.from_ttl(ttl, fs=1000.0, t0=1.0)

    assert [event.label for event in events] == [
        "ttl_rising",
        "ttl_falling",
        "ttl_rising",
    ]
    np.testing.assert_allclose(events.onsets, [1.002, 1.004, 1.005])
    assert [event.sample_index for event in events] == [2, 4, 5]


def test_event_series_from_ttl_supports_hysteresis_and_debounce():
    ttl = np.array([0.0, 0.8, 0.2, 0.8, 0.9, 0.85, 0.5, 0.1, 0.05, 0.8, 0.9])
    events = EventSeries.from_ttl(
        ttl,
        fs=1000.0,
        low_threshold=0.2,
        high_threshold=0.7,
        debounce=0.002,
    )

    assert [event.label for event in events] == ["ttl_rising", "ttl_falling", "ttl_rising"]
    assert [event.sample_index for event in events] == [3, 7, 9]


def test_event_series_from_ttl_initial_state_and_validation():
    events = EventSeries.from_ttl(
        np.array([1.0, 1.0, 0.0, 0.0]),
        fs=1000.0,
        initial_high=True,
    )
    assert [event.label for event in events] == ["ttl_falling"]
    assert events[0].sample_index == 2
    with pytest.raises(ValidationError, match="low_threshold"):
        EventSeries.from_ttl(
            np.ones(4),
            fs=1000.0,
            low_threshold=0.8,
            high_threshold=0.2,
        )


def test_event_and_trial_validation():
    event = Event(onset=1.0, duration=0.5, label="cue")
    assert event.stop == 1.5
    assert event.is_interval

    with pytest.raises(ValidationError):
        Event(onset=-1.0)

    trials = TrialTable([Trial(trial_id=0, start=0.0, stop=1.0, target_id=2)])
    assert trials[0].duration == 1.0

    with pytest.raises(ValidationError):
        Trial(trial_id=1, start=2.0, stop=1.0)


def test_event_series_requires_ordered_unique_events():
    first = Event(0.1, label="cue")
    second = Event(0.2, label="go")

    with pytest.raises(ValidationError, match="ordered"):
        EventSeries([second, first])
    with pytest.raises(ValidationError, match="duplicates"):
        EventSeries([first, first])
