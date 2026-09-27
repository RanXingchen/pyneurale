#!/usr/bin/env python3

import numpy as np
import pytest

from neurale.data import Clock, EventSeries, Frame, Recording, SignalArray
from neurale.exceptions import ValidationError


def test_recording_from_arrays_builds_neural_signal():
    recording = Recording.from_arrays(
        np.zeros((10, 2)),
        fs=1000.0,
        channel_names=["A-000", "A-001"],
        channel_type="ecog",
        unit="uV",
        subject={"id": "S01"},
        session={"paradigm": "centerout2d"},
    )

    signal = recording.signal("neural")
    assert signal.n_samples == 10
    assert signal.channel_names == ["A-000", "A-001"]
    assert recording.subject == {"id": "S01"}


def test_recording_from_arrays_preserves_per_channel_units():
    recording = Recording.from_arrays(
        np.zeros((10, 2)),
        fs=1000.0,
        unit=["uV", "count"],
    )

    signal = recording.signal()
    assert signal.unit == ("uV", "count")
    assert signal.channels.units == ["uV", "count"]


def test_recording_and_frame_reject_wrong_value_types():
    with pytest.raises(ValidationError):
        Recording(signals={"neural": object()})

    signal = Recording.from_arrays(np.zeros((2, 1)), fs=1.0).signal()
    frame = Frame(signals={"neural": signal}, events=EventSeries(), sequence=1, received_at=2.0)
    assert frame.sequence == 1

    with pytest.raises(ValidationError):
        Frame(signals={"neural": object()})


def test_clock_validation_and_signal_acceptance():
    clock = Clock(name="acq", type="device", rate=30000.0, offset=0.01)
    recording = Recording.from_arrays(np.zeros((2, 1)), fs=1.0)
    signal = recording.signal().copy()

    signal = SignalArray(
        data=signal.data,
        fs=signal.fs,
        time=signal.time,
        t0=signal.t0,
        clock=clock,
        channels=signal.channels,
        unit=signal.unit,
        name=signal.name,
    )

    assert signal.clock.name == "acq"


def test_recording_mapping_rejects_later_mutation():
    recording = Recording()

    with pytest.raises(ValidationError):
        recording.signals["broken"] = object()
