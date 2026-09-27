#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pickle

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, Clock, SpikeWaveformBatch
from neurale.exceptions import ValidationError
from neurale.sorting import (
    OnlineDetectorDescriptor,
    spike_block_to_waveform_batch,
)
from neurale.sorting import (
    online as online_module,
)


def _decoded(group_ids=(0, 0)) -> dict:
    return {
        "waveforms": np.arange(12, dtype=np.float64).reshape(2, 3, 2),
        "sample_indices": np.array([100, 105], dtype=np.int64),
        "times": np.array([1.0, 1.005], dtype=np.float64),
        "peak_channel_indices": np.array([0, 1], dtype=np.uint32),
        "electrode_group_ids": np.array(group_ids, dtype=np.uint32),
        "amps": np.array([-4.0, 5.0]),
        "scores": np.array([4.0, 5.0]),
        "spike_polarities": np.array([-1, 1], dtype=np.int8),
        "segment_id": 7,
        "pre_samples": 1,
        "post_samples": 1,
        "polarity": "both",
        "capacity": 8,
        "overflow_count": 3,
        "overflowed": True,
    }


def _channels() -> ChannelTable:
    return ChannelTable(
        (
            ChannelInfo("a", 0, "spike", "uV"),
            ChannelInfo("b", 1, "spike", "uV"),
        )
    )


def test_spike_block_conversion_builds_immutable_offline_batch(monkeypatch) -> None:
    monkeypatch.setattr(online_module, "_decode_native_block", lambda payload: _decoded())
    channels = _channels()
    clock = Clock("device", "device", rate=1_000.0)
    descriptor = OnlineDetectorDescriptor(electrode_groups=((0, 1),))

    batch = spike_block_to_waveform_batch(
        b"opaque",
        channels=channels,
        fs=1_000.0,
        clock=clock,
        source_stream="wideband",
        descriptor=descriptor,
    )

    assert batch.waveforms.shape == (2, 3, 2)
    assert batch.segment_id == 7
    assert batch.polarity == "both"
    np.testing.assert_array_equal(batch.spike_polarities, [-1, 1])
    assert batch.attrs["overflow_count"] == 3
    assert batch.attrs["overflowed"] is True
    assert batch.attrs["electrode_groups"] == ((0, 1),)
    assert not batch.waveforms.flags.writeable


def test_conversion_requires_descriptor_and_validates_partition(monkeypatch) -> None:
    monkeypatch.setattr(online_module, "_decode_native_block", lambda payload: _decoded())
    channels = _channels()

    with pytest.raises(ValidationError, match="descriptor must be an OnlineDetectorDescriptor"):
        spike_block_to_waveform_batch(
            b"opaque",
            channels=channels,
            fs=1_000.0,
            clock=None,
            source_stream="wideband",
            descriptor=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="cover every channel exactly once"):
        spike_block_to_waveform_batch(
            b"opaque",
            channels=channels,
            fs=1_000.0,
            clock=None,
            source_stream="wideband",
            descriptor=OnlineDetectorDescriptor(electrode_groups=((0,),)),
        )
    with pytest.raises(ValidationError, match="outside the ChannelTable"):
        spike_block_to_waveform_batch(
            b"opaque",
            channels=channels,
            fs=1_000.0,
            clock=None,
            source_stream="wideband",
            descriptor=OnlineDetectorDescriptor(electrode_groups=((0,), (1,), (2,))),
        )


def test_conversion_rejects_block_group_ids_without_definition(monkeypatch) -> None:
    # group id 2 has no definition under a 2-channel partition into one group.
    monkeypatch.setattr(online_module, "_decode_native_block", lambda payload: _decoded((0, 2)))
    channels = _channels()
    with pytest.raises(ValidationError, match="no defined group"):
        spike_block_to_waveform_batch(
            b"opaque",
            channels=channels,
            fs=1_000.0,
            clock=None,
            source_stream="wideband",
            descriptor=OnlineDetectorDescriptor(electrode_groups=((0, 1),)),
        )


def test_conversion_carries_provenance_through_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(online_module, "_decode_native_block", lambda payload: _decoded())
    channels = _channels()
    descriptor = OnlineDetectorDescriptor(
        electrode_groups=((0,), (1,)),
        channel_centers=(0.0, 0.5),
        channel_thresholds=(1.0, 2.0),
        refractory_samples=4,
        alignment_search_radius=2,
        boundary_behavior="drop",
        overflow_policy="fault",
    )

    batch = spike_block_to_waveform_batch(
        b"opaque",
        channels=channels,
        fs=1_000.0,
        clock=None,
        source_stream="wideband",
        descriptor=descriptor,
    )

    assert batch.attrs["electrode_groups"] == ((0,), (1,))
    np.testing.assert_allclose(batch.attrs["channel_centers"], [0.0, 0.5])
    np.testing.assert_allclose(batch.attrs["channel_thresholds"], [1.0, 2.0])
    assert batch.attrs["refractory_samples"] == 4
    assert batch.attrs["alignment_search_radius"] == 2
    assert batch.attrs["boundary_behavior"] == "drop"
    assert batch.attrs["overflow_policy"] == "fault"
    assert not batch.attrs["channel_centers"].flags.writeable

    # The electrode-group definition is immutable metadata and survives pickle,
    # slicing, and concatenation -- the cross-cutting contract the offline batch
    # already enforces for the reference path.
    restored = pickle.loads(pickle.dumps(batch))
    assert restored.attrs["electrode_groups"] == ((0,), (1,))
    sliced = batch[:1]
    assert sliced.attrs["electrode_groups"] == ((0,), (1,))
    concatenated = SpikeWaveformBatch.concatenate([batch[:1], batch[1:]])
    assert concatenated.attrs["electrode_groups"] == ((0,), (1,))


def test_online_detector_descriptor_is_frozen_and_validates() -> None:
    import dataclasses

    descriptor = OnlineDetectorDescriptor(
        electrode_groups=[[0, 1], [2]],
    )
    assert descriptor.electrode_groups == ((0, 1), (2,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.electrode_groups = ((0,),)  # type: ignore[misc]
    with pytest.raises(ValidationError, match="must be provided"):
        OnlineDetectorDescriptor(electrode_groups=None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="must not be empty"):
        OnlineDetectorDescriptor(electrode_groups=[])
    with pytest.raises(ValidationError, match="boundary_behavior"):
        OnlineDetectorDescriptor(electrode_groups=((0,),), boundary_behavior="pad")
    with pytest.raises(ValidationError, match="overflow_policy"):
        OnlineDetectorDescriptor(electrode_groups=((0,),), overflow_policy="ignore")
    with pytest.raises(ValidationError, match="equal length"):
        OnlineDetectorDescriptor(
            electrode_groups=((0,),),
            channel_centers=(0.0, 1.0),
            channel_thresholds=(1.0,),
        )
