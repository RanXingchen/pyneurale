#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelTable, Clock, SpikeTrain, SpikeWaveformBatch
from neurale.exceptions import ValidationError
from neurale.sorting import waveform_batch_to_spike_train


def _batch() -> SpikeWaveformBatch:
    waveforms = np.arange(4 * 5 * 2, dtype=np.float64).reshape(4, 5, 2)
    return SpikeWaveformBatch(
        waveforms=waveforms,
        sample_indices=np.array([100, 110, 120, 130], dtype=np.int64),
        times=np.array([2.0, 2.01, 2.02, 2.03], dtype=np.float64),
        peak_channel_indices=np.array([0, 1, 0, 1], dtype=np.int64),
        electrode_group_ids=np.array([0, 0, 0, 0], dtype=np.int64),
        amps=np.array([-4.0, -3.0, -2.0, -1.0], dtype=np.float64),
        channels=ChannelTable.from_names(
            ["a", "b"],
            type="spike",
            unit="uV",
        ),
        fs=1_000.0,
        clock=Clock("acquisition", "device", synchronization_domain="session"),
        source_stream="wideband",
        segment_id="segment-a",
        pre_samples=2,
        post_samples=2,
        polarity="negative",
    )


def test_waveform_batch_conversion_preserves_linkage() -> None:
    batch = _batch()
    labels = np.array([42, -7, -1, 42], dtype=np.int64)
    labels_before = labels.copy()

    result = waveform_batch_to_spike_train(
        batch,
        labels,
        noise_label=-1,
        retain_waveforms=True,
    )

    assert isinstance(result, SpikeTrain)
    assert result.units == ["-7", "42"]
    np.testing.assert_array_equal(result.times[0], [2.01])
    np.testing.assert_array_equal(result.times[1], [2.0, 2.03])
    np.testing.assert_array_equal(result.attrs["cluster_labels"], [-7, 42])
    np.testing.assert_array_equal(result.attrs["source_event_indices"][0], [1])
    np.testing.assert_array_equal(result.attrs["source_event_indices"][1], [0, 3])
    np.testing.assert_array_equal(result.attrs["source_sample_indices"][1], [100, 130])
    np.testing.assert_array_equal(result.attrs["source_peak_channel_indices"][1], [0, 1])
    np.testing.assert_array_equal(result.attrs["noise_event_indices"], [2])
    assert result.attrs["source_clock"] is batch.clock
    assert result.attrs["source_channels"] is batch.channels
    assert result.attrs["source_segment_id"] == "segment-a"
    assert result.fs == 1_000.0
    assert result.waveforms is not None
    np.testing.assert_array_equal(result.waveforms[1], batch.waveforms[[0, 3]])
    np.testing.assert_array_equal(labels, labels_before)


def test_waveform_batch_conversion_makes_waveform_retention_explicit() -> None:
    result = waveform_batch_to_spike_train(
        _batch(),
        np.array([-1, -1, -1, -1], dtype=np.int64),
        noise_label=-1,
    )

    assert result.times == []
    assert result.units == []
    assert result.waveforms is None
    assert result.fs is None
    assert result.attrs["waveforms_retained"] is False
    np.testing.assert_array_equal(result.attrs["noise_event_indices"], [0, 1, 2, 3])


@pytest.mark.parametrize(
    "labels,message",
    [
        (np.array([1, 2, 3], dtype=np.int64), "length"),
        (np.array([1, 2, 3, 4], dtype=np.int32), "int64"),
        ([1, 2, 3, 4], "numpy.ndarray"),
    ],
)
def test_waveform_batch_conversion_rejects_invalid_event_labels(labels, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        waveform_batch_to_spike_train(_batch(), labels, noise_label=-1)


def test_waveform_batch_conversion_rejects_implicit_coercion() -> None:
    labels = np.array([1, 1, 1, 1], dtype=np.int64)
    with pytest.raises(ValidationError, match="noise_label"):
        waveform_batch_to_spike_train(_batch(), labels, noise_label=True)
    with pytest.raises(ValidationError, match="retain_waveforms"):
        waveform_batch_to_spike_train(
            _batch(),
            labels,
            noise_label=-1,
            retain_waveforms=1,
        )
