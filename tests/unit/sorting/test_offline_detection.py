#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from neurale.data import Clock, Event, EventSeries, SignalArray, SpikeWaveformBatch
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.runtime import runtime_context
from neurale.sorting import DetectionConfig, detect_threshold_spikes
from neurale.sorting._detection_reference import threshold_detection_reference

pytestmark = pytest.mark.native


def _signal(data: np.ndarray, *, t0: float = 2.0) -> SignalArray:
    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return SignalArray.from_array(
        values,
        fs=1_000.0,
        t0=t0,
        channel_names=[f"c{idx}" for idx in range(values.shape[1])],
        channel_types="spike",
        units="uV",
        name="wideband",
        clock=Clock("acquisition", "device", synchronization_domain="session"),
        attrs={"subject": "synthetic"},
    )


def _config(**changes) -> DetectionConfig:
    values = {
        "alignment_search_radius": 2,
        "pre_samples": 3,
        "post_samples": 4,
        "threshold_multiplier": 3.0,
        "refractory_interval": 0.004,
    }
    values.update(changes)
    return DetectionConfig(**values)


def _boundary_signal() -> SignalArray:
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 24)
    data = np.column_stack((base, base + 5.0)).astype(np.float64)
    # Crossing at 15 and aligned peak at 17 cross several tested chunk edges.
    data[15:19, 0] = [-5.0, -8.0, -12.0, -7.0]
    # One active excursion spans a chunk edge and must not be emitted twice.
    data[39:44, 1] = [-5.0, -8.0, -11.0, -9.0, -6.0]
    data[72, 0] = -10.0
    return _signal(data)


def _assert_batch_equal(actual: SpikeWaveformBatch, expected: SpikeWaveformBatch) -> None:
    np.testing.assert_array_equal(actual.waveforms, expected.waveforms)
    np.testing.assert_array_equal(actual.sample_indices, expected.sample_indices)
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_array_equal(actual.peak_channel_indices, expected.peak_channel_indices)
    np.testing.assert_array_equal(actual.electrode_group_ids, expected.electrode_group_ids)
    np.testing.assert_array_equal(actual.amps, expected.amps)
    if expected.spike_polarities is None:
        assert actual.spike_polarities is None
    else:
        np.testing.assert_array_equal(actual.spike_polarities, expected.spike_polarities)
    assert actual.channels is expected.channels
    assert actual.clock is expected.clock
    assert actual.fs == expected.fs
    assert actual.source_stream == expected.source_stream
    assert actual.segment_id == expected.segment_id


def test_native_waveforms_match_reference_and_keep_metadata() -> None:
    signal = _boundary_signal()
    config = _config(polarity="both")

    (actual,) = detect_threshold_spikes(signal, config, sample_idx_offset=10_000)
    expected = threshold_detection_reference(
        signal,
        config,
        sample_idx_offset=10_000,
    )

    _assert_batch_equal(actual, expected)
    assert actual.attrs["detection_method"] == "threshold_native_cpu"
    assert actual.attrs["segment_start_sample"] == 0
    assert actual.attrs["segment_stop_sample"] == signal.n_samples
    np.testing.assert_allclose(
        actual.attrs["channel_noise"], expected.attrs["channel_noise"], rtol=1e-15
    )
    assert isinstance(actual.waveforms.base, bytes)
    assert not actual.waveforms.flags.writeable
    with pytest.raises(ValueError):
        actual.waveforms.flags.writeable = True


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 10, 16, 40, 97])
def test_chunked_results_match_whole_signal(
    chunk_size: int,
) -> None:
    signal = _boundary_signal()
    config = _config(electrode_groups=((0,), (1,)))
    (whole,) = detect_threshold_spikes(signal, config)
    (chunked,) = detect_threshold_spikes(signal, config, chunk_size=chunk_size)

    _assert_batch_equal(chunked, whole)
    np.testing.assert_array_equal(chunked.sample_indices, [17, 41, 72])
    assert np.unique(chunked.sample_indices).size == len(chunked)
    for spike, sample in enumerate(chunked.sample_indices):
        local = int(sample)
        np.testing.assert_array_equal(
            chunked.waveforms[spike],
            signal.data[local - config.pre_samples : local + config.post_samples + 1],
        )


def test_absolute_indices_remain_exact_for_chunked_nonzero_offset() -> None:
    signal = _boundary_signal()
    (batch,) = detect_threshold_spikes(
        signal,
        _config(),
        chunk_size=13,
        sample_idx_offset=2**53 + 100,
    )

    np.testing.assert_array_equal(
        batch.sample_indices,
        np.asarray([17, 41, 72], dtype=np.int64) + np.int64(2**53 + 100),
    )
    np.testing.assert_array_equal(batch.times, signal.time[[17, 41, 72]])


def test_instant_discontinuity_prevents_cross_gap_waveforms() -> None:
    signal = _boundary_signal()
    data = np.array(signal.data, copy=True)
    data[47:50, 0] = [-6.0, -12.0, -8.0]
    data[50:53, 0] = [-7.0, -13.0, -6.0]
    signal = _signal(data)
    gap = EventSeries(
        [Event(onset=float(signal.time[50]), label="gap")],
        clock=signal.clock,
    )

    whole = detect_threshold_spikes(signal, _config(), discontinuities=gap)
    chunked = detect_threshold_spikes(signal, _config(), chunk_size=11, discontinuities=gap)

    assert len(whole) == len(chunked) == 2
    for expected_segment_id, (actual, expected) in enumerate(zip(chunked, whole, strict=True)):
        _assert_batch_equal(actual, expected)
        assert actual.segment_id == expected_segment_id
    assert all(np.all(batch.sample_indices < 47) for batch in whole[:1])
    assert all(np.all(batch.sample_indices >= 53) for batch in whole[1:])
    assert whole[0].attrs["segment_stop_sample"] == 50
    assert whole[1].attrs["segment_start_sample"] == 50
    assert whole[0].attrs["sample_index_offset"] == 0
    assert whole[1].attrs["sample_index_offset"] == 50

    left_signal = signal.replace(
        data=signal.data[:50],
        time=signal.time[:50],
        t0=float(signal.time[0]),
    )
    right_signal = signal.replace(
        data=signal.data[50:],
        time=signal.time[50:],
        t0=float(signal.time[50]),
    )
    expected_left = threshold_detection_reference(left_signal, _config(), segment_id=0)
    expected_right = threshold_detection_reference(
        right_signal,
        _config(),
        segment_id=1,
        sample_idx_offset=50,
    )
    _assert_batch_equal(whole[0], expected_left)
    _assert_batch_equal(whole[1], expected_right)


def test_duration_discontinuity_keeps_absolute_indices() -> None:
    signal = _boundary_signal()
    gap = EventSeries(
        [Event(onset=float(signal.time[32]), duration=0.010, label="missing")],
        clock=signal.clock,
    )

    batches = detect_threshold_spikes(signal, _config(), chunk_size=5, discontinuities=gap)

    assert len(batches) == 2
    assert batches[0].attrs["segment_stop_sample"] == 32
    assert batches[1].attrs["segment_start_sample"] == 42
    assert all(
        not np.any((batch.sample_indices >= 32) & (batch.sample_indices < 42)) for batch in batches
    )


def test_boundary_raise_is_based_on_segment_not_logical_chunk() -> None:
    signal = _boundary_signal()
    config = _config(boundary_behavior="raise")

    whole = detect_threshold_spikes(signal, config)
    chunked = detect_threshold_spikes(signal, config, chunk_size=4)
    _assert_batch_equal(chunked[0], whole[0])

    gap = EventSeries([Event(onset=float(signal.time[18]))], clock=signal.clock)
    with pytest.raises(ValidationError, match="crosses the input boundary"):
        detect_threshold_spikes(signal, config, chunk_size=4, discontinuities=gap)


def test_empty_signal_returns_one_typed_empty_batch_in_both_modes() -> None:
    signal = _signal(np.empty((0, 2), dtype=np.float64))
    config = _config()

    (whole,) = detect_threshold_spikes(signal, config)
    (chunked,) = detect_threshold_spikes(signal, config, chunk_size=3)

    _assert_batch_equal(chunked, whole)
    assert whole.waveforms.shape == (0, 8, 2)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, "4"])
def test_invalid_chunk_size_is_rejected(chunk_size) -> None:
    with pytest.raises(ValidationError, match="chunk_size"):
        detect_threshold_spikes(_boundary_signal(), _config(), chunk_size=chunk_size)


def test_native_offline_input_rejects_dtype_and_gaps() -> None:
    with pytest.raises(ValidationError, match="float64"):
        detect_threshold_spikes(
            _signal(np.zeros((20, 2), dtype=np.float32)),
            _config(),
        )

    source = np.zeros((20, 4), dtype=np.float64)[:, ::2]
    assert not source.flags.c_contiguous
    with pytest.raises(ValidationError, match="C-contiguous"):
        detect_threshold_spikes(_signal(source), _config())

    with pytest.raises(ValidationError, match="EventSeries"):
        detect_threshold_spikes(_boundary_signal(), _config(), discontinuities=())

    signal = _boundary_signal()
    unrelated = EventSeries(
        [Event(onset=2.02)],
        clock=Clock("other", "device", synchronization_domain="unrelated"),
    )
    with pytest.raises(ValidationError, match="sync_domain"):
        detect_threshold_spikes(signal, _config(), discontinuities=unrelated)


def test_explicit_cuda_request_is_rejected_without_cpu_fallback() -> None:
    # D1 contract: the threshold detector is CPU-only, so an explicit ``cuda``
    # request must raise ``DeviceUnavailableError`` at the API boundary rather
    # than silently running the CPU kernel. ``device="cpu"`` still works. The
    # ``supports_cuda=False`` path raises without probing CUDA, so no GPU is
    # required for this assertion.
    signal = _boundary_signal()
    config = _config()
    with runtime_context(device="cpu"):
        cpu_result = detect_threshold_spikes(signal, config)
    assert len(cpu_result) >= 1

    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match=r"sorting\.detection"):
            detect_threshold_spikes(signal, config)

    code = """
import sys
import numpy as np
from neurale.data import SignalArray
from neurale.exceptions import DeviceUnavailableError
from neurale.runtime import runtime_context
from neurale.sorting import DetectionConfig, detect_threshold_spikes

signal = SignalArray.from_array(
    np.zeros((8, 1), dtype=np.float64),
    fs=1_000.0,
    channel_names=["c0"],
    channel_types="spike",
    units="uV",
    name="wideband",
)
with runtime_context(device="cuda"):
    try:
        detect_threshold_spikes(
            signal,
            DetectionConfig(alignment_search_radius=0, pre_samples=0, post_samples=0),
        )
    except DeviceUnavailableError:
        pass
    else:
        raise AssertionError("explicit CUDA request silently executed the CPU detector")
assert "neurale._native" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_sorting_import_remains_native_and_cuda_lazy() -> None:
    code = """
import sys
from neurale.sorting import DetectionConfig, detect_threshold_spikes
assert DetectionConfig is not None
assert callable(detect_threshold_spikes)
assert "neurale._native" not in sys.modules
assert "neurale._native_cuda" not in sys.modules
assert "scipy" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
