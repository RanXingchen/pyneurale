#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock, SignalArray
from neurale.sorting import DetectionConfig
from neurale.sorting._detection_reference import threshold_detection_reference
from neurale.sorting._dispatch import load_cpu_sorting_operation

pytestmark = pytest.mark.native


def _signal(data: np.ndarray, *, fs: float = 1_000.0) -> SignalArray:
    return SignalArray.from_array(
        np.asarray(data, dtype=np.float64),
        fs=fs,
        t0=2.0,
        channel_names=[f"c{idx}" for idx in range(data.shape[1])],
        channel_types="spike",
        units="uV",
        name="wideband",
        clock=Clock("acquisition", "device"),
    )


def _native(signal: SignalArray, config: DetectionConfig) -> dict[str, np.ndarray]:
    refractory_samples = int(np.ceil(config.refractory_interval * signal.fs))
    groups = None if config.electrode_groups is None else [list(g) for g in config.electrode_groups]
    return load_cpu_sorting_operation("detection")._detect_threshold(
        np.asarray(signal.data),
        config.threshold_multiplier,
        refractory_samples,
        config.alignment_search_radius,
        config.pre_samples,
        config.post_samples,
        config.polarity,
        config.boundary_behavior,
        groups,
    )


@pytest.mark.parametrize("polarity", ["negative", "positive", "both"])
def test_native_detector_matches_reference_events_and_noise_geometry(polarity: str) -> None:
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 16)
    data = np.column_stack((base, base + 4.0, base - 2.0))
    data[10:13, 0] = [-3.0, -9.0, -4.0]
    data[24:27, 1] = [7.0, 12.0, 8.0]
    data[40, 2] = -10.0
    signal = _signal(data)
    config = DetectionConfig(
        alignment_search_radius=2,
        pre_samples=2,
        post_samples=2,
        threshold_multiplier=3.0,
        polarity=polarity,
        refractory_interval=0.004,
        electrode_groups=((0, 1), (2,)),
    )

    expected = threshold_detection_reference(signal, config)
    actual = _native(signal, config)

    np.testing.assert_array_equal(actual["sample_indices"], expected.sample_indices)
    assert actual["crossing_indices"].dtype == np.int64
    np.testing.assert_array_equal(actual["peak_channel_indices"], expected.peak_channel_indices)
    np.testing.assert_array_equal(actual["electrode_group_ids"], expected.electrode_group_ids)
    np.testing.assert_allclose(actual["amps"], expected.amps, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        actual["channel_centers"], expected.attrs["channel_centers"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        actual["channel_noise"], expected.attrs["channel_noise"], rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        actual["channel_thresholds"],
        expected.attrs["channel_thresholds"],
        rtol=1e-15,
        atol=0.0,
    )
    expected_polarities = np.sign(expected.amps).astype(np.int8)
    np.testing.assert_array_equal(actual["polarities"], expected_polarities)
    np.testing.assert_allclose(
        actual["scores"],
        np.abs(expected.amps) / actual["channel_thresholds"][expected.peak_channel_indices],
        rtol=1e-15,
        atol=0.0,
    )


def test_native_detector_separates_crossing_and_peak_index() -> None:
    # ``crossing_indices`` is documented native event metadata: the first sample
    # of a threshold excursion, which need not equal the aligned peak. The
    # generic parity test only checked its dtype, so a wrong crossing value
    # would pass unnoticed. This independent case pins a known geometry -- the
    # reference ``SpikeWaveformBatch`` does not store the crossing index, so the
    # assertion is against the math directly rather than the batch. With
    # threshold ~4.4478, sample 10 (-5) is the first sample to cross and
    # sample 11 (-9) is the deeper aligned peak, so crossing == 10 and
    # sample_index == 11.
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 8).astype(np.float64)[:, np.newaxis]
    data[10, 0] = -5.0
    data[11, 0] = -9.0
    signal = _signal(data)
    config = DetectionConfig(
        alignment_search_radius=2,
        pre_samples=2,
        post_samples=2,
        threshold_multiplier=3.0,
        polarity="negative",
        refractory_interval=0.004,
    )

    actual = _native(signal, config)

    np.testing.assert_array_equal(actual["crossing_indices"], [10])
    np.testing.assert_array_equal(actual["sample_indices"], [11])
    np.testing.assert_array_equal(actual["amps"], [-9.0])


def test_native_detector_handles_empty_and_constant_input() -> None:
    config = DetectionConfig(alignment_search_radius=0, pre_samples=0, post_samples=0)
    for data in (np.empty((0, 2), dtype=np.float64), np.ones((20, 2))):
        signal = _signal(data)
        first = _native(signal, config)
        second = _native(signal, config)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        assert first["sample_indices"].shape == (0,)
        np.testing.assert_array_equal(first["channel_noise"], [0.0, 0.0])


def test_native_binding_rejects_dtype_layout_and_values() -> None:
    native = load_cpu_sorting_operation("detection")._detect_threshold
    arguments = (3.0, 1, 0, 0, 0, "negative", "drop", None)
    with pytest.raises(TypeError, match="float64"):
        native(np.zeros((10, 2), dtype=np.float32), *arguments)
    with pytest.raises(ValueError, match="C-contiguous"):
        native(np.zeros((10, 4), dtype=np.float64)[:, ::2], *arguments)
    invalid = np.zeros((10, 2), dtype=np.float64)
    invalid[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        native(invalid, *arguments)


def test_native_binding_matches_reference_boundary_and_group_validation() -> None:
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 6)[:, np.newaxis]
    data[0, 0] = -9.0
    signal = _signal(data)
    dropped = DetectionConfig(
        alignment_search_radius=0,
        pre_samples=2,
        post_samples=2,
        boundary_behavior="drop",
    )
    assert _native(signal, dropped)["sample_indices"].size == 0

    raised = DetectionConfig(
        alignment_search_radius=0,
        pre_samples=2,
        post_samples=2,
        boundary_behavior="raise",
    )
    with pytest.raises(ValueError, match="crosses the input boundary"):
        _native(signal, raised)

    native = load_cpu_sorting_operation("detection")._detect_threshold
    with pytest.raises(ValueError, match="cover every"):
        native(
            np.zeros((10, 2), dtype=np.float64),
            3.5,
            1,
            0,
            0,
            0,
            "negative",
            "drop",
            [[0]],
        )


def test_private_waveform_extraction_stage_matches_fused_binding() -> None:
    data = np.zeros((40, 2), dtype=np.float64)
    data[:, 0] = np.tile(np.array([-1.0, 0.0, 1.0, 0.0]), 10)
    data[:, 1] = data[:, 0]
    data[10, 0] = -10.0
    data[25, 1] = -9.0
    native = load_cpu_sorting_operation("detection")
    arguments = (3.0, 1, 0, 2, 2, "negative", "drop", None)

    events = native._detect_threshold(data, *arguments)
    extracted = native._extract_threshold_waveforms(
        data,
        events["sample_indices"],
        2,
        2,
    )
    fused = native._detect_threshold_waveforms([data], *arguments)

    np.testing.assert_array_equal(extracted, fused["waveforms"])
    assert extracted.shape == (len(events["sample_indices"]), 5, 2)
    assert isinstance(extracted.base, bytes)
    assert not extracted.flags.writeable


def test_waveform_extraction_rejects_dtype_and_indices() -> None:
    native = load_cpu_sorting_operation("detection")
    data = np.zeros((8, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="int64"):
        native._extract_threshold_waveforms(data, np.array([3], dtype=np.int32), 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        native._extract_threshold_waveforms(data, np.array([-1], dtype=np.int64), 1, 1)
    with pytest.raises(ValueError, match="C-contiguous"):
        native._extract_threshold_waveforms(
            data,
            np.array([1, 2, 3, 4], dtype=np.int64)[::2],
            1,
            1,
        )
