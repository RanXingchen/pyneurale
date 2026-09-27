#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import pickle
import subprocess
import sys

import numpy as np
import pytest

from neurale.data import ChannelTable, Clock, SignalArray, SpikeWaveformBatch
from neurale.exceptions import ValidationError
from neurale.sorting import DetectionConfig
from neurale.sorting._detection_reference import threshold_detection_reference

_MAD_NORMALIZER = 0.6744897501960817


def _signal(data, *, fs=1_000.0, t0=2.0, time=None) -> SignalArray:
    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return SignalArray.from_array(
        values,
        fs=fs,
        time=time,
        t0=t0,
        channel_names=[f"c{idx}" for idx in range(values.shape[1])],
        channel_types="spike",
        units="uV",
        name="wideband",
        clock=Clock("acquisition", "device", synchronization_domain="session"),
        attrs={"session": "synthetic"},
    )


def _config(**changes) -> DetectionConfig:
    values = {
        "alignment_search_radius": 2,
        "pre_samples": 2,
        "post_samples": 2,
        "threshold_multiplier": 3.0,
        "refractory_interval": 0.004,
    }
    values.update(changes)
    return DetectionConfig(**values)


def test_config_is_frozen_normalized_with_defaults() -> None:
    config = DetectionConfig(
        alignment_search_radius=np.int64(4),
        pre_samples=8,
        post_samples=12,
        electrode_groups=[[0, 1], [2]],
    )

    assert config.noise_estimator == "mad"
    assert config.threshold_multiplier == 3.5
    assert config.polarity == "negative"
    assert config.refractory_interval == 1.5e-3
    assert config.electrode_groups == ((0, 1), (2,))
    assert config.boundary_behavior == "drop"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.pre_samples = 9


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"alignment_search_radius": -1}, "alignment_search_radius"),
        ({"pre_samples": 1.5}, "pre_samples"),
        ({"post_samples": True}, "post_samples"),
        ({"noise_estimator": "rms"}, "noise_estimator"),
        ({"threshold_multiplier": 0.0}, "threshold_multiplier"),
        ({"threshold_multiplier": np.inf}, "threshold_multiplier"),
        ({"polarity": "neg"}, "polarity"),
        ({"refractory_interval": -0.1}, "refractory_interval"),
        ({"electrode_groups": []}, "must not be empty"),
        ({"electrode_groups": [[0], []]}, "must not be empty"),
        ({"electrode_groups": [[0, 0]]}, "duplicate"),
        ({"electrode_groups": [[0, 1], [1, 2]]}, "multiple"),
        ({"boundary_behavior": "pad"}, "boundary_behavior"),
    ],
)
def test_config_rejects_ambiguous_or_invalid_contracts(changes, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**changes)


def test_negative_detection_matches_direct_mad_threshold() -> None:
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 8)
    data[10:13] = [-3.0, -8.0, -4.0]
    source = _signal(data)
    config = _config()

    batch = threshold_detection_reference(
        source,
        config,
        segment_id="segment-a",
        sample_idx_offset=10_000,
    )

    center = np.median(data)
    expected_noise = np.median(np.abs(data - center)) / _MAD_NORMALIZER
    assert isinstance(batch, SpikeWaveformBatch)
    np.testing.assert_array_equal(batch.sample_indices, [10_011])
    np.testing.assert_allclose(batch.times, [2.011])
    np.testing.assert_array_equal(batch.peak_channel_indices, [0])
    np.testing.assert_array_equal(batch.electrode_group_ids, [0])
    np.testing.assert_array_equal(batch.amps, [-8.0])
    np.testing.assert_array_equal(batch.waveforms[0, :, 0], data[9:14])
    assert batch.waveforms.shape == (1, 5, 1)
    assert batch.channels is source.channels
    assert batch.clock is source.clock
    assert batch.source_stream == "wideband"
    assert batch.segment_id == "segment-a"
    assert batch.attrs["detection_method"] == "threshold_reference"
    np.testing.assert_allclose(batch.attrs["channel_noise"], [expected_noise])
    np.testing.assert_allclose(batch.attrs["channel_thresholds"], [3.0 * expected_noise])
    assert batch.attrs["sample_index_offset"] == 10_000


def test_detection_is_offset_invariant_and_keeps_dtype() -> None:
    data = np.tile(np.asarray([-1, 0, 1, 0], dtype=np.int16), 8)
    data[11] = -20

    original = threshold_detection_reference(_signal(data), _config())
    shifted = threshold_detection_reference(_signal(data + 100), _config())

    np.testing.assert_array_equal(original.sample_indices, shifted.sample_indices)
    np.testing.assert_array_equal(original.waveforms + 100, shifted.waveforms)
    assert original.waveforms.dtype == np.int16
    assert shifted.waveforms.dtype == np.int16


@pytest.mark.parametrize(
    ("offset", "expect_invariant"),
    [(0, True), (1_000_000, True), (2**53, False)],
)
def test_location_invariance_or_rejects_unsafe_integer_geometry(offset, expect_invariant) -> None:
    # The centered-MAD geometry is documented as location invariant, but that
    # only holds while the integer waveform is exactly representable as
    # float64. Past the 2**53 precision limit the small oscillations collapse
    # onto the DC offset and the same waveform silently returns an empty batch.
    # Safe offsets must produce the identical result; an unsafe offset must
    # raise rather than return a wrong empty batch.
    base = np.tile(np.array([-1, 0, 1, 0], dtype=np.int64), 8)
    base[11] = -20
    data = base + np.int64(offset)

    if expect_invariant:
        batch = threshold_detection_reference(_signal(data), _config())
        np.testing.assert_array_equal(batch.sample_indices, [11])
    else:
        with pytest.raises(ValidationError, match="exactly representable as float64"):
            threshold_detection_reference(_signal(data), _config())


@pytest.mark.parametrize(
    ("polarity", "expected_idx", "expected_amp"),
    [
        ("negative", 9, -9.0),
        ("positive", 21, 10.0),
    ],
)
def test_signed_polarity_selects_requested_excursion(
    polarity,
    expected_idx,
    expected_amp,
) -> None:
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 8)
    data[9] = -9.0
    data[21] = 10.0

    batch = threshold_detection_reference(
        _signal(data),
        _config(polarity=polarity, alignment_search_radius=1),
    )

    np.testing.assert_array_equal(batch.sample_indices, [expected_idx])
    np.testing.assert_array_equal(batch.amps, [expected_amp])
    assert batch.spike_polarities is None


def test_amplitudes_are_centered_polarity_matched_excursions() -> None:
    # ``amplitudes`` are baseline-centered detection excursions -- the same
    # centered signal used for thresholding and polarity -- so their sign matches
    # ``spike_polarities`` and they are invariant to a DC offset. With a non-zero
    # baseline a negative spike has a positive raw value; the amplitudes field
    # records the centered excursion, and the raw value stays recoverable from
    # the waveform.
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 8)
    base[11] = -8.0
    config = _config(polarity="both", alignment_search_radius=2)

    zero = threshold_detection_reference(_signal(base), config)
    shifted = threshold_detection_reference(_signal(base + 100.0), config)

    for batch in (zero, shifted):
        np.testing.assert_array_equal(batch.sample_indices, [11])
        np.testing.assert_array_equal(batch.amps, [-8.0])
        np.testing.assert_array_equal(batch.spike_polarities, [-1])

    # The raw aligned sample on the peak channel survives in the waveform.
    np.testing.assert_array_equal(zero.waveforms[0, zero.pre_samples, 0], -8.0)
    np.testing.assert_array_equal(shifted.waveforms[0, shifted.pre_samples, 0], 92.0)


def test_both_polarity_records_sign_with_inclusive_crossing() -> None:
    data = np.tile(np.asarray([-1.0, 1.0]), 20)
    noise = np.median(np.abs(data - np.median(data))) / _MAD_NORMALIZER
    threshold = 3.0 * noise
    data[8] = -threshold
    data[29] = threshold

    batch = threshold_detection_reference(
        _signal(data),
        _config(polarity="both", alignment_search_radius=0),
    )

    np.testing.assert_array_equal(batch.sample_indices, [8, 29])
    np.testing.assert_array_equal(batch.spike_polarities, [-1, 1])


def test_grouping_picks_strongest_channel_then_refractory() -> None:
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 10)
    data = np.column_stack((base, base, base))
    data[10, 0] = -7.0
    data[10, 1] = -10.0
    data[12, 0] = -12.0
    data[20, 2] = -8.0
    config = _config(
        alignment_search_radius=0,
        refractory_interval=0.004,
        electrode_groups=((0, 1), (2,)),
    )

    batch = threshold_detection_reference(_signal(data), config)

    np.testing.assert_array_equal(batch.sample_indices, [10, 20])
    np.testing.assert_array_equal(batch.peak_channel_indices, [1, 2])
    np.testing.assert_array_equal(batch.electrode_group_ids, [0, 1])
    assert batch.waveforms.shape == (2, 5, 3)


def test_result_preserves_resolved_electrode_group_definition() -> None:
    # ``electrode_group_ids`` alone cannot recover group membership: a
    # ``SpikeWaveformBatch`` is a cross-cutting contract for sorting, I/O, and
    # visualization, so the resolved electrode-group definition must travel with
    # it as self-contained metadata. Even ``electrode_groups=None`` resolves to
    # one singleton group per channel so group IDs are always interpretable.
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 10)
    data = np.column_stack((base, base, base))
    data[10, 0] = -7.0
    data[20, 2] = -8.0
    signal = _signal(data)

    config = _config(
        alignment_search_radius=0,
        refractory_interval=0.004,
        electrode_groups=((0, 2), (1,)),
    )
    batch = threshold_detection_reference(signal, config)
    assert batch.attrs["electrode_groups"] == ((0, 2), (1,))

    default_batch = threshold_detection_reference(signal, _config())
    assert default_batch.attrs["electrode_groups"] == ((0,), (1,), (2,))

    # The metadata is immutable and survives pickle, slicing, and concatenation.
    restored = pickle.loads(pickle.dumps(batch))
    assert restored.attrs["electrode_groups"] == ((0, 2), (1,))

    sliced = batch[:1]
    assert sliced.attrs["electrode_groups"] == ((0, 2), (1,))

    concatenated = SpikeWaveformBatch.concatenate([batch[:1], batch[1:]])
    assert concatenated.attrs["electrode_groups"] == ((0, 2), (1,))


def test_refractory_uses_ceil_and_allows_boundary_event() -> None:
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 10)
    data[[8, 12]] = [-8.0, -10.0]

    batch = threshold_detection_reference(
        _signal(data, fs=1_000.0),
        _config(alignment_search_radius=0, refractory_interval=0.0031),
    )

    assert batch.attrs["refractory_samples"] == 4
    np.testing.assert_array_equal(batch.sample_indices, [8, 12])


def test_group_refractory_uses_crossings_before_peak_alignment() -> None:
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 10)
    data = np.column_stack((base, base))
    data[8, 0] = -9.0
    data[11:14, 1] = [-5.0, -7.0, -12.0]

    batch = threshold_detection_reference(
        _signal(data),
        _config(
            alignment_search_radius=2,
            refractory_interval=0.004,
            electrode_groups=((0, 1),),
        ),
    )

    # Crossings 8 and 11 are inside the four-sample refractory interval, so
    # the second is suppressed before its later peak at sample 13 is aligned.
    np.testing.assert_array_equal(batch.sample_indices, [8])
    np.testing.assert_array_equal(batch.peak_channel_indices, [0])


def test_alignment_ties_choose_earliest_extremum() -> None:
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 8)
    data[10:13] = [-5.0, -8.0, -8.0]

    batch = threshold_detection_reference(_signal(data), _config())

    np.testing.assert_array_equal(batch.sample_indices, [11])


def test_boundary_drop_discards_events_and_raise_reports_them() -> None:
    data = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 6)
    data[0] = -9.0

    dropped = threshold_detection_reference(
        _signal(data),
        _config(alignment_search_radius=0, boundary_behavior="drop"),
    )
    assert dropped.waveforms.shape == (0, 5, 1)

    with pytest.raises(ValidationError, match="crosses the input boundary"):
        threshold_detection_reference(
            _signal(data),
            _config(alignment_search_radius=0, boundary_behavior="raise"),
        )


def test_constant_and_empty_signals_return_typed_empty_batches() -> None:
    for data in (np.ones(20), np.empty(0)):
        batch = threshold_detection_reference(_signal(data), _config())
        assert batch.waveforms.shape == (0, 5, 1)
        assert batch.sample_indices.dtype == np.int64
        assert batch.times.dtype == np.float64


def test_explicit_groups_must_partition_current_signal_channels() -> None:
    source = _signal(np.zeros((20, 3)))
    with pytest.raises(ValidationError, match="cover every"):
        threshold_detection_reference(
            source,
            _config(electrode_groups=((0, 1),)),
        )
    with pytest.raises(ValidationError, match="outside"):
        threshold_detection_reference(
            source,
            _config(electrode_groups=((0, 1), (2, 3))),
        )


def test_reference_rejects_irregular_or_untyped_input() -> None:
    irregular = _signal(
        np.zeros(4),
        time=np.asarray([1.0, 1.001, 1.003, 1.004]),
    )
    with pytest.raises(ValidationError, match="regularly sampled"):
        threshold_detection_reference(irregular, _config())

    nonfinite = _signal(np.asarray([0.0, np.nan, 0.0]))
    with pytest.raises(ValidationError, match="finite"):
        threshold_detection_reference(nonfinite, _config())
    with pytest.raises(ValidationError, match="SignalArray"):
        threshold_detection_reference(np.zeros((10, 1)), _config())
    with pytest.raises(ValidationError, match="DetectionConfig"):
        threshold_detection_reference(_signal(np.zeros(10)), object())

    channel_less = SignalArray(
        data=np.empty((10, 0)),
        fs=1_000.0,
        time=None,
        t0=0.0,
        clock=None,
        channels=ChannelTable(),
        unit=(),
        name="empty-channels",
    )
    with pytest.raises(ValidationError, match="at least one channel"):
        threshold_detection_reference(channel_less, _config())

    extreme = _signal(np.asarray([-np.finfo(float).max, np.finfo(float).max]))
    with pytest.raises(ValidationError, match="threshold geometry"):
        threshold_detection_reference(extreme, _config(pre_samples=0, post_samples=0))


@pytest.mark.parametrize("offset", [-1, 1.5, True, np.iinfo(np.int64).max])
def test_sample_index_offset_preserves_int64_indices(offset) -> None:
    with pytest.raises(ValidationError, match="sample_index_offset"):
        threshold_detection_reference(
            _signal(np.zeros(4)),
            _config(),
            sample_idx_offset=offset,
        )


def test_sorting_import_does_not_publish_reference_oracle() -> None:
    code = """
import sys
import neurale.sorting as sorting
assert sorting.__all__ == [
    "ClusterIsiMetrics",
    "CurationOperationRecord",
    "CurationResult",
    "DetectionConfig",
    "MatchingEventsResult",
    "OfflineSortingResult",
    "OnlineDetectorDescriptor",
    "ValleySeekingResult",
    "WaveformOutlierResult",
    "WaveformProjector",
    "detect_threshold_spikes",
    "isi_metrics",
    "matching_events",
    "merge_clusters",
    "project_waveform_features",
    "reject_waveform_outliers",
    "remove_tiny_clusters",
    "run_offline_sorting",
    "spike_block_to_waveform_batch",
    "split_cluster",
    "valley_seeking",
    "waveform_batch_to_spike_train",
]
assert not hasattr(sorting, "threshold_detection_reference")
assert not hasattr(sorting, "valley_seeking_reference")
assert not hasattr(sorting, "_valley_seeking")
assert callable(sorting.WaveformProjector)
assert callable(sorting.detect_threshold_spikes)
assert callable(sorting.isi_metrics)
assert callable(sorting.matching_events)
assert callable(sorting.merge_clusters)
assert callable(sorting.project_waveform_features)
assert callable(sorting.reject_waveform_outliers)
assert callable(sorting.remove_tiny_clusters)
assert callable(sorting.run_offline_sorting)
assert callable(sorting.split_cluster)
assert callable(sorting.spike_block_to_waveform_batch)
assert callable(sorting.valley_seeking)
assert callable(sorting.waveform_batch_to_spike_train)
assert "neurale._native" not in sys.modules
assert "scipy" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
