#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _subprocess_probe import probe_json

import neurale.features.tuning as tuning_module
from neurale.data import (
    Clock,
    EventSeries,
    FeatureMatrix,
    SignalArray,
    SpikeTrain,
    Trial,
    TrialTable,
)
from neurale.exceptions import ValidationError
from neurale.features.tuning import (
    DirectionalTuningResult,
    TuningResult,
    binned_tuning,
    directional_tuning,
)


def _features(
    data,
    time,
    *,
    names: list[str] | None = None,
    units: str | list[str] = "Hz",
    attrs: dict[str, object] | None = None,
    window_size: float | None = 0.1,
    shift: float | None = 0.1,
) -> FeatureMatrix:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    observation_time = np.asarray(time, dtype=float)
    fs: float | None = 10.0
    if observation_time.size >= 2:
        intervals = np.diff(observation_time)
        fs = (
            1.0 / float(intervals[0])
            if intervals[0] > 0.0 and np.allclose(intervals, intervals[0], rtol=1e-7, atol=1e-12)
            else None
        )
    return FeatureMatrix(
        data=values,
        fs=fs,
        time=observation_time,
        feature_names=names or [f"unit{idx}" for idx in range(values.shape[1])],
        source_signal="motor-cortex",
        window_size=window_size,
        shift=shift,
        unit=units,
        attrs={"session": "synthetic"} if attrs is None else attrs,
    )


def _behavior(
    data,
    time,
    *,
    names: list[str] | None = None,
    units: str | list[str] = "a.u.",
    clock: Clock | None = None,
    attrs: dict[str, object] | None = None,
) -> SignalArray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return SignalArray.from_array(
        values,
        fs=10.0,
        time=np.asarray(time, dtype=float),
        channel_names=names or [f"behavior{idx}" for idx in range(values.shape[1])],
        channel_types="behavior",
        units=units,
        name="behavior",
        clock=clock,
        attrs={"task": "reach"} if attrs is None else attrs,
        copy_data=True,
    )


def test_binned_tuning_rejects_irregular_feature_observation_rate() -> None:
    neural = FeatureMatrix(
        data=np.ones((3, 1)),
        fs=None,
        time=np.array([0.0, 0.1, 0.35]),
        feature_names=["unit0"],
        source_signal="motor-cortex",
        unit="Hz",
    )
    behavior = _behavior([0.0, 0.5, 1.0], [0.0, 0.1, 0.35])

    with pytest.raises(ValidationError, match="regular FeatureMatrix sampling rate"):
        binned_tuning(
            neural,
            behavior,
            bin_edges=[-0.5, 0.5, 1.5],
            alignment_tol=0.01,
        )


def test_binned_tuning_recovers_profiles_and_metadata() -> None:
    edges = np.array([-1.0, 0.0, 1.0, 2.0])
    centers = (edges[:-1] + edges[1:]) * 0.5
    behavioral_values = np.repeat(centers, [3, 4, 5])
    time = np.arange(behavioral_values.size, dtype=float) / 10.0
    neural_values = np.column_stack((1.0 + 2.0 * behavioral_values, 5.0 - 0.5 * behavioral_values))
    clock = Clock("experiment", "monotonic", synchronization_domain="session")
    neural = _features(
        neural_values,
        time,
        names=["unit-a", "unit-b"],
        units=["spikes/s", "uV"],
        attrs={"array": "M1"},
    )
    behavior = _behavior(
        behavioral_values,
        time,
        names=["speed"],
        units="m/s",
        clock=clock,
        attrs={"sensor": "camera"},
    )

    result = binned_tuning(
        neural,
        behavior,
        bin_edges=edges,
        alignment_tol=0.0,
        neural_clock=clock,
    )

    expected = np.column_stack((1.0 + 2.0 * centers, 5.0 - 0.5 * centers))
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_array_equal(result.occupancy, [3, 4, 5])
    np.testing.assert_array_equal(result.bin_edges, edges)
    np.testing.assert_array_equal(result.bin_coordinates[:, 0], centers)
    assert result.axis_order == ("bin", "feature_or_unit")
    assert result.bin_coordinate_axis_order == ("bin", "behavioral_dimension")
    assert result.feature_names == ("unit-a", "unit-b")
    assert result.feature_units == ("spikes/s", "uV")
    assert result.feature_channels == (None, None)
    assert result.behavior_names == ("speed",)
    assert result.behavior_units == ("m/s",)
    assert result.source_name == "motor-cortex"
    assert result.neural_clock is clock
    assert result.behavior_clock is clock
    assert result.source_attrs["array"] == "M1"
    assert result.behavior_attrs["sensor"] == "camera"
    assert result.attrs["source_fs"] == 10.0
    assert result.attrs["source_window_size"] == 0.1
    assert result.attrs["source_shift"] == 0.1


def test_directional_tuning_recovers_cosine_preference() -> None:
    n_bins = 12
    centers = (np.arange(n_bins) + 0.5) * (2.0 * np.pi / n_bins)
    angles = np.repeat(centers, 6)
    time = np.arange(angles.size, dtype=float) / 10.0
    preferences = np.array([0.7, 4.2])
    amps = np.array([3.0, 1.5])
    responses = 5.0 + amps * np.cos(angles[:, np.newaxis] - preferences)

    result = directional_tuning(
        _features(responses, time, names=["u0", "u1"]),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
    )

    assert isinstance(result, DirectionalTuningResult)
    circular_error = np.angle(np.exp(1j * (result.preferred_direction - preferences)))
    np.testing.assert_allclose(circular_error, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.modulation_depth, amps, atol=1e-12)
    np.testing.assert_allclose(result.r_squared, 1.0, atol=1e-12)
    np.testing.assert_array_equal(result.occupancy, np.full(n_bins, 6))
    assert result.tuning.attrs["bin_interval"] == "[low, high)"
    assert result.tuning.attrs["tuning_kind"] == "directional"
    assert result.tuning.attrs["circular_period"] == 2.0 * np.pi
    assert result.tuning.attrs["circular_interval"] == "[0, 2*pi)"
    assert result.tuning.attrs["fit_weighting"] == "uniform_valid_bins"

    normalized = directional_tuning(
        _features(responses, time, names=["u0", "u1"]),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
        normalization="zscore",
    )
    assert normalized.tuning.feature_units == ("z-score", "z-score")
    assert normalized.tuning.attrs["source_feature_units"] == ("Hz", "Hz")


def test_uneven_occupancy_and_empty_bins_are_explicit() -> None:
    behavior_values = np.array([0.1, 0.2, 0.7, 1.5, -0.1])
    time = np.arange(behavior_values.size, dtype=float) / 10.0
    result = binned_tuning(
        _features(np.array([1.0, 3.0, 9.0, 100.0, 200.0]), time),
        _behavior(behavior_values, time, names=["position"]),
        bin_edges=np.array([0.0, 0.5, 1.0, 1.5]),
        alignment_tol=0.0,
        minimum_sample_count=2,
    )

    np.testing.assert_array_equal(result.occupancy, [2, 1, 0])
    np.testing.assert_allclose(result.values[0], [2.0])
    assert np.all(np.isnan(result.values[1:]))
    assert result.minimum_sample_count == 2
    assert result.empty_bin_behavior == "nan"


def test_zscore_matches_independent_calculation() -> None:
    behavior_values = np.array([0.1, 0.2, 0.6, 0.7])
    neural_values = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0], [7.0, 22.0]])
    time = np.arange(4, dtype=float)

    result = binned_tuning(
        _features(neural_values, time, names=["a", "b"]),
        _behavior(behavior_values, time, names=["x"]),
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.0,
        normalization="zscore",
    )

    normalized = (neural_values - neural_values.mean(axis=0)) / neural_values.std(axis=0)
    expected = np.stack((normalized[:2].mean(axis=0), normalized[2:].mean(axis=0)))
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_allclose(result.attrs["normalization_center"], neural_values.mean(axis=0))
    np.testing.assert_allclose(result.attrs["normalization_scale"], neural_values.std(axis=0))
    assert result.feature_units == ("z-score", "z-score")
    assert result.attrs["source_feature_units"] == ("Hz", "Hz")
    with pytest.raises(ValidationError, match="must use 'z-score'"):
        dataclasses.replace(result, feature_units=("Hz", "Hz"))
    missing_provenance = dict(result.attrs)
    missing_provenance.pop("normalization_center")
    with pytest.raises(ValidationError, match="complete normalization provenance"):
        dataclasses.replace(result, attrs=missing_provenance)
    invalid_scale = dict(result.attrs)
    invalid_scale["normalization_scale"] = np.array([-1.0, 1.0])
    with pytest.raises(ValidationError, match="must be non-negative"):
        dataclasses.replace(result, attrs=invalid_scale)


def test_constant_feature_zscore_is_explicitly_zero() -> None:
    time = np.arange(4, dtype=float)
    result = binned_tuning(
        _features(np.full(4, 7.0), time),
        _behavior([0.1, 0.2, 0.6, 0.7], time),
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.0,
        normalization="zscore",
    )

    np.testing.assert_array_equal(result.values, np.zeros((2, 1)))
    assert result.attrs["constant_features_normalized_to_zero"] == (0,)
    incomplete = dict(result.attrs)
    incomplete["constant_features_normalized_to_zero"] = ()
    with pytest.raises(ValidationError, match="exactly match"):
        dataclasses.replace(result, attrs=incomplete)


def test_zscore_without_observations_has_empty_provenance() -> None:
    time = np.arange(2, dtype=float)
    result = binned_tuning(
        _features([1.0, 2.0], time),
        _behavior([10.0, 11.0], time),
        bin_edges=np.array([0.0, 1.0]),
        alignment_tol=0.0,
        normalization="zscore",
    )

    np.testing.assert_array_equal(result.occupancy, [0])
    assert np.isnan(result.values[0, 0])
    assert result.attrs["normalization_center"] is None
    assert result.attrs["normalization_scale"] is None
    assert result.attrs["constant_features_normalized_to_zero"] == ()


def test_zscore_rejects_unrepresentable_scale() -> None:
    smallest = np.nextafter(0.0, 1.0)
    with pytest.raises(ValidationError, match="scale is not representable"):
        binned_tuning(
            _features([0.0, smallest], [0.0, 1.0]),
            _behavior([0.1, 0.2], [0.0, 1.0]),
            bin_edges=np.array([0.0, 1.0]),
            alignment_tol=0.0,
            normalization="zscore",
        )


def test_extreme_finite_bins_and_statistics_have_stable_behavior() -> None:
    time = np.arange(2, dtype=float)
    centered = binned_tuning(
        _features([1.0, 3.0], time),
        _behavior([-5e307, 5e307], time),
        bin_edges=np.array([-1e308, 1e308]),
        alignment_tol=0.0,
    )
    assert centered.bin_coordinates[0, 0] == 0.0

    result = binned_tuning(
        _features([1e300, 1e300], time),
        _behavior([0.1, 0.2], time),
        bin_edges=np.array([0.0, 1.0]),
        alignment_tol=0.0,
    )
    assert result.values[0, 0] == 1e300


def test_nan_policy_raise_and_complete_case_omit_are_explicit() -> None:
    time = np.arange(4, dtype=float)
    neural = _features(np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0], [7.0, 8.0]]), time)
    behavior = _behavior(np.array([0.1, 0.2, np.nan, 0.8]), time)

    with pytest.raises(ValidationError, match="contain NaN"):
        binned_tuning(
            neural,
            behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            alignment_tol=0.0,
        )
    omitted = binned_tuning(
        neural,
        behavior,
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.0,
        nan_policy="omit",
    )
    np.testing.assert_array_equal(omitted.occupancy, [1, 1])
    np.testing.assert_array_equal(omitted.values[:, 0], [1.0, 7.0])


def test_spike_train_uses_explicit_half_open_observation_bins() -> None:
    spikes = SpikeTrain(
        times=[np.array([0.1, 0.9, 1.0, 2.9, 3.0]), np.array([0.2, 1.2, 1.8])],
        units=["cell-a", "cell-b"],
        channels=[4, 7],
        attrs={"sorter": "synthetic"},
    )
    behavior = _behavior(
        [0.2, 0.2, 0.8],
        [0.5, 1.5, 2.5],
        names=["target"],
    )

    result = binned_tuning(
        spikes,
        behavior,
        bin_edges=np.array([0.0, 0.5, 1.0]),
        spike_observation_edges=np.array([0.0, 1.0, 2.0, 3.0]),
        alignment_tol=0.0,
    )

    # Spike at 1.0 belongs to [1, 2); spike at the final edge 3.0 is excluded.
    np.testing.assert_allclose(result.values[0], [1.5, 1.5])
    np.testing.assert_allclose(result.values[1], [1.0, 0.0])
    np.testing.assert_array_equal(result.occupancy, [2, 1])
    assert result.source_kind == "spikes"
    assert result.feature_names == ("cell-a", "cell-b")
    assert result.feature_units == ("Hz", "Hz")
    assert result.feature_channels == (4, 7)
    assert result.source_attrs["sorter"] == "synthetic"
    np.testing.assert_array_equal(
        result.attrs["spike_observation_edges"], np.array([0.0, 1.0, 2.0, 3.0])
    )
    assert result.attrs["spike_observation_interval"] == "[start, end)"
    assert result.attrs["spike_observation_time_reference"] == "center"
    with pytest.raises(ValueError):
        result.attrs["spike_observation_edges"][0] = 1.0
    for name in (
        "spike_observation_edges",
        "spike_observation_interval",
        "spike_observation_time_reference",
    ):
        incomplete = dict(result.attrs)
        incomplete.pop(name)
        with pytest.raises(ValidationError, match="complete observation geometry"):
            dataclasses.replace(result, attrs=incomplete)
    with pytest.raises(ValidationError, match="cannot contain trials"):
        dataclasses.replace(result, trials=(Trial(0, 0.0, 1.0),))
    invalid_units = dict(result.attrs)
    invalid_units["source_feature_units"] = ("mV", "mV")
    with pytest.raises(ValidationError, match="source units must be 'Hz'"):
        dataclasses.replace(result, attrs=invalid_units)
    invalid_interval = dict(result.attrs)
    invalid_interval["spike_observation_interval"] = "closed"
    with pytest.raises(ValidationError, match="spike_observation_interval"):
        dataclasses.replace(result, attrs=invalid_interval)
    invalid_reference = dict(result.attrs)
    invalid_reference["spike_observation_time_reference"] = "start"
    with pytest.raises(ValidationError, match="spike_observation_time_reference"):
        dataclasses.replace(result, attrs=invalid_reference)


def test_trials_delegate_to_shared_slicing(monkeypatch) -> None:
    time = np.arange(20, dtype=float) / 10.0
    neural = _features(1.0 + time, time, window_size=None, shift=None)
    behavior = _behavior(np.mod(time, 1.0), time, names=["phase"])
    first = Trial(10, 0.0, 1.0, label="outbound", target_id=1, attrs={"side": "left"})
    second = Trial(20, 1.0, 2.0, label="return", target_id=2, attrs={"side": "right"})
    trials = TrialTable([first, second], attrs={"block_set": "training"})
    calls = {"split": 0, "align": 0}
    shared_split = tuning_module.split_recording_trials
    shared_align = tuning_module.align_neural_behavior_nearest

    def wrapped_split(*args, **kwargs):
        calls["split"] += 1
        return shared_split(*args, **kwargs)

    def wrapped_align(*args, **kwargs):
        calls["align"] += 1
        return shared_align(*args, **kwargs)

    monkeypatch.setattr(tuning_module, "split_recording_trials", wrapped_split)
    monkeypatch.setattr(tuning_module, "align_neural_behavior_nearest", wrapped_align)

    result = binned_tuning(
        neural,
        behavior,
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.0,
        trials=trials,
    )

    assert calls == {"split": 1, "align": 2}
    assert result.trials == (first, second)
    assert result.trials[0].label == "outbound"
    assert result.trials[0].target_id == 1
    assert result.trials[0].attrs["side"] == "left"
    assert result.trial_table_attrs["block_set"] == "training"
    np.testing.assert_array_equal(result.occupancy, [10, 10])


@pytest.mark.parametrize("time_reference", ["start", "center", "end"])
@pytest.mark.parametrize(
    ("feature_time", "trial_start", "trial_stop"),
    [
        (0.5, 0.0, 1.0),
        (0.5, 0.0, 2.0),
        (0.5, 0.5, 1.0),
        (0.5, 0.25, 0.75),
    ],
)
def test_windowed_feature_trials_require_support_slicing(
    time_reference: str,
    feature_time: float,
    trial_start: float,
    trial_stop: float,
) -> None:
    neural = _features(
        [1.0],
        [feature_time],
        window_size=1.0,
        shift=1.0,
        attrs={"timestamp_reference": time_reference},
    )
    trials = TrialTable([Trial(0, trial_start, trial_stop)])

    with pytest.raises(ValidationError, match="explicit feature time-reference"):
        binned_tuning(
            neural,
            _behavior([0.2], [feature_time]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            alignment_tol=0.0,
            trials=trials,
        )


def test_spike_trials_on_observation_edges_are_unsupported() -> None:
    trials = TrialTable([Trial(0, 0.0, 1.0), Trial(1, 1.0, 2.0)])
    with pytest.raises(ValidationError, match="support-aware observation slicing"):
        binned_tuning(
            SpikeTrain(times=[np.array([0.1, 1.1])]),
            _behavior([0.2, 0.8], [0.5, 1.5]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            spike_observation_edges=np.array([0.0, 1.0, 2.0]),
            alignment_tol=0.0,
            trials=trials,
        )


def test_spike_observation_crossing_boundary_is_unsupported() -> None:
    trials = TrialTable([Trial(0, 0.5, 1.5)])
    with pytest.raises(ValidationError, match="support-aware observation slicing"):
        binned_tuning(
            SpikeTrain(times=[np.array([0.1, 1.1])]),
            _behavior([0.2], [1.0]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            spike_observation_edges=np.array([0.0, 1.0, 2.0]),
            alignment_tol=0.0,
            trials=trials,
        )


def test_behavior_dimension_selection_preserves_requested_identity() -> None:
    time = np.arange(4, dtype=float)
    behavior = _behavior(
        np.column_stack(([100.0, 100.0, 100.0, 100.0], [0.1, 0.2, 0.7, 0.8])),
        time,
        names=["unused", "speed"],
        units=["px", "m/s"],
    )
    result = binned_tuning(
        _features([1.0, 3.0, 5.0, 7.0], time),
        behavior,
        behavior_dim="speed",
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.0,
    )

    assert result.behavior_names == ("speed",)
    assert result.behavior_units == ("m/s",)
    np.testing.assert_allclose(result.values[:, 0], [2.0, 6.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bin_edges": np.array([0.0, 0.0, 1.0])}, "strictly increasing"),
        ({"minimum_sample_count": 0}, "positive"),
        ({"normalization": "minmax"}, "normalization must be one of"),
        ({"smoothing": "gaussian"}, "smoothing must be one of"),
        ({"empty_bin_behavior": "zero"}, "empty_bin_behavior must be one of"),
        ({"behavior_dim": "missing"}, "unknown behavior_dimension"),
    ],
)
def test_invalid_binning_contracts_are_rejected(kwargs, message) -> None:
    time = np.arange(3, dtype=float)
    arguments = {
        "bin_edges": np.array([0.0, 0.5, 1.0]),
        "alignment_tol": 0.0,
        **kwargs,
    }
    with pytest.raises(ValidationError, match=message):
        binned_tuning(
            _features([1.0, 2.0, 3.0], time), _behavior([0.1, 0.5, 0.9], time), **arguments
        )


def test_spike_and_direction_specific_contracts_are_rejected() -> None:
    time = np.arange(3, dtype=float)
    behavior = _behavior([0.1, 0.5, 0.9], time)
    with pytest.raises(ValidationError, match="requires spike_observation_edges"):
        binned_tuning(
            SpikeTrain(times=[np.array([0.2])]),
            behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            alignment_tol=0.0,
        )
    with pytest.raises(ValidationError, match="must be unique"):
        binned_tuning(
            SpikeTrain(
                times=[np.array([0.2]), np.array([0.4])],
                units=["duplicate", "duplicate"],
            ),
            behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            spike_observation_edges=np.array([0.0, 1.0, 2.0, 3.0]),
            alignment_tol=0.0,
        )
    with pytest.raises(ValidationError, match="equal-width"):
        binned_tuning(
            SpikeTrain(times=[np.array([0.2])]),
            behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            spike_observation_edges=np.array([0.0, 0.5, 2.0, 3.0]),
            alignment_tol=0.0,
        )
    with pytest.raises(ValidationError, match="equal-width"):
        binned_tuning(
            SpikeTrain(times=[np.array([])]),
            behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            spike_observation_edges=np.array([0.0, 1e-13, 3e-13]),
            alignment_tol=0.0,
        )
    with pytest.raises(ValidationError, match="unrepresentable firing rates"):
        binned_tuning(
            SpikeTrain(times=[np.array([])]),
            behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            spike_observation_edges=np.array([0.0, 1e-320, 2e-320]),
            alignment_tol=0.0,
        )
    with pytest.raises(ValidationError, match="unit 'rad'"):
        directional_tuning(
            _features([1.0, 2.0, 3.0], time),
            behavior,
            alignment_tol=0.0,
        )


def test_result_arrays_and_metadata_are_immutable() -> None:
    time = np.arange(4, dtype=float)
    result = binned_tuning(
        _features([1.0, 2.0, 3.0, 4.0], time),
        _behavior([0.1, 0.2, 0.7, 0.8], time),
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.0,
    )

    with pytest.raises(ValueError):
        result.values[0, 0] = 0.0
    with pytest.raises(TypeError):
        result.attrs["new"] = "value"
    with pytest.raises(TypeError):
        result.trial_table_attrs["new"] = "value"
    with pytest.raises(ValidationError, match="values bin axis"):
        dataclasses.replace(result, values=np.ones((3, 1)))
    with pytest.raises(ValidationError, match="midpoints"):
        dataclasses.replace(result, bin_coordinates=np.zeros((2, 1)))
    with pytest.raises(ValidationError, match="below minimum_sample_count"):
        dataclasses.replace(result, minimum_sample_count=3)
    with pytest.raises(ValidationError, match="meeting minimum_sample_count"):
        dataclasses.replace(result, values=np.full_like(result.values, np.nan))
    with pytest.raises(ValidationError, match="fit in int64"):
        dataclasses.replace(
            result,
            occupancy=np.array([np.iinfo(np.uint64).max, 1], dtype=np.uint64),
        )
    for name in ("source_fs", "source_window_size", "source_shift"):
        incomplete = dict(result.attrs)
        incomplete.pop(name)
        with pytest.raises(ValidationError, match="complete source geometry"):
            dataclasses.replace(result, attrs=incomplete)
    invalid_rate = dict(result.attrs)
    invalid_rate["source_fs"] = 0.0
    with pytest.raises(ValidationError, match="source_fs"):
        dataclasses.replace(result, attrs=invalid_rate)
    invalid_window = dict(result.attrs)
    invalid_window["source_window_size"] = -1.0
    with pytest.raises(ValidationError, match="source_window_size"):
        dataclasses.replace(result, attrs=invalid_window)
    with pytest.raises(ValidationError, match="cannot contain trials"):
        dataclasses.replace(result, trials=(Trial(0, 0.0, 1.0),))


def test_result_accepts_equivalent_midpoint_and_canonicalizes_it() -> None:
    result = binned_tuning(
        _features([1.0], [0.0]),
        _behavior([-5.0], [0.0]),
        bin_edges=np.array([-10.0, 2.2]),
        alignment_tol=0.0,
    )
    alternate_midpoint = np.array([[-3.9]])
    assert alternate_midpoint[0, 0] != result.bin_coordinates[0, 0]

    reconstructed = dataclasses.replace(result, bin_coordinates=alternate_midpoint)

    np.testing.assert_array_equal(reconstructed.bin_coordinates, result.bin_coordinates)


def test_constant_profile_has_no_fabricated_preference() -> None:
    n_bins = 8
    angles = (np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins
    time = np.arange(n_bins, dtype=float)
    result = directional_tuning(
        _features(np.ones(n_bins), time),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
    )

    assert np.isnan(result.preferred_direction[0])
    assert result.modulation_depth[0] == pytest.approx(0.0, abs=1e-14)
    assert np.isnan(result.r_squared[0])
    with pytest.raises(ValidationError, match=r"\[0, 2\*pi\)"):
        dataclasses.replace(result, preferred_direction=np.array([2.0 * np.pi]))
    with pytest.raises(ValidationError, match="non-negative"):
        dataclasses.replace(result, modulation_depth=np.array([-1.0]))


@pytest.mark.parametrize("preference", [0.0, np.pi])
def test_directional_preference_boundary_values_are_canonical(preference: float) -> None:
    n_bins = 16
    angles = (np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins
    time = np.arange(n_bins, dtype=float)
    result = directional_tuning(
        _features(2.0 + np.cos(angles - preference), time),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
    )

    error = np.angle(np.exp(1j * (result.preferred_direction[0] - preference)))
    assert error == pytest.approx(0.0, abs=1e-14)
    assert 0.0 <= result.preferred_direction[0] < 2.0 * np.pi


def test_direction_wrapping_distinguishes_small_negatives() -> None:
    values = np.array([-1e-20, 0.0, 2.0 * np.pi, -2.0 * np.pi, 4.0 * np.pi, -4.0 * np.pi])
    time = np.arange(values.size, dtype=float)
    result = directional_tuning(
        _features(np.ones(values.size), time),
        _behavior(values, time, names=["direction"], units="rad"),
        n_bins=8,
        alignment_tol=0.0,
    )

    assert result.occupancy[-1] == 1
    assert result.occupancy[0] == 5
    assert result.occupancy.sum() == values.size


@pytest.mark.parametrize("scale", [1e-20, 1e-10, 1.0, 1e10, 1e150, 1e200, 1e300])
def test_directional_fit_is_scale_invariant_and_overflow_safe(scale: float) -> None:
    n_bins = 16
    preference = 0.7
    angles = np.repeat((np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins, 3)
    time = np.arange(angles.size, dtype=float)
    response = scale * (1.0 + 0.5 * np.cos(angles - preference))
    result = directional_tuning(
        _features(response, time),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
    )

    error = np.angle(np.exp(1j * (result.preferred_direction[0] - preference)))
    assert error == pytest.approx(0.0, abs=1e-12)
    assert result.modulation_depth[0] == pytest.approx(0.5 * scale, rel=1e-12)
    assert result.r_squared[0] == pytest.approx(1.0, abs=1e-12)


def test_directional_summaries_must_be_derived_from_tuning() -> None:
    n_bins = 8
    angles = (np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins
    time = np.arange(n_bins, dtype=float)
    result = directional_tuning(
        _features(2.0 + np.cos(angles - 0.5), time),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
    )

    with pytest.raises(ValidationError, match="preferred_direction must match"):
        dataclasses.replace(result, preferred_direction=np.array([1.0]))
    with pytest.raises(ValidationError, match="modulation_depth must match"):
        dataclasses.replace(result, modulation_depth=np.array([2.0]))
    with pytest.raises(ValidationError, match="cosine_coefs must match"):
        dataclasses.replace(result, cosine_coefs=np.zeros((1, 3)))
    with pytest.raises(ValidationError, match=r"r_squared must be in \[0, 1\]"):
        dataclasses.replace(result, r_squared=np.array([2.0]))


def test_directional_result_requires_circular_geometry() -> None:
    n_bins = 8
    angles = (np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins
    time = np.arange(n_bins, dtype=float)
    result = directional_tuning(
        _features(2.0 + np.cos(angles), time),
        _behavior(angles, time, names=["direction"], units="rad"),
        n_bins=n_bins,
        alignment_tol=0.0,
    )

    invalid_edges = result.tuning.bin_edges.copy()
    invalid_edges[1] += 0.01
    invalid_tuning = dataclasses.replace(
        result.tuning,
        bin_edges=invalid_edges,
        bin_coordinates=tuning_module._bin_centers(invalid_edges)[:, np.newaxis],
    )
    with pytest.raises(ValidationError, match="uniform bin edges"):
        dataclasses.replace(result, tuning=invalid_tuning)

    missing_kind = dict(result.tuning.attrs)
    missing_kind.pop("tuning_kind")
    invalid_tuning = dataclasses.replace(result.tuning, attrs=missing_kind)
    with pytest.raises(ValidationError, match="tuning_kind"):
        dataclasses.replace(result, tuning=invalid_tuning)


def test_invalid_behavior_and_gap_clocks_are_rejected() -> None:
    time = np.arange(3, dtype=float)
    bad_behavior = _behavior([0.1, 0.5, 0.9], time, clock="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"behavior\.clock"):
        binned_tuning(
            _features([1.0, 2.0, 3.0], time),
            bad_behavior,
            bin_edges=np.array([0.0, 0.5, 1.0]),
            alignment_tol=0.0,
        )

    discontinuities = EventSeries(clock="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"discontinuities\.clock"):
        binned_tuning(
            _features([1.0, 2.0, 3.0], time),
            _behavior([0.1, 0.5, 0.9], time),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            alignment_tol=0.0,
            discontinuities=discontinuities,
        )


def test_tuning_module_import_does_not_load_native_extensions() -> None:
    code = """
import importlib
import json
import sys

module = importlib.import_module("neurale.features.tuning")
print(json.dumps({
    "has_api": hasattr(module, "TuningResult"),
    "native": "neurale._native" in sys.modules,
    "cuda": "neurale._native_cuda" in sys.modules,
}))
"""
    result = probe_json(code)
    assert result == {"has_api": True, "native": False, "cuda": False}


def test_public_feature_exports_are_available() -> None:
    import neurale.features as features

    assert features.TuningResult is TuningResult
    assert features.DirectionalTuningResult is DirectionalTuningResult
    assert features.binned_tuning is binned_tuning
    assert features.directional_tuning is directional_tuning
