#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _subprocess_probe import probe_json

from neurale.data import Clock, Recording, SignalArray, Trial, TrialTable
from neurale.exceptions import ValidationError
from neurale.features import (
    KinematicSeries,
    acceleration,
    as_kinematic_series,
    direction,
    kinematic_derivative,
    kinematic_trials,
    speed,
    velocity,
)


def _signal(
    data: np.ndarray,
    time: np.ndarray,
    *,
    fs: float = 10.0,
    clock: Clock | None = None,
    names: list[str] | None = None,
    units: str | list[str] = "m",
    name: str = "hand_position",
    attrs: dict[str, object] | None = None,
) -> SignalArray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return SignalArray.from_array(
        values,
        fs=fs,
        time=np.asarray(time, dtype=float),
        channel_names=names or [f"d{idx}" for idx in range(values.shape[1])],
        channel_types="behavior",
        units=units,
        name=name,
        clock=clock,
        attrs={"subject": "synthetic"} if attrs is None else attrs,
    )


def test_derivatives_preserve_typed_metadata() -> None:
    time = np.arange(11, dtype=float) / 10.0
    clock = Clock("motion", "camera", synchronization_domain="session")
    source = _signal(
        np.column_stack((1.0 + 2.0 * time, -3.0 * time)),
        time,
        clock=clock,
        names=["x", "y"],
    )

    trajectory = as_kinematic_series(
        source,
        quantity="position",
        trial=Trial(7, 0.0, 1.1, label="linear"),
    )
    result_velocity = velocity(trajectory)
    result_acceleration = acceleration(trajectory)

    assert isinstance(trajectory, KinematicSeries)
    assert trajectory.data.shape == (1, 11, 2)
    assert trajectory.time.shape == (1, 11)
    assert trajectory.trial_ids == (7,)
    assert trajectory.trials[0] is not None
    assert trajectory.trials[0].label == "linear"
    assert trajectory.dimension_names == ("x", "y")
    assert trajectory.units == ("m", "m")
    assert trajectory.clock is clock
    assert trajectory.fs == 10.0
    np.testing.assert_array_equal(trajectory.time[0], time)
    np.testing.assert_allclose(
        result_velocity.data[0],
        np.tile([2.0, -3.0], (time.size, 1)),
        atol=1e-14,
    )
    np.testing.assert_allclose(result_acceleration.data, 0.0, atol=1e-13)
    assert result_velocity.units == ("m/s", "m/s")
    assert result_acceleration.units == ("m/s^2", "m/s^2")
    assert result_velocity.attrs["derivative_method"] == "gradient"
    assert result_velocity.attrs["boundary"] == "one-sided"
    assert result_velocity.attrs["smoothing"] == "none"


def test_constant_trajectory_has_zero_velocity() -> None:
    time = np.arange(6, dtype=float) / 5.0
    source = _signal(np.full((6, 2), 4.0), time, fs=5.0, names=["x", "y"])

    result_velocity = velocity(source)
    result_speed = speed(source)
    result_direction = direction(source)

    np.testing.assert_array_equal(result_velocity.data, np.zeros((1, 6, 2)))
    np.testing.assert_array_equal(result_speed.data, np.zeros((1, 6, 1)))
    assert np.all(np.isnan(result_direction.data))
    assert result_speed.dimension_names == ("speed",)
    assert result_direction.dimension_names == ("direction",)
    assert result_direction.units == ("rad",)


def test_typed_velocity_supports_speed_and_acceleration() -> None:
    time = np.arange(5, dtype=float)
    source = _signal(
        np.tile([3.0, 4.0], (time.size, 1)),
        time,
        names=["vx", "vy"],
        units="m/s",
        name="hand_velocity",
    )
    typed_velocity = as_kinematic_series(source, quantity="velocity")

    assert np.all(speed(typed_velocity).data[:, :, 0] == 5.0)
    np.testing.assert_allclose(
        direction(typed_velocity).data[:, :, 0],
        np.arctan2(4.0, 3.0),
    )
    derived_acceleration = acceleration(typed_velocity)
    assert np.all(derived_acceleration.data == 0.0)
    assert derived_acceleration.quantity == "acceleration"
    assert derived_acceleration.units == ("m/s^2", "m/s^2")


@pytest.mark.parametrize(
    ("components", "expected_speed"),
    [
        ([1e200, 1e200], np.hypot(1e200, 1e200)),
        ([1e308, 1e308], np.hypot(1e308, 1e308)),
        ([1e-308, 1e-308], np.hypot(1e-308, 1e-308)),
    ],
)
def test_speed_and_direction_are_stable_at_extremes(
    components,
    expected_speed,
) -> None:
    time = np.array([0.0, 1.0])
    source = _signal(
        np.tile(components, (2, 1)),
        time,
        names=["vx", "vy"],
        units="m/s",
        name="velocity",
    )
    typed_velocity = as_kinematic_series(source, quantity="velocity")

    result_speed = speed(typed_velocity)
    result_direction = direction(typed_velocity)

    np.testing.assert_allclose(
        result_speed.data[:, :, 0],
        np.full((1, 2), expected_speed),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        result_direction.data[:, :, 0],
        np.full((1, 2), np.pi / 4.0),
        rtol=1e-15,
    )
    assert np.all(np.isfinite(result_speed.data))
    assert np.all(np.isfinite(result_direction.data))


@pytest.mark.parametrize("quantity", ["trajectory", "position", "velocity", "acceleration"])
def test_typed_source_quantities_are_explicit(quantity) -> None:
    time = np.arange(3, dtype=float)

    result = as_kinematic_series(_signal(time, time, names=["x"]), quantity=quantity)

    assert result.quantity == quantity


def test_circular_trajectory_matches_analytic_interior() -> None:
    time = np.linspace(0.0, 2.0 * np.pi, 401)
    source = _signal(
        np.column_stack((np.cos(time), np.sin(time))),
        time,
        fs=400.0 / (2.0 * np.pi),
        names=["x", "y"],
    )

    result_speed = speed(source)
    result_direction = direction(source)

    np.testing.assert_allclose(result_speed.data[0, 1:-1, 0], 1.0, rtol=5e-5, atol=5e-5)
    expected_direction = np.unwrap(np.arctan2(np.cos(time), -np.sin(time)))
    actual_direction = np.unwrap(result_direction.data[0, :, 0])
    np.testing.assert_allclose(actual_direction[1:-1], expected_direction[1:-1], atol=1e-12)


def test_irregular_time_uses_explicit_timestamps() -> None:
    time = np.array([0.0, 0.1, 0.4, 1.0])
    source = _signal(time**2, time, fs=10.0, names=["x"])

    result = velocity(source)

    # First-order one-sided boundaries and the exact irregular-grid central
    # derivative of a quadratic at the two interior samples.
    expected = np.array([0.1, 0.2, 0.8, 1.4])
    np.testing.assert_allclose(result.data[0, :, 0], expected, atol=1e-14)
    np.testing.assert_array_equal(result.time[0], time)
    assert result.fs == source.fs


def test_two_samples_use_one_sided_boundary() -> None:
    two = _signal(np.array([1.0, 4.0]), np.array([2.0, 2.5]), names=["x"])
    one = _signal(np.array([1.0]), np.array([2.0]), names=["x"])

    np.testing.assert_array_equal(velocity(two).data[0, :, 0], [6.0, 6.0])
    with pytest.raises(ValidationError, match="order 1 requires at least 2 samples"):
        velocity(one)


def test_position_acceleration_requires_three_samples() -> None:
    time = np.array([0.0, 1.0])
    pos = as_kinematic_series(
        _signal(np.array([0.0, 10.0]), time, names=["x"]),
        quantity="position",
    )
    typed_velocity = as_kinematic_series(
        _signal(np.array([0.0, 10.0]), time, names=["vx"], units="m/s"),
        quantity="velocity",
    )

    with pytest.raises(ValidationError, match="order 2 requires at least 3 samples"):
        acceleration(pos)
    np.testing.assert_array_equal(acceleration(typed_velocity).data[0, :, 0], [10.0, 10.0])


@pytest.mark.parametrize(
    "time",
    [
        np.array([0.0, 1.0, 2.0]),
        np.arange(7, dtype=float) * 0.25,
        np.array([0.0, 0.1, 0.4, 1.0]),
    ],
)
def test_quadratic_acceleration_is_correct_everywhere(time) -> None:
    source = _signal(time**2, time, names=["x"])

    result = acceleration(source)

    np.testing.assert_allclose(result.data[0, :, 0], 2.0, rtol=1e-13, atol=1e-13)


def test_derivative_overflow_raises_validation_error() -> None:
    source = _signal(
        np.array([0.0, 1.0]),
        np.array([0.0, 1e-320]),
        names=["x"],
    )

    with pytest.raises(ValidationError, match="produced non-finite"):
        velocity(source)


def test_unrepresentable_speed_raises_validation_error() -> None:
    time = np.array([0.0, 1.0])
    source = _signal(
        np.full((2, 4), 1e308),
        time,
        names=["v0", "v1", "v2", "v3"],
        units="m/s",
        name="velocity",
    )
    typed_velocity = as_kinematic_series(source, quantity="velocity")

    with pytest.raises(ValidationError, match="exceeds the finite float64 range"):
        speed(typed_velocity)


def test_nan_policy_is_explicit_and_infinities_rejected() -> None:
    time = np.arange(5, dtype=float)
    source = _signal(np.array([0.0, 1.0, np.nan, 3.0, 4.0]), time, names=["x"])

    with pytest.raises(ValidationError, match="contains NaN"):
        velocity(source)
    propagated = velocity(source, nan_policy="propagate")
    assert np.any(np.isnan(propagated.data))

    series = as_kinematic_series(_signal(np.arange(5.0), time, names=["x"]))
    with pytest.raises(ValidationError, match="infinite"):
        dataclasses.replace(series, data=np.full(series.data.shape, np.inf))


def test_derivative_matches_independent_calculation() -> None:
    time = np.arange(8, dtype=float) * 0.25
    values = np.column_stack((time**3, 2.0 * time**2))
    source = _signal(values, time, fs=4.0, names=["x", "y"])

    result = kinematic_derivative(source)
    expected = np.empty_like(values)
    expected[0] = (values[1] - values[0]) / 0.25
    expected[-1] = (values[-1] - values[-2]) / 0.25
    expected[1:-1] = (values[2:] - values[:-2]) / 0.5

    np.testing.assert_allclose(result.data[0], expected, atol=1e-14)


def test_trial_extraction_reuses_shared_slicing(
    monkeypatch,
) -> None:
    time = np.arange(20, dtype=float) / 10.0
    pos = _signal(
        np.column_stack((time, -time)),
        time,
        names=["x", "y"],
        attrs={"camera_calibration": "cal-42"},
    )
    first_trial = Trial(
        10,
        0.0,
        1.0,
        label="reach",
        target_id=3,
        target=np.array([1.0, 2.0]),
        outcome="success",
        block=4,
        attrs={"condition": "left"},
    )
    second_trial = Trial(20, 1.0, 2.0, label="return", outcome="success", block=4)
    recording = Recording(
        signals={"position": pos},
        trials=TrialTable([first_trial, second_trial]),
        metadata={"session": "synthetic"},
    )
    from neurale.data import split_recording_trials as shared_split

    calls = []

    def _tracked_split(*args, **kwargs):
        calls.append((args, kwargs))
        return shared_split(*args, **kwargs)

    monkeypatch.setattr("neurale.features.kinematics.split_recording_trials", _tracked_split)

    result = kinematic_trials(recording, reference_clock=None)

    assert len(calls) == 1
    assert calls[0][1]["signal_names"] == ("position",)
    assert result.data.shape == (2, 10, 2)
    assert result.time.shape == (2, 10)
    assert result.trial_ids == (10, 20)
    assert result.trials == (first_trial, second_trial)
    assert result.trials[0].start == 0.0
    assert result.trials[0].stop == 1.0
    assert result.trials[0].label == "reach"
    assert result.trials[0].target_id == 3
    np.testing.assert_array_equal(result.trials[0].target, [1.0, 2.0])
    assert result.trials[0].outcome == "success"
    assert result.trials[0].block == 4
    assert result.trials[0].attrs["condition"] == "left"
    assert result.dimension_names == ("x", "y")
    assert result.attrs["recording_metadata"]["session"] == "synthetic"
    assert result.source_attrs["camera_calibration"] == "cal-42"


def test_unequal_trial_lengths_are_rejected() -> None:
    time = np.arange(20, dtype=float) / 10.0
    recording = Recording(
        signals={"position": _signal(time, time, names=["x"])},
        trials=TrialTable([Trial(0, 0.0, 0.5), Trial(1, 0.5, 2.0)]),
    )

    with pytest.raises(ValidationError, match="equal sample counts"):
        kinematic_trials(recording, reference_clock=None)


def test_invalid_time_options_and_shapes_are_rejected() -> None:
    time = np.arange(4, dtype=float)
    one_dim = as_kinematic_series(_signal(time, time, names=["x"]))

    with pytest.raises(ValidationError, match="strictly increasing"):
        dataclasses.replace(one_dim, time=np.array([[0.0, 1.0, 1.0, 2.0]]))
    with pytest.raises(ValidationError, match="method must be"):
        kinematic_derivative(one_dim, method="diff")
    with pytest.raises(ValidationError, match="boundary must be"):
        kinematic_derivative(one_dim, boundary="zero-pad")
    with pytest.raises(ValidationError, match="smoothing must be"):
        kinematic_derivative(one_dim, smoothing="moving-average")
    with pytest.raises(ValidationError, match="exactly two"):
        direction(one_dim)

    velocity_series = velocity(one_dim)
    with pytest.raises(ValidationError, match="requires position"):
        velocity(velocity_series)


def test_undefined_derivative_quantity_combinations_are_rejected() -> None:
    time = np.arange(5, dtype=float)
    pos = as_kinematic_series(_signal(time, time, names=["x"]), quantity="position")
    velocity_series = velocity(pos)
    acceleration_series = acceleration(pos)
    speed_series = speed(pos)
    planar = as_kinematic_series(
        _signal(np.column_stack((time, time)), time, names=["x", "y"]),
        quantity="position",
    )
    direction_series = direction(planar)

    for series, order in (
        (velocity_series, 2),
        (acceleration_series, 1),
        (pos, 3),
        (speed_series, 1),
        (direction_series, 1),
    ):
        with pytest.raises(ValidationError, match="not defined"):
            kinematic_derivative(series, order=order, nan_policy="propagate")


def test_scalar_quantity_shape_and_direction_unit_are_validated() -> None:
    time = np.arange(4, dtype=float)
    vector = as_kinematic_series(_signal(np.column_stack((time, time)), time, names=["x", "y"]))
    scalar = as_kinematic_series(_signal(time, time, names=["x"]))

    with pytest.raises(ValidationError, match="requires exactly one dimension"):
        dataclasses.replace(vector, quantity="speed")
    with pytest.raises(ValidationError, match="requires exactly one dimension"):
        dataclasses.replace(vector, quantity="direction")
    with pytest.raises(ValidationError, match="requires unit 'rad'"):
        dataclasses.replace(scalar, quantity="direction")


def test_mixed_dimension_units_are_rejected() -> None:
    time = np.arange(4, dtype=float)
    source = _signal(
        np.column_stack((time, time)),
        time,
        names=["x", "angle"],
        units=["m", "rad"],
    )

    with pytest.raises(ValidationError, match="identical dimension units"):
        speed(source)


def test_result_arrays_and_metadata_are_immutable() -> None:
    time = np.arange(4, dtype=float)
    result = velocity(_signal(time, time, names=["x"]))

    with pytest.raises(ValueError, match="read-only"):
        result.data[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        result.time[0, 0] = 1.0
    with pytest.raises(TypeError):
        result.attrs["new"] = "value"
    with pytest.raises(TypeError):
        result.source_attrs["new"] = "value"


def test_module_import_does_not_load_native_extensions() -> None:
    code = """
import importlib
import json
import sys

module = importlib.import_module("neurale.features.kinematics")
print(json.dumps({
    "has_api": hasattr(module, "KinematicSeries"),
    "native": "neurale._native" in sys.modules,
    "cuda": "neurale._native_cuda" in sys.modules,
}))
"""
    result = probe_json(code)
    assert result == {"has_api": True, "native": False, "cuda": False}
