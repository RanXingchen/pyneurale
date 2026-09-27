#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline typed kinematic transforms.

Kinematic arrays use the fixed axis order ``(trial, sample, spatial_dimension)``.
Each trial has an explicit timestamp row with shape ``(trial, sample)``. Numerical
derivatives are evaluated independently within each trial and never cross trial
boundaries.

The transforms do not append zero samples or assume one scalar ``dt``. They
retain the source time grid and apply repeated time differentiation for the
explicitly supported typed quantity transitions. Label binning and in-place
outlier clipping are outside this API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_positive_float,
    validate_real_array,
)
from neurale.data import (
    ChannelInfo,
    ChannelTable,
    Clock,
    EventSeries,
    Recording,
    SignalArray,
    Trial,
    split_recording_trials,
)
from neurale.data._helpers import freeze_metadata, immutable_array_copy
from neurale.exceptions import ValidationError

KinematicQuantity = Literal[
    "trajectory",
    "position",
    "velocity",
    "acceleration",
    "speed",
    "direction",
]
DerivativeMethod = Literal["gradient"]
BoundaryMode = Literal["one-sided"]
SmoothingMode = Literal["none"]
NanPolicy = Literal["raise", "propagate"]

_QUANTITIES = (
    "trajectory",
    "position",
    "velocity",
    "acceleration",
    "speed",
    "direction",
)


@dataclass(frozen=True, slots=True)
class KinematicSeries:
    """Typed trial-by-sample kinematic data.

    ``data`` has shape ``(trial, sample, spatial_dimension)`` and ``time`` has
    shape ``(trial, sample)``. ``fs`` is preserved as nominal source
    metadata; derivatives always use the explicit timestamps, including for an
    irregular but strictly increasing time grid. ``trials`` retains complete
    typed trial metadata; ``trial_ids`` mirrors those values and contains
    ``None`` for the single ungrouped trajectory. ``source_attrs`` retains the
    selected signal's metadata separately from result-operation ``attrs``.

    NaN values are representable so a transform can apply its explicit
    ``nan_policy``. Infinite values are never accepted. ``source_attrs`` and
    ``attrs`` use the closed deep-frozen metadata value domain documented by
    :mod:`neurale.data`.
    """

    data: np.ndarray
    time: np.ndarray
    fs: float
    clock: Clock | None
    dims: ChannelTable
    trial_ids: tuple[int | None, ...]
    trials: tuple[Trial | None, ...]
    quantity: KinematicQuantity
    source_signal: str
    source_attrs: Mapping[str, object]
    attrs: Mapping[str, object]

    def __post_init__(self) -> None:
        data = validate_real_array(self.data, "KinematicSeries.data", ndim=3, finite=False)
        if np.any(np.isinf(data)):
            raise ValidationError("KinematicSeries.data must not contain infinite values.")
        time = validate_real_array(self.time, "KinematicSeries.time", ndim=2, finite=True)
        if time.shape != data.shape[:2]:
            raise ValidationError(
                "KinematicSeries.time must match the trial and sample axes of data."
            )
        if data.shape[0] == 0:
            raise ValidationError("KinematicSeries requires at least one trial.")
        if data.shape[2] == 0:
            raise ValidationError("KinematicSeries requires at least one spatial dimension.")
        rate = validate_positive_float(self.fs, "fs")
        if self.clock is not None and not isinstance(self.clock, Clock):
            raise ValidationError("clock must be a Clock or None.")
        if not isinstance(self.dims, ChannelTable) or len(self.dims) != data.shape[2]:
            raise ValidationError(
                "dimensions must be a ChannelTable matching the spatial-dimension axis."
            )
        if not isinstance(self.trial_ids, tuple) or len(self.trial_ids) != data.shape[0]:
            raise ValidationError("trial_ids must be a tuple matching the trial axis.")
        concrete_ids: list[int] = []
        normalized_ids: list[int | None] = []
        for trial_id in self.trial_ids:
            if trial_id is None:
                normalized_ids.append(None)
                continue
            value = validate_integer(trial_id, "trial_id", minimum=0)
            concrete_ids.append(value)
            normalized_ids.append(value)
        if len(concrete_ids) != len(set(concrete_ids)):
            raise ValidationError("non-None trial_ids must be unique.")
        if len(normalized_ids) > 1 and any(value is None for value in normalized_ids):
            raise ValidationError("None trial_id is only valid for a single ungrouped trial.")
        if not isinstance(self.trials, tuple) or len(self.trials) != data.shape[0]:
            raise ValidationError("trials must be a tuple matching the trial axis.")
        normalized_trials: list[Trial | None] = []
        for i, trial in enumerate(self.trials):
            if trial is None:
                if normalized_ids[i] is not None:
                    raise ValidationError(
                        "trial_ids must be None when the corresponding trial is None."
                    )
            elif not isinstance(trial, Trial):
                raise ValidationError("trials must contain Trial instances or None.")
            elif normalized_ids[i] != trial.trial_id:
                raise ValidationError(
                    "trial_ids must match the corresponding Trial.trial_id values."
                )
            normalized_trials.append(trial)
        quantity = validate_choice(self.quantity, _QUANTITIES, "quantity")
        if quantity in ("speed", "direction") and data.shape[2] != 1:
            raise ValidationError(f"quantity={quantity!r} requires exactly one dimension.")
        if quantity == "direction" and self.dims.units != ["rad"]:
            raise ValidationError("quantity='direction' requires unit 'rad'.")
        if not isinstance(self.source_signal, str) or not self.source_signal:
            raise ValidationError("source_signal must be a non-empty string.")
        if time.shape[1] > 1 and np.any(np.diff(time, axis=1) <= 0.0):
            raise ValidationError("time must be strictly increasing within every trial.")
        if not isinstance(self.source_attrs, Mapping):
            raise ValidationError("source_attrs must be a mapping.")
        if not isinstance(self.attrs, Mapping):
            raise ValidationError("attrs must be a mapping.")

        object.__setattr__(self, "data", immutable_array_copy(data))
        object.__setattr__(self, "time", immutable_array_copy(time))
        object.__setattr__(self, "fs", rate)
        object.__setattr__(self, "trial_ids", tuple(normalized_ids))
        object.__setattr__(self, "trials", tuple(normalized_trials))
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "source_attrs", freeze_metadata(self.source_attrs))
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @property
    def n_trials(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_dims(self) -> int:
        return int(self.data.shape[2])

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return tuple(self.dims.names)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(self.dims.units)


def as_kinematic_series(
    source: SignalArray | Recording,
    *,
    signal_name: str = "position",
    quantity: KinematicQuantity = "position",
    trial: Trial | None = None,
) -> KinematicSeries:
    """Convert one typed signal into a one-trial kinematic series."""
    signal = _selected_signal(source, signal_name)
    signal.validate()
    if signal.clock is not None and not isinstance(signal.clock, Clock):
        raise ValidationError("signal.clock must be a Clock or None.")
    quantity = validate_choice(quantity, _QUANTITIES, "quantity")
    if trial is not None and not isinstance(trial, Trial):
        raise ValidationError("trial must be a Trial or None.")
    return KinematicSeries(
        data=np.asarray(signal.data)[np.newaxis, :, :],
        time=np.asarray(signal.time)[np.newaxis, :],
        fs=signal.fs,
        clock=signal.clock,
        dims=signal.channels,
        trial_ids=(None if trial is None else trial.trial_id,),
        trials=(trial,),
        quantity=quantity,
        source_signal=signal.name,
        source_attrs=signal.attrs,
        attrs={"axis_order": ("trial", "sample", "dimension")},
    )


def kinematic_trials(
    recording: Recording,
    *,
    signal_name: str = "position",
    quantity: KinematicQuantity = "position",
    reference_clock: Clock | None,
    discontinuities: EventSeries | None = None,
) -> KinematicSeries:
    """Extract equal-length typed trial trajectories through the shared trial-slicing API."""
    if not isinstance(recording, Recording):
        raise ValidationError("recording must be a Recording.")
    quantity = validate_choice(quantity, _QUANTITIES, "quantity")
    epochs = split_recording_trials(
        recording,
        reference_clock=reference_clock,
        signal_names=(signal_name,),
        discontinuities=discontinuities,
    )
    if not epochs:
        raise ValidationError("recording must contain at least one trial.")
    signals = [epoch.signals[signal_name] for epoch in epochs]
    first = signals[0]
    for signal in signals:
        signal.validate()
        if signal.clock != first.clock:
            raise ValidationError("all trial signals must use the same clock.")
        if signal.fs != first.fs:
            raise ValidationError("all trial signals must use the same sampling rate.")
        if signal.channels != first.channels:
            raise ValidationError("all trial signals must use the same dimensions.")
        if signal.n_samples != first.n_samples:
            raise ValidationError("kinematic trials must have equal sample counts.")
    return KinematicSeries(
        data=np.stack([signal.data for signal in signals], axis=0),
        time=np.stack([signal.time for signal in signals], axis=0),
        fs=first.fs,
        clock=first.clock,
        dims=first.channels,
        trial_ids=tuple(epoch.trial.trial_id for epoch in epochs),
        trials=tuple(epoch.trial for epoch in epochs),
        quantity=quantity,
        source_signal=first.name,
        source_attrs=first.attrs,
        attrs={
            "recording_metadata": recording.metadata,
            "axis_order": ("trial", "sample", "dimension"),
        },
    )


def kinematic_derivative(
    source: KinematicSeries | SignalArray | Recording,
    order: int = 1,
    *,
    signal_name: str = "position",
    input_quantity: KinematicQuantity | None = None,
    method: DerivativeMethod = "gradient",
    boundary: BoundaryMode = "one-sided",
    smoothing: SmoothingMode = "none",
    nan_policy: NanPolicy = "raise",
) -> KinematicSeries:
    """Differentiate each trial on its explicit time grid.

    The only first-version method is NumPy-style gradient differentiation:
    central differences for interior samples and one-sided differences at both
    boundaries. First derivatives use first-order boundary estimates. Second
    derivatives use repeated gradients with second-order boundary estimates.
    No smoothing is performed. Irregular time is supported because each
    trial's timestamp vector is passed directly to the derivative. With
    ``nan_policy='propagate'``, NumPy propagation is retained; the default
    rejects NaNs before calculation.
    """
    series = _coerce_series(source, signal_name, input_quantity)
    derivative_order = validate_integer(order, "order", minimum=1)
    _validate_transform_options(method, boundary, smoothing, nan_policy)
    output_quantity = _derivative_quantity(series.quantity, derivative_order)
    _require_derivative_input(series, nan_policy, derivative_order)

    edge_order = 2 if derivative_order == 2 else 1
    values = np.asarray(series.data, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(derivative_order):
            values = np.stack(
                [
                    np.gradient(
                        values[idx],
                        series.time[idx],
                        axis=0,
                        edge_order=edge_order,
                    )
                    for idx in range(series.n_trials)
                ],
                axis=0,
            )
    if np.any(np.isinf(values)) or (nan_policy == "raise" and np.any(np.isnan(values))):
        raise ValidationError("kinematic derivative produced non-finite values.")
    return _derived_series(
        series,
        values,
        dims=_derivative_dimensions(series.dims, derivative_order),
        quantity=output_quantity,
        operation="derivative",
        operation_attrs={
            "derivative_order": derivative_order,
            "derivative_method": method,
            "boundary": boundary,
            "smoothing": smoothing,
            "nan_policy": nan_policy,
        },
    )


def velocity(
    source: KinematicSeries | SignalArray | Recording,
    *,
    signal_name: str = "position",
    input_quantity: KinematicQuantity | None = None,
    nan_policy: NanPolicy = "raise",
) -> KinematicSeries:
    """Return sample-aligned velocity from position or trajectory data."""
    series = _coerce_series(source, signal_name, input_quantity)
    if series.quantity not in ("position", "trajectory"):
        raise ValidationError("velocity requires position or trajectory input.")
    return kinematic_derivative(series, order=1, nan_policy=nan_policy)


def acceleration(
    source: KinematicSeries | SignalArray | Recording,
    *,
    signal_name: str = "position",
    input_quantity: KinematicQuantity | None = None,
    nan_policy: NanPolicy = "raise",
) -> KinematicSeries:
    """Return sample-aligned acceleration from position/trajectory or velocity."""
    series = _coerce_series(source, signal_name, input_quantity)
    if series.quantity in ("position", "trajectory"):
        return kinematic_derivative(series, order=2, nan_policy=nan_policy)
    if series.quantity == "velocity":
        return kinematic_derivative(series, order=1, nan_policy=nan_policy)
    raise ValidationError("acceleration requires position, trajectory, or velocity input.")


def speed(
    source: KinematicSeries | SignalArray | Recording,
    *,
    signal_name: str = "position",
    input_quantity: KinematicQuantity | None = None,
    nan_policy: NanPolicy = "raise",
) -> KinematicSeries:
    """Return Euclidean speed with shape ``(trial, sample, 1)``."""
    series = _velocity_input(source, signal_name, input_quantity, nan_policy)
    _require_compatible_vector_units(series)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.hypot.reduce(series.data, axis=2, keepdims=True)
    if np.any(np.isinf(values)):
        raise ValidationError("kinematic speed exceeds the finite float64 range.")
    unit = series.units[0]
    dims = ChannelTable(
        [
            ChannelInfo(
                "speed", 0, "behavior", unit, attrs={"source_dimensions": series.dimension_names}
            )
        ]
    )
    return _derived_series(
        series,
        values,
        dims=dims,
        quantity="speed",
        operation="speed",
        operation_attrs={"norm": "euclidean", "nan_policy": nan_policy},
    )


def direction(
    source: KinematicSeries | SignalArray | Recording,
    *,
    signal_name: str = "position",
    input_quantity: KinematicQuantity | None = None,
    nan_policy: NanPolicy = "raise",
) -> KinematicSeries:
    """Return planar direction ``atan2(v_y, v_x)`` in ``[0, 2*pi)`` radians.

    Exactly two spatial dimensions are required. Stationary samples have no
    defined movement direction and are represented by NaN.
    """
    series = _velocity_input(source, signal_name, input_quantity, nan_policy)
    if series.n_dims != 2:
        raise ValidationError("direction requires exactly two spatial dimensions.")
    _require_compatible_vector_units(series)
    planar = np.mod(np.arctan2(series.data[:, :, 1], series.data[:, :, 0]), 2.0 * np.pi)
    stationary = np.all(series.data == 0.0, axis=2)
    planar[stationary] = np.nan
    values = planar[:, :, np.newaxis]
    dims = ChannelTable(
        [
            ChannelInfo(
                "direction",
                0,
                "behavior",
                "rad",
                attrs={"source_dimensions": series.dimension_names},
            )
        ]
    )
    return _derived_series(
        series,
        values,
        dims=dims,
        quantity="direction",
        operation="direction",
        operation_attrs={
            "definition": "atan2(v_y, v_x) in [0, 2*pi); stationary is NaN",
            "nan_policy": nan_policy,
        },
    )


def _selected_signal(source: SignalArray | Recording, signal_name: str) -> SignalArray:
    if not isinstance(signal_name, str) or not signal_name:
        raise ValidationError("signal_name must be a non-empty string.")
    if isinstance(source, SignalArray):
        return source
    if isinstance(source, Recording):
        try:
            return source.signal(signal_name)
        except KeyError as exc:
            raise ValidationError(f"recording does not contain signal {signal_name!r}.") from exc
    raise ValidationError("source must be a KinematicSeries, SignalArray, or Recording.")


def _coerce_series(
    source: KinematicSeries | SignalArray | Recording,
    signal_name: str,
    input_quantity: KinematicQuantity | None,
) -> KinematicSeries:
    if isinstance(source, KinematicSeries):
        if input_quantity is not None and input_quantity != source.quantity:
            raise ValidationError("input_quantity must match KinematicSeries.quantity.")
        return source
    quantity = "position" if input_quantity is None else input_quantity
    return as_kinematic_series(source, signal_name=signal_name, quantity=quantity)


def _validate_transform_options(
    method: DerivativeMethod,
    boundary: BoundaryMode,
    smoothing: SmoothingMode,
    nan_policy: NanPolicy,
) -> None:
    validate_choice(method, ("gradient",), "method")
    validate_choice(boundary, ("one-sided",), "boundary")
    validate_choice(smoothing, ("none",), "smoothing")
    validate_choice(nan_policy, ("raise", "propagate"), "nan_policy")


def _require_derivative_input(
    series: KinematicSeries,
    nan_policy: NanPolicy,
    derivative_order: int,
) -> None:
    required_samples = 3 if derivative_order == 2 else 2
    if series.n_samples < required_samples:
        raise ValidationError(
            f"derivative order {derivative_order} requires at least "
            f"{required_samples} samples per trial."
        )
    _require_no_nan(series, nan_policy)


def _require_no_nan(series: KinematicSeries, nan_policy: NanPolicy) -> None:
    if nan_policy == "raise" and np.any(np.isnan(series.data)):
        raise ValidationError("kinematic data contains NaN; use nan_policy='propagate' explicitly.")


def _velocity_input(
    source: KinematicSeries | SignalArray | Recording,
    signal_name: str,
    input_quantity: KinematicQuantity | None,
    nan_policy: NanPolicy,
) -> KinematicSeries:
    series = _coerce_series(source, signal_name, input_quantity)
    if series.quantity == "velocity":
        _validate_transform_options("gradient", "one-sided", "none", nan_policy)
        _require_no_nan(series, nan_policy)
        return series
    if series.quantity in ("position", "trajectory"):
        return velocity(series, nan_policy=nan_policy)
    raise ValidationError("speed and direction require position, trajectory, or velocity input.")


def _derivative_quantity(quantity: KinematicQuantity, order: int) -> KinematicQuantity:
    if quantity in ("position", "trajectory"):
        if order == 1:
            return "velocity"
        if order == 2:
            return "acceleration"
    if quantity == "velocity" and order == 1:
        return "acceleration"
    raise ValidationError(
        f"derivative order {order} is not defined for quantity={quantity!r}; "
        "the first-version typed contract supports position/trajectory orders 1-2 "
        "and velocity order 1 only."
    )


def _derivative_dimensions(dims: ChannelTable, order: int) -> ChannelTable:
    return ChannelTable([replace(dim, unit=_derivative_unit(dim.unit, order)) for dim in dims])


def _derivative_unit(unit: str, order: int) -> str:
    if order == 1 and unit.endswith("/s") and not unit.endswith("/s^2"):
        return unit.removesuffix("/s") + "/s^2"
    suffix = "/s" if order == 1 else f"/s^{order}"
    return unit + suffix


def _require_compatible_vector_units(series: KinematicSeries) -> None:
    if len(set(series.units)) != 1:
        raise ValidationError("vector magnitude/direction requires identical dimension units.")


def _derived_series(
    source: KinematicSeries,
    values: np.ndarray,
    *,
    dims: ChannelTable,
    quantity: KinematicQuantity,
    operation: str,
    operation_attrs: Mapping[str, object],
) -> KinematicSeries:
    return KinematicSeries(
        data=np.asarray(values, dtype=float),
        time=source.time,
        fs=source.fs,
        clock=source.clock,
        dims=dims,
        trial_ids=source.trial_ids,
        trials=source.trials,
        quantity=quantity,
        source_signal=source.source_signal,
        source_attrs=source.source_attrs,
        attrs={
            **dict(source.attrs),
            "operation": operation,
            **dict(operation_attrs),
            "axis_order": ("trial", "sample", "dimension"),
        },
    )


__all__ = [
    "KinematicSeries",
    "acceleration",
    "as_kinematic_series",
    "direction",
    "kinematic_derivative",
    "kinematic_trials",
    "speed",
    "velocity",
]
