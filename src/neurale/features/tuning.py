#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline typed neural tuning analyses.

The public inputs use observation-major neural data and sample-major behavior.
The first implementation deliberately supports one selected behavioral
dimension per result. Generic tuning values use ``(bin, feature_or_unit)``
order, bin coordinates use ``(bin, behavioral_dimension)``, and occupancy uses
``(bin,)``.

Clock and trial alignment are delegated to :mod:`neurale.data.alignment`.
Binning is deterministic, and tuning analysis performs neither random
subsampling nor decoder fitting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

import numpy as np

from neurale._validation import validate_choice, validate_integer, validate_real_array
from neurale.data import (
    Clock,
    EventSeries,
    FeatureMatrix,
    Recording,
    SignalArray,
    SpikeTrain,
    Trial,
    TrialTable,
    align_neural_behavior_nearest,
    split_recording_trials,
)
from neurale.data._helpers import freeze_metadata, immutable_array_copy
from neurale.exceptions import ValidationError

Normalization = Literal["none", "zscore"]
Smoothing = Literal["none"]
EmptyBinBehavior = Literal["nan"]
NanPolicy = Literal["raise", "omit"]
NeuralSourceKind = Literal["features", "spikes"]
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Immutable binned tuning result.

    ``values`` has shape ``(n_bins, n_features_or_units)``.
    ``bin_coordinates`` has shape ``(n_bins, 1)`` and stores bin centers for
    the selected behavioral dimension. ``occupancy`` counts complete, aligned,
    in-range observations before the minimum-count rule is applied. Metadata
    mapping fields use the closed deep-frozen value domain documented by
    :mod:`neurale.data`.
    """

    values: np.ndarray
    bin_edges: np.ndarray
    bin_coordinates: np.ndarray
    occupancy: np.ndarray
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    feature_channels: tuple[int | None, ...]
    behavior_names: tuple[str, ...]
    behavior_units: tuple[str, ...]
    source_kind: NeuralSourceKind
    source_name: str
    neural_clock: Clock | None
    behavior_clock: Clock | None
    trials: tuple[Trial, ...]
    trial_table_attrs: Mapping[str, object]
    minimum_sample_count: int
    normalization: Normalization
    smoothing: Smoothing
    empty_bin_behavior: EmptyBinBehavior
    source_attrs: Mapping[str, object]
    behavior_attrs: Mapping[str, object]
    attrs: Mapping[str, object]

    def __post_init__(self) -> None:
        values = validate_real_array(np.asarray(self.values), "values", ndim=2, finite=False)
        if np.any(np.isinf(values)):
            raise ValidationError("values must not contain infinite values.")
        edges = validate_real_array(np.asarray(self.bin_edges), "bin_edges", ndim=1, finite=True)
        if edges.size < 2 or np.any(edges[1:] <= edges[:-1]):
            raise ValidationError("bin_edges must contain at least two strictly increasing values.")
        coordinates = validate_real_array(
            np.asarray(self.bin_coordinates), "bin_coordinates", ndim=2, finite=True
        )
        if coordinates.shape != (edges.size - 1, 1):
            raise ValidationError("bin_coordinates must have shape (n_bins, 1).")
        expected_coordinates = _bin_centers(edges)[:, np.newaxis]
        if not _within_ulp_tol(coordinates, expected_coordinates):
            raise ValidationError("bin_coordinates must equal the bin-edge midpoints.")
        coordinates = expected_coordinates
        if values.shape[0] != edges.size - 1:
            raise ValidationError("values bin axis must match bin_edges.")
        if values.shape[1] == 0:
            raise ValidationError("values must contain at least one feature or unit.")

        occupancy = np.asarray(self.occupancy)
        if occupancy.ndim != 1 or occupancy.shape[0] != values.shape[0]:
            raise ValidationError("occupancy must have shape (n_bins,).")
        if not np.issubdtype(occupancy.dtype, np.integer) or np.any(occupancy < 0):
            raise ValidationError("occupancy must contain non-negative integers.")
        if np.any(occupancy > np.iinfo(np.int64).max):
            raise ValidationError("occupancy values must fit in int64.")

        feature_names = _validated_names(self.feature_names, values.shape[1], "feature_names")
        feature_units = _validated_names(
            self.feature_units, values.shape[1], "feature_units", unique=False
        )
        feature_channels = _validated_feature_channels(self.feature_channels, values.shape[1])
        behavior_names = _validated_names(self.behavior_names, 1, "behavior_names")
        behavior_units = _validated_names(self.behavior_units, 1, "behavior_units", unique=False)
        source_kind = validate_choice(self.source_kind, ("features", "spikes"), "source_kind")
        source_name = _non_empty_string(self.source_name, "source_name")
        _require_optional_clock(self.neural_clock, "neural_clock")
        _require_optional_clock(self.behavior_clock, "behavior_clock")
        trials = _validated_trials(self.trials)
        minimum_sample_count = validate_integer(
            self.minimum_sample_count, "minimum_sample_count", minimum=1
        )
        if np.any(np.isfinite(values[occupancy < minimum_sample_count])):
            raise ValidationError(
                "bins below minimum_sample_count must contain only NaN tuning values."
            )
        if np.any(~np.isfinite(values[occupancy >= minimum_sample_count])):
            raise ValidationError(
                "bins meeting minimum_sample_count must contain finite tuning values."
            )
        normalization = validate_choice(self.normalization, ("none", "zscore"), "normalization")
        smoothing = validate_choice(self.smoothing, ("none",), "smoothing")
        empty_bin_behavior = validate_choice(
            self.empty_bin_behavior, ("nan",), "empty_bin_behavior"
        )
        for name, value in (
            ("trial_table_attrs", self.trial_table_attrs),
            ("source_attrs", self.source_attrs),
            ("behavior_attrs", self.behavior_attrs),
            ("attrs", self.attrs),
        ):
            if not isinstance(value, Mapping):
                raise ValidationError(f"{name} must be a mapping.")
        source_feature_units = _validated_names(
            self.attrs.get("source_feature_units"),
            values.shape[1],
            "attrs['source_feature_units']",
            unique=False,
        )
        _validate_source_provenance(source_kind, self.attrs, source_feature_units, trials)
        if normalization == "zscore":
            if feature_units != ("z-score",) * values.shape[1]:
                raise ValidationError("zscore tuning values must use 'z-score' feature_units.")
            _validate_zscore_provenance(self.attrs, values.shape[1], occupancy)
        elif feature_units != source_feature_units:
            raise ValidationError(
                "unnormalized feature_units must match attrs['source_feature_units']."
            )

        object.__setattr__(self, "values", immutable_array_copy(values))
        object.__setattr__(self, "bin_edges", immutable_array_copy(edges))
        object.__setattr__(self, "bin_coordinates", immutable_array_copy(coordinates))
        object.__setattr__(self, "occupancy", immutable_array_copy(occupancy.astype(np.int64)))
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_units", feature_units)
        object.__setattr__(self, "feature_channels", feature_channels)
        object.__setattr__(self, "behavior_names", behavior_names)
        object.__setattr__(self, "behavior_units", behavior_units)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "trial_table_attrs", freeze_metadata(self.trial_table_attrs))
        object.__setattr__(self, "minimum_sample_count", minimum_sample_count)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "empty_bin_behavior", empty_bin_behavior)
        object.__setattr__(self, "source_attrs", freeze_metadata(self.source_attrs))
        object.__setattr__(self, "behavior_attrs", freeze_metadata(self.behavior_attrs))
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @property
    def axis_order(self) -> tuple[str, str]:
        return ("bin", "feature_or_unit")

    @property
    def bin_coordinate_axis_order(self) -> tuple[str, str]:
        return ("bin", "behavioral_dimension")

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class DirectionalTuningResult:
    """Binned directional tuning plus deterministic cosine-fit summaries."""

    tuning: TuningResult
    preferred_direction: np.ndarray
    modulation_depth: np.ndarray
    cosine_coefs: np.ndarray
    r_squared: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.tuning, TuningResult):
            raise ValidationError("tuning must be a TuningResult.")
        if self.tuning.behavior_units != ("rad",):
            raise ValidationError("directional tuning requires behavior unit 'rad'.")
        _validate_directional_tuning_contract(self.tuning)
        n_features = self.tuning.n_features
        preferred = _summary_array(self.preferred_direction, "preferred_direction", n_features)
        modulation = _summary_array(self.modulation_depth, "modulation_depth", n_features)
        r_squared = _summary_array(self.r_squared, "r_squared", n_features)
        finite_preferred = preferred[np.isfinite(preferred)]
        if np.any((finite_preferred < 0.0) | (finite_preferred >= 2.0 * np.pi)):
            raise ValidationError("preferred_direction must be in [0, 2*pi) or NaN.")
        if np.any(modulation[np.isfinite(modulation)] < 0.0):
            raise ValidationError("modulation_depth must be non-negative or NaN.")
        finite_r_squared = r_squared[np.isfinite(r_squared)]
        if np.any((finite_r_squared < 0.0) | (finite_r_squared > 1.0)):
            raise ValidationError("r_squared must be in [0, 1] or NaN.")
        coefs = validate_real_array(
            np.asarray(self.cosine_coefs), "cosine_coefs", ndim=2, finite=False
        )
        if coefs.shape != (n_features, 3) or np.any(np.isinf(coefs)):
            raise ValidationError("cosine_coefs must have shape (n_features, 3) without inf.")
        expected = _cosine_fits(self.tuning)
        for name, actual, derived in (
            ("preferred_direction", preferred, expected[0]),
            ("modulation_depth", modulation, expected[1]),
            ("cosine_coefs", coefs, expected[2]),
            ("r_squared", r_squared, expected[3]),
        ):
            if not np.array_equal(actual, derived, equal_nan=True):
                raise ValidationError(f"{name} must match the summaries derived from tuning.")
        object.__setattr__(self, "preferred_direction", immutable_array_copy(preferred))
        object.__setattr__(self, "modulation_depth", immutable_array_copy(modulation))
        object.__setattr__(self, "cosine_coefs", immutable_array_copy(coefs))
        object.__setattr__(self, "r_squared", immutable_array_copy(r_squared))

    @property
    def values(self) -> np.ndarray:
        return self.tuning.values

    @property
    def bin_coordinates(self) -> np.ndarray:
        return self.tuning.bin_coordinates

    @property
    def occupancy(self) -> np.ndarray:
        return self.tuning.occupancy


def binned_tuning(
    neural: FeatureMatrix | SpikeTrain,
    behavior: SignalArray,
    *,
    behavior_dim: str | int = 0,
    bin_edges: np.ndarray,
    alignment_tol: float,
    neural_clock: Clock | None = None,
    trials: TrialTable | None = None,
    reference_clock: Clock | None = None,
    discontinuities: EventSeries | None = None,
    spike_observation_edges: np.ndarray | None = None,
    minimum_sample_count: int = 1,
    normalization: Normalization = "none",
    smoothing: Smoothing = "none",
    empty_bin_behavior: EmptyBinBehavior = "nan",
    nan_policy: NanPolicy = "raise",
    source_name: str | None = None,
) -> TuningResult:
    """Calculate deterministic 1D binned tuning.

    Behavioral bins are half-open ``[low, high)`` for every bin, including the
    final bin. Out-of-range observations do not contribute. Empty bins and bins
    below ``minimum_sample_count`` contain NaN while retaining their true
    occupancy. ``zscore`` normalization is per neural feature/unit over the
    complete aligned observations that fall inside the requested bin range.

    For :class:`SpikeTrain`, ``spike_observation_edges`` is required. Each
    interval is half-open and becomes one firing-rate observation at its center.
    """

    if not isinstance(behavior, SignalArray):
        raise ValidationError("behavior must be a SignalArray.")
    selected_behavior = _select_behavior(behavior, behavior_dim)
    _require_optional_clock(selected_behavior.clock, "behavior.clock")
    edges = _validated_bin_edges(bin_edges, "bin_edges")
    tol = _non_negative_float(alignment_tol, "alignment_tol")
    _require_optional_clock(neural_clock, "neural_clock")
    _require_optional_clock(reference_clock, "reference_clock")
    if trials is not None and not isinstance(trials, TrialTable):
        raise ValidationError("trials must be a TrialTable or None.")
    if isinstance(neural, SpikeTrain) and trials is not None:
        raise ValidationError(
            "SpikeTrain tuning with trials requires support-aware observation slicing."
        )
    if isinstance(neural, FeatureMatrix) and trials is not None and neural.window_size is not None:
        raise ValidationError(
            "windowed FeatureMatrix tuning with trials requires an explicit feature "
            "time-reference and support-aware slicing."
        )
    if discontinuities is not None and not isinstance(discontinuities, EventSeries):
        raise ValidationError("discontinuities must be an EventSeries or None.")
    if discontinuities is not None:
        _require_optional_clock(discontinuities.clock, "discontinuities.clock")
    n_minimum = validate_integer(minimum_sample_count, "minimum_sample_count", minimum=1)
    normalization = validate_choice(normalization, ("none", "zscore"), "normalization")
    smoothing = validate_choice(smoothing, ("none",), "smoothing")
    empty_bin_behavior = validate_choice(empty_bin_behavior, ("nan",), "empty_bin_behavior")
    nan_policy = validate_choice(nan_policy, ("raise", "omit"), "nan_policy")

    neural_signal, source = _neural_signal(
        neural,
        neural_clock=neural_clock,
        spike_observation_edges=spike_observation_edges,
        source_name=source_name,
    )
    neural_values, behavior_values, selected_trials, trial_table_attrs = _aligned_observations(
        neural_signal,
        selected_behavior,
        tol=tol,
        trials=trials,
        reference_clock=reference_clock,
        discontinuities=discontinuities,
    )
    tuning_values, occupancy, normalization_attrs = _bin_means(
        neural_values,
        behavior_values[:, 0],
        edges,
        minimum_sample_count=n_minimum,
        normalization=normalization,
        nan_policy=nan_policy,
    )
    coordinates = _bin_centers(edges)[:, np.newaxis]
    output_units = (
        ("z-score",) * len(source.feature_units)
        if normalization == "zscore"
        else source.feature_units
    )
    return TuningResult(
        values=tuning_values,
        bin_edges=edges,
        bin_coordinates=coordinates,
        occupancy=occupancy,
        feature_names=source.feature_names,
        feature_units=output_units,
        feature_channels=source.feature_channels,
        behavior_names=(selected_behavior.channels.names[0],),
        behavior_units=(selected_behavior.channels.units[0],),
        source_kind=source.kind,
        source_name=source.name,
        neural_clock=neural_clock,
        behavior_clock=selected_behavior.clock,
        trials=selected_trials,
        trial_table_attrs=trial_table_attrs,
        minimum_sample_count=n_minimum,
        normalization=normalization,
        smoothing=smoothing,
        empty_bin_behavior=empty_bin_behavior,
        source_attrs=source.attrs,
        behavior_attrs=selected_behavior.attrs,
        attrs={
            "axis_order": ("bin", "feature_or_unit"),
            "bin_coordinate_axis_order": ("bin", "behavioral_dimension"),
            "bin_interval": "[low, high)",
            "nan_policy": nan_policy,
            "alignment": "nearest_one_to_one",
            "alignment_tol": tol,
            "aligned_observation_count": int(neural_values.shape[0]),
            "source_feature_units": source.feature_units,
            **source.result_attrs,
            **normalization_attrs,
        },
    )


def directional_tuning(
    neural: FeatureMatrix | SpikeTrain,
    direction: SignalArray,
    *,
    direction_dim: str | int = 0,
    n_bins: int = 8,
    alignment_tol: float,
    neural_clock: Clock | None = None,
    trials: TrialTable | None = None,
    reference_clock: Clock | None = None,
    discontinuities: EventSeries | None = None,
    spike_observation_edges: np.ndarray | None = None,
    minimum_sample_count: int = 1,
    normalization: Normalization = "none",
    smoothing: Smoothing = "none",
    empty_bin_behavior: EmptyBinBehavior = "nan",
    nan_policy: NanPolicy = "raise",
    source_name: str | None = None,
) -> DirectionalTuningResult:
    """Calculate circular directional tuning and a cosine fit per feature.

    Input angles are radians and are wrapped into ``[0, 2*pi)``. At least three
    finite bins are required for a cosine fit. Constant or underdetermined
    profiles retain their binned values but report NaN fit summaries.
    """

    bins = validate_integer(n_bins, "n_bins", minimum=3)
    if not isinstance(direction, SignalArray):
        raise ValidationError("direction must be a SignalArray.")
    selected = _select_behavior(direction, direction_dim)
    _require_optional_clock(selected.clock, "direction.clock")
    if selected.channels.units != ["rad"]:
        raise ValidationError("directional tuning requires behavior unit 'rad'.")
    wrapped = SignalArray.from_array(
        _wrap_direction_angles(np.asarray(selected.data, dtype=float)),
        fs=selected.fs,
        time=selected.time,
        channel_names=selected.channels.names,
        channel_types=[channel.type for channel in selected.channels],
        units=selected.channels.units,
        name=selected.name,
        clock=selected.clock,
        attrs=dict(selected.attrs),
        copy_data=True,
    )
    tuning = binned_tuning(
        neural,
        wrapped,
        behavior_dim=0,
        bin_edges=np.linspace(0.0, 2.0 * np.pi, bins + 1),
        alignment_tol=alignment_tol,
        neural_clock=neural_clock,
        trials=trials,
        reference_clock=reference_clock,
        discontinuities=discontinuities,
        spike_observation_edges=spike_observation_edges,
        minimum_sample_count=minimum_sample_count,
        normalization=normalization,
        smoothing=smoothing,
        empty_bin_behavior=empty_bin_behavior,
        nan_policy=nan_policy,
        source_name=source_name,
    )
    tuning = replace(
        tuning,
        attrs={
            **dict(tuning.attrs),
            "tuning_kind": "directional",
            "circular_period": _TWO_PI,
            "circular_interval": "[0, 2*pi)",
            "fit_weighting": "uniform_valid_bins",
        },
    )
    preferred, modulation, coefs, r_squared = _cosine_fits(tuning)
    return DirectionalTuningResult(
        tuning=tuning,
        preferred_direction=preferred,
        modulation_depth=modulation,
        cosine_coefs=coefs,
        r_squared=r_squared,
    )


@dataclass(frozen=True, slots=True)
class _NeuralSource:
    kind: NeuralSourceKind
    name: str
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    feature_channels: tuple[int | None, ...]
    attrs: Mapping[str, object]
    result_attrs: Mapping[str, object]


def _neural_signal(
    neural: FeatureMatrix | SpikeTrain,
    *,
    neural_clock: Clock | None,
    spike_observation_edges: np.ndarray | None,
    source_name: str | None,
) -> tuple[SignalArray, _NeuralSource]:
    if isinstance(neural, FeatureMatrix):
        if spike_observation_edges is not None:
            raise ValidationError("spike_observation_edges is only valid for SpikeTrain input.")
        if neural.fs is None:
            raise ValidationError("binned tuning requires a regular FeatureMatrix sampling rate.")
        values = validate_real_array(
            np.asarray(neural.data), "FeatureMatrix.data", ndim=2, finite=False
        )
        units = _feature_units(neural)
        name = source_name or neural.source_signal or "features"
        source = _NeuralSource(
            "features",
            name,
            tuple(neural.feature_names),
            units,
            (None,) * neural.n_features,
            neural.attrs,
            {
                "source_fs": neural.fs,
                "source_window_size": neural.window_size,
                "source_shift": neural.shift,
            },
        )
        signal = SignalArray.from_array(
            values,
            fs=neural.fs,
            time=neural.time,
            channel_names=neural.feature_names,
            channel_types="decoder",
            units=units,
            name=name,
            clock=neural_clock,
            attrs=dict(neural.attrs),
            copy_data=True,
        )
        return signal, source
    if not isinstance(neural, SpikeTrain):
        raise ValidationError("neural must be a FeatureMatrix or SpikeTrain.")
    if spike_observation_edges is None:
        raise ValidationError("SpikeTrain input requires spike_observation_edges.")
    edges, widths, observation_rate = _validated_spike_observation_geometry(spike_observation_edges)
    n_units = len(neural.times)
    if n_units == 0:
        raise ValidationError("SpikeTrain must contain at least one unit.")
    unit_names = (
        tuple(neural.units)
        if neural.units is not None
        else tuple(f"spike{idx:03d}" for idx in range(n_units))
    )
    _validated_names(unit_names, n_units, "SpikeTrain unit names")
    rates = np.empty((widths.size, n_units), dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for unit_idx, spike_times in enumerate(neural.times):
            indices = np.searchsorted(edges, spike_times, side="right") - 1
            valid = (indices >= 0) & (indices < widths.size) & (spike_times < edges[-1])
            counts = np.bincount(indices[valid], minlength=widths.size)
            rates[:, unit_idx] = counts / widths
    if not np.all(np.isfinite(rates)):
        raise ValidationError("spike observation geometry produces unrepresentable firing rates.")
    centers = _bin_centers(edges)
    name = source_name or "spikes"
    channels = tuple(neural.channels) if neural.channels is not None else (None,) * n_units
    source = _NeuralSource(
        "spikes",
        name,
        unit_names,
        ("Hz",) * n_units,
        channels,
        neural.attrs,
        {
            "spike_observation_edges": edges,
            "spike_observation_interval": "[start, end)",
            "spike_observation_time_reference": "center",
        },
    )
    signal = SignalArray.from_array(
        rates,
        fs=observation_rate,
        time=centers,
        channel_names=unit_names,
        channel_types="spike",
        units="Hz",
        name=name,
        clock=neural_clock,
        attrs=dict(neural.attrs),
        copy_data=True,
    )
    return signal, source


def _aligned_observations(
    neural: SignalArray,
    behavior: SignalArray,
    *,
    tol: float,
    trials: TrialTable | None,
    reference_clock: Clock | None,
    discontinuities: EventSeries | None,
) -> tuple[np.ndarray, np.ndarray, tuple[Trial, ...], Mapping[str, object]]:
    pairs: list[tuple[SignalArray, SignalArray]] = []
    selected_trials: tuple[Trial, ...] = ()
    trial_table_attrs: Mapping[str, object] = {}
    if trials is None:
        pairs.append((neural, behavior))
    else:
        recording = Recording(
            signals={"tuning_neural": neural, "tuning_behavior": behavior},
            trials=trials,
        )
        alignment_clock = reference_clock or neural.clock or behavior.clock
        epochs = split_recording_trials(
            recording,
            reference_clock=alignment_clock,
            signal_names=("tuning_neural", "tuning_behavior"),
            discontinuities=discontinuities,
        )
        selected_trials = tuple(epoch.trial for epoch in epochs)
        trial_table_attrs = trials.attrs
        pairs.extend(
            (epoch.signals["tuning_neural"], epoch.signals["tuning_behavior"]) for epoch in epochs
        )

    neural_rows: list[np.ndarray] = []
    behavior_rows: list[np.ndarray] = []
    for neural_epoch, behavior_epoch in pairs:
        alignment = align_neural_behavior_nearest(
            neural_epoch,
            behavior_epoch,
            target="neural",
            tol=tol,
            discontinuities=() if trials is not None else discontinuities,
        )
        neural_indices = alignment.neural_indices[alignment.matched]
        behavior_indices = alignment.behavior_indices[alignment.matched]
        if neural_indices.size:
            neural_rows.append(np.asarray(neural_epoch.data)[neural_indices])
            behavior_rows.append(np.asarray(behavior_epoch.data)[behavior_indices])
    if not neural_rows:
        raise ValidationError("neural and behavior inputs have no aligned observations.")
    return (
        np.concatenate(neural_rows),
        np.concatenate(behavior_rows),
        selected_trials,
        trial_table_attrs,
    )


def _bin_means(
    neural: np.ndarray,
    behavior: np.ndarray,
    edges: np.ndarray,
    *,
    minimum_sample_count: int,
    normalization: Normalization,
    nan_policy: NanPolicy,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if np.any(np.isinf(neural)) or np.any(np.isinf(behavior)):
        raise ValidationError("aligned neural and behavior values must not contain infinity.")
    complete = np.isfinite(behavior) & np.all(np.isfinite(neural), axis=1)
    if nan_policy == "raise" and not np.all(complete):
        raise ValidationError("aligned neural or behavior values contain NaN.")
    in_range = (behavior >= edges[0]) & (behavior < edges[-1])
    included = complete & in_range
    included_neural = np.asarray(neural[included], dtype=float)
    normalization_attrs: dict[str, object] = {}
    if normalization == "zscore":
        normalization_attrs = {
            "normalization_center": None,
            "normalization_scale": None,
            "constant_features_normalized_to_zero": (),
        }
        if included_neural.size:
            feature_scale = np.max(np.abs(included_neural), axis=0)
            safe_feature_scale = np.where(feature_scale == 0.0, 1.0, feature_scale)
            scaled_neural = included_neural / safe_feature_scale
            center_scaled = np.mean(scaled_neural, axis=0)
            scale_scaled = np.std(scaled_neural, axis=0)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                center = center_scaled * safe_feature_scale
                scale = scale_scaled * safe_feature_scale
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
                raise ValidationError("zscore normalization exceeds the finite float64 range.")
            if np.any((scale_scaled > 0.0) & (scale == 0.0)):
                raise ValidationError(
                    "zscore normalization scale is not representable as finite float64."
                )
            safe_scale_scaled = np.where(scale_scaled == 0.0, 1.0, scale_scaled)
            included_neural = (scaled_neural - center_scaled) / safe_scale_scaled
            normalization_attrs = {
                "normalization_center": center,
                "normalization_scale": scale,
                "constant_features_normalized_to_zero": tuple(np.flatnonzero(scale == 0.0)),
            }

    bin_indices = np.searchsorted(edges, behavior[included], side="right") - 1
    occupancy = np.bincount(bin_indices, minlength=edges.size - 1).astype(np.int64)
    values = np.full((edges.size - 1, neural.shape[1]), np.nan, dtype=float)
    for bin_idx in range(values.shape[0]):
        if occupancy[bin_idx] >= minimum_sample_count:
            values[bin_idx] = _stable_column_mean(included_neural[bin_indices == bin_idx])
    return values, occupancy, normalization_attrs


def _cosine_fits(
    tuning: TuningResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = tuning.bin_coordinates[:, 0]
    n_features = tuning.n_features
    preferred = np.full(n_features, np.nan)
    modulation = np.full(n_features, np.nan)
    coefs = np.full((n_features, 3), np.nan)
    r_squared = np.full(n_features, np.nan)
    for feature_idx in range(n_features):
        response = tuning.values[:, feature_idx]
        valid = np.isfinite(response)
        if np.count_nonzero(valid) < 3:
            continue
        design = np.column_stack(
            (np.cos(angles[valid]), np.sin(angles[valid]), np.ones(np.count_nonzero(valid)))
        )
        if np.linalg.matrix_rank(design) < 3:
            continue
        response_values = response[valid]
        response_scale = float(np.max(np.abs(response_values)))
        if response_scale == 0.0:
            coefs[feature_idx] = 0.0
            modulation[feature_idx] = 0.0
            continue
        scaled_response = response_values / response_scale
        scaled_coef, _, _, _ = np.linalg.lstsq(design, scaled_response, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            coef = scaled_coef * response_scale
        if not np.all(np.isfinite(coef)):
            raise ValidationError("cosine coefficients exceed the finite float64 range.")
        coefs[feature_idx] = coef
        with np.errstate(over="ignore", invalid="ignore"):
            amp = float(np.hypot(coef[0], coef[1]))
        if not np.isfinite(amp):
            raise ValidationError("directional modulation exceeds the finite float64 range.")
        modulation[feature_idx] = amp
        coef_scale = float(np.max(np.abs(coef)))
        threshold = np.finfo(float).eps * coef_scale * 100.0
        if amp > threshold:
            preferred[feature_idx] = _canonical_preferred_direction(
                float(np.arctan2(coef[1], coef[0]))
            )
        predicted = design @ scaled_coef
        residual = scaled_response - predicted
        centered = scaled_response - np.mean(scaled_response)
        residual_sum = float(np.dot(residual, residual))
        total_sum = float(np.dot(centered, centered))
        if total_sum > 0.0:
            r_squared[feature_idx] = float(np.clip(1.0 - residual_sum / total_sum, 0.0, 1.0))
    return preferred, modulation, coefs, r_squared


def _stable_column_mean(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values), axis=0)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.mean(values / safe_scale, axis=0) * safe_scale
    if not np.all(np.isfinite(result)):
        raise ValidationError("binned tuning mean exceeds the finite float64 range.")
    return result


def _wrap_direction_angles(values: np.ndarray) -> np.ndarray:
    if np.any(np.isinf(values)):
        raise ValidationError("direction values must not contain infinity.")
    with np.errstate(invalid="ignore"):
        wrapped = np.remainder(values, _TWO_PI)
    wrapped = np.asarray(wrapped, dtype=float)
    negative_to_upper_edge = (values < 0.0) & (wrapped >= _TWO_PI)
    wrapped[negative_to_upper_edge] = np.nextafter(_TWO_PI, 0.0)
    wrapped[wrapped >= _TWO_PI] = 0.0
    return wrapped


def _canonical_preferred_direction(value: float) -> float:
    if value < 0.0:
        value += _TWO_PI
    if value >= _TWO_PI:
        return 0.0
    return value


def _select_behavior(behavior: SignalArray, key: str | int) -> SignalArray:
    try:
        selected = behavior.select_channels([key])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError(f"unknown behavior_dimension: {key!r}.") from exc
    return selected


def _validated_bin_edges(value: np.ndarray, name: str) -> np.ndarray:
    edges = validate_real_array(np.asarray(value), name, ndim=1, finite=True)
    if edges.size < 2 or np.any(edges[1:] <= edges[:-1]):
        raise ValidationError(f"{name} must contain at least two strictly increasing values.")
    return edges


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    lower = edges[:-1]
    upper = edges[1:]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        direct = lower + (upper - lower) * 0.5
        fallback = lower * 0.5 + upper * 0.5
    centers = np.where(np.isfinite(direct), direct, fallback)
    if not np.all(np.isfinite(centers)):
        raise ValidationError("bin centers must be representable as finite float64 values.")
    return centers


def _within_ulp_tol(values: np.ndarray, expected: np.ndarray) -> bool:
    if values.shape != expected.shape:
        return False
    toward_zero = np.nextafter(expected, 0.0)
    ulp = np.abs(expected - toward_zero)
    ulp = np.where(expected == 0.0, np.nextafter(0.0, 1.0), ulp)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(values - expected)
    return bool(np.all(np.isfinite(difference)) and np.all(difference <= 2.0 * ulp))


def _validate_directional_tuning_contract(tuning: TuningResult) -> None:
    expected_edges = np.linspace(0.0, _TWO_PI, tuning.n_bins + 1)
    if not _within_ulp_tol(tuning.bin_edges, expected_edges):
        raise ValidationError("directional tuning requires uniform bin edges covering [0, 2*pi).")
    expected_metadata = {
        "tuning_kind": "directional",
        "circular_period": _TWO_PI,
        "circular_interval": "[0, 2*pi)",
        "fit_weighting": "uniform_valid_bins",
    }
    for name, expected in expected_metadata.items():
        if tuning.attrs.get(name) != expected:
            raise ValidationError(f"directional tuning attrs[{name!r}] must equal {expected!r}.")


def _validate_source_provenance(
    source_kind: NeuralSourceKind,
    attrs: Mapping[str, object],
    source_feature_units: tuple[str, ...],
    trials: tuple[Trial, ...],
) -> None:
    if source_kind == "features":
        names = ("source_fs", "source_window_size", "source_shift")
        if any(name not in attrs for name in names):
            raise ValidationError("FeatureMatrix tuning requires complete source geometry.")
        _positive_metadata_float(attrs["source_fs"], "source_fs")
        window_size = _positive_metadata_float(
            attrs["source_window_size"], "source_window_size", optional=True
        )
        _positive_metadata_float(attrs["source_shift"], "source_shift", optional=True)
        if trials and window_size is not None:
            raise ValidationError(
                "windowed FeatureMatrix tuning results cannot contain trials without "
                "support-aware observation slicing."
            )
        return

    if trials:
        raise ValidationError("SpikeTrain tuning results cannot contain trials.")
    if source_feature_units != ("Hz",) * len(source_feature_units):
        raise ValidationError("SpikeTrain tuning source units must be 'Hz'.")
    names = (
        "spike_observation_edges",
        "spike_observation_interval",
        "spike_observation_time_reference",
    )
    if any(name not in attrs for name in names):
        raise ValidationError("SpikeTrain tuning requires complete observation geometry.")
    _validated_spike_observation_geometry(attrs["spike_observation_edges"])
    if attrs["spike_observation_interval"] != "[start, end)":
        raise ValidationError("spike_observation_interval must be '[start, end)'.")
    if attrs["spike_observation_time_reference"] != "center":
        raise ValidationError("spike_observation_time_reference must be 'center'.")


def _validated_spike_observation_geometry(
    value: object,
) -> tuple[np.ndarray, np.ndarray, float]:
    edges = _validated_bin_edges(np.asarray(value), "spike_observation_edges")
    if edges[0] < 0.0:
        raise ValidationError("spike_observation_edges must be non-negative.")
    widths = np.diff(edges)
    reference_width = float(widths[0])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        relative_width_error = np.abs(widths / reference_width - 1.0)
    if not np.all(np.isfinite(relative_width_error)) or np.any(relative_width_error > 1e-9):
        raise ValidationError("spike_observation_edges must define equal-width intervals.")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        observation_rate = 1.0 / reference_width
    if not np.isfinite(observation_rate):
        raise ValidationError("spike observation geometry produces unrepresentable firing rates.")
    return edges, widths, observation_rate


def _positive_metadata_float(value: object, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValidationError(
            f"{name} must be a finite positive number{_optional_suffix(optional)}."
        )
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError(
            f"{name} must be a finite positive number{_optional_suffix(optional)}."
        )
    return result


def _optional_suffix(optional: bool) -> str:
    return " or None" if optional else ""


def _validate_zscore_provenance(
    attrs: Mapping[str, object], n_features: int, occupancy: np.ndarray
) -> None:
    names = (
        "normalization_center",
        "normalization_scale",
        "constant_features_normalized_to_zero",
    )
    if any(name not in attrs for name in names):
        raise ValidationError("zscore tuning requires complete normalization provenance.")
    n_observations = sum(int(value) for value in occupancy)
    center = attrs["normalization_center"]
    scale = attrs["normalization_scale"]
    constant_features = attrs["constant_features_normalized_to_zero"]
    if n_observations == 0:
        if center is not None or scale is not None or constant_features != ():
            raise ValidationError(
                "zscore tuning without in-range observations requires empty provenance."
            )
        return
    center_array = validate_real_array(
        np.asarray(center), "normalization_center", ndim=1, finite=True
    )
    scale_array = validate_real_array(np.asarray(scale), "normalization_scale", ndim=1, finite=True)
    if center_array.shape != (n_features,) or scale_array.shape != (n_features,):
        raise ValidationError("normalization center and scale must have shape (n_features,).")
    if np.any(scale_array < 0.0):
        raise ValidationError("normalization_scale must be non-negative.")
    if not isinstance(constant_features, tuple):
        raise ValidationError("constant_features_normalized_to_zero must be a tuple.")
    normalized_indices = tuple(
        validate_integer(idx, "constant feature index", minimum=0) for idx in constant_features
    )
    if any(idx >= n_features for idx in normalized_indices):
        raise ValidationError("constant feature indices must be smaller than n_features.")
    expected_indices = tuple(int(idx) for idx in np.flatnonzero(scale_array == 0.0))
    if normalized_indices != expected_indices:
        raise ValidationError(
            "constant_features_normalized_to_zero must exactly match the zero normalization scales."
        )


def _feature_units(matrix: FeatureMatrix) -> tuple[str, ...]:
    if isinstance(matrix.unit, str):
        return (matrix.unit,) * matrix.n_features
    return tuple(matrix.unit)


def _validated_names(
    values: tuple[str, ...],
    n_expected: int,
    name: str,
    *,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) != n_expected:
        raise ValidationError(f"{name} must be a tuple containing {n_expected} entries.")
    for value in values:
        _non_empty_string(value, name)
    if unique and len(set(values)) != len(values):
        raise ValidationError(f"{name} must be unique.")
    return values


def _validated_trials(trials: tuple[Trial, ...]) -> tuple[Trial, ...]:
    if not isinstance(trials, tuple) or any(not isinstance(trial, Trial) for trial in trials):
        raise ValidationError("trials must be a tuple of Trial instances.")
    trial_ids = [trial.trial_id for trial in trials]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValidationError("trials must have unique trial IDs.")
    return trials


def _validated_feature_channels(
    channels: tuple[int | None, ...], n_expected: int
) -> tuple[int | None, ...]:
    if not isinstance(channels, tuple) or len(channels) != n_expected:
        raise ValidationError(f"feature_channels must be a tuple containing {n_expected} entries.")
    return tuple(
        None if channel is None else validate_integer(channel, "feature channel", minimum=0)
        for channel in channels
    )


def _summary_array(value: np.ndarray, name: str, n_features: int) -> np.ndarray:
    arr = validate_real_array(np.asarray(value), name, ndim=1, finite=False)
    if arr.shape != (n_features,) or np.any(np.isinf(arr)):
        raise ValidationError(f"{name} must have shape (n_features,) without inf.")
    return arr


def _non_empty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} must be a non-empty string.")
    return value


def _non_negative_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a finite non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite non-negative number.") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be a finite non-negative number.")
    return result


def _require_optional_clock(value: Clock | None, name: str) -> None:
    if value is not None and not isinstance(value, Clock):
        raise ValidationError(f"{name} must be a Clock or None.")


__all__ = [
    "DirectionalTuningResult",
    "TuningResult",
    "binned_tuning",
    "directional_tuning",
]
