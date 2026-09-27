#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Array-backed signal, feature, and spike data structures."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

import numpy as np

import neurale.data.channels
import neurale.data.time
from neurale._validation import validate_integer
from neurale.exceptions import ValidationError

from ._helpers import (
    as_1d_float_array,
    as_2d_array,
    build_sample_time_grid,
    copy_attrs,
    ensure_finite_float,
    ensure_monotonic_non_decreasing,
    ensure_non_empty_string,
    ensure_positive_float,
    freeze_metadata,
    immutable_array_copy,
    is_strongly_immutable,
    validate_unit,
)
from ._time_validation import sampled_times_match_rate
from .channels import ChannelInfo, ChannelTable
from .time import Clock


@dataclass(frozen=True, slots=True)
class SignalArray:
    """Store continuous sampled data with channel and timing metadata.

    Parameters
    ----------
    data : numpy.ndarray
        2D array with shape ``(n_samples, n_channels)``.
    fs : float
        Sampling rate in hertz.
    time : numpy.ndarray or None
        Optional timestamp vector with one value per sample. When omitted,
        timestamps are generated from ``t0`` and ``fs``.
    t0 : float or None
        Time of the first sample when timestamps are derived from ``fs``.
    clock : neurale.data.time.Clock or None
        neurale.data.time.Clock metadata associated with the signal.
    channels : neurale.data.channels.ChannelTable
        Metadata for each column in ``data``.
    unit : str or list of str
        Shared unit or one unit per channel.
    name : str
        Non-empty signal name.
    attrs : dict, optional
        Deep-frozen application metadata using the closed value domain
        documented by :mod:`neurale.data`.
    copy_data : bool, default=False
        Copy the data buffer when ``True``. The default borrows the supplied
        array to preserve zero-copy and explicit in-place workflows.

    Raises
    ------
    neurale.exceptions.ValidationError
        If array dimensions, timing information, channels, or units are invalid.

    Notes
    -----
    ``fs`` is required. ``time`` is always normalized to a timestamp
    vector after construction.
    """

    data: np.ndarray
    fs: float
    time: np.ndarray | None
    t0: float | None
    clock: neurale.data.time.Clock | None
    channels: neurale.data.channels.ChannelTable
    unit: str | tuple[str, ...]
    name: str
    attrs: Mapping[str, Any] = field(default_factory=dict)
    copy_data: bool = field(default=False, repr=False, compare=False)
    _validated_shape: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.copy_data, bool):
            raise ValidationError("copy_data must be a bool.")
        data = as_2d_array(self.data, "SignalArray.data")
        if self.copy_data:
            data = np.array(data, copy=True, order="K")
        fs = ensure_positive_float(self.fs, "fs")
        if fs is None:
            raise ValidationError("fs must be provided.")
        t0 = 0.0 if self.t0 is None else ensure_finite_float(self.t0, "t0")
        name = ensure_non_empty_string(self.name, "signal name")

        if len(self.channels) != data.shape[1]:
            raise ValidationError("channels length must match data.shape[1].")
        unit = validate_unit(self.unit, data.shape[1], "unit")
        expected_units = [unit] * data.shape[1] if isinstance(unit, str) else list(unit)
        if self.channels.units != expected_units:
            raise ValidationError("SignalArray unit metadata must match channel units.")

        time = build_sample_time_grid(self.time, t0, data.shape[0], fs)

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(
            self,
            "unit",
            unit if isinstance(unit, str) else tuple(unit),
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))
        object.__setattr__(self, "_validated_shape", data.shape)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def channel_names(self) -> list[str]:
        return self.channels.names

    @property
    def duration(self) -> float:
        if self.time.size < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def copy(self, *, data: np.ndarray | None = None) -> SignalArray:
        """Return a copy of the signal, optionally replacing its data.

        Parameters
        ----------
        data : numpy.ndarray or None, optional
            Replacement data. If omitted, the existing array is copied.

        Returns
        -------
        neurale.data.arrays.SignalArray
            New signal with copied metadata and channel information.

        Raises
        ------
        neurale.exceptions.ValidationError
            If replacement data is incompatible with the existing metadata.
        """
        return SignalArray(
            data=self.data.copy() if data is None else data,
            fs=self.fs,
            time=self.time.copy(),
            t0=self.t0,
            clock=self.clock,
            channels=self.channels.copy(),
            unit=self.unit,
            name=self.name,
            attrs=dict(self.attrs),
            copy_data=False,
        )

    def validate(self) -> None:
        """Validate that borrowed buffers still satisfy signal invariants.

        This check is useful at API and persistence boundaries when callers
        retain ownership of the writable ``data`` array.

        Raises
        ------
        neurale.exceptions.ValidationError
            If data shape, time, channels, units, or metadata are inconsistent.
        """
        data = as_2d_array(self.data, "SignalArray.data")
        if data.shape != self._validated_shape:
            raise ValidationError(
                "SignalArray.data shape changed after construction; "
                f"expected {self._validated_shape}, got {data.shape}."
            )
        if len(self.channels) != data.shape[1]:
            raise ValidationError("channels length must match data.shape[1].")
        expected_units = (
            [self.unit] * data.shape[1] if isinstance(self.unit, str) else list(self.unit)
        )
        if self.channels.units != expected_units:
            raise ValidationError("SignalArray unit metadata must match channel units.")
        if self.fs is None:
            raise ValidationError("fs must be provided.")
        if self.time.size != data.shape[0]:
            raise ValidationError("time length must match data.shape[0].")
        ensure_monotonic_non_decreasing(self.time, "time")

    def with_data(
        self,
        data: np.ndarray,
        *,
        copy_data: bool = False,
    ) -> SignalArray:
        """Return a validated signal with a replacement data buffer.

        Parameters
        ----------
        data : numpy.ndarray
            Replacement sample-by-channel data.
        copy_data : bool, default=False
            Copy ``data`` before storing it when ``True``.

        Returns
        -------
        neurale.data.arrays.SignalArray
            New signal sharing existing metadata with the replacement data.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``data`` is incompatible with the existing metadata.
        """
        return self.replace(data=data, copy_data=copy_data)

    def replace(
        self,
        *,
        copy_data: bool = False,
        **changes: Any,
    ) -> SignalArray:
        """Return a validated copy with selected fields replaced.

        Parameters
        ----------
        copy_data : bool, default=False
            Copy the selected data buffer before storing it when ``True``.
        **changes : Any
            Field replacements accepted by :class:`SignalArray`.

        Returns
        -------
        neurale.data.arrays.SignalArray
            New validated signal with updated fields.

        Raises
        ------
        TypeError
            If an unknown field name is supplied.
        neurale.exceptions.ValidationError
            If the resulting signal fields are inconsistent.
        """
        values = {
            "data": self.data,
            "fs": self.fs,
            "time": self.time,
            "t0": self.t0,
            "clock": self.clock,
            "channels": self.channels,
            "unit": self.unit,
            "name": self.name,
            "attrs": self.attrs,
        }
        unknown = set(changes) - set(values)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown SignalArray fields: {names}.")
        values.update(changes)
        return SignalArray(**values, copy_data=copy_data)

    def select_channels(self, keys: Sequence[int | str] | np.ndarray) -> SignalArray:
        """Select channels by index, name, or boolean mask.

        Parameters
        ----------
        keys : sequence of int or str, or numpy.ndarray
            Channel indices, channel names, or a boolean mask in table order.

        Returns
        -------
        neurale.data.arrays.SignalArray
            New signal containing the selected channels in requested order.

        Raises
        ------
        KeyError
            If a requested channel name or index is unknown.
        neurale.exceptions.ValidationError
            If a boolean mask has an invalid shape.
        """
        positions = self.channels.positions(keys)
        unit: str | tuple[str, ...]
        if not isinstance(self.unit, str):
            unit = tuple(self.unit[pos] for pos in positions)
        else:
            unit = self.unit
        return SignalArray(
            data=self.data[:, positions],
            fs=self.fs,
            time=self.time.copy(),
            t0=self.t0,
            clock=self.clock,
            channels=self.channels.select(keys),
            unit=unit,
            name=self.name,
            attrs=dict(self.attrs),
        )

    def select_time(self, start: float, stop: float) -> SignalArray:
        """Select samples in the absolute half-open interval ``[start, stop)``.

        Parameters
        ----------
        start : float
            Absolute start time in seconds, inclusive.
        stop : float
            Absolute stop time in seconds, exclusive.

        Returns
        -------
        neurale.data.arrays.SignalArray
            New signal containing samples in the selected interval.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the interval is invalid or selects no samples.
        """
        start = ensure_finite_float(start, "start")
        stop = ensure_finite_float(stop, "stop")
        if stop <= start:
            raise ValidationError("stop must be greater than start.")
        times = self.time
        positions = np.flatnonzero((times >= start) & (times < stop))
        if positions.size == 0:
            raise ValidationError("selected time interval contains no samples.")
        first = int(positions[0])
        last = int(positions[-1]) + 1
        return SignalArray(
            data=self.data[first:last].copy(),
            fs=self.fs,
            time=self.time[first:last].copy(),
            t0=float(times[first]),
            clock=self.clock,
            channels=self.channels.copy(),
            unit=self.unit,
            name=self.name,
            attrs=dict(self.attrs),
        )

    def crop(
        self,
        tmin: float = 0.0,
        tmax: float | None = None,
    ) -> SignalArray:
        """Crop using seconds relative to the first sample.

        Parameters
        ----------
        tmin : float, default=0.0
            Start time in seconds relative to the first sample.
        tmax : float or None, optional
            Stop time in seconds relative to the first sample. When omitted,
            the crop extends through the last sample.

        Returns
        -------
        neurale.data.arrays.SignalArray
            Cropped signal.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the crop range is invalid or selects no samples.
        """
        if self.n_samples == 0:
            raise ValidationError("cannot crop an empty signal.")
        tmin = ensure_finite_float(tmin, "tmin")
        if tmin < 0:
            raise ValidationError("tmin must be non-negative.")
        start = float(self.time[0]) + tmin
        if tmax is None:
            stop = float(np.nextafter(self.time[-1], np.inf))
        else:
            tmax = ensure_finite_float(tmax, "tmax")
            if tmax <= tmin:
                raise ValidationError("tmax must be greater than tmin.")
            start_time = float(self.time[0])
            stop = start_time + tmax
        return self.select_time(start, stop)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        *,
        fs: float,
        time: np.ndarray | None = None,
        t0: float | None = 0.0,
        channel_names: Sequence[str] | None = None,
        channel_types: str | Sequence[str] = "aux",
        units: str | Sequence[str] = "a.u.",
        name: str = "signal",
        clock: Clock | None = None,
        attrs: dict[str, Any] | None = None,
        copy_data: bool = False,
    ) -> SignalArray:
        """Construct a signal while generating channel metadata.

        Parameters
        ----------
        data : numpy.ndarray
            One- or two-dimensional signal data. 1D input is
            treated as a single channel.
        fs : float
            Sampling rate in hertz.
        time : numpy.ndarray or None, optional
            Optional timestamp vector with one value per sample.
        t0 : float or None, default=0.0
            Time of the first sample when timestamps are generated.
        channel_names : sequence of str or None, optional
            Channel names. Stable names are generated when omitted.
        channel_types : str or sequence of str, default="aux"
            Shared channel type or one type per channel.
        units : str or sequence of str, default="a.u."
            Shared unit or one unit per channel.
        name : str, default="signal"
            Signal name.
        clock : neurale.data.time.Clock or None, optional
            Clock metadata associated with the signal.
        attrs : dict or None, optional
            Additional application-specific metadata.
        copy_data : bool, default=False
            Copy the data buffer before storing it when ``True``.

        Returns
        -------
        neurale.data.arrays.SignalArray
            Constructed signal with generated channel metadata.

        Raises
        ------
        neurale.exceptions.ValidationError
            If array dimensions, metadata lengths, timing, or units are invalid.
        """
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValidationError("data must be a one- or two-dimensional array.")
        n_channels = arr.shape[1]
        if channel_names is None:
            width = max(3, len(str(max(n_channels - 1, 0))))
            names = [f"ch{idx:0{width}d}" for idx in range(n_channels)]
        else:
            names = list(channel_names)
            if len(names) != n_channels:
                raise ValidationError("channel_names length must match the channel count.")
        types = (
            [channel_types] * n_channels if isinstance(channel_types, str) else list(channel_types)
        )
        unit_values = [units] * n_channels if isinstance(units, str) else list(units)
        if len(types) != n_channels:
            raise ValidationError("channel_types length must match the channel count.")
        if len(unit_values) != n_channels:
            raise ValidationError("units length must match the channel count.")
        channels = ChannelTable(
            ChannelInfo(
                name=names[idx],
                index=idx,
                type=types[idx],
                unit=unit_values[idx],
            )
            for idx in range(n_channels)
        )
        return cls(
            data=arr,
            fs=fs,
            time=time,
            t0=t0,
            clock=clock,
            channels=channels,
            unit=units if isinstance(units, str) else unit_values,
            name=name,
            attrs={} if attrs is None else attrs,
            copy_data=copy_data,
        )


@dataclass(slots=True)
class FeatureMatrix:
    """Store frame-wise features derived from signals, spikes, or behavior.

    Parameters
    ----------
    data : numpy.ndarray
        2D array with shape ``(n_frames, n_features)``.
    fs : float or None
        Feature-frame rate in hertz for regularly timed observations. Use
        ``None`` for irregular observations and provide explicit ``time``.
        When both are provided, ``time`` must match the declared rate.
    time : numpy.ndarray or None, optional
        Optional timestamp vector with one value per frame. When omitted,
        timestamps are generated from ``t0`` and ``fs``.
    t0 : float or None
        Time of the first frame when timestamps are derived from ``fs``.
    feature_names : list of str, optional
        Unique feature names. Stable names are generated when omitted.
    source_signal : str or None, optional
        Name of the signal from which features were derived.
    window_size : float or None, optional
        Analysis-window duration in seconds.
    shift : float or None, optional
        Time shift between adjacent feature frames in seconds.
    unit : str or list of str, default="a.u."
        Shared unit or one unit per feature.
    attrs : dict, optional
        Additional application-specific metadata.

    Raises
    ------
    neurale.exceptions.ValidationError
        If data, names, timing information, units, or window metadata are invalid.

    Notes
    -----
    ``fs`` is required when ``time`` is omitted. Explicitly timed
    irregular observations use ``fs=None``. ``time`` is always
    normalized to a timestamp vector after construction.
    """

    data: np.ndarray
    fs: float | None
    time: np.ndarray | None = None
    t0: float | None = 0.0
    feature_names: list[str] = field(default_factory=list)
    source_signal: str | None = None
    window_size: float | None = None
    shift: float | None = None
    unit: str | list[str] = "a.u."
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = as_2d_array(self.data, "FeatureMatrix.data")
        fs = ensure_positive_float(self.fs, "fs")
        if fs is None and self.time is None:
            raise ValidationError("fs must be provided when FeatureMatrix.time is omitted.")
        self.fs = fs
        self.t0 = 0.0 if self.t0 is None else ensure_finite_float(self.t0, "t0")
        self.feature_names = list(self.feature_names)
        if not self.feature_names:
            width = max(3, len(str(max(self.data.shape[1] - 1, 0))))
            self.feature_names = [f"feature{idx:0{width}d}" for idx in range(self.data.shape[1])]
        if len(self.feature_names) != self.data.shape[1]:
            raise ValidationError("feature_names length must match data.shape[1].")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValidationError("feature_names must be unique.")
        self.unit = validate_unit(self.unit, self.data.shape[1], "unit")
        if self.source_signal is not None:
            self.source_signal = ensure_non_empty_string(self.source_signal, "source_signal")
        self.window_size = ensure_positive_float(self.window_size, "window_size")
        self.shift = ensure_positive_float(self.shift, "shift")
        self.attrs = copy_attrs(self.attrs)

        time = build_sample_time_grid(self.time, self.t0, self.data.shape[0], fs)
        if fs is not None and not sampled_times_match_rate(time, fs):
            raise ValidationError(
                "FeatureMatrix.time must match fs; use fs=None for irregular observations."
            )
        self.time = time

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])


SpikePolarity = Literal["negative", "positive", "both"]
SpikeSelector = int | slice | Sequence[int] | np.ndarray


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class SpikeWaveformBatch:
    """Store an immutable offline batch of aligned spike waveforms.

    Waveforms use ``(spike, sample, channel)`` axis order. The aligned source
    sample is ``waveforms[:, pre_samples, :]`` and the sample axis must contain
    exactly ``pre_samples + 1 + post_samples`` entries. ``sample_indices`` are
    absolute source indices; equal adjacent indices are allowed for simultaneous
    spikes, but indices and times must otherwise be non-decreasing.

    ``segment_id`` is batch-level, so one object cannot represent spikes from
    multiple continuous segments. Concatenation rejects different segments or
    any other incompatible batch metadata. Integer indexing returns a one-spike
    batch rather than dropping the typed batch contract.

    Times and absolute indices must describe one regular source grid. Relative
    time offsets are compared with ``rtol=1e-12`` and an absolute tolerance of
    at least ``1e-12`` seconds, enlarged to cover eight floating-point ULPs at
    the represented time scale.

    ``attrs`` is deep-frozen using the closed metadata value domain documented
    by :mod:`neurale.data`; unsupported leaf values raise ``ValidationError``.
    """

    waveforms: np.ndarray
    sample_indices: np.ndarray
    times: np.ndarray
    peak_channel_indices: np.ndarray
    electrode_group_ids: np.ndarray
    amps: np.ndarray
    channels: ChannelTable
    fs: float
    clock: Clock | None
    source_stream: str
    segment_id: int | str
    pre_samples: int
    post_samples: int
    polarity: SpikePolarity
    spike_polarities: np.ndarray | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._initialize(freeze_arrays=True)

    def _initialize(self, *, freeze_arrays: bool) -> None:
        waveforms = _real_array(self.waveforms, "waveforms", ndim=3)
        if waveforms.shape[1] == 0:
            raise ValidationError("waveforms must contain at least one sample per spike.")
        if waveforms.shape[2] == 0:
            raise ValidationError("waveforms must contain at least one channel.")

        sample_indices = _int64_array(self.sample_indices, "sample_indices")
        peak_channels = _int64_array(
            self.peak_channel_indices,
            "peak_channel_indices",
        )
        group_ids = _int64_array(self.electrode_group_ids, "electrode_group_ids")
        times = _finite_float_array(self.times, "times")
        amps = _real_array(self.amps, "amps", ndim=1)
        n_spikes = waveforms.shape[0]
        for name, values in (
            ("sample_indices", sample_indices),
            ("times", times),
            ("peak_channel_indices", peak_channels),
            ("electrode_group_ids", group_ids),
            ("amps", amps),
        ):
            if values.shape[0] != n_spikes:
                raise ValidationError(f"{name} length must match waveforms.shape[0].")

        if np.any(sample_indices < 0):
            raise ValidationError("sample_indices must be non-negative.")
        ensure_monotonic_non_decreasing(sample_indices, "sample_indices")
        if np.any(times < 0.0):
            raise ValidationError("times must be non-negative.")
        ensure_monotonic_non_decreasing(times, "times")

        if not isinstance(self.channels, ChannelTable):
            raise ValidationError("channels must be a ChannelTable.")
        if len(self.channels) != waveforms.shape[2]:
            raise ValidationError("channels length must match waveforms.shape[2].")
        if np.any(peak_channels < 0) or np.any(peak_channels >= waveforms.shape[2]):
            raise ValidationError(
                "peak_channel_indices must refer to positions in the batch ChannelTable."
            )
        if np.any(group_ids < 0):
            raise ValidationError("electrode_group_ids must be non-negative.")

        fs = ensure_positive_float(self.fs, "fs")
        if fs is None:  # pragma: no cover - required argument
            raise ValidationError("fs must be provided.")
        if self.clock is not None and not isinstance(self.clock, Clock):
            raise ValidationError("clock must be a Clock or None.")
        source_stream = ensure_non_empty_string(self.source_stream, "source_stream")
        segment_id = _segment_id(self.segment_id)
        pre_samples = validate_integer(self.pre_samples, "pre_samples", minimum=0)
        post_samples = validate_integer(self.post_samples, "post_samples", minimum=0)
        expected_samples = pre_samples + 1 + post_samples
        if waveforms.shape[1] != expected_samples:
            raise ValidationError("waveforms.shape[1] must equal pre_samples + 1 + post_samples.")

        polarity = self.polarity
        if polarity not in ("negative", "positive", "both"):
            raise ValidationError("polarity must be one of: negative, positive, both.")
        spike_polarities = self.spike_polarities
        if polarity == "both":
            if spike_polarities is None:
                raise ValidationError("spike_polarities is required when polarity='both'.")
            spike_polarities = _int8_polarity_array(spike_polarities, n_spikes)
        elif spike_polarities is not None:
            raise ValidationError("spike_polarities is only valid when polarity='both'.")

        if not isinstance(self.attrs, Mapping):
            raise ValidationError("attrs must be a mapping.")
        _validate_regular_spike_times(sample_indices, times, fs)
        if freeze_arrays:
            # Native sorting may provide a bytes-backed array that is already
            # strongly immutable. Reusing that final backing avoids a second
            # full waveform payload during typed batch construction. Mutable
            # and merely read-only caller arrays still receive the defensive
            # copy required by the public ownership contract.
            if not is_strongly_immutable(waveforms):
                waveforms = immutable_array_copy(waveforms)
            sample_indices = immutable_array_copy(sample_indices)
            times = immutable_array_copy(times)
            peak_channels = immutable_array_copy(peak_channels)
            group_ids = immutable_array_copy(group_ids)
            amps = immutable_array_copy(amps)
            spike_polarities = (
                None if spike_polarities is None else immutable_array_copy(spike_polarities)
            )
        else:
            # Trusted internal path used by contiguous-slice derivation: the
            # arrays are views of this batch's already strongly immutable
            # backing, so they are re-used without a copy. ``flags.writeable``
            # alone is not a sufficient proof -- a view of a writable base can
            # be marked read-only yet still allow ``flags.writeable = True`` --
            # so each array is checked with ``is_strongly_immutable``, which
            # walks the ``.base`` chain to a ``bytes`` buffer. A soft-frozen view
            # or a writable array reaching this path fails loudly instead of
            # producing a silently mutable batch.
            for name, arr in (
                ("waveforms", waveforms),
                ("sample_indices", sample_indices),
                ("times", times),
                ("peak_channel_indices", peak_channels),
                ("electrode_group_ids", group_ids),
                ("amps", amps),
            ):
                if not is_strongly_immutable(arr):
                    raise RuntimeError(f"trusted {name} must be strongly immutable")
            if spike_polarities is not None and not is_strongly_immutable(spike_polarities):
                raise RuntimeError("trusted spike_polarities must be strongly immutable")
        object.__setattr__(self, "waveforms", waveforms)
        object.__setattr__(self, "sample_indices", sample_indices)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "peak_channel_indices", peak_channels)
        object.__setattr__(self, "electrode_group_ids", group_ids)
        object.__setattr__(self, "amps", amps)
        object.__setattr__(
            self,
            "spike_polarities",
            spike_polarities,
        )
        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "source_stream", source_stream)
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "pre_samples", pre_samples)
        object.__setattr__(self, "post_samples", post_samples)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    def _rebuild_kwargs(self) -> dict[str, Any]:
        # Collect the constructor arguments needed to rebuild an equivalent
        # batch. ``pickle.loads`` and ``copy.deepcopy`` skip ``__post_init__``,
        # so without an explicit rebuild protocol they restore the backing
        # arrays as writable ndarrays and the metadata as plain mutable
        # containers, breaking the immutable batch contract. Routing
        # deserialization and deep copy back through the public constructor
        # re-runs ``__post_init__`` and re-freezes every array and metadata
        # container via ``immutable_array_copy`` / ``freeze_metadata``.
        return {
            "waveforms": self.waveforms,
            "sample_indices": self.sample_indices,
            "times": self.times,
            "peak_channel_indices": self.peak_channel_indices,
            "electrode_group_ids": self.electrode_group_ids,
            "amps": self.amps,
            "channels": self.channels,
            "fs": self.fs,
            "clock": self.clock,
            "source_stream": self.source_stream,
            "segment_id": self.segment_id,
            "pre_samples": self.pre_samples,
            "post_samples": self.post_samples,
            "polarity": self.polarity,
            "spike_polarities": self.spike_polarities,
            "attrs": self.attrs,
        }

    def __reduce__(self):
        return _rebuild_spike_waveform_batch, (self._rebuild_kwargs(),)

    def __deepcopy__(self, memo: dict[int, Any]) -> SpikeWaveformBatch:
        # Rebuild through the constructor so the copy re-freezes arrays and
        # metadata. Arrays and ``attrs`` are copied fresh by ``__post_init__``;
        # the immutable ``channels`` and ``clock`` are deep-copied so the
        # result is an independent object graph rather than shared references.
        kwargs = self._rebuild_kwargs()
        # The constructor safely reuses bytes-backed waveforms, whereas a deep
        # copy promises an independent immutable backing. Make exactly that one
        # payload copy before reconstruction.
        kwargs["waveforms"] = immutable_array_copy(self.waveforms)
        kwargs["channels"] = copy.deepcopy(self.channels, memo)
        kwargs["clock"] = copy.deepcopy(self.clock, memo)
        rebuilt = _rebuild_spike_waveform_batch(kwargs)
        memo[id(self)] = rebuilt
        return rebuilt

    def _contiguous_slice(self, key: slice) -> SpikeWaveformBatch:
        # Derive a contiguous-slice subset from this batch's already strongly
        # immutable backing arrays. Provenance is structural -- every array
        # below is a view of ``self``'s bytes-backed arrays, whose writeable
        # flag cannot be re-enabled -- so the subset shares the backing without
        # a copy. There is intentionally no general "trusted" entry that accepts
        # caller-supplied arrays: a caller could pass a soft-frozen view of a
        # writable base (``flags.writeable == False`` but re-enableable), which
        # would sneak a mutable array past the immutable-batch contract.
        # Deriving the arrays from ``self`` makes that impossible; the
        # ``is_strongly_immutable`` check in ``_initialize`` is a defense-in-
        # depth net, not the source of the guarantee.
        polarities = None if self.spike_polarities is None else self.spike_polarities[key]
        obj = object.__new__(type(self))
        object.__setattr__(obj, "waveforms", self.waveforms[key])
        object.__setattr__(obj, "sample_indices", self.sample_indices[key])
        object.__setattr__(obj, "times", self.times[key])
        object.__setattr__(obj, "peak_channel_indices", self.peak_channel_indices[key])
        object.__setattr__(obj, "electrode_group_ids", self.electrode_group_ids[key])
        object.__setattr__(obj, "amps", self.amps[key])
        object.__setattr__(obj, "spike_polarities", polarities)
        object.__setattr__(obj, "channels", self.channels)
        object.__setattr__(obj, "fs", self.fs)
        object.__setattr__(obj, "clock", self.clock)
        object.__setattr__(obj, "source_stream", self.source_stream)
        object.__setattr__(obj, "segment_id", self.segment_id)
        object.__setattr__(obj, "pre_samples", self.pre_samples)
        object.__setattr__(obj, "post_samples", self.post_samples)
        object.__setattr__(obj, "polarity", self.polarity)
        object.__setattr__(obj, "attrs", self.attrs)
        obj._initialize(freeze_arrays=False)
        return obj

    def __len__(self) -> int:
        return self.n_spikes

    def __getitem__(self, key: SpikeSelector) -> SpikeWaveformBatch:
        """Return an ordered spike subset while preserving batch metadata."""
        # Fast path: a contiguous slice (default step) produces views of the
        # already-immutable backing arrays. The views are strongly immutable
        # (the writeable flag cannot be re-enabled because the ultimate base is
        # a bytes buffer), so the subset can share the backing without a copy.
        # ``_contiguous_slice`` derives the arrays from ``self`` rather than
        # accepting caller-supplied arrays, so the immutable provenance is
        # structural.
        if isinstance(key, slice) and (key.step is None or key.step == 1):
            return self._contiguous_slice(key)
        # General path: integer, fancy, or boolean selectors copy via advanced
        # indexing; the public constructor re-freezes those copies.
        positions = _spike_positions(key, self.n_spikes)
        polarities = None if self.spike_polarities is None else self.spike_polarities[positions]
        return SpikeWaveformBatch(
            waveforms=self.waveforms[positions],
            sample_indices=self.sample_indices[positions],
            times=self.times[positions],
            peak_channel_indices=self.peak_channel_indices[positions],
            electrode_group_ids=self.electrode_group_ids[positions],
            amps=self.amps[positions],
            channels=self.channels,
            fs=self.fs,
            clock=self.clock,
            source_stream=self.source_stream,
            segment_id=self.segment_id,
            pre_samples=self.pre_samples,
            post_samples=self.post_samples,
            polarity=self.polarity,
            spike_polarities=polarities,
            attrs=self.attrs,
        )

    def __repr__(self) -> str:
        return (
            f"SpikeWaveformBatch(n_spikes={self.n_spikes}, "
            f"n_samples={self.n_samples}, n_channels={self.n_channels}, "
            f"source_stream={self.source_stream!r}, segment_id={self.segment_id!r}, "
            f"polarity={self.polarity!r})"
        )

    @property
    def n_spikes(self) -> int:
        return int(self.waveforms.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.waveforms.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.waveforms.shape[2])

    @property
    def axis_order(self) -> tuple[str, str, str]:
        return ("spike", "sample", "channel")

    @classmethod
    def concatenate(cls, batches: Sequence[SpikeWaveformBatch]) -> SpikeWaveformBatch:
        """Concatenate compatible, already ordered batches from one segment."""
        values = tuple(batches)
        if not values:
            raise ValidationError("batches must contain at least one SpikeWaveformBatch.")
        if not all(isinstance(batch, cls) for batch in values):
            raise ValidationError("batches must contain only SpikeWaveformBatch instances.")
        first = values[0]
        for batch in values[1:]:
            _require_compatible_spike_batches(first, batch)
        polarities = (
            None
            if first.spike_polarities is None
            else np.concatenate([batch.spike_polarities for batch in values])
        )
        return cls(
            waveforms=np.concatenate([batch.waveforms for batch in values], axis=0),
            sample_indices=np.concatenate([batch.sample_indices for batch in values]),
            times=np.concatenate([batch.times for batch in values]),
            peak_channel_indices=np.concatenate([batch.peak_channel_indices for batch in values]),
            electrode_group_ids=np.concatenate([batch.electrode_group_ids for batch in values]),
            amps=np.concatenate([batch.amps for batch in values]),
            channels=first.channels,
            fs=first.fs,
            clock=first.clock,
            source_stream=first.source_stream,
            segment_id=first.segment_id,
            pre_samples=first.pre_samples,
            post_samples=first.post_samples,
            polarity=first.polarity,
            spike_polarities=polarities,
            attrs=first.attrs,
        )


def _rebuild_spike_waveform_batch(kwargs: dict[str, Any]) -> SpikeWaveformBatch:
    # Module-level rebuild entry point referenced by
    # ``SpikeWaveformBatch.__reduce__`` / ``__deepcopy__``. Pickle resolves it
    # by qualified name, so it must live at module scope. Re-entering the public
    # constructor re-runs ``__post_init__`` and re-freezes every array and
    # metadata container.
    return SpikeWaveformBatch(**kwargs)


@dataclass(slots=True)
class SpikeTrain:
    """Store spike timestamps and optional unit or waveform metadata.

    Parameters
    ----------
    times : list of numpy.ndarray
        One non-negative, monotonically ordered timestamp vector per spike train.
    units : list of str or None, optional
        Unit identifier for each spike train.
    channels : list of int or None, optional
        Non-negative channel index for each spike train.
    waveforms : list of numpy.ndarray or None, optional
        One 3D waveform array per spike train.
    fs : float or None, optional
        Waveform sampling rate in hertz.
    attrs : dict, optional
        Additional application-specific metadata.

    Raises
    ------
    neurale.exceptions.ValidationError
        If timestamps or optional per-train metadata are inconsistent.
    """

    times: list[np.ndarray]
    units: list[str] | None = None
    channels: list[int] | None = None
    waveforms: list[np.ndarray] | None = None
    fs: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = [as_1d_float_array(values, "spike times") for values in self.times]
        for values in self.times:
            if np.any(values < 0):
                raise ValidationError("spike times must be non-negative.")
            ensure_monotonic_non_decreasing(values, "spike times")

        n_trains = len(self.times)
        if self.units is not None:
            self.units = [ensure_non_empty_string(unit, "unit id") for unit in self.units]
            if len(self.units) != n_trains:
                raise ValidationError("units length must match times length.")
        if self.channels is not None:
            self.channels = [
                validate_integer(channel, "spike channel", minimum=0) for channel in self.channels
            ]
            if len(self.channels) != n_trains:
                raise ValidationError("channels length must match times length.")
        if self.waveforms is not None:
            self.waveforms = [np.asarray(waveform) for waveform in self.waveforms]
            if len(self.waveforms) != n_trains:
                raise ValidationError("waveforms length must match times length.")
            for waveform in self.waveforms:
                if waveform.ndim != 3:
                    raise ValidationError("each waveform array must be 3D.")
        self.fs = ensure_positive_float(self.fs, "fs")
        self.attrs = copy_attrs(self.attrs)


def _real_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != ndim:
        raise ValidationError(f"{name} must be a {ndim}D array.")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValidationError(f"{name} must contain real numeric values.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def _finite_float_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValidationError(f"{name} must contain finite real values.")
    try:
        arr = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain finite real values.") from exc
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def _int64_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array.")
    if arr.dtype != np.dtype(np.int64):
        raise ValidationError(f"{name} must have dtype int64.")
    return arr


def _int8_polarity_array(value: Any, n_spikes: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape[0] != n_spikes:
        raise ValidationError("spike_polarities length must match waveforms.shape[0].")
    if arr.dtype != np.dtype(np.int8):
        raise ValidationError("spike_polarities must have dtype int8.")
    if np.any((arr != -1) & (arr != 1)):
        raise ValidationError("spike_polarities values must be -1 or 1.")
    return arr


def _segment_id(value: int | str) -> int | str:
    if isinstance(value, str):
        return ensure_non_empty_string(value, "segment_id")
    return validate_integer(value, "segment_id", minimum=0)


def _validate_regular_spike_times(
    sample_indices: np.ndarray,
    times: np.ndarray,
    fs: float,
) -> None:
    if sample_indices.size < 2:
        return
    relative_indices = (sample_indices - sample_indices[0]).astype(np.float64)
    expected = relative_indices / fs
    actual = times - times[0]
    scale = max(
        1.0,
        float(np.max(np.abs(times))),
        float(np.max(np.abs(expected))),
    )
    tol = max(1e-12, 8.0 * np.spacing(scale))
    if not np.allclose(actual, expected, rtol=1e-12, atol=tol):
        raise ValidationError("times must agree with sample_indices and fs on one regular grid.")


def _spike_positions(key: SpikeSelector, n_spikes: int) -> np.ndarray:
    if isinstance(key, (bool, np.bool_)):
        raise TypeError("boolean scalar is not a valid spike index.")
    if isinstance(key, (int, np.integer)):
        pos = int(key)
        if pos < 0:
            pos += n_spikes
        if pos < 0 or pos >= n_spikes:
            raise IndexError("spike index out of range")
        return np.asarray([pos], dtype=np.int64)
    if isinstance(key, slice):
        return np.arange(n_spikes, dtype=np.int64)[key]

    selector = np.asarray(key)
    if selector.ndim != 1:
        raise ValidationError("spike selector must be 1D.")
    if selector.dtype == np.dtype(bool):
        if selector.shape[0] != n_spikes:
            raise ValidationError("boolean spike selector has the wrong length.")
        return np.flatnonzero(selector).astype(np.int64, copy=False)
    # An empty selector selects no spikes regardless of the dtype ``np.asarray``
    # inferred: a Python ``[]`` becomes a zero-length float64 array, but it is a
    # legal empty integer subset and must round-trip to an empty batch.
    if selector.size == 0:
        return np.asarray([], dtype=np.int64)
    if not np.issubdtype(selector.dtype, np.integer):
        raise ValidationError("spike selector must contain integer positions or booleans.")
    # Validate unsigned selectors on the original dtype before the int64 cast.
    # A uint64 value above int64 max wraps to a negative int64, which the
    # negative-index rewrite below would reinterpret as a valid trailing
    # position -- silently returning the wrong spike instead of rejecting an
    # out-of-range index.
    if np.issubdtype(selector.dtype, np.unsignedinteger) and np.any(
        selector > np.iinfo(np.int64).max
    ):
        raise IndexError("spike index out of range")
    positions = selector.astype(np.int64)
    positions[positions < 0] += n_spikes
    if np.any(positions < 0) or np.any(positions >= n_spikes):
        raise IndexError("spike index out of range")
    return positions


def _require_compatible_spike_batches(
    first: SpikeWaveformBatch,
    other: SpikeWaveformBatch,
) -> None:
    scalar_fields = (
        "fs",
        "source_stream",
        "segment_id",
        "pre_samples",
        "post_samples",
        "polarity",
    )
    for name in scalar_fields:
        if getattr(first, name) != getattr(other, name):
            raise ValidationError(f"cannot concatenate batches with different {name}.")
    if first.waveforms.shape[1:] != other.waveforms.shape[1:]:
        raise ValidationError("cannot concatenate batches with different waveform shapes.")
    if first.waveforms.dtype != other.waveforms.dtype:
        raise ValidationError("cannot concatenate batches with different waveform dtypes.")
    if first.amps.dtype != other.amps.dtype:
        raise ValidationError("cannot concatenate batches with different amplitude dtypes.")
    if not _metadata_equal(first.channels, other.channels):
        raise ValidationError("cannot concatenate batches with different channels.")
    if not _metadata_equal(first.clock, other.clock):
        raise ValidationError("cannot concatenate batches with different clocks.")
    if not _metadata_equal(first.attrs, other.attrs):
        raise ValidationError("cannot concatenate batches with different attrs.")


def _metadata_equal(first: Any, other: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(other, np.ndarray):
        if not (isinstance(first, np.ndarray) and isinstance(other, np.ndarray)):
            return False
        return bool(np.array_equal(first, other, equal_nan=True))
    if isinstance(first, Mapping) or isinstance(other, Mapping):
        if not (isinstance(first, Mapping) and isinstance(other, Mapping)):
            return False
        # Mapping equality is order-independent: two attrs that differ only in
        # insertion order (e.g. ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``)
        # are semantically identical and must concatenate. ``keys()`` views
        # compare as sets, so this holds for ``dict`` and ``FrozenMapping`` alike.
        if len(first) != len(other):
            return False
        if first.keys() != other.keys():
            return False
        return all(_metadata_equal(first[key], other[key]) for key in first)
    if is_dataclass(first) or is_dataclass(other):
        if type(first) is not type(other):
            return False
        return all(
            _metadata_equal(getattr(first, item.name), getattr(other, item.name))
            for item in fields(first)
        )
    if isinstance(first, (tuple, list)) or isinstance(other, (tuple, list)):
        if not (isinstance(first, (tuple, list)) and isinstance(other, (tuple, list))):
            return False
        return len(first) == len(other) and all(
            _metadata_equal(left, right) for left, right in zip(first, other, strict=True)
        )
    try:
        result = first == other
    except (TypeError, ValueError):
        return False
    return bool(result)
