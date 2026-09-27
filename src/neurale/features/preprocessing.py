#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Composable offline and streaming neural-signal preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from neurale._validation import (
    validate_axis,
    validate_choice,
    validate_integer,
    validate_positive_float,
)
from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import ValidationError
from neurale.signal import (
    Resampler,
    SosCoefficients,
    SosFilter,
    butter,
    common_reference,
    notch,
    resample,
    sos_filtfilt,
)

from ._streaming_input import normalize_stream_block

ChannelKeys = Sequence[int | str] | np.ndarray
ReferenceMethod = Literal["mean", "median"] | None
FilterBand = Literal["lowpass", "bandpass"]
_MAX_RATE_DENOMINATOR = 1_000_000


@dataclass(frozen=True, slots=True)
class _FilterSpec:
    reference: ReferenceMethod
    notch_freqs: tuple[float, ...]
    notch_bandwidth: float
    filter_order: int | None
    filter_cutoff: float | tuple[float, float] | None
    filter_band: FilterBand


def select_good_channels(
    signal: np.ndarray | SignalArray,
    *,
    bad_channels: ChannelKeys | None = None,
    exclude_bad: bool = True,
    axis: int = 0,
) -> np.ndarray | SignalArray:
    """Remove explicit and metadata-marked bad channels.

    ``SignalArray`` selectors are channel names or channel indices. ndarray
    selectors are zero-based positions; a boolean selector marks bad channels.
    """
    if not isinstance(exclude_bad, bool):
        raise ValidationError("exclude_bad must be a bool.")

    if isinstance(signal, SignalArray):
        normalized_axis = validate_axis(axis, 2)
        if normalized_axis != 0:
            raise ValidationError("SignalArray uses axis 0 as its fixed sample axis.")
        keep = signal.channels.good_mask if exclude_bad else np.ones(signal.n_channels, dtype=bool)
        if bad_channels is not None:
            keep = keep.copy()
            keep[signal.channels.positions(bad_channels)] = False
        if not np.any(keep):
            raise ValidationError("channel selection removed every channel.")
        return signal.select_channels(keep)

    if not isinstance(signal, np.ndarray) or signal.ndim not in (1, 2):
        raise ValidationError("signal must be a one- or two-dimensional ndarray or SignalArray.")
    sample_axis = validate_axis(axis, signal.ndim)
    if signal.ndim == 1:
        if bad_channels is not None and _bad_positions(bad_channels, 1):
            raise ValidationError("channel selection removed every channel.")
        return signal.copy()

    channel_axis = 1 - sample_axis
    n_channels = signal.shape[channel_axis]
    bad = _bad_positions(bad_channels, n_channels)
    keep = np.ones(n_channels, dtype=bool)
    keep[bad] = False
    if not np.any(keep):
        raise ValidationError("channel selection removed every channel.")
    return np.compress(keep, signal, axis=channel_axis)


def offline_preprocess(
    signal: np.ndarray | SignalArray,
    *,
    fs: float | None = None,
    bad_channels: ChannelKeys | None = None,
    exclude_bad: bool = True,
    target_rate: float | None = None,
    reference: ReferenceMethod = "mean",
    notch_freqs: Sequence[float] = (),
    notch_bandwidth: float = 2.0,
    filter_order: int | None = None,
    filter_cutoff: float | Sequence[float] | None = None,
    filter_band: FilterBand = "bandpass",
    axis: int = 0,
) -> np.ndarray | SignalArray:
    """Run the zero-phase offline preprocessing pipeline."""
    spec = _filter_spec(
        reference,
        notch_freqs,
        notch_bandwidth,
        filter_order,
        filter_cutoff,
        filter_band,
    )
    rate = _input_rate(signal, fs)
    result = select_good_channels(
        signal,
        bad_channels=bad_channels,
        exclude_bad=exclude_bad,
        axis=axis,
    )

    if target_rate is not None:
        requested_rate = validate_positive_float(target_rate, "target_rate")
        if rate is None:
            raise ValidationError("target_rate requires fs.")
        if isinstance(result, SignalArray):
            result = resample(result, target_rate=requested_rate)
            rate = result.fs
        else:
            up, down, rate = _rate_ratio(rate, requested_rate)
            result = resample(result, up, down, axis=axis)

    if spec.reference is not None:
        result = common_reference(result, method=spec.reference, axis=axis)

    filters = _design_filters(rate, spec)
    for coefs in filters:
        result = sos_filtfilt(result, coefs, axis=axis)
    return result


class NeuralPreprocessor:
    """Causal stateful preprocessing for sample-major streaming blocks.

    All filters and optional resampling state are constructed eagerly. Pass a
    ``ChannelTable`` to apply its quality mask and retain channel metadata.
    Consecutive non-empty ``SignalArray`` chunks must be time-contiguous;
    ``reset()`` starts a new stream and clears that timing contract.
    """

    def __init__(
        self,
        fs: float,
        channels: int | ChannelTable,
        *,
        bad_channels: ChannelKeys | None = None,
        exclude_bad: bool = True,
        target_rate: float | None = None,
        reference: ReferenceMethod = "mean",
        notch_freqs: Sequence[float] = (),
        notch_bandwidth: float = 2.0,
        filter_order: int | None = None,
        filter_cutoff: float | Sequence[float] | None = None,
        filter_band: FilterBand = "bandpass",
        max_input_samples: int | None = None,
    ) -> None:
        self.input_rate = validate_positive_float(fs, "fs")
        (
            self._output_channels,
            self._positions,
            self.n_input_channels,
            self._channel_signature,
        ) = _streaming_channels(channels, bad_channels, exclude_bad)
        self.n_channels = len(self._positions)
        self.reference = _validate_reference(reference)
        spec = _filter_spec(
            self.reference,
            notch_freqs,
            notch_bandwidth,
            filter_order,
            filter_cutoff,
            filter_band,
        )

        self._resampler: Resampler | None = None
        self.fs = self.input_rate
        if target_rate is not None:
            requested_rate = validate_positive_float(target_rate, "target_rate")
            up, down, self.fs = _rate_ratio(self.input_rate, requested_rate)
            self._resampler = Resampler(up, down)
            if max_input_samples is not None:
                block_size = validate_integer(max_input_samples, "max_input_samples", minimum=0)
                self._resampler.prepare(self.n_channels, block_size)
        elif max_input_samples is not None:
            raise ValidationError("max_input_samples is only used when target_rate is provided.")

        self._filters = tuple(
            SosFilter(coefs, n_channels=self.n_channels) for coefs in _design_filters(self.fs, spec)
        )
        self._template: SignalArray | None = None
        self._output_start: float | None = None
        self._output_samples = 0
        self._next_input_time: float | None = None
        self._stream_kind: type[np.ndarray] | type[SignalArray] | None = None
        self._source_signal: str | None = None
        self._failed = False

    def process(
        self,
        block: np.ndarray | SignalArray,
    ) -> np.ndarray | SignalArray:
        """Process one causal sample-major block and retain stream state."""
        if self._failed:
            raise RuntimeError("processor failed; call reset() before continuing")
        candidate = normalize_stream_block(
            block,
            n_channels=self.n_input_channels,
            fs=self.input_rate,
            channel_signature=self._channel_signature,
            kind=self._stream_kind,
            stream_start=self._output_start,
            next_input_time=self._next_input_time,
            source_signal=self._source_signal,
        )
        selected = np.array(
            candidate.data[:, self._positions],
            dtype=float,
            order="C",
            copy=True,
        )
        try:
            if self._resampler is not None:
                selected = self._resampler.process(selected)
            output = self._process_selected(selected)
        except (ValueError, OverflowError, RuntimeError):
            self._failed = True
            raise
        self._stream_kind = candidate.kind
        self._output_start = candidate.stream_start
        self._next_input_time = candidate.next_input_time
        self._source_signal = candidate.source_signal
        if candidate.template is not None and candidate.template.n_samples:
            self._template = candidate.template
        return self._restore_block(output, candidate.template)

    def flush(self) -> np.ndarray | SignalArray:
        """Emit pending resampler output using the active stream layout."""
        if self._failed:
            raise RuntimeError("processor failed; call reset() before continuing")
        try:
            if self._resampler is None:
                output = np.empty((0, self.n_channels), dtype=float)
            else:
                output = self._process_selected(self._resampler.flush())
        except (ValueError, OverflowError, RuntimeError):
            self._failed = True
            raise
        return self._restore_block(output, self._template)

    def reset(self) -> None:
        """Reset causal filter, resampler, and output timing state."""
        if self._resampler is not None:
            self._resampler.reset()
        for processor in self._filters:
            processor.reset()
        self._template = None
        self._output_start = None
        self._output_samples = 0
        self._next_input_time = None
        self._stream_kind = None
        self._source_signal = None
        self._failed = False

    def _process_selected(self, data: np.ndarray) -> np.ndarray:
        if self.reference is not None and data.shape[0]:
            data = common_reference(data, method=self.reference, axis=0, inplace=True)
        for processor in self._filters:
            processor.process(data)
        return data

    def _restore_block(
        self, data: np.ndarray, template: SignalArray | None
    ) -> np.ndarray | SignalArray:
        if template is None:
            return data
        channels = (
            self._output_channels
            if self._output_channels is not None
            else template.channels.select(self._positions)
        )
        if self._resampler is None:
            time = (
                template.time.copy()
                if data.shape[0] == template.n_samples
                else np.empty(0, dtype=float)
            )
        else:
            start = self._output_start if self._output_start is not None else template.t0
            time = start + (self._output_samples + np.arange(data.shape[0], dtype=float)) / self.fs
            self._output_samples += data.shape[0]
        unit: str | tuple[str, ...] = (
            template.unit if isinstance(template.unit, str) else tuple(channels.units)
        )
        t0 = (
            float(time[0])
            if time.size
            else (self._output_start if self._output_start is not None else template.t0)
        )
        return SignalArray(
            data=data,
            fs=self.fs,
            time=time,
            t0=t0,
            clock=template.clock,
            channels=channels,
            unit=unit,
            name=template.name,
            attrs=dict(template.attrs),
        )


def _bad_positions(keys: ChannelKeys | None, n_channels: int) -> list[int]:
    if keys is None:
        return []
    arr = np.asarray(keys)
    if arr.dtype == bool:
        if arr.ndim != 1 or arr.size != n_channels:
            raise ValidationError("boolean bad-channel mask has the wrong shape.")
        return np.flatnonzero(arr).tolist()
    if any(isinstance(key, str) for key in list(keys)):
        raise ValidationError("ndarray bad_channels must be integer positions.")
    positions = [validate_integer(key, "bad channel") for key in list(keys)]
    if any(pos < 0 or pos >= n_channels for pos in positions):
        raise ValidationError("bad channel position is out of range.")
    return positions


def _streaming_channels(
    channels: int | ChannelTable,
    bad_channels: ChannelKeys | None,
    exclude_bad: bool,
) -> tuple[
    ChannelTable | None,
    tuple[int, ...],
    int,
    tuple[tuple[str, int], ...] | None,
]:
    if not isinstance(exclude_bad, bool):
        raise ValidationError("exclude_bad must be a bool.")
    if isinstance(channels, ChannelTable):
        keep = channels.good_mask if exclude_bad else np.ones(len(channels), dtype=bool)
        if bad_channels is not None:
            keep = keep.copy()
            keep[channels.positions(bad_channels)] = False
        positions = tuple(np.flatnonzero(keep).tolist())
        table = channels.copy()
        output_table = table.select([table.channels[pos].index for pos in positions])
        count = len(table)
        signature = tuple(zip(table.names, table.indices, strict=False))
    else:
        count = validate_integer(channels, "channels", minimum=1)
        bad = _bad_positions(bad_channels, count)
        bad_set = set(bad)
        positions = tuple(idx for idx in range(count) if idx not in bad_set)
        table = None
        output_table = None
        signature = None
    if not positions:
        raise ValidationError("channel selection removed every channel.")
    return output_table, positions, count, signature


def _filter_spec(
    reference: ReferenceMethod,
    notch_freqs: Sequence[float],
    notch_bandwidth: float,
    filter_order: int | None,
    filter_cutoff: float | Sequence[float] | None,
    filter_band: FilterBand,
) -> _FilterSpec:
    reference = _validate_reference(reference)
    freqs = tuple(validate_positive_float(value, "notch frequency") for value in notch_freqs)
    bandwidth = validate_positive_float(notch_bandwidth, "notch_bandwidth")
    band = validate_choice(filter_band, ("lowpass", "bandpass"), "filter_band")
    if (filter_order is None) != (filter_cutoff is None):
        raise ValidationError("filter_order and filter_cutoff must be provided together.")
    order = (
        None if filter_order is None else validate_integer(filter_order, "filter_order", minimum=1)
    )
    cutoff: float | tuple[float, float] | None
    if filter_cutoff is None:
        cutoff = None
    elif band == "lowpass":
        cutoff = validate_positive_float(filter_cutoff, "filter_cutoff")
    else:
        values = np.asarray(filter_cutoff)
        if values.ndim != 1 or values.size != 2:
            raise ValidationError("bandpass filter_cutoff must contain two values.")
        cutoff = (
            validate_positive_float(values[0], "filter_cutoff"),
            validate_positive_float(values[1], "filter_cutoff"),
        )
    return _FilterSpec(reference, freqs, bandwidth, order, cutoff, band)


def _validate_reference(reference: ReferenceMethod) -> ReferenceMethod:
    if reference is None:
        return None
    return validate_choice(reference, ("mean", "median"), "reference")


def _input_rate(signal: np.ndarray | SignalArray, fs: float | None) -> float | None:
    if isinstance(signal, SignalArray):
        if fs is not None and not np.isclose(
            validate_positive_float(fs, "fs"),
            signal.fs,
        ):
            raise ValidationError("fs does not match SignalArray metadata.")
        return signal.fs
    return None if fs is None else validate_positive_float(fs, "fs")


def _rate_ratio(source_rate: float, target_rate: float) -> tuple[int, int, float]:
    ratio = Fraction(target_rate / source_rate).limit_denominator(_MAX_RATE_DENOMINATOR)
    output_rate = source_rate * ratio.numerator / ratio.denominator
    if not np.isclose(
        output_rate,
        target_rate,
        rtol=1e-12,
        atol=max(1e-12, target_rate * 1e-15),
    ):
        raise ValidationError("target_rate cannot be represented accurately.")
    return ratio.numerator, ratio.denominator, output_rate


def _design_filters(fs: float | None, spec: _FilterSpec) -> tuple[SosCoefficients, ...]:
    if not spec.notch_freqs and spec.filter_order is None:
        return ()
    if fs is None:
        raise ValidationError("filtering requires fs.")
    filters = [
        notch(
            freq,
            bandwidth=spec.notch_bandwidth,
            fs=fs,
            output="sos",
        )
        for freq in spec.notch_freqs
    ]
    if spec.filter_order is not None and spec.filter_cutoff is not None:
        filters.append(
            butter(
                spec.filter_order,
                spec.filter_cutoff,
                band=spec.filter_band,
                fs=fs,
                output="sos",
            )
        )
    return tuple(filters)
