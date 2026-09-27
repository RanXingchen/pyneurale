#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Polyphase signal resampling."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import gcd

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_integer,
    validate_number,
    validate_positive_float,
)
from neurale.data import SignalArray
from neurale.exceptions import ValidationError

from ._arrays import native_array
from ._input import normalize_signal_input, uniform_sampling_rate
from .filtering.design import firls

_MAX_RATE_DENOMINATOR = 1_000_000
_MAX_FILTER_TAPS = 4097
_KAISER_BETA = 5.0
_NEIGHBOR_TERMS = 10


def _restore_array_layout(values: np.ndarray, ndim: int, sample_axis: int) -> np.ndarray:
    if ndim == 1:
        return values[:, 0]
    return np.moveaxis(values, 0, sample_axis)


class Resampler:
    """Reusable native polyphase resampler.

    Parameters
    ----------
    up : int
        Positive upsampling factor.
    down : int
        Positive downsampling factor.
    neighbor_terms : int, default=10
        Number of neighboring samples included on each side of the
        anti-aliasing kernel.
    beta : float, default=5.0
        Kaiser-window beta used by the anti-aliasing filter.
    dtype : numpy.dtype or type, default=float
        Output dtype for resampled blocks.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native resampling extension is unavailable.
    neurale.exceptions.ValidationError
        If rate factors or filter parameters are invalid.
    """

    def __init__(
        self,
        up: int,
        down: int,
        *,
        neighbor_terms: int = _NEIGHBOR_TERMS,
        beta: float = _KAISER_BETA,
        dtype: np.dtype | type = float,
    ) -> None:
        up = validate_integer(up, "up", minimum=1)
        down = validate_integer(down, "down", minimum=1)
        common = gcd(up, down)
        self.up = up // common
        self.down = down // common
        self.neighbor_terms = validate_integer(neighbor_terms, "neighbor_terms", minimum=0)
        self.beta = float(
            validate_number(
                beta,
                "beta",
                kind="real",
                minimum=0,
                coerce=True,
            )
        )
        self.dtype = _resample_output_dtype(np.dtype(dtype))
        taps = _design_antialias_filter(
            self.up,
            self.down,
            self.dtype,
            neighbor_terms=self.neighbor_terms,
            beta=self.beta,
        )
        self._native = _native_resampling().Resampler(
            self.up,
            self.down,
            native_array(taps, float),
        )
        self._layout: tuple[int, int] | None = None

    def output_length(self, input_length: int) -> int:
        """Return the final sample count after processing and flushing.

        Parameters
        ----------
        input_length : int
            Number of input samples.

        Returns
        -------
        int
            Output samples after all data are processed and ``flush()`` is
            called.
        """
        input_length = validate_integer(input_length, "input_length", minimum=0)
        return int(self._native.output_length(input_length))

    def max_output_length(self, input_length: int) -> int:
        """Return the maximum samples emitted by one process call.

        Parameters
        ----------
        input_length : int
            Number of input samples in the next block.

        Returns
        -------
        int
            Upper bound on samples emitted by ``process()`` for that block.
        """
        input_length = validate_integer(input_length, "input_length", minimum=0)
        return int(self._native.max_output_length(input_length))

    def max_flush_length(self) -> int:
        """Return the maximum samples emitted by ``flush()``.

        Returns
        -------
        int
            Upper bound on pending samples emitted by ``flush()``.
        """
        return int(self._native.max_flush_length())

    @property
    def initial_output_trim(self) -> int:
        return int(self._native.initial_output_trim())

    def prepare(
        self,
        n_channels: int,
        max_input_samples: int,
    ) -> None:
        """Preallocate native buffers for a fixed streaming shape.

        Parameters
        ----------
        n_channels : int
            Number of channels in later ``process()`` calls.
        max_input_samples : int
            Maximum input samples per later ``process()`` call.

        Raises
        ------
        neurale.exceptions.ValidationError
            If dimensions are invalid.
        """
        n_channels = validate_integer(n_channels, "n_channels", minimum=1)
        max_input_samples = validate_integer(
            max_input_samples,
            "max_input_samples",
            minimum=0,
        )
        self._native.prepare(n_channels, max_input_samples)

    def process(
        self,
        x: np.ndarray,
        *,
        axis: int = 0,
    ) -> np.ndarray:
        """Resample one streaming block and keep tail state for later calls.

        Parameters
        ----------
        x : numpy.ndarray
            One- or two-dimensional real-valued input block.
        axis : int, default=0
            Sample axis for ``x``.

        Returns
        -------
        numpy.ndarray
            Resampled block with the same channel layout convention as
            ``x``.

        Raises
        ------
        TypeError
            If ``x`` is a ``SignalArray``.
        neurale.exceptions.ValidationError
            If input data or streaming layout is invalid.
        """
        if isinstance(x, SignalArray):
            raise TypeError(
                "Resampler.process() expects ndarray input; use resample() "
                "to preserve SignalArray metadata."
            )
        if isinstance(x, np.ndarray) and axis == 0:
            if x.ndim == 1:
                self._remember_layout(1, 0)
                result = self._process_normalized(x.reshape(-1, 1))
                return result[:, 0]
            if x.ndim == 2:
                self._remember_layout(2, 0)
                return self._process_normalized(x)
        normalized, context = normalize_signal_input(x, axis=axis)
        if isinstance(context.source, np.ndarray):
            self._remember_layout(context.source.ndim, context.sample_axis)
        result = self._process_normalized(normalized)
        return self._restore_layout(result)

    def flush(self) -> np.ndarray:
        """Emit pending samples and clear the remembered input layout.

        Returns
        -------
        numpy.ndarray
            Pending resampled samples using the layout of previous input
            blocks.
        """
        result = self._native.flush()
        restored = self._restore_layout(result)
        self._layout = None
        return restored

    def reset(self) -> None:
        """Clear native resampling state and remembered input layout."""
        self._native.reset()
        self._layout = None

    def _process_normalized(self, x: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(x):
            raise ValidationError("resample requires real-valued data.")
        if x.shape[0] == 0:
            return np.empty((0, x.shape[1]), dtype=self.dtype)
        result = self._native.process(native_array(x, float))
        if result.dtype != self.dtype:
            result = result.astype(self.dtype, copy=False)
        return result

    def _remember_layout(self, ndim: int, sample_axis: int) -> None:
        layout = (ndim, sample_axis)
        if self._layout is None:
            self._layout = layout
        elif self._layout != layout:
            raise ValidationError("resampler input layout must stay constant until flush or reset.")

    def _restore_layout(self, values: np.ndarray) -> np.ndarray:
        if self._layout is None:
            return values
        ndim, sample_axis = self._layout
        return _restore_array_layout(values, ndim, sample_axis)


def resample(
    x: np.ndarray | SignalArray,
    up: int | None = None,
    down: int | None = None,
    *,
    target_rate: float | None = None,
    axis: int = 0,
    neighbor_terms: int = _NEIGHBOR_TERMS,
    beta: float = _KAISER_BETA,
) -> np.ndarray | SignalArray:
    """Resample a signal with a zero-phase native polyphase FIR filter.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional real-valued signal.
    up : int or None, optional
        Positive upsampling factor. Required when ``target_rate`` is omitted.
    down : int or None, optional
        Positive downsampling factor. Required when ``target_rate`` is omitted.
    target_rate : float or None, optional
        Target sampling rate in hertz for ``SignalArray`` input. Cannot be
        combined with ``up`` or ``down``.
    axis : int, default=0
        Sample axis for ndarray input.
    neighbor_terms : int, default=10
        Number of neighboring samples included on each side of the
        anti-aliasing kernel.
    beta : float, default=5.0
        Kaiser-window beta used by the anti-aliasing filter.

    Returns
    -------
    numpy.ndarray or neurale.data.SignalArray
        Resampled signal with input layout and metadata preserved. For
        ``SignalArray`` input, sampling metadata are updated.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native resampling extension is unavailable.
    neurale.exceptions.ValidationError
        If input data, rate arguments, or filter parameters are invalid.
    """
    data, context = normalize_signal_input(x, axis=axis)
    if np.iscomplexobj(data):
        raise ValidationError("resample requires real-valued data.")
    up, down, source_rate, output_rate = _resolve_rate_ratio(x, up, down, target_rate)
    processor = Resampler(
        up,
        down,
        neighbor_terms=neighbor_terms,
        beta=beta,
        dtype=data.dtype,
    )
    result = _resample_normalized(data, processor)

    if isinstance(context.source, np.ndarray):
        return _restore_array_layout(result, context.source.ndim, context.sample_axis)

    source = context.source
    if not isinstance(source, SignalArray):
        raise TypeError("resample metadata restoration requires SignalArray input.")
    if source_rate is None or output_rate is None:
        raise ValidationError("SignalArray resampling requires sampling metadata.")
    start = float(source.time[0]) if source.time.size else source.t0
    explicit_time = start + np.arange(result.shape[0]) / output_rate
    return SignalArray(
        data=result,
        fs=output_rate,
        time=explicit_time,
        t0=start,
        clock=source.clock,
        channels=source.channels.copy(),
        unit=source.unit,
        name=source.name,
        attrs=dict(source.attrs),
    )


def _native_resampling():
    return load_native_namespace("signal.resampling")


def _resample_normalized(
    x: np.ndarray,
    processor: Resampler,
) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty((0, x.shape[1]), dtype=processor.dtype)
    head = processor._process_normalized(x)
    tail = processor._native.flush()
    if tail.dtype != processor.dtype:
        tail = tail.astype(processor.dtype, copy=False)
    if tail.shape[0] == 0:
        return head
    if head.shape[0] == 0:
        return tail
    return np.concatenate((head, tail), axis=0)


def _resample_output_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.inexact):
        return dtype
    return np.dtype(float)


def _resolve_rate_ratio(
    x: np.ndarray | SignalArray,
    up: int | None,
    down: int | None,
    target_rate: float | None,
) -> tuple[int, int, float | None, float | None]:
    if target_rate is not None:
        if up is not None or down is not None:
            raise ValidationError("target_rate cannot be combined with up or down.")
        if not isinstance(x, SignalArray):
            raise ValidationError("target_rate requires SignalArray sampling metadata.")
        source_rate = uniform_sampling_rate(x, required=True)
        if source_rate is None:
            raise ValidationError("target_rate requires sampling metadata.")
        requested_rate = validate_positive_float(target_rate, "target_rate")
        ratio = Fraction(requested_rate / source_rate).limit_denominator(_MAX_RATE_DENOMINATOR)
        up, down = ratio.numerator, ratio.denominator
        output_rate = source_rate * up / down
        if not np.isclose(
            output_rate,
            requested_rate,
            rtol=1e-12,
            atol=max(1e-12, requested_rate * 1e-15),
        ):
            raise ValidationError(
                "target_rate cannot be represented accurately with the "
                f"maximum denominator {_MAX_RATE_DENOMINATOR}."
            )
        return up, down, source_rate, output_rate

    if up is None or down is None:
        raise ValidationError("up and down are both required when target_rate is omitted.")
    up = validate_integer(up, "up", minimum=1)
    down = validate_integer(down, "down", minimum=1)
    common = gcd(up, down)
    up //= common
    down //= common
    if isinstance(x, SignalArray):
        source_rate = uniform_sampling_rate(x, required=True)
        if source_rate is None:
            raise ValidationError("SignalArray resampling requires sampling metadata.")
        return up, down, source_rate, source_rate * up / down
    return up, down, None, None


def _design_antialias_filter(
    up: int,
    down: int,
    dtype: np.dtype,
    *,
    neighbor_terms: int = _NEIGHBOR_TERMS,
    beta: float = _KAISER_BETA,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    if up == down == 1:
        taps = np.ones(1, dtype=np.result_type(dtype, float))
        taps.setflags(write=False)
        return taps
    return _cached_antialias_filter(
        up,
        down,
        dtype.str,
        neighbor_terms,
        float(beta),
    )


@lru_cache(maxsize=64)
def _cached_antialias_filter(
    up: int,
    down: int,
    dtype_str: str,
    neighbor_terms: int,
    beta: float,
) -> np.ndarray:
    dtype = np.dtype(dtype_str)
    if neighbor_terms == 0:
        taps = np.full(up, 1.0 / up, dtype=np.result_type(dtype, float))
        taps.setflags(write=False)
        return taps

    max_rate = max(up, down)
    length = 2 * neighbor_terms * max_rate + 1
    if length > _MAX_FILTER_TAPS:
        raise ValidationError(
            "resampling ratio requires an anti-alias filter with "
            f"{length} taps, exceeding the supported limit "
            f"{_MAX_FILTER_TAPS}; use a staged rate conversion."
        )
    cutoff = 1.0 / max_rate
    taps = firls(
        length - 1,
        [0.0, cutoff, cutoff, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ).taps
    taps = taps * np.kaiser(length, beta)
    taps /= np.sum(taps)
    if np.issubdtype(dtype, np.inexact):
        taps = taps.astype(dtype, copy=False)
    taps.setflags(write=False)
    return taps
