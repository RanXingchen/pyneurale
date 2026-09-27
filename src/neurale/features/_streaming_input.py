#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared validation for fixed-rate feature streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurale._validation import validate_real_array
from neurale.data import SignalArray
from neurale.data._time_validation import (
    sampled_times_are_contiguous,
    sampled_times_match_rate,
)
from neurale.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class NormalizedStreamBlock:
    data: np.ndarray
    template: SignalArray | None
    kind: type[np.ndarray] | type[SignalArray] | None
    stream_start: float | None
    next_input_time: float | None
    source_signal: str | None


def normalize_stream_block(
    block: np.ndarray | SignalArray,
    *,
    n_channels: int,
    fs: float,
    channel_signature: tuple[tuple[str, int], ...] | None,
    kind: type[np.ndarray] | type[SignalArray] | None,
    stream_start: float | None,
    next_input_time: float | None,
    source_signal: str | None,
) -> NormalizedStreamBlock:
    """Validate a block and return stream state to commit after processing."""
    if isinstance(block, SignalArray):
        block.validate()
        if channel_signature is None:
            raise ValidationError("initialize SignalArray streams with a ChannelTable.")
        if block.n_channels != n_channels:
            raise ValidationError(f"stream block must have {n_channels} channels.")
        signature = tuple(zip(block.channels.names, block.channels.indices, strict=False))
        if signature != channel_signature:
            raise ValidationError("stream block channel layout changed.")
        if not np.isclose(block.fs, fs):
            raise ValidationError("stream block fs changed.")
        data = validate_real_array(block.data, "stream block", ndim=2)
        if block.n_samples and not sampled_times_match_rate(block.time, fs):
            raise ValidationError("SignalArray stream samples must match fs.")
        if not block.n_samples:
            return NormalizedStreamBlock(
                data,
                block,
                kind,
                stream_start,
                next_input_time,
                source_signal,
            )
        if kind is not None and kind is not SignalArray:
            raise ValidationError("cannot mix ndarray and SignalArray blocks in one stream.")
        if source_signal is not None and block.name != source_signal:
            raise ValidationError("stream block signal name changed.")
        start = float(block.time[0])
        if next_input_time is not None and not sampled_times_are_contiguous(
            start, next_input_time, fs
        ):
            raise ValidationError("SignalArray stream chunks must be time-contiguous.")
        return NormalizedStreamBlock(
            data,
            block,
            SignalArray,
            start if stream_start is None else stream_start,
            float(block.time[-1]) + 1.0 / fs,
            block.name if source_signal is None else source_signal,
        )

    if not isinstance(block, np.ndarray) or block.ndim != 2:
        raise ValidationError("stream block must be a 2D ndarray or SignalArray.")
    if block.shape[1] != n_channels:
        raise ValidationError(f"stream block must have {n_channels} channels.")
    data = validate_real_array(block, "stream block", ndim=2)
    if not block.shape[0]:
        return NormalizedStreamBlock(
            data,
            None,
            kind,
            stream_start,
            next_input_time,
            source_signal,
        )
    if kind is not None and kind is not np.ndarray:
        raise ValidationError("cannot mix ndarray and SignalArray blocks in one stream.")
    return NormalizedStreamBlock(
        data,
        None,
        np.ndarray,
        0.0 if stream_start is None else stream_start,
        next_input_time,
        source_signal,
    )
