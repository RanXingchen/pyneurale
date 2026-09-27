#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Define the data frame exchanged by streaming components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from neurale._validation import validate_integer
from neurale.exceptions import ValidationError

from ._helpers import (
    FrozenMapping,
    ValidatedNamedDict,
    ensure_non_negative_float,
    freeze_metadata,
)
from .arrays import SignalArray
from .events import EventSeries
from .time import Clock


@dataclass(frozen=True, slots=True)
class Frame:
    """Represent one online block of signals and events.

    Parameters
    ----------
    signals : mapping of str to neurale.data.arrays.SignalArray, optional
        Named signal blocks carried by the frame.
    events : neurale.data.events.EventSeries or None, optional
        Events associated with the frame.
    sequence : int or None, optional
        Non-negative frame sequence number.
    received_at : float or None, optional
        Host monotonic reception timestamp in seconds. The streaming runtime
        replaces source-provided values at its input boundary.
    source_received_at : float or None, optional
        Non-negative source-provided reception timestamp, preserved only as
        source metadata and never used for host latency calculations.
    source_received_clock : neurale.data.time.Clock or None, optional
        Clock domain for ``source_received_at``. The two fields must be
        provided together.
    sample_idx_start : int or None, optional
        Absolute sample index of the first sample, required when frame events
        carry ``sample_index`` values.
    attrs : mapping, optional
        Deep-frozen application metadata using the closed value domain
        documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If names, signal values, events, sequence, or timestamp are invalid.

    Notes
    -----
    Frame structure and metadata are immutable. Signal data remains writable
    for the exclusive processor owner until streaming publication.
    """

    signals: Mapping[str, SignalArray] = field(default_factory=dict)
    events: EventSeries | None = None
    sequence: int | None = None
    received_at: float | None = None
    source_received_at: float | None = None
    source_received_clock: Clock | None = None
    sample_idx_start: int | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signals = ValidatedNamedDict(
            self.signals,
            value_type=SignalArray,
            label="frame signal",
        )
        if self.events is not None and not isinstance(self.events, EventSeries):
            raise ValidationError("events must be an EventSeries.")
        sequence = (
            None
            if self.sequence is None
            else validate_integer(self.sequence, "sequence", minimum=0)
        )
        received_at = self.received_at
        if received_at is not None:
            received_at = ensure_non_negative_float(received_at, "received_at")
        source_received_at = self.source_received_at
        if source_received_at is not None:
            source_received_at = ensure_non_negative_float(source_received_at, "source_received_at")
        if (source_received_at is None) != (self.source_received_clock is None):
            raise ValidationError(
                "source_received_at and source_received_clock must be provided together."
            )
        if self.source_received_clock is not None and not isinstance(
            self.source_received_clock, Clock
        ):
            raise ValidationError("source_received_clock must be a Clock.")
        sample_idx_start = (
            None
            if self.sample_idx_start is None
            else validate_integer(self.sample_idx_start, "sample_idx_start", minimum=0)
        )
        object.__setattr__(self, "signals", FrozenMapping(signals))
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "received_at", received_at)
        object.__setattr__(self, "source_received_at", source_received_at)
        object.__setattr__(self, "sample_idx_start", sample_idx_start)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))
