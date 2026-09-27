#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Event and trial data structures."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from neurale._validation import validate_integer
from neurale.exceptions import ValidationError

from ._helpers import (
    as_1d_float_array,
    copy_attrs,
    ensure_finite_float,
    ensure_monotonic_non_decreasing,
    ensure_non_negative_float,
    freeze_metadata,
    immutable_array_copy,
)
from .time import Clock


@dataclass(frozen=True, slots=True)
class Event:
    """Describe an instantaneous or interval event in seconds.

    Parameters
    ----------
    onset : float
        Non-negative event onset in seconds.
    duration : float, default=0.0
        Non-negative event duration in seconds.
    label : str or None, optional
        Human-readable event label.
    code : int or None, optional
        Numeric event code.
    source : str or None, optional
        Name of the event source.
    value : float or str or None, optional
        Source-specific event value.
    sample_index : int or None, optional
        Non-negative source sample index.
    confidence : float or None, optional
        Confidence score in the inclusive interval ``[0, 1]``.
    attrs : dict, optional
        Deep-frozen application metadata using the closed value domain
        documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If onset, duration, sample index, or confidence is invalid.
    """

    onset: float
    duration: float = 0.0
    label: str | None = None
    code: int | None = None
    source: str | None = None
    value: float | str | None = None
    sample_index: int | None = None
    confidence: float | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        onset = ensure_non_negative_float(self.onset, "event onset")
        duration = ensure_non_negative_float(self.duration, "event duration")
        sample_idx = (
            None
            if self.sample_index is None
            else validate_integer(self.sample_index, "sample_index", minimum=0)
        )
        code = None if self.code is None else validate_integer(self.code, "event code")
        confidence = self.confidence
        if confidence is not None:
            confidence = ensure_finite_float(confidence, "event confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValidationError("event confidence must be in [0, 1].")
        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "sample_index", sample_idx)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @property
    def stop(self) -> float:
        return self.onset + self.duration

    @property
    def is_interval(self) -> bool:
        return self.duration > 0.0


@dataclass(frozen=True, slots=True)
class EventSeries:
    """Store an ordered collection of events.

    Parameters
    ----------
    events : iterable of Event, optional
        Events in collection order.
    clock : neurale.data.time.Clock or None, optional
        neurale.data.time.Clock metadata shared by the events.
    attrs : dict or None, optional
        Deep-frozen collection metadata using the closed value domain
        documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If an item is not an :class:`Event`, events are out of order, or an
        event is duplicated.
    """

    events: tuple[Event, ...] = field(default_factory=tuple)
    clock: Clock | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        events: Iterable[Event] = (),
        clock: Clock | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        event_values = tuple(events)
        for event in event_values:
            if not isinstance(event, Event):
                raise ValidationError("events must contain Event instances.")
        if any(current.onset < previous.onset for previous, current in pairwise(event_values)):
            raise ValidationError("events must be ordered by onset.")
        identities = [_event_identity(event) for event in event_values]
        if len(set(identities)) != len(identities):
            raise ValidationError("events must not contain duplicates.")
        object.__setattr__(self, "events", event_values)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "attrs", freeze_metadata(attrs))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __getitem__(self, key: int | slice) -> Event | EventSeries:
        if isinstance(key, slice):
            return EventSeries(self.events[key], clock=self.clock, attrs=self.attrs)
        return self.events[key]

    @property
    def onsets(self) -> np.ndarray:
        return np.asarray([event.onset for event in self.events], dtype=float)

    @property
    def durations(self) -> np.ndarray:
        return np.asarray([event.duration for event in self.events], dtype=float)

    @property
    def labels(self) -> list[str | None]:
        return [event.label for event in self.events]

    def by_label(self, label: str) -> EventSeries:
        """Select events with an exact label match.

        Parameters
        ----------
        label : str
            Label to match.

        Returns
        -------
        neurale.data.events.EventSeries
            New series containing matching events.
        """
        return EventSeries(
            [event for event in self.events if event.label == label],
            clock=self.clock,
            attrs=self.attrs,
        )

    def as_lines(self) -> np.ndarray:
        """Return onsets of instantaneous events.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of onset times for zero-duration events.
        """
        return np.asarray(
            [event.onset for event in self.events if not event.is_interval], dtype=float
        )

    def as_regions(self) -> np.ndarray:
        """Return intervals for non-instantaneous events.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_intervals, 2)`` containing onset and stop times.
        """
        regions = [[event.onset, event.stop] for event in self.events if event.is_interval]
        return np.asarray(regions, dtype=float).reshape(-1, 2)

    @classmethod
    def from_ttl(
        cls,
        ttl: np.ndarray,
        fs: float,
        *,
        channel: int = 0,
        t0: float = 0.0,
        threshold: float = 0.5,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
        debounce: float = 0.0,
        min_high_duration: float = 0.0,
        min_low_duration: float = 0.0,
        initial_high: bool = False,
        include_rising: bool = True,
        include_falling: bool = True,
        source: str = "ttl",
    ) -> EventSeries:
        """Build events from threshold crossings in a TTL trace.

        Parameters
        ----------
        ttl : numpy.ndarray
            One-dimensional TTL trace or 2D sample-by-channel array.
        fs : float
            Sampling rate in hertz.
        channel : int, default=0
            Channel selected when ``ttl`` is 2D.
        t0 : float, default=0.0
            Time of the first TTL sample in seconds.
        threshold : float, default=0.5
            Threshold separating low and high states.
        low_threshold, high_threshold : float or None, optional
            Hysteresis thresholds. When omitted, both use ``threshold``.
        debounce : float, default=0
            Minimum stable transition duration in seconds.
        min_high_duration, min_low_duration : float, default=0
            Minimum stable high or low duration required to accept the
            corresponding transition.
        initial_high : bool, default=False
            Treat the state before the first sample as high.
        include_rising : bool, default=True
            Whether to emit events for low-to-high transitions.
        include_falling : bool, default=True
            Whether to emit events for high-to-low transitions.
        source : str, default="ttl"
            Source name stored on generated events.

        Returns
        -------
        neurale.data.events.EventSeries
            Chronologically sorted threshold-crossing events.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the sampling rate, TTL dimensions, or channel index is invalid.
        """

        options = _validate_ttl_options(
            fs=fs,
            threshold=threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            debounce=debounce,
            min_high_duration=min_high_duration,
            min_low_duration=min_low_duration,
            initial_high=initial_high,
            include_rising=include_rising,
            include_falling=include_falling,
        )
        trace = _ttl_trace(ttl, channel)
        transitions = _ttl_transitions(
            trace,
            low_threshold=options.low_threshold,
            high_threshold=options.high_threshold,
            high_samples=options.high_samples,
            low_samples=options.low_samples,
            initial_high=options.initial_high,
        )
        return cls(_ttl_events(trace, transitions, t0, source, options))


def _event_identity(event: Event) -> tuple[object, ...]:
    return (
        event.onset,
        event.duration,
        event.label,
        event.code,
        event.source,
        event.value,
        event.sample_index,
    )


@dataclass(frozen=True, slots=True)
class _TtlOptions:
    fs: float
    low_threshold: float
    high_threshold: float
    high_samples: int
    low_samples: int
    initial_high: bool
    include_rising: bool
    include_falling: bool


def _validate_ttl_options(
    *,
    fs: float,
    threshold: float,
    low_threshold: float | None,
    high_threshold: float | None,
    debounce: float,
    min_high_duration: float,
    min_low_duration: float,
    initial_high: bool,
    include_rising: bool,
    include_falling: bool,
) -> _TtlOptions:
    rate = ensure_finite_float(fs, "fs")
    if rate <= 0:
        raise ValidationError("fs must be positive.")
    threshold = ensure_finite_float(threshold, "threshold")
    low = (
        threshold if low_threshold is None else ensure_finite_float(low_threshold, "low_threshold")
    )
    high = (
        threshold
        if high_threshold is None
        else ensure_finite_float(high_threshold, "high_threshold")
    )
    if low > high:
        raise ValidationError("low_threshold must not exceed high_threshold.")
    debounce = ensure_non_negative_float(debounce, "debounce")
    minimum_high = ensure_non_negative_float(min_high_duration, "min_high_duration")
    minimum_low = ensure_non_negative_float(min_low_duration, "min_low_duration")
    if not isinstance(initial_high, bool):
        raise ValidationError("initial_high must be a bool.")
    if not isinstance(include_rising, bool) or not isinstance(include_falling, bool):
        raise ValidationError("include_rising and include_falling must be bool values.")
    debounce_samples = math.ceil(debounce * rate)
    return _TtlOptions(
        fs=rate,
        low_threshold=low,
        high_threshold=high,
        high_samples=max(
            1,
            debounce_samples,
            math.ceil(minimum_high * rate),
        ),
        low_samples=max(
            1,
            debounce_samples,
            math.ceil(minimum_low * rate),
        ),
        initial_high=initial_high,
        include_rising=include_rising,
        include_falling=include_falling,
    )


def _ttl_trace(ttl: np.ndarray, channel: int) -> np.ndarray:
    arr = np.asarray(ttl)
    if arr.ndim == 1:
        trace = arr
    elif arr.ndim == 2:
        if not 0 <= channel < arr.shape[1]:
            raise ValidationError("channel is out of bounds for ttl data.")
        trace = arr[:, channel]
    else:
        raise ValidationError("ttl must be a 1D or 2D array.")
    try:
        result = np.asarray(trace, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("ttl must contain numeric values.") from exc
    if np.any(~np.isfinite(result)):
        raise ValidationError("ttl must contain only finite values.")
    return result


def _ttl_events(
    trace: np.ndarray,
    transitions: list[tuple[int, bool]],
    t0: float,
    source: str,
    options: _TtlOptions,
) -> list[Event]:
    events = []
    for sample_idx, is_high in transitions:
        if is_high and not options.include_rising:
            continue
        if not is_high and not options.include_falling:
            continue
        events.append(
            Event(
                onset=t0 + sample_idx / options.fs,
                label="ttl_rising" if is_high else "ttl_falling",
                source=source,
                value=float(trace[sample_idx]),
                sample_index=int(sample_idx),
            )
        )
    return sorted(events, key=lambda event: (event.onset, event.label or ""))


def _ttl_transitions(
    trace: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
    high_samples: int,
    low_samples: int,
    initial_high: bool,
) -> list[tuple[int, bool]]:
    # Detect debounced hysteretic transitions and report candidate start times.
    state_high = initial_high
    candidate_start: int | None = None
    transitions: list[tuple[int, bool]] = []
    for i, value in enumerate(trace):
        qualifies = value > high_threshold if not state_high else value <= low_threshold
        if not qualifies:
            candidate_start = None
            continue
        if candidate_start is None:
            candidate_start = i
        required = low_samples if state_high else high_samples
        if i - candidate_start + 1 >= required:
            state_high = not state_high
            transitions.append((candidate_start, state_high))
            candidate_start = None
    return transitions


@dataclass(frozen=True, slots=True)
class Trial:
    """Describe a task-level trial interval and its annotations.

    Parameters
    ----------
    trial_id : int
        Unique non-negative trial identifier.
    start : float
        Non-negative trial start time in seconds.
    stop : float
        Trial stop time in seconds, strictly greater than ``start``.
    label : str or None, optional
        Human-readable trial label.
    target_id : int or None, optional
        Non-negative target identifier.
    target : numpy.ndarray or None, optional
        Target value or coordinates.
    outcome : str or None, optional
        Trial outcome label.
    block : int or None, optional
        Non-negative block identifier.
    attrs : dict, optional
        Deep-frozen application metadata using the closed value domain
        documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If identifiers or trial timing are invalid.
    """

    trial_id: int
    start: float
    stop: float
    label: str | None = None
    target_id: int | None = None
    target: np.ndarray | None = None
    outcome: str | None = None
    block: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trial_id = validate_integer(self.trial_id, "trial_id", minimum=0)
        start = ensure_non_negative_float(self.start, "trial start")
        stop = ensure_non_negative_float(self.stop, "trial stop")
        if stop <= start:
            raise ValidationError("trial stop must be greater than trial start.")
        target_id = (
            None
            if self.target_id is None
            else validate_integer(self.target_id, "target_id", minimum=0)
        )
        block = None if self.block is None else validate_integer(self.block, "block", minimum=0)
        target = None if self.target is None else immutable_array_copy(self.target)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "target", target)
        # Freeze attrs (deep, like Event/EventSeries) so a frozen Trial cannot be
        # mutated through shared references such as TrialEpoch.trial.
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @property
    def duration(self) -> float:
        return self.stop - self.start


@dataclass(slots=True)
class TrialTable:
    """Store an ordered collection of trials with unique identifiers.

    Parameters
    ----------
    trials : iterable of Trial, optional
        Trials in collection order.
    attrs : dict or None, optional
        Additional collection metadata.

    Raises
    ------
    neurale.exceptions.ValidationError
        If an item is not a :class:`Trial` or identifiers are duplicated.
    """

    trials: tuple[Trial, ...] = field(default_factory=tuple)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, trials: Iterable[Trial] = (), attrs: dict[str, Any] | None = None) -> None:
        self.trials = tuple(trials)
        self.attrs = copy_attrs(attrs)
        for trial in self.trials:
            if not isinstance(trial, Trial):
                raise ValidationError("trials must contain Trial instances.")
        ids = [trial.trial_id for trial in self.trials]
        if len(ids) != len(set(ids)):
            raise ValidationError("trial_id values must be unique.")

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self) -> Iterator[Trial]:
        return iter(self.trials)

    def __getitem__(self, key: int | slice) -> Trial | TrialTable:
        if isinstance(key, slice):
            return TrialTable(self.trials[key], attrs=self.attrs)
        return self.trials[key]

    @property
    def starts(self) -> np.ndarray:
        values = as_1d_float_array([trial.start for trial in self.trials], "trial starts")
        ensure_monotonic_non_decreasing(values, "trial starts")
        return values
