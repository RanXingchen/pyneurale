#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Trial slicing, epoch extraction, and neural/behavior alignment.

This module is the public owner of the alignment contract: the typed
result dataclasses, the public functions, and the ``AlignmentTarget`` alias are
defined here so that ``__module__`` and pickle paths reference
``neurale.data.alignment`` rather than a provisional private module.
Implementation helpers live in :mod:`neurale.data._alignment`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from neurale._validation import validate_choice, validate_number, validate_real_array
from neurale.exceptions import ValidationError

from ._alignment import (
    _common_reference_coverage,
    _convert_time,
    _converted_interval,
    _coverage_slice_bounds,
    _covered_interval,
    _finite_time,
    _from_reference_time,
    _nearest_one_to_one,
    _reject_any_discontinuity,
    _reject_discontinuity_overlap,
    _require_alignment_clock_compatibility,
    _require_bool,
    _require_compatible_clocks,
    _require_discontinuity_clock_compatibility,
    _require_discontinuity_sequence,
    _require_optional_clock,
    _require_optional_event_series,
    _require_regular_time_axis,
    _require_signal,
    _require_strictly_increasing_time,
    _require_trial_splitting_clock_compatibility,
    _required_labels,
    _selected_signal_names,
    _stable_reference_time,
    _to_reference_time,
    _to_reference_time_array,
    _validate_trial_table_for_splitting,
    _within_common_coverage,
)
from ._helpers import FrozenMapping, freeze_metadata, immutable_array_copy
from .arrays import SignalArray
from .events import EventSeries, Trial
from .recording import Recording
from .time import Clock

AlignmentTarget = Literal["neural", "behavior"]


@dataclass(frozen=True, slots=True)
class TrialEpoch:
    """One trial's slicing result from :func:`split_recording_trials`.

    The epoch is a frozen, self-consistent view: ``signals`` is an immutable
    mapping and ``recording_metadata`` / ``trial_attrs`` are deep-frozen copies,
    so mutating a result cannot leak back into the source :class:`Recording`
    or :class:`Trial`.

    Parameters
    ----------
    trial : neurale.data.events.Trial
        The source trial value.
    start, stop : float
        The actual epoch interval in recording-reference seconds, half-open
        ``[start, stop)``. ``split_recording_trials`` is strict-only, so this
        equals the source trial interval. ``start < stop`` is enforced.
    signals : Mapping[str, neurale.data.arrays.SignalArray]
        Mapping from non-empty signal name to that signal's own epoch. Each
        value must be a :class:`SignalArray`.
    events : neurale.data.events.EventSeries or None
        Events filtered to the epoch interval, or ``None`` if the recording has
        no events.
    recording_metadata : Mapping[str, object]
        Recording-level metadata; deep-frozen on construction using the closed
        value domain documented by :mod:`neurale.data`.
    trial_attrs : Mapping[str, object]
        Trial-level attrs; deep-frozen on construction using the same closed
        metadata value domain.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``trial`` is not a :class:`Trial`, ``start >= stop``, the interval
        lies outside ``[trial.start, trial.stop]``, ``signals`` is not a
        non-empty mapping, a signal name is empty, a signal value is not a
        :class:`SignalArray`, ``events`` is neither an :class:`EventSeries` nor
        ``None``, or ``recording_metadata``/``trial_attrs`` is not a mapping.
    """

    trial: Trial
    start: float
    stop: float
    signals: Mapping[str, SignalArray]
    events: EventSeries | None
    recording_metadata: Mapping[str, object]
    trial_attrs: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.trial, Trial):
            raise ValidationError("TrialEpoch.trial must be a Trial.")
        start = _finite_time(self.start, "start")
        stop = _finite_time(self.stop, "stop")
        if stop <= start:
            raise ValidationError("TrialEpoch requires start < stop.")
        if not (self.trial.start <= start and stop <= self.trial.stop):
            raise ValidationError("TrialEpoch interval must lie within the source trial.")
        if not isinstance(self.signals, Mapping):
            raise ValidationError("TrialEpoch.signals must be a mapping.")
        if not self.signals:
            raise ValidationError("TrialEpoch.signals must not be empty.")
        frozen_signals: dict[str, SignalArray] = {}
        for name, signal in self.signals.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("TrialEpoch signal names must be non-empty strings.")
            if not isinstance(signal, SignalArray):
                raise ValidationError("TrialEpoch signal values must be SignalArray instances.")
            frozen_signals[name] = signal
        if self.events is not None and not isinstance(self.events, EventSeries):
            raise ValidationError("TrialEpoch.events must be an EventSeries or None.")
        if not isinstance(self.recording_metadata, Mapping):
            raise ValidationError("TrialEpoch.recording_metadata must be a mapping.")
        if not isinstance(self.trial_attrs, Mapping):
            raise ValidationError("TrialEpoch.trial_attrs must be a mapping.")
        # Deep-freeze metadata so mutating a result cannot leak back into the
        # source Recording/Trial, and wrap signals in an immutable mapping so a
        # frozen TrialEpoch cannot be broken via signals.clear().
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "signals", FrozenMapping(frozen_signals))
        object.__setattr__(self, "recording_metadata", freeze_metadata(self.recording_metadata))
        object.__setattr__(self, "trial_attrs", freeze_metadata(self.trial_attrs))


@dataclass(frozen=True, slots=True)
class AlignmentIndex:
    """Typed neural/behavior nearest-neighbor alignment index from
    :func:`align_neural_behavior_nearest`.

    The index is frozen and internally consistent: the four arrays are backed
    by immutable bytes (writeability cannot be re-enabled), so a result cannot
    be edited into a self-contradictory state such as ``behavior_indices[i] ==
    -1`` with ``matched[i] == True``.

    Parameters
    ----------
    mode : "nearest"
        Currently always ``"nearest"``.
    target : {"neural", "behavior"}
        Which signal's grid is the target whose rows are all present.
    tol : float
        Inclusive match tolerance in recording-reference seconds (``>= 0``).
    reference_time : numpy.ndarray
        Target grid in recording-reference seconds, shape ``(n_rows,)``.
    neural_indices, behavior_indices : numpy.ndarray
        Integer index arrays, shape ``(n_rows,)``. The target-side array equals
        ``arange(n_rows)``. The source-side array holds the matched source row
        index or ``-1`` for unmatched rows.
    matched : numpy.ndarray
        Boolean mask, shape ``(n_rows,)``. ``matched[i]`` is ``True`` exactly
        when the source-side index is valid (``>= 0``).

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``mode``/``target`` are invalid, ``tol`` is negative, the
        arrays are not equal-length 1-D ndarrays, indices are not integer,
        ``matched`` is not boolean, ``reference_time`` is not real/finite/
        strictly increasing, the target-side indices are not ``arange(n_rows)``
        (including small-int-dtype overflow), ``matched`` does not equal
        source-index validity, an unmatched source index is not ``-1``, or
        matched indices on either side are not strictly increasing.
    """

    mode: Literal["nearest"]
    target: AlignmentTarget
    tol: float
    reference_time: np.ndarray
    neural_indices: np.ndarray
    behavior_indices: np.ndarray
    matched: np.ndarray

    def __post_init__(self) -> None:
        validate_choice(self.mode, ("nearest",), "mode")
        validate_choice(self.target, ("neural", "behavior"), "target")
        tol = validate_number(
            self.tol,
            "tol",
            kind="real",
            minimum=0,
            minimum_inclusive=True,
            finite=True,
            coerce=True,
        )

        arrays = (
            self.reference_time,
            self.neural_indices,
            self.behavior_indices,
            self.matched,
        )
        for arr in arrays:
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValidationError("AlignmentIndex arrays must be 1D ndarrays.")
        n_rows = self.reference_time.shape[0]
        for arr in arrays:
            if arr.shape[0] != n_rows:
                raise ValidationError("AlignmentIndex arrays must share the same length.")
        if not np.issubdtype(self.neural_indices.dtype, np.integer):
            raise ValidationError("AlignmentIndex.neural_indices must be integer.")
        if not np.issubdtype(self.behavior_indices.dtype, np.integer):
            raise ValidationError("AlignmentIndex.behavior_indices must be integer.")
        if not np.issubdtype(self.matched.dtype, np.bool_):
            raise ValidationError("AlignmentIndex.matched must be boolean.")

        # Normalize dtypes so caller-provided int8/float32 do not silently pass
        # (e.g. an int8 target index overflows past 127 but would match an arange
        # built in the same dtype). reference_time is checked for real/finite and
        # promoted to float64 via validate_real_array; complex/object dtypes are
        # rejected there rather than leaking a ValueError out of immutable copies.
        reference_time = validate_real_array(
            self.reference_time, "AlignmentIndex.reference_time", ndim=1, finite=True
        )
        if n_rows >= 2 and not np.all(np.diff(reference_time) > 0):
            raise ValidationError("AlignmentIndex.reference_time must be strictly increasing.")
        neural_indices = np.asarray(self.neural_indices, dtype=np.intp)
        behavior_indices = np.asarray(self.behavior_indices, dtype=np.intp)
        matched = np.asarray(self.matched, dtype=np.bool_)

        if self.target == "neural":
            target_side = neural_indices
            source_side = behavior_indices
        else:
            target_side = behavior_indices
            source_side = neural_indices
        expected_target = np.arange(n_rows, dtype=np.intp)
        if not np.array_equal(target_side, expected_target):
            raise ValidationError("AlignmentIndex target-side indices must be arange(n_rows).")
        # matched must be exactly source-index validity, and unmatched rows must
        # carry -1 (not some other negative) on the source side.
        if not np.array_equal(matched, source_side >= 0):
            raise ValidationError("AlignmentIndex.matched must equal source-index validity.")
        if source_side.size and np.any(source_side[~matched] != -1):
            raise ValidationError("AlignmentIndex unmatched source indices must be -1.")
        # Matched indices on both sides must be strictly increasing.
        matched_target = target_side[matched]
        matched_source = source_side[matched]
        if matched_target.size >= 2 and not np.all(np.diff(matched_target) > 0):
            raise ValidationError(
                "AlignmentIndex matched target indices must be strictly increasing."
            )
        if matched_source.size >= 2 and not np.all(np.diff(matched_source) > 0):
            raise ValidationError(
                "AlignmentIndex matched source indices must be strictly increasing."
            )

        # Back arrays with immutable bytes so writeability cannot be re-enabled,
        # keeping a frozen AlignmentIndex internally consistent.
        object.__setattr__(self, "tol", tol)
        object.__setattr__(self, "reference_time", immutable_array_copy(reference_time))
        object.__setattr__(self, "neural_indices", immutable_array_copy(neural_indices))
        object.__setattr__(self, "behavior_indices", immutable_array_copy(behavior_indices))
        object.__setattr__(self, "matched", immutable_array_copy(matched))


def to_reference_time(value: float, source: Clock | None) -> float:
    """Map local clock seconds into recording-reference seconds.

    ``reference_time = clock.offset + local_time * (1 + clock.drift)``. With
    ``source=None`` the value is already reference time and is returned
    unchanged (after a finiteness check).

    Parameters
    ----------
    value : float
        Local-clock seconds to convert. Must be finite.
    source : neurale.data.time.Clock or None
        Source clock, or ``None`` when ``value`` is already reference time.

    Returns
    -------
    float
        ``value`` expressed in recording-reference seconds.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``source`` is neither a :class:`Clock` nor ``None``, ``value`` is
        not finite, or the affine result overflows to a non-finite value.
    """

    source = _require_optional_clock(source, "source")
    return _to_reference_time(value, source)


def from_reference_time(value: float, target: Clock | None) -> float:
    """Map recording-reference seconds into local clock seconds.

    ``local_time = (reference_time - clock.offset) / (1 + clock.drift)``. With
    ``target=None`` the result stays in reference time.

    Parameters
    ----------
    value : float
        Recording-reference seconds to convert. Must be finite.
    target : neurale.data.time.Clock or None
        Target clock, or ``None`` to leave the result in reference time.

    Returns
    -------
    float
        ``value`` expressed in target-local seconds.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``target`` is neither a :class:`Clock` nor ``None``, ``value`` is
        not finite, or the affine result overflows to a non-finite value.
    """

    target = _require_optional_clock(target, "target")
    return _from_reference_time(value, target)


def convert_time(value: float, *, source: Clock | None, target: Clock | None) -> float:
    """Convert seconds between two synchronized clocks.

    Equivalent to
    ``from_reference_time(to_reference_time(value, source), target)``. When both
    clocks are present they must belong to the same synchronization domain.

    Parameters
    ----------
    value : float
        Time in ``source``-local seconds. Must be finite.
    source : neurale.data.time.Clock or None
        Source clock, or ``None`` when ``value`` is already reference time.
    target : neurale.data.time.Clock or None
        Target clock, or ``None`` to return reference time.

    Returns
    -------
    float
        ``value`` expressed in ``target``-local seconds.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``source`` or ``target`` is neither a :class:`Clock` nor ``None``,
        ``value`` is not finite, the two clocks belong to different
        synchronization domains, or either affine step overflows to a
        non-finite value.
    """

    source = _require_optional_clock(source, "source")
    target = _require_optional_clock(target, "target")
    return _convert_time(value, source=source, target=target)


def extract_signal_epoch(
    signal: SignalArray,
    start: float,
    stop: float,
    *,
    clock: Clock | None = None,
    discontinuities: EventSeries | None = None,
) -> SignalArray:
    """Extract a strict half-open ``[start, stop)`` epoch from a signal.

    ``start`` and ``stop`` are interpreted in ``clock`` coordinates and
    converted to the signal's own clock before slicing; ``clock=None`` means
    they are already in reference time. The interval must lie entirely within
    the signal's coverage ``[time[0], time[-1] + 1/fs)``; an
    out-of-range request is rejected rather than clipped. Any discontinuity
    overlapping the (converted) half-open interval rejects the epoch.

    Parameters
    ----------
    signal : neurale.data.arrays.SignalArray
        Signal to slice. Must have a regular time axis.
    start, stop : float
        Epoch boundaries in ``clock`` coordinates, half-open ``[start, stop)``.
    clock : neurale.data.time.Clock or None, default None
        Clock of ``start``/``stop``, or ``None`` for reference time.
    discontinuities : neurale.data.events.EventSeries or None, default None
        Gaps to reject on. Instantaneous gaps are barriers at their onset: an
        epoch ending at the barrier is valid, one spanning it is not. Sustained
        gaps use half-open overlap.

    Returns
    -------
    neurale.data.arrays.SignalArray
        The sliced epoch, preserving channel table, units, clock, name,
        sampling rate, and attrs.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``signal`` is not a :class:`SignalArray`,
        ``clock``/``discontinuities`` have wrong type, the time axis is
        irregular, the interval is out of range or selects no samples, or a
        discontinuity overlaps the interval.
    """

    signal = _require_signal(signal, "signal")
    clock = _require_optional_clock(clock, "clock")
    discontinuities = _require_optional_event_series(discontinuities, "discontinuities")
    _require_regular_time_axis(signal)
    start_local, stop_local = _converted_interval(start, stop, source=clock, target=signal.clock)
    if signal.n_samples == 0:
        raise ValidationError("cannot extract an epoch from an empty signal.")

    interval_start, interval_stop = _covered_interval(signal)
    if start_local < interval_start or stop_local > interval_stop:
        raise ValidationError("epoch interval is outside the signal time range.")
    if stop_local <= start_local:
        raise ValidationError("epoch interval selects no samples.")

    _reject_discontinuity_overlap(
        discontinuities,
        start_local,
        stop_local,
        source_clock=discontinuities.clock if discontinuities is not None else None,
        target_clock=signal.clock,
    )
    return signal.select_time(start_local, stop_local)


def extract_trial_epoch(
    signal: SignalArray,
    trial: Trial,
    *,
    tmin: float = 0.0,
    tmax: float | None = None,
    reference_clock: Clock | None = None,
    discontinuities: EventSeries | None = None,
) -> SignalArray:
    """Extract a trial-relative signal epoch
    ``[trial.start + tmin, trial.start + tmax)``.

    Trial boundaries are recording-reference seconds, so the epoch is converted
    to the signal's clock with ``source_clock=None``. ``reference_clock`` is
    used only to validate synchronization identity, not to transform the trial
    values. Negative ``tmin`` is allowed for pre-trial baseline windows; those
    samples are still subject to signal coverage and discontinuity checks.
    ``tmax=None`` uses ``trial.stop`` as the stop boundary, and the epoch stop
    must not exceed ``trial.stop``.

    A signal that carries a :class:`Clock` requires an explicit
    ``reference_clock`` sharing its synchronization domain — the same contract
    :func:`split_recording_trials` enforces — so a clocked trial source cannot
    be sliced without verifying its domain.

    Parameters
    ----------
    signal : neurale.data.arrays.SignalArray
        Signal to slice.
    trial : neurale.data.events.Trial
        Source trial. ``trial.start``/``trial.stop`` are reference seconds.
    tmin : float, default 0.0
        Start offset relative to ``trial.start``. May be negative.
    tmax : float or None, default None
        Stop offset relative to ``trial.start``; ``None`` means ``trial.stop``.
    reference_clock : neurale.data.time.Clock or None, default None
        Reference clock for synchronization-domain validation only. Required
        when ``signal`` carries a :class:`Clock`.
    discontinuities : neurale.data.events.EventSeries or None, default None
        Gaps to reject on (see :func:`extract_signal_epoch`).

    Returns
    -------
    neurale.data.arrays.SignalArray
        The sliced trial-relative epoch.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``trial`` is not a :class:`Trial`, ``reference_clock``/
        ``discontinuities`` have wrong type, ``signal`` carries a
        :class:`Clock` without an explicit ``reference_clock`` or with a
        mismatched synchronization domain, ``tmax <= tmin``, the epoch stop
        exceeds ``trial.stop``, or any :func:`extract_signal_epoch` check
        fails.
    """

    if not isinstance(trial, Trial):
        raise ValidationError("trial must be a Trial.")
    reference_clock = _require_optional_clock(reference_clock, "reference_clock")
    discontinuities = _require_optional_event_series(discontinuities, "discontinuities")
    if signal.clock is not None:
        # Mirror ``split_recording_trials``'s contract: a clocked signal
        # cannot be sliced on trial-relative reference seconds without an
        # explicit reference clock to validate its synchronization domain.
        # Without this gate a clocked ERP trial source would silently succeed
        # here while ``split_recording_trials`` rejects it.
        if reference_clock is None:
            raise ValidationError(
                "trial-relative epoch requires an explicit reference clock when "
                "the signal carries a clock."
            )
        _require_compatible_clocks(signal.clock, reference_clock)
    if discontinuities is not None and discontinuities.clock is not None:
        # Trial bounds live in the recording-reference domain, so a clocked
        # discontinuity must be verifiable against that domain. Fall back to the
        # signal clock (already checked compatible with reference_clock above);
        # if neither is available the discontinuity domain cannot be verified.
        reference_domain = reference_clock if reference_clock is not None else signal.clock
        _require_discontinuity_clock_compatibility(discontinuities.clock, reference_domain)
    start_offset = _finite_time(tmin, "tmin")
    stop_offset = trial.stop - trial.start if tmax is None else _finite_time(tmax, "tmax")
    if stop_offset <= start_offset:
        raise ValidationError("tmax must be greater than tmin.")
    if trial.start + stop_offset > trial.stop:
        raise ValidationError("trial-relative epoch stop must not exceed trial.stop.")
    start = _stable_reference_time(trial.start + start_offset)
    stop = _stable_reference_time(trial.start + stop_offset)
    return extract_signal_epoch(
        signal,
        start,
        stop,
        clock=None,
        discontinuities=discontinuities,
    )


def events_in_interval(
    events: EventSeries,
    start: float,
    stop: float,
    *,
    clock: Clock | None = None,
    required_labels: Iterable[str] = (),
    allow_missing: bool = False,
) -> EventSeries:
    """Filter events to a half-open ``[start, stop)`` onset interval.

    ``start``/``stop`` are interpreted in ``clock`` coordinates and converted
    to the events' own clock; ``clock=None`` means reference time. Required
    labels are strict by default: if a required label is absent from the
    selected events, the call raises unless ``allow_missing=True``.

    Parameters
    ----------
    events : neurale.data.events.EventSeries
        Events to filter.
    start, stop : float
        Onset interval boundaries in ``clock`` coordinates, half-open
        ``[start, stop)``.
    clock : neurale.data.time.Clock or None, default None
        Clock of ``start``/``stop``, or ``None`` for reference time.
    required_labels : Iterable[str], default ()
        Labels that must appear in the selected events.
    allow_missing : bool, default False
        If ``True``, do not raise when a required label is absent.

    Returns
    -------
    neurale.data.events.EventSeries
        Events whose onset lies in ``[start, stop)``, preserving the input
        clock and collection attrs.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``events`` is not an :class:`EventSeries`, ``allow_missing``/
        ``clock`` have wrong type, ``required_labels`` is a bare string or
        contains non-empty non-strings, ``stop <= start``, or a required label
        is missing and ``allow_missing`` is False.
    """

    if not isinstance(events, EventSeries):
        raise ValidationError("events must be an EventSeries.")
    allow_missing = _require_bool(allow_missing, "allow_missing")
    clock = _require_optional_clock(clock, "clock")
    required = _required_labels(required_labels)
    start_local, stop_local = _converted_interval(start, stop, source=clock, target=events.clock)
    selected = [
        event for event in events if event.onset >= start_local and event.onset < stop_local
    ]
    present = {event.label for event in selected}
    missing = [label for label in required if label not in present]
    if missing and not allow_missing:
        raise ValidationError("missing required events: " + ", ".join(missing) + ".")
    return EventSeries(selected, clock=events.clock, attrs=events.attrs)


def split_recording_trials(
    recording: Recording,
    *,
    reference_clock: Clock | None,
    signal_names: Sequence[str] | None = None,
    required_event_labels: Iterable[str] = (),
    discontinuities: EventSeries | None = None,
) -> tuple[TrialEpoch, ...]:
    """Split a recording into typed :class:`TrialEpoch` results.

    Each trial is sliced independently on every selected signal's own clock and
    sampling grid; trial boundaries (reference seconds) are converted to
    signal-local time with ``source_clock=None``. ``reference_clock`` validates
    synchronization identity only. It may be ``None`` only when every selected
    signal, the recording events, and the discontinuity series also have
    ``clock=None``; if any selected input carries a :class:`Clock`, an explicit
    ``reference_clock`` is required, and every clocked input must share its
    synchronization domain.

    Parameters
    ----------
    recording : neurale.data.recording.Recording
        Recording with a :class:`TrialTable` to split.
    reference_clock : neurale.data.time.Clock or None
        Reference clock for synchronization-domain validation. Required when
        any selected input carries a :class:`Clock`.
    signal_names : Sequence[str] or None, default None
        Signals to slice; ``None`` selects all. Must be unique non-empty
        strings; a bare string, duplicates, or unknown names are rejected
        before slicing.
    required_event_labels : Iterable[str], default ()
        Event labels required in every trial epoch.
    discontinuities : neurale.data.events.EventSeries or None, default None
        Gaps to reject on when they overlap any trial epoch interval.

    Returns
    -------
    tuple[TrialEpoch, ...]
        One frozen :class:`TrialEpoch` per trial, in trial order.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``recording`` is not a :class:`Recording` or has no trials,
        ``reference_clock``/``discontinuities`` have wrong type, the trial
        table is not sorted/non-overlapping, signal names are invalid,
        required labels are requested without events, a clocked input lacks
        an explicit reference clock or crosses synchronization domains, or any
        per-trial :func:`extract_signal_epoch`/:func:`events_in_interval` check
        fails.
    """

    if not isinstance(recording, Recording):
        raise ValidationError("recording must be a Recording.")
    if recording.trials is None:
        raise ValidationError("recording.trials is required for trial splitting.")
    reference_clock = _require_optional_clock(reference_clock, "reference_clock")
    discontinuities = _require_optional_event_series(discontinuities, "discontinuities")
    _validate_trial_table_for_splitting(recording.trials)
    names = _selected_signal_names(recording, signal_names)
    required_labels = _required_labels(required_event_labels)
    if required_labels and recording.events is None:
        raise ValidationError(
            "recording has no events but required event labels were requested: "
            + ", ".join(required_labels)
            + "."
        )
    _require_trial_splitting_clock_compatibility(
        recording,
        names,
        reference_clock,
        discontinuities,
    )

    epochs: list[TrialEpoch] = []
    for trial in recording.trials:
        epoch_start, epoch_stop = trial.start, trial.stop
        signals = {
            name: extract_signal_epoch(
                recording.signal(name),
                epoch_start,
                epoch_stop,
                clock=None,
                discontinuities=discontinuities,
            )
            for name in names
        }
        events = (
            None
            if recording.events is None
            else events_in_interval(
                recording.events,
                epoch_start,
                epoch_stop,
                clock=None,
                required_labels=required_labels,
            )
        )
        epochs.append(
            TrialEpoch(
                trial=trial,
                start=epoch_start,
                stop=epoch_stop,
                signals=signals,
                events=events,
                recording_metadata=recording.metadata,
                trial_attrs=trial.attrs,
            )
        )
    return tuple(epochs)


def align_neural_behavior_nearest(
    neural: SignalArray,
    behavior: SignalArray,
    *,
    target: AlignmentTarget = "neural",
    tol: float,
    discontinuities: EventSeries | Sequence[EventSeries] | None = (),
) -> AlignmentIndex:
    """Align neural and behavior samples by monotonic one-to-one nearest
    matching.

    Both signals' timestamps are mapped to recording-reference seconds;
    matching runs only on rows that fall in the two signals' common coverage
    ``intersection(neural_coverage, behavior_coverage)``. Target rows outside
    common coverage stay present but unmatched (``matched=False``, source
    index ``-1``), and source samples in a source-only tail never participate
    in a match -- so a gap in an ungated tail cannot feed a cross-gap match.

    For each in-coverage target row the matcher picks the nearest unused
    source row whose index is greater than the previous match (one-to-one,
    monotonic, no source reuse). A match is accepted when the absolute time
    error is ``<= tol`` (inclusive within float comparison tolerance);
    distance ties choose the earlier source row. Unmatched target rows keep
    source index ``-1``.

    Any discontinuity overlapping the common coverage in any input stream
    rejects the whole alignment; discontinuities in target-only or
    source-only tails do not. A discontinuity with a :class:`Clock` must share
    the alignment synchronization domain.

    Parameters
    ----------
    neural, behavior : neurale.data.arrays.SignalArray
        Signals to align. Both must be non-empty with strictly increasing
        timestamps.
    target : {"neural", "behavior"}, default "neural"
        Which signal's grid is the target. Its rows are all present in the
        result; the other signal is the source.
    tol : float
        Inclusive match tolerance in recording-reference seconds (``>= 0``).
    discontinuities : EventSeries or Sequence[EventSeries] or None, default ()
        Gap series to gate on. A bare :class:`EventSeries` is accepted and
        treated as a single-element sequence.

    Returns
    -------
    AlignmentIndex
        Frozen alignment index. The target-side index array equals
        ``arange(n_rows)``; the source-side array holds matched source rows or
        ``-1``; ``matched`` marks valid source indices.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``neural``/``behavior`` are not :class:`SignalArray`, ``target`` is
        invalid, ``tol`` is negative, ``discontinuities`` is not an
        :class:`EventSeries`/sequence of :class:`EventSeries`/``None``, either
        signal is empty or has non-strictly-increasing timestamps, the clocks
        belong to different synchronization domains, the signals have no common
        coverage, or a discontinuity overlaps the common coverage.
    """

    neural = _require_signal(neural, "neural")
    behavior = _require_signal(behavior, "behavior")
    target = validate_choice(target, ("neural", "behavior"), "target")
    tol = validate_number(
        tol,
        "tol",
        kind="real",
        minimum=0,
        minimum_inclusive=True,
        finite=True,
        coerce=True,
    )
    discontinuities = _require_discontinuity_sequence(discontinuities)
    if neural.n_samples == 0 or behavior.n_samples == 0:
        raise ValidationError("alignment requires non-empty neural and behavior signals.")
    _require_strictly_increasing_time(neural, "neural")
    _require_strictly_increasing_time(behavior, "behavior")
    alignment_clock = _require_alignment_clock_compatibility(neural.clock, behavior.clock)
    common_start, common_stop = _common_reference_coverage(neural, behavior)
    _reject_any_discontinuity(discontinuities, common_start, common_stop, alignment_clock)

    neural_ref = _to_reference_time_array(neural.time, neural.clock)
    behavior_ref = _to_reference_time_array(behavior.time, behavior.clock)
    target_ref = neural_ref if target == "neural" else behavior_ref
    source_ref = behavior_ref if target == "neural" else neural_ref
    target_indices = np.arange(target_ref.size, dtype=int)
    target_in_coverage = _within_common_coverage(target_ref, common_start, common_stop)
    # Both sides must live in common coverage: a source sample in a source-only
    # tail is not gated by discontinuities, so it must not participate in matches
    # either. In-coverage source rows are contiguous, so match on that slice and
    # restore original source indices afterwards.
    source_start, source_stop = _coverage_slice_bounds(
        _within_common_coverage(source_ref, common_start, common_stop)
    )
    slice_indices, matched = _nearest_one_to_one(
        target_ref,
        source_ref[source_start:source_stop],
        tol,
        target_in_coverage,
    )
    source_indices = np.where(matched, slice_indices + source_start, -1)

    if target == "neural":
        neural_indices = target_indices
        behavior_indices = source_indices
    else:
        neural_indices = source_indices
        behavior_indices = target_indices
    return AlignmentIndex(
        mode="nearest",
        target=target,
        tol=tol,
        reference_time=target_ref.copy(),
        neural_indices=neural_indices,
        behavior_indices=behavior_indices,
        matched=matched,
    )


__all__ = [
    "AlignmentIndex",
    "AlignmentTarget",
    "TrialEpoch",
    "align_neural_behavior_nearest",
    "convert_time",
    "events_in_interval",
    "extract_signal_epoch",
    "extract_trial_epoch",
    "from_reference_time",
    "split_recording_trials",
    "to_reference_time",
]
