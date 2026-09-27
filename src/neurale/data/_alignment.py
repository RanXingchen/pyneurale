#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Private implementation helpers for trial slicing and alignment.

The public dataclasses, functions, and type aliases live in
:mod:`neurale.data.alignment`; this module holds the underlying matching,
coverage, clock/discontinuity, and trial-splitting helpers plus the low-level
time-conversion primitives they share. The public time functions in
``alignment`` are thin wrappers over ``_to_reference_time`` /
``_from_reference_time`` / ``_convert_time`` so that the public module owns the
API surface (and pickle ``__module__``) while helpers call the private
primitives directly, keeping the dependency direction one-way.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from neurale.exceptions import ValidationError

from ._time_validation import _require_unified_sync_domain
from .arrays import SignalArray
from .events import EventSeries, TrialTable
from .recording import Recording
from .time import Clock


def _finite_time(value: float, name: str) -> float:
    try:
        time = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be finite.") from exc
    if not np.isfinite(time):
        raise ValidationError(f"{name} must be finite.")
    return time


def _stable_reference_time(value: float) -> float:
    # Trial-relative offsets are often decimal values; normalize addition
    # artifacts so intended sample-boundary intervals remain half-open.
    return round(value, 12)


def _to_reference_time(value: float, source: Clock | None) -> float:
    """Map local clock seconds into recording-reference seconds."""

    time = _finite_time(value, "value")
    if source is None:
        return time
    result = source.offset + time * (1.0 + source.drift)
    if not np.isfinite(result):
        raise ValidationError("to_reference_time produced a non-finite value.")
    return result


def _to_reference_time_array(time: np.ndarray, source: Clock | None) -> np.ndarray:
    """Vectorized affine mapping of a whole time grid into reference seconds.

    Equivalent to ``_to_reference_time`` applied per element, but performed as a
    single array affine transform to avoid per-sample Python call overhead in
    the alignment hot path. ``time`` is a :class:`SignalArray` grid, which is
    finite by construction, so only the affine result is checked for overflow.
    """

    if source is None:
        return time.astype(float, copy=False)
    result = source.offset + time * (1.0 + source.drift)
    if not np.all(np.isfinite(result)):
        raise ValidationError("to_reference_time produced a non-finite value.")
    return result


def _from_reference_time(value: float, target: Clock | None) -> float:
    """Map recording-reference seconds into local clock seconds."""

    time = _finite_time(value, "value")
    if target is None:
        return time
    result = (time - target.offset) / (1.0 + target.drift)
    if not np.isfinite(result):
        raise ValidationError("from_reference_time produced a non-finite value.")
    return result


def _convert_time(value: float, *, source: Clock | None, target: Clock | None) -> float:
    """Convert seconds between synchronized clocks."""

    if source is not None and target is not None:
        _require_compatible_clocks(source, target)
    return _from_reference_time(_to_reference_time(value, source), target)


def _require_compatible_clocks(source: Clock, target: Clock) -> None:
    _require_unified_sync_domain(source, target)


def _require_alignment_clock_compatibility(
    neural_clock: Clock | None,
    behavior_clock: Clock | None,
) -> Clock | None:
    if neural_clock is not None and behavior_clock is not None:
        _require_compatible_clocks(neural_clock, behavior_clock)
        return neural_clock
    return neural_clock or behavior_clock


def _require_discontinuity_clock_compatibility(
    discontinuity_clock: Clock | None,
    alignment_clock: Clock | None,
) -> None:
    if discontinuity_clock is None:
        return
    if alignment_clock is None:
        raise ValidationError("discontinuity clock requires an explicit alignment reference clock.")
    _require_compatible_clocks(discontinuity_clock, alignment_clock)


def _require_regular_time_axis(signal: SignalArray) -> None:
    if signal.n_samples < 2:
        return
    expected = 1.0 / signal.fs
    deltas = np.diff(signal.time)
    if not np.allclose(deltas, expected, rtol=1e-9, atol=1e-12):
        raise ValidationError("epoch extraction requires a regular SignalArray time axis.")


def _require_strictly_increasing_time(signal: SignalArray, name: str) -> None:
    # SignalArray only guarantees monotonic non-decreasing time, but the
    # nearest matcher assumes a strictly increasing grid (its forward-moving
    # pointer would skip over repeated timestamps and break the earlier-source
    # tiebreak). Reject duplicates explicitly rather than mis-matching.
    if signal.n_samples >= 2 and not np.all(np.diff(signal.time) > 0):
        raise ValidationError(
            f"nearest alignment requires strictly increasing observation timestamps in {name}."
        )


def _require_signal(value: object, name: str) -> SignalArray:
    if not isinstance(value, SignalArray):
        raise ValidationError(f"{name} must be a SignalArray.")
    return value


def _require_optional_clock(value: object, name: str) -> Clock | None:
    if value is None:
        return None
    if not isinstance(value, Clock):
        raise ValidationError(f"{name} must be a Clock.")
    return value


def _require_bool(value: object, name: str) -> bool:
    # Guard against truthy non-bools (e.g. allow_missing="false") being silently
    # interpreted as True.
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a bool.")
    return value


def _require_optional_event_series(value: object, name: str) -> EventSeries | None:
    if value is None:
        return None
    if not isinstance(value, EventSeries):
        raise ValidationError(f"{name} must be an EventSeries or None.")
    return value


def _require_discontinuity_sequence(value: object) -> tuple[EventSeries, ...]:
    # Accept None, a bare EventSeries, or a sequence of EventSeries, and
    # normalize to a tuple of EventSeries. A bare EventSeries is iterable as a
    # sequence of events, so it must be caught before the Sequence branch to
    # avoid it being iterated and failing deep on .clock access.
    if value is None:
        return ()
    if isinstance(value, EventSeries):
        return (value,)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError(
            "discontinuities must be an EventSeries, a sequence of EventSeries, or None."
        )
    items = tuple(value)
    for item in items:
        if not isinstance(item, EventSeries):
            raise ValidationError("discontinuities must contain only EventSeries instances.")
    return items


def _required_labels(required_labels: Iterable[str] | None) -> tuple[str, ...]:
    # ``None`` means "no required labels" (consistent with discontinuities=None);
    # bare strings and other non-iterables are rejected with ValidationError
    # rather than leaking a TypeError out of tuple().
    if required_labels is None:
        return ()
    if isinstance(required_labels, str):
        raise ValidationError("required_labels must be an iterable of labels, not a string.")
    try:
        labels = tuple(required_labels)
    except TypeError as exc:
        raise ValidationError("required_labels must be an iterable of non-empty strings.") from exc
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ValidationError("required_labels must contain non-empty strings.")
    return labels


def _strictly_before(left: float, right: float) -> bool:
    return left < right and not np.isclose(left, right, rtol=1e-12, atol=1e-15)


def _strictly_after(left: float, right: float) -> bool:
    return left > right and not np.isclose(left, right, rtol=1e-12, atol=1e-15)


def _covered_interval(signal: SignalArray) -> tuple[float, float]:
    sample_period = 1.0 / signal.fs
    # The start is a raw sample time with no addition artifact, so it is left
    # unrounded: rounding it could push a boundary sample (e.g. the first sample
    # of the later-starting signal, which sits exactly at common coverage start)
    # just outside the interval. The stop is time[-1] + period, an addition that
    # can accumulate float error, so it is normalized to keep the half-open
    # boundary stable.
    return (
        float(signal.time[0]),
        _stable_reference_time(float(signal.time[-1]) + sample_period),
    )


def _reference_coverage(signal: SignalArray) -> tuple[float, float]:
    start, stop = _covered_interval(signal)
    return (
        _to_reference_time(start, signal.clock),
        _to_reference_time(stop, signal.clock),
    )


def _common_reference_coverage(
    neural: SignalArray,
    behavior: SignalArray,
) -> tuple[float, float]:
    neural_start, neural_stop = _reference_coverage(neural)
    behavior_start, behavior_stop = _reference_coverage(behavior)
    start = max(neural_start, behavior_start)
    stop = min(neural_stop, behavior_stop)
    if stop <= start:
        raise ValidationError("neural and behavior signals have no common time coverage.")
    return start, stop


def _within_common_coverage(
    values: np.ndarray,
    start: float,
    stop: float,
) -> np.ndarray:
    # Half-open [start, stop) membership with the same float tolerance the
    # module uses elsewhere, so sample-boundary target rows are classified
    # consistently with the discontinuity gate.
    lower_ok = (values >= start) | np.isclose(values, start, rtol=1e-12, atol=1e-15)
    upper_ok = (values < stop) & ~np.isclose(values, stop, rtol=1e-12, atol=1e-15)
    return lower_ok & upper_ok


def _coverage_slice_bounds(in_coverage: np.ndarray) -> tuple[int, int]:
    # In-coverage rows of a sorted reference grid form a single contiguous block
    # (the grid is strictly increasing and common coverage is an interval), so
    # the eligible slice is [first True, last True + 1). Returns an empty slice
    # when no row is in coverage.
    indices = np.flatnonzero(in_coverage)
    if indices.size == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1]) + 1


def _converted_interval(
    start: float,
    stop: float,
    *,
    source: Clock | None,
    target: Clock | None,
) -> tuple[float, float]:
    start_value = _convert_time(start, source=source, target=target)
    stop_value = _convert_time(stop, source=source, target=target)
    if stop_value <= start_value:
        raise ValidationError("stop must be greater than start.")
    return start_value, stop_value


def _reject_discontinuity_overlap(
    discontinuities: EventSeries | None,
    start: float,
    stop: float,
    *,
    source_clock: Clock | None,
    target_clock: Clock | None,
) -> None:
    if discontinuities is None:
        return
    for event in discontinuities:
        event_start = _convert_time(event.onset, source=source_clock, target=target_clock)
        event_stop = _convert_time(event.stop, source=source_clock, target=target_clock)
        if event.duration == 0.0:
            overlaps = event_start >= start and _strictly_before(event_start, stop)
        else:
            overlaps = _strictly_before(event_start, stop) and _strictly_after(event_stop, start)
        if overlaps:
            raise ValidationError("epoch interval crosses a discontinuity.")


def _reject_any_discontinuity(
    discontinuities: Sequence[EventSeries],
    common_start: float,
    common_stop: float,
    alignment_clock: Clock | None,
) -> None:
    for series in discontinuities:
        _require_discontinuity_clock_compatibility(series.clock, alignment_clock)
        _reject_discontinuity_overlap(
            series,
            common_start,
            common_stop,
            source_clock=series.clock,
            target_clock=None,
        )


def _validate_trial_table_for_splitting(trials: TrialTable) -> None:
    previous_stop: float | None = None
    for trial in trials:
        if previous_stop is not None and trial.start < previous_stop:
            raise ValidationError("trial splitting requires non-overlapping trials in start order.")
        previous_stop = trial.stop


def _selected_signal_names(
    recording: Recording,
    signal_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if signal_names is None:
        names = tuple(recording.signals)
    elif isinstance(signal_names, str):
        raise ValidationError("signal_names must be an iterable of names, not a string.")
    else:
        try:
            names = tuple(signal_names)
        except TypeError as exc:
            raise ValidationError("signal_names must be an iterable of names.") from exc
    if not names:
        raise ValidationError("at least one signal is required for trial splitting.")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValidationError("signal_names must contain non-empty strings.")
    if len(names) != len(set(names)):
        raise ValidationError("signal_names must not contain duplicate names.")
    missing = [name for name in names if name not in recording.signals]
    if missing:
        raise ValidationError("unknown signal names: " + ", ".join(missing) + ".")
    return names


def _require_trial_splitting_clock_compatibility(
    recording: Recording,
    signal_names: Sequence[str],
    reference_clock: Clock | None,
    discontinuities: EventSeries | None,
) -> None:
    clocked_signals = [name for name in signal_names if recording.signal(name).clock is not None]
    events_clock = recording.events.clock if recording.events is not None else None
    discontinuity_clock = discontinuities.clock if discontinuities is not None else None

    if reference_clock is None:
        if clocked_signals or events_clock is not None or discontinuity_clock is not None:
            raise ValidationError(
                "trial splitting requires an explicit reference clock when any "
                "selected signal, events, or discontinuities carry a clock."
            )
        return

    for name in clocked_signals:
        _require_compatible_clocks(recording.signal(name).clock, reference_clock)
    if events_clock is not None:
        _require_compatible_clocks(events_clock, reference_clock)
    if discontinuity_clock is not None:
        _require_compatible_clocks(discontinuity_clock, reference_clock)


def _nearest_one_to_one(
    target_time: np.ndarray,
    source_time: np.ndarray,
    tol: float,
    in_coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # One-to-one monotonic nearest matching. Both grids are strictly increasing
    # (signal time is monotonic and to_reference_time preserves order), and the
    # matched source index strictly increases across targets, so the eligible
    # sources for a target are exactly the suffix ``source_time[last+1:]``. The
    # nearest eligible source must bracket the target value, so it is one of the
    # two candidates around the insertion point; a single forward-moving
    # pointer ``p`` locates them. This is O(n_targets + n_sources) with no
    # per-target distance/sort allocations, replacing the prior per-target scan
    # and lexsort of the whole source grid.
    n_targets = target_time.shape[0]
    n_sources = source_time.shape[0]
    indices = np.full(n_targets, -1, dtype=int)
    matched = np.zeros(n_targets, dtype=bool)
    if n_sources == 0:
        return indices, matched
    last_source_idx = -1
    p = 0  # first source index >= the current target value (monotonic, never rewinds)
    for target_idx in range(n_targets):
        if in_coverage is not None and not in_coverage[target_idx]:
            # Target rows outside common coverage stay present but unmatched.
            continue
        value = target_time[target_idx]
        lo = last_source_idx + 1
        if lo >= n_sources:
            continue  # no eligible sources remain
        if p < lo:
            p = lo
        while p < n_sources and source_time[p] < value:
            p += 1
        # Candidates that bracket ``value``: ``left`` is the greatest eligible
        # source < value, ``right`` is the least eligible source >= value.
        best = -1
        best_distance = 0.0
        left = p - 1
        if left >= lo:
            best = left
            best_distance = abs(source_time[left] - value)
        if p < n_sources:
            distance = abs(source_time[p] - value)
            # Strictly closer wins; on a tie keep ``left`` (smaller index),
            # matching the old lexsort tiebreak by ascending source index.
            if best == -1 or distance < best_distance:
                best = p
                best_distance = distance
        if best == -1:
            continue
        if best_distance <= tol or np.isclose(best_distance, tol, rtol=1e-12, atol=1e-15):
            indices[target_idx] = best
            matched[target_idx] = True
            last_source_idx = best
    return indices, matched
