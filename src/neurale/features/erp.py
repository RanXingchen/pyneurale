#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Offline typed event-related-potential extraction.

This module performs only epoch extraction, optional baseline correction, and
averaging. Filtering,
Hilbert envelopes, resampling, and whole-recording z-scoring remain explicit
signal/feature operations and are not hidden inside ERP extraction.

The epoch tensor axis order is ``(epoch, sample, channel)``. An averaged ERP
uses ``(sample, channel)``. ``time`` is a 1D sample axis relative
to each selected event onset or trial start, expressed on the source signal's
time grid. Multiple epochs must have equal lengths and matching relative grids;
no padding, interpolation, resampling, or NaN omission is performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

import neurale.data._time_validation as _time_validation
import neurale.data.alignment as _alignment
from neurale._validation import validate_number, validate_real_array
from neurale.data import (
    ChannelTable,
    Clock,
    Event,
    EventSeries,
    Recording,
    SignalArray,
    Trial,
    TrialEpoch,
    TrialTable,
)
from neurale.data._helpers import FrozenMapping, freeze_metadata, immutable_array_copy
from neurale.exceptions import ValidationError

ERPAnchor = Event | Trial
ERPAnchorSource = EventSeries | TrialTable | Literal["events", "trials"] | None
BaselineInterval = tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class ERPWaveform:
    """Trial-averaged ERP with sample-by-channel data and source metadata.

    The clock and anchor-metadata fields carry the same meaning as
    :class:`ERPEpochs`; see its docstring. ``selections`` here still names
    every epoch that went into the average.
    """

    data: np.ndarray
    time: np.ndarray
    fs: float
    source_clock: Clock | None
    channels: ChannelTable
    unit: str | tuple[str, ...]
    signal_name: str
    selections: tuple[ERPAnchor, ...]
    signal_attrs: tuple[Mapping[str, object], ...]
    recording_metadata: tuple[Mapping[str, object], ...]
    anchor_clock: Clock | None
    anchor_reference_time: np.ndarray
    anchor_collection_attrs: Mapping[str, object]
    baseline: BaselineInterval

    def __post_init__(self) -> None:
        _validate_and_commit_result(self, ndim=2, name="ERPWaveform.data", channel_axis=1)

    @property
    def n_epochs(self) -> int:
        """Return the number of single-trial epochs in the average."""
        return len(self.selections)


@dataclass(frozen=True, slots=True)
class ERPEpochs:
    """ERP epochs with epoch-by-sample-by-channel data and typed metadata.

    ``time`` is the anchor-relative sample axis shared across epochs.
    ``source_clock`` is the original signal clock kept as source metadata only;
    its offset does not apply to the relative axis. ``anchor_clock`` is the
    clock that interprets the raw anchor onsets in ``selections`` (the
    event-series clock for event anchors, ``None`` for trial anchors, whose
    time is already in recording reference coordinates).
    ``anchor_reference_time`` holds each anchor's recording-reference
    timestamp, so the full anchor coordinate is preserved losslessly alongside
    ``selections``. ``anchor_collection_attrs`` carries the source
    :class:`EventSeries` or :class:`TrialTable` collection attributes. Metadata
    mappings use the closed deep-frozen value domain documented by
    :mod:`neurale.data`.
    """

    data: np.ndarray
    time: np.ndarray
    fs: float
    source_clock: Clock | None
    channels: ChannelTable
    unit: str | tuple[str, ...]
    signal_name: str
    selections: tuple[ERPAnchor, ...]
    signal_attrs: tuple[Mapping[str, object], ...]
    recording_metadata: tuple[Mapping[str, object], ...]
    anchor_clock: Clock | None
    anchor_reference_time: np.ndarray
    anchor_collection_attrs: Mapping[str, object]
    baseline: BaselineInterval = None

    def __post_init__(self) -> None:
        _validate_and_commit_result(self, ndim=3, name="ERPEpochs.data", channel_axis=2)

    @property
    def n_epochs(self) -> int:
        """Return the number of selected epochs."""
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        """Return the number of samples per epoch."""
        return int(self.data.shape[1])

    @property
    def n_channels(self) -> int:
        """Return the number of channels."""
        return int(self.data.shape[2])

    def average(self) -> ERPWaveform:
        """Average trials with NumPy semantics, including NaN propagation."""
        return average_erp(self)


def erp_epochs(
    source: SignalArray | Recording | TrialEpoch | Sequence[TrialEpoch],
    anchors: ERPAnchorSource = None,
    *,
    tmin: float,
    tmax: float,
    signal_name: str = "neural",
    label: str | None = None,
    baseline: Sequence[float] | None = None,
    reference_clock: Clock | None = None,
    discontinuities: EventSeries | None = None,
) -> ERPEpochs:
    """Extract offline ERP epochs through the shared typed slicing API.

    ``SignalArray`` sources require an :class:`EventSeries` or
    :class:`TrialTable` in ``anchors``. ``Recording`` sources accept either
    collection directly or the explicit strings ``"events"`` / ``"trials"``
    to use the recording-owned collection. A single :class:`TrialEpoch` or a
    sequence of them uses its typed trial and selected signal; ``anchors`` must
    then be ``None``.

    ``tmin`` and ``tmax`` form a strict half-open interval relative to each
    anchor, with ``tmax > tmin``. They are always **recording-reference
    durations**: for an event anchor the raw onset is first mapped to reference
    time (through the event-series clock), then offset by ``tmin``/``tmax`` in
    reference seconds, so event-locked and trial-locked windows have identical
    duration semantics regardless of the event clock's drift. Explicit
    ``baseline=(start, end)`` is another half-open interval on the returned
    relative sample axis. It must lie within that axis and select at least one
    sample.

    An out-of-coverage epoch is rejected; there is no partial-clipping mode.
    The final selection is still rejected if epochs have unequal lengths or
    different relative sample grids. Empty selections and missing requested
    labels are errors. NaN values are retained, baseline means propagate them,
    and trial averaging uses :func:`numpy.mean` rather than ``nanmean``.
    """
    start_offset, stop_offset = _validated_epoch_interval(tmin, tmax)
    label = _validated_label(label)
    baseline_interval = _validated_baseline(baseline)
    reference_clock = _validated_optional_type(reference_clock, Clock, "reference_clock")
    discontinuities = _validated_optional_type(discontinuities, EventSeries, "discontinuities")
    signal_name = _validated_signal_name(signal_name)

    prepared = _prepare_sources(source, anchors, signal_name, label)
    _validate_prepared_clock_domains(prepared, reference_clock, discontinuities)
    extracted: list[SignalArray] = []
    relative_times: list[np.ndarray] = []
    for signal, anchor in zip(prepared.signals, prepared.selections, strict=True):
        epoch, relative_time = _extract_anchor_epoch(
            signal,
            anchor,
            anchor_clock=prepared.anchor_clock,
            tmin=start_offset,
            tmax=stop_offset,
            reference_clock=reference_clock,
            discontinuities=discontinuities,
        )
        extracted.append(epoch)
        relative_times.append(relative_time)

    _validate_compatible_epochs(extracted, relative_times)
    values = np.stack([epoch.data for epoch in extracted], axis=0)
    time = relative_times[0]
    if baseline_interval is not None:
        values = _apply_baseline(values, time, extracted[0].fs, baseline_interval)

    # Per-anchor reference timestamp (recording-reference seconds). For events
    # this maps the raw onset through the anchor collection's clock; for trials
    # it is ``trial.start`` (already reference seconds). Stored alongside the
    # raw ``selections`` and ``anchor_clock`` so a reader can recover the full
    # time coordinate of every anchor losslessly.
    anchor_reference_time = np.array(
        [
            _alignment.to_reference_time(anchor.onset, prepared.anchor_clock)
            if isinstance(anchor, Event)
            else float(anchor.start)
            for anchor in prepared.selections
        ],
        dtype=float,
    )

    return ERPEpochs(
        data=values,
        time=time,
        fs=extracted[0].fs,
        source_clock=extracted[0].clock,
        channels=extracted[0].channels,
        unit=extracted[0].unit,
        signal_name=extracted[0].name,
        selections=prepared.selections,
        signal_attrs=prepared.signal_attrs,
        recording_metadata=prepared.recording_metadata,
        anchor_clock=prepared.anchor_clock,
        anchor_reference_time=anchor_reference_time,
        anchor_collection_attrs=prepared.anchor_collection_attrs,
        baseline=baseline_interval,
    )


def average_erp(epochs: ERPEpochs) -> ERPWaveform:
    """Return the sample-by-channel arithmetic mean of typed ERP epochs."""
    if not isinstance(epochs, ERPEpochs):
        raise ValidationError("epochs must be an ERPEpochs instance.")
    return ERPWaveform(
        data=np.mean(epochs.data, axis=0),
        time=epochs.time,
        fs=epochs.fs,
        source_clock=epochs.source_clock,
        channels=epochs.channels,
        unit=epochs.unit,
        signal_name=epochs.signal_name,
        selections=epochs.selections,
        signal_attrs=epochs.signal_attrs,
        recording_metadata=epochs.recording_metadata,
        anchor_clock=epochs.anchor_clock,
        anchor_reference_time=epochs.anchor_reference_time,
        anchor_collection_attrs=epochs.anchor_collection_attrs,
        baseline=epochs.baseline,
    )


@dataclass(frozen=True, slots=True)
class _PreparedSources:
    signals: tuple[SignalArray, ...]
    selections: tuple[ERPAnchor, ...]
    anchor_clock: Clock | None
    anchor_collection_attrs: Mapping[str, object]
    signal_attrs: tuple[Mapping[str, object], ...]
    recording_metadata: tuple[Mapping[str, object], ...]


def _prepare_sources(
    source: SignalArray | Recording | TrialEpoch | Sequence[TrialEpoch],
    anchors: ERPAnchorSource,
    signal_name: str,
    label: str | None,
) -> _PreparedSources:
    if isinstance(source, SignalArray):
        collection = _validated_anchor_collection(anchors, owner="SignalArray")
        selections = _selected_anchors(collection, label)
        signals = (source,) * len(selections)
        return _PreparedSources(
            signals,
            selections,
            collection.clock if isinstance(collection, EventSeries) else None,
            _collection_attrs(collection),
            tuple(source.attrs for _ in selections),
            tuple(FrozenMapping() for _ in selections),
        )

    if isinstance(source, Recording):
        try:
            signal = source.signal(signal_name)
        except KeyError as exc:
            raise ValidationError(f"recording has no signal named {signal_name!r}.") from exc
        collection = _recording_anchor_collection(source, anchors)
        selections = _selected_anchors(collection, label)
        signals = (signal,) * len(selections)
        return _PreparedSources(
            signals,
            selections,
            collection.clock if isinstance(collection, EventSeries) else None,
            _collection_attrs(collection),
            tuple(signal.attrs for _ in selections),
            tuple(source.metadata for _ in selections),
        )

    trial_epochs = _validated_trial_epochs(source, anchors)
    selected_epochs = tuple(
        epoch for epoch in trial_epochs if label is None or epoch.trial.label == label
    )
    if not selected_epochs:
        raise ValidationError("ERP selection is empty or the requested label is missing.")
    signals: list[SignalArray] = []
    for epoch in selected_epochs:
        try:
            signals.append(epoch.signals[signal_name])
        except KeyError as exc:
            raise ValidationError(f"TrialEpoch has no signal named {signal_name!r}.") from exc
    return _PreparedSources(
        tuple(signals),
        tuple(epoch.trial for epoch in selected_epochs),
        None,
        FrozenMapping(),
        tuple(signal.attrs for signal in signals),
        tuple(epoch.recording_metadata for epoch in selected_epochs),
    )


def _validated_anchor_collection(
    anchors: ERPAnchorSource,
    *,
    owner: str,
) -> EventSeries | TrialTable:
    if not isinstance(anchors, (EventSeries, TrialTable)):
        raise ValidationError(f"{owner} ERP requires EventSeries or TrialTable anchors.")
    return anchors


def _recording_anchor_collection(
    recording: Recording,
    anchors: ERPAnchorSource,
) -> EventSeries | TrialTable:
    if anchors == "events":
        if recording.events is None:
            raise ValidationError("recording has no events for ERP selection.")
        return recording.events
    if anchors == "trials":
        if recording.trials is None:
            raise ValidationError("recording has no trials for ERP selection.")
        return recording.trials
    return _validated_anchor_collection(anchors, owner="Recording")


def _selected_anchors(
    collection: EventSeries | TrialTable,
    label: str | None,
) -> tuple[ERPAnchor, ...]:
    selected = tuple(item for item in collection if label is None or item.label == label)
    if not selected:
        raise ValidationError("ERP selection is empty or the requested label is missing.")
    return selected


def _collection_attrs(collection: EventSeries | TrialTable) -> Mapping[str, object]:
    # ``EventSeries`` stores ``attrs`` as ``freeze_metadata(...)`` which is ``None``
    # when the caller passed no attrs; ``TrialTable`` stores a (possibly empty)
    # dict. Normalize to a non-None Mapping so the result always carries typed
    # collection metadata.
    attrs = collection.attrs
    return attrs if isinstance(attrs, Mapping) else FrozenMapping()


def _validated_trial_epochs(
    source: TrialEpoch | Sequence[TrialEpoch],
    anchors: ERPAnchorSource,
) -> tuple[TrialEpoch, ...]:
    if anchors is not None:
        raise ValidationError("anchors must be None when source contains TrialEpoch values.")
    if isinstance(source, TrialEpoch):
        return (source,)
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise ValidationError(
            "source must be a SignalArray, Recording, TrialEpoch, or sequence of TrialEpoch."
        )
    values = tuple(source)
    if not values:
        raise ValidationError("TrialEpoch source must not be empty.")
    if not all(isinstance(value, TrialEpoch) for value in values):
        raise ValidationError("TrialEpoch source must contain only TrialEpoch values.")
    return values


def _extract_anchor_epoch(
    signal: SignalArray,
    anchor: ERPAnchor,
    *,
    anchor_clock: Clock | None,
    tmin: float,
    tmax: float,
    reference_clock: Clock | None,
    discontinuities: EventSeries | None,
) -> tuple[SignalArray, np.ndarray]:
    if isinstance(anchor, Event):
        # ``tmin``/``tmax`` are recording-reference durations for both event and
        # trial anchors. Map the raw onset to reference time first, then offset
        # by ``tmin``/``tmax`` in reference seconds, so an event-series clock with
        # drift does not stretch the window width the way it would if the offsets
        # were treated as event-clock-local durations.
        onset_reference = _alignment.to_reference_time(anchor.onset, anchor_clock)
        epoch = _alignment.extract_signal_epoch(
            signal,
            _stable_anchor_boundary(onset_reference, tmin),
            _stable_anchor_boundary(onset_reference, tmax),
            clock=None,
            discontinuities=discontinuities,
        )
        anchor_local = _alignment.from_reference_time(onset_reference, signal.clock)
    else:
        epoch_stop = _stable_anchor_boundary(anchor.start, tmax)
        if epoch_stop > anchor.stop:
            raise ValidationError("trial-relative ERP stop must not exceed trial.stop.")
        epoch = _alignment.extract_trial_epoch(
            signal,
            anchor,
            tmin=tmin,
            tmax=tmax,
            reference_clock=reference_clock,
            discontinuities=discontinuities,
        )
        anchor_local = _alignment.from_reference_time(anchor.start, signal.clock)
    return epoch, np.asarray(epoch.time - anchor_local, dtype=float)


def _validate_prepared_clock_domains(
    prepared: _PreparedSources,
    reference_clock: Clock | None,
    discontinuities: EventSeries | None,
) -> None:
    # Event anchors all come from one EventSeries and every prepared signal is
    # the same selected source. Validate their invariant clock set once before
    # entering the per-anchor extraction loop. Trial extraction retains its own
    # per-epoch validation because typed TrialEpoch inputs may carry distinct
    # signals and clocks.
    if not isinstance(prepared.selections[0], Event):
        return
    _time_validation._require_unified_sync_domain(
        reference_clock,
        prepared.signals[0].clock,
        prepared.anchor_clock,
        discontinuities.clock if discontinuities is not None else None,
    )


def _stable_anchor_boundary(anchor: float, offset: float) -> float:
    # Round to guard against decimal addition drift accidentally including the
    # half-open stop sample.
    return round(anchor + offset, 12)


def _validate_compatible_epochs(
    epochs: Sequence[SignalArray],
    relative_times: Sequence[np.ndarray],
) -> None:
    reference = epochs[0]
    reference_time = relative_times[0]
    tol = np.finfo(float).eps * max(16.0, float(np.max(np.abs(reference_time))))
    for epoch, time in zip(epochs[1:], relative_times[1:], strict=True):
        if epoch.data.shape != reference.data.shape:
            raise ValidationError("ERP epochs have unequal sample or channel counts.")
        if epoch.fs != reference.fs:
            raise ValidationError("ERP epochs have unequal sampling rates.")
        if epoch.channels != reference.channels or epoch.unit != reference.unit:
            raise ValidationError("ERP epochs have incompatible channel metadata or units.")
        if epoch.clock != reference.clock:
            raise ValidationError("ERP epochs have incompatible clocks.")
        if epoch.name != reference.name:
            raise ValidationError("ERP epochs have incompatible signal names.")
        if not np.allclose(time, reference_time, rtol=0.0, atol=tol):
            raise ValidationError("ERP epochs have unequal relative sample grids.")


def _baseline_mask(time: np.ndarray, fs: float, baseline: tuple[float, float]) -> np.ndarray:
    start, stop = baseline
    coverage_start = float(time[0])
    coverage_stop = float(time[-1] + 1.0 / fs)
    tol = np.finfo(float).eps * max(16.0, abs(coverage_start), abs(coverage_stop))
    if start < coverage_start - tol or stop > coverage_stop + tol:
        raise ValidationError("baseline interval is outside the extracted epoch.")
    mask = (time >= start) & (time < stop)
    if not np.any(mask):
        raise ValidationError("baseline interval selects no samples.")
    return mask


def _apply_baseline(
    values: np.ndarray,
    time: np.ndarray,
    fs: float,
    baseline: tuple[float, float],
) -> np.ndarray:
    mask = _baseline_mask(time, fs, baseline)
    corrected = values.astype(np.result_type(values.dtype, np.float64), copy=True)
    corrected -= np.mean(corrected[:, mask, :], axis=1, keepdims=True)
    return corrected


def _validated_epoch_interval(tmin: float, tmax: float) -> tuple[float, float]:
    start = float(validate_number(tmin, "tmin", kind="real", finite=True, coerce=True))
    stop = float(validate_number(tmax, "tmax", kind="real", finite=True, coerce=True))
    if stop <= start:
        raise ValidationError("tmax must be greater than tmin.")
    return start, stop


def _validated_baseline(baseline: Sequence[float] | None) -> BaselineInterval:
    if baseline is None:
        return None
    if isinstance(baseline, (str, bytes)) or not isinstance(baseline, Sequence):
        raise ValidationError("baseline must be a two-value [start, end) interval or None.")
    if len(baseline) != 2:
        raise ValidationError("baseline must contain exactly start and end.")
    start, stop = _validated_epoch_interval(baseline[0], baseline[1])
    return start, stop


def _validated_baseline_coverage(
    baseline: tuple[float, float],
    time: np.ndarray,
    fs: float,
) -> None:
    # Mirror the coverage/mask logic of ``_apply_baseline`` so a hand-constructed
    # result object cannot carry a baseline interval that lies outside the
    # extracted epoch or selects no samples. ``_apply_baseline`` only runs on the
    # ``erp_epochs`` construction path; the public dataclass constructor must
    # enforce the same invariant independently.
    _baseline_mask(time, fs, baseline)


def _validated_label(label: str | None) -> str | None:
    if label is not None and (not isinstance(label, str) or not label):
        raise ValidationError("label must be a non-empty string or None.")
    return label


def _validated_signal_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValidationError("signal_name must be a non-empty string.")
    return name


def _validated_optional_type(value, expected_type, name: str):
    if value is not None and not isinstance(value, expected_type):
        raise ValidationError(f"{name} must be {expected_type.__name__} or None.")
    return value


def _validated_result_data(data: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    if not isinstance(data, np.ndarray) or data.ndim != ndim:
        raise ValidationError(f"{name} must be a {ndim}D numpy.ndarray.")
    if any(size == 0 for size in data.shape):
        raise ValidationError(f"{name} dimensions must not be empty.")
    if not (np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)):
        raise ValidationError(f"{name} must contain numeric data.")
    return data


def _validated_result_time(
    time: np.ndarray,
    n_samples: int,
    fs: float,
) -> tuple[np.ndarray, float]:
    time = validate_real_array(time, "ERP time", ndim=1, finite=True)
    if time.size != n_samples:
        raise ValidationError("ERP time must be a 1D ndarray matching the sample axis.")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValidationError("ERP time must be strictly increasing.")
    rate = float(
        validate_number(
            fs,
            "fs",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    # The relative axis inherits the source signal's regular grid (it is
    # ``epoch.time - anchor_local``), so its spacing must equal ``1/fs``.
    # Mirror the tolerance used by ``_alignment._require_regular_time_axis`` so the
    # public dataclass cannot be hand-assembled with a grid that contradicts its
    # declared sampling rate (e.g. fs=10 with a 0.15 s spacing).
    if time.size > 1:
        expected = 1.0 / rate
        if not np.allclose(np.diff(time), expected, rtol=1e-9, atol=1e-12):
            raise ValidationError("ERP time grid must be regular with period 1/fs.")
    return np.asarray(time, dtype=float), rate


def _validated_anchor_reference_time(value: object, n_epochs: int) -> np.ndarray:
    # Reuse the project's real-array validator so a complex dtype is rejected
    # (``"anchor_reference_time must contain real-valued data."``) instead of
    # being silently cast to float and dropping its imaginary part.
    arr = validate_real_array(value, "anchor_reference_time", ndim=1, finite=True)
    if arr.size != n_epochs:
        raise ValidationError("anchor_reference_time must be a 1D ndarray matching the epoch axis.")
    return arr


def _anchor_timestamp_matches(stored: float, expected: float) -> bool:
    # Absolute timestamp consistency uses a float-ULP-scale absolute tolerance
    # only (rtol=0). A non-zero rtol would grow with the timestamp magnitude and
    # admit sub-second -- at 1e9 s even ~1 s -- drift, breaking the "lossless
    # anchor coordinate" promise. At 1e9 s this tolerance is sub-microsecond.
    tol = np.finfo(float).eps * max(16.0, abs(expected))
    return np.isclose(stored, expected, rtol=0.0, atol=tol)


def _validated_anchor_consistency(
    selections: tuple[ERPAnchor, ...],
    anchor_clock: Clock | None,
    anchor_reference_time: np.ndarray,
) -> None:
    # The (selections, anchor_clock, anchor_reference_time) triple must be
    # self-consistent so the docstring's "full anchor coordinate preserved
    # losslessly" promise holds for hand-constructed results too, not only for
    # those produced by ``erp_epochs``. Event and Trial anchors obey different
    # rules and must not be mixed within one result.
    are_events = [isinstance(item, Event) for item in selections]
    if all(are_events):
        for selection, stored in zip(selections, anchor_reference_time, strict=True):
            expected = _alignment.to_reference_time(selection.onset, anchor_clock)
            if not _anchor_timestamp_matches(float(stored), expected):
                raise ValidationError(
                    "anchor_reference_time must equal to_reference_time(event.onset, anchor_clock)."
                )
        return
    if not any(are_events):
        # All trials: their bounds are already recording-reference seconds, so no
        # anchor clock may be present and each reference timestamp is trial.start.
        if anchor_clock is not None:
            raise ValidationError("anchor_clock must be None for trial selections.")
        for selection, stored in zip(selections, anchor_reference_time, strict=True):
            if not _anchor_timestamp_matches(float(stored), float(selection.start)):
                raise ValidationError(
                    "anchor_reference_time must equal trial.start for trial selections."
                )
        return
    raise ValidationError("ERP selections must not mix Event and Trial anchors.")


def _validated_result_metadata(
    *,
    fs: float,
    time: np.ndarray,
    source_clock: Clock | None,
    anchor_clock: Clock | None,
    anchor_collection_attrs: Mapping[str, object],
    channels: ChannelTable,
    unit: str | tuple[str, ...],
    signal_name: str,
    selections: tuple[ERPAnchor, ...],
    signal_attrs: tuple[Mapping[str, object], ...],
    recording_metadata: tuple[Mapping[str, object], ...],
    baseline: BaselineInterval,
    n_channels: int,
    n_epochs: int,
) -> dict[str, object]:
    rate = float(
        validate_number(
            fs,
            "fs",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            coerce=True,
        )
    )
    if source_clock is not None and not isinstance(source_clock, Clock):
        raise ValidationError("source_clock must be a Clock or None.")
    if anchor_clock is not None and not isinstance(anchor_clock, Clock):
        raise ValidationError("anchor_clock must be a Clock or None.")
    # The signal time axis and the anchor onsets must live in the same
    # synchronization domain, or the result's "full anchor coordinate preserved
    # losslessly" promise is broken: a reader could not recover a consistent
    # reference coordinate from two clocks of different sessions. Trial results
    # carry ``anchor_clock=None`` and are unconstrained.
    _time_validation._require_unified_sync_domain(source_clock, anchor_clock)
    if not isinstance(anchor_collection_attrs, Mapping):
        raise ValidationError("anchor_collection_attrs must be a mapping.")
    if not isinstance(channels, ChannelTable) or len(channels) != n_channels:
        raise ValidationError("channels must match the ERP channel axis.")
    units = (unit,) * n_channels if isinstance(unit, str) else tuple(unit)
    if len(units) != n_channels or list(units) != channels.units:
        raise ValidationError("unit metadata must match ERP channels.")
    if n_epochs == 0:
        raise ValidationError("ERP selections must not be empty.")
    if len(selections) != n_epochs or not all(
        isinstance(item, (Event, Trial)) for item in selections
    ):
        raise ValidationError("selections must match the ERP epoch axis.")
    if len(signal_attrs) != n_epochs or len(recording_metadata) != n_epochs:
        raise ValidationError("per-epoch metadata must match the ERP epoch axis.")
    if not all(isinstance(value, Mapping) for value in (*signal_attrs, *recording_metadata)):
        raise ValidationError("ERP metadata entries must be mappings.")
    baseline_interval = _validated_baseline(baseline)
    if baseline_interval is not None:
        _validated_baseline_coverage(baseline_interval, time, rate)
    return {
        "fs": rate,
        "source_clock": source_clock,
        "anchor_clock": anchor_clock,
        "anchor_collection_attrs": freeze_metadata(anchor_collection_attrs),
        "channels": channels,
        "unit": unit if isinstance(unit, str) else tuple(unit),
        "signal_name": _validated_signal_name(signal_name),
        "selections": tuple(selections),
        "signal_attrs": tuple(freeze_metadata(value) for value in signal_attrs),
        "recording_metadata": tuple(freeze_metadata(value) for value in recording_metadata),
        "baseline": baseline_interval,
    }


def _set_result_metadata(result: ERPEpochs | ERPWaveform, metadata: Mapping[str, object]) -> None:
    for name, value in metadata.items():
        object.__setattr__(result, name, value)


def _validate_and_commit_result(
    result: ERPEpochs | ERPWaveform,
    *,
    ndim: int,
    name: str,
    channel_axis: int,
) -> None:
    """Validate and freeze the shared ERP result fields in place.

    ``ERPWaveform`` (2D, sample-by-channel) and ``ERPEpochs`` (3D,
    epoch-by-sample-by-channel) share one validation, metadata commit, and
    anchor-consistency check. The shape contract is derived from ``ndim``:
    2D uses ``data.shape[0]`` for the time axis and ``len(selections)`` for the
    epoch count; 3D uses ``data.shape[1]`` for the time axis and ``data.shape[0]``
    for the epoch count.
    """
    data = _validated_result_data(result.data, ndim=ndim, name=name)
    sample_axis = 0 if ndim == 2 else 1
    time, _rate = _validated_result_time(result.time, data.shape[sample_axis], result.fs)
    n_epochs = len(result.selections) if ndim == 2 else data.shape[0]
    anchor_reference_time = _validated_anchor_reference_time(result.anchor_reference_time, n_epochs)
    metadata = _validated_result_metadata(
        fs=result.fs,
        time=time,
        source_clock=result.source_clock,
        anchor_clock=result.anchor_clock,
        anchor_collection_attrs=result.anchor_collection_attrs,
        channels=result.channels,
        unit=result.unit,
        signal_name=result.signal_name,
        selections=result.selections,
        signal_attrs=result.signal_attrs,
        recording_metadata=result.recording_metadata,
        baseline=result.baseline,
        n_channels=data.shape[channel_axis],
        n_epochs=n_epochs,
    )
    object.__setattr__(result, "data", immutable_array_copy(data))
    object.__setattr__(result, "time", immutable_array_copy(time))
    object.__setattr__(result, "anchor_reference_time", immutable_array_copy(anchor_reference_time))
    _set_result_metadata(result, metadata)
    _validated_anchor_consistency(result.selections, result.anchor_clock, anchor_reference_time)


__all__ = ["ERPEpochs", "ERPWaveform", "average_erp", "erp_epochs"]
