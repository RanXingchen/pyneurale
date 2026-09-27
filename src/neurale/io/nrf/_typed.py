#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Convert committed NRF data into the typed objects in :mod:`neurale.data`.

Specification reference: ``README.md`` section 10.

The dependency direction is ``neurale.io -> neurale.data`` and never the
reverse: the data model knows nothing about storage.

These helpers convert only mappings the specification makes unambiguous. Where a
record kind has no single typed home -- experiment state, commands, task
variables, targets, assistance, drops -- the reader exposes the rows rather than
inventing a mapping, because silently dropping a recorded record would make an
incomplete reconstruction look complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neurale.data import (
    ChannelInfo,
    ChannelTable,
    Clock,
    Event,
    EventSeries,
    FeatureMatrix,
    Recording,
    SignalArray,
    Trial,
    TrialTable,
)

from ._errors import NrfSemanticError

NANOSECONDS = 1_000_000_000

#: Record kinds with an unambiguous typed destination.
MAPPED_RECORD_KINDS = frozenset({"events", "trials"})

#: Manifest metadata key holding everything a :class:`Recording` carries that
#: the NRF registries have no field for. Namespaced so it can never collide
#: with a format-defined manifest member.
RECORDING_METADATA_KEY = "recording"

#: Recording-metadata keys this module generates. The prefix is reserved: the
#: writer strips it before persisting user metadata, so ``write(read(x))`` does
#: not accumulate bookkeeping.
RESERVED_METADATA_PREFIX = "nrf_"

#: Clock-descriptor metadata keys carrying typed values the NRF clock
#: vocabulary cannot express verbatim.
CLOCK_TYPE_KEY = "neurale_clock_type"
CLOCK_DOMAIN_KEY = "neurale_synchronization_domain"
SYNTHETIC_CLOCK_KEY = "neurale_synthetic_clock"


def _seconds(value: Any) -> float:
    return float(value) / NANOSECONDS


def _seconds_array(timestamps: np.ndarray | None) -> np.ndarray | None:
    """Return nanosecond *timestamps* as seconds, preserving ``None``."""
    if timestamps is None:
        return None
    return np.asarray(timestamps, dtype=float) / NANOSECONDS


def _units_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {unit["id"]: unit for unit in manifest["units"]}


def _regular_timing(timing: Mapping[str, Any]) -> tuple[float, float]:
    """Return the ``(fs, t0)`` a regular timing block declares."""
    rate = timing["rate"]
    return (
        rate["numerator"] / rate["denominator"],
        _seconds(timing.get("segment_start_time_ns", 0)),
    )


def _collapsed_unit(symbols: Sequence[str], default: str) -> Any:
    """Return one unit symbol when the columns agree, else *symbols* unchanged.

    The per-column sequence is returned as given rather than normalized, so a
    caller's tuple stays a tuple and its list stays a list.
    """
    if len(set(symbols)) > 1:
        return symbols
    return symbols[0] if symbols else default


def clock_from_descriptor(descriptor: Mapping[str, Any]) -> Clock:
    """Return a :class:`~neurale.data.Clock` for one manifest clock entry.

    NRF fixes a clock-type vocabulary and requires a synchronization domain to
    be an identifier, so a writer that had to reshape either one records the
    original in the descriptor metadata. Those spellings are preferred here;
    a session written by anything else simply has no such metadata and the
    format's own values are used.
    """
    rate = descriptor.get("rate")
    metadata = descriptor.get("metadata") or {}
    return Clock(
        name=descriptor.get("name", descriptor["id"]),
        type=metadata.get(CLOCK_TYPE_KEY, descriptor["type"]),
        rate=(rate["numerator"] / rate["denominator"]) if rate else None,
        epoch=descriptor.get("epoch"),
        offset=_seconds(descriptor.get("offset_ns", 0)),
        drift=float(descriptor.get("drift_ppb", 0)) / 1e9,
        synchronization_domain=metadata.get(
            CLOCK_DOMAIN_KEY, descriptor.get("synchronization_domain")
        ),
    )


def optional_clock(descriptor: Mapping[str, Any] | None) -> Clock | None:
    """Return the clock a typed object declared, or ``None`` if it declared none.

    NRF requires every stream and record set to name a clock, while a
    :class:`~neurale.data.SignalArray` may legitimately carry none. A writer
    that had to supply a placeholder marks it, so restoring ``None`` here is
    reading a recorded fact rather than second-guessing the registry.
    """
    if descriptor is None:
        return None
    if (descriptor.get("metadata") or {}).get(SYNTHETIC_CLOCK_KEY):
        return None
    return clock_from_descriptor(descriptor)


def stream_timing_hint(manifest: Mapping[str, Any], stream_id: str) -> Mapping[str, Any]:
    """Return the declared rate and ``t0`` recorded for an explicit stream.

    NRF's explicit timing block stores timestamps and nothing else. A typed
    writer records the declarations it could not otherwise keep beside the
    payload; an empty mapping means this session did not, and the values are
    derived from the timestamps instead.
    """
    typed = (manifest.get("metadata") or {}).get(RECORDING_METADATA_KEY)
    if not isinstance(typed, Mapping):
        return {}
    streams = typed.get("streams")
    if not isinstance(streams, Mapping):
        return {}
    hint = streams.get(stream_id)
    return hint if isinstance(hint, Mapping) else {}


def channel_table(
    manifest: Mapping[str, Any],
    channel_ids: Sequence[str],
    *,
    width: int | None = None,
    unit_ids: Sequence[str] = (),
    channel_type: str = "analog",
) -> ChannelTable:
    """Return the channel table for one stream, in the stream's own order.

    NRF permits a stream to declare no channels -- a 2D cursor is
    not an electrode -- while :class:`~neurale.data.ChannelTable` requires one
    entry per column. In that case positional placeholders are produced, and
    the caller marks them as derived so they are never mistaken for recorded
    channel metadata.
    """
    units = _units_by_id(manifest)
    if not channel_ids and width is not None:
        return ChannelTable(
            [
                ChannelInfo(
                    name=f"column-{idx:04d}",
                    index=idx,
                    type=channel_type,
                    unit=units.get(unit_ids[idx] if idx < len(unit_ids) else "", {}).get(
                        "symbol", ""
                    ),
                )
                for idx in range(width)
            ]
        )

    by_id = {channel["id"]: channel for channel in manifest["channels"]}
    infos: list[ChannelInfo] = []
    for channel_id in channel_ids:
        descriptor = by_id.get(channel_id)
        if descriptor is None:
            raise NrfSemanticError(f"stream references unknown channel {channel_id!r}")
        unit = units.get(descriptor["unit_id"], {})
        infos.append(
            ChannelInfo(
                name=descriptor.get("name", channel_id),
                index=descriptor["index"],
                type=descriptor.get("type", "analog"),
                unit=unit.get("symbol", ""),
                valid=descriptor.get("valid", True),
                bad=descriptor.get("bad", False),
                electrode=descriptor.get("electrode_id"),
                contact=descriptor.get("contact"),
            )
        )
    return ChannelTable(infos)


def signal_array(
    manifest: Mapping[str, Any],
    stream: Mapping[str, Any],
    values: np.ndarray,
    timestamps: np.ndarray | None,
) -> SignalArray:
    """Return a :class:`~neurale.data.SignalArray` for a sampled stream."""
    if stream["kind"] not in {"neural", "behavioral"}:
        raise NrfSemanticError(
            f"{stream['id']} is a {stream['kind']} stream and has no SignalArray mapping"
        )
    timing = stream["timing"]
    clocks = {clock["id"]: clock for clock in manifest["clocks"]}
    units = _units_by_id(manifest)
    symbols = tuple(units.get(unit_id, {}).get("symbol", "") for unit_id in stream["unit_ids"])

    if timing["mode"] == "regular":
        fs, t0 = _regular_timing(timing)
        time = None
    else:
        hint = stream_timing_hint(manifest, stream["id"])
        # A SignalArray requires a positive rate, which explicit timing does not
        # record. The rate the writer declared is preferred; an average of the
        # timestamps is a fallback for sessions that declared none, and is not
        # the same number.
        fs = hint.get("fs") or _inferred_rate(timestamps)
        if fs is None:
            raise NrfSemanticError(
                f"{stream['id']} uses explicit timing, declares no sampling rate, and has "
                "fewer than two committed timestamps, so no SignalArray rate can be honest"
            )
        time = _seconds_array(timestamps)
        t0 = hint.get("t0")

    return SignalArray(
        data=values,
        fs=fs,
        time=time,
        t0=t0,
        clock=optional_clock(clocks.get(stream["clock_id"])),
        channels=channel_table(
            manifest,
            stream["channel_ids"],
            width=int(values.shape[1]) if values.ndim > 1 else len(stream["unit_ids"]),
            unit_ids=stream["unit_ids"],
            channel_type="behavior" if stream["kind"] == "behavioral" else "analog",
        ),
        unit=_collapsed_unit(symbols, ""),
        name=stream.get("name", stream["id"]),
        attrs={
            "nrf_stream_id": stream["id"],
            "nrf_segment_id": stream["segment_policy"].get("initial_segment_id"),
            "nrf_source_stream_ids": list(stream.get("source_stream_ids", ())),
            # Placeholder channels are derived from the payload width, not
            # recorded metadata; the flag keeps that distinction visible.
            "nrf_channels_declared": bool(stream["channel_ids"]),
        },
    )


def _inferred_rate(timestamps: np.ndarray | None) -> float | None:
    """Return an average rate for explicitly timed data, or ``None``.

    An explicitly timed stream has no declared rate; a single average is a
    convenience for consumers that need one and never replaces the timestamps.
    """
    if timestamps is None or len(timestamps) < 2:
        return None
    span = float(timestamps[-1] - timestamps[0]) / NANOSECONDS
    if span <= 0:
        return None
    return (len(timestamps) - 1) / span


def feature_matrix(
    manifest: Mapping[str, Any],
    stream: Mapping[str, Any],
    values: np.ndarray,
    timestamps: np.ndarray | None,
) -> FeatureMatrix:
    """Return a :class:`~neurale.data.FeatureMatrix` for a feature stream."""
    if stream["kind"] != "feature":
        raise NrfSemanticError(f"{stream['id']} is not a feature stream")
    descriptors = {entry["id"]: entry for entry in manifest["feature_sets"]}
    descriptor = descriptors.get(stream.get("feature_set_id", ""))
    if descriptor is None:
        raise NrfSemanticError(f"{stream['id']} has no feature-set descriptor")
    units = _units_by_id(manifest)
    symbols = [units.get(unit_id, {}).get("symbol", "") for unit_id in descriptor["unit_ids"]]

    timing = stream["timing"]
    if timing["mode"] == "regular":
        fs, t0 = _regular_timing(timing)
        time = None
    else:
        # Unlike a signal, a FeatureMatrix accepts ``fs=None`` and
        # that is what explicit timing means: irregular observations. Averaging
        # the timestamps into a rate here would also fail the container's own
        # rate/time consistency check for genuinely irregular data.
        hint = stream_timing_hint(manifest, stream["id"])
        fs = hint.get("fs")
        time = _seconds_array(timestamps)
        t0 = hint.get("t0", 0.0)

    return FeatureMatrix(
        data=values,
        fs=fs,
        time=time,
        t0=t0,
        feature_names=list(descriptor["feature_names"]),
        source_signal=descriptor["source_stream_id"],
        window_size=_seconds(descriptor["window_length_ns"]),
        shift=_seconds(descriptor["shift_ns"]),
        unit=_collapsed_unit(symbols, "a.u."),
        attrs={
            "nrf_stream_id": stream["id"],
            "nrf_feature_set_id": descriptor["id"],
            "nrf_timestamp_reference": descriptor["timestamp_reference"],
            "nrf_algorithm": dict(descriptor["algorithm"]),
        },
    )


def event_series(columns: Mapping[str, Sequence[Any]], clock: Clock | None = None) -> EventSeries:
    """Return an :class:`~neurale.data.EventSeries` from ``events`` columns."""
    onsets = columns.get("time_ns") or columns.get("onset_ns")
    if onsets is None:
        raise NrfSemanticError("events record set has no onset column")
    durations = columns.get("duration_ns") or [0] * len(onsets)
    events = [
        Event(
            onset=_seconds(onset),
            duration=_seconds(duration),
            label=_optional(columns.get("label"), idx),
            code=_optional(columns.get("code"), idx),
            source=_optional(columns.get("source"), idx),
            value=_optional(columns.get("value"), idx),
            sample_index=_optional(columns.get("sample_index"), idx),
            confidence=_optional(columns.get("confidence"), idx),
            attrs={"nrf_event_id": columns["event_id"][idx]},
        )
        for idx, (onset, duration) in enumerate(zip(onsets, durations, strict=True))
    ]
    return EventSeries(events, clock=clock)


def trial_table(columns: Mapping[str, Sequence[Any]]) -> TrialTable:
    """Return a :class:`~neurale.data.TrialTable` from ``trials`` columns."""
    starts = columns.get("start_ns")
    stops = columns.get("stop_ns")
    if starts is None or stops is None:
        raise NrfSemanticError("trials record set has no start/stop columns")
    trials = [
        Trial(
            trial_id=idx,
            start=_seconds(start),
            stop=_seconds(stop),
            label=_optional(columns.get("label"), idx),
            target_id=_optional(columns.get("target_id"), idx),
            outcome=_optional(columns.get("outcome"), idx),
            block=_optional(columns.get("block"), idx),
            attrs={"nrf_trial_id": columns["trial_id"][idx]},
        )
        for idx, (start, stop) in enumerate(zip(starts, stops, strict=True))
    ]
    return TrialTable(trials)


def _optional(column: Sequence[Any] | None, idx: int) -> Any:
    if column is None:
        return None
    return column[idx]


def build_recording(
    manifest: Mapping[str, Any],
    *,
    signals: Mapping[str, SignalArray],
    features: Mapping[str, FeatureMatrix],
    events: EventSeries | None,
    trials: TrialTable | None,
    unmapped_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Recording:
    """Assemble a :class:`~neurale.data.Recording` from converted parts.

    ``Recording.metadata``, ``Recording.subject``, and ``Recording.session``
    have no NRF registry of their own, so a typed writer records them verbatim
    under :data:`RECORDING_METADATA_KEY` and they are restored from there. A
    session written by anything else falls back to the manifest's own session
    descriptor, which is the only place such a session can carry them.

    ``unmapped_records`` is placed in the recording metadata rather than
    discarded: a record kind without a typed destination must remain visible,
    not vanish.
    """
    descriptor = dict(manifest["session"])
    manifest_metadata = dict(manifest.get("metadata") or {})
    typed = manifest_metadata.pop(RECORDING_METADATA_KEY, None)

    if isinstance(typed, Mapping):
        session = _mapping_or_none(typed.get("session"))
        subject = _mapping_or_none(typed.get("subject"))
        metadata: dict[str, Any] = dict(typed.get("metadata") or {})
    else:
        session = descriptor
        subject = _mapping_or_none(descriptor.get("subject"))
        metadata = {}

    metadata.update(
        {
            "nrf_format_version": dict(manifest["version"]),
            "nrf_writer": dict(manifest["writer"]),
            "nrf_unmapped_records": {
                name: [dict(row) for row in rows] for name, rows in unmapped_records.items()
            },
            "nrf_session": descriptor,
        }
    )
    if manifest_metadata:
        metadata["nrf_metadata"] = manifest_metadata
    return Recording(
        signals=dict(signals),
        features=dict(features),
        events=events,
        trials=trials,
        metadata=metadata,
        subject=subject,
        session=session,
    )


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None
