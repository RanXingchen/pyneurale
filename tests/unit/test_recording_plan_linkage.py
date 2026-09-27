#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Feature source linkage in the compiled recording plan.

The native ``FeatureSetDescriptor.source_stream_id`` is a ``SignalId``; the plan
maps it to the NRF ``stream_id`` of the declaration that records that signal.
The descriptor's ``source_stream`` text is provenance only, so an NRF stream id
that spells a native source differently must not change the linkage, and the
same declarations in a different order must resolve to the same linkage and the
same fingerprint.

These tests use a lightweight stand-in for the prepared native schema so they
run without the native extension; ``compile_recording_plan`` only reads the
attributes the stand-in exposes.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from neurale.recording import (
    RecorderConfig,
    RecorderConfigError,
    StreamMetadata,
    StreamProvenanceConfig,
    StreamRecording,
    StreamStorageConfig,
    compile_recording_plan,
)


def _signal(
    signal_id: int,
    *,
    kind: str = "SAMPLED",
    feature_set_id: int = 0,
    channels: int = 2,
    observation: str = "NOT_APPLICABLE",
    rate: tuple[int, int] = (1000, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        id=signal_id,
        clock_domain=7,
        n_channels=channels,
        nominal_block_samples=8,
        max_block_samples=16,
        fs=SimpleNamespace(numerator=rate[0], denominator=rate[1]),
        dtype="INT16",
        layout="SAMPLE_MAJOR",
        device_tick_tracking="SAMPLE_COUNTER",
        kind=kind,
        # A feature signal's unit comes from the unit registry, so the native
        # StreamSchema constructor refuses one that also carries a physical
        # unit. The stand-in follows that rule: a schema the runtime would have
        # rejected is not an input the compiler can be asked about.
        physical_unit="UNSPECIFIED" if kind == "FEATURE" else "VOLTS",
        channel_set_id=0,
        calibration_id=0,
        reference_id=0,
        feature_set_id=feature_set_id,
        observation_timing=observation,
        fixed_block_bytes=0,
        max_block_bytes=channels * 16 * 2,
    )


def _feature_set(feature_set_id: int, source_signal_id: int, source_stream: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=feature_set_id,
        feature_names=["a", "b"],
        unit_ids=[1, 1],
        source_stream_id=source_signal_id,
        source_stream=source_stream,
        algorithm_name="mt",
        algorithm_version="1",
        window_length_ns=20_000_000,
        shift_ns=20_000_000,
        timestamp_reference="WINDOW_CENTER",
    )


def _schema(signals: list[Any], feature_sets: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=11,
        signals=signals,
        feature_sets=feature_sets,
        units=[SimpleNamespace(id=1, symbol="V^2", description="volt squared")],
    )


def _base_signals() -> list[SimpleNamespace]:
    """Two neural signals plus a feature signal described by feature set 9."""
    return [
        _signal(1),
        _signal(2),
        _signal(3, kind="FEATURE", feature_set_id=9, observation="REGULAR", rate=(50, 1)),
    ]


def _linked_schema() -> SimpleNamespace:
    """The base signals, with feature set 9 sourced from signal 2."""
    return _schema(
        _base_signals(),
        [_feature_set(9, source_signal_id=2, source_stream="native-source-two")],
    )


def _config(streams: list[StreamRecording]) -> RecorderConfig:
    return RecorderConfig(path="/tmp/unused.nrf", streams=streams)


def _base_streams() -> list[StreamRecording]:
    first, second, feature = _base_signals()
    return [
        StreamRecording(
            stream_id="stream-a",
            signal=first,
            metadata=StreamMetadata(channel_names=("a0", "a1")),
            storage=StreamStorageConfig(capacity=4096),
        ),
        StreamRecording(
            stream_id="stream-b",
            signal=second,
            metadata=StreamMetadata(channel_names=("b0", "b1")),
            storage=StreamStorageConfig(capacity=4096),
        ),
        StreamRecording(
            stream_id="bandpower",
            signal=feature,
            storage=StreamStorageConfig(capacity=1024),
        ),
    ]


def test_descriptor_source_maps_to_nrf_stream_id() -> None:
    """The native source_stream text can differ from the NRF stream id; the
    linkage follows the numeric source_stream_id, not the text."""
    schema = _linked_schema()
    plan = compile_recording_plan(_config(_base_streams()), schema)

    assert plan.stream_for(3).source_stream_ids == ("stream-b",)


def test_linkage_and_fingerprint_ignore_declaration_order() -> None:
    schema = _linked_schema()
    streams = _base_streams()
    plans = [
        compile_recording_plan(_config([streams[i] for i in order]), schema)
        for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0))
    ]

    assert len({plan.fingerprint for plan in plans}) == 1
    assert {plan.stream_for(3).source_stream_ids for plan in plans} == {("stream-b",)}


def test_unrecorded_explicit_source_is_rejected() -> None:
    schema = _linked_schema()
    streams = _base_streams()
    ghost = StreamRecording(stream_id="ghost", signal=_signal(99))
    streams[2] = replace(
        streams[2],
        provenance=StreamProvenanceConfig(sources=(ghost,)),
    )
    with pytest.raises(RecorderConfigError, match="ghost"):
        compile_recording_plan(_config(streams), schema)


def test_unrecorded_descriptor_source_is_rejected() -> None:
    schema = _schema(
        _base_signals(), [_feature_set(9, source_signal_id=99, source_stream="missing")]
    )
    with pytest.raises(RecorderConfigError, match="99"):
        compile_recording_plan(_config(_base_streams()), schema)


def test_non_feature_stream_with_missing_source_is_rejected() -> None:
    """Source linkage is checked on every stream, not only on feature streams.

    A neural stream may name the stream it was recorded alongside. A name that
    resolves to nothing cannot be written into a legal manifest, so the plan is
    refused while it is still a plan -- the manifest freezes before the first
    append, and a session that cannot describe itself has no later chance.
    """
    schema = _linked_schema()
    streams = _base_streams()
    ghost = StreamRecording(stream_id="ghost", signal=_signal(99))
    streams[0] = replace(
        streams[0],
        provenance=StreamProvenanceConfig(sources=(ghost,)),
    )
    with pytest.raises(RecorderConfigError, match="'ghost', which this plan does not record"):
        compile_recording_plan(_config(streams), schema)


def test_non_feature_stream_may_declare_recorded_source() -> None:
    """The rule refuses dangling references, not source linkage itself."""
    schema = _linked_schema()
    streams = _base_streams()
    streams[0] = replace(
        streams[0],
        provenance=StreamProvenanceConfig(sources=(streams[1],)),
    )
    plan = compile_recording_plan(_config(streams), schema)

    assert plan.stream_for(1).source_stream_ids == ("stream-b",)


def test_declared_and_descriptor_sources_are_both_linked() -> None:
    schema = _linked_schema()
    streams = _base_streams()
    streams[2] = replace(
        streams[2],
        provenance=StreamProvenanceConfig(sources=(streams[0],)),
    )
    plan = compile_recording_plan(_config(streams), schema)

    assert plan.stream_for(3).source_stream_ids == ("stream-a", "stream-b")


def test_feature_signal_without_descriptor_is_configuration_error() -> None:
    """A missing descriptor is refused as configuration, at one place.

    The native ``StreamSchema`` constructor already refuses a feature signal
    whose ``feature_set_id`` resolves to nothing, so this schema cannot arrive
    from a live prepared object -- it can only arrive from a caller that built
    one by hand. Such a caller is owed the same error class as every other
    undescribable plan, and the compiler no longer carries per-field fallbacks
    for a descriptor that is absent: a feature stream is described from its
    descriptor or not at all, so resolution fails once, here, instead of
    quietly substituting invented metadata further down.
    """
    schema = _schema(_base_signals(), [])

    with pytest.raises(RecorderConfigError, match="missing feature-set descriptor 9"):
        compile_recording_plan(_config(_base_streams()), schema)


def test_descriptor_with_zero_window_is_configuration_error() -> None:
    """The window and shift are read from the descriptor, never invented.

    Native validation refuses a zero window or shift, so this too can only
    arrive from a hand-built schema; the compiler refuses it rather than
    writing a window this session never used into a descriptor a later
    analysis is entitled to trust.
    """
    descriptor = _feature_set(9, source_signal_id=2, source_stream="native-source-two")
    descriptor.window_length_ns = 0
    schema = _schema(_base_signals(), [descriptor])

    with pytest.raises(RecorderConfigError, match="window length and shift"):
        compile_recording_plan(_config(_base_streams()), schema)
