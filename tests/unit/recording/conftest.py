#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared fixtures for the streaming-to-NRF recorder tests.

The frames here are built the way the observer bridge builds them -- a single
flat payload with one signal block pointing into it -- so the tests exercise the
same decoding path a live session does, including the channel-major layout that
a transposing source produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import neurale.streaming as streaming
from neurale.io.nrf import NrfWriter
from neurale.recording import (
    RecorderConfig,
    RecorderLimits,
    StreamMetadata,
    StreamProvenanceConfig,
    StreamRecording,
    StreamRole,
    StreamStorageConfig,
    StreamTimingConfig,
    StreamTimingMode,
    compile_recording_plan,
)
from neurale.recording._registry import build_session

#: Native signal identifiers used throughout.
NEURAL = 1
CURSOR = 2
BANDPOWER = 3

CLOCK_DOMAIN = 7
SCHEMA_ID = 11
FEATURE_SET_ID = 9

NEURAL_RATE = 1_000
CURSOR_RATE = 100
BANDPOWER_RATE = 50

NEURAL_CHANNELS = 4
CURSOR_CHANNELS = 2
BANDPOWER_FEATURES = 3

NANOSECONDS = 1_000_000_000

#: Native dtype of each signal, and the NumPy dtype it decodes to.
DTYPES: dict[int, str] = {NEURAL: "int16", CURSOR: "float32", BANDPOWER: "float64"}
CHANNELS: dict[int, int] = {
    NEURAL: NEURAL_CHANNELS,
    CURSOR: CURSOR_CHANNELS,
    BANDPOWER: BANDPOWER_FEATURES,
}
RATES: dict[int, int] = {NEURAL: NEURAL_RATE, CURSOR: CURSOR_RATE, BANDPOWER: BANDPOWER_RATE}
#: ``cursor`` is deliberately channel-major: a recorder that ignored the layout
#: would still produce plausible-looking numbers, just transposed.
CHANNEL_MAJOR: frozenset[int] = frozenset({CURSOR})


def native_schema() -> Any:
    """Return the prepared native schema the recorder is configured against."""
    neural = streaming.SignalSchema(
        NEURAL,
        streaming.SignalDType.INT16,
        NEURAL_CHANNELS,
        8,
        16,
        streaming.RationalRate(NEURAL_RATE, 1),
        CLOCK_DOMAIN,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
    )
    cursor = streaming.SignalSchema(
        CURSOR,
        streaming.SignalDType.FLOAT32,
        CURSOR_CHANNELS,
        2,
        4,
        streaming.RationalRate(CURSOR_RATE, 1),
        CLOCK_DOMAIN,
        layout=streaming.SignalLayout.CHANNEL_MAJOR,
    )
    bandpower = streaming.SignalSchema(
        BANDPOWER,
        streaming.SignalDType.FLOAT64,
        BANDPOWER_FEATURES,
        1,
        2,
        streaming.RationalRate(BANDPOWER_RATE, 1),
        CLOCK_DOMAIN,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=FEATURE_SET_ID,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    descriptor = streaming.FeatureSetDescriptor(
        FEATURE_SET_ID,
        ["lfp.beta", "lfp.gamma", "lfp.hfa"],
        [1, 1, 1],
        NEURAL,
        "neural",
        20_000_000,
        20_000_000,
        "multitaper-bandpower",
        "1",
    )
    return streaming.StreamSchema(
        SCHEMA_ID,
        [neural, cursor, bandpower],
        [descriptor],
        [streaming.UnitDescriptor(1, "V^2", "volt squared")],
    )


def stream_specs(**overrides: Any) -> list[StreamRecording]:
    """Return the default recording plan for :func:`native_schema`."""
    block_index = bool(overrides.pop("block_index", True))
    timing_mode = overrides.pop("timing", "explicit")
    if overrides:
        raise AssertionError(f"unsupported stream fixture overrides: {sorted(overrides)}")
    timing = StreamTimingConfig(mode=StreamTimingMode(timing_mode))
    schema = native_schema()
    neural_signal, cursor_signal, feature_signal = schema.signals
    neural = StreamRecording(
        stream_id="neural",
        signal=neural_signal,
        metadata=StreamMetadata(
            unit="V",
            channel_names=tuple(f"electrode-{idx}" for idx in range(NEURAL_CHANNELS)),
        ),
        storage=StreamStorageConfig(capacity=4096, chunk_length=8, block_index_chunk_length=4),
        timing=timing,
        provenance=StreamProvenanceConfig(block_index=block_index),
    )
    cursor = StreamRecording(
        stream_id="cursor",
        signal=cursor_signal,
        metadata=StreamMetadata(
            role=StreamRole.BEHAVIORAL,
            unit=("m", "m"),
            channel_names=("cursor-x", "cursor-y"),
        ),
        storage=StreamStorageConfig(capacity=1024, chunk_length=8, block_index_chunk_length=4),
        timing=timing,
        provenance=StreamProvenanceConfig(block_index=block_index),
    )
    feature = StreamRecording(
        stream_id="bandpower",
        signal=feature_signal,
        storage=StreamStorageConfig(capacity=1024, chunk_length=8, block_index_chunk_length=4),
        timing=timing,
        provenance=StreamProvenanceConfig(block_index=block_index, sources=(neural,)),
    )
    return [neural, cursor, feature]


def recorder_config(path: Path, **overrides: Any) -> RecorderConfig:
    """Return a :class:`RecorderConfig` with small, test-sized bounds."""
    limit_values: dict[str, Any] = {
        "frame_queue_capacity": 64,
        "control_queue_capacity": 64,
        "max_control_records": 4 * 1024,
        "checkpoint_interval": 0,
    }
    for name in tuple(limit_values):
        if name in overrides:
            limit_values[name] = overrides.pop(name)
    if "spool_capacity_bytes" in overrides:
        limit_values["spool_capacity_bytes"] = overrides.pop("spool_capacity_bytes")
    if "control_chunk_length" in overrides:
        limit_values["max_control_records"] = overrides.pop("control_chunk_length") * 1024
    arguments: dict[str, Any] = {
        "path": path,
        "streams": stream_specs(),
        "limits": RecorderLimits(**limit_values),
    }
    arguments.update(overrides)
    return RecorderConfig(**arguments)


def recording_runner(schema: Any, config: Any, source: Any) -> Any:
    """Return the runner a recorder attaches to.

    Every recorder test drives the same fixed topology -- identity processing
    into a counting sink, under the safety controller recording installs -- so
    what a test varies is its schema, its runtime config, and its source.
    """
    return streaming.StreamRunner(
        schema,
        config,
        source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )


@dataclass(frozen=True, slots=True)
class Block:
    """One signal block, stated in the sample-major form a caller thinks in."""

    signal_id: int
    values: np.ndarray
    sample_idx_start: int
    observation_ns: int
    device_tick: int = 0

    def encoded(self) -> bytes:
        stored = self.values.T if self.signal_id in CHANNEL_MAJOR else self.values
        return np.ascontiguousarray(stored, dtype=DTYPES[self.signal_id]).tobytes()


def make_frame(sequence: int, blocks: list[Block], *, host_ns: int | None = None) -> Any:
    """Assemble the frame snapshot the observer bridge would have delivered."""
    payload = b""
    descriptors = []
    for block in blocks:
        raw = block.encoded()
        descriptors.append(
            streaming.PythonObserverSignalBlock(
                block.sample_idx_start,
                block.device_tick,
                len(payload),
                len(raw),
                block.signal_id,
                int(block.values.shape[0]),
                streaming.ClockSyncSnapshot(),
                block.observation_ns,
                block.sample_idx_start + int(block.values.shape[0]) - 1,
            )
        )
        payload += raw
    return streaming.PythonObserverFrame(
        session_id=1,
        sequence=sequence,
        host_received_ns=blocks[0].observation_ns if host_ns is None else host_ns,
        schema_id=SCHEMA_ID,
        source_clock_domain=CLOCK_DOMAIN,
        blocks=descriptors,
        payload=np.frombuffer(payload, dtype=np.uint8),
    )


def observation_ns(signal_id: int, sample_idx: int) -> int:
    """Return the observation time of one sample on its declared grid."""
    return sample_idx * NANOSECONDS // RATES[signal_id]


def sample_values(signal_id: int, start: int, count: int) -> np.ndarray:
    """Return a distinct, exactly representable payload for every sample."""
    width = CHANNELS[signal_id]
    base = np.arange(start, start + count, dtype="int64").reshape(count, 1)
    columns = np.arange(width, dtype="int64").reshape(1, width)
    return (base * 16 + columns + signal_id * 4096).astype(DTYPES[signal_id])


def signal_block(signal_id: int, start: int, count: int, *, device_tick: int = 0) -> Block:
    """Return *count* consecutive samples of one signal, on its declared grid."""
    return Block(
        signal_id=signal_id,
        values=sample_values(signal_id, start, count),
        sample_idx_start=start,
        observation_ns=observation_ns(signal_id, start),
        device_tick=device_tick,
    )


def neural_frame(sequence: int, start: int, count: int, *, device_tick: int = 0) -> Any:
    """Return the frame one contiguous block of neural samples arrives in."""
    return make_frame(sequence, [signal_block(NEURAL, start, count, device_tick=device_tick)])


@dataclass(slots=True)
class SignalGapStandIn:
    """A per-signal gap, shaped exactly like the native snapshot.

    The native ``PythonObserverSignalGap`` is read-only from Python, so a test
    that wants a specific gap has to state one. The runtime integration tests
    use real ones.
    """

    signal_id: int
    reason: Any = streaming.GapReason.SAMPLE_GAP
    expected_sample_idx: int = 0
    actual_sample_idx: int = 0
    missing_samples: int | None = None
    expected_device_tick: int | None = None
    actual_device_tick: int | None = None


@dataclass(slots=True)
class DiscontinuityStandIn:
    """A discontinuity snapshot, shaped exactly like the native one."""

    previous_frame_sequence: int
    actual_frame_sequence: int
    reason: Any = streaming.GapReason.SAMPLE_GAP
    signal_gaps: list[SignalGapStandIn] = field(default_factory=list)
    session_id: int = 1


@pytest.fixture
def schema() -> Any:
    return native_schema()


@pytest.fixture
def session_path(tmp_path: Path) -> Path:
    return tmp_path / "session.nrf"


class LegacySessionFixture:
    """Create one pre-native-replay NRF session without a live recorder."""

    def __init__(self, path: Path, schema: Any) -> None:
        self.path = path
        self.schema = schema
        self.closed = False

    def stop(self) -> None:
        if self.closed:
            return
        config = recorder_config(self.path)
        plan = compile_recording_plan(config, self.schema)
        writer = NrfWriter.create(
            self.path,
            session_id=plan.session.session_id,
            created_at=plan.session.created_at,
            writer_name=plan.session.writer_name,
            writer_version=plan.session.writer_version,
        )
        build_session(
            writer.registry,
            self.schema,
            config.streams,
            control_chunk_length=plan.resource_bounds.control_chunk_length,
        )
        writer.freeze()
        writer.finalize(kind="normal", reason="legacy fixture")
        writer.close()
        self.closed = True


@pytest.fixture
def recorder(session_path: Path, schema: Any) -> LegacySessionFixture:
    return LegacySessionFixture(session_path, schema)
