#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The spool-to-NRF finalizer.

Everything here starts from a spool a **real native recorder wrote**. A
hand-built spool would be testing the finalizer against this file's idea of the
format, and the one thing a converter must be right about is the bytes the other
engine actually produces. Where a case cannot be recorded -- a corrupted
counter, an unknown record kind -- a genuine spool is patched and re-checksummed
rather than synthesized, so every byte outside the injected fault is still the
recorder's.

Two spool shapes appear, and they are not interchangeable:

* **runtime-driven**, for the data plane. A runtime with a critical recorder
  attached is the only thing that produces frames, blocks, and discontinuities.
* **standalone**, for the control plane. The window between ``start()`` and a
  synthetic source running out is scheduling, so a runtime-driven control test
  would be testing the scheduler; the recorder's own standalone entry points
  reach ``recording`` with no runtime and no race.
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import neurale.streaming as streaming
from neurale.io.nrf import (
    NATIVE_REPLAY_NAMESPACE,
    RECOVERY_NAMESPACE,
    NrfReader,
    SessionIdentity,
    diagnose_session,
    ledger_target_paths,
    plan_from_spool_document,
)
from neurale.recording import (
    FINALIZATION_NAMESPACE,
    FinalizationError,
    FinalizationProgress,
    FinalizerOptions,
    RecorderConfig,
    RecorderLimits,
    SpoolSourceError,
    StreamMetadata,
    StreamProvenanceConfig,
    StreamRecording,
    StreamStorageConfig,
    StreamTimingConfig,
    StreamTimingMode,
    finalize_spool,
)
from neurale.recording import _spool_format as spool
from neurale.recording._finalizer import (
    FAULT_CODE_CAPTURE_FAULTED,
    PROGRESS_FILE,
    STAGING_SESSION,
    STAGING_SUFFIX,
    _bounds_for,
    _publish_session,
    _reject_unmaterializable_streams,
)
from neurale.recording._native_recorder import NativeRecorderOptions, NativeSessionRecorder
from neurale.streaming import _native as streaming_native

from .conftest import native_schema as multi_schema
from .conftest import recorder_config as multi_recorder_config
from .conftest import recording_runner


def _package_text(path: Path, key: str) -> str:
    from neurale.io.nrf._package import Package

    package = Package(path)
    try:
        return (package.root / key).read_text()
    finally:
        package.close()


SIGNAL_ID = 1
SCHEMA_ID = 11
CLOCK_DOMAIN = 7
CHANNELS = 4
BLOCK_SAMPLES = 8

#: The three-stream fixture's maximum frame payload, in bytes. The runtime's
#: buffer is sized for the worst case a schema allows, not for what the source
#: nominally emits.
MULTI_MAX_FRAME_PAYLOAD_BYTES = 4 * 16 * 2 + 2 * 4 * 4 + 3 * 2 * 8


# --- producing real spools ---------------------------------------------------


def _schema() -> Any:
    signal = streaming.SignalSchema(
        SIGNAL_ID,
        streaming.SignalDType.INT16,
        CHANNELS,
        BLOCK_SAMPLES,
        BLOCK_SAMPLES,
        streaming.RationalRate(1_000, 1),
        CLOCK_DOMAIN,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
    )
    return streaming.StreamSchema(SCHEMA_ID, [signal])


def _realtime_config(frames: int, *, blocks: int = 1, payload_bytes: int | None = None) -> Any:
    capacity = max(8, frames * 2)
    budget = streaming_native.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = capacity
    budget.processor_owned = 2
    budget.critical_edge_capacity = capacity + 1
    budget.actuator_owned = 1
    budget.observer_edge_capacity = capacity

    config = streaming_native.RealtimeConfig()
    config.pool_capacity = budget
    config.buffer_size = BLOCK_SAMPLES * CHANNELS * 2 if payload_bytes is None else payload_bytes
    config.max_signal_blocks = blocks
    config.discontinuity_capacity = 8
    config.gaps_per_discontinuity = 4
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 4
    config.platform.mode = streaming_native.RealtimeConfigMode.STRICT
    config.validate()
    return config


def _recorder_config(path: Path, **overrides: Any) -> RecorderConfig:
    block_idx = overrides.pop("block_index", True)
    limit_values: dict[str, Any] = {"checkpoint_interval": 0}
    for name in (
        "frame_queue_capacity",
        "control_queue_capacity",
        "spool_capacity_bytes",
        "checkpoint_interval",
        "max_control_records",
    ):
        if name in overrides:
            limit_values[name] = overrides.pop(name)
    if "control_chunk_length" in overrides:
        limit_values["max_control_records"] = overrides.pop("control_chunk_length") * 1024
    arguments: dict[str, Any] = {
        "path": path,
        "streams": [
            StreamRecording(
                metadata=StreamMetadata(unit="V"),
                storage=StreamStorageConfig(
                    capacity=8192, chunk_length=8, block_index_chunk_length=8
                ),
                timing=StreamTimingConfig(mode=StreamTimingMode.REGULAR),
                provenance=StreamProvenanceConfig(block_index=block_idx),
            )
        ],
        "limits": RecorderLimits(**limit_values),
    }
    arguments.update(overrides)
    return RecorderConfig(**arguments)


def _record_session(
    tmp_path: Path,
    frames: int,
    *,
    multi_stream: bool = False,
    sequence_gap_at: int | None = None,
    abort: bool = False,
    block_idx: bool = True,
) -> bytes:
    """Run one real native session and return the spool it wrote."""
    if multi_stream:
        schema = multi_schema()
        config = _realtime_config(frames, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES)
        recorder_config = multi_recorder_config(tmp_path / "unused.nrf", checkpoint_interval=0)
    else:
        schema = _schema()
        config = _realtime_config(frames)
        recorder_config = _recorder_config(tmp_path / "unused.nrf", block_index=block_idx)

    source = streaming.SyntheticNativeSource(schema, frames, 1, None, sequence_gap_at)
    runner = recording_runner(schema, config, source)
    recorder = NativeSessionRecorder.create(
        recorder_config,
        schema,
        options=NativeRecorderOptions(spool="memory", edge_capacity=max(8, frames * 2)),
    )
    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    if abort:
        recorder.abort("operator-abort")
    else:
        runner.join()
        recorder.stop("done")
    snapshot = recorder._recorder.spool_snapshot()
    recorder.close()
    return snapshot


def _control_session(tmp_path: Path, *, abort: bool = False, **overrides: Any) -> bytes:
    """Drive a recorder with no runtime and return the spool of its control plane."""
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "unused.nrf", **overrides),
        _schema(),
        options=NativeRecorderOptions(spool="memory"),
    )
    recorder.prepare()
    native = recorder._native
    assert recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
    assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK

    assert recorder.record_event("trial-start", value=1.5, text="go") is True
    assert recorder.record_state("phase", text="hold") is True
    assert recorder.record_command("cursor", value=-2.0) is True
    assert recorder.record_task_variable("gain", value=0.25) is True
    assert recorder.record_label("target", text="left") is True
    assert recorder.record_target("goal", value=3.0) is True
    assert recorder.record_assistance("share", value=0.5) is True
    assert recorder.record_trial(start_ns=10, stop_ns=20, label="reach", outcome="hit") is True
    assert recorder.record_fault("edge", stage="observer", frame_sequence=7, signal_id=1) is True

    if abort:
        recorder.abort("operator-abort")
    else:
        recorder.stop("done")
    snapshot = recorder._recorder.spool_snapshot()
    recorder.close()
    return snapshot


# --- patching a real spool ---------------------------------------------------


def _transaction_extents(scan) -> dict:
    """Map each committed transaction's offset to where it ends.

    The record cursor reports which transaction made a record visible but not
    where that transaction stops, because nothing on the read path needs to
    know: the trailer has already been checked by the scan. A test that repairs
    checksums does need it, so it derives the boundaries from the walk -- a
    transaction ends where the next one starts, and the last one ends at the
    committed prefix.
    """
    starts = []
    for record in scan.records():
        if not starts or starts[-1] != record.transaction_offset:
            starts.append(record.transaction_offset)
    ends = [*starts[1:], scan.committed_prefix_end]
    return dict(zip(starts, ends, strict=True))


def _patch_payload(data: bytes, kind: int, mutate) -> bytes:
    """Rewrite one record's payload in place and repair every checksum over it.

    Everything outside the injected fault stays the recorder's own bytes, so a
    test of "the finalizer refuses this" is a test against a spool that is
    otherwise exactly what the recorder writes.
    """
    scan = spool.scan_spool(data)
    extents = _transaction_extents(scan)
    buffer = bytearray(data)
    for record in scan.records():
        if record.kind != kind:
            continue
        original = data[record.payload_offset : record.payload_offset + record.payload_bytes]
        payload = mutate(bytearray(original))
        assert len(payload) == len(original)
        body = record.offset + spool.RECORD_HEADER_BYTES
        buffer[body : body + len(payload)] = payload
        struct.pack_into("<I", buffer, record.offset + 24, spool.crc32c(bytes(payload)))
        struct.pack_into(
            "<I",
            buffer,
            record.offset + 28,
            spool.crc32c(bytes(buffer[record.offset : record.offset + 28])),
        )
        trailer = extents[record.transaction_offset] - spool.TRANSACTION_TRAILER_BYTES
        struct.pack_into(
            "<I",
            buffer,
            trailer + 36,
            spool.crc32c(bytes(buffer[record.transaction_offset : trailer])),
        )
        struct.pack_into(
            "<I", buffer, trailer + 44, spool.crc32c(bytes(buffer[trailer : trailer + 44]))
        )
        return bytes(buffer)
    raise AssertionError(f"the spool holds no record of kind {kind}")


def _patch_kind(data: bytes, kind: int, replacement: int) -> bytes:
    """Rewrite one record's *kind* field, repairing the checksums over it."""
    scan = spool.scan_spool(data)
    extents = _transaction_extents(scan)
    buffer = bytearray(data)
    for record in scan.records():
        if record.kind != kind:
            continue
        struct.pack_into("<H", buffer, record.offset, replacement)
        struct.pack_into(
            "<I",
            buffer,
            record.offset + 28,
            spool.crc32c(bytes(buffer[record.offset : record.offset + 28])),
        )
        trailer = extents[record.transaction_offset] - spool.TRANSACTION_TRAILER_BYTES
        struct.pack_into(
            "<I",
            buffer,
            trailer + 36,
            spool.crc32c(bytes(buffer[record.transaction_offset : trailer])),
        )
        struct.pack_into(
            "<I", buffer, trailer + 44, spool.crc32c(bytes(buffer[trailer : trailer + 44]))
        )
        return bytes(buffer)
    raise AssertionError(f"the spool holds no record of kind {kind}")


# --- a complete session -------------------------------------------------------


def test_complete_session_becomes_readable_nrf(tmp_path: Path) -> None:
    """The whole point: a committed spool becomes a session an ordinary reader opens.

    Read back through the public :class:`~neurale.io.nrf.NrfReader`, not through
    the finalizer's own bookkeeping, because an artifact only counts as produced
    when the format's reader accepts it.
    """
    data = _record_session(tmp_path, 64)
    report = finalize_spool(data, tmp_path / "session.nrf")

    assert report.termination_kind == "normal"
    assert report.termination_origin == "recorder"
    assert report.source_session_end_present is True
    assert report.accounting_origin == "recorder"
    assert report.counts.frames == 64
    assert report.counts.signal_blocks == 64

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        values = reader.read_stream("neural")
        assert values.shape == (64 * BLOCK_SAMPLES, CHANNELS)
        assert values.dtype == np.dtype("int16")
        assert reader.manifest["version"] == {"major": 1, "minor": 1}
    assert diagnose_session(tmp_path / "session.nrf").diagnostics == ()


def test_finalized_session_declares_spool_plan(tmp_path: Path) -> None:
    """The plan is rebuilt from the spool, not from a live recorder.

    The spool stores the fingerprinted plan document, and the fingerprint is the
    SHA-256 of exactly those bytes. A session whose extension reproduces the
    fingerprint is one whose plan the spool identifies -- which is what makes an
    offline finalizer possible at all.
    """
    data = _record_session(tmp_path, 32)
    finalize_spool(data, tmp_path / "session.nrf")

    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    extension = manifest["extensions"][NATIVE_REPLAY_NAMESPACE]
    scan = spool.scan_spool(data)
    assert extension["plan_fingerprint"] == scan.superblock.plan_fingerprint
    assert [entry["stream_id"] for entry in extension["streams"]] == ["neural"]

    finalization = manifest["extensions"][FINALIZATION_NAMESPACE]
    assert finalization["termination_origin"] == "recorder"
    assert finalization["source_session_end_present"] is True
    assert finalization["spool"]["committed_transactions"] == scan.committed_transactions
    assert finalization["spool"]["durability_policy"] == "buffered"


def test_every_native_replay_ledger_is_written(tmp_path: Path) -> None:
    """Five ledgers or none: a session with four cannot have its replay evaluated."""
    data = _record_session(tmp_path, 16)
    report = finalize_spool(data, tmp_path / "session.nrf")

    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    extents = manifest["commit"]["committed_extents"]
    for path in ledger_target_paths():
        assert path in extents
    assert extents["records/native_frames/native-frames-v1"] == report.counts.frames
    assert extents["records/native_signal_blocks/native-signal-blocks-v1"] == (
        report.counts.signal_blocks
    )
    assert extents["records/session_accounting/session-accounting-v1"] == 1


def test_frame_ledger_row_carries_unrecorded_fields(tmp_path: Path) -> None:
    """The frame ledger exists because the per-stream records lose the topology.

    A stream keeps samples; it does not keep which frame they arrived in, how
    many blocks that frame carried, or how many of them the plan recorded. That
    is what exact replay needs and what this row restores.
    """
    data = _record_session(tmp_path, 8)
    finalize_spool(data, tmp_path / "session.nrf")

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        rows = reader.read_records("native-frames-v1")
    assert len(rows["data_message_ordinal"]) == 8
    assert list(rows["frame_ordinal"]) == list(range(8))
    assert list(rows["signal_block_count"]) == [1] * 8
    assert list(rows["recorded_signal_block_count"]) == [1] * 8
    # Null rather than zero: the header flags say the frame carried no source
    # tick, and zero is a legal tick.
    assert all(value is None for value in rows["source_tick"])
    assert list(rows["first_signal_block_ordinal"]) == list(range(8))


# --- multi-block frames and discontinuities -----------------------------------


def test_multi_block_frame_is_one_item(tmp_path: Path) -> None:
    """One frame carrying three blocks is one data item, and three ledger rows.

    The distinction is the acceptance ladder's: substituting a block count into
    an item count would make every multi-signal session over-report what the
    runtime handed over.
    """
    data = _record_session(tmp_path, 24, multi_stream=True)
    report = finalize_spool(data, tmp_path / "session.nrf")

    assert report.counts.frames == 24
    assert report.counts.signal_blocks == 24 * 3
    assert report.counts.data_items == 24

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        blocks = reader.read_records("native-signal-blocks-v1")
        assert sorted(set(blocks["stream_id"])) == ["bandpower", "cursor", "neural"]
        assert reader.read_stream("neural").shape[1] == 4
        assert reader.read_stream("cursor").shape[1] == 2
        assert reader.read_stream("bandpower").shape[1] == 3


def test_discontinuity_reaches_ledger_and_stream_records(
    tmp_path: Path,
) -> None:
    """One native message, one ledger row, and one record per affected stream.

    Contract section 5.2 in executable form: the per-stream projection cannot
    say how many messages there were, so the ledger counts messages and the
    per-stream records stay for readers that consume streams.
    """
    data = _record_session(tmp_path, 32, multi_stream=True, sequence_gap_at=8)
    scan = spool.scan_spool(data)
    assert any(r.kind == spool.RECORD_DISCONTINUITY for r in scan.records())

    report = finalize_spool(data, tmp_path / "session.nrf")
    assert report.counts.discontinuities >= 1
    # A frame-level gap with no per-signal detail affects every recorded stream,
    # so the fan-out is a stream count and never the item count.
    assert sum(report.counts.stream_discontinuity_rows.values()) >= 3

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        ledger = reader.read_records("native-discontinuities-v1")
        assert len(ledger["data_message_ordinal"]) == report.counts.discontinuities
        assert all(value > 0 for value in ledger["runtime_accepted_host_time_ns"])
        # Zero gaps is information, not its absence: a frame-level gap has no
        # first child, so its anchor is null rather than gap ordinal zero.
        for count, first in zip(
            ledger["signal_gap_count"], ledger["first_signal_gap_ordinal"], strict=True
        ):
            assert (first is None) == (count == 0)


# --- the control plane --------------------------------------------------------


def test_control_kinds_reach_own_record_sets(tmp_path: Path) -> None:
    """Nine typed submissions become rows in the record sets NRF names for them.

    The producer-identity number is the only bridge from a spool position to an
    NRF kind string, and ``experiment_states``/``experiment_state`` is the one
    place the two vocabularies disagree in spelling -- a mapping, not a guess.
    """
    data = _control_session(tmp_path)
    report = finalize_spool(data, tmp_path / "session.nrf")
    assert report.counts.control_records == 9

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        assert list(reader.read_records("events-v1")["name"]) == ["trial-start"]
        assert list(reader.read_records("experiment-state-v1")["name"]) == ["phase"]
        assert list(reader.read_records("commands-v1")["value"]) == [-2.0]
        assert list(reader.read_records("task-variables-v1")["name"]) == ["gain"]
        assert list(reader.read_records("labels-v1")["text"]) == ["left"]
        assert list(reader.read_records("targets-v1")["value"]) == [3.0]
        assert list(reader.read_records("assistance-v1")["value"]) == [0.5]
        trials = reader.read_records("trials-v1")
        assert list(trials["start_ns"]) == [10]
        assert list(trials["stop_ns"]) == [20]
        assert list(trials["outcome"]) == ["hit"]
        faults = reader.read_records("faults-v1")
        assert list(faults["code"]) == ["edge"]
        assert list(faults["frame_sequence"]) == [7]


# --- an abnormal session ------------------------------------------------------


def test_aborted_session_finalizes_as_aborted(tmp_path: Path) -> None:
    """The finalizer states the outcome the session-end record froze.

    It may not invent a ``normal`` termination, and it may not invent any other
    one either: the capture outcome is frozen at the session-end record and the
    finalizer copies it rather than deciding it.
    """
    data = _control_session(tmp_path, abort=True)
    report = finalize_spool(data, tmp_path / "session.nrf")

    assert report.termination_kind == "aborted"
    assert report.source_session_end_present is True
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        assert list(reader.read_records("termination-v1")["termination_kind"]) == ["aborted"]


def test_spool_without_session_end_finalizes_abnormal(tmp_path: Path) -> None:
    """A committed prefix with no clean end must never be finalized as normal.

    Produced by truncating a real spool at the transaction that would have
    sealed it -- the shape a process crash leaves. The accounting is rebuilt and
    says so; the acceptance columns are the surviving prefix, and
    ``producer_acceptance_known`` is the field that stops them from being read
    as evidence of what a producer handed over.
    """
    data = _record_session(tmp_path, 16)
    scan = spool.scan_spool(data)
    sealing = next(
        record.transaction_offset
        for record in scan.records()
        if record.kind == spool.RECORD_SESSION_END
    )
    truncated = data[:sealing]

    report = finalize_spool(truncated, tmp_path / "session.nrf")
    assert report.termination_kind == "aborted"
    assert report.termination_origin == "recovery"
    assert report.source_session_end_present is False
    assert report.accounting_origin == "recovery_rebuilt"

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        accounting = reader.read_records("session-accounting-v1")
        assert list(accounting["accounting_origin"]) == ["recovery_rebuilt"]
        assert list(accounting["producer_acceptance_known"]) == [False]
        assert list(accounting["termination_origin"]) == ["recovery"]
    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    assert manifest["extensions"][FINALIZATION_NAMESPACE]["source_session_end_present"] is False


# --- the accounting summary ---------------------------------------------------


def test_accounting_summary_holds_identities(tmp_path: Path) -> None:
    """Per handoff, never one total, and never a stored verdict.

    ``accounting_verified`` is deliberately not a column: it is a judgement the
    reader answering the call derives, and a stored copy would let an artifact
    assert a verdict its reader disagrees with.
    """
    data = _record_session(tmp_path, 40)
    report = finalize_spool(data, tmp_path / "session.nrf")

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        row = {
            name: values[0] for name, values in reader.read_records("session-accounting-v1").items()
        }
    assert "accounting_verified" not in row
    assert row["accounting_origin"] == "recorder"
    assert row["producer_acceptance_known"] is True
    assert (
        row["runtime_accepted"]
        == row["recorder_accepted"] + row["failed_between_runtime_and_recorder"]
    )
    assert (
        row["recorder_accepted"] == row["spool_committed"] + row["lost_between_recorder_and_spool"]
    )
    assert row["spool_committed"] == row["nrf_committed"] + row["lost_during_finalization"]
    assert row["nrf_committed"] == report.counts.data_items == 40
    assert row["lost_during_finalization"] == 0


def test_summary_is_sealed_with_session(
    tmp_path: Path,
) -> None:
    """A session cannot be sealed with accounting that disagrees with it.

    Both the accounting row and the termination row are committed by the same
    final transaction, which is what makes "the accounting was sealed with the
    session" checkable from the journal rather than promised by the writer.
    """
    data = _record_session(tmp_path, 8)
    finalize_spool(data, tmp_path / "session.nrf")

    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    last = manifest["commit"]["last_transaction_id"]
    journal = _package_text(tmp_path / "session.nrf", "journal/transactions.jsonl")
    records = (json.loads(line) for line in journal.splitlines())
    prepare = next(
        record
        for record in records
        if record.get("transaction_id") == last and record["kind"] == "prepare"
    )
    sealed = {
        entry["target_path"] for entry in prepare["extents"] if entry["after"] > entry["before"]
    }
    assert "records/session_accounting/session-accounting-v1" in sealed
    assert "records/session_termination/termination-v1" in sealed


def test_accounting_snapshot_beyond_prefix_is_rejected(tmp_path: Path) -> None:
    """A counter larger than the container holds is a failed verification.

    Never reconciled by trusting the number: layer 1 alone is satisfied by a
    snapshot claiming ninety-nine items over a prefix holding two, which is
    exactly the failure the second layer exists for.
    """
    data = _record_session(tmp_path, 12)
    inflated = _patch_payload(
        data,
        spool.RECORD_ACCOUNTING_SNAPSHOT,
        lambda payload: _set_u64(payload, 16, struct.unpack_from("<Q", payload, 16)[0] + 5),
    )
    assert "SPOOL-035" in spool.scan_spool(inflated).codes

    with pytest.raises(SpoolSourceError, match="SPOOL-035"):
        finalize_spool(inflated, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def _set_u64(payload: bytearray, offset: int, value: int) -> bytearray:
    struct.pack_into("<Q", payload, offset, value)
    return payload


# --- source corruption --------------------------------------------------------


def test_unknown_record_kind_blocks_finalization(tmp_path: Path) -> None:
    """Skipping what a reader does not understand is safe for a cache, not a spool.

    The writer committed that record, so it is part of what the session
    captured; promoting a prefix that stepped over it would produce a session
    missing a record nobody counted.
    """
    data = _record_session(tmp_path, 8)
    unknown = _patch_kind(data, spool.RECORD_SIGNAL_BLOCK, 42)
    assert "SPOOL-030" in spool.scan_spool(unknown).codes

    with pytest.raises(SpoolSourceError, match="SPOOL-030"):
        finalize_spool(unknown, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_invalid_superblock_is_rejected(tmp_path: Path) -> None:
    """Nothing in it is readable, so nothing can be finalized from it."""
    data = bytearray(_record_session(tmp_path, 4))
    data[0:8] = b"NOTSPOOL"

    with pytest.raises(SpoolSourceError, match="superblock"):
        finalize_spool(bytes(data), tmp_path / "session.nrf")


def test_torn_tail_is_finalized(tmp_path: Path) -> None:
    """A torn tail is the ordinary crash path, not a reason to refuse.

    Recovery starts from the last valid committed transaction; refusing here
    would leave every interrupted session unconvertible, which is the opposite
    of what the spool is for.
    """
    data = _record_session(tmp_path, 24)
    torn = data[:-16]
    scan = spool.scan_spool(torn)
    assert scan.status == "torn_tail"
    assert scan.finalizable is True

    report = finalize_spool(torn, tmp_path / "session.nrf")
    assert report.spool_codes  # the tail is reported, not hidden
    assert (tmp_path / "session.nrf").exists()


# --- retry, publication, and retention ----------------------------------------


def test_failed_attempt_publishes_nothing_retry_succeeds(tmp_path: Path, monkeypatch) -> None:
    """A retryable failure leaves the target absent and the spool untouched.

    The failure is injected at the cross-check, which is the last step before
    publication and therefore the one where a session has been fully written and
    still must not appear.
    """
    from neurale.recording import _finalizer

    data = _record_session(tmp_path, 16)
    calls: list[int] = []
    original = _finalizer._cross_check

    def failing(target, counts, termination_kind="normal"):
        calls.append(1)
        if len(calls) == 1:
            raise FinalizationError("injected", category="validation")
        return original(target, counts, termination_kind)

    monkeypatch.setattr(_finalizer, "_cross_check", failing)

    with pytest.raises(FinalizationError, match="injected"):
        finalize_spool(data, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()

    staging = tmp_path / ("session.nrf" + STAGING_SUFFIX)
    progress = FinalizationProgress.load(staging / PROGRESS_FILE)
    assert progress.status == "failed_retryable"
    assert progress.category == "validation"
    assert progress.attempts == 1
    assert (staging / STAGING_SESSION).exists()

    report = finalize_spool(data, tmp_path / "session.nrf")
    assert report.progress.attempts == 2
    assert report.progress.status == "succeeded"
    assert (tmp_path / "session.nrf").is_file()
    # The staging directory is the attempt, not the artifact: once the artifact
    # exists there is nothing left for it to hold.
    assert not staging.exists()


def test_retry_over_different_spool_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """Contract section 7: any identity mismatch rejects the resume.

    A target built from one spool must never absorb a second one, and the
    recorded progress is the only thing that can tell the two apart.
    """
    from neurale.recording import _finalizer

    first = _record_session(tmp_path, 8)
    monkeypatch.setattr(
        _finalizer,
        "_cross_check",
        lambda *arguments: (_ for _ in ()).throw(FinalizationError("injected")),
    )
    with pytest.raises(FinalizationError):
        finalize_spool(first, tmp_path / "session.nrf")
    monkeypatch.undo()

    other = _control_session(tmp_path)
    with pytest.raises(FinalizationError, match="different source"):
        finalize_spool(other, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_publication_never_replaces_existing_session(tmp_path: Path) -> None:
    """An NRF session is never deleted or replaced by recording or finalization."""
    data = _record_session(tmp_path, 8)
    finalize_spool(data, tmp_path / "session.nrf")

    with pytest.raises(FinalizationError, match="already exists"):
        finalize_spool(data, tmp_path / "session.nrf")


def test_spool_is_retained_when_requested(tmp_path: Path) -> None:
    """Debug retention is explicit."""
    path = tmp_path / "session.nspool"
    path.write_bytes(_record_session(tmp_path, 8))

    report = finalize_spool(
        path, tmp_path / "session.nrf", options=FinalizerOptions(spool_retention="retain")
    )
    assert report.spool_retained is True
    assert path.exists()


def test_spool_is_deleted_only_after_validated_publication(tmp_path: Path) -> None:
    """Every condition of contract section 7, in order, before anything is removed."""
    path = tmp_path / "session.nspool"
    path.write_bytes(_record_session(tmp_path, 8))

    report = finalize_spool(
        path,
        tmp_path / "session.nrf",
    )
    assert report.spool_retained is False
    assert not path.exists()
    assert (tmp_path / "session.nrf").is_file()


def test_package_progress_and_postpublication_progress_failure(tmp_path: Path, monkeypatch) -> None:
    from neurale.recording import _finalizer

    path = tmp_path / "session.nspool"
    path.write_bytes(_record_session(tmp_path, 8))
    target = tmp_path / "session.nrf"
    phases = []
    original = _finalizer._write_progress

    def record(staging, progress):
        phases.append(progress.phase)
        if progress.phase == "verifying_package":
            assert progress.processed_bytes == progress.total_bytes > 0
        if progress.phase == "cleaning":
            assert target.is_file()
            raise PermissionError("progress update refused after publication")
        return original(staging, progress)

    monkeypatch.setattr(_finalizer, "_write_progress", record)
    report = finalize_spool(path, target)
    expected = (
        "converting",
        "packing",
        "verifying_package",
        "cross_checking",
        "publishing",
        "cleaning",
    )
    assert [phases.index(phase) for phase in expected] == sorted(
        phases.index(phase) for phase in expected
    )
    assert report.progress.status == "succeeded"
    assert not report.spool_retained
    assert not path.exists()


def test_failed_packing_is_not_retried_implicitly_by_close(tmp_path: Path, monkeypatch) -> None:
    from neurale.io.nrf import _package

    path = tmp_path / "session.nspool"
    path.write_bytes(_record_session(tmp_path, 8))
    target = tmp_path / "session.nrf"
    original = _package.publish_package
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("packing interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(_package, "publish_package", fail_once)
    with pytest.raises(FinalizationError, match="packing interrupted"):
        finalize_spool(path, target)
    assert calls == 1
    assert not target.exists()
    assert path.exists()
    report = finalize_spool(path, target)
    assert report.progress.attempts == 2
    assert calls == 2
    assert not report.spool_retained
    assert list(tmp_path.glob("*.finalizing")) == []


def test_conversion_streams_payloads_and_verifies_them_once(tmp_path: Path, monkeypatch) -> None:
    from neurale.io.nrf import NrfReader
    from neurale.io.nrf._package import PayloadArchive
    from neurale.recording import _finalizer

    source = _record_session(tmp_path, 8)
    original_write = PayloadArchive.write
    original_diagnose = _finalizer.diagnose_session
    payload_keys = []
    diagnoses = []

    def write(self, key, raw):
        assert not (self.path.parent / key).exists()
        payload_keys.append(key)
        return original_write(self, key, raw)

    def diagnose(path, **kwargs):
        assert kwargs.get("verify_checksums", True)
        diagnoses.append(path)
        return original_diagnose(path, **kwargs)

    def redundant_verification(*args, **kwargs):
        raise AssertionError("payload verification belongs to the final cross-check")

    monkeypatch.setattr(PayloadArchive, "write", write)
    monkeypatch.setattr(_finalizer, "diagnose_session", diagnose)
    monkeypatch.setattr(NrfReader, "verify_committed_objects", redundant_verification)
    target = tmp_path / "session.nrf"
    finalize_spool(source, target)
    assert any("/c/" in key for key in payload_keys)
    assert len(payload_keys) == len(set(payload_keys))
    assert len(diagnoses) == 1
    assert not list(tmp_path.glob("*.finalizing"))


def test_direct_payload_corruption_blocks_publication_and_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    from neurale.io.nrf._package import PayloadArchive

    source = tmp_path / "capture.spool"
    source.write_bytes(_record_session(tmp_path, 8))
    original = PayloadArchive.write
    corrupted = False

    def damage(self, key, raw):
        nonlocal corrupted
        if not corrupted and "/c/" in key and raw:
            raw = bytes([raw[0] ^ 0xFF]) + raw[1:]
            corrupted = True
        return original(self, key, raw)

    monkeypatch.setattr(PayloadArchive, "write", damage)
    target = tmp_path / "session.nrf"
    with pytest.raises(FinalizationError, match="object_checksum_invalid"):
        finalize_spool(source, target)
    assert corrupted
    assert not target.exists()
    assert source.exists()


def _refuse_to_unlink_the_sidecar(monkeypatch) -> None:
    real_unlink = Path.unlink

    def refuse(self: Path, *arguments: Any, **keywords: Any):
        if self.name.endswith(".plan.json"):
            raise PermissionError("injected sidecar cleanup failure")
        return real_unlink(self, *arguments, **keywords)

    monkeypatch.setattr(Path, "unlink", refuse)


def test_sidecar_is_removed_before_spool_pathname(tmp_path: Path, monkeypatch) -> None:
    """One pathname claims both files, so the order decides whose files are removed.

    The spool wins its path with ``O_EXCL`` and the sidecar's name is derived
    from it. Unlinking the spool first frees that claim, and the next creator
    can take the path -- and write *its* sidecar -- before this cleanup reaches
    its second unlink, which then deletes that recorder's file. The window is
    staged directly here: a competing ``O_EXCL`` create is attempted at the one
    moment it could win, and must not.
    """
    from neurale.recording._finalizer import discard_spool_bundle

    spool = tmp_path / "bundle.nspool"
    sidecar = tmp_path / "bundle.nspool.plan.json"
    spool.write_bytes(b"spool")
    sidecar.write_text("{}", encoding="utf-8")

    real_unlink = Path.unlink
    order: list[str] = []
    refused: list[str] = []

    def watched(self: Path, *arguments: Any, **keywords: Any):
        order.append(self.name)
        result = real_unlink(self, *arguments, **keywords)
        if self.name.endswith(".plan.json"):
            # The next creator, racing at exactly the point the old ordering
            # left open. It must lose: the spool still holds the pathname.
            with pytest.raises(FileExistsError):
                os.close(os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            refused.append(self.name)
        return result

    monkeypatch.setattr(Path, "unlink", watched)
    assert discard_spool_bundle(spool) is None

    assert order == [sidecar.name, spool.name]
    assert refused == [sidecar.name]
    assert not spool.exists()
    assert not sidecar.exists()


def test_unremovable_bundle_keeps_spool_pathname(tmp_path: Path, monkeypatch) -> None:
    """A cleanup that cannot finish must not release the claim it is cleaning up under.

    Unlinking the spool anyway would free the pathname with the sidecar still
    sitting on it, which is the case the ordering exists to prevent -- the next
    creator would win the path and find a document it never wrote.
    """
    from neurale.recording._finalizer import discard_spool_bundle

    spool = tmp_path / "bundle.nspool"
    sidecar = tmp_path / "bundle.nspool.plan.json"
    spool.write_bytes(b"spool")
    sidecar.write_text("{}", encoding="utf-8")
    _refuse_to_unlink_the_sidecar(monkeypatch)

    error = discard_spool_bundle(spool)

    assert isinstance(error, PermissionError)
    assert spool.exists()
    assert sidecar.exists()


def test_sidecar_cleanup_failure_keeps_publication(tmp_path: Path, monkeypatch) -> None:
    """Publication is irreversible. Removing the source spool's sidecar is not.

    By this point the session is on disk, sealed, validated, and the progress
    document already says ``succeeded``. Raising out of the cleanup that comes
    afterwards would be latched by the caller as a failed finalization over a
    session that succeeded -- and the retry it invites would then fail
    non-retryably on a target that already exists, leaving a recorder reporting
    ``failed`` about a published NRF session. The cleanup result is reported as
    ``spool_retained`` instead.
    """
    from neurale.recording._finalizer import _created_at, write_plan_sidecar
    from neurale.recording._plan import compile_recording_plan

    path = tmp_path / "session.nspool"
    data = _record_session(tmp_path, 8)
    path.write_bytes(data)
    scan = spool.scan_spool(data)
    assert scan.superblock is not None
    plan = compile_recording_plan(_recorder_config(tmp_path / "unused.nrf"), _schema())
    plan = dataclasses.replace(
        plan,
        session=dataclasses.replace(
            plan.session,
            session_id=scan.superblock.session_id,
            created_at=_created_at(scan.superblock.created_unix_nanos),
        ),
    )
    sidecar = write_plan_sidecar(path, plan)
    _refuse_to_unlink_the_sidecar(monkeypatch)

    report = finalize_spool(
        path,
        tmp_path / "session.nrf",
        options=FinalizerOptions(spool_retention="delete_after_validated_finalization"),
    )

    assert report.progress.status == "succeeded"
    assert (tmp_path / "session.nrf").is_file()
    # The failure is reported where a cleanup failure belongs, and the spool is
    # still whole: its sidecar could not go, so neither did it.
    assert report.spool_retained is True
    assert path.exists()
    assert sidecar.exists()
    assert set(report.cleanup_paths) == {path, sidecar}


def test_failed_finalization_never_deletes_spool(tmp_path: Path, monkeypatch) -> None:
    """The spool must be retained when finalization, validation, or publication fails.

    This is the failure mode the rule exists to prevent: deleting the only
    reconstruction input because finalization *appeared* to work.
    """
    from neurale.recording import _finalizer

    path = tmp_path / "session.nspool"
    path.write_bytes(_record_session(tmp_path, 8))
    monkeypatch.setattr(
        _finalizer,
        "_cross_check",
        lambda *arguments: (_ for _ in ()).throw(FinalizationError("injected")),
    )

    with pytest.raises(FinalizationError):
        finalize_spool(
            path,
            tmp_path / "session.nrf",
            options=FinalizerOptions(spool_retention="delete_after_validated_finalization"),
        )
    assert path.exists()


# --- typed round trip ---------------------------------------------------------


def test_finalized_session_reconstructs_typed_data(tmp_path: Path) -> None:
    """The end of the road: the artifact becomes the package's own typed objects.

    A session that reads back as arrays but not as a ``Recording`` is one whose
    descriptors do not survive, and the descriptors are most of what a recording
    is for.
    """
    # The block-index stream is left out here on purpose: the synthetic source
    # stamps no observation time on a sampled block, so every block-index row
    # would carry the same timestamp and a typed reconstruction has no honest
    # rate to give it. That is a property of the fixture's source, not of the
    # finalizer -- the Python recorder writes the same rows from the same input.
    data = _record_session(tmp_path, 24, block_idx=False)
    finalize_spool(data, tmp_path / "session.nrf")

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        recording = reader.to_recording()
    signal = recording.signals["neural"]
    assert signal.data.shape == (24 * BLOCK_SAMPLES, CHANNELS)
    assert signal.fs == 1_000
    assert [channel.name for channel in signal.channels] == [
        f"neural-{idx}" for idx in range(CHANNELS)
    ]


def test_abandoning_failed_attempt_destroys_nothing(tmp_path: Path, monkeypatch) -> None:
    """A caller who never retries loses nothing, which is what makes stopping safe.

    Abandoning a finalization is a decision recovery will surface as an operation;
    what the finalizer owes it is that the decision has no cost -- the spool is still
    there, the failed attempt is still on disk as evidence, and no half-session
    was published for anyone to find.
    """
    from neurale.recording import _finalizer

    path = tmp_path / "session.nspool"
    path.write_bytes(_record_session(tmp_path, 12))
    monkeypatch.setattr(
        _finalizer,
        "_cross_check",
        lambda *arguments: (_ for _ in ()).throw(FinalizationError("injected")),
    )
    with pytest.raises(FinalizationError):
        finalize_spool(path, tmp_path / "session.nrf")

    staging = tmp_path / ("session.nrf" + STAGING_SUFFIX)
    assert path.exists()
    assert (staging / PROGRESS_FILE).exists()
    assert not (tmp_path / "session.nrf").exists()
    # The spool is still exactly what the recorder wrote, byte for byte: a
    # failed conversion may not have edited its own source.
    assert spool.scan_spool(path.read_bytes()).finalizable is True


def test_successful_finalization_is_not_rerun(tmp_path: Path) -> None:
    """One spool must not become two sessions.

    The state reconstructed here is the one a crash between the publishing
    rename and the cleanup leaves: the artifact exists and the staging directory
    still records success. Moving the artifact away afterwards -- archiving it,
    say -- is what makes the target-exists check stop applying, and the recorded
    progress is then the only thing that knows the work was already done.
    """
    from neurale.recording._finalizer import _write_progress

    data = _record_session(tmp_path, 8)
    report = finalize_spool(data, tmp_path / "session.nrf")

    staging = tmp_path / ("session.nrf" + STAGING_SUFFIX)
    _write_progress(staging, report.progress)
    (tmp_path / "session.nrf").rename(tmp_path / "archived.nrf")

    with pytest.raises(FinalizationError, match="already succeeded"):
        finalize_spool(data, tmp_path / "session.nrf")


def test_faulted_session_names_fault_row(tmp_path: Path) -> None:
    """A faulted termination must reference a fault, and NRF requires the id.

    The fault here is real: a control body over the plan's whole-record bound is
    a plan violation, which a lossless-until-fault recorder faults on rather than
    dropping. The finalizer copies that outcome and points the termination at the
    row written from the recorder's own committed fault record -- not at a
    control-plane fault a caller submitted earlier, which is a different event.
    """
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "unused.nrf"),
        _schema(),
        options=NativeRecorderOptions(
            spool="memory", max_control_payload_bytes=64, max_control_string_bytes=32
        ),
    )
    recorder.prepare()
    native = recorder._native
    assert recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
    assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
    assert recorder.record_event("x" * 30, text="y" * 30) is False
    status = recorder.stop("after-fault")
    assert status.effective_session_outcome == native.EffectiveSessionOutcome.FAULTED
    data = recorder._recorder.spool_snapshot()
    recorder.close()

    assert spool.scan_spool(data).session_end.capture_outcome_name == "faulted"
    report = finalize_spool(data, tmp_path / "session.nrf")
    assert report.termination_kind == "faulted"
    assert report.counts.fault_rows >= 1

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        termination = reader.read_records("termination-v1")
        faults = reader.read_records("faults-v1")
        assert termination["termination_kind"][0] == "faulted"
        assert termination["fault_id"][0] in set(faults["fault_id"])
        assert any(code.startswith("recorder.") for code in faults["code"])


def test_cross_check_compares_each_record_set(tmp_path: Path) -> None:
    """Layer 2 is per target, because a total hides two errors that cancel.

    Every count the check compares is produced while writing and compared with
    the committed extent the sealed session reports -- two independently
    produced numbers. A check that read the manifest twice would pass for a
    writer that miscounted consistently.
    """
    data = _control_session(tmp_path)
    report = finalize_spool(data, tmp_path / "session.nrf")

    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    extents = manifest["commit"]["committed_extents"]
    paths = {schema["id"]: schema["path"] for schema in manifest["record_schemas"]}
    assert report.counts.record_rows
    for schema_id, rows in report.counts.record_rows.items():
        assert extents[paths[schema_id]] == rows
    # Nine control items, and the fault row among them is a control submission
    # rather than the recorder's own fault record.
    assert sum(report.counts.record_rows.values()) == 9
    assert report.counts.control_records == 9


# --- topology validation, outcome dimensions, publication ---


def _termination_record(session: Path) -> dict[str, Any]:
    """The sealed session's journal termination record."""
    journal = _package_text(session, "journal/transactions.jsonl").splitlines()
    records = [json.loads(line) for line in journal]
    return next(record for record in records if record["kind"] == "termination")


def _set_u32(payload: bytearray, offset: int, value: int) -> bytearray:
    struct.pack_into("<I", payload, offset, value)
    return payload


def _set_u8(payload: bytearray, offset: int, value: int) -> bytearray:
    struct.pack_into("<B", payload, offset, value)
    return payload


def test_frame_claiming_missing_blocks_is_rejected(tmp_path: Path) -> None:
    """A frame whose claimed recorded blocks did not commit is not a committed item.

    The spool container validates framing and owner ordinals but treats record
    payloads as opaque; the finalizer validates the interior topology before any
    NRF is written, so a frame that claims a recorded_signal_block_count it
    cannot back is refused as a source problem rather than published with an
    nrf_committed count its artifact contradicts.
    """
    data = _record_session(tmp_path, 8)
    inflated = _patch_payload(
        data,
        spool.RECORD_FRAME,
        lambda payload: _set_u32(payload, 72, struct.unpack_from("<I", payload, 72)[0] + 1),
    )
    with pytest.raises(SpoolSourceError, match="recorded signal block"):
        finalize_spool(inflated, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_journal_termination_carries_outcome_dimensions(tmp_path: Path) -> None:
    """The outcome fields travel in the termination record's signed extension.

    NRF v1's closed termination_kind enum cannot carry the frozen capture
    outcome, the latched request, the effective outcome, the finalization status
    or whether the primary fault row committed, so they travel in the journal
    record's extensions member and are covered by its checksum. The origin of
    the termination itself travels with them, for the same reason and under the
    same checksum.
    """
    data = _record_session(tmp_path, 16)
    finalize_spool(data, tmp_path / "session.nrf")
    record = _termination_record(tmp_path / "session.nrf")
    extension = record["extensions"][NATIVE_REPLAY_NAMESPACE]
    scan = spool.scan_spool(data)
    assert extension["plan_fingerprint"] == scan.superblock.plan_fingerprint
    assert extension["capture_outcome"] == "normal"
    assert extension["effective_session_outcome"] == "normal"
    assert extension["requested_terminal_intent"] == "normal"
    assert extension["finalization_status"] == "succeeded"
    assert extension["primary_fault_row_committed"] is False
    # Contract section 4.5 requires termination_origin on *both* kinds of
    # termination -- "recorder" in the ordinary case. It is written rather than
    # left to be inferred from the absence of a recovery extension, because
    # provenance read out of a missing field is provenance no checksum signs.
    assert extension["termination_origin"] == "recorder"
    from neurale.io.nrf._canonical import verify_record_checksum

    assert verify_record_checksum(record) is True


def test_termination_kind_is_effective_outcome(
    tmp_path: Path,
) -> None:
    """A recovered session carries capture=unknown but terminates as aborted.

    The capture outcome is frozen and stays unknown; the effective outcome is
    what the NRF termination_kind states. Folding them would either publish a
    recovered session as normal or erase that the recording ran but was never
    sealed by its recorder.
    """
    data = _record_session(tmp_path, 16)
    scan = spool.scan_spool(data)
    sealing = next(
        record.transaction_offset
        for record in scan.records()
        if record.kind == spool.RECORD_SESSION_END
    )
    truncated = data[:sealing]
    report = finalize_spool(truncated, tmp_path / "session.nrf")
    assert report.termination_kind == "aborted"
    # A recovered spool's termination record declares its own origin in the
    # record's extensions (contract section 4.5): the recovery origin cannot
    # live only in the manifest, because the termination record is the thing
    # that says the session ended. The frozen neurale.native_replay schema
    # cannot carry a recovered session -- no latched intent -- so the record
    # carries a neurale.recovery extension instead, signed by its checksum, and
    # a reader can tell a recorder termination from a recovery one in the record
    # itself.
    term = _termination_record(tmp_path / "session.nrf")
    recovery = term["extensions"][RECOVERY_NAMESPACE]
    assert recovery["termination_origin"] == "recovery"
    assert recovery["source_session_end_present"] is False
    assert recovery["capture_outcome"] == "unknown"
    assert recovery["recovery_reason"] == "process_crash"
    assert NATIVE_REPLAY_NAMESPACE not in term["extensions"]
    from neurale.io.nrf._canonical import verify_record_checksum

    assert verify_record_checksum(term) is True
    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    finalization = manifest["extensions"][FINALIZATION_NAMESPACE]
    assert finalization["termination_origin"] == "recovery"
    assert finalization["source_session_end_present"] is False


def test_committed_fault_row_escalates_to_faulted(
    tmp_path: Path,
) -> None:
    """A fault is a fact about the session that outranks the capture's claim.

    The contract's load-bearing case: a capture that ended normal, but a
    committed fault row proves the session faulted. The capture outcome stays
    frozen as normal and the effective outcome escalates to faulted, which is the
    termination_kind -- neither collapsing the two nor publishing a known-short
    session as normal.
    """
    # Record a session that faulted, so a recorder fault row committed and the
    # session-end froze capture=faulted.
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "unused.nrf"),
        _schema(),
        options=NativeRecorderOptions(
            spool="memory", max_control_payload_bytes=64, max_control_string_bytes=32
        ),
    )
    recorder.prepare()
    native = recorder._native
    assert recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
    assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
    assert recorder.record_event("x" * 30, text="y" * 30) is False
    recorder.stop("after-fault")
    data = recorder._recorder.spool_snapshot()
    recorder.close()
    assert spool.scan_spool(data).session_end.capture_outcome_name == "faulted"
    # Rewrite the frozen capture outcome to normal, leaving the committed fault
    # row in place: every byte outside the injected fault is still the recorder's.
    patched = _patch_payload(
        data,
        spool.RECORD_SESSION_END,
        lambda payload: _set_u8(payload, 1, 0),
    )
    assert spool.scan_spool(patched).session_end.capture_outcome_name == "normal"
    report = finalize_spool(patched, tmp_path / "session.nrf")
    assert report.termination_kind == "faulted"
    extension = _termination_record(tmp_path / "session.nrf")["extensions"][NATIVE_REPLAY_NAMESPACE]
    assert extension["capture_outcome"] == "normal"
    assert extension["effective_session_outcome"] == "faulted"
    assert extension["primary_fault_row_committed"] is True


def test_publication_failure_is_retryable(tmp_path: Path, monkeypatch) -> None:
    """A rename failure is in the retryable state machine, not past it.

    The spool and the staged session are intact, so the progress document records
    a retryable publication failure rather than stalling at "running", and no
    half-session is published for a reader to open. A retry succeeds once the
    publication fault clears.
    """
    import os as _os

    import neurale.recording._finalizer as _finalizer

    data = _record_session(tmp_path, 8)
    path = tmp_path / "session.nspool"
    path.write_bytes(data)

    target = tmp_path / "session.nrf"
    operation = "rename" if _os.name == "nt" else "link"
    real_rename = getattr(_os, operation)
    # NrfWriter itself renames cache files atomically, so the hook must fail only
    # the publication rename -- the one whose destination is the final session --
    # and let the writer's own renames through, on this attempt and the retry.
    published = {"attempted": False}

    def failing(src, dst):
        if str(dst) == str(target) and not published["attempted"]:
            published["attempted"] = True
            error = PermissionError("simulated non-transient publication failure")
            error.winerror = 3
            raise error
        return real_rename(src, dst)

    monkeypatch.setattr(_finalizer.os, operation, failing)

    with pytest.raises(FinalizationError, match="publication"):
        finalize_spool(path, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()
    staging = tmp_path / ("session.nrf" + STAGING_SUFFIX)
    progress = FinalizationProgress.load(staging / PROGRESS_FILE)
    assert progress.status == "failed_retryable"
    assert progress.category == "publication"
    # a publication failure never deletes the only reconstruction input
    assert path.exists()

    report = finalize_spool(path, tmp_path / "session.nrf")
    assert report.progress.attempts == 2
    assert report.progress.status == "succeeded"
    assert (tmp_path / "session.nrf").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing errors are platform-specific")
def test_publication_retries_transient_windows_sharing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged.nrf"
    target = tmp_path / "session.nrf"
    staged.write_bytes(b"publication-test")
    real_rename = os.rename
    attempts = 0

    def transient(src: Path, dst: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            error = PermissionError("simulated sharing violation")
            error.winerror = 32
            raise error
        real_rename(src, dst)

    import neurale.recording._finalizer as _finalizer

    monkeypatch.setattr(_finalizer.os, "rename", transient)
    _publish_session(staged, target)
    assert attempts == 3
    assert target.is_file()


def test_non_retryable_failure_is_not_marked_retryable(tmp_path: Path, monkeypatch) -> None:
    """A failure a retry cannot fix is not stamped "failed_retryable".

    The progress document and the exception must agree: a source fault whose
    retry reads the same bytes leaves "failed", so recovery is not told to try again
    over a fault that says trying again cannot help.
    """
    import neurale.recording._finalizer as _finalizer

    data = _record_session(tmp_path, 8)
    path = tmp_path / "session.nspool"
    path.write_bytes(data)

    def non_retryable(*arguments):
        raise FinalizationError("injected source fault", category="source", retryable=False)

    monkeypatch.setattr(_finalizer, "_cross_check", non_retryable)

    with pytest.raises(FinalizationError, match="injected source fault"):
        finalize_spool(path, tmp_path / "session.nrf")
    staging = tmp_path / ("session.nrf" + STAGING_SUFFIX)
    progress = FinalizationProgress.load(staging / PROGRESS_FILE)
    assert progress.status == "failed"
    assert progress.category == "source"
    assert path.exists()


# --- outcome escalation, session/schema identity, typed records ---


def _inject_data_loss(payload: bytearray) -> bytearray:
    """Make a clean spool's accounting prove one item lost between recorder and spool.

    The recorder accepted one more data item than the spool committed, so every
    accounting identity still holds (``recorder_accepted == spool_committed +
    lost_between_recorder_and_spool``) and ``spool_committed`` still equals the
    committed prefix, but the loss counter is non-zero -- the case a clean
    capture whose artifact turned out short is. The first-loss position is set to
    a present ordinal, which is the form the spool's position check requires for
    a non-zero loss counter.
    """
    struct.pack_into("<Q", payload, 0, struct.unpack_from("<Q", payload, 0)[0] + 1)
    struct.pack_into("<Q", payload, 8, struct.unpack_from("<Q", payload, 8)[0] + 1)
    struct.pack_into("<Q", payload, 40, 1)
    payload[112] = 1
    payload[113:116] = b"\x00\x00\x00"
    struct.pack_into("<I", payload, 116, 0)
    struct.pack_into("<Q", payload, 120, 16)
    struct.pack_into("<Q", payload, 128, 0)
    return payload


def _raw_control_session(tmp_path: Path, kind: str, body: dict[str, Any]) -> bytes:
    """Drive a recorder with one raw ``submit_control`` and return its spool."""
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "unused.nrf"),
        _schema(),
        options=NativeRecorderOptions(spool="memory"),
    )
    recorder.prepare()
    native = recorder._native
    assert recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
    assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
    assert recorder.submit_control(kind, body) is True
    recorder.stop("done")
    snapshot = recorder._recorder.spool_snapshot()
    recorder.close()
    return snapshot


def test_required_data_loss_escalates_to_faulted(tmp_path: Path) -> None:
    """A clean capture whose artifact proves short is sealed as faulted, not normal.

    The capture outcome is frozen at the session-end record and stays normal; the
    effective outcome is what the NRF ``termination_kind`` states, and finalization
    proving required data lost escalates it to ``faulted`` (contract section 3.2
    rule 4). The accounting identities still hold -- the recorder accepted one
    more item than the spool committed -- so this is the load-bearing case the two
    fields exist to state, not an accounting fault, and the frozen capture
    outcome is left exactly as the session-end record latched it.
    """
    data = _record_session(tmp_path, 16)
    lossy = _patch_payload(data, spool.RECORD_ACCOUNTING_SNAPSHOT, _inject_data_loss)
    report = finalize_spool(lossy, tmp_path / "session.nrf")
    assert report.termination_kind == "faulted"
    extension = _termination_record(tmp_path / "session.nrf")["extensions"][NATIVE_REPLAY_NAMESPACE]
    assert extension["capture_outcome"] == "normal"
    assert extension["effective_session_outcome"] == "faulted"
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        termination = reader.read_records("termination-v1")
        faults = reader.read_records("faults-v1")
        assert termination["termination_kind"][0] == "faulted"
        assert termination["fault_id"][0] in set(faults["fault_id"])


def test_two_native_sessions_in_one_spool_are_rejected(tmp_path: Path) -> None:
    """A canonical NRF session may not fold two native runtime sessions into one.

    Every frame, discontinuity, and recorder fault carries the native runtime
    session id; the committed prefix is one session, and a frame that claims
    another session's id is a source the finalizer cannot account for, refused
    before any NRF is written rather than sealed as a session it cannot stand
    behind.
    """
    data = _record_session(tmp_path, 16)
    mixed = _patch_payload(data, spool.RECORD_FRAME, lambda p: _set_u64(p, 8, 999))
    with pytest.raises(SpoolSourceError, match="native_session_id"):
        finalize_spool(mixed, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_gap_referencing_undeclared_signal_is_rejected(
    tmp_path: Path,
) -> None:
    """A signal gap may name an unselected signal, but not one the schema never had.

    The gap ledger preserves gaps for signals the recording plan did not select
    (contract section 5.2), so the check is against the native schema's full
    signal set, not the recorded one. A gap that names a signal the schema does
    not declare would persist a reference nothing in the session recognises, and
    is refused before it reaches the ``native-signal-gaps-v1`` ledger.
    """
    data = _record_session(tmp_path, 16, sequence_gap_at=8)
    assert any(r.kind == spool.RECORD_SIGNAL_GAP for r in spool.scan_spool(data).records())
    bad = _patch_payload(data, spool.RECORD_SIGNAL_GAP, lambda p: _set_u32(p, 60, 999))
    with pytest.raises(SpoolSourceError, match="999"):
        finalize_spool(bad, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_control_records_must_agree_on_one_clock_domain(tmp_path: Path) -> None:
    """A control record's clock domain is time semantics, not optional metadata.

    Every control record carries the clock domain its ``time_ns`` belongs to, and
    a session has one control clock, distinct from every data clock. A spool whose
    control records disagree about their clock has a ``time_ns`` whose semantics
    the finalizer cannot seal, so it is refused rather than written with the
    domain silently ignored (contract section 1.2).
    """
    data = _control_session(tmp_path)
    mismatched = _patch_payload(data, spool.RECORD_CONTROL, lambda p: _set_u32(p, 28, 77))
    with pytest.raises(SpoolSourceError, match="clock_domain"):
        finalize_spool(mismatched, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_control_body_with_unknown_field_is_rejected(
    tmp_path: Path,
) -> None:
    """A typed control record must not silently lose a field it cannot map.

    The frozen NRF control schemas have columns for the fields each kind carries
    and no column for anything else. A body offered through the provisional
    ``submit_control`` that names a field no schema has a column for is refused,
    not silently dropped -- silent drop is the one failure this contract does not
    tolerate (section 1.3).
    """
    data = _raw_control_session(
        tmp_path,
        "events",
        {"name": "evt", "value": 1.0, "text": "ok", "unexpected": "MUST_SURVIVE_OR_REJECT"},
    )
    with pytest.raises(FinalizationError, match="unexpected"):
        finalize_spool(data, tmp_path / "session.nrf")
    assert not (tmp_path / "session.nrf").exists()


def test_recorder_fault_keeps_native_provenance(tmp_path: Path) -> None:
    """A recorder fault row carries the native payload the standard columns cannot.

    The frozen ``faults-v1`` schema has columns for when, what code, what stage,
    and which frame and signal, but not for ``runtime_generation``,
    ``sample_index``, ``device_tick``, ``component_id``, ``detail``,
    ``schema_id``, or ``clock_domain``. Those travel as a deterministic
    structured JSON object in the row's ``text`` column, so a fault record loses
    no native provenance and a reader recovers the clock its time belongs to.
    """
    recorder = NativeSessionRecorder.create(
        _recorder_config(tmp_path / "unused.nrf"),
        _schema(),
        options=NativeRecorderOptions(
            spool="memory", max_control_payload_bytes=64, max_control_string_bytes=32
        ),
    )
    recorder.prepare()
    native = recorder._native
    assert recorder._recorder.standalone_pass_readiness_gate(True) == native.RecorderStatusCode.OK
    assert recorder._recorder.standalone_start() == native.RecorderStatusCode.OK
    assert recorder.record_event("x" * 30, text="y" * 30) is False
    recorder.stop("after-fault")
    data = recorder._recorder.spool_snapshot()
    recorder.close()
    finalize_spool(data, tmp_path / "session.nrf")
    with NrfReader.open(tmp_path / "session.nrf") as reader:
        faults = reader.read_records("faults-v1")
        recorder_rows = [
            text
            for code, text in zip(faults["code"], faults["text"], strict=True)
            if str(code).startswith("recorder.")
        ]
        assert recorder_rows, "the recorder's own fault row must be written"
        provenance = json.loads(recorder_rows[0])
        for field in (
            "native_session_id",
            "runtime_generation",
            "sample_index",
            "device_tick",
            "component_id",
            "detail",
            "schema_id",
            "clock_domain",
        ):
            assert field in provenance, f"{field} must survive the conversion"


# --- synthetic primary-fault attribution ----------------


def test_synthetic_fault_row_matches_termination(tmp_path: Path) -> None:
    """A synthesized ``capture_faulted`` row must be the fault the termination names.

    The load-bearing case is a session that already holds a fault row that did
    not end it: a caller submitted a control-plane fault earlier, no recorder
    primary fault record committed, and the effective outcome is nonetheless
    faulted. NRF requires a faulted termination to name a fault, so the finalizer
    writes one from the session-end evidence -- and the termination must point at
    *that* row. Pointing at the session's first fault row instead would seal an
    artifact claiming an unrelated, earlier event ended the capture, which is a
    provenance error in the canonical artifact rather than a missing diagnostic.
    """
    data = _control_session(tmp_path)

    def faulted_without_a_primary_row(payload: bytearray) -> bytearray:
        payload[1] = 2  # capture_outcome = faulted
        payload[2] = 0  # primary_fault_committed = false
        return payload

    faulted = _patch_payload(data, spool.RECORD_SESSION_END, faulted_without_a_primary_row)
    scan = spool.scan_spool(faulted)
    assert scan.session_end.capture_outcome_name == "faulted"
    assert scan.session_end.primary_fault_committed == 0

    report = finalize_spool(faulted, tmp_path / "session.nrf")
    assert report.termination_kind == "faulted"

    with NrfReader.open(tmp_path / "session.nrf") as reader:
        faults = reader.read_records("faults-v1")
        termination = reader.read_records("termination-v1")
    ids = [str(value) for value in faults["fault_id"]]
    codes = [str(value) for value in faults["code"]]
    named = str(termination["fault_id"][0])

    # The control-plane fault came first and is not the fault that ended the
    # session; the synthesized row came last and is.
    assert codes[0] == "edge"
    assert codes[-1] == FAULT_CODE_CAPTURE_FAULTED
    assert named == ids[-1]
    assert named != ids[0]

    manifest = json.loads(_package_text(tmp_path / "session.nrf", "manifest.json"))
    assert manifest["extensions"][FINALIZATION_NAMESPACE]["primary_fault_record_committed"] is False


def test_plan_with_unbuildable_stream_kind_is_rejected(
    tmp_path: Path,
) -> None:
    """A ``spike`` stream is refused rather than materialized as a dense block.

    ``_write_frame`` builds a block as a dense ``n_samples x n_channels``
    array and skips a zero-sample block. A sparse spike block is neither: its
    payload size does not follow from a sample count, and a block carrying no
    spikes is a real committed block whose parent ledger counts it. The recorder
    cannot declare such a stream today, but a spool's plan document is source
    data this process did not write, so the finalizer refuses it up front instead
    of publishing a session whose block ledgers no stream record backs.
    """
    data = _record_session(tmp_path, 8)
    scan = spool.scan_spool(data)
    superblock = scan.superblock
    prepared = plan_from_spool_document(
        superblock.plan_document,
        session=SessionIdentity(
            session_id=superblock.session_id,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            writer_name="neurale",
            writer_version="0",
        ),
        resource_bounds=_bounds_for(scan, FinalizerOptions()),
    )
    assert [stream.kind for stream in prepared.streams] == ["neural"]

    spike = dataclasses.replace(prepared.streams[0], kind="spike")
    with pytest.raises(SpoolSourceError, match="spike"):
        _reject_unmaterializable_streams(dataclasses.replace(prepared, streams=(spike,)))
