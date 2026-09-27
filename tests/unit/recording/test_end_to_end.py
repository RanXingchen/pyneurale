#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The whole lifecycle, end to end: record -> spool -> NRF -> replay.

Every other recording suite owns one stage and stops at its boundary. The
recorder suite asks whether the recorder accepted what the runtime produced,
the finalizer suite whether the finalizer converted the spool faithfully, the
image-building suite whether the image is the compiled form of the committed
session, and the native allocation gate what a run does with an image it
was handed. Each of them is honest about its own stage and blind to the
composition: three stages can each be faithful to the thing in front of them
and still lose the data between them, and nothing in those suites would notice.

So this module asks one question the others cannot: **do the samples a session
recorded come back?** Not "does the image agree with the NRF", which the
image-building suite covers, but does the number a source produced survive the
runtime, the recorder, the spool, the finalizer, the image builder and a
native replay run, and arrive at a consumer as the same number, attributed to
the same sample index of the same signal.

Answering that needed two things the repository did not have.

The first is a payload worth comparing. ``SyntheticNativeSource`` wrote zeroes,
which is the right default for a test about frame plumbing and useless here: a
transposed, misaligned or misattributed block of zeroes is still a block of
zeroes. ``payload_pattern=True`` gives every (signal, sample, channel) position
a different, exactly representable value, in that signal's declared layout,
matching :func:`conftest.sample_values`.

The second is an oracle that does not come from the artifact under test. The
same flag makes the source keep a **manifest**: for every frame it emitted, the
frame's identity and, per block, the position, the device tick, the observation
time, the clock-sync snapshot and a digest of the bytes -- all captured at the
moment of writing. Without it, a parity check can only compare a block with the
metadata that block itself carries, which a whole block moved into a
neighbouring frame satisfies perfectly. With it, "this block belongs to *that*
frame" is a question with an answer. The manifest is itself checked against
:func:`_expected_bytes` before anything is judged against it, so an oracle
wrong in the same direction as the source cannot certify the chain.

The replay leg runs natively. ``NativeReplaySource`` has no Python binding
(the C++ test suite owns that), so the run happens in ``neurale_recording_end_to_end_test``,
which prints each emitted message as JSON, the schema it can recover from an
image alone, and -- through a real ``NativeStreamRunner`` -- the ordered,
per-block record of what a *consumer* at the far end received. The tests that
need it are skipped, not failed, in a checkout with no C++ build. Everything
before the run -- recording, finalization, image building, the accounting -- is
exercised here in Python either way.

**Not claimed anywhere in this file**: throughput, latency or any realtime
property (no benchmark is run), or device, experiment, actuator, or platform
safety beyond the exact native chains executed here. The platform file backend
is refused by the critical-recorder readiness gate on every platform because
requesting cancellation does not itself bound I/O completion and worker
reclamation. Process-crash evidence uses the separate fixed-capacity Linux
tmpfs-mapped backend: its named tmpfs pages are committed and locked before
prepare, its critical append is a bounded memory copy without a backing-file
writeback path, and it supports only the ``buffered`` guarantee. Windows and
disk-backed mappings are refused. No reboot, power-loss, or stronger durability
claim follows from that test.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _native_harness import find_native_test_binary

import neurale.streaming as streaming
from neurale.io.nrf import NrfReader
from neurale.io.nrf._canonical import canonical_json_bytes
from neurale.recording import (
    ACTION_FINALIZE,
    ACTION_QUARANTINE,
    ACTION_REPAIR,
    ReplayConfig,
    abandon_finalization,
    build_replay_image,
    compile_recording_plan,
    diagnose_finalization,
    diagnose_spool,
    finalize_spool,
    load_replay_image,
    open_replay_image,
    repair_spool,
    resume_finalization,
)
from neurale.recording._errors import FinalizationError, RecorderError
from neurale.recording._native_recorder import NativeRecorderOptions, NativeSessionRecorder
from neurale.recording._replay_config import (
    InjectedFault,
    MessageFaultTarget,
    MessageRange,
    StreamReplayRange,
)
from neurale.recording._spool_format import crc32c, scan_spool

from .conftest import (
    BANDPOWER,
    CHANNEL_MAJOR,
    CURSOR,
    DTYPES,
    NEURAL,
    native_schema,
    recorder_config,
    recording_runner,
    sample_values,
    stream_specs,
)
from .test_finalizer import (
    MULTI_MAX_FRAME_PAYLOAD_BYTES,
    _control_session,
    _realtime_config,
)
from .test_recovery import _unsealed, _with_torn_tail

#: What one native session records, by stream id.
SIGNAL_OF_STREAM = {"neural": NEURAL, "cursor": CURSOR, "bandpower": BANDPOWER}

BINARY_NAME = "neurale_recording_end_to_end_test"
BINARY_ENV = "NEURALE_END_TO_END_TEST_BINARY"


# --- the native harness --------------------------------------------------------


def _binary() -> Path | None:
    # Return the built end-to-end harness, or None when there is no C++ build.

    return find_native_test_binary(
        BINARY_NAME,
        explicit_env=BINARY_ENV,
        repo_root=Path(__file__).resolve().parents[3],
    )


@pytest.fixture(scope="module")
def harness() -> Path:
    binary = _binary()
    if binary is None:
        pytest.skip(f"{BINARY_NAME} is not built; configure the C++ tests to run this")
    return binary


@dataclass(frozen=True, slots=True)
class Run:
    """One native replay run, as the harness reported it."""

    summary: dict[str, Any]
    messages: tuple[dict[str, Any], ...]
    tail: dict[str, Any]
    returncode: int

    @property
    def frames(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.messages if item["kind"] == "frame")

    @property
    def discontinuities(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.messages if item["kind"] == "discontinuity")


def _replay(harness: Path, image: Path, *arguments: str) -> Run:
    """Run one image natively and parse what the harness printed."""
    result = subprocess.run(
        [str(harness), "--replay", str(image), *arguments],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"the harness printed nothing (exit {result.returncode}): {result.stderr}"
    assert lines[0].get("status") == "ok", f"the run refused the image: {lines[0]}"
    return Run(lines[0], tuple(lines[1:-1]), lines[-1], result.returncode)


def _through_runtime(harness: Path, image: Path) -> dict[str, Any]:
    """Run one image through a real ``NativeStreamRunner`` and return the report."""
    result = subprocess.run(
        [str(harness), "--runner", str(image)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"the harness printed nothing (exit {result.returncode}): {result.stderr}"
    report = lines[0]
    assert result.returncode == 0, (report, result.stderr)
    assert report.get("status") == "ok", f"the runtime refused the image: {report}"
    assert report["join"] == "ok"
    assert report["runtime_state"] == "stopped"
    assert report["primary_fault_present"] is False
    return report


def _cancel_through_runtime(harness: Path, image: Path) -> dict[str, Any]:
    """Cancel a step-blocked native replay while its runtime is active."""
    result = subprocess.run(
        [str(harness), "--runner", str(image), "--cancel"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"the cancellation harness printed nothing ({result.returncode}): {result.stderr}"
    assert result.returncode == 0, (lines[-1], result.stderr)
    return lines[-1]


def _rerecord(
    harness: Path,
    image: Path,
    plan_document: Path,
    spool: Path,
    *arguments: str,
) -> dict[str, Any]:
    """Replay through a native runtime and critical recorder into *spool*."""
    result = subprocess.run(
        [str(harness), "--rerecord", str(image), str(plan_document), str(spool), *arguments],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"the re-record harness printed nothing ({result.returncode}): {result.stderr}"
    assert result.returncode == 0, (lines[-1], result.stderr)
    return lines[-1]


def _assert_native_resources_released(report: Mapping[str, Any]) -> None:
    """Check resources in the native process that owned the tested chain."""
    assert report["outstanding_frames"] == 0
    assert report["outstanding_discontinuities"] == 0
    assert report["resource_counts_supported"] is True
    assert report["threads_after"] <= report["threads_before"]
    assert report["handles_after"] <= report["handles_before"]


# --- real sessions --------------------------------------------------------------


@dataclass(slots=True)
class Recorded:
    """One real native session and everything the tests need to judge it."""

    spool: bytes
    frames_delivered: int
    #: What the source wrote, recorded as it wrote it. See :func:`_record`.
    manifest: tuple[dict[str, Any], ...]
    status: Any = None
    #: The runtime's own verdict, read before the runner was released, so a
    #: recorder fault can be followed to the abort it is required to cause.
    runtime_fault: Any = None
    runtime_state: Any = None
    outstanding_frames: int = 0
    outstanding_discontinuities: int = 0


def _record(
    tmp_path: Path,
    frames: int,
    *,
    schema: Any = None,
    streams: list[Any] | None = None,
    sequence_gap_at: int | None = None,
    fail_at: int | None = None,
    abort: bool = False,
    block_idx: bool = True,
    vary_block_samples: bool = False,
    recorder_overrides: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    during: Callable[[Any], None] | None = None,
    expect_ok: bool = True,
) -> Recorded:
    """Run one real native session over the three-signal schema.

    The source writes the position-encoding pattern, so every recorded byte has
    a value this file can state independently of what the recorder did with it,
    and it keeps a **manifest**: for every frame it emitted, the identity of that
    frame and, for each block in it, the position, the timing, the clock-sync
    snapshot and a digest of the bytes -- captured at the moment of writing,
    before the runtime has seen them.

    That manifest is the oracle. Judging a replay by its own metadata can only
    establish that the replay agrees with itself: a block moved wholesale into a
    neighbouring frame, provenance and payload together, still carries the
    samples its own header claims. The manifest is what makes "this block
    belongs to *that* frame" a question with an answer.
    """
    schema = native_schema() if schema is None else schema
    config = _realtime_config(frames, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES)
    specs = stream_specs(block_index=block_idx) if streams is None else streams
    overrides: dict[str, Any] = {"checkpoint_interval": 0}
    overrides.update(recorder_overrides or {})
    native_config = recorder_config(tmp_path / "unused.nrf", streams=specs, **overrides)
    option_values: dict[str, Any] = {"spool": "memory", "edge_capacity": max(16, frames * 4)}
    option_values.update(options or {})

    source = streaming.SyntheticNativeSource(
        schema,
        frames,
        1,
        fail_at,
        sequence_gap_at,
        payload_pattern=True,
        vary_block_samples=vary_block_samples,
    )
    runner = recording_runner(schema, config, source)
    recorder = NativeSessionRecorder.create(
        native_config, schema, options=NativeRecorderOptions(**option_values)
    )
    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    started = runner.start()
    if expect_ok:
        assert started == streaming.StreamStatus.OK

    if during is not None:
        # Handed the source, because anything this hook wants to observe about
        # the run has to be observed *during* it: everything after join() is
        # true of any implementation.
        during(source)
    if abort:
        recorder.abort("operator-abort")
    else:
        runner.join()
        recorder.stop("done")
    delivered = source.frames_emitted
    manifest = tuple(source.manifest)
    status = recorder.status
    runtime_fault = runner.primary_fault
    runtime_state = runner.state
    outstanding_frames = runner.outstanding_frames
    outstanding_discontinuities = runner.outstanding_discontinuities
    snapshot = recorder._recorder.spool_snapshot()
    recorder.close()
    return Recorded(
        spool=snapshot,
        frames_delivered=delivered,
        manifest=manifest,
        status=status,
        runtime_fault=runtime_fault,
        runtime_state=runtime_state,
        outstanding_frames=outstanding_frames,
        outstanding_discontinuities=outstanding_discontinuities,
    )


def _session(tmp_path: Path, frames: int = 6, name: str = "session.nrf", **keywords: Any) -> Path:
    """Record and finalize one session, returning the published NRF directory."""
    path = tmp_path / name
    finalize_spool(_record(tmp_path, frames, **keywords).spool, path)
    return path


# --- what the samples should be -------------------------------------------------


def _expected_bytes(signal_id: int, start: int, count: int) -> bytes:
    """The bytes one block of *count* samples should carry, in its own layout.

    Stated from the formula, never read back from the session: a check that
    derived its expectation from the artifact under test would agree with any
    consistent corruption.
    """
    values = sample_values(signal_id, start, count)
    stored = values.T if signal_id in CHANNEL_MAJOR else values
    return np.ascontiguousarray(stored, dtype=DTYPES[signal_id]).tobytes()


def _fnv1a(data: bytes) -> int:
    """The digest the source records in its manifest, restated independently.

    Three lines, no table and no import, so the value the source computed over
    the bytes it wrote can be checked against the bytes the formula predicts
    without either side borrowing the other's implementation.
    """
    digest = 0xCBF29CE484222325
    for byte in data:
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest


#: Everything about a block that the whole chain is required to preserve.
#:
#: ``payload_offset`` is deliberately absent: a replay repacks its frames, so
#: where a block sits inside a payload is the replay's business. Everything
#: else -- where the samples are, when they were observed, which device tick
#: they carry, and under which clock-sync snapshot -- belongs to the recording.
PRESERVED_BLOCK_FIELDS = (
    "signal_id",
    "sample_idx_start",
    "last_sample_idx",
    "n_samples",
    "device_tick_start",
    "observation_time_start_ns",
    "payload_byte_count",
    "payload_digest",
    "clock_sync_device_tick_reference",
    "clock_sync_host_time_reference_ns",
    "clock_sync_rate_numerator",
    "clock_sync_rate_denominator",
    "clock_sync_uncertainty_ns",
    "clock_sync_clock_domain",
    "clock_sync_generation",
    "clock_sync_flags",
)


def assert_manifest_states_the_formula(manifest: tuple[dict[str, Any], ...]) -> None:
    """The oracle is checked before anything is judged against it.

    An oracle taken on trust is a second implementation nobody verified: if the
    source's pattern writer and this file's formula ever disagreed, every parity
    assertion below would fail and blame the chain. So the digest the source
    recorded over the bytes it wrote is compared with a digest of the bytes
    :func:`_expected_bytes` says those samples should be.
    """
    assert manifest, "the source recorded no manifest at all"
    for frame in manifest:
        assert frame["blocks"], "a recorded frame carried no block"
        for block in frame["blocks"]:
            stated = _expected_bytes(
                block["signal_id"], block["sample_idx_start"], block["n_samples"]
            )
            assert len(stated) == block["payload_byte_count"]
            assert _fnv1a(stated) == block["payload_digest"], (
                f"the source wrote different bytes than the formula states for signal "
                f"{block['signal_id']} at sample {block['sample_idx_start']}"
            )


def oracle_frames(
    manifest: tuple[dict[str, Any], ...],
    *,
    signals: set[int] | None = None,
    first: int = 0,
    count: int | None = None,
) -> list[list[dict[str, Any]]]:
    """The blocks the source wrote, grouped by the frame that carried them."""
    window = manifest[first : None if count is None else first + count]
    return [
        [block for block in frame["blocks"] if signals is None or block["signal_id"] in signals]
        for frame in window
    ]


#: The subset of :data:`PRESERVED_BLOCK_FIELDS` a synthesized replay drops.
#:
#: ``stream_frames`` builds its frames from the block index, which stores no
#: clock-sync snapshot, so the mode reports the absence through its fidelity
#: entry rather than inventing one. That is a contract the tests check by
#: asserting the fields are *zero* and the fidelity says why -- not by looking
#: away from them.
CLOCK_SYNC_FIELDS = tuple(
    field for field in PRESERVED_BLOCK_FIELDS if field.startswith("clock_sync_")
)


def assert_block_is_the_recorded_one(
    observed: dict[str, Any], expected: dict[str, Any], *, clock_sync: bool = True
) -> None:
    """One replayed block against the one the source wrote, field by field."""
    for field in PRESERVED_BLOCK_FIELDS:
        if not clock_sync and field in CLOCK_SYNC_FIELDS:
            # The absent snapshot is the default-constructed one, whose rate
            # denominator is 1 rather than 0 -- a rate of x/0 would be a worse
            # lie than no rate at all.
            assert observed[field] == (1 if field.endswith("_rate_denominator") else 0), (
                f"a synthesized replay reported a {field} the block index cannot carry"
            )
            continue
        assert observed[field] == expected[field], (
            f"{field} changed between recording and replay for signal "
            f"{expected['signal_id']} at sample {expected['sample_idx_start']}: "
            f"recorded {expected[field]}, replayed {observed[field]}"
        )
    # And once more from the formula rather than from the source, so a recording
    # and a replay that agreed on the wrong bytes still fail.
    stated = _expected_bytes(
        expected["signal_id"], expected["sample_idx_start"], expected["n_samples"]
    )
    assert crc32c(stated) == observed["payload_crc32c"], (
        f"signal {expected['signal_id']} samples {expected['sample_idx_start']}.."
        f"{expected['sample_idx_start'] + expected['n_samples']} came back changed"
    )


def assert_frames_are_the_recorded_ones(run: Run, expected: list[list[dict[str, Any]]]) -> int:
    """Every replayed frame carries the blocks the source put in *that* frame.

    This is the assertion the mission turns on. Comparing a block with the
    samples its own metadata names cannot see a block that moved to another
    frame with all of its metadata; comparing frame *n* of the replay with frame
    *n* of the source's manifest can.
    """
    assert len(run.frames) == len(expected), (
        f"the replay delivered {len(run.frames)} frames where the source recorded {len(expected)}"
    )
    checked = 0
    for pos, (frame, blocks) in enumerate(zip(run.frames, expected, strict=True)):
        assert len(frame["blocks"]) == len(blocks), (
            f"frame {pos} came back with {len(frame['blocks'])} blocks, recorded with {len(blocks)}"
        )
        for observed, recorded in zip(frame["blocks"], blocks, strict=True):
            assert_block_is_the_recorded_one(observed, recorded)
            checked += 1
    assert checked, "the parity check needs at least one block"
    return checked


def assert_blocks_are_the_recorded_ones(
    run: Run, expected: list[dict[str, Any]], *, clock_sync: bool = True
) -> int:
    """The same comparison where framing is not preserved by design.

    ``stream_frames`` synthesizes its frames, so there is no recorded frame to
    compare one against. The blocks and their order still are the recording's,
    and that is what is checked.
    """
    observed = [block for frame in run.frames for block in frame["blocks"]]
    assert len(observed) == len(expected), (
        f"the replay delivered {len(observed)} blocks where the source recorded {len(expected)}"
    )
    for one, other in zip(observed, expected, strict=True):
        assert_block_is_the_recorded_one(one, other, clock_sync=clock_sync)
    assert observed, "the parity check needs at least one block"
    return len(observed)


# --- the whole chain, one mode at a time ----------------------------------------


def test_recorded_samples_return_through_exact_replay(tmp_path: Path, harness: Path) -> None:
    """The end-to-end claim, in its strongest form.

    A real native session records three signals of three dtypes at three rates,
    one of them channel-major, in multi-block frames. It is finalized, compiled
    to an ``exact_frames`` image, and replayed by the native source. Every block
    that arrives is compared -- position, timing, device tick, clock-sync
    snapshot and bytes -- against the block the *source* recorded in that frame,
    so a block that arrived intact but in the wrong frame fails here.
    """
    recorded = _record(tmp_path, 6)
    assert_manifest_states_the_formula(recorded.manifest)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image)
    assert run.summary["mode"] == "exact_frames"
    assert len(run.frames) == recorded.frames_delivered
    blocks = assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest))
    assert blocks == recorded.frames_delivered * 3

    # Frame-level provenance too: an exact replay keeps the recorded sequence
    # and the recorded source tick, and reports the clock domain the frame was
    # produced in.
    for frame, source_frame in zip(run.frames, recorded.manifest, strict=True):
        assert frame["sequence"] == source_frame["sequence"]
        assert frame["source_tick"] == source_frame["source_tick"]
        assert frame["clock_domain"] == source_frame["clock_domain"]
        assert frame["schema_id"] == source_frame["schema_id"]
        assert frame["flags"] == source_frame["flags"]


def test_full_coverage_projection_matches_exact_replay(tmp_path: Path, harness: Path) -> None:
    """A projection over everything differs from an exact replay in numbering only.

    That is the one difference the contract names for a full-coverage
    projection, so it is the only one allowed to appear: same blocks, same
    bytes, same attribution, renumbered from zero.
    """
    recorded = _record(tmp_path, 6)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    exact = tmp_path / "exact.nrimg"
    projected = tmp_path / "projected.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=exact).close()
    build_replay_image(
        session,
        ReplayConfig(mode="recorded_projection", selected_streams=tuple(SIGNAL_OF_STREAM)),
        path=projected,
    ).close()

    left = _replay(harness, exact)
    right = _replay(harness, projected)
    assert len(left.frames) == len(right.frames)
    for first, second in zip(left.frames, right.frames, strict=True):
        assert first["blocks"] == second["blocks"]
        assert first["payload_crc32c"] == second["payload_crc32c"]
    assert_frames_are_the_recorded_ones(right, oracle_frames(recorded.manifest))


def test_single_stream_projection_carries_one_stream(tmp_path: Path, harness: Path) -> None:
    """Projection is a selection, not a rewrite: the kept bytes are untouched."""
    recorded = _record(tmp_path, 6)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "cursor.nrimg"
    build_replay_image(
        session,
        ReplayConfig(mode="recorded_projection", selected_streams=("cursor",)),
        path=image,
    ).close()

    run = _replay(harness, image)
    # Channel-major is the layout that would survive a transposition unnoticed
    # if the payload were uniform, which is the reason the pattern exists.
    assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest, signals={CURSOR}))
    assert all(len(frame["blocks"]) == 1 for frame in run.frames)


def test_synthesized_replay_carries_recorded_samples(tmp_path: Path, harness: Path) -> None:
    """``stream_frames`` builds frames the recording never had, from real blocks.

    The frames are synthesized; the *samples* are not, and that is the line the
    mode has to hold.
    """
    recorded = _record(tmp_path, 6)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "stream.nrimg"
    build_replay_image(
        session,
        ReplayConfig(
            mode="stream_frames",
            selected_streams=("neural",),
            stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 6)},
        ),
        path=image,
    ).close()

    run = _replay(harness, image)
    assert run.summary["mode"] == "stream_frames"
    assert not run.summary["ledger_based"]
    expected = [
        block for frame in oracle_frames(recorded.manifest, signals={NEURAL}) for block in frame
    ]
    # The synthesized shape is built from the block index, which stores no
    # clock-sync snapshot; the mode declares that rather than inventing one.
    with open_replay_image(image) as opened:
        fidelity = {entry.stream_id: entry for entry in opened.fidelity()}
        assert fidelity["neural"].clock_sync_available is False
    assert_blocks_are_the_recorded_ones(run, expected, clock_sync=False)
    assert [frame["sequence"] for frame in run.frames] == list(range(len(run.frames)))


# --- provenance, status and accounting ------------------------------------------


def test_replay_never_presents_recording_identity(tmp_path: Path, harness: Path) -> None:
    """Section 8.12, checked over a real recording rather than a stated image.

    The run's identity is its own, it is on every message it emits, and it
    survives a reset -- a consumer must never be able to mistake a replay for
    the session it replays.
    """
    session = _session(tmp_path, 4)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image, "--reset")
    recorded = run.summary["recorded_session_id"]
    identity = run.summary["run_session_id"]
    assert identity != 0
    assert identity != recorded
    assert {message["session_id"] for message in run.messages} == {identity}
    assert run.tail["session_id_kept"] is True


def test_run_counters_agree_with_emitted_items(tmp_path: Path, harness: Path) -> None:
    """Status is checked against the messages, not against itself.

    A counter compared only with the image's own summary would agree with a run
    that emitted nothing at all.
    """
    recorded = _record(tmp_path, 6, sequence_gap_at=3)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image)
    assert run.tail["terminal"] == "end_of_data"
    assert run.tail["items_emitted"] == len(run.messages)
    assert run.tail["frames_emitted"] == len(run.frames)
    assert run.tail["discontinuities_emitted"] == len(run.discontinuities)
    assert run.summary["n_items"] == len(run.messages)
    assert run.summary["n_frames"] == len(run.frames)
    assert run.summary["n_discontinuities"] == len(run.discontinuities)


def test_recorded_discontinuity_reaches_replay_with_gaps(tmp_path: Path, harness: Path) -> None:
    """A gap is an item of its own, and it survives the whole chain as one."""
    recorded = _record(tmp_path, 6, sequence_gap_at=3)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image)
    assert run.discontinuities, "the recorded sequence gap did not reach the replay"
    for discontinuity in run.discontinuities:
        assert discontinuity["gaps"], "a discontinuity arrived with no gap at all"
        assert discontinuity["sequence"] > discontinuity["previous_sequence"]
    # A discontinuity is ordered against the frames, not appended to them: the
    # frame it precedes is the one whose sequence it names.
    kinds = [message["kind"] for message in run.messages]
    assert kinds.index("discontinuity") < len(kinds) - 1


def test_control_kinds_survive_conversion_off_data_plane(
    tmp_path: Path,
) -> None:
    """The control plane goes through the same spool and the same finalizer.

    The session here is control-only because it verifies every typed control
    mapping without mixing those assertions with data-plane parity. The
    attached-recorder path is exercised separately by the deterministic
    control-queue-saturation case below.

    What it does establish end to end is the separation: nine control records
    reach the finalized session, and the replay image built over that session
    carries no items at all, because replay is a data-plane story.
    """
    data = _control_session(tmp_path)
    session = tmp_path / "control.nrf"
    report = finalize_spool(data, session)
    assert report.termination_kind == "normal"
    assert report.counts.control_records == 9
    assert report.counts.frames == 0

    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is True
        for schema_id in (
            "events-v1",
            "experiment-state-v1",
            "commands-v1",
            "task-variables-v1",
            "labels-v1",
            "targets-v1",
            "assistance-v1",
            "trials-v1",
            "faults-v1",
        ):
            records = reader.read_records(schema_id)
            assert len(next(iter(records.values()))) >= 1, f"{schema_id} lost its rows"

    image = build_replay_image(
        session, ReplayConfig(mode="exact_frames"), path=tmp_path / "c.nrimg"
    )
    try:
        assert image.metadata.n_items == 0
        assert image.metadata.n_frames == 0
    finally:
        image.close()


# --- committed data only --------------------------------------------------------


def test_committed_prefix_replays_exactly(tmp_path: Path, harness: Path) -> None:
    """Committed-data-only visibility, over a session that really has a prefix.

    A real spool truncated at the transaction that would have sealed it -- the
    shape a crash leaves -- is finalized as an abnormal end. Its replay is
    allowed to be short. It is not allowed to invent, it is not allowed to end
    as though the data had simply run out, and every sample it does deliver is
    the recorded one.
    """
    recorded = _record(tmp_path, 8)
    session = tmp_path / "prefix.nrf"
    report = finalize_spool(_unsealed(recorded.spool), session)
    assert report.termination_kind == "aborted"
    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is False
        committed = len(np.asarray(reader.read_stream("neural").data))

    image = tmp_path / "prefix.nrimg"
    build_replay_image(
        session, ReplayConfig(mode="exact_frames", allow_incomplete=True), path=image
    ).close()
    run = _replay(harness, image)

    assert run.summary["abnormal_end_required"] is True
    assert run.tail["terminal"] == "abnormal_end"
    # A prefix of the recording, compared with the same prefix of the source's
    # manifest: short is allowed, different is not.
    assert_frames_are_the_recorded_ones(
        run, oracle_frames(recorded.manifest, count=len(run.frames))
    )
    replayed = sum(
        block["n_samples"]
        for frame in run.frames
        for block in frame["blocks"]
        if block["signal_id"] == NEURAL
    )
    assert replayed == committed


def test_incomplete_session_replay_needs_opt_in(
    tmp_path: Path,
) -> None:
    """The default answer for an incomplete session is a refusal, not a prefix.

    Two ways to end a session incompletely, one answer to both: a runtime source
    failure, and a crash-shaped spool. Operator abort has its own deterministic
    step-paced native case below; it is not reused as a generic damaged-input
    fixture here.
    """
    recorded = _record(tmp_path, 8)
    sources = {
        "faulted": _record(tmp_path, 8, fail_at=5).spool,
        "unsealed": _unsealed(recorded.spool),
    }
    for label, data in sources.items():
        session = tmp_path / f"{label}.nrf"
        finalize_spool(data, session)
        with NrfReader.open(session) as reader:
            assert reader.completeness.complete is False, label
        with pytest.raises(Exception) as refusal:
            build_replay_image(
                session, ReplayConfig(mode="exact_frames"), path=tmp_path / f"{label}.nrimg"
            )
        assert "incomplete" in str(refusal.value).lower(), label


# --- determinism ----------------------------------------------------------------


def test_reset_run_reproduces_same_run(tmp_path: Path, harness: Path) -> None:
    """Section 8.10 over a real session: every message, every byte, again."""
    session = _session(tmp_path, 6, sequence_gap_at=3)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image, "--reset")
    assert run.tail["reset"] == "ok"
    assert run.tail["second_run_identical"] is True


def test_replays_agree_across_processes(tmp_path: Path, harness: Path) -> None:
    """Determinism across processes, not only across a reset within one.

    A run that depended on process state -- an address, an allocation order, a
    clock read -- could still be identical to itself after a reset.
    """
    session = _session(tmp_path, 6)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    first = _replay(harness, image, "--session-id", "77")
    second = _replay(harness, image, "--session-id", "77")
    assert first.messages == second.messages
    assert first.summary == second.summary


# --- through a real runtime ------------------------------------------------------


@pytest.mark.parametrize("mode", ["exact_frames", "recorded_projection"])
def test_projection_does_not_infer_break(tmp_path: Path, harness: Path, mode: str) -> None:
    """The claim no direct read of the source can check.

    ``recorded_projection`` renumbers the frame sequence from zero. If that
    renumbering left a hole -- or if the recorded discontinuity were dropped and
    the frames left to imply one -- the runtime's continuity checker would infer
    a break of its own and the consumer would see more discontinuities than the
    recording had. So the run goes through a real ``NativeStreamRunner`` and the
    consumer's count is compared with the image's own.
    """
    session = _session(tmp_path, 6, sequence_gap_at=3)
    image = tmp_path / f"{mode}.nrimg"
    keywords: dict[str, Any] = {}
    if mode == "recorded_projection":
        keywords["selected_streams"] = tuple(SIGNAL_OF_STREAM)
    build_replay_image(session, ReplayConfig(mode=mode, **keywords), path=image).close()

    report = _through_runtime(harness, image)
    # Without this the comparison below could pass by both sides being zero,
    # which is exactly the shape a dropped discontinuity would take.
    assert report["image_discontinuity_count"] >= 1
    assert report["consumed_frames"] == report["image_frame_count"]
    assert report["consumed_discontinuities"] == report["image_discontinuity_count"]
    assert report["processor_discontinuities"] == report["image_discontinuity_count"]
    assert report["terminal"] == "end_of_data"
    # The identity a consumer sees is the run's, all the way through the chain.
    assert report["consumer_session_id"] == report["run_session_id"]


def test_schema_rebuilt_from_image_belongs_to_image(tmp_path: Path, harness: Path) -> None:
    """The runner entry only starts if ``check_schema`` accepts the rebuild.

    Which makes the runtime cases above evidence for something narrower and
    worth stating: a schema recovered from nothing but a replay image is the
    schema the recording declared, feature-set descriptors and units included.
    """
    session = _session(tmp_path, 4)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()
    report = _through_runtime(harness, image)
    assert report["status"] == "ok"


def test_runner_entry_reports_flush_fault(tmp_path: Path, harness: Path) -> None:
    """The command-line verdict includes lifecycle completion, not just startup."""
    session = _session(tmp_path, 4)
    image = tmp_path / "flush-fault.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    result = subprocess.run(
        [str(harness), "--runner", str(image), "--stage-fault", "flush"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    reports = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert reports, result.stderr
    report = reports[-1]
    assert result.returncode != 0
    assert report["status"] == "faulted"
    assert report["join"] == "processor_failure"
    assert report["runtime_state"] == "failed"
    assert report["primary_fault_present"] is True


# --- steady state ----------------------------------------------------------------


def test_steady_state_replay_allocates_nothing(tmp_path: Path, harness: Path) -> None:
    """The native allocation gate, run over a recorded image instead of a stated one.

    Multi-block frames, three signals, whatever payload sizes the session
    happened to produce. The tracker's backend is reported rather than assumed:
    under a sanitizer build the ``--wrap`` tracker is off and this is the
    weaker ``operator new`` measurement, which the validation matrix says.
    """
    session = _session(tmp_path, 8)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    result = subprocess.run(
        [str(harness), "--allocate", str(image)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    report = json.loads(result.stdout.splitlines()[0])
    assert report["allocations"] == 0, report
    assert report["items"] == 8
    assert report["backend"] in {"operator_new", "operator_new+malloc"}
    assert result.returncode == 0


def test_session_records_while_python_holds_gil(tmp_path: Path) -> None:
    """The data plane never enters Python, checked where that would show.

    A recording that reached Python per frame would need the GIL per frame. So
    the main thread takes the GIL and does not give it back: ``math.factorial``
    of a large number is a single C call that never releases it. The whole
    session -- source, runtime, continuity checker, recorder, spool -- has to
    run to completion inside that window, and the count is read on the very next
    statement after the call returns, before the runtime thread could have made
    up the difference.

    It is a necessary condition, not a proof: it shows the data path needed no
    Python to finish, not that no Python is reachable from it. The sufficient
    argument is structural and belongs to the recorder suite.
    """
    frames = 64
    observed: list[int] = []

    def hold_the_gil(source: Any) -> None:
        # 120000! is on the order of a second of pure C work in one call that
        # never releases the GIL. The read on the next line is the first Python
        # statement after it, so the runtime thread has had no window of its own
        # in which to catch up.
        math.factorial(120_000)
        observed.append(source.frames_emitted)

    recorded = _record(tmp_path, frames, during=hold_the_gil)
    assert observed == [frames], (
        f"only {observed} of {frames} frames had been produced when the GIL came "
        "back, so the data path did not run independently of Python"
    )
    assert recorded.frames_delivered == frames

    session = tmp_path / "session.nrf"
    report = finalize_spool(recorded.spool, session)
    assert report.counts.frames == frames


# --- the failure matrix -----------------------------------------------------------


def test_clean_stop_is_outcome_control(
    tmp_path: Path,
) -> None:
    """The baseline: complete, sealed, nothing lost."""
    recorded = _record(tmp_path, 8)
    session = tmp_path / "session.nrf"
    report = finalize_spool(recorded.spool, session)
    assert report.termination_kind == "normal"
    assert report.counts.frames == recorded.frames_delivered
    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is True


def test_recorder_and_runtime_faults_both_end_faulted(
    tmp_path: Path,
) -> None:
    """Two origins, one verdict, and the origin is recoverable from the session.

    The **recorder** fault is a control body over the plan's whole-record bound,
    which a lossless-until-fault recorder faults on rather than dropping. It is
    submitted on a standalone-started recorder for the same reason the control
    test above gives: the attached window is not observable from Python.

    The **runtime** fault is a source that fails mid-run, which ends the session
    without a seal. Both are faulted, both are incomplete, and the fault rows
    say which was which -- a verdict that lost the origin would leave an
    operator unable to tell a broken device from a broken recorder.
    """
    recorder_session = tmp_path / "recorder-fault.nrf"
    report = finalize_spool(_control_session_that_faults(tmp_path), recorder_session)
    assert report.termination_kind == "faulted"
    assert report.counts.fault_rows >= 1

    runtime_faulted = _record(tmp_path, 8, fail_at=5)
    assert runtime_faulted.frames_delivered == 5
    runtime_session = tmp_path / "runtime-fault.nrf"
    runtime_report = finalize_spool(runtime_faulted.spool, runtime_session)
    assert runtime_report.termination_kind == "faulted"
    assert runtime_report.counts.fault_rows >= 1

    with NrfReader.open(recorder_session) as reader:
        codes = set(reader.read_records("faults-v1")["code"])
        assert any(code.startswith("recorder.") for code in codes), codes
    with NrfReader.open(runtime_session) as reader:
        assert reader.completeness.complete is False
        assert len(reader.read_records("faults-v1")["code"]) >= 1


@pytest.mark.parametrize(
    ("injected", "expected_stage"),
    [
        ("source", "source"),
        ("processor", "processor"),
        ("consumer", "consumer"),
        ("actuator", "actuator"),
        ("watchdog", "processor"),
    ],
)
def test_runtime_fault_origins_survive_finalization(
    tmp_path: Path, harness: Path, injected: str, expected_stage: str
) -> None:
    """Stage-local faults retain provenance across every recording artifact seam."""
    recorded = _record(tmp_path, 12, sequence_gap_at=4)
    source_session = tmp_path / f"source-{injected}.nrf"
    finalize_spool(recorded.spool, source_session)
    source_image = tmp_path / f"source-{injected}.nrimg"
    build_replay_image(source_session, ReplayConfig(mode="exact_frames"), path=source_image).close()

    plan = compile_recording_plan(
        recorder_config(tmp_path / f"fault-{injected}.nrf"), native_schema()
    )
    plan_path = tmp_path / f"fault-{injected}-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    spool = tmp_path / f"fault-{injected}.spool"
    native = _rerecord(harness, source_image, plan_path, spool, "--stage-fault", injected)
    assert native["status"] == "faulted"
    assert native["fault_stage"] == expected_stage
    _assert_native_resources_released(native)

    session = tmp_path / f"fault-{injected}-final.nrf"
    report = finalize_spool(spool.read_bytes(), session)
    assert report.termination_kind == "faulted"
    with NrfReader.open(session) as reader:
        faults = reader.read_records("faults-v1")
        assert expected_stage in faults["stage"]
        assert reader.completeness.complete is False

    if report.counts.frames:
        image = tmp_path / f"fault-{injected}-prefix.nrimg"
        build_replay_image(
            session, ReplayConfig(mode="exact_frames", allow_incomplete=True), path=image
        ).close()
        replayed = _replay(harness, image)
        assert_frames_are_the_recorded_ones(
            replayed, oracle_frames(recorded.manifest, count=len(replayed.frames))
        )


@pytest.mark.parametrize("injected", ["short_write", "writer_stall", "queue_saturation"])
def test_storage_failures_publish_abnormal_prefix(
    tmp_path: Path, harness: Path, injected: str
) -> None:
    """Writer and queue failures cannot become normal-looking NRF sessions."""
    recorded = _record(tmp_path, 16)
    source_session = tmp_path / f"storage-source-{injected}.nrf"
    finalize_spool(recorded.spool, source_session)
    source_image = tmp_path / f"storage-source-{injected}.nrimg"
    build_replay_image(source_session, ReplayConfig(mode="exact_frames"), path=source_image).close()
    plan = compile_recording_plan(
        recorder_config(tmp_path / f"storage-{injected}.nrf"), native_schema()
    )
    plan_path = tmp_path / f"storage-{injected}-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    spool = tmp_path / f"storage-{injected}.spool"
    native = _rerecord(harness, source_image, plan_path, spool, "--storage-fault", injected)
    assert native["status"] == "faulted"
    assert native["fault_stage"] == "observer"
    _assert_native_resources_released(native)

    session = tmp_path / f"storage-{injected}-final.nrf"
    report = finalize_spool(spool.read_bytes(), session)
    assert report.termination_kind != "normal"
    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is False
    if report.counts.frames:
        image = tmp_path / f"storage-{injected}-prefix.nrimg"
        build_replay_image(
            session, ReplayConfig(mode="exact_frames", allow_incomplete=True), path=image
        ).close()
        replayed = _replay(harness, image)
        assert_frames_are_the_recorded_ones(
            replayed, oracle_frames(recorded.manifest, count=len(replayed.frames))
        )


def test_control_queue_saturation_accounts_both_planes(tmp_path: Path, harness: Path) -> None:
    """Control saturation is exercised while the recorder is attached and accepting data."""
    recorded = _record(tmp_path, 16)
    source_session = tmp_path / "control-saturation-source.nrf"
    finalize_spool(recorded.spool, source_session)
    source_image = tmp_path / "control-saturation-source.nrimg"
    build_replay_image(source_session, ReplayConfig(mode="exact_frames"), path=source_image).close()
    plan = compile_recording_plan(
        recorder_config(tmp_path / "control-saturation.nrf"), native_schema()
    )
    plan_path = tmp_path / "control-saturation-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    spool = tmp_path / "control-saturation.spool"

    native = _rerecord(
        harness,
        source_image,
        plan_path,
        spool,
        "--storage-fault",
        "control_queue_saturation",
    )
    assert native["status"] == "faulted"
    assert native["fault_stage"] == "observer"
    assert native["recorder_fault_reason"] == "control_queue_saturated"
    assert native["control_saturation_observed"] is True
    assert native["join"] != "deadline_exceeded"
    assert native["runtime_state"] == "failed"
    assert native["control_offered"] == native["control_accepted"] + 1
    assert native["control_rejected"] == 1
    assert native["control_spool_committed"] == native["control_accepted"]
    _assert_native_resources_released(native)

    session = tmp_path / "control-saturation-final.nrf"
    report = finalize_spool(spool.read_bytes(), session)
    assert report.termination_kind == "faulted"
    assert report.counts.control_records == native["control_spool_committed"]
    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is False
        assert "recorder.control_queue_saturated" in set(reader.read_records("faults-v1")["code"])

    if report.counts.frames:
        prefix_image = tmp_path / "control-saturation-prefix.nrimg"
        build_replay_image(
            session,
            ReplayConfig(mode="exact_frames", allow_incomplete=True),
            path=prefix_image,
        ).close()
        replayed = _replay(harness, prefix_image)
        assert_frames_are_the_recorded_ones(
            replayed, oracle_frames(recorded.manifest, count=len(replayed.frames))
        )


def test_operator_abort_drains_prefix_without_fault(tmp_path: Path, harness: Path) -> None:
    """An explicit runtime abort is distinct from a recorder or runtime failure."""
    recorded = _record(tmp_path, 16)
    source_session = tmp_path / "operator-abort-source.nrf"
    finalize_spool(recorded.spool, source_session)
    source_image = tmp_path / "operator-abort-source.nrimg"
    build_replay_image(source_session, ReplayConfig(mode="exact_frames"), path=source_image).close()
    plan = compile_recording_plan(recorder_config(tmp_path / "operator-abort.nrf"), native_schema())
    plan_path = tmp_path / "operator-abort-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    spool = tmp_path / "operator-abort.spool"

    native = _rerecord(harness, source_image, plan_path, spool, "--operator-abort-at", "3")
    assert native["status"] == "aborted"
    assert native["operator_aborted"] is True
    assert native["runtime_state"] == "stopped"
    assert native["fault_stage"] == "none"
    assert native["recorder_fault_reason"] == "none"
    assert 3 <= native["runtime_accepted"] < 16
    assert native["spool_committed"] == native["runtime_accepted"]
    _assert_native_resources_released(native)

    session = tmp_path / "operator-abort-final.nrf"
    report = finalize_spool(spool.read_bytes(), session)
    assert report.termination_kind == "aborted"
    assert report.counts.frames == native["spool_committed"]
    assert report.counts.fault_rows == 0
    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is False

    prefix_image = tmp_path / "operator-abort-prefix.nrimg"
    build_replay_image(
        session, ReplayConfig(mode="exact_frames", allow_incomplete=True), path=prefix_image
    ).close()
    replayed = _replay(harness, prefix_image)
    assert replayed.tail["terminal"] == "abnormal_end"
    assert_frames_are_the_recorded_ones(
        replayed, oracle_frames(recorded.manifest, count=len(replayed.frames))
    )


def test_retryable_finalization_failure_keeps_capture_clean(
    tmp_path: Path, harness: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real publication I/O failure retains the clean spool and retries."""
    from neurale.recording import _finalizer

    recorded = _record(tmp_path, 10)
    frozen_spool = recorded.spool
    assert scan_spool(frozen_spool).session_end.capture_outcome_name == "normal"
    spool = tmp_path / "retry.spool"
    spool.write_bytes(frozen_spool)
    target = tmp_path / "retry.nrf"
    operation = "rename" if os.name == "nt" else "link"
    real_rename = getattr(os, operation)
    publication_attempted = False

    def fail_publication_once(source: Any, destination: Any) -> Any:
        nonlocal publication_attempted
        if Path(destination) == target and not publication_attempted:
            publication_attempted = True
            error = PermissionError("simulated publication I/O failure")
            error.winerror = 3
            raise error
        return real_rename(source, destination)

    monkeypatch.setattr(_finalizer.os, operation, fail_publication_once)
    with pytest.raises(FinalizationError, match="publication") as failed:
        finalize_spool(spool, target)
    assert failed.value.retryable
    assert publication_attempted
    assert not target.exists()
    assert spool.read_bytes() == frozen_spool
    assert scan_spool(spool.read_bytes()).session_end.capture_outcome_name == "normal"

    report = finalize_spool(spool, target)
    assert report.progress.attempts == 2
    assert report.termination_kind == "normal"
    assert spool.exists()
    assert spool.read_bytes() == frozen_spool
    with NrfReader.open(target) as reader:
        assert reader.completeness.complete is True
        assert len(reader.read_records("faults-v1")["code"]) == 0

    image = tmp_path / "retry.nrimg"
    build_replay_image(target, ReplayConfig(mode="exact_frames"), path=image).close()
    replayed = _replay(harness, image)
    assert_frames_are_the_recorded_ones(replayed, oracle_frames(recorded.manifest))


def test_replay_cancellation_wakes_and_releases(tmp_path: Path, harness: Path) -> None:
    """Cancellation crosses source, runtime, terminal state, and cleanup."""
    recorded = _record(tmp_path, 12)
    session = tmp_path / "cancel-source.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "cancel-source.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    report = _cancel_through_runtime(harness, image)
    assert report["status"] == "cancelled"
    assert report["cancel_requested"] is True
    assert report["terminal"] == "cancelled"
    assert report["join"] == "ok"
    assert report["runtime_state"] == "stopped"
    assert report["primary_fault_present"] is False
    assert report["cancel_elapsed_ms"] < 2_000
    assert report["outstanding_frames"] == 0
    assert report["outstanding_discontinuities"] == 0


def test_mismatched_staged_target_leaves_spool_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publication interruption cannot be resumed through another session."""
    from neurale.recording import _finalizer

    recorded = _record(tmp_path, 8)
    spool = tmp_path / "mismatch.spool"
    spool.write_bytes(recorded.spool)
    frozen_spool = spool.read_bytes()
    target = tmp_path / "mismatch.nrf"
    operation = "rename" if os.name == "nt" else "link"
    real_rename = getattr(os, operation)
    failed = False

    def stop_at_publication(source: Any, destination: Any) -> Any:
        nonlocal failed
        if Path(destination) == target and not failed:
            failed = True
            error = PermissionError("interrupted publication")
            error.winerror = 3
            raise error
        return real_rename(source, destination)

    monkeypatch.setattr(_finalizer.os, operation, stop_at_publication)
    with pytest.raises(FinalizationError, match="publication"):
        finalize_spool(spool, target)
    assert failed

    other_recorded = _record(tmp_path, 8)
    other = tmp_path / "other.nrf"
    finalize_spool(other_recorded.spool, other)
    staged = target.with_name(target.name + _finalizer.STAGING_SUFFIX) / _finalizer.STAGING_SESSION
    staged.unlink()
    shutil.copyfile(other, staged)
    staged_before = staged.read_bytes()

    with pytest.raises(FinalizationError, match="not the session this finalization staged") as bad:
        resume_finalization(spool, target)
    assert bad.value.category == "resume_mismatch"
    assert not bad.value.retryable
    assert not target.exists()
    assert spool.read_bytes() == frozen_spool
    assert staged.read_bytes() == staged_before


def test_stale_replay_image_is_rebuilt(tmp_path: Path, harness: Path) -> None:
    """One cache path cannot make an old session current after source replacement."""
    first = _record(tmp_path, 4)
    current = tmp_path / "current.nrf"
    finalize_spool(first.spool, current)
    image = tmp_path / "current.nrimg"
    config = ReplayConfig(mode="exact_frames")
    build_replay_image(current, config, path=image).close()
    assert len(_replay(harness, image).frames) == 4

    second = _record(tmp_path, 7)
    current.unlink()
    finalize_spool(second.spool, current)
    rebuilt = load_replay_image(current, config, path=image)
    assert rebuilt.reused is False
    rebuilt.image.close()

    replayed = _replay(harness, image)
    assert len(replayed.frames) == 7
    assert_frames_are_the_recorded_ones(replayed, oracle_frames(second.manifest))


@pytest.mark.skipif(
    sys.platform not in {"linux", "win32"}
    or (sys.platform == "linux" and not Path("/dev/shm").is_dir()),
    reason="process-crash critical recording requires Linux tmpfs or Windows local storage",
)
def test_killed_recording_is_recovered_next_process(
    tmp_path: Path, harness: Path, request: pytest.FixtureRequest
) -> None:
    """Kill an active critical recorder, then repair, finalize, and replay its prefix."""
    recorded = _record(tmp_path, 16)
    source_session = tmp_path / "process-crash-source.nrf"
    finalize_spool(recorded.spool, source_session)
    source_image = tmp_path / "process-crash-source.nrimg"
    build_replay_image(source_session, ReplayConfig(mode="exact_frames"), path=source_image).close()
    plan = compile_recording_plan(
        recorder_config(tmp_path / "process-crash.nrf", checkpoint_interval=0), native_schema()
    )
    plan_path = tmp_path / "process-crash-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    crash_id = uuid.uuid4().hex
    spool_root = Path("/dev/shm") if sys.platform == "linux" else tmp_path
    spool = spool_root / f"pyneurale-process-crash-{crash_id}.spool"
    quarantine = spool_root / f"pyneurale-process-crash-quarantine-{crash_id}"
    request.addfinalizer(lambda: spool.unlink(missing_ok=True))
    request.addfinalizer(lambda: shutil.rmtree(quarantine, ignore_errors=True))

    process = subprocess.Popen(
        [
            str(harness),
            "--rerecord",
            str(source_image),
            str(plan_path),
            str(spool),
            "--crash-after-committed",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()

    def read_status() -> None:
        assert process.stdout is not None
        lines.put(process.stdout.readline())

    threading.Thread(target=read_status, daemon=True).start()
    try:
        line = lines.get(timeout=30)
        assert line, "the recording process exited before reporting an active committed prefix"
        active = json.loads(line)
        assert active["status"] == "recording_active", active
        assert active["spool_committed"] >= 3
        assert process.poll() is None
        process.kill()
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    assert process.returncode != 0
    assert spool.exists()
    diagnosis = diagnose_spool(spool)
    assert diagnosis.recommended_action in {ACTION_REPAIR, ACTION_FINALIZE}
    if diagnosis.recommended_action == ACTION_REPAIR:
        assert repair_spool(spool, quarantine_dir=quarantine).repaired
    assert diagnose_spool(spool).recommended_action == ACTION_FINALIZE

    recovered = tmp_path / "process-crash-recovered.nrf"
    report = finalize_spool(spool, recovered)
    assert report.counts.frames == active["spool_committed"] == active["runtime_accepted"]
    with NrfReader.open(recovered) as reader:
        assert reader.completeness.complete is False
    replay_image = tmp_path / "process-crash-recovered.nrimg"
    build_replay_image(
        recovered,
        ReplayConfig(mode="exact_frames", allow_incomplete=True),
        path=replay_image,
    ).close()
    replayed = _replay(harness, replay_image)
    assert_frames_are_the_recorded_ones(
        replayed, oracle_frames(recorded.manifest, count=len(replayed.frames))
    )


def _control_session_that_faults(tmp_path: Path) -> bytes:
    """A real recorder faulted by a control body over its whole-record bound."""
    recorder = NativeSessionRecorder.create(
        recorder_config(tmp_path / "unused.nrf", streams=stream_specs(), checkpoint_interval=0),
        native_schema(),
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
    snapshot = recorder._recorder.spool_snapshot()
    recorder.close()
    return snapshot


def test_crash_shaped_spool_is_repaired_and_replayed(tmp_path: Path, harness: Path) -> None:
    """The recovery path, end to end.

    A real spool with a torn tail is diagnosed, repaired, finalized and
    replayed, and what comes back is the committed prefix -- the same samples,
    at the same positions, as the untorn recording delivers.
    """
    recorded = _record(tmp_path, 8)
    torn = tmp_path / "torn.spool"
    torn.write_bytes(_with_torn_tail(_unsealed(recorded.spool)))

    assert diagnose_spool(torn).recommended_action == ACTION_REPAIR
    # Repair truncates in place and quarantines the original, so afterwards the
    # path holds the committed prefix and the tail is preserved beside it.
    assert repair_spool(torn, quarantine_dir=tmp_path / "quarantine").repaired
    assert diagnose_spool(torn).recommended_action == ACTION_FINALIZE

    session = tmp_path / "recovered.nrf"
    report = finalize_spool(torn.read_bytes(), session)
    assert report.counts.frames == recorded.frames_delivered
    image = tmp_path / "recovered.nrimg"
    build_replay_image(
        session, ReplayConfig(mode="exact_frames", allow_incomplete=True), path=image
    ).close()

    run = _replay(harness, image)
    assert run.frames, "a repaired session replayed nothing at all"
    assert_frames_are_the_recorded_ones(
        run, oracle_frames(recorded.manifest, count=len(run.frames))
    )


def test_successive_recordings_do_not_leak(tmp_path: Path, harness: Path) -> None:
    """Two recordings, two sessions, two replays -- and no leakage between them.

    The second session is deliberately shorter, so a replay that reached the
    wrong session's data would come back with the wrong number of frames rather
    than merely the wrong bytes.

    This remains the independent-session isolation case. The actual
    replay-to-recorder path is covered separately below.
    """
    first = _record(tmp_path, 8)
    second = _record(tmp_path, 4)
    sessions = []
    for i, recorded in enumerate((first, second)):
        session = tmp_path / f"session-{i}.nrf"
        finalize_spool(recorded.spool, session)
        image = tmp_path / f"session-{i}.nrimg"
        build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()
        sessions.append((recorded, _replay(harness, image)))

    for recorded, run in sessions:
        assert len(run.frames) == recorded.frames_delivered
        assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest))
    assert len(sessions[0][1].frames) != len(sessions[1][1].frames)


def test_replay_can_be_recorded_and_replayed_again(tmp_path: Path, harness: Path) -> None:
    """record -> NRF -> replay -> runtime -> recorder -> NRF -> replay.

    The source manifest from the first run remains the oracle. The C++ harness
    owns the integration seam that rebuilds the runtime schema from the image
    and prepares a second native recording plan; the replay library itself
    remains independent of the recorder target.
    """
    first = _record(tmp_path, 10, sequence_gap_at=5)
    first_session = tmp_path / "first.nrf"
    finalize_spool(first.spool, first_session)
    first_image = tmp_path / "first.nrimg"
    build_replay_image(first_session, ReplayConfig(mode="exact_frames"), path=first_image).close()

    plan = compile_recording_plan(recorder_config(tmp_path / "second.nrf"), native_schema())
    plan_path = tmp_path / "second-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    second_spool = tmp_path / "second.spool"
    native = _rerecord(harness, first_image, plan_path, second_spool)
    assert native["runtime_accepted"] == len(first.manifest) + 1  # frames plus the gap
    assert native["spool_committed"] == native["runtime_accepted"]
    _assert_native_resources_released(native)

    second_session = tmp_path / "second.nrf"
    finalize_spool(second_spool.read_bytes(), second_session)
    second_image = tmp_path / "second.nrimg"
    build_replay_image(second_session, ReplayConfig(mode="exact_frames"), path=second_image).close()
    second = _replay(harness, second_image)
    assert_frames_are_the_recorded_ones(second, oracle_frames(first.manifest))
    assert len(second.discontinuities) == 1


# --- injected faults, over a real recording ---------------------------------------


def test_injected_read_failure_ends_replay_at_target(tmp_path: Path, harness: Path) -> None:
    """A fault positioned by recorded identity, fired during a real replay."""
    session = _session(tmp_path, 6)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image, "--fault", "ledger:frame:3:read_failure")
    assert run.tail["terminal"] == "faulted"
    assert run.tail["faults"][0]["fired"] is True
    assert run.tail["faults"][0]["emitted"] is False
    assert len(run.frames) == 3


def test_fault_rejected_in_python_is_rejected_natively(tmp_path: Path, harness: Path) -> None:
    """One legality, two front doors.

    Two faults on one recorded position are refused by ``ReplayConfig`` before
    an image is ever built, and by the native source when it is handed the same
    pair directly. A native API that accepted what the Python one refuses would
    make the contract a matter of which door a caller used.
    """
    session = _session(tmp_path, 6)
    with pytest.raises(Exception) as refusal:
        ReplayConfig(
            mode="exact_frames",
            faults=(
                InjectedFault(target=MessageFaultTarget("frame", 3), effect="stall", stall_ns=1),
                InjectedFault(target=MessageFaultTarget("frame", 3), effect="read_failure"),
            ),
        )
    assert "same position" in str(refusal.value)

    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()
    result = subprocess.run(
        [
            str(harness),
            "--replay",
            str(image),
            "--fault",
            "ledger:frame:3:stall:1",
            "--fault",
            "ledger:frame:3:read_failure",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert json.loads(result.stdout.splitlines()[0]) == {"status": "duplicate_fault_target"}


# --- resources --------------------------------------------------------------------


def test_chain_leaves_no_temporary(tmp_path: Path, harness: Path) -> None:
    """A published session and a published image, and nothing else.

    Every stage of this chain writes through a temporary and renames. A stage
    that left one behind would be invisible to every assertion above and would
    fill a real deployment's disk.
    """
    recorded = _record(tmp_path, 6)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()
    _replay(harness, image)

    leftovers = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and (path.name.endswith(".tmp") or ".staging" in path.name)
    ]
    assert not leftovers, f"the chain left {leftovers} behind"


def test_replayed_session_can_be_read_again(tmp_path: Path, harness: Path) -> None:
    """Building and running an image is read-only over the session it replays."""
    session = _session(tmp_path, 6)
    with NrfReader.open(session) as reader:
        before = np.asarray(reader.read_stream("neural").data).copy()

    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()
    _replay(harness, image, "--reset")

    with NrfReader.open(session) as reader:
        after = np.asarray(reader.read_stream("neural").data)
        assert reader.completeness.complete is True
    assert np.array_equal(before, after)


def test_range_replay_covers_requested_messages(tmp_path: Path, harness: Path) -> None:
    """A range is a request about recorded positions, honoured exactly."""
    recorded = _record(tmp_path, 8)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "range.nrimg"
    build_replay_image(
        session,
        ReplayConfig(
            mode="recorded_projection",
            selected_streams=tuple(SIGNAL_OF_STREAM),
            message_range=MessageRange(2, 6),
        ),
        path=image,
    ).close()

    run = _replay(harness, image)
    assert len(run.messages) == 4
    # Messages 2..5 of the recording, compared with recorded frames 2..5 -- a
    # range that delivered the right *count* from the wrong offset fails here.
    assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest, first=2, count=4))
    # Renumbered from zero, and the samples still those of messages 2..5.
    assert [frame["sequence"] for frame in run.frames] == list(range(len(run.frames)))
    assert run.frames[0]["blocks"][0]["sample_idx_start"] == 2 * 8


# --- what a consumer at the far end of a real runtime receives ---------------------


def _consumed_blocks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The blocks the runner entry's consumer actually received, in order."""
    assert report["consumer_record_truncated"] is False, (
        "the consumer's record filled up, so what follows would be a partial comparison"
    )
    return list(report["consumed"])


@pytest.mark.parametrize("mode", ["exact_frames", "recorded_projection"])
def test_consumer_receives_recorded_bytes(tmp_path: Path, harness: Path, mode: str) -> None:
    """The claim the validation matrix makes, checked where it is made.

    A direct read of ``NativeReplaySource`` shows what the *source* produced. It
    says nothing about what survives the runtime's queues, the processor stage
    and the consumer edge: a seam that corrupted or reordered payloads there
    would leave every frame count intact and every direct-read assertion above
    passing. So the consumer at the end of a real ``NativeStreamRunner`` keeps
    an ordered, per-block record of what arrived, and it is compared with what
    the source recorded -- position, device tick, size and a digest of the bytes.
    """
    recorded = _record(tmp_path, 6)
    assert_manifest_states_the_formula(recorded.manifest)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / f"{mode}.nrimg"
    keywords: dict[str, Any] = {}
    if mode == "recorded_projection":
        keywords["selected_streams"] = tuple(SIGNAL_OF_STREAM)
    build_replay_image(session, ReplayConfig(mode=mode, **keywords), path=image).close()

    report = _through_runtime(harness, image)
    consumed = _consumed_blocks(report)
    expected = [block for frame in oracle_frames(recorded.manifest) for block in frame]
    assert len(consumed) == len(expected), (
        f"the consumer received {len(consumed)} blocks where the source recorded {len(expected)}"
    )
    for arrived, source_block in zip(consumed, expected, strict=True):
        for field in (
            "signal_id",
            "sample_idx_start",
            "n_samples",
            "device_tick_start",
            "payload_byte_count",
            "payload_digest",
        ):
            assert arrived[field] == source_block[field], (
                f"{field} differs at the consumer for signal {source_block['signal_id']} "
                f"sample {source_block['sample_idx_start']}"
            )

    # And the framing the consumer saw is the recording's: a run that delivered
    # every byte in one enormous frame would pass the comparison above.
    grouping: dict[int, list[dict[str, Any]]] = {}
    for arrived in consumed:
        grouping.setdefault(arrived["frame_ordinal"], []).append(arrived)
    assert len(grouping) == len(recorded.manifest)
    assert all(len(blocks) == 3 for blocks in grouping.values())


def test_synthesized_replay_reaches_consumer(tmp_path: Path, harness: Path) -> None:
    """``stream_frames`` through a real runtime, not only through a direct read.

    The mode invents its framing, so it is the one most likely to be refused by
    a runtime or to arrive rearranged. The frames are the image's; the samples
    have to be the recording's.
    """
    recorded = _record(tmp_path, 6)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "stream.nrimg"
    build_replay_image(
        session,
        ReplayConfig(
            mode="stream_frames",
            selected_streams=("neural",),
            stream_ranges={"neural": StreamReplayRange("block_ordinal", 0, 6)},
        ),
        path=image,
    ).close()

    report = _through_runtime(harness, image)
    consumed = _consumed_blocks(report)
    expected = [
        block for frame in oracle_frames(recorded.manifest, signals={NEURAL}) for block in frame
    ]
    assert [block["sample_idx_start"] for block in consumed] == [
        block["sample_idx_start"] for block in expected
    ]
    assert [block["payload_digest"] for block in consumed] == [
        block["payload_digest"] for block in expected
    ]


# --- provenance the image alone cannot vouch for -----------------------------------


def test_schema_from_image_matches_recording(tmp_path: Path, harness: Path) -> None:
    """Descriptor provenance, compared with the declaration rather than the image.

    Rebuilding a schema from an image and checking it against that same image
    establishes only that the image is internally consistent, which the
    image-building suite owns. What no single stage can answer is whether the
    recorder, the finalizer or the image builder rewrote a descriptor
    *consistently* -- a feature set pointing at the wrong source stream, a unit
    id renumbered, a rate carried as its reciprocal. So the schema recovered
    from the image is compared with the :class:`StreamSchema` the recording
    was configured with.
    """
    declared = native_schema()
    session = _session(tmp_path, 4)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    result = subprocess.run(
        [str(harness), "--schema", str(image)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    recovered = json.loads(result.stdout.splitlines()[0])
    assert recovered["status"] == "ok", recovered
    assert recovered["schema_id"] == declared.id

    assert len(recovered["signals"]) == len(declared.signals)
    for signal, source in zip(recovered["signals"], declared.signals, strict=True):
        assert signal["id"] == source.id
        assert signal["dtype"] == int(source.dtype)
        assert signal["layout"] == int(source.layout)
        assert signal["kind"] == int(source.kind)
        assert signal["n_channels"] == source.n_channels
        assert signal["nominal_block_samples"] == source.nominal_block_samples
        assert signal["max_block_samples"] == source.max_block_samples
        assert signal["rate_numerator"] == source.fs.numerator
        assert signal["rate_denominator"] == source.fs.denominator
        assert signal["clock_domain"] == source.clock_domain
        assert signal["device_tick_tracking"] == int(source.device_tick_tracking)
        assert signal["observation_timing"] == int(source.observation_timing)
        assert signal["feature_set_id"] == source.feature_set_id

    assert len(recovered["feature_sets"]) == len(declared.feature_sets)
    for feature, source in zip(recovered["feature_sets"], declared.feature_sets, strict=True):
        assert feature["id"] == source.id
        # The source linkage is the field a renumbering would break silently:
        # a descriptor pointing at the wrong stream still validates.
        assert feature["source_stream_id"] == source.source_stream_id
        assert feature["source_stream"] == source.source_stream
        assert feature["algorithm_name"] == source.algorithm_name
        assert feature["algorithm_version"] == source.algorithm_version
        assert feature["window_length_ns"] == source.window_length_ns
        assert feature["shift_ns"] == source.shift_ns
        assert feature["feature_names"] == list(source.feature_names)
        assert feature["unit_ids"] == list(source.unit_ids)

    assert len(recovered["units"]) == len(declared.units)
    for unit, source in zip(recovered["units"], declared.units, strict=True):
        assert unit["id"] == source.id
        assert unit["symbol"] == source.symbol
        assert unit["description"] == source.description


def test_replayed_gap_carries_ledger_ticks_and_flags(tmp_path: Path, harness: Path) -> None:
    """Gap provenance, compared across stages rather than within one.

    The source never sees the discontinuity: the runtime's continuity checker
    forms it, the recorder writes it, the finalizer stores it in the session's
    gap ledger. So the oracle here is the *session*, read through
    :class:`NrfReader`, and the replayed gap is compared with the ledger row --
    every field of it, including the device ticks and the flags that say which
    of them are meaningful.
    """
    recorded = _record(tmp_path, 6, sequence_gap_at=3)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    with NrfReader.open(session) as reader:
        rows = reader.read_records("native-signal-gaps-v1")
        ledger = [
            {name: rows[name][idx] for name in rows}
            for idx in range(len(rows["signal_gap_ordinal"]))
        ]
    assert ledger, "the recording wrote no gap row, so this comparison would be empty"

    run = _replay(harness, image)
    replayed = [gap for message in run.discontinuities for gap in message["gaps"]]
    assert len(replayed) == len(ledger)
    for gap, row in zip(replayed, ledger, strict=True):
        assert gap["signal_id"] == row["native_signal_id"]
        assert gap["reason"] == row["reason"]
        assert gap["expected_sample_idx"] == row["expected_sample_index"]
        assert gap["actual_sample_idx"] == row["actual_sample_index"]
        assert gap["expected_device_tick"] == row["expected_device_tick"]
        assert gap["actual_device_tick"] == row["actual_device_tick"]
        assert gap["flags"] == row["gap_flags"]


# --- clocks, and frames that are not all the same size -----------------------------


def _two_clock_domain_schema() -> Any:
    """The three-signal schema with the feature stream on its own clock domain."""
    signals = native_schema().signals
    rebuilt = [
        streaming.SignalSchema(
            signal.id,
            signal.dtype,
            signal.n_channels,
            signal.nominal_block_samples,
            signal.max_block_samples,
            streaming.RationalRate(signal.fs.numerator, signal.fs.denominator),
            signal.clock_domain if signal.id != BANDPOWER else SECOND_CLOCK_DOMAIN,
            layout=signal.layout,
            kind=signal.kind,
            feature_set_id=signal.feature_set_id,
            device_tick_tracking=signal.device_tick_tracking,
            observation_timing=signal.observation_timing,
        )
        for signal in signals
    ]
    declared = native_schema()
    return streaming.StreamSchema(
        declared.id, rebuilt, list(declared.feature_sets), list(declared.units)
    )


#: A second device clock, distinct from the first and from the host domain.
SECOND_CLOCK_DOMAIN = 8


def test_two_clock_domains_replay_under_their_own(tmp_path: Path, harness: Path) -> None:
    """Two clocks in one session, and neither block borrows the other's.

    A chain that stored one clock domain per session rather than per block would
    pass every single-domain test in this file. Here the feature stream runs on
    its own device clock, and the domain of every replayed block is compared
    with the one the source stamped.
    """
    schema = _two_clock_domain_schema()
    recorded = _record(tmp_path, 6, schema=schema)
    assert_manifest_states_the_formula(recorded.manifest)
    domains = {
        block["signal_id"]: block["clock_sync_clock_domain"]
        for block in recorded.manifest[0]["blocks"]
    }
    assert domains[BANDPOWER] == SECOND_CLOCK_DOMAIN
    assert domains[NEURAL] != domains[BANDPOWER]

    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image)
    assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest))
    for frame in run.frames:
        for block in frame["blocks"]:
            assert block["clock_sync_clock_domain"] == domains[block["signal_id"]]


def test_varying_frame_sizes_survive_chain(tmp_path: Path, harness: Path) -> None:
    """Every frame the same size is the easy case, and not the promised one.

    A chain that assumed a fixed block size -- a stride computed once, a bound
    taken from the first frame -- would pass every other test in this file. Here
    each signal's block size walks its declared range, no two signals change
    together, and the sample sequence per signal still has to come back
    contiguous and in order.
    """
    recorded = _record(tmp_path, 8, vary_block_samples=True)
    assert_manifest_states_the_formula(recorded.manifest)
    sizes = {
        block["signal_id"]: {frame["blocks"][idx]["n_samples"] for frame in recorded.manifest}
        for idx, block in enumerate(recorded.manifest[0]["blocks"])
    }
    assert any(len(values) > 1 for values in sizes.values()), (
        "the source produced one block size after all, so this test would be the fixed case"
    )

    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()

    run = _replay(harness, image)
    assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest))
    # Contiguity per signal, stated over the replay rather than the manifest:
    # varying sizes are exactly where an off-by-one stride would show.
    for signal in (NEURAL, CURSOR, BANDPOWER):
        cursor = 0
        for frame in run.frames:
            for block in frame["blocks"]:
                if block["signal_id"] != signal:
                    continue
                assert block["sample_idx_start"] == cursor
                cursor += block["n_samples"]


_KILL_SCRIPT = """
import os
import signal
import sys
from pathlib import Path

from neurale.recording import _finalizer, finalize_spool

spool_path, target = sys.argv[1], sys.argv[2]


def die(*arguments, **keywords):
    # SIGKILL, so nothing unwinds and nothing gets a chance to tidy up: what the
    # next process finds is whatever was already on disk.
    os.kill(os.getpid(), signal.SIGKILL)


_finalizer._cross_check = die
finalize_spool(Path(spool_path).read_bytes(), target)
"""


def _kill_during_finalization(tmp_path: Path, recorded: Recorded, target: Path) -> Path:
    """Finalize a real recording in a subprocess that is killed part-way."""
    import signal as signal_module

    spool = tmp_path / "killed.spool"
    spool.write_bytes(recorded.spool)
    script = tmp_path / "kill.py"
    script.write_text(_KILL_SCRIPT, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), str(spool), str(target)],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=180,
        check=False,
    )
    assert result.returncode == -signal_module.SIGKILL, result.stderr.decode()
    return spool


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGKILL"), reason="SIGKILL is POSIX-only")
def test_killed_finalization_resumes_replayable(tmp_path: Path, harness: Path) -> None:
    """A real abrupt termination, and the whole chain carried on across it.

    The recovery suite establishes that a killed finalization leaves a
    resumable attempt. What it cannot establish is that resuming produces a
    session a *replay* then reproduces sample for sample: that question spans
    the finalizer, the image builder and the native source. So the process
    really is killed, a second process resumes the attempt, and the replay of
    what it published is compared with the manifest the recording's source
    wrote.
    """
    recorded = _record(tmp_path, 12)
    target = tmp_path / "session.nrf"
    spool = _kill_during_finalization(tmp_path, recorded, target)

    state = diagnose_finalization(target)
    assert not state.published
    assert state.resumable

    resumed = resume_finalization(spool.read_bytes(), target)
    assert resumed.progress.attempts == 2
    with NrfReader.open(target) as reader:
        assert reader.completeness.complete is True

    image = tmp_path / "resumed.nrimg"
    build_replay_image(target, ReplayConfig(mode="exact_frames"), path=image).close()
    run = _replay(harness, image)
    assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest))


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGKILL"), reason="SIGKILL is POSIX-only")
def test_abandoned_finalization_leaves_spool_replayable(tmp_path: Path, harness: Path) -> None:
    """Abandonment is a decision about an attempt, not about the recording.

    After a killed finalization is abandoned, the operator's remaining option is
    to finalize the same spool again from nothing. The session that produces has
    to replay as the recording, or abandonment would be a one-way door.
    """
    recorded = _record(tmp_path, 8)
    target = tmp_path / "session.nrf"
    spool = _kill_during_finalization(tmp_path, recorded, target)

    abandon_finalization(target, reason="operator gave up on this attempt")
    assert diagnose_finalization(target).status == "abandoned"

    fresh = tmp_path / "fresh.nrf"
    finalize_spool(spool.read_bytes(), fresh)
    image = tmp_path / "fresh.nrimg"
    build_replay_image(fresh, ReplayConfig(mode="exact_frames"), path=image).close()
    run = _replay(harness, image)
    assert_frames_are_the_recorded_ones(run, oracle_frames(recorded.manifest))


# --- images that are not the image ---------------------------------------------------


def _bytes_with(path: Path, offset: int, value: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] = value
    path.write_bytes(bytes(data))


def test_corrupt_or_truncated_image_is_rejected(tmp_path: Path, harness: Path) -> None:
    """Nothing downstream of a damaged image is allowed to look like data.

    Three ways to damage one -- a flipped payload byte, a truncated file, and a
    file that is not an image at all -- and one required answer to all three: a
    named refusal before a single message is emitted.
    """
    session = _session(tmp_path, 6)
    good = tmp_path / "good.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=good).close()
    intact = good.read_bytes()

    flipped = tmp_path / "flipped.nrimg"
    flipped.write_bytes(intact)
    _bytes_with(flipped, len(intact) - 24, intact[len(intact) - 24] ^ 0xFF)

    truncated = tmp_path / "truncated.nrimg"
    truncated.write_bytes(intact[: len(intact) // 2])

    nonsense = tmp_path / "nonsense.nrimg"
    nonsense.write_bytes(b"not an image at all" * 64)

    for damaged in (flipped, truncated, nonsense):
        result = subprocess.run(
            [str(harness), "--replay", str(damaged)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        first = json.loads(result.stdout.splitlines()[0])
        assert first["status"] != "ok", f"{damaged.name} was replayed as though it were intact"
        assert result.returncode != 0, damaged.name
        assert len(result.stdout.splitlines()) == 1, (
            f"{damaged.name} emitted messages after refusing"
        )


# --- resources, and what a critical fault leaves behind -------------------------------


def test_chain_returns_threads_and_file_handles(tmp_path: Path, harness: Path) -> None:
    """The native half measures its own threads, handles, frames, and gaps.

    Measuring the Python parent cannot see a ``std::thread`` or handle leaked
    inside a child. The re-record harness therefore samples OS resources in the
    same native process before constructing the source/runtime/recorder chain
    and after all three have been destroyed.
    """

    recorded = _record(tmp_path, 6)
    session = tmp_path / "session.nrf"
    finalize_spool(recorded.spool, session)
    image = tmp_path / "exact.nrimg"
    build_replay_image(session, ReplayConfig(mode="exact_frames"), path=image).close()
    plan = compile_recording_plan(recorder_config(tmp_path / "rerecorded.nrf"), native_schema())
    plan_path = tmp_path / "resource-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.document()))
    report = _rerecord(harness, image, plan_path, tmp_path / "resource.spool")

    assert recorded.outstanding_frames == 0
    assert recorded.outstanding_discontinuities == 0
    assert recorded.runtime_state.name.lower() == "stopped"
    _assert_native_resources_released(report)
    leftovers = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and (path.name.endswith(".tmp") or ".staging" in path.name)
    ]
    assert not leftovers, f"the chain left {leftovers} behind"


def test_spool_exhaustion_aborts_and_accounts(
    tmp_path: Path,
) -> None:
    """A critical recorder fault, taken all the way to its consequences.

    The recorder here is attached to a live runtime, and its store is too small
    for the session -- the shape a full disk has. A lossless-until-fault
    recorder may not drop the frame it cannot write, so it faults; a critical
    observer's fault aborts the runtime; and what the session then has to say is
    an *accounting*: how many frames the runtime accepted, how many the recorder
    took, how many reached the spool. A recorder that faulted and then reported
    having committed everything would be the worst of the failure modes, because
    nothing downstream could tell.
    """
    recorded = _record(
        tmp_path,
        4096,
        options={"spool_capacity_bytes": 96 * 1024},
        expect_ok=False,
    )
    status = recorded.status

    assert status.primary_fault.present, "the undersized store never faulted"
    assert status.effective_session_outcome.name.lower() == "faulted"
    assert status.spool_committed <= status.recorder_accepted <= status.runtime_accepted
    assert status.spool_committed >= 1, "nothing at all was committed, so there is no prefix"

    # The fault did not stay inside the recorder: a critical observer's failure
    # is the runtime's failure, and the runtime has to say so and let go of
    # every frame and discontinuity it was holding.
    assert recorded.runtime_fault is not None, (
        "the recorder faulted and the runtime carried on as though nothing had happened"
    )
    assert recorded.runtime_state.name.lower() == "failed"
    assert recorded.outstanding_frames == 0
    assert recorded.outstanding_discontinuities == 0

    # The committed prefix is still a session, and it still says it is short.
    session = tmp_path / "faulted.nrf"
    report = finalize_spool(recorded.spool, session)
    assert report.termination_kind == "faulted"
    assert report.counts.frames == status.spool_committed
    with NrfReader.open(session) as reader:
        assert reader.completeness.complete is False
        assert len(reader.read_records("faults-v1")["code"]) >= 1


def test_stop_and_close_are_idempotent(
    tmp_path: Path,
) -> None:
    """Repeated stop and close, at the level a caller reaches them.

    These are the operations an error path takes, which is where they are least
    likely to have been exercised: a second stop or close raising would turn a
    recoverable failure into a crash inside the handler for it.
    """
    schema = native_schema()
    config = _realtime_config(8, blocks=3, payload_bytes=MULTI_MAX_FRAME_PAYLOAD_BYTES)
    source = streaming.SyntheticNativeSource(schema, 8, 1, None, None, payload_pattern=True)
    runner = recording_runner(schema, config, source)
    recorder = NativeSessionRecorder.create(
        recorder_config(tmp_path / "unused.nrf", streams=stream_specs(), checkpoint_interval=0),
        schema,
        options=NativeRecorderOptions(spool="memory", edge_capacity=64),
    )
    recorder.prepare()
    recorder.attach(runner)
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.start() == streaming.StreamStatus.OK
    runner.join()
    recorder.stop("done")
    snapshot = recorder._recorder.spool_snapshot()

    # Every teardown call an error handler might reach, twice, in the order it
    # would reach them.
    runner.stop()
    runner.stop()
    recorder.close()
    recorder.close()
    runner.close()
    runner.close()
    assert runner.outstanding_frames == 0
    assert runner.outstanding_discontinuities == 0

    # And the session the run produced is unaffected by any of it.
    session = tmp_path / "session.nrf"
    assert finalize_spool(snapshot, session).termination_kind == "normal"


# --- damaged artifacts, at each stage of the chain -------------------------------------


def _damaged_spools(recorded: Recorded) -> dict[str, bytes]:
    """One real spool, damaged four ways a device or a crash damages one."""
    torn = bytearray(_with_torn_tail(_unsealed(recorded.spool)))
    superblock = bytearray(recorded.spool)
    superblock[24] ^= 0xFF
    body = bytearray(recorded.spool)
    body[len(body) // 2] ^= 0xFF
    return {
        "torn_tail": bytes(torn),
        "broken_superblock": bytes(superblock),
        "corrupt_body": bytes(body),
        "empty": b"",
    }


def test_damaged_spools_get_verdicts(tmp_path: Path, harness: Path) -> None:
    """Four damaged spools, one rule: whatever survives has to be the recording.

    The shapes differ -- a torn tail, a broken superblock, a flipped byte in the
    committed body, an empty file -- and so do the verdicts: repair, quarantine,
    discard. What must not differ is the consequence. Whatever a damaged spool
    is allowed to yield has to be a **prefix of the recording**, replayed sample
    for sample against the source's manifest; what it may never yield is data
    the session never had, or a normal-looking session. Reporting the verdict
    belongs to the recovery suite; carrying it through finalization, image
    building and a native replay is this suite's.
    """
    recorded = _record(tmp_path, 8)
    verdicts: dict[str, str] = {}
    for label, data in _damaged_spools(recorded).items():
        path = tmp_path / f"{label}.spool"
        path.write_bytes(data)
        diagnosis = diagnose_spool(path)
        verdicts[label] = diagnosis.recommended_action

        if diagnosis.recommended_action == ACTION_QUARANTINE:
            # Unaccountable from its first byte: refused outright, and nothing
            # published from it.
            with pytest.raises(RecorderError):
                finalize_spool(data, tmp_path / f"{label}.nrf")
            assert not (tmp_path / f"{label}.nrf").exists()
            continue
        if diagnosis.recommended_action == ACTION_REPAIR:
            assert repair_spool(path, quarantine_dir=tmp_path / f"q-{label}").repaired
            data = path.read_bytes()

        session = tmp_path / f"{label}.nrf"
        report = finalize_spool(data, session)
        if diagnosis.recommended_action != ACTION_FINALIZE:
            assert report.termination_kind != "normal", f"{label} was published as a normal session"
        if report.counts.frames == 0:
            # A damaged spool is allowed to account for nothing. It is not
            # allowed to account for something that was never recorded, which
            # is what the branch below checks.
            continue

        image = tmp_path / f"{label}.nrimg"
        build_replay_image(
            session, ReplayConfig(mode="exact_frames", allow_incomplete=True), path=image
        ).close()
        run = _replay(harness, image)
        assert_frames_are_the_recorded_ones(
            run, oracle_frames(recorded.manifest, count=len(run.frames))
        )

    # The four shapes must not have collapsed into one verdict, or this test
    # would be the torn-tail test written four times.
    assert len(set(verdicts.values())) > 1, verdicts


def test_altered_session_bytes_get_no_image(
    tmp_path: Path,
) -> None:
    """The image builder reads a session; it does not get to trust one.

    A published NRF that was altered afterwards -- a bad block, a partial
    restore, an edit -- must not compile into a replay image that would then
    look exactly as authoritative as any other.
    """
    session = _session(tmp_path, 6)
    data = bytearray(session.read_bytes())
    data[len(data) // 2] ^= 0xFF
    session.write_bytes(bytes(data))

    with pytest.raises((RecorderError, ValueError, OSError)):
        build_replay_image(
            session, ReplayConfig(mode="exact_frames"), path=tmp_path / "altered.nrimg"
        )
