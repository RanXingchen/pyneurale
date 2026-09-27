#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The compiled recording plan: coverage, capability, and fingerprint.

Three things are being pinned here, and they are different claims:

- what the plan *is* -- an immutable, canonically ordered statement compiled
  against the prepared schema, which rejects a stream the schema does not have;
- what a session recorded under it may be **replayed** as, which the lifecycle
  contract decides from coverage rather than from the replay request;
- what the fingerprint **identifies** -- the plan, not the session. It must move
  when what gets recorded changes and stay put when only how it is stored does,
  or it cannot answer "was this the same recording plan?".
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neurale.io.nrf import RecordingPlan, plan_from_manifest_extension
from neurale.io.nrf._canonical import canonical_json_bytes
from neurale.recording import (
    RecorderConfig,
    RecorderConfigError,
    RecorderLimits,
    RecorderSessionMetadata,
    StreamMetadata,
    StreamRecording,
    StreamTimingConfig,
    StreamTimingMode,
    compile_recording_plan,
)

from .conftest import BANDPOWER, CURSOR, NEURAL, SCHEMA_ID, stream_specs

FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "specifications"
    / "nrf"
    / "v1"
    / "extensions"
    / "native-replay-v1"
    / "fixtures"
)


def _config(tmp_path: Path, streams: list[StreamRecording], **overrides: Any) -> RecorderConfig:
    limit_values = {}
    for name in (
        "frame_queue_capacity",
        "control_queue_capacity",
        "spool_capacity_bytes",
        "checkpoint_interval",
        "max_control_records",
    ):
        if name in overrides:
            limit_values[name] = overrides.pop(name)
    if limit_values:
        overrides["limits"] = RecorderLimits(**limit_values)
    return RecorderConfig(path=tmp_path / "session.nrf", streams=streams, **overrides)


def _plan(
    tmp_path: Path, schema: Any, streams: list[StreamRecording] | None = None
) -> RecordingPlan:
    return compile_recording_plan(_config(tmp_path, streams or stream_specs()), schema)


def _replace_neural(specs: list[StreamRecording], neural: StreamRecording) -> list[StreamRecording]:
    feature = replace(
        specs[2],
        provenance=replace(specs[2].provenance, sources=(neural,)),
    )
    return [neural, specs[1], feature]


# --- compilation ---------------------------------------------------------


def test_stream_recording_public_api_contains_only_recording_intent() -> None:
    assert tuple(inspect.signature(StreamRecording).parameters) == (
        "stream_id",
        "signal",
        "metadata",
        "storage",
        "timing",
        "provenance",
    )
    with pytest.raises(RecorderConfigError, match="not a native integer"):
        StreamRecording(signal=1)


def test_recorder_config_public_api_contains_only_recording_intent(tmp_path: Path) -> None:
    assert tuple(inspect.signature(RecorderConfig).parameters) == (
        "path",
        "streams",
        "session",
        "metadata",
        "limits",
        "spool_retention",
    )
    assert tuple(inspect.signature(RecorderSessionMetadata).parameters) == (
        "subject",
        "experiment",
        "metadata",
        "extensions",
    )
    config = RecorderConfig(path=tmp_path / "session.nrf")
    assert config.streams == (StreamRecording(),)
    assert config.spool_retention == "delete_after_validated_finalization"
    assert RecorderConfig(path=config.path, spool_retention="retain").spool_retention == "retain"
    with pytest.raises(RecorderConfigError, match="spool_retention"):
        RecorderConfig(path=config.path, spool_retention="delete")
    with pytest.raises(RecorderConfigError, match="RecorderSessionMetadata"):
        RecorderConfig(path=config.path, session={"id": "not-public"})  # type: ignore[arg-type]


def test_recorder_limits_derive_nrf_control_layout() -> None:
    limits = RecorderLimits(max_control_records=1025)

    assert limits.spool_capacity_bytes is None
    assert limits.control_chunk_length == 2
    assert limits.control_capacity == 2048
    assert RecorderLimits(spool_capacity_bytes=128 << 20).spool_capacity_bytes == 128 << 20
    with pytest.raises(RecorderConfigError, match="exact-integer"):
        RecorderLimits(spool_capacity_bytes=1 << 53)


def test_plan_compiles_against_prepared_schema(tmp_path: Path, schema: Any) -> None:
    """Every recorded signal is one the schema declared, and it says which."""
    plan = _plan(tmp_path, schema)

    assert plan.native_schema_id == SCHEMA_ID
    assert plan.planned_signal_ids == (NEURAL, CURSOR, BANDPOWER)
    assert plan.recorded_signal_ids == (NEURAL, CURSOR, BANDPOWER)
    assert plan.stream_for(NEURAL) is not None
    assert plan.stream_for(4096) is None
    feature = plan.stream_for(BANDPOWER)
    assert feature is not None
    assert feature.unit == ("V^2", "V^2", "V^2")
    assert feature.feature_names == ("lfp.beta", "lfp.gamma", "lfp.hfa")
    assert feature.algorithm_name == "multitaper-bandpower"
    assert feature.algorithm_version == "1"


def test_feature_metadata_cannot_override_native_units(tmp_path: Path, schema: Any) -> None:
    specs = stream_specs()
    specs[-1] = replace(specs[-1], metadata=StreamMetadata(unit="invented"))

    with pytest.raises(RecorderConfigError, match="feature units come from"):
        _plan(tmp_path, schema, specs)


def test_undefined_stream_is_rejected(tmp_path: Path, schema: Any) -> None:
    """Refused at compile time: the manifest freezes before the first append."""
    streams = [*stream_specs(), StreamRecording(stream_id="ghost", signal=SimpleNamespace(id=99))]

    with pytest.raises(RecorderConfigError) as error:
        _plan(tmp_path, schema, streams)

    assert "99" in str(error.value)
    assert "is not declared" in str(error.value)


def test_plan_is_immutable(tmp_path: Path, schema: Any) -> None:
    plan = _plan(tmp_path, schema)

    with pytest.raises(FrozenInstanceError):
        plan.streams = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.streams[0].stream_id = "renamed"  # type: ignore[misc]
    # Deep immutability: storage bounds are frozen value objects, not dicts a
    # caller can mutate after the plan is prepared.
    with pytest.raises(FrozenInstanceError):
        plan.resource_bounds.streams[0].capacity = 7  # type: ignore[misc]


# --- coverage and capability ---------------------------------------------


def test_full_coverage_permits_exact_replay(tmp_path: Path, schema: Any) -> None:
    plan = _plan(tmp_path, schema)

    assert plan.coverage == "full"
    assert plan.capability("exact_frames").available
    assert plan.capability("exact_frames").reason is None
    assert plan.capability("recorded_projection").available


def test_partial_plan_rejects_exact_replay(tmp_path: Path, schema: Any) -> None:
    """A partial plan is not exact replay with a caveat; it is a projection."""
    partial = [spec for spec in stream_specs() if int(spec.signal.id) != BANDPOWER]

    plan = _plan(tmp_path, schema, partial)

    assert plan.coverage == "partial"
    exact = plan.capability("exact_frames")
    assert not exact.available
    assert "recorded_projection" in (exact.reason or "")
    # The projection itself stays available: that is the whole point of the mode.
    assert plan.capability("recorded_projection").available


def test_synthesized_replay_follows_block_index(tmp_path: Path, schema: Any) -> None:
    """`stream_frames` needs a committed per-stream block index, so a plan that
    turns it off everywhere cannot promise it."""
    without = stream_specs(block_index=False)

    plan = _plan(tmp_path, schema, without)

    capability = plan.capability("stream_frames")
    assert not capability.available
    assert "block index" in (capability.reason or "")
    assert _plan(tmp_path, schema).capability("stream_frames").available


def test_unknown_replay_mode_is_error(tmp_path: Path, schema: Any) -> None:
    with pytest.raises(RecorderConfigError, match="unknown replay mode"):
        _plan(tmp_path, schema).capability("exact_replay")


# --- fingerprint ---------------------------------------------------------


def test_fingerprint_is_deterministic_and_order_independent(tmp_path: Path, schema: Any) -> None:
    """Two callers who declared the same plan agree without comparing objects."""
    declared = stream_specs()
    reversed_order = list(reversed(declared))

    first = _plan(tmp_path, schema, declared)
    second = _plan(tmp_path, schema, reversed_order)

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.fingerprint == first.fingerprint.lower()
    # The feature stream's source linkage follows the descriptor's numeric
    # source_stream_id, not a declaration-order fallback, so it is stable too.
    assert first.stream_for(BANDPOWER).source_stream_ids == ("neural",)
    assert (
        first.stream_for(BANDPOWER).source_stream_ids
        == second.stream_for(BANDPOWER).source_stream_ids
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda specs: specs[:-1], id="stream_removed"),
        pytest.param(
            lambda specs: [
                *specs[:-1],
                replace(specs[-1], stream_id="x"),
            ],
            id="stream_renamed",
        ),
        pytest.param(
            lambda specs: [
                *specs[:-1],
                replace(
                    specs[-1],
                    provenance=replace(specs[-1].provenance, block_index=False),
                ),
            ],
            id="block_index_disabled",
        ),
        pytest.param(
            lambda specs: [
                *specs[:-1],
                replace(
                    specs[-1],
                    timing=StreamTimingConfig(mode=StreamTimingMode.REGULAR),
                ),
            ],
            id="timing_changed",
        ),
        pytest.param(
            lambda specs: [
                *specs[:-1],
                replace(
                    specs[-1],
                    metadata=StreamMetadata(name="renamed"),
                ),
            ],
            id="name_changed",
        ),
    ],
)
def test_fingerprint_moves_with_recorded_content(tmp_path: Path, schema: Any, mutate: Any) -> None:
    baseline = _plan(tmp_path, schema).fingerprint

    assert _plan(tmp_path, schema, mutate(stream_specs())).fingerprint != baseline


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacity", 1 << 16),
        ("chunk_length", 64),
        ("block_index_chunk_length", 32),
    ],
)
def test_fingerprint_ignores_storage_layout(
    tmp_path: Path, schema: Any, field: str, value: Any
) -> None:
    """Storage sizing changes how a session is written, not what it records."""
    baseline = _plan(tmp_path, schema).fingerprint
    specs = stream_specs()
    storage = replace(specs[-1].storage, **{field: value})
    altered = [*specs[:-1], replace(specs[-1], storage=storage)]

    assert _plan(tmp_path, schema, altered).fingerprint == baseline


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unit", ["m", "m", "m", "m"]),
        ("channel_names", ["a", "b", "c", "d"]),
    ],
)
def test_stream_metadata_moves_fingerprint(
    tmp_path: Path, schema: Any, field: str, value: Any
) -> None:
    """Unit and channel names reach the manifest, so two declarations that differ
    only in them record different things and fingerprint differently."""
    baseline = _plan(tmp_path, schema).fingerprint
    specs = stream_specs()
    neural = specs[0]
    metadata = replace(neural.metadata, **{field: value})
    altered = _replace_neural(specs, replace(neural, metadata=metadata))

    assert _plan(tmp_path, schema, altered).fingerprint != baseline


def test_regular_timing_origin_moves_fingerprint(tmp_path: Path, schema: Any) -> None:
    """``segment_start_time_ns`` anchors a regular stream's time axis, so two
    declarations that differ only in it record different things. An explicit
    stream ignores the origin, so it must not move the fingerprint."""
    specs = stream_specs()
    neural = specs[0]
    base = _replace_neural(
        specs,
        replace(
            neural,
            timing=StreamTimingConfig(mode=StreamTimingMode.REGULAR, segment_start_time_ns=1000),
        ),
    )
    shifted = _replace_neural(
        specs,
        replace(
            neural,
            timing=StreamTimingConfig(mode=StreamTimingMode.REGULAR, segment_start_time_ns=2000),
        ),
    )
    assert _plan(tmp_path, schema, base).fingerprint != _plan(tmp_path, schema, shifted).fingerprint

    with pytest.raises(RecorderConfigError, match="explicit timing"):
        StreamTimingConfig(segment_start_time_ns=1000)


def test_fingerprint_ignores_session_identity(tmp_path: Path, schema: Any) -> None:
    """Two sessions of the same plan share a fingerprint; that is what makes it
    usable to ask whether a finalization target belongs to this plan."""
    config = _config(tmp_path, stream_specs())
    first = compile_recording_plan(config, schema)
    second = compile_recording_plan(config, schema)

    assert first.fingerprint == second.fingerprint
    assert first.session.session_id != second.session.session_id


def test_fingerprint_covers_plan_document_only(tmp_path: Path, schema: Any) -> None:
    """Capabilities are derived, so they are not in the fingerprinted document.

    Including them would let a document carry both a fact and a conclusion drawn
    from it, with no rule for which wins when they disagree.
    """
    document = _plan(tmp_path, schema).document()

    assert "replay_capabilities" not in document
    assert "plan_fingerprint" not in document


# --- manifest extension --------------------------------------------------


def test_manifest_extension_round_trips_fingerprint(tmp_path: Path, schema: Any) -> None:
    """A reader with only the manifest reaches the plan the session was
    recorded under, which is what makes the stored fingerprint checkable."""
    plan = _plan(tmp_path, schema)
    extension = plan.manifest_extension()

    rebuilt = plan_from_manifest_extension(extension)

    assert rebuilt.fingerprint == plan.fingerprint == extension["plan_fingerprint"]
    assert rebuilt.manifest_extension() == extension


def test_non_default_spool_bound_keeps_fingerprint(tmp_path: Path, schema: Any) -> None:
    baseline = _plan(tmp_path, schema)
    configured = compile_recording_plan(
        _config(tmp_path, stream_specs(), spool_capacity_bytes=8 << 20), schema
    )

    extension = configured.manifest_extension()
    rebuilt = plan_from_manifest_extension(extension)

    assert extension["resource_bounds"]["spool_capacity_bytes"] == 8 << 20
    assert rebuilt.resource_bounds.spool_capacity_bytes == 8 << 20
    assert configured.fingerprint == baseline.fingerprint


def test_manifest_extension_matches_specification_fixtures() -> None:
    """The fixtures are produced by the specification's own tool, from its own
    tables, importing nothing from ``neurale``."""
    for name in ("full", "partial"):
        fixture = json.loads((FIXTURES / f"manifest-extension.{name}-coverage.json").read_text())

        rebuilt = plan_from_manifest_extension(fixture)

        assert rebuilt.manifest_extension() == fixture
        assert rebuilt.coverage == name


def test_malformed_manifest_extension_is_rejected() -> None:
    with pytest.raises(RecorderConfigError, match="not a recording plan"):
        plan_from_manifest_extension({"streams": [{"stream_id": "neural"}]})


def test_rejection_states_reason_once(tmp_path: Path, schema: Any) -> None:
    """The parser's own diagnostics are not re-wrapped on the way out.

    ``RecorderConfigError`` is a ``ValueError``, so a handler that catches
    ``ValueError`` to describe a malformed document will also catch the parser's
    considered rejections and stack a second copy of its prefix on them. What
    reaches the caller then reads as two failures rather than one, and the rule
    that actually fired is buried mid-sentence.
    """
    extension = _plan(tmp_path, schema).manifest_extension()
    extension["coverage"] = "partial"

    with pytest.raises(RecorderConfigError) as error:
        plan_from_manifest_extension(extension)

    assert str(error.value).count("is not a recording plan") == 1


def test_extension_with_unrecorded_source_is_rejected(tmp_path: Path, schema: Any) -> None:
    """A dangling source is refused by the rule, not by a stale fingerprint.

    The specification requires every explicit ``source_stream_ids`` entry to
    resolve to a recorded stream, on streams of every kind. The mutated
    extension is re-fingerprinted first, so it is internally consistent and
    the hash cannot be what rejects it -- exactly the document a forged or
    hand-edited manifest would present. The core NRF validator would also
    refuse the dangling reference eventually, but the plan is what a finalizer
    reads back before it writes anything, so it has to be refusable on its own.
    """
    plan = _plan(tmp_path, schema)
    extension = plan.manifest_extension()
    extension["streams"][0]["source_stream_ids"] = ["ghost"]
    # Derive the fingerprinted key set from the plan itself rather than
    # restating the recipe: a test that hard-codes it stops noticing when the
    # document changes.
    document = {key: extension[key] for key in plan.document()}
    extension["plan_fingerprint"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()

    with pytest.raises(RecorderConfigError, match="'ghost', which this plan does not record"):
        plan_from_manifest_extension(extension)
