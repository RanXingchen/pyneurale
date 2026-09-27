#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared fixtures for the NRF v1 test modules.

Every NRF test needs the same two things: the specification's normative test
vectors, and a session whose manifest declares enough to be valid. Restating
either one per module is how the two drift apart, so both live here.

This module is imported by name rather than through ``conftest.py`` because the
session builder is needed at module scope -- ``test_nrf_corruption_recovery``
damages a freshly written session in each fixture, and the payload constants
below are what its assertions compare against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurale.io.nrf import ManifestBuilder, NrfWriter, record_field
from neurale.io.nrf._paths import record_set_path, stream_discontinuities_path

#: Stated once so that moving this directory breaks in one place rather than in
#: every module that wants to reach the checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SPEC_DIR = REPOSITORY_ROOT / "specifications" / "nrf" / "v1"
VECTORS: dict[str, Any] = json.loads((SPEC_DIR / "test-vectors.json").read_text(encoding="utf-8"))

SESSION_ID = "018f4f30-6f9d-7b36-8c61-3f96cf5c5a40"
CREATED_AT = "2026-08-01T00:00:00Z"

#: The payloads :func:`build_writer` sessions are filled with. They are shared so
#: that a test which damages a session can state what a healthy read returns.
NEURAL = np.arange(1, 21, dtype="int16").reshape(10, 2)
CURSOR = np.arange(6, dtype="float32").reshape(3, 2)
CURSOR_TIMES = np.array([0, 10_000_000, 20_000_000], dtype="int64")
BANDPOWER = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")

#: Every record kind NRF v1 requires a session to declare, as ``kind ->
#: (schema_id, primary_key, chunk_length, fields)``. Each field is
#: ``(name, dtype, nullable, reference)``.
_Field = tuple[str, str, bool, str | None]
REQUIRED_RECORD_KINDS: dict[str, tuple[str, str, int, list[_Field]]] = {
    "events": (
        "events-v1",
        "event_id",
        2,
        [
            ("event_id", "utf8", False, None),
            ("time_ns", "int64", False, None),
            ("duration_ns", "int64", False, None),
            ("label", "utf8", True, None),
        ],
    ),
    "trials": (
        "trials-v1",
        "trial_id",
        2,
        [
            ("trial_id", "utf8", False, None),
            ("start_ns", "int64", False, None),
            ("stop_ns", "int64", False, None),
            ("label", "utf8", True, None),
        ],
    ),
    "experiment_state": (
        "experiment-state-v1",
        "state_transition_id",
        2,
        [
            ("state_transition_id", "utf8", False, None),
            ("time_ns", "int64", False, None),
            ("new_state", "utf8", False, None),
        ],
    ),
    "commands": (
        "commands-v1",
        "command_id",
        2,
        [
            ("command_id", "utf8", False, None),
            ("time_ns", "int64", False, None),
            ("command_type", "utf8", False, None),
        ],
    ),
    "faults": (
        "faults-v1",
        "fault_id",
        2,
        [
            ("fault_id", "utf8", False, None),
            ("time_ns", "int64", False, None),
            ("code", "utf8", False, None),
        ],
    ),
    "session_termination": (
        "termination-v1",
        "termination_id",
        1,
        [
            ("termination_id", "utf8", False, None),
            ("time_ns", "int64", False, None),
            ("termination_kind", "utf8", False, None),
            ("reason", "utf8", False, None),
            ("fault_id", "utf8", True, "fault"),
            ("last_transaction_id", "utf8", False, "transaction"),
        ],
    ),
}

#: The columns every per-stream discontinuity record set declares.
_DISCONTINUITY_FIELDS = (("discontinuity_id", "utf8"), ("time_ns", "int64"), ("reason", "utf8"))


def register_records(registry: ManifestBuilder) -> None:
    """Register every record kind NRF v1 requires a session to declare."""
    for kind, (schema_id, primary_key, chunk_length, fields) in REQUIRED_RECORD_KINDS.items():
        path = record_set_path(kind, schema_id)
        registry.register_record_schema(
            schema_id,
            kind=kind,
            primary_key=primary_key,
            clock_id="host-clock",
            chunk_length=chunk_length,
            fields=[
                record_field(path, name, dtype, nullable=nullable, reference=reference)
                for name, dtype, nullable, reference in fields
            ],
        )


def register_discontinuities(registry: ManifestBuilder, stream_id: str) -> str:
    """Register the discontinuity record schema one stream needs, and name it."""
    schema_id = f"discontinuities-{stream_id}-v1"
    path = stream_discontinuities_path(stream_id)
    registry.register_record_schema(
        schema_id,
        kind="discontinuities",
        stream_id=stream_id,
        primary_key="discontinuity_id",
        clock_id="host-clock",
        chunk_length=1,
        fields=[record_field(path, name, dtype) for name, dtype in _DISCONTINUITY_FIELDS],
    )
    return schema_id


def build_writer(root: Path) -> NrfWriter:
    """Create a session with a neural stream, a cursor stream, and features."""
    writer = NrfWriter.create(
        root,
        session_id=SESSION_ID,
        created_at=CREATED_AT,
        session_metadata={"subject": {"id": "subject-01"}},
    )
    registry = writer.registry
    registry.register_clock(
        "host-clock",
        clock_type="host_monotonic",
        rate={"numerator": 1_000_000_000, "denominator": 1},
        synchronization_domain="session-domain",
    )
    registry.register_unit("volt", symbol="V", description="volt")
    registry.register_unit("volt-squared", symbol="V^2", description="volt squared")
    registry.register_channel("channel-0000", idx=0, unit_id="volt", electrode_id="grid-a")
    registry.register_channel("channel-0001", idx=1, unit_id="volt", electrode_id="grid-a")
    registry.register_electrode("grid-a", channel_ids=["channel-0000", "channel-0001"])
    registry.register_schema("schema-source", stream_ids=["neural", "cursor", "bandpower"])
    register_records(registry)
    for stream_id in ("neural", "cursor", "bandpower"):
        register_discontinuities(registry, stream_id)

    registry.register_stream(
        "neural",
        kind="neural",
        dtype="int16",
        channel_ids=["channel-0000", "channel-0001"],
        unit_ids=["volt", "volt"],
        clock_id="host-clock",
        schema_id="schema-source",
        chunk_length=4,
        capacity=64,
        discontinuity_record_schema_id="discontinuities-neural-v1",
        rate={"numerator": 4000, "denominator": 1},
    )
    registry.register_stream(
        "cursor",
        kind="behavioral",
        dtype="float32",
        channel_ids=[],
        unit_ids=["volt", "volt"],
        clock_id="host-clock",
        schema_id="schema-source",
        chunk_length=2,
        capacity=32,
        discontinuity_record_schema_id="discontinuities-cursor-v1",
        timing_mode="explicit",
    )
    registry.register_feature_set(
        "bandpower-70-200",
        feature_names=["70-200:E1", "70-200:E2"],
        unit_ids=["volt-squared", "volt-squared"],
        source_stream_id="neural",
        algorithm_name="multitaper_bandpower",
        algorithm_version="1",
        window_length_ns=100_000_000,
        shift_ns=10_000_000,
    )
    registry.register_stream(
        "bandpower",
        kind="feature",
        dtype="float64",
        channel_ids=[],
        unit_ids=["volt-squared", "volt-squared"],
        clock_id="host-clock",
        schema_id="schema-source",
        chunk_length=1,
        capacity=16,
        discontinuity_record_schema_id="discontinuities-bandpower-v1",
        rate={"numerator": 100, "denominator": 1},
        source_stream_ids=["neural"],
        feature_set_id="bandpower-70-200",
    )
    writer.freeze()
    return writer


#: The record columns every session written by :func:`build_writer` carries.
EVENT_ROWS: dict[str, list[Any]] = {
    "event_id": ["event-0001", "event-0002"],
    "time_ns": [10_000_000, 20_000_000],
    "duration_ns": [0, 5_000_000],
    "label": ["go", None],
}


def first_file(root: Path, relative: str) -> Path:
    """Return one existing file under *relative*, in sorted order."""
    return next(path for path in sorted((root / relative).rglob("*")) if path.is_file())
