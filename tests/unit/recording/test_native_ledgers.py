#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The five native replay ledgers, checked against the specification.

The ledgers are declared in two independent places on purpose: the extension
specification under ``specifications/nrf/v1/extensions/native-replay-v1``, whose
README tables are normative and whose fixtures are produced by a tool that
imports nothing from ``neurale``; and :mod:`neurale.io.nrf._ledgers`, which
builds the descriptors a writer will register. A test that only checked the
implementation against itself would prove nothing, so every assertion here
crosses that boundary.

Old sessions get their own section. The extension widens an enum and adds an
optional manifest member; an early session predates all of it and must still
validate, still read, and still say -- without guessing -- that it cannot be
replayed exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from neurale.io.nrf import (
    LEDGER_KINDS,
    LEDGERS,
    NATIVE_REPLAY_EXTENSION_VERSION,
    NATIVE_REPLAY_MINOR_VERSION,
    NATIVE_REPLAY_NAMESPACE,
    declares_native_replay,
    ledger_record_schema,
    ledger_record_schemas,
    ledger_schema_ids,
    ledger_target_paths,
)
from neurale.io.nrf._ledgers import FORBIDDEN_ACCOUNTING_FIELD, SESSION_ACCOUNTING
from neurale.io.nrf._schemas import MANIFEST_SCHEMA, validate_document
from neurale.io.nrf._schemas import schema as nrf_schema
from neurale.io.nrf._semantics import validate_manifest

REPOSITORY = Path(__file__).resolve().parents[3]
EXTENSION = REPOSITORY / "specifications" / "nrf" / "v1" / "extensions" / "native-replay-v1"
FIXTURES = EXTENSION / "fixtures"
EXTENSION_SCHEMA = json.loads((EXTENSION / "native-replay.schema.json").read_text("utf-8"))

SESSION_CLOCK = "session.clock"

jsonschema = pytest.importorskip("jsonschema", reason="the JSON Schema layer needs jsonschema")


def _validator(pointer: str) -> Any:
    """Return a validator for one ``$defs`` entry of the extension schema."""
    node: Any = EXTENSION_SCHEMA
    for step in pointer.split("/"):
        node = node[step]
    return jsonschema.Draft202012Validator({**node, "$defs": EXTENSION_SCHEMA["$defs"]})


# --- the implementation against the specification ------------------------


@pytest.mark.parametrize("ledger", LEDGERS, ids=lambda ledger: ledger.kind)
def test_ledger_descriptor_matches_specification_fixture(ledger: Any) -> None:
    """Byte-for-value equality with the fixture the specification tool wrote."""
    fixture = json.loads((FIXTURES / f"{ledger.schema_id}.json").read_text("utf-8"))

    built = ledger_record_schema(ledger, clock_id=SESSION_CLOCK)

    assert built == fixture


@pytest.mark.parametrize("ledger", LEDGERS, ids=lambda ledger: ledger.kind)
def test_ledger_descriptor_satisfies_extension_schema(ledger: Any) -> None:
    """Field names, dtypes, nullability, order, and paths, all machine-checked."""
    validator = _validator(f"$defs/ledgerRecordSchemas/{ledger.kind}")

    validator.validate(ledger_record_schema(ledger, clock_id=SESSION_CLOCK))


@pytest.mark.parametrize("ledger", LEDGERS, ids=lambda ledger: ledger.kind)
def test_ledger_descriptor_is_valid_record_schema(ledger: Any) -> None:
    """The ledgers are ordinary NRF record sets, not a parallel format."""
    record_schema = ledger_record_schema(ledger, clock_id=SESSION_CLOCK)
    definition = nrf_schema(MANIFEST_SCHEMA)["$defs"]["recordSchema"]
    validator = jsonschema.Draft202012Validator(
        {**definition, "$defs": nrf_schema(MANIFEST_SCHEMA)["$defs"]}
    )

    validator.validate(record_schema)

    assert record_schema["path"] == f"records/{ledger.kind}/{ledger.schema_id}"
    assert record_schema["primary_key"] in ledger.field_names


def test_generated_specification_artifacts_are_current() -> None:
    """The README tables are the source; a field added there and missed by the
    schema or the fixtures would be a specification that disagrees with itself."""
    result = subprocess.run(
        [sys.executable, str(EXTENSION / "tools" / "generate_fixtures.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ledger_set_is_all_or_nothing() -> None:
    """Four ledgers is not a reduced capability; it is an unanswerable question."""
    complete = [{"kind": kind} for kind in LEDGER_KINDS]

    assert declares_native_replay(complete)
    assert not declares_native_replay(complete[:-1])
    assert not declares_native_replay([])


def test_accounting_summary_stores_no_verdict() -> None:
    """``accounting_verified`` is derived by the reader answering the call. A
    stored copy would let an artifact assert a verdict its reader disagrees
    with, and there would be no rule saying which one a caller sees."""
    assert FORBIDDEN_ACCOUNTING_FIELD not in SESSION_ACCOUNTING.field_names

    fixture = json.loads((FIXTURES / "session-accounting-v1.json").read_text("utf-8"))

    assert all(field["name"] != FORBIDDEN_ACCOUNTING_FIELD for field in fixture["fields"])


def test_spool_adjacent_counters_are_nullable() -> None:
    """A recording path with no spool writes null rather than a synthesized
    stage-3 number. Dropping a stage is legal; inventing one is not."""
    nullable = {field.name for field in SESSION_ACCOUNTING.fields if field.nullable}

    assert {
        "spool_committed",
        "lost_between_recorder_and_spool",
        "control_spool_committed",
        "lost_between_control_acceptance_and_spool",
    } <= nullable


def test_first_failed_position_keeps_forms_apart() -> None:
    """An item refused before acceptance never received an ordinal, so the two
    forms are separate nullable fields rather than one reused number."""
    names = SESSION_ACCOUNTING.field_names

    for field in (
        "data_first_lost_ordinal",
        "data_first_rejected_message_kind",
        "data_first_rejected_frame_sequence",
        "control_first_lost_ordinal",
        "control_first_rejected_kind",
        "control_first_rejected_identity",
    ):
        assert field in names


# --- fixtures ------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["manifest-extension.full-coverage.json", "manifest-extension.partial-coverage.json"],
)
def test_manifest_extension_fixtures_satisfy_schema(name: str) -> None:
    _validator("$defs/manifestExtension").validate(json.loads((FIXTURES / name).read_text("utf-8")))


def test_journal_extension_fixture_satisfies_schema() -> None:
    fixture = json.loads((FIXTURES / "journal-termination-extension.json").read_text("utf-8"))

    _validator("$defs/journalTerminationExtension").validate(fixture)

    assert fixture["extension_version"] == NATIVE_REPLAY_EXTENSION_VERSION


def test_unavailable_capability_states_reason() -> None:
    """A capability that is false without a reason tells a caller nothing about
    which mode to use instead."""
    validator = _validator("$defs/manifestExtension")
    fixture = json.loads((FIXTURES / "manifest-extension.partial-coverage.json").read_text("utf-8"))
    fixture["replay_capabilities"]["exact_frames"]["reason"] = None

    with pytest.raises(jsonschema.ValidationError):
        validator.validate(fixture)


def test_available_capability_carries_no_reason() -> None:
    validator = _validator("$defs/manifestExtension")
    fixture = json.loads((FIXTURES / "manifest-extension.full-coverage.json").read_text("utf-8"))
    fixture["replay_capabilities"]["exact_frames"]["reason"] = "why not"

    with pytest.raises(jsonschema.ValidationError):
        validator.validate(fixture)


# --- a manifest that declares the ledgers --------------------------------


def _session_manifest(recorder: Any, session_path: Path) -> dict[str, Any]:
    recorder.stop()
    from neurale.io.nrf import NrfReader

    with NrfReader.open(session_path) as reader:
        return dict(reader.manifest)


def test_ledger_paths_are_unique_and_canonical() -> None:
    paths = ledger_target_paths()

    assert len(set(paths)) == len(paths)
    assert set(paths) == {
        f"records/{kind}/{schema_id}" for kind, schema_id in ledger_schema_ids().items()
    }


def test_manifest_with_ledgers_passes_both_layers(
    recorder: Any, session_path: Path, schema: Any
) -> None:
    """The ledgers are added to a real session manifest, and both NRF layers --
    JSON Schema and the semantic rules -- accept the result. This is what
    "backward-compatible manifest extension" has to mean to be worth claiming.

    The plan is compiled from the *same* configuration the recorder wrote the
    manifest from. That is not a convenience: the semantic layer now cross-checks
    every fact the extension and the core manifest both state, so a plan built
    from a different configuration is a contradictory session and is supposed to
    be rejected. Grafting one on would test the wrong thing.
    """
    from neurale.recording import compile_recording_plan

    from .conftest import recorder_config

    manifest = _session_manifest(recorder, session_path)
    plan = compile_recording_plan(recorder_config(session_path), schema)
    plan = replace(
        plan,
        session=replace(
            plan.session,
            session_id=manifest["session"]["id"],
            created_at=manifest["session"]["created_at"],
            writer_name=manifest["writer"]["name"],
            writer_version=manifest["writer"]["version"],
        ),
    )

    manifest["version"] = {"major": 1, "minor": plan.nrf_minor_version}
    manifest["record_schemas"] = [
        *manifest["record_schemas"],
        *ledger_record_schemas(clock_id=SESSION_CLOCK),
    ]
    manifest["extensions"] = {**manifest.get("extensions", {}), **plan.manifest_extensions()}

    validate_document(manifest, MANIFEST_SCHEMA)
    validate_manifest(manifest)

    assert declares_native_replay(manifest["record_schemas"])
    _validator("$defs/manifestExtension").validate(manifest["extensions"][NATIVE_REPLAY_NAMESPACE])


def test_manifest_schema_accepts_new_record_kinds() -> None:
    """The enum widening is what lets the ledgers be ordinary record sets. If it
    is missing from the installed schema, this fails rather than the writer."""
    kinds = set(nrf_schema(MANIFEST_SCHEMA)["$defs"]["recordSchema"]["properties"]["kind"]["enum"])

    assert set(LEDGER_KINDS) <= kinds


def test_session_with_ledgers_declares_new_minor_version() -> None:
    """A v1.0 reader must reject such a session: the ledgers hold committed data,
    so they are an unknown required feature rather than an ignorable addition."""
    assert NATIVE_REPLAY_MINOR_VERSION == 1


# --- old sessions --------------------------------------------------------


def test_legacy_session_manifest_still_validates(recorder: Any, session_path: Path) -> None:
    """The extension widens an enum and adds an optional member. A session
    written before any of it exists must be untouched by all of it."""
    manifest = _session_manifest(recorder, session_path)

    validate_document(manifest, MANIFEST_SCHEMA)
    validate_manifest(manifest)

    assert manifest["version"] == {"major": 1, "minor": 0}
    assert NATIVE_REPLAY_NAMESPACE not in manifest.get("extensions", {})
    assert not declares_native_replay(manifest["record_schemas"])


def test_legacy_session_claims_no_replay_capability(recorder: Any, session_path: Path) -> None:
    """Exact replay needs the ledgers, and an early session has none. The answer
    is a refusal, not a silent downgrade to a synthesized replay."""
    manifest = _session_manifest(recorder, session_path)

    assert not declares_native_replay(manifest["record_schemas"])
    # The per-stream block index is what such a session *does* have, and it is
    # the one thing that makes stream_frames available for it.
    assert any(entry["kind"] == "discontinuities" for entry in manifest["record_schemas"])
