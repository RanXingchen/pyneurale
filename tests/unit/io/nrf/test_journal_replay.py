#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Journal storage and replay for NRF v1 sessions.

The strongest checks here do not compare replay against itself: they require
that replaying the normative journal reproduces the manifest's commit cache and
the specification's expected recovery report. Both are values the specification
fixed independently of this implementation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from neurale.io.nrf._canonical import canonical_json_bytes, sign_record
from neurale.io.nrf._errors import NrfSemanticError
from neurale.io.nrf._journal import (
    JournalTail,
    append_records,
    build_checkpoint,
    build_commit,
    build_prepare,
    build_termination,
    extent_transition,
    prepared_object,
    read_journal,
)
from neurale.io.nrf._replay import CommittedState, replay, require_cache_does_not_lead

from .nrf_support import VECTORS

_SEQUENCE: list[dict[str, Any]] = VECTORS["complete_sequence"]
_TAIL_VECTOR = VECTORS["recoverable_uncommitted_tail"]


# --- replay against independently fixed expectations ----------------------


def test_replay_reproduces_manifest_commit_cache() -> None:
    state = replay(_SEQUENCE)
    commit = VECTORS["manifest"]["commit"]
    assert state.committed_extents == commit["committed_extents"]
    assert state.sealed_targets == set(commit["sealed_targets"])
    assert state.last_transaction_id == commit["last_transaction_id"]
    assert state.last_checkpoint_id == commit["last_checkpoint_id"]


def test_replay_records_normal_termination() -> None:
    state = replay(_SEQUENCE)
    assert state.terminated
    assert state.termination is not None
    assert state.termination.kind == "normal"
    assert state.termination.fault_id is None
    assert state.termination.last_transaction_id == state.last_transaction_id


def test_replay_matches_recovery_report_for_uncommitted_tail() -> None:
    report = _TAIL_VECTOR["expected_report"]
    state = replay(_TAIL_VECTOR["journal_records"])
    assert state.committed_extents == report["committed_extents"]
    assert state.last_transaction_id == report["last_committed_transaction_id"]
    assert not state.terminated


def test_prepare_without_commit_is_invisible() -> None:
    records = _TAIL_VECTOR["journal_records"]
    assert records[-1]["kind"] == "prepare"
    dangling = records[-1]["transaction_id"]
    committed = replay(records)
    without = replay(records[:-1])
    # The dangling prepare changes nothing a reader can see.
    assert committed.committed_extents == without.committed_extents
    assert committed.last_transaction_id != dangling


def test_head_cache_may_lag_but_never_lead() -> None:
    state = replay(_SEQUENCE)
    head = VECTORS["head"]
    require_cache_does_not_lead(
        cache_name="head.json",
        cache_sequence=head["journal_sequence"],
        cache_extents=head["committed_extents"],
        state=state,
    )
    # Lagging is fine.
    require_cache_does_not_lead(
        cache_name="head.json", cache_sequence=1, cache_extents={}, state=state
    )
    with pytest.raises(NrfSemanticError, match="leads replay"):
        require_cache_does_not_lead(
            cache_name="head.json",
            cache_sequence=state.journal_sequence + 1,
            cache_extents={},
            state=state,
        )


def test_cache_extent_beyond_replay_is_rejected() -> None:
    state = replay(_SEQUENCE)
    with pytest.raises(NrfSemanticError, match="leads replayed committed state"):
        require_cache_does_not_lead(
            cache_name="manifest.json",
            cache_sequence=state.journal_sequence,
            cache_extents={"streams/neural/data": 99},
            state=state,
        )


# --- replay rejects malformed journals ------------------------------------


def _mutated(mutate) -> list[dict[str, Any]]:
    records = deepcopy(_SEQUENCE)
    mutate(records)
    return records


def test_replay_rejects_sequence_gap() -> None:
    records = _mutated(lambda r: r.__setitem__(2, {**r[2], "sequence": 99}))
    with pytest.raises(NrfSemanticError, match="journal sequence"):
        replay(records)


def test_replay_rejects_commit_without_prepare() -> None:
    records = _mutated(lambda r: r.pop(0))
    with pytest.raises(NrfSemanticError):
        replay(records)


def test_replay_rejects_double_commit() -> None:
    records = deepcopy(_SEQUENCE[:2])
    duplicate = dict(records[1], sequence=3)
    with pytest.raises(NrfSemanticError, match="commits without an earlier prepare"):
        replay([*records, duplicate])


def test_replay_rejects_prepare_out_of_order() -> None:
    records = deepcopy(_SEQUENCE[:3])
    records[2] = {**records[2], "previous_committed_transaction_id": None}
    with pytest.raises(NrfSemanticError, match="does not follow the last committed transaction"):
        replay(records)


def test_replay_rejects_record_after_termination() -> None:
    records = deepcopy(_SEQUENCE)
    trailing = dict(records[-1], sequence=records[-1]["sequence"] + 1)
    with pytest.raises(NrfSemanticError, match="may follow a termination"):
        replay([*records, trailing])


def test_replay_rejects_unassigned_prepared_object() -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        prepare = records[0]
        prepare["extents"] = [
            dict(transition, object_paths=[]) for transition in prepare["extents"]
        ]

    with pytest.raises(NrfSemanticError, match="unassigned objects"):
        replay(_mutated(mutate))


def test_replay_rejects_doubly_claimed_object() -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        prepare = records[0]
        first = prepare["extents"][0]
        prepare["extents"] = [first, dict(first, target_path="streams/cursor/data")]

    with pytest.raises(NrfSemanticError, match="claimed by multiple transitions"):
        replay(_mutated(mutate))


def test_replay_rejects_underived_staged_path() -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        records[0]["objects"][0]["staged_path"] = ".staging/elsewhere/object"

    with pytest.raises(NrfSemanticError, match="not derived from the final path"):
        replay(_mutated(mutate))


def test_replay_rejects_advancing_transition_without_objects() -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        prepare = records[0]
        transition = prepare["extents"][0]
        prepare["objects"] = [
            entry for entry in prepare["objects"] if entry["path"] not in transition["object_paths"]
        ]
        prepare["extents"] = [dict(transition, object_paths=[]), *prepare["extents"][1:]]

    with pytest.raises(NrfSemanticError, match="backed by no staged object"):
        replay(_mutated(mutate))


def test_replay_rejects_incomplete_chunk_coverage() -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        prepare = records[0]
        transition = prepare["extents"][0]
        dropped = next(
            entry
            for entry in prepare["objects"]
            if entry["path"] in transition["object_paths"]
            and entry["logical_role"] == "array_chunk"
        )
        prepare["objects"] = [e for e in prepare["objects"] if e is not dropped]
        transition["object_paths"] = [
            path for path in transition["object_paths"] if path != dropped["path"]
        ]

    with pytest.raises(NrfSemanticError, match="chunk indices do not cover"):
        replay(_mutated(mutate))


def test_replay_rejects_append_to_sealed_target() -> None:
    state = replay(_SEQUENCE[:-1])
    assert "streams/neural/data" in state.sealed_targets
    prepare = build_prepare(
        sequence=state.journal_sequence + 1,
        transaction_id="tx-0000000000000099",
        previous_committed_transaction_id=state.last_transaction_id,
        extents=[
            extent_transition(
                target_path="streams/neural/data",
                target_kind="array",
                before=state.extent("streams/neural/data"),
                after=state.extent("streams/neural/data") + 4,
                chunk_length=4,
                required_array_paths=["streams/neural/data"],
                object_paths=[],
                seals_target=False,
            )
        ],
        objects=[],
    )
    with pytest.raises(NrfSemanticError, match="sealed and cannot be appended"):
        replay([*_SEQUENCE[:-1], prepare])


# --- journal storage ------------------------------------------------------


def test_journal_round_trip(tmp_path: Path) -> None:
    journal = tmp_path / "journal" / "transactions.jsonl"
    records = [
        build_prepare(
            sequence=1,
            transaction_id="tx-0000000000000001",
            previous_committed_transaction_id=None,
            extents=[
                extent_transition(
                    target_path="records/events/events-v1",
                    target_kind="record_set",
                    before=0,
                    after=0,
                    chunk_length=4,
                    required_array_paths=["records/events/events-v1/columns/event_id"],
                    object_paths=[],
                    seals_target=True,
                    record_schema_id="events-v1",
                )
            ],
            objects=[],
        ),
        build_commit(sequence=2, transaction_id="tx-0000000000000001", prepare_sequence=1),
    ]
    append_records(journal, records)
    loaded, tail = read_journal(journal)
    assert loaded == records
    assert tail.clean


def test_partial_final_line_is_invisible(tmp_path: Path) -> None:
    journal = tmp_path / "transactions.jsonl"
    record = build_commit(sequence=2, transaction_id="tx-0000000000000001", prepare_sequence=1)
    append_records(journal, [record])
    # Simulate a crash part-way through the next record.
    with journal.open("ab") as handle:
        handle.write(canonical_json_bytes({"kind": "commit", "sequence": 2})[:20])

    loaded, tail = read_journal(journal)
    assert loaded == [record]
    assert tail.partial_line_bytes > 0
    assert not tail.clean


def test_invalid_checksum_truncates_readable_journal(tmp_path: Path) -> None:
    journal = tmp_path / "transactions.jsonl"
    good = build_commit(sequence=2, transaction_id="tx-0000000000000001", prepare_sequence=1)
    append_records(journal, [good])
    tampered = dict(
        build_commit(sequence=4, transaction_id="tx-0000000000000002", prepare_sequence=3),
        transaction_id="tx-0000000000000003",
    )
    append_records(journal, [tampered])

    loaded, tail = read_journal(journal)
    assert loaded == [good]
    assert tail.invalid_checksum_at == 2


def test_missing_journal_reads_as_empty(tmp_path: Path) -> None:
    loaded, tail = read_journal(tmp_path / "absent.jsonl")
    assert loaded == []
    assert tail == JournalTail()


def test_appended_records_are_signed_and_verifiable(tmp_path: Path) -> None:
    journal = tmp_path / "transactions.jsonl"
    checkpoint = build_checkpoint(
        sequence=1,
        checkpoint_id="checkpoint-0000000000000001",
        checkpoint_path="journal/checkpoints/checkpoint-0000000000000001.json",
        checkpoint_sha256="0" * 64,
    )
    append_records(journal, [checkpoint])
    loaded, _ = read_journal(journal)
    assert loaded == [checkpoint]
    assert "record_checksum" in loaded[0]


# --- record construction contracts ----------------------------------------


def test_prepared_object_derives_staging_path() -> None:
    entry = prepared_object(
        path="streams/neural/data/c/0/0",
        transaction_id="tx-0000000000000001",
        target_path="streams/neural/data",
        array_path="streams/neural/data",
        logical_role="array_chunk",
        content_kind="zarr_chunk",
        sha256="0" * 64,
        byte_length=16,
        chunk_coordinate=(0, 0),
    )
    assert entry["staged_path"] == ".staging/tx-0000000000000001/streams/neural/data/c/0/0"
    assert entry["disposition"] == "create"


@pytest.mark.parametrize(
    ("kind", "fault_id"),
    [("normal", "fault-1"), ("faulted", None)],
)
def test_termination_fault_contract_is_enforced(kind: str, fault_id: str | None) -> None:
    from neurale.io.nrf._errors import NrfSchemaError

    with pytest.raises(NrfSchemaError):
        build_termination(
            sequence=1,
            termination_record_id="termination-0001",
            termination_kind=kind,
            time_ns=0,
            reason="done",
            fault_id=fault_id,
            last_transaction_id="tx-0000000000000001",
        )


def test_aborted_termination_may_omit_fault() -> None:
    record = build_termination(
        sequence=1,
        termination_record_id="termination-0001",
        termination_kind="aborted",
        time_ns=0,
        reason="operator abort",
        fault_id=None,
        last_transaction_id="tx-0000000000000001",
    )
    assert record["fault_id"] is None


def test_empty_state_starts_at_sequence_zero() -> None:
    state = CommittedState()
    assert state.journal_sequence == 0
    assert state.extent("streams/neural/data") == 0
    assert not state.terminated
    assert replay([], initial=state) is state


def test_signed_records_are_stable_under_resigning() -> None:
    record = build_commit(sequence=1, transaction_id="tx-0000000000000001", prepare_sequence=0)
    assert sign_record(record) == record
