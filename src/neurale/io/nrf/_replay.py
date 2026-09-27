#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Journal replay: the authoritative committed state of an NRF v1 session.

Specification reference: ``README.md`` sections 8 and 9, and
``semantic-validation.md`` rules NRF-TX-001 through NRF-TX-012.

Replay -- not ``manifest.json`` and not ``journal/head.json`` -- decides what a
reader may see. Both of those files are caches: each may lag replay, and a
cache that *leads* replay is a validation failure rather than newer truth.

Like the semantic layer, this is an implementation independent of the
specification's replay oracle in
``tests/specification/test_nrf_v1_specification.py``; the two are pinned
together by ``test_nrf_v1_package_conformance.py`` rather than by sharing code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._errors import NrfSemanticError
from ._journal import CHECKPOINT, COMMIT, PREPARE, TERMINATION
from ._paths import portable_path_key, staged_path
from ._zarr import grid_coordinates

#: Roles that carry chunk data. A stream data or timestamp array uses
#: ``array_chunk``; a record set uses one role per column and one per nullable
#: field's validity array, and all of them must advance together.
CHUNK_ROLES = frozenset({"array_chunk", "record_column_chunk", "record_validity_chunk"})

METADATA_ROLE = "metadata_snapshot"


@dataclass(frozen=True, slots=True)
class Termination:
    """The terminal state a replayed session reached."""

    record_id: str
    kind: str
    time_ns: int
    reason: str
    fault_id: str | None
    last_transaction_id: str


@dataclass
class CommittedState:
    """Everything a reader is allowed to see, derived only from the journal."""

    journal_sequence: int = 0
    last_transaction_id: str | None = None
    last_checkpoint_id: str | None = None
    #: Journal sequence of the newest commit, so an incremental replay can still
    #: enforce that termination immediately follows its commit.
    last_commit_sequence: int | None = None
    committed_extents: dict[str, int] = field(default_factory=dict)
    sealed_targets: set[str] = field(default_factory=set)
    committed_objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    termination: Termination | None = None

    @property
    def terminated(self) -> bool:
        return self.termination is not None

    def extent(self, target_path: str) -> int:
        """Return the committed extent of *target_path*, zero when untouched."""
        return self.committed_extents.get(target_path, 0)


#: The members a checkpoint records for one committed object. Optional ones are
#: carried only when the object has them: a metadata snapshot has no chunk
#: coordinate, and a stream chunk has no record schema or column.
_CHECKPOINT_OBJECT_MEMBERS = (
    "path",
    "byte_length",
    "sha256",
    "content_kind",
    "target_path",
    "array_path",
    "logical_role",
)
_CHECKPOINT_OBJECT_OPTIONAL = ("chunk_coordinate", "record_schema_id", "column_name")


def checkpoint_object(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return one committed object exactly as a checkpoint must record it.

    Both the writer that emits checkpoints and the diagnosis that judges them
    project committed state through this one function, so "the checkpoint
    equals replay" is a comparison of identical shapes rather than of two
    hand-written ideas of the same thing.
    """
    descriptor = {member: entry[member] for member in _CHECKPOINT_OBJECT_MEMBERS}
    for optional in _CHECKPOINT_OBJECT_OPTIONAL:
        if entry.get(optional) is not None:
            descriptor[optional] = entry[optional]
    return descriptor


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise NrfSemanticError(message)


def _expected_leading_indices(before: int, after: int, chunk_length: int) -> set[int]:
    """Return the leading-axis chunk indices that must back ``[before, after)``."""
    if after <= before:
        return set()
    return set(range(before // chunk_length, ((after - 1) // chunk_length) + 1))


def _check_chunk_coverage(
    *,
    array_path: str,
    coordinates: set[tuple[int, ...]],
    before: int,
    after: int,
    chunk_length: int,
    trailing_grid: Sequence[int] | None,
) -> None:
    """Enforce NRF-TX-005 chunk coverage for one physical array.

    The leading axis must be covered exactly. Full trailing coverage can only
    be *proved* against the frozen chunk grid, so when no manifest contract is
    supplied this falls back to requiring that every leading index carries the
    same trailing coordinates -- enough to catch a partially advanced array,
    but not a claim that the grid itself is right. A reader that has the
    manifest always passes the contract.
    """
    expected_leading = _expected_leading_indices(before, after, chunk_length)
    actual_leading = {coordinate[0] for coordinate in coordinates}
    _check(
        actual_leading == expected_leading,
        f"{array_path} leading chunk indices do not cover [{before}, {after}); "
        f"missing {sorted(expected_leading - actual_leading)}, "
        f"unexpected {sorted(actual_leading - expected_leading)}",
    )
    if not coordinates:
        return

    ranks = {len(coordinate) for coordinate in coordinates}
    _check(len(ranks) == 1, f"{array_path} chunk coordinates have inconsistent rank")

    if trailing_grid is not None:
        expected = {
            (idx, *trailing)
            for idx in expected_leading
            for trailing in grid_coordinates(trailing_grid)
        }
        _check(
            coordinates == expected,
            f"{array_path} chunk coordinates do not cover the frozen chunk grid; "
            f"missing {sorted(expected - coordinates)}, "
            f"unexpected {sorted(coordinates - expected)}",
        )
        return

    by_leading: dict[int, set[tuple[int, ...]]] = {}
    for coordinate in coordinates:
        by_leading.setdefault(coordinate[0], set()).add(coordinate[1:])
    trailing_sets = list(by_leading.values())
    _check(
        all(trailing == trailing_sets[0] for trailing in trailing_sets),
        f"{array_path} advances some trailing chunks but not others",
    )


def _validate_prepare(
    record: Mapping[str, Any],
    state: CommittedState,
    *,
    contracts: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    transaction_id = record["transaction_id"]
    extents = record["extents"]
    objects = record["objects"]

    _check(
        record["previous_committed_transaction_id"] == state.last_transaction_id,
        f"{transaction_id} does not follow the last committed transaction",
    )

    # NRF-TX-007: final and staged paths are independently unique within one
    # prepare, and each staged path is derived from its final path.
    final_paths = [entry["path"] for entry in objects]
    staged_paths = [entry["staged_path"] for entry in objects]
    _check(len(final_paths) == len(set(final_paths)), "duplicate final object path in prepare")
    _check(len(staged_paths) == len(set(staged_paths)), "duplicate staged object path in prepare")
    portable_finals = {portable_path_key(path) for path in final_paths}
    _check(len(portable_finals) == len(final_paths), "portable final object path collision")
    for entry in objects:
        _check(
            entry["staged_path"] == staged_path(transaction_id, entry["path"]),
            f"staged path is not derived from the final path: {entry['path']!r}",
        )
        _check(
            entry["staged_path"] != entry["path"],
            "staged path must differ from the final path",
        )
        _check(
            entry.get("disposition", "create") == "create",
            "NRF v1 has no replace or supersede disposition",
        )
        # Object paths are unique across transactions, not just within one.
        _check(
            entry["path"] not in state.committed_objects,
            f"object path was already committed: {entry['path']!r}",
        )

    # NRF-TX-001: every object belongs to exactly one transition.
    by_path = {entry["path"]: entry for entry in objects}
    assigned: dict[str, int] = {}
    for i, transition in enumerate(extents):
        for path in transition["object_paths"]:
            _check(path in by_path, f"transition names an object not in this prepare: {path!r}")
            _check(path not in assigned, f"object is claimed by multiple transitions: {path!r}")
            assigned[path] = i
    unassigned = set(by_path) - set(assigned)
    _check(not unassigned, f"prepare has unassigned objects: {sorted(unassigned)}")

    for transition in extents:
        target = transition["target_path"]
        before = transition["before"]
        after = transition["after"]
        chunk_length = transition["chunk_length"]
        required = list(transition["required_array_paths"])

        # Extents never decrease and must continue the committed extent.
        _check(after >= before, f"{target} transition decreases its extent")
        _check(
            before == state.extent(target),
            f"{target} prepare 'before' {before} does not equal the committed extent "
            f"{state.extent(target)}",
        )
        _check(target not in state.sealed_targets, f"{target} is sealed and cannot be appended")

        # NRF-TX-003: the required arrays must equal the frozen manifest
        # contract when one was supplied.
        if contracts is not None:
            contract = contracts.get(target)
            _check(contract is not None, f"{target} is not an appendable target of this session")
            assert contract is not None
            _check(
                sorted(required) == sorted(contract["required_array_paths"]),
                f"{target} required arrays disagree with the frozen manifest",
            )
            _check(
                transition.get("record_schema_id") == contract.get("record_schema_id"),
                f"{target} record schema disagrees with the frozen manifest",
            )
            _check(
                chunk_length == contract["chunk_length"],
                f"{target} chunk length disagrees with the frozen manifest",
            )

        transition_objects = [by_path[path] for path in transition["object_paths"]]
        # NRF-TX-002.
        for entry in transition_objects:
            _check(
                entry["target_path"] == target,
                f"object {entry['path']!r} names a different target than its transition",
            )
            _check(
                entry["array_path"] in required,
                f"object {entry['path']!r} names an array outside the transition contract",
            )

        if after == before:
            # NRF-TX-006 / section 8.3: only a seal-only transition may be
            # objectless, and it must actually seal.
            _check(
                transition["seals_target"],
                f"{target} transition neither advances nor seals",
            )
            _check(
                not transition_objects,
                f"{target} seal-only transition must carry no objects",
            )
            continue

        _check(
            transition_objects,
            f"{target} advances its extent but is backed by no staged object",
        )
        # Section 8.3: an open target starts chunk aligned, and stays aligned
        # unless this transition seals it with a final short chunk.
        _check(before % chunk_length == 0, f"{target} 'before' is not chunk aligned")
        if not transition["seals_target"]:
            _check(after % chunk_length == 0, f"{target} unsealed 'after' is not chunk aligned")

        # NRF-TX-005 / NRF-TX-006, per required physical array.
        for array_path in required:
            chunks = [
                entry
                for entry in transition_objects
                if entry["array_path"] == array_path and entry["logical_role"] in CHUNK_ROLES
            ]
            snapshots = [
                entry
                for entry in transition_objects
                if entry["array_path"] == array_path and entry["logical_role"] == METADATA_ROLE
            ]
            _check(
                len(snapshots) == 1,
                f"{array_path} must have exactly one metadata snapshot per advancing "
                f"transaction, found {len(snapshots)}",
            )
            _check_chunk_coverage(
                array_path=array_path,
                coordinates={tuple(entry["chunk_coordinate"]) for entry in chunks},
                before=before,
                after=after,
                chunk_length=chunk_length,
                trailing_grid=_trailing_grid(contracts, target, array_path),
            )


def _trailing_grid(
    contracts: Mapping[str, Mapping[str, Any]] | None,
    target: str,
    array_path: str,
) -> Sequence[int] | None:
    """Return the frozen trailing chunk-grid counts, or ``None`` if unknown.

    ``None`` means "no manifest contract was supplied", which weakens the
    coverage check rather than silently passing an empty grid.
    """
    if contracts is None:
        return None
    contract = contracts.get(target)
    if contract is None:
        return None
    return contract.get("trailing_chunk_grid", {}).get(array_path)


def _apply_commit(
    record: Mapping[str, Any], prepare: Mapping[str, Any], state: CommittedState
) -> None:
    for transition in prepare["extents"]:
        target = transition["target_path"]
        state.committed_extents[target] = transition["after"]
        if transition["seals_target"]:
            state.sealed_targets.add(target)
    for entry in prepare["objects"]:
        state.committed_objects[entry["path"]] = dict(entry)
    state.last_transaction_id = record["transaction_id"]


class ReplayCursor:
    """Apply journal records one at a time and keep the state they build.

    :func:`replay` validates a whole journal and raises on the first violation,
    which is what a reader needs: a journal it cannot fully validate is not one
    it may read through. Recovery needs the complementary view -- how far the
    journal *was* valid, and which record stopped it -- so both are driven from
    this one cursor rather than from two rule sets that could drift apart.

    A record that violates a rule leaves the state untouched: every check runs
    before any mutation, so a caller may report the failure and keep the state
    the previous records established.
    """

    __slots__ = ("_committed_transactions", "_contracts", "_expected_sequence", "_pending", "state")

    def __init__(
        self,
        state: CommittedState | None = None,
        *,
        contracts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.state = state if state is not None else CommittedState()
        self._contracts = contracts
        self._pending: dict[str, tuple[int, Mapping[str, Any]]] = {}
        self._committed_transactions: set[str] = set()
        self._expected_sequence = self.state.journal_sequence + 1

    @property
    def pending_prepares(self) -> dict[str, int]:
        """Transaction ID -> prepare sequence for prepares that never committed."""
        return {transaction_id: sequence for transaction_id, (sequence, _) in self._pending.items()}

    @property
    def committed_transaction_ids(self) -> list[str]:
        """Transactions whose commit record was applied, in transaction order."""
        return sorted(self._committed_transactions)

    def apply(self, record: Mapping[str, Any]) -> None:
        """Apply one journal record.

        Raises
        ------
        NrfSemanticError
            If the record violates an NRF-TX rule. Replay never repairs a
            journal, and the cursor's state is unchanged by a rejected record.
        """
        sequence = record["sequence"]
        _check(
            sequence == self._expected_sequence,
            f"journal sequence {sequence} is not the expected {self._expected_sequence}",
        )
        kind = record["kind"]
        state = self.state

        _check(
            state.termination is None,
            "no journal record may follow a termination record",
        )

        if kind == PREPARE:
            transaction_id = record["transaction_id"]
            _check(
                transaction_id not in self._committed_transactions,
                f"{transaction_id} has already committed",
            )
            _check(transaction_id not in self._pending, f"{transaction_id} was already prepared")
            _validate_prepare(record, state, contracts=self._contracts)
            self._pending[transaction_id] = (sequence, record)

        elif kind == COMMIT:
            transaction_id = record["transaction_id"]
            _check(
                transaction_id in self._pending,
                f"{transaction_id} commits without an earlier prepare",
            )
            prepare_sequence, prepare = self._pending[transaction_id]
            _check(
                record["prepare_sequence"] == prepare_sequence,
                f"{transaction_id} commit references the wrong prepare sequence",
            )
            _check(
                _transaction_number(transaction_id) > _last_committed_number(state),
                f"{transaction_id} commits out of transaction order",
            )
            del self._pending[transaction_id]
            _apply_commit(record, prepare, state)
            self._committed_transactions.add(transaction_id)
            state.last_commit_sequence = sequence

        elif kind == CHECKPOINT:
            state.last_checkpoint_id = record["checkpoint_id"]

        elif kind == TERMINATION:
            _check(
                record["last_transaction_id"] == state.last_transaction_id,
                "termination does not reference the last committed transaction",
            )
            _check(
                state.last_commit_sequence is not None
                and sequence == state.last_commit_sequence + 1,
                "termination must immediately follow the commit it references",
            )
            state.termination = Termination(
                record_id=record["termination_record_id"],
                kind=record["termination_kind"],
                time_ns=record["time_ns"],
                reason=record["reason"],
                fault_id=record["fault_id"],
                last_transaction_id=record["last_transaction_id"],
            )

        else:
            # Unreachable through ``read_journal``, which validates every record
            # against its schema first. Stated anyway so that a record kind this
            # implementation does not understand can never advance the journal
            # sequence by being silently skipped.
            _check(False, f"unknown journal record kind {kind!r}")

        self._expected_sequence = sequence + 1
        state.journal_sequence = sequence


def replay(
    records: Sequence[Mapping[str, Any]],
    *,
    contracts: Mapping[str, Mapping[str, Any]] | None = None,
    initial: CommittedState | None = None,
) -> CommittedState:
    """Replay journal *records* into the committed state they describe.

    Parameters
    ----------
    records
        Complete, checksum-valid records in journal order.
    contracts
        Frozen per-target contract from the manifest, keyed by target path.
        When omitted, only the record-internal rules are enforced; a reader
        with a manifest should always pass it so NRF-TX-003 applies.
    initial
        State to continue from, normally the state a valid checkpoint restored.

    Returns
    -------
    CommittedState
        Committed extents, sealed targets, and termination as of the last
        valid commit.

    Raises
    ------
    NrfSemanticError
        If any NRF-TX rule is violated. Replay never repairs a journal.
    """
    cursor = ReplayCursor(initial, contracts=contracts)
    for record in records:
        cursor.apply(record)
    return cursor.state


def _transaction_number(transaction_id: str) -> int:
    return int(transaction_id.removeprefix("tx-"))


def _last_committed_number(state: CommittedState) -> int:
    if state.last_transaction_id is None:
        return -1
    return _transaction_number(state.last_transaction_id)


def require_cache_does_not_lead(
    *,
    cache_name: str,
    cache_sequence: int,
    cache_extents: Mapping[str, int],
    state: CommittedState,
) -> None:
    """Reject a cache that claims more than journal replay supports.

    ``manifest.json`` and ``journal/head.json`` are independent caches replaced
    by separate atomic renames, so a crash between them can legitimately leave
    them at different sequences. Lagging is recoverable; leading is not, because
    it would expose data no commit record backs.
    """
    _check(
        cache_sequence <= state.journal_sequence,
        f"{cache_name} journal sequence {cache_sequence} leads replay {state.journal_sequence}",
    )
    for target, extent in cache_extents.items():
        _check(
            extent <= state.extent(target),
            f"{cache_name} extent for {target!r} leads replayed committed state",
        )
