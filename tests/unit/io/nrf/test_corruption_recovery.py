#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Corruption detection and explicit recovery for NRF v1 sessions.

Each fixture here damages a healthy session in exactly one way and then asks
two questions of it: what does an ordinary read do, and what does recovery do?
Those answers are the whole contract. A reader must never hand back a fill
value where a committed chunk used to be, and recovery must never quietly turn
uncommitted material into data by promoting it.

The fixtures are built by damaging a real written session rather than by
hand-assembling a broken one, so each test is about a state a crash or a bad
disk can actually produce.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neurale.io.nrf import (
    SEVERITY_CORRUPT,
    SEVERITY_INCOMPLETE,
    SEVERITY_STALE,
    STATUS_PARTIALLY_RECOVERED,
    STATUS_RECOVERED,
    STATUS_UNRECOVERABLE,
    NrfCorruptionError,
    NrfReader,
    NrfSchemaError,
    NrfSemanticError,
    NrfWriter,
    diagnose_session,
    read_report,
    recover,
)
from neurale.io.nrf.__main__ import EXIT_CORRUPT, EXIT_OK, EXIT_UNUSABLE, main
from neurale.io.nrf._canonical import canonical_json_bytes, encode_journal_line, sign_record
from neurale.io.nrf._diagnostics import Diagnostic, _diagnose_extents
from neurale.io.nrf._journal import read_journal
from neurale.io.nrf._manifest import TargetContract
from neurale.io.nrf._paths import JOURNAL_HEAD, JOURNAL_TRANSACTIONS, MANIFEST, staged_path
from neurale.io.nrf._replay import CommittedState
from neurale.io.nrf._zarr import ArraySpec

from .nrf_support import BANDPOWER, CURSOR, CURSOR_TIMES, EVENT_ROWS, NEURAL, build_writer
from .package_objects import Objects, snapshot

# --- fixtures --------------------------------------------------------------


def _first_transaction(root: Path, *, checkpoint: bool = True) -> NrfWriter:
    """Write and commit the streams, leaving the writer open."""
    writer = build_writer(root)
    writer.append_stream("neural", NEURAL)
    writer.append_stream("cursor", CURSOR, timestamps=CURSOR_TIMES)
    writer.append_stream("bandpower", BANDPOWER)
    writer.commit()
    if checkpoint:
        writer.checkpoint()
    return writer


def _healthy(root: Path, *, checkpoint: bool = True, finalize: bool = True) -> Path:
    """Write a session with two committed transactions.

    The second transaction exists so a test can drop its commit and be left
    with a prepare that never became visible -- the ordinary shape of a crash
    between promotion and the commit line.
    """
    root = Objects(root)
    writer = _first_transaction(root, checkpoint=checkpoint)
    writer.append_records("events-v1", EVENT_ROWS)
    writer.commit()
    if finalize:
        writer.finalize()
    else:
        snapshot(writer)
    writer.close()
    return root


@pytest.fixture
def session(tmp_path: Path) -> Path:
    return _healthy(tmp_path / "healthy.nrf")


#: What a session that was never finalized has committed. The trailing rows of
#: each stream do not fill a chunk, so they stay in the writer-owned tail and
#: are invisible -- which is the point: an interrupted recording must be
#: detectably short rather than silently complete.
OPEN_NEURAL = NEURAL[:8]

#: Recovery may rewrite caches. It may not touch any of this.
_IMMUTABLE_TREES = ("streams", "records", "metadata", "journal/transactions.jsonl")


def _payload_digests(root: Path) -> dict[str, str]:
    """Return a digest of every committed payload byte and the journal itself."""
    digests: dict[str, str] = {}
    for tree in _IMMUTABLE_TREES:
        target = root / tree
        if target.is_file():
            digests[tree] = hashlib.sha256(target.read_bytes()).hexdigest()
            continue
        if not target.is_dir():
            continue
        for item in sorted(target.rglob("*")):
            if not item.is_file() or item.name == "zarr.json":
                continue
            relative = item.relative_to(root).as_posix()
            digests[relative] = hashlib.sha256(item.read_bytes()).hexdigest()
    return digests


def _file_digests(root: Path) -> dict[str, str]:
    return {
        item.relative_to(root).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def _journal_records(root: Path) -> list[dict[str, Any]]:
    return read_journal(root / JOURNAL_TRANSACTIONS)[0]


def _rewrite_journal(
    root: Path, transform: Callable[[dict[str, Any]], dict[str, Any] | None]
) -> None:
    """Rewrite the journal, re-signing every record *transform* keeps.

    Re-signing matters: a fixture that damaged a record's meaning without
    fixing its checksum would be testing the checksum rule instead of the rule
    it meant to test.
    """
    updated = []
    for record in _journal_records(root):
        changed = transform(dict(record))
        if changed is not None:
            updated.append(sign_record(changed))
    (root / JOURNAL_TRANSACTIONS).write_bytes(
        b"".join(encode_journal_line(record) for record in updated)
    )


def _committed_chunk(root: Path, prefix: str) -> str:
    """Return one committed chunk path under *prefix*."""
    diagnosis = diagnose_session(root, verify_checksums=False)
    for path in sorted(diagnosis.state.committed_objects):
        if path.startswith(prefix) and "/c/" in path:
            return path
    raise AssertionError(f"no committed chunk under {prefix}")


# --- 1. malformed or truncated manifest ------------------------------------


def test_truncated_manifest_is_reported_and_never_read(session: Path) -> None:
    raw = (session / MANIFEST).read_bytes()
    (session / MANIFEST).write_bytes(raw[: len(raw) // 2])

    diagnosis = diagnose_session(session)
    assert diagnosis.has("manifest_unreadable")
    assert not diagnosis.readable
    with pytest.raises(NrfCorruptionError):
        NrfReader.open(session)


def test_missing_manifest_is_not_nrf_session(session: Path) -> None:
    (session / MANIFEST).unlink()
    assert diagnose_session(session).has("invalid_package")
    with pytest.raises(NrfCorruptionError):
        NrfReader.open(session)


# --- 2. unsupported major version ------------------------------------------


def test_unsupported_major_version_is_rejected(session: Path) -> None:
    manifest = json.loads((session / MANIFEST).read_bytes())
    manifest["version"]["major"] = 2
    (session / MANIFEST).write_bytes(canonical_json_bytes(manifest))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("unsupported_version")
    assert not diagnosis.readable
    with pytest.raises(NrfSchemaError, match="unsupported major version"):
        NrfReader.open(session)
    assert recover(session, created_at=_NOW).status == STATUS_UNRECOVERABLE


# --- 3. missing chunk -------------------------------------------------------


def test_missing_committed_chunk_is_corruption(session: Path) -> None:
    # Zarr answers a missing chunk with the fill value. Returning that as data
    # would hand a caller a block of zeros where samples used to be.
    path = _committed_chunk(session, "streams/neural/data")
    (session / path).unlink()

    diagnosis = diagnose_session(session)
    assert path in diagnosis.missing_objects
    assert diagnosis.has("object_missing")
    assert not diagnosis.readable

    with NrfReader.open(session) as reader:
        with pytest.raises(NrfCorruptionError, match="missing"):
            reader.read_stream("neural")


def test_missing_chunk_leaves_other_streams_readable(session: Path) -> None:
    (session / _committed_chunk(session, "streams/neural/data")).unlink()
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("bandpower"), BANDPOWER)


# --- 4. truncated chunk -----------------------------------------------------


def test_truncated_committed_chunk_is_rejected(session: Path) -> None:
    path = _committed_chunk(session, "streams/neural/data")
    absolute = session / path
    absolute.write_bytes(absolute.read_bytes()[:-4])

    diagnosis = diagnose_session(session)
    assert path in diagnosis.corrupt_objects
    assert diagnosis.has("object_truncated")

    with NrfReader.open(session) as reader:
        with pytest.raises(NrfCorruptionError, match=r"truncated|bytes, the commit records"):
            reader.read_stream("neural")


# --- 5. bad checksum --------------------------------------------------------


def test_in_place_chunk_edit_fails_checksum(session: Path) -> None:
    # Same byte length, different bytes: only the checksum can catch this, so
    # it is the one case a cheap existence check is not enough for.
    path = _committed_chunk(session, "streams/neural/data")
    absolute = session / path
    raw = bytearray(absolute.read_bytes())
    raw[0] ^= 0xFF
    absolute.write_bytes(bytes(raw))

    diagnosis = diagnose_session(session)
    assert path in diagnosis.corrupt_objects
    assert diagnosis.has("object_checksum_invalid")

    with pytest.raises(NrfCorruptionError, match="checksum"):
        NrfReader.open(session, verify_checksums=True)


def test_bad_journal_checksum_caps_session(session: Path) -> None:
    lines = (session / JOURNAL_TRANSACTIONS).read_bytes().splitlines(keepends=True)
    damaged = json.loads(lines[-1])
    damaged["record_checksum"] = "0" * 64
    lines[-1] = canonical_json_bytes(damaged) + b"\n"
    (session / JOURNAL_TRANSACTIONS).write_bytes(b"".join(lines))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("journal_record_checksum_invalid")
    assert not diagnosis.readable
    assert any(item.reason == "bad_record_checksum" for item in diagnosis.ignored_records)


@pytest.mark.parametrize(
    ("kind", "drop"),
    [
        ("prepare", "extents"),
        ("commit", "prepare_sequence"),
        ("checkpoint", "checkpoint_path"),
        ("termination", "termination_kind"),
    ],
)
def test_signed_non_record_stops_replay(session: Path, kind: str, drop: str) -> None:
    """A valid checksum proves the bytes were not edited, not that they mean anything.

    So structure is checked too, before replay ever indexes a record. Without
    that, a signed record missing a required member reaches replay and fails
    there as a raw ``KeyError`` -- which would escape a diagnosis that promises
    never to raise.
    """
    _rewrite_journal(
        session,
        lambda record: (
            {key: value for key, value in record.items() if key != drop}
            if record["kind"] == kind
            else record
        ),
    )

    diagnosis = diagnose_session(session)
    assert diagnosis.has("journal_record_schema_invalid")
    assert drop in diagnosis.find("journal_record_schema_invalid")[0].message
    assert not diagnosis.readable
    assert any(item.reason == "malformed_order" for item in diagnosis.ignored_records)
    # Whatever the damaged record would have made visible is simply not there.
    assert kind not in {record["kind"] for record in _journal_records(session)}


def test_unknown_record_kind_is_schema_failure(session: Path) -> None:
    # A v1 reader must reject an unknown record kind. Which rule rejected it
    # matters: reporting a checksum failure for a record whose checksum is
    # perfectly good sends whoever reads the report after the wrong evidence.
    _rewrite_journal(
        session,
        lambda record: {**record, "kind": "rollback"} if record["kind"] == "commit" else record,
    )

    diagnosis = diagnose_session(session)
    assert diagnosis.has("journal_record_schema_invalid")
    assert not diagnosis.has("journal_record_checksum_invalid")


def test_schema_invalid_record_hides_later_data(session: Path) -> None:
    # Damage the very first prepare. Replay reaches nothing, so nothing is
    # visible -- and the manifest cache, which still describes the whole
    # session, now leads replay and must be refused rather than believed.
    _rewrite_journal(
        session,
        lambda record: (
            {key: value for key, value in record.items() if key != "extents"}
            if record["kind"] == "prepare"
            else record
        ),
    )

    diagnosis = diagnose_session(session)
    assert diagnosis.has("journal_record_schema_invalid")
    assert diagnosis.has("cache_leads_replay")
    assert diagnosis.state.extent("streams/neural/data") == 0
    with pytest.raises(NrfSemanticError, match="leads replay"):
        NrfReader.open(session)


# --- 6. incomplete and contradictory transactions ---------------------------


@pytest.fixture
def uncommitted_prepare(tmp_path: Path) -> Path:
    """A session that crashed between promoting objects and writing the commit.

    The append protocol stages, promotes, then commits, then refreshes caches.
    So the state a crash at that point leaves behind is: final objects on disk,
    a prepare in the journal, no commit line, and caches still describing the
    *previous* transaction. Rewinding the caches too is what makes this a state
    a conforming writer can actually produce rather than an impossible one.
    """
    root = Objects(tmp_path / "prepared.nrf")
    writer = _first_transaction(root)
    manifest_before = (root / MANIFEST).read_bytes()
    head_before = (root / JOURNAL_HEAD).read_bytes()

    writer.append_records("events-v1", EVENT_ROWS)
    writer.commit()
    writer.close()

    journal = root / JOURNAL_TRANSACTIONS
    lines = journal.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[-1])["kind"] == "commit"
    journal.write_bytes(b"".join(lines[:-1]))
    (root / MANIFEST).write_bytes(manifest_before)
    (root / JOURNAL_HEAD).write_bytes(head_before)
    snapshot(writer)
    return root


def test_uncommitted_prepare_stays_invisible(
    uncommitted_prepare: Path,
) -> None:
    diagnosis = diagnose_session(uncommitted_prepare)

    assert diagnosis.has("uncommitted_prepare")
    assert any(item.reason == "uncommitted_prepare" for item in diagnosis.ignored_records)
    # The chunks were already renamed into place, but no commit line backs them.
    assert diagnosis.orphan_objects
    assert diagnosis.readable  # nothing committed is damaged
    assert not diagnosis.complete

    with NrfReader.open(uncommitted_prepare) as reader:
        assert reader.record_extent("events-v1") == 0
        assert not reader.complete


def test_recovery_never_promotes_uncommitted_prepare(uncommitted_prepare: Path) -> None:
    before = diagnose_session(uncommitted_prepare).state.committed_extents
    result = recover(uncommitted_prepare, created_at=_NOW)

    assert result.report["ignored_transaction_ids"]
    assert result.report["committed_extents"] == dict(before)
    with NrfReader.open(uncommitted_prepare) as reader:
        assert reader.record_extent("events-v1") == 0


def test_duplicate_commit_record_stops_replay(session: Path) -> None:
    raw = (session / JOURNAL_TRANSACTIONS).read_bytes()
    lines = raw.splitlines(keepends=True)
    commit = next(line for line in reversed(lines) if json.loads(line)["kind"] == "commit")
    (session / JOURNAL_TRANSACTIONS).write_bytes(raw + commit)

    diagnosis = diagnose_session(session)
    assert diagnosis.has("journal_replay_failed")
    assert any(item.reason == "malformed_order" for item in diagnosis.ignored_records)
    assert not diagnosis.readable
    # The reader refuses the whole journal rather than replaying part of it.
    with pytest.raises(NrfSemanticError, match="journal sequence"):
        NrfReader.open(session)


def test_prepare_with_vanished_objects_is_ignored(
    uncommitted_prepare: Path,
) -> None:
    # "Missing temporary chunk": the prepare named staged objects that no
    # longer exist anywhere. Nothing committed depends on them, so the only
    # correct outcome is that the transaction is ignored -- not that the
    # session is declared corrupt.
    diagnosis = diagnose_session(uncommitted_prepare)
    for relative in diagnosis.orphan_objects:
        (uncommitted_prepare / relative).unlink()
    assert not [item for item in (uncommitted_prepare / ".staging").rglob("*") if item.is_file()]

    after = diagnose_session(uncommitted_prepare)
    assert after.has("uncommitted_prepare")
    assert after.missing_objects == ()
    assert after.corrupt_objects == ()
    assert after.readable


# --- 7. checkpoints ---------------------------------------------------------


def _reseal_checkpoint(root: Path, relative: str, document: dict[str, Any]) -> None:
    """Re-sign an edited checkpoint and re-point the journal record at it.

    Without this a fixture would only be testing the file checksum, which is a
    different rule from "the checkpoint equals replay".
    """
    raw = canonical_json_bytes(sign_record(document))
    (root / relative).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    _rewrite_journal(
        root,
        lambda record: (
            {**record, "checkpoint_sha256": digest} if record["kind"] == "checkpoint" else record
        ),
    )


def test_valid_checkpoint_is_selected_and_replayed_forward(session: Path) -> None:
    diagnosis = diagnose_session(session)
    assert diagnosis.base_checkpoint_id == "checkpoint-0000000000000001"
    # The checkpoint was taken after the first transaction; the extents come
    # from replaying every commit after it as well.
    assert len(diagnosis.retained_transaction_ids) == 3
    assert diagnosis.state.extent("records/events/events-v1") == 2

    report = recover(session, dry_run=True, created_at=_NOW).report
    assert report["base_checkpoint_id"] == "checkpoint-0000000000000001"
    assert report["last_committed_transaction_id"] == diagnosis.state.last_transaction_id


def test_missing_checkpoint_falls_back_without_loss(session: Path) -> None:
    checkpoint = session / "journal/checkpoints/checkpoint-0000000000000001.json"
    expected = diagnose_session(session).state.committed_extents
    checkpoint.unlink()

    diagnosis = diagnose_session(session)
    assert diagnosis.has("checkpoint_missing")
    assert diagnosis.base_checkpoint_id is None
    assert diagnosis.readable  # a checkpoint is an optimization, not data
    assert diagnosis.state.committed_extents == expected


def test_checkpoint_disagreeing_with_replay_is_stale(session: Path) -> None:
    relative = "journal/checkpoints/checkpoint-0000000000000001.json"
    document = json.loads((session / relative).read_bytes())
    target = next(iter(document["committed_extents"]))
    document["committed_extents"][target] += 4
    _reseal_checkpoint(session, relative, document)

    diagnosis = diagnose_session(session)
    assert diagnosis.has("checkpoint_stale")
    assert diagnosis.base_checkpoint_id is None
    assert diagnosis.readable  # committed data is untouched


def test_checkpoint_with_edited_digest_is_stale(session: Path) -> None:
    # The object set is right and the extents are right; only one object's
    # recorded digest is wrong. A checkpoint is a summary of committed state,
    # so a summary that misdescribes an object is not the newest valid one.
    relative = "journal/checkpoints/checkpoint-0000000000000001.json"
    document = json.loads((session / relative).read_bytes())
    document["committed_objects"][0]["sha256"] = "0" * 64
    _reseal_checkpoint(session, relative, document)

    diagnosis = diagnose_session(session)
    assert diagnosis.has("checkpoint_stale")
    assert "sha256" in diagnosis.find("checkpoint_stale")[0].message
    assert diagnosis.base_checkpoint_id is None
    assert diagnosis.readable  # the objects themselves are intact


def test_checkpoint_claiming_index_extents_is_stale(session: Path) -> None:
    # No v1 journal record can advance an index position, so replay can never
    # reach a non-empty index_extents and a checkpoint must not claim one.
    relative = "journal/checkpoints/checkpoint-0000000000000001.json"
    document = json.loads((session / relative).read_bytes())
    document["index_extents"] = {"indexes/spike-index/part-0000": 4}
    _reseal_checkpoint(session, relative, document)

    diagnosis = diagnose_session(session)
    assert diagnosis.has("checkpoint_stale")
    assert "index extents" in diagnosis.find("checkpoint_stale")[0].message
    assert diagnosis.base_checkpoint_id is None


def test_malformed_checkpoint_is_rejected(session: Path) -> None:
    relative = "journal/checkpoints/checkpoint-0000000000000001.json"
    (session / relative).write_bytes(b"{ not json")

    diagnosis = diagnose_session(session)
    assert diagnosis.has("checkpoint_checksum_mismatch")
    assert diagnosis.base_checkpoint_id is None


# --- 8. committed extents beyond valid data ---------------------------------


def test_manifest_cache_leading_replay_is_rejected(session: Path) -> None:
    manifest = json.loads((session / MANIFEST).read_bytes())
    target = "streams/neural/data"
    manifest["commit"]["committed_extents"][target] += 4
    for stream in manifest["streams"]:
        if stream["data"]["path"] == target:
            stream["committed_extent"] += 4
    (session / MANIFEST).write_bytes(canonical_json_bytes(manifest))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("cache_leads_replay")
    assert not diagnosis.readable
    with pytest.raises(NrfSemanticError, match="leads replayed committed state"):
        NrfReader.open(session)


#: One frozen neural array of eight rows, stated the way the manifest layer
#: hands it to the extent diagnosis.
_NEURAL_CONTRACT = TargetContract(
    target_path="streams/neural/data",
    target_kind="array",
    chunk_length=4,
    arrays=(
        ArraySpec(
            path="streams/neural/data",
            shape=(8, 2),
            chunk_shape=(4, 2),
            dtype="int16",
            endianness="little",
            codec_ids=("bytes-le",),
        ),
    ),
    capacity=8,
)


def _extent_diagnostics(state: CommittedState) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    _diagnose_extents({"streams/neural/data": _NEURAL_CONTRACT}, state, diagnostics)
    return diagnostics


def test_extent_beyond_frozen_array_is_reported() -> None:
    # The manifest's own semantic layer makes this unreachable through file
    # damage -- it requires committed_extent <= shape[0] -- so the check is
    # exercised directly. It exists because journal replay alone never bounds
    # an extent by the array that has to hold it.
    diagnostics = _extent_diagnostics(CommittedState(committed_extents={"streams/neural/data": 12}))

    assert [item.code for item in diagnostics] == ["extent_exceeds_capacity"]
    assert diagnostics[0].severity == SEVERITY_CORRUPT


def test_unbacked_extent_is_reported() -> None:
    diagnostics = _extent_diagnostics(
        CommittedState(
            committed_extents={"streams/neural/data": 8},
            committed_objects={
                "streams/neural/data/c/0/0": {"sha256": "0" * 64, "byte_length": 16}
            },
        )
    )

    assert [item.code for item in diagnostics] == ["extent_not_backed_by_objects"]


# --- 9. Zarr metadata -------------------------------------------------------


def test_missing_zarr_metadata_is_rebuilt(session: Path) -> None:
    metadata = session / "streams/neural/data/zarr.json"
    metadata.unlink()

    diagnosis = diagnose_session(session)
    assert diagnosis.has("zarr_metadata_stale")
    assert "streams/neural/data" in diagnosis.stale_metadata_arrays
    assert diagnosis.readable  # the cache is rebuildable; no committed byte is lost
    with NrfReader.open(session) as reader:
        with pytest.raises(NrfCorruptionError, match=r"active zarr\.json"):
            reader.read_stream("neural")

    recover(session, created_at=_NOW)
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)


def test_zarr_metadata_disagreeing_with_manifest_is_rejected(session: Path) -> None:
    metadata = session / "streams/neural/data/zarr.json"
    document = json.loads(metadata.read_bytes())
    document["shape"] = [999, 2]
    metadata.write_bytes(json.dumps(document).encode("utf-8"))

    assert diagnose_session(session).has("zarr_metadata_stale")
    with NrfReader.open(session) as reader:
        with pytest.raises(NrfCorruptionError, match="declares shape"):
            reader.read_stream("neural")

    recover(session, created_at=_NOW)
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)


def _replace(**members: Any) -> Callable[[dict[str, Any]], None]:
    """Return an edit that overwrites top-level members of the metadata."""

    def edit(document: dict[str, Any]) -> None:
        document.update(members)

    return edit


def _edit_endianness(document: dict[str, Any]) -> None:
    for codec in document["codecs"]:
        if codec["name"] == "bytes":
            codec["configuration"]["endian"] = "big"


@pytest.mark.parametrize(
    ("member", "edit"),
    [
        # The bytes on disk are int16; decoding them as int32 halves the sample
        # count and reinterprets every pair of samples as one value.
        ("data_type", _replace(data_type="int32")),
        (
            "codecs",
            _replace(
                codecs=[
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
                ]
            ),
        ),
        ("codecs", _edit_endianness),
        # Zarr would look for streams/neural/data/c/0.0, find nothing, and hand
        # back the fill value for every committed chunk.
        (
            "chunk_key_encoding",
            _replace(chunk_key_encoding={"name": "default", "configuration": {"separator": "."}}),
        ),
        ("node_type", _replace(node_type="group")),
        ("fill_value", _replace(fill_value=7)),
    ],
    ids=["dtype", "codec", "endianness", "chunk-key-separator", "node-type", "fill-value"],
)
def test_decoding_members_of_metadata_are_checked(
    session: Path, member: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    """A member that changes decoding must be caught even when the shape agrees.

    Each of these edits leaves shape and chunk shape untouched, so a check that
    compared only those would let Zarr reinterpret committed bytes -- or look
    for chunks at paths that do not exist and answer with fill values.
    """
    metadata = session / "streams/neural/data/zarr.json"
    document = json.loads(metadata.read_bytes())
    edit(document)
    metadata.write_bytes(json.dumps(document).encode("utf-8"))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("zarr_metadata_stale")
    assert member in diagnosis.find("zarr_metadata_stale")[0].message
    with NrfReader.open(session) as reader:
        with pytest.raises(NrfCorruptionError, match=f"declares {member}"):
            reader.read_stream("neural")

    recover(session, created_at=_NOW)
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)


def test_equivalent_metadata_spelling_is_accepted(session: Path) -> None:
    """A legal alternative spelling must not read as corruption.

    Zarr v3 lets a named object omit an empty ``configuration``, and
    ``attributes`` is an annotation no decoder consults. A conforming session
    written by another implementation must open rather than be quarantined --
    and the leniency stops there, at exactly what zarr-python itself accepts.
    """
    metadata = next(
        path
        for path in sorted(session.rglob("zarr.json"))
        if json.loads(path.read_bytes()).get("data_type") == "string"
    )
    document = json.loads(metadata.read_bytes())
    assert any(codec.get("configuration") == {} for codec in document["codecs"])
    document["codecs"] = [
        {key: value for key, value in codec.items() if value != {}} for codec in document["codecs"]
    ]
    document["attributes"] = {"written_by": "some other implementation"}
    metadata.write_bytes(json.dumps(document).encode("utf-8"))

    assert not diagnose_session(session).has("zarr_metadata_stale")
    with NrfReader.open(session) as reader:
        assert reader.read_records("events-v1")["label"][0] == "go"


# --- 10. index caches -------------------------------------------------------

IDX_PATH = "indexes/spike-index/part-0000"


def _commit_an_index(root: Path, payload: bytes) -> None:
    """Commit one index object through the session's last real transaction.

    NRF v1 lets a transaction write into ``indexes/`` with the ``index``
    logical role, which makes the result ordinary committed data rather than a
    derived cache. This writer has no API for that, so the fixture adds the
    object to the last prepare and re-signs the journal -- which is what a
    writer that did have one would have produced.
    """
    (root / IDX_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / IDX_PATH).write_bytes(payload)
    last = max(
        record["sequence"] for record in _journal_records(root) if record["kind"] == "prepare"
    )

    def carry_the_index(record: dict[str, Any]) -> dict[str, Any]:
        if record["kind"] != "prepare" or record["sequence"] != last:
            return record
        transition, *rest = record["extents"]
        entry = {
            "path": IDX_PATH,
            "staged_path": staged_path(record["transaction_id"], IDX_PATH),
            "target_path": transition["target_path"],
            "array_path": transition["required_array_paths"][0],
            "logical_role": "index",
            "content_kind": "index",
            "disposition": "create",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_length": len(payload),
        }
        return {
            **record,
            "objects": [*record["objects"], entry],
            "extents": [
                {**transition, "object_paths": [*transition["object_paths"], IDX_PATH]},
                *rest,
            ],
        }

    _rewrite_journal(root, carry_the_index)


def test_committed_index_is_never_quarantined(session: Path) -> None:
    _commit_an_index(session, b"a real committed index")

    diagnosis = diagnose_session(session)
    assert IDX_PATH in diagnosis.state.committed_objects
    assert IDX_PATH not in diagnosis.index_cache_paths
    assert not diagnosis.has("index_cache_unrebuildable")
    assert diagnosis.readable

    result = recover(session, quarantine=True, created_at=_NOW)
    assert IDX_PATH not in result.quarantined_paths
    assert (session / IDX_PATH).read_bytes() == b"a real committed index"


def test_damaged_committed_index_is_reported_not_moved(session: Path) -> None:
    # Recovery must not remove the evidence of corruption, and it must not
    # quietly drop an object the journal says the session holds.
    _commit_an_index(session, b"a real committed index")
    (session / IDX_PATH).write_bytes(b"truncated")

    diagnosis = diagnose_session(session)
    assert IDX_PATH in diagnosis.corrupt_objects
    assert diagnosis.has("object_truncated")
    assert not diagnosis.readable

    result = recover(session, quarantine=True, created_at=_NOW)
    assert result.status == STATUS_PARTIALLY_RECOVERED
    assert IDX_PATH not in result.quarantined_paths
    assert (session / IDX_PATH).exists()


def test_corrupt_index_does_not_affect_visibility(
    session: Path,
) -> None:
    idx = session / IDX_PATH
    idx.parent.mkdir(parents=True)
    idx.write_bytes(b"\x00 not a real index")

    diagnosis = diagnose_session(session)
    assert diagnosis.has("index_cache_unrebuildable")
    assert diagnosis.readable
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)

    result = recover(session, quarantine=True, created_at=_NOW)
    # NRF v1 journals carry no index transitions, so nothing describes how to
    # rebuild one; claiming otherwise would be the dishonest answer.
    assert result.report["rebuilt_indexes"] == []
    assert IDX_PATH in result.quarantined_paths
    assert not idx.exists()


# --- 11. writer tails -------------------------------------------------------


def test_writer_tail_is_invisible_and_never_promoted(session: Path) -> None:
    tail = session / ".staging/tails/streams-neural-data/pending-0000"
    tail.parent.mkdir(parents=True)
    tail.write_bytes(b"uncommitted rows")
    expected = diagnose_session(session).state.extent("streams/neural/data")

    diagnosis = diagnose_session(session)
    assert diagnosis.has("staged_objects_present")
    assert ".staging/tails/streams-neural-data/pending-0000" in diagnosis.staged_paths
    assert diagnosis.readable

    result = recover(session, created_at=_NOW)
    assert result.report["committed_extents"]["streams/neural/data"] == expected
    assert tail.exists()  # not quarantined unless asked, and never promoted

    quarantined = recover(session, quarantine=True, created_at=_NOW, recovery_id="recovery-0002")
    assert ".staging/tails/streams-neural-data/pending-0000" in quarantined.quarantined_paths
    assert not tail.exists()
    assert (
        session / "recovery/quarantine/recovery-0002/.staging/tails/"
        "streams-neural-data/pending-0000"
    ).exists()


# --- recovery guarantees ----------------------------------------------------

_NOW = "2026-08-01T12:00:00Z"


def test_recovery_leaves_committed_data_unchanged(session: Path) -> None:
    (session / JOURNAL_HEAD).unlink()  # give recovery a real cache to rebuild
    before = _payload_digests(session)

    result = recover(session, quarantine=True, created_at=_NOW)

    assert _payload_digests(session) == before
    assert result.report["committed_data_modified"] is False
    assert JOURNAL_HEAD in result.rebuilt_caches
    with NrfReader.open(session, verify_checksums=True) as reader:
        assert np.array_equal(reader.read_stream("neural"), NEURAL)
        assert reader.legacy_termination_normal


def test_recovery_report_validates_and_verifies(session: Path) -> None:
    result = recover(session, created_at=_NOW)

    assert result.report_path is not None
    assert result.report_path == Path(session)
    assert (session / "recovery/recovery-0001.json").exists()
    document = read_report(result.report_path)  # schema + record_checksum
    assert document == dict(result.report)
    assert document["extensions"]["neurale"]["status"] == STATUS_RECOVERED


def test_dry_run_writes_nothing_and_reports_plan(
    tmp_path: Path,
) -> None:
    root = _healthy(tmp_path / "dry.nrf")
    (root / JOURNAL_HEAD).unlink()
    (root / "streams/neural/data/zarr.json").unlink()
    tail = root / ".staging/tails/pending-0000"
    tail.parent.mkdir(parents=True)
    tail.write_bytes(b"tail")
    before = _file_digests(root)

    proposed = recover(root, dry_run=True, quarantine=True, created_at=_NOW)
    assert _file_digests(root) == before
    assert proposed.report_path is None

    applied = recover(root, quarantine=True, created_at=_NOW)
    assert dict(applied.report) == dict(proposed.report)
    assert applied.rebuilt_caches == proposed.rebuilt_caches
    assert not tail.exists()
    assert (root / "streams/neural/data/zarr.json").exists()


def test_unrecoverable_session_rewrites_nothing(
    session: Path,
) -> None:
    raw = (session / MANIFEST).read_bytes()
    (session / MANIFEST).write_bytes(raw[:40])
    before = _payload_digests(session)

    result = recover(session, quarantine=True, created_at=_NOW)

    assert result.status == STATUS_UNRECOVERABLE
    assert result.rebuilt_caches == ()
    assert (session / MANIFEST).read_bytes() == raw[:40]
    assert _payload_digests(session) == before
    assert result.report_path is None  # never rewrite an unrepairable package to add a report
    document = result.report
    assert document["committed_data_modified"] is False
    assert any(
        item["code"] == "manifest_unreadable"
        for item in document["extensions"]["neurale"]["diagnostics"]
    )


def test_damaged_session_recovery_is_partial(session: Path) -> None:
    path = _committed_chunk(session, "streams/neural/data")
    (session / path).unlink()

    result = recover(session, created_at=_NOW)

    assert result.status == STATUS_PARTIALLY_RECOVERED
    assert path in result.report["missing_objects"]
    # The rest of the session is untouched and still readable.
    with NrfReader.open(session) as reader:
        assert np.array_equal(reader.read_stream("bandpower"), BANDPOWER)


def test_recovery_refreshes_lagging_cache(session: Path) -> None:
    head = json.loads((session / JOURNAL_HEAD).read_bytes())
    head["journal_sequence"] = 1
    head["committed_extents"] = {}
    head["sealed_targets"] = []
    (session / JOURNAL_HEAD).write_bytes(canonical_json_bytes(head))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("cache_lags_replay")
    assert diagnosis.readable  # lagging is recoverable, leading is not

    recover(session, created_at=_NOW)
    refreshed = json.loads((session / JOURNAL_HEAD).read_bytes())
    assert refreshed["journal_sequence"] == diagnosis.state.journal_sequence
    assert refreshed["committed_extents"] == dict(diagnosis.state.committed_extents)
    assert diagnose_session(session).lagging_caches == ()


@pytest.mark.parametrize(
    "damage",
    [
        {"journal_sequence": "invalid"},
        {"committed_extents": []},
        {"unexpected_member": 1},
    ],
    ids=["wrong-type", "wrong-container", "unknown-member"],
)
def test_invalid_head_cache_is_rebuilt(session: Path, damage: dict[str, Any]) -> None:
    """A malformed pointer cache must not reach a comparison it cannot survive.

    ``head.json`` carries no checksum; its integrity is agreement with replay.
    One that is not a head document at all makes no claim to compare, so it is
    ignored and rebuilt -- but it must be *recognized* as malformed first,
    rather than reaching the sequence comparison and surfacing as a raw
    ``TypeError`` from Python.
    """
    head = json.loads((session / JOURNAL_HEAD).read_bytes())
    (session / JOURNAL_HEAD).write_bytes(canonical_json_bytes({**head, **damage}))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("head_cache_invalid")
    assert diagnosis.readable  # a rebuildable cache never costs committed data

    with NrfReader.open(session) as reader:  # no TypeError escapes
        assert np.array_equal(reader.read_stream("neural"), NEURAL)

    recover(session, created_at=_NOW)
    assert not diagnose_session(session).has("head_cache_invalid")


# --- incompleteness is not corruption ---------------------------------------


def test_unfinalized_session_is_incomplete_but_readable(tmp_path: Path) -> None:
    root = _healthy(tmp_path / "open.nrf", finalize=False)

    diagnosis = diagnose_session(root)
    assert diagnosis.has("session_not_terminated")
    assert diagnosis.of_severity(SEVERITY_INCOMPLETE)
    assert diagnosis.readable
    assert not diagnosis.complete

    with NrfReader.open(root, verify_checksums=True) as reader:
        assert not reader.complete
        assert np.array_equal(reader.read_stream("neural"), OPEN_NEURAL)


def test_partial_journal_line_is_crash_outcome(tmp_path: Path) -> None:
    root = _healthy(tmp_path / "torn.nrf", finalize=False)
    journal = root / JOURNAL_TRANSACTIONS
    journal.write_bytes(journal.read_bytes() + b'{"kind":"commit","seq')

    diagnosis = diagnose_session(root)
    assert diagnosis.has("journal_partial_line")
    assert diagnosis.find("journal_partial_line")[0].severity == SEVERITY_STALE
    assert any(item.reason == "partial_line" for item in diagnosis.ignored_records)
    assert diagnosis.readable

    with NrfReader.open(root) as reader:
        assert np.array_equal(reader.read_stream("neural"), OPEN_NEURAL)


def test_healthy_session_reports_nothing(session: Path) -> None:
    diagnosis = diagnose_session(session)
    assert diagnosis.diagnostics == ()
    assert diagnosis.readable and diagnosis.complete
    assert not diagnosis.repairable


def test_diagnosis_never_mutates_session(session: Path) -> None:
    before = _file_digests(session)
    diagnose_session(session)
    assert _file_digests(session) == before


def test_opening_reader_never_mutates_session(session: Path) -> None:
    before = sorted(item.relative_to(session).as_posix() for item in session.rglob("*"))
    with NrfReader.open(session, verify_checksums=True) as reader:
        reader.to_recording()
    assert sorted(item.relative_to(session).as_posix() for item in session.rglob("*")) == before


def test_severity_separates_unreadable_from_unfinished(tmp_path: Path) -> None:
    unfinished = diagnose_session(_healthy(tmp_path / "a.nrf", finalize=False))
    damaged_root = _healthy(tmp_path / "b.nrf")
    (damaged_root / _committed_chunk(damaged_root, "streams/neural/data")).unlink()
    damaged = diagnose_session(damaged_root)

    assert unfinished.of_severity(SEVERITY_INCOMPLETE) and unfinished.readable
    assert damaged.of_severity(SEVERITY_CORRUPT) and not damaged.readable


# --- command-line tool ------------------------------------------------------


def test_tool_reports_healthy_and_damaged_sessions(
    session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["diagnose", str(session)]) == EXIT_OK
    assert "no findings" in capsys.readouterr().out

    (session / _committed_chunk(session, "streams/neural/data")).unlink()
    assert main(["diagnose", str(session)]) == EXIT_CORRUPT
    assert "object_missing" in capsys.readouterr().out


def test_tool_dry_run_writes_nothing(session: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (session / JOURNAL_HEAD).unlink()
    before = _file_digests(session)
    main(["recover", str(session), "--dry-run"])
    assert _file_digests(session) == before
    assert json.loads(capsys.readouterr().out)["format"] == "nrf-recovery-report"


def test_tool_rejects_uninterpretable_session(
    session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (session / MANIFEST).write_bytes(b"{")
    assert main(["recover", str(session)]) == EXIT_UNUSABLE
    capsys.readouterr()


def test_quarantine_moves_evidence(session: Path) -> None:
    tail = session / ".staging/tails/pending-0000"
    tail.parent.mkdir(parents=True)
    payload = b"rows that never reached a commit"
    tail.write_bytes(payload)

    recover(session, quarantine=True, created_at=_NOW, recovery_id="recovery-0001")

    moved = session / "recovery/quarantine/recovery-0001/.staging/tails/pending-0000"
    assert moved.read_bytes() == payload
    assert not tail.exists()


# --- broken descriptor references -------------------------------------------


def test_broken_clock_reference_is_semantic_failure(session: Path) -> None:
    manifest = json.loads((session / MANIFEST).read_bytes())
    manifest["streams"][0]["clock_id"] = "no-such-clock"
    (session / MANIFEST).write_bytes(canonical_json_bytes(manifest))

    diagnosis = diagnose_session(session)
    assert diagnosis.has("manifest_semantics_invalid")
    assert not diagnosis.readable
    with pytest.raises(NrfSemanticError, match="clock"):
        NrfReader.open(session)

    result = recover(session, created_at=_NOW)
    assert result.status == STATUS_UNRECOVERABLE
    assert result.rebuilt_caches == ()


def test_missing_record_column_chunk_stops_read(session: Path) -> None:
    path = _committed_chunk(session, "records/events/events-v1")
    (session / path).unlink()

    assert path in diagnose_session(session).missing_objects
    with NrfReader.open(session) as reader:
        with pytest.raises(NrfCorruptionError, match="missing"):
            reader.read_records("events-v1")
        # A different record set is unaffected.
        assert reader.record_extent("trials-v1") == 0
