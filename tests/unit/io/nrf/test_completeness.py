#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The completeness verdict over sessions that carry no accounting.

Normative source: ``docs/development/native_recording_replay.md`` section 5.1.

These are the **legacy** cases, and they are here rather than in the recording
tree because they need no recorder at all: a session written by any generic
``NrfWriter`` carries no accounting summary, which is the state every session
written before the summary existed is in. The verification of an accounting
summary that *is* present needs a valid NRF v1.1 ``neurale.native_replay``
extension, so those cases are built from real finalized sessions in
``tests/unit/recording/test_recovery.py``.

The rule these pin: a legacy session terminated ``normal`` reads
``unverified_legacy`` with ``complete is None``. It must not read ``True`` -- the
session is perfectly readable, it is the completeness claim that is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurale.io.nrf import NrfReader, NrfWriter

from .nrf_support import CREATED_AT, SESSION_ID, register_discontinuities, register_records
from .package_objects import snapshot


def _legacy_session(root: Path, *, kind: str | None) -> None:
    """A session with no accounting summary at all, sealed or left open."""
    writer = NrfWriter.create(root, session_id=SESSION_ID, created_at=CREATED_AT)
    registry = writer.registry
    registry.register_clock(
        "host-clock",
        clock_type="host_monotonic",
        rate={"numerator": 1_000_000_000, "denominator": 1},
        synchronization_domain="session-domain",
    )
    registry.register_unit("volt", symbol="V", description="volt")
    registry.register_channel("channel-0000", idx=0, unit_id="volt", electrode_id="grid-a")
    registry.register_channel("channel-0001", idx=1, unit_id="volt", electrode_id="grid-a")
    registry.register_electrode("grid-a", channel_ids=["channel-0000", "channel-0001"])
    registry.register_schema("schema-source", stream_ids=["neural"])
    register_records(registry)
    register_discontinuities(registry, "neural")
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
    writer.freeze()
    writer.append_stream("neural", np.arange(8, dtype=np.int16).reshape(4, 2))
    writer.commit()
    if kind is not None:
        writer.finalize(kind=kind, reason="fixture", fault_id=None)
    else:
        snapshot(writer)  # intentionally unsealed package for verdict tests
    writer.close()


def test_legacy_normal_termination_is_unverified(tmp_path: Path) -> None:
    """It must not read ``complete is True``, and it must not raise.

    What the artifact does say is exposed separately as
    ``legacy_termination_normal`` -- a fact about the termination record, not a
    completeness verdict. Calling that fact ``complete`` would return the API to
    a boolean plus a flag every caller must remember to combine, and would assert
    exactly what a legacy artifact cannot support.
    """
    _legacy_session(tmp_path / "legacy.nrf", kind="normal")
    with NrfReader.open(tmp_path / "legacy.nrf") as reader:
        assert reader.completeness_verdict == "unverified_legacy"
        assert reader.complete is None
        assert not reader.accounting_verified
        assert reader.legacy_termination_normal
        assert reader.read_stream("neural").shape == (4, 2)


def test_legacy_abnormal_termination_verifies_incomplete(
    tmp_path: Path,
) -> None:
    """Its own termination record proves it incomplete -- no accounting needed.

    Incompleteness has cheaper proofs than completeness, which is why the verdict
    is not bound to ``accounting_verified``: binding it would leave
    ``complete = False`` with unverified accounting unrepresentable, and that is
    the state every crash-recovered session is in.
    """
    _legacy_session(tmp_path / "legacy.nrf", kind="aborted")
    with NrfReader.open(tmp_path / "legacy.nrf") as reader:
        assert reader.completeness_verdict == "verified_incomplete"
        assert reader.complete is False
        assert not reader.accounting_verified
        assert not reader.legacy_termination_normal


def test_unsealed_session_has_no_verdict(tmp_path: Path) -> None:
    """Completeness is a property of a sealed session (stage 5).

    ``None`` is not a defect report and must not be reported as
    ``unverified_legacy``: one means there is no sealed session to judge, the
    other that there is one and it carries no evidence.
    """
    _legacy_session(tmp_path / "open.nrf", kind=None)
    with NrfReader.open(tmp_path / "open.nrf") as reader:
        assert reader.completeness_verdict is None
        assert reader.complete is None
        assert not reader.accounting_verified
        assert not reader.legacy_termination_normal


def test_falsiness_keeps_callers_refusing_same_sessions(
    tmp_path: Path,
) -> None:
    """``if not reader.complete`` rejects sessions without verified completeness.

    ``None`` is falsy, so ``not complete`` rejects both unverified and
    verified-incomplete sessions. ``complete is False`` distinguishes a
    verified-incomplete session from an unverified one.
    """
    _legacy_session(tmp_path / "legacy.nrf", kind="normal")
    _legacy_session(tmp_path / "aborted.nrf", kind="aborted")
    with NrfReader.open(tmp_path / "legacy.nrf") as legacy:
        assert not legacy.complete
        assert legacy.complete is not False
    with NrfReader.open(tmp_path / "aborted.nrf") as aborted:
        assert not aborted.complete
        assert aborted.complete is False


def test_reader_judges_session_without_recorder() -> None:
    """The verdict is derived by the reader, from the artifact, and nothing else.

    Stated as a test because the dependency direction is a rule rather than a
    preference: ``neurale.io.nrf`` must be able to judge a session without
    ``neurale.recording``, so the module that computes the verdict names the NRF
    record kinds itself instead of importing the recorder's ledger definitions.
    """
    import ast

    from neurale.io.nrf import _completeness

    tree = ast.parse(Path(_completeness.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("neurale.recording") for name in imported), imported
