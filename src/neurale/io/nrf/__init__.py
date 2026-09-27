#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Neurale Recording Format (NRF) v1 offline sessions.

This package implements the normative specification in
``specifications/nrf/v1/`` and its single-file envelope. A published recording
is one ``<session>.nrf`` file containing a JSON manifest, Zarr v3 chunks,
a transaction journal and checkpoints. Native capture uses a separate spool;
compression and validation run after capture stops.

Importing :mod:`neurale.io` does not import this package, and importing this
package does not import Zarr; the optional dependencies in the ``nrf`` extra are
resolved at first use.

Two contracts are worth stating up front, because they are what make a recording
trustworthy rather than merely present:

* **Visibility comes from the journal.** ``manifest.json`` and
  ``journal/head.json`` are caches. A reader replays the journal, refuses a
  cache that leads replay, and never returns data beyond a committed extent.
* **An interrupted session is detectably incomplete.** Rows that never filled a
  chunk stay in a writer-owned tail and are invisible; a session that was not
  finalized reports ``complete == False`` rather than looking finished.

Implemented here: the offline writer and reader, both mandatory validation
layers, session diagnosis, and explicit recovery with a ``python -m
neurale.io.nrf`` entry point. Not implemented: any connection to the streaming
runtime -- the native critical recorder, the native spool, and paced replay.

Two things this package owns are easy to mistake for the recorder's, because the
recorder is their only producer: the **recording plan** and the **native-replay
ledgers**. Both are NRF artifacts. A plan document is written into the manifest
and read back by anything that opens the session, and a ledger is a record set
with a schema this package validates -- so their value objects, their codecs and
their schema accessors are defined and exported here. :mod:`neurale.recording`
consumes them; it does not own them, and it must not be the package that decides
whether they are public. Keeping the dependency pointing this way is what lets a
reader parse a plan without the runtime ever being importable.

Examples
--------
Write and read one session::

    from neurale.io.nrf import NrfWriter, NrfReader

    with NrfWriter.create(path, session_id=..., created_at=...) as writer:
        writer.registry.register_clock(...)
        ...
        writer.freeze()
        writer.append_stream("neural", block)
        writer.commit()

    with NrfReader.open(path) as reader:
        # complete is True/False/None -- see NrfReader.complete. ``is not False``
        # refuses all non-complete cases, unchanged.
        assert reader.complete is not False
        data = reader.read_stream("neural")

Diagnose a session a reader refuses, then recover it explicitly::

    from neurale.io.nrf import diagnose_session, recover

    diagnosis = diagnose_session(path)  # never raises, never mutates
    if not diagnosis.complete:
        result = recover(path, dry_run=True)  # proposes, writes nothing
        recover(path)  # rebuilds caches, writes a report
"""

from __future__ import annotations

from ._completeness import (
    AccountingFinding,
    CompletenessVerdict,
    SessionCompleteness,
    evaluate_completeness,
)
from ._diagnostics import (
    SEVERITY_CORRUPT,
    SEVERITY_INCOMPLETE,
    SEVERITY_STALE,
    Diagnostic,
    IgnoredRecord,
    SessionDiagnosis,
    diagnose_session,
)
from ._errors import (
    NrfCorruptionError,
    NrfError,
    NrfSchemaError,
    NrfSemanticError,
    NrfStateError,
)
from ._fields import record_field
from ._ledgers import (
    LEDGER_KINDS,
    LEDGERS,
    NATIVE_REPLAY_EXTENSION_VERSION,
    NATIVE_REPLAY_MINOR_VERSION,
    NATIVE_REPLAY_NAMESPACE,
    RECOVERY_NAMESPACE,
    LedgerDefinition,
    LedgerField,
    declares_native_replay,
    ledger_record_schema,
    ledger_record_schemas,
    ledger_schema_ids,
    ledger_target_paths,
)
from ._manifest import ManifestBuilder
from ._plan_document import (
    REPLAY_MODES,
    NativeSchema,
    PlannedFeatureSet,
    PlannedSignal,
    PlannedUnit,
    PreparedMetadata,
    PreparedStream,
    PreparedStreamBounds,
    RecordingPlan,
    ReplayCapability,
    ResourceBounds,
    SessionIdentity,
    plan_from_manifest_extension,
    plan_from_spool_document,
)
from ._reader import NrfReader
from ._recording import write_recording
from ._recovery import (
    STATUS_PARTIALLY_RECOVERED,
    STATUS_RECOVERED,
    STATUS_UNRECOVERABLE,
    RecoveryResult,
    read_report,
    recover,
)
from ._writer import NrfWriter

__all__ = [
    "LEDGERS",
    "LEDGER_KINDS",
    "NATIVE_REPLAY_EXTENSION_VERSION",
    "NATIVE_REPLAY_MINOR_VERSION",
    "NATIVE_REPLAY_NAMESPACE",
    "RECOVERY_NAMESPACE",
    "REPLAY_MODES",
    "SEVERITY_CORRUPT",
    "SEVERITY_INCOMPLETE",
    "SEVERITY_STALE",
    "STATUS_PARTIALLY_RECOVERED",
    "STATUS_RECOVERED",
    "STATUS_UNRECOVERABLE",
    "AccountingFinding",
    "CompletenessVerdict",
    "Diagnostic",
    "IgnoredRecord",
    "LedgerDefinition",
    "LedgerField",
    "ManifestBuilder",
    "NativeSchema",
    "NrfCorruptionError",
    "NrfError",
    "NrfReader",
    "NrfSchemaError",
    "NrfSemanticError",
    "NrfStateError",
    "NrfWriter",
    "PlannedFeatureSet",
    "PlannedSignal",
    "PlannedUnit",
    "PreparedMetadata",
    "PreparedStream",
    "PreparedStreamBounds",
    "RecordingPlan",
    "RecoveryResult",
    "ReplayCapability",
    "ResourceBounds",
    "SessionCompleteness",
    "SessionDiagnosis",
    "SessionIdentity",
    "declares_native_replay",
    "diagnose_session",
    "evaluate_completeness",
    "ledger_record_schema",
    "ledger_record_schemas",
    "ledger_schema_ids",
    "ledger_target_paths",
    "plan_from_manifest_extension",
    "plan_from_spool_document",
    "read_report",
    "record_field",
    "recover",
    "write_recording",
]
