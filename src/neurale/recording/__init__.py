#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Record a live native streaming session into an NRF v1 session directory.

This is the integration layer, and it exists because neither side may depend on
the other: the streaming core must not know about :mod:`neurale.io`, and
:mod:`neurale.io.nrf` must not know about the runtime. Everything that has to
know about both lives here.

The public recorder is a thin orchestration facade over the native critical
recorder. Frames and discontinuities stay on the native path and are first
committed to its bounded spool. After capture stops, the existing offline
finalizer converts that spool into canonical NRF.

Being the integration layer is not the same as owning both sides. The recording
plan (:class:`~neurale.io.nrf.RecordingPlan` and its value objects) and the
native-replay ledgers (:data:`~neurale.io.nrf.LEDGERS` and the schema
accessors) are NRF artifacts that this package produces and consumes but does
not define, so they are imported from :mod:`neurale.io.nrf` rather than
re-exported here. Only :func:`compile_recording_plan`, which turns a
:class:`RecorderConfig` into a plan, belongs to the recorder.

The vocabularies the submodules validate against -- the legal pacing modes,
progress phases, range units, and the like -- stay in the module that enforces
them, because a caller that needs to *read* one is usually a caller that should
have been handed a checked value instead. They remain importable from there;
they are simply not part of this package's public surface.

Read a recorded session back with the ordinary offline reader::

    from neurale.io.nrf import NrfReader

    with NrfReader.open(path) as reader:
        assert reader.complete is True
        data = reader.read_stream("neural")
        for start, block in reader.iter_blocks("neural"):
            ...
        recording = reader.to_recording()

Example::

    from neurale.recording import RecorderConfig, SessionRecorder

    config = RecorderConfig(path=tmp / "session.nrf")
    recorder = SessionRecorder.create(config, device)
    recorder.prepare()
    recorder.attach(runner)
    runner.prepare()
    runner.arm()
    runner.start()
    ...
    status = recorder.close()
    assert status.complete
"""

from __future__ import annotations

from ._block_index import block_index_columns, read_block_index
from ._errors import (
    FinalizationError,
    RecorderConfigError,
    RecorderError,
    RecorderRuntimeShutdownError,
    RecorderStateError,
    ReplayConfigError,
    ReplayError,
    ReplayImageError,
    SpoolSourceError,
)
from ._finalizer import (
    FINALIZATION_NAMESPACE,
    FinalizationCounts,
    FinalizationProgress,
    FinalizationReport,
    FinalizerOptions,
    StagedConversion,
    finalize_spool,
)
from ._plan import compile_recording_plan
from ._recovery import (
    ACTION_DISCARD,
    ACTION_FINALIZE,
    ACTION_QUARANTINE,
    ACTION_REPAIR,
    RESUMABLE_STATUSES,
    FinalizationState,
    SpoolDiagnosis,
    SpoolRepairReport,
    abandon_finalization,
    diagnose_finalization,
    diagnose_spool,
    read_repair_report,
    repair_spool,
    resume_finalization,
)
from ._replay_build import (
    ReplayImageBuild,
    ResolvedFault,
    build_replay_image,
    load_replay_image,
    open_replay_image,
    resolve_replay_faults,
)
from ._replay_config import (
    InjectedFault,
    MessageFaultTarget,
    MessageRange,
    ReplayConfig,
    StreamDiscontinuityFaultTarget,
    StreamFrameFaultTarget,
    StreamReplayRange,
)
from ._replay_image import (
    OmittedMessage,
    ReplayBlock,
    ReplayGap,
    ReplayImage,
    ReplayImageMetadata,
    ReplayItem,
    StreamFidelity,
)
from ._session_recorder import FinalizationAttempt, RecorderStatus, SessionRecorder
from ._spec import (
    RecorderConfig,
    RecorderLimits,
    RecorderSessionMetadata,
    RecorderState,
    StreamMetadata,
    StreamProvenanceConfig,
    StreamRecording,
    StreamRole,
    StreamStorageConfig,
    StreamTimingConfig,
    StreamTimingMode,
)

__all__ = [
    "ACTION_DISCARD",
    "ACTION_FINALIZE",
    "ACTION_QUARANTINE",
    "ACTION_REPAIR",
    "FINALIZATION_NAMESPACE",
    "RESUMABLE_STATUSES",
    "FinalizationAttempt",
    "FinalizationCounts",
    "FinalizationError",
    "FinalizationProgress",
    "FinalizationReport",
    "FinalizationState",
    "FinalizerOptions",
    "InjectedFault",
    "MessageFaultTarget",
    "MessageRange",
    "OmittedMessage",
    "RecorderConfig",
    "RecorderConfigError",
    "RecorderError",
    "RecorderLimits",
    "RecorderRuntimeShutdownError",
    "RecorderSessionMetadata",
    "RecorderState",
    "RecorderStateError",
    "RecorderStatus",
    "ReplayBlock",
    "ReplayConfig",
    "ReplayConfigError",
    "ReplayError",
    "ReplayGap",
    "ReplayImage",
    "ReplayImageBuild",
    "ReplayImageError",
    "ReplayImageMetadata",
    "ReplayItem",
    "ResolvedFault",
    "SessionRecorder",
    "SpoolDiagnosis",
    "SpoolRepairReport",
    "SpoolSourceError",
    "StagedConversion",
    "StreamDiscontinuityFaultTarget",
    "StreamFidelity",
    "StreamFrameFaultTarget",
    "StreamMetadata",
    "StreamProvenanceConfig",
    "StreamRecording",
    "StreamReplayRange",
    "StreamRole",
    "StreamStorageConfig",
    "StreamTimingConfig",
    "StreamTimingMode",
    "abandon_finalization",
    "block_index_columns",
    "build_replay_image",
    "compile_recording_plan",
    "diagnose_finalization",
    "diagnose_spool",
    "finalize_spool",
    "load_replay_image",
    "open_replay_image",
    "read_block_index",
    "read_repair_report",
    "repair_spool",
    "resolve_replay_faults",
    "resume_finalization",
]
