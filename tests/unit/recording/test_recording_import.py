#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The import contract for :mod:`neurale.recording`.

Every other public subpackage pins its exports this way -- ``streaming``,
``signal``, ``decoding``, ``sorting``. ``recording`` did not, and it is the one
that grew to a hundred names. The snapshot below is not decoration: it is the
only thing that makes adding a public name a visible edit rather than a side
effect of adding an import, and the only thing that makes removing one show up
in a diff.

Adding a name here is a long-term contract. Prefer leaving a type importable
from its own module over re-exporting it at the top level, and never re-export
a name another package owns -- that package decides its own public surface.
"""

from _subprocess_probe import probe_json


def _import_recording_in_a_fresh_interpreter() -> dict[str, object]:
    code = """
import importlib
import json
import sys

recording = importlib.import_module("neurale.recording")
forbidden = (
    "neurale._native",
    "neurale._native_cuda",
    "neurale.streaming",
    "neurale.devices",
    "neurale.visualization",
    "neurale.pipeline",
    "neurale.signal",
    "neurale.features",
    "neurale.models",
    "neurale.decoding",
    "neurale.sorting",
    "zarr",
    "torch",
    "PySide6",
    "PyQt6",
)
print(json.dumps({
    "exports": sorted(recording.__all__),
    "loaded_forbidden": [name for name in forbidden if name in sys.modules],
}))
"""
    return probe_json(code)


def test_recording_exports_are_pinned() -> None:
    result = _import_recording_in_a_fresh_interpreter()

    assert result["exports"] == [
        "ACTION_DISCARD",
        "ACTION_FINALIZE",
        "ACTION_QUARANTINE",
        "ACTION_REPAIR",
        "FINALIZATION_NAMESPACE",
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
        "RESUMABLE_STATUSES",
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


def test_recording_import_loads_neither_runtime_nor_storage_backend() -> None:
    # ``recording`` is the integration layer between the streaming runtime and
    # :mod:`neurale.io.nrf`, so it is the one package positioned to drag both in
    # at import time. It must drag in neither: no native extension, no
    # streaming runtime, and no Zarr -- the last one transitively preserves
    # ``neurale.io.nrf``'s own promise that importing it does not import Zarr.
    result = _import_recording_in_a_fresh_interpreter()

    assert result["loaded_forbidden"] == []
