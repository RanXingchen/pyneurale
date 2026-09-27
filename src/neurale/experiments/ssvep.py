# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Native SSVEP task and acquisition-to-feedback closed-loop session.

Fixed stimulation, at most one discrete selection, bounded decision wait,
feedback and rest. Task durations are expressed in seconds. Machine observation times and event
timestamps use integer nanoseconds, as in Center-Out. Pure-time steps can be drained at the same time until ``settled``.
The module loads the native extension only when an exported name is accessed.
"""

from __future__ import annotations

_NATIVE_EXPORTS = {
    "MAX_SSVEP_TARGETS",
    "SSVEPCause",
    "SSVEPTask",
    "SSVEPMachine",
    "SSVEPMarker",
    "SSVEPPhase",
    "SSVEPPhaseInterval",
    "SSVEPPresentationRequest",
    "SSVEPReason",
    "SSVEPSelectionDisposition",
    "SSVEPSnapshot",
    "SSVEPState",
    "SSVEPStepResult",
    "SSVEPTarget",
    "SSVEPTrial",
    "SSVEPTrialSchedule",
    "prepare_trial",
    "validate",
}


_SESSION_EXPORTS = {"SSVEPSession", "SSVEPOutcome", "SSVEPProtocol"}


def __getattr__(name: str) -> object:
    if name in _SESSION_EXPORTS:
        from . import _ssvep_session

        value = getattr(_ssvep_session, name)
        globals()[name] = value
        return value
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from neurale._native_loader import load_native_namespace

    value = getattr(load_native_namespace("experiments.ssvep"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS | _SESSION_EXPORTS)


__all__ = sorted(_NATIVE_EXPORTS | _SESSION_EXPORTS)
