#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Errors raised while recording a streaming session.

A recorder fails in two distinguishable ways, and conflating them would hide
the one that matters. A configuration that cannot describe a valid session
fails before the session directory exists and destroys nothing. A failure while
recording has already produced bytes on disk, so it never raises out of the
data path: it marks the recorder faulted, terminates the session as ``aborted``,
and leaves a session recovery can open.
"""

from __future__ import annotations

from neurale.exceptions import (
    RecorderConfigError,
    RecorderError,
    StreamStateError,
)

#: Re-exported so that this module stays the one place recording code names its
#: errors. Both live in :mod:`neurale.exceptions` because ``neurale.io.nrf``
#: raises them when it parses a native-replay plan document, and it sits below
#: this package. They are the same classes, not aliases of a copy.
__all__ = [
    "FinalizationError",
    "RecorderConfigError",
    "RecorderError",
    "RecorderRuntimeShutdownError",
    "RecorderStateError",
    "ReplayConfigError",
    "ReplayError",
    "ReplayImageError",
    "SpoolSourceError",
]


class RecorderStateError(RecorderError, StreamStateError):
    """A recorder operation was attempted in the wrong lifecycle state."""


class FinalizationError(RecorderError):
    """Converting a committed spool into a sealed NRF session did not finish.

    **This is not a capture fault, and the two must never be conflated.**
    Capture and finalization are two dimensions of a session's outcome
    (contract section 3.2): a finalization failure moves ``finalization_status``
    and leaves the capture outcome exactly where the session-end record froze
    it. A session whose capture ran cleanly and whose finalization failed is a
    clean recording that is not yet in NRF form -- not a damaged one.

    ``retryable`` says whether trying again can succeed, and it is what decides
    which value ``finalization_status`` moves to. A full disk, a permission
    failure, or an interrupted finalizer is retryable: the status becomes
    ``failed_retryable``, the recorder is ``finalization_failed``, and the spool
    stays finalizable by that same recorder. A spool whose committed prefix
    cannot be accounted for is not retryable, because a retry reads the same
    bytes: the status becomes ``failed``, the recorder is ``failed``, and the
    retained spool goes to diagnosis, repair, or quarantine instead. Reporting
    the second as the first would ask a caller to retry what cannot succeed --
    and would contradict the progress document, which records the same
    distinction and is what offline recovery reads to decide resumability.

    ``category`` names the failure in one stable word so a caller can route it
    without parsing the message.
    """

    #: Whether another attempt over the same inputs could succeed.
    retryable: bool = True
    #: Stable one-word classification, mirrored into the progress document.
    category: str = "finalization"

    def __init__(self, message: str, *, category: str | None = None, retryable: bool | None = None):
        super().__init__(message)
        if category is not None:
            self.category = category
        if retryable is not None:
            self.retryable = retryable


class SpoolSourceError(FinalizationError):
    """The spool cannot be finalized, and no retry over it will change that.

    Distinct from a retryable failure because the answer is different: a
    retryable failure asks the caller to try again, while this one asks for
    diagnosis, repair, or quarantine (contract section 7 and spool
    specification section 8). The spool is retained either way -- nothing here
    ever deletes the only reconstruction input.
    """

    retryable = False
    category = "source"


class ReplayError(RecorderError):
    """Base class for failures of the replay path (contract section 8)."""


class ReplayConfigError(ReplayError, ValueError):
    """The replay was not described as a legal run over this session.

    Raised before any run starts, and deliberately the only answer to an
    unsupported request: contract section 8 forbids answering a request with a
    different mode, a widened range, or a silently skipped stream, so every one
    of those is this error rather than a degraded result.
    """


class ReplayImageError(ReplayError):
    """A replay image could not be read as one this build understands.

    Distinct from :class:`ReplayConfigError` because the subject is different:
    the configuration was legal and the *artifact* is unusable -- truncated,
    corrupt, written by another format version, or built for another source.
    Rebuilding is the remedy, which is what the cache path does with it.
    """


class RecorderRuntimeShutdownError(RecorderStateError):
    """An attached streaming runtime did not reach a terminal state.

    Raised *instead of* running the recorder's own shutdown, and that
    substitution is the whole point. The lifecycle order is ``runner
    stop/fault/join`` and only then ``recorder drain/sync/close``, so a runtime
    whose shutdown returned ``deadline_exceeded`` -- or that is simply not
    terminal yet -- has not completed the first stage, and entering the second
    one would stop a recorder while a worker may still be producing into it.
    That turns a runtime shutdown timeout into a critical-edge fault charged to
    the recording.

    Nothing was done to the recorder: it is still owned by the runtime's own
    shutdown and fault protocol, and a caller who resolves the runtime can end
    the session normally afterwards.
    """
