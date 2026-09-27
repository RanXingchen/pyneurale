#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Public error types for NRF v1 sessions.

The specification requires an implementation to report *which* validation layer
rejected an input (``semantic-validation.md`` section 2), so schema violations,
cross-object semantic violations, and physical corruption are distinct types
rather than one generic error.
"""

from __future__ import annotations

from neurale.exceptions import NeuraleError


class NrfError(NeuraleError):
    """Base class for every NRF session error."""


class NrfSchemaError(NrfError, ValueError):
    """A document failed Draft 2020-12 validation against its NRF schema.

    This is the first of the two mandatory validation layers.
    """


class NrfSemanticError(NrfError, ValueError):
    """A document failed the NRF v1 cross-object semantic rules.

    This is the second mandatory layer: registry references, rank/shape
    relationships, extent agreement, path ownership, and the other rules in
    ``semantic-validation.md`` that JSON Schema cannot express.
    """


class NrfCorruptionError(NrfError, ValueError):
    """A committed object is missing, truncated, or fails its checksum.

    Corruption is never an uncommitted tail: the specification requires a
    missing or checksum-invalid *committed* object to make the affected
    committed range unreadable rather than to be silently ignored.
    """


class NrfStateError(NrfError, RuntimeError):
    """A session operation was attempted in the wrong lifecycle state.

    Covers registering after the registry freeze, appending to a sealed target,
    and using a closed session.
    """
