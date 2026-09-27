#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Error types for decoder artifacts.

A load can fail for three different reasons, and a caller reacts to each
differently: the artifact was written by something this release cannot read, it
is well-formed but says something invalid, or its bytes no longer are what was
written. Collapsing the three into one error would leave "upgrade the library",
"fix the writer", and "the file is damaged" indistinguishable.
"""

from __future__ import annotations

from neurale.exceptions import NeuraleError


class DecoderArtifactError(NeuraleError):
    """Base class for every decoder-artifact error."""


class DecoderArtifactVersionError(DecoderArtifactError, ValueError):
    """The artifact declares a format or decoder version this release cannot read.

    Raised for an unknown format major version and for an unknown or
    unsupported decoder type. This is the "upgrade, or write with an older
    release" case: the artifact may be perfectly valid, just not for this
    reader.
    """


class DecoderArtifactFormatError(DecoderArtifactError, ValueError):
    """The artifact is malformed: a required field is missing or invalid.

    Also covers a stored feature schema whose fingerprint does not match the
    fields it carries, which means the manifest describes two different inputs
    at once.
    """


class DecoderArtifactCorruptionError(DecoderArtifactError, ValueError):
    """A stored array is missing, truncated, altered, or not a plain array.

    Includes an array whose dtype or shape disagrees with the manifest, a file
    whose digest does not match, and a payload that would require unpickling --
    which this format never does, whatever a file claims to contain.
    """


__all__ = [
    "DecoderArtifactCorruptionError",
    "DecoderArtifactError",
    "DecoderArtifactFormatError",
    "DecoderArtifactVersionError",
]
