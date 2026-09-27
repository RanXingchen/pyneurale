#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The envelope every decoder artifact carries, and the versions it admits.

The envelope answers three questions before a single array is read: is this a
decoder artifact at all, is it a format this release can read, and which
decoder does it describe. Each has its own failure, because "upgrade PyNeurale"
and "this directory is not an artifact" are different problems.

Version policy is the usual one and is stated rather than implied. A *major*
version change means the layout is not readable by an older reader, so an
unknown major is refused outright. A *minor* change adds fields an older reader
can ignore, so a newer minor is read, and its unknown fields are left alone. A
decoder type carries its own version for the same reason and on the same terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neurale import __version__

from . import _canonical as canonical
from ._codecs import BY_ID, DecoderCodec
from ._errors import DecoderArtifactFormatError, DecoderArtifactVersionError

#: Marks a directory as this format. Checked first, so a directory that is
#: something else entirely is reported as such rather than as a missing field.
ARTIFACT = "neurale.decoder"

#: Layout version of the envelope and the sections around each payload.
FORMAT_MAJOR = 1
FORMAT_MINOR = 0


def build(
    codec: DecoderCodec,
    payload: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    runtime_state: bool,
) -> dict[str, Any]:
    """Assemble the manifest of one artifact."""

    return {
        "artifact": ARTIFACT,
        "format_version": {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR},
        "writer": {"library": "neurale", "version": __version__},
        "decoder": {"type": codec.type_id, "type_version": codec.type_version},
        "runtime_state": bool(runtime_state),
        "arrays": dict(arrays),
        "payload": dict(payload),
    }


def codec_of(manifest: Mapping[str, Any]) -> DecoderCodec:
    """Return the codec named by a manifest, after checking every version.

    Raises
    ------
    DecoderArtifactFormatError
        If the directory is not a decoder artifact or the envelope is malformed.
    DecoderArtifactVersionError
        If the format major version, or the decoder type or its version, is not
        one this release reads.
    """

    marker = canonical.require_typed(manifest, "artifact", str, path="manifest")
    if marker != ARTIFACT:
        raise DecoderArtifactFormatError(
            f"manifest.artifact is {marker!r}, not {ARTIFACT!r}; this directory is not a "
            "decoder artifact."
        )

    version = canonical.require_typed(manifest, "format_version", dict, path="manifest")
    major = canonical.require_typed(version, "major", int, path="manifest.format_version")
    canonical.require_typed(version, "minor", int, path="manifest.format_version")
    if major != FORMAT_MAJOR:
        raise DecoderArtifactVersionError(
            f"the artifact declares format version {major}, and this release reads version "
            f"{FORMAT_MAJOR}. A major version means the layout changed in a way an older "
            "reader cannot interpret, so it is refused rather than guessed at."
        )

    decoder = canonical.require_typed(manifest, "decoder", dict, path="manifest")
    type_id = canonical.require_typed(decoder, "type", str, path="manifest.decoder")
    type_version = canonical.require_typed(decoder, "type_version", int, path="manifest.decoder")

    codec = BY_ID.get(type_id)
    if codec is None:
        raise DecoderArtifactVersionError(
            f"the artifact holds a {type_id!r} decoder, which this release cannot read. "
            f"Supported types are {sorted(BY_ID)}."
        )
    if type_version != codec.type_version:
        raise DecoderArtifactVersionError(
            f"the artifact holds a {type_id!r} decoder at version {type_version}, and this "
            f"release reads version {codec.type_version}."
        )
    return codec


def payload_of(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the decoder-specific section of a manifest."""

    payload = canonical.require(manifest, "payload", path="manifest")
    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError("manifest.payload must be a JSON object.")
    return payload


def check_writer(manifest: Mapping[str, Any]) -> None:
    """Check the writer stamp is present and well formed.

    Which version wrote an artifact is informational -- nothing is read
    differently because of it -- but it is the first thing anyone looks at when
    an artifact behaves oddly, so it must at least be there and be readable.
    """

    writer = canonical.require_typed(manifest, "writer", dict, path="manifest")
    canonical.require_typed(writer, "library", str, path="manifest.writer")
    canonical.require_typed(writer, "version", str, path="manifest.writer")


def check_runtime_state(
    manifest: Mapping[str, Any],
    codec: DecoderCodec,
    payload: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> bool:
    """Check the declared ``runtime_state`` against what the artifact holds.

    The flag is what a caller inspects to decide whether an artifact can resume
    an online session, so it has to mean something: it must agree with the
    decoder-specific ``runtime`` section, with the presence of ``runtime.*``
    arrays, and with whether this decoder has any runtime state at all. An
    artifact that says one thing in the envelope and another in the payload is
    refused rather than quietly resolved in favour of either.

    Returns
    -------
    bool
        The declared flag, once it is known to describe the artifact.
    """

    declared = canonical.require_typed(manifest, "runtime_state", bool, path="manifest")
    if declared and not codec.has_runtime_state:
        raise DecoderArtifactFormatError(
            f"manifest.runtime_state is true, but a {codec.type_id} decoder carries no runtime "
            "state. Prediction is stateless here, so there is nothing the flag could describe."
        )

    stored = (
        canonical.require(payload, "runtime", path="payload") is not None
        if codec.has_runtime_state
        else False
    )
    if declared != stored:
        raise DecoderArtifactFormatError(
            f"manifest.runtime_state is {declared}, but payload.runtime is "
            f"{'present' if stored else 'absent'}. The envelope and the payload disagree about "
            "what the artifact holds, so neither can be trusted to describe it."
        )

    runtime_arrays = sorted(name for name in arrays if name.startswith("runtime."))
    if not declared and runtime_arrays:
        raise DecoderArtifactFormatError(
            f"manifest.runtime_state is false, but the artifact declares runtime arrays "
            f"{runtime_arrays}. An artifact that stores no runtime state must not carry one."
        )
    if declared and not runtime_arrays:
        raise DecoderArtifactFormatError(
            "manifest.runtime_state is true, but the artifact declares no runtime arrays."
        )
    return declared


__all__ = [
    "ARTIFACT",
    "FORMAT_MAJOR",
    "FORMAT_MINOR",
    "build",
    "check_runtime_state",
    "check_writer",
    "codec_of",
    "payload_of",
]
