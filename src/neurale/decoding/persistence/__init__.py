#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""One versioned, non-executable format for saving a fitted decoder.

An artifact is a directory holding a canonical JSON manifest and one ``.npy``
file per array:

.. code-block:: text

    <artifact>/
        manifest.json
        arrays/model.coef.npy
        arrays/fitted.scaler.mean.npy
        ...

**Loading never executes anything the artifact chose.** Arrays are read with
``allow_pickle=False``, the manifest carries no import path or class name, and
the decoder is selected from a fixed table of type identifiers this release
knows. A hostile artifact can therefore make a load *fail*, which is the point,
but it cannot make it import, construct, or call anything a normal load would
not. This is the whole reason the format exists rather than a pickle.

What is stored
--------------

The manifest records the artifact and format version, the PyNeurale version
that wrote it, the decoder type and *its* version, and for a feature-based
decoder the device the fit actually ran on, the canonical feature schema with
its fingerprint, the fitted timeline, and the target or class metadata a
prediction is built from. A sequence decoder stores its vocabulary and the
score schema it consumes instead, because it has none of the former. Alongside
that go the owned preprocessing configuration *and* its fitted state, and the
model parameters.

What is not stored, by default
------------------------------

Transient online state. A saved decoder comes back in its post-reset condition,
because that is what a fitted decoder means outside the session that produced
it: a Kalman filter's evolving state belongs to the recording it was decoding,
not to the model. ``runtime_state=True`` stores it anyway for callers resuming
a session, and the manifest records which of the two it holds.

Guarantees
----------

A loaded decoder predicts *exactly* what the saved one predicted -- the native
models are rebuilt from their own parameters rather than reimplemented, so the
outputs are equal bit for bit, not merely close. :attr:`~BaseDecoder.device_`
is restored as recorded and is never re-resolved, so loading under a different
ambient runtime cannot silently move a fitted decoder onto another device. And
the feature schema is re-derived from the stored fields and checked against the
recorded fingerprint, so an artifact cannot describe two different inputs.

Scope
-----

This is a decoder artifact and nothing more. It is not a
:mod:`neurale.io` backend, not an NRF extension, and not a container for
recordings: it holds one fitted decoder, and the formats that hold data are a
separate concern with separate guarantees.

Examples
--------
>>> import numpy as np
>>> from neurale.data import FeatureMatrix, SignalArray
>>> from neurale.decoding import LinearDecoder
>>> from neurale.decoding.persistence import load_decoder, save_decoder
>>> rng = np.random.default_rng(0)
>>> position = np.cumsum(rng.normal(scale=0.05, size=(60, 2)), axis=0)
>>> X = FeatureMatrix(
...     data=position @ rng.normal(size=(2, 4)),
...     fs=50.0,
...     feature_names=[f"rate_{i}" for i in range(4)],
...     unit="Hz",
...     shift=0.02,
... )
>>> y = SignalArray.from_array(
...     position,
...     fs=50.0,
...     time=X.time.copy(),
...     channel_names=["x", "y"],
...     channel_types="behavior",
...     units="m",
...     name="cursor",
... )
>>> decoder = LinearDecoder().fit(X, y)
>>> import tempfile, pathlib
>>> directory = pathlib.Path(tempfile.mkdtemp()) / "cursor-decoder"
>>> _ = save_decoder(decoder, directory)
>>> restored = load_decoder(directory)
>>> restored.device_, restored.feature_names_in_ == decoder.feature_names_in_
('cpu', True)
>>> bool(np.array_equal(restored.predict(X).data, decoder.predict(X).data))
True
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neurale.exceptions import ValidationError

from ._artifact import (
    ArrayReader,
    ArrayWriter,
    declared_arrays,
    publishing,
    read_manifest_bytes,
    write_manifest,
)
from ._codecs import BY_ID, BY_TYPE
from ._errors import (
    DecoderArtifactCorruptionError,
    DecoderArtifactError,
    DecoderArtifactFormatError,
    DecoderArtifactVersionError,
)
from ._manifest import (
    ARTIFACT,
    FORMAT_MAJOR,
    FORMAT_MINOR,
    build,
    check_runtime_state,
    check_writer,
    codec_of,
    payload_of,
)


def save_decoder(
    decoder: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
    runtime_state: bool = False,
) -> Path:
    """Save one fitted decoder as an artifact directory.

    Parameters
    ----------
    decoder : object
        A fitted decoder of a supported type. Which types those are is an
        explicit table, not a protocol: a decoder without a codec is refused
        rather than saved partially.
    path : str or pathlib.Path
        Directory to publish. It is built under a temporary name beside itself
        and renamed into place, so a reader never sees a partial artifact.
    overwrite : bool, optional
        Whether to replace an existing artifact at ``path``. The default
        refuses: a save never silently discards one.
    runtime_state : bool, optional
        Whether to store the transient state a prediction has moved, for a
        decoder that has any. The default stores the post-reset condition.

    Returns
    -------
    pathlib.Path
        The published directory.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the decoder is unfitted or of an unsupported type.
    DecoderArtifactFormatError
        If some metadata the decoder carries cannot be represented without
        pickling it.
    FileExistsError
        If ``path`` exists and ``overwrite`` is false.
    """

    codec = BY_TYPE.get(type(decoder))
    if codec is None:
        raise ValidationError(
            f"{type(decoder).__name__} has no decoder artifact codec. This format supports "
            f"{sorted(BY_ID)}, each with an explicit reader; a decoder outside that set would "
            "have to be pickled, and this format never is."
        )
    # A sequence decoder has no fitted state of its own -- its language model
    # is required to be fitted before it can be constructed at all -- so the
    # check applies only where there is one.
    if hasattr(decoder, "is_fitted") and not decoder.is_fitted:
        raise ValidationError(
            f"{type(decoder).__name__} instance is not fitted; there is nothing to save."
        )
    if not isinstance(runtime_state, bool):
        raise ValidationError("runtime_state must be a bool.")
    if not isinstance(overwrite, bool):
        raise ValidationError("overwrite must be a bool.")
    if runtime_state and not codec.has_runtime_state:
        raise ValidationError(
            f"a {codec.type_id} decoder carries no runtime state, so runtime_state=True would "
            "record nothing. Prediction is stateless here."
        )

    destination = Path(path)
    with publishing(destination, overwrite=overwrite) as staging:
        writer = ArrayWriter(staging)
        payload = codec.write(decoder, writer, runtime_state=runtime_state)
        write_manifest(
            staging,
            build(codec, payload, writer.entries, runtime_state=runtime_state),
        )
    return destination


def load_decoder(path: str | Path) -> Any:
    """Load a fitted decoder from an artifact directory.

    Nothing the artifact names is imported or executed: the decoder type comes
    from a fixed table, and every array is read with ``allow_pickle=False`` and
    checked against the dtype, shape, and digest the manifest declares.

    Parameters
    ----------
    path : str or pathlib.Path
        The artifact directory.

    Returns
    -------
    object
        The restored decoder, fitted, reporting the device it was fitted on.

    Raises
    ------
    DecoderArtifactVersionError
        If the format or decoder version is not one this release reads.
    DecoderArtifactFormatError
        If the directory is not an artifact, or the manifest is missing a
        required field or describes something invalid.
    DecoderArtifactCorruptionError
        If a stored array is missing, truncated, altered, of the wrong dtype or
        shape, or would require unpickling.
    """

    directory = Path(path)
    manifest = read_manifest_bytes(directory)
    codec = codec_of(manifest)
    check_writer(manifest)
    payload = payload_of(manifest)
    arrays = declared_arrays(manifest)
    check_runtime_state(manifest, codec, payload, arrays)
    return codec.read(payload, ArrayReader(directory, arrays))


def read_manifest(path: str | Path) -> Mapping[str, Any]:
    """Read an artifact's manifest without loading the decoder.

    This is how a caller inspects what an artifact holds -- which decoder, which
    versions, which writer, which arrays -- including one this release cannot
    load, which is why the manifest is *parsed but not validated*: a version
    mismatch has to be something a caller can read and report, not an exception
    with nothing behind it. Only that the directory holds a ``manifest.json``
    of valid UTF-8 JSON whose top level is an object is checked here. Every
    other guarantee in this module -- the artifact marker, the versions, the
    envelope, the arrays -- belongs to :func:`load_decoder`, so a field read out
    of this result is a field a caller should check before trusting it.

    Returns
    -------
    Mapping
        The parsed manifest, exactly as stored.
    """

    return read_manifest_bytes(Path(path))


def supported_types() -> tuple[str, ...]:
    """Return the decoder type identifiers this release can read and write."""

    return tuple(sorted(BY_ID))


__all__ = [
    "ARTIFACT",
    "FORMAT_MAJOR",
    "FORMAT_MINOR",
    "DecoderArtifactCorruptionError",
    "DecoderArtifactError",
    "DecoderArtifactFormatError",
    "DecoderArtifactVersionError",
    "load_decoder",
    "read_manifest",
    "save_decoder",
    "supported_types",
]
