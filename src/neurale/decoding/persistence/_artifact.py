#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The on-disk shape of a decoder artifact: a manifest and plain arrays.

An artifact is a directory, not an archive:

.. code-block:: text

    <artifact>/
        manifest.json          canonical JSON, one trailing newline
        arrays/<name>.npy      one array per file, never pickled

Two properties are load-bearing. Arrays are ``.npy`` read with
``allow_pickle=False``, so nothing in an artifact can execute on load however
the file is spelled. And the manifest declares each array's dtype, shape, and
digest, so a file that is missing, truncated, altered, or of the wrong type
fails with a message naming the array rather than surfacing later as a wrong
answer.

Writing publishes atomically where the platform supports it: the whole
directory is built under a temporary name and renamed into place, so a reader
never observes a half-written artifact.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical import decode, encode, require, require_typed
from ._errors import DecoderArtifactCorruptionError, DecoderArtifactFormatError

MANIFEST_NAME = "manifest.json"
ARRAYS_DIRECTORY = "arrays"

#: Array element types an artifact stores. Numeric and boolean data, plus fixed
#: width text for class labels. Everything else -- object arrays above all --
#: would need pickling to round-trip, which this format never does.
_ALLOWED_KINDS = frozenset({"b", "i", "u", "f", "U"})


class ArrayWriter:
    """Collects the arrays of one artifact and records what it wrote."""

    __slots__ = ("_directory", "_entries")

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._entries: dict[str, dict[str, Any]] = {}

    def add(self, name: str, values: np.ndarray) -> str:
        """Write one array and return the name the manifest refers to it by."""

        arr = np.asarray(values)
        if arr.dtype.hasobject or arr.dtype.kind not in _ALLOWED_KINDS:
            raise DecoderArtifactFormatError(
                f"array {name!r} has dtype {arr.dtype!r}, which this format does not store. "
                "Only numeric, boolean, and fixed-width text arrays round-trip without pickling."
            )
        if name in self._entries:  # pragma: no cover - defensive
            raise DecoderArtifactFormatError(f"array {name!r} is written twice.")

        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, arr, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

        self._entries[name] = {
            "dtype": arr.dtype.str,
            "shape": [int(size) for size in arr.shape],
            "sha256": _digest(path),
        }
        return name

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        """The ``arrays`` section of the manifest, in name order."""

        return {name: self._entries[name] for name in sorted(self._entries)}

    def _path(self, name: str) -> Path:
        return self._directory / ARRAYS_DIRECTORY / f"{name}.npy"


class ArrayReader:
    """Reads the arrays of one artifact, checking each against the manifest."""

    __slots__ = ("_declared", "_directory")

    def __init__(self, directory: Path, declared: Mapping[str, Any]) -> None:
        self._directory = directory
        self._declared = declared

    def get(self, name: str) -> np.ndarray:
        """Return one declared array, verified against what the manifest says.

        The dtype, shape, and digest are all checked before the values are
        handed back, so a caller never has to ask whether what it received is
        what was written.
        """

        entry = self._declared.get(name)
        if entry is None:
            raise DecoderArtifactFormatError(
                f"array {name!r} is referenced by the manifest but not declared in its "
                "'arrays' section."
            )
        if not isinstance(entry, Mapping):
            raise DecoderArtifactFormatError(f"arrays.{name} must be a JSON object.")

        path = self._directory / ARRAYS_DIRECTORY / f"{name}.npy"
        if not path.is_file():
            raise DecoderArtifactCorruptionError(f"array {name!r} is missing from the artifact.")

        digest = require_typed(entry, "sha256", str, path=f"arrays.{name}")
        found = _digest(path)
        if found != digest:
            raise DecoderArtifactCorruptionError(
                f"array {name!r} does not match its recorded digest: the file has been "
                f"truncated or altered since it was written."
            )

        values = _load(path, name)
        expected_dtype = _declared_dtype(entry, name)
        if values.dtype != expected_dtype:
            raise DecoderArtifactCorruptionError(
                f"array {name!r} has dtype {values.dtype.str!r}; the manifest declares "
                f"{expected_dtype.str!r}."
            )
        expected_shape = _declared_shape(entry, name)
        if values.shape != expected_shape:
            raise DecoderArtifactCorruptionError(
                f"array {name!r} has shape {values.shape}; the manifest declares {expected_shape}."
            )
        return values


def _declared_dtype(entry: Mapping[str, Any], name: str) -> np.dtype:
    # A dtype string is data from the file, so it is parsed defensively: NumPy
    # raises a bare TypeError for one it does not recognize, and a malformed
    # artifact must surface as an artifact error with the field that named it.
    text = require_typed(entry, "dtype", str, path=f"arrays.{name}")
    try:
        dtype = np.dtype(text)
    except TypeError as exc:
        raise DecoderArtifactFormatError(
            f"arrays.{name}.dtype is {text!r}, which is not a NumPy dtype: {exc}"
        ) from exc
    if dtype.hasobject or dtype.kind not in _ALLOWED_KINDS:
        raise DecoderArtifactFormatError(
            f"arrays.{name}.dtype is {text!r}, which this format never stores. Only numeric, "
            "boolean, and fixed-width text arrays round-trip without pickling."
        )
    return dtype


def _declared_shape(entry: Mapping[str, Any], name: str) -> tuple[int, ...]:
    values = require_typed(entry, "shape", list, path=f"arrays.{name}")
    for pos, size in enumerate(values):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DecoderArtifactFormatError(
                f"arrays.{name}.shape[{pos}] must be a non-negative integer; got {size!r}."
            )
    return tuple(int(size) for size in values)


def _load(path: Path, name: str) -> np.ndarray:
    with path.open("rb") as handle:
        try:
            values = np.load(handle, allow_pickle=False)
        except ValueError as exc:
            # NumPy raises ValueError both for a pickled payload under
            # allow_pickle=False and for a header it cannot parse. Either way
            # the artifact does not hold a plain array here.
            raise DecoderArtifactCorruptionError(
                f"array {name!r} could not be read as a plain array: {exc}. This format never "
                "unpickles, so a pickled payload is refused rather than executed."
            ) from exc
        except (OSError, EOFError) as exc:
            raise DecoderArtifactCorruptionError(
                f"array {name!r} is truncated or unreadable: {exc}"
            ) from exc
    if not isinstance(values, np.ndarray):  # pragma: no cover - defensive
        raise DecoderArtifactCorruptionError(f"array {name!r} is not an array.")
    return values


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


@contextmanager
def publishing(path: Path, *, overwrite: bool) -> Iterator[Path]:
    """Build an artifact under a temporary name and publish it atomically.

    The body writes into the yielded directory. On success the directory is
    renamed onto ``path``; on any failure the temporary directory is removed and
    an existing artifact at ``path`` is left exactly as it was.

    An overwrite moves the previous artifact aside and renames the new one into
    place. Both steps are atomic individually, which is as far as a directory
    publish goes on a POSIX filesystem: there is no rename that replaces a
    non-empty directory in one operation.
    """

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists. Pass overwrite=True to replace it; a save never "
            "silently discards an artifact."
        )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = parent / f".{destination.name}.saving-{token}"
    superseded = parent / f".{destination.name}.superseded-{token}"

    staging.mkdir()
    try:
        yield staging
        _sync_directory(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    moved = False
    if destination.exists():
        os.replace(destination, superseded)
        moved = True
    try:
        os.replace(staging, destination)
    except OSError:
        if moved:
            os.replace(superseded, destination)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if moved:
        shutil.rmtree(superseded, ignore_errors=True)
    _sync_directory(parent)


def write_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Write the canonical manifest of an artifact under construction."""

    path = directory / MANIFEST_NAME
    with path.open("wb") as handle:
        handle.write(encode(manifest))
        handle.flush()
        os.fsync(handle.fileno())


def read_manifest_bytes(path: Path) -> dict[str, Any]:
    """Read and parse the manifest of an artifact directory."""

    directory = Path(path)
    if not directory.is_dir():
        raise DecoderArtifactFormatError(
            f"{directory} is not a decoder artifact directory. An artifact is a directory "
            f"holding a {MANIFEST_NAME} and an {ARRAYS_DIRECTORY}/ payload."
        )
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        raise DecoderArtifactFormatError(f"{directory} has no {MANIFEST_NAME}.")
    return decode(manifest.read_bytes())


def declared_arrays(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the manifest's ``arrays`` section."""

    entries = require(manifest, "arrays", path="manifest")
    if not isinstance(entries, Mapping):
        raise DecoderArtifactFormatError("manifest.arrays must be a JSON object.")
    return entries


def _sync_directory(path: Path) -> None:
    # Renaming a directory that has not been synced can publish an entry whose
    # contents are not durable yet. Not every platform lets a directory be
    # opened for fsync, so a failure here is not fatal.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(fd)


__all__ = [
    "ARRAYS_DIRECTORY",
    "MANIFEST_NAME",
    "ArrayReader",
    "ArrayWriter",
    "declared_arrays",
    "publishing",
    "read_manifest_bytes",
    "write_manifest",
]
