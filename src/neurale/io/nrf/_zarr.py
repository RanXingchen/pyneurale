#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Zarr v3 arrays: the only chunked encoding NRF v1 defines.

Specification reference: ``README.md`` sections 2, 5, and 7.

Chunk bytes are produced by zarr-python itself, through an in-memory store,
rather than by hand-rolling the codec pipeline. That matters for two reasons:

* the checksum must cover *the exact final object bytes after codec
  processing*, so the encoder and the eventual reader must be the same
  implementation; and
* the result has to be a conforming Zarr v3 array that any Zarr reader can
  open, not merely something this package can read back.

Only the codecs the NRF codec registry declares are supported: ``bytes`` with
an explicit endianness for numeric data, and ``vlen-utf8`` for UTF-8 columns.
No compressor is applied unless the manifest declares one, because an undeclared
codec would change the committed bytes.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from neurale.runtime.dependencies import require_dependency

from ._errors import NrfCorruptionError, NrfSchemaError, NrfSemanticError

#: NRF dtype -> NumPy dtype. ``utf8`` has no fixed-width NumPy equivalent and is
#: handled through the variable-length string path.
NUMPY_DTYPES: Mapping[str, str] = {
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint32": "uint32",
    "uint64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "bool": "bool",
}

_ENDIAN_SUFFIX = {"little": "<", "big": ">", "not_applicable": "|"}


@cache
def _zarr():
    # Cached because this is called once per array access, and the probe behind
    # it is not cheap: ``require_dependency`` asks ``importlib.metadata`` for the
    # installed version, which parses the distribution's METADATA as an email
    # message. Reading one session issued this thousands of times. Whether zarr
    # is importable cannot change usefully within a process, and ``_validator``
    # in :mod:`._schemas` already resolves its own dependency the same way.
    return require_dependency("zarr", extra="nrf")


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """One physical Zarr array exactly as the frozen manifest declares it."""

    path: str
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: str
    endianness: str
    codec_ids: tuple[str, ...]
    fill_value: Any = None

    def __post_init__(self) -> None:
        if self.dtype not in NUMPY_DTYPES and self.dtype != "utf8":
            raise NrfSchemaError(f"unsupported NRF dtype {self.dtype!r}")
        if len(self.shape) != len(self.chunk_shape):
            raise NrfSemanticError(f"{self.path} shape and chunk shape have different ranks")
        if any(size <= 0 for size in self.chunk_shape):
            raise NrfSemanticError(f"{self.path} chunk shape must be positive")

    @property
    def is_text(self) -> bool:
        return self.dtype == "utf8"

    @property
    def numpy_dtype(self) -> np.dtype:
        """Return the NumPy dtype including the declared byte order."""
        if self.is_text:
            return np.dtype(object)
        base = NUMPY_DTYPES[self.dtype]
        if self.dtype in {"bool", "uint8"}:
            return np.dtype(base)
        return np.dtype(_ENDIAN_SUFFIX[self.endianness] + np.dtype(base).str[1:])

    @property
    def chunk_length(self) -> int:
        """Leading-axis chunk length, the unit every extent advances in."""
        return self.chunk_shape[0]

    def trailing_chunk_grid(self) -> tuple[int, ...]:
        """Return the number of chunks along each trailing axis."""
        return tuple(
            -(-size // chunk)
            for size, chunk in zip(self.shape[1:], self.chunk_shape[1:], strict=True)
        )

    def chunk_coordinates(self, start: int, stop: int) -> list[tuple[int, ...]]:
        """Return every chunk coordinate covering ``[start, stop)``."""
        if stop <= start:
            return []
        leading = range(start // self.chunk_length, (stop - 1) // self.chunk_length + 1)
        trailing = grid_coordinates(self.trailing_chunk_grid())
        return [(idx, *rest) for idx in leading for rest in trailing]


def grid_coordinates(counts: Sequence[int]) -> list[tuple[int, ...]]:
    """Return every coordinate of a chunk grid with *counts* chunks per axis.

    An empty *counts* yields the single empty coordinate, which is what a
    1D array's trailing grid is.
    """
    return list(itertools.product(*(range(count) for count in counts)))


def array_spec(descriptor: Mapping[str, Any], *, dtype: str, endianness: str) -> ArraySpec:
    """Return the array one manifest Zarr array descriptor declares.

    The descriptor carries the path, shape, chunk shape, and codecs; the dtype
    and byte order live on the owning stream or record field, so they are passed
    in rather than guessed from the array.
    """
    return ArraySpec(
        path=descriptor["path"],
        shape=tuple(descriptor["shape"]),
        chunk_shape=tuple(descriptor["chunk_shape"]),
        dtype=dtype,
        endianness=endianness,
        codec_ids=tuple(descriptor["codec_ids"]),
    )


def _serializer(spec: ArraySpec):
    zarr = _zarr()
    if spec.is_text:
        return zarr.codecs.VLenUTF8Codec()
    # NRF says a single-byte dtype has no byte order, but the Zarr bytes codec
    # still requires one. It cannot change the stored bytes for a one-byte
    # type, so the declaration stays "not_applicable" while the codec is given
    # the session's default.
    endian = "little" if spec.endianness == "not_applicable" else spec.endianness
    return zarr.codecs.BytesCodec(endian=endian)


def _dtype_argument(spec: ArraySpec):
    return str if spec.is_text else spec.numpy_dtype


def _fill_value(spec: ArraySpec) -> Any:
    if spec.fill_value is not None:
        return spec.fill_value
    if spec.is_text:
        return ""
    if spec.dtype == "bool":
        return False
    return 0


async def _dump_store(store) -> dict[str, bytes]:
    from zarr.core.buffer import default_buffer_prototype

    contents: dict[str, bytes] = {}
    async for key in store.list():
        buffer = await store.get(key, prototype=default_buffer_prototype())
        if buffer is not None:
            contents[key] = buffer.to_bytes()
    return contents


def _encode_in_memory(
    spec: ArraySpec, values: np.ndarray, shape: tuple[int, ...]
) -> dict[str, bytes]:
    """Create a one-chunk array in memory and return its raw store contents."""
    zarr = _zarr()
    from zarr.core.sync import sync
    from zarr.storage import MemoryStore

    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        name="a",
        shape=shape,
        chunks=shape,
        dtype=_dtype_argument(spec),
        serializer=_serializer(spec),
        compressors=None,
        fill_value=_fill_value(spec),
        # Zarr normally omits a chunk whose values all equal the fill value.
        # NRF cannot use that optimization: every advancing extent must be
        # backed by a real object with a checksum, and a legitimately all-zero
        # chunk is data, not absence.
        config={"write_empty_chunks": True},
    )
    arr[...] = values
    # Zarr's synchronous API already owns one process-wide event loop on its
    # ``zarr_io`` thread.  Creating another loop with ``asyncio.run`` for every
    # metadata snapshot and chunk dump is both redundant and unsafe on Windows:
    # each Proactor loop creates a socket pair, and repeated recorder
    # transactions can block indefinitely in CPython's fallback ``accept``.
    # Submit the store walk to the same loop used by ``create_array`` and the
    # array assignment above instead.
    return sync(_dump_store(store))


def encode_chunk(spec: ArraySpec, values: np.ndarray) -> bytes:
    """Return the exact stored bytes of one chunk of *spec*.

    *values* must already have the chunk's full shape. A short final chunk is
    padded with the array's fill value by the caller, because the specification
    represents an unwritten suffix only by the fill value.
    """
    expected = tuple(spec.chunk_shape)
    if tuple(values.shape) != expected:
        raise NrfSemanticError(
            f"{spec.path} chunk payload has shape {values.shape}, expected {expected}"
        )
    contents = _encode_in_memory(spec, values, expected)
    chunk_keys = [key for key in contents if "/c/" in key or key.endswith("/c")]
    if len(chunk_keys) != 1:
        raise NrfCorruptionError(f"{spec.path} chunk encoding produced {len(chunk_keys)} objects")
    return contents[chunk_keys[0]]


def array_metadata(spec: ArraySpec) -> dict[str, Any]:
    """Return the Zarr v3 ``zarr.json`` document for *spec*.

    Built by zarr-python for the array's full shape so the committed metadata
    snapshot and the active cache cannot disagree about codecs or fill value.
    """
    contents = _encode_in_memory(
        spec,
        _empty(spec, spec.chunk_shape),
        tuple(spec.chunk_shape),
    )
    metadata = json.loads(contents["a/zarr.json"].decode("utf-8"))
    metadata["shape"] = list(spec.shape)
    metadata["chunk_grid"]["configuration"]["chunk_shape"] = list(spec.chunk_shape)
    return metadata


def array_metadata_bytes(spec: ArraySpec) -> bytes:
    """Return the canonical serialization of the array's Zarr v3 metadata."""
    return json.dumps(array_metadata(spec), indent=None, separators=(",", ":")).encode("utf-8")


def _empty(spec: ArraySpec, shape: Sequence[int]) -> np.ndarray:
    if spec.is_text:
        return np.full(tuple(shape), "", dtype=object)
    return np.zeros(tuple(shape), dtype=spec.numpy_dtype)


def pad_to_chunk(spec: ArraySpec, values: np.ndarray) -> np.ndarray:
    """Pad *values* along the leading axis up to one full chunk.

    Only a sealing transaction may commit a short final chunk; its unwritten
    suffix is represented by the fill value and is never an observation.
    """
    rows = values.shape[0]
    if rows == spec.chunk_length:
        return values
    if rows > spec.chunk_length:
        raise NrfSemanticError(f"{spec.path} payload exceeds one chunk")
    padded = _empty(spec, (spec.chunk_length, *values.shape[1:]))
    fill = _fill_value(spec)
    if not spec.is_text:
        padded[...] = fill
    padded[:rows] = values
    return padded


def write_array_metadata(session_root: Path, spec: ArraySpec) -> None:
    """Materialize the active ``zarr.json`` cache for one array."""
    target = session_root / spec.path / "zarr.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(array_metadata_bytes(spec))


class ArrayHandles:
    """The Zarr arrays one reader has opened, released together at its close.

    Opening an array is not free: it builds a store and parses the array's
    metadata through zarr-python's async machinery. A reader issues one read per
    block, so paying that per read dominated the cost of reading a session --
    thousands of opens for one image build -- while every one of them returned a
    handle onto the same unchanging arrays.

    Deterministic release is the reason this is owned by a reader rather than
    being a module-level cache. Zarr's ZIP store retains a file handle until
    explicitly closed, and finalization publishes a package that a reader may
    just have finished reading; a cache with no
    lifetime would make that rename intermittently fail with a sharing
    violation. A reader has a lifetime, and :meth:`close` ends it.

    Holding a handle open cannot serve stale bytes: ``NrfReader`` proves the
    array's on-disk metadata still equals the manifest's frozen spec before
    every read, so metadata that changed under a held handle is refused before
    the handle is used at all.
    """

    __slots__ = ("_arrays", "_root", "_store")

    def __init__(self, session_root: Path) -> None:
        self._root = session_root
        self._store: Any = None
        self._arrays: dict[str, Any] = {}

    def array(self, spec: ArraySpec) -> Any:
        """Return the open array for *spec*, opening it on first use."""
        cached = self._arrays.get(spec.path)
        if cached is not None:
            return cached
        zarr = _zarr()
        if self._store is None:
            from zarr.storage import ZipStore

            self._store = ZipStore(str(self._root.package.path), mode="r")
        try:
            arr = zarr.open_array(store=self._store, path=spec.path, mode="r")
        except Exception as error:  # pragma: no cover - surfaced as corruption
            raise NrfCorruptionError(f"{spec.path} cannot be opened as a Zarr v3 array") from error
        self._arrays[spec.path] = arr
        return arr

    def close(self) -> None:
        """Release every handle. Idempotent."""
        self._arrays.clear()
        store, self._store = self._store, None
        if store is not None:
            store.close()


def read_range(
    session_root: Path,
    spec: ArraySpec,
    start: int,
    stop: int,
    *,
    handles: ArrayHandles | None = None,
) -> np.ndarray:
    """Read ``[start, stop)`` of one array's leading axis.

    The caller is responsible for capping *stop* at the committed extent; this
    function does not know what is committed.

    Pass *handles* to read through a reader's open arrays. Without it the array
    is opened and closed around this one read, which is what a caller that has
    no lifetime of its own must do.
    """
    if start < 0 or stop < start:
        raise NrfSemanticError(f"invalid range [{start}, {stop}) for {spec.path}")
    if stop == start:
        # An empty committed extent is a valid state that predates any chunk,
        # so this must not require the array node to exist yet.
        return _empty(spec, (0, *spec.shape[1:]))
    if handles is not None:
        return np.asarray(handles.array(spec)[start:stop])
    own = ArrayHandles(session_root)
    try:
        return np.asarray(own.array(spec)[start:stop])
    finally:
        # Zarr's ZIP store retains its package handle until explicitly
        # closed. Finalization cross-checks a sealed package and then
        # publishes it, so relying on garbage collection
        # makes publication intermittently fail with a sharing violation.
        own.close()
