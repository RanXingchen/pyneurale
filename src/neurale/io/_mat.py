#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""SciPy-based MATLAB compatibility I/O.

This module intentionally implements only MAT interoperability, not a
PyNeurale recording format. Recursive conversion reaches structs inside
top-level object/struct arrays and preserves ordinary numeric ndarrays nested in
structs. Cell/object arrays remain ordinary nested
Python lists. MATLAB v7.3/HDF5 files receive an explicit unsupported-format
error rather than SciPy's backend-specific failure.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from os import PathLike
from typing import Any

import numpy as np

from neurale.runtime.dependencies import require_dependency

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_MATLAB_73_HEADER = b"MATLAB 7.3 MAT-file"


def loadmat(
    file_path: str | PathLike[str],
    squeeze_me: bool = True,
) -> dict[str, Any]:
    """Load a SciPy-supported MAT file into ordinary Python containers.

    SciPy is called with ``struct_as_record=False`` and the caller-selected
    ``squeeze_me`` value. MATLAB structs become dictionaries, while cell and
    object arrays become shape-preserving nested Python lists. Numeric,
    logical, character, complex, empty, and multidimensional arrays remain in
    the representation returned by SciPy unless they are object containers
    requiring recursive conversion. SciPy metadata entries such as
    ``__header__``, ``__version__``, and ``__globals__`` are preserved.

    MATLAB v7.3/HDF5 files are not supported by SciPy's MAT reader and raise
    ``NotImplementedError`` with an explicit message.
    """

    path = _validate_mat_path(file_path, mode="read")
    if _is_hdf5_mat(path):
        raise NotImplementedError(
            "MATLAB v7.3/HDF5 MAT files are not supported by neurale.io.loadmat; "
            "convert the file to a SciPy-supported MAT version first."
        )

    scipy_io, mat_struct_type = _scipy_mat_api()
    try:
        loaded = scipy_io.loadmat(
            path,
            squeeze_me=squeeze_me,
            struct_as_record=False,
        )
    except NotImplementedError as exc:
        raise NotImplementedError(
            "This MAT variant is not supported by neurale.io.loadmat; MATLAB v7.3/HDF5 "
            "files require a separate HDF5 reader."
        ) from exc
    return {key: _convert_mat_value(value, mat_struct_type) for key, value in loaded.items()}


def savemat(file_path: str | PathLike[str], data: Mapping[str, Any]) -> None:
    """Save a mapping through :func:`scipy.io.savemat` without a custom schema."""

    path = _validate_mat_path(file_path, mode="write")
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping of MATLAB variable names to values.")
    scipy_io, _ = _scipy_mat_api()
    scipy_io.savemat(path, data)


def _scipy_mat_api() -> tuple[Any, type[Any]]:
    scipy_io = require_dependency("scipy", import_name="scipy.io")
    scipy_matlab = require_dependency("scipy", import_name="scipy.io.matlab")
    return scipy_io, scipy_matlab.mat_struct


def _convert_mat_value(value: Any, mat_struct_type: type[Any]) -> Any:
    if isinstance(value, mat_struct_type):
        field_names = value._fieldnames or ()
        return {
            name: _convert_mat_value(getattr(value, name), mat_struct_type) for name in field_names
        }
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        return _convert_object_container(value.tolist(), mat_struct_type)
    return value


def _convert_object_container(value: Any, mat_struct_type: type[Any]) -> Any:
    if isinstance(value, list):
        return [_convert_object_container(item, mat_struct_type) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _convert_object_container(value.item(), mat_struct_type)
        return [_convert_object_container(item, mat_struct_type) for item in value]
    return _convert_mat_value(value, mat_struct_type)


def _validate_mat_path(
    file_path: str | PathLike[str],
    *,
    mode: str,
) -> str:
    path = os.fspath(file_path)
    if not isinstance(path, str):
        raise TypeError("file_path must resolve to a text filesystem path.")

    if mode == "read":
        if not os.path.exists(path):
            raise FileNotFoundError(f"MAT file does not exist: {path}")
    elif mode == "write":
        parent = os.path.dirname(path) or "."
        if not os.path.exists(parent):
            raise FileNotFoundError(f"MAT output directory does not exist: {parent}")
    else:  # pragma: no cover - private programming error
        raise ValueError("mode must be 'read' or 'write'.")

    extension = os.path.splitext(path)[1]
    if extension != ".mat":
        raise ValueError(f"file_path must have extension '.mat', got {extension!r}.")
    return path


def _is_hdf5_mat(path: str) -> bool:
    with open(path, "rb") as stream:
        header = stream.read(520)
    return (
        header.startswith((_MATLAB_73_HEADER, _HDF5_SIGNATURE))
        or header[512:520] == _HDF5_SIGNATURE
    )


__all__ = ["loadmat", "savemat"]
