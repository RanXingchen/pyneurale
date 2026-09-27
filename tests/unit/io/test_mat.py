#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io

from neurale.exceptions import DependencyError
from neurale.io import loadmat, savemat


@pytest.fixture
def representative_mat(tmp_path: Path) -> Path:
    struct_array = np.empty(
        (1, 2),
        dtype=[("name", object), ("value", object)],
    )
    struct_array[0, 0] = ("left", np.array([1.0, 2.0]))
    struct_array[0, 1] = ("right", np.array([3.0, 4.0]))

    cells = np.empty((2, 2), dtype=object)
    cells[0, 0] = {"label": "alpha", "samples": np.array([1, 2], dtype=np.int16)}
    cells[0, 1] = np.array([[5.0, 6.0]])
    cells[1, 0] = "plain"
    cells[1, 1] = np.array([{"depth": 2}], dtype=object)

    path = tmp_path / "representative.mat"
    scipy.io.savemat(
        path,
        {
            "numeric": np.arange(6, dtype=np.float64).reshape(2, 3),
            "scalar": np.int32(7),
            "text": "hello",
            "logical": np.array([[True, False], [False, True]], dtype=np.bool_),
            "complex_values": np.array([1.0 + 2.0j, 3.0 - 4.0j]),
            "empty": np.empty((0, 2), dtype=np.float64),
            "nested": {
                "title": "outer",
                "inner": {
                    "values": np.array([[10.0, 20.0], [30.0, 40.0]]),
                    "flag": True,
                },
            },
            "struct_array": struct_array,
            "cells": cells,
        },
    )
    return path


def _assert_no_mat_struct(value: Any) -> None:
    assert not isinstance(value, scipy.io.matlab.mat_struct)
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_mat_struct(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_mat_struct(item)
    elif isinstance(value, np.ndarray) and value.dtype.hasobject:
        for item in value.flat:
            _assert_no_mat_struct(item)


def test_loadmat_squeezed_recursively_converts_structs_and_cells(
    representative_mat: Path,
) -> None:
    result = loadmat(representative_mat)

    assert {"__header__", "__version__", "__globals__"} <= result.keys()
    np.testing.assert_array_equal(result["numeric"], np.arange(6).reshape(2, 3))
    assert result["scalar"] == 7
    assert result["text"] == "hello"
    np.testing.assert_array_equal(result["logical"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(result["complex_values"], [1.0 + 2.0j, 3.0 - 4.0j])
    assert result["empty"].shape == (0,)

    assert result["nested"]["title"] == "outer"
    assert isinstance(result["nested"]["inner"]["values"], np.ndarray)
    np.testing.assert_array_equal(
        result["nested"]["inner"]["values"],
        [[10.0, 20.0], [30.0, 40.0]],
    )
    assert [item["name"] for item in result["struct_array"]] == ["left", "right"]
    assert result["cells"][0][0]["label"] == "alpha"
    assert isinstance(result["cells"][0][1], list)
    np.testing.assert_array_equal(result["cells"][0][1], [5.0, 6.0])
    assert result["cells"][1][1]["depth"] == 2
    _assert_no_mat_struct(result)


def test_loadmat_without_squeeze_preserves_scipy_shapes(
    representative_mat: Path,
) -> None:
    result = loadmat(representative_mat, squeeze_me=False)

    assert result["scalar"].shape == (1, 1)
    assert result["text"].shape == (1,)
    assert result["empty"].shape == (0, 2)
    assert isinstance(result["nested"], list)
    assert result["nested"][0][0]["inner"][0][0]["values"].shape == (2, 2)
    assert len(result["struct_array"]) == 1
    assert [item["name"][0] for item in result["struct_array"][0]] == ["left", "right"]
    assert len(result["cells"]) == 2
    assert len(result["cells"][0]) == 2
    _assert_no_mat_struct(result)


def test_savemat_delegates_mapping_and_round_trips_with_scipy(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.mat"
    data = {
        "samples": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "metadata": {"subject": "synthetic", "valid": True},
        "complex_value": np.array([2.0 + 3.0j]),
    }

    savemat(path, data)

    scipy_result = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    np.testing.assert_array_equal(scipy_result["samples"], data["samples"])
    public_result = loadmat(path)
    assert public_result["metadata"] == {"subject": "synthetic", "valid": 1}
    np.testing.assert_array_equal(public_result["complex_value"], [2.0 + 3.0j])


def test_mat_path_and_input_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        loadmat(tmp_path / "missing.mat")

    malformed = tmp_path / "malformed.mat"
    malformed.write_bytes(b"not a MAT file")
    with pytest.raises((ValueError, OSError, IndexError, scipy.io.matlab.MatReadError)):
        loadmat(malformed)

    wrong_extension = tmp_path / "wrong.bin"
    wrong_extension.write_bytes(b"not a MAT file")
    with pytest.raises(ValueError, match=r"extension '\.mat'"):
        loadmat(wrong_extension)

    missing_parent = tmp_path / "missing" / "output.mat"
    with pytest.raises(FileNotFoundError, match="output directory"):
        savemat(missing_parent, {})

    with pytest.raises(TypeError, match="must be a mapping"):
        savemat(tmp_path / "invalid.mat", [("value", 1)])  # type: ignore[arg-type]


def test_savemat_accepts_bare_relative_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    savemat("result.mat", {"sample": np.array([1.0, 2.0])})

    np.testing.assert_array_equal(loadmat("result.mat")["sample"], [1.0, 2.0])


@pytest.mark.parametrize(
    "header",
    [
        b"MATLAB 7.3 MAT-file" + b"\0" * 512,
        b"\x89HDF\r\n\x1a\n" + b"\0" * 512,
        b"\0" * 512 + b"\x89HDF\r\n\x1a\n",
    ],
)
def test_loadmat_rejects_hdf5_variants_clearly(tmp_path: Path, header: bytes) -> None:
    path = tmp_path / "v73.mat"
    path.write_bytes(header)

    with pytest.raises(NotImplementedError, match=r"v7\.3/HDF5"):
        loadmat(path)


def test_mat_api_uses_optional_dependency_error_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dependency.mat"
    path.write_bytes(b"not a MAT file")
    import neurale.io._mat as mat_module

    def unavailable(*args: object, **kwargs: object) -> object:
        raise DependencyError("scipy is required for this operation")

    monkeypatch.setattr(mat_module, "require_dependency", unavailable)
    from neurale.io import loadmat as current_loadmat

    with pytest.raises(DependencyError, match="scipy is required"):
        current_loadmat(path)


def test_public_mat_import_is_scipy_lazy() -> None:
    source_root = Path(__file__).parents[3] / "src"
    code = """
import sys
import neurale
assert 'scipy' not in sys.modules
from neurale.io import loadmat, savemat
assert callable(loadmat) and callable(savemat)
assert 'scipy' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
