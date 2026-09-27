#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neurale.io import loadcsv, savecsv


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8", newline="")
    return path


def test_loadcsv_numeric_columns_are_float_values(tmp_path: Path) -> None:
    path = _write(tmp_path / "numeric.csv", "x,y\n1,2.5\n-3,4e2\n")

    assert loadcsv(path) == {"x": [1.0, -3.0], "y": [2.5, 400.0]}


def test_loadcsv_preserves_strings_mixed_values_and_unicode(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mixed.csv",
        "name,value,note\n神经,1.25,alpha\ncafé,not-a-number,βeta\n",
    )

    assert loadcsv(path) == {
        "name": ["神经", "café"],
        "value": [1.25, "not-a-number"],
        "note": ["alpha", "βeta"],
    }


def test_loadcsv_custom_delimiter(tmp_path: Path) -> None:
    path = _write(tmp_path / "custom.csv", "left;right\n1;two\n")

    assert loadcsv(path, delimiter=";") == {"left": [1.0], "right": ["two"]}


def test_headerless_load_uses_integer_keys(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "headerless.csv", "1,,name\n2,3,4\n5,,word\n")

    assert loadcsv(path, include_header=False) == {
        0: ["1", 2.0, 5.0],
        1: ["", 3.0],
        2: ["name", 4.0, "word"],
    }


def test_empty_cells_are_skipped_and_shorten_individual_columns(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty-cells.csv", "a,b,c\n1,,x\n,2,\n3,4,y\n")

    assert loadcsv(path) == {
        "a": [1.0, 3.0],
        "b": [2.0, 4.0],
        "c": ["x", "y"],
    }


def test_float_conversion_uses_python_float_boundaries(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "float-boundaries.csv",
        "value\n 2.5 \n1e309\nnan\n0x10\n1,000\n",
    )

    with pytest.raises(ValueError, match="row 6 has width 2"):
        loadcsv(path)

    path = _write(
        tmp_path / "float-boundaries.csv",
        "value\n 2.5 \n1e309\nnan\n0x10\nnot numeric\n",
    )
    result = loadcsv(path)["value"]
    assert result[:2] == [2.5, float("inf")]
    assert math.isnan(result[2])
    assert result[3:] == ["0x10", "not numeric"]


def test_single_row_files_follow_header_selection(tmp_path: Path) -> None:
    path = _write(tmp_path / "single.csv", "first,second\n")

    assert loadcsv(path) == {"first": [], "second": []}
    assert loadcsv(path, include_header=False) == {0: ["first"], 1: ["second"]}


def test_savecsv_pads_columns_and_load_skips(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unequal.csv"

    savecsv(path, {"long": [1, 2, 3], "short": ["x"], "empty": []})

    assert path.read_text(encoding="utf-8").splitlines() == [
        "long,short,empty",
        "1,x,",
        "2,,",
        "3,,",
    ]
    assert loadcsv(path) == {
        "long": [1.0, 2.0, 3.0],
        "short": ["x"],
        "empty": [],
    }


def test_savecsv_supports_unicode_and_custom_delimiter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unicode.csv"

    savecsv(
        path,
        {"名称": (value for value in ["通道一", "café"]), "值": [1, 2]},
        delimiter=";",
    )

    assert loadcsv(path, delimiter=";") == {
        "名称": ["通道一", "café"],
        "值": [1.0, 2.0],
    }


def test_loadcsv_rejects_empty_duplicate_and_malformed_inputs(tmp_path: Path) -> None:
    empty = _write(tmp_path / "empty.csv", "")
    with pytest.raises(ValueError, match="non-empty first row"):
        loadcsv(empty)

    duplicate = _write(tmp_path / "duplicate.csv", "same,same\n1,2\n")
    with pytest.raises(ValueError, match="header names must be unique"):
        loadcsv(duplicate)

    too_short = _write(tmp_path / "short.csv", "a,b\n1\n")
    with pytest.raises(ValueError, match="row 2 has width 1; expected 2"):
        loadcsv(too_short)

    too_wide = _write(tmp_path / "wide.csv", "a,b\n1,2,3\n")
    with pytest.raises(ValueError, match="row 2 has width 3; expected 2"):
        loadcsv(too_wide)


def test_csv_paths_and_save_inputs_are_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        loadcsv(tmp_path / "missing.csv")

    wrong_extension = _write(tmp_path / "wrong.txt", "a\n")
    with pytest.raises(ValueError, match=r"extension '\.csv'"):
        loadcsv(wrong_extension)

    with pytest.raises(FileNotFoundError, match="output directory"):
        savecsv(tmp_path / "missing" / "output.csv", {"a": [1]})

    with pytest.raises(ValueError, match="at least one column"):
        savecsv(tmp_path / "empty-output.csv", {})

    with pytest.raises(TypeError, match="must be a mapping"):
        savecsv(tmp_path / "not-mapping.csv", [("a", [1])])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="column 'a' must be iterable"):
        savecsv(tmp_path / "bad-column.csv", {"a": 1})  # type: ignore[dict-item]


def test_savecsv_accepts_bare_relative_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    savecsv("result.csv", {"sample": [1, 2]})

    assert loadcsv("result.csv") == {"sample": [1.0, 2.0]}


def test_public_csv_import_remains_light() -> None:
    source_root = Path(__file__).parents[3] / "src"
    code = """
import sys
import neurale
assert 'scipy' not in sys.modules
from neurale.io import loadcsv, loadmat, savecsv, savemat
assert all(callable(value) for value in (loadcsv, loadmat, savecsv, savemat))
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
