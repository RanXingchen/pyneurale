#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Dictionary-of-columns CSV compatibility helpers."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any

CsvKey = str | int
CsvValue = float | str


def loadcsv(
    file_path: str | PathLike[str],
    delimiter: str = ",",
    include_header: bool = True,
) -> dict[CsvKey, list[CsvValue]]:
    """Load a CSV file as an insertion-ordered dictionary of columns.

    With ``include_header=True``, the first row supplies string keys and is not
    included as data. With ``include_header=False``, keys are zero-based
    integers and every first-row cell is retained verbatim as a string.

    For subsequent rows, each non-empty cell is converted with ``float`` when
    possible and otherwise retained as ``str``. An empty cell is skipped
    completely; no placeholder is appended, so
    columns containing empty cells become shorter.

    Rows must all have the first row's width. Empty files and duplicate headers
    are rejected rather than producing an indirect ``StopIteration`` or
    silently overwriting an earlier column.
    """

    path = _validate_csv_path(file_path, mode="read")
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        first_row = next(reader, None)
        if first_row is None or not first_row:
            raise ValueError("CSV input must contain a non-empty first row.")

        if include_header:
            if len(set(first_row)) != len(first_row):
                raise ValueError("CSV header names must be unique.")
            result: dict[CsvKey, list[CsvValue]] = {header: [] for header in first_row}
        else:
            result = {idx: [content] for idx, content in enumerate(first_row)}

        keys = tuple(result)
        expected_width = len(first_row)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != expected_width:
                raise ValueError(
                    f"CSV row {row_number} has width {len(row)}; expected {expected_width}."
                )
            for key, cell in zip(keys, row, strict=True):
                if cell:
                    result[key].append(_parse_cell(cell))
        return result


def savecsv(
    file_path: str | PathLike[str],
    data_dict: Mapping[Any, Iterable[Any]],
    delimiter: str = ",",
) -> None:
    """Save a mapping of columns, padding shorter columns with empty cells.

    Column iterables are materialized once before writing. Empty mappings and
    non-iterable column values receive explicit validation errors; the old
    implementation failed indirectly through ``max``, ``len``, or indexing.
    """

    path = _validate_csv_path(file_path, mode="write")
    if not isinstance(data_dict, Mapping):
        raise TypeError("data_dict must be a mapping of headers to column values.")
    if not data_dict:
        raise ValueError("data_dict must contain at least one column.")

    columns: dict[Any, tuple[Any, ...]] = {}
    for key, values in data_dict.items():
        try:
            columns[key] = tuple(values)
        except TypeError as exc:
            raise TypeError(f"CSV column {key!r} must be iterable.") from exc

    longest = max(len(values) for values in columns.values())
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(columns)
        for i in range(longest):
            writer.writerow(values[i] if i < len(values) else "" for values in columns.values())


def _parse_cell(value: str) -> CsvValue:
    try:
        return float(value)
    except ValueError:
        return value


def _validate_csv_path(
    file_path: str | PathLike[str],
    *,
    mode: str,
) -> str:
    path = os.fspath(file_path)
    if not isinstance(path, str):
        raise TypeError("file_path must resolve to a text filesystem path.")

    if mode == "read":
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file does not exist: {path}")
    elif mode == "write":
        parent = os.path.dirname(path) or "."
        if not os.path.exists(parent):
            raise FileNotFoundError(f"CSV output directory does not exist: {parent}")
    else:  # pragma: no cover - private programming error
        raise ValueError("mode must be 'read' or 'write'.")

    extension = os.path.splitext(path)[1]
    if extension != ".csv":
        raise ValueError(f"file_path must have extension '.csv', got {extension!r}.")
    return path


__all__ = ["loadcsv", "savecsv"]
