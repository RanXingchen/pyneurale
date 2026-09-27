#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Read-only helpers for the canonical per-block provenance stream."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ._registry import BLOCK_IDX_COLUMNS


def block_index_columns() -> tuple[str, ...]:
    """Return the stable column order of a recording block-index stream."""
    return BLOCK_IDX_COLUMNS


def read_block_index(reader: Any, stream_id: str) -> Iterator[dict[str, int]]:
    """Yield rows from *stream_id*'s committed block-index companion stream."""
    block_stream = reader.read_stream(f"{stream_id}.blocks")
    for row in block_stream.data:
        yield {name: int(value) for name, value in zip(BLOCK_IDX_COLUMNS, row, strict=True)}
