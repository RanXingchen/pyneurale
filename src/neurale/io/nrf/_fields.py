#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Record-schema field descriptors and the codec choices they freeze.

Split out of :mod:`._manifest` so that a module which only builds record
schemas does not have to import the manifest builder. :mod:`._manifest` builds
manifests *and* validates them, so it imports :mod:`._semantics`; a schema
builder that reached for :func:`record_field` therefore could not be imported
from the semantic layer without a cycle. Nothing here validates or reads a
manifest, so this module sits below both.

:mod:`._manifest` re-exports every name defined here, which is what keeps
``neurale.io.nrf.record_field`` and the codec constants where callers already
find them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._paths import record_column_path, record_validity_path

BYTES_LE_CODEC = "bytes-le"
UTF8_CODEC = "utf8-vlen"


def codec_for(dtype: str) -> str:
    """Return the codec ID the default registry uses for *dtype*."""
    return UTF8_CODEC if dtype == "utf8" else BYTES_LE_CODEC


def endianness_for(dtype: str) -> str:
    """Return the byte order NRF requires for *dtype*."""
    return "not_applicable" if dtype in {"bool", "uint8", "utf8"} else "little"


def record_field(
    record_path: str,
    name: str,
    dtype: str,
    *,
    nullable: bool = False,
    unit_id: str | None = None,
    reference: str | None = None,
    shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return one record-schema field descriptor with its frozen array paths."""
    descriptor: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "endianness": endianness_for(dtype),
        "nullable": nullable,
        "array_path": record_column_path(record_path, name),
        "codec_ids": [codec_for(dtype)],
    }
    if nullable:
        descriptor["validity_path"] = record_validity_path(record_path, name)
    if unit_id is not None:
        descriptor["unit_id"] = unit_id
    if reference is not None:
        descriptor["reference"] = reference
    if shape:
        descriptor["shape"] = list(shape)
    return descriptor
