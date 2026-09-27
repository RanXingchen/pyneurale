#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The canonical JSON form of a decoder manifest, and the values it admits.

A manifest is canonical so that two saves of the same fitted decoder produce
the same bytes: keys sorted, no insignificant whitespace, UTF-8 without escape
padding, and one trailing newline. That is what makes an artifact comparable,
diffable, and checksummable rather than merely readable.

The admitted value domain is deliberately narrow -- ``None``, ``bool``, ``int``,
``float``, ``str``, and lists and string-keyed mappings of those. Numeric data
belongs in the ``.npy`` payload, not in the manifest, and a metadata value
outside this domain is *refused at save time* rather than coerced. A format
that silently reinterprets what it was handed is a format whose round trip
cannot be trusted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._errors import DecoderArtifactFormatError

#: Scalars a manifest may carry directly. NumPy scalars are converted to their
#: Python equivalents on the way in, so a manifest never depends on NumPy's
#: repr.
_SCALARS = (bool, int, float, str)


def encode(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes of one manifest."""

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode()


def decode(data: bytes) -> dict[str, Any]:
    """Parse manifest bytes, rejecting anything that is not a JSON object."""

    try:
        payload = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise DecoderArtifactFormatError(f"the manifest is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecoderArtifactFormatError(f"the manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecoderArtifactFormatError("the manifest must be a JSON object.")
    return payload


def portable(value: object, *, path: str) -> Any:
    """Convert one metadata value into the manifest's value domain.

    Raises rather than coerces. ``path`` names where the offending value sits,
    because a rejected ``attrs`` entry is otherwise very hard to find.
    """

    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    # bool before int: bool is an int subclass, and a manifest that turned True
    # into 1 would not round-trip the flag it was handed.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecoderArtifactFormatError(
                f"{path} is {value!r}, which a manifest cannot carry. Non-finite values belong "
                "in an array payload, where the format stores them exactly."
            )
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DecoderArtifactFormatError(
                    f"{path} has a non-string key {key!r}; a manifest object is keyed by strings."
                )
            converted[key] = portable(item, path=f"{path}[{key!r}]")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [portable(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    raise DecoderArtifactFormatError(
        f"{path} is a {type(value).__name__}, which this format does not store. A decoder "
        "artifact carries JSON metadata and plain arrays; anything else would have to be "
        "pickled, and this format never is."
    )


def require(payload: Mapping[str, Any], key: str, *, path: str) -> Any:
    """Return a required manifest field, or say precisely which one is missing."""

    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object.")
    if key not in payload:
        raise DecoderArtifactFormatError(f"{path}.{key} is missing from the manifest.")
    return payload[key]


def require_typed(
    payload: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    *,
    path: str,
) -> Any:
    """Return a required field after checking its JSON type."""

    value = require(payload, key, path=path)
    # bool is an int subclass; a field declared as an int must not accept one.
    if kind is int and isinstance(value, bool):
        raise DecoderArtifactFormatError(f"{path}.{key} must be an integer; got a bool.")
    if not isinstance(value, kind):
        names = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise DecoderArtifactFormatError(
            f"{path}.{key} must be {names}; got {type(value).__name__}."
        )
    return value


def optional_text(payload: Mapping[str, Any], key: str, *, path: str) -> str | None:
    """Return an optional string field that must be present but may be null."""

    value = require(payload, key, path=path)
    if value is None or isinstance(value, str):
        return value
    raise DecoderArtifactFormatError(f"{path}.{key} must be a string or null.")


def optional_number(payload: Mapping[str, Any], key: str, *, path: str) -> float | None:
    """Return an optional finite number field that must be present but may be null."""

    value = require(payload, key, path=path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DecoderArtifactFormatError(f"{path}.{key} must be a number or null.")
    return float(value)


def text_sequence(payload: Mapping[str, Any], key: str, *, path: str) -> list[str]:
    """Return a required list-of-strings field."""

    value = require_typed(payload, key, list, path=path)
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise DecoderArtifactFormatError(f"{path}.{key}[{i}] must be a string.")
    return list(value)


__all__ = [
    "decode",
    "encode",
    "optional_number",
    "optional_text",
    "portable",
    "require",
    "require_typed",
    "text_sequence",
]
