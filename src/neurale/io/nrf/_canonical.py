#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""RFC 8785 canonical JSON, the I-JSON domain, and SHA-256 checksums.

Specification reference: ``specifications/nrf/v1/README.md`` section 7.

Two rules here are easy to get wrong and are the reason this module exists
instead of ``json.dumps(..., sort_keys=True)``:

* Ordinary key-sorted JSON is explicitly *not* conforming. RFC 8785 sorts
  property names by UTF-16 code units and has its own number serialization, so
  the checksum input must come from a real JCS implementation.
* A ``record_checksum`` covers the canonical JSON of its own object with the
  ``record_checksum`` member removed -- not the stored bytes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from neurale.runtime.dependencies import require_dependency

from ._errors import NrfSchemaError

# RFC 8785 / I-JSON exact integer range. Anything wider belongs in a typed Zarr
# array, or in a schema-defined decimal string when it must live in checksummed
# JSON.
SAFE_INTEGER_MIN = -9007199254740991
SAFE_INTEGER_MAX = 9007199254740991

_CHECKSUM_FIELD = "record_checksum"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# UTC only, mandatory trailing Z, seconds 00..59. Leap seconds and year zero are
# not supported by NRF v1.
_UTC_DATE_TIME = re.compile(
    r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])"
    r"T(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)(\.\d+)?Z$"
)
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@cache
def _rfc8785():
    # Same reason as ``_zarr`` in :mod:`._zarr`: this runs once per canonical
    # encode, and the dependency probe behind it parses distribution metadata.
    return require_dependency("rfc8785", extra="nrf")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 encoding of *value*."""
    return _rfc8785().dumps(value)


def canonical_json(value: Any) -> str:
    """Return the RFC 8785 canonical encoding of *value* as text."""
    return canonical_json_bytes(value).decode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lower-case hexadecimal SHA-256 digest of *data*.

    SHA-256 is the only checksum algorithm NRF v1 defines.
    """
    return hashlib.sha256(data).hexdigest()


def is_sha256_hex(value: object) -> bool:
    """Return whether *value* is a 64-character lower-case hex digest."""
    return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None


def is_leap_year(year: int) -> bool:
    """Return whether *year* is a Gregorian leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def is_utc_date_time(value: object) -> bool:
    """Return whether *value* is a valid NRF v1 UTC ``date-time``.

    The regular expression bounds the components; this also rejects impossible
    calendar days such as ``2026-02-30``, which a pattern alone cannot catch.
    """
    if not isinstance(value, str):
        return False
    match = _UTC_DATE_TIME.fullmatch(value)
    if match is None:
        return False
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    if year == 0:
        return False
    days = _DAYS_IN_MONTH[month - 1]
    if month == 2 and is_leap_year(year):
        days = 29
    return day <= days


def check_ijson_domain(value: Any, path: str = "$") -> None:
    """Recursively reject values RFC 8785 cannot canonicalize interoperably.

    JSON Schema bounds the integers it names, but free-form ``metadata``,
    ``extensions``, and Zarr ``fill_value`` members can carry an out-of-range
    integer or a non-finite float that no schema can constrain.

    Raises
    ------
    NrfSchemaError
        If a duplicate property name, non-finite number, or out-of-range
        integer is present.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if not SAFE_INTEGER_MIN <= value <= SAFE_INTEGER_MAX:
            raise NrfSchemaError(
                f"{path} is outside the I-JSON exact integer range; "
                "wide integers belong in a typed array or a decimal string"
            )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NrfSchemaError(f"{path} is a non-finite number")
        return
    if isinstance(value, Mapping):
        seen: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                raise NrfSchemaError(f"{path} has a non-string property name")
            if key in seen:
                raise NrfSchemaError(f"{path} has the duplicate property name {key!r}")
            seen.add(key)
            check_ijson_domain(item, f"{path}.{key}")
        return
    if isinstance(value, (str, bytes)):
        return
    if isinstance(value, Sequence):
        for i, item in enumerate(value):
            check_ijson_domain(item, f"{path}[{i}]")
        return


def _without_checksum(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != _CHECKSUM_FIELD}


def record_checksum(record: Mapping[str, Any]) -> str:
    """Return the ``record_checksum`` for a journal, checkpoint, or report record.

    The digest covers the canonical JSON of *record* with any existing
    ``record_checksum`` member removed, so the value can be recomputed from a
    stored record without knowing how it was serialized.
    """
    payload = _without_checksum(record)
    check_ijson_domain(payload)
    return sha256_hex(canonical_json_bytes(payload))


def sign_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return *record* with a freshly computed ``record_checksum``."""
    signed = _without_checksum(record)
    signed[_CHECKSUM_FIELD] = record_checksum(signed)
    return signed


def verify_record_checksum(record: Mapping[str, Any]) -> bool:
    """Return whether *record* carries a matching ``record_checksum``."""
    stored = record.get(_CHECKSUM_FIELD)
    if not is_sha256_hex(stored):
        return False
    try:
        return record_checksum(record) == stored
    except NrfSchemaError:
        return False


def encode_journal_line(record: Mapping[str, Any]) -> bytes:
    """Return one JSONL journal line: canonical JSON plus a single LF byte.

    A partial line is invalid and invisible, so the terminating LF is part of
    the record rather than a separator added by the caller.
    """
    return canonical_json_bytes(record) + b"\n"
