#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Canonical relative paths and namespace ownership for NRF v1.

Specification reference: ``README.md`` section 2, ``semantic-validation.md``
rules NRF-SEM-015, NRF-SEM-016, NRF-TX-010, and NRF-TX-012.

The single most important rule here: a reader **rejects** a non-canonical path
rather than normalizing it. Normalizing would let two spellings of the same
path both appear valid, which breaks object ownership, checksum attribution,
and the create-if-absent promotion that makes a commit atomic.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

from ._errors import NrfSemanticError

# One canonical spelling: lower-case ASCII letters, digits, '.', '_', '-',
# separated by single '/'. No leading/trailing slash and no empty segment.
_SEGMENT_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")

WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)

CONTROL_FILES = frozenset({"manifest.json", "zarr.json"})

#: Top-level trees that cannot hold manifest-owned payload arrays.
CONTROL_NAMESPACES = frozenset(
    {
        ".staging",
        "journal",
        "metadata",
        "recovery",
        "checksums",
        "indexes",
        "feature_sets",
    }
)

#: Top-level trees that hold manifest-owned payload arrays.
PAYLOAD_NAMESPACES = frozenset({"streams", "records"})

STAGING_ROOT = ".staging"
JOURNAL_TRANSACTIONS = "journal/transactions.jsonl"
JOURNAL_HEAD = "journal/head.json"
JOURNAL_CHECKPOINTS = "journal/checkpoints"
MANIFEST = "manifest.json"
SESSION_TERMINATION_PATH = "records/session_termination/termination-v1"


def is_canonical_relative_path(path: object) -> bool:
    """Return whether *path* is already in the one canonical NRF spelling."""
    if not isinstance(path, str) or not path:
        return False
    posix = PurePosixPath(path)
    if path != posix.as_posix() or posix.is_absolute() or PureWindowsPath(path).drive:
        return False
    for segment in path.split("/"):
        if not segment or segment in {".", ".."}:
            return False
        if segment.endswith("."):
            return False
        if not _SEGMENT_CHARACTERS.issuperset(segment):
            return False
        if segment.split(".", maxsplit=1)[0] in WINDOWS_RESERVED_BASENAMES:
            return False
    return True


def require_canonical_relative_path(path: object, label: str) -> str:
    """Return *path* unchanged, or raise if it is not canonical.

    Raises
    ------
    NrfSemanticError
        If *path* is not the canonical spelling. It is never normalized.
    """
    if not is_canonical_relative_path(path):
        raise NrfSemanticError(f"{label} is not a canonical relative NRF path: {path!r}")
    assert isinstance(path, str)
    return path


def portable_path_key(path: str) -> tuple[str, ...]:
    """Return the cross-platform comparison key for a validated NRF path.

    Ownership and collision checks run on this key so that two paths which
    would collide on a case-insensitive or trailing-dot-stripping filesystem
    are treated as the same object even though a conforming path is already
    lower-case and has neither suffix.
    """
    return tuple(segment.rstrip(" .").casefold() for segment in path.split("/"))


def require_payload_path(path: object, label: str) -> str:
    """Return *path* if it may hold a manifest-owned payload array."""
    value = require_canonical_relative_path(path, label)
    top_level = value.split("/", maxsplit=1)[0]
    if value in CONTROL_FILES or top_level in CONTROL_NAMESPACES:
        raise NrfSemanticError(f"{label} occupies a reserved NRF control path: {value!r}")
    if top_level not in PAYLOAD_NAMESPACES:
        raise NrfSemanticError(f"{label} must live under {sorted(PAYLOAD_NAMESPACES)}: {value!r}")
    return value


def paths_overlap(left: str, right: str) -> bool:
    """Return whether either validated path is an ancestor of the other.

    Two physical arrays may not nest: a chunk of one would then be inside the
    node of the other, which makes ownership ambiguous.
    """
    left_key = portable_path_key(left)
    right_key = portable_path_key(right)
    shortest = min(len(left_key), len(right_key))
    return left_key[:shortest] == right_key[:shortest]


def stream_data_path(stream_id: str) -> str:
    """Return the frozen payload path for a stream's data array."""
    return f"streams/{stream_id}/data"


def stream_timestamps_path(stream_id: str) -> str:
    """Return the frozen payload path for a stream's explicit timestamps."""
    return f"streams/{stream_id}/timestamps"


def stream_discontinuities_path(stream_id: str) -> str:
    """Return the frozen record-set path for a stream's discontinuities."""
    return f"streams/{stream_id}/discontinuities"


def record_set_path(record_kind: str, schema_id: str) -> str:
    """Return the frozen record-set path for a non-discontinuity record kind."""
    return f"records/{record_kind}/{schema_id}"


def record_column_path(record_set: str, field_name: str) -> str:
    """Return the exact column array path required by NRF-SEM-015."""
    return f"{record_set}/columns/{field_name}"


def record_validity_path(record_set: str, field_name: str) -> str:
    """Return the exact validity array path required by NRF-SEM-015."""
    return f"{record_set}/validity/{field_name}"


def chunk_path(array_path: str, coordinate: tuple[int, ...]) -> str:
    """Return the final path of one chunk object (NRF-TX-012)."""
    return f"{array_path}/c/" + "/".join(str(idx) for idx in coordinate)


def metadata_snapshot_path(transaction_id: str, metadata_object_id: str) -> str:
    """Return the transaction-unique metadata snapshot path (NRF-TX-009)."""
    return f"metadata/transactions/{transaction_id}/{metadata_object_id}.json"


def staged_path(transaction_id: str, final_path: str) -> str:
    """Return the only staging path a prepared object may use.

    The specification fixes this as ``.staging/<transaction-id>/<final-path>``
    so every promotion is a one-to-one create-if-absent rename.
    """
    return f"{STAGING_ROOT}/{transaction_id}/{final_path}"


def transaction_id(number: int) -> str:
    """Return the ``tx-`` identifier for transaction *number*."""
    if number < 0:
        raise NrfSemanticError("transaction number must be non-negative")
    return f"tx-{number:016d}"


def checkpoint_id(number: int) -> str:
    """Return the ``checkpoint-`` identifier for checkpoint *number*."""
    if number < 0:
        raise NrfSemanticError("checkpoint number must be non-negative")
    return f"checkpoint-{number:016d}"
