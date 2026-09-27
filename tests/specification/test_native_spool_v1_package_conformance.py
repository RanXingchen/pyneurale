#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""``neurale.recording``'s spool reader against the normative vectors.

This is the **only** place the native-spool specification suite touches the
installed package, and it is deliberately narrow: it compares the production
reader's verdict with ``vectors/index.json`` -- the verdict section 12 says a
conforming reader must produce -- and nothing else.

What it deliberately does not do is compare the production reader with the
specification's reference reader in ``specifications/native-spool/v1/tools/``.
Two implementations agreeing proves they share an author's assumptions; both
agreeing with a frozen binary vector and a frozen expectation proves they read
the format. The vectors are the oracle, in both directions.
"""

from __future__ import annotations

import pytest
from _native_spool_v1_reference import VECTOR_BYTES, VECTOR_EXPECTATIONS

from neurale.recording._spool_format import crc32c, scan_spool

#: Known answers from section 3, so an implementation checks itself before it
#: checks a spool. A reader whose CRC is subtly wrong would report every vector
#: as corrupt and look strict rather than broken.
CRC32C_KNOWN_ANSWERS = (
    (b"", 0x00000000),
    (b"123456789", 0xE3069283),
    (bytes(32), 0x8A9136AA),
    (b"\xff" * 32, 0x62A8AB43),
)


@pytest.mark.parametrize(("data", "expected"), CRC32C_KNOWN_ANSWERS)
def test_package_crc32c_reproduces_known_answers(data: bytes, expected: int) -> None:
    assert crc32c(data) == expected


def _verdict(data: bytes) -> dict[str, object]:
    """Render one scan in the shape ``vectors/index.json`` states expectations in."""
    scan = scan_spool(data)
    end = scan.session_end
    return {
        "status": scan.status,
        "committed_prefix_end": scan.committed_prefix_end,
        "committed_transactions": scan.committed_transactions,
        "last_transaction_id": scan.last_transaction_id,
        "data_items": scan.data_items,
        "control_items": scan.control_items,
        "session_end_present": scan.session_end_present,
        "capture_outcome": end.capture_outcome_name if end else None,
        "requested_terminal_intent": end.requested_terminal_intent_name if end else None,
        "primary_fault_committed": bool(end.primary_fault_committed) if end else None,
        "terminal_reason": end.terminal_reason if end else None,
        "durability_policy": scan.superblock.durability_policy_name if scan.superblock else None,
        "durable_extent_bytes": scan.durable_extent_bytes,
        "finalizable": scan.finalizable,
        "codes": list(scan.codes),
    }


@pytest.mark.parametrize("name", sorted(VECTOR_BYTES))
def test_package_reader_reproduces_normative_verdicts(name: str) -> None:
    """Section 12: a conforming reader reproduces the verdict in the index.

    Every field is compared, not a summary of them. ``codes`` in particular is
    compared as an ordered list including repeats: two accounting identities
    that both fail are two findings, and a reader that collapsed them would
    report a spool with one broken identity and a spool with four as the same.
    """
    assert _verdict(VECTOR_BYTES[name]) == VECTOR_EXPECTATIONS[name]


def test_reader_never_promotes_past_committed_prefix() -> None:
    """The promotion rule, checked over every vector at once.

    A record is visible if and only if the transaction holding it passed every
    framing and checksum check. Stated as an invariant over the whole corpus
    rather than per vector, because the failure it guards against -- one code
    path promoting one byte too far -- shows up as a single outlier.
    """
    for name, data in VECTOR_BYTES.items():
        scan = scan_spool(data)
        if scan.superblock is None:
            assert scan.committed_prefix_end == 0, name
            assert not list(scan.records()), name
            continue
        for record in scan.records():
            assert record.transaction_offset <= record.offset, name
            assert record.payload_offset + record.payload_bytes <= scan.committed_prefix_end, name
        assert scan.durable_extent_bytes <= scan.committed_prefix_end, name


def test_reading_spool_never_mutates_it() -> None:
    """Reading, diagnosing, and finalizing are read-only over a spool.

    Cheap to check and worth checking: the reader takes bytes and the only way
    it could mutate them is by holding a mutable view, which is exactly the kind
    of accident a defensive copy exists to prevent.
    """
    for name, data in VECTOR_BYTES.items():
        mutable = bytearray(data)
        scan_spool(mutable)
        assert bytes(mutable) == data, name
