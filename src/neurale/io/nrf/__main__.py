#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Command-line diagnosis and recovery for NRF v1 sessions.

``python -m neurale.io.nrf diagnose <session>.nrf`` never touches the session.
``python -m neurale.io.nrf recover <session>.nrf`` rebuilds caches and writes a
report; ``--dry-run`` prints the same report and writes nothing, so the change
can be reviewed before it is made.

Exit status is the answer to "can this session be trusted?": ``0`` for a
readable session, ``1`` when committed data is missing or unverifiable, and
``2`` when the session could not be interpreted at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ._diagnostics import SEVERITY_CORRUPT, SEVERITY_INCOMPLETE, diagnose_session
from ._errors import NrfError
from ._recovery import STATUS_UNRECOVERABLE, recover

EXIT_OK = 0
EXIT_CORRUPT = 1
EXIT_UNUSABLE = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neurale.io.nrf",
        description="Diagnose or recover a single-file NRF session.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    diagnose = commands.add_parser("diagnose", help="report what is wrong; never mutates")
    diagnose.add_argument("session", help="path to a <session>.nrf file")
    diagnose.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="skip re-hashing committed objects (keeps existence and size checks)",
    )
    diagnose.add_argument("--json", action="store_true", help="emit the findings as JSON")

    repair = commands.add_parser("recover", help="rebuild caches and write a recovery report")
    repair.add_argument("session", help="path to a <session>.nrf file")
    repair.add_argument(
        "--dry-run", action="store_true", help="print the report without writing anything"
    )
    repair.add_argument(
        "--quarantine",
        action="store_true",
        help="move staged, orphan, and index-cache objects into recovery/quarantine/",
    )
    repair.add_argument(
        "--no-verify-checksums", action="store_true", help="skip re-hashing committed objects"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tool and return its exit status."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "diagnose":
            return _diagnose(arguments)
        return _recover(arguments)
    except NrfError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_UNUSABLE


def _diagnose(arguments: argparse.Namespace) -> int:
    diagnosis = diagnose_session(
        arguments.session, verify_checksums=not arguments.no_verify_checksums
    )
    if arguments.json:
        print(
            json.dumps(
                {
                    "readable": diagnosis.readable,
                    "complete": diagnosis.complete,
                    "last_valid_journal_sequence": diagnosis.state.journal_sequence,
                    "base_checkpoint_id": diagnosis.base_checkpoint_id,
                    "committed_extents": dict(diagnosis.state.committed_extents),
                    "diagnostics": [item.as_json() for item in diagnosis.diagnostics],
                },
                indent=2,
            )
        )
    else:
        print(f"{arguments.session}: readable={diagnosis.readable} complete={diagnosis.complete}")
        for item in diagnosis.diagnostics:
            print(f"  [{item.severity}] {item.code}: {item.message}")
        if not diagnosis.diagnostics:
            print("  no findings")
    return EXIT_OK if diagnosis.readable else EXIT_CORRUPT


def _recover(arguments: argparse.Namespace) -> int:
    result = recover(
        arguments.session,
        dry_run=arguments.dry_run,
        quarantine=arguments.quarantine,
        verify_checksums=not arguments.no_verify_checksums,
    )
    print(json.dumps(dict(result.report), indent=2))
    if result.report_path is not None:
        print(f"report written to {result.report_path}", file=sys.stderr)
    if result.status == STATUS_UNRECOVERABLE:
        return EXIT_UNUSABLE
    corrupt = result.diagnosis.of_severity(SEVERITY_CORRUPT)
    incomplete = result.diagnosis.of_severity(SEVERITY_INCOMPLETE)
    if incomplete and not corrupt:
        # An unfinished recording is a fact about the session, not a failure of
        # recovery: everything committed came back intact.
        return EXIT_OK
    return EXIT_CORRUPT if corrupt else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
