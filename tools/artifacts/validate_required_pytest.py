# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Reject a required pytest JUnit result with skipped or missing tests."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def validate_required_pytest(path: Path, minimum: int) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit report contains no test suites")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    failed = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    if tests < minimum or skipped or failed or errors:
        raise ValueError(
            f"required tests did not pass: tests={tests}, skipped={skipped}, "
            f"failures={failed}, errors={errors}; minimum={minimum}"
        )
    return tests, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum", type=int, required=True)
    args = parser.parse_args()
    try:
        tests, skipped = validate_required_pytest(args.report, args.minimum)
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"required pytest validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"validated required pytest result: {tests} tests, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
