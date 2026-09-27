#!/usr/bin/env python3

"""Isolated-process ``import neurale`` time.

This one stays at the root of ``benchmarks/`` because it has no owning domain:
it measures the package import contract itself, which is what
``docs/architecture/overview.md`` specifies and ``tests/unit/test_package_import.py``
checks. The measurement includes interpreter startup and runs under ``python -I``
so every repeat is a clean process.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _harness import benchmark_metadata, timed_row, write_records

import neurale._native as runtime_native


def _import_neurale() -> None:
    subprocess.run(
        [sys.executable, "-I", "-c", "import neurale"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/import_benchmark.jsonl"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    metadata = benchmark_metadata(runtime_native)
    rows = [
        timed_row(
            metadata,
            benchmark="import",
            operation="python_isolated_import_neurale",
            function=_import_neurale,
            warmups=args.warmups,
            repeats=args.repeats,
            provider="python",
            dtype="n/a",
            shape="n/a",
            items_per_call=1,
        )
    ]

    write_records(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
