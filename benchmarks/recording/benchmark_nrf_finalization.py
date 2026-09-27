#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Time repeated finalization of the same retained spool, without changing its bytes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neurale.io.nrf import NrfReader
from neurale.recording import FinalizerOptions, finalize_spool


def main(source: Path, output: Path, runs: int) -> None:
    if runs < 1:
        raise ValueError("runs must be positive")
    output.mkdir(parents=True, exist_ok=True)
    for run in range(runs):
        target = output / f"run-{run + 1}.nrf"
        started = time.perf_counter()
        report = finalize_spool(source, target, options=FinalizerOptions(spool_retention="retain"))
        elapsed = time.perf_counter() - started
        # Validation outside the measured interval, in addition to the mandatory
        # finalizer cross-check. Do not benchmark an artifact that cannot be read.
        with NrfReader.open(target, verify_checksums=True) as reader:
            complete = reader.complete
        print(
            json.dumps(
                {
                    "run": run + 1,
                    "seconds": elapsed,
                    "package_bytes": target.stat().st_size,
                    "frames": report.counts.frames,
                    "complete": complete,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    main(args.source, args.output, args.runs)
