#!/usr/bin/env python3

"""The automatic-dispatch regression check both signal kernel matrices run.

A dispatch matrix times every kernel over one geometry and records what the
``auto`` backend picked. Two questions follow from that, and they are the same
two whichever kernel family produced the matrix:

* did automatic dispatch pick something slower than plain builtin, by enough to
  matter rather than by timer noise -- the regression gate; and
* across a whole matrix, how much did it win by, and how far off the best
  available kernel did it land -- the speedup and regret it reports.

What differs between families is the geometry that names a case, which kernel
counts as "builtin", and how much slack the gate allows. Those are arguments.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def add_arguments(parser: argparse.ArgumentParser, *, max_regression: float) -> None:
    """Add the arguments every dispatch analysis takes."""
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--max-regression", type=float, default=max_regression)
    parser.add_argument(
        "--min-regression-seconds",
        type=float,
        default=1e-6,
        help="Ignore sub-microsecond timer noise when enforcing regressions.",
    )


def analyze(
    args: argparse.Namespace,
    *,
    key_fields: Sequence[str],
    builtin_kernel: str,
    report_auto_overhead: bool = False,
) -> None:
    """Report the matrix and exit non-zero if automatic dispatch regressed.

    @param key_fields  The CSV columns that identify one geometry; the first is
                       the thread count the report groups by.
    @param builtin_kernel  The kernel a regression is measured against.
    @param report_auto_overhead  Also report what ``auto`` itself cost against
                       the kernel it chose, which is the dispatch overhead the
                       measurement -- not the choice -- added.
    """
    grouped: dict[tuple[int, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    with args.matrix.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[name]) for name in key_fields)
            grouped[key][row["kernel"]] = row

    by_threads: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    regressions = []
    for key, rows in grouped.items():
        candidates = {
            name: float(row["warm_median_seconds"]) for name, row in rows.items() if name != "auto"
        }
        selected = rows["auto"]["selected_kernel"]
        selected_time = candidates[selected]
        builtin = candidates[builtin_kernel]
        best = min(candidates.values())
        measurement = [builtin / selected_time, selected_time / best]
        if report_auto_overhead:
            measurement.append(float(rows["auto"]["warm_median_seconds"]) / selected_time)
        by_threads[key[0]].append(tuple(measurement))
        if (
            selected_time > builtin * args.max_regression
            and selected_time - builtin > args.min_regression_seconds
        ):
            regressions.append((key, selected_time / builtin))

    for threads, measurements in sorted(by_threads.items()):
        speedups = [measurement[0] for measurement in measurements]
        regrets = [measurement[1] for measurement in measurements]
        report = (
            f"threads={threads}: "
            f"geomean_speedup={geometric_mean(speedups):.3f}x, "
            f"geomean_regret={geometric_mean(regrets):.3f}x, "
            f"max_regret={max(regrets):.3f}x"
        )
        if report_auto_overhead:
            overheads = [measurement[2] for measurement in measurements]
            report += f", measured_auto_overhead={geometric_mean(overheads):.3f}x"
        print(report)

    if regressions:
        for key, regression in regressions:
            print(f"regression {key}: {regression:.3f}x slower than builtin")
        raise SystemExit(1)
