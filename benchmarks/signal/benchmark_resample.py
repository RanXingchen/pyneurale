#!/usr/bin/env python3

"""Public 2:3 resampling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The shared benchmark helpers stay at the root of ``benchmarks/``; this script
# lives one level down, in the directory of the domain that owns it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import benchmark_metadata, timed_row, write_records

import neurale._native as runtime_native
from neurale.signal import resample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/signal_benchmark.jsonl"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    metadata = {
        **benchmark_metadata(runtime_native),
        "input_seed": args.seed,
    }
    rng = np.random.default_rng(args.seed)
    values = np.ascontiguousarray(
        rng.standard_normal((args.samples, args.channels)),
        dtype=np.float64,
    )

    rows = [
        timed_row(
            metadata,
            benchmark="resampler",
            operation="resample_2_3",
            function=lambda: resample(values, 2, 3),
            warmups=args.warmups,
            repeats=args.repeats,
            dtype=str(values.dtype),
            shape=f"{args.samples}x{args.channels}",
            items_per_call=args.samples * args.channels,
            up=2,
            down=3,
        )
    ]

    write_records(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
