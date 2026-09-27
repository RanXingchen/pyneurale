#!/usr/bin/env python3

"""Offline LMP and whole-block online LMP.

Both records carry sampling rate, cutoff, window size, and shift so the measured
input contract can be reconstructed from the JSON Lines artifact alone.
"""

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
from neurale.features import LmpProcessor, lmp_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/features_benchmark.jsonl"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
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

    fs = args.sampling_rate
    lmp_kwargs = {
        "cutoff": 200.0,
        "window_size": 0.05,
        "shift": 0.01,
    }
    case = {
        "fs_hz": fs,
        "cutoff_hz": lmp_kwargs["cutoff"],
        "window_size_s": lmp_kwargs["window_size"],
        "shift_s": lmp_kwargs["shift"],
    }
    processor = LmpProcessor(fs, args.channels, **lmp_kwargs)

    def online_lmp() -> object:
        processor.reset()
        return processor.process(values)

    rows = [
        timed_row(
            metadata,
            benchmark="lmp",
            operation="offline",
            function=lambda: lmp_features(values, fs=fs, **lmp_kwargs),
            warmups=args.warmups,
            repeats=args.repeats,
            dtype=str(values.dtype),
            shape=f"{args.samples}x{args.channels}",
            items_per_call=args.samples * args.channels,
            **case,
        ),
        timed_row(
            metadata,
            benchmark="lmp",
            operation="online_whole_block",
            function=online_lmp,
            warmups=args.warmups,
            repeats=args.repeats,
            dtype=str(values.dtype),
            shape=f"{args.samples}x{args.channels}",
            items_per_call=args.samples * args.channels,
            **case,
        ),
    ]

    write_records(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
