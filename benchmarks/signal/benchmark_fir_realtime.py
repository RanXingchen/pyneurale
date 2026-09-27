#!/usr/bin/env python3

"""Small public-API benchmark for realtime FIR calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import benchmark_metadata, emit_csv, measure_call
from scipy import signal as scipy_signal

import neurale._native as runtime_native
from neurale.signal import (
    FirCoefficients,
    FirFilter,
    fir_filter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    values = np.ascontiguousarray(
        rng.standard_normal((args.samples, args.channels)), dtype=np.float64
    )
    fir = FirCoefficients(scipy_signal.firwin(17, 0.25))
    processor = FirFilter(fir, n_channels=args.channels)
    block_out = np.empty_like(values)

    def process_frame() -> np.ndarray:
        np.copyto(block_out, values)
        return processor.process(block_out)

    cases = (
        (
            "fir_filter",
            lambda: fir_filter(
                values,
                fir,
                check_finite=False,
            ),
        ),
        (
            "FirFilter.process",
            process_frame,
        ),
    )

    metadata = benchmark_metadata(runtime_native)
    rows = []
    for name, function in cases:
        timing = measure_call(
            function,
            args.repeats,
            warmups=args.warmups,
        )
        rows.append(
            {
                **metadata,
                "benchmark": "stateful_fir",
                "operation": name,
                "provider": "public_cpu",
                "dtype": str(values.dtype),
                "shape": f"{args.samples}x{args.channels}",
                "samples": args.samples,
                "channels": args.channels,
                "warmups": args.warmups,
                "repeats": args.repeats,
                **timing.fields("us"),
                "throughput_samples_per_second": timing.throughput(args.samples * args.channels),
            }
        )
    emit_csv(rows)


if __name__ == "__main__":
    main()
