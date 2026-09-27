#!/usr/bin/env python3

"""Small public-API benchmark for realtime IIR and SOS calls."""

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
    IirCoefficients,
    IirFilter,
    SosCoefficients,
    SosFilter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    values = np.ascontiguousarray(
        rng.standard_normal((args.samples, args.channels)), dtype=np.float64
    )
    out = np.empty_like(values)
    iir = IirFilter(
        IirCoefficients(*scipy_signal.butter(4, 0.2)),
        n_channels=args.channels,
    )
    sos = SosFilter(
        SosCoefficients(scipy_signal.butter(4, 0.2, output="sos")),
        n_channels=args.channels,
    )

    def process_iir() -> np.ndarray:
        np.copyto(out, values)
        return iir.process(out)

    def process_sos() -> np.ndarray:
        np.copyto(out, values)
        return sos.process(out)

    cases = (
        (
            "IirFilter.process",
            process_iir,
        ),
        (
            "SosFilter.process",
            process_sos,
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
                "benchmark": "stateful_iir_sos",
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
