#!/usr/bin/env python3

"""Benchmark matrix for direct-form IIR and SOS execution."""

from __future__ import annotations

import argparse
import importlib
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    benchmark_metadata,
    emit_csv,
    measure_call,
    write_records,
)
from scipy import signal

from neurale.signal import (
    IirCoefficients,
    SosCoefficients,
    iir_filter,
    sos_filter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    runtime_native = importlib.import_module("neurale._native")
    native = runtime_native.signal.filtering
    kernels = tuple(native.iir_available_kernels())
    metadata = benchmark_metadata(runtime_native)
    rng = np.random.default_rng(10)
    rows = []
    for samples in (32, 256, 4096):
        for channels in (1, 4, 64, 256):
            values = rng.standard_normal((samples, channels))
            for order in (1, 4, 12, 32):
                coefs = IirCoefficients(*signal.butter(order, 0.2))
                state = np.zeros((order, channels))
                for kernel in ("public", *kernels, "auto"):
                    if kernel == "public":

                        def function(v, c, s):
                            return iir_filter(
                                v,
                                c,
                                check_finite=False,
                            )
                    else:

                        def function(v, c, s, kernel=kernel):
                            return native.iir_filter(
                                v,
                                c.b,
                                c.a,
                                s,
                                kernel,
                                False,
                            )

                    timing = measure_call(
                        partial(function, values, coefs, state),
                        args.repeats,
                        warmups=args.warmups,
                    )
                    selected = (
                        kernel
                        if kernel != "auto"
                        else native.iir_kernel_name(samples, channels, order)
                    )
                    rows.append(
                        {
                            **metadata,
                            "benchmark": "iir",
                            "form": "iir",
                            "provider": selected,
                            "dtype": str(values.dtype),
                            "shape": f"{samples}x{channels}",
                            "samples": samples,
                            "channels": channels,
                            "complexity": order,
                            "kernel": f"{kernel}:{selected}",
                            "seconds": timing.minimum_ns * 1e-9,
                            "warmups": args.warmups,
                            "repeats": args.repeats,
                            **timing.fields("seconds"),
                            "throughput_samples_per_second": timing.throughput(samples * channels),
                        }
                    )
            for sections in (1, 2, 6, 16):
                coefs = SosCoefficients(signal.butter(sections * 2, 0.2, output="sos"))
                state = np.zeros((sections, 2, channels))
                for kernel in ("public", *kernels, "auto"):
                    if kernel == "public":

                        def function(v, c, s):
                            return sos_filter(
                                v,
                                c,
                                check_finite=False,
                            )
                    else:

                        def function(v, c, s, kernel=kernel):
                            return native.sos_filter(
                                v,
                                c.sos,
                                s,
                                kernel,
                                False,
                            )

                    timing = measure_call(
                        partial(function, values, coefs, state),
                        args.repeats,
                        warmups=args.warmups,
                    )
                    selected = (
                        kernel
                        if kernel != "auto"
                        else native.sos_kernel_name(samples, channels, sections)
                    )
                    rows.append(
                        {
                            **metadata,
                            "benchmark": "sos",
                            "form": "sos",
                            "provider": selected,
                            "dtype": str(values.dtype),
                            "shape": f"{samples}x{channels}",
                            "samples": samples,
                            "channels": channels,
                            "complexity": sections,
                            "kernel": f"{kernel}:{selected}",
                            "seconds": timing.minimum_ns * 1e-9,
                            "warmups": args.warmups,
                            "repeats": args.repeats,
                            **timing.fields("seconds"),
                            "throughput_samples_per_second": timing.throughput(samples * channels),
                        }
                    )
    if args.output is None:
        emit_csv(rows)
    else:
        write_records(args.output, rows)
        print(args.output)


if __name__ == "__main__":
    main()
