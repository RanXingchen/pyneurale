#!/usr/bin/env python3

"""Benchmark builtin and MKL FFT kernels used by signal transforms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    TimingSummary,
    benchmark_metadata,
    measure_cold_and_warm,
    write_records,
)

import neurale._native as runtime_native

signal_native = runtime_native.signal


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _measure(
    values: np.ndarray,
    kernel: str,
    repeats: int,
    warmups: int,
) -> tuple[float, TimingSummary, str]:
    spectral = signal_native.spectral
    selected = kernel

    def call() -> None:
        nonlocal selected
        _, selected = spectral.fft(values, False, kernel)

    cold_ns, timing = measure_cold_and_warm(
        call,
        repeats,
        warmups=warmups,
    )
    return cold_ns * 1e-9, timing, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/signal_fft_benchmark.csv"),
    )
    parser.add_argument(
        "--lengths",
        default="8,16,32,64,127,128,256,1000,1024,4097,8192",
    )
    parser.add_argument("--channels", default="1,4,16,64")
    parser.add_argument("--threads", default="1,24")
    parser.add_argument("--target-elements", type=int, default=1_000_000)
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=0)
    args = parser.parse_args()

    lengths = _parse_ints(args.lengths)
    channels = _parse_ints(args.channels)
    threads = _parse_ints(args.threads)
    kernels = tuple(signal_native.spectral.fft_available_kernels())
    metadata = benchmark_metadata(runtime_native)
    rng = np.random.default_rng(0)
    rows = []
    for thread_count in threads:
        runtime_native.set_num_threads(thread_count)
        for length in lengths:
            for channel_count in channels:
                values = rng.standard_normal((length, channel_count)) + 1j * rng.standard_normal(
                    (length, channel_count)
                )
                repeats = max(
                    3,
                    min(
                        args.max_repeats,
                        args.target_elements // max(length * channel_count, 1),
                    ),
                )
                for kernel in (*kernels, "auto"):
                    cold, timing, selected = _measure(
                        values,
                        kernel,
                        repeats,
                        args.warmups,
                    )
                    row = {
                        **metadata,
                        "benchmark": "fft",
                        "provider": selected,
                        "dtype": str(values.dtype),
                        "shape": f"{length}x{channel_count}",
                        "threads": thread_count,
                        "length": length,
                        "channels": channel_count,
                        "kernel": kernel,
                        "selected_kernel": selected,
                        "cold_seconds": cold,
                        "warm_min_seconds": timing.minimum_ns * 1e-9,
                        "warm_median_seconds": timing.median_ns * 1e-9,
                        "warmups": args.warmups + 1,
                        "repeats": repeats,
                        "throughput_elements_per_second": timing.throughput(length * channel_count),
                    }
                    row.update(timing.fields("seconds", prefix="warm_"))
                    rows.append(row)
    write_records(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
