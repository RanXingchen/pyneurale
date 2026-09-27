#!/usr/bin/env python3

"""Benchmark native FIR kernels across frame, channel, tap, and thread counts."""

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
    taps: np.ndarray,
    state: np.ndarray,
    kernel: str,
    repeats: int,
    warmups: int,
) -> tuple[float, TimingSummary, str]:
    filtering = signal_native.filtering
    selected = kernel

    def call() -> None:
        nonlocal selected
        _, _, selected = filtering.fir_filter(values, taps, state, kernel)

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
        default=Path("build/fir_benchmark.csv"),
    )
    parser.add_argument("--samples", default="32,64,96,128,192,256,512,2048,8192")
    parser.add_argument("--channels", default="1,4,16,64")
    parser.add_argument("--taps", default="5,17,33,65,129,257,1025")
    parser.add_argument("--threads", default="1,24")
    parser.add_argument("--target-operations", type=int, default=10_000_000)
    parser.add_argument("--max-repeats", type=int, default=20)
    parser.add_argument("--max-operations", type=int, default=100_000_000)
    parser.add_argument("--warmups", type=int, default=0)
    args = parser.parse_args()

    samples_values = _parse_ints(args.samples)
    channel_values = _parse_ints(args.channels)
    tap_values = _parse_ints(args.taps)
    thread_values = _parse_ints(args.threads)
    filtering = signal_native.filtering
    kernels = tuple(filtering.fir_available_kernels())
    metadata = benchmark_metadata(runtime_native)
    rng = np.random.default_rng(0)
    rows = []
    for threads in thread_values:
        runtime_native.set_num_threads(threads)
        for samples in samples_values:
            for channels in channel_values:
                for n_taps in tap_values:
                    operations = samples * channels * n_taps
                    if operations > args.max_operations:
                        continue
                    values = rng.standard_normal((samples, channels))
                    taps = rng.standard_normal(n_taps)
                    state = np.zeros((n_taps - 1, channels))
                    repeats = max(
                        3,
                        min(
                            args.max_repeats,
                            args.target_operations // max(operations, 1),
                        ),
                    )
                    for kernel in (*kernels, "auto"):
                        cold, timing, selected = _measure(
                            values,
                            taps,
                            state,
                            kernel,
                            repeats,
                            args.warmups,
                        )
                        row = {
                            **metadata,
                            "benchmark": "fir",
                            "provider": selected,
                            "dtype": str(values.dtype),
                            "shape": f"{samples}x{channels}",
                            "threads": threads,
                            "samples": samples,
                            "channels": channels,
                            "taps": n_taps,
                            "operations": operations,
                            "kernel": kernel,
                            "selected_kernel": selected,
                            "cold_seconds": cold,
                            "warm_min_seconds": timing.minimum_ns * 1e-9,
                            "warm_median_seconds": timing.median_ns * 1e-9,
                            "warmups": args.warmups + 1,
                            "repeats": repeats,
                            "throughput_operations_per_second": timing.throughput(operations),
                        }
                        row.update(timing.fields("seconds", prefix="warm_"))
                        rows.append(row)

    write_records(args.output, rows)

    print(args.output)


if __name__ == "__main__":
    main()
