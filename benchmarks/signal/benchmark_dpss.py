#!/usr/bin/env python3

"""Measure public DPSS generation with and without cache hits."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    TimingSummary,
    benchmark_metadata,
    summarize_ns,
    write_records,
)

import neurale._native as runtime_native
from neurale.signal.windows import _cached_dpss, dpss


def _parse_lengths(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _measure(
    length: int,
    time_bandwidth: float,
    n_tapers: int,
    repeats: int,
) -> tuple[TimingSummary, TimingSummary]:
    uncached = []
    for _ in range(repeats):
        _cached_dpss.cache_clear()
        start = time.perf_counter_ns()
        dpss(
            length,
            time_bandwidth,
            n_tapers,
        )
        uncached.append(time.perf_counter_ns() - start)

    _cached_dpss.cache_clear()
    dpss(length, time_bandwidth, n_tapers)
    cached = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        dpss(
            length,
            time_bandwidth,
            n_tapers,
        )
        cached.append(time.perf_counter_ns() - start)
    return summarize_ns(uncached), summarize_ns(cached)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/signal_dpss_benchmark.csv"),
    )
    parser.add_argument("--lengths", default="256,1024,4096,16384")
    parser.add_argument("--time-bandwidth", type=float, default=4.0)
    parser.add_argument("--n-tapers", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    metadata = benchmark_metadata(runtime_native)
    rows = []
    for length in _parse_lengths(args.lengths):
        uncached, cached = _measure(
            length,
            args.time_bandwidth,
            args.n_tapers,
            args.repeats,
        )
        rows.append(
            {
                **metadata,
                "benchmark": "dpss",
                "provider": "native_cpu",
                "dtype": "float64",
                "shape": f"{length}x{args.n_tapers}",
                "length": length,
                "time_bandwidth": args.time_bandwidth,
                "n_tapers": args.n_tapers,
                "warmups": 1,
                "repeats": args.repeats,
                "uncached_seconds": uncached.median_ns * 1e-9,
                "cached_seconds": cached.median_ns * 1e-9,
                **uncached.fields("seconds", prefix="uncached_"),
                **cached.fields("seconds", prefix="cached_"),
                "cache_speedup": uncached.median_ns / cached.median_ns,
                "uncached_output_values_per_second": uncached.throughput(length * args.n_tapers),
                "cached_output_values_per_second": cached.throughput(length * args.n_tapers),
            }
        )

    write_records(args.output, rows)
    print(args.output)
    for row in rows:
        print(f"N={row['length']:>5}: cache {row['cache_speedup']:.2f}x")


if __name__ == "__main__":
    main()
