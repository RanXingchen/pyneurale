#!/usr/bin/env python3

"""Benchmark public CPU and CUDA exact KNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    TimingSummary,
    benchmark_metadata,
    measure_call,
    write_records,
)

import neurale._native as runtime_native
from neurale.models import knn
from neurale.runtime import runtime_context

DEFAULT_CASES = (
    (256, 8, 8),
    (1024, 8, 8),
    (4096, 8, 8),
    (4096, 32, 8),
    (4096, 8, 32),
    (8192, 8, 8),
)

LARGE_CASES = (
    (50_000, 16, 5),
    (50_000, 61, 5),
    (100_000, 61, 5),
    (50_000, 128, 5),
    (50_000, 61, 16),
    (50_000, 61, 64),
)


def _parse_cases(value: str) -> tuple[tuple[int, int, int], ...]:
    cases = []
    for item in value.split(","):
        n_samples, n_features, k = (int(part) for part in item.split(":"))
        cases.append((n_samples, n_features, k))
    return tuple(cases)


def _measure(
    samples: np.ndarray,
    k: int,
    device: str,
    warmups: int,
    repeats: int,
    include_self: bool,
) -> TimingSummary:
    with runtime_context(device=device, cuda_device=0):
        return measure_call(
            lambda: knn(samples, k, include_self=include_self),
            repeats,
            warmups=warmups,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite",
        choices=("standard", "large"),
        default="standard",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "both"),
        help="Defaults to both for the standard suite and CUDA for large N.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated n_samples:n_features:k cases.",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--include-self",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    default_cases = LARGE_CASES if args.suite == "large" else DEFAULT_CASES
    cases = _parse_cases(args.cases) if args.cases else default_cases
    device = args.device or ("cuda" if args.suite == "large" else "both")
    devices = ("cpu", "cuda") if device == "both" else (device,)
    output = args.output or Path(
        "build/models_knn_large_benchmark.csv"
        if args.suite == "large"
        else "build/models_knn_benchmark.csv"
    )
    metadata = benchmark_metadata(
        runtime_native,
        include_cuda="cuda" in devices,
    )

    rng = np.random.default_rng(123)
    rows = []
    for n_samples, n_features, k in cases:
        samples = np.ascontiguousarray(
            rng.normal(size=(n_samples, n_features)),
            dtype=np.float64,
        )
        row = {
            **metadata,
            "benchmark": "knn",
            "dtype": str(samples.dtype),
            "shape": f"{n_samples}x{n_features}",
            "n_samples": n_samples,
            "n_features": n_features,
            "k": k,
            "include_self": args.include_self,
            "warmups": args.warmups,
            "repeats": args.repeats,
        }
        for selected_device in devices:
            timing = _measure(
                samples,
                k,
                selected_device,
                args.warmups,
                args.repeats,
                args.include_self,
            )
            row.update(timing.fields("ms", prefix=f"{selected_device}_"))
            row[f"{selected_device}_provider"] = selected_device
            row[f"{selected_device}_queries_per_second"] = timing.throughput(n_samples)
        if device == "both":
            row["cuda_speedup"] = row["cpu_median_ms"] / row["cuda_median_ms"]
        rows.append(row)

    write_records(output, rows)

    print(output)
    for row in rows:
        timings = ", ".join(
            f"{selected_device.upper()} {row[f'{selected_device}_median_ms']:.3f} ms"
            for selected_device in devices
        )
        speedup = f", speedup {row['cuda_speedup']:.2f}x" if device == "both" else ""
        print(f"n={row['n_samples']}, d={row['n_features']}, k={row['k']}: {timings}{speedup}")


if __name__ == "__main__":
    main()
