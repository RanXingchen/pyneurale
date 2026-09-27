#!/usr/bin/env python3

"""Benchmark public CPU and CUDA Gaussian KDE evaluation."""

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
from neurale.models import GaussianKDE
from neurale.runtime import runtime_context

DEFAULT_CASES = (
    (512, 32, 4),
    (2048, 512, 8),
    (5000, 5000, 8),
    (10000, 2000, 16),
)


def _parse_cases(value: str) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(int(part) for part in case.split(":")) for case in value.split(","))


def _measure(
    model: GaussianKDE,
    points: np.ndarray,
    warmups: int,
    repeats: int,
) -> TimingSummary:
    output = np.empty(points.shape[0], dtype=np.float64)
    return measure_call(
        lambda: model.logpdf(points, out=output),
        repeats,
        warmups=warmups,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/models_kde_benchmark.csv"),
    )
    parser.add_argument(
        "--cases",
        default=",".join(":".join(map(str, case)) for case in DEFAULT_CASES),
        help="Comma-separated n_samples:n_points:n_features cases.",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    metadata = benchmark_metadata(runtime_native, include_cuda=True)

    rng = np.random.default_rng(42)
    rows = []
    for n_samples, n_points, n_features in _parse_cases(args.cases):
        samples = np.ascontiguousarray(
            rng.normal(size=(n_samples, n_features)),
            dtype=np.float64,
        )
        points = np.ascontiguousarray(
            rng.normal(size=(n_points, n_features)),
            dtype=np.float64,
        )
        timings: dict[str, TimingSummary] = {}
        for device in ("cpu", "cuda"):
            with runtime_context(device=device):
                model = GaussianKDE().fit(samples)
            timings[device] = _measure(
                model,
                points,
                args.warmups,
                args.repeats,
            )
        rows.append(
            {
                **metadata,
                "benchmark": "gaussian_kde_logpdf",
                "dtype": str(samples.dtype),
                "shape": f"{n_samples}x{n_features}",
                "n_samples": n_samples,
                "n_points": n_points,
                "n_features": n_features,
                "warmups": args.warmups,
                "repeats": args.repeats,
                **timings["cpu"].fields("ms", prefix="cpu_"),
                **timings["cuda"].fields("ms", prefix="cuda_"),
                "cpu_provider": "cpu",
                "cuda_provider": "cuda",
                "cpu_points_per_second": timings["cpu"].throughput(n_points),
                "cuda_points_per_second": timings["cuda"].throughput(n_points),
                "cuda_speedup": (timings["cpu"].median_ns / timings["cuda"].median_ns),
            }
        )

    write_records(args.output, rows)

    print(args.output)
    for row in rows:
        print(
            f"n={row['n_samples']}, m={row['n_points']}, "
            f"d={row['n_features']}: CPU {row['cpu_median_ms']:.3f} ms, "
            f"CUDA {row['cuda_median_ms']:.3f} ms, "
            f"speedup {row['cuda_speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
