#!/usr/bin/env python3

"""Shared timing, metadata, measurement, and result helpers for benchmark scripts."""

from __future__ import annotations

import csv
import ctypes
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")


class WindowsProcessMemoryCounters(ctypes.Structure):
    """The Win32 ``PROCESS_MEMORY_COUNTERS_EX`` layout, for GetProcessMemoryInfo.

    Declared once because it is an ABI, not a convenience: the field list and
    their order have to match what the OS writes, and a second hand-maintained
    copy is a second chance to get a field width wrong and read a plausible but
    wrong number out of the middle of the struct. Which member a caller reads
    still differs -- ``WorkingSetSize`` is the current resident set, sampled by
    a benchmark that wants a peak of its own sampling; ``PeakWorkingSetSize`` is
    the peak the OS itself tracked.
    """

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class TimingSummary:
    """Distribution of one timed operation, stored in nanoseconds."""

    samples: int
    minimum_ns: int
    median_ns: float
    p95_ns: float
    p99_ns: float
    maximum_ns: int

    def fields(self, unit: str, *, prefix: str = "") -> dict[str, int | float]:
        scale = {"seconds": 1e-9, "ms": 1e-6, "us": 1e-3, "ns": 1.0}[unit]
        return {
            f"{prefix}min_{unit}": self.minimum_ns * scale,
            f"{prefix}median_{unit}": self.median_ns * scale,
            f"{prefix}p95_{unit}": self.p95_ns * scale,
            f"{prefix}p99_{unit}": self.p99_ns * scale,
            f"{prefix}max_{unit}": self.maximum_ns * scale,
        }

    def throughput(self, items_per_call: int) -> float:
        """Return median processed items per second."""
        if self.median_ns <= 0:
            return 0.0
        return items_per_call * 1e9 / self.median_ns


def _percentile(sorted_values: list[int], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = fraction * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_ns(timings: Iterable[int]) -> TimingSummary:
    values = sorted(timings)
    if not values:
        raise ValueError("at least one timing sample is required")
    return TimingSummary(
        samples=len(values),
        minimum_ns=values[0],
        median_ns=statistics.median(values),
        p95_ns=_percentile(values, 0.95),
        p99_ns=_percentile(values, 0.99),
        maximum_ns=values[-1],
    )


def measure_call(
    function: Callable[[], _T],
    repeats: int,
    *,
    warmups: int = 1,
) -> TimingSummary:
    """Warm an operation, then summarize independent wall-clock calls."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for _ in range(warmups):
        function()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        timings.append(time.perf_counter_ns() - started)
    return summarize_ns(timings)


def measure_cold_and_warm(
    function: Callable[[], _T],
    repeats: int,
    *,
    warmups: int = 0,
) -> tuple[int, TimingSummary]:
    """Measure the first call separately and then the warmed distribution."""
    started = time.perf_counter_ns()
    function()
    cold_ns = time.perf_counter_ns() - started
    return cold_ns, measure_call(function, repeats, warmups=warmups)


def timed_row(
    metadata: Mapping[str, Any],
    *,
    benchmark: str,
    operation: str,
    function: Callable[[], Any],
    warmups: int,
    repeats: int,
    dtype: str,
    shape: str,
    items_per_call: int,
    provider: str = "public_cpu",
    **case: Any,
) -> dict[str, Any]:
    """Time one operation and shape it into a complete JSON Lines record.

    The per-domain benchmark scripts each own one domain's operations but must keep
    emitting the same record schema, so the shaping lives here rather than
    being copied into every one of them.
    """
    timing = measure_call(function, repeats, warmups=warmups)
    return {
        **metadata,
        "benchmark": benchmark,
        "operation": operation,
        "provider": provider,
        "dtype": dtype,
        "shape": shape,
        "warmups": warmups,
        "repeats": repeats,
        **timing.fields("ms"),
        "throughput_items_per_second": timing.throughput(items_per_call),
        **case,
    }


def benchmark_metadata(
    native: Any | None = None,
    *,
    include_cuda: bool = False,
) -> dict[str, Any]:
    """Collect stable host/build metadata outside measured intervals."""
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "allocation_metric": "not_measured",
    }
    try:
        metadata["pyneurale_version"] = importlib.metadata.version("pyneurale")
    except importlib.metadata.PackageNotFoundError:
        metadata["pyneurale_version"] = None
    if native is None:
        return metadata

    build = native.build_info()
    cpu = native.cpu_info()
    threading = native.threading_info()
    metadata.update(
        {
            "native_version": build["version"],
            "native_abi_version": build["abi_version"],
            "compiler": build["compiler"],
            "build_type": build["build_type"],
            "cpu": cpu.get("model") or cpu["architecture"],
            "cpu_vendor": cpu.get("vendor"),
            "logical_cores": cpu["logical_cores"],
            "cpu_math_backend": build["cpu_math_backend"],
            "fft_backend": build["fft_backend"],
            "threading_backend": threading["backend"],
            "threads": threading["num_threads"],
            "cuda_compiled": build["cuda_compiled"],
            "cuda_toolkit_version": build.get("cuda_toolkit_version"),
        }
    )
    if include_cuda:
        cuda_namespace = getattr(native, "cuda", None)
        info = getattr(cuda_namespace, "info", None)
        if not callable(info):
            raise RuntimeError("CUDA benchmark metadata requires neurale._native.cuda.info()")
        cuda = info()
        metadata.update(
            {
                "cuda_available": cuda["available"],
                "cuda_runtime_version": cuda.get("runtime_version"),
                "cuda_driver_version": cuda.get("driver_version"),
                "cuda_devices": ";".join(device["name"] for device in cuda.get("devices", ())),
            }
        )
    return metadata


def write_records(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write flat records as CSV or JSON Lines according to the suffix."""
    if not rows:
        raise ValueError("at least one benchmark result is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), sort_keys=True, default=str))
                stream.write("\n")
        return
    if path.suffix != ".csv":
        raise ValueError("benchmark output must use .csv or .jsonl")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def emit_csv(rows: list[Mapping[str, Any]]) -> None:
    """Write flat records to standard output for legacy stdout benchmarks."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
