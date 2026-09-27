#!/usr/bin/env python3

"""Stage-by-stage memory and latency benchmark for the spike-sorting workflow."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    WindowsProcessMemoryCounters,
    benchmark_metadata,
    summarize_ns,
    write_records,
)

import neurale._native as runtime_native
from neurale.data import ChannelTable, Clock, SpikeWaveformBatch
from neurale.models import LPP, PCA
from neurale.sorting import (
    WaveformProjector,
    project_waveform_features,
    reject_waveform_outliers,
    remove_tiny_clusters,
    valley_seeking,
)
from neurale.sorting._dispatch import load_cpu_sorting_operation

_STAGES = (
    "detection",
    "waveform_extraction",
    "pca",
    "lpp",
    "clustering",
    "tiny_cluster_curation",
    "waveform_outlier",
)

# Provider metadata must distinguish native CPU kernels from pure Python/NumPy
# stages so cross-provider comparisons are not distorted. The curation stages
# (remove_tiny_clusters, reject_waveform_outliers) are Python/NumPy, not native.
_STAGE_PROVIDER = {
    "detection": "native_cpu",
    "waveform_extraction": "native_cpu",
    "pca": "native_cpu",
    "lpp": "native_cpu",
    "clustering": "native_cpu",
    "tiny_cluster_curation": "python_numpy",
    "waveform_outlier": "python_numpy",
}
assert set(_STAGES) == set(_STAGE_PROVIDER), "every stage must declare a provider"


def _resident_memory_bytes() -> tuple[int, str]:
    if sys.platform == "win32":
        counters = WindowsProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize), "windows_working_set_sampled"
    if sys.platform.startswith("linux"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * page_size, "linux_proc_statm_rss_sampled"
    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if platform.system() == "Darwin" else 1024
    return int(maximum * scale), "process_peak_rss_fallback"


class _MemorySampler:
    def __init__(self, interval_seconds: float) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self.peak_bytes = 0
        self.metric = ""

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def sample(self) -> None:
        resident, metric = _resident_memory_bytes()
        self.peak_bytes = max(self.peak_bytes, resident)
        self.metric = metric

    def stop(self) -> None:
        self.sample()
        self._stop.set()
        self._thread.join()
        self.sample()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()


def _array_backing_bytes(value: object) -> int:
    seen_objects: set[int] = set()
    seen_backings: set[int] = set()

    def visit(item: object) -> int:
        if item is None or isinstance(item, (str, bytes, bytearray, bool, int, float, np.generic)):
            return 0
        item_id = id(item)
        if item_id in seen_objects:
            return 0
        seen_objects.add(item_id)
        if isinstance(item, np.ndarray):
            root: object = item
            while isinstance(getattr(root, "base", None), np.ndarray):
                root = root.base
            backing = getattr(root, "base", None)
            owner = root if backing is None else backing
            owner_id = id(owner)
            if owner_id in seen_backings:
                return 0
            seen_backings.add(owner_id)
            if isinstance(owner, bytes):
                return len(owner)
            return int(root.nbytes)
        if isinstance(item, Mapping):
            return sum(visit(key) + visit(member) for key, member in item.items())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return sum(visit(member) for member in item)
        if is_dataclass(item):
            return sum(visit(getattr(item, field.name)) for field in fields(item))
        return 0

    return visit(value)


def _source(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    template = rng.uniform(-1.0, 1.0, size=(args.samples, args.channels))
    blocks = np.empty((args.spikes, args.samples, args.channels), dtype=np.float64)
    blocks[:] = template
    events = np.arange(args.spikes, dtype=np.int64)
    peak_channels = events % args.channels
    blocks[events, args.pre_samples, peak_channels] = -8.0 - (events % 1000) * 0.001
    return blocks.reshape(args.spikes * args.samples, args.channels)


def _detection_arguments(args: argparse.Namespace) -> tuple[object, ...]:
    return (
        args.threshold_multiplier,
        args.refractory_samples,
        args.alignment_search_radius,
        args.pre_samples,
        args.samples - args.pre_samples - 1,
        "negative",
        "raise",
        None,
    )


def _detect(source: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    native = load_cpu_sorting_operation("detection")
    result = native._detect_threshold(source, *_detection_arguments(args))
    if np.asarray(result["sample_indices"]).size != args.spikes:
        raise RuntimeError(
            f"synthetic source produced {np.asarray(result['sample_indices']).size} spikes, "
            f"expected {args.spikes}"
        )
    return result


def _expected_sample_indices(args: argparse.Namespace) -> np.ndarray:
    return args.pre_samples + np.arange(args.spikes, dtype=np.int64) * args.samples


def _extract(source: np.ndarray, detections: Mapping[str, object], args: argparse.Namespace):
    native = load_cpu_sorting_operation("detection")
    return native._extract_threshold_waveforms(
        source,
        detections["sample_indices"],
        args.pre_samples,
        args.samples - args.pre_samples - 1,
    )


def _batch(args: argparse.Namespace) -> SpikeWaveformBatch:
    source = _source(args)
    sample_indices = _expected_sample_indices(args)
    peak_channels = np.arange(args.spikes, dtype=np.int64) % args.channels
    detections = {"sample_indices": sample_indices}
    waveforms = _extract(source, detections, args)
    return SpikeWaveformBatch(
        waveforms=waveforms,
        sample_indices=sample_indices,
        times=2.0 + sample_indices / args.fs,
        peak_channel_indices=peak_channels,
        electrode_group_ids=peak_channels,
        amps=-8.0 - (np.arange(args.spikes, dtype=np.float64) % 1000) * 0.001,
        channels=ChannelTable.from_names(
            [f"channel-{idx}" for idx in range(args.channels)],
            type="spike",
            unit="uV",
        ),
        fs=args.fs,
        clock=Clock("acquisition", "device", synchronization_domain="benchmark"),
        source_stream="synthetic-wideband",
        segment_id=0,
        pre_samples=args.pre_samples,
        post_samples=args.samples - args.pre_samples - 1,
        polarity="negative",
    )


def _projector_features(
    args: argparse.Namespace,
    estimator: PCA | LPP,
) -> tuple[SpikeWaveformBatch, WaveformProjector]:
    batch = _batch(args)
    n_calibrations = min(args.calibration_spikes, args.spikes)
    if isinstance(estimator, LPP) and n_calibrations < 2:
        raise ValueError("LPP calibration requires at least two spikes")
    projector = WaveformProjector(estimator)
    project_waveform_features(
        batch[:n_calibrations],
        projector,
        fit=True,
        measurements=(),
    )
    return batch, projector


def _resolve_outlier_clusters(args: argparse.Namespace) -> int:
    # ``--outlier-clusters`` defaults to the clustering cluster count so the
    # outlier stage matches the clustering geometry unless a caller (e.g. the
    # single-large-cluster probe) overrides it.
    return args.outlier_clusters if args.outlier_clusters is not None else args.clusters


def _stage_operation(args: argparse.Namespace, stage: str):
    if stage == "detection":
        source = _source(args)
        return lambda: _detect(source, args)
    if stage == "waveform_extraction":
        source = _source(args)
        detections = {"sample_indices": _expected_sample_indices(args)}
        return lambda: _extract(source, detections, args)
    if stage == "pca":
        batch, projector = _projector_features(args, PCA(args.components))
        return lambda: project_waveform_features(batch, projector, fit=False, measurements=())
    if stage == "lpp":
        batch, projector = _projector_features(
            args,
            LPP(
                args.components,
                n_neighbors=min(args.neighbors, args.calibration_spikes),
                proj_method="LPP",
            ),
        )
        return lambda: project_waveform_features(batch, projector, fit=False, measurements=())
    if stage == "clustering":
        batch, projector = _projector_features(args, PCA(args.components))
        features = project_waveform_features(batch, projector, fit=False, measurements=())
        initial_labels = np.arange(args.spikes, dtype=np.int64) % args.clusters
        return lambda: valley_seeking(
            features,
            initial_labels,
            radius=args.cluster_radius,
            max_iterations=args.max_iterations,
        )
    if stage == "tiny_cluster_curation":
        labels = np.arange(args.spikes, dtype=np.int64) % args.clusters
        labels[: min(5, args.spikes)] = args.clusters
        return lambda: remove_tiny_clusters(
            labels,
            min_cluster_size=args.min_cluster_size,
            noise_label=-1,
        )
    if stage == "waveform_outlier":
        # The outlier stage is a scale probe: it allocates per-cluster
        # waveform temporaries, so peak RSS is driven by the largest cluster.
        # Default to the clustering cluster count; ``--outlier-clusters 1`` forces
        # a single 100k-spike cluster that exercises the full waveform payload.
        batch = _batch(args)
        labels = np.arange(args.spikes, dtype=np.int64) % _resolve_outlier_clusters(args)
        return lambda: reject_waveform_outliers(
            batch,
            labels,
            threshold_multiplier=args.outlier_threshold_multiplier,
            minimum_cluster_size=args.outlier_minimum_cluster_size,
            noise_label=-1,
        )
    raise ValueError(f"unknown stage: {stage}")


def _run_worker(args: argparse.Namespace, stage: str) -> dict[str, object]:
    operation = _stage_operation(args, stage)
    for _ in range(args.warmups):
        warm = operation()
        del warm
    gc.collect()
    baseline_rss, rss_metric = _resident_memory_bytes()
    sampler = _MemorySampler(args.memory_sample_interval)
    timings: list[int] = []
    result: object | None = None
    sampler.start()
    try:
        for _ in range(args.repetitions):
            if result is not None:
                del result
                result = None
                gc.collect()

            started = time.perf_counter_ns()
            result = operation()
            timings.append(time.perf_counter_ns() - started)
    finally:
        sampler.stop()
    final_rss, _ = _resident_memory_bytes()
    output_bytes = _array_backing_bytes(result)
    tmp_bytes = max(0, sampler.peak_bytes - max(baseline_rss, final_rss))
    timing = summarize_ns(timings)
    row: dict[str, object] = {
        **benchmark_metadata(runtime_native),
        "benchmark": "sorting_100k_spikes",
        "stage": stage,
        "status": "completed",
        "operation": "transform" if stage in ("pca", "lpp") else stage,
        "provider": _STAGE_PROVIDER[stage],
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        **timing.fields("ms"),
        "spikes_per_second": timing.throughput(args.spikes),
        "peak_resident_memory_bytes": sampler.peak_bytes,
        "stage_baseline_resident_memory_bytes": baseline_rss,
        "stage_final_resident_memory_bytes": final_rss,
        "resident_memory_metric": rss_metric or sampler.metric,
        "allocation_metric": "sampled_process_rss_derived",
        "derived_tmp_resident_delta_bytes": tmp_bytes,
        "tmp_resident_metric": ("sampled_peak_rss_minus_max_of_stage_baseline_and_final_rss"),
        # No native allocator instrumentation is connected: a single-allocation
        # size is not available and allocation events are not tracked. The
        # resident delta above is a sampled RSS aggregate, not a malloc trace.
        "largest_single_allocation_bytes": None,
        "allocation_event_tracking": "not_measured",
        "output_memory_bytes": output_bytes,
        "output_memory_metric": "unique_ndarray_backing_bytes",
        "dtype": "float64",
        "n_spikes": args.spikes,
        "waveform_samples": args.samples,
        "channels": args.channels,
        "waveform_shape": f"{args.spikes}x{args.samples}x{args.channels}",
        "window_pre_samples": args.pre_samples,
        "window_post_samples": args.samples - args.pre_samples - 1,
        "fs_hz": args.fs,
        "input_seed": args.seed,
        "memory_sample_interval_ms": args.memory_sample_interval * 1000.0,
    }
    row["stage_input"] = {
        "detection": "sample_major_signal",
        "waveform_extraction": "sample_major_signal_and_known_aligned_indices",
        "pca": "spike_waveform_batch",
        "lpp": "spike_waveform_batch",
        "clustering": "projected_feature_matrix",
        "tiny_cluster_curation": "event_aligned_cluster_labels",
        "waveform_outlier": "spike_waveform_batch_and_event_aligned_labels",
    }[stage]
    if stage in ("pca", "lpp"):
        row.update(
            {
                "projection_components": args.components,
                "calibration_spikes": min(args.calibration_spikes, args.spikes),
                "calibration_fit_in_timing": False,
            }
        )
    if stage == "lpp":
        row["lpp_neighbors"] = min(args.neighbors, args.calibration_spikes)
    if stage == "clustering":
        row.update(
            {
                "clusters": args.clusters,
                "cluster_radius": args.cluster_radius,
                "max_iterations": args.max_iterations,
                "workspace_bytes": result.workspace_bytes,
                "neighbor_pairs": result.neighbor_pairs,
                "converged": result.converged,
            }
        )
    if stage == "tiny_cluster_curation":
        operation = result.operations[0]
        row.update(
            {
                "clusters": args.clusters,
                "min_cluster_size": args.min_cluster_size,
                "removed_cluster_count": len(operation.source_labels),
                "affected_event_count": operation.affected_count,
            }
        )
    if stage == "waveform_outlier":
        # Per-cluster temporaries make the largest cluster the peak-memory
        # driver; record the geometry and rejection outcome so the row answers
        # the peak-memory scale question directly.
        outlier_clusters = _resolve_outlier_clusters(args)
        labels = np.arange(args.spikes, dtype=np.int64) % outlier_clusters
        cluster_sizes = np.bincount(labels)
        operation = result.operations[0]
        row.update(
            {
                "outlier_clusters": int(np.unique(labels).size),
                "largest_cluster_size": int(cluster_sizes.max()),
                "outlier_threshold_multiplier": args.outlier_threshold_multiplier,
                "outlier_minimum_cluster_size": args.outlier_minimum_cluster_size,
                "waveform_unit": operation.parameters["waveform_unit"],
                "rejected_spike_count": int(result.rejection_mask.sum()),
                "score_output_bytes": _array_backing_bytes(result.scores),
                "threshold_output_bytes": _array_backing_bytes(result.thresholds),
            }
        )
    if stage == "detection":
        row["detected_spikes"] = int(np.asarray(result["sample_indices"]).size)
        row["scalar_samples_per_second"] = timing.throughput(
            args.spikes * args.samples * args.channels
        )
    if stage == "waveform_extraction":
        row["extracted_spikes"] = int(np.asarray(result).shape[0])
        row["scalar_samples_per_second"] = timing.throughput(
            args.spikes * args.samples * args.channels
        )
    return row


def _worker_command(args: argparse.Namespace, stage: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-stage",
        stage,
        "--spikes",
        str(args.spikes),
        "--samples",
        str(args.samples),
        "--channels",
        str(args.channels),
        "--pre-samples",
        str(args.pre_samples),
        "--fs",
        str(args.fs),
        "--threshold-multiplier",
        str(args.threshold_multiplier),
        "--refractory-samples",
        str(args.refractory_samples),
        "--alignment-search-radius",
        str(args.alignment_search_radius),
        "--components",
        str(args.components),
        "--calibration-spikes",
        str(args.calibration_spikes),
        "--neighbors",
        str(args.neighbors),
        "--clusters",
        str(args.clusters),
        "--cluster-radius",
        str(args.cluster_radius),
        "--max-iterations",
        str(args.max_iterations),
        "--min-cluster-size",
        str(args.min_cluster_size),
        "--outlier-clusters",
        str(_resolve_outlier_clusters(args)),
        "--outlier-threshold-multiplier",
        str(args.outlier_threshold_multiplier),
        "--outlier-minimum-cluster-size",
        str(args.outlier_minimum_cluster_size),
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--memory-sample-interval",
        str(args.memory_sample_interval),
        "--seed",
        str(args.seed),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/sorting_100k.jsonl"))
    parser.add_argument("--stages", default=",".join(_STAGES))
    parser.add_argument("--worker-stage", choices=_STAGES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--spikes", type=int, default=100_000)
    parser.add_argument("--samples", type=int, default=61)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--pre-samples", type=int, default=30)
    parser.add_argument("--fs", type=float, default=30_000.0)
    parser.add_argument("--threshold-multiplier", type=float, default=5.0)
    parser.add_argument("--refractory-samples", type=int, default=1)
    parser.add_argument("--alignment-search-radius", type=int, default=0)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--calibration-spikes", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--cluster-radius", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument(
        "--outlier-clusters",
        type=int,
        default=None,
        help="waveform_outlier cluster count; defaults to --clusters; 1 forces a single large cluster",
    )
    parser.add_argument("--outlier-threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--outlier-minimum-cluster-size", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--memory-sample-interval", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.spikes < 1 or args.samples < 1 or args.channels < 1:
        raise ValueError("spikes, samples, and channels must be positive")
    if not 0 <= args.pre_samples < args.samples:
        raise ValueError("pre-samples must identify a sample inside the waveform window")
    if args.calibration_spikes < 2 or args.calibration_spikes > args.spikes:
        raise ValueError("calibration-spikes must be in [2, spikes]")
    if args.components < 1 or args.components > args.samples * args.channels:
        raise ValueError("components must fit the flattened waveform dimension")
    if args.neighbors < 1 or args.neighbors > args.calibration_spikes:
        raise ValueError("neighbors must be in [1, calibration-spikes]")
    if args.clusters < 1 or args.clusters > args.spikes:
        raise ValueError("clusters must be in [1, spikes]")
    outlier_clusters = _resolve_outlier_clusters(args)
    if outlier_clusters < 1 or outlier_clusters > args.spikes:
        raise ValueError("outlier-clusters must be in [1, spikes]")
    if args.outlier_threshold_multiplier < 0.0:
        raise ValueError("outlier-threshold-multiplier must be non-negative")
    if args.outlier_minimum_cluster_size < 1:
        raise ValueError("outlier-minimum-cluster-size must be positive")
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    if args.memory_sample_interval <= 0.0:
        raise ValueError("memory-sample-interval must be positive")
    stages = tuple(value.strip() for value in args.stages.split(",") if value.strip())
    if not stages or any(stage not in _STAGES for stage in stages):
        raise ValueError(f"stages must contain only: {', '.join(_STAGES)}")
    return stages


def main() -> None:
    args = _parser().parse_args()
    stages = _validate_args(args)
    if args.worker_stage is not None:
        print(json.dumps(_run_worker(args, args.worker_stage), sort_keys=True))
        return

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for stage in stages:
        completed = subprocess.run(
            _worker_command(args, stage),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            failures.append(stage)
            rows.append(
                {
                    "schema_version": 1,
                    "benchmark": "sorting_100k_spikes",
                    "stage": stage,
                    "status": "failed",
                    "error": completed.stderr.strip() or completed.stdout.strip(),
                    "n_spikes": args.spikes,
                    "waveform_samples": args.samples,
                    "channels": args.channels,
                    "dtype": "float64",
                }
            )
            continue
        rows.append(json.loads(completed.stdout))
    write_records(args.output, rows)
    print(args.output)
    if failures:
        raise SystemExit(f"sorting benchmark stages failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
