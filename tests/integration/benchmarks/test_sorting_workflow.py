#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.native

# The benchmark script lives under repository ``benchmarks/``, not ``tests/``,
# filed under the directory of the domain that owns it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_SCRIPT = _REPO_ROOT / "benchmarks" / "sorting" / "benchmark_workflow.py"


def test_benchmark_emits_stage_memory_and_shape_records(tmp_path: Path) -> None:
    output = tmp_path / "sorting.jsonl"
    script = _BENCHMARK_SCRIPT
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--spikes",
            "32",
            "--samples",
            "5",
            "--channels",
            "2",
            "--pre-samples",
            "2",
            "--components",
            "1",
            "--calibration-spikes",
            "16",
            "--neighbors",
            "3",
            "--clusters",
            "4",
            "--cluster-radius",
            "0.000001",
            "--max-iterations",
            "2",
            "--min-cluster-size",
            "6",
            "--outlier-minimum-cluster-size",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == [
        "detection",
        "waveform_extraction",
        "pca",
        "lpp",
        "clustering",
        "tiny_cluster_curation",
        "waveform_outlier",
    ]
    expected_provider = {
        "detection": "native_cpu",
        "waveform_extraction": "native_cpu",
        "pca": "native_cpu",
        "lpp": "native_cpu",
        "clustering": "native_cpu",
        "tiny_cluster_curation": "python_numpy",
        "waveform_outlier": "python_numpy",
    }
    for row in rows:
        assert row["status"] == "completed"
        assert row["waveform_shape"] == "32x5x2"
        assert row["dtype"] == "float64"
        # Curation is pure Python/NumPy, not a native CPU kernel; provider must
        # not claim native_cpu for the curation stages.
        assert row["provider"] == expected_provider[row["stage"]]
        assert row["peak_resident_memory_bytes"] > 0
        assert row["derived_tmp_resident_delta_bytes"] >= 0
        assert row["tmp_resident_metric"]
        assert row["largest_single_allocation_bytes"] is None
        assert row["allocation_event_tracking"] == "not_measured"
        assert row["output_memory_bytes"] >= 0
        assert row["output_memory_metric"] == "unique_ndarray_backing_bytes"
        assert row["median_ms"] >= 0.0
        assert row["p95_ms"] >= 0.0
        assert row["p99_ms"] >= 0.0
        assert row["spikes_per_second"] >= 0.0
        if row["stage"] in ("detection", "waveform_extraction"):
            assert row["scalar_samples_per_second"] >= 0.0
        else:
            assert "scalar_samples_per_second" not in row

    tiny = next(row for row in rows if row["stage"] == "tiny_cluster_curation")
    assert tiny["clusters"] == 4
    assert tiny["min_cluster_size"] == 6
    assert tiny["removed_cluster_count"] == 1  # the 5 relabeled spikes form one tiny cluster
    assert tiny["affected_event_count"] == 5

    outlier = next(row for row in rows if row["stage"] == "waveform_outlier")
    assert outlier["outlier_clusters"] == 4  # defaults to --clusters
    assert outlier["largest_cluster_size"] == 8  # 32 spikes / 4 clusters
    assert outlier["outlier_threshold_multiplier"] == 3.0
    assert outlier["outlier_minimum_cluster_size"] == 2
    assert outlier["waveform_unit"] == "uV"
    assert outlier["rejected_spike_count"] >= 0
    assert outlier["score_output_bytes"] >= 0
    assert outlier["threshold_output_bytes"] >= 0


def test_waveform_outlier_probe_records_payload_geometry(
    tmp_path: Path,
) -> None:
    # --outlier-clusters 1 collapses every spike into one cluster, so the
    # per-cluster temporaries cover the whole waveform payload.
    output = tmp_path / "sorting.jsonl"
    script = _BENCHMARK_SCRIPT
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--stages",
            "waveform_outlier",
            "--spikes",
            "32",
            "--samples",
            "5",
            "--channels",
            "2",
            "--pre-samples",
            "2",
            "--components",
            "1",
            "--calibration-spikes",
            "16",
            "--neighbors",
            "3",
            "--clusters",
            "4",
            "--outlier-clusters",
            "1",
            "--outlier-minimum-cluster-size",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    (row,) = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert row["stage"] == "waveform_outlier"
    assert row["status"] == "completed"
    assert row["outlier_clusters"] == 1
    assert row["largest_cluster_size"] == 32
    assert row["waveform_unit"] == "uV"


def test_default_shape_is_hundred_thousand_spikes() -> None:
    # The smoke test pins a tiny geometry, but the benchmark's *defaults* must
    # remain the required 100k-spike case so a manual run measures the real load.
    source = _BENCHMARK_SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--spikes", type=int, default=100_000)' in source
    assert 'parser.add_argument("--samples", type=int, default=61)' in source
    assert 'parser.add_argument("--channels", type=int, default=32)' in source
