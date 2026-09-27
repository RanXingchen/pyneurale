#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _repository_module import load_repository_module

pytestmark = pytest.mark.native

_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS = _ROOT / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS))
benchmark = load_repository_module(
    "benchmark_record_replay", _BENCHMARKS / "recording" / "benchmark_record_replay.py"
)


def test_physical_profile_has_exact_multirate_shapes_and_payload() -> None:
    profile = benchmark.PhysicalProfile(duration_seconds=1.0)

    assert profile.document()["neural_shape_per_frame"] == [4, 256]
    assert profile.document()["cursor_shape_per_block"] == [1, 2]
    assert profile.document()["audio_shape_per_block"] == [441, 1]
    assert profile.document()["feature_shape_per_block"] == [1, 256]
    assert profile.neural_samples == 4_000
    assert profile.cursor_samples == 100
    assert profile.audio_samples == 44_100
    assert profile.feature_observations == 100
    assert profile.payload_bytes == 2_341_800


def test_native_source_profile_preserves_stream_rates() -> None:
    profile = benchmark.PhysicalProfile(duration_seconds=0.02)
    signals = {signal.id: signal for signal in benchmark._schema(profile).signals}
    native = benchmark._find_native_executable(None)
    actual = benchmark._run_json(
        [str(native), "source-checksums", "--seed", str(profile.seed)], cwd=_ROOT
    )

    assert actual["frame_period_ns"] == 1_000_000
    assert actual["behavior_block_period_ns"] == 10_000_000
    assert actual["neural_only_signal_blocks"] == 1
    assert actual["multimodal_signal_blocks"] == 4
    assert actual["neural_frame_samples"] == signals[benchmark.NEURAL].nominal_block_samples == 4
    assert actual["cursor_block_samples"] == signals[benchmark.CURSOR].nominal_block_samples == 1
    assert actual["audio_block_samples"] == signals[benchmark.AUDIO].nominal_block_samples == 441
    assert (
        actual["bandpower_block_samples"] == signals[benchmark.BANDPOWER].nominal_block_samples == 1
    )
    assert (
        actual["neural_frame_payload_bytes"] == signals[benchmark.NEURAL].max_block_bytes == 2_048
    )
    assert (
        actual["multimodal_frame_payload_bytes"]
        == sum(signal.max_block_bytes for signal in signals.values())
        == 4_986
    )
    assert profile.frames == 20
    assert profile.behavior_blocks == 2
    assert profile.cursor_samples == 2
    assert profile.audio_samples == 882
    assert profile.feature_observations == 2


def test_native_and_python_source_payloads_are_byte_identical() -> None:
    profile = benchmark.PhysicalProfile(duration_seconds=1.0, seed=613)
    expected = benchmark._template_checksums(benchmark._templates(profile))
    native = benchmark._find_native_executable(None)
    actual = benchmark._run_json(
        [str(native), "source-checksums", "--seed", str(profile.seed)], cwd=_ROOT
    )

    assert actual["payload_generator"] == "m6-13-counter-v1"
    assert {stream_id: actual[f"{stream_id}_sha256"] for stream_id in benchmark.STREAM_IDS} == (
        expected
    )
    different = benchmark._template_checksums(
        benchmark._templates(benchmark.PhysicalProfile(duration_seconds=1.0, seed=614))
    )
    assert all(different[stream_id] != expected[stream_id] for stream_id in benchmark.STREAM_IDS)


def test_spool_timing_samples_survive_dense_checkpoints(tmp_path: Path) -> None:
    profile = benchmark.PhysicalProfile(duration_seconds=1.1, seed=613)
    native = benchmark._find_native_executable(None)
    plan = tmp_path / "plan.json"
    spool = tmp_path / "recording.spool"
    benchmark._write_plan(plan, profile, queue_capacity=2_048)
    row = benchmark._run_json(
        [
            str(native),
            "record",
            "--frames",
            str(profile.frames),
            "--seed",
            str(profile.seed),
            "--plan",
            str(plan),
            "--spool",
            str(spool),
            "--backend",
            "memory",
            "--spool-capacity",
            str(profile.payload_bytes * 2),
            "--queue-capacity",
            "2048",
            "--checkpoint-interval",
            "1",
            "--worker-poll-ns",
            "50000",
            "--drain-timeout-ns",
            "30000000000",
        ],
        cwd=_ROOT,
    )

    assert row["spool_append_timing_dropped"] == 0
    assert row["checkpoint_append_timing_dropped"] == 0
    assert row["spool_sync_timing_dropped"] == 0
    assert row["timing_sample_capacity"] >= row["spool_append_latency_samples"]
    assert row["spool_append_latency_samples"] == row["committed_transactions"] + 1


def test_replay_configs_cover_modes_without_full_load() -> None:
    profile = benchmark.PhysicalProfile(duration_seconds=60.0)

    exact = benchmark._image_config("exact_frames", profile, range_seconds=None)
    projection = benchmark._image_config("recorded_projection", profile, range_seconds=2.0)
    streams = benchmark._image_config("stream_frames", profile, range_seconds=2.0)

    assert exact["message_range"] == {"start": 0, "stop": 60_000}
    assert projection["selected_streams"] == list(benchmark.STREAM_IDS)
    assert projection["message_range"] == {"start": 0, "stop": 2_000}
    assert streams["stream_ranges"] == {
        "neural": {"unit": "block_ordinal", "start": 0, "stop": 2_000},
        "cursor": {"unit": "block_ordinal", "start": 0, "stop": 200},
        "audio": {"unit": "block_ordinal", "start": 0, "stop": 200},
        "bandpower": {"unit": "block_ordinal", "start": 0, "stop": 200},
    }


def test_aggregate_retains_commands_behind_distributions() -> None:
    rows = [
        {
            "stage": "native_recording",
            "status": "ok",
            "elapsed_ns": 10,
            "exact_command": "a",
            "throughput_payload_bytes_per_second": 30.0,
        },
        {
            "stage": "native_recording",
            "status": "ok",
            "elapsed_ns": 20,
            "exact_command": "b",
            "throughput_payload_bytes_per_second": 10.0,
        },
    ]

    (aggregate,) = benchmark._aggregate(rows)

    assert aggregate["repetitions"] == 2
    assert aggregate["elapsed_median_ns"] == 15.0
    assert aggregate["throughput_payload_bytes_per_second_median"] == 20.0
    assert aggregate["exact_commands"] == ["a", "b"]


def test_same_build_verification_rejects_provider_mismatch() -> None:
    extension = {
        "native_version": "1",
        "native_abi_version": 1,
        "compiler": "compiler",
        "build_type": "Release",
        "cpu_math_backend": "builtin",
        "fft_backend": "builtin",
        "cuda_compiled": False,
        "threading_backend": "std",
        "threads": 4,
    }
    helper = {
        **extension,
        "cpu_math_backend": "mkl",
        "native_threads": extension["threads"],
    }
    helper.pop("threads")

    with pytest.raises(RuntimeError, match="different builds"):
        benchmark._verify_same_native_build(extension, helper)
