#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Simulation and simulated-device characterization benchmark.

Two halves make one artifact. The native half
(``neurale_devices_simulation_benchmark``) owns generation cost, simulated
source read latency, pacing error, cancellation, scheduled loss, and the
simulated device -> native sink integration: those paths must be timed without
a Python callback or the GIL on them. This script owns the second small
integration, simulated device -> ``SessionRecorder``, because the recorder is a
Python-owned control-plane object, and it merges both halves into one JSON
Lines file.

It deliberately does not re-measure filters, features, decoders, or the
record/replay characterization; those keep their owners in
``docs/development/benchmarks.md``.

The recorder's bounded spool uses disk space. Its configured capacity must
hold the complete session; the script does not silently shrink the workload.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import benchmark_metadata, summarize_ns, write_records

_NATIVE_BINARY = "neurale_devices_simulation_benchmark"
# The device ships raw 16-bit ADC counts, so payload bytes are half what the
# generator produces internally. Frame-pool and spool sizing follow the payload.
_SAMPLE_DTYPE = "int16"
_BYTES_PER_SAMPLE = 2


def _native_module() -> Any:
    import neurale._native as native

    return native


def _resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_seconds": time.process_time(),
        "peak_resident_memory_bytes": 0,
    }
    if sys.platform == "win32":
        return snapshot
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if platform.system() == "Darwin" else 1024
    snapshot["peak_resident_memory_bytes"] = int(usage.ru_maxrss * scale)
    return snapshot


def _find_native_executable(requested: str | None) -> Path | None:
    if requested:
        candidate = Path(requested)
        return candidate if candidate.is_file() else None
    root = Path(__file__).resolve().parents[2]
    for directory in sorted(root.glob("build*/cpp/benchmarks")):
        for name in (_NATIVE_BINARY, f"{_NATIVE_BINARY}.exe"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    found = shutil.which(_NATIVE_BINARY)
    return None if found is None else Path(found)


def _run_native(executable: Path, arguments: argparse.Namespace) -> list[dict[str, Any]]:
    command = [
        str(executable),
        "--channels",
        str(arguments.channels),
        "--fs",
        str(arguments.fs),
        "--frame-samples",
        str(arguments.frame_samples),
        "--warmup-frames",
        str(arguments.warmup_frames),
        "--pipeline-frames",
        str(arguments.frames),
        "--loss-period-frames",
        str(arguments.loss_period_frames),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not rows:
        raise RuntimeError(
            f"{_NATIVE_BINARY} exited {completed.returncode}: {completed.stderr.strip()}"
        )
    for row in rows:
        row["half"] = "native"
    return rows


def _generator(channels: int, fs: float) -> Any:
    from neurale.signal.simulation import SignalGenerator

    # Two tones per channel, matching the native half's default generator so the
    # two integrations are fed comparable work. Amplitudes are in ADC counts and
    # stay inside the int16 range, so quantization rounds rather than saturates.
    low = [7.0 + (i % 13) for i in range(channels)]
    high = [83.0 + (i % 29) for i in range(channels)]
    return SignalGenerator.tones(
        channels,
        fs,
        [low, high],
        amps=[[2_000.0] * channels, [500.0] * channels],
        phases=[
            [i * 0.01 for i in range(channels)],
            [-i * 0.02 for i in range(channels)],
        ],
    )


def _runner_config(payload_bytes: int, queue_capacity: int) -> Any:
    from neurale.streaming import _native as streaming_native

    budget = streaming_native.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = queue_capacity
    budget.processor_owned = 8
    budget.critical_edge_capacity = queue_capacity + 1
    budget.actuator_owned = 1
    budget.observer_edge_capacity = queue_capacity
    budget.reserve = 4
    config = streaming_native.RealtimeConfig()
    config.platform.mode = streaming_native.RealtimeConfigMode.STRICT
    config.platform.prefault_pools = True
    config.pool_capacity = budget
    config.buffer_size = payload_bytes
    config.max_signal_blocks = 1
    config.discontinuity_capacity = 8
    config.gaps_per_discontinuity = 1
    config.max_process_outputs = 1
    config.max_flush_outputs = 0
    config.fault_history_capacity = 8
    config.validate()
    return config


@dataclass(frozen=True)
class _Geometry:
    """One acquisition shape. The two halves do not have to share one.

    The native half is bounded only by patience; the recorder half also needs
    enough disk-spool capacity for the complete session.
    """

    channels: int
    fs: int
    frame_samples: int
    frames: int


def _recording_geometry(arguments: argparse.Namespace) -> _Geometry:
    return _Geometry(
        channels=arguments.record_channels or arguments.channels,
        fs=arguments.record_fs or arguments.fs,
        frame_samples=arguments.record_frame_samples or arguments.frame_samples,
        frames=arguments.record_frames or arguments.frames,
    )


class _RecordingRun:
    """One recorded session, timed in the two intervals that mean something.

    ``run_ns`` is the acquisition/critical-recording interval: everything from
    ``start()`` to the runner having stopped, which is the part that competes
    with the realtime chain. ``finalize_ns`` is the control-plane interval that
    turns the spool into a canonical NRF session, which happens after
    acquisition has ended and is therefore not a realtime cost.
    """

    def __init__(self, run_ns: int, finalize_ns: int, facts: dict[str, Any]) -> None:
        self.run_ns = run_ns
        self.finalize_ns = finalize_ns
        self.facts = facts


def _record_once(
    output: Path,
    *,
    channels: int,
    fs: float,
    frame_samples: int,
    frames: int,
    spool_capacity_bytes: int,
    queue_capacity: int,
    events: Sequence[Any],
) -> _RecordingRun:
    import neurale.streaming as streaming
    from neurale.devices.simulation import (
        SimulationFaultPlan,
        SimulationTimingConfig,
        _finite_simulated_neural_device,
    )
    from neurale.io.nrf import NrfReader
    from neurale.recording import RecorderConfig, RecorderLimits, SessionRecorder, StreamRecording

    total_samples = frames * frame_samples
    device = _finite_simulated_neural_device(
        _generator(channels, fs),
        samples_per_frame=frame_samples,
        total_sample_count=total_samples,
        sample_dtype=_SAMPLE_DTYPE,
        # Raw counts with no calibration: declaring volts here would relabel an
        # uncalibrated integer as a physical measurement.
        physical_unit="dimensionless",
        timing=SimulationTimingConfig(
            initial_sample_idx=0,
            initial_device_tick=0,
            clock_offset_ns=250,
            clock_drift_ppm=100,
            clock_sync_uncertainty_ns=50,
        ),
        # Paced at the acquisition rate, which is what makes this an integration
        # measurement rather than a saturation one: a recorder fed by an unpaced
        # source is not slow, it is simply overrun, and the question here is
        # whether the recording chain keeps up with a device and at what cost.
        paced=True,
        faults=SimulationFaultPlan(tuple(events)),
    )
    config = RecorderConfig(
        path=output,
        streams=(StreamRecording(stream_id="neural"),),
        limits=RecorderLimits(
            frame_queue_capacity=queue_capacity,
            control_queue_capacity=queue_capacity,
            spool_capacity_bytes=spool_capacity_bytes,
            max_control_records=16 * 1024,
            checkpoint_interval=0,
        ),
    )
    recorder = SessionRecorder.create(config, device)
    runner = streaming.StreamRunner(
        device.schema,
        _runner_config(frame_samples * channels * _BYTES_PER_SAMPLE, queue_capacity),
        device.source,
        streaming.IdentityNativeProcessor(),
        streaming.CountingNativeConsumer(),
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    recorder.prepare()
    recorder.attach(runner)
    if runner.prepare() != streaming.StreamStatus.OK or runner.arm() != streaming.StreamStatus.OK:
        raise RuntimeError("the recorded runner could not be prepared")

    started = time.perf_counter_ns()
    if runner.start() != streaming.StreamStatus.OK:
        raise RuntimeError("the recorded runner could not be started")
    join_status = runner.join()
    run_ns = time.perf_counter_ns() - started
    if join_status != streaming.StreamStatus.OK:
        # The usual cause is a spool that could not hold the whole session: the
        # recorder faults, its safety controller inhibits the runtime, and the
        # runner stops. Say so, because "STOPPED" alone points at the runner.
        fault = runner.primary_fault
        raise RuntimeError(
            f"the recorded run ended with {join_status} (primary fault: {fault}). "
            f"The spool held {spool_capacity_bytes} bytes for {frames} frames of "
            f"{frame_samples * channels * _BYTES_PER_SAMPLE} payload bytes; a session "
            "must fit its spool in full. Raise --spool-capacity-bytes or lower --frames."
        )

    recorder.stop("m7-simulation-benchmark")
    finalize_started = time.perf_counter_ns()
    status = recorder.finalize()
    finalize_ns = time.perf_counter_ns() - finalize_started
    stats = runner.stats
    recorder.close()
    if not status.complete or not status.accounting_verified:
        raise RuntimeError("the recorded session did not finalize completely")

    with NrfReader.open(output) as reader:
        recorded_samples = int(reader.stream_extent("neural"))
        discontinuities = sum(
            int(reader.record_extent(schema["id"]))
            for schema in reader.manifest["record_schemas"]
            if schema["kind"] == "discontinuities"
        )
    payload_bytes = total_samples * channels * _BYTES_PER_SAMPLE
    session_bytes = output.stat().st_size
    facts = {
        "recorded_samples": recorded_samples,
        "recorded_discontinuities": discontinuities,
        "nrf_committed": int(status.nrf_committed),
        "raw_payload_bytes": payload_bytes,
        "session_bytes": session_bytes,
        "session_to_payload_ratio": session_bytes / payload_bytes if payload_bytes else 0.0,
        "ingress_high_water_mark": int(stats.ingress_high_water_mark),
        "frames_consumed": int(stats.frames_consumed),
        "queue_overruns": int(stats.queue_overruns),
        "pool_exhaustions": int(stats.pool_exhaustions),
        "outstanding_frames": int(runner.outstanding_frames),
    }
    return _RecordingRun(run_ns, finalize_ns, facts)


def _recording_rows(
    arguments: argparse.Namespace,
    geometry: _Geometry,
    metadata: dict[str, Any],
    work_dir: Path,
    *,
    scenario: str,
    events: Sequence[Any],
    schedule: str,
) -> list[dict[str, Any]]:
    frame_bytes = geometry.frame_samples * geometry.channels * _BYTES_PER_SAMPLE
    # Leave room for timing, block-index, and control records beyond the payload.
    spool_capacity = arguments.spool_capacity_bytes or max(
        1 << 20, geometry.frames * (frame_bytes + 1_024) * 2
    )
    # Import before the CPU snapshot. The recorder and NRF reader pull in large
    # dependencies on first use, and charging that one-time import to the
    # recording chain would overstate its cost several times over.
    import neurale.io.nrf
    import neurale.recording
    import neurale.streaming  # noqa: F401
    from neurale.devices.simulation import SimulatedNeuralDevice  # noqa: F401

    before = _resource_snapshot()
    runs: list[_RecordingRun] = []
    for i in range(arguments.runs):
        output = work_dir / f"{scenario}-{i}.nrf"
        if output.exists():
            raise FileExistsError(f"benchmark session already exists: {output}")
        runs.append(
            _record_once(
                output,
                channels=geometry.channels,
                fs=float(geometry.fs),
                frame_samples=geometry.frame_samples,
                frames=geometry.frames,
                spool_capacity_bytes=spool_capacity,
                queue_capacity=arguments.queue_capacity,
                events=events,
            )
        )
        if not arguments.keep_sessions:
            output.unlink()
    after = _resource_snapshot()

    run_summary = summarize_ns(run.run_ns for run in runs)
    finalize_summary = summarize_ns(run.finalize_ns for run in runs)
    facts = runs[-1].facts
    acquired_seconds = geometry.frames * geometry.frame_samples / geometry.fs
    shared = {
        **metadata,
        "half": "python",
        "benchmark": "simulated_device_to_session_recorder",
        "scenario": scenario,
        "generator_kind": "tones",
        "channels": geometry.channels,
        "fs": geometry.fs,
        "dtype": _SAMPLE_DTYPE,
        "frame_samples": geometry.frame_samples,
        "frame_payload_bytes": frame_bytes,
        "frames": geometry.frames,
        "pacing_mode": "paced_steady_clock",
        "clock_mode": "steady_host_clock",
        "clock_offset_ns": 250,
        "clock_drift_ppm": 100,
        "clock_sync_uncertainty_ns": 50,
        "fault_schedule": schedule,
        "spool_capacity_bytes": spool_capacity,
        "queue_capacity": arguments.queue_capacity,
        "runs": arguments.runs,
        "warmups": 0,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        # Scenario-wide, not per-metric: every run's acquisition, finalization,
        # and read-back is inside this interval, and finalization is threaded,
        # so the value can exceed the scenario's wall time.
        "process_cpu_seconds": max(0.0, after["cpu_seconds"] - before["cpu_seconds"]),
        "peak_resident_memory_bytes": after["peak_resident_memory_bytes"],
        # Python-level work is on the control plane here; the critical recording
        # path is native. This script measures no allocation events.
        "allocation_tracking": "not_measured",
        **facts,
    }
    return [
        {
            **shared,
            "metric": "record_session",
            "unit": "ns",
            "samples": run_summary.samples,
            **run_summary.fields("ns"),
            "throughput_item": "frames",
            "throughput_per_second": run_summary.throughput(geometry.frames),
            "realtime_factor": (
                acquired_seconds * 1e9 / run_summary.median_ns if run_summary.median_ns else 0.0
            ),
        },
        {
            **shared,
            "metric": "finalize_session",
            "unit": "ns",
            "samples": finalize_summary.samples,
            **finalize_summary.fields("ns"),
            "throughput_item": "samples",
            "throughput_per_second": finalize_summary.throughput(
                geometry.frames * geometry.frame_samples
            ),
            "realtime_factor": 0.0,
        },
    ]


def run(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not arguments.skip_native:
        executable = _find_native_executable(arguments.native_executable)
        if executable is None:
            raise RuntimeError(
                f"{_NATIVE_BINARY} was not found. Build it with "
                f"'cmake --build <build> --target {_NATIVE_BINARY}', pass "
                "--native-executable, or pass --skip-native to record only the "
                "SessionRecorder half."
            )
        rows.extend(_run_native(executable, arguments))

    if arguments.skip_recording:
        return rows

    from neurale.devices.simulation import KnownSampleLoss

    metadata = benchmark_metadata(_native_module())
    geometry = _recording_geometry(arguments)
    work_dir = arguments.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    rows.extend(
        _recording_rows(
            arguments,
            geometry,
            metadata,
            work_dir,
            scenario="continuous",
            events=(),
            schedule="none",
        )
    )
    losses = tuple(
        KnownSampleLoss(ordinal, arguments.loss_samples)
        for ordinal in range(
            arguments.loss_period_frames, geometry.frames, arguments.loss_period_frames
        )
    )
    if losses:
        rows.extend(
            _recording_rows(
                arguments,
                geometry,
                metadata,
                work_dir,
                scenario="scheduled_sample_loss",
                events=losses,
                schedule=(
                    f"sample_loss:every_{arguments.loss_period_frames}_frames:"
                    f"{arguments.loss_samples}_samples"
                ),
            )
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/m7-simulation.jsonl"))
    parser.add_argument("--work-dir", type=Path, default=Path("build/m7-simulation-work"))
    parser.add_argument("--native-executable")
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--fs", type=int, default=30_000)
    parser.add_argument("--frame-samples", type=int, default=30)
    parser.add_argument("--frames", type=int, default=2_000)
    parser.add_argument("--warmup-frames", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--queue-capacity", type=int, default=64)
    for name in ("channels", "fs", "frame-samples", "frames"):
        parser.add_argument(
            f"--record-{name}",
            type=int,
            default=0,
            help=f"--{name} for the SessionRecorder half only; 0 reuses --{name}.",
        )
    parser.add_argument("--loss-period-frames", type=int, default=64)
    parser.add_argument("--loss-samples", type=int, default=4)
    parser.add_argument(
        "--spool-capacity-bytes",
        type=int,
        default=0,
        help="0 derives the disk spool capacity from the session size.",
    )
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-recording", action="store_true")
    parser.add_argument("--keep-sessions", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.skip_native and arguments.skip_recording:
        print("nothing to measure: both halves were skipped", file=sys.stderr)
        return 2
    for name in ("channels", "fs", "frame_samples", "frames", "runs", "queue_capacity"):
        if getattr(arguments, name) < 1:
            print(f"--{name.replace('_', '-')} must be positive", file=sys.stderr)
            return 2
    rows = run(arguments)
    write_records(arguments.output, rows)
    print(f"wrote {len(rows)} records to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
