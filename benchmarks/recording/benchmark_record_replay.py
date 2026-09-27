#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Native recording, finalization, recovery, and replay benchmark.

The default physical profile represents sixty seconds of four typed streams:

* 4 kHz x 256-channel ``int16`` neural acquisition in 1 ms frames;
* 100 Hz x 2D ``float32`` cursor trajectory;
* 44.1 kHz x 1-channel ``int16`` audio in exact 10 ms / 441-sample blocks;
* 100 Hz x 256-feature ``float64`` bandpower observations, with a 100 ms
  window and 10 ms shift.

The native executable generates the typed physical profile through
``NativeStreamRunner`` and the critical recorder. Finalization, torn-tail
recovery, and replay-image construction each run in an isolated Python process
so peak RSS and process I/O describe one stage.

This is manual performance characterization.  It defines no threshold and its
output supports no platform-wide or universal realtime claim.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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

import neurale.streaming as streaming
from neurale.io.nrf._canonical import canonical_json_bytes
from neurale.recording import (
    MessageRange,
    RecorderConfig,
    RecorderLimits,
    ReplayConfig,
    StreamMetadata,
    StreamRecording,
    StreamReplayRange,
    StreamRole,
    StreamStorageConfig,
    StreamTimingConfig,
    StreamTimingMode,
    build_replay_image,
    compile_recording_plan,
    diagnose_spool,
    finalize_spool,
    repair_spool,
)

FRAMES_PER_SECOND = 1_000

SCHEMA_ID = 613
NEURAL = 1
CURSOR = 2
AUDIO = 3
BANDPOWER = 4
FEATURE_SET_ID = 61301

STREAM_IDS = ("neural", "cursor", "audio", "bandpower")


@dataclass(frozen=True, slots=True)
class PhysicalProfile:
    duration_seconds: float = 60.0
    seed: int = 613
    neural_rate_hz: int = 4_000
    neural_channels: int = 256
    neural_frame_samples: int = 4
    cursor_rate_hz: int = 100
    cursor_dims: int = 2
    audio_rate_hz: int = 44_100
    audio_channels: int = 1
    audio_block_samples: int = 441
    feature_rate_hz: int = 100
    n_features: int = 256
    feature_window_ms: int = 100
    feature_shift_ms: int = 10
    source_dtype: str = "int16"
    cursor_dtype: str = "float32"
    audio_dtype: str = "int16"
    feature_dtype: str = "float64"

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must fit uint64")

    @property
    def frames(self) -> int:
        value = self.duration_seconds * FRAMES_PER_SECOND
        rounded = round(value)
        if not math.isclose(value, rounded, abs_tol=1e-9):
            raise ValueError("duration_seconds must contain a whole number of 1 ms frames")
        if rounded < 1:
            raise ValueError("duration_seconds must include at least one frame")
        return int(rounded)

    @property
    def behavior_blocks(self) -> int:
        return (self.frames + 9) // 10

    @property
    def neural_samples(self) -> int:
        return self.frames * self.neural_frame_samples

    @property
    def cursor_samples(self) -> int:
        return self.behavior_blocks

    @property
    def audio_samples(self) -> int:
        return self.behavior_blocks * self.audio_block_samples

    @property
    def feature_observations(self) -> int:
        return self.behavior_blocks

    @property
    def payload_bytes(self) -> int:
        return (
            self.neural_samples * self.neural_channels * np.dtype(self.source_dtype).itemsize
            + self.cursor_samples * self.cursor_dims * np.dtype(self.cursor_dtype).itemsize
            + self.audio_samples * self.audio_channels * np.dtype(self.audio_dtype).itemsize
            + self.feature_observations * self.n_features * np.dtype(self.feature_dtype).itemsize
        )

    def document(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "frames": self.frames,
            "frame_period_ms": 1.0,
            "neural_shape_per_frame": [self.neural_frame_samples, self.neural_channels],
            "cursor_shape_per_block": [1, self.cursor_dims],
            "cursor_block_period_ms": 10.0,
            "audio_shape_per_block": [self.audio_block_samples, self.audio_channels],
            "audio_block_period_ms": 10.0,
            "feature_shape_per_block": [1, self.n_features],
            "feature_block_period_ms": 10.0,
            "neural_samples": self.neural_samples,
            "cursor_samples": self.cursor_samples,
            "audio_samples": self.audio_samples,
            "feature_observations": self.feature_observations,
            "payload_bytes": self.payload_bytes,
            "payload_axis_order": "sample_major",
        }


def _schema(profile: PhysicalProfile) -> Any:
    neural = streaming.SignalSchema(
        NEURAL,
        streaming.SignalDType.INT16,
        profile.neural_channels,
        profile.neural_frame_samples,
        profile.neural_frame_samples,
        streaming.RationalRate(profile.neural_rate_hz, 1),
        1,
        layout=streaming.SignalLayout.SAMPLE_MAJOR,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
        physical_unit=streaming.PhysicalUnit.VOLTS,
    )
    cursor = streaming.SignalSchema(
        CURSOR,
        streaming.SignalDType.FLOAT32,
        profile.cursor_dims,
        1,
        1,
        streaming.RationalRate(profile.cursor_rate_hz, 1),
        2,
        layout=streaming.SignalLayout.SAMPLE_MAJOR,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
        physical_unit=streaming.PhysicalUnit.UNSPECIFIED,
    )
    audio = streaming.SignalSchema(
        AUDIO,
        streaming.SignalDType.INT16,
        profile.audio_channels,
        profile.audio_block_samples,
        profile.audio_block_samples,
        streaming.RationalRate(profile.audio_rate_hz, 1),
        3,
        layout=streaming.SignalLayout.SAMPLE_MAJOR,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
        physical_unit=streaming.PhysicalUnit.UNSPECIFIED,
    )
    feature = streaming.SignalSchema(
        BANDPOWER,
        streaming.SignalDType.FLOAT64,
        profile.n_features,
        1,
        1,
        streaming.RationalRate(profile.feature_rate_hz, 1),
        1,
        layout=streaming.SignalLayout.SAMPLE_MAJOR,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=FEATURE_SET_ID,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    descriptor = streaming.FeatureSetDescriptor(
        FEATURE_SET_ID,
        [f"bandpower:channel-{idx}" for idx in range(profile.n_features)],
        [1] * profile.n_features,
        NEURAL,
        "neural",
        profile.feature_window_ms * 1_000_000,
        profile.feature_shift_ms * 1_000_000,
        "multitaper-bandpower",
        "1",
    )
    return streaming.StreamSchema(
        SCHEMA_ID,
        [neural, cursor, audio, feature],
        [descriptor],
        [streaming.UnitDescriptor(1, "V^2", "volt squared")],
    )


def _stream_specs(profile: PhysicalProfile, schema: Any) -> list[StreamRecording]:
    # Capacity is a fixed upper bound, not an allocation.  One chunk of margin
    # keeps sub-second structural runs legal without changing the physical
    # profile used by the default run.
    signals = {int(signal.id): signal for signal in schema.signals}
    regular = StreamTimingConfig(mode=StreamTimingMode.REGULAR)
    return [
        StreamRecording(
            stream_id="neural",
            signal=signals[NEURAL],
            metadata=StreamMetadata(
                channel_names=[f"electrode-{idx}" for idx in range(profile.neural_channels)]
            ),
            storage=StreamStorageConfig(
                capacity=profile.neural_samples + profile.neural_rate_hz,
                chunk_length=profile.neural_rate_hz,
                block_index_chunk_length=1_000,
            ),
            timing=regular,
        ),
        StreamRecording(
            stream_id="cursor",
            signal=signals[CURSOR],
            metadata=StreamMetadata(
                role=StreamRole.BEHAVIORAL,
                unit=("m", "m"),
                channel_names=("cursor-x", "cursor-y"),
            ),
            storage=StreamStorageConfig(
                capacity=profile.cursor_samples + profile.cursor_rate_hz,
                chunk_length=profile.cursor_rate_hz,
                block_index_chunk_length=100,
            ),
            timing=regular,
        ),
        StreamRecording(
            stream_id="audio",
            signal=signals[AUDIO],
            metadata=StreamMetadata(
                role=StreamRole.BEHAVIORAL, unit="pcm-count", channel_names=("audio",)
            ),
            storage=StreamStorageConfig(
                capacity=profile.audio_samples + profile.audio_rate_hz,
                chunk_length=profile.audio_rate_hz,
                block_index_chunk_length=100,
            ),
            timing=regular,
        ),
        StreamRecording(
            stream_id="bandpower",
            signal=signals[BANDPOWER],
            storage=StreamStorageConfig(
                capacity=profile.feature_observations + profile.feature_rate_hz,
                chunk_length=profile.feature_rate_hz,
                block_index_chunk_length=100,
            ),
            timing=StreamTimingConfig(
                mode=StreamTimingMode.REGULAR,
                segment_start_time_ns=profile.feature_window_ms * 1_000_000 // 2,
            ),
        ),
    ]


def _config(
    path: Path,
    profile: PhysicalProfile,
    schema: Any,
    *,
    frame_queue_capacity: int,
) -> RecorderConfig:
    return RecorderConfig(
        path=path,
        streams=_stream_specs(profile, schema),
        limits=RecorderLimits(
            frame_queue_capacity=frame_queue_capacity,
            control_queue_capacity=64,
            checkpoint_interval=32,
        ),
        metadata={"benchmark": "M6-13", "physical_profile": profile.document()},
    )


@dataclass(frozen=True, slots=True)
class _Templates:
    neural: np.ndarray
    cursor: np.ndarray
    audio: np.ndarray
    feature: np.ndarray


def _templates(profile: PhysicalProfile) -> _Templates:
    neural_words = _counter_words(
        profile.seed, NEURAL, profile.neural_rate_hz * profile.neural_channels
    )
    neural = (
        ((neural_words & 0xFFF).astype(np.int32) - 2_048)
        .astype("<i2")
        .reshape(profile.neural_rate_hz, profile.neural_channels)
    )

    cursor_words = _counter_words(
        profile.seed, CURSOR, profile.cursor_rate_hz * profile.cursor_dims
    )
    cursor_integer = (cursor_words & 0xFFFF).astype(np.int32) - 32_768
    cursor = np.ldexp(cursor_integer.astype(np.float32), -17).reshape(
        profile.cursor_rate_hz, profile.cursor_dims
    )

    audio_words = _counter_words(
        profile.seed, AUDIO, profile.audio_rate_hz * profile.audio_channels
    )
    audio = (
        ((audio_words & 0x3FFF).astype(np.int32) - 8_192)
        .astype("<i2")
        .reshape(profile.audio_rate_hz, profile.audio_channels)
    )

    feature_words = _counter_words(
        profile.seed, BANDPOWER, profile.feature_rate_hz * profile.n_features
    )
    feature_integer = 1_024 + (feature_words & 0xFFFF).astype(np.int64)
    feature = np.ldexp(feature_integer.astype(np.float64), -36).reshape(
        profile.feature_rate_hz, profile.n_features
    )
    return _Templates(
        np.ascontiguousarray(neural),
        np.ascontiguousarray(cursor, dtype="<f4"),
        np.ascontiguousarray(audio),
        np.ascontiguousarray(feature, dtype="<f8"),
    )


def _counter_words(seed: int, stream_id: int, count: int) -> np.ndarray:
    """Generate the frozen counter payload words shared with the native benchmark.

    The algorithm must stay bit-identical across Python and native so their
    payload checksums can be compared.
    """
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    idxs = np.arange(count, dtype=np.uint64)
    values = (
        np.uint64(seed)
        ^ np.uint64((stream_id * 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF)
        ^ (idxs * np.uint64(0x9E3779B97F4A7C15))
    ) & mask
    values ^= values >> np.uint64(30)
    values = (values * np.uint64(0xBF58476D1CE4E5B9)) & mask
    values ^= values >> np.uint64(27)
    values = (values * np.uint64(0x94D049BB133111EB)) & mask
    values ^= values >> np.uint64(31)
    return values


def _template_checksums(templates: _Templates) -> dict[str, str]:
    return {
        name: hashlib.sha256(memoryview(value).cast("B")).hexdigest()
        for name, value in (
            ("neural", templates.neural),
            ("cursor", templates.cursor),
            ("audio", templates.audio),
            ("bandpower", templates.feature),
        )
    }


class _WindowsIoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


def _resource_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cpu_seconds": time.process_time(),
        "peak_resident_memory_bytes": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "io_supported": False,
    }
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        kernel32.GetProcessIoCounters.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.GetProcessIoCounters.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        memory = WindowsProcessMemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
            result["peak_resident_memory_bytes"] = int(memory.PeakWorkingSetSize)
        io = _WindowsIoCounters()
        if kernel32.GetProcessIoCounters(handle, ctypes.byref(io)):
            result.update(
                read_bytes=int(io.ReadTransferCount),
                write_bytes=int(io.WriteTransferCount),
                io_supported=True,
            )
        return result
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if platform.system() == "Darwin" else 1024
    result["peak_resident_memory_bytes"] = int(usage.ru_maxrss * scale)
    if sys.platform.startswith("linux"):
        values = {}
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip())
        result.update(
            read_bytes=values.get("read_bytes", 0),
            write_bytes=values.get("write_bytes", 0),
            io_supported=True,
        )
    return result


def _resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "process_cpu_seconds": max(0.0, after["cpu_seconds"] - before["cpu_seconds"]),
        "peak_resident_memory_bytes": after["peak_resident_memory_bytes"],
        "process_io_supported": bool(before["io_supported"] and after["io_supported"]),
        "process_read_bytes": max(0, after["read_bytes"] - before["read_bytes"]),
        "process_write_bytes": max(0, after["write_bytes"] - before["write_bytes"]),
    }


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _codec_metadata(root: Path) -> dict[str, Any]:
    codecs: set[str] = set()
    for metadata in root.rglob("zarr.json"):
        value = json.loads(metadata.read_text(encoding="utf-8"))
        for codec in value.get("codecs", ()):
            if isinstance(codec, dict):
                name = codec.get("name") or codec.get("configuration", {}).get("name")
                codecs.add(str(name or json.dumps(codec, sort_keys=True)))
    return {
        "array_encoding": "zarr_v3",
        "zarr_codecs": sorted(codecs),
        "nrf_object_checksum": "sha256",
        "native_spool_checksum": "crc32c",
    }


def _finalize_worker(source: Path, output: Path, payload_bytes: int) -> dict[str, Any]:
    before = _resource_snapshot()
    started = time.perf_counter_ns()
    report = finalize_spool(source, output)
    elapsed_ns = time.perf_counter_ns() - started
    after = _resource_snapshot()
    output_bytes = _directory_bytes(output)
    return {
        "schema_version": 1,
        "stage": "native_finalization",
        "status": "ok",
        "elapsed_ns": elapsed_ns,
        "report_duration_seconds": report.duration_seconds,
        "payload_bytes": payload_bytes,
        "spool_bytes": source.stat().st_size,
        "output_bytes": output_bytes,
        "output_to_payload_size_ratio": output_bytes / payload_bytes,
        "throughput_payload_bytes_per_second": payload_bytes * 1e9 / elapsed_ns,
        "frames": report.counts.frames,
        "signal_blocks": report.counts.signal_blocks,
        "termination_kind": report.termination_kind,
        "accounting_origin": report.accounting_origin,
        **_codec_metadata(output),
        **_resource_delta(before, after),
    }


def _recovery_worker(source: Path, output: Path, payload_bytes: int) -> dict[str, Any]:
    before = _resource_snapshot()
    started = time.perf_counter_ns()
    diagnose_started = time.perf_counter_ns()
    diagnosis = diagnose_spool(source)
    diagnose_ns = time.perf_counter_ns() - diagnose_started
    repair_started = time.perf_counter_ns()
    repair = repair_spool(source)
    repair_ns = time.perf_counter_ns() - repair_started
    finalize_started = time.perf_counter_ns()
    report = finalize_spool(source, output)
    finalize_ns = time.perf_counter_ns() - finalize_started
    elapsed_ns = time.perf_counter_ns() - started
    after = _resource_snapshot()
    output_bytes = _directory_bytes(output)
    return {
        "schema_version": 1,
        "stage": "torn_tail_recovery",
        "status": "ok",
        "elapsed_ns": elapsed_ns,
        "diagnose_ns": diagnose_ns,
        "repair_ns": repair_ns,
        "finalize_ns": finalize_ns,
        "payload_bytes": payload_bytes,
        "repaired_spool_bytes": source.stat().st_size,
        "output_bytes": output_bytes,
        "output_to_payload_size_ratio": output_bytes / payload_bytes,
        "throughput_payload_bytes_per_second": payload_bytes * 1e9 / elapsed_ns,
        "diagnosis_status": diagnosis.status,
        "diagnosis_codes": list(diagnosis.codes),
        "invalid_tail_bytes": diagnosis.invalid_tail_bytes,
        "repair_removed_bytes": repair.removed_bytes,
        "repair_codes": list(repair.codes),
        "termination_kind": report.termination_kind,
        **_codec_metadata(output),
        **_resource_delta(before, after),
    }


def _image_worker(session: Path, image: Path, config_path: Path) -> dict[str, Any]:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if document["mode"] in ("exact_frames", "recorded_projection"):
        message = document.get("message_range")
        config = ReplayConfig(
            mode=document["mode"],
            selected_streams=tuple(document.get("selected_streams", ())),
            message_range=None if message is None else MessageRange(**message),
        )
    else:
        config = ReplayConfig(
            mode="stream_frames",
            stream_ranges={
                key: StreamReplayRange(**value) for key, value in document["stream_ranges"].items()
            },
        )
    before = _resource_snapshot()
    started = time.perf_counter_ns()
    built = build_replay_image(session, config, path=image)
    built.close()
    elapsed_ns = time.perf_counter_ns() - started
    after = _resource_snapshot()
    image_bytes = image.stat().st_size
    return {
        "schema_version": 1,
        "stage": "replay_image_build",
        "status": "ok",
        "mode": document["mode"],
        "range": document.get("range_name", "full"),
        "elapsed_ns": elapsed_ns,
        "image_bytes": image_bytes,
        "image_checksum": "sha256",
        "throughput_image_bytes_per_second": image_bytes * 1e9 / elapsed_ns,
        **_resource_delta(before, after),
    }


def _worker_main(arguments: argparse.Namespace) -> int:
    if arguments.worker == "finalize":
        row = _finalize_worker(
            Path(arguments.source_path), Path(arguments.output_path), arguments.payload_bytes
        )
    elif arguments.worker == "recover":
        row = _recovery_worker(
            Path(arguments.source_path), Path(arguments.output_path), arguments.payload_bytes
        )
    elif arguments.worker == "image":
        row = _image_worker(
            Path(arguments.source_path), Path(arguments.output_path), Path(arguments.config_path)
        )
    else:  # pragma: no cover - argparse restricts it
        raise AssertionError(arguments.worker)
    print(json.dumps(row, sort_keys=True, default=str))
    return 0 if row["status"] == "ok" else 1


def _find_native_executable(requested: str | None) -> Path:
    if requested:
        path = Path(requested).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    suffix = ".exe" if os.name == "nt" else ""
    candidates = sorted(
        Path("build").rglob(f"neurale_recording_replay_benchmark{suffix}"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "neurale_recording_replay_benchmark is not built; pass --native-executable"
        )
    return candidates[0].resolve()


def _run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {subprocess.list2cmdline(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    row = json.loads(lines[0])
    row["exact_command"] = subprocess.list2cmdline(command)
    row["command_cwd"] = str(cwd)
    return row


def _storage_metadata(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    result: dict[str, Any] = {
        "storage_path": str(path.resolve()),
        "storage_total_bytes": usage.total,
        "storage_free_bytes_at_start": usage.free,
        "storage_filesystem": None,
        "storage_device": None,
    }
    if os.name == "nt":
        root = Path(path.resolve().anchor)
        volume = ctypes.create_unicode_buffer(256)
        filesystem = ctypes.create_unicode_buffer(256)
        serial = ctypes.c_ulong()
        max_component = ctypes.c_ulong()
        flags = ctypes.c_ulong()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if kernel32.GetVolumeInformationW(
            str(root),
            volume,
            len(volume),
            ctypes.byref(serial),
            ctypes.byref(max_component),
            ctypes.byref(flags),
            filesystem,
            len(filesystem),
        ):
            result["storage_filesystem"] = filesystem.value
            result["storage_device"] = volume.value or str(root)
    elif sys.platform.startswith("linux"):
        query = subprocess.run(
            ["stat", "-f", "-c", "%T|%m", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if query.returncode == 0:
            filesystem, _, device = query.stdout.strip().partition("|")
            result["storage_filesystem"] = filesystem or None
            result["storage_device"] = device or None
    return result


def _physical_memory_bytes() -> int | None:
    if os.name == "nt":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    names = os.sysconf_names
    if "SC_PHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    return None


def _host_metadata(native_executable: Path, work_dir: Path) -> dict[str, Any]:
    import neurale._native as native

    metadata = benchmark_metadata(native)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], text=True, capture_output=True, check=False
    )
    metadata.update(
        benchmark="m6_record_replay",
        benchmark_schema_version=1,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        git_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        native_executable=str(native_executable),
        native_executable_sha256=hashlib.sha256(native_executable.read_bytes()).hexdigest(),
        physical_memory_bytes=_physical_memory_bytes(),
        performance_thresholds="none",
        claim_scope="this benchmark profile, build, host, OS, and storage only",
        universal_realtime_claim=False,
        **_storage_metadata(work_dir),
    )
    return metadata


def _verify_same_native_build(
    extension_metadata: dict[str, Any], helper_metadata: dict[str, Any]
) -> None:
    key_pairs = (
        ("native_version", "native_version"),
        ("native_abi_version", "native_abi_version"),
        ("compiler", "compiler"),
        ("build_type", "build_type"),
        ("cpu_math_backend", "cpu_math_backend"),
        ("fft_backend", "fft_backend"),
        ("cuda_compiled", "cuda_compiled"),
        ("threading_backend", "threading_backend"),
        ("threads", "native_threads"),
    )
    mismatches = {
        extension_key: {
            "python_extension": extension_metadata.get(extension_key),
            "native_helper": helper_metadata.get(helper_key),
        }
        for extension_key, helper_key in key_pairs
        if extension_metadata.get(extension_key) != helper_metadata.get(helper_key)
    }
    if mismatches:
        raise RuntimeError(
            "neurale._native and neurale_recording_replay_benchmark are from different "
            f"builds: {json.dumps(mismatches, sort_keys=True)}"
        )


def _write_plan(path: Path, profile: PhysicalProfile, queue_capacity: int) -> None:
    schema = _schema(profile)
    plan = compile_recording_plan(
        _config(
            path.parent / "unused-native-output.nrf",
            profile,
            schema,
            frame_queue_capacity=queue_capacity,
        ),
        schema,
    )
    path.write_bytes(canonical_json_bytes(plan.document()))


def _image_config(
    mode: str, profile: PhysicalProfile, *, range_seconds: float | None
) -> dict[str, Any]:
    frames = profile.frames
    if range_seconds is not None:
        frames = min(profile.frames, max(1, round(range_seconds * FRAMES_PER_SECOND)))
    result: dict[str, Any] = {
        "mode": mode,
        "range_name": "full" if range_seconds is None else f"{range_seconds:g}_seconds",
    }
    if mode in ("exact_frames", "recorded_projection"):
        result["message_range"] = {"start": 0, "stop": frames}
        if mode == "recorded_projection":
            result["selected_streams"] = list(STREAM_IDS)
    else:
        behavior = (frames + 9) // 10
        result["stream_ranges"] = {
            "neural": {"unit": "block_ordinal", "start": 0, "stop": frames},
            "cursor": {"unit": "block_ordinal", "start": 0, "stop": behavior},
            "audio": {"unit": "block_ordinal", "start": 0, "stop": behavior},
            "bandpower": {"unit": "block_ordinal", "start": 0, "stop": behavior},
        }
    return result


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or "elapsed_ns" not in row or row.get("aggregate"):
            continue
        key = (str(row["stage"]), str(row.get("mode", "")), str(row.get("pacing", "")))
        groups.setdefault(key, []).append(row)
    aggregates = []
    for (stage, mode, pacing), group in groups.items():
        values = [int(row["elapsed_ns"]) for row in group]
        timing = summarize_ns(values)
        commands = [str(row["exact_command"]) for row in group if row.get("exact_command")]
        build_keys = (
            "native_version",
            "native_abi_version",
            "compiler",
            "build_type",
            "cpu_math_backend",
            "fft_backend",
            "cuda_compiled",
            "cpu_architecture",
            "cpu_vendor",
            "cpu_model",
            "logical_cores",
            "threading_backend",
            "native_threads",
        )
        measured_build = {key: group[0][key] for key in build_keys if key in group[0]}
        throughput_medians = {}
        throughput_keys = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if key.startswith("throughput_") and isinstance(value, (int, float))
            }
        )
        for key in throughput_keys:
            values_for_key = [row.get(key) for row in group]
            if all(isinstance(value, (int, float)) for value in values_for_key):
                throughput_medians[f"{key}_median"] = statistics.median(values_for_key)
        aggregates.append(
            {
                "schema_version": 1,
                "stage": stage,
                "mode": mode or None,
                "pacing": pacing or None,
                "status": "ok",
                "aggregate": True,
                "repetitions": timing.samples,
                "exact_command": "derived aggregate; see exact_commands",
                "exact_commands": commands,
                **measured_build,
                **timing.fields("ns", prefix="elapsed_"),
                **throughput_medians,
            }
        )
    return aggregates


def run(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    native = _find_native_executable(arguments.native_executable)
    profile = PhysicalProfile(arguments.duration_seconds, arguments.seed)
    work = Path(arguments.work_dir).resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"benchmark work directory must be empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    metadata = _host_metadata(native, work)
    build_probe = _run_json([str(native), "metadata"], cwd=root)
    _verify_same_native_build(metadata, build_probe)
    templates = _templates(profile)
    python_payload_checksums = _template_checksums(templates)
    source_probe = _run_json(
        [str(native), "source-checksums", "--seed", str(arguments.seed)], cwd=root
    )
    native_payload_checksums = {
        stream_id: str(source_probe[f"{stream_id}_sha256"]) for stream_id in STREAM_IDS
    }
    if native_payload_checksums != python_payload_checksums:
        raise RuntimeError(
            "Python and native M6-13 physical payloads differ: "
            f"{json.dumps({'python': python_payload_checksums, 'native': native_payload_checksums}, sort_keys=True)}"
        )
    common = {
        **metadata,
        "physical_profile": profile.document(),
        "repetitions": arguments.runs,
        "warmup_seconds": arguments.warmup_seconds,
        "payload_generator": "m6-13-counter-v1",
        "payload_template_sha256": python_payload_checksums,
        "native_build_probe_command": build_probe["exact_command"],
        "native_payload_probe_command": source_probe["exact_command"],
        "queue_configuration": {
            "native_frame_capacity": arguments.native_queue_capacity,
            "native_control_capacity": 64,
            "critical_edge_capacity": arguments.native_queue_capacity,
        },
        "payload_configuration": profile.document(),
        "replay_range_seconds": arguments.replay_range_seconds,
        "codec_configuration": {
            "nrf": "writer_default_zarr_v3_codecs",
            "native_spool": "none",
        },
        "checksum_configuration": {
            "nrf_objects": "sha256",
            "native_spool": "crc32c",
            "replay_image": "sha256+crc32c",
        },
        "durability_configuration": {
            "critical_spool": "buffered",
            "memory_backend_crash_durable": False,
            "mapped_backend": "Linux tmpfs only; not selected unless explicitly requested",
        },
        "warmup_policy": (
            "native recording and each native replay mode/pacing pair run one "
            "separate process-level warm-up; one-shot finalization, recovery, and image builds "
            "have no discarded warm-up"
        ),
        "warmup_commands": [],
    }
    rows: list[dict[str, Any]] = []

    def python_worker(*extra: str) -> dict[str, Any]:
        command = [sys.executable, str(Path(__file__).resolve()), *extra]
        return _run_json(command, cwd=root)

    plan_path = work / "native-plan.json"
    _write_plan(plan_path, profile, arguments.native_queue_capacity)
    spool_path = work / "native.spool"
    spool_capacity = max(arguments.spool_capacity_bytes, profile.payload_bytes * 2)
    if arguments.warmup_seconds > 0:
        warmup_frames = PhysicalProfile(arguments.warmup_seconds, arguments.seed).frames
        warmup_spool = work / "warmup-native.spool"
        warmup_command = [
            str(native),
            "record",
            "--frames",
            str(warmup_frames),
            "--seed",
            str(arguments.seed),
            "--plan",
            str(plan_path),
            "--spool",
            str(warmup_spool),
            "--backend",
            arguments.spool_backend,
            "--spool-capacity",
            str(spool_capacity),
            "--queue-capacity",
            str(arguments.native_queue_capacity),
            "--checkpoint-interval",
            str(arguments.checkpoint_interval),
            "--worker-poll-ns",
            str(arguments.worker_poll_ns),
            "--drain-timeout-ns",
            str(arguments.drain_timeout_ns),
        ]
        _run_json(warmup_command, cwd=root)
        common["warmup_commands"].append(subprocess.list2cmdline(warmup_command))
        warmup_spool.unlink(missing_ok=True)
    for repetition in range(arguments.runs):
        output = (
            spool_path if repetition + 1 == arguments.runs else work / f"native-{repetition}.spool"
        )
        command = [
            str(native),
            "record",
            "--frames",
            str(profile.frames),
            "--seed",
            str(arguments.seed),
            "--plan",
            str(plan_path),
            "--spool",
            str(output),
            "--backend",
            arguments.spool_backend,
            "--spool-capacity",
            str(spool_capacity),
            "--queue-capacity",
            str(arguments.native_queue_capacity),
            "--checkpoint-interval",
            str(arguments.checkpoint_interval),
            "--worker-poll-ns",
            str(arguments.worker_poll_ns),
            "--drain-timeout-ns",
            str(arguments.drain_timeout_ns),
        ]
        row = _run_json(command, cwd=root)
        row = {
            **common,
            **row,
            "repetition": repetition,
            "spool_capacity_bytes": spool_capacity,
        }
        rows.append(row)
        if output != spool_path:
            output.unlink(missing_ok=True)

    native_session = work / "native-final.nrf"
    for repetition in range(arguments.runs):
        output = (
            native_session if repetition + 1 == arguments.runs else work / f"final-{repetition}.nrf"
        )
        row = python_worker(
            "--worker",
            "finalize",
            "--source-path",
            str(spool_path),
            "--output-path",
            str(output),
            "--payload-bytes",
            str(profile.payload_bytes),
        )
        row = {**common, **row, "repetition": repetition}
        rows.append(row)
        if output != native_session:
            shutil.rmtree(output, ignore_errors=True)

    for repetition in range(arguments.runs):
        damaged = work / f"recovery-{repetition}.spool"
        shutil.copyfile(spool_path, damaged)
        with damaged.open("ab") as stream:
            stream.write(b"NRF-M6-13-RECOVERABLE-UNCOMMITTED-TAIL")
        output = work / f"recovered-{repetition}.nrf"
        row = python_worker(
            "--worker",
            "recover",
            "--source-path",
            str(damaged),
            "--output-path",
            str(output),
            "--payload-bytes",
            str(profile.payload_bytes),
        )
        row = {**common, **row, "repetition": repetition}
        rows.append(row)
        shutil.rmtree(output, ignore_errors=True)
        damaged.unlink(missing_ok=True)
        for quarantine in work.glob(f"recovery-{repetition}.spool.quarantined*"):
            quarantine.unlink(missing_ok=True)

    for mode in ("exact_frames", "recorded_projection", "stream_frames"):
        config_path = work / f"{mode}-replay-range.json"
        range_document = _image_config(mode, profile, range_seconds=arguments.replay_range_seconds)
        config_path.write_text(json.dumps(range_document, sort_keys=True), encoding="utf-8")
        image = work / f"{mode}-replay-range.nrimg"
        for repetition in range(arguments.runs):
            output = (
                image
                if repetition + 1 == arguments.runs
                else work / f"{mode}-replay-range-{repetition}.nrimg"
            )
            row = python_worker(
                "--worker",
                "image",
                "--source-path",
                str(native_session),
                "--output-path",
                str(output),
                "--config-path",
                str(config_path),
            )
            row = {**common, **row, "repetition": repetition}
            rows.append(row)
            if output != image:
                output.unlink(missing_ok=True)

        for pacing in ("as_fast_as_possible", "recorded", "step"):
            if arguments.warmup_seconds > 0:
                warmup_command = [
                    str(native),
                    "replay",
                    "--image",
                    str(image),
                    "--pacing",
                    pacing,
                    "--speed-factor",
                    str(arguments.replay_speed_factor),
                ]
                _run_json(warmup_command, cwd=root)
                common["warmup_commands"].append(subprocess.list2cmdline(warmup_command))
            for repetition in range(arguments.runs):
                command = [
                    str(native),
                    "replay",
                    "--image",
                    str(image),
                    "--pacing",
                    pacing,
                    "--speed-factor",
                    str(arguments.replay_speed_factor),
                ]
                row = _run_json(command, cwd=root)
                row = {
                    **common,
                    **row,
                    "repetition": repetition,
                    "replay_range": range_document["range_name"],
                }
                rows.append(row)
        image.unlink(missing_ok=True)

    rows.extend({**common, **row} for row in _aggregate(rows))
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/m6-record-replay.jsonl"))
    parser.add_argument("--work-dir", type=Path, default=Path("build/m6-record-replay-work"))
    parser.add_argument("--native-executable")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--native-queue-capacity", type=int, default=8_192)
    parser.add_argument("--spool-backend", choices=("memory", "mapped"), default="memory")
    parser.add_argument("--spool-capacity-bytes", type=int, default=256 << 20)
    parser.add_argument("--checkpoint-interval", type=int, default=32)
    parser.add_argument("--worker-poll-ns", type=int, default=50_000)
    parser.add_argument("--drain-timeout-ns", type=int, default=30_000_000_000)
    parser.add_argument("--replay-speed-factor", type=float, default=1.0)
    parser.add_argument("--replay-range-seconds", type=float, default=2.0)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="print the resolved physical profile without running a benchmark",
    )
    parser.add_argument(
        "--worker", choices=("finalize", "recover", "image"), help=argparse.SUPPRESS
    )
    parser.add_argument("--source-path", help=argparse.SUPPRESS)
    parser.add_argument("--output-path", help=argparse.SUPPRESS)
    parser.add_argument("--config-path", help=argparse.SUPPRESS)
    parser.add_argument("--payload-bytes", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.runs < 1:
        raise SystemExit("--runs must be positive")
    if not math.isfinite(arguments.warmup_seconds) or arguments.warmup_seconds < 0:
        raise SystemExit("--warmup-seconds must be finite and non-negative")
    if not math.isfinite(arguments.replay_range_seconds) or arguments.replay_range_seconds <= 0:
        raise SystemExit("--replay-range-seconds must be finite and positive")
    if arguments.worker:
        return _worker_main(arguments)
    if arguments.profile_only:
        print(
            json.dumps(
                PhysicalProfile(arguments.duration_seconds, arguments.seed).document(),
                sort_keys=True,
            )
        )
        return 0
    rows = run(arguments)
    write_records(arguments.output, rows)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
