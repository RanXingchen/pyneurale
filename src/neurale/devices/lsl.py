# SPDX-License-Identifier: MIT
"""Optional, single-stream LSL acquisition. Importing this module loads no LSL library."""

from __future__ import annotations

import ctypes
import math
import threading
import time
from typing import Any
from xml.etree import ElementTree

import numpy as np

from neurale.exceptions import DependencyError

from .provider import PythonProvider, Signal, host_time_ns

_FORMATS = {1: "float32", 2: "float64", 4: "int32", 5: "int16"}
_SELECTORS = {"source_id": "source_id", "stream_name": "name", "stream_type": "type", "uid": "uid"}


def provider() -> PythonProvider:
    return PythonProvider("neurale.devices.lsl:LSLProvider")


def _lsl():
    try:
        import pylsl
    except (ImportError, RuntimeError, OSError) as error:
        raise DependencyError(
            'LSL requires pip install "pyneurale[lsl]" and a liblsl shared library; '
            "set PYLSL_LIB to its path if necessary."
        ) from error
    return pylsl


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return value


def _summary(info) -> dict[str, Any]:
    return {
        key: getattr(info, key)()
        for key in (
            "name",
            "type",
            "source_id",
            "uid",
            "hostname",
            "channel_count",
            "nominal_srate",
            "channel_format",
        )
    }


def discover(timeout: float = 1.0) -> list[dict[str, Any]]:
    """List visible streams without subscribing to their samples."""
    return [
        _summary(info) for info in _lsl().resolve_streams(wait_time=_positive(timeout, "timeout"))
    ]


def _resolve(lsl, selectors, timeout):
    if not selectors or any(not isinstance(v, str) or not v for v in selectors.values()):
        raise ValueError("provide at least one nonempty LSL stream selector")
    matches = [
        info
        for info in lsl.resolve_streams(wait_time=timeout)
        if all(getattr(info, _SELECTORS[key])() == value for key, value in selectors.items())
    ]
    if not matches:
        raise TimeoutError("no LSL stream matches the selectors")
    if len(matches) != 1:
        raise ValueError(
            f"LSL selector is ambiguous: {len(matches)} streams match; use source_id or uid"
        )
    return matches[0]


def _channels(info):
    nodes = ElementTree.fromstring(info.as_xml()).findall("./desc/channels/channel")
    return [
        {
            "label": (nodes[i].findtext("label") if i < len(nodes) else None) or f"ch{i + 1}",
            "unit": (nodes[i].findtext("unit") if i < len(nodes) else None) or "",
        }
        for i in range(info.channel_count())
    ]


def inspect(*, timeout: float = 1.0, **selectors) -> dict[str, Any]:
    """Resolve one stream and obtain its full XML and original channel units."""
    if selectors.keys() - _SELECTORS.keys():
        raise ValueError("unknown LSL selector")
    lsl = _lsl()
    inlet = lsl.StreamInlet(_resolve(lsl, selectors, _positive(timeout, "timeout")), recover=False)
    try:
        info = inlet.info(timeout=timeout)
        return {**_summary(info), "xml": info.as_xml(), "channels": _channels(info)}
    finally:
        inlet.close_stream()


def _correction(inlet, timeout):
    # pylsl exposes only the offset; use the same loaded library for uncertainty.
    from pylsl.lib import lib

    fn = lib.lsl_time_correction_ex
    fn.restype = ctypes.c_double
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
    ]
    remote, uncertainty, error = ctypes.c_double(), ctypes.c_double(), ctypes.c_int()
    offset = fn(
        inlet.obj, ctypes.byref(remote), ctypes.byref(uncertainty), timeout, ctypes.byref(error)
    )
    if error.value == -1:
        raise TimeoutError("LSL clock correction timed out")
    if error.value:
        raise RuntimeError(f"LSL clock correction failed: {error.value}")
    if (
        not all(math.isfinite(v) for v in (offset, remote.value, uncertainty.value))
        or uncertainty.value < 0
    ):
        raise RuntimeError("invalid LSL clock correction")
    return offset, remote.value, uncertainty.value


class LSLProvider:
    """Producer-process adapter. Data is signal 1; paired timestamps are signal 2."""

    def __init__(self):
        self._inlet = None
        self._thread = None
        self._stop = threading.Event()
        self._error = None
        self._remote_measurement = None

    def open(self, config):
        allowed = {
            *_SELECTORS,
            "max_samples",
            "buffer_seconds",
            "resolve_timeout",
            "data_timeout",
            "max_lag",
        }
        if config.keys() - allowed:
            raise ValueError(f"unknown LSL options: {sorted(config.keys() - allowed)}")
        self._max_samples = config.get("max_samples", 256)
        buffer_seconds = config.get("buffer_seconds", 2)
        for name, value in (("max_samples", self._max_samples), ("buffer_seconds", buffer_seconds)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        timeout = _positive(config.get("resolve_timeout", 1.0), "resolve_timeout")
        self._data_timeout = _positive(config.get("data_timeout", 5.0), "data_timeout")
        self._max_lag = _positive(config.get("max_lag", 0.5), "max_lag")
        self._lsl = _lsl()
        info = _resolve(self._lsl, {k: v for k, v in config.items() if k in _SELECTORS}, timeout)
        self._inlet = self._lsl.StreamInlet(
            info, max_buflen=buffer_seconds, recover=False, processing_flags=0
        )
        info = self._inlet.info(timeout=timeout)
        rate = _positive(info.nominal_srate(), "nominal sample rate")
        if info.channel_format() not in _FORMATS:
            raise ValueError(
                "LSL supports only int16, int32, float32 and float64 continuous streams"
            )
        channels = _channels(info)
        units = {c["unit"] for c in channels}
        unit = {"V": "volts", "volts": "volts", "A": "amperes", "1": "dimensionless"}.get(
            next(iter(units)) if len(units) == 1 else "", "unspecified"
        )
        self._signals = (
            Signal(
                "data",
                info.channel_count(),
                rate,
                self._max_samples,
                dtype=_FORMATS[info.channel_format()],
                unit=unit,
                channel_names=tuple(c["label"] for c in channels),
            ),
            Signal(
                "timestamps",
                2,
                rate,
                self._max_samples,
                dtype="float64",
                channel_names=("lsl_source_seconds", "host_monotonic_seconds"),
            ),
        )
        self._data = np.empty(
            (self._max_samples, info.channel_count()), dtype=self._signals[0].dtype
        )
        self._times = np.empty((self._max_samples, 2), dtype=np.float64)

    def describe(self):
        return self._signals

    def _sync(self, timeout):
        correction, remote, uncertainty = _correction(self._inlet, timeout)
        if self._remote_measurement is not None:
            if remote < self._remote_measurement:
                raise RuntimeError("LSL clock measurement moved backwards; reopen the device")
            if remote == self._remote_measurement:
                return  # liblsl can return the same cached estimate indefinitely.
        before = host_time_ns()
        local = self._lsl.local_clock()
        after = host_time_ns()
        self._mapping = (
            local - correction,
            (before + after) // 2,
            math.ceil(uncertainty * 1e9) + (after - before + 1) // 2,
        )
        self._remote_measurement = remote
        # Account for an estimate already aged when first observed. Translate its
        # remote measurement time to the local LSL clock before comparing it.
        age = max(0.0, local - (remote + correction))
        self._synced_at = time.monotonic() - age

    def start(self, ingress):
        self._sync(1.0)
        self._inlet.open_stream(timeout=1.0)
        self._inlet.flush()  # Explicit acquisition boundary; never flush while running.
        self._thread = threading.Thread(target=self._run, args=(ingress,), name="lsl-acquisition")
        self._thread.start()

    def _run(self, ingress):
        try:
            self._acquire(ingress)
        except BaseException as error:
            if not self._stop.is_set() and not ingress.cancelled:
                self._error = error
                ingress.fail()

    def _acquire(self, ingress):
        last_data = last_sync_attempt = time.monotonic()
        sample_index = 0
        origin = previous = previous_host = None
        while not self._stop.is_set() and not ingress.cancelled:
            now = time.monotonic()
            if self._inlet.was_clock_reset():
                raise RuntimeError("LSL source clock reset; reopen the device")
            if now - last_sync_attempt >= 1.0:
                last_sync_attempt = now
                try:
                    self._sync(0.0)
                except TimeoutError:
                    pass
            if now - self._synced_at > 5.0:
                raise TimeoutError("LSL clock mapping is stale")
            _, timestamps = self._inlet.pull_chunk(
                timeout=0.0, max_samples=self._max_samples, dest_obj=self._data
            )
            count = len(timestamps)
            if not count:
                if now - last_data > self._data_timeout:
                    raise TimeoutError("LSL stream stopped producing samples")
                self._stop.wait(0.001)
                continue
            last_data = now
            raw = self._times[:count, 0]
            raw[:] = timestamps
            if (
                not np.isfinite(raw).all()
                or np.any(np.diff(raw) < 0)
                or (previous is not None and raw[0] < previous)
            ):
                raise RuntimeError("invalid or backwards LSL timestamps")
            if origin is None:
                origin = raw[0]
            remote_ref, host_ref, uncertainty = self._mapping
            corrected = self._times[:count, 1]
            corrected[:] = (raw - remote_ref) + host_ref * 1e-9
            if corrected[0] < 0 or (previous_host is not None and corrected[0] < previous_host):
                raise RuntimeError("LSL clock mapping moved backwards")
            lag = host_time_ns() * 1e-9 - corrected[0]
            if lag > self._max_lag or lag < -(uncertainty * 1e-9 + self._max_lag):
                raise RuntimeError("LSL sample time is outside the permitted lag")
            tick = round((raw[0] - origin) * 1e9)
            metadata = {
                "sample_index": sample_index,
                "device_tick": tick,
                "tick_reference": tick,
                "host_reference_ns": round(corrected[0] * 1e9),
                "tick_rate_numerator": 1_000_000_000,
                "tick_rate_denominator": 1,
                "uncertainty_ns": uncertainty,
                "clock_generation": 1,
                "synchronized": 1,
            }
            ingress.publish_frame(
                ((0, self._data[:count], metadata), (1, self._times[:count], metadata))
            )
            sample_index += count
            previous, previous_host = raw[-1], corrected[-1]

    def cancel(self):
        self._stop.set()

    def close(self):
        self.cancel()
        try:
            if self._thread is not None:
                self._thread.join()
        finally:
            if self._inlet is not None:
                self._inlet.close_stream()
                self._inlet = None
        if self._error is not None:
            raise RuntimeError(f"LSL acquisition failed: {self._error}") from self._error


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Discover LSL streams or inspect one stream's XML")
    parser.add_argument("--source-id")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()
    result = (
        inspect(source_id=args.source_id, timeout=args.timeout)
        if args.source_id
        else discover(args.timeout)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
