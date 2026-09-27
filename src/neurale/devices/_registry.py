# SPDX-License-Identifier: MIT
"""Device discovery and control plane; vendor imports are deferred until open."""

from __future__ import annotations

import json
import math
import multiprocessing
import shutil
import tempfile
import threading
import weakref
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from neurale.exceptions import (
    DependencyError,
    StreamExecutionError,
    StreamStateError,
    StreamTimeoutError,
)

from .provider import NativeProvider, PythonProvider, _native_signals


def list() -> tuple[str, ...]:
    """List installed provider names without loading entry points or scanning hardware."""
    return tuple(sorted({entry.name for entry in entry_points(group="neurale.devices")}))


def open(name: str, *, queue_capacity: int = 256, timeout: float = 5.0, **config: Any) -> Device:
    """Connect a registered device; call start after preparing its StreamRunner."""
    matches = tuple(e for e in entry_points(group="neurale.devices") if e.name == name)
    if len(matches) != 1:
        raise DependencyError(f"expected one installed provider for {name!r}, found {len(matches)}")
    try:
        definition = matches[0].load()()
    except ImportError as error:
        raise DependencyError(f"cannot load device provider {name!r}: {error}") from error
    return Device(definition, config=config, queue_capacity=queue_capacity, timeout=timeout)


class Device:
    """Own an external provider and its native source.

    The context manager closes resources; acquisition is explicitly started after
    runner.prepare(). Stop and join the runner before closing the device.
    """

    def __init__(
        self,
        provider: NativeProvider | PythonProvider,
        *,
        config: dict[str, Any] | None = None,
        queue_capacity: int = 256,
        timeout: float = 5.0,
    ) -> None:
        from neurale._native_loader import load_native_namespace

        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity < 2
        ):
            raise ValueError("queue_capacity must be an integer of at least two")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not isinstance(provider, (NativeProvider, PythonProvider)):
            raise TypeError("provider must be NativeProvider or PythonProvider")
        self._provider = provider
        self._config = dict(config or {})
        self._capacity = queue_capacity
        self._timeout = timeout
        self._closed = False
        self._last_stats = None
        self._process = None
        self._connection = None
        self._monitor = None
        self._stopping = threading.Event()
        self._directory = tempfile.mkdtemp(prefix="neurale-device-")
        self._path = str(Path(self._directory, "ingress.bin"))
        self._source = None
        try:
            native = load_native_namespace("devices")
            if isinstance(provider, NativeProvider):
                try:
                    library = Path(provider.library).resolve(strict=True)
                except OSError as error:
                    raise DependencyError(
                        f"device library is unavailable: {provider.library}"
                    ) from error
                try:
                    self._source = native.GenericDeviceSource(
                        str(library),
                        json.dumps(self._config, allow_nan=False),
                        self._path,
                        queue_capacity,
                    )
                except RuntimeError as error:
                    raise StreamExecutionError(str(error)) from error
            else:
                self._signals = self._spawn()
                self._source = native.GenericDeviceSource(
                    _native_signals(self._signals), self._path, queue_capacity
                )
                reference = weakref.ref(self)

                def reset_process() -> None:
                    device = reference()
                    if device is None or device._closed:
                        raise StreamStateError("device is closed")
                    shutdown_error = device._stop_process()
                    if shutdown_error is not None:
                        raise shutdown_error
                    try:
                        signals = device._spawn()
                        if signals != device._signals:
                            raise StreamStateError("device schema changed; open a new device")
                    except BaseException:
                        device._stop_process()
                        raise

                self._source.set_reset_action(reset_process)
            self._source.ingress.remove_directory_on_destroy()
        except BaseException:
            self._stop_process()
            self._source = None
            shutil.rmtree(self._directory)
            raise

    @property
    def source(self) -> Any:
        if self._closed:
            raise StreamStateError("device is closed")
        return self._source

    @property
    def _native_session_id(self) -> int:
        return self.source._native_session_id

    @property
    def schema(self) -> Any:
        return self.source.schema

    @property
    def stats(self) -> dict[str, int]:
        if self._closed:
            return dict(self._last_stats)
        return self.source.ingress.stats

    def _receive(self, expected: str) -> Any:
        if not self._connection.poll(self._timeout):
            raise StreamTimeoutError(f"device process did not return {expected}")
        try:
            kind, result = self._connection.recv()
        except (EOFError, OSError) as error:
            raise StreamExecutionError("device process exited") from error
        if kind != expected:
            raise StreamExecutionError(f"device process: {result}")
        return result

    def _spawn(self) -> tuple[Any, ...]:
        from ._process import worker

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=worker,
            args=(self._provider.factory, self._config, child),
            name="neurale-device",
            daemon=True,
        )
        self._stopping.clear()
        try:
            self._process.start()
        finally:
            child.close()
        return self._receive("description")

    def start(self) -> None:
        try:
            self.source.start()
        except RuntimeError as error:
            raise StreamExecutionError(str(error)) from error
        if self._process is not None:
            try:
                self._connection.send(("start", (self._path, self._capacity)))
                self._receive("started")
            except BaseException:
                self.source.ingress.fail()
                self._stop_process()
                raise
            process, ingress, stopping = self._process, self.source.ingress, self._stopping

            def monitor() -> None:
                from multiprocessing.connection import wait

                wait([process.sentinel])
                if not stopping.is_set() and not ingress.cancelled:
                    ingress.fail()

            self._monitor = threading.Thread(
                target=monitor, name="neurale-device-monitor", daemon=True
            )
            self._monitor.start()

    def cancel(self) -> None:
        if not self._closed:
            self.source.cancel()

    def _stop_process(self) -> StreamExecutionError | StreamTimeoutError | None:
        if self._process is None:
            return None
        self._stopping.set()
        if self._process.pid is None:
            self._connection.close()
            self._process.close()
            self._process = self._connection = None
            return None
        try:
            self._connection.send(("close", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._process.join(self._timeout)
        forced = self._process.is_alive()
        if forced:
            self._process.terminate()
            self._process.join(self._timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        if self._monitor is not None:
            self._monitor.join()
            self._monitor = None
        exitcode = self._process.exitcode
        shutdown_message = None
        try:
            while self._connection.poll():
                kind, message = self._connection.recv()
                if kind == "shutdown_error":
                    shutdown_message = message
        except (EOFError, OSError):
            pass
        self._process.close()
        self._connection.close()
        self._process = self._connection = None
        error = None
        if forced:
            error = StreamTimeoutError(
                "device process exceeded shutdown timeout and was terminated"
            )
        elif shutdown_message is not None:
            error = StreamExecutionError(f"device shutdown failed: {shutdown_message}")
        elif exitcode != 0:
            error = StreamExecutionError(f"device process exited with code {exitcode}")
        if error is not None and self._source is not None:
            self._source.ingress.fail()
        return error

    def close(self) -> None:
        if self._closed:
            return
        self.cancel()
        shutdown_error = self._stop_process()
        self._source.close()
        self._last_stats = self._source.ingress.stats
        self._closed = True
        # Native mapping ownership defers cleanup until the runner also releases it.
        self._source = None
        if shutdown_error is not None:
            raise shutdown_error

    def __enter__(self) -> Device:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
