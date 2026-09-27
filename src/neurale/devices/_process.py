# SPDX-License-Identifier: MIT
"""Spawn-only Python SDK host. No Python runs on the native consumer thread."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def worker(factory: str, config: dict[str, Any], connection: Any) -> None:
    from neurale._native_loader import load_native_namespace

    from .provider import _native_signals

    provider = None
    ingress = None
    try:
        module, name = factory.split(":", 1)
        provider = getattr(import_module(module), name)()
        provider.open(config)
        signals = tuple(provider.describe())
        native_signals = _native_signals(signals)
        connection.send(("description", signals))
        while True:
            if ingress is not None and ingress.cancelled:
                break
            if not connection.poll(0.01):
                continue
            command, value = connection.recv()
            if command == "close":
                break
            if command != "start" or ingress is not None:
                raise RuntimeError("invalid device process command")
            path, capacity = value
            ingress = load_native_namespace("devices").DeviceIngress(
                path, native_signals, capacity, False
            )
            provider.start(ingress)
            connection.send(("started", None))
    except BaseException as error:
        if ingress is not None:
            ingress.fail()
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        try:
            failures = []
            if provider is not None:
                for operation in ("cancel", "close"):
                    try:
                        getattr(provider, operation)()
                    except BaseException as error:
                        failures.append(f"{operation}: {type(error).__name__}: {error}"[:1024])
            if failures:
                message = "; ".join(failures)
                if ingress is not None:
                    ingress.fail()
                try:
                    connection.send(("shutdown_error", message))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                raise RuntimeError(message)
        finally:
            connection.close()
