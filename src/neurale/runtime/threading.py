#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Configure CPU math-library and OpenMP thread limits."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock

from neurale._native_loader import load_native_extension
from neurale._validation import validate_integer
from neurale.exceptions import ConfigurationError, NativeUnavailableError

from ._native_probe import probe_native_threading
from .context import get_runtime_config

_THREAD_ENVIRONMENT = (
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
)
_THREAD_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class ThreadingInfo:
    """Describe configured environment and native thread limits.

    Parameters
    ----------
    requested_threads : int or None
        Thread limit requested by runtime configuration.
    backend : str or None
        Native threading backend name.
    native_threads : int or None
        Thread count reported by the native backend.
    environment : tuple of tuple
        Thread-related environment variable names and values.
    """

    requested_threads: int | None
    backend: str | None
    native_threads: int | None
    environment: tuple[tuple[str, str | None], ...]


def _validate_num_threads(num_threads: int) -> int:
    return validate_integer(
        num_threads,
        "num_threads",
        minimum=1,
        error_type=ConfigurationError,
    )


def _native_thread_setter(native: object):
    namespace = getattr(native, "threading", None)
    setter = getattr(namespace, "set_num_threads", None)
    if callable(setter):
        return setter
    setter = getattr(native, "set_num_threads", None)
    return setter if callable(setter) else None


def _native_thread_count(native: object) -> int | None:
    info = probe_native_threading(native)  # type: ignore[arg-type]
    if info is None:
        return None
    value = info.get("num_threads")
    return int(value) if value is not None else None


def _optional_native() -> object | None:
    try:
        return load_native_extension()
    except NativeUnavailableError:
        return None


def threading_info(*, probe_native: bool = False) -> ThreadingInfo:
    """Return current thread-related environment and native state.

    Parameters
    ----------
    probe_native : bool, default=False
        Load and query the optional native extension.

    Returns
    -------
    neurale.runtime.threading.ThreadingInfo
        Current requested, environment, and native thread settings.
    """

    native = _optional_native() if probe_native else None
    native_info = probe_native_threading(native) if native is not None else None
    backend = None if native_info is None else native_info.get("backend")
    native_threads = None
    if native_info is not None:
        count = native_info.get("num_threads")
        native_threads = int(count) if count is not None else None
    return ThreadingInfo(
        requested_threads=get_runtime_config().num_threads,
        backend=None if backend is None else str(backend),
        native_threads=native_threads,
        environment=tuple((name, os.environ.get(name)) for name in _THREAD_ENVIRONMENT),
    )


def configure_threading(
    num_threads: int,
    *,
    strict: bool | None = None,
) -> ThreadingInfo:
    """Apply CPU thread limits to environment and native backends.

    Parameters
    ----------
    num_threads : int
        Positive thread limit.
    strict : bool or None, optional
        Raise instead of warning when NumPy is already loaded. Defaults to runtime
        configuration.

    Returns
    -------
    neurale.runtime.threading.ThreadingInfo
        Thread settings after configuration.

    Raises
    ------
    neurale.exceptions.ConfigurationError
        If the thread count is invalid or strict reconfiguration is unsafe.
    """

    selected = _validate_num_threads(num_threads)
    use_strict = get_runtime_config().strict if strict is None else strict

    with _THREAD_LOCK:
        if "numpy" in sys.modules:
            message = (
                "NumPy is already loaded; MKL_NUM_THREADS and OMP_NUM_THREADS may "
                "not reconfigure its existing thread pools."
            )
            if use_strict:
                raise ConfigurationError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        for name in _THREAD_ENVIRONMENT:
            os.environ[name] = str(selected)
        os.environ["MKL_DYNAMIC"] = "FALSE"

        native = _optional_native()
        if native is not None:
            setter = _native_thread_setter(native)
            if setter is not None:
                setter(selected)

    info = threading_info(probe_native=native is not None)
    logging.getLogger("neurale.runtime.threading").debug(
        "Configured CPU thread limit=%s native_backend=%s.",
        selected,
        info.backend,
    )
    return info


@contextmanager
def thread_limit(
    num_threads: int,
    *,
    strict: bool | None = None,
) -> Iterator[ThreadingInfo]:
    """Temporarily apply a process-wide CPU thread limit.

    Parameters
    ----------
    num_threads : int
        Positive temporary thread limit.
    strict : bool or None, optional
        Strict reconfiguration policy.

    Yields
    ------
    neurale.runtime.threading.ThreadingInfo
        Thread settings active inside the context.

    Notes
    -----
    Environment variables are process-wide. PyNeurale serializes its own changes,
    but external libraries may still alter their thread settings.
    """

    selected = _validate_num_threads(num_threads)
    with _THREAD_LOCK:
        previous_environment = {
            name: os.environ.get(name) for name in (*_THREAD_ENVIRONMENT, "MKL_DYNAMIC")
        }
        native = _optional_native()
        previous_native = _native_thread_count(native) if native is not None else None
        info = configure_threading(selected, strict=strict)
        try:
            yield info
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            if native is not None and previous_native is not None:
                setter = _native_thread_setter(native)
                if setter is not None:
                    setter(previous_native)
