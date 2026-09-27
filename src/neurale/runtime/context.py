#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Manage global and context-local runtime configuration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import _state
from .config import RuntimeConfig


def get_runtime_config() -> RuntimeConfig:
    """Return the runtime configuration active in the current context.

    Returns
    -------
    neurale.runtime.config.RuntimeConfig
        Context-local configuration when present, otherwise the process default.
    """

    local = _state.CONTEXT_CONFIG.get()
    if local is not None:
        return local
    with _state.LOCK:
        return _state.GLOBAL_CONFIG


def configure_runtime(
    config: RuntimeConfig | None = None,
    **overrides: Any,
) -> RuntimeConfig:
    """Set the process-wide default runtime configuration.

    Parameters
    ----------
    config : neurale.runtime.config.RuntimeConfig or None, optional
        Base configuration. The current process default is used when omitted.
    **overrides : Any
        Fields to replace on the base configuration.

    Returns
    -------
    neurale.runtime.config.RuntimeConfig
        New process-wide default configuration.

    Raises
    ------
    TypeError
        If ``config`` has an incompatible type.
    neurale.exceptions.ConfigurationError
        If an override is invalid.
    """

    if config is not None and not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig or None.")
    with _state.LOCK:
        base = _state.GLOBAL_CONFIG if config is None else config
        updated = base.with_overrides(**overrides)
        _state.GLOBAL_CONFIG = updated
    return updated


def reset_runtime(*, reload_environment: bool = True) -> RuntimeConfig:
    """Reset global configuration and initialization state.

    Parameters
    ----------
    reload_environment : bool, default=True
        Re-read ``NEURALE_*`` variables instead of using built-in defaults.

    Returns
    -------
    neurale.runtime.config.RuntimeConfig
        New process-wide configuration.

    Notes
    -----
    Context-local overrides in other threads or asynchronous tasks are unchanged.
    """

    config = RuntimeConfig.from_environment() if reload_environment else RuntimeConfig()
    with _state.LOCK:
        _state.RESOLVED_DEVICE = None
        _state.GLOBAL_CONFIG = config
        _state.INITIALIZED = False
        _state.INITIALIZED_CONFIG = None
    _state.CONTEXT_CONFIG.set(None)
    return config


@contextmanager
def runtime_context(
    config: RuntimeConfig | None = None,
    **overrides: Any,
) -> Iterator[RuntimeConfig]:
    """Temporarily override runtime configuration in the current context.

    Parameters
    ----------
    config : neurale.runtime.config.RuntimeConfig or None, optional
        Base configuration. The currently active configuration is used when omitted.
    **overrides : Any
        Fields to replace for the lifetime of the context.

    Yields
    ------
    neurale.runtime.config.RuntimeConfig
        Active temporary configuration.

    Raises
    ------
    TypeError
        If ``config`` has an incompatible type.
    neurale.exceptions.ConfigurationError
        If an override is invalid.
    """

    if config is not None and not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig or None.")
    base = get_runtime_config() if config is None else config
    selected = base.with_overrides(**overrides)
    token = _state.CONTEXT_CONFIG.set(selected)
    try:
        yield selected
    finally:
        _state.CONTEXT_CONFIG.reset(token)
