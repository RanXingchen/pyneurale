#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Coordinate explicit initialization of runtime services."""

from __future__ import annotations

import logging
from dataclasses import replace

from neurale.exceptions import ConfigurationError

from . import _state
from ._device import DeviceResolution, resolve_device
from ._logging import configure_logging as configure_package_logging
from .config import RuntimeConfig
from .context import configure_runtime, get_runtime_config, runtime_context
from .diagnostics import RuntimeInfo, runtime_info
from .random import configure_random_seed
from .threading import configure_threading


def initialize_runtime(
    config: RuntimeConfig | None = None,
    *,
    configure_logging: bool = True,
    configure_threads: bool = True,
    configure_random: bool = True,
    probe_native: bool = False,
    probe_cuda: bool = False,
    force: bool = False,
) -> RuntimeInfo:
    """Initialize configured runtime services explicitly.

    Parameters
    ----------
    config : neurale.runtime.config.RuntimeConfig or None, optional
        Configuration to initialize. The active configuration is used when omitted.
    configure_logging : bool, default=True
        Configure the package logger.
    configure_threads : bool, default=True
        Apply the configured CPU thread limit.
    configure_random : bool, default=True
        Apply the configured random seed.
    probe_native : bool, default=False
        Include native extension diagnostics.
    probe_cuda : bool, default=False
        Include CUDA diagnostics.
    force : bool, default=False
        Reinitialize even when runtime services were previously initialized.

    Returns
    -------
    neurale.runtime.diagnostics.RuntimeInfo
        Diagnostics collected after initialization.

    Raises
    ------
    TypeError
        If ``config`` has an incompatible type.
    neurale.exceptions.ConfigurationError
        If reinitialization conflicts with prior state or configuration is invalid.
    neurale.exceptions.DeviceUnavailableError
        If required CUDA capabilities are unavailable.
    """

    if config is not None and not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig or None.")
    selected = get_runtime_config() if config is None else config

    with _state.LOCK:
        if _state.INITIALIZED and not force:
            if selected != _state.INITIALIZED_CONFIG:
                raise ConfigurationError(
                    "runtime is already initialized with a different "
                    "configuration; pass force=True to reinitialize it."
                )
            return runtime_info(
                probe_native=probe_native,
                probe_cuda=probe_cuda,
            )

        resolution = resolve_device(selected)

        with runtime_context(selected):
            _apply_runtime_services(
                selected,
                configure_logging=configure_logging,
                configure_threads=configure_threads,
                configure_random=configure_random,
                force=force,
            )
            info = _collect_initialized_info(probe_native, probe_cuda, resolution)

        if config is not None:
            configure_runtime(selected)
        _state.INITIALIZED = True
        _state.INITIALIZED_CONFIG = selected
        _state.RESOLVED_DEVICE = resolution.resolved
        logging.getLogger("neurale.runtime").debug(
            "Runtime initialized requested_device=%s resolved_device=%s "
            "threads=%s deterministic=%s.",
            selected.device,
            resolution.resolved,
            selected.num_threads,
            selected.deterministic,
        )
        return info


def _apply_runtime_services(
    config: RuntimeConfig,
    *,
    configure_logging: bool,
    configure_threads: bool,
    configure_random: bool,
    force: bool,
) -> None:
    if configure_logging:
        configure_package_logging(level=config.log_level, force=force)
    if configure_threads and config.num_threads is not None:
        configure_threading(config.num_threads, strict=config.strict)
    if configure_random and config.random_seed is not None:
        configure_random_seed(
            config.random_seed,
            torch="auto",
            deterministic=config.deterministic,
        )


def _collect_initialized_info(
    probe_native: bool,
    probe_cuda: bool,
    resolution: DeviceResolution,
) -> RuntimeInfo:
    info = runtime_info(
        probe_native=probe_native or resolution.resolved == "cuda",
        probe_cuda=probe_cuda,
    )
    return replace(
        info,
        resolved_device=resolution.resolved,
        cuda=info.cuda if info.cuda is not None else resolution.cuda,
    )
