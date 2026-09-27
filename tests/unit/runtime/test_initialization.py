#!/usr/bin/env python3

from __future__ import annotations

import logging

import pytest

from neurale.exceptions import ConfigurationError, DeviceUnavailableError
from neurale.runtime import (
    CudaInfo,
    RuntimeConfig,
    get_runtime_config,
    initialize_runtime,
    reset_logging,
    reset_runtime,
)
from neurale.runtime import _device as device_module
from neurale.runtime import initialization as initialization_module


def teardown_function() -> None:
    reset_runtime(reload_environment=False)
    reset_logging()


def test_initialize_runtime_applies_logging_and_random_seed() -> None:
    info = initialize_runtime(
        RuntimeConfig(log_level="INFO", random_seed=42),
        configure_threads=False,
        force=True,
    )

    assert info.config.random_seed == 42
    assert logging.getLogger("neurale").level == logging.INFO


@pytest.mark.parametrize("requested", ("auto", "cpu"))
def test_initialize_runtime_resolves_cpu_without_cuda_probe(
    monkeypatch,
    requested,
) -> None:
    def fail_probe():
        raise AssertionError("CUDA must not be probed")

    monkeypatch.setattr(device_module, "_probe_cuda", fail_probe)

    info = initialize_runtime(
        RuntimeConfig(device=requested),
        configure_logging=False,
        configure_threads=False,
        configure_random=False,
        force=True,
    )

    assert info.config.device == requested
    assert info.resolved_device == "cpu"


def test_initialize_runtime_requires_requested_cuda(monkeypatch) -> None:
    config = RuntimeConfig(device="cuda")
    monkeypatch.setattr(
        device_module,
        "_probe_cuda",
        lambda: CudaInfo(
            compiled=False,
            available=False,
            reason="test CUDA unavailable",
        ),
    )
    logging_calls = []
    monkeypatch.setattr(
        initialization_module,
        "configure_package_logging",
        lambda **kwargs: logging_calls.append(kwargs),
    )

    with pytest.raises(
        DeviceUnavailableError,
        match="test CUDA unavailable",
    ):
        initialize_runtime(
            config,
            configure_logging=True,
            configure_threads=False,
            configure_random=False,
            force=True,
        )

    assert logging_calls == []
    assert get_runtime_config() != config


def test_initialize_runtime_is_idempotent_per_configuration() -> None:
    config = RuntimeConfig(device="cpu")
    first = initialize_runtime(
        config,
        configure_logging=False,
        configure_threads=False,
        configure_random=False,
        force=True,
    )
    second = initialize_runtime(
        config,
        configure_logging=False,
        configure_threads=False,
        configure_random=False,
    )

    assert second.config == first.config
    assert second.resolved_device == "cpu"

    with pytest.raises(ConfigurationError, match="different configuration"):
        initialize_runtime(
            RuntimeConfig(device="auto"),
            configure_logging=False,
            configure_threads=False,
            configure_random=False,
        )
