#!/usr/bin/env python3

from __future__ import annotations

import asyncio

from neurale.runtime import (
    RuntimeConfig,
    configure_runtime,
    get_runtime_config,
    reset_runtime,
    runtime_context,
)


def teardown_function() -> None:
    reset_runtime(reload_environment=False)


def test_global_and_nested_context_configuration() -> None:
    configure_runtime(RuntimeConfig(device="cpu", num_threads=4))

    with runtime_context(num_threads=1):
        assert get_runtime_config().device == "cpu"
        assert get_runtime_config().num_threads == 1
        with runtime_context(allow_cuda=False):
            assert not get_runtime_config().allow_cuda
        assert get_runtime_config().allow_cuda

    assert get_runtime_config().num_threads == 4


def test_runtime_context_restores_after_exception() -> None:
    configure_runtime(device="cpu")

    try:
        with runtime_context(device="auto"):
            raise RuntimeError("stop")
    except RuntimeError:
        pass

    assert get_runtime_config().device == "cpu"


def test_runtime_context_isolated_between_async_tasks() -> None:
    async def read_device(device: str) -> str:
        with runtime_context(device=device):
            await asyncio.sleep(0)
            return get_runtime_config().device

    async def run() -> tuple[str, str]:
        first, second = await asyncio.gather(
            read_device("cpu"),
            read_device("cuda"),
        )
        return first, second

    assert asyncio.run(run()) == ("cpu", "cuda")
