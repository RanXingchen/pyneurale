#!/usr/bin/env python3

from __future__ import annotations

import inspect

import pytest

from neurale.exceptions import ConfigurationError
from neurale.runtime import RuntimeConfig


def test_runtime_config_defaults_are_safe() -> None:
    config = RuntimeConfig()

    assert config.device == "auto"
    assert config.allow_cuda
    assert "require_cuda" not in inspect.signature(RuntimeConfig).parameters
    assert config.num_threads is None
    assert not config.deterministic


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"device": "gpu"}, "device"),
        ({"cuda_device": -1}, "cuda_device"),
        ({"num_threads": 0}, "num_threads"),
        ({"random_seed": -1}, "random_seed"),
        ({"device": "cuda", "allow_cuda": False}, "conflicts"),
        ({"deterministic": True}, "requires random_seed"),
    ],
)
def test_runtime_config_rejects_inconsistent_values(values, message) -> None:
    with pytest.raises(ConfigurationError, match=message):
        RuntimeConfig(**values)


def test_legacy_require_cuda_configuration_cannot_be_constructed() -> None:
    with pytest.raises(TypeError, match="require_cuda"):
        RuntimeConfig(device="cpu", require_cuda=True)


def test_runtime_config_with_overrides_is_immutable() -> None:
    original = RuntimeConfig()
    updated = original.with_overrides(device="cpu", num_threads=1)

    assert original.device == "auto"
    assert updated.device == "cpu"
    assert updated.num_threads == 1

    with pytest.raises(ConfigurationError, match="unknown"):
        original.with_overrides(implementation="native")
