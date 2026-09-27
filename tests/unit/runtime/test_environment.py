#!/usr/bin/env python3

from __future__ import annotations

import pytest

from neurale.exceptions import ConfigurationError
from neurale.runtime._environment import environment_overrides


def test_environment_overrides_parses_supported_values() -> None:
    values = environment_overrides(
        {
            "NEURALE_DEVICE": "cuda",
            "NEURALE_CUDA_DEVICE": "2",
            "NEURALE_ALLOW_CUDA": "yes",
            "NEURALE_NUM_THREADS": "4",
            "NEURALE_RANDOM_SEED": "42",
            "NEURALE_DETERMINISTIC": "true",
            "NEURALE_LOG_LEVEL": "INFO",
            "NEURALE_STRICT": "on",
        }
    )

    assert values == {
        "device": "cuda",
        "cuda_device": 2,
        "allow_cuda": True,
        "num_threads": 4,
        "random_seed": 42,
        "deterministic": True,
        "log_level": "INFO",
        "strict": True,
    }


def test_legacy_require_cuda_environment_variable_is_ignored() -> None:
    assert environment_overrides({"NEURALE_REQUIRE_CUDA": "1"}) == {}


def test_environment_overrides_rejects_invalid_values() -> None:
    with pytest.raises(ConfigurationError, match="NEURALE_ALLOW_CUDA"):
        environment_overrides({"NEURALE_ALLOW_CUDA": "sometimes"})

    with pytest.raises(ConfigurationError, match="positive"):
        environment_overrides({"NEURALE_NUM_THREADS": "0"})
