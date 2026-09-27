#!/usr/bin/env python3

from __future__ import annotations

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

from neurale.exceptions import ConfigurationError, DependencyError
from neurale.runtime import configure_random_seed
from neurale.runtime import random as random_module


def test_configure_random_seed_seeds_python_and_numpy() -> None:
    state = configure_random_seed(42, torch=False)
    python_value = random.random()
    numpy_value = np.random.random()

    configure_random_seed(42, torch=False)

    assert random.random() == python_value
    assert np.random.random() == numpy_value
    assert state.python_seeded
    assert state.numpy_seeded
    assert not state.torch_seeded


def test_configure_random_seed_configures_optional_torch(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("seed", seed)),
        use_deterministic_algorithms=lambda enabled: calls.append(("deterministic", enabled)),
        backends=SimpleNamespace(cudnn=cudnn),
    )
    monkeypatch.setattr(
        random_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else None,
    )
    monkeypatch.setattr(
        random_module.importlib,
        "import_module",
        lambda name: fake_torch,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    state = configure_random_seed(7, torch="auto", deterministic=True)

    assert calls == [("seed", 7), ("deterministic", True)]
    assert cudnn.deterministic
    assert not cudnn.benchmark
    assert state.torch_seeded
    assert state.deterministic
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert "PYTHONHASHSEED" not in os.environ


def test_configure_random_seed_requires_torch_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(random_module.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(DependencyError, match="torch is not installed"):
        configure_random_seed(1, torch=True)


def test_configure_random_seed_validates_arguments() -> None:
    with pytest.raises(ConfigurationError, match="non-negative"):
        configure_random_seed(-1, torch=False)

    with pytest.raises(ConfigurationError, match="True, False"):
        configure_random_seed(1, torch="required")  # type: ignore[arg-type]
