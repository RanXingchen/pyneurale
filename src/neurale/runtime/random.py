#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Configure random generators and deterministic execution options."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random as _random
from dataclasses import dataclass
from typing import Literal

from neurale._validation import validate_choice, validate_integer
from neurale.exceptions import ConfigurationError, DependencyError

TorchSeedMode = bool | Literal["auto"]


@dataclass(frozen=True, slots=True)
class RandomState:
    """Describe configured random generators and deterministic options.

    Parameters
    ----------
    seed : int
        Applied random seed.
    python_seeded : bool
        Whether Python's random generator was seeded.
    numpy_seeded : bool
        Whether NumPy's random generator was seeded.
    torch_seeded : bool
        Whether PyTorch random generators were seeded.
    deterministic : bool
        Whether deterministic PyTorch algorithms were enabled.
    """

    seed: int
    python_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    deterministic: bool


def _validate_seed(seed: int) -> int:
    return validate_integer(
        seed,
        "random seed",
        minimum=0,
        error_type=ConfigurationError,
    )


def _load_torch(mode: TorchSeedMode):
    validate_choice(
        mode,
        (True, False, "auto"),
        "torch",
        error_type=ConfigurationError,
    )
    if mode is False:
        return None

    try:
        available = importlib.util.find_spec("torch") is not None
    except (ImportError, AttributeError, ValueError):
        available = False

    if not available:
        if mode is True:
            raise DependencyError(
                "PyTorch random seeding was requested, but torch is not installed."
            )
        return None

    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "PyTorch was found but could not be imported for random seeding."
        ) from exc


def _configure_torch(torch_module, seed: int, deterministic: bool) -> None:
    # torch.manual_seed seeds CPU and CUDA generators for all devices.
    torch_module.manual_seed(seed)
    if not deterministic:
        return

    torch_module.use_deterministic_algorithms(True)
    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


def configure_random_seed(
    seed: int,
    *,
    torch: TorchSeedMode = "auto",
    deterministic: bool = False,
) -> RandomState:
    """Seed Python, NumPy, and optionally PyTorch generators.

    Parameters
    ----------
    seed : int
        Non-negative random seed.
    torch : {True, False, "auto"}, default="auto"
        Require, disable, or opportunistically configure PyTorch.
    deterministic : bool, default=False
        Enable deterministic PyTorch algorithms when PyTorch is configured.

    Returns
    -------
    neurale.runtime.random.RandomState
        Record of configured generators.

    Raises
    ------
    neurale.exceptions.ConfigurationError
        If the seed or PyTorch mode is invalid.
    neurale.exceptions.DependencyError
        If PyTorch is required but unavailable or cannot be imported.

    Notes
    -----
    ``PYTHONHASHSEED`` must be set before process startup. Deterministic mode sets
    ``CUBLAS_WORKSPACE_CONFIG`` before importing PyTorch.
    """

    selected_seed = _validate_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    _random.seed(selected_seed)

    # NumPy is required, but remains an explicit initialization side effect.
    import numpy as np

    np.random.seed(selected_seed)
    torch_module = _load_torch(torch)
    if torch_module is not None:
        _configure_torch(torch_module, selected_seed, deterministic)

    return RandomState(
        seed=selected_seed,
        python_seeded=True,
        numpy_seeded=True,
        torch_seeded=torch_module is not None,
        deterministic=deterministic,
    )
