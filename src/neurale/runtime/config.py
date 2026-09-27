#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Define and validate immutable runtime configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any

from neurale._validation import validate_choice, validate_integer
from neurale.exceptions import ConfigurationError

from ._environment import environment_overrides
from ._types import DevicePreference

_DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Describe process and context-level execution preferences.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda"}, default="auto"
        Requested execution device. "auto" resolves to CPU.
    cuda_device : int or None, optional
        Non-negative CUDA device ordinal.
    allow_cuda : bool, default=True
        Whether CUDA execution may be selected.
    num_threads : int or None, optional
        Positive CPU thread limit.
    random_seed : int or None, optional
        Non-negative random seed.
    deterministic : bool, default=False
        Whether deterministic algorithms are required.
    log_level : int or str, default="WARNING"
        Python logging level.
    strict : bool, default=False
        Whether recoverable runtime configuration issues should raise errors.

    Raises
    ------
    neurale.exceptions.ConfigurationError
        If a field value or cross-field constraint is invalid.
    """

    device: DevicePreference = "auto"
    cuda_device: int | None = None
    allow_cuda: bool = True
    num_threads: int | None = None
    random_seed: int | None = None
    deterministic: bool = False
    log_level: int | str = "WARNING"
    strict: bool = False

    def __post_init__(self) -> None:
        validate_choice(
            self.device,
            _DEVICES,
            "device",
            error_type=ConfigurationError,
        )
        if self.cuda_device is not None:
            validate_integer(
                self.cuda_device,
                "cuda_device",
                minimum=0,
                error_type=ConfigurationError,
            )
        if self.num_threads is not None:
            validate_integer(
                self.num_threads,
                "num_threads",
                minimum=1,
                error_type=ConfigurationError,
            )
        if self.random_seed is not None:
            validate_integer(
                self.random_seed,
                "random_seed",
                minimum=0,
                error_type=ConfigurationError,
            )
        if self.device == "cuda" and not self.allow_cuda:
            raise ConfigurationError("device='cuda' conflicts with allow_cuda=False.")
        if self.deterministic and self.random_seed is None:
            raise ConfigurationError("deterministic=True requires random_seed to be configured.")
        try:
            logging._checkLevel(self.log_level)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"log_level is not a valid logging level: {self.log_level!r}."
            ) from exc

    def with_overrides(self, **overrides: Any) -> RuntimeConfig:
        """Return a validated copy with selected fields replaced.

        Parameters
        ----------
        **overrides : Any
            Runtime configuration fields to replace.

        Returns
        -------
        neurale.runtime.config.RuntimeConfig
            New validated configuration.

        Raises
        ------
        neurale.exceptions.ConfigurationError
            If an override name or value is invalid.
        """

        valid_fields = {item.name for item in fields(self)}
        unknown = set(overrides) - valid_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ConfigurationError(f"unknown runtime configuration fields: {names}.")
        return replace(self, **overrides)

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        """Construct configuration from ``NEURALE_*`` environment variables.

        Returns
        -------
        neurale.runtime.config.RuntimeConfig
            Validated environment-derived configuration.

        Raises
        ------
        neurale.exceptions.ConfigurationError
            If an environment value is invalid.
        """

        return cls(**environment_overrides())
