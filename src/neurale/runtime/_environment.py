#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Parse environment variables that override runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from neurale._validation import validate_choice, validate_integer
from neurale.exceptions import ConfigurationError

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

ENVIRONMENT_FIELDS = {
    "device": "NEURALE_DEVICE",
    "cuda_device": "NEURALE_CUDA_DEVICE",
    "allow_cuda": "NEURALE_ALLOW_CUDA",
    "num_threads": "NEURALE_NUM_THREADS",
    "random_seed": "NEURALE_RANDOM_SEED",
    "deterministic": "NEURALE_DETERMINISTIC",
    "log_level": "NEURALE_LOG_LEVEL",
    "strict": "NEURALE_STRICT",
}


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    validate_choice(
        normalized,
        _TRUE_VALUES | _FALSE_VALUES,
        name,
        error_type=ConfigurationError,
    )
    return normalized in _TRUE_VALUES


def _parse_int(value: str, name: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; got {value!r}.") from exc
    return validate_integer(
        parsed,
        name,
        minimum=minimum,
        error_type=ConfigurationError,
    )


def environment_overrides(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}

    for field, environment_name in ENVIRONMENT_FIELDS.items():
        if environment_name not in values:
            continue
        raw = values[environment_name]
        if field in {"allow_cuda", "deterministic", "strict"}:
            overrides[field] = _parse_bool(raw, environment_name)
        elif field == "cuda_device":
            overrides[field] = _parse_int(raw, environment_name, minimum=0)
        elif field == "num_threads":
            overrides[field] = _parse_int(raw, environment_name, minimum=1)
        elif field == "random_seed":
            overrides[field] = _parse_int(raw, environment_name, minimum=0)
        else:
            overrides[field] = raw.strip()

    return overrides
