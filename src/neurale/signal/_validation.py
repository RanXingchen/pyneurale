#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Signal-specific validation helpers."""

from __future__ import annotations

import numpy as np

from neurale._validation import (
    validate_numeric_array,
    validate_positive_float,
)
from neurale.exceptions import ValidationError


def validate_signal_matrix(x: np.ndarray, name: str = "data") -> np.ndarray:
    return validate_numeric_array(x, name, ndim=2)


def validate_fs(
    fs: float | None,
    name: str = "fs",
    *,
    optional: bool = False,
) -> float | None:
    if fs is None:
        if optional:
            return None
        raise ValidationError(f"{name} is required.")
    return validate_positive_float(fs, name)
