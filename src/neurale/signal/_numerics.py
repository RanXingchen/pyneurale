#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Small numerical helpers shared by signal algorithms."""

from __future__ import annotations

from neurale._validation import validate_integer


def next_power_of_two(value: int) -> int:
    value = validate_integer(value, "value", minimum=0)
    return 1 if value <= 1 else 1 << (value - 1).bit_length()
