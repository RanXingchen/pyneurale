#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Array preparation helpers for native signal kernels."""

from __future__ import annotations

import numpy as np


def native_array(value, dtype):
    arr = np.asarray(value)
    target = np.dtype(dtype)
    if arr.dtype == target and arr.flags.c_contiguous:
        return arr
    return np.asarray(arr, dtype=target, order="C")
