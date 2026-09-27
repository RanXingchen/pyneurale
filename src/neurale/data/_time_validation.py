#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared private time-grid and clock-domain validation helpers."""

from __future__ import annotations

import numpy as np

from neurale.exceptions import ValidationError

from .time import Clock


def _require_unified_sync_domain(*clocks: Clock | None) -> None:
    """Require all present clocks to identify one synchronization domain."""
    present = [clock for clock in clocks if clock is not None]
    if len(present) < 2:
        return
    domain = present[0].synchronization_domain
    for clock in present[1:]:
        other = clock.synchronization_domain
        if not isinstance(domain, str) or not isinstance(other, str):
            raise ValidationError("clock conversion requires explicit sync_domain.")
        if domain != other:
            raise ValidationError("clock conversion requires matching sync_domain.")


def sampled_times_are_contiguous(
    start: float,
    expected: float,
    fs: float,
) -> bool:
    tol = max(
        1e-12,
        abs(1.0 / fs) * 1e-7,
        np.finfo(float).eps * max(1.0, abs(start), abs(expected)) * 16.0,
    )
    return bool(np.isclose(start, expected, rtol=0.0, atol=tol))


def sampled_times_match_rate(
    time: np.ndarray,
    fs: float,
) -> bool:
    if time.size < 2:
        return True
    interval = 1.0 / fs
    tol = max(
        1e-12,
        np.finfo(float).eps * max(1.0, abs(interval)) * 16.0,
    )
    return bool(
        np.allclose(
            np.diff(time),
            interval,
            rtol=1e-7,
            atol=tol,
        )
    )
