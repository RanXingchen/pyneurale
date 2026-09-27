#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Private validation helpers shared across the sorting domain.

These helpers are intentionally private (``_``-prefixed) and live inside the
sorting package rather than in a cross-domain ``utils``/``common`` module.
They are the small, exact-duplicate validation routines that several sorting
submodules (``offline``, ``_detection_reference``, ``curation``, ``metrics``) each
re-implemented; collecting them here keeps one source of truth for the shared
error contracts. Helpers used by a single submodule stay in that submodule.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from neurale.data import SignalArray
from neurale.exceptions import ValidationError

from .detection import DetectionConfig


def _resolved_groups(
    config: DetectionConfig,
    n_channels: int,
) -> tuple[tuple[int, ...], ...]:
    # Resolve an electrode-group partition against a channel count. ``None``
    # means one independent group per channel. Supplied groups must cover every
    # channel exactly once and stay in bounds; the returned partition preserves
    # the caller's group order, which defines the stable integer group id.
    if config.electrode_groups is None:
        return tuple((channel,) for channel in range(n_channels))
    flattened = tuple(channel for group in config.electrode_groups for channel in group)
    if any(channel >= n_channels for channel in flattened):
        raise ValidationError("electrode_groups contains a channel position outside the signal.")
    if set(flattened) != set(range(n_channels)):
        raise ValidationError("electrode_groups must cover every signal channel exactly once.")
    return tuple(tuple(group) for group in config.electrode_groups)


def _validate_regular_time(signal: SignalArray) -> None:
    # Threshold detection needs a regularly sampled grid. The tolerance is the
    # looser of a quarter sample and a scale-aware floating-point bound, so a
    # well-formed clock neither over-tolerates drift nor rejects honest binary64.
    if signal.n_samples < 2:
        return
    actual = (np.asarray(signal.time) - float(signal.time[0])) * signal.fs
    expected = np.arange(signal.n_samples, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected))))
    tol = min(0.25, max(1e-10, np.finfo(float).eps * scale * 16.0))
    if not np.allclose(actual, expected, rtol=0.0, atol=tol):
        raise ValidationError("threshold detection requires a regularly sampled SignalArray.")


def _sample_index_offset(value: int, n_samples: int) -> int:
    # Validate a non-negative absolute index for ``signal.data[0]`` that must
    # combine with the signal length to stay inside int64.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError("sample_index_offset must be a non-negative integer.")
    offset = int(value)
    maximum = np.iinfo(np.int64).max
    if offset < 0 or offset > maximum:
        raise ValidationError("sample_index_offset must fit non-negative int64.")
    if n_samples and offset + n_samples - 1 > maximum:
        raise ValidationError("sample_index_offset plus signal length must fit int64.")
    return offset


def _int64_scalar(value: object, name: str) -> int:
    # Coerce a scalar label/index to ``int`` with explicit int64 bounds. ``bool``
    # is rejected so a stray ``True``/``False`` is never silently relabeled ``1``/``0``.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer.")
    integer = int(value)
    if integer < np.iinfo(np.int64).min or integer > np.iinfo(np.int64).max:
        raise ValidationError(f"{name} must fit in int64.")
    return integer


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be finite and non-negative.")
    return result
