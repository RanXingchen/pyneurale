#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Window functions for spectral analysis and FIR design."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_choice,
    validate_integer,
    validate_number,
)
from neurale.exceptions import ValidationError

WindowKind = Literal["hann", "hamming", "blackman", "flattop"]

_WINDOW_KINDS: tuple[WindowKind, ...] = (
    "hann",
    "hamming",
    "blackman",
    "flattop",
)


def dpss(
    length: int,
    nw: float,
    n_tapers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DPSS tapers and their spectral concentration ratios.

    Repeated calls cache the immutable native result and return independent
    writable copies to prevent cache corruption.

    Parameters
    ----------
    length : int
        Positive taper length.
    nw : float
        Time-half-bandwidth product, strictly between zero and ``length / 2``.
    n_tapers : int or None, optional
        Number of tapers. The default is ``floor(2*nw - 1)``.
    Returns
    -------
    tapers : numpy.ndarray
        L2-normalized tapers with shape ``(length, n_tapers)``.
    concentration_ratios : numpy.ndarray
        Spectral concentration ratio for each taper.
    """
    length = validate_integer(length, "length", minimum=2)
    nw = _validate_nw(length, nw)
    default_tapers = min(length, math.floor(2.0 * nw - 1.0))
    if n_tapers is None:
        if default_tapers < 1:
            raise ValidationError(
                "nw must allow at least one default taper (floor(2*nw - 1) >= 1)."
            )
        n_tapers = default_tapers
    else:
        n_tapers = validate_integer(n_tapers, "n_tapers", minimum=1)
        if n_tapers > length:
            raise ValidationError("n_tapers must not exceed length.")

    tapers, ratios = _cached_dpss(length, nw, n_tapers)
    return tapers.copy(), ratios.copy()


@lru_cache(maxsize=32)
def _cached_dpss(
    length: int,
    nw: float,
    n_tapers: int,
) -> tuple[np.ndarray, np.ndarray]:
    native = load_native_namespace("signal.windows")
    tapers, ratios = native.dpss(length, nw, n_tapers)
    tapers = np.array(tapers, dtype=float, order="C", copy=True)
    ratios = np.array(ratios, dtype=float, order="C", copy=True)
    tapers.setflags(write=False)
    ratios.setflags(write=False)
    return tapers, ratios


def _validate_nw(length: int, value: float) -> float:
    result = float(
        validate_number(
            value,
            "nw",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            finite=True,
            coerce=True,
        )
    )
    if result >= length / 2.0:
        raise ValidationError("nw must be strictly less than length / 2.")
    return result


def cosine_window(
    kind: WindowKind,
    length: int,
    *,
    symmetric: bool = True,
) -> np.ndarray:
    """Return a standard generalized-cosine window.

    Parameters
    ----------
    kind : {"hann", "hamming", "blackman", "flattop"}
        Window coefficient set.
    length : int
        Non-negative output length.
    symmetric : bool, default=True
        Generate a symmetric filter-design window; use ``False`` for a periodic
        spectral-analysis window.
    Returns
    -------
    numpy.ndarray
        Floating-point window.

    Raises
    ------
    neurale.exceptions.ValidationError
        If kind, length, or symmetry is invalid.
    """
    kind = validate_choice(kind, _WINDOW_KINDS, "kind")
    length = validate_integer(length, "length", minimum=0)
    symmetric = _validate_symmetric(symmetric)
    native = load_native_namespace("signal.windows")
    return native.cosine_window(kind, length, symmetric)


def kaiser_window(
    length: int,
    beta: float,
    *,
    symmetric: bool = True,
) -> np.ndarray:
    """Return a Kaiser window.

    Parameters
    ----------
    length : int
        Non-negative output length.
    beta : float
        Finite shape parameter.
    symmetric : bool, default=True
        Generate a symmetric filter-design window; use ``False`` for a periodic
        spectral-analysis window.
    Returns
    -------
    numpy.ndarray
        Floating-point Kaiser window.

    Raises
    ------
    neurale.exceptions.ValidationError
        If length, beta, or symmetry is invalid.
    """
    length = validate_integer(length, "length", minimum=0)
    beta = float(validate_number(beta, "beta", kind="real", coerce=True))
    symmetric = _validate_symmetric(symmetric)
    native = load_native_namespace("signal.windows")
    return native.kaiser_window(length, beta, symmetric)


def _validate_symmetric(symmetric: bool) -> bool:
    if not isinstance(symmetric, bool):
        raise ValidationError("symmetric must be a bool.")
    return symmetric
