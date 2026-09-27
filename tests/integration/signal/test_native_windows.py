#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.signal import windows as scipy_windows

from neurale.signal.windows import (
    _cached_dpss,
    cosine_window,
    dpss,
    kaiser_window,
)


def _native() -> object:
    return importlib.import_module("neurale._native").signal


@pytest.mark.parametrize(
    ("length", "nw", "n_tapers"),
    [(16, 2.5, 4), (64, 3.5, 5), (256, 3.0, 5), (1024, 4.0, 7)],
)
def test_native_dpss_matches_scipy(
    length: int,
    nw: float,
    n_tapers: int,
) -> None:
    _native()
    _cached_dpss.cache_clear()

    tapers, ratios = dpss(length, nw, n_tapers)
    expected_tapers, expected_ratios = scipy_windows.dpss(
        length,
        nw,
        Kmax=n_tapers,
        sym=True,
        norm=2,
        return_ratios=True,
    )

    np.testing.assert_allclose(tapers, expected_tapers.T, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(ratios, expected_ratios, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("kind", "length", "symmetric", "reference"),
    [
        ("hann", 0, True, scipy_windows.hann),
        ("hann", 1, False, scipy_windows.hann),
        ("hamming", 2, True, scipy_windows.hamming),
        ("blackman", 17, False, scipy_windows.blackman),
        ("flattop", 18, True, scipy_windows.flattop),
    ],
)
def test_native_cosine_window_matches_scipy(
    kind: str,
    length: int,
    symmetric: bool,
    reference,
) -> None:
    _native()

    result = cosine_window(kind, length, symmetric=symmetric)

    np.testing.assert_allclose(
        result,
        reference(length, sym=symmetric),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("length", "beta", "symmetric"),
    [(0, 0.0, True), (1, 5.0, False), (2, 5.0, True), (31, 14.0, False)],
)
def test_native_kaiser_matches_scipy(
    length: int,
    beta: float,
    symmetric: bool,
) -> None:
    _native()

    result = kaiser_window(length, beta, symmetric=symmetric)
    reference = scipy_windows.kaiser(length, beta, sym=symmetric)

    np.testing.assert_allclose(result, reference, atol=1e-14)
