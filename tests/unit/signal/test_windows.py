#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import windows as scipy_windows

from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import windows as windows_module
from neurale.signal.windows import (
    _cached_dpss,
    cosine_window,
    dpss,
    kaiser_window,
)


def _native_unavailable(*args, **kwargs):
    raise NativeUnavailableError("native unavailable")


def test_dpss_matches_scipy_reference_and_sample_major_layout() -> None:
    tapers, ratios = dpss(64, 3.5, 5)
    expected_tapers, expected_ratios = scipy_windows.dpss(
        64,
        3.5,
        Kmax=5,
        sym=True,
        norm=2,
        return_ratios=True,
    )

    assert tapers.shape == (64, 5)
    np.testing.assert_allclose(tapers, expected_tapers.T, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(ratios, expected_ratios, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tapers.T @ tapers, np.eye(5), atol=1e-10)


def test_dpss_uses_two_nw_minus_one_default() -> None:
    tapers, ratios = dpss(128, 4.0)

    assert tapers.shape == (128, 7)
    assert ratios.shape == (7,)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((1, 0.25, None), "length"),
        ((8, 4.0, None), "strictly less"),
        ((16, 0.5, None), "at least one"),
        ((32, 2.5, 33), "must not exceed"),
    ],
)
def test_dpss_validates_parameters(args, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        dpss(*args)


def test_dpss_allows_explicit_tapers_beyond_pmtm_default() -> None:
    tapers, ratios = dpss(32, 2.5, 5)

    assert tapers.shape == (32, 5)
    assert ratios.shape == (5,)


def test_dpss_cache_reuses_computation_without_sharing(
    monkeypatch,
) -> None:
    _cached_dpss.cache_clear()
    calls = 0

    def counted(length, nw, n_tapers):
        nonlocal calls
        calls += 1
        scipy_tapers, ratios = scipy_windows.dpss(
            length,
            nw,
            Kmax=n_tapers,
            sym=True,
            norm=2,
            return_ratios=True,
        )
        return np.asarray(scipy_tapers).T, ratios

    native = SimpleNamespace(dpss=counted)
    monkeypatch.setattr(windows_module, "load_native_namespace", lambda namespace: native)
    first_tapers, first_ratios = dpss(64, 3.0, 5)
    second_tapers, second_ratios = dpss(64, 3.0, 5)

    assert calls == 1
    first_tapers[0, 0] = 100.0
    first_ratios[0] = 0.0
    assert second_tapers[0, 0] != 100.0
    assert second_ratios[0] != 0.0


def test_dpss_requires_native_extension(monkeypatch) -> None:
    _cached_dpss.cache_clear()
    monkeypatch.setattr(windows_module, "load_native_namespace", _native_unavailable)
    with pytest.raises(NativeUnavailableError):
        dpss(32, 2.5)


@pytest.mark.parametrize(
    ("kind", "reference"),
    [
        ("hann", scipy_windows.hann),
        ("hamming", scipy_windows.hamming),
        ("blackman", scipy_windows.blackman),
        ("flattop", scipy_windows.flattop),
    ],
)
@pytest.mark.parametrize("length", [0, 1, 2, 7, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_cosine_windows_match_scipy(
    kind: str,
    reference,
    length: int,
    symmetric: bool,
) -> None:
    result = cosine_window(kind, length, symmetric=symmetric)

    np.testing.assert_allclose(
        result,
        reference(length, sym=symmetric),
        atol=1e-15,
    )
    assert result.dtype == np.float64
    assert result.shape == (length,)


@pytest.mark.parametrize("length", [0, 1, 2, 9, 10])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("beta", [0.0, 5.0, 14.0, -5.0])
def test_kaiser_window_matches_scipy(
    length: int,
    symmetric: bool,
    beta: float,
) -> None:
    result = kaiser_window(length, beta, symmetric=symmetric)

    np.testing.assert_allclose(
        result,
        scipy_windows.kaiser(length, beta, sym=symmetric),
        atol=1e-15,
    )


def test_window_validation() -> None:
    with pytest.raises(ValidationError, match="kind"):
        cosine_window("triangle", 8)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="non-negative"):
        cosine_window("hann", -1)
    with pytest.raises(ValidationError, match="symmetric"):
        cosine_window("hann", 8, symmetric=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="finite"):
        kaiser_window(8, np.inf)


@pytest.mark.parametrize("beta", [710.0, 1000.0])
def test_kaiser_large_beta_remains_finite(beta: float) -> None:
    native = kaiser_window(17, beta)
    assert np.all(np.isfinite(native))
    np.testing.assert_allclose(native, native[::-1], atol=0.0)
    assert native[8] == pytest.approx(1.0)


def test_window_functions_require_native_extension(monkeypatch) -> None:
    monkeypatch.setattr(windows_module, "load_native_namespace", _native_unavailable)

    with pytest.raises(NativeUnavailableError):
        cosine_window("hann", 8)
    with pytest.raises(NativeUnavailableError):
        kaiser_window(8, 1.0)
