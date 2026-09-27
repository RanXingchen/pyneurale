#!/usr/bin/env python3

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import fft as scipy_fft

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import multitaper_psd
from neurale.signal.spectral import multitaper as multitaper_module
from neurale.signal.windows import dpss


@pytest.mark.parametrize("weighting", ["unity", "eigen"])
def test_multitaper_linear_weighting_matches_definition(
    weighting: str,
) -> None:
    rng = np.random.default_rng(42)
    values = rng.standard_normal(48)
    tapers, ratios = dpss(48, 3.0, 4)
    individual = np.abs(scipy_fft.fft(tapers.T * values, n=64, axis=1)) ** 2
    weights = np.ones(4) if weighting == "unity" else ratios
    whole = np.sum(weights[:, None] * individual, axis=0) / 4
    expected = whole[:33].copy() / 200.0
    expected[1:-1] *= 2.0

    result, freqs = multitaper_psd(
        values,
        200.0,
        nw=3.0,
        n_tapers=4,
        nfft=64,
        weighting=weighting,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(freqs, np.arange(33) * 200.0 / 64)


def test_multitaper_adaptive_zero_signal_is_stable() -> None:
    result, freqs = multitaper_psd(
        np.zeros((32, 2)),
        100.0,
        nw=2.5,
        nfft=32,
    )

    np.testing.assert_array_equal(result, np.zeros((17, 2)))
    assert freqs.shape == (17,)


def test_multitaper_complex_input_is_two_sided_and_preserves_axis() -> None:
    rng = np.random.default_rng(8)
    values = rng.standard_normal((3, 40)) + 1j * rng.standard_normal((3, 40))

    result, freqs = multitaper_psd(
        values,
        80.0,
        nw=2.5,
        n_tapers=3,
        nfft=32,
        weighting="unity",
        axis=1,
    )

    assert result.shape == (3, 32)
    np.testing.assert_allclose(freqs, np.arange(32) * 2.5)
    with pytest.raises(ValidationError, match="real-valued"):
        multitaper_psd(
            values,
            80.0,
            nw=2.5,
            n_tapers=3,
            nfft=32,
            sides="one-sided",
            axis=1,
        )


def test_multitaper_signal_array_uses_uniform_sampling_metadata() -> None:
    source = SignalArray(
        data=np.arange(64.0).reshape(32, 2),
        fs=128.0,
        time=1.0 + np.arange(32) / 128.0,
        t0=1.0,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="source",
    )

    result, freqs = multitaper_psd(
        source,
        nw=2.5,
        nfft=32,
        weighting="unity",
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (17, 2)
    assert freqs[-1] == pytest.approx(64.0)
    with pytest.raises(ValidationError, match="inconsistent"):
        multitaper_psd(
            source,
            100.0,
            nw=2.5,
            weighting="unity",
        )


def test_multitaper_validates_estimator_parameters() -> None:
    values = np.ones(32)
    with pytest.raises(ValidationError, match="at least one sample"):
        multitaper_psd(np.empty(0), nw=2.5)
    with pytest.raises(ValidationError, match="finite"):
        multitaper_psd(
            np.array([1.0, np.nan] * 16),
            nw=0.75,
            weighting="unity",
        )
    with pytest.raises(ValidationError, match="at least two"):
        multitaper_psd(
            values,
            nw=1.0,
            n_tapers=1,
            weighting="adaptive",
        )
    with pytest.raises(ValidationError, match="must not exceed"):
        multitaper_psd(
            values,
            nw=2.5,
            n_tapers=5,
            weighting="unity",
        )
    with pytest.raises(ValidationError, match="weighting"):
        multitaper_psd(
            values,
            nw=2.5,
            weighting="median",  # type: ignore[arg-type]
        )


def test_multitaper_psd_does_not_expose_backend() -> None:
    assert "backend" not in inspect.signature(multitaper_psd).parameters


def test_multitaper_high_level_calls_do_not_share_processor_state() -> None:
    rng = np.random.default_rng(12)
    values = rng.standard_normal((96, 3))
    barrier = threading.Barrier(4)
    expected, expected_freqs = multitaper_psd(
        values,
        500.0,
        nw=3.0,
        n_tapers=5,
        nfft=128,
        weighting="adaptive",
    )

    def run_once() -> tuple[np.ndarray, np.ndarray]:
        barrier.wait()
        return multitaper_psd(
            values,
            500.0,
            nw=3.0,
            n_tapers=5,
            nfft=128,
            weighting="adaptive",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: run_once(), range(4)))

    for psd, freqs in results:
        np.testing.assert_allclose(freqs, expected_freqs)
        np.testing.assert_allclose(psd, expected, rtol=1e-12, atol=1e-12)


def test_multitaper_native_backend_error_when_extension_is_absent(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(multitaper_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        multitaper_psd(
            np.ones(32),
            nw=2.5,
        )
