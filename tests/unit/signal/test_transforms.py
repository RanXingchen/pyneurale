#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import transforms as transforms_module
from neurale.signal.transforms import fft_integrate


@pytest.mark.parametrize("n_samples", [127, 128])
def test_fft_integrate_even_and_odd_periodic_signals(
    n_samples: int,
) -> None:
    fs = float(n_samples)
    freq = 3.0
    time = np.arange(n_samples) / fs
    values = np.cos(2.0 * np.pi * freq * time)
    expected = np.sin(2.0 * np.pi * freq * time) / (2.0 * np.pi * freq)

    result = fft_integrate(values, 1.0 / fs)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert np.isrealobj(result)


def test_fft_integrate_multichannel_axis_and_dc_handling() -> None:
    n_samples = 64
    time = np.arange(n_samples) / n_samples
    values = np.vstack(
        [
            np.cos(2.0 * np.pi * 2.0 * time) + 5.0,
            np.cos(2.0 * np.pi * 5.0 * time) - 3.0,
        ]
    )

    result = fft_integrate(values, 1.0 / n_samples, axis=1)

    expected = np.vstack(
        [
            np.sin(2.0 * np.pi * 2.0 * time) / (4.0 * np.pi),
            np.sin(2.0 * np.pi * 5.0 * time) / (10.0 * np.pi),
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(np.mean(result, axis=1), 0.0, atol=1e-15)


def test_fft_integrate_supports_repeated_integration() -> None:
    n_samples = 64
    time = np.arange(n_samples) / n_samples
    values = np.cos(2.0 * np.pi * 3.0 * time)

    result = fft_integrate(
        values,
        1.0 / n_samples,
        order=2,
    )
    expected = -values / (2.0 * np.pi * 3.0) ** 2

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_fft_integrate_preserves_complex_results() -> None:
    rng = np.random.default_rng(44)
    values = rng.standard_normal(31) + 1j * rng.standard_normal(31)

    result = fft_integrate(values, dt=0.02)

    assert np.iscomplexobj(result)
    assert result.shape == values.shape


def test_fft_integrate_uses_signal_array_fs() -> None:
    n_samples = 32
    source = SignalArray(
        data=np.cos(2.0 * np.pi * 2.0 * np.arange(n_samples) / n_samples)[:, None],
        fs=float(n_samples),
        time=None,
        t0=4.0,
        clock=None,
        channels=ChannelTable.default(1, unit="uV"),
        unit="uV",
        name="neural",
    )

    result = fft_integrate(source)

    assert isinstance(result, SignalArray)
    assert result.fs == source.fs
    assert result.t0 == source.t0
    assert result.channel_names == source.channel_names
    assert result.unit == "uV*s"
    assert result.channels.units == ["uV*s"]
    np.testing.assert_array_equal(
        source.data,
        np.cos(2.0 * np.pi * 2.0 * np.arange(n_samples) / n_samples)[:, None],
    )


def test_fft_integrate_infers_uniform_explicit_time() -> None:
    n_samples = 32
    time = np.arange(n_samples) / n_samples
    source = SignalArray(
        data=np.cos(2.0 * np.pi * 2.0 * time)[:, None],
        fs=float(n_samples),
        time=time,
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1, unit="mV"),
        unit="mV",
        name="explicit-time",
    )
    result = fft_integrate(source)
    assert isinstance(result, SignalArray)
    assert result.unit == "mV*s"
    assert result.channels.units == ["mV*s"]
    assert result.fs == pytest.approx(float(n_samples))


def test_fft_integrate_transforms_per_channel_units() -> None:
    source = SignalArray(
        data=np.ones((8, 2)),
        fs=8.0,
        time=None,
        t0=0.0,
        clock=None,
        channels=ChannelTable.from_names(["ch000", "ch001"], unit=["uV", "mV"]),
        unit=["uV", "mV"],
        name="mixed-units",
    )
    result = fft_integrate(source)
    assert isinstance(result, SignalArray)
    assert result.unit == ("uV*s", "mV*s")
    assert result.channels.units == ["uV*s", "mV*s"]


def test_repeated_integration_updates_units_for_each_order() -> None:
    source = SignalArray(
        data=np.ones((8, 1)),
        fs=8.0,
        time=None,
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1, unit="uV"),
        unit="uV",
        name="repeated",
    )

    result = fft_integrate(source, order=2)

    assert isinstance(result, SignalArray)
    assert result.unit == "(uV*s)*s"
    assert result.channels.units == ["(uV*s)*s"]


def test_fft_integrate_rejects_irregular_time() -> None:
    source = SignalArray(
        data=np.ones((4, 1)),
        fs=10.0,
        time=np.array([0.0, 0.1, 0.21, 0.3]),
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1),
        unit="a.u.",
        name="irregular",
    )
    with pytest.raises(ValidationError, match="uniformly sampled"):
        fft_integrate(source)


def test_fft_integrate_validates_time_step() -> None:
    with pytest.raises(ValidationError, match="dt is required"):
        fft_integrate(np.arange(8.0))
    with pytest.raises(ValidationError):
        fft_integrate(np.arange(8.0), 0.0)

    source = SignalArray(
        data=np.ones((8, 1)),
        fs=8.0,
        time=None,
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1),
        unit="a.u.",
        name="test",
    )
    with pytest.raises(ValidationError, match="inconsistent"):
        fft_integrate(source, dt=0.5)
    with pytest.raises(ValidationError, match="order"):
        fft_integrate(np.ones(8), dt=1.0, order=0)


def test_fft_integrate_native_backend_error(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(transforms_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        fft_integrate(np.ones(8), dt=1.0)
