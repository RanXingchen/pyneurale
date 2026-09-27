from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import analytic_signal, czt
from neurale.signal.spectral import transforms as spectral_transforms_module


@pytest.mark.parametrize("length", [1, 7, 16])
def test_chirp_z_default_matches_fft(length: int) -> None:
    rng = np.random.default_rng(18)
    values = rng.standard_normal(length) + 1j * rng.standard_normal(length)

    result = czt(values)

    np.testing.assert_allclose(result, np.fft.fft(values), atol=1e-12)


def test_chirp_z_supports_arbitrary_parameters_and_axis() -> None:
    rng = np.random.default_rng(31)
    values = rng.standard_normal((3, 11))
    m = 17
    w = 0.98 * np.exp(-1.7j / m)
    a = 0.9 * np.exp(0.2j)

    result = czt(values, m, w=w, a=a, axis=1)
    expected = scipy_signal.czt(values, m=m, w=w, a=a, axis=1)

    assert result.shape == (3, m)
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_chirp_z_validates_inputs() -> None:
    with pytest.raises(ValidationError, match="one- or two-dimensional"):
        czt(np.ones((2, 3, 4)))
    with pytest.raises(ValidationError, match="at least one"):
        czt(np.empty(0))
    with pytest.raises(ValidationError, match="nonzero"):
        czt(np.ones(4), w=0)
    with pytest.raises(ValidationError, match="nonzero"):
        czt(np.ones(4), a=0)


@pytest.mark.parametrize(("length", "nfft"), [(7, 7), (7, 8), (8, 16)])
def test_analytic_signal_matches_scipy(length: int, nfft: int) -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal(length)

    result = analytic_signal(values, nfft=nfft)
    expected = scipy_signal.hilbert(values, N=nfft)

    assert result.shape == (nfft,)
    np.testing.assert_allclose(result, expected, atol=1e-13)


def test_analytic_signal_supports_multichannel_axis() -> None:
    rng = np.random.default_rng(8)
    values = rng.standard_normal((3, 15))

    result = analytic_signal(values, axis=1, nfft=16)
    expected = scipy_signal.hilbert(values, N=16, axis=1)

    assert result.shape == (3, 16)
    np.testing.assert_allclose(result, expected, atol=1e-13)


def test_analytic_signal_updates_signal_array_time_metadata() -> None:
    source = SignalArray(
        data=np.arange(12.0).reshape(6, 2),
        fs=100.0,
        time=2.0 + np.arange(6) / 100.0,
        t0=2.0,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="neural",
    )

    result = analytic_signal(source, nfft=8)

    assert isinstance(result, SignalArray)
    assert result.data.shape == (8, 2)
    assert result.fs == source.fs
    np.testing.assert_allclose(result.time, 2.0 + np.arange(8) / 100.0)
    assert result.unit == source.unit
    assert result.channels == source.channels


def test_analytic_signal_validates_length_and_real_input() -> None:
    with pytest.raises(ValidationError, match="real-valued"):
        analytic_signal(np.ones(4, dtype=complex))
    with pytest.raises(ValidationError, match="greater than or equal"):
        analytic_signal(np.ones(8), nfft=7)

    irregular = SignalArray(
        data=np.ones((4, 1)),
        fs=10.0,
        time=np.array([0.0, 0.1, 0.21, 0.3]),
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1, unit="uV"),
        unit="uV",
        name="irregular",
    )
    preserved = analytic_signal(irregular)
    assert isinstance(preserved, SignalArray)
    np.testing.assert_array_equal(preserved.time, irregular.time)
    with pytest.raises(ValidationError, match="uniformly sampled"):
        analytic_signal(irregular, nfft=8)


def test_spectral_native_backend_error_when_extension_is_absent(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(spectral_transforms_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        czt(np.ones(4))
    with pytest.raises(NativeUnavailableError):
        analytic_signal(np.ones(4))
