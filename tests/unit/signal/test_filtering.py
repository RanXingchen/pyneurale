#!/usr/bin/env python3

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal.filtering import (
    FirCoefficients,
    FirFilter,
    IirCoefficients,
    SosCoefficients,
    StateSpaceCoefficients,
    ZpkCoefficients,
    fir_filter,
    fir_order,
    fir_transition_width,
    firls,
    firwin,
    phase_comp_fir,
)
from neurale.signal.filtering import fir as fir_module


def _signal(data: np.ndarray, fs: float = 1000.0) -> SignalArray:
    return SignalArray(
        data=data,
        fs=fs,
        time=None,
        t0=1.0,
        clock=None,
        channels=ChannelTable.default(data.shape[1], unit="uV"),
        unit="uV",
        name="neural",
        attrs={"fixture": True},
    )


def test_fir_filter_signature_is_native_only() -> None:
    assert set(inspect.signature(fir_filter).parameters) == {
        "x",
        "coefs",
        "axis",
        "zi",
        "return_state",
        "check_finite",
    }


def test_fir_coefficients_are_copied_read_only_and_typed() -> None:
    source = np.array([1, 2, 3])
    coefs = FirCoefficients(source, fs=1000)
    source[0] = 99

    np.testing.assert_array_equal(coefs.taps, [1.0, 2.0, 3.0])
    assert coefs.order == 2
    assert coefs.fs == 1000.0
    assert not coefs.taps.flags.writeable
    assert coefs == FirCoefficients([1, 2, 3], fs=1000)


def test_trusted_coefficients_do_not_freeze_source_arrays() -> None:
    taps = np.array([0.25, 0.5, 0.25])
    b = np.array([1.0, 0.5])
    a = np.array([1.0, -0.2])
    sos = np.array([[1.0, 0.0, 0.0, 1.0, -0.1, 0.2]])
    z = np.array([0.1 + 0.2j])
    p = np.array([0.3 + 0.4j])
    ss_a = np.eye(2)
    ss_b = np.ones(2)
    ss_c = np.ones(2)

    fir = FirCoefficients._from_trusted_array(taps)
    iir = IirCoefficients._from_trusted_arrays(b, a)
    sos_coefs = SosCoefficients._from_trusted_array(sos)
    zpk = ZpkCoefficients._from_trusted_arrays(z, p, 1.0)
    state_space = StateSpaceCoefficients._from_trusted_arrays(ss_a, ss_b, ss_c, 0.0)

    assert taps.flags.writeable
    assert b.flags.writeable
    assert a.flags.writeable
    assert sos.flags.writeable
    assert z.flags.writeable
    assert p.flags.writeable
    assert ss_a.flags.writeable
    assert ss_b.flags.writeable
    assert ss_c.flags.writeable
    assert not fir.taps.flags.writeable
    assert not iir.b.flags.writeable
    assert not iir.a.flags.writeable
    assert not sos_coefs.sos.flags.writeable
    assert not zpk.z.flags.writeable
    assert not zpk.p.flags.writeable
    assert not state_space.a.flags.writeable
    assert not state_space.b.flags.writeable
    assert not state_space.c.flags.writeable


@pytest.mark.parametrize("order", [10, 20])
def test_firls_type_one_matches_scipy(order: int) -> None:
    bands = np.array([0.0, 0.2, 0.3, 1.0])
    desired = np.array([1.0, 1.0, 0.0, 0.0])
    weights = np.array([1.0, 2.0])

    coefs = firls(order, bands, desired, weight=weights)
    expected = scipy_signal.firls(order + 1, bands, desired, weight=weights, fs=2.0)

    np.testing.assert_allclose(coefs.taps, expected, atol=1e-12)


def test_firwin_and_order_helpers() -> None:
    with pytest.warns(RuntimeWarning, match="Increasing FIR order"):
        coefs = firwin(19, 100.0, band="highpass", fs=1000.0)
    assert coefs.order == 20
    assert fir_order(50.0, 1000.0, window="hamming") == 66
    assert fir_transition_width(66, 1000.0, window="hamming") == pytest.approx(50.0)


def test_fir_filter_matches_scipy_for_1d_and_multichannel_axis() -> None:
    rng = np.random.default_rng(42)
    taps = scipy_signal.firwin(11, 0.2)
    coefs = FirCoefficients(taps)
    values = rng.standard_normal((3, 100))

    result = fir_filter(values, coefs, axis=1)
    expected = scipy_signal.lfilter(taps, [1.0], values, axis=1)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(fir_filter(values[0], coefs), expected[0], atol=1e-12)


def test_fir_filter_preserves_signal_array_metadata_and_checks_rate() -> None:
    source = _signal(np.arange(40.0).reshape(20, 2))
    coefs = FirCoefficients([0.25, 0.5, 0.25], 1000.0)

    result = fir_filter(source, coefs)

    assert isinstance(result, SignalArray)
    assert result.fs == source.fs
    assert result.channel_names == source.channel_names
    assert result.attrs == source.attrs
    np.testing.assert_array_equal(source.data, np.arange(40.0).reshape(20, 2))

    with pytest.raises(ValidationError, match="fs"):
        fir_filter(source, FirCoefficients([1.0], 500.0))


def test_fir_filter_state_can_continue_processing() -> None:
    values = np.arange(30.0).reshape(15, 2)
    coefs = FirCoefficients([0.2, 0.3, 0.5])
    first, state = fir_filter(values[:6], coefs, return_state=True)
    second = fir_filter(values[6:], coefs, zi=state)
    expected = fir_filter(values, coefs)
    np.testing.assert_allclose(np.concatenate((first, second)), expected)


def test_fir_filter_realtime_processes_in_place_and_matches_offline() -> None:
    rng = np.random.default_rng(1)
    values = np.ascontiguousarray(rng.standard_normal((101, 4)), dtype=np.float64)
    source = values.copy()
    coefs = FirCoefficients(scipy_signal.firwin(13, 0.25))
    expected = fir_filter(source, coefs)
    processor = FirFilter(coefs, n_channels=4)

    chunks = [values[:17], values[17:63], values[63:]]
    for chunk in chunks:
        assert processor.process(chunk) is chunk

    np.testing.assert_allclose(values, expected, atol=1e-12)
    state = processor.get_state()
    processor.reset()
    np.testing.assert_array_equal(processor.get_state(), np.zeros_like(state))
    processor.set_state(state)
    np.testing.assert_array_equal(processor.get_state(), state)


def test_fir_processor_validates_channels_and_state() -> None:
    processor = FirFilter(FirCoefficients([1.0, 0.5]), n_channels=2)
    with pytest.raises(ValueError, match="channels"):
        processor.process(np.ones((3, 1)))
    with pytest.raises(ValidationError, match="shape"):
        processor.set_state(np.zeros((2, 2)))


def test_phase_compensator_returns_fir_coefficients() -> None:
    phase = np.linspace(0.0, -np.pi / 2.0, 128)
    coefs = phase_comp_fir(phase, 1000.0, num_taps=81)
    assert isinstance(coefs, FirCoefficients)
    assert coefs.taps.size == 81
    assert np.all(np.isfinite(coefs.taps))


def test_filtering_extension_is_required(monkeypatch) -> None:

    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(fir_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        fir_filter(np.arange(5.0), FirCoefficients([1.0]))
