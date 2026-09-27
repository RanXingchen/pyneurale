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
    IirFilter,
    SosCoefficients,
    SosFilter,
    iir_filter,
    sos_filter,
    sos_filtfilt,
)
from neurale.signal.filtering import iir as iir_module


def _signal(data, fs=1000.0):
    return SignalArray(
        data=np.asarray(data),
        fs=fs,
        time=None,
        t0=2.0,
        clock=None,
        channels=ChannelTable.default(np.asarray(data).shape[1], unit="uV"),
        unit="uV",
        name="source",
        attrs={"stage": 10},
    )


def test_iir_coefficients_normalize_pad_copy_and_freeze() -> None:
    b = np.asarray([2.0, 1.0])
    a = np.asarray([2.0, -0.5, 0.25])
    coefs = IirCoefficients(b, a, fs=1000.0)
    b[0] = 99
    np.testing.assert_array_equal(coefs.b, [1.0, 0.5, 0.0])
    np.testing.assert_array_equal(coefs.a, [1.0, -0.25, 0.125])
    assert coefs.order == 2
    assert not coefs.b.flags.writeable
    assert coefs == IirCoefficients([2, 1], [2, -0.5, 0.25], fs=1000)


def test_sos_coefficients_normalize_copy_and_freeze() -> None:
    source = np.asarray([[2.0, 1, 0, 2, -0.5, 0.25]])
    coefs = SosCoefficients(source, fs=500)
    source[0, 0] = 99
    np.testing.assert_array_equal(coefs.sos, [[1.0, 0.5, 0, 1, -0.25, 0.125]])
    assert coefs.n_sections == 1
    assert not coefs.sos.flags.writeable


@pytest.mark.parametrize(
    ("factory", "arguments", "message"),
    [
        (IirCoefficients, ([], [1]), "at least one"),
        (IirCoefficients, ([1], [0]), "nonzero"),
        (IirCoefficients, ([np.nan], [1]), "finite"),
        (SosCoefficients, (np.ones((2, 5)),), "shape"),
        (SosCoefficients, (np.asarray([[1, 0, 0, 0, 0, 0]]),), "a0"),
    ],
)
def test_iir_coefficient_validation(factory, arguments, message) -> None:
    with pytest.raises(ValidationError, match=message):
        factory(*arguments)


@pytest.mark.parametrize("channels", [1, 3, 67])
@pytest.mark.parametrize("order", [0, 1, 4, 12])
def test_iir_filter_matches_scipy_all_channel_and_order_paths(channels: int, order: int) -> None:
    rng = np.random.default_rng(order * 100 + channels)
    values = rng.standard_normal((97, channels))
    if order == 0:
        b, a = np.asarray([0.75]), np.asarray([1.0])
    else:
        b, a = scipy_signal.butter(order, 0.2, output="ba")
    coefs = IirCoefficients(b, a)

    result = iir_filter(values, coefs)
    expected = scipy_signal.lfilter(coefs.b, coefs.a, values, axis=0)

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("sections", [1, 2, 8])
@pytest.mark.parametrize("channels", [1, 4, 67])
def test_sos_filter_matches_scipy_all_section_and_channel_paths(
    sections: int, channels: int
) -> None:
    rng = np.random.default_rng(sections * 100 + channels)
    values = rng.standard_normal((113, channels))
    sos = scipy_signal.butter(sections * 2, 0.25, output="sos")
    coefs = SosCoefficients(sos)

    result = sos_filter(values, coefs)
    expected = scipy_signal.sosfilt(sos, values, axis=0)

    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)


def test_zero_phase_sos_filter_matches_scipy_and_preserves_metadata() -> None:
    rng = np.random.default_rng(17)
    source = _signal(rng.standard_normal((160, 3)), fs=500.0)
    coefs = SosCoefficients(
        scipy_signal.butter(4, 40, fs=500, output="sos"),
        fs=500,
    )

    result = sos_filtfilt(source, coefs)

    assert isinstance(result, SignalArray)
    np.testing.assert_allclose(
        result.data,
        scipy_signal.sosfiltfilt(np.array(coefs.sos, copy=True), source.data, axis=0),
    )
    assert result.channel_names == source.channel_names
    assert result.attrs == source.attrs
    assert result.fs == source.fs


def test_zero_phase_sos_preserves_axis_and_validates_rate() -> None:
    rng = np.random.default_rng(18)
    values = rng.standard_normal((2, 128))
    coefs = SosCoefficients(scipy_signal.butter(2, 0.2, output="sos"))
    result = sos_filtfilt(values, coefs, axis=1)
    expected = scipy_signal.sosfiltfilt(np.array(coefs.sos, copy=True), values, axis=1)
    np.testing.assert_allclose(result, expected)

    with pytest.raises(ValidationError, match="fs"):
        sos_filtfilt(
            _signal(values.T, fs=500),
            SosCoefficients(coefs.sos, fs=250),
        )


@pytest.mark.parametrize("order", [1, 3, 4, 9])
@pytest.mark.parametrize("channels", [1, 4, 128])
def test_zero_phase_sos_filter_matches_scipy_across_native_paths(order: int, channels: int) -> None:
    rng = np.random.default_rng(order * 1000 + channels)
    values = rng.standard_normal((257, channels))
    source = values.copy()
    sos = scipy_signal.butter(order, 0.23, output="sos")

    result = sos_filtfilt(values, SosCoefficients(sos))
    expected = scipy_signal.sosfiltfilt(sos, values, axis=0)

    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(values, source)


@pytest.mark.parametrize(
    "sos",
    [
        scipy_signal.butter(5, 0.2, btype="highpass", output="sos"),
        scipy_signal.butter(6, [0.15, 0.35], btype="bandpass", output="sos"),
        scipy_signal.butter(5, [0.2, 0.4], btype="bandstop", output="sos"),
        scipy_signal.ellip(6, 1.0, 60.0, 0.25, output="sos"),
        scipy_signal.tf2sos(*scipy_signal.iirnotch(0.2, 20.0)),
    ],
)
def test_zero_phase_sos_filter_matches_scipy_filter_families(
    sos: np.ndarray,
) -> None:
    rng = np.random.default_rng(91)
    values = rng.standard_normal((301, 3))

    np.testing.assert_allclose(
        sos_filtfilt(values, SosCoefficients(sos)),
        scipy_signal.sosfiltfilt(sos, values, axis=0),
        rtol=3e-12,
        atol=3e-12,
    )


def test_zero_phase_sos_filter_padding_boundary_and_finite_check() -> None:
    sos = scipy_signal.butter(3, 0.2, output="sos")
    coefs = SosCoefficients(sos)
    ntaps = (
        2 * len(sos)
        + 1
        - min(
            np.count_nonzero(sos[:, 2] == 0),
            np.count_nonzero(sos[:, 5] == 0),
        )
    )
    edge = 3 * ntaps

    with pytest.raises(ValidationError, match="padding length"):
        sos_filtfilt(np.ones(edge), coefs)

    values = np.linspace(-1.0, 1.0, edge + 1)
    np.testing.assert_allclose(
        sos_filtfilt(values, coefs),
        scipy_signal.sosfiltfilt(sos, values),
        rtol=2e-12,
        atol=2e-12,
    )

    invalid = values.copy()
    invalid[-1] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        sos_filtfilt(invalid, coefs)
    assert np.isnan(
        sos_filtfilt(
            invalid,
            coefs,
            check_finite=False,
        )
    ).any()


def test_iir_and_sos_filter_initial_state_and_nonzero_axis() -> None:
    rng = np.random.default_rng(42)
    values = rng.standard_normal((3, 80))
    b, a = scipy_signal.butter(4, 0.2)
    iir_state = rng.standard_normal((4, 3))
    result, final_state = iir_filter(
        values,
        IirCoefficients(b, a),
        axis=1,
        zi=iir_state,
        return_state=True,
    )
    expected, expected_state = scipy_signal.lfilter(b, a, values.T, axis=0, zi=iir_state)
    np.testing.assert_allclose(result, expected.T)
    np.testing.assert_allclose(final_state, expected_state)

    sos = scipy_signal.butter(6, 0.3, output="sos")
    sos_state = rng.standard_normal((3, 2, 3))
    result, final_state = sos_filter(
        values,
        SosCoefficients(sos),
        axis=1,
        zi=sos_state,
        return_state=True,
    )
    expected, expected_state = scipy_signal.sosfilt(sos, values.T, axis=0, zi=sos_state)
    np.testing.assert_allclose(result, expected.T)
    np.testing.assert_allclose(final_state, expected_state)


def test_iir_filters_validate_state_shapes_types_and_complex_values() -> None:
    with pytest.raises(ValidationError, match="IirCoefficients"):
        iir_filter(np.ones(5), ([1], [1]))
    with pytest.raises(ValidationError, match="SosCoefficients"):
        sos_filter(np.ones(5), np.ones((1, 6)))
    with pytest.raises(ValidationError, match="shape"):
        iir_filter(
            np.ones(10),
            IirCoefficients([1, 0], [1, -0.5]),
            zi=np.zeros((2, 1)),
        )
    with pytest.raises(ValidationError, match="real-valued"):
        iir_filter(
            np.ones(5, dtype=complex),
            IirCoefficients([1], [1]),
        )


def test_iir_and_sos_filter_preserve_signal_array_metadata_and_rate() -> None:
    source = _signal(np.arange(60.0).reshape(30, 2))
    iir = IirCoefficients([0.5], [1, -0.5], fs=1000)
    sos = SosCoefficients([[0.5, 0, 0, 1, -0.5, 0]], fs=1000)
    for result in (iir_filter(source, iir), sos_filter(source, sos)):
        assert isinstance(result, SignalArray)
        assert result.channel_names == source.channel_names
        assert result.attrs == source.attrs
        assert result.fs == source.fs
    with pytest.raises(ValidationError, match="fs"):
        iir_filter(source, IirCoefficients([1], [1], 500))


@pytest.mark.parametrize(
    ("processor_type", "coef_type", "raw", "offline"),
    [
        (IirFilter, IirCoefficients, ([0.2, 0.1], [1, -0.7]), iir_filter),
        (
            SosFilter,
            SosCoefficients,
            scipy_signal.butter(6, 0.25, output="sos"),
            sos_filter,
        ),
    ],
)
def test_iir_processors_process_in_place_and_match_offline(
    processor_type, coef_type, raw, offline
) -> None:
    rng = np.random.default_rng(8)
    values = np.ascontiguousarray(rng.standard_normal((131, 3)), dtype=np.float64)
    source = values.copy()
    coefs = coef_type(*raw) if coef_type is IirCoefficients else coef_type(raw)
    processor = processor_type(coefs, n_channels=3)

    for chunk in (values[:1], values[1:37], values[37:]):
        assert processor.process(chunk) is chunk

    np.testing.assert_allclose(values, offline(source, coefs), atol=2e-12)
    saved = processor.get_state()
    processor.reset()
    np.testing.assert_array_equal(processor.get_state(), np.zeros_like(saved))
    processor.set_state(saved)
    np.testing.assert_array_equal(processor.get_state(), saved)


@pytest.mark.parametrize(
    ("processor_type", "coefs"),
    [
        (IirFilter, IirCoefficients([0.2, 0.1], [1, -0.7])),
        (
            SosFilter,
            SosCoefficients(scipy_signal.butter(4, 0.25, output="sos")),
        ),
    ],
)
def test_iir_processors_accept_realtime_sample_and_chunk_shapes(processor_type, coefs) -> None:
    chunk = np.linspace(0.0, 1.0, 8)
    processor = processor_type(coefs, n_channels=1)
    assert processor.process(chunk) is chunk

    row = np.ascontiguousarray(np.arange(3.0), dtype=np.float64)
    processor = processor_type(coefs, n_channels=3)
    assert processor.process(row) is row


@pytest.mark.parametrize(
    "frame",
    [
        np.arange(5, dtype=np.float32),
        np.arange(6.0).reshape(3, 2).T,
        np.arange(4.0).reshape(2, 2),
    ],
)
def test_iir_realtime_process_rejects_non_native_frames(frame: np.ndarray) -> None:
    processor = IirFilter(IirCoefficients([1], [1]), n_channels=1)

    with pytest.raises((TypeError, ValueError)):
        processor.process(frame)


def test_fir_iir_sos_share_public_lifecycle_protocol() -> None:
    processors = [
        FirFilter(FirCoefficients([1]), n_channels=1),
        IirFilter(IirCoefficients([1], [1]), n_channels=1),
        SosFilter(SosCoefficients([[1, 0, 0, 1, 0, 0]]), n_channels=1),
    ]
    for processor in processors:
        values = np.arange(5.0)
        assert processor.process(values) is values
        np.testing.assert_array_equal(values, np.arange(5.0))
        assert callable(processor.get_state)
        assert callable(processor.set_state)
        assert callable(processor.reset)


def test_empty_frames_preserve_state() -> None:
    iir = IirCoefficients([1, 0], [1, -0.5])
    state = np.asarray([[2.0, 3.0]])
    output, final = iir_filter(np.empty((0, 2)), iir, zi=state, return_state=True)
    assert output.shape == (0, 2)
    np.testing.assert_array_equal(final, state)


def test_filtering_extension_is_required(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(iir_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        sos_filter(np.ones(5), SosCoefficients([[1, 0, 0, 1, 0, 0]]))


def test_public_signatures_are_native_only() -> None:
    expected = {
        "x",
        "coefs",
        "axis",
        "zi",
        "return_state",
        "check_finite",
    }
    assert set(inspect.signature(iir_filter).parameters) == expected
    assert set(inspect.signature(sos_filter).parameters) == expected


def test_iir_coefficients_use_standard_b_a_names() -> None:
    parameters = inspect.signature(IirCoefficients).parameters
    assert list(parameters) == ["b", "a", "fs"]
    coefs = IirCoefficients([1.0], [1.0, -0.5])
    assert not hasattr(coefs, "numerator")
    assert not hasattr(coefs, "denominator")
