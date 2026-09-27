#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.signal.filtering import (
    IirCoefficients,
    IirFilter,
    SosCoefficients,
    SosFilter,
    iir_filter,
    sos_filter,
)


def _native() -> object:
    native = importlib.import_module("neurale._native").signal
    native.filtering.iir_filter  # noqa: B018
    return native


@pytest.mark.parametrize("order", [0, 1, 4, 12])
@pytest.mark.parametrize("channels", [1, 4, 67])
@pytest.mark.parametrize("samples", [0, 1, 127])
def test_iir_filter_matches_scipy(
    samples: int,
    channels: int,
    order: int,
) -> None:
    _native()
    rng = np.random.default_rng(samples * 1000 + channels * 10 + order)
    values = rng.standard_normal((samples, channels))
    if order == 0:
        b, a = np.asarray([0.5]), np.asarray([1.0])
    else:
        b, a = scipy_signal.butter(order, 0.25)
    coefs = IirCoefficients(b, a)
    state = rng.standard_normal((coefs.order, channels))

    actual, actual_state = iir_filter(values, coefs, zi=state, return_state=True)
    if samples == 0:
        expected = np.empty_like(values)
        expected_state = state
    else:
        expected, expected_state = scipy_signal.lfilter(coefs.b, coefs.a, values, axis=0, zi=state)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual_state, expected_state, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("sections", [1, 2, 8])
@pytest.mark.parametrize("channels", [1, 4, 67])
@pytest.mark.parametrize("samples", [0, 1, 127])
def test_sos_filter_matches_scipy(
    samples: int,
    channels: int,
    sections: int,
) -> None:
    _native()
    rng = np.random.default_rng(samples * 1000 + channels * 10 + sections)
    values = rng.standard_normal((samples, channels))
    coefs = SosCoefficients(scipy_signal.butter(sections * 2, 0.25, output="sos"))
    state = rng.standard_normal((coefs.n_sections, 2, channels))

    actual, actual_state = sos_filter(values, coefs, zi=state, return_state=True)
    if samples == 0:
        expected = np.empty_like(values)
        expected_state = state
    else:
        expected, expected_state = scipy_signal.sosfilt(coefs.sos.copy(), values, axis=0, zi=state)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_state, expected_state, rtol=1e-10, atol=1e-10)


def test_all_native_iir_kernels_match_builtin() -> None:
    native = _native().filtering
    rng = np.random.default_rng(99)
    values = rng.standard_normal((257, 8))
    b, a = scipy_signal.butter(8, 0.2)
    state = rng.standard_normal((8, 8))
    reference, reference_state = native.iir_filter(values, b, a, state, "builtin-df2t")[:2]
    for kernel in native.iir_available_kernels():
        result, final_state, selected = native.iir_filter(values, b, a, state, kernel)
        assert selected == kernel
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(final_state, reference_state, rtol=1e-10, atol=1e-10)


def test_all_native_sos_kernels_match_builtin() -> None:
    native = _native().filtering
    rng = np.random.default_rng(101)
    values = rng.standard_normal((257, 8))
    sos = scipy_signal.butter(8, 0.2, output="sos")
    state = rng.standard_normal((4, 2, 8))
    reference, reference_state = native.sos_filter(values, sos, state, "builtin-df2t")[:2]
    for kernel in native.iir_available_kernels():
        result, final_state, selected = native.sos_filter(values, sos, state, kernel)
        assert selected == kernel
        np.testing.assert_allclose(result, reference, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(final_state, reference_state, rtol=2e-12, atol=2e-12)


def test_iir_kernel_selection_depends_on_channels_only() -> None:
    native = _native().filtering
    assert native.iir_kernel_name(1, 1, 64) == "builtin-df2t"
    if "mkl-blas-df2t" in native.iir_available_kernels():
        low_order = native.iir_kernel_name(1, 256, 1)
        high_order = native.iir_kernel_name(1, 256, 64)
        assert low_order == high_order == "mkl-blas-df2t"


def test_native_iir_sample_processor_filters_multichannel_frames() -> None:
    native = _native().filtering
    rng = np.random.default_rng(196)
    values = rng.standard_normal((41, 5))
    coefs = IirCoefficients(*scipy_signal.butter(4, 0.2))
    state = rng.standard_normal((coefs.order, values.shape[1]))
    processor = native._IirSampleProcessor(coefs.b, coefs.a, state)
    actual = values.copy()
    processor.process(actual)
    expected, expected_state = iir_filter(values, coefs, zi=state, return_state=True)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(processor.state(), expected_state)


def test_native_iir_sample_processor_uses_selected_realtime_kernel() -> None:
    native = _native().filtering
    rng = np.random.default_rng(201)
    values = rng.standard_normal((7, 256))
    coefs = IirCoefficients(*scipy_signal.butter(4, 0.2))
    processor = native._IirSampleProcessor(
        coefs.b,
        coefs.a,
        np.zeros((coefs.order, values.shape[1])),
    )
    assert processor.kernel == native.iir_kernel_name(1, values.shape[1], coefs.order)

    actual = values.copy()
    processor.process(actual)
    np.testing.assert_allclose(
        actual,
        iir_filter(values, coefs, check_finite=False),
    )


def test_native_sos_sample_processor_filters_multichannel_frames() -> None:
    native = _native().filtering
    rng = np.random.default_rng(197)
    values = rng.standard_normal((41, 5))
    coefs = SosCoefficients(scipy_signal.butter(6, 0.2, output="sos"))
    state = rng.standard_normal((coefs.n_sections, 2, values.shape[1]))
    processor = native._SosSampleProcessor(coefs.sos, state)
    actual = values.copy()
    processor.process(actual)
    expected, expected_state = sos_filter(values, coefs, zi=state, return_state=True)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(processor.state(), expected_state)


def test_native_sos_sample_processor_uses_selected_realtime_kernel() -> None:
    native = _native().filtering
    rng = np.random.default_rng(202)
    values = rng.standard_normal((7, 256))
    coefs = SosCoefficients(scipy_signal.butter(12, 0.2, output="sos"))
    state = rng.standard_normal((coefs.n_sections, 2, values.shape[1]))
    processor = native._SosSampleProcessor(coefs.sos, state)
    assert processor.kernel == native.sos_kernel_name(1, values.shape[1], coefs.n_sections)

    actual = values.copy()
    processor.process(actual)
    expected, expected_state = native.sos_filter(values, coefs.sos, state, "builtin-df2t")[:2]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(processor.state(), expected_state, rtol=2e-12, atol=2e-12)


def test_python_realtime_processors_process_in_place() -> None:
    _native()
    rng = np.random.default_rng(198)
    values = np.ascontiguousarray(rng.standard_normal((41, 5)), dtype=np.float64)
    source = values.copy()
    iir_coefs = IirCoefficients(*scipy_signal.butter(4, 0.2))
    iir = IirFilter(iir_coefs, n_channels=values.shape[1])
    assert iir.process(values) is values
    np.testing.assert_allclose(values, iir_filter(source, iir_coefs))

    values = np.ascontiguousarray(source.copy(), dtype=np.float64)
    sos_coefs = SosCoefficients(scipy_signal.butter(6, 0.2, output="sos"))
    sos = SosFilter(sos_coefs, n_channels=values.shape[1])
    assert sos.process(values) is values
    np.testing.assert_allclose(values, sos_filter(source, sos_coefs))


def test_native_bindings_validate_raw_buffer_contracts() -> None:
    native = _native().filtering
    with pytest.raises(ValueError):
        native.iir_filter(np.ones((4, 1)), [1.0], [1.0, 0.0], np.zeros((0, 1)))
    with pytest.raises(ValueError):
        native.sos_filter(np.ones((4, 1)), np.ones((1, 5)), np.zeros((1, 2, 1)))
