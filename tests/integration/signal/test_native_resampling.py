#!/usr/bin/env python3

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.signal import resample
from neurale.signal.resampling import _design_antialias_filter

_WEIRD_RATIOS = [(2, 1), (1, 3), (2, 3), (5, 7), (32, 125), (125, 256)]


def _native() -> object:
    return importlib.import_module("neurale._native").signal.resampling


def _stream_native(
    values: np.ndarray,
    *,
    up: int,
    down: int,
    taps: np.ndarray,
    chunks: tuple[int, ...],
) -> tuple[np.ndarray, object]:
    native = _native()
    processor = native.Resampler(up, down, taps)
    outputs = []
    start = 0
    for chunk_size in chunks:
        stop = min(start + chunk_size, values.shape[0])
        if stop > start:
            outputs.append(processor.process(values[start:stop]))
        start = stop
    if start < values.shape[0]:
        outputs.append(processor.process(values[start:]))
    outputs.append(processor.flush())
    return np.concatenate(outputs, axis=0), processor


@pytest.mark.parametrize(
    ("length", "channels", "up", "down"),
    [
        (1, 1, 3, 2),
        (2, 3, 1, 4),
        (17, 2, 4, 1),
        (31, 5, 1, 3),
        (64, 4, 147, 160),
        (65, 2, 6, 4),
    ],
)
def test_native_resample_matches_scipy_reference(
    length: int,
    channels: int,
    up: int,
    down: int,
) -> None:
    _native()
    rng = np.random.default_rng(731)
    values = rng.standard_normal((length, channels))

    common = np.gcd(up, down)
    reduced_up, reduced_down = up // common, down // common
    native = resample(values, up, down)
    reference = scipy_signal.resample_poly(
        values,
        reduced_up,
        reduced_down,
        axis=0,
        window=_design_antialias_filter(reduced_up, reduced_down, values.dtype),
    )

    assert native.shape == reference.shape
    np.testing.assert_allclose(native, reference, rtol=1e-12, atol=1e-12)


def test_native_resampled_length_uses_ceil_rule() -> None:
    native = _native()
    assert native.resampled_length(0, 3, 2) == 0
    assert native.resampled_length(1, 3, 2) == 2
    assert native.resampled_length(10, 2, 3) == 7


@pytest.mark.parametrize("neighbor_terms", [0, 8])
def test_native_resample_supports_filter_design_controls(
    neighbor_terms: int,
) -> None:
    _native()
    rng = np.random.default_rng(91)
    values = rng.standard_normal((37, 3))

    native = resample(
        values,
        5,
        4,
        neighbor_terms=neighbor_terms,
        beta=3.2,
    )
    reference = scipy_signal.resample_poly(
        values,
        5,
        4,
        axis=0,
        window=_design_antialias_filter(
            5,
            4,
            values.dtype,
            neighbor_terms=neighbor_terms,
            beta=3.2,
        ),
    )

    np.testing.assert_allclose(native, reference, rtol=1e-12, atol=1e-12)


def test_native_resample_preserves_float32_contract() -> None:
    _native()
    values = np.arange(32, dtype=np.float32)

    native = resample(values, 3, 2)
    reference = scipy_signal.resample_poly(
        values,
        3,
        2,
        window=_design_antialias_filter(3, 2, values.dtype),
    ).astype(np.float32, copy=False)

    assert native.dtype == reference.dtype == np.float32
    np.testing.assert_allclose(native, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("filter_length", [1, 2, 3])
@pytest.mark.parametrize(("up", "down"), _WEIRD_RATIOS)
def test_native_resampler_short_filter_full_vs_chunked_consistency(
    filter_length: int,
    up: int,
    down: int,
) -> None:
    values = np.arange(38.0).reshape(19, 2)
    taps = np.linspace(1.0, 2.0, filter_length)
    native = _native()

    full_processor = native.Resampler(up, down, taps)
    full = np.concatenate(
        (full_processor.process(values), full_processor.flush()),
        axis=0,
    )
    chunked, chunked_processor = _stream_native(
        values,
        up=up,
        down=down,
        taps=taps,
        chunks=(1, 2, 5, 3),
    )

    assert full.shape[0] == full_processor.output_length(values.shape[0])
    assert chunked.shape[0] == chunked_processor.output_length(values.shape[0])
    np.testing.assert_allclose(chunked, full, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(("up", "down"), _WEIRD_RATIOS)
def test_native_resampler_uses_exact_max_process_and_flush_buffers(
    up: int,
    down: int,
) -> None:
    native = _native()
    values = np.arange(45.0).reshape(15, 3)
    taps = np.array([1.0])
    processor = native.Resampler(up, down, taps)

    first = values[:1]
    rest = values[1:]
    first_output = processor.process(first)
    rest_output = processor.process(rest)
    tail = processor.flush()

    assert first_output.shape[0] <= processor.max_output_length(first.shape[0])
    assert rest_output.shape[0] <= processor.max_output_length(rest.shape[0])
    assert tail.shape[0] <= processor.max_flush_length()
    assert first_output.shape[0] + rest_output.shape[0] + tail.shape[0] == processor.output_length(
        values.shape[0]
    )


@pytest.mark.parametrize(("up", "down"), _WEIRD_RATIOS)
def test_native_resampler_interleaved_matches_per_channel(
    up: int,
    down: int,
) -> None:
    rng = np.random.default_rng(814)
    values = rng.standard_normal((23, 4))
    taps = np.array([0.25, 0.5, 0.25])

    multichannel, _ = _stream_native(
        values,
        up=up,
        down=down,
        taps=taps,
        chunks=(4, 1, 7),
    )
    per_channel = []
    for channel in range(values.shape[1]):
        channel_result, _ = _stream_native(
            values[:, channel : channel + 1],
            up=up,
            down=down,
            taps=taps,
            chunks=(4, 1, 7),
        )
        per_channel.append(channel_result[:, 0])

    np.testing.assert_allclose(
        multichannel,
        np.column_stack(per_channel),
        rtol=1e-13,
        atol=1e-13,
    )
