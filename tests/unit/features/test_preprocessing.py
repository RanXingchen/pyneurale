#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, SignalArray
from neurale.exceptions import ValidationError
from neurale.features.preprocessing import (
    NeuralPreprocessor,
    offline_preprocess,
    select_good_channels,
)
from neurale.signal import (
    butter,
    common_reference,
    notch,
    resample,
    sos_filtfilt,
)


def _channels() -> ChannelTable:
    return ChannelTable(
        [
            ChannelInfo("a", 10, "ecog", "uV"),
            ChannelInfo("b", 20, "ecog", "uV", bad=True),
            ChannelInfo("c", 30, "ecog", "uV"),
            ChannelInfo("d", 40, "ecog", "uV", valid=False),
        ]
    )


def _signal(data: np.ndarray, *, rate: float = 400.0, t0: float = 1.5) -> SignalArray:
    return SignalArray(
        data=data,
        fs=rate,
        time=None,
        t0=t0,
        clock=None,
        channels=_channels(),
        unit="uV",
        name="ecog",
        attrs={"session": "fresh"},
    )


def _chunk(
    source: SignalArray,
    start: int,
    stop: int,
    *,
    time_offset: float = 0.0,
) -> SignalArray:
    time = source.time[start:stop] + time_offset
    return SignalArray(
        data=source.data[start:stop],
        fs=source.fs,
        time=time,
        t0=float(time[0]) if time.size else source.t0,
        clock=source.clock,
        channels=source.channels.copy(),
        unit=source.unit,
        name=source.name,
        attrs=dict(source.attrs),
    )


def test_select_good_channels_uses_quality_and_named_bad_channels() -> None:
    source = _signal(np.arange(80.0).reshape(20, 4))

    result = select_good_channels(source, bad_channels=["c"])

    assert isinstance(result, SignalArray)
    assert result.channel_names == ["a"]
    np.testing.assert_array_equal(result.data[:, 0], source.data[:, 0])
    assert result.attrs == source.attrs


def test_select_good_channels_preserves_ndarray_layout() -> None:
    values = np.arange(24.0).reshape(3, 8)

    result = select_good_channels(values, bad_channels=[1], axis=1)

    np.testing.assert_array_equal(result, values[[0, 2], :])
    assert result.shape == (2, 8)


def test_offline_preprocess_matches_explicit_composition() -> None:
    rng = np.random.default_rng(41)
    source = _signal(rng.standard_normal((600, 4)))

    result = offline_preprocess(
        source,
        target_rate=200,
        reference="median",
        notch_freqs=(50,),
        filter_order=4,
        filter_cutoff=(8, 70),
    )

    expected = select_good_channels(source)
    expected = resample(expected, target_rate=200)
    expected = common_reference(expected, method="median")
    expected = sos_filtfilt(
        expected,
        notch(50, bandwidth=2, fs=200, output="sos"),
    )
    expected = sos_filtfilt(
        expected,
        butter(4, (8, 70), band="bandpass", fs=200, output="sos"),
    )

    assert isinstance(result, SignalArray)
    np.testing.assert_allclose(result.data, expected.data)
    assert result.channel_names == ["a", "c"]
    assert result.fs == 200
    np.testing.assert_array_equal(result.time, expected.time)


def test_offline_preprocess_runs_complete_ndarray_nonzero_axis_path() -> None:
    rng = np.random.default_rng(42)
    values = rng.standard_normal((3, 600))
    source = values.copy()

    result = offline_preprocess(
        values,
        fs=400,
        bad_channels=[1],
        target_rate=200,
        reference="median",
        notch_freqs=(50,),
        filter_order=4,
        filter_cutoff=70,
        filter_band="lowpass",
        axis=1,
    )

    expected = select_good_channels(values, bad_channels=[1], axis=1)
    expected = resample(expected, 1, 2, axis=1)
    expected = common_reference(expected, method="median", axis=1)
    expected = sos_filtfilt(
        expected,
        notch(50, bandwidth=2, fs=200, output="sos"),
        axis=1,
    )
    expected = sos_filtfilt(
        expected,
        butter(4, 70, band="lowpass", fs=200, output="sos"),
        axis=1,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 300)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(values, source)


def test_streaming_chunks_match_single_causal_block() -> None:
    rng = np.random.default_rng(43)
    values = rng.standard_normal((360, 4))
    options = dict(
        reference="mean",
        notch_freqs=(50,),
        filter_order=4,
        filter_cutoff=(5, 90),
    )
    chunked = NeuralPreprocessor(400, 4, **options)
    whole = NeuralPreprocessor(400, 4, **options)

    actual = np.concatenate([chunked.process(values[:73]), chunked.process(values[73:])], axis=0)
    expected = whole.process(values)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_streaming_signal_array_preserves_selected_metadata() -> None:
    rng = np.random.default_rng(44)
    source = _signal(rng.standard_normal((120, 4)))
    processor = NeuralPreprocessor(
        source.fs,
        source.channels,
        reference=None,
    )

    result = processor.process(source)

    assert isinstance(result, SignalArray)
    assert result.channel_names == ["a", "c"]
    assert result.attrs == source.attrs
    np.testing.assert_array_equal(result.time, source.time)
    np.testing.assert_array_equal(result.data, source.data[:, [0, 2]])


def test_streaming_resample_and_flush_preserve_metadata() -> None:
    rng = np.random.default_rng(45)
    source = _signal(rng.standard_normal((257, 4)), rate=500)
    processor = NeuralPreprocessor(
        source.fs,
        source.channels,
        target_rate=200,
        reference=None,
        max_input_samples=91,
    )

    outputs = [
        processor.process(_chunk(source, 0, 91)),
        processor.process(_chunk(source, 91, 180)),
        processor.process(_chunk(source, 180, source.n_samples)),
        processor.flush(),
    ]

    assert all(isinstance(output, SignalArray) for output in outputs)
    combined_data = np.concatenate([output.data for output in outputs], axis=0)
    combined_time = np.concatenate([output.time for output in outputs])
    expected = offline_preprocess(source, target_rate=200, reference=None)
    assert isinstance(expected, SignalArray)
    np.testing.assert_allclose(combined_data, expected.data)
    np.testing.assert_allclose(combined_time, expected.time)
    assert np.all(np.diff(combined_time) > 0)
    for output in outputs:
        assert output.fs == 200
        assert output.channel_names == ["a", "c"]
        assert output.unit == "uV"
        assert output.attrs == source.attrs


@pytest.mark.parametrize("time_offset", [-1 / 400, 1 / 400])
def test_streaming_signal_array_requires_time_contiguous_chunks(
    time_offset: float,
) -> None:
    source = _signal(np.ones((60, 4)))
    processor = NeuralPreprocessor(
        source.fs,
        source.channels,
        reference=None,
    )
    processor.process(_chunk(source, 0, 20))

    discontinuous = _chunk(source, 20, 40, time_offset=time_offset)
    with pytest.raises(ValidationError, match="time-contiguous"):
        processor.process(discontinuous)

    processor.reset()
    result = processor.process(discontinuous)
    assert isinstance(result, SignalArray)


def test_streaming_requires_and_preserves_channel_layout() -> None:
    source = _signal(np.ones((20, 4)))
    positional = NeuralPreprocessor(source.fs, 4)
    with pytest.raises(ValidationError, match="ChannelTable"):
        positional.process(source)

    reordered = source.select_channels([20, 10, 30, 40])
    configured = NeuralPreprocessor(source.fs, source.channels)
    with pytest.raises(ValidationError, match="layout changed"):
        configured.process(reordered)


def test_streaming_resample_chunks_and_flush_match_single_block() -> None:
    rng = np.random.default_rng(45)
    values = rng.standard_normal((257, 3))
    chunked = NeuralPreprocessor(
        500,
        3,
        target_rate=200,
        reference="median",
        max_input_samples=100,
    )
    whole = NeuralPreprocessor(500, 3, target_rate=200, reference="median")

    actual = np.concatenate(
        [
            chunked.process(values[:80]),
            chunked.process(values[80:170]),
            chunked.process(values[170:]),
            chunked.flush(),
        ],
        axis=0,
    )
    expected = np.concatenate([whole.process(values), whole.flush()], axis=0)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_streaming_reset_restarts_filter_state() -> None:
    rng = np.random.default_rng(46)
    values = rng.standard_normal((180, 2))
    processor = NeuralPreprocessor(
        400,
        2,
        notch_freqs=(50,),
        reference=None,
    )

    first = processor.process(values)
    processor.reset()
    second = processor.process(values)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "values,message",
    [
        (np.ones((8, 4), dtype=complex), "real-valued"),
        (np.full((8, 4), np.nan), "finite"),
        (np.full((8, 4), np.inf), "finite"),
    ],
)
def test_streaming_signal_array_rejects_invalid_numeric_data(
    values: np.ndarray,
    message: str,
) -> None:
    processor = NeuralPreprocessor(400, _channels(), reference=None)

    with pytest.raises(ValidationError, match=message):
        processor.process(_signal(values))

    result = processor.process(_signal(np.ones((8, 4))))
    assert isinstance(result, SignalArray)


def test_streaming_preprocessor_rejects_irregular_signal_times() -> None:
    source = _signal(np.ones((3, 4)))
    irregular = SignalArray(
        data=source.data,
        fs=source.fs,
        time=np.array([1.5, 1.5025, 1.5075]),
        t0=1.5,
        clock=source.clock,
        channels=source.channels,
        unit=source.unit,
        name=source.name,
        attrs=dict(source.attrs),
    )
    processor = NeuralPreprocessor(400, source.channels, reference=None)

    with pytest.raises(ValidationError, match="match fs"):
        processor.process(irregular)


def test_streaming_preprocessor_empty_block_does_not_fix_kind() -> None:
    source = _signal(np.ones((8, 4)))
    processor = NeuralPreprocessor(400, source.channels, reference=None)

    empty = processor.process(np.empty((0, 4)))
    assert isinstance(empty, np.ndarray)
    assert isinstance(processor.process(source), SignalArray)

    with pytest.raises(ValidationError, match="cannot mix"):
        processor.process(np.ones((1, 4)))
