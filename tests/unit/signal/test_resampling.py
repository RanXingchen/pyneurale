from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as scipy_signal

from neurale.data import ChannelTable, SignalArray
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.signal import Resampler, resample
from neurale.signal import resampling as resampling_module
from neurale.signal.resampling import _design_antialias_filter


@pytest.mark.parametrize(
    ("length", "up", "down"),
    [
        (1, 3, 2),
        (2, 1, 4),
        (17, 4, 1),
        (31, 1, 3),
        (64, 147, 160),
        (65, 6, 4),
    ],
)
def test_resample_matches_scipy(length: int, up: int, down: int) -> None:
    rng = np.random.default_rng(123)
    values = rng.standard_normal(length)

    result = resample(values, up, down)
    common = np.gcd(up, down)
    reduced_up, reduced_down = up // common, down // common
    expected = scipy_signal.resample_poly(
        values,
        reduced_up,
        reduced_down,
        window=_design_antialias_filter(reduced_up, reduced_down, values.dtype),
    )

    assert result.shape == (int(np.ceil(length * up / down)),)
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_resample_supports_multichannel_nonzero_axis() -> None:
    rng = np.random.default_rng(42)
    values = rng.standard_normal((3, 41))

    result = resample(values, 5, 3, axis=1)
    expected = scipy_signal.resample_poly(
        values,
        5,
        3,
        axis=1,
        window=_design_antialias_filter(5, 3, values.dtype),
    )

    assert result.shape == (3, 69)
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_resample_supports_legacy_filter_design_controls() -> None:
    rng = np.random.default_rng(9)
    values = rng.standard_normal(57)
    taps = _design_antialias_filter(
        5,
        4,
        values.dtype,
        neighbor_terms=8,
        beta=3.2,
    )

    result = resample(
        values,
        5,
        4,
        neighbor_terms=8,
        beta=3.2,
    )
    expected = scipy_signal.resample_poly(values, 5, 4, window=taps)

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_resample_neighbor_terms_zero_uses_nearest_polyphase_kernel() -> None:
    values = np.arange(6.0)
    taps = _design_antialias_filter(3, 2, values.dtype, neighbor_terms=0, beta=5.0)

    result = resample(values, 3, 2, neighbor_terms=0)
    expected = scipy_signal.resample_poly(values, 3, 2, window=taps)

    np.testing.assert_array_equal(result, expected)


def test_resample_reduces_ratio_and_handles_identity() -> None:
    values = np.arange(12.0)

    reduced = resample(values, 6, 4)
    expected = resample(values, 3, 2)
    identity = resample(values, 8, 8)

    np.testing.assert_array_equal(reduced, expected)
    np.testing.assert_array_equal(identity, values)
    assert identity is not values


def test_resample_updates_signal_array_rate_time_and_metadata() -> None:
    source = SignalArray(
        data=np.arange(20.0).reshape(10, 2),
        fs=1000.0,
        time=2.0 + np.arange(10) / 1000.0,
        t0=2.0,
        clock=None,
        channels=ChannelTable.from_names(["ch000", "ch001"], unit=["uV", "mV"]),
        unit=["uV", "mV"],
        name="neural",
        attrs={"source": "fixture"},
    )

    result = resample(source, target_rate=400.0)

    assert isinstance(result, SignalArray)
    assert result.data.shape == (4, 2)
    assert result.fs == pytest.approx(400.0)
    np.testing.assert_allclose(result.time, 2.0 + np.arange(4) / 400.0)
    assert result.channels == source.channels
    assert result.unit == source.unit
    assert result.name == source.name
    assert result.attrs == source.attrs


def test_resample_signal_array_explicit_time_sets_matching_t0() -> None:
    source = SignalArray(
        data=np.arange(20.0).reshape(10, 2),
        fs=1000.0,
        time=2.0 + np.arange(10) / 1000.0,
        t0=1.5,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="offset-time",
    )

    result = resample(source, target_rate=500.0)

    assert isinstance(result, SignalArray)
    assert result.t0 == pytest.approx(2.0)
    np.testing.assert_allclose(result.time, 2.0 + np.arange(5) / 500.0)


def test_resample_target_rate_infers_uniform_explicit_time() -> None:
    source = SignalArray(
        data=np.arange(8.0).reshape(4, 2),
        fs=200.0,
        time=1.0 + np.arange(4) / 200.0,
        t0=1.0,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="explicit-time",
    )

    result = resample(source, target_rate=125.0)

    assert isinstance(result, SignalArray)
    assert result.fs == pytest.approx(125.0)
    assert result.n_samples == 3
    np.testing.assert_allclose(result.time, 1.0 + np.arange(3) / 125.0)


def test_resample_converts_noninteger_rate_ratio_to_reduced_fraction() -> None:
    source = SignalArray(
        data=np.arange(200.0).reshape(100, 2),
        fs=48000.0,
        time=None,
        t0=0.25,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="audio-rate",
    )

    result = resample(source, target_rate=44100.0)
    expected = resample(source.data, 147, 160)

    assert isinstance(result, SignalArray)
    assert result.fs == pytest.approx(44100.0)
    assert result.t0 == source.t0
    assert result.n_samples == 92
    np.testing.assert_allclose(result.time, 0.25 + np.arange(92) / 44100.0)
    np.testing.assert_allclose(result.data, expected)


def test_resample_validates_ratio_and_sampling_contract() -> None:
    values = np.arange(5.0)
    with pytest.raises(ValidationError, match="both required"):
        resample(values, up=2)
    with pytest.raises(ValidationError, match="positive"):
        resample(values, 0, 1)
    with pytest.raises(ValidationError, match="target_rate requires"):
        resample(values, target_rate=100.0)
    with pytest.raises(ValidationError, match="cannot be combined"):
        resample(values, 2, 1, target_rate=100.0)
    with pytest.raises(ValidationError, match="neighbor_terms"):
        resample(values, 2, 1, neighbor_terms=-1)
    with pytest.raises(ValidationError, match="beta"):
        resample(values, 2, 1, beta=-1.0)

    irregular = SignalArray(
        data=np.ones((4, 1)),
        fs=100.0,
        time=np.array([0.0, 0.01, 0.021, 0.03]),
        t0=0.0,
        clock=None,
        channels=ChannelTable.default(1, unit="uV"),
        unit="uV",
        name="irregular",
    )
    with pytest.raises(ValidationError, match="uniformly sampled"):
        resample(irregular, target_rate=50.0)


def test_resample_rejects_complex_input() -> None:
    with pytest.raises(ValidationError, match="real-valued"):
        resample(np.arange(5.0) + 1j, 2, 1)
    with pytest.raises(ValidationError, match="real-valued"):
        Resampler(2, 1).process(np.arange(5.0) + 1j)


def test_resampler_process_rejects_signal_array() -> None:
    source = SignalArray(
        data=np.arange(12.0).reshape(6, 2),
        fs=100.0,
        time=None,
        t0=0.25,
        clock=None,
        channels=ChannelTable.default(2, unit="uV"),
        unit="uV",
        name="stream",
    )

    with pytest.raises(TypeError, match=r"Resampler\.process"):
        Resampler(2, 1).process(source)


def test_resample_requires_native_backend(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(resampling_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        resample(np.arange(5.0), 2, 1)


@pytest.mark.parametrize("values", [np.array([], dtype=np.float32), np.arange(4)])
def test_resample_identity_and_empty_require_native_backend(values, monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise NativeUnavailableError("native unavailable")

    monkeypatch.setattr(resampling_module, "load_native_namespace", unavailable)
    with pytest.raises(NativeUnavailableError):
        resample(values, 1, 1)


def test_resample_preserves_float32_dtype() -> None:
    values = np.arange(16, dtype=np.float32)

    assert resample(values, 3, 2).dtype == np.float32
    assert resample(values, 1, 1).dtype == np.float32
    assert resample(values[:0], 1, 1).dtype == np.float32


def test_resampler_reuses_native_plan() -> None:
    values = np.arange(12.0)
    processor = Resampler(3, 2)

    result = np.concatenate((processor.process(values), processor.flush()))
    expected = resample(values, 3, 2)

    assert processor.output_length(values.size) == expected.size
    assert processor.initial_output_trim >= 0
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("up", "down", "chunks"),
    [
        (3, 2, [5, 7, 13, 39]),
        (1, 3, [1] * 64),
        (147, 160, [3, 4, 5, 52]),
        (2, 1, [1, 3, 5, 55]),
        (2, 3, [2, 3, 7, 52]),
        (5, 7, [4, 6, 9, 45]),
        (32, 125, [1, 8, 13, 42]),
        (125, 256, [5, 11, 17, 31]),
    ],
)
def test_resampler_chunked_stream_matches_offline(up: int, down: int, chunks: list[int]) -> None:
    values = np.arange(128.0).reshape(64, 2)
    expected = resample(values, up, down, neighbor_terms=8)
    processor = Resampler(up, down, neighbor_terms=8)
    processor.prepare(n_channels=2, max_input_samples=max(chunks))

    outputs = []
    start = 0
    for chunk_size in chunks:
        stop = start + chunk_size
        outputs.append(processor.process(values[start:stop]))
        start = stop
    outputs.append(processor.flush())

    result = np.concatenate(outputs, axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("up", "down"),
    [(2, 1), (1, 3), (2, 3), (5, 7), (32, 125), (125, 256)],
)
def test_resampler_flush_reaches_exact_output_length(
    up: int,
    down: int,
) -> None:
    values = np.arange(222.0).reshape(74, 3)
    processor = Resampler(up, down, neighbor_terms=8)

    head = processor.process(values[:17])
    middle = processor.process(values[17:51])
    tail = processor.process(values[51:])
    flushed = processor.flush()
    result = np.concatenate((head, middle, tail, flushed), axis=0)

    assert result.shape[0] == processor.output_length(values.shape[0])
    np.testing.assert_allclose(
        result,
        resample(values, up, down, neighbor_terms=8),
        rtol=1e-13,
        atol=1e-13,
    )


def test_resampler_max_lengths_cover_process_and_flush_outputs() -> None:
    values = np.arange(90.0).reshape(30, 3)
    processor = Resampler(5, 7)

    first = values[:13]
    second = values[13:]
    first_output = processor.process(first)
    second_output = processor.process(second)
    tail = processor.flush()

    assert first_output.shape[0] <= processor.max_output_length(first.shape[0])
    assert second_output.shape[0] <= processor.max_output_length(second.shape[0])
    assert tail.shape[0] <= processor.max_flush_length()
    assert first_output.shape[0] + second_output.shape[0] + tail.shape[
        0
    ] == processor.output_length(values.shape[0])


def test_resampler_multichannel_matches_independent_channels() -> None:
    rng = np.random.default_rng(614)
    values = rng.standard_normal((53, 4))

    result = resample(values, 5, 7)
    expected = np.column_stack(
        [resample(values[:, channel], 5, 7) for channel in range(values.shape[1])]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_resampler_reset_allows_new_channel_count() -> None:
    processor = Resampler(3, 2)
    first = np.arange(16.0).reshape(8, 2)
    second = np.arange(6.0)

    np.concatenate((processor.process(first), processor.flush()))
    result = np.concatenate((processor.process(second), processor.flush()))

    np.testing.assert_allclose(
        result,
        resample(second, 3, 2),
        rtol=1e-13,
        atol=1e-13,
    )


def test_resampler_prepare_validates_arguments() -> None:
    processor = Resampler(3, 2)
    with pytest.raises(ValidationError, match="n_channels"):
        processor.prepare(0, 16)
    with pytest.raises(ValidationError, match="max_input_samples"):
        processor.prepare(2, -1)


def test_resample_rejects_pathological_filter_size() -> None:
    with pytest.raises(ValidationError, match="staged rate conversion"):
        _design_antialias_filter(10_000, 1, np.dtype(float))
