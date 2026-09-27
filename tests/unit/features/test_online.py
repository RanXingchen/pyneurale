#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, SignalArray
from neurale.exceptions import ValidationError
from neurale.features import (
    BandpowerProcessor,
    HilbertEnvelopeProcessor,
    LmpProcessor,
    bandpower_features,
)
from neurale.signal import butter, sos_filter

_RATE = 256.0
_CHANNELS = ChannelTable(
    [
        ChannelInfo("motor-a", 10, "ecog", "uV"),
        ChannelInfo("motor-b", 20, "ecog", "uV"),
    ]
)
_BANDS = {"alpha": (8.0, 13.0), "gamma": (35.0, 45.0)}


def _values(samples: int = 640) -> np.ndarray:
    time = np.arange(samples) / _RATE
    return np.column_stack(
        (
            2.0 * np.sin(2.0 * np.pi * 10.0 * time) + 0.1 * np.sin(2.0 * np.pi * 40.0 * time),
            0.1 * np.sin(2.0 * np.pi * 10.0 * time) + 3.0 * np.sin(2.0 * np.pi * 40.0 * time),
        )
    )


def _signal(data: np.ndarray, start: int = 0) -> SignalArray:
    return SignalArray(
        data=data,
        fs=_RATE,
        time=None,
        t0=3.0 + start / _RATE,
        clock=None,
        channels=_CHANNELS,
        unit="uV",
        name="motor-cortex",
    )


def _factories() -> tuple[Callable[[], object], ...]:
    return (
        lambda: BandpowerProcessor(
            _RATE,
            2,
            _BANDS,
            0.5,
            0.125,
            weighting="unity",
        ),
        lambda: HilbertEnvelopeProcessor(_RATE, 2, _BANDS, 0.5, 0.125),
        lambda: LmpProcessor(_RATE, 2, cutoff=20.0, window_size=0.5, shift=0.125),
    )


@pytest.mark.parametrize("factory", _factories())
def test_online_processors_are_invariant_to_chunk_boundaries(factory) -> None:
    values = _values()
    whole = factory().process(values)
    chunked_processor = factory()
    outputs = []
    times = []
    start = 0
    for size in (1, 17, 64, 3, 129, 11, 211, 204):
        stop = min(start + size, values.shape[0])
        result = chunked_processor.process(values[start:stop])
        outputs.append(result.data)
        times.append(result.time)
        start = stop
        if start == values.shape[0]:
            break

    np.testing.assert_allclose(np.vstack(outputs), whole.data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.concatenate(times), whole.time)


def test_bandpower_processor_matches_offline_complete_windows() -> None:
    values = _values()
    online = BandpowerProcessor(
        _RATE,
        _CHANNELS,
        _BANDS,
        0.5,
        0.125,
        weighting="unity",
        detrend="mean",
    ).process(_signal(values))
    offline = bandpower_features(
        _signal(values),
        _BANDS,
        0.5,
        0.125,
        weighting="unity",
        detrend="mean",
    )

    np.testing.assert_allclose(online.data, offline.data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(online.time, offline.time)
    assert online.feature_names == offline.feature_names


def test_lmp_processor_matches_causal_filter_and_window_means() -> None:
    values = _values()
    processor = LmpProcessor(_RATE, 2, cutoff=20.0, window_size=0.5, shift=0.125)
    result = processor.process(values)
    filtered = sos_filter(
        values,
        butter(4, 20.0, band="lowpass", fs=_RATE, output="sos"),
    )
    starts = np.arange(result.n_frames) * processor.hop_samples
    expected = np.asarray(
        [np.mean(filtered[start : start + processor.window_samples], axis=0) for start in starts]
    )

    np.testing.assert_allclose(result.data, expected, rtol=1e-12, atol=1e-12)


def test_hilbert_processor_separates_bands_after_causal_warmup() -> None:
    result = HilbertEnvelopeProcessor(_RATE, 2, _BANDS, 0.5, 0.125).process(_values(1280))
    interior = result.data[4:]

    assert np.mean(interior[:, 0]) > 10.0 * np.mean(interior[:, 1])
    assert np.mean(interior[:, 3]) > 10.0 * np.mean(interior[:, 2])


def test_signal_array_metadata_and_continuity_contract() -> None:
    values = _values(256)
    processor = LmpProcessor(_RATE, _CHANNELS, cutoff=20.0, window_size=0.5, shift=0.125)
    first = processor.process(_signal(values[:100], 0))
    second = processor.process(_signal(values[100:], 100))

    assert first.n_frames == 0
    assert second.feature_names == ["lmp:motor-a", "lmp:motor-b"]
    assert second.source_signal == "motor-cortex"
    np.testing.assert_allclose(second.time, 3.25 + np.arange(second.n_frames) * 0.125)
    with pytest.raises(ValidationError, match="time-contiguous"):
        processor.process(_signal(values[:8], 300))


def test_flush_discards_partial_window_and_reanchors_stream() -> None:
    values = _values(128)
    processor = BandpowerProcessor(_RATE, 2, _BANDS, 0.5, 0.125, weighting="unity")

    assert processor.process(values[:127]).n_frames == 0
    assert processor.flush().n_frames == 0
    assert processor.process(values[127:]).n_frames == 0

    processor.flush()
    assert processor.process(values).n_frames == 1


def test_empty_block_does_not_fix_online_stream_kind() -> None:
    processor = LmpProcessor(_RATE, _CHANNELS, cutoff=20.0)
    assert processor.process(np.empty((0, 2))).n_frames == 0
    assert processor.process(_signal(_values(1))).n_frames == 0

    with pytest.raises(ValidationError, match="cannot mix"):
        processor.process(np.ones((1, 2)))


def test_rejected_online_block_does_not_commit_stream_state() -> None:
    processor = LmpProcessor(_RATE, _CHANNELS, cutoff=20.0)
    invalid = _signal(_values(8).astype(complex))
    invalid.data.imag[:] = 1.0

    with pytest.raises(ValidationError, match="real-valued"):
        processor.process(invalid)

    result = processor.process(_signal(_values(8)))
    assert result.source_signal == "motor-cortex"


def test_online_processor_rejects_irregular_signal_times() -> None:
    signal = SignalArray(
        data=_values(3),
        fs=_RATE,
        time=np.array([3.0, 3.0 + 1.0 / _RATE, 3.0 + 3.0 / _RATE]),
        t0=3.0,
        clock=None,
        channels=_CHANNELS,
        unit="uV",
        name="motor-cortex",
    )
    processor = LmpProcessor(_RATE, _CHANNELS, cutoff=20.0)

    with pytest.raises(ValidationError, match="match fs"):
        processor.process(signal)
