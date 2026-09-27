#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurale.data import FeatureMatrix, SignalArray
from neurale.decoding import KalmanDecoder
from neurale.decoding.persistence import load_decoder, save_decoder
from neurale.models import StandardScaler


def _workflow_data() -> tuple[FeatureMatrix, SignalArray]:
    rng = np.random.default_rng(20260804)
    frames = 160
    rate = 50.0
    pos = np.cumsum(rng.normal(scale=0.05, size=(frames, 2)), axis=0)
    features = FeatureMatrix(
        data=pos @ rng.normal(size=(2, 6)) + rng.normal(scale=0.1, size=(frames, 6)),
        fs=rate,
        feature_names=[f"rate_{idx}" for idx in range(6)],
        unit="Hz",
        shift=1.0 / rate,
        source_signal="units",
    )
    target = SignalArray.from_array(
        pos,
        fs=rate,
        time=features.time.copy(),
        channel_names=("x", "y"),
        channel_types="behavior",
        units="m",
        name="cursor",
    )
    return features, target


def _fitted_decoder(features: FeatureMatrix, target: SignalArray) -> KalmanDecoder:
    return KalmanDecoder(
        scaler=StandardScaler(),
        feature_names=("rate_4", "rate_1", "rate_0"),
        innovation_jitter=1e-9,
    ).fit(features, target)


def _chunk(features: FeatureMatrix, start: int, stop: int | None = None) -> FeatureMatrix:
    return FeatureMatrix(
        data=features.data[start:stop],
        fs=features.fs,
        time=features.time[start:stop].copy(),
        feature_names=features.feature_names,
        source_signal=features.source_signal,
        window_size=features.window_size,
        shift=features.shift,
        unit=features.unit,
        attrs=features.attrs,
    )


def test_kalman_workflow_is_chunk_invariant_and_complete() -> None:
    features, target = _workflow_data()
    decoder = _fitted_decoder(features, target)

    decoder.reset()
    whole = decoder.predict(features)
    decoder.reset()
    first = decoder.predict(_chunk(features, 0, 73))
    second = decoder.predict(_chunk(features, 73))

    assert np.array_equal(np.vstack((first.data, second.data)), whole.data)
    assert np.array_equal(np.concatenate((first.time, second.time)), whole.time)
    assert whole.channels.names == target.channels.names
    assert whole.unit == target.unit
    assert whole.name == target.name
    assert whole.fs == features.fs
    assert decoder.selected_feature_names_ == ("rate_4", "rate_1", "rate_0")


def test_kalman_workflow_survives_artifact_round_trip(tmp_path: Path) -> None:
    features, target = _workflow_data()
    decoder = _fitted_decoder(features, target)
    decoder.reset()
    expected = decoder.predict(features)
    decoder.reset()

    artifact = tmp_path / "kalman-workflow"
    save_decoder(decoder, artifact)
    restored = load_decoder(artifact)
    found = restored.predict(features)

    assert type(restored) is KalmanDecoder
    assert np.array_equal(found.data, expected.data)
    assert np.array_equal(found.time, expected.time)
    assert restored.feature_schema_.fingerprint == decoder.feature_schema_.fingerprint
    assert restored.selected_feature_names_ == decoder.selected_feature_names_
