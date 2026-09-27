#!/usr/bin/env python3

"""Fitted decoders of every supported type, and the data they were fitted on.

One dataset for the continuous decoders and one for the classifier, both small
enough to fit instantly and structured enough that a decoder actually learns
something -- a round trip that compares two decoders predicting noise would
pass however wrong the format was.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from neurale.data import FeatureMatrix, SignalArray
from neurale.decoding import (
    ClassificationTarget,
    KalmanDecoder,
    LDADecoder,
    LinearDecoder,
    RidgeDecoder,
)
from neurale.decoding.sequence import BeamSearchDecoder, NgramLanguageModel, Vocabulary
from neurale.models import MinMaxScaler, StandardScaler
from neurale.models.preprocessing import TemporalContext

_RATE = 50.0
_FRAMES = 160


@dataclass(frozen=True)
class ContinuousFixture:
    """Aligned features and a continuous target a decoder can reconstruct."""

    features: FeatureMatrix
    target: SignalArray


@dataclass(frozen=True)
class ClassificationFixture:
    """Aligned features and a separable three-class target."""

    features: FeatureMatrix
    target: ClassificationTarget


@pytest.fixture(scope="session")
def continuous() -> ContinuousFixture:
    rng = np.random.default_rng(20260808)
    pos = np.cumsum(rng.normal(scale=0.05, size=(_FRAMES, 2)), axis=0)
    features = FeatureMatrix(
        data=pos @ rng.normal(size=(2, 6)) + rng.normal(scale=0.1, size=(_FRAMES, 6)),
        fs=_RATE,
        feature_names=[f"rate_{idx}" for idx in range(6)],
        unit="Hz",
        shift=1.0 / _RATE,
        source_signal="units",
    )
    target = SignalArray.from_array(
        pos,
        fs=_RATE,
        time=features.time.copy(),
        channel_names=["x", "y"],
        channel_types="behavior",
        units="m",
        name="cursor",
    )
    return ContinuousFixture(features=features, target=target)


@pytest.fixture(scope="session")
def classification() -> ClassificationFixture:
    rng = np.random.default_rng(20260809)
    labels = np.array(["left", "right", "rest"] * (_FRAMES // 3 + 1))[:_FRAMES]
    centers = {"left": [-1.0, 0.5, 0.0], "right": [1.0, -0.5, 0.0], "rest": [0.0, 0.0, 1.0]}
    data = np.array([centers[label] for label in labels]) + rng.normal(scale=0.3, size=(_FRAMES, 3))
    features = FeatureMatrix(
        data=data,
        fs=_RATE,
        feature_names=["beta", "gamma", "hfa"],
        unit="uV^2",
        shift=1.0 / _RATE,
    )
    return ClassificationFixture(
        features=features,
        target=ClassificationTarget(labels=labels, time=features.time.copy()),
    )


@pytest.fixture
def kalman(continuous: ContinuousFixture) -> KalmanDecoder:
    return KalmanDecoder(scaler=StandardScaler(), innovation_jitter=1e-9).fit(
        continuous.features, continuous.target
    )


@pytest.fixture
def kalman_selected(continuous: ContinuousFixture) -> KalmanDecoder:
    """A Kalman decoder that consumes a reordered subset of the features.

    Feature identity is the part of the model contract a dimension check cannot
    see, so the tests that damage it need a decoder whose selection is neither
    complete nor in schema order.
    """

    return KalmanDecoder(
        scaler=StandardScaler(),
        feature_names=["rate_4", "rate_1", "rate_0"],
    ).fit(continuous.features, continuous.target)


@pytest.fixture
def linear(continuous: ContinuousFixture) -> LinearDecoder:
    return LinearDecoder(
        scaler=MinMaxScaler(feature_range=(-1.0, 1.0)),
        context=TemporalContext(left=2, right=1),
    ).fit(continuous.features, continuous.target)


@pytest.fixture
def ridge(continuous: ContinuousFixture) -> RidgeDecoder:
    return RidgeDecoder(0.7, scaler=StandardScaler(with_mean=True, with_std=False)).fit(
        continuous.features, continuous.target
    )


@pytest.fixture
def lda(classification: ClassificationFixture) -> LDADecoder:
    return LDADecoder(scaler=StandardScaler(), shrinkage="auto").fit(
        classification.features, classification.target
    )


@pytest.fixture(scope="session")
def language_model() -> NgramLanguageModel:
    """The shared prior. Read it; never refit it -- the scope is the session."""

    return _fitted_language_model()


@pytest.fixture
def adaptive_language_model() -> NgramLanguageModel:
    """A private copy for the tests that refit a model mid-sequence.

    An adaptive model is state, so a test that moves one needs its own: the
    session fixture is shared, and a refit through it would reach every other
    test that reads a prior.
    """

    return _fitted_language_model()


def _fitted_language_model() -> NgramLanguageModel:
    vocabulary = Vocabulary(["a", "b", "<unk>", "stop"], bos="<s>", eos="stop", unknown="<unk>")
    return NgramLanguageModel(vocabulary, smoothing=0.5).fit(
        [["a", "b"], ["b", "a", "b"], ["a"], ["b", "b", "a"]]
    )


@pytest.fixture
def beam(language_model: NgramLanguageModel) -> BeamSearchDecoder:
    return _decoder(language_model)


@pytest.fixture
def adaptive_beam(adaptive_language_model: NgramLanguageModel) -> BeamSearchDecoder:
    return _decoder(adaptive_language_model)


def _decoder(model: NgramLanguageModel) -> BeamSearchDecoder:
    return BeamSearchDecoder(
        model,
        beam_width=4,
        max_completed=3,
        language_weight=0.8,
        length_normalization=0.5,
    )


@pytest.fixture(scope="session")
def steps() -> np.ndarray:
    """Four score rows over the four-token vocabulary, each summing to one."""

    return np.array(
        [
            [0.50, 0.30, 0.10, 0.10],
            [0.20, 0.55, 0.10, 0.15],
            [0.30, 0.20, 0.10, 0.40],
            [0.10, 0.10, 0.10, 0.70],
        ]
    )
