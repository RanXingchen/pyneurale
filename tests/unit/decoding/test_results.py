#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock
from neurale.decoding import ClassificationPrediction
from neurale.exceptions import ValidationError

_CLASSES = np.array([10, 20, 30])
_SCORES = np.array(
    [
        [2.0, 1.0, 0.5],
        [0.1, 3.0, 0.2],
        [0.0, 0.0, 4.0],
    ]
)


def _probabilities() -> np.ndarray:
    exponentiated = np.exp(_SCORES)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _prediction(**overrides: object) -> ClassificationPrediction:
    settings: dict[str, object] = {
        "labels": np.array([10, 20, 30]),
        "classes": _CLASSES,
        "time": np.array([0.0, 0.05, 0.1]),
        "scores": _SCORES,
        "probabilities": _probabilities(),
    }
    settings.update(overrides)
    return ClassificationPrediction(**settings)


# --------------------------------------------------------------------------------------
# Result metadata
# --------------------------------------------------------------------------------------


def test_prediction_carries_labels_scores_and_times() -> None:
    prediction = _prediction()

    assert np.array_equal(prediction.labels, np.array([10, 20, 30]))
    assert np.array_equal(prediction.classes, _CLASSES)
    assert np.array_equal(prediction.time, np.array([0.0, 0.05, 0.1]))
    assert prediction.scores.shape == (3, 3)
    assert prediction.probabilities.shape == (3, 3)
    assert prediction.n_samples == 3
    assert prediction.n_classes == 3


def test_score_columns_follow_declared_class_order() -> None:
    prediction = _prediction()
    pos = int(np.flatnonzero(prediction.classes == 20)[0])

    assert prediction.scores[1, pos] == pytest.approx(3.0)


def test_class_indices_address_declared_order() -> None:
    prediction = _prediction(labels=np.array([30, 30, 10]))

    assert np.array_equal(prediction.class_indices, np.array([2, 2, 0]))


def test_scores_and_probabilities_are_optional() -> None:
    prediction = _prediction(scores=None, probabilities=None)

    assert prediction.scores is None
    assert prediction.probabilities is None


def test_prediction_carries_reference_clock_metadata() -> None:
    clock = Clock(name="task", type="host")

    assert _prediction(clock=clock).clock is clock


def test_labels_need_not_be_score_argmax() -> None:
    """A decoder may abstain or threshold; forcing agreement would hide that."""

    prediction = _prediction(labels=np.array([10, 10, 10]))

    assert np.array_equal(prediction.labels, np.array([10, 10, 10]))


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def test_predicted_label_must_be_declared_class() -> None:
    with pytest.raises(ValidationError, match="does not declare"):
        _prediction(labels=np.array([10, 20, 40]))


@pytest.mark.parametrize("name", ["scores", "probabilities"])
def test_matrix_shape_matches_samples_and_classes(name: str) -> None:
    with pytest.raises(ValidationError, match=rf"{name} must have shape \(3, 3\)"):
        _prediction(**{name: np.zeros((3, 2))})


@pytest.mark.parametrize("name", ["scores", "probabilities"])
def test_matrix_must_be_finite_floating_point(name: str) -> None:
    with pytest.raises(ValidationError, match="finite values"):
        _prediction(**{name: np.full((3, 3), np.nan)})
    with pytest.raises(ValidationError, match="floating-point"):
        _prediction(**{name: np.zeros((3, 3), dtype=np.int64)})


def test_probabilities_lie_in_unit_interval() -> None:
    values = _probabilities()
    values[0] = [-0.1, 0.6, 0.5]

    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        _prediction(probabilities=values)


def test_probability_rows_must_sum_to_one() -> None:
    values = _probabilities()
    values[1] *= 0.5

    with pytest.raises(ValidationError, match="sum to one"):
        _prediction(probabilities=values)


def test_probability_rows_tolerate_few_ulps() -> None:
    """A float64 normalization leaves slack; renormalizing here would hide it."""

    values = _probabilities()
    values[0, 0] = np.nextafter(values[0, 0], 1.0)

    assert _prediction(probabilities=values).probabilities is not None


def test_time_vector_must_match_labels() -> None:
    with pytest.raises(ValidationError, match="length must be 3"):
        _prediction(time=np.array([0.0, 0.1]))


def test_empty_prediction_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one sample"):
        ClassificationPrediction(
            labels=np.array([], dtype=np.int64),
            classes=_CLASSES,
            time=np.array([]),
        )


# --------------------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------------------


def test_prediction_owns_non_writable_arrays() -> None:
    scores = _SCORES.copy()
    prediction = _prediction(scores=scores)

    scores[0, 0] = 99.0

    assert prediction.scores[0, 0] == pytest.approx(2.0)
    for arr in (prediction.labels, prediction.classes, prediction.time, prediction.scores):
        assert not arr.flags.writeable


def test_prediction_is_immutable() -> None:
    prediction = _prediction()

    with pytest.raises(AttributeError):
        prediction.labels = np.array([30, 30, 30])


def test_prediction_metadata_is_deep_frozen() -> None:
    prediction = _prediction(attrs={"run": 7, "nested": {"block": 1}})

    with pytest.raises(TypeError):
        prediction.attrs["run"] = 8
    with pytest.raises(TypeError):
        prediction.attrs["nested"]["block"] = 2
