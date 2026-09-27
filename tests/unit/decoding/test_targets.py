#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock
from neurale.decoding import ClassificationTarget
from neurale.exceptions import ValidationError


def _labels() -> np.ndarray:
    return np.array([2, 0, 2, 1, 0, 1])


def _time() -> np.ndarray:
    return np.arange(6, dtype=np.float64) * 0.05


# --------------------------------------------------------------------------------------
# Class ordering
# --------------------------------------------------------------------------------------


def test_default_class_order_ignores_row_order() -> None:
    forward = ClassificationTarget(labels=_labels(), time=_time())
    reversed_rows = ClassificationTarget(labels=_labels()[::-1].copy(), time=_time())

    assert np.array_equal(forward.classes, np.array([0, 1, 2]))
    assert np.array_equal(forward.classes, reversed_rows.classes)


def test_explicit_class_order_is_kept_exactly() -> None:
    """The class order is the column order of every later score, so it is not re-sorted."""

    target = ClassificationTarget(labels=_labels(), time=_time(), classes=[2, 1, 0])

    assert np.array_equal(target.classes, np.array([2, 1, 0]))
    assert np.array_equal(target.class_indices, np.array([0, 2, 0, 1, 2, 1]))


def test_explicit_class_order_may_declare_unobserved_class() -> None:
    target = ClassificationTarget(labels=_labels(), time=_time(), classes=[0, 1, 2, 3])

    assert target.n_classes == 4
    assert np.array_equal(target.class_indices, np.array([2, 0, 2, 1, 0, 1]))


def test_explicit_class_order_cannot_omit_observed_class() -> None:
    with pytest.raises(ValidationError, match="does not declare"):
        ClassificationTarget(labels=_labels(), time=_time(), classes=[0, 1])


def test_class_indices_address_declared_order() -> None:
    target = ClassificationTarget(labels=_labels(), time=_time())

    assert np.array_equal(target.classes[target.class_indices], target.labels)


def test_string_labels_are_supported() -> None:
    labels = np.array(["left", "right", "left"])
    target = ClassificationTarget(labels=labels, time=np.array([0.0, 0.1, 0.2]))

    assert list(target.classes) == ["left", "right"]
    assert np.array_equal(target.class_indices, np.array([0, 1, 0]))


@pytest.mark.parametrize(
    ("classes", "message"),
    [
        pytest.param([0, 0, 1], "must be unique", id="duplicate"),
        pytest.param([], "at least one class", id="empty"),
        pytest.param([[0, 1], [1, 2]], "1D array", id="2D"),
        pytest.param(["a", "b", "c"], "same kind of values", id="mismatched-kind"),
    ],
)
def test_invalid_class_order_is_rejected(classes: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ClassificationTarget(labels=_labels(), time=_time(), classes=classes)


# --------------------------------------------------------------------------------------
# Labels, time, and clock
# --------------------------------------------------------------------------------------


def test_target_reports_size() -> None:
    target = ClassificationTarget(labels=_labels(), time=_time())

    assert target.n_samples == 6
    assert target.n_classes == 3


def test_target_carries_reference_clock_metadata() -> None:
    clock = Clock(name="task", type="host", synchronization_domain="rig")
    target = ClassificationTarget(labels=_labels(), time=_time(), clock=clock)

    assert target.clock is clock


def test_target_rejects_non_clock() -> None:
    with pytest.raises(ValidationError, match="clock must be a Clock"):
        ClassificationTarget(labels=_labels(), time=_time(), clock="task")


@pytest.mark.parametrize(
    ("time", "message"),
    [
        pytest.param(np.arange(5, dtype=np.float64), "length must be 6", id="length"),
        pytest.param(np.full(6, np.nan), "finite timestamps", id="non-finite"),
        pytest.param(np.array([0.0, 0.2, 0.1, 0.3, 0.4, 0.5]), "non-decreasing", id="backwards"),
        pytest.param(np.zeros((6, 1)), "1D array", id="2D"),
        pytest.param([0.0] * 6, "numpy.ndarray", id="list"),
    ],
)
def test_invalid_time_vector_is_rejected(time: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ClassificationTarget(labels=_labels(), time=time)


def test_repeated_timestamps_are_allowed() -> None:
    """Simultaneous frames are a data property, not an alignment error."""

    target = ClassificationTarget(
        labels=np.array([0, 1, 0]),
        time=np.array([0.0, 0.0, 0.1]),
    )

    assert target.n_samples == 3


def test_empty_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one sample"):
        ClassificationTarget(labels=np.array([], dtype=np.int64), time=np.array([]))


def test_two_dimensional_labels_are_rejected() -> None:
    with pytest.raises(ValidationError, match="1D array"):
        ClassificationTarget(labels=_labels().reshape(3, 2), time=np.zeros(3))


# --------------------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------------------


def test_target_is_unchanged_by_fitted_decoder() -> None:
    labels = _labels()
    time = _time()
    target = ClassificationTarget(labels=labels, time=time)

    labels[0] = 1
    time[0] = 99.0

    assert target.labels[0] == 2
    assert target.time[0] == 0.0
    for arr in (target.labels, target.time, target.classes, target.class_indices):
        assert not arr.flags.writeable


def test_target_is_immutable() -> None:
    target = ClassificationTarget(labels=_labels(), time=_time())

    with pytest.raises(AttributeError):
        target.labels = _labels()


def test_target_metadata_is_deep_frozen() -> None:
    target = ClassificationTarget(
        labels=_labels(),
        time=_time(),
        attrs={"session": "s1", "nested": {"block": 1}},
    )

    with pytest.raises(TypeError):
        target.attrs["session"] = "s2"
    with pytest.raises(TypeError):
        target.attrs["nested"]["block"] = 2
