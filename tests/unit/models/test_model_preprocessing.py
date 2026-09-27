#!/usr/bin/env python3

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import MinMaxScaler, StandardScaler, TemporalContext
from neurale.runtime import runtime_context


def _sample_matrix() -> np.ndarray:
    rng = np.random.default_rng(20260802)
    return rng.normal(loc=[1.0, -3.0, 10.0], scale=[0.5, 4.0, 0.01], size=(64, 3))


def _reference_standard(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    variance = ((X - mean) ** 2).mean(axis=0)
    scale = np.sqrt(variance)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, variance, scale


def _reference_min_max(
    X: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    data_range = data_max - data_min
    scale = (high - low) / np.where(data_range == 0.0, 1.0, data_range)
    return data_min, data_max, scale


# --------------------------------------------------------------------------------------
# StandardScaler
# --------------------------------------------------------------------------------------


def test_standard_scaler_matches_direct_numpy_scaling() -> None:
    X = _sample_matrix()
    mean, variance, scale = _reference_standard(X)

    scaler = StandardScaler().fit(X)

    assert scaler.n_samples_ == X.shape[0]
    assert scaler.n_features_in_ == X.shape[1]
    assert np.allclose(scaler.mean_, mean)
    assert np.allclose(scaler.var_, variance)
    assert np.allclose(scaler.scale_, scale)
    assert np.allclose(scaler.transform(X), (X - mean) / scale)


def test_standard_scaler_inverse_transform_round_trips() -> None:
    X = _sample_matrix()
    scaler = StandardScaler().fit(X)

    scaled = scaler.transform(X)

    assert np.allclose(scaler.inverse_transform(scaled), X)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaled.std(axis=0), 1.0)


def test_standard_scaler_fit_transform_matches_two_steps() -> None:
    X = _sample_matrix()

    combined = StandardScaler().fit_transform(X)

    assert np.allclose(combined, StandardScaler().fit(X).transform(X))


@pytest.mark.parametrize(
    ("with_mean", "with_std"),
    [(False, True), (True, False), (False, False)],
    ids=["no-mean", "no-std", "identity"],
)
def test_standard_scaler_honors_disabled_statistics(with_mean: bool, with_std: bool) -> None:
    X = _sample_matrix()
    mean, variance, scale = _reference_standard(X)
    expected_mean = mean if with_mean else np.zeros(X.shape[1])
    expected_scale = scale if with_std else np.ones(X.shape[1])

    scaler = StandardScaler(with_mean=with_mean, with_std=with_std).fit(X)

    # var_ describes the fit data, not the transform: disabling a statistic
    # neutralizes the corresponding term of the map without hiding what was seen.
    assert np.allclose(scaler.mean_, expected_mean)
    assert np.allclose(scaler.var_, variance)
    assert np.allclose(scaler.scale_, expected_scale)
    assert np.allclose(scaler.transform(X), (X - expected_mean) / expected_scale)
    assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)


def test_standard_scaler_uses_safe_scale_for_constant_features() -> None:
    X = np.column_stack([np.full(8, 4.25), np.arange(8.0)])

    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)

    assert scaler.var_[0] == 0.0
    assert scaler.scale_[0] == 1.0
    assert np.allclose(scaled[:, 0], 0.0)
    assert np.allclose(scaler.inverse_transform(scaled), X)


def test_standard_scaler_transform_does_not_update_statistics() -> None:
    X = _sample_matrix()
    scaler = StandardScaler().fit(X)
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    for _ in range(3):
        scaler.transform(X + 100.0)
        scaler.inverse_transform(X * -7.0)

    assert np.array_equal(scaler.mean_, mean_before)
    assert np.array_equal(scaler.scale_, scale_before)
    assert scaler.n_samples_ == X.shape[0]


def test_standard_scaler_fits_overflowing_magnitudes() -> None:
    # The column sums to 4e308, so a straight np.mean() would report inf. The
    # statistics are representable, so the fit must succeed and round-trip.
    huge = np.finfo(np.float64).max / 2.0
    X = np.column_stack([np.full(4, huge), np.arange(4.0)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler = StandardScaler().fit(X)

    assert scaler.mean_[0] == huge
    assert scaler.var_[0] == 0.0
    assert scaler.scale_[0] == 1.0
    assert np.array_equal(scaler.inverse_transform(scaler.transform(X)), X)


def test_standard_scaler_rejects_unrepresentable_variance() -> None:
    # Every element is finite and passes input validation, but the variance
    # overflows float64, which used to be stored as inf and turn the round trip
    # into NaN.
    limit = np.finfo(np.float64).max
    X = np.array([[-limit], [limit]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError, match="variance is not representable"):
            StandardScaler().fit(X)


def test_min_max_scaler_rejects_unrepresentable_data_range() -> None:
    limit = np.finfo(np.float64).max
    X = np.array([[-limit], [limit]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError, match="data_range is not representable"):
            MinMaxScaler().fit(X)


def test_min_max_scaler_rejects_unrepresentable_scale() -> None:
    # A subnormal data range is finite, but the multiplier it implies is not.
    X = np.array([[0.0], [np.finfo(np.float64).smallest_subnormal]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError, match="scale is not representable"):
            MinMaxScaler().fit(X)


def test_standard_scaler_fitted_attributes_are_read_only() -> None:
    scaler = StandardScaler().fit(_sample_matrix())

    for attribute in (scaler.mean_, scaler.var_, scaler.scale_):
        assert not attribute.flags.writeable
        with pytest.raises(ValueError):
            attribute[0] = 0.0


# --------------------------------------------------------------------------------------
# MinMaxScaler
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature_range",
    [(0.0, 1.0), (-1.0, 1.0), (2.5, 7.5)],
    ids=["unit", "symmetric", "shifted"],
)
def test_min_max_scaler_matches_direct_numpy_scaling(feature_range: tuple[float, float]) -> None:
    X = _sample_matrix()
    low, high = feature_range
    data_min, data_max, scale = _reference_min_max(X, low, high)

    scaler = MinMaxScaler(feature_range=feature_range).fit(X)
    scaled = scaler.transform(X)

    assert np.allclose(scaler.data_min_, data_min)
    assert np.allclose(scaler.data_max_, data_max)
    assert np.allclose(scaler.data_range_, data_max - data_min)
    assert np.allclose(scaler.scale_, scale)
    assert np.allclose(scaler.min_, low - data_min * scale)
    assert np.allclose(scaled, (X - data_min) * scale + low)
    assert np.allclose(scaled.min(axis=0), low)
    assert np.allclose(scaled.max(axis=0), high)
    assert np.allclose(scaler.inverse_transform(scaled), X)


def test_min_max_scaler_defaults_to_unit_range() -> None:
    scaler = MinMaxScaler()

    assert scaler.feature_range == (0.0, 1.0)


def test_min_max_scaler_uses_safe_scale_for_constant_features() -> None:
    X = np.column_stack([np.full(6, -2.0), np.linspace(0.0, 5.0, 6)])

    scaler = MinMaxScaler(feature_range=(-1.0, 3.0)).fit(X)
    scaled = scaler.transform(X)

    assert scaler.data_range_[0] == 0.0
    assert np.allclose(scaled[:, 0], -1.0)
    assert np.allclose(scaler.inverse_transform(scaled), X)


@pytest.mark.parametrize(
    "feature_range",
    [(1.0, 1.0), (1.0, 0.0), (0.0, np.inf), (-np.inf, 1.0), (0.0, np.nan)],
    ids=["degenerate", "reversed", "inf-high", "inf-low", "nan-high"],
)
def test_min_max_scaler_rejects_invalid_feature_range(feature_range) -> None:
    with pytest.raises(ValidationError, match="feature_range"):
        MinMaxScaler(feature_range=feature_range)


@pytest.mark.parametrize(
    "feature_range",
    [[0.0, 1.0], (0.0,), (0.0, 1.0, 2.0), "01", None],
    ids=["list", "short", "long", "string", "none"],
)
def test_min_max_scaler_requires_two_element_tuple(feature_range) -> None:
    with pytest.raises(ValidationError, match="feature_range must be a tuple of two numbers"):
        MinMaxScaler(feature_range=feature_range)


def test_min_max_scaler_transform_does_not_update_statistics() -> None:
    X = _sample_matrix()
    scaler = MinMaxScaler().fit(X)
    data_min_before = scaler.data_min_.copy()
    scale_before = scaler.scale_.copy()

    scaler.transform(X * 1000.0)
    scaler.inverse_transform(X - 1000.0)

    assert np.array_equal(scaler.data_min_, data_min_before)
    assert np.array_equal(scaler.scale_, scale_before)


def test_min_max_scaler_fitted_attributes_are_read_only() -> None:
    scaler = MinMaxScaler().fit(_sample_matrix())

    for attribute in (
        scaler.data_min_,
        scaler.data_max_,
        scaler.data_range_,
        scaler.scale_,
        scaler.min_,
    ):
        assert not attribute.flags.writeable
        with pytest.raises(ValueError):
            attribute[0] = 0.0


# --------------------------------------------------------------------------------------
# Shared scaler validation and device contract
# --------------------------------------------------------------------------------------


def _scaler_factories() -> list:
    return [StandardScaler, MinMaxScaler]


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize(
    "method",
    ["transform", "inverse_transform"],
    ids=["transform", "inverse"],
)
def test_scalers_reject_use_before_fit(factory, method: str) -> None:
    scaler = factory()

    with pytest.raises(ValidationError, match="is not fitted"):
        getattr(scaler, method)(_sample_matrix())

    for attribute in ("device_", "n_samples_", "n_features_in_", "scale_"):
        with pytest.raises(ValidationError, match="is not fitted"):
            getattr(scaler, attribute)


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (np.zeros((0, 3)), "must not be empty"),
        (np.zeros((3, 0)), "must not be empty"),
        (np.array([[np.nan, 1.0]]), "must contain finite values"),
        (np.array([[np.inf, 1.0]]), "must contain finite values"),
        (np.zeros(4), "must be a 2D array"),
        (np.zeros((2, 2, 2)), "must be a 2D array"),
        (np.array([[1.0 + 1.0j]]), "must contain real-valued data"),
        ([[1.0, 2.0]], "must be a numpy.ndarray"),
    ],
    ids=["no-rows", "no-columns", "nan", "inf", "1d", "3d", "complex", "not-array"],
)
def test_scalers_reject_invalid_fit_input(factory, invalid, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory().fit(invalid)


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize(
    "method",
    ["transform", "inverse_transform"],
    ids=["transform", "inverse"],
)
def test_scalers_reject_wrong_feature_count(factory, method: str) -> None:
    scaler = factory().fit(_sample_matrix())

    for wrong in (np.zeros((5, 2)), np.zeros((5, 4))):
        with pytest.raises(ValidationError, match="same number of features as fit data"):
            getattr(scaler, method)(wrong)


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize(
    "method",
    ["transform", "inverse_transform"],
    ids=["transform", "inverse"],
)
def test_scalers_reject_non_finite_transform_input(factory, method: str) -> None:
    scaler = factory().fit(_sample_matrix())
    invalid = np.zeros((2, 3))
    invalid[1, 2] = np.nan

    with pytest.raises(ValidationError, match="must contain finite values"):
        getattr(scaler, method)(invalid)


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
def test_scalers_report_cpu_device_after_fit(factory) -> None:
    scaler = factory().fit(_sample_matrix())

    assert scaler.device_ == "cpu"


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"], ids=["fit", "fit-transform"])
def test_scalers_reject_explicit_cuda_requests(factory, method: str) -> None:
    X = _sample_matrix()

    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"models\.preprocessing\..*no CUDA implementation",
        ):
            getattr(factory(), method)(X)


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize("requested", ["auto", "cpu", "cuda"], ids=["auto", "cpu", "cuda"])
def test_ambient_runtime_context_does_not_change_fitted_device(factory, requested: str) -> None:
    X = _sample_matrix()
    scaler = factory().fit(X)
    expected = scaler.transform(X)

    with runtime_context(device=requested):
        assert scaler.device_ == "cpu"
        assert np.array_equal(scaler.transform(X), expected)


# --------------------------------------------------------------------------------------
# Freezing a published fit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
def test_frozen_scaler_still_transforms_and_reports(factory) -> None:
    X = _sample_matrix()
    scaler = factory().fit(X)
    expected = scaler.transform(X)
    n_samples = scaler.n_samples_

    assert scaler.freeze() is scaler
    assert scaler.is_frozen

    assert np.array_equal(scaler.transform(X), expected)
    assert np.allclose(scaler.inverse_transform(expected), X)
    assert scaler.n_samples_ == n_samples
    assert scaler.device_ == "cpu"


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"], ids=["fit", "fit-transform"])
def test_frozen_scaler_cannot_be_refitted(factory, method: str) -> None:
    X = _sample_matrix()
    scaler = factory().fit(X).freeze()
    expected = scaler.transform(X)

    with pytest.raises(ValidationError, match="is frozen"):
        getattr(scaler, method)(X * 7.0 + 3.0)

    # The rejected refit changed nothing: the published map is still the one
    # that was published.
    assert np.array_equal(scaler.transform(X), expected)


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
def test_frozen_scaler_rejects_rebinding_fitted_state(factory) -> None:
    scaler = factory().fit(_sample_matrix()).freeze()

    with pytest.raises(AttributeError, match="is frozen"):
        scaler._scale = np.ones(3)
    with pytest.raises(AttributeError, match="is frozen"):
        del scaler._scale


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
def test_freezing_is_idempotent_and_survives_copy(factory) -> None:
    X = _sample_matrix()
    scaler = factory().fit(X).freeze()

    scaler.freeze()
    duplicate = copy.deepcopy(scaler)

    assert scaler.is_frozen
    assert duplicate.is_frozen
    assert np.array_equal(duplicate.transform(X), scaler.transform(X))


@pytest.mark.parametrize("factory", _scaler_factories(), ids=["standard", "min-max"])
def test_unfitted_scaler_cannot_be_frozen(factory) -> None:
    scaler = factory()

    with pytest.raises(ValidationError, match="is not fitted"):
        scaler.freeze()

    # Refused, not half-applied: the scaler is still a usable configuration.
    assert not scaler.is_frozen
    assert scaler.fit(_sample_matrix()) is scaler


# --------------------------------------------------------------------------------------
# TemporalContext
# --------------------------------------------------------------------------------------


def _reference_context(X: np.ndarray, left: int, right: int) -> np.ndarray:
    rows = [
        np.concatenate([X[idx + offset] for offset in range(left + right + 1)])
        for idx in range(X.shape[0] - left - right)
    ]
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize(
    ("left", "right"),
    [(0, 0), (1, 0), (3, 0), (0, 2), (2, 3)],
    ids=["identity", "one-past", "three-past", "two-future", "two-past-three-future"],
)
def test_temporal_context_matches_direct_stacking(left: int, right: int) -> None:
    X = _sample_matrix()
    context = TemporalContext(left, right)

    stacked = context.transform(X)

    assert stacked.shape == (X.shape[0] - left - right, X.shape[1] * (left + right + 1))
    assert np.array_equal(stacked, _reference_context(X, left, right))


def test_temporal_context_orders_frames_and_features() -> None:
    X = np.arange(20.0).reshape(10, 2)
    context = TemporalContext(left=2, right=1)

    stacked = context.transform(X)

    # Row 0 aligns with source row 2 and spans source rows 0..3.
    assert np.array_equal(stacked[0], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert np.array_equal(stacked[-1], X[6:10].reshape(-1))
    # The newest frame of a causal context is the aligned source row itself.
    causal = TemporalContext(left=3, right=0).transform(X)
    assert np.array_equal(causal[:, -X.shape[1] :], X[3:])


def test_temporal_context_identity_returns_equal_values() -> None:
    X = _sample_matrix()

    stacked = TemporalContext().transform(X)

    assert np.array_equal(stacked, X)
    assert stacked is not X


def test_temporal_context_output_is_writable_and_independent() -> None:
    X = _sample_matrix()

    stacked = TemporalContext(left=1, right=1).transform(X)
    stacked[0, 0] = 1234.5

    assert stacked.flags.writeable
    assert X[0, 0] != 1234.5


@pytest.mark.parametrize(
    ("left", "right", "n_samples", "expected"),
    [(0, 0, 5, 5), (2, 0, 5, 3), (0, 2, 5, 3), (2, 2, 5, 1), (2, 2, 4, 0), (1, 1, 0, 0)],
    ids=["identity", "past", "future", "exact", "too-short", "empty"],
)
def test_temporal_context_output_length_is_deterministic(
    left: int,
    right: int,
    n_samples: int,
    expected: int,
) -> None:
    assert TemporalContext(left, right).output_length(n_samples) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [(0, 0), (2, 0), (0, 2), (1, 3)],
    ids=["identity", "past", "future", "mixed"],
)
def test_temporal_context_trims_targets_to_transformed_rows(left: int, right: int) -> None:
    X = _sample_matrix()
    y = np.arange(X.shape[0], dtype=np.float64)
    y2d = np.column_stack([y, -y])
    context = TemporalContext(left, right)

    stacked = context.transform(X)
    trimmed = context.trim(y)
    trimmed2d = context.trim(y2d)

    assert trimmed.shape[0] == stacked.shape[0]
    assert np.array_equal(trimmed, y[left : X.shape[0] - right])
    assert np.array_equal(trimmed2d, y2d[left : X.shape[0] - right])
    # Each kept row's target is the target of the newest causal frame's aligned sample.
    assert trimmed[0] == float(left)


def test_temporal_context_boundaries_use_only_available_frames() -> None:
    X = np.arange(12.0).reshape(6, 2)
    context = TemporalContext(left=1, right=1)

    stacked = context.transform(X)

    assert stacked.shape[0] == 4
    assert np.array_equal(stacked[0], X[0:3].reshape(-1))
    assert np.array_equal(stacked[-1], X[3:6].reshape(-1))
    # Exactly one output row when the input has the minimum usable length.
    minimal = context.transform(X[:3])
    assert minimal.shape == (1, 6)
    assert np.array_equal(minimal[0], X[:3].reshape(-1))


@pytest.mark.parametrize(
    ("left", "right", "causal"),
    [(0, 0, True), (5, 0, True), (0, 1, False), (2, 2, False)],
    ids=["identity", "past-only", "one-future", "mixed"],
)
def test_temporal_context_reports_causality(left: int, right: int, causal: bool) -> None:
    context = TemporalContext(left, right)

    assert context.is_causal is causal
    assert context.n_frames == left + right + 1


def test_temporal_context_is_immutable() -> None:
    context = TemporalContext(left=2, right=1)

    with pytest.raises(AttributeError):
        context.left = 5
    with pytest.raises(AttributeError):
        context.right = 5
    assert context == TemporalContext(left=2, right=1)


@pytest.mark.parametrize(
    ("left", "right"),
    [(-1, 0), (0, -1), (1.5, 0), (0, 1.5), (True, 0), ("1", 0)],
    ids=["negative-left", "negative-right", "float-left", "float-right", "bool", "string"],
)
def test_temporal_context_rejects_invalid_widths(left, right) -> None:
    with pytest.raises(ValidationError, match=r"left|right"):
        TemporalContext(left, right)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (np.zeros((0, 3)), "must not be empty"),
        (np.zeros((3, 0)), "must not be empty"),
        (np.array([[np.nan, 1.0]]), "must contain finite values"),
        (np.zeros(4), "must be a 2D array"),
        ([[1.0, 2.0]], "must be a numpy.ndarray"),
    ],
    ids=["no-rows", "no-columns", "nan", "1d", "not-array"],
)
def test_temporal_context_rejects_invalid_transform_input(invalid, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TemporalContext(left=1, right=1).transform(invalid)


def test_temporal_context_rejects_input_shorter_than_window() -> None:
    context = TemporalContext(left=2, right=1)

    with pytest.raises(ValidationError, match=r"at least left \+ right \+ 1 = 4 samples"):
        context.transform(np.zeros((3, 2)))
    with pytest.raises(ValidationError, match=r"at least left \+ right \+ 1 = 4 samples"):
        context.trim(np.zeros(3))


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (np.zeros((0, 2)), "must not be empty"),
        (np.zeros((2, 2, 2)), "must be a 1D or 2D array"),
        ([0.0, 1.0], "must be a numpy.ndarray"),
    ],
    ids=["empty", "3d", "not-array"],
)
def test_temporal_context_rejects_invalid_trim_input(invalid, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TemporalContext(left=1, right=0).trim(invalid)


def test_temporal_context_trim_preserves_dtype_and_labels() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0])

    trimmed = TemporalContext(left=1, right=2).trim(labels)

    assert trimmed.dtype == labels.dtype
    assert np.array_equal(trimmed, labels[1:4])


def test_causal_transform_never_uses_future_frames() -> None:
    X = np.arange(24.0).reshape(12, 2)
    context = TemporalContext(left=3, right=0)

    stacked = context.transform(X)

    for i, row in enumerate(stacked):
        aligned = i + context.left
        assert np.array_equal(row, X[aligned - context.left : aligned + 1].reshape(-1))
        # Streaming a prefix of the signal yields exactly the same causal rows.
        prefix = context.transform(X[: aligned + 1])
        assert np.array_equal(prefix[-1], row)
