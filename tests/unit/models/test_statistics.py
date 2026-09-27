#!/usr/bin/env python3

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.models.statistics import (
    DipStatistic,
    DipTestResult,
    LedoitWolfCov,
    dip_statistic,
    dip_test,
    empirical_cov,
    rayleigh_test,
)
from neurale.runtime import runtime_context


def test_empirical_cov_matches_legacy_reference() -> None:
    X = np.array(
        [
            [-0.03207008, 0.58788027, 0.44092057, -1.74731715, -0.91512972],
            [1.41947410, 1.83519003, 0.85557546, 3.48225733, 2.05689851],
            [0.07503051, -0.45059105, -1.67781366, 0.56455264, -0.98276245],
            [0.52474022, 0.64649282, 2.11410146, -1.10211797, 0.87034958],
            [0.26365594, 0.08061475, -0.26249384, 0.58429384, -0.53454199],
        ]
    )
    expected = np.array(
        [
            [0.27063676, 0.33951521, 0.29054217, 0.76300988, 0.59366981],
            [0.33951521, 0.57669117, 0.62742306, 0.69632916, 0.78641221],
            [0.29054217, 0.62742306, 1.56949129, -0.34911699, 0.96800838],
            [0.76300988, 0.69632916, -0.34911699, 3.28382954, 1.35179485],
            [0.59366981, 0.78641221, 0.96800838, 1.35179485, 1.40567825],
        ]
    )

    assert np.allclose(empirical_cov(X), expected)


def test_empirical_cov_handles_uncentered_and_single_feature_inputs() -> None:
    X = np.array([[1.0], [2.0], [3.0]])

    assert np.allclose(empirical_cov(X), [[2.0 / 3.0]])
    assert np.allclose(empirical_cov(X, center=False), [[14.0 / 3.0]])
    assert np.allclose(
        empirical_cov(np.array([[2.0, 3.0]])),
        [[0.0, 0.0], [0.0, 0.0]],
    )


def test_empirical_cov_rejects_complex_input() -> None:
    with pytest.raises(ValidationError, match="real-valued"):
        empirical_cov(np.array([[1.0 + 1.0j, 2.0]]))


def test_ledoit_wolf_cov_matches_legacy_reference() -> None:
    X = np.array(
        [
            [-0.03207008, 0.58788027, 0.44092057, -1.74731715, -0.91512972],
            [1.41947410, 1.83519003, 0.85557546, 3.48225733, 2.05689851],
            [0.07503051, -0.45059105, -1.67781366, 0.56455264, -0.98276245],
            [0.52474022, 0.64649282, 2.11410146, -1.10211797, 0.87034958],
            [0.26365594, 0.08061475, -0.26249384, 0.58429384, -0.53454199],
        ]
    )
    expected_cov = np.array(
        [
            [0.96293893, 0.13523808, 0.11573080, 0.30392745, 0.23647472],
            [0.13523808, 1.08484867, 0.24991956, 0.27736671, 0.31324923],
            [0.11573080, 0.24991956, 1.48030778, -0.13906273, 0.38558389],
            [0.30392745, 0.27736671, -0.13906273, 2.16317501, 0.53845641],
            [0.23647472, 0.31324923, 0.38558389, 0.53845641, 1.41505662],
        ]
    )

    estimator = LedoitWolfCov().fit(X)

    assert np.allclose(estimator.covariance_, expected_cov)
    assert np.allclose(estimator.location_, np.mean(X, axis=0))
    assert estimator.shrinkage_ == pytest.approx(0.6016729845008)
    assert estimator.device_ == "cpu"
    with pytest.raises(AttributeError):
        estimator.device_ = "cuda"
    with runtime_context(device="cuda"):
        assert estimator.device_ == "cpu"
        assert np.allclose(estimator.covariance_, expected_cov)


def test_ledoit_wolf_shrunk_validates_shrinkage() -> None:
    covariance = np.eye(2)

    assert np.allclose(LedoitWolfCov.shrunk(covariance, 0.5), covariance)
    with pytest.raises(ValidationError, match="shrinkage"):
        LedoitWolfCov.shrunk(covariance, 1.5)


def test_ledoit_wolf_fit_does_not_copy_when_center_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    estimator = LedoitWolfCov(center=False)

    def fake_ledoit_wolf(centered: np.ndarray) -> tuple[np.ndarray, float]:
        assert np.shares_memory(centered, X)
        return np.eye(2), 0.0

    monkeypatch.setattr(estimator, "_ledoit_wolf", fake_ledoit_wolf)

    estimator.fit(X)

    assert np.allclose(estimator.covariance_, np.eye(2))


def test_ledoit_wolf_clips_shrinkage_when_covariance_delta_is_zero() -> None:
    estimator = LedoitWolfCov().fit(np.ones((4, 3)))

    assert estimator.shrinkage_ == 0.0
    assert np.allclose(estimator.covariance_, np.zeros((3, 3)))


def test_rayleigh_test_matches_legacy_reference_and_axis_shape() -> None:
    uniform_short = np.array(
        [
            5.50371100310229,
            3.45673425035102,
            3.91112631444826,
            3.68851066216938,
            1.30528332137997,
            1.89278651625386,
            2.95889866421354,
            1.44819982172012,
            5.30494860098620,
            1.22374012257095,
        ]
    )
    nonuniform_short = np.array(
        [
            0.0,
            1.35100807537622,
            1.17175445918672,
            4.07021614858383,
            5.75053609151296,
            2.92417598002763,
            4.71581997018994,
            4.29622172738443,
            1.78417640824356,
            4.06333118023541,
            2.25560601449391,
            5.82298845394547,
            3.38697110510583,
            6.01130508052712,
            2.76965663319211,
            4.36525739657257,
            3.33943343350203,
            0.258039076274217,
            1.89348778693319,
            4.40155454846279,
            3.35186663017059,
            1.67864148404351,
            2.45301887965822,
            3.83063542926874,
            1.87636387164171,
            4.97160133972592,
            2.43047902082102,
            6.28318530717959,
            1.73976984368099,
            3.75147868915786,
            1.04590944059539,
            2.30005924768618,
        ]
    )
    stacked = np.vstack([uniform_short, uniform_short])
    uniform_long = np.random.default_rng(123).uniform(0.0, 2.0 * np.pi, 512)
    long_result = rayleigh_test(uniform_long)

    assert rayleigh_test(uniform_short) == pytest.approx(0.662328124105056)
    assert rayleigh_test(nonuniform_short) == pytest.approx(0.345863528546573)
    assert 0.0 <= long_result <= 1.0
    assert np.allclose(
        rayleigh_test(stacked, axis=1),
        [0.662328124105056, 0.662328124105056],
    )


def test_rayleigh_test_rejects_empty_or_nonfinite_data() -> None:
    with pytest.raises(ValidationError, match="empty"):
        rayleigh_test(np.array([]))
    with pytest.raises(ValidationError, match="finite"):
        rayleigh_test(np.array([0.0, np.nan]))
    with pytest.raises(ValidationError, match="at least two"):
        rayleigh_test(np.array([0.0]))
    with pytest.raises(ValidationError, match="real-valued"):
        rayleigh_test(np.array([0.0, 1.0 + 0.0j]))


def test_dip_statistic_detects_unimodal_and_multimodal_samples() -> None:
    uniform = np.linspace(0.0, 1.0, 64)
    multimodal = np.r_[
        np.linspace(-2.0, -1.0, 32),
        np.linspace(1.0, 2.0, 32),
    ]

    uniform_result = dip_statistic(uniform)
    result = dip_statistic(multimodal)

    assert isinstance(result, DipStatistic)
    assert uniform_result.dip >= 0.0
    assert result.dip > uniform_result.dip
    assert result.lower <= result.upper
    assert not hasattr(result, "gcm")
    assert not hasattr(result, "lcm")


def test_dip_statistic_handles_small_and_degenerate_samples() -> None:
    result = dip_statistic(np.arange(4.0))
    degenerate = dip_statistic(np.ones(4))

    assert result.dip == pytest.approx(0.125)
    assert degenerate == DipStatistic(dip=0.0, lower=1.0, upper=1.0)
    with pytest.raises(ValidationError, match="at least four"):
        dip_statistic(np.array([0.0, 0.5, 1.0]))


def test_dip_test_uses_explicit_random_state() -> None:
    multimodal = np.r_[
        np.linspace(-2.0, -1.0, 32),
        np.linspace(1.0, 2.0, 32),
    ]

    result1 = dip_test(multimodal, n_boot=32, random_state=123)
    result2 = dip_test(multimodal, n_boot=32, random_state=123)

    assert isinstance(result1, DipTestResult)
    assert result1 == result2
    assert result1.statistic == pytest.approx(dip_statistic(multimodal).dip)
    assert result1.n_boot == 32
    assert 0.0 <= result1.p_value <= 1.0


def test_dip_test_returns_one_for_degenerate_sample() -> None:
    result = dip_test(np.ones(4), n_boot=8, random_state=123)

    assert result == DipTestResult(statistic=0.0, p_value=1.0, n_boot=8)


def test_dip_test_uses_right_tail_monte_carlo_correction() -> None:
    result = dip_test(np.arange(4.0), n_boot=8, random_state=123)

    assert result.statistic == pytest.approx(0.125)
    assert result.p_value == pytest.approx(1.0)


def test_statistics_reject_invalid_inputs() -> None:
    with pytest.raises(ValidationError, match="2D array"):
        empirical_cov(np.array([1.0, 2.0]))
    with pytest.raises(ValidationError, match="numeric"):
        empirical_cov(np.array([["x"]]))
    with pytest.raises(ValidationError, match="at least four"):
        dip_statistic(np.array([1.0]))
    with pytest.raises(ValidationError, match="at least four"):
        dip_test(np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValidationError, match="real-valued"):
        dip_statistic(np.array([0.0, 1.0 + 0.0j]))
