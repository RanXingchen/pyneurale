#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import LinearRegression, Ridge
from neurale.runtime import runtime_context


def _design(n_samples: int = 64, n_features: int = 5, seed: int = 20260802) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features))


def _exact_system(
    n_samples: int = 64,
    n_features: int = 5,
    n_outputs: int = 3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=(n_outputs, n_features))
    intercept = rng.normal(size=n_outputs)
    return X, X @ coef.T + intercept, coef, intercept


def _reference_least_squares(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the same problem with ``numpy.linalg.lstsq``.

    The intercept is estimated by centering, matching the documented estimator
    semantics: the minimum-norm tie-break applies to the coefficients only, never
    to the intercept. For a full-rank design this agrees with a least-squares fit
    of the augmented ``[X, 1]`` design, which
    :func:`test_regression_matches_numpy_least_squares`
    checks separately.
    """

    targets = y.reshape(y.shape[0], -1)
    if not fit_intercept:
        solution, *_ = np.linalg.lstsq(X, targets, rcond=None)
        return solution.T, np.zeros(targets.shape[1])

    feature_mean = X.mean(axis=0)
    target_mean = targets.mean(axis=0)
    solution, *_ = np.linalg.lstsq(X - feature_mean, targets - target_mean, rcond=None)
    return solution.T, target_mean - solution.T @ feature_mean


def _reference_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the penalized normal equations directly, leaving the intercept free."""

    targets = y.reshape(y.shape[0], -1)
    if fit_intercept:
        feature_mean = X.mean(axis=0)
        target_mean = targets.mean(axis=0)
        design = X - feature_mean
        response = targets - target_mean
    else:
        feature_mean = np.zeros(X.shape[1])
        target_mean = np.zeros(targets.shape[1])
        design = X
        response = targets
    gram = design.T @ design + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(gram, design.T @ response).T
    return coef, target_mean - coef @ feature_mean


# --------------------------------------------------------------------------------------
# Exact and noisy recovery
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_outputs", [1, 2, 4], ids=["single", "two", "four"])
def test_regression_recovers_exact_multi_output_systems(n_outputs: int) -> None:
    X, y, coef, intercept = _exact_system(n_outputs=n_outputs)

    model = LinearRegression().fit(X, y)

    assert model.n_samples_ == X.shape[0]
    assert model.n_features_in_ == X.shape[1]
    assert model.n_outputs_ == n_outputs
    assert model.rank_ == X.shape[1]
    assert model.coef_.shape == (n_outputs, X.shape[1])
    assert model.intercept_.shape == (n_outputs,)
    assert np.allclose(model.coef_, coef)
    assert np.allclose(model.intercept_, intercept)
    assert np.allclose(model.predict(X), y)


def test_regression_accepts_one_dimensional_targets() -> None:
    X, y, coef, intercept = _exact_system(n_outputs=1)
    flat = y[:, 0]

    model = LinearRegression().fit(X, flat)

    assert model.n_outputs_ == 1
    assert model.coef_.shape == (1, X.shape[1])
    assert np.allclose(model.coef_[0], coef[0])
    assert np.allclose(model.intercept_, intercept)
    assert model.predict(X).shape == (X.shape[0], 1)
    assert np.allclose(model.predict(X)[:, 0], flat)


def test_regression_without_intercept_leaves_zero_offset() -> None:
    X, _y, coef, _ = _exact_system()
    centered_targets = X @ coef.T

    model = LinearRegression(fit_intercept=False).fit(X, centered_targets)

    assert np.allclose(model.intercept_, 0.0)
    assert np.allclose(model.coef_, coef)


def test_regression_matches_numpy_least_squares() -> None:
    rng = np.random.default_rng(11)
    X, y, _, _ = _exact_system(n_outputs=2)
    noisy = y + rng.normal(scale=0.5, size=y.shape)

    for fit_intercept in (True, False):
        model = LinearRegression(fit_intercept=fit_intercept).fit(X, noisy)
        coef, intercept = _reference_least_squares(X, noisy, fit_intercept=fit_intercept)

        assert np.allclose(model.coef_, coef)
        assert np.allclose(model.intercept_, intercept)

    # The design has full column rank, so the fit also equals a plain least-squares
    # solve of the augmented design that carries the intercept as a constant column.
    augmented, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(X.shape[0])]), noisy, rcond=None)
    fitted = LinearRegression().fit(X, noisy)
    assert np.allclose(fitted.coef_, augmented[:-1].T)
    assert np.allclose(fitted.intercept_, augmented[-1])


def test_regression_predict_matches_affine_definition() -> None:
    rng = np.random.default_rng(3)
    X, y, _, _ = _exact_system(n_outputs=3)
    model = LinearRegression().fit(X, y + rng.normal(scale=0.1, size=y.shape))
    unseen = rng.normal(size=(9, X.shape[1]))

    predicted = model.predict(unseen)

    assert np.allclose(predicted, unseen @ model.coef_.T + model.intercept_)


# --------------------------------------------------------------------------------------
# Rank-deficient behavior
# --------------------------------------------------------------------------------------


def _rank_deficient_design() -> np.ndarray:
    base = _design(n_features=3)
    # The fourth column duplicates the first, so the design has rank three.
    return np.column_stack([base, base[:, 0]])


def test_regression_returns_minimum_norm_for_duplicate_columns() -> None:
    X = _rank_deficient_design()
    y = X @ np.array([1.0, 2.0, 3.0, 0.0]) + 4.0

    model = LinearRegression().fit(X, y)
    coef, intercept = _reference_least_squares(X, y, fit_intercept=True)

    assert model.rank_ == 3
    assert model.singular_values_.shape == (X.shape[1],)
    assert np.allclose(model.coef_, coef)
    assert np.allclose(model.intercept_, intercept)
    # The duplicated pair shares the coefficient equally, which is the minimum-norm split.
    assert np.isclose(model.coef_[0, 0], model.coef_[0, 3])
    assert np.allclose(model.predict(X)[:, 0], y)


def test_regression_handles_more_features_than_samples() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 10))
    y = rng.normal(size=(6, 2))

    model = LinearRegression().fit(X, y)
    coef, intercept = _reference_least_squares(X, y, fit_intercept=True)

    assert model.rank_ == 5  # centering removes one degree of freedom
    assert model.singular_values_.shape == (min(X.shape),)
    assert np.allclose(model.predict(X), y)
    assert np.allclose(model.coef_, coef)
    assert np.allclose(model.intercept_, intercept)


def test_regression_reports_zero_column_rank_deficient() -> None:
    X = np.column_stack([_design(n_features=2), np.zeros(64)])
    y = X @ np.array([2.0, -1.0, 0.0])

    model = LinearRegression().fit(X, y)

    assert model.rank_ == 2
    assert model.coef_[0, 2] == 0.0
    assert np.isclose(model.singular_values_[-1], 0.0)


def test_singular_values_come_from_centered_design() -> None:
    X = _design()

    fitted = LinearRegression().fit(X, X @ np.arange(1.0, X.shape[1] + 1.0))
    uncentered = LinearRegression(fit_intercept=False).fit(X, X @ np.arange(1.0, X.shape[1] + 1.0))

    assert np.allclose(fitted.singular_values_, np.linalg.svd(X - X.mean(axis=0))[1])
    assert np.allclose(uncentered.singular_values_, np.linalg.svd(X)[1])


# --------------------------------------------------------------------------------------
# Numerical robustness
# --------------------------------------------------------------------------------------


def _near_singular_design(spread: float) -> np.ndarray:
    """Diagonal design whose second direction is ``spread`` times the first."""

    return np.diag([1.0, spread])


def _spectral_design(spread: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a near-singular design from its spectral decomposition.

    A diagonal design cannot expose cancellation, because each feature is already
    a singular direction. Rotating the same spectrum mixes the two directions
    into every entry, so recovering the small one requires the solver not to
    cancel it away.
    """

    first = np.array([np.cos(angle), np.sin(angle)])
    second = np.array([-np.sin(angle), np.cos(angle)])
    design = np.outer(first, first) + spread * np.outer(second, second)
    return design, first, second


def _spectral_ridge_solution(
    spread: float,
    first: np.ndarray,
    second: np.ndarray,
    beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Closed-form ridge solution of ``_spectral_design`` for ``y = X @ beta``.

    Written in the spectral basis, the penalized solution shrinks each direction
    by ``s ** 2 / (s ** 2 + alpha)``, so the expectation needs no solve of its own
    and cannot inherit the solver's conditioning.
    """

    return (first @ beta) / (1.0 + alpha) * first + (second @ beta) * spread**2 / (
        spread**2 + alpha
    ) * second


@pytest.mark.parametrize(
    ("spread", "expected_rank"),
    [(1e-8, 2), (1e-20, 1)],
    ids=["above-rcond", "below-rcond"],
)
def test_least_squares_truncates_at_rank_tol(
    spread: float,
    expected_rank: int,
) -> None:
    X = _near_singular_design(spread)
    y = X @ np.array([1.0, 1.0])

    model = LinearRegression(fit_intercept=False).fit(X, y)

    # Above the tolerance the direction is inverted; below it the minimum-norm
    # solution drops it instead of amplifying noise by 1 / s.
    assert model.rank_ == expected_rank
    assert np.allclose(model.singular_values_, [1.0, spread])
    assert np.isclose(model.coef_[0, 1], 1.0 if expected_rank == 2 else 0.0)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(1e-18, 0.99009901), (1e-16, 0.5), (1e-14, 0.00990099)],
    ids=["alpha-below-s2", "alpha-equals-s2", "alpha-above-s2"],
)
def test_ridge_inverts_small_singular_value_with_penalty(
    alpha: float,
    expected: float,
) -> None:
    # The whole answer lives in the s = 1e-8 direction, and it swings from 0.99
    # to 0.0099 as alpha crosses s ** 2 = 1e-16. A factorization that resolves
    # small singular values only to sqrt(eps) * s_max cannot see this direction
    # at all and returns zero for every alpha.
    X = _near_singular_design(1e-8)
    y = X @ np.array([0.0, 1.0])

    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)

    assert model.rank_ == 2
    assert np.isclose(model.coef_[0, 0], 0.0)
    assert np.isclose(model.coef_[0, 1], expected)
    assert np.allclose(model.coef_, _reference_ridge(X, y, alpha, fit_intercept=False)[0])


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(1e-45, 0.99999), (1e-40, 0.5), (0.0, 0.0)],
    ids=["penalized-below-s2", "penalized-at-s2", "unpenalized"],
)
def test_only_unpenalized_solve_truncates_below_rank_tol(
    alpha: float,
    expected: float,
) -> None:
    # s = 1e-20 is below max(n, p) * eps * s_max, so rank_ reports one direction
    # for every alpha. A penalized solve still uses the second one, because the
    # penalty -- not the rank rule -- is what bounds 1 / (s ** 2 + alpha) there.
    X = _near_singular_design(1e-20)
    y = X @ np.array([0.0, 1.0])

    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)

    assert model.rank_ == 1
    assert np.isclose(model.coef_[0, 1], expected)


@pytest.mark.parametrize("angle", [np.pi / 4.0, 0.3], ids=["symmetric", "tilted"])
def test_least_squares_recovers_rotated_near_singular_system(angle: float) -> None:
    # Every entry of this design is within 1e-8 of every other, so forming
    # X.T @ y cancels the small direction away entirely and returns the
    # projection onto the dominant one instead of the exact solution.
    spread = 1e-8
    X, _first, _second = _spectral_design(spread, angle)
    beta = np.array([1.0, 2.0])

    model = LinearRegression(fit_intercept=False).fit(X, beta @ X.T)

    assert model.rank_ == 2
    assert np.allclose(model.singular_values_, [1.0, spread], rtol=1e-12)
    assert np.allclose(model.coef_[0], beta, rtol=1e-6)


@pytest.mark.parametrize("angle", [np.pi / 4.0, 0.3], ids=["symmetric", "tilted"])
@pytest.mark.parametrize("alpha", [1e-16, 1e-12, 1.0], ids=["below-s2", "above-s2", "unit"])
def test_ridge_recovers_rotated_near_singular_system(angle: float, alpha: float) -> None:
    spread = 1e-8
    X, first, second = _spectral_design(spread, angle)
    beta = np.array([1.0, 2.0])
    expected = _spectral_ridge_solution(spread, first, second, beta, alpha)

    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, beta @ X.T)

    assert np.allclose(model.coef_[0], expected, rtol=1e-6, atol=1e-9)


def test_ridge_reproduces_documented_rotated_example() -> None:
    # The spectral solution of the symmetric case at alpha = s ** 2 is exactly
    # halfway between the least-squares solution [1, 2] and the projection onto
    # the dominant direction [1.5, 1.5].
    spread = 1e-8
    X, _first, _second = _spectral_design(spread, np.pi / 4.0)

    model = Ridge(alpha=spread**2, fit_intercept=False).fit(X, np.array([1.0, 2.0]) @ X.T)

    assert np.allclose(model.coef_[0], [1.25, 1.75], rtol=1e-6)


def test_coefficients_match_numpy_on_near_singular_design() -> None:
    spread = 1e-8
    X, _first, _second = _spectral_design(spread, 0.7)
    y = np.array([1.0, 2.0]) @ X.T

    model = LinearRegression(fit_intercept=False).fit(X, y)
    reference, *_ = np.linalg.lstsq(X, y, rcond=None)

    assert np.allclose(model.coef_[0], reference, rtol=1e-6)


@pytest.mark.parametrize(
    "spread",
    [1e-4, 1e-8, 1e-12, 1e-16],
    ids=["mild", "near-singular", "very-small", "below-eps"],
)
def test_singular_values_match_lapack_factorization(spread: float) -> None:
    """Pin the solver's own factorization to the one MKL would hand it.

    The MKL provider returns LAPACK singular values unmodified, and NumPy calls
    the same routine, so this is the runnable half of a builtin/MKL parity check:
    it fails if the builtin path resolves small singular values differently from
    the LAPACK ones the MKL build would use to make the same rank decision. The
    two agree to the absolute accuracy LAPACK itself offers, ``eps * s_max``,
    which is also the scale the shared rank rule is stated in.
    """

    rng = np.random.default_rng(11)
    basis = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    X = basis @ np.diag([1.0, 0.5, spread]) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    y = X @ np.arange(1.0, 4.0)

    model = LinearRegression(fit_intercept=False).fit(X, y)
    reference = np.linalg.svd(X, compute_uv=False)

    assert np.allclose(
        model.singular_values_,
        reference,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps * reference[0],
    )
    assert model.rank_ == int(np.linalg.matrix_rank(X))


def test_fit_survives_overflowing_gram_matrix() -> None:
    # X.T @ X is 2e308 and would report the design as rank zero, but the singular
    # value and the exact solution are both representable.
    X = np.array([[1e154], [-1e154]])
    y = np.array([1.0, -1.0])

    model = LinearRegression(fit_intercept=False).fit(X, y)

    assert model.rank_ == 1
    assert np.isclose(model.singular_values_[0], np.sqrt(2.0) * 1e154)
    assert np.isclose(model.coef_[0, 0], 1e-154, rtol=1e-12)


def test_fit_survives_underflowing_gram_matrix() -> None:
    X = np.array([[1e-170, 0.0], [0.0, 2e-170], [1e-170, 2e-170]])
    y = X @ np.array([3.0, -5.0])

    model = LinearRegression(fit_intercept=False).fit(X, y)

    assert model.rank_ == 2
    assert np.allclose(model.coef_[0], [3.0, -5.0], rtol=1e-8)


def test_fit_survives_overflowing_target_sum() -> None:
    # The target column sums to 1.5 * float64 max, so a mean taken by summing
    # first would center the targets on inf and report nan parameters.
    offset = np.finfo(np.float64).max / 2.0
    X = np.arange(3.0)[:, None]
    y = np.full(3, offset)

    model = LinearRegression().fit(X, y)

    assert np.isfinite(model.coef_).all()
    assert model.coef_[0, 0] == 0.0
    assert model.intercept_[0] == offset


def test_fit_recovers_exact_solution_at_extreme_scales() -> None:
    rng = np.random.default_rng(3)
    coef = rng.normal(size=(2, 4))
    X = rng.normal(size=(32, 4)) * 1e-160
    y = X @ coef.T * 1e150

    model = LinearRegression(fit_intercept=False).fit(X, y)

    assert np.allclose(model.coef_, coef * 1e150, rtol=1e-10)


def test_fit_preserves_small_target_entries() -> None:
    # A well-conditioned design with an exactly representable answer. Rescaling
    # the targets by their largest entry would flush the second one to zero, so
    # the solver may only rescale as far as its own working range requires.
    X = np.eye(2)
    y = np.array([1e224, 1e-100])

    model = LinearRegression(fit_intercept=False).fit(X, y)

    assert np.allclose(model.coef_[0], y, rtol=1e-12)


def test_fit_rejects_design_out_of_range() -> None:
    # 1e500 of dynamic range: bringing the large entry into the factorization's
    # working range underflows the small one, so there is no scaling that keeps
    # both. That fails rather than silently fitting a design with a lost column.
    X = np.diag([1e300, 1e-200])
    y = np.array([1.0, 1.0])

    with pytest.raises(ValidationError, match="spans more magnitudes"):
        LinearRegression(fit_intercept=False).fit(X, y)


def test_fit_rejects_unrepresentable_solution() -> None:
    # Both operands are finite and the design is well conditioned, but the
    # coefficients it implies are about 1e308 / 1e-100.
    limit = np.finfo(np.float64).max
    X = np.array([[1e-100], [2e-100]])
    y = np.array([limit / 2.0, limit])

    with pytest.raises(ValidationError, match="not representable"):
        LinearRegression(fit_intercept=False).fit(X, y)


def test_ridge_saturates_to_least_squares_at_tiny_penalty() -> None:
    X = _design(n_samples=16, n_features=3) * 1e30
    y = X @ np.array([1.0, -2.0, 0.5])

    penalized = Ridge(alpha=1e-300, fit_intercept=False).fit(X, y)
    unpenalized = LinearRegression(fit_intercept=False).fit(X, y)

    assert np.allclose(penalized.coef_, unpenalized.coef_)


# --------------------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1e-6, 0.1, 1.0, 25.0], ids=["tiny", "small", "unit", "large"])
@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no-intercept"])
def test_ridge_matches_direct_penalized_solve(alpha: float, fit_intercept: bool) -> None:
    rng = np.random.default_rng(13)
    X, y, _, _ = _exact_system(n_outputs=3)
    noisy = y + rng.normal(scale=0.25, size=y.shape)

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, noisy)
    coef, intercept = _reference_ridge(X, noisy, alpha, fit_intercept=fit_intercept)

    assert np.allclose(model.coef_, coef)
    assert np.allclose(model.intercept_, intercept)


def test_ridge_defaults_to_unit_alpha() -> None:
    assert Ridge().alpha == 1.0


def test_ridge_with_zero_alpha_matches_least_squares() -> None:
    X, y, _, _ = _exact_system(n_outputs=2)
    rng = np.random.default_rng(17)
    noisy = y + rng.normal(scale=0.4, size=y.shape)

    ridge = Ridge(alpha=0.0).fit(X, noisy)
    ols = LinearRegression().fit(X, noisy)

    assert np.array_equal(ridge.coef_, ols.coef_)
    assert np.array_equal(ridge.intercept_, ols.intercept_)
    assert ridge.rank_ == ols.rank_


def test_ridge_with_zero_alpha_keeps_minimum_norm() -> None:
    X = _rank_deficient_design()
    y = X @ np.array([1.0, 2.0, 3.0, 0.0])

    ridge = Ridge(alpha=0.0).fit(X, y)
    ols = LinearRegression().fit(X, y)

    assert ridge.rank_ == 3
    assert np.array_equal(ridge.coef_, ols.coef_)


def test_ridge_shrinks_coefficients_as_alpha_grows() -> None:
    rng = np.random.default_rng(19)
    X, y, _, _ = _exact_system(n_outputs=1)
    noisy = y + rng.normal(scale=0.3, size=y.shape)

    norms = [
        np.linalg.norm(Ridge(alpha=alpha).fit(X, noisy).coef_) for alpha in (0.0, 1.0, 10.0, 100.0)
    ]

    assert norms == sorted(norms, reverse=True)


def test_ridge_does_not_penalize_intercept() -> None:
    X, y, _, _ = _exact_system(n_outputs=2)
    shifted = y + 1000.0

    model = Ridge(alpha=50.0).fit(X, shifted)
    baseline = Ridge(alpha=50.0).fit(X, y)

    # A constant target shift moves the intercept only; the penalized coefficients
    # are untouched because the intercept is estimated by centering.
    assert np.allclose(model.coef_, baseline.coef_)
    assert np.allclose(model.intercept_ - baseline.intercept_, 1000.0)


def test_ridge_stabilizes_rank_deficient_design() -> None:
    X = _rank_deficient_design()
    y = X @ np.array([1.0, 2.0, 3.0, 0.0])

    model = Ridge(alpha=1.0).fit(X, y)
    coef, intercept = _reference_ridge(X, y, 1.0, fit_intercept=True)

    assert model.rank_ == 3
    assert np.allclose(model.coef_, coef, atol=1e-10)
    assert np.allclose(model.intercept_, intercept, atol=1e-10)


@pytest.mark.parametrize(
    "alpha",
    [-1.0, -1e-12, np.nan, np.inf, -np.inf],
    ids=["negative", "tiny-negative", "nan", "inf", "-inf"],
)
def test_ridge_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        Ridge(alpha=alpha)


@pytest.mark.parametrize("alpha", [None, [1.0], object()], ids=["none", "list", "object"])
def test_ridge_rejects_non_numeric_alpha(alpha) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        Ridge(alpha=alpha)


# --------------------------------------------------------------------------------------
# Shared validation, buffers, and device contract
# --------------------------------------------------------------------------------------


def _estimators() -> list:
    return [LinearRegression, lambda **kwargs: Ridge(alpha=0.5, **kwargs)]


_ESTIMATOR_IDS = ["linear", "ridge"]


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_estimators_reject_invalid_fit_intercept(factory) -> None:
    with pytest.raises(ValidationError, match="fit_intercept must be a bool"):
        factory(fit_intercept=1)


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_estimators_reject_use_before_fit(factory) -> None:
    model = factory()

    with pytest.raises(ValidationError, match="is not fitted"):
        model.predict(_design())
    for attribute in (
        "device_",
        "n_samples_",
        "n_features_in_",
        "n_outputs_",
        "rank_",
        "singular_values_",
        "coef_",
        "intercept_",
    ):
        with pytest.raises(ValidationError, match="is not fitted"):
            getattr(model, attribute)


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
@pytest.mark.parametrize(
    ("X", "message"),
    [
        (np.zeros((0, 3)), "must not be empty"),
        (np.zeros((3, 0)), "must not be empty"),
        (np.full((4, 2), np.nan), "must contain finite values"),
        (np.zeros(4), "must be a 2D array"),
        ([[1.0, 2.0]], "must be a numpy.ndarray"),
    ],
    ids=["no-rows", "no-columns", "nan", "1d", "not-array"],
)
def test_estimators_reject_invalid_designs(factory, X, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory().fit(X, np.zeros(max(np.shape(X)[0] if np.ndim(X) else 1, 1)))


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.zeros((64, 0)), "must not be empty"),
        (np.full(64, np.inf), "must contain finite values"),
        (np.zeros((64, 2, 2)), "must be a 1D or 2D array"),
        (np.zeros(63), "same number of samples"),
        (np.zeros((65, 2)), "same number of samples"),
        ([0.0] * 64, "must be a numpy.ndarray"),
    ],
    ids=["no-columns", "inf", "3d", "short", "long", "not-array"],
)
def test_estimators_reject_invalid_targets(factory, y, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory().fit(_design(), y)


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_estimators_reject_wrong_feature_count_at_predict(factory) -> None:
    X = _design()
    model = factory().fit(X, np.zeros(X.shape[0]))

    for wrong in (np.zeros((3, X.shape[1] - 1)), np.zeros((3, X.shape[1] + 1))):
        with pytest.raises(ValidationError, match="same number of features as fit data"):
            model.predict(wrong)


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_estimators_reject_invalid_predict_input(factory) -> None:
    X = _design()
    model = factory().fit(X, np.zeros(X.shape[0]))

    with pytest.raises(ValidationError, match="must not be empty"):
        model.predict(np.zeros((0, X.shape[1])))
    with pytest.raises(ValidationError, match="must be a 2D array"):
        model.predict(np.zeros(X.shape[1]))
    with pytest.raises(ValidationError, match=r"must be a numpy\.ndarray"):
        model.predict([[0.0] * X.shape[1]])


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_predict_propagates_non_finite_inputs(factory) -> None:
    # Inference shares the package-wide contract of validate_inference_array:
    # non-finite samples are not rejected, they flow through to the prediction.
    X = _design()
    model = factory().fit(X, np.zeros(X.shape[0]))
    unseen = np.zeros((2, X.shape[1]))
    unseen[1, 0] = np.nan

    predicted = model.predict(unseen)

    assert np.isfinite(predicted[0]).all()
    assert np.isnan(predicted[1]).all()


def test_predict_writes_into_supplied_output_buffer() -> None:
    X, y, _, _ = _exact_system(n_outputs=3)
    model = LinearRegression().fit(X, y)
    buffer = np.zeros((X.shape[0], 3))

    returned = model.predict(X, out=buffer)

    assert returned is buffer
    assert np.allclose(buffer, y)


@pytest.mark.parametrize(
    "buffer",
    [
        np.zeros((64, 2)),
        np.zeros((63, 3)),
        np.zeros(64),
        np.zeros((64, 3), dtype=np.float32),
        np.zeros((64, 3), order="F"),
    ],
    ids=["wrong-outputs", "wrong-rows", "1d", "wrong-dtype", "fortran-order"],
)
def test_predict_rejects_invalid_output_buffers(buffer) -> None:
    X, y, _, _ = _exact_system(n_outputs=3)
    model = LinearRegression().fit(X, y)

    with pytest.raises(ValidationError):
        model.predict(X, out=buffer)


@pytest.mark.parametrize("overlap", ["identical", "view", "partial"])
def test_predict_rejects_output_buffer_overlapping_input(overlap: str) -> None:
    # The kernel writes each output row while still reading later input rows, so
    # an overlapping buffer silently mixes old and new values instead of failing.
    X, y, _, _ = _exact_system(n_samples=8, n_features=3, n_outputs=3)
    model = LinearRegression().fit(X, y)
    base = np.zeros((9, 3))
    base[:8] = X
    design = base[:8]
    buffers = {
        "identical": design,
        "view": design[:, :],
        "partial": base[1:],
    }

    with pytest.raises(ValidationError, match="overlap"):
        model.predict(design, out=buffers[overlap])


def test_predict_into_buffer_matches_allocating_call() -> None:
    X, y, _, _ = _exact_system(n_samples=8, n_features=3, n_outputs=3)
    model = LinearRegression().fit(X, y)
    buffer = np.zeros_like(X)

    model.predict(X, out=buffer)

    assert np.array_equal(buffer, model.predict(X))


def test_predict_rejects_read_only_output_buffer() -> None:
    X, y, _, _ = _exact_system(n_outputs=3)
    model = LinearRegression().fit(X, y)
    buffer = np.zeros((X.shape[0], 3))
    buffer.setflags(write=False)

    with pytest.raises(ValidationError, match="writable"):
        model.predict(X, out=buffer)


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_fitted_parameters_are_read_only(factory) -> None:
    X, y, _, _ = _exact_system(n_outputs=2)
    model = factory().fit(X, y)

    for attribute in (model.coef_, model.intercept_, model.singular_values_):
        assert not attribute.flags.writeable
        with pytest.raises(ValueError):
            attribute.reshape(-1)[0] = 0.0


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_predict_does_not_mutate_fitted_parameters(factory) -> None:
    X, y, _, _ = _exact_system(n_outputs=2)
    model = factory().fit(X, y)
    coef_before = np.array(model.coef_)
    intercept_before = np.array(model.intercept_)

    for _ in range(3):
        model.predict(X * 3.0 + 1.0)

    assert np.array_equal(model.coef_, coef_before)
    assert np.array_equal(model.intercept_, intercept_before)


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_estimators_report_cpu_device_after_fit(factory) -> None:
    X = _design()

    assert factory().fit(X, np.zeros(X.shape[0])).device_ == "cpu"


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
def test_estimators_reject_explicit_cuda_before_native_dispatch(factory) -> None:
    X = _design()

    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"models\.linear_model.*no CUDA implementation",
        ):
            factory().fit(X, np.zeros(X.shape[0]))


@pytest.mark.parametrize("factory", _estimators(), ids=_ESTIMATOR_IDS)
@pytest.mark.parametrize("requested", ["auto", "cpu", "cuda"], ids=["auto", "cpu", "cuda"])
def test_ambient_runtime_context_does_not_change_fitted_device(factory, requested: str) -> None:
    X, y, _, _ = _exact_system(n_outputs=2)
    model = factory().fit(X, y)
    expected = model.predict(X)

    with runtime_context(device=requested):
        assert model.device_ == "cpu"
        assert np.array_equal(model.predict(X), expected)
