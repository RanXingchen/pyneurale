/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace neurale::models
{

/// Fitted parameters and solver diagnostics of one linear or ridge fit.
struct LinearFitResult
{
    std::size_t n_samples{};
    std::size_t n_features{};
    std::size_t n_outputs{};
    /// Number of singular values of the (optionally centered) design above tolerance.
    std::size_t rank{};
    /// Row-major ``n_outputs x n_features`` coefficients.
    std::vector<double> coef;
    /// Per-output intercept; all zero when the fit did not estimate one.
    std::vector<double> intercept;
    /// Descending singular values of the (optionally centered) design.
    std::vector<double> singular_values;
};

/// Immutable fitted state of a multi-output affine model.
struct LinearModelState
{
    std::size_t n_features{};
    std::size_t n_outputs{};
    std::vector<double> coef;
    std::vector<double> intercept;
};

/// Multi-output affine predictor ``Y = X * coef^T + intercept``.
class LinearModel
{
  public:
    explicit LinearModel(LinearModelState state);

    [[nodiscard]] std::size_t n_features() const noexcept;
    [[nodiscard]] std::size_t n_outputs() const noexcept;
    [[nodiscard]] const std::vector<double>& coef() const noexcept;
    [[nodiscard]] const std::vector<double>& intercept() const noexcept;

    /// Write ``n_samples x n_outputs`` predictions for a row-major design.
    ///
    /// ``X`` and ``output`` must not overlap: the accumulation writes an
    /// output row while later X rows are still to be read, so an
    /// overlapping pair would be computed from half-overwritten data. An
    /// overlap throws rather than producing that result.
    ///
    /// Under MKL the product runs with the local thread count pinned to one.
    /// That is what makes this callable from the real-time data plane at all:
    /// an unpinned BLAS call enters MKL's shared pool and synchronizes it on
    /// every invocation, and the measured tail of that is worse than the
    /// pinned kernel's whole runtime. It is also what makes the result
    /// reproducible -- this target links libgomp while static MKL brings
    /// libiomp5, and a threaded kernel with two OpenMP runtimes in the process
    /// can answer differently for the same input.
    ///
    /// Allocation-free and bounded once the model is constructed. It still
    /// throws for a shape it cannot evaluate, which is a programming error
    /// rather than an ordinary result; a real-time caller sizes its spans from
    /// the prepared geometry so that check cannot fire.
    void predict(std::span<const double> X, std::size_t n_samples, std::span<double> output) const;

  private:
    LinearModelState state_;
};

/// Fit a multi-output linear model with optional L2 penalty on the coefficients.
///
/// The design is factorized once with ``accurate_thin_svd``, whose small
/// singular values are meaningful under either provider, and the coefficients
/// are read off the projection of the targets onto the left singular vectors:
/// ``coef = V diag(s / (s^2 + alpha)) U^T y``. Nothing forms ``X^T X`` or
/// ``X^T y``, which would square the condition number and cancel away the
/// near-singular directions the factorization just resolved. The filter itself
/// is evaluated as ``1 / (s + alpha / s)``, which agrees with the definition
/// wherever both are representable and stays correct where ``s^2`` is not.
///
/// ``rank`` counts the singular values above
/// ``max(n_samples, n_features) * eps * s_max`` under both providers and
/// describes the design alone; it does not depend on ``alpha``.
///
/// An operand is rescaled only if the factorization cannot take it as given:
/// the sweep sums squares of design entries and the projections sum target
/// entries, so each has a band it must fall in, and an operand already inside
/// its band is used untouched. An operand outside is divided by a power of two,
/// which is exact for every value that stays normal. An operand that spans more
/// magnitudes than double precision can carry -- one whose smallest nonzero
/// entries underflow when its largest are brought into range -- throws, as does
/// a fit whose result is not representable. Neither is reported as a solution.
///
/// Truncation does depend on ``alpha``. An unpenalized fit drops the directions
/// at or below that tolerance, which is what makes its result the minimum-norm
/// least-squares solution. A penalized fit keeps every direction the design
/// spans, because ``1 / (s^2 + alpha)`` is finite wherever ``s`` is and a
/// direction below the rank tolerance still dominates the solution when
/// ``alpha`` is smaller than ``s^2``. The two agree as ``alpha`` approaches
/// zero, since a penalty too small to represent against the design scale
/// reproduces the unpenalized solve exactly.
///
/// When ``fit_intercept`` is set, the design and targets are centered first, so
/// the intercept is never penalized.
LinearFitResult fit_linear_model(std::span<const double> X, std::size_t n_samples,
                                 std::size_t n_features, std::span<const double> y,
                                 std::size_t n_outputs, double alpha, bool fit_intercept);

} // namespace neurale::models
