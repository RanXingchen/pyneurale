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

struct SymmetricEigenResult
{
    std::vector<double> values;
    std::vector<double> vectors;
};

struct GeneralizedEigenResult
{
    std::vector<double> values;
    std::vector<double> vectors;
};

struct SvdResult
{
    std::vector<double> singular_values;
    std::vector<double> right_singular_vectors;
};

/// Thin SVD including the left singular vectors.
struct FullSvdResult
{
    std::vector<double> singular_values;
    /// Row-major ``n_components x rows``; zero where the singular value is zero.
    std::vector<double> left_singular_vectors;
    /// Row-major ``n_components x cols``.
    std::vector<double> right_singular_vectors;
};

/// Thin SVD whose singular values below ``sqrt(eps) * s_max`` are zeroed.
///
/// The builtin path factorizes a Gram matrix, which resolves small singular
/// values only to about ``sqrt(eps) * s_max``; the zeroing keeps callers such as
/// PCA from reading a value the factorization cannot support. Callers that need
/// the small singular values themselves want ``accurate_thin_svd`` instead.
SvdResult thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols);

/// Thin SVD that resolves small singular values to high relative accuracy.
///
/// No Gram matrix is formed and no singular value is zeroed, so both providers
/// return the same quantities and a caller can apply one rank rule to either.
/// The builtin path uses one-sided Jacobi rotations, which cost more sweeps than
/// the Gram factorization but keep tiny singular values meaningful; the MKL path
/// returns the LAPACK values unmodified. Both singular vector sets are returned
/// so that a caller can solve from ``U^T y`` rather than from a cross product
/// that would square the condition number. Only the vectors of the nonzero
/// directions are defined.
FullSvdResult accurate_thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols);

SymmetricEigenResult symmetric_eigen(std::span<const double> matrix, std::size_t n);

GeneralizedEigenResult generalized_symmetric_eigen(std::span<const double> numerator,
                                                   std::span<const double> denominator,
                                                   std::size_t n);

void canonicalize_vector_signs(std::span<double> vectors, std::size_t n_vectors,
                               std::size_t vector_size);

std::vector<double> solve_spd(std::span<const double> matrix, std::span<const double> rhs,
                              std::size_t n, std::size_t nrhs);

std::vector<double> solve_linear(std::span<const double> matrix, std::span<const double> rhs,
                                 std::size_t n, std::size_t nrhs);

/// Cholesky solve that allocates nothing and throws nothing.
///
/// ``matrix`` is overwritten with its lower Cholesky factor and ``rhs`` with the
/// solution, both in the caller's own storage, so a prepared real-time step can
/// solve without touching the allocator. A matrix that is not numerically
/// positive definite returns ``false`` and leaves both operands partially
/// factorized: there is no fallback here, because the caller on a real-time
/// thread is the one that decides whether a second factorization is affordable
/// or the step is a fault. This is the same factorization and the same
/// substitutions the allocating ``solve_spd`` builtin path runs.
[[nodiscard]] bool solve_spd_in_place(std::span<double> matrix, std::span<double> rhs,
                                      std::size_t n, std::size_t nrhs) noexcept;

/// Partially pivoted LU solve that allocates nothing and throws nothing.
///
/// ``matrix`` and ``rhs`` are overwritten as in ``solve_spd_in_place``. A pivot
/// that falls to the matrix tolerance returns ``false`` rather than dividing by
/// it.
[[nodiscard]] bool solve_linear_in_place(std::span<double> matrix, std::span<double> rhs,
                                         std::size_t n, std::size_t nrhs) noexcept;

void gemm_nt(std::span<const double> left, std::span<const double> right, std::size_t rows,
             std::size_t cols, std::size_t inner, std::span<double> output);

void gemm_nt_unchecked(std::span<const double> left, std::span<const double> right,
                       std::size_t rows, std::size_t cols, std::size_t inner,
                       std::span<double> output);

void gemv_unchecked(std::span<const double> left, std::span<const double> right, std::size_t rows,
                    std::size_t inner, std::span<double> output, std::size_t output_stride);

} // namespace neurale::models
