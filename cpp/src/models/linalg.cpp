/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "linalg.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef NEURALE_MODELS_WITH_MKL
#include <mkl.h>
#endif

namespace neurale::models
{
namespace
{

using detail::matrix_index;

constexpr double kTolerance = 100.0 * std::numeric_limits<double>::epsilon();

void require_square_size(std::span<const double> matrix, std::size_t n)
{
    if (n == 0 || matrix.size() != n * n)
    {
        throw std::invalid_argument("matrix shape is invalid");
    }
}

double matrix_scale(std::span<const double> values)
{
    double scale = 0.0;
    for (double value : values)
    {
        scale = std::max(scale, std::abs(value));
    }
    return scale;
}

double relative_tolerance(std::span<const double> values)
{
    return kTolerance * matrix_scale(values);
}

double vector_norm(std::span<const double> values)
{
    double squared_norm = 0.0;
    for (double value : values)
    {
        squared_norm += value * value;
    }
    return std::sqrt(squared_norm);
}

void fill_orthonormal_complement(std::span<double> vectors, std::span<const double> singular_values,
                                 std::size_t rank, std::size_t n_features, double singular_tol,
                                 double basis_tol)
{
    for (std::size_t component = 0; component < rank; ++component)
    {
        if (singular_values[component] > singular_tol)
        {
            continue;
        }

        bool filled = false;
        for (std::size_t candidate = 0; candidate < n_features && !filled; ++candidate)
        {
            auto basis = std::vector<double>(n_features, 0.0);
            basis[candidate] = 1.0;
            for (std::size_t previous = 0; previous < component; ++previous)
            {
                double projection = 0.0;
                for (std::size_t feature = 0; feature < n_features; ++feature)
                {
                    projection +=
                        basis[feature] * vectors[matrix_index(previous, feature, n_features)];
                }
                for (std::size_t feature = 0; feature < n_features; ++feature)
                {
                    basis[feature] -=
                        projection * vectors[matrix_index(previous, feature, n_features)];
                }
            }

            const double norm = vector_norm(basis);
            if (norm <= basis_tol)
            {
                continue;
            }
            for (std::size_t feature = 0; feature < n_features; ++feature)
            {
                vectors[matrix_index(component, feature, n_features)] = basis[feature] / norm;
            }
            filled = true;
        }
        if (!filled)
        {
            throw std::runtime_error("failed to build PCA null-space basis");
        }
    }
}

double singular_value_tolerance(std::span<const double> singular_values, std::size_t rows,
                                std::size_t cols)
{
    if (singular_values.empty())
    {
        return 0.0;
    }
    const double dim = static_cast<double>(std::max(rows, cols));
    return std::sqrt(std::numeric_limits<double>::epsilon() * dim) * singular_values.front();
}

double basis_norm_tolerance(std::size_t n_features)
{
    return kTolerance * static_cast<double>(n_features);
}

void zero_unreliable_singular_values(std::span<double> singular_values, std::size_t rows,
                                     std::size_t cols)
{
    const double tol = singular_value_tolerance(singular_values, rows, cols);
    for (double& singular : singular_values)
    {
        if (singular <= tol)
        {
            singular = 0.0;
        }
    }
}

SymmetricEigenResult builtin_symmetric_eigen(std::span<const double> matrix, std::size_t n)
{
    auto a = std::vector<double>(matrix.begin(), matrix.end());
    auto vectors = std::vector<double>(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        vectors[matrix_index(i, i, n)] = 1.0;
    }

    if (n == 1)
    {
        return {{a[0]}, {1.0}};
    }

    const double tol = relative_tolerance(matrix);
    bool converged = false;
    const auto max_iterations = 50 * n * n;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration)
    {
        std::size_t p = 0;
        std::size_t q = 1;
        double largest = 0.0;
        for (std::size_t row = 0; row < n; ++row)
        {
            for (std::size_t col = row + 1; col < n; ++col)
            {
                const double value = std::abs(a[matrix_index(row, col, n)]);
                if (value > largest)
                {
                    largest = value;
                    p = row;
                    q = col;
                }
            }
        }
        if (largest <= tol)
        {
            converged = true;
            break;
        }

        const double app = a[matrix_index(p, p, n)];
        const double aqq = a[matrix_index(q, q, n)];
        const double apq = a[matrix_index(p, q, n)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double sign = tau < 0.0 ? -1.0 : 1.0;
        const double t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (std::size_t k = 0; k < n; ++k)
        {
            if (k != p && k != q)
            {
                const double akp = a[matrix_index(k, p, n)];
                const double akq = a[matrix_index(k, q, n)];
                a[matrix_index(k, p, n)] = c * akp - s * akq;
                a[matrix_index(p, k, n)] = a[matrix_index(k, p, n)];
                a[matrix_index(k, q, n)] = s * akp + c * akq;
                a[matrix_index(q, k, n)] = a[matrix_index(k, q, n)];
            }
        }
        a[matrix_index(p, p, n)] = app - t * apq;
        a[matrix_index(q, q, n)] = aqq + t * apq;
        a[matrix_index(p, q, n)] = 0.0;
        a[matrix_index(q, p, n)] = 0.0;

        for (std::size_t k = 0; k < n; ++k)
        {
            const double vkp = vectors[matrix_index(k, p, n)];
            const double vkq = vectors[matrix_index(k, q, n)];
            vectors[matrix_index(k, p, n)] = c * vkp - s * vkq;
            vectors[matrix_index(k, q, n)] = s * vkp + c * vkq;
        }
    }
    if (!converged)
    {
        throw std::runtime_error("symmetric eigensolver failed to converge");
    }

    auto order = std::vector<std::size_t>(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right)
              { return a[matrix_index(left, left, n)] > a[matrix_index(right, right, n)]; });

    SymmetricEigenResult result;
    result.values.resize(n);
    result.vectors.resize(n * n);
    for (std::size_t row = 0; row < n; ++row)
    {
        const std::size_t source = order[row];
        result.values[row] = a[matrix_index(source, source, n)];
        for (std::size_t col = 0; col < n; ++col)
        {
            result.vectors[matrix_index(row, col, n)] = vectors[matrix_index(col, source, n)];
        }
    }
    return result;
}

SvdResult builtin_thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols)
{
    const std::size_t rank = std::min(rows, cols);
    if (cols <= rows)
    {
        auto gram = std::vector<double>(cols * cols, 0.0);
        for (std::size_t row = 0; row < cols; ++row)
        {
            for (std::size_t col = row; col < cols; ++col)
            {
                double value = 0.0;
                for (std::size_t sample = 0; sample < rows; ++sample)
                {
                    value += matrix[matrix_index(sample, row, cols)] *
                             matrix[matrix_index(sample, col, cols)];
                }
                gram[matrix_index(row, col, cols)] = value;
                gram[matrix_index(col, row, cols)] = value;
            }
        }
        auto eigen = builtin_symmetric_eigen(gram, cols);
        SvdResult result;
        result.singular_values.resize(rank);
        result.right_singular_vectors.assign(eigen.vectors.begin(),
                                             eigen.vectors.begin() +
                                                 static_cast<std::ptrdiff_t>(rank * cols));
        for (std::size_t i = 0; i < rank; ++i)
        {
            result.singular_values[i] = std::sqrt(std::max(eigen.values[i], 0.0));
        }
        zero_unreliable_singular_values(result.singular_values, rows, cols);
        return result;
    }

    auto gram = std::vector<double>(rows * rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = row; col < rows; ++col)
        {
            double value = 0.0;
            for (std::size_t feature = 0; feature < cols; ++feature)
            {
                value += matrix[matrix_index(row, feature, cols)] *
                         matrix[matrix_index(col, feature, cols)];
            }
            gram[matrix_index(row, col, rows)] = value;
            gram[matrix_index(col, row, rows)] = value;
        }
    }
    auto eigen = builtin_symmetric_eigen(gram, rows);
    SvdResult result;
    result.singular_values.resize(rank);
    result.right_singular_vectors.assign(rank * cols, 0.0);
    for (std::size_t component = 0; component < rank; ++component)
    {
        result.singular_values[component] = std::sqrt(std::max(eigen.values[component], 0.0));
    }

    zero_unreliable_singular_values(result.singular_values, rows, cols);
    for (std::size_t component = 0; component < rank; ++component)
    {
        const double singular = result.singular_values[component];
        if (singular == 0.0)
        {
            continue;
        }
        for (std::size_t feature = 0; feature < cols; ++feature)
        {
            double value = 0.0;
            for (std::size_t sample = 0; sample < rows; ++sample)
            {
                value += eigen.vectors[matrix_index(component, sample, rows)] *
                         matrix[matrix_index(sample, feature, cols)];
            }
            result.right_singular_vectors[matrix_index(component, feature, cols)] =
                value / singular;
        }
    }
    fill_orthonormal_complement(result.right_singular_vectors, result.singular_values, rank, cols,
                                0.0, basis_norm_tolerance(cols));
    return result;
}

/// Orthogonalize the columns of ``a`` in place by plane rotations.
///
/// The accumulated rotations are appended to ``rotations``, which starts as the
/// identity, so ``a_initial = a_final * rotations^T``. Every quantity the sweep
/// looks at is a dot product of two columns, never an entry of a Gram matrix
/// that was formed and stored, which is what preserves the small singular
/// values.
void one_sided_jacobi(std::vector<double>& a, std::size_t rows, std::size_t cols,
                      std::vector<double>& rotations)
{
    constexpr std::size_t kMaxSweeps = 60;
    const double eps = std::numeric_limits<double>::epsilon();

    rotations.assign(cols * cols, 0.0);
    for (std::size_t i = 0; i < cols; ++i)
    {
        rotations[matrix_index(i, i, cols)] = 1.0;
    }
    if (cols < 2)
    {
        return;
    }

    for (std::size_t sweep = 0; sweep < kMaxSweeps; ++sweep)
    {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < cols; ++p)
        {
            for (std::size_t q = p + 1; q < cols; ++q)
            {
                double app = 0.0;
                double aqq = 0.0;
                double apq = 0.0;
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const double left = a[matrix_index(row, p, cols)];
                    const double right = a[matrix_index(row, q, cols)];
                    app += left * left;
                    aqq += right * right;
                    apq += left * right;
                }
                // Compare against the geometric mean of the column norms so that
                // the test stays scale free and cannot overflow on its own.
                if (apq == 0.0 || std::abs(apq) <= eps * std::sqrt(app) * std::sqrt(aqq))
                {
                    continue;
                }

                const double zeta = (aqq - app) / (2.0 * apq);
                const double sign = zeta < 0.0 ? -1.0 : 1.0;
                const double t = sign / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;

                for (std::size_t row = 0; row < rows; ++row)
                {
                    const double left = a[matrix_index(row, p, cols)];
                    const double right = a[matrix_index(row, q, cols)];
                    a[matrix_index(row, p, cols)] = c * left - s * right;
                    a[matrix_index(row, q, cols)] = s * left + c * right;
                }
                for (std::size_t row = 0; row < cols; ++row)
                {
                    const double left = rotations[matrix_index(row, p, cols)];
                    const double right = rotations[matrix_index(row, q, cols)];
                    rotations[matrix_index(row, p, cols)] = c * left - s * right;
                    rotations[matrix_index(row, q, cols)] = s * left + c * right;
                }
                rotated = true;
            }
        }
        if (!rotated)
        {
            return;
        }
    }
    throw std::runtime_error("one-sided Jacobi SVD failed to converge");
}

std::vector<std::size_t> descending_order(std::span<const double> values)
{
    auto order = std::vector<std::size_t>(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t left, std::size_t right) { return values[left] > values[right]; });
    return order;
}

double column_norm(std::span<const double> matrix, std::size_t rows, std::size_t cols,
                   std::size_t col)
{
    double squared_norm = 0.0;
    for (std::size_t row = 0; row < rows; ++row)
    {
        const double value = matrix[matrix_index(row, col, cols)];
        squared_norm += value * value;
    }
    return std::sqrt(squared_norm);
}

/// The two singular vector sets of a tall ``m x k`` matrix, ordered by
/// descending singular value.
struct JacobiFactors
{
    std::vector<double> singular_values;
    /// Row-major ``k x m``: the orthogonalized columns, normalized. Zero where
    /// the singular value is, because there is nothing to normalize by.
    std::vector<double> column_vectors;
    /// Row-major ``k x k``: the columns of the accumulated rotation, which stay
    /// defined even where the singular value is zero.
    std::vector<double> rotation_vectors;
};

/// Read the thin SVD of a tall ``m x k`` matrix off a one-sided Jacobi sweep.
///
/// ``work`` is consumed. Whether the two vector sets are the left and right ones
/// or the other way round depends on the orientation the caller sweeps in, which
/// is why they are named after what produced them rather than after a side.
JacobiFactors jacobi_factors(std::vector<double> work, std::size_t m, std::size_t k)
{
    auto rotations = std::vector<double>();
    one_sided_jacobi(work, m, k, rotations);

    auto norms = std::vector<double>(k, 0.0);
    for (std::size_t col = 0; col < k; ++col)
    {
        norms[col] = column_norm(work, m, k, col);
    }

    JacobiFactors factors;
    factors.singular_values.assign(k, 0.0);
    factors.column_vectors.assign(k * m, 0.0);
    factors.rotation_vectors.assign(k * k, 0.0);
    const auto order = descending_order(norms);
    for (std::size_t component = 0; component < k; ++component)
    {
        const std::size_t source = order[component];
        const double singular = norms[source];
        factors.singular_values[component] = singular;
        for (std::size_t i = 0; i < k; ++i)
        {
            factors.rotation_vectors[matrix_index(component, i, k)] =
                rotations[matrix_index(i, source, k)];
        }
        if (singular == 0.0)
        {
            continue;
        }
        for (std::size_t i = 0; i < m; ++i)
        {
            factors.column_vectors[matrix_index(component, i, m)] =
                work[matrix_index(i, source, k)] / singular;
        }
    }
    return factors;
}

FullSvdResult jacobi_thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols)
{
    if (cols <= rows)
    {
        auto factors = jacobi_factors({matrix.begin(), matrix.end()}, rows, cols);
        return {std::move(factors.singular_values), std::move(factors.column_vectors),
                std::move(factors.rotation_vectors)};
    }

    // Fewer samples than features: sweep the transpose instead, so the wide case
    // never materializes a features-by-features basis. Transposing the input
    // swaps the roles of the two vector sets the sweep produces.
    auto transposed = std::vector<double>(cols * rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            transposed[matrix_index(col, row, rows)] = matrix[matrix_index(row, col, cols)];
        }
    }
    auto factors = jacobi_factors(std::move(transposed), cols, rows);
    return {std::move(factors.singular_values), std::move(factors.rotation_vectors),
            std::move(factors.column_vectors)};
}

std::vector<double> builtin_solve_linear(std::span<const double> matrix,
                                         std::span<const double> rhs, std::size_t n,
                                         std::size_t nrhs)
{
    auto a = std::vector<double>(matrix.begin(), matrix.end());
    auto solution = std::vector<double>(rhs.begin(), rhs.end());
    if (!solve_linear_in_place(a, solution, n, nrhs))
    {
        throw std::runtime_error("linear solve failed: singular matrix");
    }
    return solution;
}

std::vector<double> builtin_solve_spd(std::span<const double> matrix, std::span<const double> rhs,
                                      std::size_t n, std::size_t nrhs)
{
    auto factor = std::vector<double>(matrix.begin(), matrix.end());
    auto solution = std::vector<double>(rhs.begin(), rhs.end());
    if (solve_spd_in_place(factor, solution, n, nrhs))
    {
        return solution;
    }
    // The factorization stopped on a non-positive pivot, so the matrix is not
    // usable as a Cholesky factor; the general solve still is, and it is run on
    // the untouched original rather than on the partial factor above.
    return builtin_solve_linear(matrix, rhs, n, nrhs);
}

#ifdef NEURALE_MODELS_WITH_MKL
/// Transpose the LAPACK ``rows x rank`` left factor into ``rank x rows`` rows.
std::vector<double> transposed_left_vectors(std::span<const double> lapack_left, std::size_t rows,
                                            std::size_t rank)
{
    auto left = std::vector<double>(rank * rows, 0.0);
    for (std::size_t sample = 0; sample < rows; ++sample)
    {
        for (std::size_t component = 0; component < rank; ++component)
        {
            left[matrix_index(component, sample, rows)] =
                lapack_left[matrix_index(sample, component, rank)];
        }
    }
    return left;
}

FullSvdResult lapack_thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols)
{
    auto values = std::vector<double>(matrix.begin(), matrix.end());
    const auto rank = std::min(rows, cols);
    auto singular_values = std::vector<double>(rank);
    auto right_singular_vectors = std::vector<double>(rank * cols);
    auto left_singular_vectors = std::vector<double>(rows * rank);

    auto status = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', static_cast<MKL_INT>(rows),
                                 static_cast<MKL_INT>(cols), values.data(),
                                 static_cast<MKL_INT>(cols), singular_values.data(),
                                 left_singular_vectors.data(), static_cast<MKL_INT>(rank),
                                 right_singular_vectors.data(), static_cast<MKL_INT>(cols));
    if (status != 0)
    {
        values.assign(matrix.begin(), matrix.end());
        std::fill(singular_values.begin(), singular_values.end(), 0.0);
        std::fill(right_singular_vectors.begin(), right_singular_vectors.end(), 0.0);
        std::fill(left_singular_vectors.begin(), left_singular_vectors.end(), 0.0);
        auto superb = std::vector<double>(rank > 0 ? rank - 1 : 0);
        status = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'S', 'S', static_cast<MKL_INT>(rows), static_cast<MKL_INT>(cols),
            values.data(), static_cast<MKL_INT>(cols), singular_values.data(),
            left_singular_vectors.data(), static_cast<MKL_INT>(rank), right_singular_vectors.data(),
            static_cast<MKL_INT>(cols), superb.data());
        if (status != 0)
        {
            throw std::runtime_error("SVD failed to converge");
        }
    }
    return {std::move(singular_values), transposed_left_vectors(left_singular_vectors, rows, rank),
            std::move(right_singular_vectors)};
}
#endif

void require_matrix_shape(std::span<const double> matrix, std::size_t rows, std::size_t cols)
{
    if (rows == 0 || cols == 0 || matrix.size() != rows * cols)
    {
        throw std::invalid_argument("matrix shape is invalid");
    }
}

} // namespace

SvdResult thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols)
{
    require_matrix_shape(matrix, rows, cols);
#ifdef NEURALE_MODELS_WITH_MKL
    auto full = lapack_thin_svd(matrix, rows, cols);
    return {std::move(full.singular_values), std::move(full.right_singular_vectors)};
#else
    return builtin_thin_svd(matrix, rows, cols);
#endif
}

FullSvdResult accurate_thin_svd(std::span<const double> matrix, std::size_t rows, std::size_t cols)
{
    require_matrix_shape(matrix, rows, cols);
#ifdef NEURALE_MODELS_WITH_MKL
    return lapack_thin_svd(matrix, rows, cols);
#else
    return jacobi_thin_svd(matrix, rows, cols);
#endif
}

SymmetricEigenResult symmetric_eigen(std::span<const double> matrix, std::size_t n)
{
    require_square_size(matrix, n);
#ifdef NEURALE_MODELS_WITH_MKL
    auto a = std::vector<double>(matrix.begin(), matrix.end());
    auto values = std::vector<double>(n);
    const auto status = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', static_cast<MKL_INT>(n), a.data(),
                                      static_cast<MKL_INT>(n), values.data());
    if (status != 0)
    {
        throw std::runtime_error("symmetric eigensolver failed");
    }
    SymmetricEigenResult result;
    result.values.resize(n);
    result.vectors.resize(n * n);
    for (std::size_t row = 0; row < n; ++row)
    {
        const std::size_t source = n - row - 1;
        result.values[row] = values[source];
        for (std::size_t col = 0; col < n; ++col)
        {
            result.vectors[matrix_index(row, col, n)] = a[matrix_index(col, source, n)];
        }
    }
    return result;
#else
    return builtin_symmetric_eigen(matrix, n);
#endif
}

GeneralizedEigenResult generalized_symmetric_eigen(std::span<const double> numerator,
                                                   std::span<const double> denominator,
                                                   std::size_t n)
{
    require_square_size(numerator, n);
    require_square_size(denominator, n);

    const auto denominator_eigen = symmetric_eigen(denominator, n);
    const double threshold = denominator_eigen.values.front() *
                             std::numeric_limits<double>::epsilon() * static_cast<double>(n);
    const auto rank = static_cast<std::size_t>(
        std::count_if(denominator_eigen.values.begin(), denominator_eigen.values.end(),
                      [threshold](double value) { return value > threshold; }));
    if (rank == 0)
    {
        throw std::runtime_error("generalized eigensolver denominator has zero rank");
    }

    auto whitening = std::vector<double>(rank * n);
    for (std::size_t row = 0; row < rank; ++row)
    {
        const double scale = 1.0 / std::sqrt(denominator_eigen.values[row]);
        for (std::size_t col = 0; col < n; ++col)
        {
            whitening[matrix_index(row, col, n)] =
                denominator_eigen.vectors[matrix_index(row, col, n)] * scale;
        }
    }

    auto workspace = std::vector<double>(rank * n);
    gemm_nt(whitening, numerator, rank, n, n, workspace);
    auto reduced = std::vector<double>(rank * rank);
    gemm_nt(workspace, whitening, rank, rank, n, reduced);
    const auto reduced_eigen = symmetric_eigen(reduced, rank);

    GeneralizedEigenResult result;
    result.values = reduced_eigen.values;
    result.vectors.resize(rank * n);
    for (std::size_t row = 0; row < rank; ++row)
    {
        for (std::size_t col = 0; col < n; ++col)
        {
            double value = 0.0;
            for (std::size_t inner = 0; inner < rank; ++inner)
            {
                value += reduced_eigen.vectors[matrix_index(row, inner, rank)] *
                         whitening[matrix_index(inner, col, n)];
            }
            result.vectors[matrix_index(row, col, n)] = value;
        }
    }
    return result;
}

void canonicalize_vector_signs(std::span<double> vectors, std::size_t n_vectors,
                               std::size_t vector_size)
{
    if (vectors.size() != n_vectors * vector_size)
    {
        throw std::invalid_argument("vector matrix shape is invalid");
    }
    for (std::size_t row = 0; row < n_vectors; ++row)
    {
        std::size_t pivot = 0;
        double largest = 0.0;
        for (std::size_t col = 0; col < vector_size; ++col)
        {
            const double value = std::abs(vectors[matrix_index(row, col, vector_size)]);
            if (value > largest)
            {
                largest = value;
                pivot = col;
            }
        }
        if (vectors[matrix_index(row, pivot, vector_size)] < 0.0)
        {
            for (std::size_t col = 0; col < vector_size; ++col)
            {
                vectors[matrix_index(row, col, vector_size)] =
                    -vectors[matrix_index(row, col, vector_size)];
            }
        }
    }
}

std::vector<double> solve_spd(std::span<const double> matrix, std::span<const double> rhs,
                              std::size_t n, std::size_t nrhs)
{
    require_square_size(matrix, n);
    if (nrhs == 0 || rhs.size() != n * nrhs)
    {
        throw std::invalid_argument("right-hand side shape is invalid");
    }
#ifdef NEURALE_MODELS_WITH_MKL
    auto a = std::vector<double>(matrix.begin(), matrix.end());
    auto solution = std::vector<double>(rhs.begin(), rhs.end());
    const auto status = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'L', static_cast<MKL_INT>(n),
                                      static_cast<MKL_INT>(nrhs), a.data(), static_cast<MKL_INT>(n),
                                      solution.data(), static_cast<MKL_INT>(nrhs));
    if (status == 0)
    {
        return solution;
    }
    return solve_linear(matrix, rhs, n, nrhs);
#else
    return builtin_solve_spd(matrix, rhs, n, nrhs);
#endif
}

std::vector<double> solve_linear(std::span<const double> matrix, std::span<const double> rhs,
                                 std::size_t n, std::size_t nrhs)
{
    require_square_size(matrix, n);
    if (nrhs == 0 || rhs.size() != n * nrhs)
    {
        throw std::invalid_argument("right-hand side shape is invalid");
    }
#ifdef NEURALE_MODELS_WITH_MKL
    auto a = std::vector<double>(matrix.begin(), matrix.end());
    auto solution = std::vector<double>(rhs.begin(), rhs.end());
    auto pivots = std::vector<MKL_INT>(n);
    const auto status = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<MKL_INT>(n),
                                      static_cast<MKL_INT>(nrhs), a.data(), static_cast<MKL_INT>(n),
                                      pivots.data(), solution.data(), static_cast<MKL_INT>(nrhs));
    if (status != 0)
    {
        throw std::runtime_error("linear solve failed");
    }
    return solution;
#else
    return builtin_solve_linear(matrix, rhs, n, nrhs);
#endif
}

bool solve_spd_in_place(std::span<double> matrix, std::span<double> rhs, std::size_t n,
                        std::size_t nrhs) noexcept
{
    if (n == 0 || nrhs == 0 || matrix.size() != n * n || rhs.size() != n * nrhs)
    {
        return false;
    }
    // Taken before the factorization overwrites the lower triangle, so the
    // pivot threshold is the one belonging to the matrix that was handed in.
    const double tol = relative_tolerance(matrix);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t col = 0; col <= row; ++col)
        {
            // The entry is read before it is written and every factor this sum
            // touches sits in an already-completed column, which is what lets
            // the factor overwrite the matrix it came from.
            double value = matrix[matrix_index(row, col, n)];
            for (std::size_t k = 0; k < col; ++k)
            {
                value -= matrix[matrix_index(row, k, n)] * matrix[matrix_index(col, k, n)];
            }
            if (row == col)
            {
                if (value <= tol)
                {
                    return false;
                }
                matrix[matrix_index(row, col, n)] = std::sqrt(value);
            }
            else
            {
                matrix[matrix_index(row, col, n)] = value / matrix[matrix_index(col, col, n)];
            }
        }
    }

    for (std::size_t rhs_col = 0; rhs_col < nrhs; ++rhs_col)
    {
        for (std::size_t row = 0; row < n; ++row)
        {
            double value = rhs[matrix_index(row, rhs_col, nrhs)];
            for (std::size_t col = 0; col < row; ++col)
            {
                value -= matrix[matrix_index(row, col, n)] * rhs[matrix_index(col, rhs_col, nrhs)];
            }
            rhs[matrix_index(row, rhs_col, nrhs)] = value / matrix[matrix_index(row, row, n)];
        }
        for (std::size_t reverse = 0; reverse < n; ++reverse)
        {
            const std::size_t row = n - reverse - 1;
            double value = rhs[matrix_index(row, rhs_col, nrhs)];
            for (std::size_t col = row + 1; col < n; ++col)
            {
                value -= matrix[matrix_index(col, row, n)] * rhs[matrix_index(col, rhs_col, nrhs)];
            }
            rhs[matrix_index(row, rhs_col, nrhs)] = value / matrix[matrix_index(row, row, n)];
        }
    }
    return true;
}

bool solve_linear_in_place(std::span<double> matrix, std::span<double> rhs, std::size_t n,
                           std::size_t nrhs) noexcept
{
    if (n == 0 || nrhs == 0 || matrix.size() != n * n || rhs.size() != n * nrhs)
    {
        return false;
    }
    const double tol = relative_tolerance(matrix);
    for (std::size_t pivot_col = 0; pivot_col < n; ++pivot_col)
    {
        std::size_t pivot = pivot_col;
        double pivot_abs = std::abs(matrix[matrix_index(pivot_col, pivot_col, n)]);
        for (std::size_t row = pivot_col + 1; row < n; ++row)
        {
            const double candidate = std::abs(matrix[matrix_index(row, pivot_col, n)]);
            if (candidate > pivot_abs)
            {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= tol)
        {
            return false;
        }
        if (pivot != pivot_col)
        {
            for (std::size_t col = 0; col < n; ++col)
            {
                std::swap(matrix[matrix_index(pivot_col, col, n)],
                          matrix[matrix_index(pivot, col, n)]);
            }
            for (std::size_t col = 0; col < nrhs; ++col)
            {
                std::swap(rhs[matrix_index(pivot_col, col, nrhs)],
                          rhs[matrix_index(pivot, col, nrhs)]);
            }
        }

        for (std::size_t row = pivot_col + 1; row < n; ++row)
        {
            const double factor = matrix[matrix_index(row, pivot_col, n)] /
                                  matrix[matrix_index(pivot_col, pivot_col, n)];
            matrix[matrix_index(row, pivot_col, n)] = 0.0;
            for (std::size_t col = pivot_col + 1; col < n; ++col)
            {
                matrix[matrix_index(row, col, n)] -=
                    factor * matrix[matrix_index(pivot_col, col, n)];
            }
            for (std::size_t col = 0; col < nrhs; ++col)
            {
                rhs[matrix_index(row, col, nrhs)] -=
                    factor * rhs[matrix_index(pivot_col, col, nrhs)];
            }
        }
    }

    for (std::size_t rhs_col = 0; rhs_col < nrhs; ++rhs_col)
    {
        for (std::size_t reverse = 0; reverse < n; ++reverse)
        {
            const std::size_t row = n - reverse - 1;
            double value = rhs[matrix_index(row, rhs_col, nrhs)];
            for (std::size_t col = row + 1; col < n; ++col)
            {
                value -= matrix[matrix_index(row, col, n)] * rhs[matrix_index(col, rhs_col, nrhs)];
            }
            rhs[matrix_index(row, rhs_col, nrhs)] = value / matrix[matrix_index(row, row, n)];
        }
    }
    return true;
}

void gemm_nt(std::span<const double> left, std::span<const double> right, std::size_t rows,
             std::size_t cols, std::size_t inner, std::span<double> output)
{
    if (left.size() != rows * inner || right.size() != cols * inner || output.size() != rows * cols)
    {
        throw std::invalid_argument("matrix multiplication shape is invalid");
    }
    gemm_nt_unchecked(left, right, rows, cols, inner, output);
}

void gemm_nt_unchecked(std::span<const double> left, std::span<const double> right,
                       std::size_t rows, std::size_t cols, std::size_t inner,
                       std::span<double> output)
{
#ifdef NEURALE_MODELS_WITH_MKL
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(rows),
                static_cast<MKL_INT>(cols), static_cast<MKL_INT>(inner), 1.0, left.data(),
                static_cast<MKL_INT>(inner), right.data(), static_cast<MKL_INT>(inner), 0.0,
                output.data(), static_cast<MKL_INT>(cols));
#else
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            double value = 0.0;
            for (std::size_t k = 0; k < inner; ++k)
            {
                value += left[matrix_index(row, k, inner)] * right[matrix_index(col, k, inner)];
            }
            output[matrix_index(row, col, cols)] = value;
        }
    }
#endif
}

void gemv_unchecked(std::span<const double> left, std::span<const double> right, std::size_t rows,
                    std::size_t inner, std::span<double> output, std::size_t output_stride)
{
#ifdef NEURALE_MODELS_WITH_MKL
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<MKL_INT>(rows),
                static_cast<MKL_INT>(inner), 1.0, left.data(), static_cast<MKL_INT>(inner),
                right.data(), 1, 0.0, output.data(), static_cast<MKL_INT>(output_stride));
#else
    for (std::size_t row = 0; row < rows; ++row)
    {
        double value = 0.0;
        for (std::size_t feature = 0; feature < inner; ++feature)
        {
            value += left[matrix_index(row, feature, inner)] * right[feature];
        }
        output[row * output_stride] = value;
    }
#endif
}

} // namespace neurale::models
