/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/linear_model.h>

#ifdef NEURALE_MODELS_WITH_MKL
#include "mkl_utils.h"
#endif

#include "linalg.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace neurale::models
{
namespace
{

using detail::matrix_index;

void require_fit_shapes(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                        std::span<const double> y, std::size_t n_outputs, double alpha)
{
    if (n_samples == 0 || n_features == 0 || X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (n_outputs == 0 || y.size() != n_samples * n_outputs)
    {
        throw std::invalid_argument("y shape is invalid");
    }
    if (!(alpha >= 0.0) || !std::isfinite(alpha))
    {
        throw std::invalid_argument("alpha must be a finite non-negative number");
    }
}

void require_model_state(const LinearModelState& state)
{
    if (state.n_features == 0 || state.n_outputs == 0 ||
        state.coef.size() != state.n_outputs * state.n_features ||
        state.intercept.size() != state.n_outputs)
    {
        throw std::invalid_argument("linear model state shape is invalid");
    }
}

/// Largest magnitude in ``values``; zero when they are all zero.
double magnitude(std::span<const double> values)
{
    double largest = 0.0;
    for (double value : values)
    {
        largest = std::max(largest, std::abs(value));
    }
    return largest;
}

/// Power-of-two factor that brings ``largest`` inside ``[low, high]``.
///
/// Returning a power of two makes the rescaling exact for every value that stays
/// normal, and returning 1.0 for operands that are already in range means the
/// common case is not rescaled at all. Rescaling only as far as the band
/// requires, rather than down to unity, is what preserves the dynamic range
/// below the largest entry.
double band_scale(double largest, double low, double high)
{
    if (largest == 0.0 || (largest >= low && largest <= high))
    {
        return 1.0;
    }
    int exponent = 0;
    std::frexp(largest / (largest < low ? low : high), &exponent);
    return std::ldexp(1.0, exponent);
}

/// Copy ``values`` divided by ``scale``, rejecting a copy that loses a value.
std::vector<double> rescaled_copy(std::span<const double> values, double scale, const char* name)
{
    auto scaled = std::vector<double>(values.begin(), values.end());
    if (scale == 1.0)
    {
        return scaled;
    }
    for (double& value : scaled)
    {
        const double original = value;
        value /= scale;
        if (value == 0.0 && original != 0.0)
        {
            throw std::runtime_error(std::string(name) +
                                     " spans more magnitudes than double precision can carry: "
                                     "bringing its largest values into range underflows its "
                                     "smallest nonzero ones");
        }
    }
    return scaled;
}

/// Column means accumulated incrementally, so no partial sum leaves the range
/// spanned by the column itself.
std::vector<double> column_means(std::span<const double> values, std::size_t n_rows,
                                 std::size_t n_cols)
{
    auto means = std::vector<double>(n_cols, 0.0);
    for (std::size_t row = 0; row < n_rows; ++row)
    {
        const double count = static_cast<double>(row + 1);
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            means[col] += (values[matrix_index(row, col, n_cols)] - means[col]) / count;
        }
    }
    return means;
}

void center_in_place(std::span<double> values, std::size_t n_rows, std::size_t n_cols,
                     std::span<const double> means)
{
    for (std::size_t row = 0; row < n_rows; ++row)
    {
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            values[matrix_index(row, col, n_cols)] -= means[col];
        }
    }
}

void require_finite_fit(const LinearFitResult& result)
{
    const auto finite = [](std::span<const double> values)
    {
        return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
    };
    if (!finite(result.singular_values) || !finite(result.coef) || !finite(result.intercept))
    {
        throw std::runtime_error("linear fit produced non-finite parameters; X and y are finite "
                                 "but the solution they imply is not representable");
    }
}

/// Compute ``U^T Y`` for a row-major ``n_components x n_samples`` left factor.
std::vector<double> project_targets(std::span<const double> left_singular_vectors,
                                    std::span<const double> y, std::size_t n_samples,
                                    std::size_t n_components, std::size_t n_outputs)
{
    auto projected = std::vector<double>(n_components * n_outputs, 0.0);
    for (std::size_t component = 0; component < n_components; ++component)
    {
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            const double weight = left_singular_vectors[matrix_index(component, sample, n_samples)];
            if (weight == 0.0)
            {
                continue;
            }
            for (std::size_t output = 0; output < n_outputs; ++output)
            {
                projected[matrix_index(component, output, n_outputs)] +=
                    weight * y[matrix_index(sample, output, n_outputs)];
            }
        }
    }
    return projected;
}

/// Ridge filter ``s / (s^2 + alpha)`` evaluated without squaring ``s``.
///
/// The reciprocal form ``1 / (s + alpha / s)`` agrees with the definition
/// wherever both are representable, and stays correct where the square is not:
/// a large ``s`` still yields ``1 / s`` instead of dividing by an overflowed
/// denominator, and a small one still yields ``s / alpha`` instead of dividing
/// by an underflowed one. ``s`` is nonzero by construction here.
double ridge_filter(double singular, double alpha)
{
    if (alpha == 0.0)
    {
        return 1.0 / singular;
    }
    return 1.0 / (singular + alpha / singular);
}

double rank_tolerance(std::span<const double> singular_values, std::size_t n_samples,
                      std::size_t n_features)
{
    if (singular_values.empty() || singular_values.front() <= 0.0)
    {
        return 0.0;
    }
    const auto dim = static_cast<double>(std::max(n_samples, n_features));
    return dim * std::numeric_limits<double>::epsilon() * singular_values.front();
}

} // namespace

LinearModel::LinearModel(LinearModelState state)
{
    require_model_state(state);
    state_ = std::move(state);
}

std::size_t LinearModel::n_features() const noexcept
{
    return state_.n_features;
}

std::size_t LinearModel::n_outputs() const noexcept
{
    return state_.n_outputs;
}

const std::vector<double>& LinearModel::coef() const noexcept
{
    return state_.coef;
}

const std::vector<double>& LinearModel::intercept() const noexcept
{
    return state_.intercept;
}

void LinearModel::predict(std::span<const double> X, std::size_t n_samples,
                          std::span<double> output) const
{
    const std::size_t n_features = state_.n_features;
    const std::size_t n_outputs = state_.n_outputs;
    if (X.size() != n_samples * n_features || output.size() != n_samples * n_outputs)
    {
        throw std::invalid_argument("prediction shape is invalid");
    }
    const auto X_begin = reinterpret_cast<std::uintptr_t>(X.data());
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data());
    if (X_begin < output_begin + output.size_bytes() && output_begin < X_begin + X.size_bytes())
    {
        throw std::invalid_argument("prediction X and output must not overlap");
    }
#ifdef NEURALE_MODELS_WITH_MKL
    // Pinned for the reasons on the declaration: a bounded tail on the
    // real-time thread, and one answer per input whatever the machine load.
    const mkl::LocalThreadLimit thread_limit;
#endif
    gemm_nt(X, state_.coef, n_samples, n_outputs, n_features, output);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t i = 0; i < n_outputs; ++i)
        {
            output[matrix_index(sample, i, n_outputs)] += state_.intercept[i];
        }
    }
}

LinearFitResult fit_linear_model(std::span<const double> X, std::size_t n_samples,
                                 std::size_t n_features, std::span<const double> y,
                                 std::size_t n_outputs, double alpha, bool fit_intercept)
{
    require_fit_shapes(X, n_samples, n_features, y, n_outputs, alpha);

    // Rescale only what the factorization cannot take as given. The sweep sums
    // squares of design entries, so the design has to fit in a band whose square
    // survives accumulation over the samples; the projections only sum target
    // entries, so the targets fit in a much wider one. Both bounds leave a factor
    // of four for the centering. Operands already inside their band are used
    // untouched, and the ones outside move by a power of two, so nothing is
    // rescaled that does not have to be and nothing that is rescaled is rounded.
    const auto samples = static_cast<double>(n_samples);
    const double design_high = std::sqrt(std::numeric_limits<double>::max() / (16.0 * samples));
    const double design_low = std::sqrt(std::numeric_limits<double>::min()) * 1024.0;
    const double target_high = std::numeric_limits<double>::max() / (16.0 * samples);
    const double target_low = std::numeric_limits<double>::min() * 1024.0;
    const double design_scale = band_scale(magnitude(X), design_low, design_high);
    const double target_scale = band_scale(magnitude(y), target_low, target_high);
    auto design = rescaled_copy(X, design_scale, "X");
    auto response = rescaled_copy(y, target_scale, "y");

    auto feature_means = std::vector<double>(n_features, 0.0);
    auto target_means = std::vector<double>(n_outputs, 0.0);
    if (fit_intercept)
    {
        feature_means = column_means(design, n_samples, n_features);
        target_means = column_means(response, n_samples, n_outputs);
        center_in_place(design, n_samples, n_features, feature_means);
        center_in_place(response, n_samples, n_outputs, target_means);
    }

    auto svd = accurate_thin_svd(design, n_samples, n_features);
    const std::size_t n_components = svd.singular_values.size();
    // Read the singular values back on the scale of the design the caller gave,
    // so that the penalty is weighed against them directly and never has to be
    // transported into the working variables itself.
    auto singular_values = std::move(svd.singular_values);
    for (double& singular : singular_values)
    {
        singular *= design_scale;
    }
    const double tol = rank_tolerance(singular_values, n_samples, n_features);

    // Solve from the projection of the targets onto the left singular vectors.
    // Going through X^T y instead would square the condition number and cancel
    // away exactly the near-singular directions the factorization just resolved.
    const auto projected_targets =
        project_targets(svd.left_singular_vectors, response, n_samples, n_components, n_outputs);
    auto projected = std::vector<double>(n_components * n_outputs, 0.0);
    std::size_t rank = 0;
    for (std::size_t component = 0; component < n_components; ++component)
    {
        const double singular = singular_values[component];
        if (singular > tol)
        {
            ++rank;
        }
        // A penalized solve keeps every direction the design actually spans: the
        // filter is finite wherever s is, and a direction below the rank
        // tolerance can still dominate the solution when alpha is smaller than
        // s^2. Only an unpenalized solve truncates, which is what makes its
        // result the minimum-norm one.
        const bool discard = singular == 0.0 || (alpha == 0.0 && singular <= tol);
        if (discard)
        {
            continue;
        }
        const double filter = ridge_filter(singular, alpha);
        for (std::size_t output = 0; output < n_outputs; ++output)
        {
            projected[matrix_index(component, output, n_outputs)] =
                filter * projected_targets[matrix_index(component, output, n_outputs)];
        }
    }

    LinearFitResult result;
    result.n_samples = n_samples;
    result.n_features = n_features;
    result.n_outputs = n_outputs;
    result.rank = rank;
    result.singular_values = std::move(singular_values);
    result.coef.assign(n_outputs * n_features, 0.0);
    for (std::size_t output = 0; output < n_outputs; ++output)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            double value = 0.0;
            for (std::size_t component = 0; component < n_components; ++component)
            {
                value += svd.right_singular_vectors[matrix_index(component, feature, n_features)] *
                         projected[matrix_index(component, output, n_outputs)];
            }
            // The filter already carries the design scale, so only the target
            // rescaling has to be undone here.
            result.coef[matrix_index(output, feature, n_features)] = value * target_scale;
        }
    }

    result.intercept.assign(n_outputs, 0.0);
    if (fit_intercept)
    {
        for (std::size_t output = 0; output < n_outputs; ++output)
        {
            double value = target_means[output] * target_scale;
            for (std::size_t feature = 0; feature < n_features; ++feature)
            {
                value -= result.coef[matrix_index(output, feature, n_features)] *
                         (feature_means[feature] * design_scale);
            }
            result.intercept[output] = value;
        }
    }
    require_finite_fit(result);
    return result;
}

} // namespace neurale::models
