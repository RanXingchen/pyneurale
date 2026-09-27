/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/density.h>

#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace neurale::models
{
namespace
{

using detail::checked_product;
using detail::matrix_index;
using detail::saturating_product;

constexpr std::size_t kParallelWorkThreshold = 1'000'000;

struct CholeskyResult
{
    std::vector<double> lower;
    double log_det{};
};

struct CovarianceResult
{
    std::vector<double> mean;
    std::vector<double> covariance;
};

void require_samples(std::span<const double> X, std::size_t n_samples, std::size_t n_features)
{
    if (n_samples < 2 || n_features == 0 ||
        X.size() != checked_product(n_samples, n_features, "X shape is too large"))
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (n_samples <= n_features)
    {
        throw std::invalid_argument("full-covariance KDE requires n_samples > n_features");
    }
}

void require_finite(std::span<const double> values, const char* name)
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(name) + " must contain finite values");
        }
    }
}

void require_points(std::span<const double> X, std::size_t nX, std::size_t n_features,
                    std::span<double> output)
{
    if (n_features == 0 || X.size() != checked_product(nX, n_features, "X shape is too large") ||
        output.size() != nX)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (detail::spans_overlap(X, output))
    {
        throw std::invalid_argument("X and output must not overlap");
    }
}

double compute_bandwidth_factor(std::size_t n_samples, std::size_t n_features, BandwidthRule rule)
{
    const double samples = static_cast<double>(n_samples);
    const double features = static_cast<double>(n_features);
    if (rule == BandwidthRule::Scott)
    {
        return std::pow(samples, -1.0 / (features + 4.0));
    }
    if (rule == BandwidthRule::Silverman)
    {
        return std::pow(samples * (features + 2.0) / 4.0, -1.0 / (features + 4.0));
    }
    throw std::invalid_argument("bw_method is invalid");
}

double validate_bandwidth_factor(double factor)
{
    if (factor > 0.0 && std::isfinite(factor))
    {
        return factor;
    }
    throw std::invalid_argument("bw_method is invalid");
}

CovarianceResult compute_kernel_covariance(std::span<const double> X, std::size_t n_samples,
                                           std::size_t n_features, double factor)
{
    const double factor_squared = factor * factor;
    if (!std::isfinite(factor_squared) || factor_squared <= 0.0)
    {
        throw std::invalid_argument("bw_method is invalid");
    }

    auto mean = std::vector<double>(n_features, 0.0);
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        const double base = X[matrix_index(0, feature, n_features)];
        double sum = 0.0;
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            sum += X[matrix_index(sample, feature, n_features)] - base;
        }
        mean[feature] = base + sum / static_cast<double>(n_samples);
    }

    auto result = std::vector<double>(
        checked_product(n_features, n_features, "covariance shape is too large"), 0.0);
    const double scale = factor_squared / static_cast<double>(n_samples - 1);
    if (!std::isfinite(scale) || scale <= 0.0)
    {
        throw std::invalid_argument("bw_method is invalid");
    }
    for (std::size_t row = 0; row < n_features; ++row)
    {
        for (std::size_t col = row; col < n_features; ++col)
        {
            double value = 0.0;
            for (std::size_t sample = 0; sample < n_samples; ++sample)
            {
                value += (X[matrix_index(sample, row, n_features)] - mean[row]) *
                         (X[matrix_index(sample, col, n_features)] - mean[col]);
            }
            result[matrix_index(row, col, n_features)] = value * scale;
            result[matrix_index(col, row, n_features)] = value * scale;
        }
    }
    return CovarianceResult{
        .mean = std::move(mean),
        .covariance = std::move(result),
    };
}

CholeskyResult cholesky_factor(std::span<const double> matrix, std::size_t n)
{
    CholeskyResult result;
    result.lower.assign(checked_product(n, n, "Cholesky shape is too large"), 0.0);
    double max_diagonal = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        max_diagonal = std::max(max_diagonal, std::abs(matrix[matrix_index(i, i, n)]));
    }
    const double tol =
        std::numeric_limits<double>::epsilon() * static_cast<double>(n) * max_diagonal;

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t col = 0; col <= row; ++col)
        {
            double value = matrix[matrix_index(row, col, n)];
            for (std::size_t k = 0; k < col; ++k)
            {
                value -=
                    result.lower[matrix_index(row, k, n)] * result.lower[matrix_index(col, k, n)];
            }
            if (row == col)
            {
                if (!std::isfinite(value) || value <= tol)
                {
                    throw std::invalid_argument(
                        "KDE covariance must be finite and positive definite");
                }
                result.lower[matrix_index(row, col, n)] = std::sqrt(value);
                result.log_det += 2.0 * std::log(result.lower[matrix_index(row, col, n)]);
            }
            else
            {
                result.lower[matrix_index(row, col, n)] =
                    value / result.lower[matrix_index(col, col, n)];
            }
        }
    }
    return result;
}

void whiten_row(std::span<const double> values, std::size_t row_idx, std::span<const double> mean,
                std::span<const double> lower, std::size_t n_features, std::span<double> output)
{
    for (std::size_t row = 0; row < n_features; ++row)
    {
        double value = values[matrix_index(row_idx, row, n_features)] - mean[row];
        for (std::size_t col = 0; col < row; ++col)
        {
            value -= lower[matrix_index(row, col, n_features)] * output[col];
        }
        output[row] = value / lower[matrix_index(row, row, n_features)];
    }
}

std::vector<double> whiten_samples(std::span<const double> samples, std::span<const double> mean,
                                   std::span<const double> lower, std::size_t n_samples,
                                   std::size_t n_features)
{
    auto output = std::vector<double>(
        checked_product(n_samples, n_features, "whitened samples shape is too large"));
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        whiten_row(samples, sample, mean, lower, n_features,
                   std::span<double>(output.data() + sample * n_features, n_features));
    }
    return output;
}

double squared_distance(std::span<const double> left, std::span<const double> right)
{
    double value = 0.0;
    for (std::size_t feature = 0; feature < left.size(); ++feature)
    {
        const double delta = left[feature] - right[feature];
        value += delta * delta;
    }
    return value;
}

std::size_t kde_work_size(std::size_t nX, std::size_t n_samples, std::size_t n_features) noexcept
{
    return saturating_product(saturating_product(nX, n_samples), n_features);
}

std::size_t kde_thread_count(std::size_t nX, std::size_t work_size) noexcept
{
#ifdef _OPENMP
    if (work_size < kParallelWorkThreshold || nX < 2)
    {
        return 1;
    }
    return std::min(nX, static_cast<std::size_t>(omp_get_max_threads()));
#else
    (void)nX;
    (void)work_size;
    return 1;
#endif
}

std::size_t kde_thread_index() noexcept
{
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

} // namespace

GaussianKdeModel::GaussianKdeModel(State state) : state_(std::move(state)) {}

std::size_t GaussianKdeModel::n_samples() const noexcept
{
    return state_.n_samples;
}

std::size_t GaussianKdeModel::n_features() const noexcept
{
    return state_.n_features;
}

double GaussianKdeModel::bandwidth_factor() const noexcept
{
    return state_.bandwidth_factor;
}

void GaussianKdeModel::pdf(std::span<const double> X, std::size_t nX,
                           std::span<double> output) const
{
    evaluate_logpdf(X, nX, output);
    for (double& value : output)
    {
        value = std::exp(value);
    }
}

void GaussianKdeModel::logpdf(std::span<const double> X, std::size_t nX,
                              std::span<double> output) const
{
    evaluate_logpdf(X, nX, output);
}

void GaussianKdeModel::evaluate_logpdf(std::span<const double> X, std::size_t nX,
                                       std::span<double> output) const
{
    require_points(X, nX, state_.n_features, output);
    require_finite(X, "X");
    if (nX == 0)
    {
        return;
    }
    evaluate_logpdf_impl(X, nX, output);
}

void GaussianKdeModel::evaluate_logpdf_impl(std::span<const double> X, std::size_t nX,
                                            std::span<double> output) const
{
    if (nX > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    {
        throw std::length_error("too many query points");
    }

    const std::size_t work_size = kde_work_size(nX, state_.n_samples, state_.n_features);
    const std::size_t n_threads = kde_thread_count(nX, work_size);
    auto workspace = std::vector<double>(
        checked_product(n_threads, state_.n_features, "KDE workspace shape is too large"));

#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(n_threads)) if (n_threads > 1)
#endif
    {
        const std::size_t thread = kde_thread_index();
        auto whitened_point =
            std::span<double>(workspace.data() + thread * state_.n_features, state_.n_features);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nX); ++i)
        {
            const auto point = static_cast<std::size_t>(i);
            whiten_row(X, point, state_.mean, state_.lower_cholesky, state_.n_features,
                       whitened_point);
            output[point] = point_logpdf(whitened_point);
        }
    }
}

double GaussianKdeModel::point_logpdf(std::span<const double> whitened_point) const
{
    double largest = -std::numeric_limits<double>::infinity();
    double scaled_sum = 0.0;
    for (std::size_t sample = 0; sample < state_.n_samples; ++sample)
    {
        const double distance = squared_distance(
            whitened_point,
            std::span<const double>(state_.whitened_samples.data() + sample * state_.n_features,
                                    state_.n_features));
        // Finite inputs may overflow to +inf in squared-distance arithmetic.
        // Such a kernel contributes zero density.
        if (!std::isfinite(distance))
        {
            continue;
        }
        const double value = -0.5 * distance;
        if (value > largest)
        {
            scaled_sum = largest == -std::numeric_limits<double>::infinity()
                             ? 1.0
                             : scaled_sum * std::exp(largest - value) + 1.0;
            largest = value;
        }
        else
        {
            scaled_sum += std::exp(value - largest);
        }
    }
    return largest == -std::numeric_limits<double>::infinity()
               ? largest
               : state_.log_normalizer + largest + std::log(scaled_sum);
}

GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                  std::size_t n_features, BandwidthRule bandwidth_rule)
{
    return fit_gaussian_kde(X, n_samples, n_features,
                            compute_bandwidth_factor(n_samples, n_features, bandwidth_rule));
}

GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                  std::size_t n_features, double bandwidth_factor)
{
    require_samples(X, n_samples, n_features);
    require_finite(X, "X");
    const double factor = validate_bandwidth_factor(bandwidth_factor);
    auto cov = compute_kernel_covariance(X, n_samples, n_features, factor);
    auto cholesky = cholesky_factor(cov.covariance, n_features);
    auto whitened = whiten_samples(X, cov.mean, cholesky.lower, n_samples, n_features);
    require_finite(whitened, "whitened X");
    const double log_normalizer =
        -std::log(static_cast<double>(n_samples)) -
        0.5 *
            (static_cast<double>(n_features) * std::log(2.0 * std::numbers::pi) + cholesky.log_det);

    return GaussianKdeModel(GaussianKdeModel::State{
        .n_samples = n_samples,
        .n_features = n_features,
        .bandwidth_factor = factor,
        .log_normalizer = log_normalizer,
        .mean = std::move(cov.mean),
        .whitened_samples = std::move(whitened),
        .lower_cholesky = std::move(cholesky.lower),
    });
}

} // namespace neurale::models
