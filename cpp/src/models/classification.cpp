/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/classification.h>

#include "linalg.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neurale::models
{
namespace
{

using detail::matrix_index;

void require_training_data(std::span<const double> X, std::span<const std::size_t> y,
                           std::size_t n_samples, std::size_t n_features, std::size_t n_classes)
{
    if (n_samples < 2 || n_features == 0 || X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (y.size() != n_samples || n_classes < 2 || n_samples <= n_classes)
    {
        throw std::invalid_argument("labels are invalid");
    }
    for (std::size_t label : y)
    {
        if (label >= n_classes)
        {
            throw std::invalid_argument("labels are invalid");
        }
    }
}

std::vector<double> indexed_empirical_covariance(std::span<const double> X,
                                                 std::span<const std::size_t> sample_indices,
                                                 std::span<const double> mean,
                                                 std::size_t n_features)
{
    auto covariance = std::vector<double>(n_features * n_features, 0.0);
    for (std::size_t row = 0; row < n_features; ++row)
    {
        for (std::size_t col = row; col < n_features; ++col)
        {
            double value = 0.0;
            for (std::size_t sample : sample_indices)
            {
                value += (X[matrix_index(sample, row, n_features)] - mean[row]) *
                         (X[matrix_index(sample, col, n_features)] - mean[col]);
            }
            value /= static_cast<double>(sample_indices.size());
            covariance[matrix_index(row, col, n_features)] = value;
            covariance[matrix_index(col, row, n_features)] = value;
        }
    }
    return covariance;
}

std::vector<double> indexed_standardized_covariance(std::span<const double> X,
                                                    std::span<const std::size_t> sample_indices,
                                                    std::span<const double> mean,
                                                    std::span<const double> scale,
                                                    std::size_t n_features)
{
    auto covariance = std::vector<double>(n_features * n_features, 0.0);
    for (std::size_t row = 0; row < n_features; ++row)
    {
        for (std::size_t col = row; col < n_features; ++col)
        {
            double value = 0.0;
            for (std::size_t sample : sample_indices)
            {
                const double left =
                    (X[matrix_index(sample, row, n_features)] - mean[row]) / scale[row];
                const double right =
                    (X[matrix_index(sample, col, n_features)] - mean[col]) / scale[col];
                value += left * right;
            }
            value /= static_cast<double>(sample_indices.size());
            covariance[matrix_index(row, col, n_features)] = value;
            covariance[matrix_index(col, row, n_features)] = value;
        }
    }
    return covariance;
}

void require_shrinkage(const std::string& shrinkage_kind, double shrinkage_value)
{
    if (shrinkage_kind != "none" && shrinkage_kind != "fixed" && shrinkage_kind != "auto")
    {
        throw std::invalid_argument("shrinkage kind is invalid");
    }
    if (shrinkage_kind == "fixed" &&
        (!std::isfinite(shrinkage_value) || shrinkage_value < 0.0 || shrinkage_value > 1.0))
    {
        throw std::invalid_argument("fixed shrinkage must be in [0, 1]");
    }
}

std::vector<double> shrink_covariance(std::vector<double> covariance, std::size_t n_features,
                                      double shrinkage)
{
    const double trace = [&]
    {
        double value = 0.0;
        for (std::size_t i = 0; i < n_features; ++i)
        {
            value += covariance[matrix_index(i, i, n_features)];
        }
        return value;
    }();
    const double mu = trace / static_cast<double>(n_features);
    for (double& value : covariance)
    {
        value *= 1.0 - shrinkage;
    }
    for (std::size_t i = 0; i < n_features; ++i)
    {
        covariance[matrix_index(i, i, n_features)] += shrinkage * mu;
    }
    return covariance;
}

double indexed_ledoit_wolf_shrinkage(std::span<const double> X,
                                     std::span<const std::size_t> sample_indices,
                                     std::span<const double> mean, std::span<const double> scale,
                                     std::size_t n_features)
{
    auto trace_values = std::vector<double>(n_features, 0.0);
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        for (std::size_t sample : sample_indices)
        {
            const double value =
                (X[matrix_index(sample, feature, n_features)] - mean[feature]) / scale[feature];
            trace_values[feature] += value * value;
        }
        trace_values[feature] /= static_cast<double>(sample_indices.size());
    }
    const double mu = std::accumulate(trace_values.begin(), trace_values.end(), 0.0) /
                      static_cast<double>(n_features);

    double beta_sum = 0.0;
    double delta_sum = 0.0;
    for (std::size_t row = 0; row < n_features; ++row)
    {
        for (std::size_t col = 0; col < n_features; ++col)
        {
            double beta_current = 0.0;
            double delta_current = 0.0;
            for (std::size_t sample : sample_indices)
            {
                const double row_value =
                    (X[matrix_index(sample, row, n_features)] - mean[row]) / scale[row];
                const double col_value =
                    (X[matrix_index(sample, col, n_features)] - mean[col]) / scale[col];
                beta_current += row_value * row_value * col_value * col_value;
                delta_current += row_value * col_value;
            }
            beta_sum += beta_current;
            delta_sum += delta_current * delta_current;
        }
    }

    const double samples = static_cast<double>(sample_indices.size());
    const double features = static_cast<double>(n_features);
    delta_sum /= samples * samples;
    double beta = (beta_sum / samples - delta_sum) / (features * samples);
    const double trace_sum = std::accumulate(trace_values.begin(), trace_values.end(), 0.0);
    const double delta = (delta_sum - 2.0 * mu * trace_sum + features * mu * mu) / features;
    if (delta <= 0.0)
    {
        return 0.0;
    }
    beta = std::min(std::max(beta, 0.0), delta);
    return beta / delta;
}

std::vector<double> auto_shrunk_covariance(std::span<const double> X,
                                           std::span<const std::size_t> sample_indices,
                                           std::size_t n_features)
{
    auto mean = std::vector<double>(n_features, 0.0);
    auto scale = std::vector<double>(n_features, 0.0);
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        for (std::size_t sample : sample_indices)
        {
            mean[feature] += X[matrix_index(sample, feature, n_features)];
        }
        mean[feature] /= static_cast<double>(sample_indices.size());
        for (std::size_t sample : sample_indices)
        {
            const double centered = X[matrix_index(sample, feature, n_features)] - mean[feature];
            scale[feature] += centered * centered;
        }
        const double variance = scale[feature] / static_cast<double>(sample_indices.size());
        const double samples = static_cast<double>(sample_indices.size());
        const double eps = std::numeric_limits<double>::epsilon();
        const double mean_error = samples * mean[feature] * eps;
        const double tol = samples * eps * variance + mean_error * mean_error;
        scale[feature] = variance <= tol ? 1.0 : std::sqrt(variance);
    }

    auto covariance = indexed_standardized_covariance(X, sample_indices, mean, scale, n_features);
    covariance = shrink_covariance(
        std::move(covariance), n_features,
        indexed_ledoit_wolf_shrinkage(X, sample_indices, mean, scale, n_features));
    for (std::size_t row = 0; row < n_features; ++row)
    {
        for (std::size_t col = 0; col < n_features; ++col)
        {
            covariance[matrix_index(row, col, n_features)] *= scale[row] * scale[col];
        }
    }
    return covariance;
}

void lda_decision_function_unchecked(std::span<const double> X, std::size_t n_samples,
                                     std::size_t n_features, std::span<const double> coef,
                                     std::size_t n_outputs, std::span<const double> intercept,
                                     std::span<double> output)
{
    if (n_outputs == 1)
    {
        gemv_unchecked(X, coef, n_samples, n_features, output, 1);
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            output[sample] += intercept[0];
        }
        return;
    }

    gemm_nt_unchecked(X, coef, n_samples, n_outputs, n_features, output);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t i = 0; i < n_outputs; ++i)
        {
            output[matrix_index(sample, i, n_outputs)] += intercept[i];
        }
    }
}

void lda_binary_predict_proba_unchecked(std::span<const double> X, std::size_t n_samples,
                                        std::size_t n_features, std::span<const double> coef,
                                        std::span<const double> intercept, std::span<double> output)
{
    auto scores = std::span<double>(output.data() + 1, output.size() - 1);
    gemv_unchecked(X, coef, n_samples, n_features, scores, 2);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const double score = output[matrix_index(sample, 1, 2)] + intercept[0];
        const double positive = score >= 0.0 ? 1.0 / (1.0 + std::exp(-score))
                                             : std::exp(score) / (1.0 + std::exp(score));
        output[matrix_index(sample, 0, 2)] = 1.0 - positive;
        output[matrix_index(sample, 1, 2)] = positive;
    }
}

} // namespace

LdaModel::LdaModel(LdaModelState state)
{
    if (state.n_features == 0 || state.n_classes < 2 || state.n_outputs == 0 ||
        state.coef.size() != state.n_outputs * state.n_features ||
        state.intercept.size() != state.n_outputs ||
        (state.n_classes == 2 && state.n_outputs != 1) ||
        (state.n_classes > 2 && state.n_outputs != state.n_classes))
    {
        throw std::invalid_argument("LDA state shape is invalid");
    }
    state_ = std::move(state);
}

std::size_t LdaModel::n_features() const noexcept
{
    return state_.n_features;
}

std::size_t LdaModel::n_classes() const noexcept
{
    return state_.n_classes;
}

std::size_t LdaModel::n_outputs() const noexcept
{
    return state_.n_outputs;
}

const std::vector<double>& LdaModel::coef() const noexcept
{
    return state_.coef;
}

const std::vector<double>& LdaModel::intercept() const noexcept
{
    return state_.intercept;
}

void LdaModel::decision_function(std::span<const double> X, std::size_t n_samples,
                                 std::span<double> output) const
{
    lda_decision_function(X, n_samples, state_.n_features, state_.coef, state_.n_outputs,
                          state_.intercept, output);
}

void LdaModel::predict_proba(std::span<const double> X, std::size_t n_samples,
                             std::span<double> output) const
{
    lda_predict_proba(X, n_samples, state_.n_features, state_.coef, state_.n_outputs,
                      state_.intercept, output);
}

LdaFitResult fit_lda(std::span<const double> X, std::span<const std::size_t> y,
                     std::size_t n_samples, std::size_t n_features, std::size_t n_classes,
                     const std::string& shrinkage_kind, double shrinkage_value)
{
    require_training_data(X, y, n_samples, n_features, n_classes);
    require_shrinkage(shrinkage_kind, shrinkage_value);

    LdaFitResult result;
    result.n_features = n_features;
    result.n_classes = n_classes;
    result.priors.assign(n_classes, 0.0);
    result.means.assign(n_classes * n_features, 0.0);
    auto counts = std::vector<std::size_t>(n_classes, 0);

    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const std::size_t label = y[sample];
        ++counts[label];
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            result.means[matrix_index(label, feature, n_features)] +=
                X[matrix_index(sample, feature, n_features)];
        }
    }
    for (std::size_t klass = 0; klass < n_classes; ++klass)
    {
        if (counts[klass] == 0)
        {
            throw std::invalid_argument("each class must contain samples");
        }
        result.priors[klass] = static_cast<double>(counts[klass]) / static_cast<double>(n_samples);
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            result.means[matrix_index(klass, feature, n_features)] /=
                static_cast<double>(counts[klass]);
        }
    }
    auto class_offsets = std::vector<std::size_t>(n_classes + 1, 0);
    for (std::size_t klass = 0; klass < n_classes; ++klass)
    {
        class_offsets[klass + 1] = class_offsets[klass] + counts[klass];
    }
    auto class_indices = std::vector<std::size_t>(n_samples);
    auto write_offsets = class_offsets;
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        class_indices[write_offsets[y[sample]]++] = sample;
    }

    result.covariance.assign(n_features * n_features, 0.0);
    for (std::size_t klass = 0; klass < n_classes; ++klass)
    {
        const auto class_samples = std::span<const std::size_t>(
            class_indices.data() + class_offsets[klass], counts[klass]);
        std::vector<double> class_cov;
        if (shrinkage_kind == "none" || shrinkage_kind == "fixed")
        {
            const auto class_mean =
                std::span<const double>(result.means.data() + klass * n_features, n_features);
            class_cov = indexed_empirical_covariance(X, class_samples, class_mean, n_features);
            if (shrinkage_kind == "fixed")
            {
                class_cov = shrink_covariance(std::move(class_cov), n_features, shrinkage_value);
            }
        }
        else
        {
            class_cov = auto_shrunk_covariance(X, class_samples, n_features);
        }
        for (std::size_t i = 0; i < result.covariance.size(); ++i)
        {
            result.covariance[i] += result.priors[klass] * class_cov[i];
        }
    }

    auto rhs = std::vector<double>(n_features * n_classes);
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        for (std::size_t klass = 0; klass < n_classes; ++klass)
        {
            rhs[matrix_index(feature, klass, n_classes)] =
                result.means[matrix_index(klass, feature, n_features)];
        }
    }
    const auto solution = solve_spd(result.covariance, rhs, n_features, n_classes);

    auto full_coef = std::vector<double>(n_classes * n_features);
    auto full_intercept = std::vector<double>(n_classes);
    for (std::size_t klass = 0; klass < n_classes; ++klass)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            full_coef[matrix_index(klass, feature, n_features)] =
                solution[matrix_index(feature, klass, n_classes)];
        }
        double diagonal = 0.0;
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            diagonal += result.means[matrix_index(klass, feature, n_features)] *
                        full_coef[matrix_index(klass, feature, n_features)];
        }
        full_intercept[klass] = -0.5 * diagonal + std::log(result.priors[klass]);
    }

    if (n_classes == 2)
    {
        result.n_outputs = 1;
        result.coef.assign(n_features, 0.0);
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            result.coef[feature] = full_coef[matrix_index(1, feature, n_features)] -
                                   full_coef[matrix_index(0, feature, n_features)];
        }
        result.intercept = {full_intercept[1] - full_intercept[0]};
    }
    else
    {
        result.n_outputs = n_classes;
        result.coef = std::move(full_coef);
        result.intercept = std::move(full_intercept);
    }
    return result;
}

void lda_decision_function(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                           std::span<const double> coef, std::size_t n_outputs,
                           std::span<const double> intercept, std::span<double> output)
{
    if (n_samples == 0 || n_features == 0 || n_outputs == 0 || X.size() != n_samples * n_features ||
        coef.size() != n_outputs * n_features || intercept.size() != n_outputs ||
        output.size() != n_samples * n_outputs)
    {
        throw std::invalid_argument("LDA decision shape is invalid");
    }
    if (detail::spans_overlap(X, output) || detail::spans_overlap(coef, output) ||
        detail::spans_overlap(intercept, output))
    {
        throw std::invalid_argument("LDA X and output must not overlap");
    }
    lda_decision_function_unchecked(X, n_samples, n_features, coef, n_outputs, intercept, output);
}

void lda_predict_proba(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                       std::span<const double> coef, std::size_t n_outputs,
                       std::span<const double> intercept, std::span<double> output)
{
    const std::size_t n_classes = n_outputs == 1 ? 2 : n_outputs;
    if (n_samples == 0 || n_features == 0 || n_outputs == 0 || X.size() != n_samples * n_features ||
        coef.size() != n_outputs * n_features || intercept.size() != n_outputs)
    {
        throw std::invalid_argument("LDA probability shape is invalid");
    }
    if (output.size() != n_samples * n_classes)
    {
        throw std::invalid_argument("LDA probability output shape is invalid");
    }
    if (detail::spans_overlap(X, output) || detail::spans_overlap(coef, output) ||
        detail::spans_overlap(intercept, output))
    {
        throw std::invalid_argument("LDA X and output must not overlap");
    }
    if (n_outputs == 1)
    {
        lda_binary_predict_proba_unchecked(X, n_samples, n_features, coef, intercept, output);
        return;
    }

    lda_decision_function_unchecked(X, n_samples, n_features, coef, n_outputs, intercept, output);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        double largest = output[matrix_index(sample, 0, n_outputs)];
        for (std::size_t i = 1; i < n_outputs; ++i)
        {
            largest = std::max(largest, output[matrix_index(sample, i, n_outputs)]);
        }
        double total = 0.0;
        for (std::size_t i = 0; i < n_outputs; ++i)
        {
            const double value = std::exp(output[matrix_index(sample, i, n_outputs)] - largest);
            output[matrix_index(sample, i, n_outputs)] = value;
            total += value;
        }
        for (std::size_t i = 0; i < n_outputs; ++i)
        {
            output[matrix_index(sample, i, n_outputs)] /= total;
        }
    }
}

} // namespace neurale::models
