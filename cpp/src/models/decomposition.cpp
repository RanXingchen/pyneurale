/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/decomposition.h>

#include "linalg.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neurale::models
{
namespace
{

using detail::matrix_index;

void require_matrix(std::span<const double> X, std::size_t n_samples, std::size_t n_features)
{
    if (n_samples < 2)
    {
        throw std::invalid_argument("X must contain at least two samples");
    }
    if (n_features == 0 || X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X shape is invalid");
    }
}

std::vector<double> pca_offset(std::span<const double> mean, std::span<const double> components,
                               std::size_t n_components, std::size_t n_features)
{
    auto offset = std::vector<double>(n_components, 0.0);
    for (std::size_t component = 0; component < n_components; ++component)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            offset[component] -=
                mean[feature] * components[matrix_index(component, feature, n_features)];
        }
    }
    return offset;
}

void require_pca_state(const PcaModelState& state)
{
    if (state.n_features == 0 || state.n_components == 0 || state.mean.size() != state.n_features ||
        state.components.size() != state.n_components * state.n_features ||
        state.offset.size() != state.n_components)
    {
        throw std::invalid_argument("PCA state shape is invalid");
    }
}

void transform_pca_unchecked(std::span<const double> X, std::size_t n_samples,
                             std::size_t n_features, std::span<const double> components,
                             std::span<const double> offset, std::size_t n_components,
                             std::span<double> output, bool add_offset)
{
    gemm_nt_unchecked(X, components, n_samples, n_components, n_features, output);
    if (!add_offset)
    {
        return;
    }
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t component = 0; component < n_components; ++component)
        {
            output[matrix_index(sample, component, n_components)] += offset[component];
        }
    }
}

} // namespace

PcaModel::PcaModel(PcaModelState state)
{
    require_pca_state(state);
    has_offset_ = std::any_of(state.offset.begin(), state.offset.end(),
                              [](double value) { return value != 0.0; });
    state_ = std::move(state);
}

std::size_t PcaModel::n_features() const noexcept
{
    return state_.n_features;
}

std::size_t PcaModel::n_components() const noexcept
{
    return state_.n_components;
}

bool PcaModel::center() const noexcept
{
    return state_.center;
}

const std::vector<double>& PcaModel::mean() const noexcept
{
    return state_.mean;
}

const std::vector<double>& PcaModel::components() const noexcept
{
    return state_.components;
}

const std::vector<double>& PcaModel::offset() const noexcept
{
    return state_.offset;
}

void PcaModel::transform(std::span<const double> X, std::size_t n_samples,
                         std::span<double> output) const
{
    if (n_samples == 0 || X.size() != n_samples * state_.n_features)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (output.size() != n_samples * state_.n_components)
    {
        throw std::invalid_argument("PCA state shape is invalid");
    }
    if (detail::spans_overlap(X, output) ||
        detail::spans_overlap(std::span<const double>(state_.components), output) ||
        detail::spans_overlap(std::span<const double>(state_.offset), output))
    {
        throw std::invalid_argument("PCA X and output must not overlap");
    }

    transform_pca_unchecked(X, n_samples, state_.n_features, state_.components, state_.offset,
                            state_.n_components, output, has_offset_);
}

PcaFitResult fit_pca(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t n_components, bool center)
{
    require_matrix(X, n_samples, n_features);

    const std::size_t rank_limit = std::min(n_features, center ? n_samples - 1 : n_samples);
    if (n_components == 0 || n_components > rank_limit)
    {
        throw std::invalid_argument("n_components is out of range");
    }

    PcaFitResult result;
    result.n_samples = n_samples;
    result.n_features = n_features;
    result.n_components = n_components;
    result.center = center;
    result.mean.assign(n_features, 0.0);

    SvdResult svd;
    if (center)
    {
        auto centered = std::vector<double>(X.begin(), X.end());
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            double sum = 0.0;
            for (std::size_t sample = 0; sample < n_samples; ++sample)
            {
                sum += X[matrix_index(sample, feature, n_features)];
            }
            result.mean[feature] = sum / static_cast<double>(n_samples);
        }
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            for (std::size_t feature = 0; feature < n_features; ++feature)
            {
                centered[matrix_index(sample, feature, n_features)] -= result.mean[feature];
            }
        }
        svd = thin_svd(centered, n_samples, n_features);
    }
    else
    {
        svd = thin_svd(X, n_samples, n_features);
    }

    result.components.assign(svd.right_singular_vectors.begin(),
                             svd.right_singular_vectors.begin() +
                                 static_cast<std::ptrdiff_t>(n_components * n_features));
    canonicalize_vector_signs(result.components, n_components, n_features);
    result.offset = pca_offset(result.mean, result.components, n_components, n_features);

    const double dof = static_cast<double>(center ? n_samples - 1 : n_samples);
    result.singular_values.resize(n_components);
    result.explained_variance.resize(n_components);
    auto all_variance = std::vector<double>(rank_limit);
    for (std::size_t i = 0; i < rank_limit; ++i)
    {
        all_variance[i] = svd.singular_values[i] * svd.singular_values[i] / dof;
    }
    const double total_variance = std::accumulate(all_variance.begin(), all_variance.end(), 0.0);

    result.explained_variance_ratio.resize(n_components);
    for (std::size_t i = 0; i < n_components; ++i)
    {
        const double singular_squared = svd.singular_values[i] * svd.singular_values[i];
        result.singular_values[i] = svd.singular_values[i];
        result.explained_variance[i] = singular_squared / dof;
        result.explained_variance_ratio[i] =
            total_variance == 0.0 ? 0.0 : result.explained_variance[i] / total_variance;
    }
    return result;
}

void transform_pca(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                   std::span<const double> components, std::span<const double> offset,
                   std::size_t n_components, std::span<double> output)
{
    if (n_samples == 0 || n_features == 0 || X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (n_components == 0 || components.size() != n_components * n_features ||
        offset.size() != n_components || output.size() != n_samples * n_components)
    {
        throw std::invalid_argument("PCA state shape is invalid");
    }
    if (detail::spans_overlap(X, output) || detail::spans_overlap(components, output) ||
        detail::spans_overlap(offset, output))
    {
        throw std::invalid_argument("PCA X and output must not overlap");
    }

    transform_pca_unchecked(
        X, n_samples, n_features, components, offset, n_components, output,
        std::any_of(offset.begin(), offset.end(), [](double value) { return value != 0.0; }));
}

} // namespace neurale::models
