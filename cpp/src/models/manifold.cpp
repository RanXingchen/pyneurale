/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/manifold.h>

#include "linalg.h"
#include "manifold_internal.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurale::models
{
namespace
{

using detail::matrix_index;

constexpr double kOppRegularization = 1e-6;

struct GraphForms
{
    std::vector<double> degree;
    std::vector<double> laplacian;
};

GraphForms graph_forms(std::span<const double> X, std::span<const std::int64_t> neighbor_indices,
                       std::size_t n_samples, std::size_t n_features, std::size_t n_neighbors)
{
    GraphForms result;
    result.degree.assign(n_samples, 0.0);
    auto laplacian_product = std::vector<double>(X.size(), 0.0);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t neighbor = 0; neighbor < n_neighbors; ++neighbor)
        {
            const auto other = static_cast<std::size_t>(
                neighbor_indices[matrix_index(sample, neighbor, n_neighbors)]);
            result.degree[sample] += 0.5;
            result.degree[other] += 0.5;
        }
    }
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            laplacian_product[matrix_index(sample, feature, n_features)] =
                result.degree[sample] * X[matrix_index(sample, feature, n_features)];
        }
    }
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t neighbor = 0; neighbor < n_neighbors; ++neighbor)
        {
            const auto other = static_cast<std::size_t>(
                neighbor_indices[matrix_index(sample, neighbor, n_neighbors)]);
            for (std::size_t feature = 0; feature < n_features; ++feature)
            {
                laplacian_product[matrix_index(sample, feature, n_features)] -=
                    0.5 * X[matrix_index(other, feature, n_features)];
                laplacian_product[matrix_index(other, feature, n_features)] -=
                    0.5 * X[matrix_index(sample, feature, n_features)];
            }
        }
    }

    auto input_transpose = std::vector<double>(n_features * n_samples);
    auto laplacian_transpose = std::vector<double>(n_features * n_samples);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            input_transpose[matrix_index(feature, sample, n_samples)] =
                X[matrix_index(sample, feature, n_features)];
            laplacian_transpose[matrix_index(feature, sample, n_samples)] =
                laplacian_product[matrix_index(sample, feature, n_features)];
        }
    }
    result.laplacian.resize(n_features * n_features);
    gemm_nt(input_transpose, laplacian_transpose, n_features, n_features, n_samples,
            result.laplacian);
    return result;
}

std::vector<double> degree_form(std::span<const double> X, std::span<const double> degree,
                                std::size_t n_samples, std::size_t n_features, bool center)
{
    auto mean = std::vector<double>(n_features, 0.0);
    if (center)
    {
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            for (std::size_t feature = 0; feature < n_features; ++feature)
            {
                mean[feature] += X[matrix_index(sample, feature, n_features)];
            }
        }
        for (double& value : mean)
        {
            value /= static_cast<double>(n_samples);
        }
    }

    auto weighted_transpose = std::vector<double>(n_features * n_samples);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const double scale = std::sqrt(degree[sample]);
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            weighted_transpose[matrix_index(feature, sample, n_samples)] =
                scale * (X[matrix_index(sample, feature, n_features)] - mean[feature]);
        }
    }
    auto result = std::vector<double>(n_features * n_features);
    gemm_nt(weighted_transpose, weighted_transpose, n_features, n_features, n_samples, result);
    return result;
}

std::vector<double> select_components(const GeneralizedEigenResult& eigen, std::size_t n_features,
                                      std::size_t n_components, LppProjectionMethod method)
{
    const std::size_t rank = eigen.values.size();
    auto selected = std::vector<std::size_t>();
    selected.reserve(n_components);
    if (method == LppProjectionMethod::Opp)
    {
        if (rank < n_components)
        {
            throw std::runtime_error("OPP has fewer usable directions than n_components");
        }
        for (std::size_t i = n_components; i > 0; --i)
        {
            selected.push_back(i - 1);
        }
    }
    else
    {
        const double scale =
            std::max(std::abs(eigen.values.front()), std::abs(eigen.values.back()));
        const double threshold =
            scale * std::numeric_limits<double>::epsilon() * static_cast<double>(rank);
        for (std::size_t i = rank; i-- > 0 && selected.size() < n_components;)
        {
            if (eigen.values[i] > threshold)
            {
                selected.push_back(i);
            }
        }
        if (selected.size() != n_components)
        {
            throw std::runtime_error("LPP has fewer positive directions than n_components");
        }
    }

    auto components = std::vector<double>(n_components * n_features);
    for (std::size_t row = 0; row < n_components; ++row)
    {
        const std::size_t source = selected[row];
        std::copy_n(eigen.vectors.begin() + static_cast<std::ptrdiff_t>(source * n_features),
                    n_features, components.begin() + static_cast<std::ptrdiff_t>(row * n_features));
    }
    canonicalize_vector_signs(components, n_components, n_features);
    return components;
}

void require_model_state(const LppModelState& state)
{
    if (state.n_features == 0 || state.n_components == 0 ||
        state.components.size() != state.n_components * state.n_features)
    {
        throw std::invalid_argument("LPP state shape is invalid");
    }
}

} // namespace

void detail::require_lpp_fit_problem(std::span<const double> X, std::size_t n_samples,
                                     std::size_t n_features, std::size_t n_components,
                                     std::size_t n_neighbors)
{
    if (n_samples < 2 || n_features == 0 || X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (n_components == 0 || n_components > n_features)
    {
        throw std::invalid_argument("n_components must be in [1, n_features]");
    }
    if (n_neighbors == 0 || n_neighbors > n_samples)
    {
        throw std::invalid_argument("n_neighbors must be in [1, n_samples]");
    }
}

LppModel::LppModel(LppModelState state) : state_(std::move(state))
{
    require_model_state(state_);
}

std::size_t LppModel::n_features() const noexcept
{
    return state_.n_features;
}

std::size_t LppModel::n_components() const noexcept
{
    return state_.n_components;
}

const std::vector<double>& LppModel::components() const noexcept
{
    return state_.components;
}

void LppModel::transform(std::span<const double> X, std::size_t n_samples,
                         std::span<double> output) const
{
    if (X.size() != n_samples * state_.n_features ||
        output.size() != n_samples * state_.n_components)
    {
        throw std::invalid_argument("transform shape is invalid");
    }
    if (detail::spans_overlap(X, output))
    {
        throw std::invalid_argument("X and out must not overlap");
    }
    gemm_nt_unchecked(X, state_.components, n_samples, state_.n_components, state_.n_features,
                      output);
}

LppFitResult fit_lpp(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t n_components, std::size_t n_neighbors, DistanceMetric metric,
                     LppProjectionMethod method)
{
    detail::require_lpp_fit_problem(X, n_samples, n_features, n_components, n_neighbors);
    const auto neighbors = knn(X, n_samples, n_features, n_neighbors, metric, true);
    return detail::fit_lpp_from_neighbors_unchecked(X, neighbors.indices, n_samples, n_features,
                                                    n_components, n_neighbors, method);
}

LppFitResult detail::fit_lpp_from_neighbors_unchecked(
    std::span<const double> X, std::span<const std::int64_t> neighbor_indices,
    std::size_t n_samples, std::size_t n_features, std::size_t n_components,
    std::size_t n_neighbors, LppProjectionMethod method)
{
    auto graph = graph_forms(X, neighbor_indices, n_samples, n_features, n_neighbors);

    std::vector<double> num;
    std::vector<double> den;
    if (method == LppProjectionMethod::Opp)
    {
        num = degree_form(X, graph.degree, n_samples, n_features, true);
        den = std::move(graph.laplacian);
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            den[matrix_index(feature, feature, n_features)] += kOppRegularization;
        }
    }
    else
    {
        num = std::move(graph.laplacian);
        den = degree_form(X, graph.degree, n_samples, n_features, false);
    }

    const auto eigen = generalized_symmetric_eigen(num, den, n_features);
    LppFitResult result;
    result.n_samples = n_samples;
    result.n_features = n_features;
    result.n_components = n_components;
    result.components = select_components(eigen, n_features, n_components, method);
    return result;
}

} // namespace neurale::models
