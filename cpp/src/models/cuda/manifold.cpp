/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/cuda/manifold.h>

#include <neurale/models/cuda/neighbors.h>

#include "manifold_internal.h"
#include "span_utils.h"

#include <cstdint>
#include <vector>

namespace neurale::models::cuda
{

LppFitResult fit_lpp(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t n_components, std::size_t n_neighbors, DistanceMetric metric,
                     LppProjectionMethod method, int device_id)
{
    detail::require_lpp_fit_problem(X, n_samples, n_features, n_components, n_neighbors);
    const std::size_t output_size =
        detail::checked_product(n_samples, n_neighbors, "LPP neighbor output is too large");
    auto indices = std::vector<std::int64_t>(output_size);
    auto distances = std::vector<double>(output_size);
    knn(X, n_samples, n_features, n_neighbors, metric, true, device_id, indices, distances);
    return detail::fit_lpp_from_neighbors_unchecked(X, indices, n_samples, n_features, n_components,
                                                    n_neighbors, method);
}

} // namespace neurale::models::cuda
