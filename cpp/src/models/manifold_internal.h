/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/manifold.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace neurale::models::detail
{

void require_lpp_fit_problem(std::span<const double> X, std::size_t n_samples,
                             std::size_t n_features, std::size_t n_components,
                             std::size_t n_neighbors);

LppFitResult fit_lpp_from_neighbors_unchecked(std::span<const double> X,
                                              std::span<const std::int64_t> neighbor_indices,
                                              std::size_t n_samples, std::size_t n_features,
                                              std::size_t n_components, std::size_t n_neighbors,
                                              LppProjectionMethod method);

} // namespace neurale::models::detail
