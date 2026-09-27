/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/manifold.h>

#include <cstddef>
#include <span>

namespace neurale::models::cuda
{

LppFitResult fit_lpp(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t n_components, std::size_t n_neighbors, DistanceMetric metric,
                     LppProjectionMethod method, int device_id);

} // namespace neurale::models::cuda
