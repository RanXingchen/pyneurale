/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/neighbors.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace neurale::models::cuda
{

inline constexpr std::size_t max_knn_neighbors = 64;

void knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features, std::size_t k,
         DistanceMetric metric, bool include_self, int device_id,
         std::span<std::int64_t> output_indices, std::span<double> output_distances);

} // namespace neurale::models::cuda
