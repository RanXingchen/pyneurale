/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace neurale::models
{

enum class DistanceMetric
{
    SquaredEuclidean,
    Euclidean,
};

struct KnnResult
{
    std::size_t n_samples{};
    std::size_t k{};
    std::vector<std::int64_t> indices;
    std::vector<double> distances;
};

// Exact brute-force KNN for sample-major contiguous X.
//
// X must contain n_samples * n_features finite values. output_indices and
// output_distances must both contain n_samples * k values and must not overlap
// X or each other. Neighbors are ordered by squared Euclidean distance and
// then by row index. include_self=true guarantees that each row's own index is
// present in its result set; include_self=false excludes only the same row
// index, not duplicate rows. Invalid shapes and metrics throw
// std::invalid_argument; oversized outputs throw std::length_error.
void knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features, std::size_t k,
         DistanceMetric metric, bool include_self, std::span<std::int64_t> output_indices,
         std::span<double> output_distances);

// Owning overload with the same X, ordering, metric, and include_self
// contract as the output-span overload.
KnnResult knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
              std::size_t k, DistanceMetric metric, bool include_self);

} // namespace neurale::models
