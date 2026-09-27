/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace neurale::sorting
{

inline constexpr std::size_t valley_seeking_max_observations = 100'000;
inline constexpr std::size_t valley_seeking_max_features = 1'024;
inline constexpr std::size_t valley_seeking_max_labels = 100'000;
inline constexpr std::size_t valley_seeking_max_neighbor_pairs = 4'000'000;

/// Hard workspace limits checked before or during neighborhood construction.
struct ValleySeekingLimits
{
    std::size_t max_observations{valley_seeking_max_observations};
    std::size_t max_features{valley_seeking_max_features};
    std::size_t max_labels{valley_seeking_max_labels};
    std::size_t max_neighbor_pairs{valley_seeking_max_neighbor_pairs};
};

/// Reason that the bounded synchronous iteration stopped.
enum class ValleySeekingTermination
{
    Empty,
    Converged,
    MaxIterations,
};

/// Deterministic labels, density proxy, termination, and workspace metadata.
struct ValleySeekingResult
{
    std::vector<std::int64_t> labels;
    std::vector<std::size_t> neighbor_counts;
    std::vector<std::int64_t> label_order;
    double radius{};
    std::size_t iterations{};
    bool converged{};
    ValleySeekingTermination termination{ValleySeekingTermination::Empty};
    std::size_t neighbor_pairs{};
    std::size_t workspace_bytes{};
};

/// Refine initial labels using fixed-radius synchronous neighbor voting.
/// @param features Contiguous observation-major float64 values.
/// @param n_observations Number of matrix rows.
/// @param n_features Number of matrix columns.
/// @param initial_labels One int64 label per observation.
/// @param radius Positive inclusive Euclidean neighborhood radius.
/// @param max_iterations Positive synchronous update limit.
/// @param limits Explicit bounds for inputs and compressed adjacency workspace.
/// @return Deterministic labels and execution metadata.
ValleySeekingResult valley_seeking(std::span<const double> features, std::size_t n_observations,
                                   std::size_t n_features,
                                   std::span<const std::int64_t> initial_labels, double radius,
                                   std::size_t max_iterations = 1000,
                                   const ValleySeekingLimits& limits = {});

} // namespace neurale::sorting
