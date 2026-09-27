/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/sorting/valley_seeking.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace neurale::sorting
{
namespace
{

std::size_t checked_product(std::size_t left, std::size_t right, const char* message)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::length_error(message);
    }
    return left * right;
}

void validate_limits(const ValleySeekingLimits& limits)
{
    if (limits.max_observations == 0 || limits.max_features == 0 || limits.max_labels == 0 ||
        limits.max_neighbor_pairs == 0)
    {
        throw std::invalid_argument("Valley Seeking limits must all be positive");
    }
    if (limits.max_observations > valley_seeking_max_observations ||
        limits.max_features > valley_seeking_max_features ||
        limits.max_labels > valley_seeking_max_labels ||
        limits.max_neighbor_pairs > valley_seeking_max_neighbor_pairs)
    {
        throw std::invalid_argument(
            "Valley Seeking limits may tighten but not exceed compiled maxima");
    }
}

bool within_radius(std::span<const double> features, std::size_t left, std::size_t right,
                   std::size_t n_features, double radius) noexcept
{
    double squared_scaled_distance = 0.0;
    const auto left_offset = left * n_features;
    const auto right_offset = right * n_features;
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        const auto difference =
            std::abs(features[left_offset + feature] - features[right_offset + feature]);
        if (!std::isfinite(difference) || difference > radius)
        {
            return false;
        }
        const auto normalized = difference / radius;
        squared_scaled_distance = std::fma(normalized, normalized, squared_scaled_distance);
        if (squared_scaled_distance > 1.0)
        {
            return false;
        }
    }
    return true;
}

struct NeighborhoodWorkspace
{
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> neighbors;
    std::size_t n_pairs{};
};

NeighborhoodWorkspace build_neighborhoods(std::span<const double> features,
                                          std::size_t n_observations, std::size_t n_features,
                                          double radius, std::size_t max_neighbor_pairs)
{
    NeighborhoodWorkspace workspace;
    workspace.offsets.assign(n_observations + 1, 0);
    for (std::size_t left = 0; left < n_observations; ++left)
    {
        for (std::size_t right = left + 1; right < n_observations; ++right)
        {
            if (!within_radius(features, left, right, n_features, radius))
            {
                continue;
            }
            if (workspace.n_pairs == max_neighbor_pairs)
            {
                throw std::length_error("Valley Seeking neighbor-pair limit exceeded");
            }
            ++workspace.n_pairs;
            ++workspace.offsets[left + 1];
            ++workspace.offsets[right + 1];
        }
    }

    for (std::size_t i = 0; i < n_observations; ++i)
    {
        workspace.offsets[i + 1] += workspace.offsets[i];
    }
    workspace.neighbors.resize(
        checked_product(workspace.n_pairs, 2, "Valley Seeking adjacency is too large"));
    auto positions = workspace.offsets;
    for (std::size_t left = 0; left < n_observations; ++left)
    {
        for (std::size_t right = left + 1; right < n_observations; ++right)
        {
            if (within_radius(features, left, right, n_features, radius))
            {
                workspace.neighbors[positions[left]++] = right;
                workspace.neighbors[positions[right]++] = left;
            }
        }
    }
    return workspace;
}

std::size_t workspace_bytes(const NeighborhoodWorkspace& neighborhoods, std::size_t n_observations,
                            std::size_t n_labels)
{
    auto total = checked_product(neighborhoods.offsets.capacity(), sizeof(std::size_t),
                                 "Valley Seeking workspace size overflow");
    const auto neighbor_bytes =
        checked_product(neighborhoods.neighbors.capacity(), sizeof(std::size_t),
                        "Valley Seeking workspace size overflow");
    const auto observation_arrays = checked_product(
        checked_product(n_observations, 2, "Valley Seeking workspace size overflow"),
        sizeof(std::size_t), "Valley Seeking workspace size overflow");
    const auto label_arrays =
        checked_product(checked_product(n_labels, 2, "Valley Seeking workspace size overflow"),
                        sizeof(std::size_t), "Valley Seeking workspace size overflow");
    if (neighbor_bytes > std::numeric_limits<std::size_t>::max() - total ||
        observation_arrays > std::numeric_limits<std::size_t>::max() - total - neighbor_bytes ||
        label_arrays >
            std::numeric_limits<std::size_t>::max() - total - neighbor_bytes - observation_arrays)
    {
        throw std::length_error("Valley Seeking workspace size overflow");
    }
    return total + neighbor_bytes + observation_arrays + label_arrays;
}

} // namespace

ValleySeekingResult valley_seeking(std::span<const double> features, std::size_t n_observations,
                                   std::size_t n_features,
                                   std::span<const std::int64_t> initial_labels, double radius,
                                   std::size_t max_iterations, const ValleySeekingLimits& limits)
{
    validate_limits(limits);
    if (n_observations > limits.max_observations)
    {
        throw std::length_error("Valley Seeking observation limit exceeded");
    }
    if (n_features == 0 || n_features > limits.max_features)
    {
        throw std::length_error("Valley Seeking feature count must be positive and within limit");
    }
    const auto expected_size =
        checked_product(n_observations, n_features, "Valley Seeking input shape is too large");
    if (features.size() != expected_size)
    {
        throw std::invalid_argument("Valley Seeking feature span does not match its shape");
    }
    if (initial_labels.size() != n_observations)
    {
        throw std::invalid_argument("Valley Seeking requires one initial label per observation");
    }
    if (!std::isfinite(radius) || radius <= 0.0)
    {
        throw std::invalid_argument("Valley Seeking radius must be finite and positive");
    }
    if (max_iterations == 0)
    {
        throw std::invalid_argument("Valley Seeking max_iterations must be positive");
    }
    if (!std::all_of(features.begin(), features.end(),
                     [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("Valley Seeking features must be finite");
    }

    ValleySeekingResult result;
    result.radius = radius;
    if (n_observations == 0)
    {
        result.converged = true;
        result.termination = ValleySeekingTermination::Empty;
        return result;
    }

    result.label_order.assign(initial_labels.begin(), initial_labels.end());
    std::sort(result.label_order.begin(), result.label_order.end());
    result.label_order.erase(std::unique(result.label_order.begin(), result.label_order.end()),
                             result.label_order.end());
    if (result.label_order.size() > limits.max_labels)
    {
        throw std::length_error("Valley Seeking label limit exceeded");
    }

    auto neighborhoods = build_neighborhoods(features, n_observations, n_features, radius,
                                             limits.max_neighbor_pairs);
    result.neighbor_pairs = neighborhoods.n_pairs;
    result.neighbor_counts.resize(n_observations);
    for (std::size_t observation = 0; observation < n_observations; ++observation)
    {
        result.neighbor_counts[observation] =
            neighborhoods.offsets[observation + 1] - neighborhoods.offsets[observation];
    }

    std::vector<std::size_t> current(n_observations);
    std::vector<std::size_t> following(n_observations);
    for (std::size_t observation = 0; observation < n_observations; ++observation)
    {
        current[observation] = static_cast<std::size_t>(
            std::lower_bound(result.label_order.begin(), result.label_order.end(),
                             initial_labels[observation]) -
            result.label_order.begin());
    }
    std::vector<std::size_t> votes(result.label_order.size(), 0);
    std::vector<std::size_t> touched;
    touched.reserve(result.label_order.size());
    result.workspace_bytes =
        workspace_bytes(neighborhoods, n_observations, result.label_order.size());

    for (std::size_t iteration = 1; iteration <= max_iterations; ++iteration)
    {
        bool changed = false;
        for (std::size_t observation = 0; observation < n_observations; ++observation)
        {
            touched.clear();
            std::size_t most_votes = 0;
            for (auto i = neighborhoods.offsets[observation];
                 i < neighborhoods.offsets[observation + 1]; ++i)
            {
                const auto label = current[neighborhoods.neighbors[i]];
                if (votes[label] == 0)
                {
                    touched.push_back(label);
                }
                most_votes = std::max(most_votes, ++votes[label]);
            }

            const auto incumbent = current[observation];
            auto winner = incumbent;
            if (votes[incumbent] != most_votes)
            {
                winner = result.label_order.size();
                for (const auto label : touched)
                {
                    if (votes[label] == most_votes)
                    {
                        winner = std::min(winner, label);
                    }
                }
            }
            following[observation] = winner;
            changed = changed || winner != incumbent;
            for (const auto label : touched)
            {
                votes[label] = 0;
            }
        }

        result.iterations = iteration;
        if (!changed)
        {
            result.converged = true;
            result.termination = ValleySeekingTermination::Converged;
            break;
        }
        current.swap(following);
    }
    if (!result.converged)
    {
        result.termination = ValleySeekingTermination::MaxIterations;
    }

    result.labels.resize(n_observations);
    for (std::size_t observation = 0; observation < n_observations; ++observation)
    {
        result.labels[observation] = result.label_order[current[observation]];
    }
    return result;
}

} // namespace neurale::sorting
