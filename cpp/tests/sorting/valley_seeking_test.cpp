/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/sorting/valley_seeking.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ns = neurale::sorting;

namespace
{

void matches_fixed_neighborhood_and_tie_contract()
{
    const std::vector<double> features = {0.0, 0.0, -1.0, 0.0, 1.0, 0.0};
    const std::vector<std::int64_t> labels = {8, 5, 2};
    const auto result = ns::valley_seeking(features, 3, 2, labels, 1.0, 1);

    assert(result.labels == std::vector<std::int64_t>({2, 8, 8}));
    assert(result.neighbor_counts == std::vector<std::size_t>({2, 1, 1}));
    assert(result.label_order == std::vector<std::int64_t>({2, 5, 8}));
    assert(result.neighbor_pairs == 2);
    assert(!result.converged);
    assert(result.termination == ns::ValleySeekingTermination::MaxIterations);
}

void preserves_incumbent_ties_and_isolated_labels()
{
    const std::vector<double> features = {0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 10.0, 10.0};
    const std::vector<std::int64_t> labels = {7, 3, 7, -4};
    const auto result = ns::valley_seeking(features, 4, 2, labels, 1.0, 1);

    assert(result.labels[0] == 7);
    assert(result.labels[3] == -4);
    assert(result.neighbor_counts[3] == 0);
}

void reports_empty_convergence_and_deterministic_cycles()
{
    const auto empty = ns::valley_seeking({}, 0, 2, {}, 1.0);
    assert(empty.labels.empty());
    assert(empty.converged);
    assert(empty.iterations == 0);
    assert(empty.termination == ns::ValleySeekingTermination::Empty);

    const std::vector<double> features = {0.0, 0.5};
    const std::vector<std::int64_t> labels = {0, 1};
    const auto first = ns::valley_seeking(features, 2, 1, labels, 1.0, 3);
    const auto second = ns::valley_seeking(features, 2, 1, labels, 1.0, 3);
    assert(first.labels == std::vector<std::int64_t>({1, 0}));
    assert(first.labels == second.labels);
    assert(first.neighbor_counts == second.neighbor_counts);
    assert(!first.converged);
}

void freezes_fused_multiply_add_radius_rounding()
{
    const std::vector<double> features = {
        0.0,
        0.0,
        0.68418360801139777,
        0.72930980421800595,
    };
    const std::vector<std::int64_t> labels = {0, 1};
    const auto result = ns::valley_seeking(features, 2, 2, labels, 1.0, 1);

    assert(result.neighbor_counts == std::vector<std::size_t>({0, 0}));
    assert(result.labels == labels);
}

void enforces_shape_value_and_explicit_workspace_limits()
{
    const std::vector<double> one_feature = {0.0, 0.0, 0.0};
    const std::vector<std::int64_t> three_labels = {0, 1, 2};
    ns::ValleySeekingLimits limits{
        .max_observations = 2,
        .max_features = 1,
        .max_labels = 2,
        .max_neighbor_pairs = 1,
    };

    bool rejected = false;
    try
    {
        ns::valley_seeking({}, 3, 1, {}, 1.0, 1, limits);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    assert(rejected);

    auto loose_limits = ns::ValleySeekingLimits{};
    ++loose_limits.max_observations;
    rejected = false;
    try
    {
        ns::valley_seeking({}, 0, 1, {}, 1.0, 1, loose_limits);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);

    limits.max_observations = 3;
    rejected = false;
    try
    {
        ns::valley_seeking(one_feature, 3, 1, three_labels, 1.0, 1, limits);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    assert(rejected); // unique-label bound is checked before neighborhood construction

    limits.max_labels = 3;
    rejected = false;
    try
    {
        ns::valley_seeking(one_feature, 3, 1, three_labels, 1.0, 1, limits);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    assert(rejected); // three neighbor pairs exceed the configured bound of one

    auto nonfinite = one_feature;
    nonfinite[1] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try
    {
        ns::valley_seeking(nonfinite, 3, 1, three_labels, 1.0, 1);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main()
{
    matches_fixed_neighborhood_and_tie_contract();
    preserves_incumbent_ties_and_isolated_labels();
    reports_empty_convergence_and_deterministic_cycles();
    freezes_fused_multiply_add_radius_rounding();
    enforces_shape_value_and_explicit_workspace_limits();
    return 0;
}
