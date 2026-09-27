/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/neighbors.h>

#include "numeric_utils.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace neurale::models
{
namespace
{

using detail::absolute_difference;
using detail::checked_product;
using detail::less_magnitude;
using detail::Magnitude;
using detail::magnitude_ratio;
using detail::matrix_index;
using detail::saturating_product;

#ifdef _OPENMP
constexpr std::size_t kParallelWorkThreshold = 1'000'000;
#endif

struct DistanceKey
{
    double squared{};
    double mantissa{};
    int exponent{};
    bool uses_scaled_key{};

    static DistanceKey zero() noexcept
    {
        return DistanceKey{};
    }

    static DistanceKey finite(double squared_distance) noexcept
    {
        return DistanceKey{.squared = squared_distance};
    }

    static DistanceKey scaled(double squared_distance, double normalized_mantissa,
                              int normalized_exponent) noexcept
    {
        return DistanceKey{
            .squared = squared_distance,
            .mantissa = normalized_mantissa,
            .exponent = normalized_exponent,
            .uses_scaled_key = true,
        };
    }
};

struct Neighbor
{
    DistanceKey distance{};
    std::int64_t idx{};
};

struct KnnProblem
{
    std::size_t n_samples{};
    std::size_t n_features{};
    std::size_t k{};
    std::size_t output_size{};
    DistanceMetric metric{};
    bool include_self{};
};

void require_finite(std::span<const double> values)
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("X must contain finite values");
        }
    }
}

std::size_t max_supported_samples() noexcept
{
    const auto ptrdiff_max =
        static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto int64_max = static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max());
    return static_cast<std::size_t>(std::min(ptrdiff_max, int64_max));
}

void require_valid_metric(DistanceMetric metric)
{
    switch (metric)
    {
    case DistanceMetric::SquaredEuclidean:
    case DistanceMetric::Euclidean:
        return;
    }
    throw std::invalid_argument("metric is invalid");
}

KnnProblem make_knn_problem(std::span<const double> X, std::size_t n_samples,
                            std::size_t n_features, std::size_t k, DistanceMetric metric,
                            bool include_self)
{
    const std::size_t X_size = checked_product(n_samples, n_features, "X shape is too large");
    if (n_samples == 0 || n_features == 0 || X.size() != X_size)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (n_samples > max_supported_samples())
    {
        throw std::length_error("too many samples");
    }
    if (k == 0)
    {
        throw std::invalid_argument("k must be positive");
    }
    const std::size_t max_neighbors = include_self ? n_samples : n_samples - 1;
    if (k > max_neighbors)
    {
        throw std::invalid_argument("k exceeds the available neighbor count");
    }
    const std::size_t output_size = checked_product(n_samples, k, "KNN output is too large");
    require_valid_metric(metric);
    require_finite(X);
    return KnnProblem{
        .n_samples = n_samples,
        .n_features = n_features,
        .k = k,
        .output_size = output_size,
        .metric = metric,
        .include_self = include_self,
    };
}

void require_knn_output(std::span<const double> X, std::span<const std::int64_t> output_indices,
                        std::span<const double> output_distances, std::size_t output_size)
{
    if (output_indices.size() != output_size || output_distances.size() != output_size)
    {
        throw std::invalid_argument("KNN output shape is invalid");
    }
    if (detail::spans_overlap(X, output_indices) || detail::spans_overlap(X, output_distances) ||
        detail::spans_overlap(output_indices, output_distances))
    {
        throw std::invalid_argument("KNN X and output buffers must not overlap");
    }
}

DistanceKey scaled_squared_distance(const double* left, const double* right, std::size_t n_features)
{
    Magnitude scale{};
    double sum_squares = 1.0;

    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        const Magnitude delta = absolute_difference(left[feature], right[feature]);
        if (delta.mantissa == 0.0)
        {
            continue;
        }
        if (scale.mantissa == 0.0 || less_magnitude(scale, delta))
        {
            const double ratio = scale.mantissa == 0.0 ? 0.0 : magnitude_ratio(scale, delta);
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = delta;
        }
        else
        {
            const double ratio = magnitude_ratio(delta, scale);
            sum_squares += ratio * ratio;
        }
    }

    if (scale.mantissa == 0.0)
    {
        return DistanceKey::zero();
    }

    int adjustment = 0;
    const double mantissa = std::frexp(scale.mantissa * scale.mantissa * sum_squares, &adjustment);
    const int exponent = 2 * scale.exponent + adjustment;
    return DistanceKey::scaled(std::scalbn(mantissa, exponent), mantissa, exponent);
}

DistanceKey squared_distance(const double* left, const double* right, std::size_t n_features)
{
    double squared = 0.0;

    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        const double delta = left[feature] - right[feature];
        squared += delta * delta;
    }

    if (std::isnormal(squared))
    {
        return DistanceKey::finite(squared);
    }
    if (squared == 0.0 && std::equal(left, left + n_features, right))
    {
        return DistanceKey::zero();
    }
    return scaled_squared_distance(left, right, n_features);
}

DistanceKey normalized_distance(const DistanceKey& key)
{
    if (key.uses_scaled_key || key.squared == 0.0)
    {
        return key;
    }
    int exponent = 0;
    const double mantissa = std::frexp(key.squared, &exponent);
    return DistanceKey::scaled(key.squared, mantissa, exponent);
}

int compare_distance(const DistanceKey& left, const DistanceKey& right)
{
    if (!left.uses_scaled_key && !right.uses_scaled_key)
    {
        if (left.squared < right.squared)
        {
            return -1;
        }
        if (right.squared < left.squared)
        {
            return 1;
        }
        return 0;
    }

    const DistanceKey normalized_left = normalized_distance(left);
    const DistanceKey normalized_right = normalized_distance(right);
    if (normalized_left.mantissa == 0.0 || normalized_right.mantissa == 0.0)
    {
        if (normalized_left.mantissa == normalized_right.mantissa)
        {
            return 0;
        }
        return normalized_left.mantissa == 0.0 ? -1 : 1;
    }
    if (normalized_left.exponent != normalized_right.exponent)
    {
        return normalized_left.exponent < normalized_right.exponent ? -1 : 1;
    }
    if (normalized_left.mantissa < normalized_right.mantissa)
    {
        return -1;
    }
    if (normalized_right.mantissa < normalized_left.mantissa)
    {
        return 1;
    }
    return 0;
}

double euclidean_output_distance(const DistanceKey& distance)
{
    const DistanceKey normalized = normalized_distance(distance);
    if (normalized.mantissa == 0.0)
    {
        return 0.0;
    }
    if (normalized.exponent % 2 == 0)
    {
        return std::scalbn(std::sqrt(normalized.mantissa), normalized.exponent / 2);
    }
    return std::scalbn(std::sqrt(normalized.mantissa * 2.0), (normalized.exponent - 1) / 2);
}

double format_output_distance(const DistanceKey& distance, DistanceMetric metric)
{
    switch (metric)
    {
    case DistanceMetric::SquaredEuclidean:
        return distance.squared;
    case DistanceMetric::Euclidean:
        return euclidean_output_distance(distance);
    }
    throw std::invalid_argument("metric is invalid");
}

bool is_better_neighbor(const Neighbor& left, const Neighbor& right)
{
    const int distance_order = compare_distance(left.distance, right.distance);
    if (distance_order != 0)
    {
        return distance_order < 0;
    }
    return left.idx < right.idx;
}

bool neighbor_heap_order(const Neighbor& left, const Neighbor& right)
{
    return is_better_neighbor(left, right);
}

void heap_insert(std::span<Neighbor> heap, std::size_t& heap_size, Neighbor value)
{
    heap[heap_size] = value;
    ++heap_size;
    auto active = heap.first(heap_size);
    std::push_heap(active.begin(), active.end(), neighbor_heap_order);
}

void heap_replace_top(std::span<Neighbor> heap, std::size_t heap_size, Neighbor value)
{
    auto active = heap.first(heap_size);
    std::pop_heap(active.begin(), active.end(), neighbor_heap_order);
    active.back() = value;
    std::push_heap(active.begin(), active.end(), neighbor_heap_order);
}

void sort_neighbors(std::span<Neighbor> heap, std::size_t heap_size)
{
    auto active = heap.first(heap_size);
    std::sort(active.begin(), active.end(), is_better_neighbor);
}

std::size_t knn_thread_count(std::size_t n_samples, std::size_t work_size) noexcept
{
#ifdef _OPENMP
    if (work_size < kParallelWorkThreshold || n_samples < 2)
    {
        return 1;
    }
    return std::min(n_samples, static_cast<std::size_t>(omp_get_max_threads()));
#else
    (void)n_samples;
    (void)work_size;
    return 1;
#endif
}

std::size_t knn_thread_index() noexcept
{
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

void write_self_neighbor(std::size_t sample, std::size_t k, std::span<std::int64_t> output_indices,
                         std::span<double> output_distances)
{
    const std::size_t offset = matrix_index(sample, 0, k);
    output_indices[offset] = static_cast<std::int64_t>(sample);
    output_distances[offset] = 0.0;
}

void consider_candidate(const double* sample_row, const double* candidate_row,
                        std::size_t candidate, std::size_t n_features, std::size_t n_search,
                        std::span<Neighbor> neighbors, std::size_t& heap_size)
{
    const auto candidate_idx = static_cast<std::int64_t>(candidate);
    const Neighbor candidate_neighbor{
        squared_distance(sample_row, candidate_row, n_features),
        candidate_idx,
    };
    if (heap_size < n_search)
    {
        heap_insert(neighbors, heap_size, candidate_neighbor);
        return;
    }
    if (is_better_neighbor(candidate_neighbor, neighbors.front()))
    {
        heap_replace_top(neighbors, heap_size, candidate_neighbor);
    }
}

template <bool IncludeSelf>
void knn_row(std::span<const double> X, std::size_t sample, std::size_t n_samples,
             std::size_t n_features, std::size_t k, DistanceMetric metric,
             std::span<Neighbor> neighbors, std::span<std::int64_t> output_indices,
             std::span<double> output_distances)
{
    std::size_t heap_size = 0;
    const double* sample_row = X.data() + sample * n_features;
    const std::size_t n_search = IncludeSelf ? k - 1 : k;

    if (n_search == 0)
    {
        write_self_neighbor(sample, k, output_indices, output_distances);
        return;
    }

    for (std::size_t candidate = 0; candidate < sample; ++candidate)
    {
        consider_candidate(sample_row, X.data() + candidate * n_features, candidate, n_features,
                           n_search, neighbors, heap_size);
    }
    for (std::size_t candidate = sample + 1; candidate < n_samples; ++candidate)
    {
        consider_candidate(sample_row, X.data() + candidate * n_features, candidate, n_features,
                           n_search, neighbors, heap_size);
    }

    if constexpr (IncludeSelf)
    {
        neighbors[heap_size] = Neighbor{
            .distance = DistanceKey::zero(),
            .idx = static_cast<std::int64_t>(sample),
        };
        ++heap_size;
    }
    sort_neighbors(neighbors, heap_size);

    for (std::size_t i = 0; i < k; ++i)
    {
        output_indices[matrix_index(sample, i, k)] = neighbors[i].idx;
        output_distances[matrix_index(sample, i, k)] =
            format_output_distance(neighbors[i].distance, metric);
    }
}

template <bool IncludeSelf>
void knn_rows(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
              std::size_t k, DistanceMetric metric, std::span<std::int64_t> output_indices,
              std::span<double> output_distances)
{
    const std::size_t work_size =
        saturating_product(saturating_product(n_samples, n_samples), n_features);
    const std::size_t n_threads = knn_thread_count(n_samples, work_size);
    auto workspaces =
        std::vector<Neighbor>(checked_product(n_threads, k, "KNN workspace is too large"));

    if (n_threads == 1)
    {
        auto neighbors = std::span<Neighbor>(workspaces.data(), k);
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            knn_row<IncludeSelf>(X, sample, n_samples, n_features, k, metric, neighbors,
                                 output_indices, output_distances);
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(n_threads))
#endif
    {
        const std::size_t thread = knn_thread_index();
        auto neighbors = std::span<Neighbor>(workspaces.data() + thread * k, k);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_samples); ++i)
        {
            knn_row<IncludeSelf>(X, static_cast<std::size_t>(i), n_samples, n_features, k, metric,
                                 neighbors, output_indices, output_distances);
        }
    }
}

void compute_knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                 std::size_t k, bool include_self, DistanceMetric metric,
                 std::span<std::int64_t> output_indices, std::span<double> output_distances)
{
    if (include_self)
    {
        knn_rows<true>(X, n_samples, n_features, k, metric, output_indices, output_distances);
    }
    else
    {
        knn_rows<false>(X, n_samples, n_features, k, metric, output_indices, output_distances);
    }
}

KnnProblem validate_knn_problem(std::span<const double> X, std::size_t n_samples,
                                std::size_t n_features, std::size_t k, DistanceMetric metric,
                                bool include_self)
{
    return make_knn_problem(X, n_samples, n_features, k, metric, include_self);
}

void run_knn(std::span<const double> X, const KnnProblem& problem,
             std::span<std::int64_t> output_indices, std::span<double> output_distances)
{
    compute_knn(X, problem.n_samples, problem.n_features, problem.k, problem.include_self,
                problem.metric, output_indices, output_distances);
}

} // namespace

void knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features, std::size_t k,
         DistanceMetric metric, bool include_self, std::span<std::int64_t> output_indices,
         std::span<double> output_distances)
{
    const KnnProblem problem =
        validate_knn_problem(X, n_samples, n_features, k, metric, include_self);
    require_knn_output(X, output_indices, output_distances, problem.output_size);
    run_knn(X, problem, output_indices, output_distances);
}

KnnResult knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
              std::size_t k, DistanceMetric metric, bool include_self)
{
    const KnnProblem problem =
        validate_knn_problem(X, n_samples, n_features, k, metric, include_self);
    KnnResult result;
    result.n_samples = problem.n_samples;
    result.k = problem.k;
    result.indices.resize(problem.output_size);
    result.distances.resize(problem.output_size);
    run_knn(X, problem, result.indices, result.distances);
    return result;
}

} // namespace neurale::models
