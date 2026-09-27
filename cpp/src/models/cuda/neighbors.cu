/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/cuda/neighbors.h>

#include "cuda_utils.cuh"
#include "span_utils.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__FAST_MATH__)
#error "CUDA KNN requires strict IEEE-754 floating-point semantics"
#endif

#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "CUDA KNN requires infinity and subnormal floating-point semantics"
#endif

namespace neurale::models::cuda
{
namespace
{

constexpr int kThreadsPerBlock = 128;
constexpr int kCachedFeatures = 64;

struct Magnitude
{
    double mantissa{};
    int exponent{};
};

struct DistanceKey
{
    double value{};
    int exponent{};
};

struct Neighbor
{
    double distance{};
    int exponent{};
    std::uint32_t idx{};
};

static_assert(sizeof(Neighbor) == 16);

__device__ Magnitude finite_magnitude(double value)
{
    int exponent = 0;
    return Magnitude{frexp(value, &exponent), exponent};
}

__device__ Magnitude add_magnitudes(double left, double right)
{
    if (left == 0.0)
    {
        return finite_magnitude(right);
    }
    if (right == 0.0)
    {
        return finite_magnitude(left);
    }
    int left_exponent = 0;
    int right_exponent = 0;
    const double left_mantissa = frexp(left, &left_exponent);
    const double right_mantissa = frexp(right, &right_exponent);
    int exponent = max(left_exponent, right_exponent);
    double mantissa = ldexp(left_mantissa, left_exponent - exponent) +
                      ldexp(right_mantissa, right_exponent - exponent);
    if (mantissa >= 1.0)
    {
        mantissa *= 0.5;
        ++exponent;
    }
    return Magnitude{mantissa, exponent};
}

__device__ Magnitude absolute_difference(double left, double right)
{
    if (left == right)
    {
        return Magnitude{};
    }
    const double left_abs = fabs(left);
    const double right_abs = fabs(right);
    if (signbit(left) != signbit(right))
    {
        return add_magnitudes(left_abs, right_abs);
    }
    return finite_magnitude(fabs(left_abs - right_abs));
}

__device__ bool less_magnitude(Magnitude left, Magnitude right)
{
    return left.exponent != right.exponent ? left.exponent < right.exponent
                                           : left.mantissa < right.mantissa;
}

__device__ double magnitude_ratio(Magnitude numerator, Magnitude denominator)
{
    return ldexp(numerator.mantissa / denominator.mantissa,
                 numerator.exponent - denominator.exponent);
}

__device__ DistanceKey scaled_squared_distance(const double* left, const double* right,
                                               std::size_t n_features)
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
        return DistanceKey{};
    }
    int adjustment = 0;
    const double mantissa = frexp(scale.mantissa * scale.mantissa * sum_squares, &adjustment);
    const int exponent = 2 * scale.exponent + adjustment;
    return DistanceKey{mantissa, exponent};
}

__device__ DistanceKey distance_key_from_squared(double squared, const double* left,
                                                 const double* right, std::size_t n_features)
{
    if (squared >= DBL_MIN && isfinite(squared))
    {
        int exponent = 0;
        return DistanceKey{frexp(squared, &exponent), exponent};
    }
    if (squared == 0.0)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            if (left[feature] != right[feature])
            {
                return scaled_squared_distance(left, right, n_features);
            }
        }
        return DistanceKey{};
    }
    return scaled_squared_distance(left, right, n_features);
}

__device__ DistanceKey squared_distance(const double* left, const double* right,
                                        std::size_t n_features)
{
    double squared = 0.0;
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        const double delta = left[feature] - right[feature];
        squared += delta * delta;
    }
    return distance_key_from_squared(squared, left, right, n_features);
}

template <int Count>
__device__ void squared_distance_batch(const double* left, const double* const (&right)[Count],
                                       std::size_t n_features, double (&squared)[Count])
{
#pragma unroll
    for (int item = 0; item < Count; ++item)
    {
        squared[item] = 0.0;
    }
    for (std::size_t feature = 0; feature < n_features; ++feature)
    {
        const double value = left[feature];
#pragma unroll
        for (int item = 0; item < Count; ++item)
        {
            const double delta = value - right[item][feature];
            squared[item] += delta * delta;
        }
    }
}

__device__ int compare_distance(DistanceKey left, DistanceKey right)
{
    if (left.value == 0.0 || right.value == 0.0)
    {
        if (left.value == right.value)
        {
            return 0;
        }
        return left.value == 0.0 ? -1 : 1;
    }
    if (left.exponent != right.exponent)
    {
        return left.exponent < right.exponent ? -1 : 1;
    }
    if (left.value < right.value)
    {
        return -1;
    }
    if (right.value < left.value)
    {
        return 1;
    }
    return 0;
}

__device__ bool better(Neighbor left, Neighbor right)
{
    const int order = compare_distance(DistanceKey{left.distance, left.exponent},
                                       DistanceKey{right.distance, right.exponent});
    return order != 0 ? order < 0 : left.idx < right.idx;
}

__device__ Neighbor make_neighbor(DistanceKey distance, std::uint32_t idx)
{
    return Neighbor{distance.value, distance.exponent, idx};
}

__device__ double euclidean_distance(DistanceKey distance)
{
    if (distance.value == 0.0)
    {
        return 0.0;
    }
    if (distance.exponent % 2 == 0)
    {
        return scalbn(sqrt(distance.value), distance.exponent / 2);
    }
    return scalbn(sqrt(distance.value * 2.0), (distance.exponent - 1) / 2);
}

__device__ double squared_distance_value(DistanceKey distance)
{
    return scalbn(distance.value, distance.exponent);
}

__device__ void swap_neighbors(Neighbor& left, Neighbor& right)
{
    const Neighbor value = left;
    left = right;
    right = value;
}

template <std::size_t Capacity>
__device__ void heap_insert(Neighbor (&heap)[Capacity], int& size, Neighbor value)
{
    int child = size++;
    heap[child] = value;
    while (child > 0)
    {
        const int parent = (child - 1) / 2;
        if (!better(heap[parent], heap[child]))
        {
            break;
        }
        swap_neighbors(heap[parent], heap[child]);
        child = parent;
    }
}

template <std::size_t Capacity>
__device__ void heap_replace_top(Neighbor (&heap)[Capacity], int size, Neighbor value)
{
    heap[0] = value;
    int parent = 0;
    while (true)
    {
        const int left = 2 * parent + 1;
        if (left >= size)
        {
            return;
        }
        const int right = left + 1;
        int worse_child = left;
        if (right < size && better(heap[left], heap[right]))
        {
            worse_child = right;
        }
        if (!better(heap[parent], heap[worse_child]))
        {
            return;
        }
        swap_neighbors(heap[parent], heap[worse_child]);
        parent = worse_child;
    }
}

template <std::size_t Capacity>
__device__ void insert_candidate(Neighbor (&heap)[Capacity], int& size, int limit,
                                 Neighbor candidate)
{
    if (size < limit)
    {
        heap_insert(heap, size, candidate);
    }
    else if (better(candidate, heap[0]))
    {
        heap_replace_top(heap, size, candidate);
    }
}

template <std::size_t Capacity>
__device__ void sort_neighbors(Neighbor (&neighbors)[Capacity], int size)
{
    for (int i = 1; i < size; ++i)
    {
        const Neighbor value = neighbors[i];
        int pos = i;
        while (pos > 0 && better(value, neighbors[pos - 1]))
        {
            neighbors[pos] = neighbors[pos - 1];
            --pos;
        }
        neighbors[pos] = value;
    }
}

template <bool IncludeSelf, std::size_t Capacity, int TileSize, bool CacheQuery>
__global__ void knn_kernel(const double* __restrict__ X, std::uint32_t n_samples, int n_features,
                           int k, DistanceMetric metric, std::int64_t* __restrict__ output_indices,
                           double* __restrict__ output_distances)
{
    extern __shared__ double candidate_tile[];

    const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = sample < n_samples;
    const int n_search = IncludeSelf ? k - 1 : k;
    if (n_search == 0)
    {
        if (active)
        {
            output_indices[sample] = sample;
            output_distances[sample] = 0.0;
        }
        return;
    }

    Neighbor heap[Capacity];
    int heap_size = 0;
    double cached_query[kCachedFeatures];
    const double* query = nullptr;
    if (active)
    {
        query = X + static_cast<std::size_t>(sample) * n_features;
        if constexpr (CacheQuery)
        {
            for (int feature = 0; feature < n_features; ++feature)
            {
                cached_query[feature] = query[feature];
            }
            query = cached_query;
        }
    }

    for (std::uint32_t tile_start = 0; tile_start < n_samples;)
    {
        const auto remaining = n_samples - tile_start;
        const int tile_size = remaining < static_cast<std::uint32_t>(TileSize)
                                  ? static_cast<int>(remaining)
                                  : TileSize;
        if constexpr (TileSize > 1)
        {
            const int values = tile_size * n_features;
            for (int offset = threadIdx.x; offset < values; offset += blockDim.x)
            {
                const int candidate = offset / n_features;
                const int feature = offset - candidate * n_features;
                candidate_tile[offset] =
                    X[static_cast<std::size_t>(tile_start + static_cast<std::uint32_t>(candidate)) *
                          n_features +
                      feature];
            }
        }
        __syncthreads();

        if (active)
        {
            int tile_idx = 0;
            if constexpr (TileSize > 1)
            {
                constexpr int BatchSize = 4;
                for (; tile_idx + BatchSize <= tile_size; tile_idx += BatchSize)
                {
                    const double* rows[BatchSize];
#pragma unroll
                    for (int item = 0; item < BatchSize; ++item)
                    {
                        rows[item] = candidate_tile + (tile_idx + item) * n_features;
                    }
                    double squared[BatchSize];
                    squared_distance_batch(query, rows, n_features, squared);
#pragma unroll
                    for (int item = 0; item < BatchSize; ++item)
                    {
                        const std::uint32_t candidate = tile_start + tile_idx + item;
                        if (candidate != sample)
                        {
                            insert_candidate(
                                heap, heap_size, n_search,
                                make_neighbor(distance_key_from_squared(squared[item], query,
                                                                        rows[item], n_features),
                                              candidate));
                        }
                    }
                }
            }
            for (; tile_idx < tile_size; ++tile_idx)
            {
                const std::uint32_t candidate = tile_start + tile_idx;
                if (candidate == sample)
                {
                    continue;
                }
                const double* candidate_row =
                    TileSize > 1 ? candidate_tile + tile_idx * n_features
                                 : X + static_cast<std::size_t>(candidate) * n_features;
                insert_candidate(
                    heap, heap_size, n_search,
                    make_neighbor(squared_distance(query, candidate_row, n_features), candidate));
            }
        }
        __syncthreads();
        tile_start += static_cast<std::uint32_t>(tile_size);
    }

    if constexpr (IncludeSelf)
    {
        if (active)
        {
            heap_insert(heap, heap_size, make_neighbor(DistanceKey{}, sample));
        }
    }

    if (active)
    {
        sort_neighbors(heap, heap_size);
        const std::size_t offset = static_cast<std::size_t>(sample) * k;
        for (int neighbor = 0; neighbor < k; ++neighbor)
        {
            const Neighbor value = heap[neighbor];
            output_indices[offset + neighbor] = static_cast<std::int64_t>(value.idx);
            const DistanceKey distance{value.distance, value.exponent};
            output_distances[offset + neighbor] = metric == DistanceMetric::SquaredEuclidean
                                                      ? squared_distance_value(distance)
                                                      : euclidean_distance(distance);
        }
    }
}

template <bool IncludeSelf, std::size_t Capacity, int TileSize, bool CacheQuery>
void launch_knn(const double* X, std::size_t n_samples, std::size_t n_features, std::size_t k,
                DistanceMetric metric, std::int64_t* output_indices, double* output_distances)
{
    const auto blocks =
        static_cast<unsigned int>((n_samples + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const std::size_t shared_bytes =
        TileSize > 1 ? static_cast<std::size_t>(TileSize) * n_features * sizeof(double) : 0;
    knn_kernel<IncludeSelf, Capacity, TileSize, CacheQuery>
        <<<blocks, kThreadsPerBlock, shared_bytes>>>(
            X, static_cast<std::uint32_t>(n_samples), static_cast<int>(n_features),
            static_cast<int>(k), metric, output_indices, output_distances);
}

template <bool IncludeSelf, std::size_t Capacity>
void launch_knn_for_features(const double* X, std::size_t n_samples, std::size_t n_features,
                             std::size_t k, DistanceMetric metric, std::int64_t* output_indices,
                             double* output_distances)
{
    if (n_features <= kCachedFeatures)
    {
        launch_knn<IncludeSelf, Capacity, 32, true>(X, n_samples, n_features, k, metric,
                                                    output_indices, output_distances);
    }
    else if (n_features <= 256)
    {
        launch_knn<IncludeSelf, Capacity, 16, false>(X, n_samples, n_features, k, metric,
                                                     output_indices, output_distances);
    }
    else if (n_features <= 1024)
    {
        launch_knn<IncludeSelf, Capacity, 4, false>(X, n_samples, n_features, k, metric,
                                                    output_indices, output_distances);
    }
    else
    {
        launch_knn<IncludeSelf, Capacity, 1, false>(X, n_samples, n_features, k, metric,
                                                    output_indices, output_distances);
    }
}

template <bool IncludeSelf>
void dispatch_knn(const double* X, std::size_t n_samples, std::size_t n_features, std::size_t k,
                  DistanceMetric metric, std::int64_t* output_indices, double* output_distances)
{
    if (k <= 8)
    {
        launch_knn_for_features<IncludeSelf, 8>(X, n_samples, n_features, k, metric, output_indices,
                                                output_distances);
    }
    else if (k <= 16)
    {
        launch_knn_for_features<IncludeSelf, 16>(X, n_samples, n_features, k, metric,
                                                 output_indices, output_distances);
    }
    else if (k <= 32)
    {
        launch_knn_for_features<IncludeSelf, 32>(X, n_samples, n_features, k, metric,
                                                 output_indices, output_distances);
    }
    else if (k <= 48)
    {
        launch_knn_for_features<IncludeSelf, 48>(X, n_samples, n_features, k, metric,
                                                 output_indices, output_distances);
    }
    else
    {
        launch_knn_for_features<IncludeSelf, 64>(X, n_samples, n_features, k, metric,
                                                 output_indices, output_distances);
    }
}

void require_problem(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t k, DistanceMetric metric, bool include_self,
                     std::span<std::int64_t> output_indices, std::span<double> output_distances)
{
    const std::size_t X_size =
        neurale::models::detail::checked_product(n_samples, n_features, "X shape is too large");
    const std::size_t output_size =
        neurale::models::detail::checked_product(n_samples, k, "KNN output is too large");
    if (n_samples == 0 || n_features == 0 || X.size() != X_size)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (n_samples > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        n_features > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("CUDA KNN dimensions exceed int32 range");
    }
    const std::size_t max_neighbors_count = include_self ? n_samples : n_samples - 1;
    if (k == 0 || k > max_neighbors_count)
    {
        throw std::invalid_argument("k exceeds the available neighbor count");
    }
    if (k > max_knn_neighbors)
    {
        throw std::invalid_argument("CUDA KNN supports at most 64 neighbors");
    }
    if (metric != DistanceMetric::SquaredEuclidean && metric != DistanceMetric::Euclidean)
    {
        throw std::invalid_argument("metric is invalid");
    }
    if (output_indices.size() != output_size || output_distances.size() != output_size)
    {
        throw std::invalid_argument("KNN output shape is invalid");
    }
    if (neurale::models::detail::spans_overlap(X, output_indices) ||
        neurale::models::detail::spans_overlap(X, output_distances) ||
        neurale::models::detail::spans_overlap(output_indices, output_distances))
    {
        throw std::invalid_argument("KNN X and output buffers must not overlap");
    }
    for (double value : X)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("X must contain finite values");
        }
    }
}

} // namespace

void knn(std::span<const double> X, std::size_t n_samples, std::size_t n_features, std::size_t k,
         DistanceMetric metric, bool include_self, int device_id,
         std::span<std::int64_t> output_indices, std::span<double> output_distances)
{
    require_problem(X, n_samples, n_features, k, metric, include_self, output_indices,
                    output_distances);
    detail::DeviceGuard device(device_id);

    const std::size_t X_bytes = neurale::models::detail::checked_product(X.size(), sizeof(double),
                                                                         "CUDA KNN X is too large");
    const std::size_t indices_bytes = neurale::models::detail::checked_product(
        output_indices.size(), sizeof(std::int64_t), "CUDA KNN output is too large");
    const std::size_t distances_bytes = neurale::models::detail::checked_product(
        output_distances.size(), sizeof(double), "CUDA KNN output is too large");
    const std::size_t workspace_bytes = neurale::models::detail::checked_sum(
        neurale::models::detail::checked_sum(X_bytes, indices_bytes,
                                             "CUDA KNN workspace is too large"),
        distances_bytes, "CUDA KNN workspace is too large");
    detail::PooledDeviceBuffer<std::byte> workspace(workspace_bytes);
    auto* const device_X = reinterpret_cast<double*>(workspace.data());
    auto* const device_indices = reinterpret_cast<std::int64_t*>(workspace.data() + X_bytes);
    auto* const device_distances =
        reinterpret_cast<double*>(workspace.data() + X_bytes + indices_bytes);
    detail::check_cuda(cudaMemcpy(device_X, X.data(), X_bytes, cudaMemcpyHostToDevice),
                       "CUDA KNN X copy failed");

    if (include_self)
    {
        dispatch_knn<true>(device_X, n_samples, n_features, k, metric, device_indices,
                           device_distances);
    }
    else
    {
        dispatch_knn<false>(device_X, n_samples, n_features, k, metric, device_indices,
                            device_distances);
    }
    detail::check_cuda(cudaGetLastError(), "CUDA KNN execution failed");
    detail::check_cuda(
        cudaMemcpy(output_indices.data(), device_indices, indices_bytes, cudaMemcpyDeviceToHost),
        "CUDA KNN index copy failed");
    detail::check_cuda(cudaMemcpy(output_distances.data(), device_distances, distances_bytes,
                                  cudaMemcpyDeviceToHost),
                       "CUDA KNN distance copy failed");
}

} // namespace neurale::models::cuda
