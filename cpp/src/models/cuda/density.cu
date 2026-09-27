/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/cuda/density.h>

#include "cuda_utils.cuh"
#include "span_utils.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurale::models::cuda
{
namespace
{

constexpr int kThreadsPerBlock = 256;

__device__ void accumulate_log_sum(double value, double& largest, double& scaled_sum)
{
    if (value > largest)
    {
        scaled_sum = largest == -CUDART_INF ? 1.0 : scaled_sum * exp(largest - value) + 1.0;
        largest = value;
    }
    else
    {
        scaled_sum += exp(value - largest);
    }
}

__device__ void merge_log_sums(double other_largest, double other_sum, double& largest,
                               double& scaled_sum)
{
    if (other_largest == -CUDART_INF)
    {
        return;
    }
    if (largest == -CUDART_INF)
    {
        largest = other_largest;
        scaled_sum = other_sum;
        return;
    }
    if (other_largest > largest)
    {
        scaled_sum = other_sum + scaled_sum * exp(largest - other_largest);
        largest = other_largest;
    }
    else
    {
        scaled_sum += other_sum * exp(other_largest - largest);
    }
}

template <bool Pdf>
__global__ void evaluate_density_kernel(const double* X, const double* mean, const double* lower,
                                        double* whitened, const double* samples,
                                        std::size_t n_samples, std::size_t n_features,
                                        double log_normalizer, double* output)
{
    const std::size_t point = static_cast<std::size_t>(blockIdx.x);
    const double* X_row = X + point * n_features;
    double* point_values = whitened + point * n_features;
    if (threadIdx.x == 0)
    {
        for (std::size_t row = 0; row < n_features; ++row)
        {
            double value = X_row[row] - mean[row];
            for (std::size_t col = 0; col < row; ++col)
            {
                value -= lower[row * n_features + col] * point_values[col];
            }
            point_values[row] = value / lower[row * n_features + row];
        }
    }
    __syncthreads();

    double largest = -CUDART_INF;
    double scaled_sum = 0.0;
    for (std::size_t sample = threadIdx.x; sample < n_samples; sample += blockDim.x)
    {
        double distance = 0.0;
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            const double delta = point_values[feature] - samples[feature * n_samples + sample];
            distance += delta * delta;
        }
        if (!isfinite(distance))
        {
            continue;
        }
        accumulate_log_sum(-0.5 * distance, largest, scaled_sum);
    }

    __shared__ double maxima[kThreadsPerBlock];
    __shared__ double sums[kThreadsPerBlock];
    maxima[threadIdx.x] = largest;
    sums[threadIdx.x] = scaled_sum;
    __syncthreads();

    for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            merge_log_sums(maxima[threadIdx.x + stride], sums[threadIdx.x + stride],
                           maxima[threadIdx.x], sums[threadIdx.x]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0)
    {
        return;
    }

    largest = maxima[0];
    scaled_sum = sums[0];
    const double log_density =
        largest == -CUDART_INF ? largest : log_normalizer + largest + log(scaled_sum);
    if constexpr (Pdf)
    {
        output[point] = exp(log_density);
    }
    else
    {
        output[point] = log_density;
    }
}

std::vector<double> transpose_samples(std::span<const double> samples, std::size_t n_samples,
                                      std::size_t n_features)
{
    auto output = std::vector<double>(samples.size());
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t feature = 0; feature < n_features; ++feature)
        {
            output[feature * n_samples + sample] = samples[sample * n_features + feature];
        }
    }
    return output;
}

void require_points(std::span<const double> X, std::size_t nX, std::size_t n_features,
                    std::span<double> output)
{
    const std::size_t point_size =
        neurale::models::detail::checked_product(nX, n_features, "X shape is too large");
    if (X.size() != point_size || output.size() != nX)
    {
        throw std::invalid_argument("X shape is invalid");
    }
    if (neurale::models::detail::spans_overlap(X, output))
    {
        throw std::invalid_argument("X and output must not overlap");
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

struct GaussianKdeModel::Impl
{
    Impl(std::size_t samples, std::size_t features, double factor, double normalizer,
         std::span<const double> mean, std::span<const double> whitened_samples,
         std::span<const double> lower, int requested_device)
        : n_samples(samples), n_features(features), bandwidth_factor(factor),
          log_normalizer(normalizer)
    {
        detail::DeviceGuard device(requested_device);
        device_id = device.device_id();
        const auto transposed = transpose_samples(whitened_samples, n_samples, n_features);
        device_mean.copy_from(mean.data(), mean.size());
        device_samples.copy_from(transposed.data(), transposed.size());
        device_lower.copy_from(lower.data(), lower.size());
    }

    void reserve(std::size_t nX)
    {
        const std::size_t point_size =
            neurale::models::detail::checked_product(nX, n_features, "X shape is too large");
        device_points.resize(point_size);
        device_whitened.resize(point_size);
        device_output.resize(nX);
    }

    template <bool Pdf>
    void evaluate(std::span<const double> X, std::size_t nX, std::span<double> output)
    {
        require_points(X, nX, n_features, output);
        if (nX == 0)
        {
            return;
        }
        detail::DeviceGuard device(device_id);
        reserve(nX);
        device_points.copy_from(X.data(), X.size());
        evaluate_density_kernel<Pdf><<<static_cast<unsigned int>(nX), kThreadsPerBlock>>>(
            device_points.data(), device_mean.data(), device_lower.data(), device_whitened.data(),
            device_samples.data(), n_samples, n_features, log_normalizer, device_output.data());
        detail::check_cuda(cudaGetLastError(), "CUDA KDE evaluation launch failed");
        device_output.copy_to(output.data(), output.size());
    }

    std::size_t n_samples{};
    std::size_t n_features{};
    double bandwidth_factor{};
    double log_normalizer{};
    int device_id{};
    detail::DeviceBuffer<double> device_mean;
    detail::DeviceBuffer<double> device_samples;
    detail::DeviceBuffer<double> device_lower;
    detail::DeviceBuffer<double> device_points;
    detail::DeviceBuffer<double> device_whitened;
    detail::DeviceBuffer<double> device_output;
};

GaussianKdeModel::GaussianKdeModel(neurale::models::GaussianKdeModel model, int device_id)
    : impl_(std::make_unique<Impl>(model.state_.n_samples, model.state_.n_features,
                                   model.state_.bandwidth_factor, model.state_.log_normalizer,
                                   model.state_.mean, model.state_.whitened_samples,
                                   model.state_.lower_cholesky, device_id))
{
}

GaussianKdeModel::GaussianKdeModel(GaussianKdeModel&&) noexcept = default;
GaussianKdeModel& GaussianKdeModel::operator=(GaussianKdeModel&&) noexcept = default;
GaussianKdeModel::~GaussianKdeModel() = default;

std::size_t GaussianKdeModel::n_samples() const noexcept
{
    return impl_->n_samples;
}

std::size_t GaussianKdeModel::n_features() const noexcept
{
    return impl_->n_features;
}

double GaussianKdeModel::bandwidth_factor() const noexcept
{
    return impl_->bandwidth_factor;
}

void GaussianKdeModel::pdf(std::span<const double> X, std::size_t nX, std::span<double> output)
{
    impl_->evaluate<true>(X, nX, output);
}

void GaussianKdeModel::logpdf(std::span<const double> X, std::size_t nX, std::span<double> output)
{
    impl_->evaluate<false>(X, nX, output);
}

GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                  std::size_t n_features, BandwidthRule bandwidth_rule,
                                  int device_id)
{
    return GaussianKdeModel(
        neurale::models::fit_gaussian_kde(X, n_samples, n_features, bandwidth_rule), device_id);
}

GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                  std::size_t n_features, double bandwidth_factor, int device_id)
{
    return GaussianKdeModel(
        neurale::models::fit_gaussian_kde(X, n_samples, n_features, bandwidth_factor), device_id);
}

} // namespace neurale::models::cuda
