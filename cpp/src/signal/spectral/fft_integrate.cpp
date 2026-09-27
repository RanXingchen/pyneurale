/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/transforms.h>

#include "fft_dispatch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "mkl_utils.h"
#include <mkl_dfti.h>
#endif

namespace neurale::signal
{
namespace
{

void validate_fft_integrate_args(std::size_t input_size, std::size_t output_size,
                                 std::size_t n_samples, std::size_t n_channels, double time_step,
                                 std::size_t order)
{
    if (n_samples == 0 || n_channels == 0 || order == 0 || !std::isfinite(time_step) ||
        time_step <= 0.0)
    {
        throw std::invalid_argument("invalid frequency-domain integration parameters");
    }
    if (input_size != n_samples * n_channels || output_size != input_size)
    {
        throw std::invalid_argument("integration buffer shape is incorrect");
    }
}

double signed_frequency_index(std::size_t idx, std::size_t length)
{
    return idx <= (length - 1) / 2 ? static_cast<double>(idx)
                                   : static_cast<double>(idx) - static_cast<double>(length);
}

std::complex<double> integration_factor(std::size_t idx, std::size_t length, double time_step,
                                        std::size_t order)
{
    const double omega = 2.0 * std::numbers::pi_v<double> * signed_frequency_index(idx, length) /
                         (static_cast<double>(length) * time_step);
    const std::complex<double> first_order{0.0, -1.0 / omega};
    if (order == 1)
    {
        return first_order;
    }
    if (order == 2)
    {
        return first_order * first_order;
    }
    std::complex<double> factor{1.0, 0.0};
    for (std::size_t count = 0; count < order; ++count)
    {
        factor *= first_order;
    }
    return factor;
}

void apply_multiplier(std::span<std::complex<double>> spectrum, std::size_t n_samples,
                      std::size_t n_channels, double time_step, std::size_t order)
{
    std::fill(spectrum.begin(), spectrum.begin() + static_cast<std::ptrdiff_t>(n_channels),
              std::complex<double>{0.0, 0.0});
    for (std::size_t sample = 1; sample < n_samples; ++sample)
    {
        const auto factor = integration_factor(sample, n_samples, time_step, order);
        auto* row = spectrum.data() + sample * n_channels;
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            row[channel] *= factor;
        }
    }
}

void fft_integrate_complex_builtin(std::span<const std::complex<double>> x, std::size_t n_samples,
                                   std::size_t n_channels, double time_step, std::size_t order,
                                   std::span<std::complex<double>> output)
{
    std::vector<std::complex<double>> spectrum(n_samples);
    for (std::size_t channel = 0; channel < n_channels; ++channel)
    {
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            spectrum[sample] = x[sample * n_channels + channel];
        }
        detail::fft_with_kernel(spectrum, false, detail::FftKernel::builtin);
        spectrum[0] = 0.0;
        for (std::size_t sample = 1; sample < n_samples; ++sample)
        {
            spectrum[sample] *= integration_factor(sample, n_samples, time_step, order);
        }
        detail::fft_with_kernel(spectrum, true, detail::FftKernel::builtin);
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            output[sample * n_channels + channel] = spectrum[sample];
        }
    }
}

void fft_integrate_real_builtin(std::span<const double> x, std::size_t n_samples,
                                std::size_t n_channels, double time_step, std::size_t order,
                                std::span<double> output)
{
    std::vector<std::complex<double>> spectrum(n_samples);
    for (std::size_t channel = 0; channel < n_channels; ++channel)
    {
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            spectrum[sample] = x[sample * n_channels + channel];
        }
        detail::fft_with_kernel(spectrum, false, detail::FftKernel::builtin);
        spectrum[0] = 0.0;
        for (std::size_t sample = 1; sample < n_samples; ++sample)
        {
            spectrum[sample] *= integration_factor(sample, n_samples, time_step, order);
        }
        detail::fft_with_kernel(spectrum, true, detail::FftKernel::builtin);
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            output[sample * n_channels + channel] = spectrum[sample].real();
        }
    }
}

#ifdef NEURALE_SIGNAL_WITH_MKL

using mkl::check_mkl_status;
using mkl::checked_mkl_long;

void configure_batch(DFTI_DESCRIPTOR_HANDLE handle, MKL_LONG n_channels, const MKL_LONG* stride)
{
    mkl::check_mkl_status(DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, n_channels),
                          "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS)");
    mkl::check_mkl_status(DftiSetValue(handle, DFTI_INPUT_DISTANCE, 1),
                          "DftiSetValue(DFTI_INPUT_DISTANCE)");
    mkl::check_mkl_status(DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, 1),
                          "DftiSetValue(DFTI_OUTPUT_DISTANCE)");
    mkl::check_mkl_status(DftiSetValue(handle, DFTI_INPUT_STRIDES, stride),
                          "DftiSetValue(DFTI_INPUT_STRIDES)");
    mkl::check_mkl_status(DftiSetValue(handle, DFTI_OUTPUT_STRIDES, stride),
                          "DftiSetValue(DFTI_OUTPUT_STRIDES)");
}

class ComplexDescriptor
{
  public:
    ComplexDescriptor(std::size_t n_samples, std::size_t n_channels)
    {
        const MKL_LONG length = mkl::checked_mkl_long(n_samples, "sample count");
        const MKL_LONG channels = mkl::checked_mkl_long(n_channels, "channel count");
        const MKL_LONG stride[2] = {0, channels};
        mkl::check_mkl_status(DftiCreateDescriptor(&handle_, DFTI_DOUBLE, DFTI_COMPLEX, 1, length),
                              "DftiCreateDescriptor");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_PLACEMENT, DFTI_INPLACE),
                              "DftiSetValue(DFTI_PLACEMENT)");
        mkl::check_mkl_status(
            DftiSetValue(handle_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(n_samples)),
            "DftiSetValue(DFTI_BACKWARD_SCALE)");
        configure_batch(handle_, channels, stride);
        mkl::check_mkl_status(DftiCommitDescriptor(handle_), "DftiCommitDescriptor");
    }

    ~ComplexDescriptor()
    {
        if (handle_ != nullptr)
        {
            DftiFreeDescriptor(&handle_);
        }
    }

    ComplexDescriptor(const ComplexDescriptor&) = delete;
    ComplexDescriptor& operator=(const ComplexDescriptor&) = delete;

    DFTI_DESCRIPTOR_HANDLE get() const noexcept
    {
        return handle_;
    }

  private:
    DFTI_DESCRIPTOR_HANDLE handle_ = nullptr;
};

class RealDescriptor
{
  public:
    RealDescriptor(std::size_t n_samples, std::size_t n_channels)
    {
        const MKL_LONG length = mkl::checked_mkl_long(n_samples, "sample count");
        const MKL_LONG channels = mkl::checked_mkl_long(n_channels, "channel count");
        const MKL_LONG stride[2] = {0, channels};
        mkl::check_mkl_status(DftiCreateDescriptor(&forward_, DFTI_DOUBLE, DFTI_REAL, 1, length),
                              "DftiCreateDescriptor");
        configure_real(forward_, channels, stride);
        mkl::check_mkl_status(DftiSetValue(forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE),
                              "DftiSetValue(DFTI_PLACEMENT)");
        mkl::check_mkl_status(DftiCommitDescriptor(forward_), "DftiCommitDescriptor");

        mkl::check_mkl_status(DftiCreateDescriptor(&backward_, DFTI_DOUBLE, DFTI_REAL, 1, length),
                              "DftiCreateDescriptor");
        configure_real(backward_, channels, stride);
        mkl::check_mkl_status(DftiSetValue(backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE),
                              "DftiSetValue(DFTI_PLACEMENT)");
        mkl::check_mkl_status(
            DftiSetValue(backward_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(n_samples)),
            "DftiSetValue(DFTI_BACKWARD_SCALE)");
        mkl::check_mkl_status(DftiCommitDescriptor(backward_), "DftiCommitDescriptor");
    }

    ~RealDescriptor()
    {
        if (backward_ != nullptr)
        {
            DftiFreeDescriptor(&backward_);
        }
        if (forward_ != nullptr)
        {
            DftiFreeDescriptor(&forward_);
        }
    }

    RealDescriptor(const RealDescriptor&) = delete;
    RealDescriptor& operator=(const RealDescriptor&) = delete;

    DFTI_DESCRIPTOR_HANDLE forward() const noexcept
    {
        return forward_;
    }

    DFTI_DESCRIPTOR_HANDLE backward() const noexcept
    {
        return backward_;
    }

  private:
    static void configure_real(DFTI_DESCRIPTOR_HANDLE handle, MKL_LONG n_channels,
                               const MKL_LONG* stride)
    {
        mkl::check_mkl_status(
            DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX),
            "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE)");
        mkl::check_mkl_status(DftiSetValue(handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT),
                              "DftiSetValue(DFTI_PACKED_FORMAT)");
        configure_batch(handle, n_channels, stride);
    }

    DFTI_DESCRIPTOR_HANDLE forward_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE backward_ = nullptr;
};

template <class Descriptor> struct CachedDescriptor
{
    std::size_t n_samples = 0;
    std::size_t n_channels = 0;
    std::size_t generation = 0;
    std::unique_ptr<Descriptor> descriptor;
};

template <class Descriptor>
Descriptor& cached_descriptor(std::array<CachedDescriptor<Descriptor>, 4>& cache,
                              std::size_t& generation, std::size_t n_samples,
                              std::size_t n_channels)
{
    ++generation;
    for (auto& entry : cache)
    {
        if (entry.descriptor != nullptr && entry.n_samples == n_samples &&
            entry.n_channels == n_channels)
        {
            entry.generation = generation;
            return *entry.descriptor;
        }
    }
    auto* replacement = &cache.front();
    for (auto& entry : cache)
    {
        if (entry.descriptor == nullptr || entry.generation < replacement->generation)
        {
            replacement = &entry;
        }
    }
    replacement->descriptor = std::make_unique<Descriptor>(n_samples, n_channels);
    replacement->n_samples = n_samples;
    replacement->n_channels = n_channels;
    replacement->generation = generation;
    return *replacement->descriptor;
}

ComplexDescriptor& complex_descriptor(std::size_t n_samples, std::size_t n_channels)
{
    thread_local std::array<CachedDescriptor<ComplexDescriptor>, 4> cache;
    thread_local std::size_t generation = 0;
    return cached_descriptor(cache, generation, n_samples, n_channels);
}

RealDescriptor& real_descriptor(std::size_t n_samples, std::size_t n_channels)
{
    thread_local std::array<CachedDescriptor<RealDescriptor>, 4> cache;
    thread_local std::size_t generation = 0;
    return cached_descriptor(cache, generation, n_samples, n_channels);
}

void fft_integrate_complex_mkl(std::span<const std::complex<double>> x, std::size_t n_samples,
                               std::size_t n_channels, double time_step, std::size_t order,
                               std::span<std::complex<double>> output)
{
    static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                  "MKL complex layout requires adjacent real and imaginary doubles");
    std::copy(x.begin(), x.end(), output.begin());
    auto& descriptor = complex_descriptor(n_samples, n_channels);
    mkl::check_mkl_status(DftiComputeForward(descriptor.get(), output.data()),
                          "DftiComputeForward");
    apply_multiplier(output, n_samples, n_channels, time_step, order);
    mkl::check_mkl_status(DftiComputeBackward(descriptor.get(), output.data()),
                          "DftiComputeBackward");
}

void fft_integrate_real_mkl(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, double time_step, std::size_t order,
                            std::span<double> output)
{
    static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                  "MKL complex layout requires adjacent real and imaginary doubles");
    std::vector<std::complex<double>> spectrum(n_samples * n_channels);
    auto& descriptor = real_descriptor(n_samples, n_channels);
    mkl::check_mkl_status(
        DftiComputeForward(descriptor.forward(), const_cast<double*>(x.data()), spectrum.data()),
        "DftiComputeForward");
    apply_multiplier(spectrum, n_samples, n_channels, time_step, order);
    mkl::check_mkl_status(
        DftiComputeBackward(descriptor.backward(), spectrum.data(), output.data()),
        "DftiComputeBackward");
}

#endif

} // namespace

void fft_integrate(std::span<const std::complex<double>> x, std::size_t n_samples,
                   std::size_t n_channels, double time_step, std::size_t order,
                   std::span<std::complex<double>> output)
{
    validate_fft_integrate_args(x.size(), output.size(), n_samples, n_channels, time_step, order);
#ifdef NEURALE_SIGNAL_WITH_MKL
    fft_integrate_complex_mkl(x, n_samples, n_channels, time_step, order, output);
#else
    fft_integrate_complex_builtin(x, n_samples, n_channels, time_step, order, output);
#endif
}

void fft_integrate(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                   double time_step, std::size_t order, std::span<double> output)
{
    validate_fft_integrate_args(x.size(), output.size(), n_samples, n_channels, time_step, order);
#ifdef NEURALE_SIGNAL_WITH_MKL
    fft_integrate_real_mkl(x, n_samples, n_channels, time_step, order, output);
#else
    fft_integrate_real_builtin(x, n_samples, n_channels, time_step, order, output);
#endif
}

} // namespace neurale::signal
