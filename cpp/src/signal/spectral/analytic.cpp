/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/spectral.h>

#include "fft_dispatch.h"

#include <algorithm>
#include <array>
#include <memory>
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

bool is_power_of_two(std::size_t value) noexcept
{
    return value != 0 && (value & (value - 1)) == 0;
}

void analytic_signal_builtin_with_scratch(std::span<const double> x, std::size_t n_samples,
                                          std::size_t n_channels, std::size_t fft_length,
                                          std::span<std::complex<double>> output,
                                          std::span<std::complex<double>> spectrum)
{
    if (spectrum.size() != fft_length)
    {
        throw std::invalid_argument("analytic signal workspace shape is incorrect");
    }
    for (std::size_t channel = 0; channel < n_channels; ++channel)
    {
        std::fill(spectrum.begin(), spectrum.end(), 0.0);
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            spectrum[sample] = x[sample * n_channels + channel];
        }
        detail::fft_with_kernel(spectrum, false, detail::FftKernel::builtin);
        if (fft_length % 2 == 0)
        {
            for (std::size_t i = 1; i < fft_length / 2; ++i)
            {
                spectrum[i] *= 2.0;
            }
            for (std::size_t i = fft_length / 2 + 1; i < fft_length; ++i)
            {
                spectrum[i] = 0.0;
            }
        }
        else
        {
            for (std::size_t i = 1; i <= (fft_length - 1) / 2; ++i)
            {
                spectrum[i] *= 2.0;
            }
            for (std::size_t i = (fft_length + 1) / 2; i < fft_length; ++i)
            {
                spectrum[i] = 0.0;
            }
        }
        detail::fft_with_kernel(spectrum, true, detail::FftKernel::builtin);
        for (std::size_t sample = 0; sample < fft_length; ++sample)
        {
            output[sample * n_channels + channel] = spectrum[sample];
        }
    }
}

void analytic_signal_builtin(std::span<const double> x, std::size_t n_samples,
                             std::size_t n_channels, std::size_t fft_length,
                             std::span<std::complex<double>> output)
{
    std::vector<std::complex<double>> spectrum(fft_length);
    analytic_signal_builtin_with_scratch(x, n_samples, n_channels, fft_length, output, spectrum);
}

#ifdef NEURALE_SIGNAL_WITH_MKL

using mkl::check_mkl_status;
using mkl::checked_mkl_long;

class HilbertDescriptor
{
  public:
    HilbertDescriptor(std::size_t fft_length, std::size_t n_channels)
    {
        const MKL_LONG length = mkl::checked_mkl_long(fft_length, "FFT length");
        const MKL_LONG channels = mkl::checked_mkl_long(n_channels, "channel count");
        const MKL_LONG stride[2] = {0, channels};

        mkl::check_mkl_status(DftiCreateDescriptor(&handle_, DFTI_DOUBLE, DFTI_COMPLEX, 1, length),
                              "DftiCreateDescriptor");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_PLACEMENT, DFTI_INPLACE),
                              "DftiSetValue(DFTI_PLACEMENT)");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_NUMBER_OF_TRANSFORMS, channels),
                              "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS)");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_INPUT_DISTANCE, 1),
                              "DftiSetValue(DFTI_INPUT_DISTANCE)");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_OUTPUT_DISTANCE, 1),
                              "DftiSetValue(DFTI_OUTPUT_DISTANCE)");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_INPUT_STRIDES, stride),
                              "DftiSetValue(DFTI_INPUT_STRIDES)");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_OUTPUT_STRIDES, stride),
                              "DftiSetValue(DFTI_OUTPUT_STRIDES)");
        mkl::check_mkl_status(
            DftiSetValue(handle_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(fft_length)),
            "DftiSetValue(DFTI_BACKWARD_SCALE)");
        mkl::check_mkl_status(DftiCommitDescriptor(handle_), "DftiCommitDescriptor");
    }

    ~HilbertDescriptor()
    {
        if (handle_ != nullptr)
        {
            DftiFreeDescriptor(&handle_);
        }
    }

    HilbertDescriptor(const HilbertDescriptor&) = delete;
    HilbertDescriptor& operator=(const HilbertDescriptor&) = delete;

    DFTI_DESCRIPTOR_HANDLE get() const noexcept
    {
        return handle_;
    }

  private:
    DFTI_DESCRIPTOR_HANDLE handle_ = nullptr;
};

struct CachedHilbertDescriptor
{
    std::size_t fft_length = 0;
    std::size_t n_channels = 0;
    std::size_t generation = 0;
    std::unique_ptr<HilbertDescriptor> descriptor;
};

HilbertDescriptor& cached_hilbert_descriptor(std::size_t fft_length, std::size_t n_channels)
{
    thread_local std::array<CachedHilbertDescriptor, 4> cache;
    thread_local std::size_t generation = 0;
    ++generation;

    for (auto& entry : cache)
    {
        if (entry.descriptor != nullptr && entry.fft_length == fft_length &&
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
    replacement->descriptor = std::make_unique<HilbertDescriptor>(fft_length, n_channels);
    replacement->fft_length = fft_length;
    replacement->n_channels = n_channels;
    replacement->generation = generation;
    return *replacement->descriptor;
}

void apply_hilbert_mask(std::span<std::complex<double>> values, std::size_t fft_length,
                        std::size_t n_channels)
{
    const std::size_t positive_end = fft_length % 2 == 0 ? fft_length / 2 : (fft_length + 1) / 2;
    const std::size_t zero_start = fft_length % 2 == 0 ? fft_length / 2 + 1 : positive_end;

    for (std::size_t i = 1; i < positive_end; ++i)
    {
        auto* row = values.data() + i * n_channels;
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            row[channel] *= 2.0;
        }
    }
    std::fill(values.begin() + static_cast<std::ptrdiff_t>(zero_start * n_channels), values.end(),
              std::complex<double>{0.0, 0.0});
}

void analytic_signal_mkl(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                         std::size_t fft_length, std::span<std::complex<double>> output)
{
    static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                  "MKL complex layout requires adjacent real and imaginary doubles");

    std::fill(output.begin(), output.end(), std::complex<double>{0.0, 0.0});
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const auto input_offset = sample * n_channels;
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            output[input_offset + channel] = std::complex<double>{x[input_offset + channel], 0.0};
        }
    }

    auto& descriptor = cached_hilbert_descriptor(fft_length, n_channels);
    mkl::check_mkl_status(DftiComputeForward(descriptor.get(), output.data()),
                          "DftiComputeForward");
    apply_hilbert_mask(output, fft_length, n_channels);
    mkl::check_mkl_status(DftiComputeBackward(descriptor.get(), output.data()),
                          "DftiComputeBackward");
}

#endif

} // namespace

class AnalyticSignalProcessor::Impl
{
  public:
    Impl(std::size_t n_samples, std::size_t n_channels, std::size_t fft_length,
         SpectralBackend backend)
        : n_samples_(n_samples), n_channels_(n_channels), fft_length_(fft_length), backend_(backend)
    {
        if (n_samples == 0 || n_channels == 0 || fft_length < n_samples)
        {
            throw std::invalid_argument("invalid analytic signal processor shape");
        }
#ifdef NEURALE_SIGNAL_WITH_MKL
        // Follows detail::select_fft_kernel's length threshold: routing tiny
        // FFTs through MKL pulls in the intel_thread pool, and every
        // DftiCompute call synchronises it, which can stall the realtime
        // thread for milliseconds on a scheduler wakeup. Builtin below the
        // threshold keeps the Hilbert path deterministic and allocation-free
        // (spectrum_ is a pre-sized member).
        if (backend_ == SpectralBackend::automatic &&
            detail::select_fft_kernel(fft_length_) == detail::FftKernel::mkl_dfti)
        {
            descriptor_ = std::make_unique<HilbertDescriptor>(fft_length_, n_channels_);
            return;
        }
#endif
        if (!is_power_of_two(fft_length_))
        {
            throw std::invalid_argument(
                "builtin analytic signal processor requires a power-of-two FFT length");
        }
        spectrum_.resize(fft_length_);
    }

    void process(std::span<const double> x, std::span<std::complex<double>> output)
    {
        if (x.size() != n_samples_ * n_channels_ || output.size() != fft_length_ * n_channels_)
        {
            throw std::invalid_argument("analytic signal processor buffer shape is incorrect");
        }
#ifdef NEURALE_SIGNAL_WITH_MKL
        if (descriptor_ != nullptr)
        {
            std::fill(output.begin(), output.end(), std::complex<double>{0.0, 0.0});
            for (std::size_t sample = 0; sample < n_samples_; ++sample)
            {
                const auto offset = sample * n_channels_;
                for (std::size_t channel = 0; channel < n_channels_; ++channel)
                {
                    output[offset + channel] = {x[offset + channel], 0.0};
                }
            }
            // Pin MKL to one thread for the paired forward/backward transform
            // to avoid the intel_thread pool sync latency noted above. Same
            // pattern as iir_realtime/sos_realtime process_mkl.
            mkl::LocalThreadLimit thread_limit;
            mkl::check_mkl_status(DftiComputeForward(descriptor_->get(), output.data()),
                                  "DftiComputeForward");
            apply_hilbert_mask(output, fft_length_, n_channels_);
            mkl::check_mkl_status(DftiComputeBackward(descriptor_->get(), output.data()),
                                  "DftiComputeBackward");
            return;
        }
#endif
        analytic_signal_builtin_with_scratch(x, n_samples_, n_channels_, fft_length_, output,
                                             spectrum_);
    }

  private:
    std::size_t n_samples_{};
    std::size_t n_channels_{};
    std::size_t fft_length_{};
    SpectralBackend backend_{SpectralBackend::automatic};
    std::vector<std::complex<double>> spectrum_;
#ifdef NEURALE_SIGNAL_WITH_MKL
    std::unique_ptr<HilbertDescriptor> descriptor_;
#endif
};

AnalyticSignalProcessor::AnalyticSignalProcessor(std::size_t n_samples, std::size_t n_channels,
                                                 std::size_t fft_length, SpectralBackend backend)
    : impl_(std::make_unique<Impl>(n_samples, n_channels, fft_length, backend))
{
}

AnalyticSignalProcessor::~AnalyticSignalProcessor() = default;

AnalyticSignalProcessor::AnalyticSignalProcessor(AnalyticSignalProcessor&&) noexcept = default;

AnalyticSignalProcessor&
AnalyticSignalProcessor::operator=(AnalyticSignalProcessor&&) noexcept = default;

void AnalyticSignalProcessor::process(std::span<const double> x,
                                      std::span<std::complex<double>> output)
{
    impl_->process(x, output);
}

void analytic_signal(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                     std::size_t fft_length, std::span<std::complex<double>> output)
{
    if (fft_length == 0 || fft_length < n_samples)
    {
        throw std::invalid_argument("fft_length must be at least the input length");
    }
    if (x.size() != n_samples * n_channels || output.size() != fft_length * n_channels)
    {
        throw std::invalid_argument("analytic signal buffer shape is incorrect");
    }
    if (n_channels == 0)
    {
        return;
    }
#ifdef NEURALE_SIGNAL_WITH_MKL
    if (fft_length > 1)
    {
        analytic_signal_mkl(x, n_samples, n_channels, fft_length, output);
        return;
    }
#endif
    analytic_signal_builtin(x, n_samples, n_channels, fft_length, output);
}

} // namespace neurale::signal
