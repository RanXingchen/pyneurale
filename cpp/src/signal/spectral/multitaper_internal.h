/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/signal/spectral.h>

#include "fft_dispatch.h"

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace neurale::signal::detail
{

struct MultitaperPsdSpec
{
    std::size_t n_samples;
    std::size_t n_channels;
    std::size_t fft_length;
    std::size_t n_tapers;
    MultitaperWeighting weighting;
    bool one_sided;
    bool real_input;
    double density_scale;
    std::size_t spectrum_bins;
    std::size_t output_bins;
    std::vector<double> tapers;
    std::vector<double> ratios;
    std::vector<double> one_minus_ratios;
    std::vector<std::size_t> output_source_bin;
    std::vector<double> output_scale;
};

class MultitaperPsdBackend
{
  public:
    virtual ~MultitaperPsdBackend() = default;

    [[nodiscard]] virtual std::size_t output_bins() const noexcept = 0;
    virtual void reset_adaptive_state() = 0;

    virtual void process_real(std::span<const double> x, std::span<double> output) = 0;

    virtual void process_complex(std::span<const std::complex<double>> x,
                                 std::span<double> output) = 0;
};

[[nodiscard]] MultitaperPsdSpec
make_multitaper_psd_spec(std::size_t n_samples, std::size_t n_channels, std::size_t fft_length,
                         double density_scale, std::span<const double> tapers, std::size_t n_tapers,
                         std::span<const double> concentration_ratios,
                         MultitaperWeighting weighting, bool one_sided, bool real_input);

[[nodiscard]] std::unique_ptr<MultitaperPsdBackend>
create_builtin_multitaper_psd_backend(MultitaperPsdSpec spec);

[[nodiscard]] FftKernel select_multitaper_fft_kernel(std::size_t n_samples, std::size_t fft_length,
                                                     bool real_input,
                                                     SpectralBackend backend) noexcept;

#if defined(NEURALE_SIGNAL_WITH_MKL)
[[nodiscard]] bool can_use_mkl_multitaper_psd_backend(const MultitaperPsdSpec& spec) noexcept;

[[nodiscard]] std::unique_ptr<MultitaperPsdBackend>
create_mkl_multitaper_psd_backend(MultitaperPsdSpec spec);
#endif

} // namespace neurale::signal::detail
