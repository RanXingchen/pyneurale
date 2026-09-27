/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/spectral.h>

#include "multitaper_internal.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace neurale::signal
{
namespace detail
{

namespace
{

void validate_common_psd_inputs(std::size_t n_samples, std::size_t n_channels,
                                std::size_t fft_length, double density_scale,
                                std::span<const double> tapers, std::size_t n_tapers,
                                std::span<const double> concentration_ratios,
                                MultitaperWeighting weighting, bool one_sided, bool real_input)
{
    if (n_samples == 0 || n_channels == 0 || fft_length == 0 || n_tapers == 0)
    {
        throw std::invalid_argument("multitaper PSD dimensions must be positive");
    }
    if (density_scale <= 0.0 || !std::isfinite(density_scale))
    {
        throw std::invalid_argument("multitaper PSD density scale must be positive");
    }
    if (tapers.size() != n_samples * n_tapers)
    {
        throw std::invalid_argument("multitaper PSD taper buffer shape is incorrect");
    }
    if (concentration_ratios.size() != n_tapers)
    {
        throw std::invalid_argument("multitaper PSD ratio count is incorrect");
    }
    for (const auto ratio : concentration_ratios)
    {
        if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0)
        {
            throw std::invalid_argument("multitaper PSD ratios must be finite values in [0, 1]");
        }
    }
    if (weighting == MultitaperWeighting::adaptive && n_tapers < 2)
    {
        throw std::invalid_argument("adaptive multitaper PSD requires at least two tapers");
    }
    if (!real_input && one_sided)
    {
        throw std::invalid_argument("one-sided multitaper PSD requires real input");
    }
}

void fill_real_output_map(MultitaperPsdSpec& spec)
{
    spec.output_source_bin.resize(spec.output_bins);
    spec.output_scale.resize(spec.output_bins);
    if (spec.one_sided)
    {
        for (std::size_t bin = 0; bin < spec.output_bins; ++bin)
        {
            spec.output_source_bin[bin] = bin;
            auto scale = 1.0 / spec.density_scale;
            const auto is_dc = bin == 0;
            const auto is_nyquist = spec.fft_length % 2 == 0 && bin + 1 == spec.output_bins;
            if (!is_dc && !is_nyquist)
            {
                scale *= 2.0;
            }
            spec.output_scale[bin] = scale;
        }
        return;
    }
    for (std::size_t bin = 0; bin < spec.output_bins; ++bin)
    {
        spec.output_source_bin[bin] = bin < spec.spectrum_bins ? bin : spec.fft_length - bin;
        spec.output_scale[bin] = 1.0 / spec.density_scale;
    }
}

} // namespace

MultitaperPsdSpec make_multitaper_psd_spec(std::size_t n_samples, std::size_t n_channels,
                                           std::size_t fft_length, double density_scale,
                                           std::span<const double> tapers, std::size_t n_tapers,
                                           std::span<const double> concentration_ratios,
                                           MultitaperWeighting weighting, bool one_sided,
                                           bool real_input)
{
    validate_common_psd_inputs(n_samples, n_channels, fft_length, density_scale, tapers, n_tapers,
                               concentration_ratios, weighting, one_sided, real_input);

    MultitaperPsdSpec spec{
        .n_samples = n_samples,
        .n_channels = n_channels,
        .fft_length = fft_length,
        .n_tapers = n_tapers,
        .weighting = weighting,
        .one_sided = one_sided,
        .real_input = real_input,
        .density_scale = density_scale,
        .spectrum_bins = real_input ? fft_length / 2 + 1 : fft_length,
        .output_bins = one_sided ? fft_length / 2 + 1 : fft_length,
        .tapers = {tapers.begin(), tapers.end()},
        .ratios = {concentration_ratios.begin(), concentration_ratios.end()},
    };
    spec.one_minus_ratios.resize(n_tapers);
    for (std::size_t taper = 0; taper < n_tapers; ++taper)
    {
        spec.one_minus_ratios[taper] = 1.0 - spec.ratios[taper];
    }
    if (real_input)
    {
        fill_real_output_map(spec);
    }
    return spec;
}

} // namespace detail

namespace
{

std::unique_ptr<detail::MultitaperPsdBackend> create_backend(detail::MultitaperPsdSpec spec,
                                                             SpectralBackend backend)
{
#if defined(NEURALE_SIGNAL_WITH_MKL)
    if (detail::select_multitaper_fft_kernel(spec.n_samples, spec.fft_length, spec.real_input,
                                             backend) == detail::FftKernel::mkl_dfti)
    {
        return detail::create_mkl_multitaper_psd_backend(std::move(spec));
    }
#endif
    return detail::create_builtin_multitaper_psd_backend(std::move(spec));
}

} // namespace

namespace detail
{

FftKernel select_multitaper_fft_kernel(std::size_t n_samples, std::size_t fft_length,
                                       bool real_input, SpectralBackend backend) noexcept
{
#if defined(NEURALE_SIGNAL_WITH_MKL)
    if (backend == SpectralBackend::automatic && real_input && n_samples <= fft_length)
    {
        return FftKernel::mkl_dfti;
    }
#else
    (void)n_samples;
    (void)fft_length;
    (void)real_input;
#endif
    return FftKernel::builtin;
}

} // namespace detail

class MultitaperPsdProcessor::Impl
{
  public:
    Impl(std::size_t n_samples, std::size_t n_channels, std::size_t fft_length,
         double density_scale, std::span<const double> tapers, std::size_t n_tapers,
         std::span<const double> concentration_ratios, MultitaperWeighting weighting,
         bool one_sided, bool real_input, SpectralBackend backend)
        : backend_(create_backend(
              detail::make_multitaper_psd_spec(n_samples, n_channels, fft_length, density_scale,
                                               tapers, n_tapers, concentration_ratios, weighting,
                                               one_sided, real_input),
              backend))
    {
    }

    [[nodiscard]] std::size_t output_bins() const noexcept
    {
        return backend_->output_bins();
    }

    void reset_adaptive_state()
    {
        backend_->reset_adaptive_state();
    }

    void process_real(std::span<const double> x, std::span<double> output)
    {
        backend_->process_real(x, output);
    }

    void process_complex(std::span<const std::complex<double>> x, std::span<double> output)
    {
        backend_->process_complex(x, output);
    }

  private:
    std::unique_ptr<detail::MultitaperPsdBackend> backend_;
};

MultitaperPsdProcessor::MultitaperPsdProcessor(std::size_t n_samples, std::size_t n_channels,
                                               std::size_t fft_length, double density_scale,
                                               std::span<const double> tapers, std::size_t n_tapers,
                                               std::span<const double> concentration_ratios,
                                               MultitaperWeighting weighting, bool one_sided,
                                               bool real_input, SpectralBackend backend)
    : impl_(std::make_unique<Impl>(n_samples, n_channels, fft_length, density_scale, tapers,
                                   n_tapers, concentration_ratios, weighting, one_sided, real_input,
                                   backend))
{
}

MultitaperPsdProcessor::~MultitaperPsdProcessor() = default;

MultitaperPsdProcessor::MultitaperPsdProcessor(MultitaperPsdProcessor&&) noexcept = default;

MultitaperPsdProcessor&
MultitaperPsdProcessor::operator=(MultitaperPsdProcessor&&) noexcept = default;

std::size_t MultitaperPsdProcessor::output_bins() const noexcept
{
    return impl_->output_bins();
}

void MultitaperPsdProcessor::reset_adaptive_state()
{
    impl_->reset_adaptive_state();
}

void MultitaperPsdProcessor::process_real(std::span<const double> x, std::span<double> output)
{
    impl_->process_real(x, output);
}

void MultitaperPsdProcessor::process_complex(std::span<const std::complex<double>> x,
                                             std::span<double> output)
{
    impl_->process_complex(x, output);
}

} // namespace neurale::signal
