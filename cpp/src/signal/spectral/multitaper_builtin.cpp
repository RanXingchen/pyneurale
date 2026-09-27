/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "multitaper_internal.h"

#include "fft_dispatch.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurale::signal::detail
{
namespace
{

class BuiltinMultitaperPsdBackend final : public MultitaperPsdBackend
{
  public:
    explicit BuiltinMultitaperPsdBackend(MultitaperPsdSpec spec)
        : spec_(std::move(spec)), spectra_(spec_.n_tapers * spec_.spectrum_bins),
          adaptive_state_(spec_.n_channels * spec_.spectrum_bins),
          adaptive_valid_(spec_.n_channels), combined_(spec_.spectrum_bins),
          next_(spec_.spectrum_bins), weight_sum_(spec_.spectrum_bins),
          complex_buffer_(std::max(spec_.n_samples, spec_.fft_length)),
          complex_output_(spec_.fft_length)
    {
    }

    [[nodiscard]] std::size_t output_bins() const noexcept override
    {
        return spec_.output_bins;
    }

    void reset_adaptive_state() override
    {
        std::fill(adaptive_state_.begin(), adaptive_state_.end(), 0.0);
        std::fill(adaptive_valid_.begin(), adaptive_valid_.end(), 0);
    }

    void process_real(std::span<const double> x, std::span<double> output) override
    {
        if (!spec_.real_input)
        {
            throw std::invalid_argument("processor was constructed for complex input");
        }
        validate_process_buffers(x.size(), output.size());

        for (std::size_t channel = 0; channel < spec_.n_channels; ++channel)
        {
            const auto power = real_channel_spectra(x, channel);
            combine_channel(power, channel, true);
            write_real_channel(channel, output);
        }
    }

    void process_complex(std::span<const std::complex<double>> x, std::span<double> output) override
    {
        if (spec_.real_input)
        {
            throw std::invalid_argument("processor was constructed for real input");
        }
        validate_process_buffers(x.size(), output.size());

        for (std::size_t channel = 0; channel < spec_.n_channels; ++channel)
        {
            const auto power = complex_channel_spectra(x, channel);
            combine_channel(power, channel, true);
            write_complex_channel(channel, output);
        }
    }

  private:
    void validate_process_buffers(std::size_t input_size, std::size_t output_size) const
    {
        if (input_size != spec_.n_samples * spec_.n_channels)
        {
            throw std::invalid_argument("multitaper PSD input shape is incorrect");
        }
        if (output_size != spec_.output_bins * spec_.n_channels)
        {
            throw std::invalid_argument("multitaper PSD output shape is incorrect");
        }
    }

    double taper_at(std::size_t sample, std::size_t taper) const noexcept
    {
        return spec_.tapers[sample * spec_.n_tapers + taper];
    }

    std::span<double> taper_spectrum(std::size_t taper) noexcept
    {
        return std::span<double>(spectra_.data() + taper * spec_.spectrum_bins,
                                 spec_.spectrum_bins);
    }

    std::span<const double> taper_spectrum(std::size_t taper) const noexcept
    {
        return std::span<const double>(spectra_.data() + taper * spec_.spectrum_bins,
                                       spec_.spectrum_bins);
    }

    double real_channel_spectra(std::span<const double> x, std::size_t channel)
    {
        double power = 0.0;
        for (std::size_t sample = 0; sample < spec_.n_samples; ++sample)
        {
            const auto value = x[sample * spec_.n_channels + channel];
            power += value * value;
        }

        for (std::size_t taper = 0; taper < spec_.n_tapers; ++taper)
        {
            if (spec_.n_samples <= spec_.fft_length)
            {
                real_fft_spectrum(x, channel, taper);
            }
            else
            {
                real_czt_spectrum(x, channel, taper);
            }
        }
        return power / static_cast<double>(spec_.n_samples);
    }

    void real_fft_spectrum(std::span<const double> x, std::size_t channel, std::size_t taper)
    {
        std::fill(complex_buffer_.begin(), complex_buffer_.begin() + spec_.fft_length,
                  std::complex<double>{});
        for (std::size_t sample = 0; sample < spec_.n_samples; ++sample)
        {
            complex_buffer_[sample] =
                x[sample * spec_.n_channels + channel] * taper_at(sample, taper);
        }
        fft_inplace(std::span<std::complex<double>>(complex_buffer_.data(), spec_.fft_length),
                    false);
        write_power_spectrum(
            std::span<const std::complex<double>>(complex_buffer_.data(), spec_.spectrum_bins),
            taper_spectrum(taper));
    }

    void real_czt_spectrum(std::span<const double> x, std::size_t channel, std::size_t taper)
    {
        for (std::size_t sample = 0; sample < spec_.n_samples; ++sample)
        {
            complex_buffer_[sample] =
                x[sample * spec_.n_channels + channel] * taper_at(sample, taper);
        }
        czt(std::span<const std::complex<double>>(complex_buffer_.data(), spec_.n_samples),
            spec_.n_samples, 1, spec_.fft_length,
            std::polar(1.0,
                       -2.0 * std::numbers::pi_v<double> / static_cast<double>(spec_.fft_length)),
            {1.0, 0.0}, std::span<std::complex<double>>(complex_output_.data(), spec_.fft_length));
        write_power_spectrum(
            std::span<const std::complex<double>>(complex_output_.data(), spec_.spectrum_bins),
            taper_spectrum(taper));
    }

    double complex_channel_spectra(std::span<const std::complex<double>> x, std::size_t channel)
    {
        double power = 0.0;
        for (std::size_t sample = 0; sample < spec_.n_samples; ++sample)
        {
            const auto value = x[sample * spec_.n_channels + channel];
            power += value.real() * value.real() + value.imag() * value.imag();
        }

        for (std::size_t taper = 0; taper < spec_.n_tapers; ++taper)
        {
            if (spec_.n_samples <= spec_.fft_length)
            {
                complex_fft_spectrum(x, channel, taper);
            }
            else
            {
                complex_czt_spectrum(x, channel, taper);
            }
        }
        return power / static_cast<double>(spec_.n_samples);
    }

    void complex_fft_spectrum(std::span<const std::complex<double>> x, std::size_t channel,
                              std::size_t taper)
    {
        std::fill(complex_buffer_.begin(), complex_buffer_.begin() + spec_.fft_length,
                  std::complex<double>{});
        for (std::size_t sample = 0; sample < spec_.n_samples; ++sample)
        {
            complex_buffer_[sample] =
                x[sample * spec_.n_channels + channel] * taper_at(sample, taper);
        }
        fft_inplace(std::span<std::complex<double>>(complex_buffer_.data(), spec_.fft_length),
                    false);
        write_power_spectrum(
            std::span<const std::complex<double>>(complex_buffer_.data(), spec_.fft_length),
            taper_spectrum(taper));
    }

    void complex_czt_spectrum(std::span<const std::complex<double>> x, std::size_t channel,
                              std::size_t taper)
    {
        for (std::size_t sample = 0; sample < spec_.n_samples; ++sample)
        {
            complex_buffer_[sample] =
                x[sample * spec_.n_channels + channel] * taper_at(sample, taper);
        }
        czt(std::span<const std::complex<double>>(complex_buffer_.data(), spec_.n_samples),
            spec_.n_samples, 1, spec_.fft_length,
            std::polar(1.0,
                       -2.0 * std::numbers::pi_v<double> / static_cast<double>(spec_.fft_length)),
            {1.0, 0.0}, std::span<std::complex<double>>(complex_output_.data(), spec_.fft_length));
        write_power_spectrum(
            std::span<const std::complex<double>>(complex_output_.data(), spec_.fft_length),
            taper_spectrum(taper));
    }

    static void write_power_spectrum(std::span<const std::complex<double>> values,
                                     std::span<double> output)
    {
        for (std::size_t bin = 0; bin < output.size(); ++bin)
        {
            const auto real = values[bin].real();
            const auto imag = values[bin].imag();
            output[bin] = real * real + imag * imag;
        }
    }

    void combine_channel(double power, std::size_t channel, bool use_adaptive_state)
    {
        if (spec_.weighting == MultitaperWeighting::adaptive)
        {
            adaptive_channel(power, channel, use_adaptive_state);
            return;
        }
        std::fill(combined_.begin(), combined_.end(), 0.0);
        for (std::size_t taper = 0; taper < spec_.n_tapers; ++taper)
        {
            const auto weight =
                spec_.weighting == MultitaperWeighting::unity ? 1.0 : spec_.ratios[taper];
            const auto spectrum = taper_spectrum(taper);
            for (std::size_t bin = 0; bin < spec_.spectrum_bins; ++bin)
            {
                combined_[bin] += weight * spectrum[bin];
            }
        }
        const auto scale = 1.0 / static_cast<double>(spec_.n_tapers);
        for (auto& value : combined_)
        {
            value *= scale;
        }
    }

    void adaptive_channel(double power, std::size_t channel, bool use_state)
    {
        auto state = std::span<double>(adaptive_state_.data() + channel * spec_.spectrum_bins,
                                       spec_.spectrum_bins);
        if (power == 0.0)
        {
            std::fill(combined_.begin(), combined_.end(), 0.0);
            if (use_state)
            {
                std::fill(state.begin(), state.end(), 0.0);
                adaptive_valid_[channel] = 0;
            }
            return;
        }

        if (use_state && adaptive_valid_[channel] != 0)
        {
            std::copy(state.begin(), state.end(), combined_.begin());
        }
        else
        {
            const auto first = taper_spectrum(0);
            const auto second = taper_spectrum(1);
            for (std::size_t bin = 0; bin < spec_.spectrum_bins; ++bin)
            {
                combined_[bin] = 0.5 * (first[bin] + second[bin]);
            }
        }

        const auto tol = 0.0005 * power;
        for (std::size_t iteration = 0; iteration < 100; ++iteration)
        {
            accumulate_adaptive_taper<true>(0, power);
            for (std::size_t taper = 1; taper < spec_.n_tapers; ++taper)
            {
                accumulate_adaptive_taper<false>(taper, power);
            }

            double difference = 0.0;
            for (std::size_t bin = 0; bin < spec_.spectrum_bins; ++bin)
            {
                next_[bin] = weight_sum_[bin] == 0.0 ? 0.0 : next_[bin] / weight_sum_[bin];
                difference += std::abs(next_[bin] - combined_[bin]);
            }
            if (spec_.real_input)
            {
                difference = real_two_sided_difference(difference);
            }
            combined_.swap(next_);
            if (difference <= tol)
            {
                break;
            }
        }

        if (use_state)
        {
            std::copy(combined_.begin(), combined_.end(), state.begin());
            adaptive_valid_[channel] = 1;
        }
    }

    template <bool Initialize> void accumulate_adaptive_taper(std::size_t taper, double power)
    {
        const auto ratio = spec_.ratios[taper];
        const auto leakage = power * spec_.one_minus_ratios[taper];
        const auto spectrum = taper_spectrum(taper);
        for (std::size_t bin = 0; bin < spec_.spectrum_bins; ++bin)
        {
            const auto den = ratio * combined_[bin] + leakage;
            const auto weight = den == 0.0 ? 0.0 : combined_[bin] / den;
            const auto adaptive_weight = weight * weight * ratio;
            if constexpr (Initialize)
            {
                next_[bin] = adaptive_weight * spectrum[bin];
                weight_sum_[bin] = adaptive_weight;
            }
            else
            {
                next_[bin] += adaptive_weight * spectrum[bin];
                weight_sum_[bin] += adaptive_weight;
            }
        }
    }

    double real_two_sided_difference(double one_sided_difference) const
    {
        auto difference = 2.0 * one_sided_difference;
        if (!next_.empty())
        {
            difference -= std::abs(next_[0] - combined_[0]);
        }
        if (spec_.fft_length % 2 == 0 && spec_.spectrum_bins > 1)
        {
            difference -=
                std::abs(next_[spec_.spectrum_bins - 1] - combined_[spec_.spectrum_bins - 1]);
        }
        return difference;
    }

    void write_real_channel(std::size_t channel, std::span<double> output) const
    {
        for (std::size_t bin = 0; bin < spec_.output_bins; ++bin)
        {
            output[bin * spec_.n_channels + channel] =
                combined_[spec_.output_source_bin[bin]] * spec_.output_scale[bin];
        }
    }

    void write_complex_channel(std::size_t channel, std::span<double> output) const
    {
        const auto scale = 1.0 / spec_.density_scale;
        for (std::size_t bin = 0; bin < spec_.fft_length; ++bin)
        {
            output[bin * spec_.n_channels + channel] = combined_[bin] * scale;
        }
    }

    MultitaperPsdSpec spec_;
    std::vector<double> spectra_;
    std::vector<double> adaptive_state_;
    std::vector<unsigned char> adaptive_valid_;
    std::vector<double> combined_;
    std::vector<double> next_;
    std::vector<double> weight_sum_;
    std::vector<std::complex<double>> complex_buffer_;
    std::vector<std::complex<double>> complex_output_;
};

} // namespace

std::unique_ptr<MultitaperPsdBackend> create_builtin_multitaper_psd_backend(MultitaperPsdSpec spec)
{
    return std::make_unique<BuiltinMultitaperPsdBackend>(std::move(spec));
}

} // namespace neurale::signal::detail
