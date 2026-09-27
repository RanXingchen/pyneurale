/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/features/online.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurale::features
{
namespace
{

std::size_t checked_product(std::size_t left, std::size_t right)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error("online feature buffer size overflow");
    }
    return left * right;
}

bool is_power_of_two(std::size_t value) noexcept
{
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_geometry(std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels)
{
    if (window_samples == 0 || hop_samples == 0 || n_channels == 0)
    {
        throw std::invalid_argument("window, hop, and channel counts must be positive");
    }
    if (hop_samples > window_samples)
    {
        throw std::invalid_argument("hop samples must not exceed window samples");
    }
}

class FixedWindowBuffer
{
  public:
    FixedWindowBuffer(std::size_t window_samples, std::size_t hop_samples, std::size_t width)
        : window_samples_(window_samples), hop_samples_(hop_samples), width_(width),
          ring_(checked_product(window_samples, width))
    {
        validate_geometry(window_samples, hop_samples, width);
        reset();
    }

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const
    {
        if (input_samples > std::numeric_limits<std::size_t>::max() - samples_seen_)
        {
            throw std::overflow_error("online feature sample count overflow");
        }
        const auto end = samples_seen_ + input_samples;
        if (end < next_window_end_)
        {
            return 0;
        }
        return 1 + (end - next_window_end_) / hop_samples_;
    }

    [[nodiscard]] bool full() const noexcept
    {
        return filled_ == window_samples_;
    }

    [[nodiscard]] std::span<const double> oldest_row() const noexcept
    {
        if (!full())
        {
            return {};
        }
        return std::span<const double>(ring_.data() + write_position_ * width_, width_);
    }

    [[nodiscard]] bool push(std::span<const double> row)
    {
        if (row.size() != width_)
        {
            throw std::invalid_argument("online feature row width is incorrect");
        }
        std::copy(row.begin(), row.end(),
                  ring_.begin() + static_cast<std::ptrdiff_t>(write_position_ * width_));
        write_position_ = (write_position_ + 1) % window_samples_;
        filled_ = std::min(filled_ + 1, window_samples_);
        ++samples_seen_;
        if (samples_seen_ != next_window_end_)
        {
            return false;
        }
        next_window_end_ += hop_samples_;
        return true;
    }

    void materialize(std::span<double> output) const
    {
        if (!full() || output.size() != ring_.size())
        {
            throw std::invalid_argument("online feature window is incomplete");
        }
        const auto tail = (window_samples_ - write_position_) * width_;
        std::copy_n(ring_.data() + write_position_ * width_, tail, output.data());
        std::copy_n(ring_.data(), write_position_ * width_, output.data() + tail);
    }

    void reset() noexcept
    {
        std::fill(ring_.begin(), ring_.end(), 0.0);
        write_position_ = 0;
        filled_ = 0;
        samples_seen_ = 0;
        next_window_end_ = window_samples_;
    }

  private:
    std::size_t window_samples_{};
    std::size_t hop_samples_{};
    std::size_t width_{};
    std::vector<double> ring_;
    std::size_t write_position_{};
    std::size_t filled_{};
    std::size_t samples_seen_{};
    std::size_t next_window_end_{};
};

class PreparedSosFilter
{
  public:
    PreparedSosFilter(std::span<const double> sos, std::size_t n_sections, std::size_t n_channels)
        : n_sections_(n_sections), n_channels_(n_channels),
          coefs_(checked_product(n_sections, std::size_t{5})),
          state_(checked_product(checked_product(n_sections, std::size_t{2}), n_channels))
    {
        if (n_sections == 0 || n_channels == 0 ||
            sos.size() != checked_product(n_sections, std::size_t{6}))
        {
            throw std::invalid_argument("SOS filter shape is incorrect");
        }
        for (std::size_t section = 0; section < n_sections_; ++section)
        {
            const auto* source = sos.data() + section * 6;
            const auto a0 = source[3];
            if (!std::isfinite(a0) || a0 == 0.0)
            {
                throw std::invalid_argument("SOS a0 must be finite and nonzero");
            }
            auto* destination = coefs_.data() + section * 5;
            destination[0] = source[0] / a0;
            destination[1] = source[1] / a0;
            destination[2] = source[2] / a0;
            destination[3] = source[4] / a0;
            destination[4] = source[5] / a0;
            for (std::size_t i = 0; i < 5; ++i)
            {
                if (!std::isfinite(destination[i]))
                {
                    throw std::invalid_argument("SOS coefficients must be finite");
                }
            }
        }
    }

    void process_sample(std::span<const double> input, std::span<double> output) noexcept
    {
        std::copy(input.begin(), input.end(), output.begin());
        for (std::size_t section = 0; section < n_sections_; ++section)
        {
            const auto* coefs = coefs_.data() + section * 5;
            auto* z1 = state_.data() + section * 2 * n_channels_;
            auto* z2 = z1 + n_channels_;
            for (std::size_t channel = 0; channel < n_channels_; ++channel)
            {
                const auto value = output[channel];
                const auto filtered = coefs[0] * value + z1[channel];
                z1[channel] = coefs[1] * value - coefs[3] * filtered + z2[channel];
                z2[channel] = coefs[2] * value - coefs[4] * filtered;
                output[channel] = filtered;
            }
        }
    }

    void reset() noexcept
    {
        std::fill(state_.begin(), state_.end(), 0.0);
    }

  private:
    std::size_t n_sections_{};
    std::size_t n_channels_{};
    std::vector<double> coefs_;
    std::vector<double> state_;
};

void detrend_window(std::span<double> window, std::size_t n_samples, std::size_t n_channels,
                    Detrend detrend)
{
    if (detrend == Detrend::none)
    {
        return;
    }
    const auto x_mean = (static_cast<double>(n_samples) - 1.0) / 2.0;
    const auto den =
        static_cast<double>(n_samples) * (static_cast<double>(n_samples) * n_samples - 1.0) / 12.0;
    for (std::size_t channel = 0; channel < n_channels; ++channel)
    {
        double mean = 0.0;
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            mean += window[sample * n_channels + channel];
        }
        mean /= static_cast<double>(n_samples);
        if (detrend == Detrend::mean || n_samples == 1)
        {
            for (std::size_t sample = 0; sample < n_samples; ++sample)
            {
                window[sample * n_channels + channel] -= mean;
            }
            continue;
        }
        double covariance = 0.0;
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            covariance += (static_cast<double>(sample) - x_mean) *
                          (window[sample * n_channels + channel] - mean);
        }
        const auto slope = covariance / den;
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            window[sample * n_channels + channel] -=
                mean + slope * (static_cast<double>(sample) - x_mean);
        }
    }
}

void validate_process_buffers(std::span<const double> input, std::size_t input_samples,
                              std::size_t n_channels, std::span<const double> output,
                              std::size_t output_rows, std::size_t n_features)
{
    if (input.size() != checked_product(input_samples, n_channels) ||
        output.size() != checked_product(output_rows, n_features))
    {
        throw std::invalid_argument("online feature process buffer shape is incorrect");
    }
}

} // namespace

class BandpowerProcessor::Impl
{
  public:
    Impl(std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
         std::size_t fft_length, double density_scale, std::span<const double> tapers,
         std::size_t n_tapers, std::span<const double> concentration_ratios,
         signal::MultitaperWeighting weighting, std::span<const std::size_t> band_bins,
         double freq_step, Detrend detrend, signal::SpectralBackend backend)
        : window_samples_(window_samples), hop_samples_(hop_samples), n_channels_(n_channels),
          window_(window_samples, hop_samples, n_channels),
          window_scratch_(checked_product(window_samples, n_channels)),
          processor_(window_samples, n_channels, fft_length, density_scale, tapers, n_tapers,
                     concentration_ratios, weighting, true, true, backend),
          psd_(checked_product(processor_.output_bins(), n_channels)),
          band_bins_(band_bins.begin(), band_bins.end()), freq_step_(freq_step), detrend_(detrend)
    {
        validate_geometry(window_samples, hop_samples, n_channels);
        if (fft_length < window_samples || !std::isfinite(freq_step) || freq_step <= 0.0 ||
            band_bins_.empty() || band_bins_.size() % 2 != 0)
        {
            throw std::invalid_argument("invalid online bandpower configuration");
        }
        if (backend == signal::SpectralBackend::builtin && !is_power_of_two(fft_length))
        {
            throw std::invalid_argument(
                "builtin online bandpower requires a power-of-two FFT length");
        }
        for (std::size_t band = 0; band < band_bins_.size(); band += 2)
        {
            if (band_bins_[band] >= band_bins_[band + 1] ||
                band_bins_[band + 1] > processor_.output_bins())
            {
                throw std::invalid_argument("bandpower frequency-bin range is invalid");
            }
        }
        checked_product(band_bins_.size() / 2, n_channels_);
    }

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const
    {
        return window_.output_count(input_samples);
    }

    [[nodiscard]] std::size_t feature_count() const noexcept
    {
        return band_bins_.size() / 2 * n_channels_;
    }

    void process(std::span<const double> input, std::size_t input_samples, std::span<double> output)
    {
        const auto rows = output_count(input_samples);
        validate_process_buffers(input, input_samples, n_channels_, output, rows, feature_count());
        std::size_t output_row = 0;
        for (std::size_t sample = 0; sample < input_samples; ++sample)
        {
            const auto row = input.subspan(sample * n_channels_, n_channels_);
            if (!window_.push(row))
            {
                continue;
            }
            window_.materialize(window_scratch_);
            detrend_window(window_scratch_, window_samples_, n_channels_, detrend_);
            processor_.reset_adaptive_state();
            processor_.process_real(window_scratch_, psd_);
            auto destination = output.subspan(output_row * feature_count(), feature_count());
            for (std::size_t band = 0; band < band_bins_.size() / 2; ++band)
            {
                const auto begin = band_bins_[2 * band];
                const auto end = band_bins_[2 * band + 1];
                for (std::size_t channel = 0; channel < n_channels_; ++channel)
                {
                    double power = 0.0;
                    for (std::size_t bin = begin; bin < end; ++bin)
                    {
                        power += psd_[bin * n_channels_ + channel];
                    }
                    destination[band * n_channels_ + channel] = power * freq_step_;
                }
            }
            ++output_row;
        }
    }

    void reset() noexcept
    {
        window_.reset();
        processor_.reset_adaptive_state();
    }

    std::size_t window_samples_{};
    std::size_t hop_samples_{};
    std::size_t n_channels_{};
    FixedWindowBuffer window_;
    std::vector<double> window_scratch_;
    signal::MultitaperPsdProcessor processor_;
    std::vector<double> psd_;
    std::vector<std::size_t> band_bins_;
    double freq_step_{};
    Detrend detrend_{Detrend::none};
};

class HilbertEnvelopeProcessor::Impl
{
  public:
    Impl(std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
         std::span<const double> sos, std::size_t n_bands, std::size_t n_sections,
         std::size_t fft_length, signal::SpectralBackend backend)
        : window_samples_(window_samples), hop_samples_(hop_samples), n_channels_(n_channels),
          n_bands_(n_bands), combined_width_(checked_product(n_bands, n_channels)),
          window_(window_samples, hop_samples, combined_width_), filtered_row_(combined_width_),
          combined_window_(checked_product(window_samples, combined_width_)),
          band_window_(checked_product(window_samples, n_channels)),
          analytic_output_(checked_product(fft_length, n_channels)),
          analytic_(window_samples, n_channels, fft_length, backend)
    {
        validate_geometry(window_samples, hop_samples, n_channels);
        if (n_bands == 0 || n_sections == 0 ||
            sos.size() != checked_product(checked_product(n_bands, n_sections), std::size_t{6}))
        {
            throw std::invalid_argument("Hilbert SOS bank shape is incorrect");
        }
        filters_.reserve(n_bands);
        const auto band_size = n_sections * 6;
        for (std::size_t band = 0; band < n_bands; ++band)
        {
            filters_.emplace_back(sos.subspan(band * band_size, band_size), n_sections, n_channels);
        }
    }

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const
    {
        return window_.output_count(input_samples);
    }

    [[nodiscard]] std::size_t feature_count() const noexcept
    {
        return combined_width_;
    }

    void process(std::span<const double> input, std::size_t input_samples, std::span<double> output)
    {
        const auto rows = output_count(input_samples);
        validate_process_buffers(input, input_samples, n_channels_, output, rows, feature_count());
        std::size_t output_row = 0;
        for (std::size_t sample = 0; sample < input_samples; ++sample)
        {
            const auto source = input.subspan(sample * n_channels_, n_channels_);
            for (std::size_t band = 0; band < n_bands_; ++band)
            {
                filters_[band].process_sample(
                    source,
                    std::span<double>(filtered_row_.data() + band * n_channels_, n_channels_));
            }
            if (!window_.push(filtered_row_))
            {
                continue;
            }
            window_.materialize(combined_window_);
            auto destination = output.subspan(output_row * feature_count(), feature_count());
            for (std::size_t band = 0; band < n_bands_; ++band)
            {
                for (std::size_t sample_idx = 0; sample_idx < window_samples_; ++sample_idx)
                {
                    const auto* source_row =
                        combined_window_.data() + sample_idx * combined_width_ + band * n_channels_;
                    std::copy_n(source_row, n_channels_,
                                band_window_.data() + sample_idx * n_channels_);
                }
                analytic_.process(band_window_, analytic_output_);
                for (std::size_t channel = 0; channel < n_channels_; ++channel)
                {
                    double envelope = 0.0;
                    for (std::size_t sample_idx = 0; sample_idx < window_samples_; ++sample_idx)
                    {
                        envelope += std::abs(analytic_output_[sample_idx * n_channels_ + channel]);
                    }
                    destination[band * n_channels_ + channel] =
                        envelope / static_cast<double>(window_samples_);
                }
            }
            ++output_row;
        }
    }

    void reset() noexcept
    {
        window_.reset();
        for (auto& filter : filters_)
        {
            filter.reset();
        }
    }

    std::size_t window_samples_{};
    std::size_t hop_samples_{};
    std::size_t n_channels_{};
    std::size_t n_bands_{};
    std::size_t combined_width_{};
    FixedWindowBuffer window_;
    std::vector<PreparedSosFilter> filters_;
    std::vector<double> filtered_row_;
    std::vector<double> combined_window_;
    std::vector<double> band_window_;
    std::vector<std::complex<double>> analytic_output_;
    signal::AnalyticSignalProcessor analytic_;
};

class LmpProcessor::Impl
{
  public:
    Impl(std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
         std::span<const double> sos, std::size_t n_sections)
        : window_samples_(window_samples), hop_samples_(hop_samples), n_channels_(n_channels),
          window_(window_samples, hop_samples, n_channels), filter_(sos, n_sections, n_channels),
          filtered_row_(n_channels), running_sum_(n_channels)
    {
        validate_geometry(window_samples, hop_samples, n_channels);
    }

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const
    {
        return window_.output_count(input_samples);
    }

    [[nodiscard]] std::size_t feature_count() const noexcept
    {
        return n_channels_;
    }

    void process(std::span<const double> input, std::size_t input_samples, std::span<double> output)
    {
        const auto rows = output_count(input_samples);
        validate_process_buffers(input, input_samples, n_channels_, output, rows, feature_count());
        std::size_t output_row = 0;
        for (std::size_t sample = 0; sample < input_samples; ++sample)
        {
            filter_.process_sample(input.subspan(sample * n_channels_, n_channels_), filtered_row_);
            if (window_.full())
            {
                const auto oldest = window_.oldest_row();
                for (std::size_t channel = 0; channel < n_channels_; ++channel)
                {
                    running_sum_[channel] -= oldest[channel];
                }
            }
            for (std::size_t channel = 0; channel < n_channels_; ++channel)
            {
                running_sum_[channel] += filtered_row_[channel];
            }
            if (!window_.push(filtered_row_))
            {
                continue;
            }
            auto destination = output.subspan(output_row * n_channels_, n_channels_);
            for (std::size_t channel = 0; channel < n_channels_; ++channel)
            {
                destination[channel] = running_sum_[channel] / static_cast<double>(window_samples_);
            }
            ++output_row;
        }
    }

    void reset() noexcept
    {
        window_.reset();
        filter_.reset();
        std::fill(running_sum_.begin(), running_sum_.end(), 0.0);
    }

    std::size_t window_samples_{};
    std::size_t hop_samples_{};
    std::size_t n_channels_{};
    FixedWindowBuffer window_;
    PreparedSosFilter filter_;
    std::vector<double> filtered_row_;
    std::vector<double> running_sum_;
};

BandpowerProcessor::BandpowerProcessor(std::size_t window_samples, std::size_t hop_samples,
                                       std::size_t n_channels, std::size_t fft_length,
                                       double density_scale, std::span<const double> tapers,
                                       std::size_t n_tapers,
                                       std::span<const double> concentration_ratios,
                                       signal::MultitaperWeighting weighting,
                                       std::span<const std::size_t> band_bins, double freq_step,
                                       Detrend detrend, signal::SpectralBackend backend)
    : impl_(std::make_unique<Impl>(window_samples, hop_samples, n_channels, fft_length,
                                   density_scale, tapers, n_tapers, concentration_ratios, weighting,
                                   band_bins, freq_step, detrend, backend))
{
}

BandpowerProcessor::~BandpowerProcessor() = default;
BandpowerProcessor::BandpowerProcessor(BandpowerProcessor&&) noexcept = default;
BandpowerProcessor& BandpowerProcessor::operator=(BandpowerProcessor&&) noexcept = default;

std::size_t BandpowerProcessor::output_count(std::size_t input_samples) const
{
    return impl_->output_count(input_samples);
}

std::size_t BandpowerProcessor::feature_count() const noexcept
{
    return impl_->feature_count();
}

std::size_t BandpowerProcessor::window_samples() const noexcept
{
    return impl_->window_samples_;
}

std::size_t BandpowerProcessor::hop_samples() const noexcept
{
    return impl_->hop_samples_;
}

void BandpowerProcessor::process(std::span<const double> input, std::size_t input_samples,
                                 std::span<double> output)
{
    impl_->process(input, input_samples, output);
}

void BandpowerProcessor::reset() noexcept
{
    impl_->reset();
}

HilbertEnvelopeProcessor::HilbertEnvelopeProcessor(std::size_t window_samples,
                                                   std::size_t hop_samples, std::size_t n_channels,
                                                   std::span<const double> sos, std::size_t n_bands,
                                                   std::size_t n_sections, std::size_t fft_length,
                                                   signal::SpectralBackend backend)
    : impl_(std::make_unique<Impl>(window_samples, hop_samples, n_channels, sos, n_bands,
                                   n_sections, fft_length, backend))
{
}

HilbertEnvelopeProcessor::~HilbertEnvelopeProcessor() = default;
HilbertEnvelopeProcessor::HilbertEnvelopeProcessor(HilbertEnvelopeProcessor&&) noexcept = default;
HilbertEnvelopeProcessor&
HilbertEnvelopeProcessor::operator=(HilbertEnvelopeProcessor&&) noexcept = default;

std::size_t HilbertEnvelopeProcessor::output_count(std::size_t input_samples) const
{
    return impl_->output_count(input_samples);
}

std::size_t HilbertEnvelopeProcessor::feature_count() const noexcept
{
    return impl_->feature_count();
}

std::size_t HilbertEnvelopeProcessor::window_samples() const noexcept
{
    return impl_->window_samples_;
}

std::size_t HilbertEnvelopeProcessor::hop_samples() const noexcept
{
    return impl_->hop_samples_;
}

void HilbertEnvelopeProcessor::process(std::span<const double> input, std::size_t input_samples,
                                       std::span<double> output)
{
    impl_->process(input, input_samples, output);
}

void HilbertEnvelopeProcessor::reset() noexcept
{
    impl_->reset();
}

LmpProcessor::LmpProcessor(std::size_t window_samples, std::size_t hop_samples,
                           std::size_t n_channels, std::span<const double> sos,
                           std::size_t n_sections)
    : impl_(std::make_unique<Impl>(window_samples, hop_samples, n_channels, sos, n_sections))
{
}

LmpProcessor::~LmpProcessor() = default;
LmpProcessor::LmpProcessor(LmpProcessor&&) noexcept = default;
LmpProcessor& LmpProcessor::operator=(LmpProcessor&&) noexcept = default;

std::size_t LmpProcessor::output_count(std::size_t input_samples) const
{
    return impl_->output_count(input_samples);
}

std::size_t LmpProcessor::feature_count() const noexcept
{
    return impl_->feature_count();
}

std::size_t LmpProcessor::window_samples() const noexcept
{
    return impl_->window_samples_;
}

std::size_t LmpProcessor::hop_samples() const noexcept
{
    return impl_->hop_samples_;
}

void LmpProcessor::process(std::span<const double> input, std::size_t input_samples,
                           std::span<double> output)
{
    impl_->process(input, input_samples, output);
}

void LmpProcessor::reset() noexcept
{
    impl_->reset();
}

} // namespace neurale::features
