/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <neurale/signal/spectral.h>

namespace neurale::features
{

enum class Detrend
{
    none,
    mean,
    linear,
};

class BandpowerProcessor
{
  public:
    BandpowerProcessor(std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
                       std::size_t fft_length, double density_scale, std::span<const double> tapers,
                       std::size_t n_tapers, std::span<const double> concentration_ratios,
                       signal::MultitaperWeighting weighting,
                       std::span<const std::size_t> band_bins, double freq_step, Detrend detrend,
                       signal::SpectralBackend backend);
    ~BandpowerProcessor();

    BandpowerProcessor(const BandpowerProcessor&) = delete;
    BandpowerProcessor& operator=(const BandpowerProcessor&) = delete;
    BandpowerProcessor(BandpowerProcessor&&) noexcept;
    BandpowerProcessor& operator=(BandpowerProcessor&&) noexcept;

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const;
    [[nodiscard]] std::size_t feature_count() const noexcept;
    [[nodiscard]] std::size_t window_samples() const noexcept;
    [[nodiscard]] std::size_t hop_samples() const noexcept;
    void process(std::span<const double> input, std::size_t input_samples,
                 std::span<double> output);
    void reset() noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class HilbertEnvelopeProcessor
{
  public:
    HilbertEnvelopeProcessor(std::size_t window_samples, std::size_t hop_samples,
                             std::size_t n_channels, std::span<const double> sos,
                             std::size_t n_bands, std::size_t n_sections, std::size_t fft_length,
                             signal::SpectralBackend backend);
    ~HilbertEnvelopeProcessor();

    HilbertEnvelopeProcessor(const HilbertEnvelopeProcessor&) = delete;
    HilbertEnvelopeProcessor& operator=(const HilbertEnvelopeProcessor&) = delete;
    HilbertEnvelopeProcessor(HilbertEnvelopeProcessor&&) noexcept;
    HilbertEnvelopeProcessor& operator=(HilbertEnvelopeProcessor&&) noexcept;

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const;
    [[nodiscard]] std::size_t feature_count() const noexcept;
    [[nodiscard]] std::size_t window_samples() const noexcept;
    [[nodiscard]] std::size_t hop_samples() const noexcept;
    void process(std::span<const double> input, std::size_t input_samples,
                 std::span<double> output);
    void reset() noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class LmpProcessor
{
  public:
    LmpProcessor(std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
                 std::span<const double> sos, std::size_t n_sections);
    ~LmpProcessor();

    LmpProcessor(const LmpProcessor&) = delete;
    LmpProcessor& operator=(const LmpProcessor&) = delete;
    LmpProcessor(LmpProcessor&&) noexcept;
    LmpProcessor& operator=(LmpProcessor&&) noexcept;

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const;
    [[nodiscard]] std::size_t feature_count() const noexcept;
    [[nodiscard]] std::size_t window_samples() const noexcept;
    [[nodiscard]] std::size_t hop_samples() const noexcept;
    void process(std::span<const double> input, std::size_t input_samples,
                 std::span<double> output);
    void reset() noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neurale::features
