/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <span>

namespace neurale::signal
{

/**
 * @brief Rule used to combine individual tapered power spectra.
 */
enum class MultitaperWeighting
{
    /// Give every taper equal weight.
    unity,
    /// Weight tapers by their spectral concentration ratios.
    eigen,
    /// Estimate frequency-dependent adaptive weights.
    adaptive,
};

enum class SpectralBackend
{
    automatic,
    builtin,
};

/**
 * @brief Compute the chirp Z-transform of sample-major complex data.
 *
 * @param x Complex input with `n_samples * n_channels` elements.
 * @param n_samples Number of input samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param output_length Number of transform points per channel.
 * @param ratio Ratio between consecutive contour points.
 * @param start First contour point.
 * @param output Caller-owned complex output buffer with
 *        `output_length * n_channels` elements.
 * @throws std::invalid_argument If dimensions, contour parameters, or buffer
 *         sizes are invalid.
 */
void czt(std::span<const std::complex<double>> x, std::size_t n_samples, std::size_t n_channels,
         std::size_t output_length, std::complex<double> ratio, std::complex<double> start,
         std::span<std::complex<double>> output);

/**
 * @brief Construct the analytic signal of real sample-major data.
 *
 * @param x Real input with `n_samples * n_channels` elements.
 * @param n_samples Number of input samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param fft_length Transform length, at least `n_samples`.
 * @param output Caller-owned complex output buffer with
 *        `fft_length * n_channels` elements.
 * @throws std::invalid_argument If the transform length or buffer sizes are
 *         invalid.
 */
void analytic_signal(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                     std::size_t fft_length, std::span<std::complex<double>> output);

/**
 * @brief Reusable fixed-shape analytic-signal processor.
 *
 * Construction prepares all backend state and workspace. The builtin backend
 * requires a power-of-two FFT length and does not allocate while processing.
 */
class AnalyticSignalProcessor
{
  public:
    AnalyticSignalProcessor(std::size_t n_samples, std::size_t n_channels, std::size_t fft_length,
                            SpectralBackend backend = SpectralBackend::automatic);
    ~AnalyticSignalProcessor();

    AnalyticSignalProcessor(const AnalyticSignalProcessor&) = delete;
    AnalyticSignalProcessor& operator=(const AnalyticSignalProcessor&) = delete;
    AnalyticSignalProcessor(AnalyticSignalProcessor&&) noexcept;
    AnalyticSignalProcessor& operator=(AnalyticSignalProcessor&&) noexcept;

    void process(std::span<const double> x, std::span<std::complex<double>> output);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Reusable multitaper PSD processor for fixed-shape data.
 *
 * A processor instance is not thread-safe. Do not call process_real(),
 * process_complex(), or reset_adaptive_state() concurrently on the same
 * instance.
 */
class MultitaperPsdProcessor
{
  public:
    /**
     * @brief Construct a fixed-shape multitaper PSD processor.
     *
     * @param n_samples Samples per input channel.
     * @param n_channels Number of input channels.
     * @param fft_length FFT length used for each tapered spectrum.
     * @param density_scale Scale factor used for density normalization.
     * @param tapers Flattened `(n_tapers, n_samples)` DPSS tapers.
     * @param n_tapers Number of tapers.
     * @param concentration_ratios One spectral concentration ratio per taper.
     * @param weighting Taper weighting rule.
     * @param one_sided Emit one-sided output bins.
     * @param real_input Require real-valued input processing.
     * @param backend SpectralBackend
     * @throws std::invalid_argument If dimensions or buffers are invalid.
     */
    MultitaperPsdProcessor(std::size_t n_samples, std::size_t n_channels, std::size_t fft_length,
                           double density_scale, std::span<const double> tapers,
                           std::size_t n_tapers, std::span<const double> concentration_ratios,
                           MultitaperWeighting weighting, bool one_sided, bool real_input,
                           SpectralBackend backend = SpectralBackend::automatic);

    ~MultitaperPsdProcessor();

    MultitaperPsdProcessor(const MultitaperPsdProcessor&) = delete;
    MultitaperPsdProcessor& operator=(const MultitaperPsdProcessor&) = delete;
    MultitaperPsdProcessor(MultitaperPsdProcessor&&) noexcept;
    MultitaperPsdProcessor& operator=(MultitaperPsdProcessor&&) noexcept;

    /**
     * @brief Return the number of PSD bins emitted per channel.
     */
    [[nodiscard]] std::size_t output_bins() const noexcept;

    /**
     * @brief Clear the adaptive multitaper warm-start state.
     */
    void reset_adaptive_state();

    /**
     * @brief Compute PSD output for one real-valued input block.
     *
     * @param x Real input with `n_samples * n_channels` elements.
     * @param output PSD output with `output_bins() * n_channels` elements.
     * @throws std::invalid_argument If input or output sizes are invalid.
     */
    void process_real(std::span<const double> x, std::span<double> output);

    /**
     * @brief Compute PSD output for one complex-valued input block.
     *
     * @param x Complex input with `n_samples * n_channels` elements.
     * @param output PSD output with `output_bins() * n_channels` elements.
     * @throws std::invalid_argument If input or output sizes are invalid.
     */
    void process_complex(std::span<const std::complex<double>> x, std::span<double> output);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neurale::signal
