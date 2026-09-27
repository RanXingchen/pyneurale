/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace neurale::signal::simulation
{

/** Result of a non-allocating generation call. */
enum class GenerationStatus
{
    ok,
    invalid_output,
    range_overflow,
    source_exhausted,
};

/**
 * @brief Stateless sampled-signal generator indexed by absolute sample position.
 *
 * Output uses sample-major `(sample, channel)` order. Construction may allocate;
 * generate() does not allocate and writes directly into caller-owned memory.
 */
class SignalGenerator
{
  public:
    /** Create a zero-valued generator. */
    [[nodiscard]] static SignalGenerator zeros(std::size_t n_channels, double fs);

    /** Create a per-channel constant generator. */
    [[nodiscard]] static SignalGenerator constant(std::size_t n_channels, double fs,
                                                  std::span<const double> values);

    /**
     * @brief Create a sum-of-tones generator.
     *
     * Frequency, amplitude, and phase arrays use `(tone, channel)` order and
     * must each contain `n_tones * n_channels` values.
     */
    [[nodiscard]] static SignalGenerator tones(std::size_t n_channels, double fs,
                                               std::size_t n_tones, std::span<const double> freqs,
                                               std::span<const double> amps,
                                               std::span<const double> phases);

    /** Create deterministic counter-based uniform noise in [low, high). */
    [[nodiscard]] static SignalGenerator noise(std::size_t n_channels, double fs,
                                               std::uint64_t seed, double low, double high);

    /**
     * @brief Create a finite or repeating generator from sample-major values.
     */
    [[nodiscard]] static SignalGenerator samples(std::size_t n_channels, double fs,
                                                 std::size_t n_samples, std::span<const double> x,
                                                 bool repeat);

    [[nodiscard]] std::size_t channel_count() const noexcept
    {
        return n_channels_;
    }

    [[nodiscard]] double sample_rate() const noexcept
    {
        return fs_;
    }

    /** Number of addressable samples for a non-repeating supplied source. */
    [[nodiscard]] std::optional<std::uint64_t> finite_sample_count() const noexcept;

    /**
     * @brief Fill output for `[start, start + count)` without allocating.
     *
     * output must contain exactly `count * channel_count()` doubles.
     */
    [[nodiscard]] GenerationStatus generate(std::uint64_t start, std::size_t count,
                                            std::span<double> output) const noexcept;

  private:
    enum class Kind
    {
        zeros,
        constant,
        tones,
        noise,
        samples,
    };

    explicit SignalGenerator(Kind kind, std::size_t n_channels, double fs);

    Kind kind_;
    std::size_t n_channels_;
    double fs_;
    std::size_t n_tones = 0;
    std::size_t n_samples = 0;
    bool repeat_ = false;
    std::uint64_t seed_ = 0;
    double noise_low_ = 0.0;
    double noise_scale_ = 0.0;
    std::vector<double> values_;
    std::vector<double> amps_;
    std::vector<double> phases_;
};

} // namespace neurale::signal::simulation
