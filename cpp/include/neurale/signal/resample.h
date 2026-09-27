/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace neurale::signal
{

/**
 * @brief Return the final sample count for a rational-rate conversion.
 *
 * @param input_length Number of input samples.
 * @param up Positive upsampling factor.
 * @param down Positive downsampling factor.
 * @return Output samples after processing and flushing.
 * @throws std::invalid_argument If up or down is zero.
 * @throws std::overflow_error If the computed length overflows size_t.
 */
[[nodiscard]] std::size_t resampled_length(std::size_t input_length, std::size_t up,
                                           std::size_t down);

/**
 * @brief Stateful polyphase FIR resampler for sample-major data.
 */
class Resampler
{
  public:
    /**
     * @brief Construct a rational-rate resampler.
     *
     * @param up Positive upsampling factor.
     * @param down Positive downsampling factor.
     * @param filter Prototype FIR anti-alias filter.
     * @throws std::invalid_argument If factors or filter coefficients are
     *         invalid.
     */
    Resampler(std::size_t up, std::size_t down, std::span<const double> filter);

    /**
     * @brief Return the reduced upsampling factor.
     */
    [[nodiscard]] std::size_t up() const noexcept
    {
        return up_;
    }

    /**
     * @brief Return the reduced downsampling factor.
     */
    [[nodiscard]] std::size_t down() const noexcept
    {
        return down_;
    }

    /**
     * @brief Return the initial transient samples trimmed from output.
     */
    [[nodiscard]] std::size_t initial_output_trim() const noexcept
    {
        return delay_;
    }

    /**
     * @brief Return final samples after processing input_length samples and
     *        flushing.
     */
    [[nodiscard]] std::size_t output_length(std::size_t input_length) const;

    /**
     * @brief Return the maximum samples one process() call may emit.
     */
    [[nodiscard]] std::size_t max_output_length(std::size_t input_length) const;

    /**
     * @brief Return the maximum output samples flush() may write.
     *
     * This is a buffer-capacity bound, not the number of zero input samples
     * internally consumed during flushing.
     */
    [[nodiscard]] std::size_t max_flush_length() const;

    /**
     * @brief Preallocate internal buffers for real-time processing.
     *
     * Calling process() with `n_samples <= max_input_samples` after prepare()
     * will not grow the internal workspace.
     *
     * @param n_channels Number of channels in later process() calls.
     * @param max_input_samples Maximum input samples per process() call.
     * @throws std::invalid_argument If n_channels is zero or conflicts with
     *         existing state.
     */
    void prepare(std::size_t n_channels, std::size_t max_input_samples);

    /**
     * @brief Resample one block of sample-major input.
     *
     * Input and output buffers use `(sample, channel)` order. The return value
     * is the number of output samples per channel, so written values are
     * `return_value * n_channels`.
     *
     * @throws std::invalid_argument If dimensions or output capacity are
     *         invalid.
     * @throws std::overflow_error If internal size calculations overflow.
     */
    std::size_t process(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                        std::span<double> output);

    /**
     * @brief Emit pending samples and reset the stream state.
     *
     * The return value is the number of output samples per channel, so written
     * values are `return_value * n_channels`.
     *
     * @throws std::invalid_argument If output capacity is insufficient.
     * @throws std::runtime_error If the flush path cannot produce the expected
     *         sample count.
     */
    std::size_t flush(std::span<double> output);

    /**
     * @brief Clear stream state while preserving the resampling filter.
     */
    void reset();

  private:
    void build_phases(std::span<const double> filter);
    void prepare_channels(std::size_t n_channels);
    void prepare_workspace(std::size_t n_samples);
    std::size_t process_samples(std::span<const double> x, std::size_t n_samples,
                                std::span<double> output, bool flush, std::size_t output_limit);

    std::size_t up_;
    std::size_t down_;
    std::size_t coefs_per_phase_ = 0;
    std::size_t delay_ = 0;
    std::vector<double> coefs_;
    std::size_t n_channels_ = 0;
    std::size_t phase_ = 0;
    std::size_t offset_ = 0;
    std::size_t pending_trim_ = 0;
    std::size_t total_input_ = 0;
    std::size_t emitted_output_ = 0;
    std::size_t input_overshoot_ = 0;
    std::vector<double> history_;
    std::vector<double> workspace_;
};

} // namespace neurale::signal
