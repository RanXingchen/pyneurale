/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace neurale::signal::detail
{

class FirRealtimeProcessor
{
  public:
    FirRealtimeProcessor(std::span<const double> taps, std::size_t n_channels,
                         std::span<const double> state);

    void process(std::span<double> frame, std::size_t n_samples);
    void reset();
    void set_state(std::span<const double> state);
    void state(std::span<double> output) const;

    [[nodiscard]] std::size_t order() const noexcept;
    [[nodiscard]] std::size_t n_channels() const noexcept;
    [[nodiscard]] std::string_view kernel_name() const noexcept;

  private:
    enum class Kernel
    {
        builtin_ring,
        mkl_channel_dot_ring,
        mkl_blas_ring,
    };

    void process_builtin(std::span<double> frame, std::size_t n_samples);
    void process_mkl_channel_dot(std::span<double> frame, std::size_t n_samples);
    void process_mkl(std::span<double> frame, std::size_t n_samples);
    void process_block(std::span<double> frame, std::size_t n_samples);
    void process_builtin_sample(double* row);
    void process_mkl_channel_dot_sample(double* row);
    void process_mkl_sample(double* row);
    [[nodiscard]] bool should_process_block(std::size_t n_samples) const noexcept;
    void advance() noexcept;
    void copy_public_state(std::span<const double> state);
    void copy_current_state(std::span<double> output) const;
    void copy_sample_major_state(std::span<double> output) const;
    void copy_channel_major_state(std::span<double> output) const;

    std::size_t n_channels_{1};
    std::size_t position_{0};
    Kernel kernel_{Kernel::builtin_ring};
    std::vector<double> taps_;
    std::vector<double> ring_;
    std::vector<double> channel_ring_;
    // Keep the internal class layout independent of the selected provider.
    // FirRealtimeProcessor is constructed by targets that intentionally do
    // not inherit neurale_signal's private backend compile definitions.
    int mkl_taps_{0};
    int mkl_channels_{0};
    std::vector<double> block_input_;
    std::vector<double> block_state_;
    std::vector<double> block_final_state_;
};

} // namespace neurale::signal::detail
