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

class IirRealtimeProcessor
{
  public:
    IirRealtimeProcessor(std::span<const double> b, std::span<const double> a,
                         std::size_t n_channels, std::span<const double> state);

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
        scalar_df2t,
        mkl_blas_df2t,
    };

    void process_scalar(std::span<double> frame, std::size_t n_samples);
    void process_mkl(std::span<double> frame, std::size_t n_samples);
    double process_channel(double x, double* state);
    void copy_public_state(std::span<const double> state);

    std::size_t n_channels_{1};
    Kernel kernel_{Kernel::scalar_df2t};
    std::vector<double> b_;
    std::vector<double> a_;
    std::vector<double> state_;
};

} // namespace neurale::signal::detail
