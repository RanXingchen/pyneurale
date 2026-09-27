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

class SosRealtimeProcessor
{
  public:
    SosRealtimeProcessor(std::span<const double> sos, std::size_t n_channels,
                         std::span<const double> state);

    void process(std::span<double> frame, std::size_t n_samples);
    void reset();
    void set_state(std::span<const double> state);
    void state(std::span<double> output) const;

    [[nodiscard]] std::size_t n_sections() const noexcept;
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
    void process_channel(double& value, const double* coefs, double* state);
    void copy_public_state(std::span<const double> state);
    [[nodiscard]] std::size_t state_rows() const noexcept;
    [[nodiscard]] double* section_state(std::size_t section) noexcept;
    [[nodiscard]] const double* section_state(std::size_t section) const noexcept;

    std::size_t n_sections_{0};
    std::size_t n_channels_{1};
    Kernel kernel_{Kernel::scalar_df2t};
    std::vector<double> sos_;
    std::vector<double> state_;
};

} // namespace neurale::signal::detail
