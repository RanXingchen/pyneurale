/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace neurale::signal::simulation
{

struct Intent2D
{
    double x{};
    double y{};
};

struct NeuralSignalConfig
{
    std::size_t n_channels{16};
    double fs{30'000.0};
    std::size_t units_per_channel{4};
    std::uint64_t seed{1};

    double baseline_rate_hz{12.0};
    double intent_rate_gain_hz{30.0};
    double max_rate_hz{120.0};
    double refractory_seconds{0.001};
    double preferred_direction_jitter_radians{0.08};

    double spike_amplitude_volts{80e-6};
    double spike_duration_seconds{0.0025};
    double spike_spatial_decay_channels{0.75}; ///< Exponential decay along linear channel order.

    double lfp_frequency_hz{10.0};
    double lfp_damping_time_seconds{0.25};
    double lfp_std_volts{15e-6};
    double lfp_spatial_correlation{0.6};

    double background_noise_std_volts{5e-6};
    double background_time_constant_seconds{0.004};
    double background_spatial_correlation{0.25};

    double nonstationarity_std_fraction{0.15};
    double nonstationarity_time_constant_seconds{30.0};
    double rate_correlation{0.15};

    double drift_rotation_degrees{45.0};
    double drift_per_unit_rotation_std_degrees{};
    double drift_per_unit_rotation_limit_degrees{std::numeric_limits<double>::infinity()};
    double drift_tuning_gain_scale{1.0};
    double drift_baseline_rate_shift_hz{};
};

enum class NeuralGenerationStatus
{
    ok,
    invalid_output,
    invalid_spike_output,
    invalid_control,
    range_overflow,
};

/**
 * Stateful intent-driven extracellular signal surrogate.
 *
 * Output is sample-major `(sample, channel)` voltage. Construction and reset
 * may touch prepared storage; generate() performs no allocation and advances
 * the internal absolute sample index. An optional sample-major `(sample,
 * hidden_unit)` uint8 buffer exposes spike truth for validation.
 */
class NeuralSignalGenerator
{
  public:
    explicit NeuralSignalGenerator(NeuralSignalConfig config);

    NeuralSignalGenerator(const NeuralSignalGenerator&) = delete;
    NeuralSignalGenerator& operator=(const NeuralSignalGenerator&) = delete;
    NeuralSignalGenerator(NeuralSignalGenerator&&) = delete;
    NeuralSignalGenerator& operator=(NeuralSignalGenerator&&) = delete;

    [[nodiscard]] std::size_t channel_count() const noexcept;
    [[nodiscard]] std::size_t unit_count() const noexcept;
    [[nodiscard]] double sample_rate() const noexcept;
    [[nodiscard]] std::uint64_t sample_index() const noexcept;
    [[nodiscard]] std::uint64_t drift_fingerprint() const noexcept;

    [[nodiscard]] NeuralGenerationStatus
    generate(std::size_t count, Intent2D intent, double drift_progress, std::span<double> output,
             std::span<std::uint8_t> spike_output = {}) noexcept;

    void reset() noexcept;

  private:
    void prepare();

    NeuralSignalConfig config_;
    std::size_t n_units_{};
    std::size_t waveform_samples_{};
    std::uint64_t sample_index_{};
    std::size_t spike_tail_cursor_{};
    std::uint64_t refractory_samples_{};

    double rate_alpha_{};
    double rate_innovation_{};
    double background_alpha_{};
    double background_innovation_{};
    double lfp_decay_{};
    double lfp_cos_{};
    double lfp_sin_{};
    double lfp_innovation_{};

    double common_rate_state_{};
    double common_background_state_{};
    double common_lfp_x_{};
    double common_lfp_y_{};

    std::vector<double> preferred_angles_;
    std::vector<double> drift_rotation_radians_;
    std::vector<double> directional_drive_;
    std::vector<std::size_t> unit_channel_offsets_;
    std::vector<std::size_t> unit_channels_;
    std::vector<double> unit_channel_weights_;
    std::vector<double> spike_waveform_;
    std::vector<double> spike_tail_;
    std::vector<double> rate_states_;
    std::vector<std::uint64_t> next_allowed_sample_;
    std::vector<double> background_states_;
    std::vector<double> lfp_x_;
    std::vector<double> lfp_y_;
};

} // namespace neurale::signal::simulation
