/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/neural_simulation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "counter_random.h"

namespace neurale::signal::simulation
{
namespace
{

constexpr std::uint64_t kPopulationSeed = 0x706f70756c617469ULL;
constexpr std::uint64_t kSpikeSeed = 0x7370696b655f6576ULL;
constexpr std::uint64_t kRateSeed = 0x726174655f6f755fULL;
constexpr std::uint64_t kLfpSeed = 0x6c66705f6f736369ULL;
constexpr std::uint64_t kNoiseSeed = 0x6e6f6973655f6172ULL;
constexpr std::uint64_t kInitialStateSeed = 0x696e697469616c5fULL;
constexpr std::uint64_t kDriftSeed = 0x64726966745f7064ULL;
constexpr std::uint64_t kMaxTruncatedGaussianDraws = 1'024;
constexpr double kMaximumTruncatedGaussianFailureProbability = 1e-18;
constexpr double kMinimumSpatialWeight = 1e-4;

using detail::gaussian;
using detail::kTwoPi;
using detail::uniform_open;

void require_finite(double value, const char* name)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string(name) + " must be finite");
}

void require_nonnegative(double value, const char* name)
{
    require_finite(value, name);
    if (value < 0.0)
        throw std::invalid_argument(std::string(name) + " must be non-negative");
}

void require_positive(double value, const char* name)
{
    require_finite(value, name);
    if (value <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be positive");
}

void require_correlation(double value, const char* name)
{
    require_finite(value, name);
    if (value < 0.0 || value > 1.0)
        throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
}

[[nodiscard]] NeuralSignalConfig validate(NeuralSignalConfig config)
{
    if (config.n_channels == 0)
        throw std::invalid_argument("n_channels must be positive");
    if (config.units_per_channel == 0)
        throw std::invalid_argument("units_per_channel must be positive");
    if (config.n_channels > std::numeric_limits<std::size_t>::max() / config.units_per_channel)
        throw std::invalid_argument("hidden unit count overflows size_t");
    require_positive(config.fs, "fs");
    require_nonnegative(config.baseline_rate_hz, "baseline_rate_hz");
    require_nonnegative(config.intent_rate_gain_hz, "intent_rate_gain_hz");
    require_positive(config.max_rate_hz, "max_rate_hz");
    if (config.baseline_rate_hz > config.max_rate_hz)
        throw std::invalid_argument("baseline_rate_hz must not exceed max_rate_hz");
    require_nonnegative(config.refractory_seconds, "refractory_seconds");
    const auto refractory_samples =
        std::ceil(static_cast<long double>(config.refractory_seconds) * config.fs);
    if (!std::isfinite(refractory_samples) ||
        refractory_samples > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::invalid_argument("refractory duration exceeds the uint64 sample range");
    require_nonnegative(config.preferred_direction_jitter_radians,
                        "preferred_direction_jitter_radians");
    require_nonnegative(config.spike_amplitude_volts, "spike_amplitude_volts");
    require_positive(config.spike_duration_seconds, "spike_duration_seconds");
    require_positive(config.spike_spatial_decay_channels, "spike_spatial_decay_channels");
    require_positive(config.lfp_frequency_hz, "lfp_frequency_hz");
    if (config.lfp_frequency_hz >= config.fs * 0.5)
        throw std::invalid_argument("lfp_frequency_hz must be below Nyquist");
    require_positive(config.lfp_damping_time_seconds, "lfp_damping_time_seconds");
    require_nonnegative(config.lfp_std_volts, "lfp_std_volts");
    require_correlation(config.lfp_spatial_correlation, "lfp_spatial_correlation");
    require_nonnegative(config.background_noise_std_volts, "background_noise_std_volts");
    require_positive(config.background_time_constant_seconds, "background_time_constant_seconds");
    require_correlation(config.background_spatial_correlation, "background_spatial_correlation");
    require_nonnegative(config.nonstationarity_std_fraction, "nonstationarity_std_fraction");
    require_positive(config.nonstationarity_time_constant_seconds,
                     "nonstationarity_time_constant_seconds");
    require_correlation(config.rate_correlation, "rate_correlation");
    require_finite(config.drift_rotation_degrees, "drift_rotation_degrees");
    require_nonnegative(config.drift_per_unit_rotation_std_degrees,
                        "drift_per_unit_rotation_std_degrees");
    if (!(std::isinf(config.drift_per_unit_rotation_limit_degrees) &&
          config.drift_per_unit_rotation_limit_degrees > 0.0))
    {
        require_positive(config.drift_per_unit_rotation_limit_degrees,
                         "drift_per_unit_rotation_limit_degrees");
        if (config.drift_per_unit_rotation_std_degrees > 0.0)
        {
            const auto acceptance =
                std::erf(config.drift_per_unit_rotation_limit_degrees /
                         (config.drift_per_unit_rotation_std_degrees * std::sqrt(2.0)));
            const auto failure =
                std::pow(1.0 - acceptance, static_cast<double>(kMaxTruncatedGaussianDraws));
            const auto unit_count =
                static_cast<double>(config.n_channels * config.units_per_channel);
            if (failure * unit_count > kMaximumTruncatedGaussianFailureProbability)
                throw std::invalid_argument(
                    "per-unit drift limit is too narrow for reliable truncated-normal sampling");
        }
    }
    require_nonnegative(config.drift_tuning_gain_scale, "drift_tuning_gain_scale");
    require_finite(config.drift_baseline_rate_shift_hz, "drift_baseline_rate_shift_hz");
    const auto waveform_samples = std::ceil(config.spike_duration_seconds * config.fs);
    if (!std::isfinite(waveform_samples) || waveform_samples < 3.0 ||
        waveform_samples > 1'000'000.0)
        throw std::invalid_argument("spike duration must span between 3 and 1000000 samples");
    if (waveform_samples >
        static_cast<double>(std::numeric_limits<std::size_t>::max() / config.n_channels))
        throw std::invalid_argument("spike tail size overflows size_t");
    return config;
}

} // namespace

NeuralSignalGenerator::NeuralSignalGenerator(NeuralSignalConfig config)
    : config_(validate(std::move(config)))
{
    prepare();
}

std::size_t NeuralSignalGenerator::channel_count() const noexcept
{
    return config_.n_channels;
}

std::size_t NeuralSignalGenerator::unit_count() const noexcept
{
    return n_units_;
}

double NeuralSignalGenerator::sample_rate() const noexcept
{
    return config_.fs;
}

std::uint64_t NeuralSignalGenerator::sample_index() const noexcept
{
    return sample_index_;
}

std::uint64_t NeuralSignalGenerator::drift_fingerprint() const noexcept
{
    std::uint64_t value = 1469598103934665603ULL;
    for (const auto angle : drift_rotation_radians_)
    {
        const auto bits = std::bit_cast<std::uint64_t>(angle);
        for (std::uint32_t shift = 0; shift < 64; shift += 8)
        {
            value ^= (bits >> shift) & 0xffU;
            value *= 1099511628211ULL;
        }
    }
    return value;
}

void NeuralSignalGenerator::prepare()
{
    n_units_ = config_.n_channels * config_.units_per_channel;
    waveform_samples_ =
        static_cast<std::size_t>(std::ceil(config_.spike_duration_seconds * config_.fs));
    refractory_samples_ = static_cast<std::uint64_t>(
        std::ceil(static_cast<long double>(config_.refractory_seconds) * config_.fs));

    rate_alpha_ = std::exp(-1.0 / (config_.fs * config_.nonstationarity_time_constant_seconds));
    rate_innovation_ = std::sqrt(1.0 - rate_alpha_ * rate_alpha_);
    background_alpha_ = std::exp(-1.0 / (config_.fs * config_.background_time_constant_seconds));
    background_innovation_ = std::sqrt(1.0 - background_alpha_ * background_alpha_);
    lfp_decay_ = std::exp(-1.0 / (config_.fs * config_.lfp_damping_time_seconds));
    const auto lfp_angle = kTwoPi * config_.lfp_frequency_hz / config_.fs;
    lfp_cos_ = std::cos(lfp_angle);
    lfp_sin_ = std::sin(lfp_angle);
    lfp_innovation_ = std::sqrt(1.0 - lfp_decay_ * lfp_decay_);

    preferred_angles_.resize(n_units_);
    drift_rotation_radians_.resize(n_units_);
    directional_drive_.resize(n_units_);
    unit_channel_offsets_.resize(n_units_ + 1U);
    rate_states_.resize(n_units_);
    next_allowed_sample_.resize(n_units_);
    background_states_.resize(config_.n_channels);
    lfp_x_.resize(config_.n_channels);
    lfp_y_.resize(config_.n_channels);
    spike_waveform_.resize(waveform_samples_);
    spike_tail_.resize(waveform_samples_ * config_.n_channels);

    const auto maximum_distance = static_cast<double>(config_.n_channels - 1U);
    const auto radius_value =
        std::ceil(-config_.spike_spatial_decay_channels * std::log(kMinimumSpatialWeight));
    const auto spatial_radius = !std::isfinite(radius_value) || radius_value >= maximum_distance
                                    ? config_.n_channels - 1U
                                    : static_cast<std::size_t>(radius_value);
    for (std::size_t unit = 0; unit < n_units_; ++unit)
    {
        const auto primary_channel = unit / config_.units_per_channel;
        const auto unit_on_channel = unit % config_.units_per_channel;
        const auto direction_index = unit_on_channel * config_.n_channels + primary_channel;
        const auto base_angle =
            kTwoPi * static_cast<double>(direction_index) / static_cast<double>(n_units_);
        preferred_angles_[unit] =
            base_angle + config_.preferred_direction_jitter_radians *
                             gaussian(config_.seed ^ kPopulationSeed, 0, unit);
        auto heterogeneous_rotation = 0.0;
        if (config_.drift_per_unit_rotation_std_degrees > 0.0)
        {
            const auto limit = config_.drift_per_unit_rotation_limit_degrees;
            auto sampled = false;
            for (std::uint64_t draw = 0; draw < kMaxTruncatedGaussianDraws; ++draw)
            {
                heterogeneous_rotation = config_.drift_per_unit_rotation_std_degrees *
                                         gaussian(config_.seed ^ kDriftSeed, draw, unit);
                if (std::abs(heterogeneous_rotation) <= limit)
                {
                    sampled = true;
                    break;
                }
            }
            if (!sampled)
                throw std::runtime_error("truncated-normal per-unit drift sampling failed");
        }
        drift_rotation_radians_[unit] =
            (config_.drift_rotation_degrees + heterogeneous_rotation) * kTwoPi / 360.0;
        const auto unit_amplitude =
            config_.spike_amplitude_volts *
            (0.8 + 0.4 * uniform_open(config_.seed ^ kPopulationSeed, 1, unit));
        unit_channel_offsets_[unit] = unit_channels_.size();
        const auto first_channel =
            primary_channel > spatial_radius ? primary_channel - spatial_radius : 0U;
        const auto last_channel =
            primary_channel + std::min(spatial_radius, config_.n_channels - 1U - primary_channel);
        for (auto channel = first_channel;; ++channel)
        {
            const auto distance =
                std::abs(static_cast<double>(channel) - static_cast<double>(primary_channel));
            unit_channels_.push_back(channel);
            unit_channel_weights_.push_back(
                unit_amplitude * std::exp(-distance / config_.spike_spatial_decay_channels));
            if (channel == last_channel)
                break;
        }
    }
    unit_channel_offsets_[n_units_] = unit_channels_.size();

    double peak = 0.0;
    for (std::size_t sample = 0; sample < waveform_samples_; ++sample)
    {
        const auto position =
            static_cast<double>(sample) / static_cast<double>(waveform_samples_ - 1U);
        const auto negative = -std::exp(-0.5 * std::pow((position - 0.30) / 0.07, 2.0));
        const auto positive = 0.35 * std::exp(-0.5 * std::pow((position - 0.52) / 0.12, 2.0));
        spike_waveform_[sample] = negative + positive;
        peak = std::max(peak, std::abs(spike_waveform_[sample]));
    }
    for (auto& value : spike_waveform_)
        value /= peak;
    reset();
}

NeuralGenerationStatus
NeuralSignalGenerator::generate(std::size_t count, Intent2D intent, double drift_progress,
                                std::span<double> output,
                                std::span<std::uint8_t> spike_output) noexcept
{
    if (count > std::numeric_limits<std::size_t>::max() / config_.n_channels ||
        output.size() != count * config_.n_channels)
        return NeuralGenerationStatus::invalid_output;
    if (!spike_output.empty() && (count > std::numeric_limits<std::size_t>::max() / n_units_ ||
                                  spike_output.size() != count * n_units_))
        return NeuralGenerationStatus::invalid_spike_output;
    if (!std::isfinite(intent.x) || !std::isfinite(intent.y) || !std::isfinite(drift_progress) ||
        drift_progress < 0.0 || drift_progress > 1.0)
        return NeuralGenerationStatus::invalid_control;
    const auto start = sample_index_;
    if (count > std::numeric_limits<std::uint64_t>::max() - start)
        return NeuralGenerationStatus::range_overflow;

    std::fill(spike_output.begin(), spike_output.end(), std::uint8_t{});
    if (count == 0)
        return NeuralGenerationStatus::ok;

    for (std::size_t unit = 0; unit < n_units_; ++unit)
    {
        const auto angle = preferred_angles_[unit] + drift_progress * drift_rotation_radians_[unit];
        directional_drive_[unit] = std::cos(angle) * intent.x + std::sin(angle) * intent.y;
    }
    const auto tuning_scale = 1.0 + drift_progress * (config_.drift_tuning_gain_scale - 1.0);
    const auto baseline_shift = drift_progress * config_.drift_baseline_rate_shift_hz;
    const auto rate_nonstationarity_enabled = config_.nonstationarity_std_fraction > 0.0;
    const auto background_enabled = config_.background_noise_std_volts > 0.0;
    const auto lfp_enabled = config_.lfp_std_volts > 0.0;
    const auto rate_common_weight =
        rate_nonstationarity_enabled ? std::sqrt(config_.rate_correlation) : 0.0;
    const auto rate_independent_weight =
        rate_nonstationarity_enabled ? std::sqrt(1.0 - config_.rate_correlation) : 0.0;
    const auto background_common_weight =
        background_enabled ? std::sqrt(config_.background_spatial_correlation) : 0.0;
    const auto background_independent_weight =
        background_enabled ? std::sqrt(1.0 - config_.background_spatial_correlation) : 0.0;
    const auto lfp_common_weight = lfp_enabled ? std::sqrt(config_.lfp_spatial_correlation) : 0.0;
    const auto lfp_independent_weight =
        lfp_enabled ? std::sqrt(1.0 - config_.lfp_spatial_correlation) : 0.0;

    for (std::size_t row = 0; row < count; ++row)
    {
        const auto absolute_sample = start + static_cast<std::uint64_t>(row);
        const auto output_offset = row * config_.n_channels;
        const auto tail_offset = spike_tail_cursor_ * config_.n_channels;

        if (rate_nonstationarity_enabled)
            common_rate_state_ =
                rate_alpha_ * common_rate_state_ +
                rate_innovation_ * gaussian(config_.seed ^ kRateSeed, absolute_sample, n_units_);
        if (background_enabled)
            common_background_state_ =
                background_alpha_ * common_background_state_ +
                background_innovation_ *
                    gaussian(config_.seed ^ kNoiseSeed, absolute_sample, config_.n_channels);
        if (lfp_enabled)
        {
            const auto common_lfp_x =
                lfp_decay_ * (lfp_cos_ * common_lfp_x_ - lfp_sin_ * common_lfp_y_) +
                lfp_innovation_ *
                    gaussian(config_.seed ^ kLfpSeed, absolute_sample, config_.n_channels * 2U);
            const auto common_lfp_y =
                lfp_decay_ * (lfp_sin_ * common_lfp_x_ + lfp_cos_ * common_lfp_y_) +
                lfp_innovation_ * gaussian(config_.seed ^ kLfpSeed, absolute_sample,
                                           config_.n_channels * 2U + 1U);
            common_lfp_x_ = common_lfp_x;
            common_lfp_y_ = common_lfp_y;
        }

        for (std::size_t channel = 0; channel < config_.n_channels; ++channel)
        {
            auto background = 0.0;
            if (background_enabled)
            {
                background_states_[channel] =
                    background_alpha_ * background_states_[channel] +
                    background_innovation_ *
                        gaussian(config_.seed ^ kNoiseSeed, absolute_sample, channel);
                background = config_.background_noise_std_volts *
                             (background_common_weight * common_background_state_ +
                              background_independent_weight * background_states_[channel]);
            }
            auto lfp = 0.0;
            if (lfp_enabled)
            {
                const auto channel_lfp_x =
                    lfp_decay_ * (lfp_cos_ * lfp_x_[channel] - lfp_sin_ * lfp_y_[channel]) +
                    lfp_innovation_ *
                        gaussian(config_.seed ^ kLfpSeed, absolute_sample, channel * 2U);
                const auto channel_lfp_y =
                    lfp_decay_ * (lfp_sin_ * lfp_x_[channel] + lfp_cos_ * lfp_y_[channel]) +
                    lfp_innovation_ *
                        gaussian(config_.seed ^ kLfpSeed, absolute_sample, channel * 2U + 1U);
                lfp_x_[channel] = channel_lfp_x;
                lfp_y_[channel] = channel_lfp_y;
                lfp = config_.lfp_std_volts * (lfp_common_weight * common_lfp_x_ +
                                               lfp_independent_weight * lfp_x_[channel]);
            }
            output[output_offset + channel] = spike_tail_[tail_offset + channel] + background + lfp;
            spike_tail_[tail_offset + channel] = 0.0;
        }

        for (std::size_t unit = 0; unit < n_units_; ++unit)
        {
            auto log_gain = 0.0;
            if (rate_nonstationarity_enabled)
            {
                rate_states_[unit] =
                    rate_alpha_ * rate_states_[unit] +
                    rate_innovation_ * gaussian(config_.seed ^ kRateSeed, absolute_sample, unit);
                const auto latent = rate_common_weight * common_rate_state_ +
                                    rate_independent_weight * rate_states_[unit];
                log_gain = std::clamp(config_.nonstationarity_std_fraction * latent -
                                          0.5 * config_.nonstationarity_std_fraction *
                                              config_.nonstationarity_std_fraction,
                                      -4.0, 4.0);
            }
            const auto rate = std::clamp(
                (config_.baseline_rate_hz + baseline_shift) * std::exp(log_gain) +
                    config_.intent_rate_gain_hz * tuning_scale * directional_drive_[unit],
                0.0, config_.max_rate_hz);
            if (absolute_sample < next_allowed_sample_[unit])
                continue;
            const auto probability = -std::expm1(-rate / config_.fs);
            if (uniform_open(config_.seed ^ kSpikeSeed, absolute_sample, unit) >= probability)
                continue;
            next_allowed_sample_[unit] =
                refractory_samples_ > std::numeric_limits<std::uint64_t>::max() - absolute_sample
                    ? std::numeric_limits<std::uint64_t>::max()
                    : absolute_sample + refractory_samples_;
            if (!spike_output.empty())
                spike_output[row * n_units_ + unit] = 1;
            for (std::size_t lag = 0; lag < waveform_samples_; ++lag)
            {
                const auto contribution_offset =
                    ((spike_tail_cursor_ + lag) % waveform_samples_) * config_.n_channels;
                for (auto connection = unit_channel_offsets_[unit];
                     connection < unit_channel_offsets_[unit + 1U]; ++connection)
                {
                    const auto channel = unit_channels_[connection];
                    const auto contribution =
                        unit_channel_weights_[connection] * spike_waveform_[lag];
                    if (lag == 0)
                        output[output_offset + channel] += contribution;
                    else
                        spike_tail_[contribution_offset + channel] += contribution;
                }
            }
        }
        spike_tail_cursor_ = (spike_tail_cursor_ + 1U) % waveform_samples_;
    }

    sample_index_ = start + static_cast<std::uint64_t>(count);
    return NeuralGenerationStatus::ok;
}

void NeuralSignalGenerator::reset() noexcept
{
    sample_index_ = 0;
    spike_tail_cursor_ = 0;
    const auto rate_enabled = config_.nonstationarity_std_fraction > 0.0;
    const auto background_enabled = config_.background_noise_std_volts > 0.0;
    const auto lfp_enabled = config_.lfp_std_volts > 0.0;
    const auto rate_seed = config_.seed ^ kRateSeed ^ kInitialStateSeed;
    const auto background_seed = config_.seed ^ kNoiseSeed ^ kInitialStateSeed;
    const auto lfp_seed = config_.seed ^ kLfpSeed ^ kInitialStateSeed;
    common_rate_state_ = rate_enabled ? gaussian(rate_seed, 0, n_units_) : 0.0;
    common_background_state_ =
        background_enabled ? gaussian(background_seed, 0, config_.n_channels) : 0.0;
    common_lfp_x_ = lfp_enabled ? gaussian(lfp_seed, 0, config_.n_channels * 2U) : 0.0;
    common_lfp_y_ = lfp_enabled ? gaussian(lfp_seed, 0, config_.n_channels * 2U + 1U) : 0.0;
    std::fill(spike_tail_.begin(), spike_tail_.end(), 0.0);
    std::fill(next_allowed_sample_.begin(), next_allowed_sample_.end(), 0);
    for (std::size_t unit = 0; unit < n_units_; ++unit)
        rate_states_[unit] = rate_enabled ? gaussian(rate_seed, 0, unit) : 0.0;
    for (std::size_t channel = 0; channel < config_.n_channels; ++channel)
    {
        background_states_[channel] =
            background_enabled ? gaussian(background_seed, 0, channel) : 0.0;
        lfp_x_[channel] = lfp_enabled ? gaussian(lfp_seed, 0, channel * 2U) : 0.0;
        lfp_y_[channel] = lfp_enabled ? gaussian(lfp_seed, 0, channel * 2U + 1U) : 0.0;
    }
}

} // namespace neurale::signal::simulation
