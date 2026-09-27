/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/simulation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "counter_random.h"

namespace neurale::signal::simulation
{
namespace
{

using detail::kTwoPi;

[[nodiscard]] bool valid_shape(std::size_t rows, std::size_t columns, std::size_t n_values) noexcept
{
    return rows != 0 && columns != 0 && rows <= std::numeric_limits<std::size_t>::max() / columns &&
           rows * columns == n_values;
}

void validate_common(std::size_t n_channels, double fs)
{
    if (n_channels == 0)
    {
        throw std::invalid_argument("n_channels must be positive");
    }
    if (!std::isfinite(fs) || fs <= 0.0)
    {
        throw std::invalid_argument("fs must be finite and positive");
    }
}

void validate_finite(std::span<const double> values, const char* name)
{
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
}

} // namespace

SignalGenerator::SignalGenerator(Kind kind, std::size_t n_channels, double fs)
    : kind_(kind), n_channels_(n_channels), fs_(fs)
{
    validate_common(n_channels, fs);
}

SignalGenerator SignalGenerator::zeros(std::size_t n_channels, double fs)
{
    return SignalGenerator(Kind::zeros, n_channels, fs);
}

SignalGenerator SignalGenerator::constant(std::size_t n_channels, double fs,
                                          std::span<const double> values)
{
    if (values.size() != n_channels)
    {
        throw std::invalid_argument("constant values must match n_channels");
    }
    validate_finite(values, "constant values");
    SignalGenerator generator(Kind::constant, n_channels, fs);
    generator.values_.assign(values.begin(), values.end());
    return generator;
}

SignalGenerator SignalGenerator::tones(std::size_t n_channels, double fs, std::size_t n_tones,
                                       std::span<const double> freqs, std::span<const double> amps,
                                       std::span<const double> phases)
{
    SignalGenerator generator(Kind::tones, n_channels, fs);
    if (!valid_shape(n_tones, n_channels, freqs.size()) || amps.size() != freqs.size() ||
        phases.size() != freqs.size())
    {
        throw std::invalid_argument("tone arrays must have shape (tone, channel)");
    }
    validate_finite(freqs, "frequencies");
    validate_finite(amps, "amps");
    validate_finite(phases, "phases");
    if (std::any_of(freqs.begin(), freqs.end(),
                    [fs](double freq) { return freq < 0.0 || freq > fs / 2.0; }))
    {
        throw std::invalid_argument("frequencies must be in [0, fs / 2]");
    }

    generator.n_tones = n_tones;
    generator.values_.reserve(freqs.size());
    for (const auto freq : freqs)
    {
        generator.values_.push_back((freq / fs) * kTwoPi);
    }
    generator.amps_.assign(amps.begin(), amps.end());
    generator.phases_.assign(phases.begin(), phases.end());
    return generator;
}

SignalGenerator SignalGenerator::noise(std::size_t n_channels, double fs, std::uint64_t seed,
                                       double low, double high)
{
    SignalGenerator generator(Kind::noise, n_channels, fs);
    if (!std::isfinite(low) || !std::isfinite(high) || !(low < high) || !std::isfinite(high - low))
    {
        throw std::invalid_argument("noise bounds and their span must be finite with low < high");
    }
    generator.seed_ = seed;
    generator.noise_low_ = low;
    generator.noise_scale_ = high - low;
    return generator;
}

SignalGenerator SignalGenerator::samples(std::size_t n_channels, double fs, std::size_t n_samples,
                                         std::span<const double> x, bool repeat)
{
    SignalGenerator generator(Kind::samples, n_channels, fs);
    if (!valid_shape(n_samples, n_channels, x.size()))
    {
        throw std::invalid_argument("samples must have shape (sample, channel)");
    }
    validate_finite(x, "samples");
    generator.n_samples = n_samples;
    generator.repeat_ = repeat;
    generator.values_.assign(x.begin(), x.end());
    return generator;
}

std::optional<std::uint64_t> SignalGenerator::finite_sample_count() const noexcept
{
    return kind_ == Kind::samples && !repeat_ ? std::optional<std::uint64_t>{n_samples}
                                              : std::nullopt;
}

GenerationStatus SignalGenerator::generate(std::uint64_t start, std::size_t count,
                                           std::span<double> output) const noexcept
{
    if (count > std::numeric_limits<std::size_t>::max() / n_channels_ ||
        output.size() != count * n_channels_)
    {
        return GenerationStatus::invalid_output;
    }
    if (count != 0 &&
        static_cast<std::uint64_t>(count - 1) > std::numeric_limits<std::uint64_t>::max() - start)
    {
        return GenerationStatus::range_overflow;
    }
    if (kind_ == Kind::samples && !repeat_ &&
        (start > n_samples || count > n_samples - static_cast<std::size_t>(start)))
    {
        return GenerationStatus::source_exhausted;
    }

    for (std::size_t row = 0; row < count; ++row)
    {
        const auto absolute_sample = start + static_cast<std::uint64_t>(row);
        const auto output_offset = row * n_channels_;
        for (std::size_t channel = 0; channel < n_channels_; ++channel)
        {
            double value = 0.0;
            switch (kind_)
            {
            case Kind::zeros:
                break;
            case Kind::constant:
                value = values_[channel];
                break;
            case Kind::tones:
                for (std::size_t tone = 0; tone < n_tones; ++tone)
                {
                    const auto idx = tone * n_channels_ + channel;
                    value +=
                        amps_[idx] * std::sin(phases_[idx] +
                                              values_[idx] * static_cast<double>(absolute_sample));
                }
                break;
            case Kind::noise:
                value =
                    noise_low_ + noise_scale_ * detail::uniform01(seed_, absolute_sample, channel);
                break;
            case Kind::samples:
            {
                const auto source_sample =
                    repeat_ ? static_cast<std::size_t>(absolute_sample % n_samples)
                            : static_cast<std::size_t>(absolute_sample);
                value = values_[source_sample * n_channels_ + channel];
                break;
            }
            }
            output[output_offset + channel] = value;
        }
    }
    return GenerationStatus::ok;
}

} // namespace neurale::signal::simulation
