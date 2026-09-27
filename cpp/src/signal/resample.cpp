/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/resample.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neurale::signal
{
namespace
{

std::size_t checked_add(std::size_t left, std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::overflow_error("resampling size calculation overflowed");
    }
    return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error("resampling size calculation overflowed");
    }
    return left * right;
}

std::size_t ceil_div(std::size_t numerator, std::size_t denominator)
{
    return numerator / denominator + static_cast<std::size_t>(numerator % denominator != 0);
}

} // namespace

std::size_t resampled_length(std::size_t input_length, std::size_t up, std::size_t down)
{
    if (up == 0 || down == 0)
    {
        throw std::invalid_argument("up and down must be positive");
    }
    if (input_length == 0)
    {
        return 0;
    }
    const auto common = std::gcd(up, down);
    up /= common;
    down /= common;

    const auto quotient = input_length / down;
    const auto remainder = input_length % down;
    return checked_add(checked_multiply(quotient, up),
                       ceil_div(checked_multiply(remainder, up), down));
}

Resampler::Resampler(std::size_t up, std::size_t down, std::span<const double> filter)
    : up_(up), down_(down)
{
    if (up_ == 0 || down_ == 0)
    {
        throw std::invalid_argument("up and down must be positive");
    }
    const auto common = std::gcd(up_, down_);
    up_ /= common;
    down_ /= common;
    if (up_ != 1 || down_ != 1)
    {
        build_phases(filter);
    }
    reset();
}

std::size_t Resampler::output_length(std::size_t input_length) const
{
    return resampled_length(input_length, up_, down_);
}

std::size_t Resampler::max_output_length(std::size_t input_length) const
{
    if (up_ == 1 && down_ == 1)
    {
        return input_length;
    }
    return checked_add(resampled_length(checked_add(input_length, 1), up_, down_), 1);
}

std::size_t Resampler::max_flush_length() const
{
    if (up_ == 1 && down_ == 1)
    {
        return 0;
    }
    return max_output_length(coefs_per_phase_ - 1);
}

void Resampler::prepare(std::size_t n_channels, std::size_t max_input_samples)
{
    if (n_channels == 0)
    {
        throw std::invalid_argument("n_channels must be positive");
    }
    if (up_ == 1 && down_ == 1)
    {
        return;
    }
    prepare_channels(n_channels);
    prepare_workspace(max_input_samples);
}

std::size_t Resampler::process(std::span<const double> x, std::size_t n_samples,
                               std::size_t n_channels, std::span<double> output)
{
    if (n_channels == 0)
    {
        throw std::invalid_argument("n_channels must be positive");
    }
    if (x.size() != checked_multiply(n_samples, n_channels))
    {
        throw std::invalid_argument("input shape does not match its buffer");
    }
    if (n_samples == 0)
    {
        if (n_channels_ != 0 && n_channels != n_channels_)
        {
            throw std::invalid_argument("n_channels must stay constant until reset");
        }
        return 0;
    }
    if (up_ == 1 && down_ == 1)
    {
        if (output.size() < x.size())
        {
            throw std::invalid_argument("output buffer is too small");
        }
        const auto new_total_input = checked_add(total_input_, n_samples);
        const auto new_emitted_output = checked_add(emitted_output_, n_samples);
        std::copy(x.begin(), x.end(), output.begin());
        total_input_ = new_total_input;
        emitted_output_ = new_emitted_output;
        return n_samples;
    }
    const auto required = checked_multiply(max_output_length(n_samples), n_channels);
    if (output.size() < required)
    {
        throw std::invalid_argument("output buffer is too small");
    }
    prepare_channels(n_channels);
    prepare_workspace(n_samples);

    const auto new_total_input = checked_add(total_input_, n_samples);
    total_input_ = new_total_input;
    return process_samples(x, n_samples, output, false, std::numeric_limits<std::size_t>::max());
}

std::size_t Resampler::flush(std::span<double> output)
{
    if (n_channels_ == 0 || (up_ == 1 && down_ == 1))
    {
        reset();
        return 0;
    }
    const auto target = output_length(total_input_);
    const auto remaining = emitted_output_ >= target ? 0 : target - emitted_output_;
    if (remaining == 0)
    {
        reset();
        return 0;
    }
    if (output.size() < checked_multiply(remaining, n_channels_))
    {
        throw std::invalid_argument("output buffer is too small");
    }
    const auto flush_input_samples = checked_add(coefs_per_phase_ - 1, down_);
    prepare_workspace(flush_input_samples);
    const auto written = process_samples({}, flush_input_samples, output, true, remaining);
    if (written != remaining)
    {
        reset();
        throw std::runtime_error("resampler flush produced fewer samples than expected");
    }
    reset();
    return written;
}

void Resampler::reset()
{
    phase_ = 0;
    offset_ = coefs_per_phase_ == 0 ? 0 : coefs_per_phase_ - 1;
    pending_trim_ = delay_;
    total_input_ = 0;
    emitted_output_ = 0;
    input_overshoot_ = 0;
    std::fill(history_.begin(), history_.end(), 0.0);
}

void Resampler::build_phases(std::span<const double> filter)
{
    if (filter.empty())
    {
        throw std::invalid_argument("resampling filter must have positive length");
    }
    auto sum = 0.0;
    for (const auto coef : filter)
    {
        if (!std::isfinite(coef))
        {
            throw std::invalid_argument("resampling filter must contain finite values");
        }
        sum += coef;
    }
    const auto tol = std::numeric_limits<double>::epsilon() * static_cast<double>(filter.size());
    if (!std::isfinite(sum) || std::abs(sum) <= tol)
    {
        throw std::invalid_argument("resampling filter sum must be nonzero");
    }

    const auto half_length = (filter.size() - 1) / 2;
    const auto pre_padding = down_ - half_length % down_;
    delay_ = checked_add(half_length, pre_padding) / down_;

    std::vector<double> padded(checked_add(pre_padding, filter.size()), 0.0);
    const auto scale = static_cast<double>(up_) / sum;
    for (std::size_t i = 0; i < filter.size(); ++i)
    {
        padded[pre_padding + i] = filter[i] * scale;
    }

    coefs_per_phase_ = ceil_div(padded.size(), up_);
    coefs_.assign(checked_multiply(coefs_per_phase_, up_), 0.0);
    for (std::size_t phase = 0; phase < up_; ++phase)
    {
        for (std::size_t j = 0; j < coefs_per_phase_; ++j)
        {
            const auto source = j * up_ + phase;
            if (source < padded.size())
            {
                coefs_[phase * coefs_per_phase_ + coefs_per_phase_ - 1 - j] = padded[source];
            }
        }
    }
}

void Resampler::prepare_channels(std::size_t n_channels)
{
    if (n_channels == n_channels_)
    {
        return;
    }
    if (n_channels_ == 0 || total_input_ == 0)
    {
        std::vector<double> new_history(checked_multiply(coefs_per_phase_ - 1, n_channels), 0.0);
        history_ = std::move(new_history);
        n_channels_ = n_channels;
        return;
    }
    throw std::invalid_argument("n_channels must stay constant until reset");
}

void Resampler::prepare_workspace(std::size_t n_samples)
{
    const auto history_samples = coefs_per_phase_ - 1;
    const auto total_samples = checked_add(history_samples, n_samples);
    const auto required = checked_multiply(total_samples, n_channels_);
    if (workspace_.size() < required)
    {
        workspace_.resize(required);
    }
}

std::size_t Resampler::process_samples(std::span<const double> x, std::size_t n_samples,
                                       std::span<double> output, bool flush,
                                       std::size_t output_limit)
{
    const auto history_samples = coefs_per_phase_ - 1;
    const auto total_samples = checked_add(history_samples, n_samples);
    const auto workspace_size = checked_multiply(total_samples, n_channels_);
    if (workspace_.size() < workspace_size)
    {
        workspace_.resize(workspace_size);
    }
    std::copy(history_.begin(), history_.end(), workspace_.begin());
    auto tail = workspace_.begin() + static_cast<std::ptrdiff_t>(history_.size());
    const auto tail_size = checked_multiply(n_samples, n_channels_);
    if (flush)
    {
        std::fill_n(tail, tail_size, 0.0);
    }
    else
    {
        std::copy(x.begin(), x.end(), tail);
    }

    auto input_sample = checked_add(history_samples, input_overshoot_);
    auto written = std::size_t{0};

    while (input_sample < total_samples)
    {
        if (pending_trim_ > 0)
        {
            --pending_trim_;
        }
        else
        {
            if (written >= output_limit)
            {
                break;
            }
            const auto* phase_coefs = coefs_.data() + phase_ * coefs_per_phase_ + offset_;
            const auto terms = coefs_per_phase_ - offset_;
            const auto first_input = input_sample + offset_ + 1 - coefs_per_phase_;
            auto* destination = output.data() + written * n_channels_;

            if (n_channels_ == 1)
            {
                double value = 0.0;
                const auto* source = workspace_.data() + first_input;
                for (std::size_t term = 0; term < terms; ++term)
                {
                    value += source[term] * phase_coefs[term];
                }
                destination[0] = value;
            }
            else
            {
                std::fill_n(destination, n_channels_, 0.0);
                for (std::size_t term = 0; term < terms; ++term)
                {
                    const auto coef = phase_coefs[term];
                    const auto* source = workspace_.data() + (first_input + term) * n_channels_;
                    for (std::size_t channel = 0; channel < n_channels_; ++channel)
                    {
                        destination[channel] += source[channel] * coef;
                    }
                }
            }
            ++written;
            ++emitted_output_;
        }

        phase_ += down_;
        const auto advance = phase_ / up_;
        input_sample += advance;
        phase_ %= up_;
        offset_ = offset_ > advance ? offset_ - advance : 0;
    }
    input_overshoot_ = input_sample - total_samples;

    const auto history_size = checked_multiply(history_samples, n_channels_);
    if (history_size > 0)
    {
        std::copy(workspace_.begin() + static_cast<std::ptrdiff_t>(workspace_size - history_size),
                  workspace_.begin() + static_cast<std::ptrdiff_t>(workspace_size),
                  history_.begin());
    }
    return written;
}

} // namespace neurale::signal
