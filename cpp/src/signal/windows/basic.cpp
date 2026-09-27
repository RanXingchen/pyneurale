/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/windows.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>

namespace neurale::signal
{
namespace
{

constexpr std::array<double, 2> kHann{0.5, -0.5};
constexpr std::array<double, 2> kHamming{0.54, -0.46};
constexpr std::array<double, 3> kBlackman{0.42, -0.5, 0.08};
constexpr std::array<double, 5> kFlatTop{
    0.21557895, -0.41663158, 0.277263158, -0.083578947, 0.006947368,
};

double log_modified_bessel_i0(double value)
{
    if (value == 0.0)
    {
        return 0.0;
    }
    if (value < 500.0)
    {
        return std::log(std::cyl_bessel_i(0.0, value));
    }
    const double inverse = 1.0 / value;
    const double series = 1.0 + inverse / 8.0 + 9.0 * inverse * inverse / 128.0 +
                          225.0 * inverse * inverse * inverse / 3072.0;
    return value - 0.5 * std::log(2.0 * std::numbers::pi_v<double> * value) + std::log(series);
}

void cosine_sum_window(std::span<const double> coefs, std::span<double> output, bool symmetric)
{
    if (output.empty())
    {
        return;
    }
    if (output.size() == 1)
    {
        output.front() = 1.0;
        return;
    }

    const auto den = static_cast<double>(symmetric ? output.size() - 1 : output.size());
    for (std::size_t sample = 0; sample < output.size(); ++sample)
    {
        const double phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(sample) / den;
        double value = 0.0;
        for (std::size_t harmonic = 0; harmonic < coefs.size(); ++harmonic)
        {
            value += coefs[harmonic] * std::cos(static_cast<double>(harmonic) * phase);
        }
        output[sample] = value;
    }
}

} // namespace

void cosine_window(std::string_view kind, std::span<double> output, bool symmetric)
{
    if (kind == "hann")
    {
        cosine_sum_window(kHann, output, symmetric);
        return;
    }
    if (kind == "hamming")
    {
        cosine_sum_window(kHamming, output, symmetric);
        return;
    }
    if (kind == "blackman")
    {
        cosine_sum_window(kBlackman, output, symmetric);
        return;
    }
    if (kind == "flattop")
    {
        cosine_sum_window(kFlatTop, output, symmetric);
        return;
    }
    throw std::invalid_argument("unknown generalized cosine window kind");
}

void kaiser_window(std::span<double> output, double beta, bool symmetric)
{
    if (!std::isfinite(beta))
    {
        throw std::invalid_argument("beta must be finite");
    }
    if (output.empty())
    {
        return;
    }
    if (output.size() == 1)
    {
        output.front() = 1.0;
        return;
    }

    const double shape = std::abs(beta);
    const double den = log_modified_bessel_i0(shape);
    const double sample_den = static_cast<double>(symmetric ? output.size() - 1 : output.size());
    for (std::size_t sample = 0; sample < output.size(); ++sample)
    {
        const double pos = 2.0 * static_cast<double>(sample) / sample_den - 1.0;
        const double radicand = std::max(0.0, 1.0 - pos * pos);
        output[sample] = std::exp(log_modified_bessel_i0(shape * std::sqrt(radicand)) - den);
    }
}

} // namespace neurale::signal
