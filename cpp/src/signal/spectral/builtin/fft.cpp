/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fft.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace neurale::signal::detail
{
namespace
{

bool is_power_of_two(std::size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t convolution_length(std::size_t input_length)
{
    std::size_t result = 1;
    const auto required = 2 * input_length - 1;
    while (result < required)
    {
        result <<= 1;
    }
    return result;
}

} // namespace

void fft_builtin_inplace(std::span<std::complex<double>> values, bool inverse)
{
    const auto length = values.size();
    if (length <= 1)
    {
        return;
    }
    if (inverse)
    {
        for (auto& value : values)
        {
            value = std::conj(value);
        }
        fft_builtin_inplace(values, false);
        const auto scale = 1.0 / static_cast<double>(length);
        for (auto& value : values)
        {
            value = std::conj(value) * scale;
        }
        return;
    }

    if (is_power_of_two(length))
    {
        for (std::size_t i = 1, reversed = 0; i < length; ++i)
        {
            std::size_t bit = length >> 1;
            for (; reversed & bit; bit >>= 1)
            {
                reversed ^= bit;
            }
            reversed ^= bit;
            if (i < reversed)
            {
                std::swap(values[i], values[reversed]);
            }
        }

        for (std::size_t block = 2; block <= length; block <<= 1)
        {
            const auto root =
                std::polar(1.0, -2.0 * std::numbers::pi_v<double> / static_cast<double>(block));
            for (std::size_t start = 0; start < length; start += block)
            {
                std::complex<double> factor{1.0, 0.0};
                const auto half = block / 2;
                for (std::size_t offset = 0; offset < half; ++offset)
                {
                    const auto even = values[start + offset];
                    const auto odd = values[start + offset + half] * factor;
                    values[start + offset] = even + odd;
                    values[start + offset + half] = even - odd;
                    factor *= root;
                }
            }
        }
        return;
    }

    const auto fft_length = convolution_length(length);
    std::vector<std::complex<double>> left(fft_length, 0.0);
    std::vector<std::complex<double>> right(fft_length, 0.0);
    for (std::size_t i = 0; i < length; ++i)
    {
        const auto square = static_cast<double>(i) * static_cast<double>(i);
        const auto forward_chirp =
            std::polar(1.0, -std::numbers::pi_v<double> * square / static_cast<double>(length));
        const auto convolution_chirp = std::conj(forward_chirp);
        left[i] = values[i] * forward_chirp;
        right[i] = convolution_chirp;
        if (i != 0)
        {
            right[fft_length - i] = convolution_chirp;
        }
    }
    fft_builtin_inplace(left, false);
    fft_builtin_inplace(right, false);
    for (std::size_t i = 0; i < fft_length; ++i)
    {
        left[i] *= right[i];
    }
    fft_builtin_inplace(left, true);
    for (std::size_t i = 0; i < length; ++i)
    {
        const auto square = static_cast<double>(i) * static_cast<double>(i);
        values[i] = left[i] * std::polar(1.0, -std::numbers::pi_v<double> * square /
                                                  static_cast<double>(length));
    }
}

} // namespace neurale::signal::detail
