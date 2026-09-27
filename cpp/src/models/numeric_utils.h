/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace neurale::models::detail
{

struct Magnitude
{
    double mantissa{};
    int exponent{};
};

inline Magnitude finite_magnitude(double value)
{
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    return Magnitude{.mantissa = mantissa, .exponent = exponent};
}

inline Magnitude add_magnitudes(double left, double right)
{
    if (left == 0.0)
    {
        return finite_magnitude(right);
    }
    if (right == 0.0)
    {
        return finite_magnitude(left);
    }

    int left_exponent = 0;
    int right_exponent = 0;
    const double left_mantissa = std::frexp(left, &left_exponent);
    const double right_mantissa = std::frexp(right, &right_exponent);
    const int exponent = std::max(left_exponent, right_exponent);
    double mantissa = std::ldexp(left_mantissa, left_exponent - exponent) +
                      std::ldexp(right_mantissa, right_exponent - exponent);
    int normalized_exponent = exponent;
    if (mantissa >= 1.0)
    {
        mantissa *= 0.5;
        ++normalized_exponent;
    }
    return Magnitude{
        .mantissa = mantissa,
        .exponent = normalized_exponent,
    };
}

inline Magnitude absolute_difference(double left, double right)
{
    if (left == right)
    {
        return Magnitude{};
    }

    const double left_abs = std::abs(left);
    const double right_abs = std::abs(right);
    if (std::signbit(left) != std::signbit(right))
    {
        return add_magnitudes(left_abs, right_abs);
    }
    return finite_magnitude(std::abs(left_abs - right_abs));
}

inline bool less_magnitude(const Magnitude& left, const Magnitude& right) noexcept
{
    if (left.exponent != right.exponent)
    {
        return left.exponent < right.exponent;
    }
    return left.mantissa < right.mantissa;
}

inline double magnitude_ratio(const Magnitude& numerator, const Magnitude& denominator)
{
    return std::ldexp(numerator.mantissa / denominator.mantissa,
                      numerator.exponent - denominator.exponent);
}

inline double scaled_euclidean_distance(const double* left, const double* right, std::size_t size)
{
    Magnitude scale{};
    double sum_squares = 1.0;

    for (std::size_t i = 0; i < size; ++i)
    {
        const Magnitude delta = absolute_difference(left[i], right[i]);
        if (delta.mantissa == 0.0)
        {
            continue;
        }
        if (scale.mantissa == 0.0 || less_magnitude(scale, delta))
        {
            const double ratio = scale.mantissa == 0.0 ? 0.0 : magnitude_ratio(scale, delta);
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = delta;
        }
        else
        {
            const double ratio = magnitude_ratio(delta, scale);
            sum_squares += ratio * ratio;
        }
    }

    if (scale.mantissa == 0.0)
    {
        return 0.0;
    }
    return std::scalbn(scale.mantissa * std::sqrt(sum_squares), scale.exponent);
}

} // namespace neurale::models::detail
