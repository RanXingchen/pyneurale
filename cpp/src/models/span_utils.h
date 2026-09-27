/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace neurale::models::detail
{

inline std::size_t matrix_index(std::size_t row, std::size_t col, std::size_t n_cols) noexcept
{
    return row * n_cols + col;
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* message)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::length_error(message);
    }
    return left * right;
}

inline std::size_t checked_sum(std::size_t left, std::size_t right, const char* message)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::length_error(message);
    }
    return left + right;
}

inline std::size_t saturating_product(std::size_t left, std::size_t right) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        return std::numeric_limits<std::size_t>::max();
    }
    return left * right;
}

template <typename T, typename U> bool spans_overlap(std::span<T> left, std::span<U> right)
{
    const auto left_bytes = std::as_bytes(left);
    const auto right_bytes = std::as_bytes(right);
    if (left_bytes.empty() || right_bytes.empty())
    {
        return false;
    }

    const auto left_begin = reinterpret_cast<std::uintptr_t>(left_bytes.data());
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right_bytes.data());
    const auto left_end = left_begin + static_cast<std::uintptr_t>(left_bytes.size());
    const auto right_end = right_begin + static_cast<std::uintptr_t>(right_bytes.size());
    return left_begin < right_end && right_begin < left_end;
}

} // namespace neurale::models::detail
