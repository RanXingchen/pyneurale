/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace neurale::signal::detail
{

/**
 * @brief Return the smallest power of two greater than or equal to a value.
 *
 * Zero and one both map to one.
 *
 * @param value Non-negative input value.
 * @return Smallest representable power of two not less than @p value.
 * @throws std::overflow_error If the result is not representable by
 *         ``std::size_t``.
 */
inline std::size_t next_power_of_two(std::size_t value)
{
    std::size_t result = 1;
    while (result < value)
    {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
        {
            throw std::overflow_error("next power of two is not representable");
        }
        result <<= 1;
    }
    return result;
}

} // namespace neurale::signal::detail
