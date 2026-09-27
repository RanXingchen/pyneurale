/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "crc32c.h"

#include <array>

namespace neurale::recording
{
namespace
{

constexpr std::uint32_t kReflectedPolynomial = 0x82F63B78U;

[[nodiscard]] constexpr std::array<std::uint32_t, 256> build_table() noexcept
{
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i)
    {
        std::uint32_t value = i;
        for (int bit = 0; bit < 8; ++bit)
        {
            value = (value >> 1) ^ ((value & 1U) != 0U ? kReflectedPolynomial : 0U);
        }
        table[i] = value;
    }
    return table;
}

constexpr std::array<std::uint32_t, 256> kTable = build_table();

} // namespace

std::uint32_t crc32c(std::span<const std::byte> data, std::uint32_t seed) noexcept
{
    std::uint32_t crc = seed ^ 0xFFFFFFFFU;
    for (const auto value : data)
    {
        const auto idx = (crc ^ static_cast<std::uint32_t>(value)) & 0xFFU;
        crc = (crc >> 8) ^ kTable[idx];
    }
    return crc ^ 0xFFFFFFFFU;
}

} // namespace neurale::recording
