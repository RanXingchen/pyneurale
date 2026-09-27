/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Fixed-width little-endian field access for the native spool container.
///
/// Every field of the format is read and written one byte at a time through
/// these helpers. That is deliberate: it makes the encoding independent of the
/// host byte order and of any structure padding a compiler might choose, which
/// is what "ABI-independent binary layout" has to mean in practice.

#include <cstddef>
#include <cstdint>
#include <span>

namespace neurale::recording
{

inline void store_u8(std::span<std::byte> out, std::size_t offset, std::uint8_t value) noexcept
{
    out[offset] = static_cast<std::byte>(value);
}

inline void store_u16le(std::span<std::byte> out, std::size_t offset, std::uint16_t value) noexcept
{
    out[offset] = static_cast<std::byte>(value & 0xFFU);
    out[offset + 1] = static_cast<std::byte>((value >> 8) & 0xFFU);
}

inline void store_u32le(std::span<std::byte> out, std::size_t offset, std::uint32_t value) noexcept
{
    for (std::size_t i = 0; i < 4; ++i)
    {
        out[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFU);
    }
}

inline void store_u64le(std::span<std::byte> out, std::size_t offset, std::uint64_t value) noexcept
{
    for (std::size_t i = 0; i < 8; ++i)
    {
        out[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFU);
    }
}

[[nodiscard]] inline std::uint8_t load_u8(std::span<const std::byte> data,
                                          std::size_t offset) noexcept
{
    return static_cast<std::uint8_t>(data[offset]);
}

[[nodiscard]] inline std::uint16_t load_u16le(std::span<const std::byte> data,
                                              std::size_t offset) noexcept
{
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(data[offset]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[offset + 1]) << 8));
}

[[nodiscard]] inline std::uint32_t load_u32le(std::span<const std::byte> data,
                                              std::size_t offset) noexcept
{
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        value |= static_cast<std::uint32_t>(data[offset + i]) << (8 * i);
    }
    return value;
}

[[nodiscard]] inline std::uint64_t load_u64le(std::span<const std::byte> data,
                                              std::size_t offset) noexcept
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i)
    {
        value |= static_cast<std::uint64_t>(data[offset + i]) << (8 * i);
    }
    return value;
}

} // namespace neurale::recording
