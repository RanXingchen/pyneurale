/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "sha256.h"

#include <algorithm>

namespace neurale::recording
{
namespace
{

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U,
    0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU,
    0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU,
    0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU, 0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U,
    0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU,
    0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, 0xA2BFE8A1U, 0xA81A664BU,
    0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U, 0x19A4C116U,
    0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
    0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U,
    0xC67178F2U};

[[nodiscard]] constexpr std::uint32_t rotate_right(std::uint32_t value, int bits) noexcept
{
    return (value >> bits) | (value << (32 - bits));
}

} // namespace

void Sha256::compress(const std::byte* block) noexcept
{
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t i = 0; i < 16; ++i)
    {
        schedule[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                      (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                      (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                      static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i)
    {
        const auto s0 = rotate_right(schedule[i - 15], 7) ^ rotate_right(schedule[i - 15], 18) ^
                        (schedule[i - 15] >> 3);
        const auto s1 = rotate_right(schedule[i - 2], 17) ^ rotate_right(schedule[i - 2], 19) ^
                        (schedule[i - 2] >> 10);
        schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];

    for (std::size_t i = 0; i < 64; ++i)
    {
        const auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const auto choose = (e & f) ^ (~e & g);
        const auto temp1 = h + s1 + choose + kRoundConstants[i] + schedule[i];
        const auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto temp2 = s0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::span<const std::byte> data) noexcept
{
    total_bytes_ += data.size();
    std::size_t cursor = 0;
    if (buffered_ != 0)
    {
        const auto take = std::min(data.size(), buffer_.size() - buffered_);
        std::copy_n(data.begin(), take, buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_));
        buffered_ += take;
        cursor = take;
        if (buffered_ == buffer_.size())
        {
            compress(buffer_.data());
            buffered_ = 0;
        }
    }
    while (data.size() - cursor >= buffer_.size())
    {
        compress(data.data() + cursor);
        cursor += buffer_.size();
    }
    if (cursor < data.size())
    {
        const auto remaining = data.size() - cursor;
        std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(cursor), remaining, buffer_.begin());
        buffered_ = remaining;
    }
}

std::array<std::uint8_t, 32> Sha256::finish() noexcept
{
    const auto bit_length = total_bytes_ * 8;
    std::array<std::byte, 1> one_bit{std::byte{0x80}};
    update(one_bit);
    constexpr std::array<std::byte, 1> zero{std::byte{0}};
    while (buffered_ != 56)
    {
        update(zero);
    }
    std::array<std::byte, 8> length_bytes{};
    for (std::size_t i = 0; i < 8; ++i)
    {
        length_bytes[i] = static_cast<std::byte>((bit_length >> (56 - 8 * i)) & 0xFFU);
    }
    update(length_bytes);

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < 8; ++i)
    {
        digest[i * 4] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xFFU);
        digest[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xFFU);
        digest[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xFFU);
        digest[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xFFU);
    }
    return digest;
}

std::array<std::uint8_t, 32> sha256(std::span<const std::byte> data) noexcept
{
    Sha256 hasher;
    hasher.update(data);
    return hasher.finish();
}

} // namespace neurale::recording
