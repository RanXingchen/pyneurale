/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

// Private implementation detail of the Speech stimulus catalog. The catalog
// text contract is canonical UTF-8, and this is the one decoder the contract
// trusts for that check. It is a private header under cpp/src -- not installed,
// not exported, and not part of any public include path -- so the experiment
// value targets and the Python bindings share one source of truth rather than
// drifting into two validators that agree today and diverge tomorrow.
//
// The check is strict: overlong forms, lone surrogates, truncated continuations,
// and lead bytes that start no legal sequence all fail. It runs on the prepare
// path and in the Python constructor, never on a streaming hot path.

namespace neurale::experiments::detail
{

/// Whether @p bytes of @p length form a complete, well-formed UTF-8 sequence.
[[nodiscard]] inline bool is_well_formed_utf8(const std::uint8_t* bytes,
                                              std::size_t length) noexcept
{
    const std::uint8_t* next = bytes;
    std::size_t remaining = length;
    while (remaining > 0)
    {
        const std::uint8_t lead = *next;
        ++next;
        --remaining;

        std::size_t expected = 0;
        std::uint8_t lowest = 0x80;
        std::uint8_t highest = 0xBF;
        if (lead <= 0x7F)
            expected = 0;
        else if (lead >= 0xC2 && lead <= 0xDF)
            expected = 1;
        else if (lead == 0xE0)
        {
            expected = 2;
            lowest = 0xA0;
        }
        else if (lead >= 0xE1 && lead <= 0xEC)
            expected = 2;
        else if (lead == 0xED)
        {
            expected = 2;
            highest = 0x9F;
        }
        else if (lead >= 0xEE && lead <= 0xEF)
            expected = 2;
        else if (lead == 0xF0)
        {
            expected = 3;
            lowest = 0x90;
        }
        else if (lead >= 0xF1 && lead <= 0xF3)
            expected = 3;
        else if (lead == 0xF4)
        {
            expected = 3;
            highest = 0x8F;
        }
        else
            return false; // lone continuation, overlong lead, or out of range

        if (remaining < expected)
            return false; // truncated sequence
        for (std::size_t follow = 0; follow < expected; ++follow)
        {
            const std::uint8_t byte = next[follow];
            // Only the first continuation of a multi-byte sequence is
            // range-checked against the constrained window; the rest use the
            // full continuation window.
            const std::uint8_t low = (follow == 0) ? lowest : 0x80;
            const std::uint8_t high = (follow == 0) ? highest : 0xBF;
            if (byte < low || byte > high)
                return false;
        }
        next += expected;
        remaining -= expected;
    }
    return true;
}

} // namespace neurale::experiments::detail
