/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// CRC-32C (Castagnoli), the only checksum the native spool container defines.
///
/// Reflected polynomial 0x82F63B78, reflected input and output, initial and
/// final value 0xFFFFFFFF. Known answers are in specification section 3, and
/// `recording_spool_checksum_test` checks them before anything checks a spool.
///
/// This is the parameterization the hardware `crc32c` instructions compute;
/// the implementation here is a portable table, which is correct on every
/// platform and fast enough for a store whose unit of work is a transaction.

#include <cstddef>
#include <cstdint>
#include <span>

namespace neurale::recording
{

/// Return the CRC-32C of *data*, continuing from *seed* (the value a previous
/// call returned). Streaming a byte range in chunks yields the same value as a
/// single call over the whole range.
[[nodiscard]] std::uint32_t crc32c(std::span<const std::byte> data,
                                   std::uint32_t seed = 0) noexcept;

} // namespace neurale::recording
