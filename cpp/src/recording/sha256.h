/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// SHA-256, used for exactly one thing in this component: checking that the
/// plan document stored in a spool superblock is the document whose SHA-256 is
/// the recording plan's fingerprint (specification section 2.1).
///
/// It is not a tamper check and MUST NOT be described as one. The container
/// never parses the plan document; it compares a digest computed elsewhere.
/// The incremental interface exists so a reader can stream a plan document of
/// any length through a bounded buffer.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace neurale::recording
{

class Sha256
{
  public:
    Sha256() noexcept = default;

    void update(std::span<const std::byte> data) noexcept;

    /// Finish the digest. The object must not be updated afterwards.
    [[nodiscard]] std::array<std::uint8_t, 32> finish() noexcept;

  private:
    void compress(const std::byte* block) noexcept;

    std::array<std::uint32_t, 8> state_{0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
                                        0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};
    std::array<std::byte, 64> buffer_{};
    std::size_t buffered_{};
    std::uint64_t total_bytes_{};
};

/// One-shot convenience wrapper.
[[nodiscard]] std::array<std::uint8_t, 32> sha256(std::span<const std::byte> data) noexcept;

} // namespace neurale::recording
