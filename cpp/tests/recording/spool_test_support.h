/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Test doubles and shared fixtures for the native spool tests.
///
/// `MemorySpoolFile` is the reason the writer takes a `SpoolFile` rather than
/// a path: a short write, a full store, a stalled device, a failing sync, a
/// cancelled operation, and a read error are all ordinary return values here,
/// so every failure path is exercised deterministically instead of being
/// arranged on a real filesystem and hoped for.

#include "spool_file.h"
#include "spool_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace neurale::recording::test
{

class MemorySpoolFile final : public SpoolFile
{
  public:
    // --- injection knobs ---------------------------------------------------

    /// Bytes one `append` will accept before reporting a short write.
    std::size_t max_append_bytes{std::numeric_limits<std::size_t>::max()};
    /// Total capacity. Exceeding it writes what fits and reports `out_of_space`.
    std::size_t capacity_bytes{std::numeric_limits<std::size_t>::max()};
    /// 1-based call number to fail on; 0 never fails.
    std::size_t fail_append_call{};
    SpoolIoStatus append_fault{SpoolIoStatus::io_error};
    std::size_t fail_sync_call{};
    SpoolIoStatus sync_fault{SpoolIoStatus::io_error};
    std::size_t fail_truncate_call{};
    SpoolIoStatus truncate_fault{SpoolIoStatus::io_error};
    std::size_t fail_read_call{};
    SpoolIoStatus read_fault{SpoolIoStatus::io_error};

    // --- observations ------------------------------------------------------

    std::vector<std::byte> data{};
    std::size_t append_calls{};
    std::size_t sync_calls{};
    std::size_t truncate_calls{};
    std::size_t read_calls{};
    /// Bytes the last successful sync covered.
    std::uint64_t synced_bytes{};

    SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        ++append_calls;
        if (fail_append_call != 0 && append_calls == fail_append_call)
        {
            return SpoolIoResult{.status = append_fault, .transferred = 0, .platform_error = 0};
        }
        auto take = std::min(bytes.size(), max_append_bytes);
        auto full = false;
        if (data.size() + take > capacity_bytes)
        {
            take = capacity_bytes > data.size() ? capacity_bytes - data.size() : 0;
            full = true;
        }
        data.insert(data.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(take));
        if (full)
        {
            return SpoolIoResult{
                .status = SpoolIoStatus::out_of_space, .transferred = take, .platform_error = 0};
        }
        return SpoolIoResult{.status = take == bytes.size() ? SpoolIoStatus::ok
                                                            : SpoolIoStatus::incomplete,
                             .transferred = take,
                             .platform_error = 0};
    }

    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override
    {
        ++read_calls;
        if (fail_read_call != 0 && read_calls == fail_read_call)
        {
            return SpoolIoResult{.status = read_fault, .transferred = 0, .platform_error = 0};
        }
        if (offset >= data.size())
        {
            return SpoolIoResult{
                .status = SpoolIoStatus::incomplete, .transferred = 0, .platform_error = 0};
        }
        const auto take =
            std::min<std::size_t>(out.size(), data.size() - static_cast<std::size_t>(offset));
        std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(offset), take, out.begin());
        return SpoolIoResult{.status =
                                 take == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete,
                             .transferred = take,
                             .platform_error = 0};
    }

    SpoolIoResult sync() noexcept override
    {
        ++sync_calls;
        if (fail_sync_call != 0 && sync_calls == fail_sync_call)
        {
            return SpoolIoResult{.status = sync_fault, .transferred = 0, .platform_error = 0};
        }
        synced_bytes = data.size();
        return SpoolIoResult{};
    }

    SpoolIoResult truncate(std::uint64_t bytes) noexcept override
    {
        ++truncate_calls;
        if (fail_truncate_call != 0 && truncate_calls == fail_truncate_call)
        {
            return SpoolIoResult{.status = truncate_fault, .transferred = 0, .platform_error = 0};
        }
        if (bytes < data.size())
        {
            data.resize(static_cast<std::size_t>(bytes));
        }
        return SpoolIoResult{};
    }

    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        return data.size();
    }

    // An in-memory file never blocks in append/sync, so a shutdown is bounded
    // without any cancellation primitive. (This double is used by the
    // writer/scanner/repair tests directly; the recorder's prepare-failure test
    // uses it for a spool that never reaches the readiness gate.)
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return true;
    }
};

// --- the fixed inputs the normative vector generator uses -------------------
//
// Copied from `specifications/native-spool/v1/tools/generate_vectors.py`. They
// are constants there for the same reason they are constants here: with no
// clock and no random source in the inputs, a writer either reproduces the
// normative bytes or it does not.

inline constexpr std::string_view kVectorSessionId = "spool-vector-session";
inline constexpr std::uint64_t kVectorCreatedUnixNanos = 1767225600000000000ULL;
inline constexpr std::uint64_t kVectorNanos = 1767225601000000000ULL;

inline constexpr std::array<std::uint8_t, kSessionUuidBytes> kVectorSessionUuid = {
    0x4E, 0x52, 0x4C, 0x53, 0x50, 0x4F, 0x4F, 0x4C, 0x56, 0x45, 0x43, 0x54, 0x30, 0x30, 0x30, 0x31};

inline constexpr std::string_view kVectorPlanDocument =
    R"({"coverage":"full","extension_version":1,"native_schema_id":11,)"
    R"("planned_signal_ids":[1,2],"recorded_signal_ids":[1,2]})";

[[nodiscard]] inline std::vector<std::byte> as_bytes(std::string_view text)
{
    std::vector<std::byte> out;
    out.reserve(text.size());
    for (const auto character : text)
    {
        out.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
    }
    return out;
}

/// Decode an even-length hexadecimal string. Test input only: it assumes what
/// it is given is hexadecimal.
[[nodiscard]] inline std::vector<std::byte> from_hex(std::string_view text)
{
    const auto digit = [](char character) -> int
    {
        if (character >= '0' && character <= '9')
        {
            return character - '0';
        }
        if (character >= 'a' && character <= 'f')
        {
            return character - 'a' + 10;
        }
        return character - 'A' + 10;
    };
    std::vector<std::byte> out;
    out.reserve(text.size() / 2);
    for (std::size_t i = 0; i + 1 < text.size(); i += 2)
    {
        out.push_back(static_cast<std::byte>(digit(text[i]) * 16 + digit(text[i + 1])));
    }
    return out;
}

[[nodiscard]] inline std::vector<std::byte> vector_frame_payload()
{
    return from_hex("01000000000000000200000000000000e803000000000000");
}

[[nodiscard]] inline std::vector<std::byte> vector_block_payload_a()
{
    std::vector<std::byte> out;
    for (int value = 0; value < 24; ++value)
    {
        out.push_back(static_cast<std::byte>(value));
    }
    return out;
}

[[nodiscard]] inline std::vector<std::byte> vector_block_payload_b()
{
    std::vector<std::byte> out;
    for (int value = 24; value < 44; ++value)
    {
        out.push_back(static_cast<std::byte>(value));
    }
    return out;
}

[[nodiscard]] inline std::vector<std::byte> vector_discontinuity_payload()
{
    return from_hex("03000000000000000400000000000000");
}

[[nodiscard]] inline std::vector<std::byte> vector_gap_payload()
{
    return from_hex("0500000000000000");
}

[[nodiscard]] inline std::vector<std::byte> vector_control_payload()
{
    return as_bytes("trial-start");
}

/// Read a whole file, reporting success through *ok*.
[[nodiscard]] inline std::vector<std::byte> read_file(const std::string& path, bool& ok)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        ok = false;
        return {};
    }
    std::vector<std::byte> out;
    char buffer[4096];
    while (stream.read(buffer, sizeof(buffer)) || stream.gcount() > 0)
    {
        const auto count = static_cast<std::size_t>(stream.gcount());
        for (std::size_t i = 0; i < count; ++i)
        {
            out.push_back(static_cast<std::byte>(static_cast<unsigned char>(buffer[i])));
        }
        if (!stream)
        {
            break;
        }
    }
    ok = true;
    return out;
}

/// Read a whole text file into a string.
[[nodiscard]] inline std::string read_text_file(const std::string& path, bool& ok)
{
    const auto bytes = read_file(path, ok);
    std::string out;
    out.reserve(bytes.size());
    for (const auto value : bytes)
    {
        out.push_back(static_cast<char>(value));
    }
    return out;
}

} // namespace neurale::recording::test
