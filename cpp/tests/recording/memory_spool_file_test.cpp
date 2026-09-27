/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// \file
/// The bounded, non-durable in-memory spool store.
///
/// It exists because the readiness gate refuses every backend that cannot end
/// an in-flight operation within a bound, and no production backend can promise
/// that yet (see `memory_spool_file.h`). Two of its properties are load-bearing
/// and neither is exercised by the Python surface that drives it end to end:
/// it never grows past the capacity reserved at construction, and running out
/// reports the same shape a full disk does rather than throwing or truncating
/// silently.

#include "check_returns.h"
#include "memory_spool_file.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace neurale::recording;

namespace
{

std::vector<std::byte> pattern(std::size_t count, std::uint8_t seed)
{
    std::vector<std::byte> bytes(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        bytes[i] = static_cast<std::byte>((i * 7 + seed) & 0xFF);
    }
    return bytes;
}

int test_append_reads_back_at_written_offset()
{
    MemorySpoolFile file{1024};
    CHECK(file.size() == 0);
    CHECK(file.capacity_bytes() == 1024);

    const auto first = pattern(16, 1);
    const auto second = pattern(32, 2);
    CHECK(file.append(first).status == SpoolIoStatus::ok);
    CHECK(file.append(second).status == SpoolIoStatus::ok);
    CHECK(file.size() == 48);

    std::array<std::byte, 32> out{};
    const auto read = file.read_at(16, out);
    CHECK(read.status == SpoolIoStatus::ok);
    CHECK(read.transferred == 32);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        CHECK(out[i] == second[i]);
    }
    return 0;
}

int test_out_of_room_reports_progress_and_reason()
{
    // The same shape a full disk reports: a partial transfer plus the reason.
    // A store that threw here, or silently kept the whole thing, would hide the
    // one writer-failure path the recorder must be able to reach.
    MemorySpoolFile file{40};
    CHECK(file.append(pattern(32, 3)).status == SpoolIoStatus::ok);

    const auto overflow = file.append(pattern(16, 4));
    CHECK(overflow.status == SpoolIoStatus::out_of_space);
    CHECK(overflow.transferred == 8);
    CHECK(file.size() == 40);

    // And it stays full: a second attempt takes nothing at all rather than
    // growing the store behind the bound it was given.
    const auto again = file.append(pattern(8, 5));
    CHECK(again.status == SpoolIoStatus::out_of_space);
    CHECK(again.transferred == 0);
    CHECK(file.size() == 40);
    CHECK(file.capacity_bytes() == 40);
    return 0;
}

int test_read_past_end_is_incomplete_not_error()
{
    MemorySpoolFile file{64};
    CHECK(file.append(pattern(8, 6)).status == SpoolIoStatus::ok);

    std::array<std::byte, 16> out{};
    const auto partial = file.read_at(4, out);
    CHECK(partial.status == SpoolIoStatus::incomplete);
    CHECK(partial.transferred == 4);

    const auto beyond = file.read_at(64, out);
    CHECK(beyond.status == SpoolIoStatus::incomplete);
    CHECK(beyond.transferred == 0);
    return 0;
}

int test_truncate_shortens_and_never_extends()
{
    MemorySpoolFile file{64};
    CHECK(file.append(pattern(32, 7)).status == SpoolIoStatus::ok);

    CHECK(file.truncate(8).status == SpoolIoStatus::ok);
    CHECK(file.size() == 8);
    // Truncating past the end is a no-op, not a way to grow the file: this is
    // the operation that discards a spool holding no committed record, and it
    // must never be able to add bytes.
    CHECK(file.truncate(4096).status == SpoolIoStatus::ok);
    CHECK(file.size() == 8);
    return 0;
}

int test_store_reports_honourable_bounded_cancel()
{
    // True because nothing here can block, which is the whole reason this class
    // exists: the readiness gate needs a store whose in-flight operation ends,
    // whose thread exits, and whose handle is released within a bound, and an
    // operation that cannot block satisfies all three trivially.
    MemorySpoolFile file{16};
    CHECK(file.supports_bounded_cancel());
    file.request_cancel();
    // Cancelling changes nothing about a store with nothing in flight, and the
    // file stays usable afterwards.
    CHECK(file.append(pattern(4, 8)).status == SpoolIoStatus::ok);
    CHECK(file.size() == 4);
    return 0;
}

int test_sync_succeeds_without_claiming_survival()
{
    // `sync()` returning ok is not a durability claim -- there is nothing to
    // flush. What keeps that from becoming a lie is the policy the store is
    // driven under, which the Python facade pins to `buffered`.
    MemorySpoolFile file{16};
    CHECK(file.append(pattern(4, 9)).status == SpoolIoStatus::ok);
    CHECK(file.sync().status == SpoolIoStatus::ok);
    CHECK(file.size() == 4);
    return 0;
}

int test_snapshot_is_copy_not_view()
{
    MemorySpoolFile file{32};
    CHECK(file.append(pattern(8, 10)).status == SpoolIoStatus::ok);
    const auto snapshot = file.snapshot();
    CHECK(snapshot.size() == 8);

    CHECK(file.append(pattern(8, 11)).status == SpoolIoStatus::ok);
    CHECK(snapshot.size() == 8);
    CHECK(file.size() == 16);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_append_reads_back_at_written_offset,
             test_out_of_room_reports_progress_and_reason,
             test_read_past_end_is_incomplete_not_error,
             test_truncate_shortens_and_never_extends,
             test_store_reports_honourable_bounded_cancel,
             test_sync_succeeds_without_claiming_survival,
             test_snapshot_is_copy_not_view,
         })
    {
        if (const auto line = test(); line != 0)
            return line;
    }
    return 0;
}
