/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// A bounded, non-durable `SpoolFile` that lives entirely in memory.
///
/// This exists because of a gap the contract registers rather than hides. A
/// critical recorder's readiness gate requires a backend whose in-flight
/// `append`/`sync` can be ended within a bound (contract section 4.7), and
/// `PlatformSpoolFile` cannot promise that: POSIX blocking file operations are
/// not interruptible, and Windows `CancelIoEx` is only a cancellation request,
/// not a bound on completion. The readiness gate therefore refuses it.
///
/// This class closes that hole for a **provisional** surface, and it closes
/// it honestly rather than by claiming a capability it does not have:
/// an in-memory store has no blocking operation to interrupt, so its shutdown
/// really is bounded, with no cancellation primitive and no thread left behind.
/// That is the same property the recorder-core test doubles already rely on.
///
/// What it does **not** provide is durability, and the difference matters:
///
/// | | `PlatformSpoolFile` | `MemorySpoolFile` |
/// | --- | --- | --- |
/// | bounded cancel | no -- gate refuses it | yes |
/// | survives a crash | yes, per policy | **no** |
/// | committed transactions | yes | yes, until the process ends |
///
/// So this is not the production store contract section 2 promises the native
/// path ("spool with committed transactions and a bounded recovery point"). It
/// has the committed transactions and none of the recovery point. A caller
/// therefore MUST NOT run it under a durability policy that would make the
/// writer claim a durable extent it cannot honour: `DurabilityPolicy::buffered`
/// is the only policy consistent with a store that survives nothing, and the
/// Python facade enforces exactly that.

#include "spool_file.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

namespace neurale::recording
{

/// An append-only spool held in one fixed allocation.
///
/// The capacity is reserved once at construction and never grows. That is not
/// a convenience: a store that reallocated while the recorder's worker appended
/// into it would move the very bytes a concurrent `read_at` is reading, and a
/// store that grew without bound would convert "the machine cannot keep up"
/// into "the machine ran out of memory", which is the worse failure and the
/// later one. Appending past the reserved capacity reports `out_of_space`,
/// exactly as a full disk does, which is what makes the recorder's writer-failure
/// path reachable here at all.
class MemorySpoolFile final : public SpoolFile
{
  public:
    /// Reserve *capacity_bytes* up front. Throws `std::bad_alloc` if the
    /// allocation fails, which is the one place this class is allowed to throw:
    /// it happens before any recorder holds it, on the caller's thread, and
    /// never on the worker or the critical callback.
    explicit MemorySpoolFile(std::size_t capacity_bytes);
    ~MemorySpoolFile() override = default;

    SpoolIoResult append(std::span<const std::byte> data) noexcept override;
    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override;
    /// Succeeds without doing anything, because there is nothing to flush and
    /// nothing that would survive if there were. Under `buffered` -- the only
    /// policy this store may be used with -- the writer syncs the superblock
    /// region and nothing else, so no per-transaction durability is ever
    /// claimed on the strength of this returning `ok`.
    SpoolIoResult sync() noexcept override;
    SpoolIoResult truncate(std::uint64_t bytes) noexcept override;
    [[nodiscard]] std::uint64_t size() const noexcept override;

    /// True, and true honestly: no operation on this store can block, so a
    /// shutdown reaches its worker without needing to interrupt anything. This
    /// is the bounded *cancel* contract section 4.7 asks for -- the operation
    /// ends, the thread exits, the handle is released -- not the bounded *wait*
    /// that a blocking syscall moved onto another thread would give.
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return true;
    }

    /// Bytes reserved at construction. `append` refuses past this point.
    [[nodiscard]] std::size_t capacity_bytes() const noexcept
    {
        return capacity_;
    }

    /// A copy of everything written so far. For diagnosis and tests; the
    /// recorder never calls it.
    [[nodiscard]] std::vector<std::byte> snapshot() const;

  private:
    /// Guards `data_` against the worker appending while another thread reads
    /// back or truncates. Every holder does a bounded memcpy, so the wait is
    /// bounded too -- and the critical callback never touches this class.
    mutable std::mutex mutex_{};
    std::vector<std::byte> data_{};
    std::size_t capacity_{};
};

} // namespace neurale::recording
