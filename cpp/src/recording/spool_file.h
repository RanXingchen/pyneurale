/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The file interface the spool writer, scanner, and repair operate through.
///
/// Everything that touches bytes goes through `SpoolFile`. That is what makes
/// the failure paths testable: a short write, a full disk, a stalled device, a
/// failing `fsync`, and a read error are ordinary return values of a test
/// double, not conditions a test has to arrange on a real filesystem and hope
/// for. The production implementation is one class in this header
/// (`PlatformSpoolFile`); nothing else in the component knows what a file
/// descriptor is.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace neurale::recording
{

enum class DurabilityPolicy : std::uint8_t;

/// Outcome of one file operation.
enum class SpoolIoStatus : std::uint8_t
{
    /// The whole request completed.
    ok = 0,
    /// The platform transferred fewer bytes than asked and reported no error.
    /// The caller decides whether to retry the remainder; the writer does.
    incomplete = 1,
    /// The store is full. Distinguished from `io_error` because it is the one
    /// write failure an operator can act on directly.
    out_of_space = 2,
    /// The operation did not complete within the bound its caller was willing
    /// to wait. The platform implementation never returns this -- it has no
    /// deadline mechanism and inventing one would put a clock and a wait bound
    /// on the recorder's write path. A test double returns it to prove the
    /// writer surfaces a stalled store instead of blocking on it.
    stalled = 3,
    /// The caller cancelled the operation.
    cancelled = 4,
    /// Any other platform failure.
    io_error = 5,
};

/// What one file operation transferred, and why it stopped.
struct SpoolIoResult
{
    SpoolIoStatus status{SpoolIoStatus::ok};
    /// Bytes actually transferred. Meaningful for `ok` and `incomplete`, and
    /// for a partial transfer that then failed.
    std::size_t transferred{};
    /// The platform error number, when there is one. Carried for reporting
    /// only; no control flow branches on it.
    int platform_error{};

    [[nodiscard]] constexpr bool complete() const noexcept
    {
        return status == SpoolIoStatus::ok;
    }
};

/// An append-only file with positional reads.
///
/// The interface has no seek and no positional write on purpose: a spool is
/// written strictly by appending, and a component that cannot express "write
/// at offset N" cannot accidentally rewrite a committed byte.
class SpoolFile
{
  public:
    SpoolFile() = default;
    virtual ~SpoolFile() = default;

    SpoolFile(const SpoolFile&) = delete;
    SpoolFile& operator=(const SpoolFile&) = delete;
    SpoolFile(SpoolFile&&) = delete;
    SpoolFile& operator=(SpoolFile&&) = delete;

    /// Append *data* to the end of the file. May transfer less than asked.
    virtual SpoolIoResult append(std::span<const std::byte> data) noexcept = 0;

    /// Read into *out* starting at *offset*. May transfer less than asked when
    /// the file ends first, which is reported as `incomplete`, not an error.
    virtual SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept = 0;

    /// Ask the platform to make everything written so far durable.
    virtual SpoolIoResult sync() noexcept = 0;

    /// Shorten the file to *bytes*. Only the explicit repair operation calls
    /// this, and only at a verified committed prefix end.
    virtual SpoolIoResult truncate(std::uint64_t bytes) noexcept = 0;

    /// Current length in bytes.
    [[nodiscard]] virtual std::uint64_t size() const noexcept = 0;

    /// Request cancellation, if supported. A disk backend may leave I/O in
    /// flight; its caller must retain all referenced storage until completion.
    virtual void request_cancel() noexcept {}

    /// Whether cancellation guarantees bounded I/O completion. This capability
    /// is reported in recorder status, not required by readiness. False means
    /// stop/close may time out and retain resources for a later close retry.
    [[nodiscard]] virtual bool supports_bounded_cancel() const noexcept = 0;

    /// Whether this store can honour *policy* without weakening it. Most
    /// stores support every policy and keep the default. A store whose bounded
    /// path deliberately stops at the operating system page cache overrides
    /// this and accepts only `buffered`; the recorder rejects any stronger
    /// request before committing the superblock.
    [[nodiscard]] virtual bool supports_durability_policy(DurabilityPolicy) const noexcept
    {
        return true;
    }
};

/// How an existing file is opened. Diagnosis (contract section 4.5) is
/// read-only -- a read of a spool never writes, never truncates, and never
/// repairs -- so it opens with `read_only` and cannot reach the mutating or
/// durability operations at all. Only an explicit repair opens `read_write`,
/// because only repair is allowed to truncate the tail.
enum class SpoolOpenMode : std::uint8_t
{
    /// `read_at` and `size` are available; `append`, `truncate`, and `sync`
    /// refuse with `io_error`. This is the mode diagnosis uses, and it is what
    /// lets a scanner read a spool on a read-only mount, behind 0444 evidence
    /// permissions, or from a quarantine artifact the caller only holds read.
    read_only = 0,
    /// Every operation is available. This is the mode `create()` implies and
    /// the only mode an explicit repair uses.
    read_write = 1,
};

/// The production `SpoolFile`: an ordinary file, opened once, closed by the
/// destructor. It owns exactly one descriptor for its whole lifetime, which is
/// what makes the leak check in the tests a meaningful one.
class PlatformSpoolFile final : public SpoolFile
{
  public:
    PlatformSpoolFile() noexcept = default;
    ~PlatformSpoolFile() override;

    /// Create a new file at *path*, opened read-write. Fails when it already
    /// exists: a spool is never reopened for appending by this class, because
    /// appending to a file whose tail was never scanned is how a committed
    /// prefix gets orphaned.
    [[nodiscard]] SpoolIoResult
    create(const std::string& path,
           std::uint64_t capacity = std::numeric_limits<std::int64_t>::max()) noexcept;

    /// Open an existing file under *mode*. `read_only` is for diagnosis; it
    /// opens with the platform's read-only access and refuses the mutating and
    /// durability operations. `read_write` is for explicit repair.
    [[nodiscard]] SpoolIoResult open_existing(const std::string& path, SpoolOpenMode mode) noexcept;

    /// Close the descriptor. Idempotent; the destructor calls it.
    SpoolIoResult close() noexcept;

    [[nodiscard]] bool is_open() const noexcept
    {
        return descriptor_ != -1;
    }

    /// Whether the handle permits mutating and durability operations. False
    /// for a `read_only` open; true for `create` and a `read_write` open.
    [[nodiscard]] bool is_writable() const noexcept
    {
        return writable_;
    }

    SpoolIoResult append(std::span<const std::byte> data) noexcept override;
    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override;
    SpoolIoResult sync() noexcept override;
    SpoolIoResult truncate(std::uint64_t bytes) noexcept override;
    [[nodiscard]] std::uint64_t size() const noexcept override;
    /// Ordinary disk I/O cannot guarantee bounded cancellation. Its native
    /// writer stays owned after a timeout; only completed I/O permits cleanup.
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return false;
    }

  private:
    // Keep the platform headers out of this private header: Windows stores the
    // HANDLE bit pattern in a pointer-sized integer, POSIX keeps its native fd.
#ifdef _WIN32
    std::intptr_t descriptor_{-1};
#else
    int descriptor_{-1};
#endif
    std::atomic<std::uint64_t> size_{};
    std::uint64_t capacity_{std::numeric_limits<std::int64_t>::max()};
    bool writable_{};
};

/// A fixed-capacity, page-resident store for critical recording.
///
/// Physical allocation, mapping, writable prefault, and page locking all occur
/// before the recorder is prepared. Once created, `append()` is only a bounded
/// copy into resident pages. Linux requires tmpfs so background filesystem
/// writeback cannot re-enter the append path. Windows preallocates a named local
/// file mapping and locks the whole view with `VirtualLock`, whose contract is
/// that subsequent access incurs no page fault. Both named objects survive
/// process death. There is no power-loss, reboot, or kernel-failure guarantee.
///
/// `sync()` is real and may block, but under the only accepted policy it is
/// called while committing the superblock before readiness, never from the
/// running recorder's transaction path. Checkpoint or transaction sync is
/// rejected rather than silently weakened. The capacity is fixed at creation.
/// On a clean close the backing file is shortened to the logical extent; after
/// abrupt process death its zero-filled reserved tail remains and the ordinary
/// spool scanner/repair path recovers the committed prefix.
class BoundedMappedSpoolFile final : public SpoolFile
{
  public:
    BoundedMappedSpoolFile() noexcept = default;
    ~BoundedMappedSpoolFile() override;

    /// Create a new fixed-capacity mapped object and make every page resident.
    /// Linux accepts only tmpfs. Windows accepts a local preallocated file.
    /// Other platforms, unsupported paths, and failed page locking are refused.
    [[nodiscard]] SpoolIoResult create(const std::string& path,
                                       std::size_t capacity_bytes) noexcept;

    /// Release the mapping and descriptor. Idempotent. A clean close truncates
    /// the reserved file to the logical byte count.
    SpoolIoResult close() noexcept;

    [[nodiscard]] bool is_open() const noexcept
    {
        return mapping_ != nullptr;
    }

    SpoolIoResult append(std::span<const std::byte> data) noexcept override;
    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override;
    SpoolIoResult sync() noexcept override;
    SpoolIoResult truncate(std::uint64_t bytes) noexcept override;
    [[nodiscard]] std::uint64_t size() const noexcept override;
    void request_cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return bounded_backend_ready_;
    }
    [[nodiscard]] bool supports_durability_policy(DurabilityPolicy policy) const noexcept override;

  private:
#if defined(_WIN32)
    std::intptr_t descriptor_{-1};
    std::intptr_t mapping_descriptor_{-1};
#elif defined(__linux__)
    int descriptor_{-1};
#endif
    std::byte* mapping_{};
    std::size_t capacity_{};
    std::atomic<std::uint64_t> size_{};
    std::atomic<bool> cancelled_{};
    bool pages_locked_{};
    bool bounded_backend_ready_{};
};

} // namespace neurale::recording
