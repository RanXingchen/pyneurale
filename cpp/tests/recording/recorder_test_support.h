/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Fixtures shared by the native recorder-core tests.
///
/// Everything here is deterministic on purpose. The recorder reads no clock on
/// its critical path and takes one for the worker, so a test that injects a
/// counting clock gets a spool whose bytes depend on nothing but its inputs --
/// which is what makes "the same session twice" a comparison rather than a
/// coincidence.

#include "recorder.h"
#include "recording_plan.h"
#include "sha256.h"
#include "spool_file.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/frame.h>

namespace neurale::recording::test
{

/// A clock that advances by a fixed step per read. Injected everywhere, so no
/// test depends on wall time and none of them can flake on a slow machine.
class CountingUnixClock final : public UnixClock
{
  public:
    explicit CountingUnixClock(std::uint64_t start = 1767225600000000000ULL,
                               std::uint64_t step = 1000) noexcept
        : next_(start), step_(step)
    {
    }

    [[nodiscard]] std::uint64_t unix_nanos() noexcept override
    {
        const auto value = next_.fetch_add(step_, std::memory_order_relaxed);
        return value;
    }

  private:
    std::atomic<std::uint64_t> next_;
    std::uint64_t step_;
};

/// An in-memory spool file with the same injection knobs the writer fault-injection tests use,
/// plus a mutex: unlike the writer tests, here a worker thread appends while
/// the test thread reads back what it wrote.
class RecorderMemorySpoolFile final : public SpoolFile
{
  public:
    std::size_t capacity_bytes{std::numeric_limits<std::size_t>::max()};
    /// 1-based call number to fail on; 0 never fails.
    std::atomic<std::size_t> fail_append_call{0};
    SpoolIoStatus append_fault{SpoolIoStatus::io_error};
    std::atomic<std::size_t> fail_sync_call{0};
    SpoolIoStatus sync_fault{SpoolIoStatus::io_error};

    SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        const auto call = ++append_calls_;
        const auto fail_at = fail_append_call.load(std::memory_order_relaxed);
        if (fail_at != 0 && call == fail_at)
        {
            return SpoolIoResult{.status = append_fault, .transferred = 0, .platform_error = 0};
        }
        auto take = bytes.size();
        bool full = false;
        if (data_.size() + take > capacity_bytes)
        {
            take = capacity_bytes > data_.size() ? capacity_bytes - data_.size() : 0;
            full = true;
        }
        data_.insert(data_.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(take));
        if (full)
        {
            return SpoolIoResult{
                .status = SpoolIoStatus::out_of_space, .transferred = take, .platform_error = 0};
        }
        return SpoolIoResult{.status = SpoolIoStatus::ok, .transferred = take, .platform_error = 0};
    }

    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (offset >= data_.size())
        {
            return SpoolIoResult{
                .status = SpoolIoStatus::incomplete, .transferred = 0, .platform_error = 0};
        }
        const auto take =
            std::min<std::size_t>(out.size(), data_.size() - static_cast<std::size_t>(offset));
        std::copy_n(data_.begin() + static_cast<std::ptrdiff_t>(offset), take, out.begin());
        return SpoolIoResult{.status =
                                 take == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete,
                             .transferred = take,
                             .platform_error = 0};
    }

    SpoolIoResult sync() noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        const auto call = ++sync_calls_;
        const auto fail_at = fail_sync_call.load(std::memory_order_relaxed);
        if (fail_at != 0 && call == fail_at)
        {
            return SpoolIoResult{.status = sync_fault, .transferred = 0, .platform_error = 0};
        }
        synced_bytes_ = data_.size();
        return SpoolIoResult{};
    }

    SpoolIoResult truncate(std::uint64_t bytes) noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (bytes < data_.size())
        {
            data_.resize(static_cast<std::size_t>(bytes));
        }
        return SpoolIoResult{};
    }

    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        return data_.size();
    }

    [[nodiscard]] std::vector<std::byte> snapshot() const
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        return data_;
    }

    [[nodiscard]] std::size_t append_calls() const
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        return append_calls_;
    }

    [[nodiscard]] std::uint64_t synced_bytes() const
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        return synced_bytes_;
    }

    /// Preallocate the backing store. The allocation gate measures the whole
    /// steady-state path, worker included, so a test double that grew a vector
    /// on every append would be measuring its own fixture.
    void reserve_bytes(std::size_t bytes)
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        data_.reserve(bytes);
    }

    // An in-memory file never blocks in append/sync, so its shutdown is bounded
    // without any cancellation primitive. Every recorder test that passes the
    // readiness gate uses this double, so the gate's cancellable-backend
    // requirement is satisfied.
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return true;
    }

  private:
    mutable std::mutex mutex_{};
    std::vector<std::byte> data_{};
    std::size_t append_calls_{};
    std::size_t sync_calls_{};
    std::uint64_t synced_bytes_{};
};

// --- a plan, and storage for the ranges it borrows --------------------------

inline constexpr std::uint64_t kTestNativeSessionId = 0x51ABULL;
inline constexpr std::uint32_t kTestSchemaId = 11;
inline constexpr std::uint32_t kSignalA = 1;
inline constexpr std::uint32_t kSignalB = 2;
/// Present in frames, absent from the plan: a partial-coverage session records
/// a subset, and the recorder must skip rather than fault on it.
inline constexpr std::uint32_t kUnplannedSignal = 7;

/// Owns everything a `NativeRecordingPlan` points at, so the plan can be
/// handed to `prepare()` without any borrowed range outliving its storage.
struct PlanFixture
{
    std::string session_id{"recorder-core-test-session"};
    std::string plan_document{R"({"coverage":"full","extension_version":1,"native_schema_id":11,)"
                              R"("planned_signal_ids":[1,2],"recorded_signal_ids":[1,2]})"};
    std::vector<std::byte> document_bytes{};
    std::vector<PlannedSignalRecording> signals{
        PlannedSignalRecording{
            .signal_id = kSignalA, .max_block_bytes = 512, .max_block_samples = 64},
        PlannedSignalRecording{
            .signal_id = kSignalB, .max_block_bytes = 512, .max_block_samples = 64},
    };

    std::size_t frame_queue_capacity{16};
    std::size_t control_queue_capacity{8};
    std::uint64_t checkpoint_interval{0};
    std::uint64_t drain_timeout_nanos{5000000000ULL};
    DurabilityPolicy policy{DurabilityPolicy::checkpoint_sync};

    [[nodiscard]] NativeRecordingPlan plan()
    {
        document_bytes.clear();
        for (const auto character : plan_document)
        {
            document_bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
        }

        NativeRecordingPlan out{};
        out.session_id = session_id;
        out.session_uuid = std::array<std::uint8_t, kSessionUuidBytes>{
            0x52, 0x45, 0x43, 0x4F, 0x52, 0x44, 0x45, 0x52,
            0x43, 0x4F, 0x52, 0x45, 0x30, 0x30, 0x30, 0x31};
        out.created_unix_nanos = 1767225600000000000ULL;
        out.plan_document = document_bytes;
        out.plan_fingerprint = sha256(document_bytes);
        out.native_session_id = kTestNativeSessionId;
        out.native_schema_id = kTestSchemaId;
        out.recorded_signals = signals;
        out.frame_queue_capacity = frame_queue_capacity;
        out.control_queue_capacity = control_queue_capacity;
        out.max_blocks_per_frame = 4;
        out.max_frame_payload_bytes = 1024;
        out.max_signal_gaps_per_discontinuity = 4;
        out.max_control_payload_bytes = 128;
        out.max_records_per_transaction = 64;
        out.max_transaction_bytes = 64 * 1024;
        out.checkpoint_interval_transactions = checkpoint_interval;
        out.worker_idle_poll_nanos = 50000;
        out.drain_timeout_nanos = drain_timeout_nanos;
        out.durability_policy = policy;
        return out;
    }
};

/// One frame and the storage its views point at, kept alive by the caller.
struct FrameFixture
{
    streaming::FrameHeader header{};
    std::vector<streaming::SignalBlockHeader> blocks{};
    std::vector<std::byte> payload{};

    [[nodiscard]] streaming::FrameView view() const noexcept
    {
        return streaming::FrameView{
            .header = header, .buffer = {}, .blocks = blocks, .payload = payload};
    }
};

/// Build a frame carrying *signal_ids*, each with *samples_per_block* bytes of
/// a recognisable pattern, laid out back to back in the payload.
[[nodiscard]] inline FrameFixture make_frame(std::uint64_t sequence,
                                             std::span<const std::uint32_t> signal_ids,
                                             std::size_t bytes_per_block = 16)
{
    FrameFixture fixture{};
    fixture.header.session_id = kTestNativeSessionId;
    fixture.header.sequence = sequence;
    fixture.header.host_received_ns = 900000 + sequence;
    fixture.header.source_tick = 1000 + sequence;
    fixture.header.valid_until_ns = 0;
    fixture.header.schema_id = kTestSchemaId;
    fixture.header.source_clock_domain = 3;
    fixture.header.signal_block_count = static_cast<std::uint32_t>(signal_ids.size());
    fixture.header.flags = streaming::FrameFlags::source_tick;

    fixture.payload.resize(signal_ids.size() * bytes_per_block);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < signal_ids.size(); ++i)
    {
        streaming::SignalBlockHeader block{};
        block.signal_id = signal_ids[i];
        block.sample_idx_start = sequence * 8;
        block.last_sample_idx = sequence * 8 + 7;
        block.device_tick_start = 5000 + sequence;
        block.observation_time_start_ns = 700000 + sequence;
        block.payload_offset = offset;
        block.payload_byte_count = bytes_per_block;
        block.n_samples = static_cast<std::uint32_t>(bytes_per_block / 2);
        block.clock_sync.device_tick_reference = 11;
        block.clock_sync.host_time_reference_ns = 22;
        block.clock_sync.device_tick_rate =
            streaming::RationalRate{.numerator = 30000, .denominator = 1};
        block.clock_sync.uncertainty_ns = 33;
        block.clock_sync.clock_domain = 3;
        block.clock_sync.generation = 1;
        block.clock_sync.flags = streaming::ClockSyncFlags::synchronized;
        fixture.blocks.push_back(block);

        for (std::size_t byte = 0; byte < bytes_per_block; ++byte)
        {
            fixture.payload[offset + byte] =
                static_cast<std::byte>((sequence * 31 + i * 7 + byte) & 0xFF);
        }
        offset += bytes_per_block;
    }
    return fixture;
}

/// One discontinuity and the gap storage its span points at.
struct DiscontinuityFixture
{
    std::vector<streaming::SignalGap> gaps{};
    streaming::Discontinuity discontinuity{};

    [[nodiscard]] const streaming::Discontinuity& view() noexcept
    {
        discontinuity.signal_gaps = gaps;
        return discontinuity;
    }
};

[[nodiscard]] inline DiscontinuityFixture make_discontinuity(std::uint64_t previous_sequence,
                                                             std::uint64_t actual_sequence,
                                                             std::size_t n_gaps = 2)
{
    DiscontinuityFixture fixture{};
    fixture.discontinuity.session_id = kTestNativeSessionId;
    fixture.discontinuity.previous_frame_sequence = previous_sequence;
    fixture.discontinuity.actual_frame_sequence = actual_sequence;
    fixture.discontinuity.reason = streaming::GapReason::frame_sequence_gap;
    for (std::size_t i = 0; i < n_gaps; ++i)
    {
        streaming::SignalGap gap{};
        gap.signal_id = i == 0 ? kSignalA : kSignalB;
        gap.expected_sample_idx = 100 + i;
        gap.actual_sample_idx = 200 + i;
        gap.missing_samples = 100;
        gap.expected_device_tick = 300 + i;
        gap.actual_device_tick = 400 + i;
        gap.reason = streaming::GapReason::sample_gap;
        gap.flags = streaming::SignalGapFlags::missing_samples_known;
        fixture.gaps.push_back(gap);
    }
    return fixture;
}

} // namespace neurale::recording::test
