/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Scaffolding shared by the windowed-feature adapter tests (LMP, Hilbert
/// envelope, multitaper bandpower). The three adapters converged onto one
/// data plane (``feature_adapter_stream.h``); their tests share the same
/// frame-filling, emission-collection, near-equality, and allocation-counting
/// plumbing, which lived as three verbatim copies. Each adapter test keeps its
/// algorithm-specific pieces (config, reference, the contract bodies) and
/// includes this for the rest.
///
/// The allocation counter comes from ``allocation_counter.h``, which this
/// header includes on the tests' behalf. That header replaces the global
/// allocation operators, so it may appear in only one translation unit per
/// executable -- which each feature-adapter test satisfies by including this
/// one exactly once.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/schema.h>

#include "allocation_counter.h"
#include "check_returns.h"

namespace neurale::pipeline::test
{
using namespace neurale::streaming;

inline constexpr std::size_t channels = 2;
inline constexpr std::size_t max_input_samples = 8;
inline constexpr std::size_t max_frames = 64;
inline constexpr std::size_t max_values = 4096;

[[nodiscard]] inline ProcessorPrepareContext context(const StreamSchema& schema) noexcept
{
    return {
        .input_schema = schema,
        .max_process_outputs = 32,
        .max_flush_outputs = 32,
        .available_frame_pool_leases = 4,
    };
}

[[nodiscard]] inline StreamStatus fill_frame(MutableFrame& frame, const StreamSchema& schema,
                                             std::span<const double> values, std::uint64_t sequence,
                                             SampleIndex sample_start) noexcept
{
    if (values.empty() || values.size() % channels != 0)
    {
        return StreamStatus::invalid_frame;
    }
    const auto n_samples = values.size() / channels;
    const auto rate = schema.signals().front().fs;
    frame.header() = FrameHeader{
        .session_id = 5,
        .sequence = sequence,
        .host_received_ns = 100'000'000 + sample_start,
        .source_tick = 5'000 + sample_start,
        .valid_until_ns = 200'000'000 + sample_start,
        .schema_id = schema.id(),
        .source_clock_domain = 41,
        .flags = FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_start,
        .device_tick_start = 5'000 + sample_start,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = 11,
        .n_samples = static_cast<std::uint32_t>(n_samples),
        .clock_sync =
            ClockSyncSnapshot{
                .device_tick_reference = 5'000,
                .host_time_reference_ns = 10'000'000,
                .device_tick_rate = rate,
                .uncertainty_ns = 3,
                .clock_domain = 41,
                .generation = 2,
                .flags = ClockSyncFlags::synchronized,
            },
    };
    std::memcpy(frame.payload_storage().data(), values.data(), values.size_bytes());
    return frame.set_used_sizes(1, values.size_bytes());
}

[[nodiscard]] inline bool nearly_equal(std::span<const double> left,
                                       std::span<const double> right) noexcept
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const auto scale = std::max({1.0, std::abs(left[i]), std::abs(right[i])});
        if (std::abs(left[i] - right[i]) > 1e-12 * scale)
        {
            return false;
        }
    }
    return true;
}

class CollectingEmitter final : public FrameEmitter
{
  public:
    explicit CollectingEmitter(const StreamSchema& schema)
        : pool_(1, schema.signals().front().max_block_bytes, 1), validator_(schema)
    {
    }

    void clear() noexcept
    {
        frame_count_ = 0;
        value_count_ = 0;
        static_cast<void>(acquired_.reset());
    }

    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        frame = nullptr;
        if (acquired_)
        {
            return StreamStatus::invalid_state;
        }
        const auto status = pool_.try_acquire(acquired_);
        if (status == StreamStatus::ok)
        {
            frame = &acquired_.frame();
        }
        return status;
    }

    StreamStatus publish_acquired_frame() noexcept override
    {
        return record(std::move(acquired_));
    }

    StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

    [[nodiscard]] std::size_t frame_count() const noexcept
    {
        return frame_count_;
    }
    [[nodiscard]] std::span<const double> values() const noexcept
    {
        return {values_.data(), value_count_};
    }
    [[nodiscard]] const SignalBlockHeader& block(std::size_t idx) const noexcept
    {
        return blocks_[idx];
    }

  private:
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return record(std::move(lease));
    }

    StreamStatus record(FrameLease lease) noexcept
    {
        if (!lease || frame_count_ == max_frames ||
            validator_.validate(lease.view()) != FrameValidationError::none)
        {
            return StreamStatus::invalid_frame;
        }
        const auto view = lease.view();
        const auto n_scalars = view.payload.size() / sizeof(double);
        if (n_scalars > max_values - value_count_)
        {
            return StreamStatus::invalid_frame;
        }
        blocks_[frame_count_] = view.blocks.front();
        std::memcpy(values_.data() + value_count_, view.payload.data(), view.payload.size());
        value_count_ += n_scalars;
        ++frame_count_;
        return lease.reset();
    }

    FramePool pool_;
    FrameValidator validator_;
    FrameLease acquired_{};
    std::array<SignalBlockHeader, max_frames> blocks_{};
    std::array<double, max_values> values_{};
    std::size_t frame_count_{};
    std::size_t value_count_{};
};
} // namespace neurale::pipeline::test
