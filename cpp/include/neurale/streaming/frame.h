/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include <neurale/streaming/clock.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/schema.h>

namespace neurale::streaming
{

/// Generation-checked reference to a preallocated buffer-pool slot.
struct BufferToken
{
    std::uint32_t slot{};
    std::uint64_t generation{};
};

/// Optional fields present in a frame header.
enum class FrameFlags : std::uint32_t
{
    none = 0,
    source_tick = 1U << 0U,
    valid_until = 1U << 1U,
    source_received = 1U << 2U,
};

[[nodiscard]] constexpr FrameFlags operator|(FrameFlags left, FrameFlags right) noexcept
{
    return static_cast<FrameFlags>(static_cast<std::uint32_t>(left) |
                                   static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr bool has_flag(FrameFlags value, FrameFlags flag) noexcept
{
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flag)) != 0U;
}

/// Fixed metadata carried with one native frame.
struct FrameHeader
{
    SessionId session_id{};
    std::uint64_t sequence{};
    HostTimeNs host_received_ns{};
    DeviceTick source_tick{};
    HostTimeNs valid_until_ns{};
    SchemaId schema_id{};
    ClockDomainId source_clock_domain{};
    std::uint32_t signal_block_count{};
    FrameFlags flags{FrameFlags::none};
};

/// Fixed metadata for one signal payload within a possibly multi-rate frame.
struct SignalBlockHeader
{
    SampleIndex sample_idx_start{};
    DeviceTick device_tick_start{};
    /// Window-center time of the first regular feature observation.
    HostTimeNs observation_time_start_ns{};
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    SignalId signal_id{};
    std::uint32_t n_samples{};
    ClockSyncSnapshot clock_sync{};
    /// Last absolute event index for a non-empty sparse block. Dense signals
    /// and empty sparse blocks keep this field zero.
    SampleIndex last_sample_idx{};
};

/// Read-only frame view passed to actuators and observer handoff points.
struct FrameView
{
    FrameHeader header{};
    BufferToken buffer{};
    std::span<const SignalBlockHeader> blocks{};
    std::span<const std::byte> payload{};
};

/// Writable frame view owned by the source or current processor stage.
class FramePool;
class FrameLease;

class MutableFrame
{
  public:
    MutableFrame() noexcept = default;
    MutableFrame(const MutableFrame&) = delete;
    MutableFrame& operator=(const MutableFrame&) = delete;
    MutableFrame(MutableFrame&&) noexcept = default;
    MutableFrame& operator=(MutableFrame&&) noexcept = default;

    [[nodiscard]] FrameHeader& header() noexcept
    {
        return header_;
    }
    [[nodiscard]] const FrameHeader& header() const noexcept
    {
        return header_;
    }

    [[nodiscard]] std::span<SignalBlockHeader> block_storage() noexcept
    {
        return block_storage_;
    }

    [[nodiscard]] std::span<std::byte> payload_storage() noexcept
    {
        return payload_storage_;
    }

    [[nodiscard]] std::span<SignalBlockHeader> blocks() noexcept
    {
        return block_storage_.first(n_blocks_);
    }

    [[nodiscard]] std::span<std::byte> payload() noexcept
    {
        return payload_storage_.first(payload_size_);
    }

    [[nodiscard]] StreamStatus set_used_sizes(std::size_t n_blocks,
                                              std::size_t payload_size) noexcept
    {
        if (n_blocks > block_storage_.size() ||
            n_blocks > std::numeric_limits<std::uint32_t>::max() ||
            payload_size > payload_storage_.size())
        {
            return StreamStatus::invalid_frame;
        }
        n_blocks_ = n_blocks;
        payload_size_ = payload_size;
        header_.signal_block_count = static_cast<std::uint32_t>(n_blocks);
        return StreamStatus::ok;
    }

    /// Return a read-only view without copying the payload.
    [[nodiscard]] FrameView view() const noexcept
    {
        return FrameView{
            .header = header_,
            .buffer = token_,
            .blocks = block_storage_.first(n_blocks_),
            .payload = payload_storage_.first(payload_size_),
        };
    }

  private:
    friend class FramePool;
    friend class FrameLease;

    void attach(BufferToken token, std::span<SignalBlockHeader> blocks,
                std::span<std::byte> payload) noexcept
    {
        header_ = {};
        token_ = token;
        block_storage_ = blocks;
        payload_storage_ = payload;
        n_blocks_ = 0;
        payload_size_ = 0;
    }

    void clear() noexcept
    {
        header_ = {};
        token_ = {};
        block_storage_ = {};
        payload_storage_ = {};
        n_blocks_ = 0;
        payload_size_ = 0;
    }

    FrameHeader header_{};
    BufferToken token_{};
    std::span<SignalBlockHeader> block_storage_{};
    std::span<std::byte> payload_storage_{};
    std::size_t n_blocks_{};
    std::size_t payload_size_{};
};

static_assert(sizeof(FrameHeader) == 56);
static_assert(sizeof(ClockSyncSnapshot) == 56);
static_assert(sizeof(SignalBlockHeader) == 112);
static_assert(std::is_standard_layout_v<FrameHeader>);
static_assert(std::is_standard_layout_v<SignalBlockHeader>);
static_assert(std::is_trivially_move_constructible_v<FrameHeader>);
static_assert(std::is_trivially_move_constructible_v<SignalBlockHeader>);

} // namespace neurale::streaming
