/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/frame.h>

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace neurale::streaming::python
{

namespace py = pybind11;

struct SignalBlock
{
    SampleIndex sample_idx_start{};
    DeviceTick device_tick_start{};
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    SignalId signal_id{};
    std::uint32_t n_samples{};
    ClockSyncSnapshot clock_sync{};
    HostTimeNs observation_time_start_ns{};
    SampleIndex last_sample_idx{};
};

struct Frame
{
    SessionId session_id{};
    std::uint64_t sequence{};
    HostTimeNs host_received_ns{};
    std::optional<DeviceTick> source_tick;
    std::optional<HostTimeNs> valid_until_ns;
    SchemaId schema_id{};
    ClockDomainId source_clock_domain{};
    std::vector<SignalBlock> blocks;
    py::array_t<std::uint8_t> payload;
};

struct SignalGap
{
    SampleIndex expected_sample_idx{};
    SampleIndex actual_sample_idx{};
    std::optional<std::uint64_t> missing_samples;
    std::optional<DeviceTick> expected_device_tick;
    std::optional<DeviceTick> actual_device_tick;
    SignalId signal_id{};
    GapReason reason{GapReason::frame_sequence_gap};
};

struct Discontinuity
{
    SessionId session_id{};
    std::uint64_t previous_frame_sequence{};
    std::uint64_t actual_frame_sequence{};
    GapReason reason{GapReason::frame_sequence_gap};
    std::vector<SignalGap> signal_gaps;
};

[[nodiscard]] inline Frame copy_frame(FrameView frame)
{
    std::vector<SignalBlock> blocks;
    blocks.reserve(frame.blocks.size());
    for (const auto& block : frame.blocks)
    {
        blocks.push_back(SignalBlock{
            .sample_idx_start = block.sample_idx_start,
            .device_tick_start = block.device_tick_start,
            .payload_offset = block.payload_offset,
            .payload_byte_count = block.payload_byte_count,
            .signal_id = block.signal_id,
            .n_samples = block.n_samples,
            .clock_sync = block.clock_sync,
            .observation_time_start_ns = block.observation_time_start_ns,
            .last_sample_idx = block.last_sample_idx,
        });
    }
    py::array_t<std::uint8_t> payload(frame.payload.size());
    if (!frame.payload.empty())
    {
        std::memcpy(payload.mutable_data(), frame.payload.data(), frame.payload.size());
    }
    return Frame{
        .session_id = frame.header.session_id,
        .sequence = frame.header.sequence,
        .host_received_ns = frame.header.host_received_ns,
        .source_tick = has_flag(frame.header.flags, FrameFlags::source_tick)
                           ? std::optional<DeviceTick>{frame.header.source_tick}
                           : std::nullopt,
        .valid_until_ns = has_flag(frame.header.flags, FrameFlags::valid_until)
                              ? std::optional<HostTimeNs>{frame.header.valid_until_ns}
                              : std::nullopt,
        .schema_id = frame.header.schema_id,
        .source_clock_domain = frame.header.source_clock_domain,
        .blocks = std::move(blocks),
        .payload = std::move(payload),
    };
}

[[nodiscard]] inline Discontinuity
copy_discontinuity(const neurale::streaming::Discontinuity& value)
{
    std::vector<SignalGap> gaps;
    gaps.reserve(value.signal_gaps.size());
    for (const auto& gap : value.signal_gaps)
    {
        const auto samples_known = has_flag(gap.flags, SignalGapFlags::missing_samples_known);
        const auto ticks_available = has_flag(gap.flags, SignalGapFlags::device_ticks_available);
        gaps.push_back(SignalGap{
            .expected_sample_idx = gap.expected_sample_idx,
            .actual_sample_idx = gap.actual_sample_idx,
            .missing_samples =
                samples_known ? std::optional<std::uint64_t>{gap.missing_samples} : std::nullopt,
            .expected_device_tick = ticks_available
                                        ? std::optional<DeviceTick>{gap.expected_device_tick}
                                        : std::nullopt,
            .actual_device_tick =
                ticks_available ? std::optional<DeviceTick>{gap.actual_device_tick} : std::nullopt,
            .signal_id = gap.signal_id,
            .reason = gap.reason,
        });
    }
    return Discontinuity{
        .session_id = value.session_id,
        .previous_frame_sequence = value.previous_frame_sequence,
        .actual_frame_sequence = value.actual_frame_sequence,
        .reason = value.reason,
        .signal_gaps = std::move(gaps),
    };
}

[[nodiscard]] inline StreamStatus copy_frame(const Frame& source, MutableFrame& destination)
{
    if (source.blocks.size() > destination.block_storage().size() || source.payload.ndim() != 1 ||
        static_cast<std::size_t>(source.payload.size()) > destination.payload_storage().size())
    {
        destination.header() = {};
        static_cast<void>(destination.set_used_sizes(0, 0));
        return StreamStatus::invalid_frame;
    }

    FrameFlags flags{};
    if (source.source_tick.has_value())
    {
        flags = flags | FrameFlags::source_tick;
    }
    if (source.valid_until_ns.has_value())
    {
        flags = flags | FrameFlags::valid_until;
    }
    destination.header() = FrameHeader{
        .session_id = source.session_id,
        .sequence = source.sequence,
        .host_received_ns = source.host_received_ns,
        .source_tick = source.source_tick.value_or(0),
        .valid_until_ns = source.valid_until_ns.value_or(0),
        .schema_id = source.schema_id,
        .source_clock_domain = source.source_clock_domain,
        .flags = flags,
    };
    for (std::size_t i = 0; i < source.blocks.size(); ++i)
    {
        const auto& block = source.blocks[i];
        destination.block_storage()[i] = SignalBlockHeader{
            .sample_idx_start = block.sample_idx_start,
            .device_tick_start = block.device_tick_start,
            .observation_time_start_ns = block.observation_time_start_ns,
            .payload_offset = block.payload_offset,
            .payload_byte_count = block.payload_byte_count,
            .signal_id = block.signal_id,
            .n_samples = block.n_samples,
            .clock_sync = block.clock_sync,
        };
    }
    const auto payload_size = static_cast<std::size_t>(source.payload.size());
    if (payload_size != 0)
    {
        std::memcpy(destination.payload_storage().data(), source.payload.data(), payload_size);
    }
    return destination.set_used_sizes(source.blocks.size(), payload_size);
}

} // namespace neurale::streaming::python
