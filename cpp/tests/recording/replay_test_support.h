/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// A test-only writer for the private replay-image container.
///
/// The production writer is `neurale.recording._replay_build`, and it needs a
/// committed NRF session to write anything at all. These tests are about the
/// native reader and the run it drives, so they need images they can state
/// exactly -- an image with one frame, an image whose discontinuity carries
/// three gaps, an image whose timeline runs backwards -- and building each one
/// through a recorder would test the recorder instead.
///
/// The duplication that costs is the *format*: this file states the layout a
/// second time, so a change to `_replay_image.py` that this file did not
/// follow would leave both sides self-consistent and disagreeing with each
/// other. That is what `tests/unit/recording/test_native_parity.py`
/// exists for -- it hands a native binary an image the production Python
/// writer produced. Keep the two in step.

#include "byte_order.h"
#include "crc32c.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace neurale::recording::test
{

/// A scratch directory that removes itself, so a failing assertion cannot
/// leave an image behind for the next run to trip over.
class Scratch
{
  public:
    explicit Scratch(const char* prefix = "neurale-replay")
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ =
            std::filesystem::temp_directory_path() / (std::string(prefix) + std::to_string(stamp));
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_);
    }

    ~Scratch()
    {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }

    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    [[nodiscard]] std::string path(const std::string& name) const
    {
        return (directory_ / name).string();
    }

  private:
    std::filesystem::path directory_;
};

inline constexpr std::uint32_t kAbsentU32 = 0xFFFFFFFFU;
inline constexpr std::uint64_t kAbsentU64 = 0xFFFFFFFFFFFFFFFFULL;

/// One emitted signal block, in the terms the builder writes.
struct BuiltBlock
{
    std::uint32_t native_signal_id{1};
    std::uint32_t block_idx_in_frame{};
    std::string stream{"stream"};
    std::uint32_t n_samples{1};
    std::uint64_t sample_idx_start{};
    std::uint64_t last_sample_idx{};
    std::uint64_t device_tick_start{};
    std::uint64_t observation_time_start_ns{};
    /// Offset and length inside this frame's payload region.
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    std::uint64_t original_payload_offset{kAbsentU64};
    std::uint64_t source_block_ordinal{kAbsentU64};
    std::uint64_t recorded_frame_sequence{kAbsentU64};
    bool clock_sync{true};
    std::uint64_t clock_reference_tick{};
    std::uint64_t clock_reference_host_ns{};
    std::uint64_t clock_rate_num{1};
    std::uint64_t clock_rate_den{1};
    std::uint64_t clock_uncertainty_ns{};
    std::uint32_t clock_domain{};
    std::uint32_t clock_generation{};
    std::uint32_t clock_flags{};
};

/// One per-signal gap of a discontinuity.
struct BuiltGap
{
    std::uint32_t native_signal_id{1};
    std::uint32_t gap_idx_in_message{};
    std::string reason{"sample_gap"};
    std::uint32_t gap_flags{};
    std::uint64_t expected_sample_idx{};
    std::uint64_t actual_sample_idx{};
    std::uint64_t missing_samples{kAbsentU64};
    std::uint64_t expected_device_tick{kAbsentU64};
    std::uint64_t actual_device_tick{kAbsentU64};
    std::uint64_t signal_gap_ordinal{kAbsentU64};
};

struct BuiltFrame
{
    std::uint64_t data_message_ordinal{kAbsentU64};
    std::uint64_t replay_sequence{};
    std::uint64_t original_sequence{kAbsentU64};
    std::uint64_t timeline_ns{};
    std::uint64_t original_host_received_ns{kAbsentU64};
    std::uint32_t native_schema_id{1};
    std::uint32_t source_clock_domain{};
    std::uint32_t frame_flags{};
    bool has_source_tick{};
    std::uint64_t source_tick{};
    bool has_valid_until{};
    std::uint64_t valid_until_ns{};
    std::vector<std::byte> payload;
    std::vector<BuiltBlock> blocks;
};

struct BuiltDiscontinuity
{
    std::uint64_t data_message_ordinal{kAbsentU64};
    std::uint64_t replay_sequence{};
    std::uint64_t original_sequence{kAbsentU64};
    std::uint64_t previous_replay_sequence{kAbsentU64};
    std::uint64_t original_previous_sequence{kAbsentU64};
    std::uint64_t timeline_ns{};
    std::string reason{"frame_sequence_gap"};
    bool frame_level{};
    std::vector<BuiltGap> gaps;
};

struct BuiltSignal
{
    std::uint32_t id{1};
    std::uint32_t clock_domain{};
    std::uint32_t n_channels{1};
    std::uint32_t nominal_block_samples{1};
    std::uint32_t max_block_samples{1};
    std::string dtype{"FLOAT32"};
    std::string layout{"SAMPLE_MAJOR"};
    std::string device_tick_tracking{"UNAVAILABLE"};
    std::string kind{"SAMPLED"};
    std::string physical_unit{"UNSPECIFIED"};
    std::string observation_timing{"NOT_APPLICABLE"};
    std::uint32_t channel_set_id{};
    std::uint32_t calibration_id{};
    std::uint32_t reference_id{};
    std::uint32_t feature_set_id{};
    std::uint64_t rate_num{1000};
    std::uint64_t rate_den{1};
    std::uint64_t fixed_block_bytes{};
    std::uint64_t max_block_bytes{4};
};

/// One feature-set descriptor of the declared schema.
struct BuiltFeatureSet
{
    std::uint32_t id{1};
    std::uint32_t source_stream_id{1};
    std::string source_stream{"stream"};
    std::string algorithm_name{"bandpower"};
    std::string algorithm_version{"1"};
    std::string timestamp_reference{"WINDOW_CENTER"};
    std::vector<std::string> feature_names{"alpha"};
    std::vector<std::uint32_t> unit_ids{1};
    std::uint64_t window_length_ns{2'000'000};
    std::uint64_t shift_ns{1'000'000};
};

/// One unit descriptor of the declared schema.
struct BuiltUnit
{
    std::uint32_t id{1};
    std::string symbol{"uV"};
    std::string description{"microvolts"};
};

struct BuiltFidelity
{
    std::string stream{"stream"};
    std::uint64_t native_signal_id{1};
    std::uint32_t flags{};
    std::uint64_t frames_emitted{};
    std::uint64_t blocks_emitted{};
    std::uint64_t discontinuities_emitted{};
    std::uint64_t committed_blocks{};
};

/// Assemble one image. Every deliberate corruption a container test needs is a
/// named knob rather than a byte offset in the test.
class ReplayImageBuilder
{
  public:
    std::string mode{"exact_frames"};
    std::uint32_t flags{1U << 2U}; // ledger based
    std::uint32_t schema_id{1};
    std::uint64_t message_range_start{};
    std::uint64_t message_range_stop{};
    /// The recording's own native SessionId, or absent. A replay run must
    /// never present this value as its own (contract section 8.12).
    std::uint64_t native_session_id{kAbsentU64};
    std::vector<BuiltSignal> signals;
    std::vector<BuiltFeatureSet> feature_sets;
    std::vector<BuiltUnit> units;
    std::vector<BuiltFidelity> fidelity;

    // Deliberate corruptions.
    std::uint16_t version_override{1};
    bool duplicate_items_section{};
    bool ragged_items_section{};
    bool lie_about_item_count{};
    bool lie_about_total_bytes{};
    bool break_content_checksum{};
    bool break_magic{};
    bool omit_summary_section{};

    void add_frame(BuiltFrame frame)
    {
        order_.push_back({1, frames_.size()});
        frames_.push_back(std::move(frame));
    }

    void add_discontinuity(BuiltDiscontinuity discontinuity)
    {
        order_.push_back({2, discontinuities_.size()});
        discontinuities_.push_back(std::move(discontinuity));
    }

    [[nodiscard]] bool write(const std::string& path);

  private:
    struct Entry
    {
        std::uint8_t kind;
        std::size_t idx;
    };

    std::uint32_t intern(std::string_view value)
    {
        for (std::size_t i = 0; i < strings_.size(); ++i)
        {
            if (strings_[i] == value)
            {
                return static_cast<std::uint32_t>(i);
            }
        }
        strings_.emplace_back(value);
        return static_cast<std::uint32_t>(strings_.size() - 1);
    }

    std::vector<Entry> order_;
    std::vector<BuiltFrame> frames_;
    std::vector<BuiltDiscontinuity> discontinuities_;
    std::vector<std::string> strings_;
};

namespace detail
{

inline void append_u8(std::vector<std::byte>& out, std::uint8_t value)
{
    out.push_back(static_cast<std::byte>(value));
}

inline void append_u16(std::vector<std::byte>& out, std::uint16_t value)
{
    out.resize(out.size() + 2);
    store_u16le(std::span<std::byte>(out).last(2), 0, value);
}

inline void append_u32(std::vector<std::byte>& out, std::uint32_t value)
{
    out.resize(out.size() + 4);
    store_u32le(std::span<std::byte>(out).last(4), 0, value);
}

inline void append_u64(std::vector<std::byte>& out, std::uint64_t value)
{
    out.resize(out.size() + 8);
    store_u64le(std::span<std::byte>(out).last(8), 0, value);
}

} // namespace detail

inline bool ReplayImageBuilder::write(const std::string& path)
{
    using detail::append_u16;
    using detail::append_u32;
    using detail::append_u64;
    using detail::append_u8;

    intern(""); // index 0 is the empty string, as the production writer does.

    std::vector<std::byte> items;
    std::vector<std::byte> blocks;
    std::vector<std::byte> gaps;
    std::vector<std::byte> payload;
    std::uint64_t frame_count = 0;
    std::uint64_t discontinuity_count = 0;
    std::uint64_t first_timeline = kAbsentU64;
    std::uint64_t last_timeline = kAbsentU64;

    for (const auto& entry : order_)
    {
        if (entry.kind == 1)
        {
            const auto& frame = frames_[entry.idx];
            const auto payload_offset = static_cast<std::uint64_t>(payload.size());
            payload.insert(payload.end(), frame.payload.begin(), frame.payload.end());
            const auto child_first = static_cast<std::uint64_t>(blocks.size() / 144);
            for (const auto& block : frame.blocks)
            {
                append_u32(blocks, block.native_signal_id);
                append_u32(blocks, block.block_idx_in_frame);
                append_u32(blocks, intern(block.stream));
                append_u32(blocks, block.n_samples);
                append_u64(blocks, block.sample_idx_start);
                append_u64(blocks, block.last_sample_idx);
                append_u64(blocks, block.device_tick_start);
                append_u64(blocks, block.observation_time_start_ns);
                append_u64(blocks, block.payload_offset);
                append_u64(blocks, block.payload_byte_count);
                append_u64(blocks, block.original_payload_offset);
                append_u64(blocks, block.clock_reference_tick);
                append_u64(blocks, block.clock_reference_host_ns);
                append_u64(blocks, block.clock_rate_num);
                append_u64(blocks, block.clock_rate_den);
                append_u64(blocks, block.clock_uncertainty_ns);
                append_u64(blocks, block.source_block_ordinal);
                append_u64(blocks, block.recorded_frame_sequence);
                append_u32(blocks, block.clock_domain);
                append_u32(blocks, block.clock_generation);
                append_u32(blocks, block.clock_flags);
                append_u32(blocks, block.clock_sync ? 1U : 0U);
            }
            std::uint8_t item_flags = 0;
            if (frame.has_source_tick)
            {
                item_flags |= 1U << 0U;
            }
            if (frame.has_valid_until)
            {
                item_flags |= 1U << 1U;
            }
            append_u8(items, 1);
            append_u8(items, item_flags);
            append_u16(items, 0);
            append_u32(items, static_cast<std::uint32_t>(frame.blocks.size()));
            append_u64(items, child_first);
            append_u64(items, frame.data_message_ordinal);
            append_u64(items, frame.replay_sequence);
            append_u64(items, frame.original_sequence);
            append_u64(items, kAbsentU64);
            append_u64(items, kAbsentU64);
            append_u64(items, frame.timeline_ns);
            append_u64(items, frame.original_host_received_ns);
            append_u64(items, payload_offset);
            append_u64(items, static_cast<std::uint64_t>(frame.payload.size()));
            append_u64(items, static_cast<std::uint64_t>(frame.payload.size()));
            append_u64(items, frame.has_source_tick ? frame.source_tick : kAbsentU64);
            append_u64(items, frame.has_valid_until ? frame.valid_until_ns : kAbsentU64);
            append_u32(items, frame.native_schema_id);
            append_u32(items, frame.source_clock_domain);
            append_u32(items, frame.frame_flags);
            append_u32(items, kAbsentU32);
            ++frame_count;
            if (first_timeline == kAbsentU64)
            {
                first_timeline = frame.timeline_ns;
            }
            last_timeline = frame.timeline_ns;
            continue;
        }

        const auto& record = discontinuities_[entry.idx];
        const auto child_first = static_cast<std::uint64_t>(gaps.size() / 64);
        for (const auto& gap : record.gaps)
        {
            append_u32(gaps, gap.native_signal_id);
            append_u32(gaps, gap.gap_idx_in_message);
            append_u32(gaps, intern(gap.reason));
            append_u32(gaps, gap.gap_flags);
            append_u64(gaps, gap.expected_sample_idx);
            append_u64(gaps, gap.actual_sample_idx);
            append_u64(gaps, gap.missing_samples);
            append_u64(gaps, gap.expected_device_tick);
            append_u64(gaps, gap.actual_device_tick);
            append_u64(gaps, gap.signal_gap_ordinal);
        }
        append_u8(items, 2);
        append_u8(items, record.frame_level ? (1U << 2U) : 0U);
        append_u16(items, 0);
        append_u32(items, static_cast<std::uint32_t>(record.gaps.size()));
        append_u64(items, child_first);
        append_u64(items, record.data_message_ordinal);
        append_u64(items, record.replay_sequence);
        append_u64(items, record.original_sequence);
        append_u64(items, record.previous_replay_sequence);
        append_u64(items, record.original_previous_sequence);
        append_u64(items, record.timeline_ns);
        append_u64(items, kAbsentU64);
        append_u64(items, 0);
        append_u64(items, 0);
        append_u64(items, kAbsentU64);
        append_u64(items, kAbsentU64);
        append_u64(items, kAbsentU64);
        append_u32(items, 0);
        append_u32(items, 0);
        append_u32(items, 0);
        append_u32(items, intern(record.reason));
        ++discontinuity_count;
        if (first_timeline == kAbsentU64)
        {
            first_timeline = record.timeline_ns;
        }
        last_timeline = record.timeline_ns;
    }

    std::vector<std::byte> fidelity_section;
    std::vector<std::uint32_t> lists;
    for (const auto& entry : fidelity)
    {
        append_u32(fidelity_section, intern(entry.stream));
        append_u32(fidelity_section, entry.flags);
        append_u32(fidelity_section, static_cast<std::uint32_t>(lists.size()));
        append_u32(fidelity_section, 0);
        append_u64(fidelity_section, entry.native_signal_id);
        append_u64(fidelity_section, entry.frames_emitted);
        append_u64(fidelity_section, entry.blocks_emitted);
        append_u64(fidelity_section, entry.discontinuities_emitted);
        append_u64(fidelity_section, entry.committed_blocks);
        append_u64(fidelity_section, 0);
    }

    std::vector<std::byte> schema_section;
    if (!signals.empty())
    {
        append_u32(schema_section, schema_id);
        append_u32(schema_section, static_cast<std::uint32_t>(signals.size()));
        append_u32(schema_section, static_cast<std::uint32_t>(feature_sets.size()));
        append_u32(schema_section, static_cast<std::uint32_t>(units.size()));
        for (const auto& signal : signals)
        {
            append_u32(schema_section, signal.id);
            append_u32(schema_section, signal.clock_domain);
            append_u32(schema_section, signal.n_channels);
            append_u32(schema_section, signal.nominal_block_samples);
            append_u32(schema_section, signal.max_block_samples);
            append_u32(schema_section, intern(signal.dtype));
            append_u32(schema_section, intern(signal.layout));
            append_u32(schema_section, intern(signal.device_tick_tracking));
            append_u32(schema_section, intern(signal.kind));
            append_u32(schema_section, intern(signal.physical_unit));
            append_u32(schema_section, signal.channel_set_id);
            append_u32(schema_section, signal.calibration_id);
            append_u32(schema_section, signal.reference_id);
            append_u32(schema_section, signal.feature_set_id);
            append_u32(schema_section, intern(signal.observation_timing));
            append_u32(schema_section, 0);
            append_u64(schema_section, signal.rate_num);
            append_u64(schema_section, signal.rate_den);
            append_u64(schema_section, signal.fixed_block_bytes);
            append_u64(schema_section, signal.max_block_bytes);
        }
        for (const auto& feature_set : feature_sets)
        {
            const auto names_first = static_cast<std::uint32_t>(lists.size());
            for (const auto& name : feature_set.feature_names)
            {
                lists.push_back(intern(name));
            }
            const auto units_first = static_cast<std::uint32_t>(lists.size());
            for (const auto unit_id : feature_set.unit_ids)
            {
                lists.push_back(unit_id);
            }
            append_u32(schema_section, feature_set.id);
            append_u32(schema_section, feature_set.source_stream_id);
            append_u32(schema_section, intern(feature_set.source_stream));
            append_u32(schema_section, intern(feature_set.algorithm_name));
            append_u32(schema_section, intern(feature_set.algorithm_version));
            append_u32(schema_section, intern(feature_set.timestamp_reference));
            append_u32(schema_section, names_first);
            append_u32(schema_section,
                       static_cast<std::uint32_t>(feature_set.feature_names.size()));
            append_u32(schema_section, units_first);
            append_u32(schema_section, static_cast<std::uint32_t>(feature_set.unit_ids.size()));
            append_u64(schema_section, feature_set.window_length_ns);
            append_u64(schema_section, feature_set.shift_ns);
            append_u64(schema_section, 0);
        }
        for (const auto& unit : units)
        {
            append_u32(schema_section, unit.id);
            append_u32(schema_section, intern(unit.symbol));
            append_u32(schema_section, intern(unit.description));
            append_u32(schema_section, 0);
        }
    }

    // The summary references strings, so it is packed after everything that
    // interns one.
    std::vector<std::byte> summary;
    const auto n_items = static_cast<std::uint64_t>(items.size() / 128);
    append_u32(summary, intern(mode));
    append_u32(summary, kAbsentU32);
    append_u32(summary, kAbsentU32);
    append_u32(summary, intern("session"));
    append_u32(summary, kAbsentU32);
    append_u32(summary, intern("fingerprint"));
    append_u32(summary, flags);
    append_u32(summary, 0);
    append_u32(summary, 0);
    append_u32(summary, 0);
    append_u32(summary, 0);
    append_u32(summary, 0);
    append_u32(summary, 0);
    append_u32(summary, intern("one_frame_per_block"));
    append_u32(summary, intern("original_host_received_ns"));
    append_u32(summary, 0);
    append_u64(summary, message_range_start);
    append_u64(summary, message_range_stop);
    append_u64(summary, lie_about_item_count ? n_items + 1 : n_items);
    append_u64(summary, frame_count);
    append_u64(summary, discontinuity_count);
    append_u64(summary, static_cast<std::uint64_t>(blocks.size() / 144));
    append_u64(summary, static_cast<std::uint64_t>(gaps.size() / 64));
    append_u64(summary, 0);
    append_u64(summary, static_cast<std::uint64_t>(payload.size()));
    append_u64(summary, native_session_id);
    append_u64(summary, n_items);
    append_u64(summary, first_timeline);
    append_u64(summary, last_timeline);
    summary.resize(256);

    std::vector<std::byte> string_section;
    append_u32(string_section, static_cast<std::uint32_t>(strings_.size()));
    append_u32(string_section, 0);
    std::vector<std::byte> blob;
    for (const auto& value : strings_)
    {
        append_u32(string_section, static_cast<std::uint32_t>(blob.size()));
        append_u32(string_section, static_cast<std::uint32_t>(value.size()));
        for (const char character : value)
        {
            blob.push_back(static_cast<std::byte>(character));
        }
    }
    string_section.insert(string_section.end(), blob.begin(), blob.end());

    std::vector<std::byte> list_section;
    append_u32(list_section, static_cast<std::uint32_t>(lists.size()));
    append_u32(list_section, 0);
    for (const auto value : lists)
    {
        append_u32(list_section, value);
    }

    if (ragged_items_section)
    {
        items.push_back(std::byte{0});
    }

    struct Section
    {
        std::uint32_t kind;
        std::uint32_t width;
        const std::vector<std::byte>* body;
    };
    std::vector<Section> sections{
        {1, 0, &string_section}, {2, 0, &list_section},
        {3, 256, &summary},      {4, 0, &schema_section},
        {6, 128, &items},        {7, 144, &blocks},
        {8, 64, &gaps},          {10, 64, &fidelity_section},
    };
    if (omit_summary_section)
    {
        sections.erase(sections.begin() + 2);
    }
    if (duplicate_items_section)
    {
        sections.push_back({6, 128, &items});
    }
    sections.push_back({11, 0, &payload});

    std::vector<std::byte> table;
    std::uint64_t offset = 64 + 24ULL * sections.size();
    for (const auto& section : sections)
    {
        append_u32(table, section.kind);
        append_u32(table, section.width);
        append_u64(table, offset);
        append_u64(table, static_cast<std::uint64_t>(section.body->size()));
        offset += section.body->size();
    }
    const auto total = offset;

    std::vector<std::byte> content;
    content.insert(content.end(), table.begin(), table.end());
    for (const auto& section : sections)
    {
        content.insert(content.end(), section.body->begin(), section.body->end());
    }
    auto content_crc = crc32c(std::span<const std::byte>(content));
    if (break_content_checksum)
    {
        content_crc ^= 0xFFFFFFFFU;
    }

    std::vector<std::byte> header;
    const char magic[8] = {'N', 'R', 'L', 'R', 'P', 'I', 'M', 'G'};
    for (const char character : magic)
    {
        header.push_back(static_cast<std::byte>(character));
    }
    if (break_magic)
    {
        header[0] = std::byte{'X'};
    }
    append_u16(header, version_override);
    append_u16(header, static_cast<std::uint16_t>(sections.size()));
    append_u32(header, 64);
    for (std::size_t i = 0; i < 32; ++i)
    {
        header.push_back(static_cast<std::byte>(i));
    }
    append_u64(header, lie_about_total_bytes ? total + 1 : total);
    append_u32(header, content_crc);
    append_u32(header, crc32c(std::span<const std::byte>(header)));

    std::FILE* handle = std::fopen(path.c_str(), "wb");
    if (handle == nullptr)
    {
        return false;
    }
    const auto write_all = [handle](const std::vector<std::byte>& body)
    { return body.empty() || std::fwrite(body.data(), 1, body.size(), handle) == body.size(); };
    bool ok = write_all(header) && write_all(content);
    std::fclose(handle);
    return ok;
}

} // namespace neurale::recording::test
