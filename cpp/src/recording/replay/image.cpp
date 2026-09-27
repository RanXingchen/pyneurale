/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "replay/image.h"

#include "byte_order.h"
#include "crc32c.h"

#include <cstdio>
#include <cstring>
#include <utility>

#ifdef _WIN32
#include <cstdio>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace neurale::recording
{
namespace
{

constexpr std::size_t kHeaderBytes = 64;
constexpr std::size_t kSectionEntryBytes = 24;
constexpr std::uint16_t kFormatVersion = 1;
constexpr std::size_t kFingerprintBytes = 32;
constexpr unsigned char kMagic[8] = {'N', 'R', 'L', 'R', 'P', 'I', 'M', 'G'};

// Section identifiers, mirroring `_replay_image.SECTION_*`.
constexpr std::uint32_t kSectionStrings = 1;
constexpr std::uint32_t kSectionLists = 2;
constexpr std::uint32_t kSectionSummary = 3;
constexpr std::uint32_t kSectionSchema = 4;
constexpr std::uint32_t kSectionStreamRanges = 5;
constexpr std::uint32_t kSectionItems = 6;
constexpr std::uint32_t kSectionBlocks = 7;
constexpr std::uint32_t kSectionGaps = 8;
constexpr std::uint32_t kSectionOmissions = 9;
constexpr std::uint32_t kSectionFidelity = 10;
constexpr std::uint32_t kSectionPayload = 11;
constexpr std::uint32_t kSectionIdRegistry = 12;

constexpr std::size_t kSummaryBytes = 256;
constexpr std::size_t kStreamRangeBytes = 32;
constexpr std::size_t kItemBytes = 128;
constexpr std::size_t kBlockBytes = 144;
constexpr std::size_t kGapBytes = 64;
constexpr std::size_t kOmissionBytes = 32;
constexpr std::size_t kFidelityBytes = 64;
constexpr std::size_t kSignalBytes = 96;
constexpr std::size_t kFeatureSetBytes = 64;
constexpr std::size_t kUnitBytes = 16;
constexpr std::size_t kIdRegistryHeaderBytes = 16;
constexpr std::size_t kIdRegistryEntryBytes = 8;

/// Fixed record width per section, or 0 where the section is opaque bytes.
[[nodiscard]] std::size_t section_record_bytes(std::uint32_t kind) noexcept
{
    switch (kind)
    {
    case kSectionSummary:
        return kSummaryBytes;
    case kSectionStreamRanges:
        return kStreamRangeBytes;
    case kSectionItems:
        return kItemBytes;
    case kSectionBlocks:
        return kBlockBytes;
    case kSectionGaps:
        return kGapBytes;
    case kSectionOmissions:
        return kOmissionBytes;
    case kSectionFidelity:
        return kFidelityBytes;
    default:
        return 0;
    }
}

} // namespace

const char* replay_image_status_text(ReplayImageStatus status) noexcept
{
    switch (status)
    {
    case ReplayImageStatus::ok:
        return "ok";
    case ReplayImageStatus::unreadable:
        return "unreadable";
    case ReplayImageStatus::not_an_image:
        return "not_an_image";
    case ReplayImageStatus::checksum_mismatch:
        return "checksum_mismatch";
    case ReplayImageStatus::unsupported_version:
        return "unsupported_version";
    case ReplayImageStatus::truncated:
        return "truncated";
    case ReplayImageStatus::malformed:
        return "malformed";
    }
    return "unknown";
}

ReplayImageFile::~ReplayImageFile()
{
    close();
}

ReplayImageFile::ReplayImageFile(ReplayImageFile&& other) noexcept
{
    *this = std::move(other);
}

ReplayImageFile& ReplayImageFile::operator=(ReplayImageFile&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    close();
    data_ = other.data_;
    mapping_ = other.mapping_;
    mapping_bytes_ = other.mapping_bytes_;
    buffer_ = std::move(other.buffer_);
    for (std::size_t i = 0; i < kSectionSlots; ++i)
    {
        sections_[i] = other.sections_[i];
    }
    string_offsets_ = std::move(other.string_offsets_);
    string_lengths_ = std::move(other.string_lengths_);
    string_blob_ = other.string_blob_;
    lists_ = other.lists_;
    n_lists_ = other.n_lists_;
    summary_ = other.summary_;
    signals_ = std::move(other.signals_);
    feature_sets_ = std::move(other.feature_sets_);
    units_ = std::move(other.units_);
    schema_id_ = other.schema_id_;
    n_items_ = other.n_items_;
    n_blocks_ = other.n_blocks_;
    n_gaps_ = other.n_gaps_;
    n_fidelities_ = other.n_fidelities_;
    n_omissions_ = other.n_omissions_;
    stream_range_count_ = other.stream_range_count_;
    id_registry_counts_ = other.id_registry_counts_;
    id_registry_offsets_ = other.id_registry_offsets_;
    std::memcpy(fingerprint_, other.fingerprint_, kFingerprintBytes);
    other.data_ = {};
    other.mapping_ = nullptr;
    other.mapping_bytes_ = 0;
    other.lists_ = {};
    return *this;
}

void ReplayImageFile::close() noexcept
{
#ifndef _WIN32
    if (mapping_ != nullptr)
    {
        ::munmap(mapping_, mapping_bytes_);
    }
#endif
    mapping_ = nullptr;
    mapping_bytes_ = 0;
    buffer_.clear();
    buffer_.shrink_to_fit();
    data_ = {};
    lists_ = {};
    n_lists_ = 0;
    string_offsets_.clear();
    string_lengths_.clear();
    signals_.clear();
    feature_sets_.clear();
    units_.clear();
    summary_ = ReplayImageSummary{};
    schema_id_ = 0;
    n_items_ = 0;
    n_blocks_ = 0;
    n_gaps_ = 0;
    n_fidelities_ = 0;
    n_omissions_ = 0;
    stream_range_count_ = 0;
    id_registry_counts_.fill(0);
    id_registry_offsets_.fill(0);
    for (std::size_t i = 0; i < kSectionSlots; ++i)
    {
        sections_[i] = Section{};
    }
    std::memset(fingerprint_, 0, kFingerprintBytes);
}

ReplayImageStatus ReplayImageFile::map(const char* path) noexcept
{
#ifdef _WIN32
    // No mapping here on purpose. A Windows file mapping written blind would
    // be untested code on the one path that must never hand out a partial
    // view, so the image is read whole instead and the cost is stated rather
    // than hidden.
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr)
    {
        return ReplayImageStatus::unreadable;
    }
    if (std::fseek(handle, 0, SEEK_END) != 0)
    {
        std::fclose(handle);
        return ReplayImageStatus::unreadable;
    }
    const auto size = std::ftell(handle);
    if (size < 0 || std::fseek(handle, 0, SEEK_SET) != 0)
    {
        std::fclose(handle);
        return ReplayImageStatus::unreadable;
    }
    try
    {
        buffer_.resize(static_cast<std::size_t>(size));
    }
    catch (...)
    {
        std::fclose(handle);
        return ReplayImageStatus::unreadable;
    }
    const auto read = std::fread(buffer_.data(), 1, buffer_.size(), handle);
    std::fclose(handle);
    if (read != buffer_.size())
    {
        buffer_.clear();
        return ReplayImageStatus::unreadable;
    }
    data_ = std::span<const std::byte>(buffer_.data(), buffer_.size());
    return ReplayImageStatus::ok;
#else
    const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
    {
        return ReplayImageStatus::unreadable;
    }
    struct stat info{};
    if (::fstat(descriptor, &info) != 0 || info.st_size <= 0)
    {
        ::close(descriptor);
        return ReplayImageStatus::unreadable;
    }
    const auto bytes = static_cast<std::size_t>(info.st_size);
    void* address = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
    // The mapping outlives the descriptor, so the descriptor is closed at once
    // rather than kept for the life of the image.
    ::close(descriptor);
    if (address == MAP_FAILED)
    {
        return ReplayImageStatus::unreadable;
    }
    mapping_ = address;
    mapping_bytes_ = bytes;
    data_ = std::span<const std::byte>(static_cast<const std::byte*>(address), bytes);
    return ReplayImageStatus::ok;
#endif
}

ReplayImageStatus ReplayImageFile::open(const char* path) noexcept
{
    close();
    if (path == nullptr)
    {
        return ReplayImageStatus::unreadable;
    }
    auto status = map(path);
    if (status != ReplayImageStatus::ok)
    {
        close();
        return status;
    }
    status = validate();
    if (status != ReplayImageStatus::ok)
    {
        close();
        return status;
    }
    status = read_strings();
    if (status == ReplayImageStatus::ok)
    {
        status = read_summary();
    }
    if (status == ReplayImageStatus::ok)
    {
        status = read_schema();
    }
    if (status == ReplayImageStatus::ok)
    {
        status = read_id_registry();
    }
    if (status != ReplayImageStatus::ok)
    {
        close();
    }
    return status;
}

ReplayImageStatus ReplayImageFile::validate() noexcept
{
    if (data_.size() < kHeaderBytes)
    {
        return ReplayImageStatus::truncated;
    }
    for (std::size_t i = 0; i < sizeof(kMagic); ++i)
    {
        if (load_u8(data_, i) != kMagic[i])
        {
            return ReplayImageStatus::not_an_image;
        }
    }
    const auto stored_header_crc = load_u32le(data_, kHeaderBytes - 4);
    if (stored_header_crc != crc32c(data_.first(kHeaderBytes - 4)))
    {
        return ReplayImageStatus::checksum_mismatch;
    }
    const auto version = load_u16le(data_, 8);
    if (version != kFormatVersion)
    {
        return ReplayImageStatus::unsupported_version;
    }
    const auto n_sections = load_u16le(data_, 10);
    if (load_u32le(data_, 12) != kHeaderBytes)
    {
        return ReplayImageStatus::malformed;
    }
    for (std::size_t i = 0; i < kFingerprintBytes; ++i)
    {
        fingerprint_[i] = data_[16 + i];
    }
    const auto total = load_u64le(data_, 48);
    if (total != data_.size())
    {
        return ReplayImageStatus::truncated;
    }
    const auto content_crc = load_u32le(data_, 56);

    const std::uint64_t table_end =
        kHeaderBytes + static_cast<std::uint64_t>(kSectionEntryBytes) * n_sections;
    if (table_end > total)
    {
        return ReplayImageStatus::truncated;
    }
    if (content_crc != crc32c(data_.subspan(kHeaderBytes)))
    {
        return ReplayImageStatus::checksum_mismatch;
    }

    for (std::size_t i = 0; i < n_sections; ++i)
    {
        const auto base = kHeaderBytes + kSectionEntryBytes * i;
        const auto kind = load_u32le(data_, base);
        const auto width = load_u32le(data_, base + 4);
        const auto offset = load_u64le(data_, base + 8);
        const auto length = load_u64le(data_, base + 16);
        if (offset < table_end || offset > total || length > total - offset)
        {
            return ReplayImageStatus::malformed;
        }
        const auto expected = section_record_bytes(kind);
        if (expected != 0 && (width != expected || (length % expected) != 0))
        {
            return ReplayImageStatus::malformed;
        }
        if (kind >= kSectionSlots)
        {
            // An identifier this build does not know is skipped, which is what
            // makes appending a section a compatible change. It is still
            // covered by the content checksum.
            continue;
        }
        if (sections_[kind].present)
        {
            return ReplayImageStatus::malformed;
        }
        sections_[kind] = Section{.offset = offset, .length = length, .present = true};
    }

    for (const auto required : {kSectionStrings, kSectionLists, kSectionSummary, kSectionItems})
    {
        if (!sections_[required].present)
        {
            return ReplayImageStatus::malformed;
        }
    }
    n_items_ = static_cast<std::size_t>(sections_[kSectionItems].length / kItemBytes);
    n_blocks_ = static_cast<std::size_t>(sections_[kSectionBlocks].length / kBlockBytes);
    n_gaps_ = static_cast<std::size_t>(sections_[kSectionGaps].length / kGapBytes);
    n_fidelities_ = static_cast<std::size_t>(sections_[kSectionFidelity].length / kFidelityBytes);
    n_omissions_ = static_cast<std::size_t>(sections_[kSectionOmissions].length / kOmissionBytes);
    stream_range_count_ =
        static_cast<std::size_t>(sections_[kSectionStreamRanges].length / kStreamRangeBytes);
    return ReplayImageStatus::ok;
}

std::span<const std::byte> ReplayImageFile::section(std::uint32_t kind) const noexcept
{
    if (kind >= kSectionSlots || !sections_[kind].present)
    {
        return {};
    }
    return data_.subspan(static_cast<std::size_t>(sections_[kind].offset),
                         static_cast<std::size_t>(sections_[kind].length));
}

ReplayImageStatus ReplayImageFile::read_strings() noexcept
{
    const auto body = section(kSectionStrings);
    if (body.size() < 8)
    {
        return ReplayImageStatus::truncated;
    }
    const auto count = load_u32le(body, 0);
    const std::uint64_t entries = 8;
    const std::uint64_t blob = entries + 8ULL * count;
    if (blob > body.size())
    {
        return ReplayImageStatus::truncated;
    }
    try
    {
        string_offsets_.resize(count);
        string_lengths_.resize(count);
    }
    catch (...)
    {
        return ReplayImageStatus::unreadable;
    }
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const auto base = static_cast<std::size_t>(entries + 8ULL * i);
        const auto offset = load_u32le(body, base);
        const auto length = load_u32le(body, base + 4);
        if (blob + offset + length > body.size())
        {
            return ReplayImageStatus::malformed;
        }
        string_offsets_[i] = offset;
        string_lengths_[i] = length;
    }
    string_blob_ = blob;

    const auto lists = section(kSectionLists);
    if (lists.size() < 8)
    {
        return ReplayImageStatus::truncated;
    }
    const auto n_lists = load_u32le(lists, 0);
    if (8ULL + 4ULL * n_lists > lists.size())
    {
        return ReplayImageStatus::truncated;
    }
    lists_ = lists.subspan(8, static_cast<std::size_t>(4ULL * n_lists));
    n_lists_ = n_lists;
    return ReplayImageStatus::ok;
}

ReplayImageStatus ReplayImageFile::read_summary() noexcept
{
    const auto body = section(kSectionSummary);
    if (body.size() != kSummaryBytes)
    {
        return ReplayImageStatus::malformed;
    }
    summary_.mode = load_u32le(body, 0);
    summary_.plan_coverage = load_u32le(body, 4);
    summary_.completeness = load_u32le(body, 8);
    summary_.source_session_id = load_u32le(body, 12);
    summary_.plan_fingerprint = load_u32le(body, 16);
    summary_.source_fingerprint = load_u32le(body, 20);
    summary_.flags = load_u32le(body, 24);
    summary_.selected_list = load_u32le(body, 28);
    summary_.n_selected = load_u32le(body, 32);
    summary_.planned_list = load_u32le(body, 36);
    summary_.n_planned = load_u32le(body, 40);
    summary_.recorded_list = load_u32le(body, 44);
    summary_.n_recorded = load_u32le(body, 48);
    summary_.frame_construction = load_u32le(body, 52);
    summary_.ordering_key = load_u32le(body, 56);
    summary_.range_start = load_u64le(body, 64);
    summary_.range_stop = load_u64le(body, 72);
    summary_.n_items = load_u64le(body, 80);
    summary_.n_frames = load_u64le(body, 88);
    summary_.n_discontinuities = load_u64le(body, 96);
    summary_.n_blocks = load_u64le(body, 104);
    summary_.n_gaps = load_u64le(body, 112);
    summary_.n_omissions = load_u64le(body, 120);
    summary_.payload_byte_count = load_u64le(body, 128);
    summary_.native_session_id = load_u64le(body, 136);
    summary_.source_message_count = load_u64le(body, 144);
    summary_.first_timeline_ns = load_u64le(body, 152);
    summary_.last_timeline_ns = load_u64le(body, 160);

    // The summary counts and the sections that carry the records are two
    // statements of the same fact, and an image whose two statements disagree
    // is not one this build can replay.
    if (summary_.n_items != n_items_ || summary_.n_blocks != n_blocks_ ||
        summary_.n_gaps != n_gaps_)
    {
        return ReplayImageStatus::malformed;
    }
    if (summary_.payload_byte_count != sections_[kSectionPayload].length)
    {
        return ReplayImageStatus::malformed;
    }
    return ReplayImageStatus::ok;
}

ReplayImageStatus ReplayImageFile::read_schema() noexcept
{
    const auto body = section(kSectionSchema);
    if (body.empty())
    {
        return ReplayImageStatus::ok;
    }
    if (body.size() < 16)
    {
        return ReplayImageStatus::truncated;
    }
    schema_id_ = load_u32le(body, 0);
    const auto n_signals = load_u32le(body, 4);
    const auto n_features = load_u32le(body, 8);
    const auto n_units = load_u32le(body, 12);
    const std::uint64_t declared = 16ULL + kSignalBytes * static_cast<std::uint64_t>(n_signals) +
                                   kFeatureSetBytes * static_cast<std::uint64_t>(n_features) +
                                   kUnitBytes * static_cast<std::uint64_t>(n_units);
    if (declared > body.size())
    {
        return ReplayImageStatus::truncated;
    }
    try
    {
        signals_.resize(n_signals);
        feature_sets_.resize(n_features);
        units_.resize(n_units);
    }
    catch (...)
    {
        return ReplayImageStatus::unreadable;
    }
    for (std::uint32_t i = 0; i < n_signals; ++i)
    {
        const auto base = static_cast<std::size_t>(16 + kSignalBytes * i);
        auto& signal = signals_[i];
        signal.id = load_u32le(body, base);
        signal.clock_domain = load_u32le(body, base + 4);
        signal.n_channels = load_u32le(body, base + 8);
        signal.nominal_block_samples = load_u32le(body, base + 12);
        signal.max_block_samples = load_u32le(body, base + 16);
        signal.dtype = load_u32le(body, base + 20);
        signal.layout = load_u32le(body, base + 24);
        signal.device_tick_tracking = load_u32le(body, base + 28);
        signal.kind = load_u32le(body, base + 32);
        signal.physical_unit = load_u32le(body, base + 36);
        signal.channel_set_id = load_u32le(body, base + 40);
        signal.calibration_id = load_u32le(body, base + 44);
        signal.reference_id = load_u32le(body, base + 48);
        signal.feature_set_id = load_u32le(body, base + 52);
        signal.observation_timing = load_u32le(body, base + 56);
        signal.rate_num = load_u64le(body, base + 64);
        signal.rate_den = load_u64le(body, base + 72);
        signal.fixed_block_bytes = load_u64le(body, base + 80);
        signal.max_block_bytes = load_u64le(body, base + 88);
    }

    const auto feature_base = static_cast<std::size_t>(16 + kSignalBytes * n_signals);
    for (std::uint32_t i = 0; i < n_features; ++i)
    {
        const auto base = feature_base + kFeatureSetBytes * i;
        auto& feature_set = feature_sets_[i];
        feature_set.id = load_u32le(body, base);
        feature_set.source_stream_id = load_u32le(body, base + 4);
        feature_set.source_stream = load_u32le(body, base + 8);
        feature_set.algorithm_name = load_u32le(body, base + 12);
        feature_set.algorithm_version = load_u32le(body, base + 16);
        feature_set.timestamp_reference = load_u32le(body, base + 20);
        feature_set.feature_name_first = load_u32le(body, base + 24);
        feature_set.feature_name_count = load_u32le(body, base + 28);
        feature_set.unit_id_first = load_u32le(body, base + 32);
        feature_set.unit_id_count = load_u32le(body, base + 36);
        feature_set.window_length_ns = load_u64le(body, base + 40);
        feature_set.shift_ns = load_u64le(body, base + 48);
        // A descriptor that names list entries the image does not carry is a
        // self-inconsistent container, not a schema this build merely dislikes.
        const std::uint64_t names_end = static_cast<std::uint64_t>(feature_set.feature_name_first) +
                                        feature_set.feature_name_count;
        const std::uint64_t units_end =
            static_cast<std::uint64_t>(feature_set.unit_id_first) + feature_set.unit_id_count;
        if (names_end > n_lists_ || units_end > n_lists_)
        {
            return ReplayImageStatus::malformed;
        }
    }

    const auto unit_base = feature_base + kFeatureSetBytes * n_features;
    for (std::uint32_t i = 0; i < n_units; ++i)
    {
        const auto base = unit_base + kUnitBytes * i;
        auto& unit = units_[i];
        unit.id = load_u32le(body, base);
        unit.symbol = load_u32le(body, base + 4);
        unit.description = load_u32le(body, base + 8);
    }
    return ReplayImageStatus::ok;
}

std::span<const std::byte> ReplayImageFile::fingerprint() const noexcept
{
    return {fingerprint_, kFingerprintBytes};
}

std::string_view ReplayImageFile::string(std::uint32_t idx) const noexcept
{
    if (idx >= string_offsets_.size())
    {
        return {};
    }
    const auto body = section(kSectionStrings);
    const auto start = static_cast<std::size_t>(string_blob_ + string_offsets_[idx]);
    const auto length = static_cast<std::size_t>(string_lengths_[idx]);
    return {reinterpret_cast<const char*>(body.data() + start), length};
}

std::uint32_t ReplayImageFile::list_at(std::uint64_t idx) const noexcept
{
    if (idx >= n_lists_)
    {
        return kReplayAbsentU32;
    }
    return load_u32le(lists_, static_cast<std::size_t>(4 * idx));
}

ReplayImageItem ReplayImageFile::item(std::size_t idx) const noexcept
{
    ReplayImageItem value{};
    if (idx >= n_items_)
    {
        return value;
    }
    const auto body = section(kSectionItems);
    const auto base = kItemBytes * idx;
    value.kind = load_u8(body, base) == 2 ? ReplayItemKind::discontinuity : ReplayItemKind::frame;
    value.flags = load_u8(body, base + 1);
    value.n_children = load_u32le(body, base + 4);
    value.child_first = load_u64le(body, base + 8);
    value.data_message_ordinal = load_u64le(body, base + 16);
    value.replay_sequence = load_u64le(body, base + 24);
    value.original_sequence = load_u64le(body, base + 32);
    value.previous_replay_sequence = load_u64le(body, base + 40);
    value.original_previous_sequence = load_u64le(body, base + 48);
    value.timeline_ns = load_u64le(body, base + 56);
    value.original_host_received_ns = load_u64le(body, base + 64);
    value.payload_offset = load_u64le(body, base + 72);
    value.payload_byte_count = load_u64le(body, base + 80);
    value.original_payload_byte_count = load_u64le(body, base + 88);
    value.source_tick = load_u64le(body, base + 96);
    value.valid_until_ns = load_u64le(body, base + 104);
    value.native_schema_id = load_u32le(body, base + 112);
    value.source_clock_domain = load_u32le(body, base + 116);
    value.frame_flags = load_u32le(body, base + 120);
    value.reason = load_u32le(body, base + 124);
    return value;
}

ReplayImageBlock ReplayImageFile::block(std::size_t idx) const noexcept
{
    ReplayImageBlock value{};
    if (idx >= n_blocks_)
    {
        return value;
    }
    const auto body = section(kSectionBlocks);
    const auto base = kBlockBytes * idx;
    value.native_signal_id = load_u32le(body, base);
    value.block_idx_in_frame = load_u32le(body, base + 4);
    value.stream = load_u32le(body, base + 8);
    value.n_samples = load_u32le(body, base + 12);
    value.sample_idx_start = load_u64le(body, base + 16);
    value.last_sample_idx = load_u64le(body, base + 24);
    value.device_tick_start = load_u64le(body, base + 32);
    value.observation_time_start_ns = load_u64le(body, base + 40);
    value.payload_offset = load_u64le(body, base + 48);
    value.payload_byte_count = load_u64le(body, base + 56);
    value.original_payload_offset = load_u64le(body, base + 64);
    value.clock_sync_device_tick_reference = load_u64le(body, base + 72);
    value.clock_sync_host_time_reference_ns = load_u64le(body, base + 80);
    value.clock_sync_rate_numerator = load_u64le(body, base + 88);
    value.clock_sync_rate_denominator = load_u64le(body, base + 96);
    value.clock_sync_uncertainty_ns = load_u64le(body, base + 104);
    value.source_block_ordinal = load_u64le(body, base + 112);
    value.recorded_frame_sequence = load_u64le(body, base + 120);
    value.clock_sync_clock_domain = load_u32le(body, base + 128);
    value.clock_sync_generation = load_u32le(body, base + 132);
    value.clock_sync_flags = load_u32le(body, base + 136);
    value.flags = load_u32le(body, base + 140);
    return value;
}

ReplayImageGap ReplayImageFile::gap(std::size_t idx) const noexcept
{
    ReplayImageGap value{};
    if (idx >= n_gaps_)
    {
        return value;
    }
    const auto body = section(kSectionGaps);
    const auto base = kGapBytes * idx;
    value.native_signal_id = load_u32le(body, base);
    value.gap_idx_in_message = load_u32le(body, base + 4);
    value.reason = load_u32le(body, base + 8);
    value.gap_flags = load_u32le(body, base + 12);
    value.expected_sample_idx = load_u64le(body, base + 16);
    value.actual_sample_idx = load_u64le(body, base + 24);
    value.missing_samples = load_u64le(body, base + 32);
    value.expected_device_tick = load_u64le(body, base + 40);
    value.actual_device_tick = load_u64le(body, base + 48);
    value.signal_gap_ordinal = load_u64le(body, base + 56);
    return value;
}

ReplayImageFidelity ReplayImageFile::fidelity(std::size_t idx) const noexcept
{
    ReplayImageFidelity value{};
    if (idx >= n_fidelities_)
    {
        return value;
    }
    const auto body = section(kSectionFidelity);
    const auto base = kFidelityBytes * idx;
    value.stream = load_u32le(body, base);
    value.flags = load_u32le(body, base + 4);
    value.block_idx_column_first = load_u32le(body, base + 8);
    value.block_idx_column_count = load_u32le(body, base + 12);
    value.native_signal_id = load_u64le(body, base + 16);
    value.frames_emitted = load_u64le(body, base + 24);
    value.blocks_emitted = load_u64le(body, base + 32);
    value.discontinuities_emitted = load_u64le(body, base + 40);
    value.committed_blocks = load_u64le(body, base + 48);
    return value;
}

ReplayImageOmission ReplayImageFile::omission(std::size_t idx) const noexcept
{
    ReplayImageOmission value{};
    if (idx >= n_omissions_)
    {
        return value;
    }
    const auto body = section(kSectionOmissions);
    const auto base = kOmissionBytes * idx;
    value.kind = static_cast<ReplayItemKind>(std::to_integer<std::uint8_t>(body[base]));
    value.reason = std::to_integer<std::uint8_t>(body[base + 1]);
    value.stream = load_u32le(body, base + 4);
    value.data_message_ordinal = load_u64le(body, base + 8);
    value.original_frame_sequence = load_u64le(body, base + 16);
    value.source_ordinal = load_u64le(body, base + 24);
    return value;
}

ReplayImageStreamRange ReplayImageFile::stream_range(std::size_t idx) const noexcept
{
    ReplayImageStreamRange value{};
    if (idx >= stream_range_count_)
    {
        return value;
    }
    const auto body = section(kSectionStreamRanges);
    const auto base = kStreamRangeBytes * idx;
    value.stream = load_u32le(body, base);
    value.unit = load_u32le(body, base + 4);
    value.range_start = load_u64le(body, base + 8);
    value.range_stop = load_u64le(body, base + 16);
    value.committed = load_u64le(body, base + 24);
    return value;
}

ReplayImageStatus ReplayImageFile::read_id_registry() noexcept
{
    const auto body = section(kSectionIdRegistry);
    if (body.empty())
    {
        // A ledger-based image synthesizes no ids: its names come from the
        // recording plan. Absent is not malformed.
        return ReplayImageStatus::ok;
    }
    if (body.size() < kIdRegistryHeaderBytes)
    {
        return ReplayImageStatus::truncated;
    }
    std::uint64_t offset = kIdRegistryHeaderBytes;
    for (std::size_t space = 0; space < kReplayIdNamespaceCount; ++space)
    {
        const auto count = load_u32le(body, 4 * space);
        // Bounds are resolved once, here, so every later accessor is an
        // indexed read that cannot walk off the section.
        const auto span_bytes = std::uint64_t{count} * kIdRegistryEntryBytes;
        if (span_bytes > body.size() - offset)
        {
            return ReplayImageStatus::truncated;
        }
        id_registry_counts_[space] = static_cast<std::size_t>(count);
        id_registry_offsets_[space] = offset;
        offset += span_bytes;
    }
    return ReplayImageStatus::ok;
}

std::size_t ReplayImageFile::id_registry_count(ReplayIdNamespace space) const noexcept
{
    return id_registry_counts_[static_cast<std::size_t>(space)];
}

ReplayImageIdEntry ReplayImageFile::id_registry_entry(ReplayIdNamespace space,
                                                      std::size_t idx) const noexcept
{
    ReplayImageIdEntry value{};
    const auto slot = static_cast<std::size_t>(space);
    if (idx >= id_registry_counts_[slot])
    {
        return value;
    }
    const auto body = section(kSectionIdRegistry);
    const auto base = id_registry_offsets_[slot] + kIdRegistryEntryBytes * idx;
    value.name = load_u32le(body, static_cast<std::size_t>(base));
    value.native_id = load_u32le(body, static_cast<std::size_t>(base + 4));
    return value;
}

const ReplayImageSignal& ReplayImageFile::signal(std::size_t idx) const noexcept
{
    static const ReplayImageSignal absent{};
    if (idx >= signals_.size())
    {
        return absent;
    }
    return signals_[idx];
}

const ReplayImageFeatureSet& ReplayImageFile::feature_set(std::size_t idx) const noexcept
{
    static const ReplayImageFeatureSet absent{};
    if (idx >= feature_sets_.size())
    {
        return absent;
    }
    return feature_sets_[idx];
}

const ReplayImageUnit& ReplayImageFile::unit(std::size_t idx) const noexcept
{
    static const ReplayImageUnit absent{};
    if (idx >= units_.size())
    {
        return absent;
    }
    return units_[idx];
}

std::span<const std::byte> ReplayImageFile::payload(std::uint64_t offset,
                                                    std::uint64_t count) const noexcept
{
    const auto entry = sections_[kSectionPayload];
    if (!entry.present || offset > entry.length || count > entry.length - offset)
    {
        return {};
    }
    return data_.subspan(static_cast<std::size_t>(entry.offset + offset),
                         static_cast<std::size_t>(count));
}

} // namespace neurale::recording
