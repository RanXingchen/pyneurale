/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The native reader for the private replay-image container, version 1.
///
/// The image is built by `neurale.recording._replay_build` from a committed
/// NRF session and is the *only* thing a native replay source consumes. That
/// is the point of the format: the contract
/// (`docs/development/native_recording_replay.md`, section 8) forbids JSON or
/// Zarr parsing, filesystem metadata work, logging, and dynamic allocation in
/// a steady-state replay read, and none of those can be done cheaply enough to
/// be talked out of. So everything expensive happens here, at open: the file is
/// mapped once, the container is validated once, and afterwards a record is a
/// fixed-width little-endian structure at a computed offset.
///
/// Validation at open is of the *container*, never of the semantics: magic,
/// header checksum, format version, declared length, section table bounds,
/// duplicate sections, whole-record fixed-width sections, the content
/// checksum, and the internal consistency of the string and list tables. An
/// image that fails any of those is refused whole. There is no partial read and
/// no salvage: the session, not the image, is the source of truth, and the
/// Python cache path answers a bad image by rebuilding it.
///
/// Fields are read one byte at a time through `byte_order.h`, so the reader
/// does not depend on host byte order or on any structure padding a compiler
/// might choose. The decoded structs below are the reader's own; they are not
/// the on-disk layout and must not be memcpy'd from the mapping.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace neurale::recording
{

/// Why an image could not be opened. Every value is terminal for that file.
enum class ReplayImageStatus : std::uint8_t
{
    ok,
    /// The path could not be opened, sized, or mapped.
    unreadable,
    /// The file does not begin with the replay-image magic.
    not_an_image,
    /// The header or the content does not match its own checksum.
    checksum_mismatch,
    /// A format version this build does not read.
    unsupported_version,
    /// The file is shorter than the length it declares, or a section is.
    truncated,
    /// The container is self-inconsistent: overlapping, duplicated, or
    /// partial-record sections, or a table that names what it does not carry.
    malformed,
};

[[nodiscard]] const char* replay_image_status_text(ReplayImageStatus status) noexcept;

/// The two data-message kinds of contract section 1.1.
enum class ReplayItemKind : std::uint8_t
{
    frame = 1,
    discontinuity = 2,
};

inline constexpr std::uint32_t kReplayAbsentU32 = 0xFFFFFFFFU;
inline constexpr std::uint64_t kReplayAbsentU64 = 0xFFFFFFFFFFFFFFFFULL;

/// Item flag bits, mirroring `_replay_image.ITEM_FLAG_*`.
inline constexpr std::uint8_t kReplayItemFlagSourceTick = 1U << 0U;
inline constexpr std::uint8_t kReplayItemFlagValidUntil = 1U << 1U;
inline constexpr std::uint8_t kReplayItemFlagFrameLevelGap = 1U << 2U;

/// Block flag bits, mirroring `_replay_image.BLOCK_FLAG_*`.
inline constexpr std::uint32_t kReplayBlockFlagClockSync = 1U << 0U;

/// Summary flag bits, mirroring `_replay_image.FLAG_*`.
inline constexpr std::uint32_t kReplayFlagAllowIncomplete = 1U << 0U;
inline constexpr std::uint32_t kReplayFlagAbnormalEndRequired = 1U << 1U;
inline constexpr std::uint32_t kReplayFlagLedgerBased = 1U << 2U;
inline constexpr std::uint32_t kReplayFlagClockSyncAvailable = 1U << 3U;
inline constexpr std::uint32_t kReplayFlagDescriptorMetadataAvailable = 1U << 4U;
inline constexpr std::uint32_t kReplayFlagRateFromRecordingPlan = 1U << 5U;

/// One emitted data message, decoded. Absent fields keep their absent marker.
struct ReplayImageItem
{
    ReplayItemKind kind{ReplayItemKind::frame};
    std::uint8_t flags{};
    std::uint32_t n_children{};
    std::uint64_t child_first{};
    std::uint64_t data_message_ordinal{kReplayAbsentU64};
    std::uint64_t replay_sequence{};
    std::uint64_t original_sequence{kReplayAbsentU64};
    std::uint64_t previous_replay_sequence{kReplayAbsentU64};
    std::uint64_t original_previous_sequence{kReplayAbsentU64};
    /// The recorded-timeline position pacing follows (contract section 8.7).
    std::uint64_t timeline_ns{};
    std::uint64_t original_host_received_ns{kReplayAbsentU64};
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    std::uint64_t original_payload_byte_count{kReplayAbsentU64};
    std::uint64_t source_tick{};
    std::uint64_t valid_until_ns{};
    std::uint32_t native_schema_id{};
    std::uint32_t source_clock_domain{};
    std::uint32_t frame_flags{};
    std::uint32_t reason{kReplayAbsentU32};
};

/// One emitted signal block of one frame.
struct ReplayImageBlock
{
    std::uint32_t native_signal_id{};
    std::uint32_t block_idx_in_frame{};
    std::uint32_t stream{kReplayAbsentU32};
    std::uint32_t n_samples{};
    std::uint64_t sample_idx_start{};
    std::uint64_t last_sample_idx{};
    std::uint64_t device_tick_start{};
    std::uint64_t observation_time_start_ns{};
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    std::uint64_t original_payload_offset{kReplayAbsentU64};
    std::uint64_t clock_sync_device_tick_reference{};
    std::uint64_t clock_sync_host_time_reference_ns{};
    std::uint64_t clock_sync_rate_numerator{};
    std::uint64_t clock_sync_rate_denominator{};
    std::uint64_t clock_sync_uncertainty_ns{};
    std::uint64_t source_block_ordinal{kReplayAbsentU64};
    std::uint64_t recorded_frame_sequence{kReplayAbsentU64};
    std::uint32_t clock_sync_clock_domain{};
    std::uint32_t clock_sync_generation{};
    std::uint32_t clock_sync_flags{};
    std::uint32_t flags{};
};

/// One per-signal gap carried by an emitted discontinuity.
struct ReplayImageGap
{
    std::uint32_t native_signal_id{};
    std::uint32_t gap_idx_in_message{};
    std::uint32_t reason{kReplayAbsentU32};
    std::uint32_t gap_flags{};
    std::uint64_t expected_sample_idx{};
    std::uint64_t actual_sample_idx{};
    std::uint64_t missing_samples{kReplayAbsentU64};
    std::uint64_t expected_device_tick{kReplayAbsentU64};
    std::uint64_t actual_device_tick{kReplayAbsentU64};
    /// The recorded identity a fault positions against in `stream_frames`:
    /// the discontinuity record's own ordinal within its stream.
    std::uint64_t signal_gap_ordinal{kReplayAbsentU64};
};

/// What one selected stream was reconstructed from (contract section 8.3).
struct ReplayImageFidelity
{
    std::uint32_t stream{kReplayAbsentU32};
    std::uint32_t flags{};
    /// The block-index columns this stream carried, as a run in the list table.
    /// A replay read never looks at these; they exist so the reporting path can
    /// say what a stream was reconstructed from without a second reader.
    std::uint32_t block_idx_column_first{};
    std::uint32_t block_idx_column_count{};
    std::uint64_t native_signal_id{};
    std::uint64_t frames_emitted{};
    std::uint64_t blocks_emitted{};
    std::uint64_t discontinuities_emitted{};
    std::uint64_t committed_blocks{};
};

/// One source message the projection could not emit (contract sections 8.2,
/// 8.3). Nothing on the replay read path consumes an omission: it is reported,
/// which is why it is decoded here rather than restated by a second reader.
struct ReplayImageOmission
{
    ReplayItemKind kind{ReplayItemKind::frame};
    std::uint8_t reason{};
    std::uint32_t stream{kReplayAbsentU32};
    std::uint64_t data_message_ordinal{kReplayAbsentU64};
    std::uint64_t original_frame_sequence{kReplayAbsentU64};
    std::uint64_t source_ordinal{kReplayAbsentU64};
};

/// The per-stream bound one stream-ranged request resolved to (section 8.3).
struct ReplayImageStreamRange
{
    std::uint32_t stream{kReplayAbsentU32};
    std::uint32_t unit{kReplayAbsentU32};
    std::uint64_t range_start{};
    std::uint64_t range_stop{};
    std::uint64_t committed{};
};

/// One `name -> native id` entry of a synthesized registry namespace.
struct ReplayImageIdEntry
{
    std::uint32_t name{kReplayAbsentU32};
    std::uint32_t native_id{};
};

/// The registry namespaces, in the order the section stores their counts.
enum class ReplayIdNamespace : std::uint8_t
{
    signals = 0,
    clocks = 1,
    feature_sets = 2,
    units = 3,
};

inline constexpr std::size_t kReplayIdNamespaceCount = 4;

/// The run-level summary. String fields are string-table indices.
struct ReplayImageSummary
{
    std::uint32_t mode{kReplayAbsentU32};
    std::uint32_t plan_coverage{kReplayAbsentU32};
    std::uint32_t completeness{kReplayAbsentU32};
    std::uint32_t source_session_id{kReplayAbsentU32};
    std::uint32_t plan_fingerprint{kReplayAbsentU32};
    std::uint32_t source_fingerprint{kReplayAbsentU32};
    std::uint32_t flags{};
    std::uint32_t frame_construction{kReplayAbsentU32};
    std::uint32_t ordering_key{kReplayAbsentU32};
    std::uint32_t selected_list{};
    std::uint32_t n_selected{};
    /// The planned and recorded signal-id sets, as runs in the list table. The
    /// replay read path has no use for either; the reporting path does, and a
    /// field decoded in one place is a field that cannot disagree with itself.
    std::uint32_t planned_list{};
    std::uint32_t n_planned{};
    std::uint32_t recorded_list{};
    std::uint32_t n_recorded{};
    std::uint64_t range_start{};
    std::uint64_t range_stop{};
    std::uint64_t n_items{};
    std::uint64_t n_frames{};
    std::uint64_t n_discontinuities{};
    std::uint64_t n_blocks{};
    std::uint64_t n_gaps{};
    std::uint64_t n_omissions{};
    std::uint64_t payload_byte_count{};
    std::uint64_t native_session_id{kReplayAbsentU64};
    std::uint64_t source_message_count{};
    std::uint64_t first_timeline_ns{kReplayAbsentU64};
    std::uint64_t last_timeline_ns{kReplayAbsentU64};

    [[nodiscard]] bool ledger_based() const noexcept
    {
        return (flags & kReplayFlagLedgerBased) != 0U;
    }
    [[nodiscard]] bool abnormal_end_required() const noexcept
    {
        return (flags & kReplayFlagAbnormalEndRequired) != 0U;
    }
};

/// One signal of the run's declared schema, decoded from the schema section.
struct ReplayImageSignal
{
    std::uint32_t id{};
    std::uint32_t clock_domain{};
    std::uint32_t n_channels{};
    std::uint32_t nominal_block_samples{};
    std::uint32_t max_block_samples{};
    std::uint32_t dtype{kReplayAbsentU32};
    std::uint32_t layout{kReplayAbsentU32};
    std::uint32_t device_tick_tracking{kReplayAbsentU32};
    std::uint32_t kind{kReplayAbsentU32};
    std::uint32_t physical_unit{kReplayAbsentU32};
    std::uint64_t channel_set_id{};
    std::uint64_t calibration_id{};
    std::uint64_t reference_id{};
    std::uint64_t feature_set_id{};
    std::uint32_t observation_timing{kReplayAbsentU32};
    std::uint64_t rate_num{};
    std::uint64_t rate_den{};
    std::uint64_t fixed_block_bytes{};
    std::uint64_t max_block_bytes{};
};

/// One feature-set descriptor of the run's declared schema. The string and list
/// fields are table indices, resolved through `string()` and `list_at()`.
struct ReplayImageFeatureSet
{
    std::uint32_t id{};
    std::uint32_t source_stream_id{};
    std::uint32_t source_stream{kReplayAbsentU32};
    std::uint32_t algorithm_name{kReplayAbsentU32};
    std::uint32_t algorithm_version{kReplayAbsentU32};
    std::uint32_t timestamp_reference{kReplayAbsentU32};
    std::uint32_t feature_name_first{};
    std::uint32_t feature_name_count{};
    std::uint32_t unit_id_first{};
    std::uint32_t unit_id_count{};
    std::uint64_t window_length_ns{};
    std::uint64_t shift_ns{};
};

/// One unit descriptor of the run's declared schema.
struct ReplayImageUnit
{
    std::uint32_t id{};
    std::uint32_t symbol{kReplayAbsentU32};
    std::uint32_t description{kReplayAbsentU32};
};

/// A validated, read-only view of one replay image.
class ReplayImageFile
{
  public:
    ReplayImageFile() noexcept = default;
    ~ReplayImageFile();

    ReplayImageFile(const ReplayImageFile&) = delete;
    ReplayImageFile& operator=(const ReplayImageFile&) = delete;
    ReplayImageFile(ReplayImageFile&& other) noexcept;
    ReplayImageFile& operator=(ReplayImageFile&& other) noexcept;

    /// Map and validate *path*. On any failure the object stays closed.
    [[nodiscard]] ReplayImageStatus open(const char* path) noexcept;
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept
    {
        return !data_.empty();
    }

    /// The 32-byte source-derived fingerprint (contract section 8.12).
    [[nodiscard]] std::span<const std::byte> fingerprint() const noexcept;

    [[nodiscard]] const ReplayImageSummary& summary() const noexcept
    {
        return summary_;
    }

    [[nodiscard]] std::size_t item_count() const noexcept
    {
        return n_items_;
    }
    [[nodiscard]] std::size_t block_count() const noexcept
    {
        return n_blocks_;
    }
    [[nodiscard]] std::size_t gap_count() const noexcept
    {
        return n_gaps_;
    }
    [[nodiscard]] std::size_t fidelity_count() const noexcept
    {
        return n_fidelities_;
    }
    [[nodiscard]] std::size_t omission_count() const noexcept
    {
        return n_omissions_;
    }
    [[nodiscard]] std::size_t stream_range_count() const noexcept
    {
        return stream_range_count_;
    }
    [[nodiscard]] std::size_t signal_count() const noexcept
    {
        return signals_.size();
    }
    [[nodiscard]] std::size_t feature_set_count() const noexcept
    {
        return feature_sets_.size();
    }
    [[nodiscard]] std::size_t unit_count() const noexcept
    {
        return units_.size();
    }
    [[nodiscard]] std::uint32_t schema_id() const noexcept
    {
        return schema_id_;
    }

    /// Decode one record. The index must be below the matching count; an index
    /// past the end yields a value-initialized record rather than a read out
    /// of the mapping.
    [[nodiscard]] ReplayImageItem item(std::size_t idx) const noexcept;
    [[nodiscard]] ReplayImageBlock block(std::size_t idx) const noexcept;
    [[nodiscard]] ReplayImageGap gap(std::size_t idx) const noexcept;
    [[nodiscard]] ReplayImageFidelity fidelity(std::size_t idx) const noexcept;
    [[nodiscard]] ReplayImageOmission omission(std::size_t idx) const noexcept;
    [[nodiscard]] ReplayImageStreamRange stream_range(std::size_t idx) const noexcept;
    [[nodiscard]] const ReplayImageSignal& signal(std::size_t idx) const noexcept;
    [[nodiscard]] const ReplayImageFeatureSet& feature_set(std::size_t idx) const noexcept;
    [[nodiscard]] const ReplayImageUnit& unit(std::size_t idx) const noexcept;

    /// The payload bytes of one frame, inside the mapping. An out-of-range
    /// span yields an empty result, which every caller must treat as an error.
    [[nodiscard]] std::span<const std::byte> payload(std::uint64_t offset,
                                                     std::uint64_t count) const noexcept;

    /// One interned string. The absent marker and any out-of-range index yield
    /// an empty view; validity of every index was established at open.
    [[nodiscard]] std::string_view string(std::uint32_t idx) const noexcept;

    [[nodiscard]] std::size_t string_count() const noexcept
    {
        return string_offsets_.size();
    }

    /// One entry of the flat uint32 list table. Out of range yields the absent
    /// marker; every list an image references was bounds-checked at open.
    [[nodiscard]] std::uint32_t list_at(std::uint64_t idx) const noexcept;

    [[nodiscard]] std::size_t list_count() const noexcept
    {
        return n_lists_;
    }

    /// How many entries one registry namespace holds. Zero for every namespace
    /// of a ledger-based image, whose ids come from the recording plan.
    [[nodiscard]] std::size_t id_registry_count(ReplayIdNamespace space) const noexcept;

    /// One entry of one namespace. An out-of-range index yields an entry whose
    /// name is the absent marker, which every caller must treat as an error.
    [[nodiscard]] ReplayImageIdEntry id_registry_entry(ReplayIdNamespace space,
                                                       std::size_t idx) const noexcept;

  private:
    [[nodiscard]] ReplayImageStatus map(const char* path) noexcept;
    [[nodiscard]] ReplayImageStatus validate() noexcept;
    [[nodiscard]] ReplayImageStatus read_strings() noexcept;
    [[nodiscard]] ReplayImageStatus read_summary() noexcept;
    [[nodiscard]] ReplayImageStatus read_schema() noexcept;
    [[nodiscard]] ReplayImageStatus read_id_registry() noexcept;
    [[nodiscard]] std::span<const std::byte> section(std::uint32_t kind) const noexcept;

    struct Section
    {
        std::uint64_t offset{};
        std::uint64_t length{};
        bool present{};
    };

    /// Owned mapping. `data_` is the whole file; `handle_` is the platform
    /// resource that keeps it alive.
    std::span<const std::byte> data_{};
    void* mapping_{};
    std::size_t mapping_bytes_{};
    std::vector<std::byte> buffer_;

    static constexpr std::size_t kSectionSlots = 13;
    Section sections_[kSectionSlots]{};

    std::vector<std::uint32_t> string_offsets_;
    std::vector<std::uint32_t> string_lengths_;
    std::uint64_t string_blob_{};
    std::span<const std::byte> lists_{};
    std::size_t n_lists_{};

    ReplayImageSummary summary_{};
    std::vector<ReplayImageSignal> signals_;
    std::vector<ReplayImageFeatureSet> feature_sets_;
    std::vector<ReplayImageUnit> units_;
    std::uint32_t schema_id_{};
    std::size_t n_items_{};
    std::size_t n_blocks_{};
    std::size_t n_gaps_{};
    std::size_t n_fidelities_{};
    std::size_t n_omissions_{};
    std::size_t stream_range_count_{};

    /// Per-namespace entry counts and where each namespace's entries begin,
    /// both resolved at open so an accessor is an indexed read.
    std::array<std::size_t, kReplayIdNamespaceCount> id_registry_counts_{};
    std::array<std::uint64_t, kReplayIdNamespaceCount> id_registry_offsets_{};
    std::byte fingerprint_[32]{};
};

} // namespace neurale::recording
