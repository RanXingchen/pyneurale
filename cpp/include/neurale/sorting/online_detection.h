/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include <neurale/sorting/detection.h>

namespace neurale::sorting
{

enum class SpikeBlockOverflowPolicy : std::uint8_t
{
    fault,
    drop_newest,
};

enum class OnlineDetectionStatus : std::uint8_t
{
    ok,
    overflow,
    invalid_input,
    boundary_error,
};

enum class SpikeBlockFlags : std::uint32_t
{
    none = 0,
    overflowed = 1U << 0U,
};

struct SpikeBlockHeader
{
    std::uint32_t format_version{1};
    std::uint32_t n_valid{};
    std::uint32_t capacity{};
    std::uint32_t n_channels{};
    std::uint32_t waveform_samples{};
    std::uint32_t pre_samples{};
    std::uint32_t post_samples{};
    SpikeBlockFlags flags{SpikeBlockFlags::none};
    DetectionPolarity detection_polarity{DetectionPolarity::Negative};
    std::uint8_t reserved[3]{};
    std::uint64_t overflow_count{};
    std::uint64_t segment_id{};
};

static_assert(sizeof(SpikeBlockHeader) == 56);
static_assert(std::is_trivially_copyable_v<SpikeBlockHeader>);

struct SpikeBlockSnapshot;

/// Prepare-time byte layout for one fixed-capacity online spike block.
class SpikeBlockLayout
{
  public:
    SpikeBlockLayout(std::size_t capacity, std::size_t waveform_samples, std::size_t n_channels);

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return capacity_;
    }
    [[nodiscard]] std::size_t waveform_samples() const noexcept
    {
        return waveform_samples_;
    }
    [[nodiscard]] std::size_t channel_count() const noexcept
    {
        return n_channels_;
    }
    [[nodiscard]] std::size_t payload_bytes() const noexcept
    {
        return payload_bytes_;
    }

  private:
    friend class FixedCapacitySpikeBlock;
    friend SpikeBlockSnapshot decode_spike_block(std::span<const std::byte> payload);
    std::size_t capacity_{};
    std::size_t waveform_samples_{};
    std::size_t n_channels_{};
    std::size_t sample_indices_offset_{};
    std::size_t times_offset_{};
    std::size_t peak_channels_offset_{};
    std::size_t group_ids_offset_{};
    std::size_t amps_offset_{};
    std::size_t scores_offset_{};
    std::size_t polarities_offset_{};
    std::size_t waveforms_offset_{};
    std::size_t payload_bytes_{};
};

/// Non-owning writer/reader over emitter-owned fixed-capacity payload storage.
class FixedCapacitySpikeBlock
{
  public:
    FixedCapacitySpikeBlock(std::span<std::byte> payload, const SpikeBlockLayout& layout,
                            std::size_t pre_samples, std::size_t post_samples,
                            std::uint64_t segment_id, DetectionPolarity polarity);

    void clear(std::uint64_t segment_id) noexcept;

    [[nodiscard]] bool append(std::int64_t sample_idx, double time, std::uint32_t peak_channel,
                              std::uint32_t group_id, double amp, double score,
                              std::int8_t polarity, std::int64_t crossing_sample_idx,
                              std::span<const double> waveform) noexcept;
    void note_overflow(std::uint64_t count = 1) noexcept;

    [[nodiscard]] const SpikeBlockHeader& header() const noexcept
    {
        return *header_;
    }
    [[nodiscard]] std::span<const std::int64_t> sample_indices() const noexcept;
    [[nodiscard]] std::span<const double> times() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> peak_channel_indices() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> electrode_group_ids() const noexcept;
    [[nodiscard]] std::span<const double> amplitudes() const noexcept;
    [[nodiscard]] std::span<const double> scores() const noexcept;
    [[nodiscard]] std::span<const std::int8_t> polarities() const noexcept;
    [[nodiscard]] std::span<const double> waveforms() const noexcept;

  private:
    template <typename T> [[nodiscard]] T* at(std::size_t offset) noexcept
    {
        return reinterpret_cast<T*>(payload_.data() + offset);
    }
    template <typename T> [[nodiscard]] const T* at(std::size_t offset) const noexcept
    {
        return reinterpret_cast<const T*>(payload_.data() + offset);
    }

    std::span<std::byte> payload_;
    const SpikeBlockLayout* layout_{};
    SpikeBlockHeader* header_{};
    std::int64_t last_sample_idx_{};
    std::int64_t last_crossing_sample_idx_{};
    std::uint32_t last_group_id_{};
    std::uint32_t last_peak_channel_{};
    bool has_last_sort_key_{};
};

/// Owning control-plane copy used to construct an offline typed batch.
struct SpikeBlockSnapshot
{
    SpikeBlockHeader header;
    std::vector<std::int64_t> sample_indices;
    std::vector<double> times;
    std::vector<std::uint32_t> peak_channel_indices;
    std::vector<std::uint32_t> electrode_group_ids;
    std::vector<double> amps;
    std::vector<double> scores;
    std::vector<std::int8_t> polarities;
    std::vector<double> waveforms;
};

/// Decode and copy a fixed block outside the realtime callback.
[[nodiscard]] SpikeBlockSnapshot decode_spike_block(std::span<const std::byte> payload);

struct OnlineThresholdDetectorConfig
{
    std::size_t max_input_samples{};
    std::size_t block_capacity{};
    std::size_t refractory_samples{};
    std::size_t alignment_search_radius{};
    std::size_t pre_samples{};
    std::size_t post_samples{};
    DetectionPolarity polarity{DetectionPolarity::Negative};
    BoundaryBehavior boundary_behavior{BoundaryBehavior::Drop};
    SpikeBlockOverflowPolicy overflow_policy{SpikeBlockOverflowPolicy::fault};
    std::vector<double> channel_centers;
    std::vector<double> channel_thresholds;
    std::vector<std::vector<std::size_t>> electrode_groups;
};

/// Stateful allocation-free-after-construction online threshold detector.
class OnlineThresholdDetector
{
  public:
    explicit OnlineThresholdDetector(OnlineThresholdDetectorConfig config);
    ~OnlineThresholdDetector();

    OnlineThresholdDetector(const OnlineThresholdDetector&) = delete;
    OnlineThresholdDetector& operator=(const OnlineThresholdDetector&) = delete;

    [[nodiscard]] OnlineDetectionStatus process(std::span<const double> sample_major,
                                                std::size_t n_samples,
                                                std::int64_t sample_idx_start, double time_start,
                                                double fs, std::uint64_t segment_id,
                                                FixedCapacitySpikeBlock& output) noexcept;
    void reset(std::uint64_t segment_id = 0) noexcept;

    /// Drain completed watermark-retained events in global order. Incomplete
    /// waveform tails are dropped or reported according to boundary behavior.
    [[nodiscard]] OnlineDetectionStatus finish(FixedCapacitySpikeBlock& output) noexcept;

    /// Query only the incomplete-tail boundary status. This overload does not
    /// mutate detector state and is used when no completed event needs drain.
    [[nodiscard]] OnlineDetectionStatus finish() const noexcept;

    [[nodiscard]] std::size_t ready_count() const noexcept;

    [[nodiscard]] std::size_t channel_count() const noexcept;
    [[nodiscard]] std::size_t retained_sample_capacity() const noexcept;
    [[nodiscard]] std::size_t pending_capacity() const noexcept;
    /// Total bytes of fixed workspace reserved at construction: the input
    /// buffer, pending/complete event buffers, waveform scratch, per-channel
    /// and per-group state, and the owned channel/group configuration vectors.
    /// Steady-state processing allocates nothing beyond this reservation.
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;

  private:
    struct Impl;
    Impl* impl_{};
};

} // namespace neurale::sorting
