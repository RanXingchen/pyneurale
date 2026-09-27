/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/sorting/online_detection.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace neurale::sorting
{
namespace
{

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::length_error("online spike block size is too large");
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::length_error("online spike block size is too large");
    }
    return left * right;
}

[[nodiscard]] std::size_t aligned_offset(std::size_t offset, std::size_t alignment)
{
    return checked_add(offset, alignment - 1) / alignment * alignment;
}

template <typename T>
[[nodiscard]] std::size_t append_array(std::size_t offset, std::size_t count, std::size_t& result)
{
    const auto aligned = aligned_offset(offset, alignof(T));
    result = aligned;
    return checked_add(aligned, checked_multiply(count, sizeof(T)));
}

[[nodiscard]] bool active(double value, double threshold, DetectionPolarity polarity) noexcept
{
    switch (polarity)
    {
    case DetectionPolarity::Negative:
        return value <= -threshold;
    case DetectionPolarity::Positive:
        return value >= threshold;
    case DetectionPolarity::Both:
        return std::abs(value) >= threshold;
    }
    return false;
}

[[nodiscard]] double rank(double value, DetectionPolarity polarity) noexcept
{
    switch (polarity)
    {
    case DetectionPolarity::Negative:
        return -value;
    case DetectionPolarity::Positive:
        return value;
    case DetectionPolarity::Both:
        return std::abs(value);
    }
    return 0.0;
}

} // namespace

SpikeBlockLayout::SpikeBlockLayout(std::size_t capacity, std::size_t waveform_samples,
                                   std::size_t n_channels)
    : capacity_(capacity), waveform_samples_(waveform_samples), n_channels_(n_channels)
{
    if (capacity == 0 || waveform_samples == 0 || n_channels == 0 ||
        capacity > std::numeric_limits<std::uint32_t>::max() ||
        waveform_samples > std::numeric_limits<std::uint32_t>::max() ||
        n_channels > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument("online spike block geometry must be positive and fit uint32");
    }
    auto offset = sizeof(SpikeBlockHeader);
    offset = append_array<std::int64_t>(offset, capacity, sample_indices_offset_);
    offset = append_array<double>(offset, capacity, times_offset_);
    offset = append_array<std::uint32_t>(offset, capacity, peak_channels_offset_);
    offset = append_array<std::uint32_t>(offset, capacity, group_ids_offset_);
    offset = append_array<double>(offset, capacity, amps_offset_);
    offset = append_array<double>(offset, capacity, scores_offset_);
    offset = append_array<std::int8_t>(offset, capacity, polarities_offset_);
    offset = append_array<double>(
        offset, checked_multiply(capacity, checked_multiply(waveform_samples, n_channels)),
        waveforms_offset_);
    payload_bytes_ = offset;
}

FixedCapacitySpikeBlock::FixedCapacitySpikeBlock(std::span<std::byte> payload,
                                                 const SpikeBlockLayout& layout,
                                                 std::size_t pre_samples, std::size_t post_samples,
                                                 std::uint64_t segment_id,
                                                 DetectionPolarity polarity)
    : payload_(payload), layout_(&layout)
{
    if (payload.size() != layout.payload_bytes() ||
        reinterpret_cast<std::uintptr_t>(payload.data()) % alignof(SpikeBlockHeader) != 0 ||
        checked_add(checked_add(pre_samples, 1), post_samples) != layout.waveform_samples() ||
        pre_samples > std::numeric_limits<std::uint32_t>::max() ||
        post_samples > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument("online spike payload does not match its prepared layout");
    }
    header_ = at<SpikeBlockHeader>(0);
    std::memset(payload_.data(), 0, payload_.size());
    header_->format_version = 1;
    header_->capacity = static_cast<std::uint32_t>(layout.capacity());
    header_->n_channels = static_cast<std::uint32_t>(layout.channel_count());
    header_->waveform_samples = static_cast<std::uint32_t>(layout.waveform_samples());
    header_->pre_samples = static_cast<std::uint32_t>(pre_samples);
    header_->post_samples = static_cast<std::uint32_t>(post_samples);
    header_->segment_id = segment_id;
    header_->detection_polarity = polarity;
}

void FixedCapacitySpikeBlock::clear(std::uint64_t segment_id) noexcept
{
    header_->n_valid = 0;
    header_->flags = SpikeBlockFlags::none;
    header_->overflow_count = 0;
    header_->segment_id = segment_id;
    last_sample_idx_ = 0;
    last_crossing_sample_idx_ = 0;
    last_group_id_ = 0;
    last_peak_channel_ = 0;
    has_last_sort_key_ = false;
}

bool FixedCapacitySpikeBlock::append(std::int64_t sample_idx, double time,
                                     std::uint32_t peak_channel, std::uint32_t group_id, double amp,
                                     double score, std::int8_t polarity,
                                     std::int64_t crossing_sample_idx,
                                     std::span<const double> waveform) noexcept
{
    const auto idx = static_cast<std::size_t>(header_->n_valid);
    const auto waveform_values = layout_->waveform_samples() * layout_->channel_count();
    const bool key_regressed =
        has_last_sort_key_ &&
        (sample_idx < last_sample_idx_ ||
         (sample_idx == last_sample_idx_ &&
          (group_id < last_group_id_ ||
           (group_id == last_group_id_ && (peak_channel < last_peak_channel_ ||
                                           (peak_channel == last_peak_channel_ &&
                                            crossing_sample_idx < last_crossing_sample_idx_))))));
    if (idx >= layout_->capacity() || waveform.size() != waveform_values ||
        peak_channel >= layout_->channel_count() || key_regressed)
    {
        return false;
    }
    at<std::int64_t>(layout_->sample_indices_offset_)[idx] = sample_idx;
    at<double>(layout_->times_offset_)[idx] = time;
    at<std::uint32_t>(layout_->peak_channels_offset_)[idx] = peak_channel;
    at<std::uint32_t>(layout_->group_ids_offset_)[idx] = group_id;
    at<double>(layout_->amps_offset_)[idx] = amp;
    at<double>(layout_->scores_offset_)[idx] = score;
    at<std::int8_t>(layout_->polarities_offset_)[idx] = polarity;
    std::copy(waveform.begin(), waveform.end(),
              at<double>(layout_->waveforms_offset_) + idx * waveform_values);
    ++header_->n_valid;
    last_sample_idx_ = sample_idx;
    last_crossing_sample_idx_ = crossing_sample_idx;
    last_group_id_ = group_id;
    last_peak_channel_ = peak_channel;
    has_last_sort_key_ = true;
    return true;
}

void FixedCapacitySpikeBlock::note_overflow(std::uint64_t count) noexcept
{
    if (count == 0)
        return;
    header_->flags = SpikeBlockFlags::overflowed;
    header_->overflow_count =
        count > std::numeric_limits<std::uint64_t>::max() - header_->overflow_count
            ? std::numeric_limits<std::uint64_t>::max()
            : header_->overflow_count + count;
}

#define NEURALE_BLOCK_SPAN(name, type, offset_name)                                                \
    std::span<const type> FixedCapacitySpikeBlock::name() const noexcept                           \
    {                                                                                              \
        return {at<type>(layout_->offset_name), header_->n_valid};                                 \
    }

NEURALE_BLOCK_SPAN(sample_indices, std::int64_t, sample_indices_offset_)
NEURALE_BLOCK_SPAN(times, double, times_offset_)
NEURALE_BLOCK_SPAN(peak_channel_indices, std::uint32_t, peak_channels_offset_)
NEURALE_BLOCK_SPAN(electrode_group_ids, std::uint32_t, group_ids_offset_)
NEURALE_BLOCK_SPAN(amplitudes, double, amps_offset_)
NEURALE_BLOCK_SPAN(scores, double, scores_offset_)
NEURALE_BLOCK_SPAN(polarities, std::int8_t, polarities_offset_)

std::span<const double> FixedCapacitySpikeBlock::waveforms() const noexcept
{
    const auto count = static_cast<std::size_t>(header_->n_valid) * layout_->waveform_samples() *
                       layout_->channel_count();
    return {at<double>(layout_->waveforms_offset_), count};
}

SpikeBlockSnapshot decode_spike_block(std::span<const std::byte> payload)
{
    SpikeBlockHeader header{};
    if (payload.size() < sizeof(header))
        throw std::invalid_argument("online spike payload is shorter than its header");
    std::memcpy(&header, payload.data(), sizeof(header));
    if (header.format_version != 1 || header.capacity == 0 || header.n_channels == 0 ||
        header.waveform_samples == 0 || header.n_valid > header.capacity ||
        static_cast<std::uint64_t>(header.pre_samples) + 1U + header.post_samples !=
            header.waveform_samples ||
        (header.detection_polarity != DetectionPolarity::Negative &&
         header.detection_polarity != DetectionPolarity::Positive &&
         header.detection_polarity != DetectionPolarity::Both))
        throw std::invalid_argument("online spike payload header is invalid");
    const SpikeBlockLayout layout{header.capacity, header.waveform_samples, header.n_channels};
    if (payload.size() != layout.payload_bytes())
        throw std::invalid_argument("online spike payload size does not match its header");

    auto copy = [&](auto tag, std::size_t offset, std::size_t count)
    {
        using T = decltype(tag);
        std::vector<T> values(count);
        if (count != 0)
            std::memcpy(values.data(), payload.data() + offset, count * sizeof(T));
        return values;
    };
    SpikeBlockSnapshot result;
    result.header = header;
    const auto count = static_cast<std::size_t>(header.n_valid);
    result.sample_indices = copy(std::int64_t{}, layout.sample_indices_offset_, count);
    result.times = copy(double{}, layout.times_offset_, count);
    result.peak_channel_indices = copy(std::uint32_t{}, layout.peak_channels_offset_, count);
    result.electrode_group_ids = copy(std::uint32_t{}, layout.group_ids_offset_, count);
    result.amps = copy(double{}, layout.amps_offset_, count);
    result.scores = copy(double{}, layout.scores_offset_, count);
    result.polarities = copy(std::int8_t{}, layout.polarities_offset_, count);
    result.waveforms = copy(double{}, layout.waveforms_offset_,
                            count * header.waveform_samples * header.n_channels);
    return result;
}

#undef NEURALE_BLOCK_SPAN

struct OnlineThresholdDetector::Impl
{
    struct Pending
    {
        std::int64_t crossing{};
        std::uint32_t channel{};
        std::uint32_t group{};
    };
    struct Complete
    {
        std::int64_t peak{};
        std::int64_t crossing{};
        std::uint32_t channel{};
        std::uint32_t group{};
        double amp{};
        double score{};
        std::int8_t polarity{};
    };

    explicit Impl(OnlineThresholdDetectorConfig value) : config(std::move(value)) {}

    [[nodiscard]] double value(std::int64_t sample, std::size_t channel) const noexcept
    {
        return buffer[(static_cast<std::size_t>(sample - buffer_start) * channels) + channel];
    }

    OnlineThresholdDetectorConfig config;
    std::size_t channels{};
    std::size_t retention{};
    std::vector<std::size_t> channel_groups;
    std::vector<bool> previous_active;
    std::vector<bool> has_last_crossing;
    std::vector<std::int64_t> last_crossing;
    std::vector<double> buffer;
    std::size_t retained_samples{};
    std::int64_t buffer_start{};
    std::int64_t expected_next{};
    std::int64_t segment_start{};
    double fs{};
    std::uint64_t segment_id{};
    bool anchored{};
    std::vector<Pending> pending;
    std::vector<Complete> complete;
    std::vector<double> waveform_scratch;
    std::vector<Complete> ready_events;
    std::vector<double> ready_waveforms;
    std::vector<std::size_t> ready_order;
    std::vector<std::size_t> next_ready_order;
    std::vector<std::size_t> free_ready_slots;
    std::size_t free_ready_count{};
    double last_time_start{};
    std::int64_t last_sample_idx_start{};
};

OnlineThresholdDetector::OnlineThresholdDetector(OnlineThresholdDetectorConfig config)
    : impl_(new Impl(std::move(config)))
{
    auto& state = *impl_;
    state.channels = state.config.channel_centers.size();
    if (state.config.max_input_samples == 0 || state.config.block_capacity == 0 ||
        state.channels == 0 || state.config.channel_thresholds.size() != state.channels)
    {
        throw std::invalid_argument("invalid online threshold detector geometry");
    }
    const auto max_sample_geometry =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (state.config.refractory_samples > max_sample_geometry ||
        state.config.alignment_search_radius > max_sample_geometry ||
        state.config.pre_samples > max_sample_geometry ||
        state.config.post_samples > max_sample_geometry ||
        state.config.alignment_search_radius > max_sample_geometry - state.config.post_samples)
    {
        throw std::invalid_argument("online detector sample geometry exceeds int64 range");
    }
    for (std::size_t channel = 0; channel < state.channels; ++channel)
    {
        if (!std::isfinite(state.config.channel_centers[channel]) ||
            !std::isfinite(state.config.channel_thresholds[channel]) ||
            state.config.channel_thresholds[channel] < 0.0)
        {
            throw std::invalid_argument("online channel centers and thresholds must be finite");
        }
    }
    if (state.config.electrode_groups.empty())
    {
        state.config.electrode_groups.resize(state.channels);
        for (std::size_t channel = 0; channel < state.channels; ++channel)
        {
            state.config.electrode_groups[channel].push_back(channel);
        }
    }
    state.channel_groups.assign(state.channels, std::numeric_limits<std::size_t>::max());
    for (std::size_t group = 0; group < state.config.electrode_groups.size(); ++group)
    {
        if (state.config.electrode_groups[group].empty())
        {
            throw std::invalid_argument("online electrode groups must not be empty");
        }
        for (const auto channel : state.config.electrode_groups[group])
        {
            if (channel >= state.channels ||
                state.channel_groups[channel] != std::numeric_limits<std::size_t>::max())
            {
                throw std::invalid_argument("online electrode groups must partition channels");
            }
            state.channel_groups[channel] = group;
        }
    }
    if (std::find(state.channel_groups.begin(), state.channel_groups.end(),
                  std::numeric_limits<std::size_t>::max()) != state.channel_groups.end())
    {
        throw std::invalid_argument("online electrode groups must cover every channel");
    }
    state.retention =
        checked_add(checked_add(checked_multiply(2, state.config.alignment_search_radius),
                                state.config.pre_samples),
                    checked_add(state.config.post_samples, 1));
    const auto sample_capacity = checked_add(state.retention, state.config.max_input_samples);
    state.buffer.resize(checked_multiply(sample_capacity, state.channels));
    state.previous_active.resize(state.channels);
    const auto n_groups = state.config.electrode_groups.size();
    state.has_last_crossing.resize(n_groups);
    state.last_crossing.resize(n_groups);
    const auto pending_capacity = checked_multiply(
        n_groups,
        checked_add(state.config.max_input_samples,
                    checked_add(state.config.alignment_search_radius, state.config.post_samples)));
    state.pending.reserve(std::max<std::size_t>(pending_capacity, n_groups));
    state.complete.reserve(std::max<std::size_t>(pending_capacity, n_groups));
    const auto waveform_values = checked_multiply(
        checked_add(checked_add(state.config.pre_samples, 1), state.config.post_samples),
        state.channels);
    state.waveform_scratch.resize(waveform_values);
    const auto ready_capacity = checked_multiply(
        n_groups, checked_add(checked_multiply(2, state.config.alignment_search_radius), 2));
    state.ready_events.resize(ready_capacity);
    state.ready_waveforms.resize(checked_multiply(ready_capacity, waveform_values));
    state.ready_order.reserve(ready_capacity);
    state.next_ready_order.reserve(ready_capacity);
    state.free_ready_slots.resize(ready_capacity);
    for (std::size_t slot = 0; slot < ready_capacity; ++slot)
    {
        state.free_ready_slots[slot] = ready_capacity - 1 - slot;
    }
    state.free_ready_count = ready_capacity;
}

OnlineThresholdDetector::~OnlineThresholdDetector()
{
    delete impl_;
}

std::size_t OnlineThresholdDetector::channel_count() const noexcept
{
    return impl_->channels;
}
std::size_t OnlineThresholdDetector::retained_sample_capacity() const noexcept
{
    return impl_->retention;
}
std::size_t OnlineThresholdDetector::pending_capacity() const noexcept
{
    return impl_->pending.capacity();
}

std::size_t OnlineThresholdDetector::workspace_bytes() const noexcept
{
    const auto& state = *impl_;
    // Sum the *reserved* (capacity) bytes of every container the detector
    // pre-allocates at construction. ``capacity`` is the right measure of the
    // fixed workspace the runtime must budget: steady-state processing never
    // grows these containers, so the reserved bytes are what remain resident.
    auto bytes = std::size_t{0};
    bytes += state.buffer.capacity() * sizeof(double);
    bytes += state.waveform_scratch.capacity() * sizeof(double);
    bytes += state.pending.capacity() * sizeof(decltype(state.pending)::value_type);
    bytes += state.complete.capacity() * sizeof(decltype(state.complete)::value_type);
    bytes += state.ready_events.capacity() * sizeof(decltype(state.ready_events)::value_type);
    bytes += state.ready_waveforms.capacity() * sizeof(double);
    bytes += state.ready_order.capacity() * sizeof(std::size_t);
    bytes += state.next_ready_order.capacity() * sizeof(std::size_t);
    bytes += state.free_ready_slots.capacity() * sizeof(std::size_t);
    bytes += state.channel_groups.capacity() * sizeof(std::size_t);
    bytes += state.last_crossing.capacity() * sizeof(std::int64_t);
    // ``std::vector<bool>`` packs its elements; report the backing storage
    // rather than ``capacity() * sizeof(bool)``.
    const auto packed_bool_bytes = [](const std::vector<bool>& flags) noexcept
    { return (flags.capacity() + CHAR_BIT - 1) / CHAR_BIT; };
    bytes += packed_bool_bytes(state.previous_active);
    bytes += packed_bool_bytes(state.has_last_crossing);
    // The detector owns copies of the channel/group configuration moved in at
    // construction; those vectors are part of its fixed workspace too.
    bytes += state.config.channel_centers.capacity() * sizeof(double);
    bytes += state.config.channel_thresholds.capacity() * sizeof(double);
    bytes += state.config.electrode_groups.capacity() * sizeof(std::vector<std::size_t>);
    for (const auto& group : state.config.electrode_groups)
    {
        bytes += group.capacity() * sizeof(std::size_t);
    }
    return bytes;
}

void OnlineThresholdDetector::reset(std::uint64_t segment_id) noexcept
{
    auto& state = *impl_;
    std::fill(state.previous_active.begin(), state.previous_active.end(), false);
    std::fill(state.has_last_crossing.begin(), state.has_last_crossing.end(), false);
    state.pending.clear();
    state.complete.clear();
    state.ready_order.clear();
    state.next_ready_order.clear();
    state.free_ready_count = state.free_ready_slots.size();
    for (std::size_t slot = 0; slot < state.free_ready_slots.size(); ++slot)
    {
        state.free_ready_slots[slot] = state.free_ready_slots.size() - 1 - slot;
    }
    state.retained_samples = 0;
    state.buffer_start = 0;
    state.expected_next = 0;
    state.segment_start = 0;
    state.fs = 0.0;
    state.last_time_start = 0.0;
    state.last_sample_idx_start = 0;
    state.segment_id = segment_id;
    state.anchored = false;
}

std::size_t OnlineThresholdDetector::ready_count() const noexcept
{
    return impl_->ready_order.size();
}

OnlineDetectionStatus OnlineThresholdDetector::finish() const noexcept
{
    const auto& state = *impl_;
    return !state.pending.empty() && state.config.boundary_behavior == BoundaryBehavior::Raise
               ? OnlineDetectionStatus::boundary_error
               : OnlineDetectionStatus::ok;
}

OnlineDetectionStatus
OnlineThresholdDetector::process(std::span<const double> sample_major, std::size_t n_samples,
                                 std::int64_t sample_idx_start, double time_start, double fs,
                                 std::uint64_t segment_id, FixedCapacitySpikeBlock& output) noexcept
{
    auto& state = *impl_;
    output.clear(segment_id);
    if (n_samples > state.config.max_input_samples ||
        sample_major.size() != n_samples * state.channels || !std::isfinite(time_start) ||
        !std::isfinite(fs) || fs <= 0.0 || sample_idx_start < 0 ||
        (n_samples != 0 && static_cast<std::uint64_t>(n_samples - 1) >
                               static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() -
                                                          sample_idx_start)))
    {
        return OnlineDetectionStatus::invalid_input;
    }
    for (const auto value : sample_major)
    {
        if (!std::isfinite(value))
        {
            return OnlineDetectionStatus::invalid_input;
        }
    }
    if (!state.anchored)
    {
        state.anchored = true;
        state.segment_id = segment_id;
        state.segment_start = sample_idx_start;
        state.buffer_start = sample_idx_start;
        state.expected_next = sample_idx_start;
        state.fs = fs;
    }
    if (segment_id != state.segment_id || sample_idx_start != state.expected_next || fs != state.fs)
    {
        return OnlineDetectionStatus::invalid_input;
    }
    if (n_samples == 0)
    {
        return OnlineDetectionStatus::ok;
    }

    const auto destination = state.retained_samples * state.channels;
    std::copy(sample_major.begin(), sample_major.end(), state.buffer.begin() + destination);
    const auto last_sample = sample_idx_start + static_cast<std::int64_t>(n_samples) - 1;
    const auto n_groups = state.config.electrode_groups.size();
    for (std::size_t local = 0; local < n_samples; ++local)
    {
        const auto absolute = sample_idx_start + static_cast<std::int64_t>(local);
        for (std::size_t group = 0; group < n_groups; ++group)
        {
            bool found = false;
            std::size_t best_channel{};
            double best_score{};
            for (const auto channel : state.config.electrode_groups[group])
            {
                const auto threshold = state.config.channel_thresholds[channel];
                const auto centered = sample_major[local * state.channels + channel] -
                                      state.config.channel_centers[channel];
                const auto current =
                    threshold > 0.0 && active(centered, threshold, state.config.polarity);
                if (current && !state.previous_active[channel])
                {
                    const auto score = std::abs(centered) / threshold;
                    if (!found || score > best_score ||
                        (score == best_score && channel < best_channel))
                    {
                        found = true;
                        best_channel = channel;
                        best_score = score;
                    }
                }
                state.previous_active[channel] = current;
            }
            if (found && (!state.has_last_crossing[group] ||
                          absolute - state.last_crossing[group] >=
                              static_cast<std::int64_t>(state.config.refractory_samples)))
            {
                if (state.pending.size() == state.pending.capacity())
                {
                    return OnlineDetectionStatus::overflow;
                }
                state.pending.push_back({absolute, static_cast<std::uint32_t>(best_channel),
                                         static_cast<std::uint32_t>(group)});
                state.has_last_crossing[group] = true;
                state.last_crossing[group] = absolute;
            }
        }
    }

    state.complete.clear();
    std::size_t retained_pending = 0;
    for (const auto pending : state.pending)
    {
        const auto completion =
            pending.crossing + static_cast<std::int64_t>(state.config.alignment_search_radius +
                                                         state.config.post_samples);
        if (completion > last_sample)
        {
            state.pending[retained_pending++] = pending;
            continue;
        }
        const auto search_start = std::max(
            state.segment_start,
            pending.crossing - static_cast<std::int64_t>(state.config.alignment_search_radius));
        const auto search_stop =
            pending.crossing + static_cast<std::int64_t>(state.config.alignment_search_radius);
        auto peak = search_start;
        auto amp =
            state.value(peak, pending.channel) - state.config.channel_centers[pending.channel];
        auto best_rank = rank(amp, state.config.polarity);
        for (auto sample = search_start + 1; sample <= search_stop; ++sample)
        {
            const auto candidate = state.value(sample, pending.channel) -
                                   state.config.channel_centers[pending.channel];
            const auto candidate_rank = rank(candidate, state.config.polarity);
            if (candidate_rank > best_rank)
            {
                peak = sample;
                amp = candidate;
                best_rank = candidate_rank;
            }
        }
        if (peak - static_cast<std::int64_t>(state.config.pre_samples) < state.segment_start)
        {
            if (state.config.boundary_behavior == BoundaryBehavior::Raise)
            {
                return OnlineDetectionStatus::boundary_error;
            }
            continue;
        }
        state.complete.push_back({
            peak,
            pending.crossing,
            pending.channel,
            pending.group,
            amp,
            std::abs(amp) / state.config.channel_thresholds[pending.channel],
            static_cast<std::int8_t>(amp < 0.0 ? -1 : 1),
        });
    }
    state.pending.resize(retained_pending);
    const auto waveform_samples = state.config.pre_samples + 1 + state.config.post_samples;
    const auto waveform_values = waveform_samples * state.channels;
    // The same total order is used to sort the just-completed events and to
    // merge them with the watermark-retained ready events below.
    const auto event_less = [](const auto& left, const auto& right) noexcept
    {
        if (left.peak != right.peak)
            return left.peak < right.peak;
        if (left.group != right.group)
            return left.group < right.group;
        if (left.channel != right.channel)
            return left.channel < right.channel;
        return left.crossing < right.crossing;
    };
    std::sort(state.complete.begin(), state.complete.end(), event_less);
    const auto left_alignment_bound = [&](std::int64_t sample) noexcept
    {
        const auto distance = static_cast<std::uint64_t>(sample - state.segment_start);
        return state.config.alignment_search_radius > distance
                   ? state.segment_start
                   : sample - static_cast<std::int64_t>(state.config.alignment_search_radius);
    };
    const auto future_sample =
        last_sample == std::numeric_limits<std::int64_t>::max() ? last_sample : last_sample + 1;
    auto safe_peak_watermark = left_alignment_bound(future_sample);
    for (const auto& pending : state.pending)
    {
        safe_peak_watermark = std::min(safe_peak_watermark, left_alignment_bound(pending.crossing));
    }

    const auto ready_is_safe = [&](const auto& event) noexcept
    {
        // Strict inequality is required because an unseen event may still
        // align to exactly the watermark and win a later tie-break key.
        return event.peak < safe_peak_watermark;
    };
    std::size_t n_safe{};
    for (const auto slot : state.ready_order)
    {
        n_safe += static_cast<std::size_t>(ready_is_safe(state.ready_events[slot]));
    }
    for (const auto& event : state.complete)
    {
        n_safe += static_cast<std::size_t>(ready_is_safe(event));
    }
    const auto n_total = state.ready_order.size() + state.complete.size();
    const auto n_unsafe = n_total - n_safe;
    if (state.config.overflow_policy == SpikeBlockOverflowPolicy::fault &&
        n_unsafe > state.ready_events.size())
    {
        output.note_overflow(n_unsafe - state.ready_events.size());
        return OnlineDetectionStatus::overflow;
    }

    state.next_ready_order.clear();
    std::size_t ready_idx{};
    std::size_t complete_idx{};
    std::size_t emitted{};
    std::uint64_t dropped{};
    while (ready_idx < state.ready_order.size() || complete_idx < state.complete.size())
    {
        const bool use_ready = complete_idx == state.complete.size() ||
                               (ready_idx < state.ready_order.size() &&
                                !event_less(state.complete[complete_idx],
                                            state.ready_events[state.ready_order[ready_idx]]));
        const auto ready_slot =
            use_ready ? state.ready_order[ready_idx] : std::numeric_limits<std::size_t>::max();
        const auto& event =
            use_ready ? state.ready_events[ready_slot] : state.complete[complete_idx];

        const bool safe = ready_is_safe(event);
        if (safe && emitted < output.header().capacity)
        {
            std::span<const double> waveform;
            if (use_ready)
            {
                waveform = {state.ready_waveforms.data() + ready_slot * waveform_values,
                            waveform_values};
            }
            else
            {
                std::size_t destination_idx = 0;
                const auto first = event.peak - static_cast<std::int64_t>(state.config.pre_samples);
                for (std::size_t sample = 0; sample < waveform_samples; ++sample)
                {
                    for (std::size_t channel = 0; channel < state.channels; ++channel)
                    {
                        state.waveform_scratch[destination_idx++] =
                            state.value(first + static_cast<std::int64_t>(sample), channel);
                    }
                }
                waveform = state.waveform_scratch;
            }
            if (!output.append(event.peak,
                               time_start + static_cast<double>(event.peak - sample_idx_start) / fs,
                               event.channel, event.group, event.amp, event.score, event.polarity,
                               event.crossing, waveform))
            {
                return OnlineDetectionStatus::overflow;
            }
            ++emitted;
            if (use_ready)
            {
                state.free_ready_slots[state.free_ready_count++] = ready_slot;
            }
        }
        else if (safe && state.config.overflow_policy == SpikeBlockOverflowPolicy::fault)
        {
            output.note_overflow();
            return OnlineDetectionStatus::overflow;
        }
        else if (safe)
        {
            ++dropped;
            if (use_ready)
            {
                state.free_ready_slots[state.free_ready_count++] = ready_slot;
            }
        }
        else if (state.next_ready_order.size() < state.ready_events.size())
        {
            if (use_ready)
            {
                state.next_ready_order.push_back(ready_slot);
            }
            else
            {
                // Encode a just-completed event after the fixed ready-slot
                // range. Allocation/copy is deferred until the merge has
                // released every safe or drop-newest ready slot, so a newly
                // completed earlier event can replace a later retained event
                // even when the reorder buffer began this call full.
                state.next_ready_order.push_back(state.ready_events.size() + complete_idx);
            }
        }
        else if (state.config.overflow_policy == SpikeBlockOverflowPolicy::fault)
        {
            return OnlineDetectionStatus::overflow;
        }
        else
        {
            ++dropped;
            if (use_ready)
            {
                state.free_ready_slots[state.free_ready_count++] = ready_slot;
            }
        }

        ready_idx += static_cast<std::size_t>(use_ready);
        complete_idx += static_cast<std::size_t>(!use_ready);
    }

    // Materialize selected just-completed events into the slots made free by
    // the merge. ``next_ready_order`` contains at most ready_capacity entries,
    // and every retained old slot remains occupied, so this invariant is
    // bounded and cannot require allocation or eviction here.
    for (auto& encoded : state.next_ready_order)
    {
        if (encoded < state.ready_events.size())
            continue;
        const auto selected_complete = encoded - state.ready_events.size();
        if (state.free_ready_count == 0 || selected_complete >= state.complete.size())
        {
            return OnlineDetectionStatus::overflow;
        }
        const auto slot = state.free_ready_slots[--state.free_ready_count];
        const auto& event = state.complete[selected_complete];
        state.ready_events[slot] = event;
        std::size_t destination_idx = slot * waveform_values;
        const auto first = event.peak - static_cast<std::int64_t>(state.config.pre_samples);
        for (std::size_t sample = 0; sample < waveform_samples; ++sample)
        {
            for (std::size_t channel = 0; channel < state.channels; ++channel)
            {
                state.ready_waveforms[destination_idx++] =
                    state.value(first + static_cast<std::int64_t>(sample), channel);
            }
        }
        encoded = slot;
    }
    state.ready_order.swap(state.next_ready_order);
    output.note_overflow(dropped);

    state.expected_next = sample_idx_start + static_cast<std::int64_t>(n_samples);
    state.last_time_start = time_start;
    state.last_sample_idx_start = sample_idx_start;
    const auto combined = state.retained_samples + n_samples;
    const auto keep = std::min(state.retention, combined);
    const auto discard = combined - keep;
    if (discard != 0)
    {
        std::memmove(state.buffer.data(), state.buffer.data() + discard * state.channels,
                     keep * state.channels * sizeof(double));
        state.buffer_start += static_cast<std::int64_t>(discard);
    }
    state.retained_samples = keep;
    return OnlineDetectionStatus::ok;
}

OnlineDetectionStatus OnlineThresholdDetector::finish(FixedCapacitySpikeBlock& output) noexcept
{
    auto& state = *impl_;
    output.clear(state.segment_id);
    const bool boundary_error =
        !state.pending.empty() && state.config.boundary_behavior == BoundaryBehavior::Raise;
    if (state.config.overflow_policy == SpikeBlockOverflowPolicy::fault &&
        state.ready_order.size() > output.header().capacity)
    {
        output.note_overflow(state.ready_order.size() - output.header().capacity);
        return OnlineDetectionStatus::overflow;
    }

    const auto waveform_values =
        (state.config.pre_samples + 1 + state.config.post_samples) * state.channels;
    const auto emitted = std::min<std::size_t>(state.ready_order.size(), output.header().capacity);
    for (std::size_t i = 0; i < emitted; ++i)
    {
        const auto slot = state.ready_order[i];
        const auto& event = state.ready_events[slot];
        if (!output.append(
                event.peak,
                state.last_time_start +
                    static_cast<double>(event.peak - state.last_sample_idx_start) / state.fs,
                event.channel, event.group, event.amp, event.score, event.polarity, event.crossing,
                {state.ready_waveforms.data() + slot * waveform_values, waveform_values}))
        {
            return OnlineDetectionStatus::overflow;
        }
    }
    const auto dropped = state.ready_order.size() - emitted;
    output.note_overflow(dropped);
    for (const auto slot : state.ready_order)
    {
        state.free_ready_slots[state.free_ready_count++] = slot;
    }
    state.ready_order.clear();
    state.pending.clear();
    return boundary_error ? OnlineDetectionStatus::boundary_error : OnlineDetectionStatus::ok;
}

} // namespace neurale::sorting
