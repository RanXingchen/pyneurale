/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/sorting/detection.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neurale::sorting
{
namespace
{

constexpr double normal_75th_percentile = 0.6744897501960817;

struct Crossing
{
    std::size_t sample{};
    std::size_t channel{};
    std::size_t group{};
    double score{};
};

struct Event
{
    std::size_t sample{};
    std::size_t crossing{};
    std::size_t channel{};
    std::size_t group{};
    double amp{};
    double score{};
    std::int8_t polarity{};
};

std::size_t checked_size(std::size_t rows, std::size_t columns)
{
    if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows)
    {
        throw std::length_error("threshold detection input shape is too large");
    }
    return rows * columns;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* message)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::length_error(message);
    }
    return left + right;
}

class ChunkedInput
{
  public:
    ChunkedInput(std::span<const std::span<const double>> chunks, std::size_t n_channels)
        : n_channels_(n_channels)
    {
        if (n_channels == 0)
        {
            throw std::invalid_argument("threshold detection requires at least one channel");
        }
        for (const auto chunk : chunks)
        {
            if (chunk.size() % n_channels != 0)
            {
                throw std::invalid_argument("threshold detection chunk shape is invalid");
            }
            const std::size_t rows = chunk.size() / n_channels;
            if (rows == 0)
            {
                continue;
            }
            offsets_.push_back(n_samples_);
            chunks_.push_back(chunk);
            n_samples_ =
                checked_add(n_samples_, rows, "threshold detection sample count is too large");
        }
    }

    [[nodiscard]] std::size_t n_samples() const noexcept
    {
        return n_samples_;
    }

    [[nodiscard]] std::size_t n_channels() const noexcept
    {
        return n_channels_;
    }

    [[nodiscard]] const std::vector<std::span<const double>>& chunks() const noexcept
    {
        return chunks_;
    }

    [[nodiscard]] double value(std::size_t sample, std::size_t channel) const
    {
        const auto upper = std::upper_bound(offsets_.begin(), offsets_.end(), sample);
        const std::size_t chunk_idx = static_cast<std::size_t>(upper - offsets_.begin() - 1);
        const std::size_t local_sample = sample - offsets_[chunk_idx];
        return chunks_[chunk_idx][local_sample * n_channels_ + channel];
    }

  private:
    std::size_t n_channels_{};
    std::size_t n_samples_{};
    std::vector<std::span<const double>> chunks_;
    std::vector<std::size_t> offsets_;
};

double median(std::vector<double>& values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0)
    {
        return values[middle];
    }
    return values[middle - 1] / 2.0 + values[middle] / 2.0;
}

bool active(double value, double threshold, DetectionPolarity polarity) noexcept
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

double aligned_rank(double value, DetectionPolarity polarity) noexcept
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

std::vector<std::vector<std::size_t>>
resolve_groups(const std::vector<std::vector<std::size_t>>& requested, std::size_t n_channels)
{
    if (requested.empty())
    {
        std::vector<std::vector<std::size_t>> groups(n_channels);
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            groups[channel].push_back(channel);
        }
        return groups;
    }

    std::vector<bool> claimed(n_channels, false);
    for (const auto& group : requested)
    {
        if (group.empty())
        {
            throw std::invalid_argument("electrode groups must not be empty");
        }
        for (const std::size_t channel : group)
        {
            if (channel >= n_channels)
            {
                throw std::invalid_argument("electrode group channel is outside the input");
            }
            if (claimed[channel])
            {
                throw std::invalid_argument("electrode groups must partition the channels");
            }
            claimed[channel] = true;
        }
    }
    if (std::find(claimed.begin(), claimed.end(), false) != claimed.end())
    {
        throw std::invalid_argument("electrode groups must cover every input channel");
    }
    return requested;
}

std::vector<Crossing> suppress_refractory(std::vector<Crossing> crossings,
                                          std::size_t refractory_samples)
{
    if (crossings.empty())
    {
        return {};
    }
    std::sort(crossings.begin(), crossings.end(),
              [](const Crossing& left, const Crossing& right)
              {
                  if (left.sample != right.sample)
                  {
                      return left.sample < right.sample;
                  }
                  if (left.score != right.score)
                  {
                      return left.score > right.score;
                  }
                  return left.channel < right.channel;
              });

    std::vector<Crossing> unique;
    unique.reserve(crossings.size());
    for (const Crossing& candidate : crossings)
    {
        if (unique.empty() || unique.back().sample != candidate.sample)
        {
            unique.push_back(candidate);
        }
    }

    std::vector<Crossing> accepted;
    accepted.reserve(unique.size());
    accepted.push_back(unique.front());
    for (std::size_t i = 1; i < unique.size(); ++i)
    {
        if (unique[i].sample - accepted.back().sample >= refractory_samples)
        {
            accepted.push_back(unique[i]);
        }
    }
    return accepted;
}

} // namespace

namespace
{

ThresholdDetectionResult detect_threshold_impl(const ChunkedInput& input,
                                               const ThresholdDetectionConfig& config)
{
    const std::size_t n_samples = input.n_samples();
    const std::size_t n_channels = input.n_channels();
    if (!std::isfinite(config.threshold_multiplier) || config.threshold_multiplier <= 0.0)
    {
        throw std::invalid_argument("threshold multiplier must be finite and positive");
    }
    if (config.post_samples == std::numeric_limits<std::size_t>::max() ||
        config.pre_samples > std::numeric_limits<std::size_t>::max() - config.post_samples - 1)
    {
        throw std::length_error("threshold waveform bounds are too large");
    }
    for (const auto chunk : input.chunks())
    {
        for (const double value : chunk)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("threshold detection input must contain finite values");
            }
        }
    }

    const auto groups = resolve_groups(config.electrode_groups, n_channels);
    std::vector<std::size_t> channel_groups(n_channels);
    for (std::size_t group = 0; group < groups.size(); ++group)
    {
        for (const std::size_t channel : groups[group])
        {
            channel_groups[channel] = group;
        }
    }

    ThresholdDetectionResult result;
    result.channel_centers.assign(n_channels, 0.0);
    result.channel_noise.assign(n_channels, 0.0);
    result.channel_thresholds.assign(n_channels, 0.0);
    if (n_samples == 0)
    {
        return result;
    }

    // A single reusable channel buffer avoids a full sample-major/channel-major
    // transpose. Temporary storage is bounded by n_samples plus detected crossings.
    std::vector<double> scratch(n_samples);
    for (std::size_t channel = 0; channel < n_channels; ++channel)
    {
        std::size_t destination = 0;
        for (const auto chunk : input.chunks())
        {
            const std::size_t rows = chunk.size() / n_channels;
            for (std::size_t sample = 0; sample < rows; ++sample)
            {
                scratch[destination++] = chunk[sample * n_channels + channel];
            }
        }
        const double center = median(scratch);
        result.channel_centers[channel] = center;
        destination = 0;
        for (const auto chunk : input.chunks())
        {
            const std::size_t rows = chunk.size() / n_channels;
            for (std::size_t sample = 0; sample < rows; ++sample)
            {
                scratch[destination++] = std::abs(chunk[sample * n_channels + channel] - center);
            }
        }
        const double noise = median(scratch) / normal_75th_percentile;
        const double threshold = config.threshold_multiplier * noise;
        if (!std::isfinite(noise) || !std::isfinite(threshold))
        {
            throw std::invalid_argument("threshold geometry must remain finite");
        }
        result.channel_noise[channel] = noise;
        result.channel_thresholds[channel] = threshold;
    }

    std::vector<std::vector<Crossing>> group_crossings(groups.size());
    std::vector<bool> previous_active(n_channels, false);
    std::size_t absolute_sample = 0;
    for (const auto chunk : input.chunks())
    {
        const std::size_t rows = chunk.size() / n_channels;
        for (std::size_t sample = 0; sample < rows; ++sample, ++absolute_sample)
        {
            for (std::size_t channel = 0; channel < n_channels; ++channel)
            {
                const double threshold = result.channel_thresholds[channel];
                const double value =
                    chunk[sample * n_channels + channel] - result.channel_centers[channel];
                const bool current = threshold > 0.0 && active(value, threshold, config.polarity);
                if (current && !previous_active[channel])
                {
                    group_crossings[channel_groups[channel]].push_back(Crossing{
                        .sample = absolute_sample,
                        .channel = channel,
                        .group = channel_groups[channel],
                        .score = std::abs(value) / threshold,
                    });
                }
                previous_active[channel] = current;
            }
        }
    }

    std::vector<Event> events;
    for (std::size_t group = 0; group < group_crossings.size(); ++group)
    {
        auto accepted =
            suppress_refractory(std::move(group_crossings[group]), config.refractory_samples);
        for (const Crossing& crossing : accepted)
        {
            const std::size_t start = crossing.sample > config.alignment_search_radius
                                          ? crossing.sample - config.alignment_search_radius
                                          : 0;
            const std::size_t remaining = n_samples - 1 - crossing.sample;
            const std::size_t stop =
                crossing.sample + std::min(config.alignment_search_radius, remaining);
            std::size_t peak = start;
            double amp =
                input.value(start, crossing.channel) - result.channel_centers[crossing.channel];
            double rank = aligned_rank(amp, config.polarity);
            for (std::size_t sample = start + 1; sample <= stop; ++sample)
            {
                const double candidate = input.value(sample, crossing.channel) -
                                         result.channel_centers[crossing.channel];
                const double candidate_rank = aligned_rank(candidate, config.polarity);
                if (candidate_rank > rank)
                {
                    peak = sample;
                    amp = candidate;
                    rank = candidate_rank;
                }
            }

            const bool before_start = peak < config.pre_samples;
            const bool after_end = config.post_samples > n_samples - 1 - peak;
            if (before_start || after_end)
            {
                if (config.boundary_behavior == BoundaryBehavior::Raise)
                {
                    throw std::invalid_argument(
                        "detected spike waveform crosses the input boundary");
                }
                continue;
            }
            events.push_back(Event{
                .sample = peak,
                .crossing = crossing.sample,
                .channel = crossing.channel,
                .group = group,
                .amp = amp,
                .score = std::abs(amp) / result.channel_thresholds[crossing.channel],
                .polarity = static_cast<std::int8_t>(amp < 0.0 ? -1 : 1),
            });
        }
    }

    std::sort(events.begin(), events.end(),
              [](const Event& left, const Event& right)
              {
                  if (left.sample != right.sample)
                  {
                      return left.sample < right.sample;
                  }
                  if (left.group != right.group)
                  {
                      return left.group < right.group;
                  }
                  if (left.channel != right.channel)
                  {
                      return left.channel < right.channel;
                  }
                  return left.crossing < right.crossing;
              });
    result.sample_indices.reserve(events.size());
    result.crossing_indices.reserve(events.size());
    result.peak_channel_indices.reserve(events.size());
    result.electrode_group_ids.reserve(events.size());
    result.amps.reserve(events.size());
    result.scores.reserve(events.size());
    result.polarities.reserve(events.size());
    for (const Event& event : events)
    {
        result.sample_indices.push_back(event.sample);
        result.crossing_indices.push_back(event.crossing);
        result.peak_channel_indices.push_back(event.channel);
        result.electrode_group_ids.push_back(event.group);
        result.amps.push_back(event.amp);
        result.scores.push_back(event.score);
        result.polarities.push_back(event.polarity);
    }
    return result;
}

} // namespace

ThresholdDetectionResult detect_threshold(std::span<const double> input, std::size_t n_samples,
                                          std::size_t n_channels,
                                          const ThresholdDetectionConfig& config)
{
    if (input.size() != checked_size(n_samples, n_channels))
    {
        throw std::invalid_argument("threshold detection input shape is invalid");
    }
    const std::span<const double> chunks[] = {input};
    return detect_threshold_impl(ChunkedInput(chunks, n_channels), config);
}

ThresholdDetectionResult detect_threshold_chunks(std::span<const std::span<const double>> chunks,
                                                 std::size_t n_channels,
                                                 const ThresholdDetectionConfig& config)
{
    if (chunks.empty())
    {
        throw std::invalid_argument("threshold detection chunks must not be empty");
    }
    ThresholdDetectionResult result = detect_threshold_chunk_events(chunks, n_channels, config);
    const std::size_t waveform_samples =
        checked_add(checked_add(config.pre_samples, 1, "threshold waveform shape is too large"),
                    config.post_samples, "threshold waveform shape is too large");
    const std::size_t waveform_values =
        checked_size(result.sample_indices.size(), checked_size(waveform_samples, n_channels));
    result.waveforms.resize(waveform_values);
    extract_threshold_waveforms(chunks, n_channels, config.pre_samples, config.post_samples,
                                result.sample_indices, result.waveforms);
    return result;
}

ThresholdDetectionResult
detect_threshold_chunk_events(std::span<const std::span<const double>> chunks,
                              std::size_t n_channels, const ThresholdDetectionConfig& config)
{
    if (chunks.empty())
    {
        throw std::invalid_argument("threshold detection chunks must not be empty");
    }
    return detect_threshold_impl(ChunkedInput(chunks, n_channels), config);
}

void extract_threshold_waveforms(std::span<const std::span<const double>> chunks,
                                 std::size_t n_channels, std::size_t pre_samples,
                                 std::size_t post_samples,
                                 std::span<const std::size_t> sample_indices,
                                 std::span<double> output)
{
    if (chunks.empty())
    {
        throw std::invalid_argument("threshold detection chunks must not be empty");
    }
    const ChunkedInput input(chunks, n_channels);
    const std::size_t waveform_samples =
        checked_add(checked_add(pre_samples, 1, "threshold waveform shape is too large"),
                    post_samples, "threshold waveform shape is too large");
    const std::size_t values_per_waveform = checked_size(waveform_samples, n_channels);
    if (output.size() != checked_size(sample_indices.size(), values_per_waveform))
    {
        throw std::invalid_argument("threshold waveform output shape is invalid");
    }

    std::size_t destination = 0;
    for (const std::size_t peak : sample_indices)
    {
        if (peak >= input.n_samples() || peak < pre_samples ||
            post_samples > input.n_samples() - 1 - peak)
        {
            throw std::invalid_argument("threshold waveform sample crosses the input boundary");
        }
        const std::size_t first = peak - pre_samples;
        for (std::size_t sample = first; sample < first + waveform_samples; ++sample)
        {
            for (std::size_t channel = 0; channel < n_channels; ++channel)
            {
                output[destination++] = input.value(sample, channel);
            }
        }
    }
}

} // namespace neurale::sorting
