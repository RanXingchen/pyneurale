/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/sorting/detection.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ns = neurale::sorting;

namespace
{

ns::ThresholdDetectionConfig config()
{
    return ns::ThresholdDetectionConfig{
        .threshold_multiplier = 3.0,
        .refractory_samples = 4,
        .alignment_search_radius = 2,
        .pre_samples = 2,
        .post_samples = 2,
        .polarity = ns::DetectionPolarity::Negative,
        .boundary_behavior = ns::BoundaryBehavior::Drop,
    };
}

std::vector<double> baseline(std::size_t samples, std::size_t channels)
{
    std::vector<double> values(samples * channels);
    constexpr double pattern[] = {-1.0, 0.0, 1.0, 0.0};
    for (std::size_t sample = 0; sample < samples; ++sample)
    {
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            values[sample * channels + channel] = pattern[sample % 4];
        }
    }
    return values;
}

void detects_aligned_event_deterministically()
{
    auto values = baseline(32, 1);
    values[10] = -5.0;
    values[11] = -8.0;
    values[12] = -4.0;
    const auto first = ns::detect_threshold(values, 32, 1, config());
    const auto second = ns::detect_threshold(values, 32, 1, config());
    assert(first.sample_indices == std::vector<std::size_t>{11});
    assert(first.crossing_indices == std::vector<std::size_t>{10});
    assert(first.peak_channel_indices == std::vector<std::size_t>{0});
    assert(first.amps == std::vector<double>{-8.0});
    assert(first.sample_indices == second.sample_indices);
    assert(first.amps == second.amps);
}

void suppresses_by_group_before_alignment()
{
    auto values = baseline(32, 2);
    values[8 * 2] = -7.0;
    values[8 * 2 + 1] = -10.0;
    values[10 * 2] = -12.0;
    auto grouped = config();
    grouped.alignment_search_radius = 0;
    grouped.electrode_groups = {{0, 1}};
    const auto result = ns::detect_threshold(values, 32, 2, grouped);
    assert(result.sample_indices == std::vector<std::size_t>{8});
    assert(result.peak_channel_indices == std::vector<std::size_t>{1});
    assert(result.electrode_group_ids == std::vector<std::size_t>{0});
}

void handles_empty_constant_and_invalid_inputs()
{
    const auto empty = ns::detect_threshold({}, 0, 2, config());
    assert(empty.sample_indices.empty());
    assert(empty.channel_noise == std::vector<double>({0.0, 0.0}));

    const std::vector<double> constant(20, 1.0);
    const auto none = ns::detect_threshold(constant, 10, 2, config());
    assert(none.sample_indices.empty());

    bool rejected = false;
    try
    {
        ns::detect_threshold(constant, 10, 0, config());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);

    auto nonfinite = constant;
    nonfinite[3] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try
    {
        ns::detect_threshold(nonfinite, 10, 2, config());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void chunked_detection_and_waveforms_match_whole_segment()
{
    constexpr std::size_t samples = 48;
    constexpr std::size_t channels = 2;
    auto values = baseline(samples, channels);
    // The first excursion starts at the end of chunk 0, peaks in chunk 1,
    // and needs post-samples from chunk 2. The second excursion remains active
    // across another chunk edge and must still produce one event.
    values[9 * channels] = -5.0;
    values[10 * channels] = -11.0;
    values[11 * channels] = -7.0;
    values[23 * channels + 1] = -6.0;
    values[24 * channels + 1] = -12.0;
    values[25 * channels + 1] = -8.0;

    auto detection_config = config();
    detection_config.electrode_groups = {{0}, {1}};
    const std::span<const double> whole_chunks[] = {values};
    const auto whole = ns::detect_threshold_chunks(whole_chunks, channels, detection_config);

    const std::span<const double> chunked[] = {
        std::span<const double>(values).subspan(0, 10 * channels),
        std::span<const double>(values).subspan(10 * channels, 2 * channels),
        std::span<const double>(values).subspan(12 * channels, 12 * channels),
        std::span<const double>(values).subspan(24 * channels),
    };
    const auto split = ns::detect_threshold_chunks(chunked, channels, detection_config);

    assert(split.sample_indices == whole.sample_indices);
    assert(split.crossing_indices == whole.crossing_indices);
    assert(split.peak_channel_indices == whole.peak_channel_indices);
    assert(split.electrode_group_ids == whole.electrode_group_ids);
    assert(split.amps == whole.amps);
    assert(split.polarities == whole.polarities);
    assert(split.channel_centers == whole.channel_centers);
    assert(split.channel_noise == whole.channel_noise);
    assert(split.channel_thresholds == whole.channel_thresholds);
    assert(split.waveforms == whole.waveforms);
    assert(split.sample_indices == std::vector<std::size_t>({10, 24}));
    assert(split.waveforms.size() == split.sample_indices.size() * 5 * channels);

    const auto events = ns::detect_threshold_chunk_events(chunked, channels, detection_config);
    assert(events.sample_indices == split.sample_indices);
    assert(events.waveforms.empty());
    std::vector<double> caller_owned_waveforms(split.waveforms.size());
    ns::extract_threshold_waveforms(chunked, channels, detection_config.pre_samples,
                                    detection_config.post_samples, events.sample_indices,
                                    caller_owned_waveforms);
    assert(caller_owned_waveforms == split.waveforms);
}

} // namespace

int main()
{
    detects_aligned_event_deterministically();
    suppresses_by_group_before_alignment();
    handles_empty_constant_and_invalid_inputs();
    chunked_detection_and_waveforms_match_whole_segment();
    return 0;
}
