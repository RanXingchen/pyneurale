/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace neurale::sorting
{

enum class DetectionPolarity : std::uint8_t
{
    Negative,
    Positive,
    Both,
};

enum class BoundaryBehavior
{
    Drop,
    Raise,
};

struct ThresholdDetectionConfig
{
    double threshold_multiplier{3.5};
    std::size_t refractory_samples{};
    std::size_t alignment_search_radius{};
    std::size_t pre_samples{};
    std::size_t post_samples{};
    DetectionPolarity polarity{DetectionPolarity::Negative};
    BoundaryBehavior boundary_behavior{BoundaryBehavior::Drop};
    std::vector<std::vector<std::size_t>> electrode_groups;
};

struct ThresholdDetectionResult
{
    // Flattened event-major waveform data. Populated by
    // detect_threshold_chunks() with logical shape
    // (n_events, pre_samples + 1 + post_samples, n_channels).
    std::vector<double> waveforms;
    std::vector<std::size_t> sample_indices;
    std::vector<std::size_t> crossing_indices;
    std::vector<std::size_t> peak_channel_indices;
    std::vector<std::size_t> electrode_group_ids;
    std::vector<double> amps;
    std::vector<double> scores;
    std::vector<std::int8_t> polarities;
    std::vector<double> channel_centers;
    std::vector<double> channel_noise;
    std::vector<double> channel_thresholds;
};

// Input is a contiguous sample-major matrix with shape (n_samples, n_channels).
ThresholdDetectionResult detect_threshold(std::span<const double> input, std::size_t n_samples,
                                          std::size_t n_channels,
                                          const ThresholdDetectionConfig& config);

// Process one continuous segment supplied as contiguous sample-major chunks.
// Chunk boundaries do not reset crossing, refractory, alignment, or waveform
// state and are not waveform boundaries.
ThresholdDetectionResult detect_threshold_chunks(std::span<const std::span<const double>> chunks,
                                                 std::size_t n_channels,
                                                 const ThresholdDetectionConfig& config);

// Detect event metadata without materializing waveform payloads. This is used
// by bindings that provide the final output buffer directly.
ThresholdDetectionResult
detect_threshold_chunk_events(std::span<const std::span<const double>> chunks,
                              std::size_t n_channels, const ThresholdDetectionConfig& config);

// Copy event-centered waveforms into caller-owned sample-major storage. The
// output shape is (sample_indices.size(), pre_samples + 1 + post_samples,
// n_channels), flattened in C order.
void extract_threshold_waveforms(std::span<const std::span<const double>> chunks,
                                 std::size_t n_channels, std::size_t pre_samples,
                                 std::size_t post_samples,
                                 std::span<const std::size_t> sample_indices,
                                 std::span<double> output);

} // namespace neurale::sorting
