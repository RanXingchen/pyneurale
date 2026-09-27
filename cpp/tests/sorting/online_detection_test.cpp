/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/sorting/online_detection.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <vector>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

using namespace neurale::sorting;

OnlineThresholdDetectorConfig config(std::size_t capacity, SpikeBlockOverflowPolicy policy,
                                     std::size_t post = 1,
                                     BoundaryBehavior boundary = BoundaryBehavior::Drop)
{
    return {
        .max_input_samples = 16,
        .block_capacity = capacity,
        .refractory_samples = 0,
        .alignment_search_radius = 0,
        .pre_samples = 1,
        .post_samples = post,
        .polarity = DetectionPolarity::Negative,
        .boundary_behavior = boundary,
        .overflow_policy = policy,
        .channel_centers = {0.0},
        .channel_thresholds = {1.0},
    };
}

int run()
{
    {
        // The fixed block writer itself enforces the detector's complete
        // ordering key: peak, group, channel, then crossing. A producer cannot
        // publish an internally regressed valid prefix by bypassing the
        // detector's merge ordering.
        SpikeBlockLayout layout{4, 3, 2};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 1, 0, DetectionPolarity::Positive};
        const std::array waveform{0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
        CHECK(block.append(45, 0.045, 0, 1, 1.0, 1.0, 1, 41, waveform));
        CHECK(!block.append(43, 0.043, 0, 1, 1.0, 1.0, 1, 43, waveform));
        CHECK(block.header().n_valid == 1);
        block.clear(0);
        CHECK(block.append(45, 0.045, 0, 1, 1.0, 1.0, 1, 43, waveform));
        CHECK(!block.append(45, 0.045, 0, 0, 1.0, 1.0, 1, 43, waveform));
        block.clear(0);
        CHECK(block.append(45, 0.045, 1, 0, 1.0, 1.0, 1, 43, waveform));
        CHECK(!block.append(45, 0.045, 0, 0, 1.0, 1.0, 1, 43, waveform));
        block.clear(0);
        CHECK(block.append(45, 0.045, 0, 0, 1.0, 1.0, 1, 43, waveform));
        CHECK(!block.append(45, 0.045, 0, 0, 1.0, 1.0, 1, 41, waveform));
    }
    {
        SpikeBlockLayout layout{4, 3, 1};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 1, 0, DetectionPolarity::Negative};
        OnlineThresholdDetector detector{config(4, SpikeBlockOverflowPolicy::fault)};
        const std::array values{0.0, -2.0, 0.0, -3.0, 0.0, -4.0, 0.0};
        CHECK(detector.process(values, values.size(), 100, 2.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 3);
        CHECK(block.header().overflow_count == 0);
        CHECK(block.sample_indices()[0] == 101);
        CHECK(block.sample_indices()[1] == 103);
        CHECK(block.sample_indices()[2] == 105);
        CHECK(block.times()[2] == 2.005);
        CHECK(block.waveforms().size() == 9);
        CHECK(block.waveforms()[3] == 0.0 && block.waveforms()[4] == -3.0 &&
              block.waveforms()[5] == 0.0);
        const auto snapshot = decode_spike_block(payload);
        CHECK(snapshot.header.n_valid == 3);
        CHECK(snapshot.sample_indices == std::vector<std::int64_t>({101, 103, 105}));
        CHECK(snapshot.waveforms.size() == 9);
    }
    {
        SpikeBlockLayout layout{2, 3, 1};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 1, 0, DetectionPolarity::Negative};
        const std::array values{0.0, -2.0, 0.0, -3.0, 0.0, -4.0, 0.0};
        OnlineThresholdDetector drop{config(2, SpikeBlockOverflowPolicy::drop_newest)};
        CHECK(drop.process(values, values.size(), 0, 0.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 2);
        CHECK(block.header().overflow_count == 1);
        CHECK(block.header().flags == SpikeBlockFlags::overflowed);

        OnlineThresholdDetector fault{config(2, SpikeBlockOverflowPolicy::fault)};
        CHECK(fault.process(values, values.size(), 0, 0.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::overflow);
        CHECK(block.header().n_valid == 2);
        CHECK(block.header().overflow_count == 1);
    }
    {
        SpikeBlockLayout layout{2, 4, 1};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 2, 0, DetectionPolarity::Negative};
        OnlineThresholdDetector detector{config(2, SpikeBlockOverflowPolicy::fault, 2)};
        const std::array first{0.0, -5.0};
        const std::array second{0.0, 0.0};
        CHECK(detector.process(first, first.size(), 10, 4.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 0);
        CHECK(detector.process(second, second.size(), 12, 4.002, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 1);
        CHECK(block.sample_indices()[0] == 11);
        const std::array expected{0.0, -5.0, 0.0, 0.0};
        CHECK(std::equal(block.waveforms().begin(), block.waveforms().end(), expected.begin()));

        detector.reset(1);
        CHECK(detector.process(second, second.size(), 20, 8.0, 1'000.0, 1, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 0);
        CHECK(block.header().segment_id == 1);
    }
    {
        SpikeBlockLayout layout{4, 3, 1};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 1, 0, DetectionPolarity::Negative};
        OnlineThresholdDetector detector{config(4, SpikeBlockOverflowPolicy::fault)};
        const std::array values{0.0, -2.0, 0.0, 0.0};
        CHECK(detector.process(values, values.size(), 0, 0.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        detector.reset(0);
        const auto before = allocations.load(std::memory_order_relaxed);
        for (std::size_t repetition = 0; repetition < 100; ++repetition)
        {
            CHECK(detector.process(values, values.size(), 0, 0.0, 1'000.0, 0, block) ==
                  OnlineDetectionStatus::ok);
            detector.reset(0);
        }
        CHECK(allocations.load(std::memory_order_relaxed) == before);
    }
    {
        // workspace_bytes() must account for the pending/complete event buffers
        // and per-group state that the adapter's old buffer+waveform estimate
        // omitted, and it must scale with the input capacity.
        std::vector<double> centers(32, 0.0);
        std::vector<double> thresholds(32, 1.0);
        std::vector<std::vector<std::size_t>> groups(2);
        groups[0].reserve(16);
        groups[1].reserve(16);
        for (std::size_t channel = 0; channel < 32; ++channel)
        {
            groups[channel / 16].push_back(channel);
        }
        OnlineThresholdDetectorConfig cfg{
            .max_input_samples = 1024,
            .block_capacity = 64,
            .refractory_samples = 4,
            .alignment_search_radius = 4,
            .pre_samples = 16,
            .post_samples = 31,
            .polarity = DetectionPolarity::Negative,
            .boundary_behavior = BoundaryBehavior::Drop,
            .overflow_policy = SpikeBlockOverflowPolicy::fault,
            .channel_centers = centers,
            .channel_thresholds = thresholds,
            .electrode_groups = groups,
        };
        OnlineThresholdDetector detector{cfg};
        const auto workspace = detector.workspace_bytes();

        // The old estimate counted only the retained input buffer and waveform
        // scratch. The full workspace must strictly exceed it, proving the
        // pending/complete and per-group state are now counted.
        const auto channels = cfg.channel_centers.size();
        const auto retention =
            2 * cfg.alignment_search_radius + cfg.pre_samples + cfg.post_samples + 1;
        const auto buffer_bytes = (retention + cfg.max_input_samples) * channels * sizeof(double);
        const auto waveform_bytes =
            (cfg.pre_samples + 1 + cfg.post_samples) * channels * sizeof(double);
        CHECK(workspace > buffer_bytes + waveform_bytes);
        // Each pending/complete event record carries at least one int64_t
        // crossing sample, so the reserved event buffers add a measurable
        // floor beyond the input buffer and waveform scratch.
        const auto n_groups = cfg.electrode_groups.size();
        const auto pending_capacity =
            n_groups * (cfg.max_input_samples + cfg.alignment_search_radius + cfg.post_samples);
        CHECK(workspace >=
              buffer_bytes + waveform_bytes + 2 * pending_capacity * sizeof(std::int64_t));

        // Scaling: doubling max_input_samples grows the input buffer and the
        // reserved event buffers, so the workspace must grow with it.
        OnlineThresholdDetectorConfig larger = cfg;
        larger.max_input_samples = 2048;
        OnlineThresholdDetector bigger{larger};
        CHECK(bigger.workspace_bytes() > workspace);
    }
    {
        // A spike pending in frame 1 completes in frame 2; its event time must
        // use frame 2's clock mapping (a different time_start), not the frame-1
        // anchor. Even though the peak sits in retained frame-1 samples, the
        // negative offset against frame 2's sample_idx_start is correct.
        SpikeBlockLayout layout{4, 3, 1};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 1, 0, DetectionPolarity::Negative};
        OnlineThresholdDetector detector{config(4, SpikeBlockOverflowPolicy::fault, 1)};
        const std::array first{0.0, -2.0};
        CHECK(detector.process(first, first.size(), 0, 10.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 0); // crossing at sample 1 stays pending
        const std::array second{0.0};
        CHECK(detector.process(second, second.size(), 2, 100.0, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 1);
        CHECK(block.sample_indices()[0] == 1);
        const auto t = block.times()[0];
        // Frame-2 mapping: 100.0 + (1 - 2) / 1000 = 99.999.
        CHECK(std::abs(t - 99.999) < 1e-6);
        // The stale frame-1 anchor would have produced 10.0 + (1 - 0) / 1000 = 10.001.
        CHECK(std::abs(t - 10.001) > 1e-3);
    }
    {
        // A pending spike whose post-tail never arrived cannot complete at
        // end-of-stream. ``finish()`` reports a tail boundary error under
        // ``Raise`` and succeeds under ``Drop``; with no pending spikes it
        // always succeeds. It is a query and does not clear the pending state.
        SpikeBlockLayout layout{4, 3, 1};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 1, 0, DetectionPolarity::Negative};
        const std::array pending_values{0.0, -2.0}; // crossing at sample 1; post=1 needs sample 2

        {
            OnlineThresholdDetector drop{
                config(4, SpikeBlockOverflowPolicy::fault, 1, BoundaryBehavior::Drop)};
            CHECK(drop.process(pending_values, pending_values.size(), 0, 0.0, 1'000.0, 0, block) ==
                  OnlineDetectionStatus::ok);
            CHECK(block.header().n_valid == 0); // stays pending
            CHECK(drop.finish(block) == OnlineDetectionStatus::ok);
            CHECK(block.header().n_valid == 0);
        }
        {
            OnlineThresholdDetector raise{
                config(4, SpikeBlockOverflowPolicy::fault, 1, BoundaryBehavior::Raise)};
            CHECK(raise.process(pending_values, pending_values.size(), 0, 0.0, 1'000.0, 0, block) ==
                  OnlineDetectionStatus::ok);
            CHECK(raise.finish(block) == OnlineDetectionStatus::boundary_error);
            CHECK(block.header().n_valid == 0);
        }
        // No pending spikes: finish() succeeds regardless of boundary behavior.
        {
            const std::array complete_values{0.0, -2.0,
                                             0.0}; // crossing at 1 completes (sample 2 present)
            OnlineThresholdDetector drop{
                config(4, SpikeBlockOverflowPolicy::fault, 1, BoundaryBehavior::Drop)};
            CHECK(drop.process(complete_values, complete_values.size(), 0, 0.0, 1'000.0, 0,
                               block) == OnlineDetectionStatus::ok);
            CHECK(block.header().n_valid == 1);
            CHECK(drop.finish(block) == OnlineDetectionStatus::ok);
            OnlineThresholdDetector raise{
                config(4, SpikeBlockOverflowPolicy::fault, 1, BoundaryBehavior::Raise)};
            CHECK(raise.process(complete_values, complete_values.size(), 0, 0.0, 1'000.0, 0,
                                block) == OnlineDetectionStatus::ok);
            CHECK(raise.finish(block) == OnlineDetectionStatus::ok);
        }
    }
    {
        // Completed events remain behind a strict cross-frame peak watermark
        // until no pending/future event can align before them. Event A
        // completes first at peak 45; event B completes in the next frame but
        // aligns to peak 43. They must be released together as [43, 45].
        OnlineThresholdDetectorConfig cfg{
            .max_input_samples = 64,
            .block_capacity = 8,
            .refractory_samples = 5,
            .alignment_search_radius = 4,
            .pre_samples = 1,
            .post_samples = 3,
            .polarity = DetectionPolarity::Positive,
            .boundary_behavior = BoundaryBehavior::Drop,
            .overflow_policy = SpikeBlockOverflowPolicy::fault,
            .channel_centers = {0.0, 0.0},
            .channel_thresholds = {1.0, 1.0},
        };
        SpikeBlockLayout layout{8, 5, 2};
        std::vector<std::byte> payload(layout.payload_bytes());
        FixedCapacitySpikeBlock block{payload, layout, 1, 3, 0, DetectionPolarity::Positive};
        OnlineThresholdDetector detector{cfg};
        std::vector<double> first(50 * 2, 0.0);
        first[41 * 2] = 2.0;
        first[45 * 2] = 5.0;
        first[43 * 2 + 1] = 6.0;
        CHECK(detector.process(first, 50, 0, 0.0, 1'000.0, 0, block) == OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 0);
        CHECK(detector.ready_count() == 1);
        const std::vector<double> second(6 * 2, 0.0);
        CHECK(detector.process(second, 6, 50, 0.05, 1'000.0, 0, block) ==
              OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 2);
        CHECK(block.sample_indices()[0] == 43);
        CHECK(block.sample_indices()[1] == 45);
        CHECK(detector.ready_count() == 0);

        // The same retained A is a completed event, not an incomplete tail:
        // flush must publish it even though B remains pending and is dropped.
        detector.reset(0);
        CHECK(detector.process(first, 50, 0, 0.0, 1'000.0, 0, block) == OnlineDetectionStatus::ok);
        CHECK(detector.ready_count() == 1);
        CHECK(detector.finish(block) == OnlineDetectionStatus::ok);
        CHECK(block.header().n_valid == 1);
        CHECK(block.sample_indices()[0] == 45);

        // The persistent reorder path, including reset and final drain, owns
        // all workspace established by construction.
        detector.reset(0);
        const auto before = allocations.load(std::memory_order_relaxed);
        for (std::size_t repetition = 0; repetition < 100; ++repetition)
        {
            CHECK(detector.process(first, 50, 0, 0.0, 1'000.0, 0, block) ==
                  OnlineDetectionStatus::ok);
            CHECK(detector.process(second, 6, 50, 0.05, 1'000.0, 0, block) ==
                  OnlineDetectionStatus::ok);
            CHECK(detector.finish(block) == OnlineDetectionStatus::ok);
            detector.reset(0);
        }
        CHECK(allocations.load(std::memory_order_relaxed) == before);
    }
    return 0;
}

} // namespace

int main()
{
    return run();
}
