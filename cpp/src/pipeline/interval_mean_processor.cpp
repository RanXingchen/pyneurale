// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "interval_mean_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace neurale::pipeline
{
using namespace streaming;

PreparedProcessorContract IntervalMeanProcessor::prepare(const ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1 || signals[0].kind != SignalKind::feature ||
        signals[0].dtype != SignalDType::float64 ||
        signals[0].layout != SignalLayout::sample_major || signals[0].max_block_samples != 1 ||
        signals[0].observation_timing != ObservationTiming::regular || duration_ns_ == 0 ||
        output_schema_)
        throw std::invalid_argument(
            "interval mean requires one regular float64 feature observation per frame");
    auto signal = signals[0];
    auto descriptor = *context.input_schema.feature_sets().find(signal.feature_set_id);
    window_ns_ = descriptor.window_length_ns;
    if (window_ns_ > duration_ns_ || context.input_schema.id() == UINT32_MAX ||
        signal.id == UINT32_MAX || descriptor.id == UINT64_MAX)
        throw std::invalid_argument("interval duration or schema identity is invalid");
    ++signal.id;
    ++descriptor.id;
    signal.feature_set_id = descriptor.id;
    signal.fs = {0, 1};
    signal.observation_timing = ObservationTiming::irregular;
    signal.device_tick_tracking = DeviceTickTracking::unavailable;
    descriptor.window_length_ns = duration_ns_;
    descriptor.shift_ns = 0;
    descriptor.algorithm_name = "interval_mean:" + descriptor.algorithm_name;
    descriptor.algorithm_version = "1:" + descriptor.algorithm_version;
    output_schema_ = std::make_unique<StreamSchema>(context.input_schema.id() + 1,
                                                    std::array{signal}, std::array{descriptor},
                                                    context.input_schema.units().units());
    validator_ = std::make_unique<FrameValidator>(context.input_schema);
    mean_.resize(signal.n_channels);
    return {context.input_schema.clone(),      output_schema_->clone(), 0, 0, false,
            {mean_.size() * sizeof(double), 1}};
}

StreamStatus IntervalMeanProcessor::begin(ObservationInterval interval) noexcept
{
    if (!output_schema_ || active_ || interval.key == 0 || interval.key <= last_key_ ||
        interval.end_ns <= interval.start_ns ||
        interval.end_ns - interval.start_ns != duration_ns_ || interval.start_ns < previous_end_)
        return StreamStatus::invalid_state;
    interval_ = interval;
    last_key_ = interval.key;
    previous_end_ = interval.end_ns;
    count_ = last_end_ = 0;
    std::fill(mean_.begin(), mean_.end(), 0.0);
    active_ = true;
    return StreamStatus::ok;
}

StreamStatus IntervalMeanProcessor::process(FrameBorrow& frame, FrameEmitter&) noexcept
{
    if (!validator_ || validator_->validate(frame.view()) != FrameValidationError::none)
        return StreamStatus::invalid_frame;
    const auto& block = frame.blocks()[0];
    const auto* row =
        reinterpret_cast<const double*>(frame.payload().data() + block.payload_offset);
    if (!std::all_of(row, row + mean_.size(), [](double v) { return std::isfinite(v); }))
        return StreamStatus::invalid_frame;
    if (!active_ || block.observation_time_start_ns < window_ns_ / 2)
        return StreamStatus::ok;
    const auto start = block.observation_time_start_ns - window_ns_ / 2;
    if (start < interval_.start_ns || start > UINT64_MAX - window_ns_)
        return StreamStatus::ok;
    const auto end = start + window_ns_;
    if (end > interval_.end_ns || frame.header().host_received_ns > interval_.end_ns ||
        (count_ && end <= last_end_))
        return StreamStatus::ok;
    ++count_;
    for (std::size_t i = 0; i < mean_.size(); ++i)
        mean_[i] += (row[i] - mean_[i]) / static_cast<double>(count_);
    last_end_ = end;
    last_header_ = frame.header();
    return StreamStatus::ok;
}

StreamStatus IntervalMeanProcessor::advance_time(std::uint64_t now, FrameEmitter& emitter) noexcept
{
    if (!active_ || now < interval_.end_ns)
        return StreamStatus::ok;
    active_ = false;
    if (count_ == 0)
        return StreamStatus::ok;
    FrameBorrow output;
    auto status = emitter.try_acquire(output);
    if (status != StreamStatus::ok)
        return status;
    const auto bytes = mean_.size() * sizeof(double);
    if (output->block_storage().empty() || output->payload_storage().size() < bytes)
        return StreamStatus::output_limit;
    output->header() = last_header_;
    output->header().schema_id = output_schema_->id();
    output->header().sequence = interval_.key;
    output->header().host_received_ns = now;
    output->header().flags = FrameFlags::none;
    output->header().source_tick = output->header().valid_until_ns = 0;
    output->block_storage()[0] = {.sample_idx_start = interval_.key - 1,
                                  .observation_time_start_ns =
                                      interval_.start_ns + duration_ns_ / 2,
                                  .payload_byte_count = bytes,
                                  .signal_id = output_schema_->signals()[0].id,
                                  .n_samples = 1};
    std::memcpy(output->payload_storage().data(), mean_.data(), bytes);
    status = output->set_used_sizes(1, bytes);
    return status == StreamStatus::ok ? emitter.publish_acquired() : status;
}

StreamStatus IntervalMeanProcessor::handle_discontinuity(const Discontinuity&) noexcept
{
    active_ = false;
    return StreamStatus::invalid_frame;
}
StreamStatus IntervalMeanProcessor::flush(FrameEmitter&) noexcept
{
    return StreamStatus::ok;
}
StreamStatus IntervalMeanProcessor::reset() noexcept
{
    active_ = false;
    count_ = last_end_ = last_key_ = previous_end_ = 0;
    return StreamStatus::ok;
}

} // namespace neurale::pipeline
