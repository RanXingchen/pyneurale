// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "interval_runner.h"
#include <cstring>
#include <neurale/streaming/clock.h>
#include <stdexcept>

namespace neurale::execution
{
using namespace streaming;

class IntervalRunner::ResultEmitter final : public FrameEmitter
{
  public:
    explicit ResultEmitter(const StreamSchema& schema)
        : validator_(schema), pool_(65, schema.signals()[0].max_block_bytes, 1)
    {
        pool_.prefault();
    }
    void begin(pipeline::ObservationInterval interval) noexcept
    {
        interval_ = interval;
        emitted = false;
    }
    bool emitted{};
    FrameLease take() noexcept
    {
        return std::move(decoded_);
    }
    StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

  private:
    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        auto status = pool_.try_acquire(lease_);
        frame = status == StreamStatus::ok ? &lease_.frame() : nullptr;
        return status;
    }
    bool valid(FrameView frame) noexcept
    {
        if (emitted || validator_.validate(frame) != FrameValidationError::none ||
            frame.header.sequence != interval_.key ||
            frame.blocks[0].sample_idx_start != interval_.key - 1 ||
            frame.blocks[0].observation_time_start_ns !=
                interval_.start_ns + (interval_.end_ns - interval_.start_ns) / 2)
            return false;
        return true;
    }
    StreamStatus publish_acquired_frame() noexcept override
    {
        if (!valid(lease_.view()))
        {
            static_cast<void>(lease_.reset());
            return StreamStatus::invalid_frame;
        }
        decoded_ = std::move(lease_);
        emitted = true;
        return StreamStatus::ok;
    }
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        const auto frame = lease.view();
        if (!valid(frame))
            return StreamStatus::invalid_frame;
        FrameLease copy;
        auto status = pool_.try_acquire(copy);
        if (status != StreamStatus::ok)
            return status;
        copy.frame().header() = frame.header;
        copy.frame().block_storage()[0] = frame.blocks[0];
        std::memcpy(copy.frame().payload_storage().data(), frame.payload.data(),
                    frame.payload.size());
        status = copy.frame().set_used_sizes(1, frame.payload.size());
        if (status != StreamStatus::ok)
            return status;
        decoded_ = std::move(copy);
        emitted = true;
        return StreamStatus::ok;
    }
    FrameValidator validator_;
    FramePool pool_;
    FrameLease lease_;
    FrameLease decoded_;
    pipeline::ObservationInterval interval_;
};

class IntervalRunner::MeanEmitter final : public FrameEmitter
{
  public:
    explicit MeanEmitter(IntervalRunner& owner)
        : owner_(owner), pool_(1, owner.feature_schema().signals()[0].max_block_bytes, 1)
    {
        pool_.prefault();
    }
    StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

  private:
    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        const auto status = pool_.try_acquire(lease_);
        frame = status == StreamStatus::ok ? &lease_.frame() : nullptr;
        return status;
    }
    StreamStatus forward(MutableFrame& frame) noexcept
    {
        const auto interval = owner_.mean_.interval();
        const auto width = owner_.feature_schema().signals()[0].n_channels;
        std::memcpy(owner_.features_.data() + owner_.ready() * width, frame.payload().data(),
                    width * sizeof(double));
        auto* decoder = owner_.decoder_.load(std::memory_order_acquire);
        if (!decoder)
            return StreamStatus::ok;
        owner_.result_emitter_->begin(interval);
        const auto status = decoder->process(frame, *owner_.result_emitter_);
        if (status != StreamStatus::ok)
            return status;
        return owner_.result_emitter_->emitted ? StreamStatus::ok : StreamStatus::processor_failure;
    }
    StreamStatus publish_acquired_frame() noexcept override
    {
        const auto status = forward(lease_.frame());
        const auto released = lease_.reset();
        return status == StreamStatus::ok ? released : status;
    }
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return forward(lease.frame());
    }
    IntervalRunner& owner_;
    FramePool pool_;
    FrameLease lease_;
};

IntervalRunner::IntervalRunner(const StreamSchema& schema, std::uint64_t duration,
                               std::size_t intervals)
    : input_schema_(schema.clone()), validator_(schema), mean_(duration),
      contract_(mean_.prepare({schema, 1, 0, 1})),
      input_pool_(65, schema.signals()[0].max_block_bytes, 1),
      feature_trace_(256 * schema.signals()[0].n_channels)
{
    const auto width = schema.signals()[0].n_channels;
    if (intervals == 0 || intervals > features_.max_size() / width)
        throw std::invalid_argument("invalid interval storage capacity");
    features_.resize(intervals * width);
    counts_.resize(intervals);
    completed_intervals_.resize(intervals);
    mean_emitter_ = std::make_unique<MeanEmitter>(*this);
    input_pool_.prefault();
    frames_.prefault();
    intervals_.prefault();
    results_.prefault();
    feature_trace_.prefault();
}
IntervalRunner::~IntervalRunner()
{
    stop();
}
void IntervalRunner::publish_decoder(NativeFrameProcessor& decoder)
{
    if (decoder_.load())
        throw std::invalid_argument("interval decoder already published");
    const auto contract = decoder.prepare({feature_schema(), 1, 0, 1});
    const auto signals = contract.output_schema.signals();
    if (signals.size() != 1 || signals[0].max_block_samples != 1 ||
        contract.max_process_outputs_per_input != 1 || contract.max_flush_outputs != 0 ||
        contract.can_forward_input || contract.required_resources.frame_pool_leases > 1)
        throw std::invalid_argument("interval decoder must emit one observation frame");
    result_emitter_ = std::make_unique<ResultEmitter>(contract.output_schema);
    decoder_.store(&decoder, std::memory_order_release);
}
StreamStatus IntervalRunner::schedule(pipeline::ObservationInterval interval) noexcept
{
    if (stopping_.load(std::memory_order_acquire))
        return StreamStatus::stopped;
    if (interval.key == 0 || scheduled_ == counts_.size())
        return StreamStatus::invalid_frame;
    const auto status = intervals_.try_push(std::move(interval));
    if (status == StreamStatus::ok)
        ++scheduled_;
    return status;
}
std::span<const double> IntervalRunner::features(std::size_t index) const
{
    if (index >= ready())
        throw std::out_of_range("interval features not ready");
    const auto width = feature_schema().signals()[0].n_channels;
    return {features_.data() + index * width, width};
}
std::uint64_t IntervalRunner::now_ns() const noexcept
{
    return clock_ ? clock_->now_ns() : 0;
}
StreamStatus IntervalRunner::start()
{
    if (!clock_ || worker_.joinable() || stopping_.load())
        return StreamStatus::invalid_state;
    worker_ = std::thread(
        [this]
        {
            while (!stopping_.load(std::memory_order_acquire))
            {
                const auto value = advance_time(now_ns());
                if (value == StreamStatus::stopped && stopping_.load(std::memory_order_acquire))
                    return;
                if (value != StreamStatus::ok)
                {
                    status_.store(value, std::memory_order_release);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    return StreamStatus::ok;
}
void IntervalRunner::stop() noexcept
{
    cancel();
    if (worker_.joinable())
        worker_.join();
}
StreamStatus IntervalRunner::consume(FrameView frame) noexcept
{
    if (status() != StreamStatus::ok)
        return status();
    if (stopping_.load(std::memory_order_acquire))
        return StreamStatus::stopped;
    if (validator_.validate(frame) != FrameValidationError::none)
        return StreamStatus::invalid_frame;
    FrameLease lease;
    auto status = input_pool_.try_acquire(lease);
    if (status != StreamStatus::ok)
        return status;
    if (frame.payload.size() > lease.frame().payload_storage().size())
        return StreamStatus::output_limit;
    lease.frame().header() = frame.header;
    lease.frame().block_storage()[0] = frame.blocks[0];
    std::memcpy(lease.frame().payload_storage().data(), frame.payload.data(), frame.payload.size());
    status = lease.frame().set_used_sizes(1, frame.payload.size());
    if (status != StreamStatus::ok)
        return status;
    return frames_.try_push(std::move(lease));
}
StreamStatus IntervalRunner::advance_time(std::uint64_t now) noexcept
{
    if (status() != StreamStatus::ok)
        return status();
    if (stopping_.load(std::memory_order_acquire))
        return StreamStatus::stopped;
    if (!mean_.active())
    {
        pipeline::ObservationInterval interval;
        if (intervals_.try_pop(interval) == StreamStatus::ok)
        {
            if (ready() == counts_.size())
                return StreamStatus::output_limit;
            const auto status = mean_.begin(interval);
            if (status != StreamStatus::ok)
                return status;
        }
    }
    // Bound each timer iteration even while acquisition continues producing.
    for (std::size_t i = 0, n = frames_.approximate_size(); i < n; ++i)
    {
        FrameLease frame;
        if (frames_.try_pop(frame) != StreamStatus::ok)
            break;
        const auto& block = frame.view().blocks[0];
        const auto* values =
            reinterpret_cast<const double*>(frame.view().payload.data() + block.payload_offset);
        const auto width = input_schema_.signals()[0].n_channels;
        for (std::size_t c = 0; c < width; ++c)
            if (feature_trace_.try_push({block.sample_idx_start, block.observation_time_start_ns, c,
                                         values[c]}) != StreamStatus::ok)
                return StreamStatus::queue_overflow;
        const auto status = mean_.process(frame.frame(), *mean_emitter_);
        if (status != StreamStatus::ok)
            return status;
    }
    const auto interval = mean_.interval();
    const bool finishing = mean_.active() && now >= interval.end_ns;
    const auto status = mean_.advance_time(now, *mean_emitter_);
    if (status != StreamStatus::ok || !finishing)
        return status;
    const auto index = ready();
    counts_[index] = mean_.count();
    completed_intervals_[index] = interval;
    ready_.store(index + 1, std::memory_order_release);
    return results_.try_push({interval, mean_.count(),
                              interval.start_ns + (interval.end_ns - interval.start_ns) / 2,
                              result_emitter_ ? result_emitter_->take() : FrameLease{}});
}
StreamStatus IntervalRunner::handle_discontinuity(const Discontinuity&) noexcept
{
    status_.store(StreamStatus::invalid_frame, std::memory_order_release);
    cancel();
    return StreamStatus::invalid_frame;
}
} // namespace neurale::execution
