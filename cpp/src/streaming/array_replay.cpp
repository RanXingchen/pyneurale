/* SPDX-License-Identifier: MIT */
#include <neurale/streaming/array_replay.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace neurale::streaming
{
namespace
{
SignalSchema dense_signal(const StreamSchema& schema)
{
    if (schema.signals().size() != 1)
        throw std::invalid_argument("Replay requires exactly one dense signal.");
    const auto& signal = schema.signals().front();
    if (signal.dtype != SignalDType::float64 || signal.layout != SignalLayout::sample_major ||
        (signal.kind != SignalKind::sampled && signal.kind != SignalKind::feature) ||
        signal.n_channels == 0 || signal.nominal_block_samples == 0)
        throw std::invalid_argument("Replay requires a float64 sample-major dense signal.");
    return signal;
}
} // namespace

ArrayReplaySource::ArrayReplaySource(const StreamSchema& schema, std::span<const double> data,
                                     bool paced, SessionId session_id)
    : schema_id_(schema.id()), signal_(dense_signal(schema)), data_(data.begin(), data.end()),
      paced_(paced), session_id_(session_id)
{
    const auto block_values = std::size_t{signal_.nominal_block_samples} * signal_.n_channels;
    if (signal_.kind != SignalKind::sampled ||
        signal_.device_tick_tracking != DeviceTickTracking::sample_counter || data.empty() ||
        data.size() % block_values != 0 || signal_.fs.numerator == 0 || signal_.fs.denominator == 0)
        throw std::invalid_argument(
            "Replay needs complete sampled blocks and sample-counter timing.");
    const auto limit = std::numeric_limits<HostTimeNs>::max();
    if (signal_.fs.denominator > limit / 1000000000ULL / signal_.nominal_block_samples)
        throw std::invalid_argument("Replay block duration overflows.");
    const auto numerator = 1000000000ULL * signal_.fs.denominator * signal_.nominal_block_samples;
    if (numerator % signal_.fs.numerator != 0)
        throw std::invalid_argument(
            "Replay block duration must be an integer number of nanoseconds.");
    period_ns_ = numerator / signal_.fs.numerator;
    if (period_ns_ == 0 || data.size() / block_values > (limit / 2) / period_ns_)
        throw std::invalid_argument("Replay duration is out of range.");
    timings_.resize(data.size() / block_values);
}

StreamStatus ArrayReplaySource::read(MutableFrame& frame) noexcept
{
    if (cancelled_.load(std::memory_order_acquire))
        return StreamStatus::stopped;
    if (next_ == timings_.size())
        return StreamStatus::end_of_stream;
    if (next_ == 0)
        epoch_ns_ = clock_.now_ns();
    const auto planned = epoch_ns_ + (next_ + 1) * period_ns_;
    if (paced_)
    {
        // Short waits bound cancellation even if wake races entry into wait_until.
        while (clock_.now_ns() < planned)
        {
            if (cancelled_.load(std::memory_order_acquire))
                return StreamStatus::stopped;
            clock_.wait_until(std::min(planned, clock_.now_ns() + HostTimeNs{1000000}));
        }
    }
    if (cancelled_.load(std::memory_order_acquire))
        return StreamStatus::stopped;
    const auto count = std::size_t{signal_.nominal_block_samples} * signal_.n_channels;
    if (frame.set_used_sizes(1, count * sizeof(double)) != StreamStatus::ok)
        return StreamStatus::invalid_frame;
    const auto sample = next_ * signal_.nominal_block_samples;
    frame.header() = {.session_id = session_id_,
                      .sequence = next_,
                      .source_tick = sample,
                      .schema_id = schema_id_,
                      .source_clock_domain = signal_.clock_domain,
                      .signal_block_count = 1,
                      .flags = FrameFlags::source_tick};
    frame.block_storage()[0] = {.sample_idx_start = sample,
                                .device_tick_start = sample,
                                .payload_byte_count = count * sizeof(double),
                                .signal_id = signal_.id,
                                .n_samples = signal_.nominal_block_samples,
                                .clock_sync = {.host_time_reference_ns = epoch_ns_,
                                               .device_tick_rate = signal_.fs,
                                               .clock_domain = signal_.clock_domain,
                                               .generation = 1,
                                               .flags = ClockSyncFlags::synchronized}};
    std::memcpy(frame.payload_storage().data(), data_.data() + next_ * count,
                count * sizeof(double));
    timings_[next_] = {sample, planned, clock_.now_ns()};
    ++next_;
    published_.store(next_, std::memory_order_release);
    return StreamStatus::ok;
}

void ArrayReplaySource::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
    clock_.wake();
}
StreamStatus ArrayReplaySource::reset() noexcept
{
    next_ = 0;
    epoch_ns_ = 0;
    published_.store(0, std::memory_order_release);
    cancelled_.store(false, std::memory_order_release);
    return StreamStatus::ok;
}
std::span<const ReplayBlockTiming> ArrayReplaySource::timings() const noexcept
{
    return std::span{timings_}.first(published_.load(std::memory_order_acquire));
}

NativeResultSink::NativeResultSink(const StreamSchema& schema, std::size_t capacity,
                                   std::size_t window_samples, std::size_t hop_samples,
                                   std::size_t output_channels)
    : schema_id_(schema.id()), signal_(dense_signal(schema)), window_samples_(window_samples),
      hop_samples_(hop_samples),
      output_channels_(window_samples ? output_channels : signal_.n_channels)
{
    if ((window_samples == 0) != (hop_samples == 0) || output_channels_ == 0 ||
        (window_samples &&
         (signal_.kind != SignalKind::sampled || hop_samples < signal_.max_block_samples)))
        throw std::invalid_argument(
            "Null capture needs sampled input, a window, and hop >= block size.");
    if (capacity == 0 || capacity > std::numeric_limits<std::size_t>::max() / output_channels_)
        throw std::invalid_argument("Result capacity must be positive and bounded.");
    values_.resize(capacity * output_channels_);
    timings_.resize(capacity);
}
void NativeResultSink::bind_runtime_clock(NativeClock& clock) noexcept
{
    clock_ = &clock;
}
StreamStatus NativeResultSink::consume(FrameView frame) noexcept
{
    if (frame.header.schema_id != schema_id_ || frame.blocks.size() != 1)
        return StreamStatus::invalid_frame;
    const auto& block = frame.blocks.front();
    const auto bytes = std::size_t{block.n_samples} * signal_.n_channels * sizeof(double);
    if (block.signal_id != signal_.id || block.n_samples > signal_.max_block_samples ||
        block.payload_byte_count != bytes || block.payload_offset > frame.payload.size() ||
        bytes > frame.payload.size() - block.payload_offset)
        return StreamStatus::invalid_frame;
    if (window_samples_)
    {
        if (block.sample_idx_start != samples_)
            return StreamStatus::discontinuity;
        samples_ += block.n_samples;
        if (samples_ < window_samples_ + next_ * hop_samples_)
            return StreamStatus::ok;
        if (next_ == timings_.size())
            return StreamStatus::output_limit;
        timings_[next_] = {next_, frame.header.source_tick, frame.header.host_received_ns,
                           clock_->now_ns()};
        ++next_;
        published_.store(next_, std::memory_order_release);
        return StreamStatus::ok;
    }
    if (block.n_samples > timings_.size() - next_)
        return StreamStatus::output_limit;
    std::memcpy(values_.data() + next_ * signal_.n_channels,
                frame.payload.data() + block.payload_offset, bytes);
    const auto delivered = clock_->now_ns();
    for (std::size_t i = 0; i < block.n_samples; ++i)
        timings_[next_ + i] = {block.sample_idx_start + i, frame.header.source_tick,
                               frame.header.host_received_ns, delivered};
    next_ += block.n_samples;
    published_.store(next_, std::memory_order_release);
    return StreamStatus::ok;
}
StreamStatus NativeResultSink::handle_discontinuity(const Discontinuity&) noexcept
{
    return StreamStatus::discontinuity;
}
StreamStatus NativeResultSink::flush() noexcept
{
    return StreamStatus::ok;
}
StreamStatus NativeResultSink::reset() noexcept
{
    next_ = 0;
    samples_ = 0;
    published_.store(0, std::memory_order_release);
    return StreamStatus::ok;
}
std::span<const NativeResultTiming> NativeResultSink::timings() const noexcept
{
    return std::span{timings_}.first(published_.load(std::memory_order_acquire));
}
std::span<const double> NativeResultSink::values(std::size_t count) const noexcept
{
    return std::span{values_}.first(std::min(count, published_.load(std::memory_order_acquire)) *
                                    output_channels_);
}
} // namespace neurale::streaming
