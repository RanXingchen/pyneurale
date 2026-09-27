/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/runtime.h>

#include "adapters.h"
#include "array_replay_bindings.h"
#include "observer_bridge.h"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using namespace pybind11::literals;

namespace
{

using namespace neurale::streaming;

struct SyntheticSignal
{
    SignalId id{};
    std::uint32_t n_samples{};
    std::uint32_t max_sample_count{};
    std::uint64_t payload_offset{};
    std::uint64_t payload_bytes{};
    DeviceTickTracking tick_tracking{DeviceTickTracking::unavailable};
    SignalKind kind{SignalKind::sampled};
    std::uint64_t observation_shift_ns{};
    std::uint32_t n_channels{};
    SignalDType dtype{SignalDType::int16};
    SignalLayout layout{SignalLayout::sample_major};
    ClockDomainId clock_domain{};
    RationalRate rate{};
};

/// One block exactly as the source wrote it, recorded at the moment it was
/// written.
///
/// This is the *source-side oracle*. A parity check that derives its expectation
/// from what a replay reports can only establish that the replay agrees with
/// itself: a block moved wholesale to another frame -- metadata, timing and
/// payload together -- still matches its own metadata. So the source states,
/// before anything downstream has seen it, which frame identity carried which
/// samples with which provenance, and a replay is judged against that.
struct ManifestBlock
{
    std::uint64_t signal_id{};
    std::uint64_t sample_idx_start{};
    std::uint64_t last_sample_idx{};
    std::uint64_t n_samples{};
    std::uint64_t device_tick_start{};
    std::uint64_t observation_time_start_ns{};
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    std::uint64_t payload_digest{};
    std::uint64_t clock_sync_device_tick_reference{};
    std::uint64_t clock_sync_host_time_reference_ns{};
    std::uint64_t clock_sync_rate_numerator{};
    std::uint64_t clock_sync_rate_denominator{};
    std::uint64_t clock_sync_uncertainty_ns{};
    std::uint32_t clock_sync_clock_domain{};
    std::uint32_t clock_sync_generation{};
    std::uint32_t clock_sync_flags{};
};

/// One frame exactly as the source wrote it. `blocks` counts entries in the
/// flat block storage starting at `block_first`.
struct ManifestFrame
{
    std::uint64_t sequence{};
    std::uint64_t host_received_ns{};
    std::uint64_t source_tick{};
    std::uint64_t valid_until_ns{};
    std::uint64_t payload_byte_count{};
    std::uint64_t payload_digest{};
    std::uint32_t schema_id{};
    std::uint32_t clock_domain{};
    std::uint32_t flags{};
    std::uint32_t blocks{};
    std::size_t block_first{};
};

/// FNV-1a over the bytes the source actually wrote.
///
/// Chosen because both ends must compute it and one of them is a test: three
/// lines of Python reproduce it exactly, with no table and no dependency, so
/// the digest in the manifest can be checked against the bytes the formula
/// predicts without either side borrowing the other's code.
[[nodiscard]] std::uint64_t payload_digest(std::span<const std::byte> bytes) noexcept
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : bytes)
    {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

/// The value one sample of one channel carries under `payload_pattern`.
///
/// The default payload is all zeroes, which is the right default for tests
/// about frame plumbing and the wrong one for any test about the *data*: a
/// transposed, misaligned, or wrongly attributed block of zeroes is still a
/// block of zeroes, so a payload comparison over it passes without meaning
/// anything. This pattern makes every (signal, sample, channel) position carry
/// a different, exactly representable number, so the same comparison fails on
/// any of those mistakes. It is the same formula
/// `tests/unit/recording/conftest.py::sample_values` states in Python, which is
/// what lets a Python test say what a native session should have recorded.
[[nodiscard]] double pattern_value(SignalId signal_id, std::uint64_t sample_idx,
                                   std::uint32_t channel) noexcept
{
    const auto raw = sample_idx * 16U + channel + static_cast<std::uint64_t>(signal_id) * 4096U;
    return static_cast<double>(raw);
}

/// Store *value* as one scalar of *dtype*, truncating exactly as a narrowing
/// cast does, so a Python `astype` of the same formula produces the same bytes.
void store_scalar(std::span<std::byte> destination, SignalDType dtype, double value) noexcept
{
    switch (dtype)
    {
    case SignalDType::int16:
    {
        const auto stored = static_cast<std::int16_t>(static_cast<std::int64_t>(value));
        std::memcpy(destination.data(), &stored, sizeof(stored));
        return;
    }
    case SignalDType::int32:
    {
        const auto stored = static_cast<std::int32_t>(static_cast<std::int64_t>(value));
        std::memcpy(destination.data(), &stored, sizeof(stored));
        return;
    }
    case SignalDType::float32:
    {
        const auto stored = static_cast<float>(value);
        std::memcpy(destination.data(), &stored, sizeof(stored));
        return;
    }
    case SignalDType::float64:
    {
        std::memcpy(destination.data(), &value, sizeof(value));
        return;
    }
    }
}

class SyntheticNativeSource final : public NativeFrameSource
{
  public:
    SyntheticNativeSource(const StreamSchema& schema, std::size_t n_frames, SessionId session_id,
                          std::optional<std::size_t> fail_at,
                          std::optional<std::size_t> sequence_gap_at, bool payload_pattern,
                          bool vary_block_samples)
        : schema_id_(schema.id()), session_id_(session_id), n_frames_(n_frames), fail_at_(fail_at),
          sequence_gap_at_(sequence_gap_at), payload_pattern_(payload_pattern),
          vary_block_samples_(vary_block_samples), n_signals_(schema.signals().size()),
          signals_(std::make_unique<SyntheticSignal[]>(n_signals_)),
          next_sample_idx_(std::make_unique<std::uint64_t[]>(n_signals_))
    {
        std::uint64_t payload_offset = 0;
        for (std::size_t i = 0; i < n_signals_; ++i)
        {
            const auto& signal = schema.signals()[i];
            // A varying source is bounded by the declared maximum rather than
            // the nominal one, so every sizing question below -- the payload
            // budget, the sample-index reach, the feature time reach -- is asked
            // about the largest frame this source can produce.
            const auto block_samples =
                vary_block_samples ? signal.max_block_samples : signal.nominal_block_samples;
            if (block_samples == 0)
            {
                throw std::invalid_argument("synthetic signal has no samples per block");
            }
            const auto n_scalars = static_cast<std::uint64_t>(signal.n_channels) * block_samples;
            const auto payload_bytes = n_scalars * signal_dtype_size(signal.dtype);
            if (payload_bytes > std::numeric_limits<std::uint64_t>::max() - payload_offset)
            {
                throw std::overflow_error("synthetic frame payload overflows uint64_t");
            }
            if (n_frames != 0 &&
                n_frames - 1 > std::numeric_limits<SampleIndex>::max() / block_samples)
            {
                throw std::overflow_error("synthetic sample index overflows uint64_t");
            }
            std::uint64_t observation_shift_ns = 0;
            if (signal.kind == SignalKind::feature)
            {
                const auto* descriptor = schema.feature_sets().find(signal.feature_set_id);
                if (descriptor == nullptr)
                {
                    throw std::invalid_argument("synthetic feature signal has no descriptor");
                }
                observation_shift_ns = descriptor->shift_ns;
                if (n_frames != 0 && n_frames - 1 > std::numeric_limits<HostTimeNs>::max() /
                                                        block_samples / observation_shift_ns)
                {
                    throw std::overflow_error(
                        "synthetic feature observation time overflows uint64_t");
                }
            }
            signals_[i] = SyntheticSignal{
                .id = signal.id,
                .n_samples = signal.nominal_block_samples,
                .max_sample_count = signal.max_block_samples,
                .payload_offset = payload_offset,
                .payload_bytes = payload_bytes,
                .tick_tracking = signal.device_tick_tracking,
                .kind = signal.kind,
                .observation_shift_ns = observation_shift_ns,
                .n_channels = signal.n_channels,
                .dtype = signal.dtype,
                .layout = signal.layout,
                .clock_domain = signal.clock_domain,
                .rate = signal.fs,
            };
            payload_offset += payload_bytes;
        }
        if (payload_offset > std::numeric_limits<std::size_t>::max())
        {
            throw std::overflow_error("synthetic frame payload overflows size_t");
        }
        payload_bytes_ = static_cast<std::size_t>(payload_offset);
        if (payload_pattern_)
        {
            // Sized once, here, because `read()` runs on the realtime source
            // thread: recording what was written must not allocate, grow a
            // container, or take a lock. Reserving for the whole run is what
            // makes that possible, and is only done when a manifest was asked
            // for.
            manifest_frames_.resize(n_frames_);
            manifest_blocks_.resize(n_frames_ * n_signals_);
        }
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        const auto frame_idx = frame_idx_.load(std::memory_order_relaxed);
        if (fail_at_.has_value() && frame_idx == *fail_at_)
        {
            return StreamStatus::source_failure;
        }
        if (frame_idx == n_frames_)
        {
            return StreamStatus::end_of_stream;
        }
        if (frame.block_storage().size() < n_signals_ ||
            frame.payload_storage().size() < payload_bytes_)
        {
            return StreamStatus::invalid_frame;
        }

        const auto sequence = sequence_gap_at_.has_value() && frame_idx >= *sequence_gap_at_
                                  ? frame_idx + 1
                                  : frame_idx;
        frame.header() = FrameHeader{
            .session_id = session_id_,
            .sequence = sequence,
            .host_received_ns = frame_idx,
            // Frame-level provenance is only stamped under the pattern, for the
            // same reason the payload is: a field that is zero in every frame
            // cannot tell a preserved value from a dropped one, so a parity
            // assertion over it would pass whatever the chain did with it.
            .source_tick = payload_pattern_ ? frame_idx * 3 + 1 : 0,
            .schema_id = schema_id_,
            .source_clock_domain =
                payload_pattern_ && n_signals_ != 0 ? signals_[0].clock_domain : 0,
            .flags = payload_pattern_ ? FrameFlags::source_tick : FrameFlags::none,
        };

        std::uint64_t used_bytes = 0;
        for (std::size_t i = 0; i < n_signals_; ++i)
        {
            const auto& signal = signals_[i];
            const auto n_samples = block_samples(i, frame_idx);
            const auto sample_idx = static_cast<SampleIndex>(next_sample_idx_[i]);
            const auto payload_bytes = static_cast<std::uint64_t>(n_samples) * signal.n_channels *
                                       signal_dtype_size(signal.dtype);
            auto& block = frame.block_storage()[i];
            block = SignalBlockHeader{
                .sample_idx_start = sample_idx,
                .device_tick_start =
                    signal.tick_tracking == DeviceTickTracking::sample_counter ? sample_idx : 0,
                .observation_time_start_ns = signal.kind == SignalKind::feature
                                                 ? sample_idx * signal.observation_shift_ns
                                                 : 0,
                .payload_offset = used_bytes,
                .payload_byte_count = payload_bytes,
                .signal_id = signal.id,
                .n_samples = n_samples,
            };
            if (payload_pattern_)
            {
                block.clock_sync = ClockSyncSnapshot{
                    .device_tick_reference = sample_idx,
                    .host_time_reference_ns = frame_idx * 1'000 + signal.id,
                    .device_tick_rate = signal.rate,
                    .uncertainty_ns = static_cast<HostTimeNs>(signal.id) * 7 + 1,
                    .clock_domain = signal.clock_domain,
                    .generation = static_cast<std::uint32_t>(frame_idx + 1),
                    .flags = ClockSyncFlags::synchronized,
                };
            }
            used_bytes += payload_bytes;
            next_sample_idx_[i] = sample_idx + n_samples;
        }

        std::fill_n(frame.payload_storage().begin(), static_cast<std::size_t>(used_bytes),
                    std::byte{});
        if (payload_pattern_)
        {
            write_pattern(frame.payload_storage(), frame.block_storage().first(n_signals_));
        }
        const auto status = frame.set_used_sizes(n_signals_, static_cast<std::size_t>(used_bytes));
        if (status == StreamStatus::ok)
        {
            if (payload_pattern_)
            {
                record_manifest(frame, frame_idx);
            }
            frame_idx_.store(frame_idx + 1, std::memory_order_release);
        }
        return status;
    }

    StreamStatus reset() noexcept override
    {
        cancelled_.store(false, std::memory_order_release);
        frame_idx_.store(0, std::memory_order_relaxed);
        for (std::size_t i = 0; i < n_signals_; ++i)
        {
            next_sample_idx_[i] = 0;
        }
        return StreamStatus::ok;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }

    [[nodiscard]] std::size_t frames_emitted() const noexcept
    {
        return frame_idx_.load(std::memory_order_acquire);
    }

    /// What this source actually wrote, frame by frame: the oracle.
    ///
    /// Built here, on the caller's thread, out of storage the realtime thread
    /// only ever filled in. Empty unless a pattern was asked for, because
    /// without one there is nothing in a frame worth stating.
    [[nodiscard]] py::list manifest() const
    {
        py::list frames;
        const auto produced =
            std::min(frame_idx_.load(std::memory_order_acquire), manifest_frames_.size());
        for (std::size_t i = 0; i < produced; ++i)
        {
            const auto& entry = manifest_frames_[i];
            py::list blocks;
            for (std::uint32_t child = 0; child < entry.blocks; ++child)
            {
                const auto& block = manifest_blocks_[entry.block_first + child];
                blocks.append(py::dict(
                    "signal_id"_a = block.signal_id, "sample_idx_start"_a = block.sample_idx_start,
                    "last_sample_idx"_a = block.last_sample_idx, "n_samples"_a = block.n_samples,
                    "device_tick_start"_a = block.device_tick_start,
                    "observation_time_start_ns"_a = block.observation_time_start_ns,
                    "payload_offset"_a = block.payload_offset,
                    "payload_byte_count"_a = block.payload_byte_count,
                    "payload_digest"_a = block.payload_digest,
                    "clock_sync_device_tick_reference"_a = block.clock_sync_device_tick_reference,
                    "clock_sync_host_time_reference_ns"_a = block.clock_sync_host_time_reference_ns,
                    "clock_sync_rate_numerator"_a = block.clock_sync_rate_numerator,
                    "clock_sync_rate_denominator"_a = block.clock_sync_rate_denominator,
                    "clock_sync_uncertainty_ns"_a = block.clock_sync_uncertainty_ns,
                    "clock_sync_clock_domain"_a = block.clock_sync_clock_domain,
                    "clock_sync_generation"_a = block.clock_sync_generation,
                    "clock_sync_flags"_a = block.clock_sync_flags));
            }
            frames.append(py::dict(
                "sequence"_a = entry.sequence, "host_received_ns"_a = entry.host_received_ns,
                "source_tick"_a = entry.source_tick, "valid_until_ns"_a = entry.valid_until_ns,
                "schema_id"_a = entry.schema_id, "clock_domain"_a = entry.clock_domain,
                "flags"_a = entry.flags, "payload_byte_count"_a = entry.payload_byte_count,
                "payload_digest"_a = entry.payload_digest, "blocks"_a = blocks));
        }
        return frames;
    }

  private:
    /// How many samples one signal's block carries in this frame.
    ///
    /// A source that emits its nominal block every time never exercises the
    /// variable-size frame the schema's capacities allow, so `vary_block_samples`
    /// walks the whole declared range instead -- and, because the walk depends on
    /// the signal's position as well as the frame's, no two signals in one frame
    /// change size together.
    [[nodiscard]] std::uint32_t block_samples(std::size_t idx, std::size_t frame_idx) const noexcept
    {
        const auto& signal = signals_[idx];
        if (!vary_block_samples_)
        {
            return signal.n_samples;
        }
        return 1U + static_cast<std::uint32_t>((frame_idx + idx) % signal.max_sample_count);
    }

    /// Fill this frame's payload with the position-encoding pattern, storing
    /// each signal's block in the layout that signal declares. Writing every
    /// block sample-major regardless would make a channel-major signal's
    /// recording silently transposed, which is precisely the mistake the
    /// pattern exists to catch.
    void write_pattern(std::span<std::byte> payload,
                       std::span<const SignalBlockHeader> blocks) const noexcept
    {
        for (std::size_t i = 0; i < blocks.size(); ++i)
        {
            const auto& signal = signals_[i];
            const auto& block = blocks[i];
            const auto width = signal.n_channels;
            const auto count = block.n_samples;
            const auto scalar_bytes = signal_dtype_size(signal.dtype);
            for (std::uint32_t sample = 0; sample < count; ++sample)
            {
                for (std::uint32_t channel = 0; channel < width; ++channel)
                {
                    const auto scalar = signal.layout == SignalLayout::channel_major
                                            ? static_cast<std::uint64_t>(channel) * count + sample
                                            : static_cast<std::uint64_t>(sample) * width + channel;
                    const auto offset = block.payload_offset + scalar * scalar_bytes;
                    store_scalar(
                        payload.subspan(static_cast<std::size_t>(offset), scalar_bytes),
                        signal.dtype,
                        pattern_value(signal.id, block.sample_idx_start + sample, channel));
                }
            }
        }
    }

    /// Record what this frame carried, into storage reserved at construction.
    ///
    /// Runs on the realtime source thread, so it allocates nothing, takes no
    /// lock and touches no Python: every index below is within storage the
    /// constructor already sized for the whole run.
    void record_manifest(MutableFrame& frame, std::size_t frame_idx) noexcept
    {
        if (frame_idx >= manifest_frames_.size())
        {
            return;
        }
        const auto blocks = frame.block_storage().first(n_signals_);
        const auto payload = frame.payload_storage();
        auto& entry = manifest_frames_[frame_idx];
        entry.sequence = frame.header().sequence;
        entry.host_received_ns = frame.header().host_received_ns;
        entry.source_tick = frame.header().source_tick;
        entry.valid_until_ns = frame.header().valid_until_ns;
        entry.schema_id = frame.header().schema_id;
        entry.clock_domain = frame.header().source_clock_domain;
        entry.flags = static_cast<std::uint32_t>(frame.header().flags);
        entry.blocks = static_cast<std::uint32_t>(n_signals_);
        entry.block_first = frame_idx * n_signals_;

        std::uint64_t total = 0;
        for (std::size_t i = 0; i < blocks.size(); ++i)
        {
            const auto& block = blocks[i];
            const auto bytes = payload.subspan(static_cast<std::size_t>(block.payload_offset),
                                               static_cast<std::size_t>(block.payload_byte_count));
            manifest_blocks_[entry.block_first + i] = ManifestBlock{
                .signal_id = block.signal_id,
                .sample_idx_start = block.sample_idx_start,
                .last_sample_idx = block.last_sample_idx,
                .n_samples = block.n_samples,
                .device_tick_start = block.device_tick_start,
                .observation_time_start_ns = block.observation_time_start_ns,
                .payload_offset = block.payload_offset,
                .payload_byte_count = block.payload_byte_count,
                .payload_digest = payload_digest(bytes),
                .clock_sync_device_tick_reference = block.clock_sync.device_tick_reference,
                .clock_sync_host_time_reference_ns = block.clock_sync.host_time_reference_ns,
                .clock_sync_rate_numerator = block.clock_sync.device_tick_rate.numerator,
                .clock_sync_rate_denominator = block.clock_sync.device_tick_rate.denominator,
                .clock_sync_uncertainty_ns = block.clock_sync.uncertainty_ns,
                .clock_sync_clock_domain = block.clock_sync.clock_domain,
                .clock_sync_generation = block.clock_sync.generation,
                .clock_sync_flags = static_cast<std::uint32_t>(block.clock_sync.flags),
            };
            total += block.payload_byte_count;
        }
        entry.payload_byte_count = total;
        entry.payload_digest = payload_digest(payload.first(static_cast<std::size_t>(total)));
    }

    SchemaId schema_id_{};
    SessionId session_id_{};
    std::size_t n_frames_{};
    std::optional<std::size_t> fail_at_;
    std::optional<std::size_t> sequence_gap_at_;
    bool payload_pattern_{};
    bool vary_block_samples_{};
    std::size_t n_signals_{};
    std::unique_ptr<SyntheticSignal[]> signals_;
    /// Running sample position per signal, so a varying block size still
    /// produces one contiguous, gapless sample sequence per signal.
    std::unique_ptr<std::uint64_t[]> next_sample_idx_;
    std::vector<ManifestFrame> manifest_frames_;
    std::vector<ManifestBlock> manifest_blocks_;
    std::size_t payload_bytes_{};
    std::atomic<std::size_t> frame_idx_{};
    std::atomic<bool> cancelled_{};
};

class IdentityNativeProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {context.input_schema.clone(),
                context.input_schema.clone(),
                1,
                0,
                true,
                ProcessorResourceBounds{.frame_pool_leases = 1}};
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        process_count_.fetch_add(1, std::memory_order_relaxed);
        return output.publish_input();
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuity_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus flush(FrameEmitter&) noexcept override
    {
        flush_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        process_count_.store(0, std::memory_order_relaxed);
        discontinuity_count_.store(0, std::memory_order_relaxed);
        flush_count_.store(0, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::uint64_t process_count() const noexcept
    {
        return process_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t discontinuity_count() const noexcept
    {
        return discontinuity_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t flush_count() const noexcept
    {
        return flush_count_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<std::uint64_t> process_count_{};
    std::atomic<std::uint64_t> discontinuity_count_{};
    std::atomic<std::uint64_t> flush_count_{};
};

class CountingNativeConsumer final : public NativeFrameConsumer
{
  public:
    explicit CountingNativeConsumer(std::optional<std::uint64_t> fail_at) noexcept
        : fail_at_(fail_at)
    {
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        const auto attempt = consume_attempts_.fetch_add(1, std::memory_order_relaxed);
        if (fail_at_.has_value() && attempt == *fail_at_)
        {
            return StreamStatus::consumer_failure;
        }
        frame_count_.fetch_add(1, std::memory_order_relaxed);
        last_sequence_.store(frame.header.sequence, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuity_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        flush_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        consume_attempts_.store(0, std::memory_order_relaxed);
        frame_count_.store(0, std::memory_order_relaxed);
        discontinuity_count_.store(0, std::memory_order_relaxed);
        flush_count_.store(0, std::memory_order_relaxed);
        last_sequence_.store(0, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::uint64_t frame_count() const noexcept
    {
        return frame_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t discontinuity_count() const noexcept
    {
        return discontinuity_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t flush_count() const noexcept
    {
        return flush_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t last_sequence() const noexcept
    {
        return last_sequence_.load(std::memory_order_relaxed);
    }

  private:
    std::optional<std::uint64_t> fail_at_;
    std::atomic<std::uint64_t> consume_attempts_{};
    std::atomic<std::uint64_t> frame_count_{};
    std::atomic<std::uint64_t> discontinuity_count_{};
    std::atomic<std::uint64_t> flush_count_{};
    std::atomic<std::uint64_t> last_sequence_{};
};

class CountingNativeObserver final : public NativeObserver
{
  public:
    explicit CountingNativeObserver(std::optional<std::uint64_t> fail_at) noexcept
        : fail_at_(fail_at)
    {
    }

    StreamStatus observe(FrameView frame) noexcept override
    {
        const auto attempt = attempts_.fetch_add(1, std::memory_order_relaxed);
        if (fail_at_.has_value() && attempt == *fail_at_)
        {
            return StreamStatus::consumer_failure;
        }
        frame_count_.fetch_add(1, std::memory_order_relaxed);
        last_sequence_.store(frame.header.sequence, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuity_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus flush() noexcept override
    {
        flush_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        attempts_.store(0, std::memory_order_relaxed);
        frame_count_.store(0, std::memory_order_relaxed);
        discontinuity_count_.store(0, std::memory_order_relaxed);
        flush_count_.store(0, std::memory_order_relaxed);
        last_sequence_.store(0, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}

    std::uint64_t frame_count() const noexcept
    {
        return frame_count_.load();
    }
    std::uint64_t discontinuity_count() const noexcept
    {
        return discontinuity_count_.load();
    }
    std::uint64_t flush_count() const noexcept
    {
        return flush_count_.load();
    }
    std::uint64_t last_sequence() const noexcept
    {
        return last_sequence_.load();
    }

  private:
    std::optional<std::uint64_t> fail_at_;
    std::atomic<std::uint64_t> attempts_{};
    std::atomic<std::uint64_t> frame_count_{};
    std::atomic<std::uint64_t> discontinuity_count_{};
    std::atomic<std::uint64_t> flush_count_{};
    std::atomic<std::uint64_t> last_sequence_{};
};

class RecordingNativeSafetyController final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason reason) noexcept override
    {
        last_reason_.store(reason, std::memory_order_relaxed);
        inhibit_count_.fetch_add(1, std::memory_order_relaxed);
        return fail_next_inhibit_.exchange(false, std::memory_order_relaxed)
                   ? StreamStatus::safety_failure
                   : StreamStatus::ok;
    }

    StreamStatus release() noexcept override
    {
        release_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    void fail_next_inhibit() noexcept
    {
        fail_next_inhibit_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t inhibit_count() const noexcept
    {
        return inhibit_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t release_count() const noexcept
    {
        return release_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] SafetyReason last_reason() const noexcept
    {
        return last_reason_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<std::uint64_t> inhibit_count_{};
    std::atomic<std::uint64_t> release_count_{};
    std::atomic<SafetyReason> last_reason_{SafetyReason::startup};
    std::atomic<bool> fail_next_inhibit_{};
};

[[nodiscard]] std::vector<SignalSchema> copy_signals(const StreamSchema& schema)
{
    return {schema.signals().begin(), schema.signals().end()};
}

[[nodiscard]] std::vector<FeatureSetDescriptor> copy_feature_sets(const StreamSchema& schema)
{
    const auto values = schema.feature_sets().descriptors();
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<UnitDescriptor> copy_units(const StreamSchema& schema)
{
    const auto values = schema.units().units();
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<FeatureSetDescriptor>
copy_feature_sets(const FeatureSetDescriptorRegistry& registry)
{
    const auto values = registry.descriptors();
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<UnitDescriptor> copy_units(const UnitRegistry& registry)
{
    const auto values = registry.units();
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<FaultRecord> copy_fault_history(const NativeStreamRunner& runner)
{
    std::vector<FaultRecord> history(runner.fault_history_capacity());
    history.resize(runner.copy_fault_history(history));
    return history;
}

[[nodiscard]] std::vector<ObserverDropRange>
copy_observer_drop_ranges(const NativeStreamRunner& runner, ObserverId id)
{
    const auto stats = runner.observer_stats(id);
    if (!stats.has_value())
    {
        return {};
    }
    std::vector<ObserverDropRange> ranges(static_cast<std::size_t>(stats->drop_range_count));
    ranges.resize(runner.copy_observer_drop_ranges(id, ranges));
    return ranges;
}

void bind_enums(py::module_& module)
{
    py::enum_<SignalDType>(module, "SignalDType")
        .value("INT16", SignalDType::int16)
        .value("INT32", SignalDType::int32)
        .value("FLOAT32", SignalDType::float32)
        .value("FLOAT64", SignalDType::float64);
    py::enum_<SignalLayout>(module, "SignalLayout")
        .value("SAMPLE_MAJOR", SignalLayout::sample_major)
        .value("CHANNEL_MAJOR", SignalLayout::channel_major);
    py::enum_<DeviceTickTracking>(module, "DeviceTickTracking")
        .value("UNAVAILABLE", DeviceTickTracking::unavailable)
        .value("SAMPLE_COUNTER", DeviceTickTracking::sample_counter);
    py::enum_<PhysicalUnit>(module, "PhysicalUnit")
        .value("UNSPECIFIED", PhysicalUnit::unspecified)
        .value("VOLTS", PhysicalUnit::volts)
        .value("AMPERES", PhysicalUnit::amperes)
        .value("DIMENSIONLESS", PhysicalUnit::dimensionless);
    py::enum_<SignalKind>(module, "SignalKind")
        .value("SAMPLED", SignalKind::sampled)
        .value("EVENT", SignalKind::event)
        .value("FEATURE", SignalKind::feature)
        .value("SPIKE", SignalKind::spike);
    py::enum_<ObservationTiming>(module, "ObservationTiming")
        .value("NOT_APPLICABLE", ObservationTiming::not_applicable)
        .value("REGULAR", ObservationTiming::regular)
        .value("IRREGULAR", ObservationTiming::irregular);
    py::enum_<FeatureTimestampReference>(module, "FeatureTimestampReference")
        .value("WINDOW_CENTER", FeatureTimestampReference::window_center);
    py::enum_<ClockSyncFlags>(module, "ClockSyncFlags")
        .value("NONE", ClockSyncFlags::none)
        .value("SYNCHRONIZED", ClockSyncFlags::synchronized);
    py::enum_<RuntimeState>(module, "RuntimeState")
        .value("CREATED", RuntimeState::created)
        .value("PREPARED", RuntimeState::prepared)
        .value("RUNNING", RuntimeState::running)
        .value("STOPPING", RuntimeState::stopping)
        .value("STOPPED", RuntimeState::stopped)
        .value("FAILED", RuntimeState::failed);
    py::enum_<StreamStatus>(module, "StreamStatus")
        .value("OK", StreamStatus::ok)
        .value("END_OF_STREAM", StreamStatus::end_of_stream)
        .value("WOULD_BLOCK", StreamStatus::would_block)
        .value("BUFFER_EXHAUSTED", StreamStatus::buffer_exhausted)
        .value("QUEUE_OVERFLOW", StreamStatus::queue_overflow)
        .value("DISCONTINUITY", StreamStatus::discontinuity)
        .value("INVALID_FRAME", StreamStatus::invalid_frame)
        .value("SOURCE_FAILURE", StreamStatus::source_failure)
        .value("PROCESSOR_FAILURE", StreamStatus::processor_failure)
        .value("CONSUMER_FAILURE", StreamStatus::consumer_failure)
        .value("ACTUATOR_FAILURE", StreamStatus::actuator_failure)
        .value("OBSERVER_OVERRUN", StreamStatus::observer_overrun)
        .value("STOPPED", StreamStatus::stopped)
        .value("INVALID_STATE", StreamStatus::invalid_state)
        .value("OUTPUT_LIMIT", StreamStatus::output_limit)
        .value("DEADLINE_EXCEEDED", StreamStatus::deadline_exceeded)
        .value("SAFETY_FAILURE", StreamStatus::safety_failure)
        .value("REALTIME_CONFIGURATION_FAILED", StreamStatus::realtime_configuration_failed);
    py::enum_<FaultCode>(module, "FaultCode")
        .value("NONE", FaultCode::none)
        .value("SOURCE_READ", FaultCode::source_read)
        .value("SOURCE_RESET", FaultCode::source_reset)
        .value("QUEUE_OVERRUN", FaultCode::queue_overrun)
        .value("RUNTIME_START", FaultCode::runtime_start)
        .value("SOURCE_STALL", FaultCode::source_stall)
        .value("INGRESS_DWELL_TIMEOUT", FaultCode::ingress_dwell_timeout)
        .value("PROCESSOR_DEADLINE", FaultCode::processor_deadline)
        .value("ACTUATOR_INPUT_STALE", FaultCode::actuator_input_stale)
        .value("OUTPUT_STALE", FaultCode::output_stale)
        .value("SHUTDOWN_TIMEOUT", FaultCode::shutdown_timeout)
        .value("SAFETY_CONTROLLER_FAILURE", FaultCode::safety_controller_failure)
        .value("FRAME_POOL_EXHAUSTED", FaultCode::frame_pool_exhausted)
        .value("CONTINUITY_VALIDATION", FaultCode::continuity_validation)
        .value("PROCESSOR_PREPARE", FaultCode::processor_prepare)
        .value("PROCESSOR_DISCONTINUITY", FaultCode::processor_discontinuity)
        .value("PROCESSOR_PROCESS", FaultCode::processor_process)
        .value("PROCESSOR_FLUSH", FaultCode::processor_flush)
        .value("PROCESSOR_RESET", FaultCode::processor_reset)
        .value("OUTPUT_CONTRACT", FaultCode::output_contract)
        .value("OUTPUT_VALIDATION", FaultCode::output_validation)
        .value("OUTPUT_POOL_EXHAUSTED", FaultCode::output_pool_exhausted)
        .value("CONSUMER_DISCONTINUITY", FaultCode::consumer_discontinuity)
        .value("CONSUMER_CONSUME", FaultCode::consumer_consume)
        .value("CONSUMER_FLUSH", FaultCode::consumer_flush)
        .value("CONSUMER_RESET", FaultCode::consumer_reset)
        .value("ACTUATOR_QUEUE_OVERRUN", FaultCode::actuator_queue_overrun)
        .value("ACTUATOR_DEADLINE", FaultCode::actuator_deadline)
        .value("ACTUATOR_WRITE", FaultCode::actuator_write)
        .value("ACTUATOR_FLUSH", FaultCode::actuator_flush)
        .value("ACTUATOR_RESET", FaultCode::actuator_reset)
        .value("OBSERVER_DISPATCH_OVERRUN", FaultCode::observer_dispatch_overrun)
        .value("CRITICAL_OBSERVER_OVERRUN", FaultCode::critical_observer_overrun)
        .value("CRITICAL_OBSERVER_FAILURE", FaultCode::critical_observer_failure)
        .value("REALTIME_CONFIGURATION", FaultCode::realtime_configuration);
    py::enum_<FaultStage>(module, "FaultStage")
        .value("SOURCE", FaultStage::source)
        .value("CONTINUITY", FaultStage::continuity)
        .value("PROCESSOR", FaultStage::processor)
        .value("OUTPUT", FaultStage::output)
        .value("CONSUMER", FaultStage::consumer)
        .value("ACTUATOR", FaultStage::actuator)
        .value("OBSERVER", FaultStage::observer)
        .value("RUNTIME", FaultStage::runtime);
    py::enum_<SafetyReason>(module, "SafetyReason")
        .value("STARTUP", SafetyReason::startup)
        .value("EXPLICIT_STOP", SafetyReason::explicit_stop)
        .value("END_OF_STREAM", SafetyReason::end_of_stream)
        .value("SOURCE_STALL", SafetyReason::source_stall)
        .value("INGRESS_DWELL", SafetyReason::ingress_dwell)
        .value("PROCESSOR_DEADLINE", SafetyReason::processor_deadline)
        .value("OUTPUT_STALE", SafetyReason::output_stale)
        .value("SHUTDOWN_TIMEOUT", SafetyReason::shutdown_timeout)
        .value("ACTUATOR_FAILURE", SafetyReason::actuator_failure)
        .value("CRITICAL_OBSERVER_FAILURE", SafetyReason::critical_observer_failure)
        .value("RUNTIME_FAULT", SafetyReason::runtime_fault);
    py::enum_<ObserverDropPolicy>(module, "ObserverDropPolicy")
        .value("DROP_OLDEST", ObserverDropPolicy::drop_oldest)
        .value("LATEST_VALUE", ObserverDropPolicy::latest_value)
        .value("DROP_NEWEST", ObserverDropPolicy::drop_newest)
        .value("FAULT", ObserverDropPolicy::fault);
    py::enum_<RealtimeConfigMode>(module, "RealtimeConfigMode")
        .value("DISABLED", RealtimeConfigMode::disabled)
        .value("BEST_EFFORT", RealtimeConfigMode::best_effort)
        .value("STRICT", RealtimeConfigMode::strict);
    py::enum_<RealtimeSchedulingPolicy>(module, "RealtimeSchedulingPolicy")
        .value("NORMAL", RealtimeSchedulingPolicy::normal)
        .value("FIFO", RealtimeSchedulingPolicy::fifo)
        .value("ROUND_ROBIN", RealtimeSchedulingPolicy::round_robin);
    py::enum_<RealtimeFeature>(module, "RealtimeFeature")
        .value("NONE", RealtimeFeature::none)
        .value("THREAD_NAME", RealtimeFeature::thread_name)
        .value("CPU_AFFINITY", RealtimeFeature::cpu_affinity)
        .value("SCHEDULING", RealtimeFeature::scheduling)
        .value("PRIORITY", RealtimeFeature::priority)
        .value("STACK_SIZE", RealtimeFeature::stack_size)
        .value("MEMORY_LOCK", RealtimeFeature::memory_lock)
        .value("STACK_PREFAULT", RealtimeFeature::stack_prefault)
        .value("POOL_PREFAULT", RealtimeFeature::pool_prefault);
}

void bind_schema_and_config(py::module_& module)
{
    py::class_<RealtimeThreadConfig>(module, "RealtimeThreadConfig", py::is_final())
        .def(py::init<>())
        .def_property(
            "name",
            [](const RealtimeThreadConfig& value) { return std::string{value.name.data()}; },
            [](RealtimeThreadConfig& value, const std::string& name)
            { value.set_name(name.c_str()); })
        .def_readwrite("cpu_affinity_mask", &RealtimeThreadConfig::cpu_affinity_mask)
        .def_readwrite("scheduling_policy", &RealtimeThreadConfig::scheduling_policy)
        .def_readwrite("priority", &RealtimeThreadConfig::priority)
        .def_readwrite("stack_size", &RealtimeThreadConfig::stack_size)
        .def_readwrite("prefault_stack_bytes", &RealtimeThreadConfig::prefault_stack_bytes)
        .def_property_readonly("requested_features", &RealtimeThreadConfig::requested_features)
        .def("validate", &RealtimeThreadConfig::validate);

    py::class_<RealtimePlatformConfig>(module, "RealtimePlatformConfig", py::is_final())
        .def(py::init<>())
        .def_readwrite("mode", &RealtimePlatformConfig::mode)
        .def_readwrite("lock_memory", &RealtimePlatformConfig::lock_memory)
        .def_readwrite("prefault_pools", &RealtimePlatformConfig::prefault_pools)
        .def_readwrite("acquisition", &RealtimePlatformConfig::acquisition)
        .def_readwrite("processing", &RealtimePlatformConfig::processing)
        .def_readwrite("actuator", &RealtimePlatformConfig::actuator)
        .def_readwrite("observer_dispatch", &RealtimePlatformConfig::observer_dispatch)
        .def_readwrite("watchdog", &RealtimePlatformConfig::watchdog)
        .def("validate", &RealtimePlatformConfig::validate);

    py::class_<RationalRate>(module, "RationalRate", py::is_final())
        .def(py::init<std::uint64_t, std::uint64_t>(), py::arg("numerator"),
             py::arg("denominator") = 1)
        .def_readonly("numerator", &RationalRate::numerator)
        .def_readonly("denominator", &RationalRate::denominator);

    py::class_<ClockSyncSnapshot>(module, "ClockSyncSnapshot", py::is_final())
        .def(py::init<>())
        .def_readwrite("device_tick_reference", &ClockSyncSnapshot::device_tick_reference)
        .def_readwrite("host_time_reference_ns", &ClockSyncSnapshot::host_time_reference_ns)
        .def_readwrite("device_tick_rate", &ClockSyncSnapshot::device_tick_rate)
        .def_readwrite("uncertainty_ns", &ClockSyncSnapshot::uncertainty_ns)
        .def_readwrite("clock_domain", &ClockSyncSnapshot::clock_domain)
        .def_readwrite("generation", &ClockSyncSnapshot::generation)
        .def_readwrite("flags", &ClockSyncSnapshot::flags);

    py::class_<UnitDescriptor>(module, "UnitDescriptor", py::is_final())
        .def(py::init<UnitId, std::string, std::string>(), py::arg("unit_id"), py::arg("symbol"),
             py::arg("description") = "")
        .def_readonly("id", &UnitDescriptor::id)
        .def_readonly("symbol", &UnitDescriptor::symbol)
        .def_readonly("description", &UnitDescriptor::description);

    py::class_<FeatureSetDescriptor>(module, "FeatureSetDescriptor", py::is_final())
        .def(py::init(
                 [](FeatureSetId id, std::vector<std::string> names, std::vector<UnitId> unit_ids,
                    std::uint32_t source_stream_id, std::string source_stream,
                    std::uint64_t window_length_ns, std::uint64_t shift_ns,
                    std::string algorithm_name, std::string algorithm_version,
                    FeatureTimestampReference timestamp_reference)
                 {
                     return FeatureSetDescriptor{
                         .id = id,
                         .feature_names = std::move(names),
                         .unit_ids = std::move(unit_ids),
                         .source_stream_id = source_stream_id,
                         .source_stream = std::move(source_stream),
                         .algorithm_name = std::move(algorithm_name),
                         .algorithm_version = std::move(algorithm_version),
                         .window_length_ns = window_length_ns,
                         .shift_ns = shift_ns,
                         .timestamp_reference = timestamp_reference,
                     };
                 }),
             py::arg("feature_set_id"), py::arg("feature_names"), py::arg("unit_ids"),
             py::arg("source_stream_id"), py::arg("source_stream"), py::arg("window_length_ns"),
             py::arg("shift_ns"), py::arg("algorithm_name") = "", py::arg("algorithm_version") = "",
             py::arg("timestamp_reference") = FeatureTimestampReference::window_center)
        .def_readonly("id", &FeatureSetDescriptor::id)
        .def_readonly("feature_names", &FeatureSetDescriptor::feature_names)
        .def_readonly("unit_ids", &FeatureSetDescriptor::unit_ids)
        .def_readonly("source_stream_id", &FeatureSetDescriptor::source_stream_id)
        .def_readonly("source_stream", &FeatureSetDescriptor::source_stream)
        .def_readonly("algorithm_name", &FeatureSetDescriptor::algorithm_name)
        .def_readonly("algorithm_version", &FeatureSetDescriptor::algorithm_version)
        .def_readonly("window_length_ns", &FeatureSetDescriptor::window_length_ns)
        .def_readonly("shift_ns", &FeatureSetDescriptor::shift_ns)
        .def_readonly("timestamp_reference", &FeatureSetDescriptor::timestamp_reference);

    py::class_<UnitRegistry>(module, "UnitRegistry", py::is_final())
        .def(py::init([](const std::vector<UnitDescriptor>& units) { return UnitRegistry{units}; }),
             py::arg("units"))
        .def_property_readonly("units", py::overload_cast<const UnitRegistry&>(&copy_units));

    py::class_<FeatureSetDescriptorRegistry>(module, "FeatureSetDescriptorRegistry", py::is_final())
        .def(py::init([](const std::vector<FeatureSetDescriptor>& descriptors)
                      { return FeatureSetDescriptorRegistry{descriptors}; }),
             py::arg("descriptors"))
        .def_property_readonly(
            "descriptors",
            py::overload_cast<const FeatureSetDescriptorRegistry&>(&copy_feature_sets));

    py::class_<SignalSchema>(module, "SignalSchema", py::is_final())
        .def(py::init<SignalId, SignalDType, std::uint32_t, std::uint32_t, std::uint32_t,
                      RationalRate, ClockDomainId, SignalLayout, DeviceTickTracking, PhysicalUnit,
                      ChannelSetId, CalibrationId, ReferenceId, SignalKind, FeatureSetId,
                      ObservationTiming, std::uint64_t, std::vector<std::string>,
                      std::vector<double>>(),
             py::arg("signal_id"), py::arg("dtype"), py::arg("n_channels"),
             py::arg("nominal_block_samples"), py::arg("max_block_samples"), py::arg("fs"),
             py::arg("clock_domain"), py::arg("layout") = SignalLayout::sample_major,
             py::arg("device_tick_tracking") = DeviceTickTracking::unavailable,
             py::arg("physical_unit") = PhysicalUnit::unspecified, py::arg("channel_set_id") = 0,
             py::arg("calibration_id") = 0, py::arg("reference_id") = 0,
             py::arg("kind") = SignalKind::sampled, py::arg("feature_set_id") = 0,
             py::arg("observation_timing") = ObservationTiming::not_applicable,
             py::arg("fixed_block_bytes") = 0,
             py::arg("channel_names") = std::vector<std::string>{},
             py::arg("channel_impedances_ohm") = std::vector<double>{})
        .def_readonly("id", &SignalSchema::id)
        .def_readonly("dtype", &SignalSchema::dtype)
        .def_readonly("n_channels", &SignalSchema::n_channels)
        .def_readonly("nominal_block_samples", &SignalSchema::nominal_block_samples)
        .def_readonly("max_block_samples", &SignalSchema::max_block_samples)
        .def_readonly("fs", &SignalSchema::fs)
        .def_readonly("clock_domain", &SignalSchema::clock_domain)
        .def_readonly("layout", &SignalSchema::layout)
        .def_readonly("device_tick_tracking", &SignalSchema::device_tick_tracking)
        .def_readonly("kind", &SignalSchema::kind)
        .def_readonly("physical_unit", &SignalSchema::physical_unit)
        .def_readonly("channel_set_id", &SignalSchema::channel_set_id)
        .def_readonly("calibration_id", &SignalSchema::calibration_id)
        .def_readonly("reference_id", &SignalSchema::reference_id)
        .def_readonly("feature_set_id", &SignalSchema::feature_set_id)
        .def_readonly("observation_timing", &SignalSchema::observation_timing)
        .def_readonly("fixed_block_bytes", &SignalSchema::fixed_block_bytes)
        .def_readonly("max_block_bytes", &SignalSchema::max_block_bytes)
        .def_readonly("channel_names", &SignalSchema::channel_names)
        .def_readonly("channel_impedances_ohm", &SignalSchema::channel_impedances_ohm);

    py::class_<StreamSchema>(module, "StreamSchema", py::is_final())
        .def(py::init([](SchemaId schema_id, const std::vector<SignalSchema>& signals,
                         const std::vector<FeatureSetDescriptor>& feature_sets,
                         const std::vector<UnitDescriptor>& units)
                      { return StreamSchema{schema_id, signals, feature_sets, units}; }),
             py::arg("schema_id"), py::arg("signals"),
             py::arg("feature_sets") = std::vector<FeatureSetDescriptor>{},
             py::arg("units") = std::vector<UnitDescriptor>{})
        .def_property_readonly("id", &StreamSchema::id)
        .def_property_readonly("signals", &copy_signals)
        .def_property_readonly("feature_sets",
                               py::overload_cast<const StreamSchema&>(&copy_feature_sets))
        .def_property_readonly("units", py::overload_cast<const StreamSchema&>(&copy_units));

    py::class_<PoolCapacityBudget>(module, "PoolCapacityBudget", py::is_final())
        .def(py::init<>())
        .def_readwrite("source_owned", &PoolCapacityBudget::source_owned)
        .def_readwrite("ingress_capacity", &PoolCapacityBudget::ingress_capacity)
        .def_readwrite("processor_owned", &PoolCapacityBudget::processor_owned)
        .def_readwrite("critical_edge_capacity", &PoolCapacityBudget::critical_edge_capacity)
        .def_readwrite("actuator_owned", &PoolCapacityBudget::actuator_owned)
        .def_readwrite("observer_edge_capacity", &PoolCapacityBudget::observer_edge_capacity)
        .def_readwrite("reserve", &PoolCapacityBudget::reserve)
        .def_property_readonly("required_buffer_count", &PoolCapacityBudget::required_buffer_count);

    py::class_<RealtimeConfig>(module, "RealtimeConfig", py::is_final())
        .def(py::init<>())
        .def_readwrite("_automatic_resources", &RealtimeConfig::automatic_resources)
        .def_readwrite("platform", &RealtimeConfig::platform)
        .def_readwrite("pool_capacity", &RealtimeConfig::pool_capacity)
        .def_readwrite("buffer_size", &RealtimeConfig::buffer_size)
        .def_readwrite("max_signal_blocks", &RealtimeConfig::max_signal_blocks)
        .def_readwrite("discontinuity_capacity", &RealtimeConfig::discontinuity_capacity)
        .def_readwrite("gaps_per_discontinuity", &RealtimeConfig::gaps_per_discontinuity)
        .def_readwrite("max_process_outputs", &RealtimeConfig::max_process_outputs)
        .def_readwrite("max_flush_outputs", &RealtimeConfig::max_flush_outputs)
        .def_readwrite("fault_history_capacity", &RealtimeConfig::fault_history_capacity)
        .def_readwrite("source_stall_timeout", &RealtimeConfig::source_stall_timeout)
        .def_readwrite("max_ingress_dwell", &RealtimeConfig::max_ingress_dwell)
        .def_readwrite("processor_execution_deadline",
                       &RealtimeConfig::processor_execution_deadline)
        .def_readwrite("max_source_to_actuator_age", &RealtimeConfig::max_source_to_actuator_age)
        .def_readwrite("max_output_age", &RealtimeConfig::max_output_age)
        .def_readwrite("actuator_deadline", &RealtimeConfig::actuator_deadline)
        .def_readwrite("shutdown_deadline", &RealtimeConfig::shutdown_deadline)
        .def_readwrite("watchdog_period", &RealtimeConfig::watchdog_period)
        .def_property_readonly("required_buffer_count", &RealtimeConfig::required_buffer_count)
        .def("validate", &RealtimeConfig::validate);

    py::class_<ObserverEdgeConfig>(module, "ObserverEdgeConfig", py::is_final())
        .def(py::init<>())
        .def_readwrite("id", &ObserverEdgeConfig::id)
        .def_readwrite("capacity", &ObserverEdgeConfig::capacity)
        .def_readwrite("drop_history_capacity", &ObserverEdgeConfig::drop_history_capacity)
        .def_readwrite("drop_policy", &ObserverEdgeConfig::drop_policy)
        .def_readwrite("critical_recorder", &ObserverEdgeConfig::critical_recorder);
}

void bind_diagnostics(py::module_& module)
{
    py::class_<RealtimePlatformCapabilities>(module, "RealtimePlatformCapabilities", py::is_final())
        .def_readonly("supported_features", &RealtimePlatformCapabilities::supported_features)
        .def_readonly("logical_cpu_count", &RealtimePlatformCapabilities::logical_cpu_count)
        .def_readonly("max_thread_name_length",
                      &RealtimePlatformCapabilities::max_thread_name_length)
        .def_readonly("linux", &RealtimePlatformCapabilities::linux)
        .def_readonly("windows", &RealtimePlatformCapabilities::windows)
        .def_readonly("hard_realtime_guaranteed",
                      &RealtimePlatformCapabilities::hard_realtime_guaranteed)
        .def("supports", &RealtimePlatformCapabilities::supports);
    py::class_<RealtimeApplyResult>(module, "RealtimeApplyResult", py::is_final())
        .def_readonly("requested", &RealtimeApplyResult::requested)
        .def_readonly("applied", &RealtimeApplyResult::applied)
        .def_readonly("unsupported", &RealtimeApplyResult::unsupported)
        .def_readonly("failed", &RealtimeApplyResult::failed)
        .def_readonly("native_error", &RealtimeApplyResult::native_error)
        .def_property_readonly("ok", &RealtimeApplyResult::ok);
    py::class_<RealtimeConfigurationStatus>(module, "RealtimeConfigurationStatus", py::is_final())
        .def_readonly("memory", &RealtimeConfigurationStatus::memory)
        .def_property_readonly("acquisition", [](const RealtimeConfigurationStatus& value)
                               { return value.thread(RealtimeThreadRole::acquisition); })
        .def_property_readonly("processing", [](const RealtimeConfigurationStatus& value)
                               { return value.thread(RealtimeThreadRole::processing); })
        .def_property_readonly("actuator", [](const RealtimeConfigurationStatus& value)
                               { return value.thread(RealtimeThreadRole::actuator); })
        .def_property_readonly("observer_dispatch", [](const RealtimeConfigurationStatus& value)
                               { return value.thread(RealtimeThreadRole::observer_dispatch); })
        .def_property_readonly("watchdog", [](const RealtimeConfigurationStatus& value)
                               { return value.thread(RealtimeThreadRole::watchdog); })
        .def_property_readonly("threads",
                               [](const RealtimeConfigurationStatus& value)
                               {
                                   return std::vector<RealtimeApplyResult>{value.threads.begin(),
                                                                           value.threads.end()};
                               })
        .def_property_readonly("ok", &RealtimeConfigurationStatus::ok)
        .def_property_readonly("warning_count", &RealtimeConfigurationStatus::warning_count);
    py::class_<RuntimeStats>(module, "RuntimeStats", py::is_final())
        .def_readonly("frames_acquired", &RuntimeStats::frames_acquired)
        .def_readonly("frames_read", &RuntimeStats::frames_read)
        .def_readonly("frames_processed", &RuntimeStats::frames_processed)
        .def_readonly("frames_published", &RuntimeStats::frames_published)
        .def_readonly("frames_consumed", &RuntimeStats::frames_consumed)
        .def_readonly("zero_output_frames", &RuntimeStats::zero_output_frames)
        .def_readonly("processor_outputs", &RuntimeStats::processor_outputs)
        .def_readonly("flush_outputs", &RuntimeStats::flush_outputs)
        .def_readonly("discontinuities", &RuntimeStats::discontinuities)
        .def_readonly("end_of_streams", &RuntimeStats::end_of_streams)
        .def_readonly("ingress_high_water_mark", &RuntimeStats::ingress_high_water_mark)
        .def_readonly("queue_overruns", &RuntimeStats::queue_overruns)
        .def_readonly("pool_exhaustions", &RuntimeStats::pool_exhaustions)
        .def_readonly("buffer_exhaustions", &RuntimeStats::buffer_exhaustions)
        .def_readonly("aborted_frames", &RuntimeStats::aborted_frames)
        .def_readonly("deadline_faults", &RuntimeStats::deadline_faults)
        .def_readonly("safety_inhibitions", &RuntimeStats::safety_inhibitions)
        .def_readonly("faults", &RuntimeStats::faults)
        .def_readonly("secondary_faults", &RuntimeStats::secondary_faults)
        .def_readonly("fault_history_dropped", &RuntimeStats::fault_history_dropped)
        .def_readonly("actuator_commands_enqueued", &RuntimeStats::actuator_commands_enqueued)
        .def_readonly("actuator_commands_applied", &RuntimeStats::actuator_commands_applied)
        .def_readonly("actuator_queue_high_water_mark",
                      &RuntimeStats::actuator_queue_high_water_mark)
        .def_readonly("actuator_failures", &RuntimeStats::actuator_failures)
        .def_readonly("actuator_deadline_misses", &RuntimeStats::actuator_deadline_misses)
        .def_readonly("observer_dispatch_drops", &RuntimeStats::observer_dispatch_drops)
        .def_readonly("max_ingress_dwell_ns", &RuntimeStats::max_ingress_dwell_ns)
        .def_readonly("max_processor_execution_ns", &RuntimeStats::max_processor_execution_ns)
        .def_readonly("max_source_to_actuator_ns", &RuntimeStats::max_source_to_actuator_ns);

    py::class_<ObserverDropRange>(module, "ObserverDropRange", py::is_final())
        .def_readonly("first_sequence", &ObserverDropRange::first_sequence)
        .def_readonly("last_sequence", &ObserverDropRange::last_sequence);
    py::class_<ObserverEdgeStats>(module, "ObserverEdgeStats", py::is_final())
        .def_readonly("id", &ObserverEdgeStats::id)
        .def_readonly("enqueued", &ObserverEdgeStats::enqueued)
        .def_readonly("delivered", &ObserverEdgeStats::delivered)
        .def_readonly("discontinuities", &ObserverEdgeStats::discontinuities)
        .def_readonly("dropped", &ObserverEdgeStats::dropped)
        .def_readonly("drop_range_count", &ObserverEdgeStats::drop_range_count)
        .def_readonly("drop_history_dropped", &ObserverEdgeStats::drop_history_dropped)
        .def_readonly("failures", &ObserverEdgeStats::failures)
        .def_readonly("high_water_mark", &ObserverEdgeStats::high_water_mark)
        .def_readonly("last_enqueued_sequence", &ObserverEdgeStats::last_enqueued_sequence)
        .def_readonly("last_delivered_sequence", &ObserverEdgeStats::last_delivered_sequence)
        .def_readonly("detached", &ObserverEdgeStats::detached);

    py::class_<RuntimeHeartbeatSnapshot>(module, "RuntimeHeartbeat", py::is_final())
        .def_readonly("last_source_heartbeat", &RuntimeHeartbeatSnapshot::last_source_heartbeat)
        .def_readonly("current_processor_start", &RuntimeHeartbeatSnapshot::current_processor_start)
        .def_readonly("last_processor_completion",
                      &RuntimeHeartbeatSnapshot::last_processor_completion)
        .def_readonly("last_valid_output", &RuntimeHeartbeatSnapshot::last_valid_output)
        .def_readonly("current_actuator_start", &RuntimeHeartbeatSnapshot::current_actuator_start)
        .def_readonly("last_inhibit", &RuntimeHeartbeatSnapshot::last_inhibit)
        .def_readonly("runtime_generation", &RuntimeHeartbeatSnapshot::runtime_generation)
        .def_readonly("processor_active", &RuntimeHeartbeatSnapshot::processor_active)
        .def_readonly("actuator_active", &RuntimeHeartbeatSnapshot::actuator_active)
        .def_readonly("output_valid", &RuntimeHeartbeatSnapshot::output_valid)
        .def_readonly("safety_inhibited", &RuntimeHeartbeatSnapshot::safety_inhibited);

    py::class_<FaultRecord>(module, "FaultRecord", py::is_final())
        .def_readonly("code", &FaultRecord::code)
        .def_readonly("status", &FaultRecord::status)
        .def_readonly("stage", &FaultRecord::stage)
        .def_readonly("component_id", &FaultRecord::component_id)
        .def_readonly("detail", &FaultRecord::detail)
        .def_readonly("session_id", &FaultRecord::session_id)
        .def_readonly("runtime_generation", &FaultRecord::runtime_generation)
        .def_readonly("frame_sequence", &FaultRecord::frame_sequence)
        .def_readonly("schema_id", &FaultRecord::schema_id)
        .def_readonly("clock_domain", &FaultRecord::clock_domain)
        .def_readonly("signal_id", &FaultRecord::signal_id)
        .def_readonly("sample_idx", &FaultRecord::sample_idx)
        .def_readonly("device_tick", &FaultRecord::device_tick)
        .def_readonly("detected_at_ns", &FaultRecord::detected_at_ns);
}

} // namespace

void bind_streaming_module(py::module_& module)
{
    auto streaming = module.def_submodule("streaming", "Native streaming control-plane facade.");
    bind_enums(streaming);
    bind_schema_and_config(streaming);
    bind_diagnostics(streaming);
    streaming.def("realtime_platform_capabilities", &realtime_platform_capabilities);

    py::class_<NativeFrameSource>(streaming, "_NativeFrameSource", py::is_final());
    py::class_<NativeFrameProcessor>(streaming, "_NativeFrameProcessor", py::is_final());
    py::class_<NativeActuator>(streaming, "_NativeActuator", py::is_final());
    py::class_<NativeFrameConsumer, NativeActuator>(streaming, "_NativeFrameConsumer",
                                                    py::is_final());
    py::class_<NativeObserver>(streaming, "_NativeObserver", py::is_final());
    py::class_<SafetyController>(streaming, "_SafetyController", py::is_final());
    bind_array_replay(streaming);

    py::class_<SyntheticNativeSource, NativeFrameSource>(streaming, "SyntheticNativeSource",
                                                         py::is_final())
        .def(py::init<const StreamSchema&, std::size_t, SessionId, std::optional<std::size_t>,
                      std::optional<std::size_t>, bool, bool>(),
             "A deterministic native source.\n\n"
             "payload_pattern turns on everything a data-parity test needs and a "
             "plumbing test does not: a position-encoding payload in each signal's "
             "declared layout, per-block clock-sync snapshots and a frame source "
             "tick, and a manifest of what was written. It is off by default, so "
             "a source without it emits exactly the bytes it always did.\n\n"
             "vary_block_samples walks each signal's block size across its "
             "declared capacity instead of emitting the nominal block every "
             "time.",
             py::arg("schema"), py::arg("n_frames"), py::arg("session_id") = 1,
             py::arg("fail_at") = py::none(), py::arg("sequence_gap_at") = py::none(),
             py::arg("payload_pattern") = false, py::arg("vary_block_samples") = false)
        .def_property_readonly("frames_emitted", &SyntheticNativeSource::frames_emitted)
        .def_property_readonly("manifest", &SyntheticNativeSource::manifest,
                               "What this source wrote, frame by frame; empty without "
                               "payload_pattern.");

    py::class_<IdentityNativeProcessor, NativeFrameProcessor>(streaming, "IdentityNativeProcessor",
                                                              py::is_final())
        .def(py::init<>())
        .def_property_readonly("process_count", &IdentityNativeProcessor::process_count)
        .def_property_readonly("discontinuity_count", &IdentityNativeProcessor::discontinuity_count)
        .def_property_readonly("flush_count", &IdentityNativeProcessor::flush_count);

    py::class_<CountingNativeConsumer, NativeFrameConsumer>(streaming, "CountingNativeConsumer",
                                                            py::is_final())
        .def(py::init<std::optional<std::uint64_t>>(), py::arg("fail_at") = py::none())
        .def_property_readonly("frame_count", &CountingNativeConsumer::frame_count)
        .def_property_readonly("discontinuity_count", &CountingNativeConsumer::discontinuity_count)
        .def_property_readonly("flush_count", &CountingNativeConsumer::flush_count)
        .def_property_readonly("last_sequence", &CountingNativeConsumer::last_sequence);

    py::class_<CountingNativeObserver, NativeObserver>(streaming, "CountingNativeObserver",
                                                       py::is_final())
        .def(py::init<std::optional<std::uint64_t>>(), py::arg("fail_at") = py::none())
        .def_property_readonly("frame_count", &CountingNativeObserver::frame_count)
        .def_property_readonly("discontinuity_count", &CountingNativeObserver::discontinuity_count)
        .def_property_readonly("flush_count", &CountingNativeObserver::flush_count)
        .def_property_readonly("last_sequence", &CountingNativeObserver::last_sequence);

    py::class_<RecordingNativeSafetyController, SafetyController>(
        streaming, "RecordingNativeSafetyController", py::is_final())
        .def(py::init<>())
        .def("fail_next_inhibit", &RecordingNativeSafetyController::fail_next_inhibit)
        .def_property_readonly("inhibit_count", &RecordingNativeSafetyController::inhibit_count)
        .def_property_readonly("release_count", &RecordingNativeSafetyController::release_count)
        .def_property_readonly("last_reason", &RecordingNativeSafetyController::last_reason);

    py::class_<NativeStreamRunner> runner(streaming, "_NativeStreamRunner", py::is_final());
    runner
        .def(py::init(
                 [](const StreamSchema& schema, RealtimeConfig config, NativeFrameSource& source,
                    NativeFrameProcessor& processor, NativeFrameConsumer& consumer)
                 {
                     return std::make_unique<NativeStreamRunner>(schema.clone(), config, source,
                                                                 processor, consumer);
                 }),
             py::arg("schema"), py::arg("config"), py::arg("source"), py::arg("processor"),
             py::arg("actuator"), py::keep_alive<1, 4>(), py::keep_alive<1, 5>(),
             py::keep_alive<1, 6>())
        .def(py::init(
                 [](const StreamSchema& schema, RealtimeConfig config, NativeFrameSource& source,
                    NativeFrameProcessor& processor, NativeFrameConsumer& consumer,
                    SafetyController& safety_controller)
                 {
                     return std::make_unique<NativeStreamRunner>(
                         schema.clone(), config, source, processor, consumer,
                         default_native_clock(), safety_controller);
                 }),
             py::arg("schema"), py::arg("config"), py::arg("source"), py::arg("processor"),
             py::arg("actuator"), py::arg("safety_controller"), py::keep_alive<1, 4>(),
             py::keep_alive<1, 5>(), py::keep_alive<1, 6>(), py::keep_alive<1, 7>())
        .def("prepare", &NativeStreamRunner::prepare)
        .def(
            "add_observer",
            [](NativeStreamRunner& runner, NativeObserver& observer, ObserverEdgeConfig config)
            { return runner.add_observer(observer, config); }, py::arg("observer"),
            py::arg("config"), py::keep_alive<1, 2>())
        .def("detach_observer", &NativeStreamRunner::detach_observer)
        .def("observer_stats", &NativeStreamRunner::observer_stats)
        .def("observer_drop_ranges", &copy_observer_drop_ranges, py::arg("observer_id"))
        .def("arm", &NativeStreamRunner::arm)
        .def("start", &NativeStreamRunner::start)
        .def("join", &NativeStreamRunner::join, py::call_guard<py::gil_scoped_release>())
        .def("run", &NativeStreamRunner::run, py::call_guard<py::gil_scoped_release>())
        .def("stop", &NativeStreamRunner::stop, py::call_guard<py::gil_scoped_release>())
        .def("abort", &NativeStreamRunner::abort, py::call_guard<py::gil_scoped_release>())
        .def("reset", &NativeStreamRunner::reset, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("state", &NativeStreamRunner::state)
        .def_property_readonly("stats", &NativeStreamRunner::stats)
        .def_property_readonly("heartbeat", &NativeStreamRunner::heartbeat)
        .def_property_readonly("realtime_configuration_status",
                               &NativeStreamRunner::realtime_configuration_status)
        .def_property_readonly("primary_fault", &NativeStreamRunner::primary_fault)
        .def_property_readonly("fault_history", &copy_fault_history)
        .def_property_readonly("outstanding_frames", &NativeStreamRunner::outstanding_frames)
        .def_property_readonly("outstanding_discontinuities",
                               &NativeStreamRunner::outstanding_discontinuities);

    bind_python_observer_bridge(streaming);
    bind_streaming_python_adapters(streaming);
}
