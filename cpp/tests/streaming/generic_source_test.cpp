/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// The generic source-message extension, through a real runtime.
///
/// `NativeFrameSource::read()` can only produce frames, and a source whose
/// device is a recording has to produce discontinuities too -- a recorded gap
/// is an ordered item of its own, and folding it into the frame that follows
/// would lose both its reason and its position. The extension is
/// `read_message()`, defaulted to the frame-only read, plus
/// `produces_discontinuities()`, which is what keeps the addition free for
/// every source that does not want it.
///
/// Two things are checked here that no replay test can check: that a source
/// discontinuity survives the runtime's ingress edge in order, and that the
/// continuity checker does not answer an explicit discontinuity by inferring a
/// second one for the same gap.
///
/// The third is the provenance a source failure carries. A source that returns
/// a failure has not filled in a frame header, so the fault has to borrow the
/// context of the last frame the source did deliver -- and when there is no
/// such frame the fault says so with a zero session ID rather than inventing
/// one. That distinction is generic streaming behavior, not a property of any
/// one device, and every recorder that reads a fault's session ID depends on
/// it, so it is frozen here.

#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{
        SignalSchema{1, SignalDType::float32, 2, 4, 4, {1'000, 1}, 2},
    };
    return StreamSchema{7, signals};
}

[[nodiscard]] RealtimeConfig make_config()
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 8;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 10;
    config.pool_capacity.actuator_owned = 1;
    config.buffer_size = 64;
    config.max_signal_blocks = 1;
    // One slot for the source's per-read lease and one for the checker.
    config.discontinuity_capacity = 4;
    config.gaps_per_discontinuity = 2;
    config.max_process_outputs = 2;
    config.max_flush_outputs = 1;
    config.fault_history_capacity = 4;
    return config;
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::uint64_t sequence,
                                      SampleIndex sample_idx) noexcept
{
    frame.header() = FrameHeader{
        .session_id = 1,
        .sequence = sequence,
        .host_received_ns = static_cast<HostTimeNs>(sequence),
        .schema_id = 7,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx,
        .device_tick_start = sample_idx,
        .payload_offset = 0,
        .payload_byte_count = 32,
        .signal_id = 1,
        .n_samples = 4,
    };
    return frame.set_used_sizes(1, 32);
}

/// Frame 0, then an explicit discontinuity announcing the jump, then frame 5.
/// The sequence hole is real, so a checker that ignored the explicit record
/// would infer one of its own and the consumer would see two.
class AnnouncingSource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame&) noexcept override
    {
        return StreamStatus::end_of_stream;
    }

    StreamStatus read_message(MutableFrame& frame,
                              DiscontinuityLease& discontinuity) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        switch (step_++)
        {
        case 0:
            return fill_frame(frame, 0, 0);
        case 1:
        {
            const SignalGap gap{
                .expected_sample_idx = 4,
                .actual_sample_idx = 20,
                .missing_samples = 16,
                .signal_id = 1,
                .reason = GapReason::sample_gap,
                .flags = SignalGapFlags::missing_samples_known,
            };
            const auto status = discontinuity.assign(1, 0, 5, GapReason::sample_gap,
                                                     std::span<const SignalGap>(&gap, 1));
            return status == StreamStatus::ok ? StreamStatus::discontinuity : status;
        }
        case 2:
            return fill_frame(frame, 5, 20);
        default:
            return StreamStatus::end_of_stream;
        }
    }

    [[nodiscard]] bool produces_discontinuities() const noexcept override
    {
        return true;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }

    StreamStatus reset() noexcept override
    {
        step_ = 0;
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::size_t step_{};
    std::atomic<bool> cancelled_{};
};

/// The frame-only source of every other suite, unchanged: it never overrides
/// read_message() and must keep behaving exactly as it did.
class LegacySource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (step_ == 2)
        {
            return StreamStatus::end_of_stream;
        }
        const auto sequence = step_++;
        return fill_frame(frame, sequence, sequence * 4);
    }

    void cancel() noexcept override {}
    StreamStatus reset() noexcept override
    {
        step_ = 0;
        return StreamStatus::ok;
    }

  private:
    std::uint64_t step_{};
};

/// Delivers `frames` good frames and then fails the way a real read error
/// does: with a status and an untouched frame header.
class FailingSource final : public NativeFrameSource
{
  public:
    explicit FailingSource(std::uint64_t frames) noexcept : frames_(frames) {}

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (step_ >= frames_)
        {
            return StreamStatus::source_failure;
        }
        const auto sequence = step_++;
        return fill_frame(frame, sequence, sequence * 4);
    }

    void cancel() noexcept override {}
    StreamStatus reset() noexcept override
    {
        step_ = 0;
        return StreamStatus::ok;
    }

  private:
    std::uint64_t frames_{};
    std::uint64_t step_{};
};

class PassthroughProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources =
                ProcessorResourceBounds{.workspace_bytes = 0, .frame_pool_leases = 1},
        };
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        return output.publish_input();
    }

    StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept override
    {
        seen_.fetch_add(1, std::memory_order_acq_rel);
        last_previous_.store(discontinuity.previous_frame_sequence, std::memory_order_release);
        last_actual_.store(discontinuity.actual_frame_sequence, std::memory_order_release);
        last_gaps_.store(discontinuity.signal_gaps.size(), std::memory_order_release);
        return StreamStatus::ok;
    }

    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        seen_.store(0, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t seen() const noexcept
    {
        return seen_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t last_previous() const noexcept
    {
        return last_previous_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t last_actual() const noexcept
    {
        return last_actual_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t last_gaps() const noexcept
    {
        return last_gaps_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<std::size_t> seen_{};
    std::atomic<std::uint64_t> last_previous_{};
    std::atomic<std::uint64_t> last_actual_{};
    std::atomic<std::size_t> last_gaps_{};
};

class CountingConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView frame) noexcept override
    {
        const auto idx = consumed_.fetch_add(1, std::memory_order_acq_rel);
        if (idx < sequences_.size())
        {
            sequences_[idx] = frame.header.sequence;
        }
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuities_.fetch_add(1, std::memory_order_acq_rel);
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        consumed_.store(0, std::memory_order_release);
        discontinuities_.store(0, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t consumed() const noexcept
    {
        return consumed_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t sequence(std::size_t idx) const noexcept
    {
        return idx < sequences_.size() ? sequences_[idx] : 0;
    }

  private:
    std::atomic<std::size_t> consumed_{};
    std::atomic<std::size_t> discontinuities_{};
    std::array<std::uint64_t, 8> sequences_{};
};

template <typename Predicate> [[nodiscard]] bool spin_until(Predicate predicate) noexcept
{
    for (std::size_t attempt = 0; attempt < 2'000'000; ++attempt)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

int run()
{
    {
        // The extension is opt-in: the default answer is the one every source
        // written before it gives, and the default read_message() forwards.
        LegacySource legacy;
        CHECK(!legacy.produces_discontinuities());
        AnnouncingSource announcing;
        CHECK(announcing.produces_discontinuities());
    }

    {
        // A generic source holds a slot per read and the checker needs its own,
        // so one between them is refused at prepare rather than found later as
        // a pool exhaustion. A frame-only source is unaffected.
        auto narrow = make_config();
        narrow.discontinuity_capacity = 1;
        AnnouncingSource announcing;
        PassthroughProcessor processor;
        CountingConsumer consumer;
        NativeStreamRunner runtime{make_schema(), narrow, announcing, processor, consumer};
        bool refused = false;
        try
        {
            static_cast<void>(runtime.prepare());
        }
        catch (const std::invalid_argument&)
        {
            refused = true;
        }
        CHECK(refused);

        LegacySource legacy;
        PassthroughProcessor other_processor;
        CountingConsumer other_consumer;
        NativeStreamRunner permitted{make_schema(), narrow, legacy, other_processor,
                                     other_consumer};
        CHECK(permitted.prepare() == StreamStatus::ok);
    }

    {
        LegacySource source;
        PassthroughProcessor processor;
        CountingConsumer consumer;
        NativeStreamRunner runtime{make_schema(), make_config(), source, processor, consumer};
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.arm() == StreamStatus::ok);
        CHECK(runtime.start() == StreamStatus::ok);
        CHECK(spin_until([&] { return consumer.consumed() == 2; }));
        CHECK(runtime.stop() == StreamStatus::ok);
        CHECK(consumer.discontinuities() == 0);
        CHECK(processor.seen() == 0);
    }

    {
        // Automatic capacities must allow ordered source messages to occupy
        // every bounded edge, rather than restricting a source to two gaps.
        AnnouncingSource source;
        PassthroughProcessor processor;
        CountingConsumer consumer;
        auto automatic = make_config();
        automatic.automatic_resources = true;
        NativeStreamRunner runtime{make_schema(), automatic, source, processor, consumer};
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.resolved_config().discontinuity_capacity >=
              runtime.resolved_config().pool_capacity.ingress_capacity + 2);
        CHECK(runtime.arm() == StreamStatus::ok);
        CHECK(runtime.run() == StreamStatus::ok);
        CHECK(consumer.discontinuities() == 1);
    }

    {
        AnnouncingSource source;
        PassthroughProcessor processor;
        CountingConsumer consumer;
        NativeStreamRunner runtime{make_schema(), make_config(), source, processor, consumer};
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.arm() == StreamStatus::ok);
        CHECK(runtime.start() == StreamStatus::ok);
        CHECK(spin_until([&] { return consumer.consumed() == 2; }));
        CHECK(runtime.stop() == StreamStatus::ok);

        // Exactly one discontinuity: the source's own. The frame that follows
        // jumps from sequence 0 to 5, which the checker would have called a
        // gap of its own -- inference yields to an explicit record.
        CHECK(processor.seen() == 1);
        CHECK(consumer.discontinuities() == 1);
        CHECK(processor.last_previous() == 0);
        CHECK(processor.last_actual() == 5);
        CHECK(processor.last_gaps() == 1);
        CHECK(consumer.sequence(0) == 0);
        CHECK(consumer.sequence(1) == 5);
    }

    {
        // A failure after two frames borrows the second one's identity: the
        // fault belongs to the same session and names the last sequence the
        // runtime actually saw, not a blank header.
        FailingSource source{2};
        PassthroughProcessor processor;
        CountingConsumer consumer;
        NativeStreamRunner runtime{make_schema(), make_config(), source, processor, consumer};
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.arm() == StreamStatus::ok);
        CHECK(runtime.run() == StreamStatus::source_failure);
        const auto fault = runtime.primary_fault();
        CHECK(fault.has_value());
        CHECK(fault->code == FaultCode::source_read);
        CHECK(fault->status == StreamStatus::source_failure);
        CHECK(fault->session_id == 1);
        CHECK(fault->frame_sequence == 1);
        CHECK(fault->schema_id == 7);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.outstanding_discontinuities() == 0);
    }

    {
        // A failure before the first frame has no context to borrow. The fault
        // reports a zero session ID, which is the runtime saying "unknown" --
        // a consumer that already knows the session identity (the recorder does)
        // must keep its own rather than adopt this zero.
        FailingSource source{0};
        PassthroughProcessor processor;
        CountingConsumer consumer;
        NativeStreamRunner runtime{make_schema(), make_config(), source, processor, consumer};
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.arm() == StreamStatus::ok);
        CHECK(runtime.run() == StreamStatus::source_failure);
        const auto fault = runtime.primary_fault();
        CHECK(fault.has_value());
        CHECK(fault->code == FaultCode::source_read);
        CHECK(fault->session_id == 0);
        CHECK(fault->frame_sequence == 0);
        CHECK(consumer.consumed() == 0);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.outstanding_discontinuities() == 0);
    }

    return 0;
}

} // namespace

int main()
{
    const int status = run();
    if (status == 0)
    {
        std::cout << "generic source extension ok\n";
        return 0;
    }
    return 1;
}
