/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/linear_processor_chain.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

template <typename T>
concept ExposesLeasePublication = requires(T& emitter, neurale::streaming::FrameLease lease) {
    emitter.publish_owned(std::move(lease));
};

static_assert(!ExposesLeasePublication<neurale::streaming::FrameEmitter>);
static_assert(!std::is_copy_constructible_v<neurale::streaming::FrameBorrow>);
} // namespace

namespace
{

using namespace neurale::streaming;

template <typename Callback> [[nodiscard]] bool throws(Callback callback)
{
    try
    {
        callback();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

[[nodiscard]] StreamSchema make_schema(SchemaId id = 7)
{
    const std::array signals{
        SignalSchema{1, SignalDType::float32, 2, 4, 4, {1'000, 1}, 2},
    };
    return StreamSchema{id, signals};
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::uint64_t sequence) noexcept
{
    frame.header() = FrameHeader{
        .session_id = 1,
        .sequence = sequence,
        .host_received_ns = sequence,
        .schema_id = 7,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sequence * 4,
        .device_tick_start = sequence * 4,
        .payload_offset = 0,
        .payload_byte_count = 32,
        .signal_id = 1,
        .n_samples = 4,
    };
    std::fill_n(frame.payload_storage().begin(), 32, static_cast<std::byte>(sequence & 0xffU));
    return frame.set_used_sizes(1, 32);
}

[[nodiscard]] StreamStatus copy_frame(FrameView source, MutableFrame& destination) noexcept
{
    destination.header() = source.header;
    std::copy(source.blocks.begin(), source.blocks.end(), destination.block_storage().begin());
    std::copy(source.payload.begin(), source.payload.end(), destination.payload_storage().begin());
    return destination.set_used_sizes(source.blocks.size(), source.payload.size());
}

struct EventLog
{
    std::array<int, 128> values{};
    std::size_t count{};

    void push(int value) noexcept
    {
        values[count++] = value;
    }
};

enum class Behavior
{
    forward,
    acquire,
    zero,
    fanout_two,
    fail,
    leak,
    retain_forwarded,
    retain_published,
    retain_unpublished,
};

class TestProcessor final : public NativeFrameProcessor
{
  public:
    TestProcessor(std::size_t id, Behavior behavior, EventLog* events = nullptr)
        : id_(id), behavior_(behavior), events_(events)
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        if (throw_prepare_)
        {
            throw std::runtime_error("injected processor prepare exception");
        }
        ++prepare_count_;
        const auto accepted_id =
            mismatch_input_ ? context.input_schema.id() + 1 : context.input_schema.id();
        const auto process_outputs = declared_process_outputs_.has_value()
                                         ? *declared_process_outputs_
                                     : behavior_ == Behavior::zero       ? 0
                                     : behavior_ == Behavior::fanout_two ? 2
                                                                         : 1;
        return {
            .accepted_input_schema = StreamSchema{accepted_id, context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = process_outputs,
            .max_flush_outputs = flush_output_ ? 1U : 0U,
            .can_forward_input = behavior_ == Behavior::forward ||
                                 behavior_ == Behavior::fanout_two ||
                                 behavior_ == Behavior::retain_forwarded,
            .required_resources =
                ProcessorResourceBounds{.workspace_bytes = 0, .frame_pool_leases = 2},
        };
    }

    StreamStatus process(FrameBorrow& frame, FrameEmitter& emitter) noexcept override
    {
        ++process_count_;
        if (events_ != nullptr)
        {
            events_->push(static_cast<int>(100 + id_));
        }
        if (fail_process_)
        {
            return StreamStatus::processor_failure;
        }
        switch (behavior_)
        {
        case Behavior::forward:
            return emitter.publish_input();
        case Behavior::acquire:
            return publish_copy(frame.view(), emitter);
        case Behavior::zero:
            return StreamStatus::ok;
        case Behavior::fanout_two:
        {
            const auto status = publish_copy(frame.view(), emitter);
            return status == StreamStatus::ok ? emitter.publish_input() : status;
        }
        case Behavior::fail:
            return StreamStatus::processor_failure;
        case Behavior::leak:
        {
            FrameBorrow unused{};
            return emitter.try_acquire(unused);
        }
        case Behavior::retain_forwarded:
            retained_input_ = std::move(frame);
            if (const auto status = emitter.publish_input(); status != StreamStatus::ok)
            {
                return status;
            }
            invalidated_during_publish_ = retained_input_.get() == nullptr;
            return StreamStatus::ok;
        case Behavior::retain_published:
        case Behavior::retain_unpublished:
        {
            FrameBorrow output{};
            auto status = emitter.try_acquire(output);
            if (status != StreamStatus::ok)
            {
                return status;
            }
            status = copy_frame(frame.view(), *output);
            if (status != StreamStatus::ok)
            {
                return status;
            }
            retained_input_ = std::move(frame);
            retained_output_ = std::move(output);
            if (behavior_ == Behavior::retain_unpublished)
            {
                return StreamStatus::ok;
            }
            status = emitter.publish_acquired();
            invalidated_during_publish_ = retained_output_.get() == nullptr;
            return status;
        }
        }
        return StreamStatus::processor_failure;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        ++discontinuity_count_;
        if (events_ != nullptr)
        {
            events_->push(static_cast<int>(300 + id_));
        }
        return discontinuity_status_;
    }

    StreamStatus flush(FrameEmitter& emitter) noexcept override
    {
        ++flush_count_;
        if (events_ != nullptr)
        {
            events_->push(static_cast<int>(200 + id_));
        }
        if (flush_status_ != StreamStatus::ok)
        {
            return flush_status_;
        }
        if (!flush_output_)
        {
            return StreamStatus::ok;
        }
        FrameBorrow frame{};
        auto status = emitter.try_acquire(frame);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        status = fill_frame(*frame, 1'000 + id_);
        return status == StreamStatus::ok ? emitter.publish_acquired() : status;
    }

    StreamStatus reset() noexcept override
    {
        ++reset_count_;
        process_count_ = 0;
        flush_count_ = 0;
        discontinuity_count_ = 0;
        if (events_ != nullptr)
        {
            events_->push(static_cast<int>(400 + id_));
        }
        return reset_status_;
    }

    void emit_on_flush() noexcept
    {
        flush_output_ = true;
    }
    void throw_on_prepare() noexcept
    {
        throw_prepare_ = true;
    }
    void mismatch_input_schema() noexcept
    {
        mismatch_input_ = true;
    }
    void declare_process_outputs(std::size_t value) noexcept
    {
        declared_process_outputs_ = value;
    }
    void set_behavior(Behavior behavior) noexcept
    {
        behavior_ = behavior;
    }
    void fail_discontinuity(StreamStatus status) noexcept
    {
        discontinuity_status_ = status;
    }
    void fail_reset(StreamStatus status) noexcept
    {
        reset_status_ = status;
    }
    void fail_flush(StreamStatus status) noexcept
    {
        flush_status_ = status;
    }
    void fail_process(bool value) noexcept
    {
        fail_process_ = value;
    }

    [[nodiscard]] bool retained_borrows_are_stale() const noexcept
    {
        return retained_input_.get() == nullptr && retained_output_.get() == nullptr;
    }

    [[nodiscard]] bool invalidated_during_publish() const noexcept
    {
        return invalidated_during_publish_;
    }

    [[nodiscard]] std::size_t process_count() const noexcept
    {
        return process_count_;
    }
    [[nodiscard]] std::size_t flush_count() const noexcept
    {
        return flush_count_;
    }
    [[nodiscard]] std::size_t discontinuity_count() const noexcept
    {
        return discontinuity_count_;
    }
    [[nodiscard]] std::size_t reset_count() const noexcept
    {
        return reset_count_;
    }

  private:
    static StreamStatus publish_copy(FrameView source, FrameEmitter& emitter) noexcept
    {
        FrameBorrow destination{};
        auto status = emitter.try_acquire(destination);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        status = copy_frame(source, *destination);
        return status == StreamStatus::ok ? emitter.publish_acquired() : status;
    }

    std::size_t id_{};
    Behavior behavior_{Behavior::forward};
    EventLog* events_{};
    std::optional<std::size_t> declared_process_outputs_;
    std::size_t prepare_count_{};
    std::size_t process_count_{};
    std::size_t flush_count_{};
    std::size_t discontinuity_count_{};
    std::size_t reset_count_{};
    StreamStatus discontinuity_status_{StreamStatus::ok};
    StreamStatus flush_status_{StreamStatus::ok};
    StreamStatus reset_status_{StreamStatus::ok};
    bool flush_output_{};
    bool throw_prepare_{};
    bool mismatch_input_{};
    bool fail_process_{};
    bool invalidated_during_publish_{};
    FrameBorrow retained_input_{};
    FrameBorrow retained_output_{};
};

class HarnessEmitter final : public FrameEmitter
{
  public:
    HarnessEmitter(const StreamSchema& schema, std::size_t capacity = 32)
        : pool_(capacity, 64, 1), validator_(schema)
    {
    }

    void begin(MutableFrame& input) noexcept
    {
        input_ = &input;
        count_ = 0;
        failure_ = StreamStatus::ok;
        invalidate_borrow(input_borrow_state_);
        invalidate_acquired_borrow();
        static_cast<void>(acquired_.reset());
        input_borrow_ = {};
        if (activate_borrow(input_borrow_state_, input, input_borrow_) != StreamStatus::ok)
        {
            failure_ = StreamStatus::invalid_state;
        }
    }

    [[nodiscard]] FrameBorrow& input_borrow() noexcept
    {
        return input_borrow_;
    }

    void fail_acquisition(StreamStatus status) noexcept
    {
        acquisition_failure_ = status;
    }

    [[nodiscard]] bool exhaust_pool() noexcept
    {
        return pool_.try_acquire(held_) == StreamStatus::ok;
    }

    void release_held() noexcept
    {
        static_cast<void>(held_.reset());
    }

    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        frame = {};
        if (acquisition_failure_ != StreamStatus::ok)
        {
            failure_ = acquisition_failure_;
            return failure_;
        }
        if (acquired_)
        {
            failure_ = StreamStatus::invalid_state;
            return failure_;
        }
        const auto status = pool_.try_acquire(acquired_);
        if (status != StreamStatus::ok)
        {
            failure_ = status;
            return status;
        }
        frame = &acquired_.frame();
        return StreamStatus::ok;
    }

    StreamStatus publish_acquired_frame() noexcept override
    {
        return record(std::move(acquired_));
    }

    StreamStatus publish_input() noexcept override
    {
        if (input_ == nullptr || validator_.validate(input_->view()) != FrameValidationError::none)
        {
            failure_ = StreamStatus::invalid_frame;
            return failure_;
        }
        sequences_[count_++] = input_->header().sequence;
        invalidate_borrow(input_borrow_state_);
        return StreamStatus::ok;
    }

    StreamStatus finish() noexcept
    {
        if (acquired_)
        {
            invalidate_acquired_borrow();
            static_cast<void>(acquired_.reset());
            failure_ = StreamStatus::invalid_state;
        }
        invalidate_borrow(input_borrow_state_);
        input_ = nullptr;
        return failure_;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return count_;
    }
    [[nodiscard]] std::uint64_t sequence(std::size_t idx) const noexcept
    {
        return sequences_[idx];
    }
    [[nodiscard]] std::size_t outstanding() const noexcept
    {
        return pool_.outstanding();
    }

  private:
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return record(std::move(lease));
    }

    StreamStatus record(FrameLease lease) noexcept
    {
        if (!lease || validator_.validate(lease.view()) != FrameValidationError::none)
        {
            failure_ = StreamStatus::invalid_frame;
            return failure_;
        }
        sequences_[count_++] = lease.frame().header().sequence;
        return StreamStatus::ok;
    }

    FramePool pool_;
    FrameValidator validator_;
    MutableFrame* input_{};
    FrameLease acquired_{};
    FrameLease held_{};
    detail::FrameBorrowState input_borrow_state_{};
    FrameBorrow input_borrow_{};
    std::array<std::uint64_t, 128> sequences_{};
    std::size_t count_{};
    StreamStatus failure_{StreamStatus::ok};
    StreamStatus acquisition_failure_{StreamStatus::ok};
};

[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema,
                                              std::size_t process_outputs = 16,
                                              std::size_t flush_outputs = 16)
{
    return {.input_schema = schema,
            .max_process_outputs = process_outputs,
            .max_flush_outputs = flush_outputs,
            .available_frame_pool_leases = 8};
}

struct FrameFixture
{
    FramePool pool{1, 64, 1};
    FrameLease lease;

    explicit FrameFixture(std::uint64_t sequence = 1)
    {
        if (pool.try_acquire(lease) != StreamStatus::ok ||
            fill_frame(lease.frame(), sequence) != StreamStatus::ok)
        {
            throw std::runtime_error("failed to prepare frame fixture");
        }
    }
};

int test_empty_chain_is_rejected()
{
    const std::span<NativeFrameProcessor* const> empty;
    CHECK(throws([&] { LinearProcessorChain chain{empty}; }));
    return 0;
}

int test_one_and_multiple_stage_propagation()
{
    auto schema = make_schema();
    {
        TestProcessor stage{0, Behavior::forward};
        std::array<NativeFrameProcessor*, 1> stages{&stage};
        LinearProcessorChain chain{stages};
        const auto contract = chain.prepare(context(schema));
        CHECK(contract.max_process_outputs_per_input == 1);
        CHECK(chain.stage_count() == 1);
        FrameFixture input{7};
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
        CHECK(output.finish() == StreamStatus::ok);
        CHECK(output.count() == 1);
        CHECK(output.sequence(0) == 7);
    }
    {
        TestProcessor first{0, Behavior::acquire};
        TestProcessor second{1, Behavior::forward};
        TestProcessor third{2, Behavior::acquire};
        std::array<NativeFrameProcessor*, 3> stages{&first, &second, &third};
        LinearProcessorChain chain{stages};
        static_cast<void>(chain.prepare(context(schema)));
        FrameFixture input{9};
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
        CHECK(output.finish() == StreamStatus::ok);
        CHECK(first.process_count() == 1);
        CHECK(second.process_count() == 1);
        CHECK(third.process_count() == 1);
        CHECK(output.count() == 1);
        CHECK(output.sequence(0) == 9);
    }
    return 0;
}

int test_zero_output_and_depth_first_fanout()
{
    auto schema = make_schema();
    {
        TestProcessor zero{0, Behavior::zero};
        TestProcessor never{1, Behavior::forward};
        std::array<NativeFrameProcessor*, 2> stages{&zero, &never};
        LinearProcessorChain chain{stages};
        const auto contract = chain.prepare(context(schema));
        CHECK(contract.max_process_outputs_per_input == 0);
        FrameFixture input;
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
        CHECK(output.finish() == StreamStatus::ok);
        CHECK(never.process_count() == 0);
        CHECK(output.count() == 0);
    }
    {
        EventLog events;
        TestProcessor first{0, Behavior::fanout_two, &events};
        TestProcessor second{1, Behavior::fanout_two, &events};
        TestProcessor third{2, Behavior::forward, &events};
        std::array<NativeFrameProcessor*, 3> stages{&first, &second, &third};
        LinearProcessorChain chain{stages};
        const auto contract = chain.prepare(context(schema, 4));
        CHECK(contract.max_process_outputs_per_input == 4);
        FrameFixture input{3};
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
        CHECK(output.finish() == StreamStatus::ok);
        CHECK(output.count() == 4);
        CHECK(events.count == 7);
        const std::array expected{100, 101, 102, 102, 101, 102, 102};
        CHECK(std::equal(expected.begin(), expected.end(), events.values.begin()));
    }
    return 0;
}

int test_prepare_schema_fanout_and_exception_checks()
{
    auto schema = make_schema();
    {
        TestProcessor first{0, Behavior::fanout_two};
        TestProcessor second{1, Behavior::fanout_two};
        std::array<NativeFrameProcessor*, 2> stages{&first, &second};
        LinearProcessorChain chain{stages};
        CHECK(throws([&] { static_cast<void>(chain.prepare(context(schema, 3))); }));
    }
    {
        TestProcessor first{0, Behavior::forward};
        TestProcessor second{1, Behavior::forward};
        second.mismatch_input_schema();
        std::array<NativeFrameProcessor*, 2> stages{&first, &second};
        LinearProcessorChain chain{stages};
        CHECK(throws([&] { static_cast<void>(chain.prepare(context(schema))); }));
    }
    {
        TestProcessor first{0, Behavior::forward};
        TestProcessor second{1, Behavior::forward};
        first.declare_process_outputs(std::numeric_limits<std::size_t>::max());
        second.declare_process_outputs(2);
        std::array<NativeFrameProcessor*, 2> stages{&first, &second};
        LinearProcessorChain chain{stages};
        CHECK(throws(
            [&]
            {
                static_cast<void>(
                    chain.prepare(context(schema, std::numeric_limits<std::size_t>::max())));
            }));
    }
    {
        TestProcessor first{0, Behavior::forward};
        TestProcessor second{1, Behavior::forward};
        first.emit_on_flush();
        second.emit_on_flush();
        second.declare_process_outputs(std::numeric_limits<std::size_t>::max());
        std::array<NativeFrameProcessor*, 2> stages{&first, &second};
        LinearProcessorChain chain{stages};
        CHECK(throws(
            [&]
            {
                static_cast<void>(
                    chain.prepare(context(schema, std::numeric_limits<std::size_t>::max(),
                                          std::numeric_limits<std::size_t>::max())));
            }));
    }
    {
        TestProcessor throwing{0, Behavior::forward};
        throwing.throw_on_prepare();
        std::array<NativeFrameProcessor*, 1> stages{&throwing};
        LinearProcessorChain chain{stages};
        CHECK(throws([&] { static_cast<void>(chain.prepare(context(schema))); }));
    }
    return 0;
}

int test_flush_ordering_and_discontinuity_barrier()
{
    auto schema = make_schema();
    EventLog events;
    TestProcessor first{0, Behavior::forward, &events};
    TestProcessor second{1, Behavior::forward, &events};
    TestProcessor third{2, Behavior::forward, &events};
    first.emit_on_flush();
    second.emit_on_flush();
    third.emit_on_flush();
    std::array<NativeFrameProcessor*, 3> stages{&first, &second, &third};
    LinearProcessorChain chain{stages};
    static_cast<void>(chain.prepare(context(schema, 8, 6)));
    FrameFixture input;
    HarnessEmitter output{schema};
    output.begin(input.lease.frame());
    CHECK(chain.flush(output) == StreamStatus::ok);
    CHECK(output.finish() == StreamStatus::ok);
    const std::array flush_expected{200, 101, 102, 201, 102, 202};
    CHECK(events.count == flush_expected.size());
    CHECK(std::equal(flush_expected.begin(), flush_expected.end(), events.values.begin()));
    CHECK(output.count() == 3);

    events.count = 0;
    const Discontinuity discontinuity{};
    CHECK(chain.handle_discontinuity(discontinuity) == StreamStatus::ok);
    const std::array discontinuity_expected{300, 301, 302};
    CHECK(events.count == discontinuity_expected.size());
    CHECK(std::equal(discontinuity_expected.begin(), discontinuity_expected.end(),
                     events.values.begin()));
    return 0;
}

int test_reset_determinism_and_fault_propagation()
{
    auto schema = make_schema();
    TestProcessor first{0, Behavior::forward};
    TestProcessor second{1, Behavior::forward};
    second.fail_process(true);
    TestProcessor third{2, Behavior::forward};
    std::array<NativeFrameProcessor*, 3> stages{&first, &second, &third};
    LinearProcessorChain chain{stages};
    static_cast<void>(chain.prepare(context(schema)));
    FrameFixture input;
    HarnessEmitter output{schema};
    output.begin(input.lease.frame());
    CHECK(chain.process(output.input_borrow(), output) == StreamStatus::processor_failure);
    CHECK(output.finish() == StreamStatus::ok);
    CHECK(third.process_count() == 0);

    CHECK(chain.reset() == StreamStatus::ok);
    CHECK(chain.process(output.input_borrow(), output) == StreamStatus::invalid_state);
    second.fail_process(false);
    static_cast<void>(chain.prepare(context(schema)));
    output.begin(input.lease.frame());
    CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
    CHECK(output.finish() == StreamStatus::ok);
    CHECK(first.process_count() == 1);
    CHECK(second.process_count() == 1);
    CHECK(third.process_count() == 1);
    return 0;
}

int test_flush_discontinuity_and_reset_fault_propagation()
{
    auto schema = make_schema();
    TestProcessor first{0, Behavior::forward};
    TestProcessor second{1, Behavior::forward};
    TestProcessor third{2, Behavior::forward};
    std::array<NativeFrameProcessor*, 3> stages{&first, &second, &third};
    LinearProcessorChain chain{stages};
    static_cast<void>(chain.prepare(context(schema)));
    FrameFixture input;
    HarnessEmitter output{schema};

    second.fail_flush(StreamStatus::processor_failure);
    output.begin(input.lease.frame());
    CHECK(chain.flush(output) == StreamStatus::processor_failure);
    CHECK(output.finish() == StreamStatus::ok);
    CHECK(first.flush_count() == 1);
    CHECK(second.flush_count() == 1);
    CHECK(third.flush_count() == 0);

    second.fail_discontinuity(StreamStatus::processor_failure);
    CHECK(chain.handle_discontinuity(Discontinuity{}) == StreamStatus::processor_failure);
    CHECK(first.discontinuity_count() == 1);
    CHECK(second.discontinuity_count() == 1);
    CHECK(third.discontinuity_count() == 0);

    first.fail_reset(StreamStatus::processor_failure);
    second.fail_reset(StreamStatus::invalid_state);
    CHECK(chain.reset() == StreamStatus::processor_failure);
    CHECK(first.reset_count() == 1);
    CHECK(second.reset_count() == 1);
    CHECK(third.reset_count() == 1);
    return 0;
}

int test_acquisition_contract_pool_and_leak_failures()
{
    auto schema = make_schema();
    {
        TestProcessor stage{0, Behavior::acquire};
        std::array<NativeFrameProcessor*, 1> stages{&stage};
        LinearProcessorChain chain{stages};
        static_cast<void>(chain.prepare(context(schema)));
        FrameFixture input;
        HarnessEmitter output{schema};
        output.fail_acquisition(StreamStatus::would_block);
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::would_block);
    }
    {
        TestProcessor stage{0, Behavior::acquire};
        std::array<NativeFrameProcessor*, 1> stages{&stage};
        LinearProcessorChain chain{stages};
        static_cast<void>(chain.prepare(context(schema)));
        FrameFixture input;
        HarnessEmitter output{schema, 1};
        output.begin(input.lease.frame());
        CHECK(output.exhaust_pool());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::buffer_exhausted);
        output.release_held();
    }
    {
        TestProcessor violating{0, Behavior::fanout_two};
        violating.declare_process_outputs(1);
        std::array<NativeFrameProcessor*, 1> stages{&violating};
        LinearProcessorChain chain{stages};
        static_cast<void>(chain.prepare(context(schema)));
        FrameFixture input;
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::output_limit);
        CHECK(output.finish() == StreamStatus::ok);
    }
    {
        TestProcessor leaking{0, Behavior::leak};
        TestProcessor downstream{1, Behavior::forward};
        std::array<NativeFrameProcessor*, 2> stages{&leaking, &downstream};
        LinearProcessorChain chain{stages};
        static_cast<void>(chain.prepare(context(schema)));
        FrameFixture input;
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::invalid_state);
        CHECK(output.finish() == StreamStatus::ok);
        leaking.set_behavior(Behavior::acquire);
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
        CHECK(output.finish() == StreamStatus::ok);
        CHECK(output.outstanding() == 0);
    }
    return 0;
}

int test_no_steady_state_allocation()
{
    auto schema = make_schema();
    TestProcessor first{0, Behavior::fanout_two};
    TestProcessor second{1, Behavior::acquire};
    TestProcessor third{2, Behavior::forward};
    std::array<NativeFrameProcessor*, 3> stages{&first, &second, &third};
    LinearProcessorChain chain{stages};
    static_cast<void>(chain.prepare(context(schema, 2)));
    FrameFixture input;
    HarnessEmitter output{schema};
    output.begin(input.lease.frame());
    const auto before = allocations.load(std::memory_order_relaxed);
    CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
    CHECK(output.finish() == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    CHECK(output.count() == 2);
    return 0;
}

int test_callback_scoped_borrow_retention_is_rejected()
{
    auto schema = make_schema();
    for (const auto behavior : {Behavior::retain_forwarded, Behavior::retain_published})
    {
        TestProcessor retaining{0, behavior};
        std::array<NativeFrameProcessor*, 1> stages{&retaining};
        LinearProcessorChain chain{stages};
        static_cast<void>(chain.prepare(context(schema)));
        FrameFixture input;
        HarnessEmitter output{schema};
        output.begin(input.lease.frame());
        CHECK(chain.process(output.input_borrow(), output) == StreamStatus::ok);
        CHECK(output.finish() == StreamStatus::ok);
        CHECK(retaining.invalidated_during_publish());
        CHECK(retaining.retained_borrows_are_stale());
        CHECK(output.outstanding() == 0);
    }

    TestProcessor directly_invoked{0, Behavior::retain_forwarded};
    FrameFixture direct_input;
    HarnessEmitter direct_output{schema};
    direct_output.begin(direct_input.lease.frame());
    NativeFrameProcessor& processor = directly_invoked;
    CHECK(processor.process(direct_input.lease.frame(), direct_output) == StreamStatus::ok);
    CHECK(!directly_invoked.invalidated_during_publish());
    CHECK(directly_invoked.retained_borrows_are_stale());
    CHECK(direct_output.finish() == StreamStatus::ok);
    CHECK(direct_output.outstanding() == 0);

    TestProcessor retaining{0, Behavior::retain_unpublished};
    std::array<NativeFrameProcessor*, 1> stages{&retaining};
    LinearProcessorChain chain{stages};
    static_cast<void>(chain.prepare(context(schema)));
    FrameFixture input;
    HarnessEmitter output{schema};
    output.begin(input.lease.frame());
    CHECK(chain.process(output.input_borrow(), output) == StreamStatus::invalid_state);
    CHECK(output.finish() == StreamStatus::invalid_state);
    CHECK(retaining.retained_borrows_are_stale());
    CHECK(output.outstanding() == 0);
    return 0;
}

} // namespace

int main()
{
    static_assert(std::is_final_v<LinearProcessorChain>);
    const std::array tests{
        test_empty_chain_is_rejected,
        test_one_and_multiple_stage_propagation,
        test_zero_output_and_depth_first_fanout,
        test_prepare_schema_fanout_and_exception_checks,
        test_flush_ordering_and_discontinuity_barrier,
        test_reset_determinism_and_fault_propagation,
        test_flush_discontinuity_and_reset_fault_propagation,
        test_acquisition_contract_pool_and_leak_failures,
        test_no_steady_state_allocation,
        test_callback_scoped_borrow_retention_is_rejected,
    };
    for (const auto test : tests)
    {
        if (const auto result = test(); result != 0)
        {
            return result;
        }
    }
    return 0;
}
