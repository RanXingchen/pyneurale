/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/linear_processor_chain.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>

namespace neurale::streaming
{
namespace
{

[[nodiscard]] bool equal_stream_schema(const StreamSchema& left, const StreamSchema& right) noexcept
{
    return left.equivalent(right);
}

[[nodiscard]] bool equal_contract(const PreparedProcessorContract& left,
                                  const PreparedProcessorContract& right)
{
    return equal_stream_schema(left.accepted_input_schema, right.accepted_input_schema) &&
           equal_stream_schema(left.output_schema, right.output_schema) &&
           left.max_process_outputs_per_input == right.max_process_outputs_per_input &&
           left.max_flush_outputs == right.max_flush_outputs &&
           left.can_forward_input == right.can_forward_input &&
           left.required_resources.workspace_bytes == right.required_resources.workspace_bytes &&
           left.required_resources.frame_pool_leases == right.required_resources.frame_pool_leases;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::overflow_error("linear processor chain bound overflows size_t");
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error("linear processor chain fan-out overflows size_t");
    }
    return left * right;
}

[[nodiscard]] std::size_t schema_payload_bytes(const StreamSchema& schema)
{
    std::size_t total = 0;
    for (const auto& signal : schema.signals())
    {
        if (signal.max_block_bytes > std::numeric_limits<std::size_t>::max())
        {
            throw std::overflow_error("linear processor schema exceeds size_t");
        }
        total = checked_add(total, static_cast<std::size_t>(signal.max_block_bytes));
    }
    return total;
}

} // namespace

class LinearProcessorChain::StageEmitter final : public FrameEmitter
{
  public:
    StageEmitter(LinearProcessorChain& owner, std::size_t stage_idx) noexcept
        : owner_(owner), stage_idx_(stage_idx)
    {
    }

    void begin(MutableFrame& input, FrameLease* input_lease, FrameEmitter& destination,
               std::size_t output_limit, bool input_forwarding) noexcept
    {
        input_ = &input;
        input_lease_ = input_lease;
        destination_ = &destination;
        output_limit_ = output_limit;
        published_count_ = 0;
        failure_ = StreamStatus::ok;
        active_ = true;
        input_forwarding_ = input_forwarding;
        delegated_acquisition_ = false;
        delegated_borrow_ = {};
        invalidate_borrow(input_borrow_state_);
        invalidate_acquired_borrow();
        static_cast<void>(acquired_.reset());
        input_borrow_ = {};
        const auto status = activate_borrow(input_borrow_state_, input, input_borrow_);
        if (status != StreamStatus::ok)
        {
            remember_failure(status);
        }
    }

    [[nodiscard]] StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override;
    [[nodiscard]] StreamStatus publish_acquired_frame() noexcept override;
    [[nodiscard]] StreamStatus publish_input() noexcept override;

    [[nodiscard]] FrameBorrow& input_borrow() noexcept
    {
        return input_borrow_;
    }

    [[nodiscard]] StreamStatus finish() noexcept
    {
        if (!active_)
        {
            return StreamStatus::invalid_state;
        }
        if (acquired_ || delegated_acquisition_)
        {
            invalidate_acquired_borrow();
            static_cast<void>(acquired_.reset());
            remember_failure(StreamStatus::invalid_state);
        }
        input_ = nullptr;
        input_lease_ = nullptr;
        destination_ = nullptr;
        invalidate_borrow(input_borrow_state_);
        invalidate_acquired_borrow();
        active_ = false;
        return failure_;
    }

  private:
    [[nodiscard]] StreamStatus publish_owned(FrameLease lease) noexcept override;
    [[nodiscard]] StreamStatus publish_intermediate(FrameLease lease) noexcept;
    [[nodiscard]] StreamStatus publish_borrowed() noexcept;
    [[nodiscard]] bool can_publish() noexcept;
    void remember_failure(StreamStatus status) noexcept
    {
        if (failure_ == StreamStatus::ok)
        {
            failure_ = status;
        }
    }

    LinearProcessorChain& owner_;
    std::size_t stage_idx_{};
    MutableFrame* input_{};
    FrameLease* input_lease_{};
    FrameEmitter* destination_{};
    FrameLease acquired_{};
    detail::FrameBorrowState input_borrow_state_{};
    FrameBorrow input_borrow_{};
    FrameBorrow delegated_borrow_{};
    std::size_t output_limit_{};
    std::size_t published_count_{};
    StreamStatus failure_{StreamStatus::ok};
    bool active_{};
    bool input_forwarding_{};
    bool delegated_acquisition_{};
};

struct LinearProcessorChain::Impl
{
    Impl(LinearProcessorChain& chain, std::span<NativeFrameProcessor* const> stage_span)
        : owner(chain), stages(stage_span.begin(), stage_span.end())
    {
        if (stages.empty())
        {
            throw std::invalid_argument("linear processor chain requires at least one stage");
        }
        if (std::any_of(stages.begin(), stages.end(),
                        [](const auto* stage) { return stage == nullptr; }))
        {
            throw std::invalid_argument("linear processor chain stages must not be null");
        }
        emitters.reserve(stages.size());
        for (std::size_t i = 0; i < stages.size(); ++i)
        {
            emitters.push_back(std::make_unique<StageEmitter>(owner, i));
        }
    }

    [[nodiscard]] StreamStatus invoke_stage(std::size_t idx, MutableFrame& frame,
                                            FrameLease* input_lease,
                                            FrameEmitter& destination) noexcept
    {
        auto& emitter = *emitters[idx];
        const auto& contract = contracts[idx];
        emitter.begin(frame, input_lease, destination, contract.max_process_outputs_per_input,
                      contract.can_forward_input);
        const auto processor_status = stages[idx]->process(emitter.input_borrow(), emitter);
        const auto emitter_status = emitter.finish();
        return emitter_status != StreamStatus::ok ? emitter_status : processor_status;
    }

    [[nodiscard]] StreamStatus invoke_next_borrowed(std::size_t idx, MutableFrame& frame,
                                                    FrameEmitter& destination) noexcept
    {
        if (idx + 1 == stages.size())
        {
            return destination.publish_input();
        }
        return invoke_stage(idx + 1, frame, nullptr, destination);
    }

    [[nodiscard]] StreamStatus invoke_next_owned(std::size_t idx, FrameLease lease,
                                                 FrameEmitter& destination) noexcept
    {
        if (idx + 1 == stages.size())
        {
            return owner.publish_owned(destination, std::move(lease));
        }
        auto& frame = lease.frame();
        return invoke_stage(idx + 1, frame, &lease, destination);
    }

    LinearProcessorChain& owner;
    std::vector<NativeFrameProcessor*> stages;
    std::vector<PreparedProcessorContract> contracts;
    std::vector<std::unique_ptr<FrameValidator>> validators;
    std::vector<std::unique_ptr<StageEmitter>> emitters;
    std::unique_ptr<FramePool> intermediate_pool;
    bool prepared{};
};

bool LinearProcessorChain::StageEmitter::can_publish() noexcept
{
    if (!active_ || destination_ == nullptr || failure_ != StreamStatus::ok)
    {
        if (!active_ || destination_ == nullptr)
        {
            remember_failure(StreamStatus::invalid_state);
        }
        return false;
    }
    if (published_count_ >= output_limit_)
    {
        remember_failure(StreamStatus::output_limit);
        return false;
    }
    return true;
}

StreamStatus LinearProcessorChain::StageEmitter::try_acquire_frame(MutableFrame*& frame) noexcept
{
    frame = nullptr;
    if (!can_publish())
    {
        return failure_;
    }
    if (acquired_ || delegated_acquisition_)
    {
        remember_failure(StreamStatus::invalid_state);
        return failure_;
    }
    if (stage_idx_ + 1 == owner_.impl_->stages.size())
    {
        const auto status = destination_->try_acquire(delegated_borrow_);
        if (status == StreamStatus::ok)
        {
            frame = delegated_borrow_.get();
            delegated_acquisition_ = true;
        }
        else
        {
            remember_failure(status);
        }
        return status;
    }
    const auto status = owner_.impl_->intermediate_pool->try_acquire(acquired_);
    if (status != StreamStatus::ok)
    {
        remember_failure(status);
        return status;
    }
    frame = &acquired_.frame();
    return StreamStatus::ok;
}

StreamStatus LinearProcessorChain::StageEmitter::publish_intermediate(FrameLease lease) noexcept
{
    if (!can_publish())
    {
        return failure_;
    }
    if (owner_.impl_->validators[stage_idx_]->validate(lease.view()) != FrameValidationError::none)
    {
        remember_failure(StreamStatus::invalid_frame);
        return failure_;
    }
    const auto status =
        owner_.impl_->invoke_next_owned(stage_idx_, std::move(lease), *destination_);
    if (status != StreamStatus::ok)
    {
        remember_failure(status);
        return status;
    }
    ++published_count_;
    return StreamStatus::ok;
}

StreamStatus LinearProcessorChain::StageEmitter::publish_borrowed() noexcept
{
    if (!can_publish())
    {
        return failure_;
    }
    if (!input_forwarding_ || input_ == nullptr)
    {
        remember_failure(StreamStatus::invalid_state);
        return failure_;
    }
    if (owner_.impl_->validators[stage_idx_]->validate(input_->view()) !=
        FrameValidationError::none)
    {
        remember_failure(StreamStatus::invalid_frame);
        return failure_;
    }

    StreamStatus status{};
    if (input_lease_ == nullptr)
    {
        status = owner_.impl_->invoke_next_borrowed(stage_idx_, *input_, *destination_);
    }
    else
    {
        FrameLease lease = std::move(*input_lease_);
        status = owner_.impl_->invoke_next_owned(stage_idx_, std::move(lease), *destination_);
    }
    if (status != StreamStatus::ok)
    {
        remember_failure(status);
        return status;
    }
    ++published_count_;
    return StreamStatus::ok;
}

StreamStatus LinearProcessorChain::StageEmitter::publish_acquired_frame() noexcept
{
    if (delegated_acquisition_)
    {
        if (!can_publish())
        {
            return failure_;
        }
        const auto status = destination_->publish_acquired();
        delegated_acquisition_ = false;
        if (status != StreamStatus::ok)
        {
            remember_failure(status);
            return status;
        }
        ++published_count_;
        return StreamStatus::ok;
    }
    if (!acquired_)
    {
        remember_failure(StreamStatus::invalid_state);
        return failure_;
    }
    return publish_intermediate(std::move(acquired_));
}

StreamStatus LinearProcessorChain::StageEmitter::publish_input() noexcept
{
    invalidate_borrow(input_borrow_state_);
    return publish_borrowed();
}

StreamStatus LinearProcessorChain::StageEmitter::publish_owned(FrameLease lease) noexcept
{
    return publish_intermediate(std::move(lease));
}

LinearProcessorChain::LinearProcessorChain(std::span<NativeFrameProcessor* const> stages)
    : impl_(std::make_unique<Impl>(*this, stages))
{
}

LinearProcessorChain::~LinearProcessorChain() = default;

PreparedProcessorContract LinearProcessorChain::prepare(const ProcessorPrepareContext& context)
{
    impl_->prepared = false;
    std::vector<PreparedProcessorContract> declared;
    declared.reserve(impl_->stages.size());

    const StreamSchema* input_schema = &context.input_schema;
    for (auto* stage : impl_->stages)
    {
        declared.push_back(stage->prepare(ProcessorPrepareContext{
            .input_schema = *input_schema,
            .max_process_outputs = context.max_process_outputs,
            .max_flush_outputs = context.max_flush_outputs,
            .available_frame_pool_leases = context.available_frame_pool_leases,
        }));
        const auto& contract = declared.back();
        if (!equal_stream_schema(contract.accepted_input_schema, *input_schema))
        {
            throw std::invalid_argument(
                "linear processor stage rejects the preceding output schema");
        }
        if (contract.required_resources.frame_pool_leases == 0 ||
            contract.required_resources.frame_pool_leases > context.available_frame_pool_leases)
        {
            throw std::invalid_argument("linear processor stage frame-pool bound is unavailable");
        }
        if (contract.can_forward_input &&
            !equal_stream_schema(contract.accepted_input_schema, contract.output_schema))
        {
            throw std::invalid_argument(
                "input forwarding requires identical input and output schemas");
        }
        input_schema = &contract.output_schema;
    }

    std::size_t process_outputs = 1;
    std::size_t flush_outputs = 0;
    std::size_t downstream_process_outputs = 1;
    std::size_t workspace_bytes = 0;
    std::size_t outer_pool_leases = 0;
    bool can_forward_input = true;
    for (auto iterator = declared.rbegin(); iterator != declared.rend(); ++iterator)
    {
        flush_outputs = checked_add(flush_outputs, checked_multiply(iterator->max_flush_outputs,
                                                                    downstream_process_outputs));
        downstream_process_outputs =
            checked_multiply(iterator->max_process_outputs_per_input, downstream_process_outputs);
    }
    process_outputs = downstream_process_outputs;
    for (const auto& contract : declared)
    {
        workspace_bytes = checked_add(workspace_bytes, contract.required_resources.workspace_bytes);
        outer_pool_leases =
            std::max(outer_pool_leases, contract.required_resources.frame_pool_leases);
        can_forward_input = can_forward_input && contract.can_forward_input;
    }
    if (process_outputs > context.max_process_outputs || flush_outputs > context.max_flush_outputs)
    {
        throw std::invalid_argument(
            "linear processor chain fan-out exceeds terminal output bounds");
    }

    if (!impl_->contracts.empty())
    {
        if (impl_->contracts.size() != declared.size())
        {
            throw std::invalid_argument("linear processor stage list changed");
        }
        for (std::size_t i = 0; i < declared.size(); ++i)
        {
            if (!equal_contract(impl_->contracts[i], declared[i]))
            {
                throw std::invalid_argument("linear processor contract changed after reset");
            }
        }
    }
    else
    {
        impl_->contracts = std::move(declared);
    }

    impl_->validators.clear();
    impl_->validators.reserve(impl_->contracts.size());
    for (const auto& contract : impl_->contracts)
    {
        impl_->validators.push_back(std::make_unique<FrameValidator>(contract.output_schema));
    }

    if (impl_->stages.size() > 1)
    {
        std::size_t payload_bytes = 0;
        std::size_t max_signal_blocks = 0;
        for (std::size_t i = 0; i + 1 < impl_->contracts.size(); ++i)
        {
            payload_bytes =
                std::max(payload_bytes, schema_payload_bytes(impl_->contracts[i].output_schema));
            max_signal_blocks =
                std::max(max_signal_blocks, impl_->contracts[i].output_schema.signals().size());
        }
        impl_->intermediate_pool =
            std::make_unique<FramePool>(impl_->stages.size() - 1, payload_bytes, max_signal_blocks);
        impl_->intermediate_pool->prefault();
    }
    else
    {
        impl_->intermediate_pool.reset();
    }
    impl_->prepared = true;

    const auto& first = impl_->contracts.front();
    const auto& last = impl_->contracts.back();
    return {first.accepted_input_schema.clone(),
            last.output_schema.clone(),
            process_outputs,
            flush_outputs,
            can_forward_input,
            ProcessorResourceBounds{.workspace_bytes = workspace_bytes,
                                    .frame_pool_leases = outer_pool_leases}};
}

StreamStatus LinearProcessorChain::process(FrameBorrow& frame, FrameEmitter& emitter) noexcept
{
    if (!impl_->prepared)
    {
        return StreamStatus::invalid_state;
    }
    auto* mutable_frame = frame.get();
    if (mutable_frame == nullptr)
    {
        return StreamStatus::invalid_state;
    }
    return impl_->invoke_stage(0, *mutable_frame, nullptr, emitter);
}

StreamStatus LinearProcessorChain::handle_discontinuity(const Discontinuity& discontinuity) noexcept
{
    if (!impl_->prepared ||
        (impl_->intermediate_pool != nullptr && impl_->intermediate_pool->outstanding() != 0))
    {
        return StreamStatus::invalid_state;
    }
    for (auto* stage : impl_->stages)
    {
        const auto status = stage->handle_discontinuity(discontinuity);
        if (status != StreamStatus::ok)
        {
            return status;
        }
    }
    return StreamStatus::ok;
}

StreamStatus LinearProcessorChain::flush(FrameEmitter& emitter) noexcept
{
    if (!impl_->prepared)
    {
        return StreamStatus::invalid_state;
    }
    for (std::size_t i = 0; i < impl_->stages.size(); ++i)
    {
        auto& stage_emitter = *impl_->emitters[i];
        const auto& contract = impl_->contracts[i];
        MutableFrame unused;
        stage_emitter.begin(unused, nullptr, emitter, contract.max_flush_outputs, false);
        const auto processor_status = impl_->stages[i]->flush(stage_emitter);
        const auto emitter_status = stage_emitter.finish();
        if (emitter_status != StreamStatus::ok)
        {
            return emitter_status;
        }
        if (processor_status != StreamStatus::ok)
        {
            return processor_status;
        }
    }
    return StreamStatus::ok;
}

StreamStatus LinearProcessorChain::reset() noexcept
{
    StreamStatus first_failure = StreamStatus::ok;
    for (auto* stage : impl_->stages)
    {
        const auto status = stage->reset();
        if (first_failure == StreamStatus::ok && status != StreamStatus::ok)
        {
            first_failure = status;
        }
    }
    impl_->prepared = false;
    if (impl_->intermediate_pool != nullptr && impl_->intermediate_pool->outstanding() != 0 &&
        first_failure == StreamStatus::ok)
    {
        first_failure = StreamStatus::invalid_state;
    }
    return first_failure;
}

std::size_t LinearProcessorChain::stage_count() const noexcept
{
    return impl_->stages.size();
}

StreamStatus LinearProcessorChain::publish_owned(FrameEmitter& destination,
                                                 FrameLease lease) noexcept
{
    return destination.publish_owned(std::move(lease));
}

} // namespace neurale::streaming
