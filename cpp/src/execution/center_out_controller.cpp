/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "center_out_controller.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <neurale/streaming/clock.h>

#include "contract_status.h"

namespace neurale::execution
{

using namespace neurale::experiments;
namespace
{

[[nodiscard]] bool compatible_velocity_space(const CenterOutControllerConfig& config) noexcept
{
    if (validate(config.velocity_space) != ContractStatus::ok || config.velocity_space.dim != 2 ||
        config.velocity_space.frame != CommandFrame::workspace_2d ||
        config.velocity_space.axes[0].name != CommandAxisName::x ||
        config.velocity_space.axes[1].name != CommandAxisName::y)
        return false;

    const auto x_unit = config.velocity_space.axes[0].unit;
    if (x_unit != config.velocity_space.axes[1].unit)
        return false;
    switch (config.task.geometry_unit)
    {
    case center_out::GeometryUnit::dimensionless:
        return x_unit == CommandUnit::dimensionless;
    case center_out::GeometryUnit::normalized:
        return x_unit == CommandUnit::normalized;
    case center_out::GeometryUnit::metres:
    case center_out::GeometryUnit::millimetres:
    case center_out::GeometryUnit::unspecified:
        return false;
    }
    return false;
}

[[nodiscard]] bool add_u64(std::uint64_t left, std::uint64_t right, std::uint64_t& value) noexcept
{
    if (left > (std::numeric_limits<std::uint64_t>::max)() - right)
        return false;
    value = left + right;
    return true;
}

[[nodiscard]] bool valid_assistance_schedule(const CenterOutControllerConfig& config) noexcept
{
    if (config.assistance_blocks.empty())
        return true;
    if (config.assistance_method != assistance::AssistanceMethod::linear_blend)
        return false;
    std::uint64_t trials = 0;
    for (const auto& block : config.assistance_blocks)
    {
        if (block.trials == 0 || assistance::validate(block.linear) != ContractStatus::ok ||
            trials > (std::numeric_limits<std::uint64_t>::max)() - block.trials)
            return false;
        trials += block.trials;
    }
    return trials == config.task.trial_limit;
}

[[nodiscard]] bool valid_cursor_bounds(const CenterOutControllerConfig& config) noexcept
{
    return std::isfinite(config.cursor_min.x) && std::isfinite(config.cursor_min.y) &&
           std::isfinite(config.cursor_max.x) && std::isfinite(config.cursor_max.y) &&
           config.cursor_min.x <= config.cursor_max.x &&
           config.cursor_min.y <= config.cursor_max.y &&
           config.initial_position.x >= config.cursor_min.x &&
           config.initial_position.x <= config.cursor_max.x &&
           config.initial_position.y >= config.cursor_min.y &&
           config.initial_position.y <= config.cursor_max.y;
}
} // namespace

streaming::StreamStatus CenterOutController::prepare(const streaming::StreamSchema& schema,
                                                     const CenterOutControllerConfig& config)
{
    if (prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    const auto signals = schema.signals();
    if (signals.size() != 1 || signals.front().id != config.decoded_signal_id)
        return streaming::StreamStatus::realtime_configuration_failed;
    const auto& signal = signals.front();
    if (signal.kind != streaming::SignalKind::sampled ||
        signal.dtype != streaming::SignalDType::float64 ||
        signal.layout != streaming::SignalLayout::sample_major || signal.n_channels != 2 ||
        signal.max_block_samples == 0 || signal.fs.numerator == 0 || signal.fs.denominator == 0 ||
        signal.physical_unit != streaming::PhysicalUnit::dimensionless)
        return streaming::StreamStatus::realtime_configuration_failed;
    if (config.paradigm == kUnsetParadigmId || config.trace_capacity == 0 ||
        validate(config.abnormal) != ContractStatus::ok ||
        center_out::validate(config.task) != ContractStatus::ok ||
        center_out::validate(config.guidance) != ContractStatus::ok ||
        center_out::validate_against(config.guidance, config.task) != ContractStatus::ok ||
        !compatible_velocity_space(config) || !valid_assistance_schedule(config) ||
        (config.assistance_method == assistance::AssistanceMethod::none
             ? config.linear_assistance.assistance != 0.0
             : (config.assistance_method != assistance::AssistanceMethod::linear_blend ||
                assistance::validate(config.linear_assistance) != ContractStatus::ok)) ||
        !std::isfinite(config.initial_position.x) || !std::isfinite(config.initial_position.y) ||
        !valid_cursor_bounds(config))
        return streaming::StreamStatus::realtime_configuration_failed;

    if (signal.fs.denominator > (std::numeric_limits<std::uint64_t>::max)() / kNanosecondsPerSecond)
        return streaming::StreamStatus::realtime_configuration_failed;
    const auto scaled = signal.fs.denominator * kNanosecondsPerSecond;
    if (scaled % signal.fs.numerator != 0)
        return streaming::StreamStatus::realtime_configuration_failed;
    const auto period = scaled / signal.fs.numerator;
    if (period == 0)
        return streaming::StreamStatus::realtime_configuration_failed;

    traces_.prepare(config.trace_capacity);
    presentation_failures_.prepare(64);
    if (config.training_capture_capacity != 0)
        training_labels_.prepare(config.training_capture_capacity);
    if (config.presentation_state_capacity != 0)
        presentation_states_.prepare(config.presentation_state_capacity);
    validator_ = std::make_unique<streaming::FrameValidator>(schema);
    config_ = config;
    schema_id_ = schema.id();
    signal_ = signal;
    observation_period_ns_ = period;
    position_ = config.initial_position;
    if (guidance_.configure(config.guidance) != ContractStatus::ok)
        return streaming::StreamStatus::realtime_configuration_failed;
    abnormal_.configure(config.abnormal, config.paradigm);
    reset_assistance_schedule();
    prepared_ = true;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus CenterOutController::start(streaming::HostTimeNs host_epoch_ns,
                                                   ExperimentTimeNs experiment_epoch_ns) noexcept
{
    if (!prepared_ || running_ || closed_)
        return streaming::StreamStatus::invalid_state;
    traces_.reset();
    presentation_failures_.reset();
    if (config_.training_capture_capacity != 0)
        training_labels_.reset();
    if (config_.presentation_state_capacity != 0)
        presentation_states_.reset();
    machine_.reset();
    guidance_.reset();
    position_ = config_.initial_position;
    host_epoch_ns_ = host_epoch_ns;
    experiment_epoch_ns_ = experiment_epoch_ns;
    last_time_ns_ = experiment_epoch_ns;
    has_last_time_ = true;
    pending_restart_ = false;
    has_trial_ = false;
    next_observation_ordinal_ = 0;
    last_observation_ordinal_ = 0;
    has_observation_ordinal_ = false;
    completed_trials_.store(0, std::memory_order_release);
    complete_.store(false, std::memory_order_release);
    trial_started_ns_ = experiment_epoch_ns;
    halt_refusal_recorded_ = false;
    halt_pending_ = false;
    cancelled_.store(false, std::memory_order_release);
    halted_.store(false, std::memory_order_release);
    abnormal_.restart();
    reset_assistance_schedule();

    center_out::CenterOutStepResult step{};
    const auto status = begin_segment(experiment_epoch_ns, step);
    if (status != streaming::StreamStatus::ok)
        return status;
    CenterOutControlTrace trace{};
    trace.kind = CenterOutTraceKind::session_start;
    trace.time_ns = experiment_epoch_ns;
    trace.position_before = position_;
    trace.position_after = position_;
    trace.step = step;
    if (traces_.try_push(trace) != streaming::StreamStatus::ok)
    {
        machine_.reset();
        return streaming::StreamStatus::queue_overflow;
    }
    running_ = true;
    pending_has_abnormal_ = false;
    pending_has_aborted_trial_ = false;
    return streaming::StreamStatus::ok;
}

void CenterOutController::close() noexcept
{
    if (closed_)
        return;
    cancelled_.store(true, std::memory_order_release);
    running_ = false;
    pending_restart_ = false;
    traces_.close();
    presentation_failures_.close();
    if (config_.training_capture_capacity != 0)
        training_labels_.close();
    if (config_.presentation_state_capacity != 0)
        presentation_states_.close();
    closed_ = true;
}

streaming::StreamStatus
CenterOutController::begin_segment(ExperimentTimeNs time_ns,
                                   center_out::CenterOutStepResult& result) noexcept
{
    machine_.reset();
    guidance_.reset();
    has_trial_ = false;
    trial_started_ns_ = time_ns;
    const auto status = machine_.start(config_.paradigm, config_.task, time_ns, result);
    if (status == ContractStatus::ok)
        note_trial_boundaries(result);
    return map_contract_status(status);
}

streaming::StreamStatus CenterOutController::process_observation(
    const double* values, std::uint64_t frame_sequence, streaming::SampleIndex sample_idx,
    ExperimentTimeNs time_ns, streaming::HostTimeNs input_ready_ns,
    const center_out::CenterOutStepResult* segment_start) noexcept
{
    if (machine_.complete())
        return streaming::StreamStatus::ok;
    // Finiteness, time order, and the interval this run will integrate across
    // are screened by consume() before a row reaches here, because each of them
    // is an abnormal condition with a policy rather than a malformed argument.
    // What is left is arithmetic on values that are already known to be usable.

    const DurationNs dt_ns = has_last_time_ ? time_ns - last_time_ns_ : 0;
    const auto before = position_;
    assistance::VelocityVector decoded{};
    decoded.space = config_.velocity_space.id;
    decoded.dim = 2;
    decoded.values[0] = values[0];
    decoded.values[1] = values[1];
    auto* runtime_clock = runtime_clock_.load(std::memory_order_acquire);
    const auto decoded_ready_ns =
        (runtime_clock == nullptr ? streaming::default_native_clock() : *runtime_clock).now_ns();
    const auto applied_linear_assistance = active_linear_assistance_;

    assistance::VelocityVector reference{};
    reference.space = config_.velocity_space.id;
    reference.dim = 2;
    center_out::CenterOutGuidanceSample guidance_sample{};
    const auto snapshot_before = machine_.snapshot();
    if (snapshot_before.active_target != kUnsetTargetId)
    {
        const center_out::TargetPlacement target{snapshot_before.active_target,
                                                 snapshot_before.active_position};
        const auto status = guidance_.update(target, position_, dt_ns, guidance_sample);
        if (status != ContractStatus::ok)
            return map_contract_status(status);
        reference.values[0] = guidance_sample.vel.x;
        reference.values[1] = guidance_sample.vel.y;
    }
    else
    {
        guidance_.reset();
    }
    intent_context_revision_.fetch_add(1, std::memory_order_acq_rel);
    std::atomic_thread_fence(std::memory_order_release);
    intent_target_id_.store(snapshot_before.active_target, std::memory_order_relaxed);
    intent_trial_key_.store(snapshot_before.trial.key, std::memory_order_relaxed);
    intent_state_->publish({next_observation_ordinal_, reference.values[0], reference.values[1],
                            snapshot_before.trial.ordinal, time_ns, frame_sequence, sample_idx,
                            snapshot_before.active_target != kUnsetTargetId});
    intent_context_revision_.fetch_add(1, std::memory_order_release);

    assistance::VelocityVector assisted = decoded;
    if (config_.assistance_method == assistance::AssistanceMethod::linear_blend)
    {
        const auto status = assistance::blend_velocity(config_.velocity_space, decoded, reference,
                                                       applied_linear_assistance, assisted);
        if (status != ContractStatus::ok)
            return map_contract_status(status);
    }

    const auto dt_seconds = static_cast<double>(dt_ns) / static_cast<double>(kNanosecondsPerSecond);
    const auto next_x = position_.x + assisted.values[0] * dt_seconds;
    const auto next_y = position_.y + assisted.values[1] * dt_seconds;
    if (!std::isfinite(next_x) || !std::isfinite(next_y))
        return streaming::StreamStatus::consumer_failure;
    position_ = {std::clamp(next_x, config_.cursor_min.x, config_.cursor_max.x),
                 std::clamp(next_y, config_.cursor_min.y, config_.cursor_max.y)};
    const center_out::WorkspacePoint cursor_velocity{
        dt_seconds == 0.0 ? 0.0 : (position_.x - before.x) / dt_seconds,
        dt_seconds == 0.0 ? 0.0 : (position_.y - before.y) / dt_seconds,
    };

    center_out::CenterOutStepResult step{};
    const auto machine_status = machine_.step(time_ns, position_, step);
    if (machine_status != ContractStatus::ok)
    {
        position_ = before;
        return map_contract_status(machine_status);
    }
    note_trial_boundaries(step);

    CenterOutControlTrace trace{};
    trace.kind = CenterOutTraceKind::observation;
    trace.observation_ordinal = next_observation_ordinal_;
    trace.frame_sequence = frame_sequence;
    trace.sample_idx = sample_idx;
    trace.decoder_version = decoder_version_;
    trace.time_ns = time_ns;
    trace.input_ready_ns = input_ready_ns;
    trace.decoded_ready_ns = decoded_ready_ns;
    trace.dt_ns = dt_ns;
    trace.position_before = before;
    trace.position_after = position_;
    trace.decoded = decoded;
    trace.guidance = reference;
    trace.assisted = assisted;
    trace.linear_assistance = applied_linear_assistance;
    trace.guidance_sample = guidance_sample;
    if (segment_start != nullptr)
        trace.segment_start = *segment_start;
    trace.step = step;
    trace.restarted_after_discontinuity = segment_start != nullptr;
    // A decision taken about this row travels with it. consume() only leaves
    // one pending when an observation is going to follow, so there is no path
    // on which a decision waits here for a record that never comes.
    if (pending_has_abnormal_)
    {
        trace.abnormal = pending_abnormal_;
        trace.has_abnormal = true;
        trace.aborted_trial = pending_aborted_trial_;
        trace.has_aborted_trial = pending_has_aborted_trial_;
    }
    const auto push_status = traces_.try_push(trace);
    if (push_status != streaming::StreamStatus::ok)
        return push_status;
    if (config_.training_capture_capacity != 0)
    {
        CenterOutTrainingLabel label{};
        label.sample_idx = sample_idx;
        label.time_ns = time_ns;
        label.trial = snapshot_before.trial;
        label.target_position = snapshot_before.active_position;
        label.cursor_position = before;
        label.cursor_velocity = cursor_velocity;
        label.guidance = reference;
        for (std::uint8_t i = 0; i < step.n_events; ++i)
            label.trial_stop |= step.events[i].kind == ExperimentEventKind::trial_stop;
        const auto label_status = training_labels_.try_push(label);
        if (label_status != streaming::StreamStatus::ok)
            return label_status;
    }
    if (config_.presentation_state_capacity != 0)
    {
        const CenterOutPresentationState presentation{machine_.snapshot(), position_,
                                                      next_observation_ordinal_, time_ns};
        // Presentation is not a critical data-plane edge. A slow window must
        // not stall acquisition; the monotonic drop count makes the resulting
        // run explicitly incomplete instead.
        static_cast<void>(presentation_states_.try_push(presentation));
    }
    pending_has_abnormal_ = false;
    pending_has_aborted_trial_ = false;
    last_observation_ordinal_ = next_observation_ordinal_;
    has_observation_ordinal_ = true;
    ++next_observation_ordinal_;
    last_time_ns_ = time_ns;
    has_last_time_ = true;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus CenterOutController::consume(streaming::FrameView frame) noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt();
    // Frame-budgeted: this is the acquisition thread, not shutdown.
    const auto presentation_status = apply_presentation_failures(kPresentationFailureDrainBudget);
    if (presentation_status != streaming::StreamStatus::ok)
        return presentation_status;
    if (!running_)
        return streaming::StreamStatus::invalid_state;
    if (validator_->validate(frame) != streaming::FrameValidationError::none ||
        frame.header.schema_id != schema_id_ || frame.blocks.size() != 1 ||
        frame.header.source_clock_domain != signal_.clock_domain ||
        frame.blocks.front().signal_id != signal_.id)
    {
        // Not the stream this run prepared against. The floor is the run rather
        // than the trial because the next frame is wrong in exactly the same
        // way: a task that restarted its segment per malformed frame would
        // restart forever while recording nothing usable.
        return report_abnormal(AbnormalCondition::input_schema_mismatch, last_time_ns_,
                               static_cast<std::uint32_t>(streaming::StreamStatus::invalid_frame),
                               streaming::StreamStatus::invalid_frame);
    }
    if (machine_.complete())
        return streaming::StreamStatus::ok;

    const auto& block = frame.blocks.front();
    if (!traces_.can_push(block.n_samples))
    {
        // Refused before a single row is processed, so the whole frame's worth
        // of observations is what the refusal cost. Counting one would report a
        // hole of one record where the trace is missing the whole frame.
        traces_.note_dropped(block.n_samples);
        return streaming::StreamStatus::queue_overflow;
    }
    if (config_.training_capture_capacity != 0 && !training_labels_.can_push(block.n_samples))
    {
        training_labels_.note_dropped(block.n_samples);
        return streaming::StreamStatus::queue_overflow;
    }
    const auto* payload = frame.payload.data() + block.payload_offset;
    for (std::uint32_t row = 0; row < block.n_samples; ++row)
    {
        if (row > (std::numeric_limits<std::uint64_t>::max)() / observation_period_ns_)
            return streaming::StreamStatus::invalid_frame;
        const auto row_offset = static_cast<std::uint64_t>(row) * observation_period_ns_;
        std::uint64_t host_time{};
        if (!add_u64(block.observation_time_start_ns, row_offset, host_time) ||
            host_time < host_epoch_ns_)
            return streaming::StreamStatus::invalid_frame;
        std::uint64_t time_ns{};
        if (!add_u64(experiment_epoch_ns_, host_time - host_epoch_ns_, time_ns))
            return streaming::StreamStatus::invalid_frame;

        double values[2]{};
        std::memcpy(values, payload + static_cast<std::size_t>(row) * 2 * sizeof(double),
                    sizeof(values));

        // A decoded value that is not a number never reaches the cursor, the
        // guidance blend, or the machine. Under the default policy the trial it
        // would have been part of is ended too: a reach whose trajectory has a
        // hole in it is not a reach anyone can score, and continuing produces a
        // longer trial rather than a better one.
        if (!std::isfinite(values[0]) || !std::isfinite(values[1]))
        {
            const auto refused =
                report_abnormal(AbnormalCondition::decoded_command_invalid, time_ns,
                                static_cast<std::uint32_t>(streaming::StreamStatus::invalid_frame),
                                streaming::StreamStatus::invalid_frame);
            if (refused != streaming::StreamStatus::ok)
                return refused;
            continue;
        }
        // Time that moved backwards is a malformed stream, not a decoder that
        // produced something odd, so it is the schema condition and it ends the
        // run.
        if (has_last_time_ && time_ns < last_time_ns_)
        {
            return report_abnormal(
                AbnormalCondition::input_schema_mismatch, last_time_ns_,
                static_cast<std::uint32_t>(streaming::StreamStatus::invalid_frame),
                streaming::StreamStatus::invalid_frame);
        }
        // More time passed than this run will apply one decoded command across.
        // The observation is not refused -- it becomes the first of a new
        // segment, with dt zero -- but the interval before it is, which is what
        // stops one sample from standing for a stretch of time nothing was
        // observed over.
        //
        // Decided here and recorded *on* the observation that follows, rather
        // than in a record of its own. The frame reserved one queue slot per
        // row before it changed anything; a row that produced two records would
        // spend a slot the reservation never covered, and the row after it
        // would be the one refused -- after the machine had already been reset
        // and restarted. One row, one slot, whatever it had to decide.
        if (config_.max_observation_interval_ns != 0 && has_last_time_ && !machine_.complete() &&
            time_ns - last_time_ns_ > config_.max_observation_interval_ns)
        {
            CenterOutControlTrace decision{};
            decision.time_ns = time_ns;
            decision.position_before = position_;
            decision.position_after = position_;
            apply_abnormal(decision, AbnormalCondition::input_stale, time_ns, 0);
            if (decision.abnormal.response == AbnormalResponse::session_aborted)
            {
                // Nothing follows it to carry it, so it takes the row's own
                // slot and the run ends here.
                decision.kind = CenterOutTraceKind::abnormal;
                const auto push = traces_.try_push(decision);
                return push != streaming::StreamStatus::ok
                           ? push
                           : streaming::StreamStatus::consumer_failure;
            }
            pending_abnormal_ = decision.abnormal;
            pending_aborted_trial_ = decision.aborted_trial;
            pending_has_abnormal_ = true;
            pending_has_aborted_trial_ = decision.has_aborted_trial;
        }

        center_out::CenterOutStepResult restart_result{};
        const center_out::CenterOutStepResult* segment_start = nullptr;
        if (pending_restart_)
        {
            const auto start_status = begin_segment(time_ns, restart_result);
            if (start_status != streaming::StreamStatus::ok)
                return start_status;
            last_time_ns_ = time_ns;
            has_last_time_ = true;
            pending_restart_ = false;
            segment_start = &restart_result;
        }
        const auto status =
            process_observation(values, frame.header.sequence, block.sample_idx_start + row,
                                time_ns, frame.header.host_received_ns, segment_start);
        if (status != streaming::StreamStatus::ok)
            return status;
    }
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
CenterOutController::handle_discontinuity(const streaming::Discontinuity& discontinuity) noexcept
{
    if (!prepared_ || !running_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt();
    if (!traces_.can_push())
    {
        traces_.note_dropped(1);
        return streaming::StreamStatus::queue_overflow;
    }
    const bool complete = machine_.complete();

    CenterOutControlTrace trace{};
    trace.kind = CenterOutTraceKind::discontinuity;
    // The last experiment instant the run actually reached. A gap has no
    // observation of its own, and stamping the record with zero would put it
    // before the session began.
    trace.time_ns = last_time_ns_;
    trace.frame_sequence = discontinuity.actual_frame_sequence;
    trace.position_before = position_;
    trace.position_after = position_;
    trace.gap_reason = discontinuity.reason;
    // The gap and the decision taken about it travel in one record. Two records
    // for one event would let a reader find the gap without the decision, or
    // the decision without what caused it.
    //
    // The floor is the trial and not the configuration's business: a hold and a
    // reach are accumulated across consecutive observations, and there are no
    // consecutive observations across a gap. What the configuration still
    // chooses is whether the run continues at all.
    apply_abnormal(trace, AbnormalCondition::source_discontinuity, last_time_ns_,
                   static_cast<std::uint32_t>(discontinuity.reason));
    // Whether a segment will restart after this gap, stated in the record
    // rather than left for a reader to infer from what follows it. A machine
    // that had already completed does not restart.
    trace.restarted_after_discontinuity = !complete && pending_restart_;
    const auto push = traces_.try_push(trace);
    if (push != streaming::StreamStatus::ok)
        return push;
    return trace.abnormal.response == AbnormalResponse::session_aborted
               ? streaming::StreamStatus::consumer_failure
               : streaming::StreamStatus::ok;
}

void CenterOutController::note_trial_boundaries(
    const center_out::CenterOutStepResult& result) noexcept
{
    // The machine's own statement of when the trial began, not this layer's
    // guess at it. A trial record built from a guessed start is a measurement
    // this layer invented, and the interval is the one thing an aborted trial
    // still carries.
    for (std::uint8_t i = 0; i < result.n_events; ++i)
    {
        const auto& event = result.events[i];
        if (event.kind == ExperimentEventKind::trial_start)
        {
            trial_started_ns_ = event.time_ns;
            has_trial_ = true;
        }
        else if (event.kind == ExperimentEventKind::trial_stop)
        {
            has_trial_ = false;
            completed_trials_.fetch_add(1, std::memory_order_release);
            if (!config_.assistance_blocks.empty())
            {
                ++trials_in_assistance_block_;
                const auto& block = config_.assistance_blocks[assistance_block_idx_];
                if (trials_in_assistance_block_ == block.trials &&
                    assistance_block_idx_ + 1 < config_.assistance_blocks.size())
                {
                    ++assistance_block_idx_;
                    trials_in_assistance_block_ = 0;
                    active_linear_assistance_ =
                        config_.assistance_blocks[assistance_block_idx_].linear;
                }
            }
        }
        else if (event.kind == ExperimentEventKind::session_stop)
        {
            complete_.store(true, std::memory_order_release);
        }
    }
}

void CenterOutController::reset_assistance_schedule() noexcept
{
    assistance_block_idx_ = 0;
    trials_in_assistance_block_ = 0;
    active_linear_assistance_ = config_.assistance_blocks.empty()
                                    ? config_.linear_assistance
                                    : config_.assistance_blocks.front().linear;
}

void CenterOutController::apply_abnormal(CenterOutControlTrace& trace, AbnormalCondition condition,
                                         ExperimentTimeNs time_ns, std::uint32_t detail) noexcept
{
    const auto snapshot = machine_.snapshot();
    const bool in_trial = running_ && !machine_.complete();
    const auto handling = center_out::center_out_condition_handling(condition);
    // Center-Out can genuinely end the trial in flight: its machine is reset and
    // a new segment begins at the next accepted observation, which is the same
    // thing a discontinuity has always done here. The response says
    // `trial_aborted` because a trial really was ended, not merely distrusted.
    trace.abnormal =
        abnormal_.observe(condition, time_ns, in_trial ? &snapshot.trial : nullptr, detail,
                          handling.floor, /*can_end_trial=*/true, handling.input_refused);
    trace.has_abnormal = true;

    const auto response = trace.abnormal.response;
    if (response == AbnormalResponse::session_aborted)
    {
        // The run ends here. Every caller of this function returns a fatal
        // status when it sees this response, but a status only ends a run if
        // someone acts on it, and a caller-stepped session has no runtime that
        // will. Latched before the record is even queued, because what stops
        // the next frame is the latch and not the record.
        stop_accepting();
    }
    const bool ended = response == AbnormalResponse::trial_aborted ||
                       response == AbnormalResponse::session_aborted;
    if (ended && in_trial && has_trial_)
    {
        trace.has_aborted_trial = true;
        trace.aborted_trial.trial = snapshot.trial;
        trace.aborted_trial.interval = TimeInterval{
            trial_started_ns_, time_ns < trial_started_ns_ ? trial_started_ns_ : time_ns};
        trace.aborted_trial.paradigm = config_.paradigm;
        trace.aborted_trial.outcome = TrialOutcome::aborted;
        // `reason` stays unset. CenterOutReason is the machine's vocabulary for
        // decisions the machine made, and this is not one of them; the
        // condition that ended the trial is in the abnormal event beside it,
        // where it does not have to be kept in step with a second enumeration.
    }
    if (response == AbnormalResponse::trial_aborted)
    {
        machine_.reset();
        guidance_.reset();
        has_last_time_ = false;
        has_trial_ = false;
        pending_restart_ = true;
    }
}

streaming::StreamStatus CenterOutController::report_abnormal(AbnormalCondition condition,
                                                             ExperimentTimeNs time_ns,
                                                             std::uint32_t detail,
                                                             streaming::StreamStatus fatal) noexcept
{
    if (!traces_.can_push())
    {
        traces_.note_dropped(1);
        return streaming::StreamStatus::queue_overflow;
    }
    CenterOutControlTrace trace{};
    trace.kind = CenterOutTraceKind::abnormal;
    trace.time_ns = time_ns;
    trace.position_before = position_;
    trace.position_after = position_;
    apply_abnormal(trace, condition, time_ns, detail);
    const auto push = traces_.try_push(trace);
    if (push != streaming::StreamStatus::ok)
        return push;
    return trace.abnormal.response == AbnormalResponse::session_aborted
               ? fatal
               : streaming::StreamStatus::ok;
}

streaming::StreamStatus CenterOutController::refuse_after_halt() noexcept
{
    // Recorded once. A runtime that keeps delivering frames after an emergency
    // stop would otherwise fill the trace queue with identical records and turn
    // the stop into a trace loss.
    if (!halt_refusal_recorded_)
    {
        halt_refusal_recorded_ = true;
        static_cast<void>(report_abnormal(AbnormalCondition::input_after_terminal, last_time_ns_, 0,
                                          streaming::StreamStatus::stopped));
    }
    return streaming::StreamStatus::stopped;
}

void CenterOutController::halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept
{
    if (closed_ || halted_.exchange(true, std::memory_order_acq_rel))
        return;
    // A latch, and deliberately only a latch. The store above is what stops the
    // next frame from reaching the cursor, the guidance blend, and the machine,
    // and it takes effect the moment it is visible -- which is the whole point
    // of an emergency stop. The record it owes is written by finish_halt(),
    // where writing into the trace queue is not a second producer.
    //
    // Nothing here touches an actuator. Whether a device is inhibited is the
    // runtime's and its safety controller's, decided from the runtime's own
    // fault state; this refuses to produce commands, which is a different thing
    // and the only one this layer owns.
    halt_time_ns_ = time_ns;
    halt_condition_ = condition;
    halt_pending_ = true;
}

void CenterOutController::finish_halt() noexcept
{
    if (!halt_pending_)
        return;
    halt_pending_ = false;
    if (!running_)
        return;
    // The trial in flight is ended here rather than left to look unfinished:
    // a run that was stopped from outside had a trial, and that trial did not
    // simply stop being recorded.
    static_cast<void>(
        report_abnormal(halt_condition_, halt_time_ns_, 0, streaming::StreamStatus::stopped));
    running_ = false;
    pending_restart_ = false;
}

streaming::StreamStatus
CenterOutController::note_presentation_failure(const PresentationFailureEvidence& evidence) noexcept
{
    if (!prepared_ || closed_ || !running_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt();
    const auto snapshot = machine_.snapshot();
    if (machine_.complete() || !same_trial(snapshot.trial, evidence.trial))
        return streaming::StreamStatus::invalid_frame;
    return report_abnormal(AbnormalCondition::presentation_failed, evidence.time_ns,
                           evidence.implementation_status,
                           streaming::StreamStatus::consumer_failure);
}

streaming::StreamStatus
CenterOutController::apply_presentation_failures(std::size_t budget) noexcept
{
    return drain_presentation_failures(presentation_failures_, budget,
                                       [this](const PresentationFailureEvidence& evidence) noexcept
                                       { return note_presentation_failure(evidence); });
}

streaming::StreamStatus CenterOutController::flush() noexcept
{
    return prepared_ && !closed_ ? streaming::StreamStatus::ok
                                 : streaming::StreamStatus::invalid_state;
}

streaming::StreamStatus CenterOutController::reset() noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    machine_.reset();
    guidance_.reset();
    traces_.reset();
    presentation_failures_.reset();
    if (config_.training_capture_capacity != 0)
        training_labels_.reset();
    if (config_.presentation_state_capacity != 0)
        presentation_states_.reset();
    position_ = config_.initial_position;
    host_epoch_ns_ = 0;
    experiment_epoch_ns_ = 0;
    last_time_ns_ = 0;
    has_last_time_ = false;
    running_ = false;
    pending_restart_ = false;
    trial_started_ns_ = 0;
    has_trial_ = false;
    next_observation_ordinal_ = 0;
    intent_context_revision_.fetch_add(1, std::memory_order_acq_rel);
    std::atomic_thread_fence(std::memory_order_release);
    intent_target_id_.store(kUnsetTargetId, std::memory_order_relaxed);
    intent_trial_key_.store(kUnsetTrialKey, std::memory_order_relaxed);
    intent_state_->reset();
    intent_context_revision_.fetch_add(1, std::memory_order_release);
    last_observation_ordinal_ = 0;
    has_observation_ordinal_ = false;
    completed_trials_.store(0, std::memory_order_release);
    complete_.store(false, std::memory_order_release);
    reset_assistance_schedule();
    halt_refusal_recorded_ = false;
    halt_pending_ = false;
    pending_has_abnormal_ = false;
    pending_has_aborted_trial_ = false;
    cancelled_.store(false, std::memory_order_release);
    halted_.store(false, std::memory_order_release);
    return streaming::StreamStatus::ok;
}

void CenterOutController::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
}

} // namespace neurale::execution
