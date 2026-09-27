/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/assistance.h>
#include <neurale/experiments/center_out.h>
#include <neurale/experiments/center_out_guidance.h>
#include <neurale/experiments/center_out_replay.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/neural_intent.h>

#include "abnormal_reporter.h"
#include "bounded_trace_queue.h"
#include "presentation_evidence.h"

namespace neurale::execution
{

using namespace neurale::experiments;

enum class CenterOutTraceKind : std::uint8_t
{
    session_start,
    observation,
    discontinuity,
    /// A condition the task rules do not describe. Carries the decision that
    /// was taken about it, and the trial it ended when it ended one.
    abnormal,
};

struct CenterOutControlTrace
{
    CenterOutTraceKind kind{CenterOutTraceKind::observation};
    /// Stable producer-assigned identity for an observation trace.
    SequenceOrdinal observation_ordinal{};
    std::uint64_t frame_sequence{};
    streaming::SampleIndex sample_idx{};
    /// Decoder model version that produced this observation.
    std::uint64_t decoder_version{};
    ExperimentTimeNs time_ns{};
    streaming::HostTimeNs input_ready_ns{};
    streaming::HostTimeNs decoded_ready_ns{};
    DurationNs dt_ns{};
    center_out::WorkspacePoint position_before{};
    center_out::WorkspacePoint position_after{};
    assistance::VelocityVector decoded{};
    assistance::VelocityVector guidance{};
    assistance::VelocityVector assisted{};
    /// Assistance fraction actually applied to this observation.
    assistance::LinearAssistance linear_assistance{};
    center_out::CenterOutGuidanceSample guidance_sample{};
    center_out::CenterOutStepResult segment_start{};
    center_out::CenterOutStepResult step{};
    streaming::GapReason gap_reason{streaming::GapReason::frame_sequence_gap};
    bool restarted_after_discontinuity{};
    /// Meaningful when `has_abnormal`, which is always true for
    /// ::CenterOutTraceKind::abnormal and ::CenterOutTraceKind::discontinuity
    /// and may be true for an observation that carries its own decision.
    AbnormalEvent abnormal{};
    /// Whether `abnormal` describes anything.
    ///
    /// An observation record can carry one. A stale interval is decided *about*
    /// the observation that revealed it, and the observation that follows is
    /// the first of the new segment -- so the decision and the observation
    /// travel in one queue entry, exactly as a discontinuity and its decision
    /// do. Two entries for one row would also mean one row could cost two
    /// slots, which is how a bounded queue turns "no room" into "half applied".
    bool has_abnormal{};
    /// The trial the condition ended, when `abnormal.response` says one was
    /// ended. Built from the machine's own trial start, never from an invented
    /// instant.
    TrialRecord aborted_trial{};
    bool has_aborted_trial{};
};

struct CenterOutIntentContext
{
    streaming::NeuralIntentSnapshot intent{};
    TargetId target_id{kUnsetTargetId};
    TrialKey trial_key{kUnsetTrialKey};
};

struct CenterOutAssistanceBlock
{
    assistance::LinearAssistance linear{};
    std::uint64_t trials{};
};

struct CenterOutTrainingLabel
{
    streaming::SampleIndex sample_idx{};
    ExperimentTimeNs time_ns{};
    TrialIdentity trial{};
    center_out::WorkspacePoint target_position{};
    center_out::WorkspacePoint cursor_position{};
    center_out::WorkspacePoint cursor_velocity{};
    assistance::VelocityVector guidance{};
    bool trial_stop{};
};

/// Immutable task state handed from the realtime task owner to presentation.
/// The source ordinal is the exact observation identity used by the control
/// trace, so a rendered state can be joined to the row that produced it.
struct CenterOutPresentationState
{
    center_out::CenterOutSnapshot snapshot{};
    center_out::WorkspacePoint cursor{};
    SequenceOrdinal source_ordinal{};
    ExperimentTimeNs time_ns{};
};

struct CenterOutControllerConfig
{
    streaming::SignalId decoded_signal_id{};
    ParadigmId paradigm{kUnsetParadigmId};
    center_out::CenterOut2DConfig task{};
    center_out::CenterOutGuidanceConfig guidance{};
    /// Zero for an explicit public config; otherwise the deterministic Python
    /// resolver version that produced ``guidance``.
    std::uint32_t guidance_resolver_version{};
    CommandSpace velocity_space{};
    assistance::LinearAssistance linear_assistance{};
    /// Optional trial-aligned schedule. Empty preserves ``linear_assistance``
    /// for every trial; otherwise the blocks must cover ``task.trial_limit``
    /// exactly and the first block takes precedence over ``linear_assistance``.
    std::vector<CenterOutAssistanceBlock> assistance_blocks;
    /// Optional critical online-training label handoff. Zero disables it.
    std::size_t training_capture_capacity{};
    /// Optional presentation-thread state handoff. Zero disables it.
    std::size_t presentation_state_capacity{};
    center_out::WorkspacePoint initial_position{};
    /// Inclusive bounds for the cursor centre. They are task-derived and
    /// already account for the rendered cursor radius.
    center_out::WorkspacePoint cursor_min{};
    center_out::WorkspacePoint cursor_max{};
    std::size_t trace_capacity{};
    assistance::AssistanceMethod assistance_method{assistance::AssistanceMethod::none};
    /// How severely this run treats each class of abnormal condition.
    AbnormalPolicySet abnormal{};
    /// Longest interval between accepted observations that a decoded command
    /// will be applied across. Zero means the caller declares no bound.
    ///
    /// The cursor is advanced by `velocity * dt`, so a gap nobody declared
    /// turns one decoded sample into a movement over an interval it was never
    /// observed over. That is the shape "a stale command kept a trial alive"
    /// actually takes here: not a command reissued, but one command asked to
    /// stand for a stretch of time the decoder said nothing about. Past this
    /// bound the observation is refused instead, and the segment restarts.
    DurationNs max_observation_interval_ns{};
};

/// Private terminal consumer for the Center-Out native path. prepare() fixes
/// one two-channel float64 decoded schema and allocates all validation/trace
/// storage. consume() performs no allocation and processes rows in frame order.
class CenterOutController final : public streaming::NativeFrameConsumer
{
  public:
    CenterOutController() noexcept = default;
    ~CenterOutController() override = default;

    CenterOutController(const CenterOutController&) = delete;
    CenterOutController& operator=(const CenterOutController&) = delete;

    [[nodiscard]] streaming::StreamStatus prepare(const streaming::StreamSchema& schema,
                                                  const CenterOutControllerConfig& config);
    [[nodiscard]] streaming::StreamStatus start(streaming::HostTimeNs host_epoch_ns,
                                                ExperimentTimeNs experiment_epoch_ns) noexcept;
    void close() noexcept;

    [[nodiscard]] streaming::StreamStatus try_pop_trace(CenterOutControlTrace& trace) noexcept
    {
        return traces_.try_pop(trace);
    }

    /// Trace records this controller could not enqueue, ever. Monotonic.
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept
    {
        return traces_.dropped() + presentation_failures_.dropped();
    }

    [[nodiscard]] streaming::StreamStatus
    try_pop_training_label(CenterOutTrainingLabel& label) noexcept
    {
        return training_labels_.try_pop(label);
    }

    [[nodiscard]] std::uint64_t dropped_training_label_count() const noexcept
    {
        return training_labels_.dropped();
    }

    [[nodiscard]] streaming::StreamStatus
    try_pop_presentation_state(CenterOutPresentationState& state) noexcept
    {
        return presentation_states_.try_pop(state);
    }

    [[nodiscard]] std::uint64_t dropped_presentation_state_count() const noexcept
    {
        return presentation_states_.dropped();
    }

    /// Abnormal conditions this controller has met, ever. Monotonic.
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept
    {
        return abnormal_.summary();
    }

    /// Refuse all further task input from this instant, and end the trial in
    /// flight.
    ///
    /// This is what an emergency stop is on the task side, and it is separate
    /// from cancel() on purpose. cancel() is the runtime's own word for "this
    /// edge is being taken down"; it says nothing about the experiment, records
    /// nothing, and is called on a perfectly ordinary shutdown. This says the
    /// run was stopped by something outside the task, records why, and leaves
    /// the trial in flight recorded as aborted rather than merely unfinished.
    ///
    /// Bounded, allocation-free, and idempotent: the first call latches, and
    /// later ones change nothing. It only latches -- the record describing the
    /// halt is produced by ::finish_halt, for the reason stated there.
    void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept;

    /// Produce the record a latched halt owes, if it has not been produced.
    ///
    /// Separate from halt() because of who is allowed to write into the trace
    /// queue. The queue has one producer, and while the run is live that
    /// producer is whichever thread is driving the paradigm -- for Center-Out,
    /// the runtime's own. halt() can be called from another thread entirely, so
    /// it latches and nothing more; this is called once the producer is
    /// provably quiet, which for an attached runtime is after its workers have
    /// been joined. ::ExperimentSession orders it that way, in shutdown step 3.
    void finish_halt() noexcept;

    /// Bounded presentation-thread handoff. The task state is untouched here.
    [[nodiscard]] bool
    enqueue_presentation_failure(const PresentationFailureEvidence& evidence) noexcept
    {
        return presentation_failures_.try_push(evidence) == streaming::StreamStatus::ok;
    }

    /// Apply queued failures on the task-owner thread.
    ///
    /// @param budget Maximum number of queued failures applied by this call.
    ///        ::kUnboundedPresentationFailureDrain means "all of them" and is
    ///        for shutdown, where no frame deadline is left to miss. The frame
    ///        consumer passes ::kPresentationFailureDrainBudget instead: that
    ///        path runs on the acquisition thread, and draining a full queue
    ///        there would put up to `presentation_failures_` capacity abnormal
    ///        decisions plus their trace pushes inside one frame callback. The
    ///        work was already bounded by the queue's capacity, but "bounded by
    ///        a 64-slot queue" is not a frame budget; the remainder is applied
    ///        by the next frame, and nothing is dropped by deferring it.
    [[nodiscard]] streaming::StreamStatus
    apply_presentation_failures(std::size_t budget = kUnboundedPresentationFailureDrain) noexcept;

    /// Identity of the most recently accepted observation. False means no
    /// observation has been accepted in this run yet.
    [[nodiscard]] bool latest_observation_ordinal(SequenceOrdinal& ordinal) const noexcept
    {
        if (!has_observation_ordinal_)
            return false;
        ordinal = last_observation_ordinal_;
        return true;
    }

    void set_decoder_version(std::uint64_t version) noexcept
    {
        decoder_version_ = version;
    }

    [[nodiscard]] std::uint64_t completed_trial_count() const noexcept
    {
        return completed_trials_.load(std::memory_order_acquire);
    }

    /// Whether the task emitted its semantic session-stop boundary.
    ///
    /// This is distinct from cancellation or a runtime failure. It is atomic
    /// because the runtime consumer publishes completion to the session thread.
    [[nodiscard]] bool complete() const noexcept
    {
        return complete_.load(std::memory_order_acquire);
    }

    /// Latch the terminal state, recording nothing.
    ///
    /// What an `abort_session` response *is*, as distinct from what it says. A
    /// condition decided under ::AbnormalPolicy::abort_session is already
    /// described by the abnormal event that carries it, and the run has to stop
    /// being a run at that instant -- not when whoever receives the fatal
    /// status this produces gets around to acting on it. Under an attached
    /// runtime that status does take the edge down; a caller-stepped run has
    /// nothing that would, and without this the paradigm would go on accepting
    /// input after having itself decided it cannot continue.
    ///
    /// Distinct from halt(), which is a stop decided *outside* the task and
    /// therefore owes a record of its own. This owes none, and deliberately
    /// writes none: the event that reached `session_aborted` is that record,
    /// and a second one would make one condition look like two.
    ///
    /// Idempotent, bounded, allocation-free, and callable from any thread.
    void stop_accepting() noexcept
    {
        halted_.store(true, std::memory_order_release);
    }

    /// Whether input is being refused because the run reached a terminal state.
    [[nodiscard]] bool halted() const noexcept
    {
        return halted_.load(std::memory_order_acquire);
    }

    /// The configuration prepare() froze.
    ///
    /// The recording bridge reads it from here rather than from a second copy
    /// handed to it by the caller: a bridge holding its own configuration can
    /// record one configuration while the controller runs another, and nothing
    /// in either object would notice.
    [[nodiscard]] const CenterOutControllerConfig& configuration() const noexcept
    {
        return config_;
    }

    [[nodiscard]] const center_out::WorkspacePoint& position() const noexcept
    {
        return position_;
    }
    [[nodiscard]] center_out::CenterOutSnapshot snapshot() const noexcept
    {
        return machine_.snapshot();
    }
    [[nodiscard]] streaming::NeuralIntentSnapshot intent_snapshot() const noexcept
    {
        return intent_state_->read_intent();
    }
    [[nodiscard]] CenterOutIntentContext intent_context() const noexcept
    {
        CenterOutIntentContext result{};
        for (;;)
        {
            const auto before = intent_context_revision_.load(std::memory_order_acquire);
            if ((before & 1U) != 0U)
                continue;
            result.intent = intent_state_->read_intent();
            result.target_id = intent_target_id_.load(std::memory_order_relaxed);
            result.trial_key = intent_trial_key_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (intent_context_revision_.load(std::memory_order_acquire) == before)
                return result;
        }
    }
    [[nodiscard]] std::shared_ptr<streaming::NeuralIntentSource> intent_source() const
    {
        return intent_state_;
    }
    [[nodiscard]] TargetId intent_target_id() const noexcept
    {
        return intent_target_id_.load(std::memory_order_acquire);
    }
    [[nodiscard]] TrialKey intent_trial_key() const noexcept
    {
        return intent_trial_key_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }

    [[nodiscard]] streaming::StreamStatus consume(streaming::FrameView frame) noexcept override;
    void bind_runtime_clock(streaming::NativeClock& clock) noexcept override
    {
        runtime_clock_.store(&clock, std::memory_order_release);
    }
    [[nodiscard]] streaming::StreamStatus
    handle_discontinuity(const streaming::Discontinuity& discontinuity) noexcept override;
    [[nodiscard]] streaming::StreamStatus flush() noexcept override;
    [[nodiscard]] streaming::StreamStatus reset() noexcept override;
    void cancel() noexcept override;

  private:
    [[nodiscard]] streaming::StreamStatus
    begin_segment(ExperimentTimeNs time_ns, center_out::CenterOutStepResult& result) noexcept;
    [[nodiscard]] streaming::StreamStatus
    process_observation(const double* values, std::uint64_t frame_sequence,
                        streaming::SampleIndex sample_idx, ExperimentTimeNs time_ns,
                        streaming::HostTimeNs input_ready_ns,
                        const center_out::CenterOutStepResult* segment_start) noexcept;
    /// Decide, record, and apply one abnormal condition.
    ///
    /// Returns `ok` when the run continues, and the status the caller should
    /// return otherwise -- `queue_overflow` when the record itself could not be
    /// queued, or the fault status when the policy ends the run.
    [[nodiscard]] streaming::StreamStatus report_abnormal(AbnormalCondition condition,
                                                          ExperimentTimeNs time_ns,
                                                          std::uint32_t detail,
                                                          streaming::StreamStatus fatal) noexcept;
    [[nodiscard]] streaming::StreamStatus
    note_presentation_failure(const PresentationFailureEvidence& evidence) noexcept;
    /// Decide one condition into @p trace, and apply what the decision changed.
    /// Split from report_abnormal() so that a discontinuity's own record
    /// carries the decision instead of a second record being written beside it.
    ///
    /// The floor and the refusal are not parameters: they are properties of the
    /// condition, and center_out_condition_handling() is the one place they are
    /// written down -- the same place a deterministic replay reads them from
    /// when it re-derives what this run did.
    void apply_abnormal(CenterOutControlTrace& trace, AbnormalCondition condition,
                        ExperimentTimeNs time_ns, std::uint32_t detail) noexcept;
    /// Refuse input that arrived after halt(), recording the refusal once.
    [[nodiscard]] streaming::StreamStatus refuse_after_halt() noexcept;
    /// Latch the machine's own statement of when the trial in flight began.
    void note_trial_boundaries(const center_out::CenterOutStepResult& result) noexcept;
    void reset_assistance_schedule() noexcept;

    AbnormalReporter abnormal_{};

    CenterOutControllerConfig config_{};
    streaming::SchemaId schema_id_{};
    streaming::SignalSchema signal_{};
    std::uint64_t observation_period_ns_{};
    std::unique_ptr<streaming::FrameValidator> validator_{};
    BoundedTraceQueue<CenterOutControlTrace> traces_{};
    BoundedTraceQueue<PresentationFailureEvidence> presentation_failures_{};
    BoundedTraceQueue<CenterOutTrainingLabel> training_labels_{};
    BoundedTraceQueue<CenterOutPresentationState> presentation_states_{};
    center_out::CenterOutMachine machine_{};
    center_out::CenterOutGuidance guidance_{};
    center_out::WorkspacePoint position_{};
    std::shared_ptr<streaming::NeuralIntentState> intent_state_{
        std::make_shared<streaming::NeuralIntentState>()};
    std::atomic<TargetId> intent_target_id_{kUnsetTargetId};
    std::atomic<TrialKey> intent_trial_key_{kUnsetTrialKey};
    std::atomic<std::uint64_t> intent_context_revision_{};
    std::atomic<streaming::NativeClock*> runtime_clock_{};
    streaming::HostTimeNs host_epoch_ns_{};
    ExperimentTimeNs experiment_epoch_ns_{};
    ExperimentTimeNs last_time_ns_{};
    bool has_last_time_{};
    bool prepared_{};
    bool running_{};
    bool pending_restart_{};
    bool closed_{};
    /// A decision taken about the row being processed, waiting for that row's
    /// observation record to carry it. Set and consumed within one iteration of
    /// the row loop; see ::CenterOutControlTrace::has_abnormal.
    AbnormalEvent pending_abnormal_{};
    TrialRecord pending_aborted_trial_{};
    bool pending_has_abnormal_{};
    bool pending_has_aborted_trial_{};
    /// Experiment instant the machine said the trial in flight began at, taken
    /// from its own ExperimentEventKind::trial_start rather than inferred.
    ExperimentTimeNs trial_started_ns_{};
    bool has_trial_{};
    SequenceOrdinal next_observation_ordinal_{};
    SequenceOrdinal last_observation_ordinal_{};
    bool has_observation_ordinal_{};
    assistance::LinearAssistance active_linear_assistance_{};
    std::size_t assistance_block_idx_{};
    std::uint64_t trials_in_assistance_block_{};
    std::uint64_t decoder_version_{};
    std::atomic<std::uint64_t> completed_trials_{};
    std::atomic<bool> complete_{};
    std::atomic<bool> cancelled_{};
    /// Latched by halt(). Atomic because an emergency stop reaches this object
    /// from whichever thread noticed, while the runtime's own thread is inside
    /// consume().
    std::atomic<bool> halted_{};
    /// Whether the refusal that follows a halt has already been recorded.
    /// Guarded by the same single-consumer discipline as the rest of the task
    /// state: only the thread driving the paradigm reads and writes it.
    bool halt_refusal_recorded_{};
    /// What a latched halt still owes a record for. Written by halt() and read
    /// by finish_halt(), both of which the session calls on its own thread.
    ExperimentTimeNs halt_time_ns_{};
    AbnormalCondition halt_condition_{AbnormalCondition::unspecified};
    bool halt_pending_{};
};

} // namespace neurale::execution
