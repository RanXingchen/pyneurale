/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/assistance.h>
#include <neurale/experiments/center_out.h>
#include <neurale/experiments/center_out_guidance.h>
#include <neurale/experiments/command.h>
#include <neurale/experiments/replay.h>

/**
 * @file
 * @brief Deterministic semantic replay of a recorded Center-Out 2D run.
 *
 * The run's semantic chain is four steps deep, and a replay re-executes all
 * four in the same order the live run did:
 *
 * ```text
 * decoded velocity -> guidance reference -> assisted velocity -> cursor -> machine
 * ```
 *
 * What it does not re-execute is the stream the decoded velocities arrived on.
 * A gap, a stale interval, and an operator's stop are conditions of a stream
 * that no longer exists; they enter the replay as recorded inputs, and what is
 * regenerated is the *response* to each -- from the same policy set, through the
 * same abnormal_response_for() the run used -- and everything that followed it.
 * A recording replayed against a different abnormal policy therefore disagrees
 * on the response, which is the point: the policy is part of the experiment.
 *
 * Nothing here submits an actuator command, sleeps, waits, or reads a clock. A
 * replay of an hour-long session takes as long as the arithmetic takes.
 */
namespace neurale::experiments::center_out
{

/// Version of the Center-Out record layout a replay understands.
///
/// The single source of truth for it: the recording bridge takes its
/// `experiment_version` from here, so the number a session records and the
/// number a replay demands cannot drift apart.
inline constexpr std::uint32_t kCenterOutRecordVersion = 1;

/// Everything a Center-Out replay re-executes under.
///
/// The same values the live run was configured with, minus the ones that were
/// about the stream rather than the task: no signal identifier, no trace
/// capacity, no observation-interval bound. The bound decided *whether* an
/// interval was stale, and that decision is recorded evidence here rather than
/// something a replay re-derives -- the observations it would re-derive it from
/// are the ones the run already refused.
struct CenterOutReplayConfig
{
    /// Provenance identifier the run was recorded under.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Task configuration, including the layout, the seed, and the sampler.
    CenterOut2DConfig task{};
    /// Guidance configuration. An unconfigured guidance is legal and means the
    /// run blended against no reference velocity.
    CenterOutGuidanceConfig guidance{};
    /// Command space the decoded and assisted velocities live in.
    CommandSpace velocity_space{};
    /// Linear assistance parameters, meaningful for
    /// ::assistance::AssistanceMethod::linear_blend.
    assistance::LinearAssistance linear_assistance{};
    /// Transform that produced the applied velocity.
    assistance::AssistanceMethod assistance_method{assistance::AssistanceMethod::none};
    /// Where the cursor started.
    WorkspacePoint initial_position{};
    /// Inclusive task-derived bounds for the cursor centre.
    WorkspacePoint cursor_min{};
    WorkspacePoint cursor_max{};
    /// Severities the run treated each class of abnormal condition under.
    AbnormalPolicySet abnormal{};
};

/// How Center-Out treats one abnormal condition, before the configuration.
///
/// Two facts that are the paradigm's and not the caller's: the severity it will
/// not go below whatever the configuration says, and whether the input carrying
/// the condition reached the task at all. Both are properties of the condition
/// -- a gap always breaks the continuity a hold is accumulated over, a
/// malformed frame is always refused -- so they belong to the condition rather
/// than to the call site that met it.
///
/// One place, because a replay derives the same two facts when it re-executes
/// a recorded condition, and two copies of them would be two paradigms.
struct CenterOutConditionHandling
{
    /// Severity this paradigm will not go below.
    AbnormalPolicy floor{AbnormalPolicy::record};
    /// Whether the input carrying the condition was refused before the task
    /// saw it.
    bool input_refused{};
};

/// How Center-Out treats @p condition.
[[nodiscard]] CenterOutConditionHandling
center_out_condition_handling(AbnormalCondition condition) noexcept;

/// What one recorded semantic input to a Center-Out run was.
enum class CenterOutReplayInputKind : std::uint8_t
{
    /// One decoded observation the run accepted and integrated.
    observation = 0,
    /// An abnormal condition the run met and decided about.
    ///
    /// One kind for all of them, because the condition already says everything
    /// that distinguishes them: center_out_condition_handling() gives the floor
    /// and the refusal, and the response the replay derives gives the effect. A
    /// second enumeration beside the condition would be a second place to keep
    /// the same distinctions in step.
    ///
    /// What is never replayed is the *stream* that produced the condition. A
    /// gap, a stale interval, and an operator's stop were decided against
    /// conditions that no longer exist; the decision is the recorded input, and
    /// what is regenerated is the response and everything that followed it.
    decision,
};

/// Whether @p kind is one of the declared CenterOutReplayInputKind values.
[[nodiscard]] constexpr bool
center_out_replay_input_kind_declared(CenterOutReplayInputKind kind) noexcept
{
    return static_cast<std::uint8_t>(kind) <=
           static_cast<std::uint8_t>(CenterOutReplayInputKind::decision);
}

/// One recorded semantic input.
struct CenterOutReplayInput
{
    /// Which kind of input this is.
    CenterOutReplayInputKind kind{CenterOutReplayInputKind::observation};
    /// Experiment instant the run stamped it with.
    ExperimentTimeNs time_ns{};
    /// Decoded velocity components, for ::CenterOutReplayInputKind::observation.
    std::array<double, 2> decoded{};
    /// Frame the observation came from.
    std::uint64_t frame_sequence{};
    /// Sample index within the decoded stream.
    std::uint64_t sample_idx{};
    /// Condition the run met, for ::CenterOutReplayInputKind::decision.
    AbnormalCondition condition{AbnormalCondition::unspecified};
    /// Response the run recorded, for ::CenterOutReplayInputKind::decision.
    ///
    /// Compared against the response the replay derives, rather than obeyed.
    /// Obeying it would make a replay agree with any recording that was
    /// internally consistent, including one produced under a policy set nobody
    /// configured.
    AbnormalResponse response{AbnormalResponse::recorded};
};

/// One recorded cursor observation and the interval it moved over.
struct CenterOutCursorSample
{
    /// Experiment instant of the observation.
    ExperimentTimeNs time_ns{};
    /// Where the cursor was before the applied velocity moved it.
    WorkspacePoint before{};
    /// Where it was after.
    WorkspacePoint after{};
    /// Interval the applied velocity was integrated across.
    DurationNs dt_ns{};
    /// Frame the observation came from.
    std::uint64_t frame_sequence{};
    /// Sample index within the decoded stream.
    std::uint64_t sample_idx{};
    /// State the machine was in afterwards.
    CenterOutState state{CenterOutState::idle};
    /// Trial the observation belonged to.
    TrialIdentity trial{};
};

/// One recorded assisted velocity and the two velocities it came from.
struct CenterOutVelocitySample
{
    /// Experiment instant of the observation.
    ExperimentTimeNs time_ns{};
    /// What the decoder produced.
    assistance::VelocityVector decoded{};
    /// What the guidance offered, or a zero vector when no target was up.
    assistance::VelocityVector guidance{};
    /// What was applied to the cursor.
    assistance::VelocityVector assisted{};
    /// Trial the observation belonged to.
    TrialIdentity trial{};
};

/// One recorded guidance sample.
struct CenterOutGuidanceObservation
{
    /// Experiment instant of the observation.
    ExperimentTimeNs time_ns{};
    /// The sample the guidance produced.
    CenterOutGuidanceSample sample{};
};

/// One recorded target onset.
struct CenterOutTargetOnset
{
    /// Experiment instant the target came up.
    ExperimentTimeNs time_ns{};
    /// Target the cursor had to acquire.
    TargetId target_id{kUnsetTargetId};
    /// Where it was.
    WorkspacePoint pos{};
    /// Leg the target belonged to.
    CenterOutPhase phase{CenterOutPhase::to_center};
    /// Outward target of the trial the onset belonged to.
    TargetId outward_target{kUnsetTargetId};
    /// Its index in the layout.
    std::uint8_t outward_idx{};
    /// Trial the onset belonged to.
    TrialIdentity trial{};
};

/// One recorded trial an abnormal condition ended.
///
/// Separate from the machine's own decided trials because it is a different
/// fact recorded in a different shape: the machine decided nothing about it,
/// and the fields a decided Center-Out trial carries -- the deciding leg, the
/// acquisition durations -- describe a decision that was never taken.
struct CenterOutReplayAbort
{
    /// The trial as the shared contract records it.
    TrialRecord record{};
    /// The condition that ended it.
    AbnormalCondition condition{AbnormalCondition::unspecified};
    /// The response that condition was decided under.
    AbnormalResponse response{AbnormalResponse::recorded};
};

/// The recorded semantic outputs a replay is checked against.
///
/// Each stream has a flag saying whether the recording carried it at all. An
/// absent stream is not an empty one: a run that produced no guidance sample
/// and a recording that did not keep guidance samples are different facts, and
/// treating the second as the first is exactly how a replay reports a match it
/// did not establish.
struct CenterOutReplayExpectation
{
    /// State transitions, in emission order.
    std::span<const StateTransition> transitions{};
    /// Semantic events, in emission order.
    std::span<const ExperimentEvent> events{};
    /// Trials the machine decided, in decision order.
    std::span<const CenterOutTrial> trials{};
    /// Trials an abnormal condition ended, in order.
    std::span<const CenterOutReplayAbort> aborts{};
    /// Target onsets, in order.
    std::span<const CenterOutTargetOnset> targets{};
    /// Cursor observations, in order.
    std::span<const CenterOutCursorSample> cursor{};
    /// Assisted velocities, in order.
    std::span<const CenterOutVelocitySample> vel{};
    /// Guidance samples, in order.
    std::span<const CenterOutGuidanceObservation> guidance{};

    /// Whether the recording carried state transitions.
    bool has_transitions{};
    /// Whether it carried semantic events.
    bool has_events{};
    /// Whether it carried decided trials.
    bool has_trials{};
    /// Whether it carried the trials conditions ended.
    bool has_aborts{};
    /// Whether it carried target onsets.
    bool has_targets{};
    /// Whether it carried cursor observations.
    bool has_cursor{};
    /// Whether it carried assisted velocities.
    bool has_velocity{};
    /// Whether it carried guidance samples.
    bool has_guidance{};
};

/// One recorded Center-Out run, as a reader recovers it.
struct CenterOutReplayRecording
{
    /// What the recording says the run was.
    ReplayProvenance provenance{};
    /// Experiment instant the run started.
    ExperimentTimeNs origin_ns{};
    /// Recorded semantic inputs, in the order the run consumed them.
    std::span<const CenterOutReplayInput> inputs{};
    /// Recorded semantic outputs.
    CenterOutReplayExpectation expected{};
    /// Whether the recording holds the run's own end.
    ///
    /// A recording that stops without one may be a truncated file rather than a
    /// short run, and the two are not distinguishable from the records.
    bool run_end_recorded{};
    /// Whether the recording reports that trace records were lost.
    bool trace_loss_recorded{};
};

/// Re-executes a recorded Center-Out run and checks it against what was recorded.
///
/// Streaming: it regenerates one input at a time and compares immediately, so
/// it holds one machine, one guidance, and a cursor into each recorded stream,
/// and never a copy of a run. Comparison stops at the first difference, which
/// is what makes ::ReplayReport::first_mismatch the earliest one rather than an
/// arbitrary one.
class CenterOutReplay
{
  public:
    CenterOutReplay() noexcept = default;

    /// Fix the configuration this replay regenerates under.
    ///
    /// @return ContractStatus::ok, ContractStatus::identity_missing for an
    ///         unset paradigm, or the first failure of the task, guidance,
    ///         command-space, assistance, or abnormal-policy validators.
    [[nodiscard]] ContractStatus prepare(const CenterOutReplayConfig& config) noexcept;

    /// Return to the state a prepared replay starts each run from.
    ///
    /// Called by run() before it replays anything, so two runs of one prepared
    /// replay produce the same report. Exposed because a caller stepping a
    /// replay for its own reasons needs the same guarantee.
    void reset() noexcept;

    /// Replay @p recording and report what it established.
    ///
    /// @param recording Recorded inputs, outputs, and provenance.
    /// @param policy    What to do when the evidence does not cover the run.
    [[nodiscard]] ReplayReport
    run(const CenterOutReplayRecording& recording,
        ReplayIncompletePolicy policy = ReplayIncompletePolicy::verify_available) noexcept;

    /// The provenance this configuration regenerates under.
    ///
    /// What check_provenance() compares the recording against. Available before
    /// a replay so a caller can decide whether to attempt one at all.
    [[nodiscard]] ReplayProvenance provenance() const noexcept;

    /// Whether a configuration has been accepted.
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }

    /// The configuration prepare() accepted.
    [[nodiscard]] const CenterOutReplayConfig& configuration() const noexcept
    {
        return config_;
    }

  private:
    struct Cursors;

    [[nodiscard]] ContractStatus begin_segment(ExperimentTimeNs time_ns,
                                               CenterOutStepResult& result) noexcept;
    void note_trial_boundaries(const CenterOutStepResult& result) noexcept;
    [[nodiscard]] ContractStatus replay_observation(const CenterOutReplayInput& input,
                                                    const CenterOutReplayRecording& recording,
                                                    Cursors& cursors,
                                                    ReplayComparator& comparator) noexcept;
    void replay_decision(const CenterOutReplayInput& input,
                         const CenterOutReplayRecording& recording, Cursors& cursors,
                         ReplayComparator& comparator) noexcept;
    void compare_step(const CenterOutStepResult& step, const CenterOutReplayRecording& recording,
                      Cursors& cursors, ReplayComparator& comparator) noexcept;
    void compare_target(const CenterOutSnapshot& snapshot, ExperimentTimeNs time_ns,
                        const CenterOutReplayRecording& recording, Cursors& cursors,
                        ReplayComparator& comparator) noexcept;
    void forget_target() noexcept;

    CenterOutReplayConfig config_{};
    CenterOutMachine machine_{};
    CenterOutGuidance guidance_{};
    WorkspacePoint position_{};
    ExperimentTimeNs last_time_ns_{};
    ExperimentTimeNs trial_started_ns_{};
    TargetId last_target_{kUnsetTargetId};
    bool has_last_time_{};
    bool has_trial_{};
    bool has_target_{};
    bool pending_restart_{};
    bool running_{};
    bool terminal_{};
    bool prepared_{};
};

} // namespace neurale::experiments::center_out
