// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <neurale/experiments/events.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/schedule.h>

namespace neurale::experiments::ssvep
{
inline constexpr std::size_t kMaxSSVEPTargets = 64;
inline constexpr DrawStream kTargetSelectionStream = 1;

/// Semantic frequency identity, not a monitor frame sequence or display configuration.
struct SSVEPTarget
{
    TargetId id{};
    double frequency_hz{};
};

/// Explicit positive durations. No renderer, decoder, or acquisition configuration.
struct SSVEPConfig
{
    std::array<SSVEPTarget, kMaxSSVEPTargets> targets{};
    std::uint8_t n_targets{};
    StimulusId stimulus_id{};
    TrialOrdinal n_trials{};
    ScheduleSeed seed{};
    SamplerVersion sampler_version{kCurrentSamplerVersion};
    DurationNs cue_duration_ns{};
    DurationNs stimulation_duration_ns{};
    DurationNs decision_timeout_ns{};
    DurationNs feedback_duration_ns{};
    DurationNs inter_trial_ns{};
};

struct SSVEPTrialSchedule
{
    TrialOrdinal ordinal{};
    TargetId target_id{};
};

[[nodiscard]] ContractStatus validate(const SSVEPTarget& target) noexcept;
[[nodiscard]] ContractStatus validate(const SSVEPConfig& config) noexcept;
[[nodiscard]] ContractStatus validate(const SSVEPTrialSchedule& schedule) noexcept;
[[nodiscard]] bool contains_target(const SSVEPConfig& config, TargetId target) noexcept;
/// Each complete cycle contains every target once; adjacent cycles may repeat a target.
/// Output is unchanged on failure. Unknown sampler versions are never substituted.
[[nodiscard]] ContractStatus prepare_trial(const SSVEPConfig& config, TrialOrdinal ordinal,
                                           SSVEPTrialSchedule& schedule) noexcept;

// State machine.
inline constexpr std::size_t kMaxStepTransitions = 8;
inline constexpr std::size_t kMaxStepEvents = 32;
inline constexpr std::size_t kMaxStepRequests = 8;
inline constexpr std::size_t kMaxSSVEPPhases = 5;

enum class SSVEPState : std::uint8_t
{
    idle = 0,
    cue,
    stimulation,
    await_decision,
    feedback,
    inter_trial,
    complete,
};
enum class SSVEPPhase : std::uint8_t
{
    none = 0,
    cue,
    stimulation,
    await_decision,
    feedback,
    inter_trial,
};
enum class SSVEPCause : std::uint32_t
{
    unspecified = 0,
    session_started,
    cue_elapsed,
    stimulation_elapsed,
    selection_received,
    decision_timeout,
    feedback_elapsed,
    rest_elapsed,
    trial_limit_reached,
    stopped,
};
enum class SSVEPReason : std::uint32_t
{
    unspecified = 0,
    correct_selection,
    incorrect_selection,
    decision_timeout,
    stopped,
};
enum class SSVEPMarker : std::uint32_t
{
    unspecified = 0,
    cue_onset,
    cue_offset,
    stimulation_onset,
    stimulation_offset,
    wait_onset,
    wait_offset,
    feedback_onset,
    feedback_offset,
    rest_onset,
    rest_offset,
};
enum class SSVEPSelectionDisposition : std::uint8_t
{
    none = 0,
    accepted,
    expired,
};

/// Only executed phases occur here. Trial phases exclude the separately emitted rest.
struct SSVEPPhaseInterval
{
    SSVEPPhase phase{SSVEPPhase::none};
    TimeInterval interval{};
};
struct SSVEPPresentationRequest
{
    PresentationRequest request{};
    TargetId selected_id{};
    TrialOutcome outcome{TrialOutcome::pending};
};
struct SSVEPTrial
{
    TrialRecord record{};
    SSVEPTrialSchedule schedule{};
    std::array<SSVEPPhaseInterval, kMaxSSVEPPhases> phases{};
    std::uint8_t n_phases{};
    bool has_selection{};
    SelectionEvent selection{};
    bool has_decision{};
    ExperimentTimeNs decision_ns{};
};
struct SSVEPSnapshot
{
    ExperimentTimeNs time_ns{};
    SSVEPState state{SSVEPState::idle};
    TrialIdentity trial{};
    TimeInterval active{};
    TimeInterval stimulation{};
    ExperimentTimeNs decision_deadline_ns{};
    bool has_selection{};
    SelectionEvent selection{};
    TrialOutcome outcome{TrialOutcome::pending};
    TrialOrdinal completed{};
};
struct SSVEPStepResult
{
    SSVEPSnapshot snapshot{};
    std::array<StateTransition, kMaxStepTransitions> transitions{};
    std::array<ExperimentEvent, kMaxStepEvents> events{};
    std::array<SSVEPPresentationRequest, kMaxStepRequests> requests{};
    std::uint8_t n_transitions{}, n_events{}, n_requests{};
    SSVEPSelectionDisposition selection_disposition{SSVEPSelectionDisposition::none};
    bool trial_decided{};
    SSVEPTrial trial{};
    bool settled{};
};

[[nodiscard]] ContractStatus validate(const SSVEPTrial& trial) noexcept;
[[nodiscard]] ContractStatus validate(const SSVEPPresentationRequest& request) noexcept;

/// Pure explicit-time machine. No allocation, clocks, I/O, rendering, or decoding.
/// Failed operations leave both the machine and the output untouched.
/// step completes at most one trial; drain unsettled results at the same time.
class SSVEPMachine
{
  public:
    [[nodiscard]] ContractStatus start(ParadigmId paradigm, const SSVEPConfig& config,
                                       ExperimentTimeNs time_ns, SSVEPStepResult& result) noexcept;
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, SSVEPStepResult& result) noexcept;
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, const SelectionEvent& selection,
                                      SSVEPStepResult& result) noexcept;
    [[nodiscard]] ContractStatus stop(ExperimentTimeNs time_ns, SSVEPStepResult& result) noexcept;
    void reset() noexcept;
    /// Extend a black inter-trial interval for off-thread model training.
    /// Called only by the owning controller before its next step.
    [[nodiscard]] ContractStatus hold_inter_trial_until(ExperimentTimeNs time_ns) noexcept;
    [[nodiscard]] SSVEPSnapshot snapshot() const noexcept;
    [[nodiscard]] const SSVEPConfig& configuration() const noexcept
    {
        return config_;
    }
    [[nodiscard]] ParadigmId paradigm() const noexcept
    {
        return paradigm_;
    }
    [[nodiscard]] SSVEPState state() const noexcept
    {
        return run_.state;
    }
    [[nodiscard]] bool complete() const noexcept
    {
        return run_.state == SSVEPState::complete;
    }

  private:
    struct Run
    {
        MonotonicTimeGate gate{};
        TrialCounter trials{};
        SequenceCounter sequence{};
        SSVEPState state{SSVEPState::idle};
        TimeInterval active{}, stimulation{};
        ExperimentTimeNs deadline_ns{};
        SSVEPTrial trial{};
        TrialOrdinal completed{};
        bool has_last_selection{};
        SequenceOrdinal last_selection_sequence{};
    };
    [[nodiscard]] ContractStatus step_impl(ExperimentTimeNs time_ns,
                                           const SelectionEvent* selection,
                                           SSVEPStepResult& result) noexcept;
    [[nodiscard]] ContractStatus advance(Run& run, ExperimentTimeNs time_ns,
                                         SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus begin_trial(Run& run, ExperimentTimeNs time_ns,
                                             SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus enter(Run& run, SSVEPState state, ExperimentTimeNs time_ns,
                                       DurationNs duration, SSVEPCause cause,
                                       SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus leave(Run& run, ExperimentTimeNs time_ns,
                                       SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus feedback(Run& run, ExperimentTimeNs time_ns,
                                          SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus finish_trial(Run& run, ExperimentTimeNs time_ns,
                                              SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus terminate(Run& run, ExperimentTimeNs time_ns, SSVEPCause cause,
                                           SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus
    event(Run& run, ExperimentTimeNs time_ns, ExperimentEventKind kind, SSVEPStepResult& result,
          SSVEPMarker marker = SSVEPMarker::unspecified) const noexcept;
    [[nodiscard]] ContractStatus transition(Run& run, SSVEPState to, ExperimentTimeNs time_ns,
                                            SSVEPCause cause,
                                            SSVEPStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus request(Run& run, SSVEPStepResult& result) const noexcept;
    [[nodiscard]] SSVEPSnapshot build_snapshot(const Run& run) const noexcept;
    ParadigmId paradigm_{};
    SSVEPConfig config_{};
    Run run_{};
};
} // namespace neurale::experiments::ssvep
