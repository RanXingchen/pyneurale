/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <span>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/replay.h>
#include <neurale/experiments/webgrid.h>

/**
 * @file
 * @brief Deterministic semantic replay of a recorded WebGrid run.
 *
 * WebGrid's semantic input is a pointer timeline and the discrete selections
 * taken on it. Both are recorded, so a replay re-derives the target sequence,
 * the selected cells, the correct and incorrect verdicts, the acquisition
 * intervals, and both the raw and the versioned derived metrics, from nothing
 * but the recording.
 *
 * The pointer *reference* is part of the evidence rather than an assumption.
 * A run may record its pointer samples in the experiment trace or name the
 * stream they already live in; either way, a replay reads them from a recorded
 * timeline, and a recording that named a stream nobody kept is incomplete
 * rather than replayable.
 *
 * A gap in that timeline enters as a recorded input, exactly as it does for
 * Center-Out: the run decided a gap had happened by looking at a stream that no
 * longer exists. What the replay regenerates is the *response*, from the same
 * policy set, and everything the response changed -- which for WebGrid means
 * whether the acquisition interval that gap crossed is still a measurement.
 */
namespace neurale::experiments::webgrid
{

/// Version of the WebGrid record layout a replay understands.
///
/// The single source of truth for it; the recording bridge takes its
/// `experiment_version` from here.
inline constexpr std::uint32_t kWebGridRecordVersion = 1;

/// Everything a WebGrid replay re-executes under.
struct WebGridReplayConfig
{
    /// Provenance identifier the run was recorded under.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Task configuration, including the grid, the seed, and the sampler.
    WebGridConfig task{};
    /// Severities the run treated each class of abnormal condition under.
    AbnormalPolicySet abnormal{};
};

/// What one recorded semantic input to a WebGrid run was.
enum class WebGridReplayInputKind : std::uint8_t
{
    /// One pointer observation.
    pointer = 0,
    /// One pointer observation carrying a discrete selection.
    selection,
    /// An interval longer than the run would accept between pointer samples.
    pointer_gap,
    /// A monitoring frame the run never saw.
    ///
    /// Recorded and nothing more, whatever the policy says, so a replay
    /// regenerates the same `record` response and the same absence of any
    /// effect on the target in flight.
    observer_drop,
    /// The run was stopped from outside the task.
    halt,
};

/// Whether @p kind is one of the declared WebGridReplayInputKind values.
[[nodiscard]] constexpr bool
webgrid_replay_input_kind_declared(WebGridReplayInputKind kind) noexcept
{
    return static_cast<std::uint8_t>(kind) <=
           static_cast<std::uint8_t>(WebGridReplayInputKind::halt);
}

/// One recorded semantic input.
struct WebGridReplayInput
{
    /// Which kind of input this is.
    WebGridReplayInputKind kind{WebGridReplayInputKind::pointer};
    /// Experiment instant the run stamped it with.
    ExperimentTimeNs time_ns{};
    /// Where the pointer was.
    PointerPosition pointer{};
    /// The selection, for ::WebGridReplayInputKind::selection.
    SelectionEvent selection{};
    /// Condition the run met, for the three decision kinds.
    AbnormalCondition condition{AbnormalCondition::unspecified};
    /// Response the run recorded, compared against the one the replay derives.
    AbnormalResponse response{AbnormalResponse::recorded};
};

/// One recorded pointer observation and what the machine made of it.
struct WebGridPointerSample
{
    /// Experiment instant of the observation.
    ExperimentTimeNs time_ns{};
    /// Where the pointer was.
    PointerPosition pointer{};
    /// State the machine was in afterwards.
    WebGridState state{WebGridState::idle};
    /// Target that was up.
    TargetId active_target{kUnsetTargetId};
    /// Trial the observation belonged to.
    TrialIdentity trial{};
};

/// One recorded selection, and whether the interval it was taken over survived.
struct WebGridReplaySelection
{
    /// The selection as the machine recorded it.
    WebGridSelectionRecord record{};
    /// Whether the acquisition interval beside it is still a measurement.
    ///
    /// The raw selection is kept whatever happened; this is the flag that says
    /// a gap crossed the interval it was selected over.
    bool acquisition_timing_valid{};
};

/// One recorded decided target.
struct WebGridReplayTrial
{
    /// The trial as the machine decided it, with the machine's own verdict.
    WebGridTrial trial{};
    /// The outcome the recording's own column carries.
    ///
    /// Not always the machine's. A target selected correctly over an interval
    /// a gap crossed is written as TrialOutcome::aborted with the machine's
    /// verdict kept beside it, and a replay that only regenerated the machine's
    /// would not notice the difference between the two.
    TrialOutcome recorded_outcome{TrialOutcome::pending};
    /// Whether the acquisition interval is still a measurement.
    bool acquisition_timing_valid{};
};

/// One recorded target onset.
struct WebGridTargetOnset
{
    /// Experiment instant the target came up.
    ExperimentTimeNs onset_ns{};
    /// The target.
    TargetId target_id{kUnsetTargetId};
    /// Targets decided before it.
    TrialOrdinal completed{};
    /// Trial it belonged to.
    TrialIdentity trial{};
};

/// The recorded semantic outputs a replay is checked against.
struct WebGridReplayExpectation
{
    /// Pointer observations, in order.
    std::span<const WebGridPointerSample> pointer{};
    /// Selections, in order.
    std::span<const WebGridReplaySelection> selections{};
    /// Decided targets, in order.
    std::span<const WebGridReplayTrial> trials{};
    /// Target onsets, in order.
    std::span<const WebGridTargetOnset> targets{};

    /// The metric set the run published at the end.
    WebGridMetrics metrics{};
    /// Targets an abnormal condition invalidated, as the run counted them.
    std::uint64_t invalidated_targets{};

    /// Whether the recording carried pointer observations.
    ///
    /// False for a run that named an external pointer stream and a reader that
    /// did not recover it. WebGrid cannot be replayed without a pointer
    /// timeline at all, so this being false is what makes such a recording
    /// unreplayable rather than merely partially checkable.
    bool has_pointer{};
    /// Whether it carried selections.
    bool has_selections{};
    /// Whether it carried decided targets.
    bool has_trials{};
    /// Whether it carried target onsets.
    bool has_targets{};
    /// Whether it carried the published metrics.
    bool has_metrics{};
};

/// One recorded WebGrid run, as a reader recovers it.
struct WebGridReplayRecording
{
    /// What the recording says the run was.
    ReplayProvenance provenance{};
    /// Experiment instant the run started.
    ExperimentTimeNs origin_ns{};
    /// Recorded semantic inputs, in the order the run consumed them.
    std::span<const WebGridReplayInput> inputs{};
    /// Recorded semantic outputs.
    WebGridReplayExpectation expected{};
    /// Whether the recording holds the run's own end.
    bool run_end_recorded{};
    /// Whether the recording reports that trace records were lost.
    bool trace_loss_recorded{};
};

/// Re-executes a recorded WebGrid run and checks it against what was recorded.
class WebGridReplay
{
  public:
    WebGridReplay() noexcept = default;

    /// Fix the configuration this replay regenerates under.
    [[nodiscard]] ContractStatus prepare(const WebGridReplayConfig& config) noexcept;

    /// Return to the state a prepared replay starts each run from.
    void reset() noexcept;

    /// Replay @p recording and report what it established.
    [[nodiscard]] ReplayReport
    run(const WebGridReplayRecording& recording,
        ReplayIncompletePolicy policy = ReplayIncompletePolicy::verify_available) noexcept;

    /// The provenance this configuration regenerates under.
    [[nodiscard]] ReplayProvenance provenance() const noexcept;

    /// Whether a configuration has been accepted.
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }

    /// The configuration prepare() accepted.
    [[nodiscard]] const WebGridReplayConfig& configuration() const noexcept
    {
        return config_;
    }

  private:
    struct Cursors;

    [[nodiscard]] ContractStatus replay_pointer(const WebGridReplayInput& input,
                                                bool with_selection,
                                                const WebGridReplayRecording& recording,
                                                Cursors& cursors,
                                                ReplayComparator& comparator) noexcept;
    void replay_decision(const WebGridReplayInput& input, ReplayComparator& comparator) noexcept;
    void compare_target(const WebGridSnapshot& snapshot, const WebGridReplayRecording& recording,
                        Cursors& cursors, ReplayComparator& comparator) noexcept;

    WebGridReplayConfig config_{};
    WebGridMachine machine_{};
    WebGridMetrics metrics_{};
    std::uint64_t invalidated_targets_{};
    ExperimentTimeNs last_onset_ns_{};
    TargetId last_target_{kUnsetTargetId};
    bool target_invalidated_{};
    bool has_target_{};
    bool running_{};
    bool terminal_{};
    bool prepared_{};
};

} // namespace neurale::experiments::webgrid
