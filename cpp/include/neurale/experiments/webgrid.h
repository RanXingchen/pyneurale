// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <neurale/experiments/events.h>
#include <neurale/experiments/schedule.h>
#include <span>

/**
 * @file
 * @brief Renderer-independent WebGrid configuration, geometry, schedule, and state machine.
 *
 * WebGrid v1 is a discrete-selection task. Pointer motion is merely an input;
 * no selection is inferred until make_selection_event() is called. This header
 * includes the explicit-time state machine but owns no browser, renderer,
 * input device, clock, or recorder.
 */
namespace neurale::experiments::webgrid
{
/// Maximum cells in one prepared grid and maximum explicit schedule length.
inline constexpr std::size_t kMaxWebGridCells = 256;

/// Draw stream reserved for WebGrid target selection.
inline constexpr DrawStream kTargetSelectionStream = 2;

/// Version 1 of the raw WebGrid benchmark metric contract.
inline constexpr std::uint16_t kMetricVersion1 = 1;
inline constexpr std::uint16_t kCurrentMetricVersion = kMetricVersion1;

enum class TargetScheduleKind : std::uint8_t
{
    unspecified = 0,
    seeded,
    explicit_sequence,
};

enum class ImmediateRepetitionPolicy : std::uint8_t
{
    unspecified = 0,
    allow,
    forbid,
};

enum class CorrectSelectionPolicy : std::uint8_t
{
    unspecified = 0,
    advance_target,
};

enum class IncorrectSelectionPolicy : std::uint8_t
{
    unspecified = 0,
    keep_current_target,
};

[[nodiscard]] constexpr bool target_schedule_kind_declared(TargetScheduleKind value) noexcept
{
    return static_cast<std::uint8_t>(value) <=
           static_cast<std::uint8_t>(TargetScheduleKind::explicit_sequence);
}

[[nodiscard]] constexpr bool
immediate_repetition_policy_declared(ImmediateRepetitionPolicy value) noexcept
{
    return static_cast<std::uint8_t>(value) <=
           static_cast<std::uint8_t>(ImmediateRepetitionPolicy::forbid);
}

[[nodiscard]] constexpr bool
correct_selection_policy_declared(CorrectSelectionPolicy value) noexcept
{
    return static_cast<std::uint8_t>(value) <=
           static_cast<std::uint8_t>(CorrectSelectionPolicy::advance_target);
}

[[nodiscard]] constexpr bool
incorrect_selection_policy_declared(IncorrectSelectionPolicy value) noexcept
{
    return static_cast<std::uint8_t>(value) <=
           static_cast<std::uint8_t>(IncorrectSelectionPolicy::keep_current_target);
}

/// Logical task-space rectangle. Both minima are included; both maxima are excluded.
struct TaskBounds
{
    double min_x{};
    double max_x{};
    double min_y{};
    double max_y{};
};

/// One pointer observation in logical task coordinates.
struct PointerPosition
{
    double x{};
    double y{};
};

/// Half-open geometry of one cell.
struct CellBounds
{
    double min_x{};
    double max_x{};
    double min_y{};
    double max_y{};
};

/// One resolved row-major cell.
struct GridCell
{
    TargetId id{kUnsetTargetId};
    std::uint16_t row{};
    std::uint16_t column{};
    CellBounds bounds{};
};

/// Immutable WebGrid v1 session configuration.
///
/// Cell identifiers are one-based row-major values: `row * columns + column + 1`.
/// Row zero starts at `bounds.min_y`, and column zero at `bounds.min_x`.
struct WebGridConfig
{
    std::uint16_t rows{};
    std::uint16_t columns{};
    TaskBounds bounds{};

    /// Explicit selectable/target candidate pool, in schedule order.
    std::uint16_t n_candidates{};
    std::array<TargetId, kMaxWebGridCells> candidates{};

    TargetScheduleKind schedule{TargetScheduleKind::unspecified};
    ImmediateRepetitionPolicy immediate_repetition{ImmediateRepetitionPolicy::unspecified};
    CorrectSelectionPolicy correct_selection{CorrectSelectionPolicy::unspecified};
    IncorrectSelectionPolicy incorrect_selection{IncorrectSelectionPolicy::unspecified};
    ScheduleSeed seed{};
    SamplerVersion sampler_version{kCurrentSamplerVersion};

    /// Finite explicit schedule. It never wraps; exhaustion is terminal.
    std::uint16_t n_explicit{};
    std::array<TargetId, kMaxWebGridCells> explicit_targets{};

    /// Optional target at ordinal zero. For an explicit schedule it must equal its first item.
    TargetId initial_target{kUnsetTargetId};
    /// Zero means no duration limit. Exact end belongs to the terminal side.
    DurationNs session_duration_ns{};
    /// Zero means no configured count limit. Explicit schedule exhaustion still terminates.
    TrialOrdinal target_count_limit{};
    /// Metric formula interpreted by WebGridMachine. Zero is unset.
    std::uint16_t metric_version{kCurrentMetricVersion};
};

[[nodiscard]] ContractStatus validate(const TaskBounds& bounds) noexcept;
[[nodiscard]] ContractStatus validate(const PointerPosition& pointer) noexcept;
[[nodiscard]] ContractStatus validate(const CellBounds& bounds) noexcept;
[[nodiscard]] ContractStatus validate(const GridCell& cell) noexcept;
[[nodiscard]] ContractStatus validate(const WebGridConfig& config) noexcept;

[[nodiscard]] constexpr std::size_t cell_count(const WebGridConfig& config) noexcept
{
    return static_cast<std::size_t>(config.rows) * static_cast<std::size_t>(config.columns);
}

/// Resolve one row and column. The output is unchanged on failure.
[[nodiscard]] ContractStatus grid_cell(const WebGridConfig& config, std::uint16_t row,
                                       std::uint16_t column, GridCell& cell) noexcept;

/// Resolve a row-major identifier. The output is unchanged on failure.
[[nodiscard]] ContractStatus grid_cell(const WebGridConfig& config, TargetId id,
                                       GridCell& cell) noexcept;

/// Map a finite pointer to any physical cell, or kUnsetTargetId outside the grid.
[[nodiscard]] ContractStatus locate_cell(const WebGridConfig& config,
                                         const PointerPosition& pointer, TargetId& id) noexcept;

/// As locate_cell(), but non-selectable cells resolve to kUnsetTargetId.
[[nodiscard]] ContractStatus locate_selectable_cell(const WebGridConfig& config,
                                                    const PointerPosition& pointer,
                                                    TargetId& id) noexcept;

[[nodiscard]] bool selectable(const WebGridConfig& config, TargetId id) noexcept;

/// Resolve target ordinal @p ordinal. kUnsetTargetId with ok means terminal.
///
/// For a seeded schedule with repetition forbidden, @p previous is required for
/// every ordinal after zero. It is ignored at ordinal zero, so reset is simply
/// the caller returning to ordinal zero; the sampler itself holds no state.
[[nodiscard]] ContractStatus select_target(const WebGridConfig& config, TrialOrdinal ordinal,
                                           TargetId previous, TargetId& target) noexcept;

[[nodiscard]] bool target_limit_reached(const WebGridConfig& config,
                                        TrialOrdinal completed) noexcept;

/// Check the optional duration limit. Exact `start + duration` is terminal.
[[nodiscard]] ContractStatus session_duration_reached(const WebGridConfig& config,
                                                      ExperimentTimeNs start_ns,
                                                      ExperimentTimeNs time_ns,
                                                      bool& reached) noexcept;

/// Construct one explicit discrete selection from the pointer at selection time.
/// Merely calling locate_cell() or moving the pointer emits no event.
[[nodiscard]] ContractStatus
make_selection_event(const WebGridConfig& config, const PointerPosition& pointer, TargetId intended,
                     TrialIdentity trial, ParadigmId paradigm, ExperimentTimeNs time_ns,
                     SequenceOrdinal sequence, SelectionEvent& event) noexcept;

[[nodiscard]] std::uint64_t configuration_fingerprint(const WebGridConfig& config) noexcept;

/**
 * @brief Pure deterministic WebGrid task machine and raw benchmark metrics.
 *
 * The machine consumes explicit time, a logical pointer position, and zero or
 * one already-constructed discrete SelectionEvent. It reads no clock and owns
 * no renderer, browser, input device, decoder, stream, or recorder.
 * SelectionEvent::sequence is the session-local selection identity; a repeated
 * or regressed sequence is rejected and never counted twice.
 */
enum class WebGridState : std::uint8_t
{
    idle = 0,
    active_target,
    complete,
};

[[nodiscard]] constexpr bool webgrid_state_declared(WebGridState state) noexcept
{
    return static_cast<std::uint8_t>(state) <= static_cast<std::uint8_t>(WebGridState::complete);
}

/// TrialRecord::reason for a correctly selected WebGrid target.
enum class WebGridReason : std::uint32_t
{
    unspecified = 0,
    target_selected,
};

/// One accepted discrete selection with the target-onset context needed to
/// recompute every published metric offline.
struct WebGridSelectionRecord
{
    SelectionEvent event{};
    ExperimentTimeNs target_onset_ns{};
    DurationNs elapsed_since_target_onset_ns{};
};

/// One successfully completed target trial. Incorrect selections never create
/// this record and leave the current trial active.
struct WebGridTrial
{
    TrialRecord record{};
    WebGridSelectionRecord selection{};
    DurationNs acquisition_ns{};
};

/// Raw WebGrid metric v1.
///
/// `correct_targets_per_minute` is
/// `correct_selections * 60e9 / elapsed_active_ns`.
/// `net_correct_targets_per_minute` is
/// `(correct_selections - incorrect_selections) * 60e9 / elapsed_active_ns`.
/// Both rates are zero and `rates_defined` is false when elapsed time is zero.
/// Acquisition summaries use correct selections only; mean nanoseconds is the
/// integer-truncated total/count. No BPS or achieved-bitrate metric is defined.
struct WebGridMetrics
{
    std::uint16_t metric_version{kCurrentMetricVersion};
    std::uint64_t correct_selections{};
    std::uint64_t incorrect_selections{};
    DurationNs elapsed_active_ns{};
    bool rates_defined{};
    double correct_targets_per_minute{};
    double net_correct_targets_per_minute{};
    std::uint64_t n_acquisitions{};
    DurationNs total_acquisition_ns{};
    DurationNs minimum_acquisition_ns{};
    DurationNs maximum_acquisition_ns{};
    DurationNs mean_acquisition_ns{};
};

struct WebGridSnapshot
{
    /// Most recent explicit step time, or start time before the first step.
    ExperimentTimeNs time_ns{};
    WebGridState state{WebGridState::idle};
    TrialIdentity trial{};
    TargetId active_target{kUnsetTargetId};
    ExperimentTimeNs target_onset_ns{};
    TrialOrdinal completed{};
    WebGridMetrics metrics{};
};

struct WebGridStepResult
{
    WebGridSnapshot snapshot{};
    bool selection_processed{};
    WebGridSelectionRecord selection{};
    bool trial_decided{};
    WebGridTrial trial{};
};

[[nodiscard]] ContractStatus validate(const WebGridSelectionRecord& record) noexcept;
[[nodiscard]] ContractStatus validate(const WebGridTrial& trial) noexcept;

/// Recompute metric v1 from raw accepted selections.
///
/// Records must be in strictly increasing SelectionEvent::sequence order and
/// nondecreasing event-time order, and their times must lie within the closed
/// observed range [start_ns, end_ns]. Incorrect selections retain the exact
/// trial identity and target onset. A correct selection closes its trial; the
/// next record, if any, belongs to ordinal + 1 and has an onset equal to the
/// closing selection time. @p metric_version is explicit so persisted v1
/// records are never reinterpreted under a future current version. Output is
/// unchanged on failure.
[[nodiscard]] ContractStatus summarize(std::span<const WebGridSelectionRecord> selections,
                                       ExperimentTimeNs start_ns, ExperimentTimeNs end_ns,
                                       std::uint16_t metric_version,
                                       WebGridMetrics& metrics) noexcept;

class WebGridMachine
{
  public:
    WebGridMachine() noexcept = default;

    /// Start one session and activate target ordinal zero.
    [[nodiscard]] ContractStatus start(ParadigmId paradigm, const WebGridConfig& config,
                                       ExperimentTimeNs time_ns,
                                       WebGridStepResult& result) noexcept;

    void reset() noexcept;

    /// Advance without a selection. Pointer movement alone never selects.
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, const PointerPosition& pointer,
                                      WebGridStepResult& result) noexcept;

    /// Advance with exactly one explicit selection at @p time_ns.
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, const PointerPosition& pointer,
                                      const SelectionEvent& selection,
                                      WebGridStepResult& result) noexcept;

    [[nodiscard]] WebGridSnapshot snapshot() const noexcept;

    [[nodiscard]] const WebGridConfig& configuration() const noexcept
    {
        return config_;
    }

    [[nodiscard]] ParadigmId paradigm() const noexcept
    {
        return paradigm_;
    }

    [[nodiscard]] WebGridState state() const noexcept
    {
        return run_.state;
    }

    [[nodiscard]] bool complete() const noexcept
    {
        return run_.state == WebGridState::complete;
    }

  private:
    struct Run;

    [[nodiscard]] ContractStatus step_impl(ExperimentTimeNs time_ns, const PointerPosition& pointer,
                                           const SelectionEvent* selection,
                                           WebGridStepResult& result) noexcept;
    [[nodiscard]] WebGridSnapshot build_snapshot(const Run& run,
                                                 ExperimentTimeNs time_ns) const noexcept;

    struct Run
    {
        MonotonicTimeGate gate{};
        WebGridState state{WebGridState::idle};
        ExperimentTimeNs session_started_ns{};
        ExperimentTimeNs ended_ns{};
        TrialIdentity trial{};
        TargetId active_target{kUnsetTargetId};
        ExperimentTimeNs target_onset_ns{};
        TrialOrdinal completed{};
        std::uint64_t incorrect{};
        bool has_last_selection{};
        SequenceOrdinal last_selection_sequence{};
        DurationNs total_acquisition_ns{};
        DurationNs minimum_acquisition_ns{};
        DurationNs maximum_acquisition_ns{};
    };

    ParadigmId paradigm_{kUnsetParadigmId};
    WebGridConfig config_{};
    Run run_{};
};
} // namespace neurale::experiments::webgrid
