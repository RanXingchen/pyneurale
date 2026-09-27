// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <neurale/experiments/contract.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/schedule.h>
#include <neurale/experiments/time.h>
#include <span>

/**
 * @file
 * @brief Center-Out 2D task configuration, geometry, schedule, and state machine.
 *
 * Configuration and geometry define the task; CenterOutMachine advances its
 * semantics from explicit time and cursor observations. Neither owns a timer,
 * device, renderer, or recording. Reference-velocity guidance remains separate.
 *
 * The radial angular step is `2*pi / max(count, kMinRadialSpokes)`, containment
 * is an axis-aligned whole-extent test, and target-selection policy states
 * explicitly whether a failed trial re-presents the same target. Geometry is
 * `double` throughout and is never narrowed for presentation.
 */
namespace neurale::experiments::center_out
{
/// Largest number of surrounding targets one layout holds.
///
/// Classical Center-Out uses eight. Sixteen leaves room for a denser ring while
/// keeping the configuration a fixed-size value that can be copied, compared,
/// and fingerprinted without owning storage.
inline constexpr std::size_t kMaxSurroundingTargets = 16;

/// Smallest divisor the radial angular step uses.
///
/// The `2*pi / max(count, 8)` floor means that with fewer than eight surrounding
/// targets the ring is *not* filled: four targets on spokes
/// 0..3 sit at 0, 45, 90 and 135 degrees and leave the other half of the circle
/// empty. Spreading them at 90 degrees would define a different layout.
inline constexpr std::uint32_t kMinRadialSpokes = 8;

/// Draw stream outward targets are sampled from.
///
/// Stream tags are paradigm-local: two paradigms sharing one ::ScheduleSeed
/// would collide on them, so a session running more than one paradigm gives each
/// its own seed rather than trying to partition a shared stream space.
inline constexpr DrawStream kTargetSelectionStream = 1;

/// Unit the workspace geometry is measured in.
///
/// Positions, acceptance half-extents, and the cursor extent are all in this one
/// unit; the contract has no mixed-unit geometry and no conversion. It is not
/// CommandUnit: that enumeration names velocity units too, and a geometry field
/// that could legally say "metres per second" is not an explicit unit.
enum class GeometryUnit : std::uint8_t
{
    /// Unset.
    unspecified = 0,
    /// A pure number with no unit.
    dimensionless,
    /// Scaled to the workspace, conventionally in `[-1, 1]`.
    normalized,
    /// Millimetres.
    millimetres,
    /// Metres.
    metres,
};

/// Which leg of a Center-Out trial a per-phase parameter applies to.
///
/// There is no unset enumerator: every leg of a trial is one of these two, so a
/// "no phase" value would name nothing. The ordinals are stable persisted
/// values.
enum class CenterOutPhase : std::uint8_t
{
    /// Moving from wherever the cursor is back to the centre target.
    to_center = 0,
    /// Moving from the centre target out to the selected surrounding target.
    to_out,
};

/// How the outward target of the next trial is chosen.
///
/// The names state whether failure repeats or advances the outward target.
enum class TargetSelectionPolicy : std::uint8_t
{
    /// Unset. Rejected by validate(const CenterOut2DConfig&); there is no
    /// default policy, because the two below produce different experiments and
    /// picking one silently would decide that for the caller.
    unspecified = 0,
    /// Step through the surrounding targets in layout order, advancing exactly
    /// one step per success.
    ///
    /// A failed trial therefore re-presents the same target until it is hit.
    /// The centre is not a member of the surrounding array.
    repeat_until_success,
    /// Draw a fresh surrounding target for every outward leg.
    ///
    /// A failure changes the target as readily as a success does.
    sample_each_trial,
    /// Shuffle the surrounding targets independently in fixed-size cycles.
    ///
    /// Every cycle contains every surrounding target exactly once. A failure
    /// still advances the schedule, so every prefix differs by at most one
    /// presentation between target directions.
    balanced_shuffled_cycles,
};

/// Whether @p unit is one of the declared GeometryUnit values.
[[nodiscard]] constexpr bool geometry_unit_declared(GeometryUnit unit) noexcept
{
    return static_cast<std::uint8_t>(unit) <= static_cast<std::uint8_t>(GeometryUnit::metres);
}

/// Whether @p phase is one of the declared CenterOutPhase values.
[[nodiscard]] constexpr bool center_out_phase_declared(CenterOutPhase phase) noexcept
{
    return static_cast<std::uint8_t>(phase) <= static_cast<std::uint8_t>(CenterOutPhase::to_out);
}

/// Whether @p policy is one of the declared TargetSelectionPolicy values.
[[nodiscard]] constexpr bool target_selection_policy_declared(TargetSelectionPolicy policy) noexcept
{
    return static_cast<std::uint8_t>(policy) <=
           static_cast<std::uint8_t>(TargetSelectionPolicy::balanced_shuffled_cycles);
}

/// One point of the paradigm's own 2D semantic workspace.
struct WorkspacePoint
{
    /// First workspace axis.
    double x{};
    /// Second workspace axis.
    double y{};
};

/// One target: a semantic identifier and where it is.
///
/// The identifier is the label a record refers the target by; it is not a
/// position in any array, and nothing derives geometry from it.
struct TargetPlacement
{
    /// Semantic identifier. ::kUnsetTargetId is rejected.
    TargetId id{kUnsetTargetId};
    /// Where the target is, in the configured GeometryUnit.
    WorkspacePoint pos{};
};

/// The resolved target geometry of one session.
///
/// The centre is a field of its own rather than an entry of the array, so
/// "the centre target cannot be selected as an outward target" holds by
/// construction instead of by an off-by-one in a selection expression. The
/// array is stored in caller order, and nothing in this header reorders it.
struct CenterOut2DLayout
{
    /// The centre target.
    TargetPlacement center{};
    /// Number of meaningful entries in `surrounding`.
    std::uint8_t count{};
    /// Surrounding targets, in caller order. Entries at or beyond `count` are
    /// unset.
    std::array<TargetPlacement, kMaxSurroundingTargets> surrounding{};
};

/// The acceptance region of a target: a rectangle centred on it.
///
/// It has no implicit default: task acceptance geometry is explicit paradigm
/// configuration and is never supplied by a renderer.
struct AcceptanceRegion
{
    /// Half-width along the first axis. Must be finite and positive.
    double half_extent_x{};
    /// Half-height along the second axis. Must be finite and positive.
    double half_extent_y{};
};

/// The only cursor geometry the hit test reads.
///
/// Half-extent of the cursor's axis-aligned box. Rendered target and cursor
/// radii are presentation values and are not part of semantic containment.
struct CursorGeometry
{
    /// Half-extent of the cursor along both axes. Must be finite and
    /// non-negative. Zero makes body containment coincide with point
    /// containment.
    double extent{};
};

/// A duration given once per trial leg.
struct PhaseDurations
{
    /// Value for CenterOutPhase::to_center.
    DurationNs to_center{};
    /// Value for CenterOutPhase::to_out.
    DurationNs to_out{};
};

/// The value of @p durations for @p phase.
///
/// @pre center_out_phase_declared(phase). The two declared legs are the only
/// values this accessor is defined for: the else-branch returns ::to_out only
/// because ::to_out is the other declared leg, not because every value falls
/// back to it. Persisted phase state that reaches this accessor must check
/// declared-ness first rather than rely on that fall-through.
[[nodiscard]] constexpr DurationNs phase_duration(const PhaseDurations& durations,
                                                  CenterOutPhase phase) noexcept
{
    return phase == CenterOutPhase::to_center ? durations.to_center : durations.to_out;
}

/// Immutable Center-Out 2D configuration.
///
/// Everything a session's geometry and schedule depend on, and nothing else.
/// There is no device type, no monitor or window, no colour, no decoder
/// parameter, no recorder path, and no streaming configuration: those are not
/// properties of the task, and a configuration that carried them could not be
/// compared across two sessions that ran the same task on different hardware.
struct CenterOut2DConfig
{
    /// Unit every geometry field below is measured in.
    GeometryUnit geometry_unit{GeometryUnit::unspecified};
    /// Centre and surrounding targets.
    CenterOut2DLayout layout{};
    /// Acceptance region, applied identically to every target.
    ///
    /// One region rather than one per target. Per-target acceptance geometry is
    /// not part of the current task contract.
    AcceptanceRegion acceptance{};
    /// The cursor extent the hit test reads.
    CursorGeometry cursor{};
    /// How long a leg of a trial may take before it has failed.
    ///
    /// Zero is rejected: a leg with no time cannot be completed, so a
    /// configuration containing one describes a session in which nothing can
    /// ever succeed.
    PhaseDurations movement_timeout{};
    /// How long the cursor must remain inside the acceptance region.
    ///
    /// Zero is admissible and means acquisition is instantaneous. How this
    /// relates to `movement_timeout` -- whether the hold
    /// runs inside the timeout or after it -- is task semantics and is decided
    /// by the state machine, not here.
    DurationNs hold_ns{};
    /// How long a successful leg dwells before the next one begins.
    PhaseDurations reward_dwell{};
    /// How long a failed leg dwells before the next one begins.
    PhaseDurations punish_dwell{};
    /// How the outward target of each trial is chosen.
    TargetSelectionPolicy selection{TargetSelectionPolicy::unspecified};
    /// Seed of this session's schedule.
    ScheduleSeed seed{};
    /// Sampler the schedule's draws are made with.
    SamplerVersion sampler_version{kCurrentSamplerVersion};
    /// Number of trials the session runs, or zero for no limit.
    TrialOrdinal trial_limit{};
};

/// A request for a radial Center-Out layout.
///
/// Held as an explicit value rather than derived from the target identifiers,
/// so that the angle of a target and the name of a target are independent. The
/// caller's value is never modified: it is read, and a new layout is written.
struct RadialLayoutRequest
{
    /// Distance from the centre to every surrounding target. Finite, positive.
    double radius{};
    /// Number of surrounding targets to place.
    std::uint8_t count{};
    /// Identifier of the centre target. ::kUnsetTargetId is rejected.
    TargetId center_id{kUnsetTargetId};
    /// Identifier of each surrounding target, in caller order.
    std::array<TargetId, kMaxSurroundingTargets> ids{};
    /// Spoke each surrounding target sits on, in caller order.
    ///
    /// Target `i` is placed at angle `spokes[i] * radial_spoke_step(count)`
    /// measured counter-clockwise from the first workspace axis. A layout may
    /// leave spokes empty or put two targets on one spoke.
    std::array<std::uint32_t, kMaxSurroundingTargets> spokes{};
};

/// Angular step, in radians, of a radial layout with @p count surrounding
/// targets.
///
/// `2*pi / max(count, kMinRadialSpokes)`. See ::kMinRadialSpokes for why the
/// floor is there and what it does to small layouts.
///
/// @pre @p count is the validated surrounding count of a radial layout (at
/// least one). A count of zero is rejected by validate(const
/// RadialLayoutRequest&) and is not a meaningful layout; it is not checked here
/// only because this is the low-level step the validated path calls. The
/// max(count, kMinRadialSpokes) floor keeps a zero finite rather than dividing
/// by zero, but that is a guard against misuse, not a licence to pass it.
[[nodiscard]] double radial_spoke_step(std::uint32_t count) noexcept;

/// Position of one spoke of a radial layout.
///
/// @param radius Distance from the centre. Must be finite and positive.
/// @param count Number of surrounding targets the step is computed from.
///              Must be the validated count of a radial layout (at least one);
///              this is a low-level helper of the validated layout path and does
///              not itself reject zero.
/// @param spoke Which spoke, counted from the first workspace axis.
/// @param pos Receives the position on ContractStatus::ok, and is left
///            unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::value_not_finite when @p radius
///         is not finite, or ContractStatus::parameter_out_of_range when it is
///         not positive.
[[nodiscard]] ContractStatus radial_position(double radius, std::uint32_t count,
                                             std::uint32_t spoke, WorkspacePoint& pos) noexcept;

/// Build the layout @p request describes.
///
/// The centre target is placed at the origin. A layout whose centre is
/// elsewhere is built from explicit coordinates; this helper is a convenience
/// for the radial case, not the only way to obtain a layout.
///
/// @param request What to build. Never modified.
/// @param layout Receives the layout on ContractStatus::ok, and is left
///               unmodified otherwise.
/// @return ContractStatus::ok, or the first failure of validate(const
///         CenterOut2DLayout&) applied to the layout that would have been
///         built, or a radial_position() failure.
[[nodiscard]] ContractStatus build_radial_layout(const RadialLayoutRequest& request,
                                                 CenterOut2DLayout& layout) noexcept;

/// Whether @p cursor, treated as a point, lies inside the acceptance region
/// centred on @p target.
///
/// The region is closed: a point exactly on the boundary is inside.
///
/// @param region Acceptance geometry.
/// @param target Centre of the region.
/// @param cursor The point tested.
/// @param inside Receives the answer on ContractStatus::ok, and is left
///               unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::value_not_finite when any input
///         is NaN or infinite, or ContractStatus::parameter_out_of_range when a
///         half-extent is not positive.
[[nodiscard]] ContractStatus contains_point(const AcceptanceRegion& region,
                                            const WorkspacePoint& target,
                                            const WorkspacePoint& cursor, bool& inside) noexcept;

/// Whether the cursor's whole extent lies inside the acceptance region centred
/// on @p target.
///
/// The semantic-coordinate predicate is:
///
/// ```text
/// cursor.x - extent >= target.x - region.half_extent_x
/// cursor.x + extent <= target.x + region.half_extent_x
/// cursor.y - extent >= target.y - region.half_extent_y
/// cursor.y + extent <= target.y + region.half_extent_y
/// ```
///
/// Four edge comparisons are used rather than two absolute values. The forms
/// are the same inequality in exact arithmetic but can differ within an ulp of
/// the boundary because they group subtractions differently.
///
/// It is not a centre-to-centre distance test, and it is not a disc-overlap
/// test: the cursor must be *contained*, not merely touching. Equality is
/// inside, as for contains_point(), which this reduces to exactly when the
/// cursor extent is zero.
///
/// Input positions are semantic values and are not clamped by presentation.
/// Comparisons use `double`.
///
/// A region smaller than the cursor is not an error for this predicate -- it
/// simply answers `false` everywhere -- but it is rejected by
/// validate(const CenterOut2DConfig&), because a session configured that way
/// cannot ever succeed.
///
/// @param region Acceptance geometry.
/// @param cursor_geometry Cursor extent.
/// @param target Centre of the region.
/// @param cursor Cursor centre.
/// @param inside Receives the answer on ContractStatus::ok, and is left
///               unmodified otherwise.
/// @return As contains_point(), and additionally
///         ContractStatus::parameter_out_of_range when the cursor extent is
///         negative.
[[nodiscard]] ContractStatus contains_cursor(const AcceptanceRegion& region,
                                             const CursorGeometry& cursor_geometry,
                                             const WorkspacePoint& target,
                                             const WorkspacePoint& cursor, bool& inside) noexcept;

/// Which surrounding target the outward leg of a trial presents.
///
/// A pure function of the configuration and two ordinals. It holds no state, so
/// resetting a schedule is the caller returning the ordinals to where they
/// started; the same ordinals then yield the same targets they yielded before,
/// and a session can be resumed from them without carrying generator state.
///
/// Under TargetSelectionPolicy::repeat_until_success the answer depends on
/// @p successes and not on @p trial, which is what makes a failed trial
/// re-present its target. Under TargetSelectionPolicy::sample_each_trial it
/// depends on @p trial and not on @p successes.
///
/// The centre target is never a possible answer, because the answer is an index
/// into the surrounding array and the centre is not in it.
///
/// @param config Session configuration.
/// @param trial Zero-based ordinal of the trial whose outward leg is being
///              selected.
/// @param successes Number of outward legs completed successfully so far.
/// @param idx Receives the index into `config.layout.surrounding` on
///              ContractStatus::ok, and is left unmodified otherwise.
/// @return ContractStatus::ok, the first failure of
///         validate(const CenterOut2DConfig&),
///         ContractStatus::version_unsupported when a draw would be needed and
///         this build cannot evaluate the configured sampler, or
///         ContractStatus::sampling_exhausted from the bounded draw.
[[nodiscard]] ContractStatus select_outward_target(const CenterOut2DConfig& config,
                                                   TrialOrdinal trial, std::uint64_t successes,
                                                   std::uint8_t& idx) noexcept;

/// Whether @p completed trials have reached the configured limit.
///
/// Always false when `config.trial_limit` is zero, which is what "no limit"
/// means. This is a question about a count, not a session state: it does not
/// know whether a trial is in progress and does not end anything.
[[nodiscard]] constexpr bool trial_limit_reached(const CenterOut2DConfig& config,
                                                 TrialOrdinal completed) noexcept
{
    return config.trial_limit != 0 && completed >= config.trial_limit;
}

/// Validate a target placement.
[[nodiscard]] ContractStatus validate(const TargetPlacement& target) noexcept;

/// Validate a target layout, including that no identifier is used twice.
[[nodiscard]] ContractStatus validate(const CenterOut2DLayout& layout) noexcept;

/// Validate an acceptance region.
[[nodiscard]] ContractStatus validate(const AcceptanceRegion& region) noexcept;

/// Validate a cursor geometry.
[[nodiscard]] ContractStatus validate(const CursorGeometry& cursor) noexcept;

/// Validate a radial layout request.
[[nodiscard]] ContractStatus validate(const RadialLayoutRequest& request) noexcept;

/// Validate a Center-Out configuration.
///
/// Beyond validating each part, this rejects two combinations that are legal
/// field by field and describe a session nothing can complete: a cursor larger
/// than the acceptance region on either axis, and a movement timeout of zero on
/// either leg.
[[nodiscard]] ContractStatus validate(const CenterOut2DConfig& config) noexcept;

/// Digest of a target layout.
///
/// Two layouts with the same digest place the same identifiers at the same
/// coordinates in the same order. Negative zero is folded onto zero before
/// absorption, so a coordinate written as `-0.0` and one written as `0.0` do not
/// produce different digests for geometry that is in the same place.
[[nodiscard]] std::uint64_t layout_fingerprint(const CenterOut2DLayout& layout) noexcept;

/// Digest of a Center-Out configuration, for ScheduleIdentity.
///
/// Every field is absorbed, so a session that changed any of them is a
/// different schedule and will not be mistaken for the one it was derived from.
[[nodiscard]] std::uint64_t configuration_fingerprint(const CenterOut2DConfig& config) noexcept;

/**
 * @brief The deterministic Center-Out 2D task state machine.
 *
 * The machine is a pure function of what it is given. Its only inputs are an
 * explicit `time_ns`, an observed cursor position, the immutable
 * CenterOut2DConfig captured at start(), and its own state. It reads no clock,
 * starts no timer, spawns no thread, touches no device, writes no recording,
 * draws nothing, and allocates nothing. Two machines given the same
 * configuration and the same sequence of `(time_ns, cursor)` observations
 * produce byte-identical transitions, events, and trials.
 *
 * It never moves the cursor. Nothing here writes a position, and the next leg
 * begins from the next cursor position supplied by the caller.
 *
 * # Time
 *
 * Every window is a half-open TimeInterval, following the rule frozen in
 * `time.h`: the exact end instant belongs to the next phase and never to both.
 * That single rule decides every boundary case below, including the exact
 * precedence between a movement timeout and a completed hold, so none of them is
 * a separate preference the state machine had to invent.
 *
 * Deadlines are instants, not counters. A leg times out because `time_ns`
 * reached the instant its window ends, never because step() was called some
 * number of times, so a caller polling at 10 Hz and one polling at 1 kHz reach
 * the same outcomes from the same observations.
 *
 * # What one step does
 *
 * step() advances the machine repeatedly while some window has already ended at
 * `time_ns`, and stops when none has -- or when one trial's outcome has been
 * decided, whichever comes first. Deciding at most one trial per step is what
 * bounds the work: a caller that stopped polling for several trials' worth of
 * time would otherwise have a single call infer an unbounded run of timeouts.
 * Such a caller instead drains them one call at a time, and
 * CenterOutStepResult::settled says whether anything is still pending.
 *
 * When `settled` is true -- the case for any caller polling faster than one
 * trial -- step() is idempotent: repeating it at the same `time_ns` with the
 * same cursor produces no transition, no event, and no trial.
 */
/// Transitions one call to CenterOutMachine::step() can emit.
///
/// The longest chain a single step can take is: leave a dwell, acquire the
/// centre, leave the centre dwell, acquire the outward target, and decide the
/// trial -- six transitions, after which the step stops because a trial has been
/// decided. The capacity is stated above that so a future state cannot silently
/// overflow it.
inline constexpr std::size_t kMaxStepTransitions = 8;

/// Semantic events one call to CenterOutMachine::step() can emit.
inline constexpr std::size_t kMaxStepEvents = 4;

/// The states of one Center-Out session.
///
/// One enumeration distinguishes task phase and hold state:
/// ::move_to_center and ::hold_center are the same leg before and after the
/// cursor arrived, and how long it has been there is a time, not a state.
enum class CenterOutState : std::uint8_t
{
    /// No session. CenterOutMachine::step() refuses; only start() leaves it.
    idle = 0,
    /// The centre leg, before the cursor has been observed inside the centre.
    move_to_center,
    /// The centre leg, with the cursor inside and the hold accumulating.
    hold_center,
    /// The interval after the centre leg was acquired.
    center_success_dwell,
    /// The interval after the centre leg timed out.
    center_failure_dwell,
    /// The outward leg, before the cursor has been observed inside the target.
    move_to_out,
    /// The outward leg, with the cursor inside and the hold accumulating.
    hold_out,
    /// The interval after the outward target was acquired.
    out_success_dwell,
    /// The interval after the outward leg timed out.
    out_failure_dwell,
    /// The session reached its configured trial limit. Terminal.
    complete,
};

/// Whether @p state is one of the declared CenterOutState values.
[[nodiscard]] constexpr bool center_out_state_declared(CenterOutState state) noexcept
{
    return static_cast<std::uint8_t>(state) <= static_cast<std::uint8_t>(CenterOutState::complete);
}

/// Which leg @p state belongs to.
///
/// ::CenterOutState::idle and ::CenterOutState::complete belong to no leg; they
/// answer CenterOutPhase::to_center because the enumeration has no third value,
/// and a caller must not read a phase out of them.
[[nodiscard]] constexpr CenterOutPhase center_out_phase_of(CenterOutState state) noexcept
{
    return state == CenterOutState::move_to_out || state == CenterOutState::hold_out ||
                   state == CenterOutState::out_success_dwell ||
                   state == CenterOutState::out_failure_dwell
               ? CenterOutPhase::to_out
               : CenterOutPhase::to_center;
}

/// Whether @p state is one in which the cursor is watched against a target.
[[nodiscard]] constexpr bool center_out_state_is_leg(CenterOutState state) noexcept
{
    return state == CenterOutState::move_to_center || state == CenterOutState::hold_center ||
           state == CenterOutState::move_to_out || state == CenterOutState::hold_out;
}

/// Whether @p state is one in which a decided leg is being dwelt on.
[[nodiscard]] constexpr bool center_out_state_is_dwell(CenterOutState state) noexcept
{
    return state == CenterOutState::center_success_dwell ||
           state == CenterOutState::center_failure_dwell ||
           state == CenterOutState::out_success_dwell || state == CenterOutState::out_failure_dwell;
}

/// Why a transition was taken, stored in `StateTransition::cause`.
enum class CenterOutCause : std::uint32_t
{
    /// Unset.
    unspecified = 0,
    /// A session began.
    session_started,
    /// The cursor was observed inside the active acceptance region.
    containment_gained,
    /// The cursor was observed outside it again, discarding the hold.
    containment_lost,
    /// The hold reached its configured duration.
    hold_completed,
    /// The leg's movement window ended.
    movement_timed_out,
    /// A success or failure dwell ended.
    dwell_elapsed,
    /// The configured trial limit was reached.
    trial_limit_reached,
};

/// Why a trial ended, stored in `TrialRecord::reason`.
enum class CenterOutReason : std::uint32_t
{
    /// Unset.
    unspecified = 0,
    /// The outward target was held for its configured duration.
    outward_acquired,
    /// The centre leg's movement window ended before the centre was held.
    center_movement_timeout,
    /// The outward leg's movement window ended before the target was held.
    outward_movement_timeout,
};

/// One decided Center-Out trial.
///
/// `record` is a canonical experiment-side TrialRecord. Persisting it through
/// the recording control plane is the job of the integration bridge; the
/// fields beside it are what a Center-Out reader needs and
/// the shared record has nowhere to put.
///
/// Which durations are meaningful follows from `record.outcome` and
/// `decided_phase`, so neither needs a validity flag of its own: a success has
/// both, a CenterOutPhase::to_out timeout has `center_acquire_ns` only, and a
/// CenterOutPhase::to_center timeout has neither. They must also be consistent
/// with `record.interval` and with the leg that decided the trial: a meaningful
/// duration never exceeds the trial's length, a leg that did not acquire reports
/// zero, a success satisfies `center_acquire_ns + outward_acquire_ns` <=
/// `record.interval.duration_ns()` (the outward leg cannot begin before the
/// centre is acquired), and a timeout cannot be a zero-length trial (every
/// movement window is strictly positive). validate(const CenterOutTrial&)
/// rejects a trial whose interval and durations describe something that could
/// not have happened.
struct CenterOutTrial
{
    /// The trial as the shared contract records it.
    TrialRecord record{};
    /// Outward target this trial presented.
    TargetId outward_target{kUnsetTargetId};
    /// Its index in `CenterOut2DLayout::surrounding`.
    std::uint8_t outward_idx{};
    /// Leg whose decision ended the trial.
    CenterOutPhase decided_phase{CenterOutPhase::to_center};
    /// Trial start to centre acquisition.
    DurationNs center_acquire_ns{};
    /// Outward leg start to target acquisition.
    DurationNs outward_acquire_ns{};
};

/// Everything a caller can read about a session between steps.
///
/// A value, produced on demand from the machine's own state. Nothing reads it
/// back, and holding one does not pin the machine.
struct CenterOutSnapshot
{
    /// Most recent instant the machine accepted.
    ExperimentTimeNs time_ns{};
    /// Where the machine is.
    CenterOutState state{CenterOutState::idle};
    /// Leg the state belongs to. Meaningless in ::idle and ::complete.
    CenterOutPhase phase{CenterOutPhase::to_center};
    /// Trial in progress, or the last one decided once the session ended.
    TrialIdentity trial{};
    /// Target the cursor must acquire now, or ::kUnsetTargetId outside a leg.
    TargetId active_target{kUnsetTargetId};
    /// Where that target is.
    WorkspacePoint active_position{};
    /// Outward target of the trial in progress.
    TargetId outward_target{kUnsetTargetId};
    /// Its index in `CenterOut2DLayout::surrounding`.
    std::uint8_t outward_idx{};
    /// Whether the last observation was inside the active acceptance region.
    bool contained{};
    /// Movement window of the current leg, `[start, timeout)`.
    TimeInterval leg{};
    /// Hold window, `[first containment, completion)`. Empty when not holding.
    TimeInterval hold{};
    /// Dwell window, `[decision, end)`. Empty outside a dwell.
    TimeInterval dwell{};
    /// Trials decided so far.
    TrialOrdinal completed{};
    /// Trials decided as TrialOutcome::success.
    ///
    /// Carried because the schedule needs it -- select_outward_target() takes it
    /// -- not because it is a statistic. Statistics are computed from the
    /// decided trials by summarize(), never accumulated here.
    std::uint64_t successes{};
};

/// What one call to CenterOutMachine::step() or start() produced.
///
/// Fixed capacity and no ownership, so a step allocates nothing. The counts say
/// how much of each array is meaningful; entries beyond them are unset.
struct CenterOutStepResult
{
    /// State after the call.
    CenterOutSnapshot snapshot{};
    /// Whether no window remained ended when the call stopped.
    ///
    /// False only when the call stopped because it had decided a trial while
    /// more had already elapsed. Stepping again at the same `time_ns` continues.
    bool settled{true};
    /// Meaningful entries in `transitions`.
    std::uint8_t n_transitions{};
    /// Transitions taken, in order.
    std::array<StateTransition, kMaxStepTransitions> transitions{};
    /// Meaningful entries in `events`.
    std::uint8_t n_events{};
    /// Semantic events emitted, in order.
    std::array<ExperimentEvent, kMaxStepEvents> events{};
    /// Whether a trial was decided.
    bool trial_decided{};
    /// The decided trial. Meaningful only when `trial_decided`.
    CenterOutTrial trial{};
};

/// Statistics derived from decided trials.
///
/// There is no running accumulator anywhere in the machine. These are a
/// pure function of the decided trials instead, so they cannot.
///
/// The mean is integer nanoseconds, truncated, and is exactly reproducible; a
/// floating-point running mean is not.
struct CenterOutStatistics
{
    /// Trials decided.
    std::uint64_t decided{};
    /// Trials decided as TrialOutcome::success.
    std::uint64_t successes{};
    /// Trials whose centre leg timed out.
    std::uint64_t center_timeouts{};
    /// Trials whose outward leg timed out.
    std::uint64_t outward_timeouts{};
    /// Sum of `outward_acquire_ns` over successful trials.
    DurationNs total_time_to_target_ns{};
    /// Mean `outward_acquire_ns` over successful trials, truncated. Zero when
    /// there were none.
    DurationNs mean_time_to_target_ns{};
};

/// Validate a decided Center-Out trial.
///
/// A decided trial is exactly one of three field combinations -- a success, a
/// centre timeout, or an outward timeout -- each with the outcome, decided
/// phase, and reason that produce it, and with the record's target identifier
/// equal to the outward target. The shared TrialRecord validator runs first, so
/// an undeclared outcome is ::enum_undeclared; an unset outward target is
/// ::identity_missing; a target identifier that disagrees with the outward
/// target, an outward index no layout can hold (at or past
/// ::kMaxSurroundingTargets), or any other legal-but-impossible combination (a
/// pending, failed, or aborted outcome, or a phase and reason that do not match
/// it) is ::outcome_invalid. The acquisition durations must then agree with the
/// record interval and with the deciding leg, and these are the strongest such
/// checks that need no configuration: neither meaningful duration exceeds
/// `record.interval.duration_ns()`; a leg that did not acquire reports zero; a
/// success satisfies `center_acquire_ns + outward_acquire_ns` <= the trial
/// duration (the outward leg cannot begin before the centre is acquired, and
/// the sum is checked by subtraction so it cannot overflow); and a timeout is
/// not a zero-length trial, with an outward timeout ending strictly after the
/// centre was acquired -- every movement window is strictly positive, so the
/// machine could only ever have produced a trial that satisfies these. A
/// hand-built trial whose interval and durations describe an impossible trial
/// is ::outcome_invalid rather than counted by summarize(), which runs this on
/// every trial because the binding lets a caller build trials by hand.
[[nodiscard]] ContractStatus validate(const CenterOutTrial& trial) noexcept;

/// Summarize decided trials.
///
/// @param trials Decided trials, in any order.
/// @param statistics Receives the summary on ContractStatus::ok, and is left
///        unmodified otherwise.
/// @return ContractStatus::ok; the first validate(const CenterOutTrial&)
///         failure of any trial; or ContractStatus::duration_overflow when the
///         successful trials' acquisition times do not sum to a representable
///         nanosecond count.
[[nodiscard]] ContractStatus summarize(std::span<const CenterOutTrial> trials,
                                       CenterOutStatistics& statistics) noexcept;

/// The Center-Out 2D task state machine.
///
/// Value semantics: copyable, comparable field by field, no pointer, no owned
/// storage, and no hidden state. Copying a machine copies a session, and a copy
/// stepped with the same observations stays identical to its original. That is
/// the whole production core; the Python binding is a thin wrapper over it and
/// reimplements none of it.
class CenterOutMachine
{
  public:
    /// Construct an ::CenterOutState::idle machine with no session.
    CenterOutMachine() noexcept = default;

    /// Begin a session.
    ///
    /// The configuration is captured, not referenced: it cannot change under a
    /// running session, and a machine is readable without the caller having kept
    /// the value it was started from.
    ///
    /// Emits the ExperimentEventKind::session_start and
    /// ExperimentEventKind::trial_start events, and the transition into the
    /// first trial's centre leg.
    ///
    /// @param paradigm Provenance identifier stamped on every record. Never
    ///        dispatch: nothing selects behaviour from it.
    /// @param config Session configuration.
    /// @param time_ns Instant the session begins.
    /// @param result Receives the transitions, events, and snapshot on
    ///        ContractStatus::ok. Left unmodified otherwise.
    /// @return ContractStatus::ok, ContractStatus::already_running when the
    ///         machine already holds a session (any state other than ::idle; a
    ///         new session needs reset() first), ContractStatus::identity_missing
    ///         when @p paradigm is unset, the first failure of
    ///         validate(const CenterOut2DConfig&), a select_outward_target()
    ///         failure, or ContractStatus::duration_overflow when the first
    ///         movement window does not fit. A refused start leaves the machine
    ///         and @p result unmodified.
    [[nodiscard]] ContractStatus start(ParadigmId paradigm, const CenterOut2DConfig& config,
                                       ExperimentTimeNs time_ns,
                                       CenterOutStepResult& result) noexcept;

    /// Return to the state a default-constructed machine has.
    ///
    /// The configuration goes too. A reset session is a *new* session rather
    /// than a rewound one, so start() decides everything again from the
    /// configuration it is given; that is what makes "reset then replay"
    /// reproduce a run exactly instead of resuming into ordinals that were
    /// already emitted.
    void reset() noexcept;

    /// Advance the session to @p time_ns given one observed cursor position.
    ///
    /// The precedence rules, all of them consequences of the half-open interval
    /// convention in `time.h`:
    ///
    /// - In a move state, the movement timeout is checked first. Containment
    ///   observed at or after the window's end instant does not start a hold,
    ///   because at that instant the window is already over.
    /// - In a hold state, a hold that completed strictly before the movement
    ///   window's end wins, even if the cursor has since left and even if the
    ///   window has since ended: it completed first. Otherwise the movement
    ///   timeout wins, including the exact tie where the hold would have
    ///   completed at the very instant the window ends.
    /// - Otherwise, containment observed to be lost discards the hold and
    ///   returns to the move state with the movement window unchanged. A later
    ///   containment starts a *new* hold from the instant it was observed.
    ///
    /// A hold begins at the instant containment is first observed and never
    /// earlier, because containment between two steps was not observed at all.
    /// With a zero hold duration a leg is therefore acquired on the first
    /// observation inside the region.
    ///
    /// Between two observations the machine assumes containment persisted: a
    /// cursor that left the region and returned unobserved is a hold this
    /// machine will credit. That is a statement about what the observations
    /// determine, not a rounding rule -- a caller that needs a hold broken by a
    /// brief excursion has to observe the excursion.
    ///
    /// @param time_ns Instant of this observation. Must not move backwards.
    /// @param cursor Where the cursor is at @p time_ns.
    /// @param result Receives everything the step produced on
    ///        ContractStatus::ok. Left unmodified otherwise, as is the machine.
    /// @return ContractStatus::ok, ContractStatus::not_running in
    ///         ::CenterOutState::idle or ::CenterOutState::complete,
    ///         ContractStatus::time_regressed, ContractStatus::value_not_finite
    ///         for a non-finite cursor, a propagated schedule, ordinal, or
    ///         overflow failure, or ContractStatus::numerical_failure if the
    ///         chain ever exceeded its transition budget -- which the
    ///         one-trial-per-step rule makes unreachable, and which is checked
    ///         rather than assumed.
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, const WorkspacePoint& cursor,
                                      CenterOutStepResult& result) noexcept;

    /// Everything readable about the session right now.
    [[nodiscard]] CenterOutSnapshot snapshot() const noexcept;

    /// The configuration this session was started with.
    ///
    /// Default-constructed while ::CenterOutState::idle.
    [[nodiscard]] const CenterOut2DConfig& configuration() const noexcept
    {
        return config_;
    }

    /// Provenance identifier stamped on this session's records.
    [[nodiscard]] ParadigmId paradigm() const noexcept
    {
        return paradigm_;
    }

    /// Where the machine is.
    [[nodiscard]] CenterOutState state() const noexcept
    {
        return run_.state;
    }

    /// Whether the session has reached ::CenterOutState::complete.
    [[nodiscard]] bool complete() const noexcept
    {
        return run_.state == CenterOutState::complete;
    }

  private:
    /// The mutable half of a session.
    ///
    /// Split out so that step() can work on a copy and commit only on success:
    /// a step that fails partway leaves the machine exactly as it was, and
    /// copying this is cheap in a way that copying the configuration beside it
    /// would not be.
    struct Run;

    [[nodiscard]] ContractStatus emit_transition(Run& run, CenterOutStepResult& result,
                                                 ExperimentTimeNs time_ns, CenterOutState to,
                                                 CenterOutCause cause) const noexcept;
    [[nodiscard]] ContractStatus emit_event(Run& run, CenterOutStepResult& result,
                                            ExperimentTimeNs time_ns,
                                            ExperimentEventKind kind) const noexcept;
    [[nodiscard]] ContractStatus begin_trial(Run& run, CenterOutStepResult& result,
                                             ExperimentTimeNs time_ns) const noexcept;
    [[nodiscard]] ContractStatus decide_trial(Run& run, CenterOutStepResult& result,
                                              TrialOutcome outcome, CenterOutReason reason,
                                              CenterOutPhase phase,
                                              ExperimentTimeNs time_ns) const noexcept;
    [[nodiscard]] ContractStatus advance(Run& run, ExperimentTimeNs time_ns,
                                         const WorkspacePoint& cursor,
                                         CenterOutStepResult& result) const noexcept;
    [[nodiscard]] CenterOutSnapshot build_snapshot(const Run& run) const noexcept;

    struct Run
    {
        MonotonicTimeGate gate{};
        TrialCounter trials{};
        SequenceCounter sequence{};
        CenterOutState state{CenterOutState::idle};
        TrialOrdinal ordinal{};
        ExperimentTimeNs trial_started_ns{};
        TargetId outward_target{kUnsetTargetId};
        std::uint8_t outward_idx{};
        bool contained{};
        bool center_acquired{};
        ExperimentTimeNs center_acquired_ns{};
        TimeInterval leg{};
        TimeInterval hold{};
        TimeInterval dwell{};
        TrialOrdinal completed{};
        std::uint64_t successes{};
    };

    ParadigmId paradigm_{kUnsetParadigmId};
    CenterOut2DConfig config_{};
    Run run_{};
};
} // namespace neurale::experiments::center_out
