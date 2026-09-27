/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/experiments/center_out.h>
#include <neurale/experiments/contract.h>

/**
 * @file
 * @brief Center-Out target-directed guidance: a reference-velocity generator.
 *
 * The profile has explicit acceleration, deceleration, maximum-speed, and
 * target-precision parameters. Phase is an enumeration, per-call storage is
 * fixed-size, and elapsed time is expressed in integer nanoseconds. Guidance
 * parameters are not derived from rendered geometry or task deadlines.
 *
 * # What it is not
 *
 * It is not assistance. There is no blend coefficient in this header, no
 * orthogonal impedance, and no knowledge that either exists: guidance produces
 * one reference velocity, and orchestration is what feeds that velocity and a
 * decoded velocity into `neurale.experiments.assistance`. Center-Out is fully
 * usable without ever calling either --
 * the task state machine never calls guidance, and guidance never advances
 * the task state machine.
 *
 * It is also not the state machine. It decides no trial, holds no clock, owns no
 * trial ordinal, and cannot end a session. And it is not a simulation, a
 * decoder, an actuator, a device, a recorder, or a renderer.
 *
 * It does not move the cursor either. The caller owns integration and
 * presentation.
 *
 * # Units and frame
 *
 * Positions and the precision radius are in CenterOutGuidanceConfig's
 * GeometryUnit -- the same unit and the same 2D semantic
 * workspace as CenterOut2DLayout, with components in `(x, y)` order. Speed is
 * that unit per second and acceleration is that unit per second squared, so a
 * configuration in millimetres produces millimetres per second. There is no
 * second unit vocabulary and no conversion anywhere in this header;
 * validate_against(const CenterOutGuidanceConfig&, const CenterOut2DConfig&)
 * checks that guidance and the session it guides agree on the unit rather than
 * leaving two records each believing something.
 *
 * This contract is two-dimensional because the Center-Out workspace is 2D.
 *
 * # Time
 *
 * The step is an explicit elapsed duration in integer nanoseconds, never an
 * instant and never a call count. Guidance therefore has no clock discipline of
 * its own: it has nothing to gate, because it is never told what time it is.
 *
 * Because the profile integrates that step, a *different* partition of the same
 * elapsed time is not the same run -- it is a different sequence of decisions,
 * each made from the velocity its own step began with. That is a property of a
 * discrete integrator, stated rather than papered over. What is guaranteed is
 * that the same sequence of `(dt_ns, position)` observations produces the same
 * result whatever the wall-clock spacing between the calls, because nothing here
 * reads a clock or counts invocations.
 */
namespace neurale::experiments::center_out
{

/// What the guidance profile is doing towards the active target.
///
/// Every value is *derived* from the configuration, the retained velocity and
/// the observed geometry on each update; no branch anywhere reads a phase back,
/// so a corrupted phase cannot change a velocity.
///
/// There is no separate "at maximum speed" phase: the maximum-speed clamp is
/// applied inside ::speeding_up, and a
/// profile pinned at the clamp is still speeding up as far as the phase is
/// concerned.
enum class CenterOutGuidancePhase : std::uint8_t
{
    /// No update has been evaluated since the state was reset.
    idle = 0,
    /// Accelerating towards the target, or moving away from it and therefore
    /// accelerating back.
    speeding_up,
    /// Braking so as to stop at the target.
    slowing_down,
    /// Within the precision radius. The velocity is exactly zero.
    at_target,
};

/// Whether @p phase is one of the declared CenterOutGuidancePhase values.
[[nodiscard]] constexpr bool
center_out_guidance_phase_declared(CenterOutGuidancePhase phase) noexcept
{
    return static_cast<std::uint8_t>(phase) <=
           static_cast<std::uint8_t>(CenterOutGuidancePhase::at_target);
}

/// A velocity in the paradigm's 2D semantic workspace.
///
/// A distinct type from WorkspacePoint, which names a place. The two have the
/// same two `double` fields and mean different things, and a signature that
/// took a point where it wanted a velocity would be one substitution away from
/// compiling and being wrong.
struct WorkspaceVelocity
{
    /// Rate along the first workspace axis, in geometry units per second.
    double x{};
    /// Rate along the second workspace axis, in geometry units per second.
    double y{};
};

/// Immutable target-directed guidance configuration.
///
/// Nothing here is derived from presentation or other session configuration.
/// A caller supplies every value explicitly.
struct CenterOutGuidanceConfig
{
    /// Unit of the precision radius, and the base of the speed and acceleration
    /// units. Must match the guided session's CenterOut2DConfig::geometry_unit.
    GeometryUnit geometry_unit{GeometryUnit::unspecified};
    /// Largest speed the profile commands, in geometry units per second.
    ///
    /// Must be finite and strictly positive: a maximum of zero describes a
    /// reference mover that never moves, which is not guidance. Applied only
    /// while ::CenterOutGuidancePhase::speeding_up, and only when the speed
    /// strictly exceeds it -- see evaluate_guidance().
    double max_speed{};
    /// Rate the speed rises at while approaching, in units per second squared.
    ///
    /// Must be finite and strictly positive.
    double acceleration{};
    /// Rate the speed falls at while braking, in units per second squared.
    ///
    /// Separate from `acceleration` because a reference mover may brake at a
    /// different rate than it accelerates. Must be finite and strictly positive
    /// because the phase decision divides by it.
    double deceleration{};
    /// Radius around the target inside which the reference velocity is exactly
    /// zero, in geometry units.
    ///
    /// Must be finite and non-negative. Zero is admissible and means the profile only stops
    /// when the cursor is exactly on the target, which no continuous approach
    /// reaches; a session that wants guidance to settle gives it a positive
    /// radius.
    double precision{};
};

/// The retained half of a guidance run.
///
/// The velocity is both what the last update produced and what the next one
/// starts from -- one value, not two that could disagree.
///
/// Direction is not retained *separately*: the direction to the target is
/// recomputed from the observed position on every update, and the retained
/// velocity is projected onto it. That projection is what makes a large change
/// of heading -- an overshoot, a cursor that has been carried past the target --
/// start the approach again from rest instead of reversing at full speed.
struct CenterOutGuidanceState
{
    /// Target the retained velocity was built towards, or ::kUnsetTargetId.
    ///
    /// Guidance carries target identity so a target change resets retained
    /// velocity explicitly.
    TargetId target{kUnsetTargetId};
    /// The velocity the last update produced, in geometry units per second.
    WorkspaceVelocity vel{};
    /// What the last update was doing. Written, never read.
    CenterOutGuidancePhase phase{CenterOutGuidancePhase::idle};
};

/// What one guidance update produced.
///
/// A value with no ownership and fixed size, so an update allocates nothing.
struct CenterOutGuidanceSample
{
    /// Reference velocity, in the workspace frame and units above.
    ///
    /// Equal to `state.velocity` by construction: both are written from the same
    /// local in the same statement, so they are one truth read two ways rather
    /// than two that could drift. A test asserts the equality on every sample it
    /// produces.
    WorkspaceVelocity vel{};
    /// State after this update. Feed it back as the next update's `previous`.
    CenterOutGuidanceState state{};
    /// Distance from the observed position to the target.
    double distance{};
    /// Component of the *entering* velocity along the direction to the target.
    ///
    /// The quantity the whole phase decision is made from; reporting it makes
    /// that decision checkable from
    /// outside rather than inferable from the velocity it produced. Negative
    /// when the retained velocity pointed away from the target. Zero and not
    /// meaningful when the phase is ::CenterOutGuidancePhase::at_target, which
    /// returns before a direction exists.
    double speed_towards_target{};
    /// Whether the retained velocity was discarded because the target changed.
    ///
    /// False on the first update after a reset, when there was nothing to
    /// discard.
    bool retargeted{};
};

/// Validate a guidance configuration.
[[nodiscard]] ContractStatus validate(const CenterOutGuidanceConfig& config) noexcept;

/// Validate a guidance state.
///
/// Beyond each field, this rejects the two combinations that describe a run that
/// could not have happened: a state with no target that nonetheless carries a
/// velocity or a phase, and a state with a target whose phase is
/// ::CenterOutGuidancePhase::idle. Either is ContractStatus::outcome_invalid.
///
/// The retained velocity is *not* checked against a maximum speed. The profile
/// applies its clamp only while speeding up, so a braking run legally
/// carries more speed than the maximum until it has slowed below it, and a check
/// here would reject states the generator itself produces.
[[nodiscard]] ContractStatus validate(const CenterOutGuidanceState& state) noexcept;

/// Validate that @p guidance can guide a session configured by @p config.
///
/// Two cross-record checks, neither of which either record can make alone:
///
/// - the geometry units must be equal, because guidance measures distance to a
///   target whose coordinates come from the other record;
/// - the precision radius must be no larger than the smaller acceptance
///   half-extent less the cursor extent, so that arriving implies being
///   contained. With `distance <= precision` and that bound, `|dx|` and `|dy|`
///   are each at most `half_extent - cursor extent`, which is exactly
///   contains_cursor(). Without it, guidance can settle at a standstill outside
///   the acceptance region and the leg times out with the reference velocity
///   reporting that it had arrived.
/// @return ContractStatus::ok, the first failure of either record's own
///         validator, ContractStatus::outcome_invalid when the units disagree,
///         or ContractStatus::parameter_out_of_range when the precision radius
///         can leave the acceptance region.
[[nodiscard]] ContractStatus validate_against(const CenterOutGuidanceConfig& guidance,
                                              const CenterOut2DConfig& config) noexcept;

/// Evaluate one guidance update.
///
/// The whole production core. Deterministic, side-effect free, bounded, and
/// allocation-free: given the same arguments it writes the same bytes, and it
/// modifies none of them.
///
/// The profile is:
///
/// ```text
/// v      = previous velocity, or (0, 0) when the target identifier changed
/// offset = target - position
/// d      = sqrt(offset.x^2 + offset.y^2)
///
/// d <= precision  ->  at_target: velocity = (0, 0), and return
///
/// u    = offset / d                       (unit direction)
/// s    = dot(v, u)                        (speed towards the target)
///
/// s <= 0                                  -> speeding_up
/// otherwise  d / s <= s / deceleration    -> slowing_down
///            d / s >  s / deceleration    -> speeding_up
///
/// w = max(s, 0) * u                       (the entering velocity, projected)
///
/// slowing_down: w -= deceleration * u * dt
///               velocity = max(dot(w, u), 0) * u
///
/// speeding_up:  w += acceleration * u * dt
///               velocity = |w| > max_speed ? max_speed * w / |w| : w
/// ```
///
/// Five details define the numerical behavior:
///
/// - **the braking test is not the minimum-time braking point.** It compares the
///   time to reach the target at the current speed against the time to brake to
///   rest, so it begins braking at `d <= s^2 / deceleration` -- twice the
///   distance a constant-deceleration stop actually needs -- and it measures to
///   the target itself, not to the edge of the precision region. The profile can
///   therefore alternate between braking and re-accelerating on successive
///   steps; the precision region prevents target oscillation;
/// - **the projection is what handles overshoot.** A retained velocity that
///   points away from the target has a negative component along it, which is
///   clamped to zero, so the approach begins again from rest rather than
///   reversing at speed. Nothing else in the profile looks at heading;
/// - **the maximum speed is applied only while speeding up**, and only when the
///   speed strictly exceeds it. A braking run above the maximum is left alone,
///   because it is already on its way down;
/// - **braking never runs the speed below zero**, but a profile standing still
///   outside the precision region is still ::CenterOutGuidancePhase::slowing_down
///   rather than arrived;
/// - **the distance is `sqrt(dx*dx + dy*dy)`, not `std::hypot`.** `hypot` avoids
///   the intermediate overflow that this rejects below, but is not required to
///   be correctly rounded, so two platforms could disagree on a reference
///   velocity. Square root is correctly rounded by IEEE 754.
///
/// Inside the precision region the velocity is exactly `(0, 0)` and the retained
/// velocity is dropped rather than decayed, so a cursor that later drifts out is
/// accelerated from rest.
///
/// @param config Guidance configuration. Never modified.
/// @param previous Retained state. Never modified; use a default-constructed
///        value for the first update after a reset.
/// @param target The target being guided towards, identifier and position.
/// @param pos Observed cursor position.
/// @param dt_ns Elapsed time since the previous update. Zero is admissible: the
///        speed is then unchanged, but the direction is recomputed and the
///        retained velocity is still projected onto it.
/// @param sample Receives everything the update produced on ContractStatus::ok,
///        and is left unmodified otherwise.
/// @return ContractStatus::ok; the first failure of validate(const
///         CenterOutGuidanceConfig&), validate(const CenterOutGuidanceState&) or
///         validate(const TargetPlacement&); ContractStatus::value_not_finite
///         when @p pos is not finite, or when an intermediate overflows
///         although every input was finite.
[[nodiscard]] ContractStatus evaluate_guidance(const CenterOutGuidanceConfig& config,
                                               const CenterOutGuidanceState& previous,
                                               const TargetPlacement& target,
                                               const WorkspacePoint& pos, DurationNs dt_ns,
                                               CenterOutGuidanceSample& sample) noexcept;

/// A configured guidance generator that retains its own state.
///
/// A thin holder over evaluate_guidance(), which stays the single
/// implementation: this class validates once at configure() and then carries the
/// configuration and the retained velocity between updates, so orchestration
/// does not have to. Value semantics -- copyable, no pointer, no owned storage --
/// mean copying it copies a run.
class CenterOutGuidance
{
  public:
    /// Construct an unconfigured generator.
    CenterOutGuidance() noexcept = default;

    /// Capture @p config and reset the retained state.
    ///
    /// @return ContractStatus::ok, or the first failure of validate(const
    ///         CenterOutGuidanceConfig&), in which case the generator is left
    ///         exactly as it was.
    [[nodiscard]] ContractStatus configure(const CenterOutGuidanceConfig& config) noexcept;

    /// Discard the retained state, keeping the configuration.
    ///
    /// This is the explicit reset an experiment reset performs. The next update
    /// therefore accelerates from rest, and reports `retargeted == false`
    /// because there was no velocity left to discard.
    void reset() noexcept;

    /// Evaluate one update against the retained state.
    ///
    /// A target different from the retained one is itself an explicit reset: the
    /// retained velocity is discarded before the update rather than carried
    /// across, and CenterOutGuidanceSample::retargeted reports that it happened.
    /// Velocity belongs to an approach, and carrying it into a new one would
    /// have the reference mover arrive at a target it had never set off for.
    /// Nothing is silent about it, and a caller that prefers to say so itself
    /// calls reset() first, which produces the same state.
    /// @return As evaluate_guidance(), and additionally
    ///         ContractStatus::not_running before configure() has succeeded. A
    ///         failed update leaves the generator unmodified.
    [[nodiscard]] ContractStatus update(const TargetPlacement& target, const WorkspacePoint& pos,
                                        DurationNs dt_ns, CenterOutGuidanceSample& sample) noexcept;

    /// The captured configuration. Default-constructed while unconfigured.
    [[nodiscard]] const CenterOutGuidanceConfig& configuration() const noexcept
    {
        return config_;
    }

    /// The retained state.
    [[nodiscard]] const CenterOutGuidanceState& state() const noexcept
    {
        return state_;
    }

    /// Whether configure() has succeeded.
    [[nodiscard]] bool configured() const noexcept
    {
        return configured_;
    }

  private:
    CenterOutGuidanceConfig config_{};
    CenterOutGuidanceState state_{};
    bool configured_{};
};

} // namespace neurale::experiments::center_out
