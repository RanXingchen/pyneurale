/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/center_out_guidance.h>

#include <cmath>

namespace neurale::experiments::center_out
{
namespace
{

[[nodiscard]] ContractStatus check_finite(double value) noexcept
{
    return std::isfinite(value) ? ContractStatus::ok : ContractStatus::value_not_finite;
}

[[nodiscard]] ContractStatus check_positive(double value) noexcept
{
    if (const ContractStatus status = check_finite(value); status != ContractStatus::ok)
        return status;
    return value > 0.0 ? ContractStatus::ok : ContractStatus::parameter_out_of_range;
}

[[nodiscard]] ContractStatus check_velocity(const WorkspaceVelocity& vel) noexcept
{
    if (const ContractStatus status = check_finite(vel.x); status != ContractStatus::ok)
        return status;
    return check_finite(vel.y);
}

// Nanoseconds to seconds. The division is by a power of ten rather than a
// multiplication by 1e-9 so that a whole number of nanoseconds converts as
// exactly as a correctly-rounded division can: 1e-9 is not representable, and
// 1'000'000'000 is.
[[nodiscard]] double seconds_of(DurationNs dt_ns) noexcept
{
    return static_cast<double>(dt_ns) / 1'000'000'000.0;
}

[[nodiscard]] double dot(double ax, double ay, double bx, double by) noexcept
{
    return ax * bx + ay * by;
}

} // namespace

ContractStatus validate(const CenterOutGuidanceConfig& config) noexcept
{
    if (!geometry_unit_declared(config.geometry_unit))
        return ContractStatus::enum_undeclared;
    if (config.geometry_unit == GeometryUnit::unspecified)
        return ContractStatus::identity_missing;

    if (const ContractStatus status = check_positive(config.max_speed);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_positive(config.acceleration);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_positive(config.deceleration);
        status != ContractStatus::ok)
        return status;

    if (const ContractStatus status = check_finite(config.precision); status != ContractStatus::ok)
        return status;
    // Zero is admissible: it names a profile that only stops exactly on the
    // target. Negative is not -- a radius is a distance, and a negative one
    // would make the arrival test unsatisfiable while looking like a setting.
    if (config.precision < 0.0)
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

ContractStatus validate(const CenterOutGuidanceState& state) noexcept
{
    if (!center_out_guidance_phase_declared(state.phase))
        return ContractStatus::enum_undeclared;
    if (const ContractStatus status = check_velocity(state.vel); status != ContractStatus::ok)
        return status;

    // Each field above is fine on its own; these two pairs are not. A state with
    // no target has never been updated, so it can carry neither a velocity nor a
    // phase; a state with a target has been updated, so its phase cannot still
    // say that it has not.
    if (state.target == kUnsetTargetId &&
        (state.vel.x != 0.0 || state.vel.y != 0.0 || state.phase != CenterOutGuidancePhase::idle))
        return ContractStatus::outcome_invalid;
    if (state.target != kUnsetTargetId && state.phase == CenterOutGuidancePhase::idle)
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate_against(const CenterOutGuidanceConfig& guidance,
                                const CenterOut2DConfig& config) noexcept
{
    if (const ContractStatus status = validate(guidance); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;

    if (guidance.geometry_unit != config.geometry_unit)
        return ContractStatus::outcome_invalid;

    // Arriving must imply being contained. validate(const CenterOut2DConfig&)
    // has already established that the cursor extent does not exceed either
    // half-extent, so both margins below are non-negative and the comparison is
    // between two distances rather than against a negative number.
    const double margin_x = config.acceptance.half_extent_x - config.cursor.extent;
    const double margin_y = config.acceptance.half_extent_y - config.cursor.extent;
    const double margin = margin_x < margin_y ? margin_x : margin_y;
    if (guidance.precision > margin)
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

ContractStatus evaluate_guidance(const CenterOutGuidanceConfig& config,
                                 const CenterOutGuidanceState& previous,
                                 const TargetPlacement& target, const WorkspacePoint& pos,
                                 DurationNs dt_ns, CenterOutGuidanceSample& sample) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(previous); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(target); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_finite(pos.x); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_finite(pos.y); status != ContractStatus::ok)
        return status;

    // A different target is a reset. The retained velocity belongs to an
    // approach that is over, and carrying it would have the reference mover
    // arrive somewhere it never set off for. An unset previous target is not a
    // change: there was nothing to discard, which is what `retargeted` reports.
    const bool retargeted = previous.target != kUnsetTargetId && previous.target != target.id;
    const WorkspaceVelocity entry = retargeted ? WorkspaceVelocity{0.0, 0.0} : previous.vel;

    CenterOutGuidanceSample produced{};
    produced.retargeted = retargeted;
    produced.state.target = target.id;

    const double offset_x = target.pos.x - pos.x;
    const double offset_y = target.pos.y - pos.y;
    const double distance = std::sqrt(dot(offset_x, offset_y, offset_x, offset_y));
    // Every input was finite, but a squared separation need not be. Reported
    // rather than answered with an infinite distance, which would look like a
    // target that is simply far away.
    if (const ContractStatus status = check_finite(distance); status != ContractStatus::ok)
        return status;
    produced.distance = distance;

    if (distance <= config.precision)
    {
        // Within precision of the target, velocity is dropped to zero rather
        // than decayed, to avoid oscillation at the target; `speed_towards_target`
        // then has nothing to report and stays zero.
        produced.state.phase = CenterOutGuidancePhase::at_target;
        produced.state.vel = WorkspaceVelocity{0.0, 0.0};
        produced.vel = produced.state.vel;
        sample = produced;
        return ContractStatus::ok;
    }

    const double unit_x = offset_x / distance;
    const double unit_y = offset_y / distance;

    // The component of the entering velocity along the direction to the target.
    // Negative when the retained velocity points away from it, which is the only
    // thing in the profile that knows about heading at all.
    const double speed_towards_target = dot(entry.x, entry.y, unit_x, unit_y);
    produced.speed_towards_target = speed_towards_target;

    CenterOutGuidancePhase phase = CenterOutGuidancePhase::speeding_up;
    if (speed_towards_target > 0.0)
    {
        // Compared as two times rather than folded into the algebraically equal
        // `distance * deceleration <= s * s`: the two group their divisions
        // differently, and the approach chatters between the branches on
        // successive steps, so a boundary that moved by an ulp would move which
        // step chatters where.
        const double time_to_target = distance / speed_towards_target;
        const double time_to_slow_down = std::abs(speed_towards_target / config.deceleration);
        if (time_to_slow_down >= time_to_target)
            phase = CenterOutGuidancePhase::slowing_down;
    }

    // The entering velocity, projected onto the direction and clamped at rest.
    // This is where an overshoot loses its speed instead of reversing at it.
    const double projected = speed_towards_target > 0.0 ? speed_towards_target : 0.0;
    double work_x = projected * unit_x;
    double work_y = projected * unit_y;

    const double dt = seconds_of(dt_ns);
    if (phase == CenterOutGuidancePhase::slowing_down)
    {
        work_x += -config.deceleration * unit_x * dt;
        work_y += -config.deceleration * unit_y * dt;
        const double braked = dot(work_x, work_y, unit_x, unit_y);
        const double speed = braked > 0.0 ? braked : 0.0;
        produced.state.vel = WorkspaceVelocity{speed * unit_x, speed * unit_y};
    }
    else
    {
        work_x += config.acceleration * unit_x * dt;
        work_y += config.acceleration * unit_y * dt;
        // The magnitude is taken from the vector rather than from the scalar it
        // was built out of -- the two differ in the last bit. No clamp is
        // applied while braking, because a braking run above the maximum is
        // already on its way down.
        const double speed = std::sqrt(dot(work_x, work_y, work_x, work_y));
        if (speed > config.max_speed)
            produced.state.vel = WorkspaceVelocity{config.max_speed * work_x / speed,
                                                   config.max_speed * work_y / speed};
        else
            produced.state.vel = WorkspaceVelocity{work_x, work_y};
    }

    if (const ContractStatus status = check_velocity(produced.state.vel);
        status != ContractStatus::ok)
        return status;

    produced.state.phase = phase;
    produced.vel = produced.state.vel;
    sample = produced;
    return ContractStatus::ok;
}

ContractStatus CenterOutGuidance::configure(const CenterOutGuidanceConfig& config) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    config_ = config;
    state_ = CenterOutGuidanceState{};
    configured_ = true;
    return ContractStatus::ok;
}

void CenterOutGuidance::reset() noexcept
{
    state_ = CenterOutGuidanceState{};
}

ContractStatus CenterOutGuidance::update(const TargetPlacement& target, const WorkspacePoint& pos,
                                         DurationNs dt_ns, CenterOutGuidanceSample& sample) noexcept
{
    if (!configured_)
        return ContractStatus::not_running;

    CenterOutGuidanceSample produced{};
    if (const ContractStatus status =
            evaluate_guidance(config_, state_, target, pos, dt_ns, produced);
        status != ContractStatus::ok)
        return status;
    // Committed only on success, so a rejected update leaves the run exactly
    // where it was rather than half-advanced.
    state_ = produced.state;
    sample = produced;
    return ContractStatus::ok;
}

} // namespace neurale::experiments::center_out
