/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/center_out.h>
#include <neurale/experiments/center_out_guidance.h>
#include <neurale/experiments/contract.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#include "allocation_counter.h"
#include "check_counts.h"

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::center_out;

// Guidance is a value, like the machine beside it. A generator that stopped
// being trivially copyable would have grown owning storage.
static_assert(std::is_trivially_copyable_v<CenterOutGuidance>);
static_assert(std::is_trivially_copyable_v<CenterOutGuidanceConfig>);
static_assert(std::is_trivially_copyable_v<CenterOutGuidanceState>);
static_assert(std::is_trivially_copyable_v<CenterOutGuidanceSample>);

int failures = 0;

// One sixty-fourth of a second. Every duration outside the differential test is
// a dyadic fraction of a second so that `dt_ns / 1e9` and its products are
// exact, and the profile is checked against arithmetic rather than against a
// tolerance that would hide a wrong coefficient.
constexpr DurationNs kStepNs = 15'625'000;
constexpr double kStepSeconds = 0.015625;

constexpr double kMaxSpeed = 100.0;
constexpr double kAcceleration = 200.0; // 3.125 per step.
constexpr double kDeceleration = 400.0; // 6.25 per step.
constexpr double kPrecision = 2.0;

constexpr TargetId kTarget = 5;
constexpr TargetId kOtherTarget = 6;

CenterOutGuidanceConfig make_config() noexcept
{
    return CenterOutGuidanceConfig{GeometryUnit::millimetres, kMaxSpeed, kAcceleration,
                                   kDeceleration, kPrecision};
}

constexpr TargetPlacement kEast{kTarget, WorkspacePoint{100.0, 0.0}};
constexpr TargetPlacement kNorth{kTarget, WorkspacePoint{0.0, 100.0}};
constexpr WorkspacePoint kOrigin{0.0, 0.0};

CenterOutGuidanceState moving(double vx, double vy, CenterOutGuidancePhase phase,
                              TargetId target = kTarget) noexcept
{
    return CenterOutGuidanceState{target, WorkspaceVelocity{vx, vy}, phase};
}

// Every sample the file produces goes through here: the reference velocity and
// the velocity carried in the state it hands back are one truth read two ways,
// and nothing may ever make them disagree.
bool consistent(const CenterOutGuidanceSample& sample) noexcept
{
    return sample.vel.x == sample.state.vel.x && sample.vel.y == sample.state.vel.y;
}

// ---------------------------------------------------------------------------

int check_reference_trajectory()
{
    const TargetPlacement target{kTarget, WorkspacePoint{60.0, 80.0}};
    CenterOutGuidance generator{};
    CHECK(generator.configure(make_config()) == ContractStatus::ok);

    WorkspacePoint cursor{};
    double speed = 0.0;
    bool clamped = false;
    bool braked = false;
    bool chattered = false;
    bool arrived = false;
    CenterOutGuidancePhase previous_phase = CenterOutGuidancePhase::idle;
    for (std::size_t step = 0; step < 120; ++step)
    {
        const double remaining = std::hypot(target.pos.x - cursor.x, target.pos.y - cursor.y);
        CenterOutGuidancePhase expected_phase = CenterOutGuidancePhase::speeding_up;
        if (remaining <= kPrecision)
        {
            speed = 0.0;
            expected_phase = CenterOutGuidancePhase::at_target;
            arrived = true;
        }
        else if (speed > 0.0 && remaining / speed <= speed / kDeceleration)
        {
            speed =
                speed > kDeceleration * kStepSeconds ? speed - kDeceleration * kStepSeconds : 0.0;
            expected_phase = CenterOutGuidancePhase::slowing_down;
            braked = true;
        }
        else
        {
            speed += kAcceleration * kStepSeconds;
            if (speed >= kMaxSpeed)
            {
                speed = kMaxSpeed;
                clamped = true;
            }
        }

        CenterOutGuidanceSample sample{};
        CHECK(generator.update(target, cursor, kStepNs, sample) == ContractStatus::ok);
        CHECK(consistent(sample));
        CHECK(sample.state.phase == expected_phase);
        CHECK(std::abs(sample.vel.x - speed * 0.6) <= 1e-9);
        CHECK(std::abs(sample.vel.y - speed * 0.8) <= 1e-9);
        if (previous_phase == CenterOutGuidancePhase::slowing_down &&
            sample.state.phase == CenterOutGuidancePhase::speeding_up)
            chattered = true;
        previous_phase = sample.state.phase;

        cursor.x += sample.vel.x * kStepSeconds;
        cursor.y += sample.vel.y * kStepSeconds;
    }
    CHECK(clamped);
    CHECK(braked);
    CHECK(chattered);
    CHECK(arrived);
    CHECK(generator.state().phase == CenterOutGuidancePhase::at_target);
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------

int check_validation()
{
    CHECK(validate(CenterOutGuidanceConfig{}) == ContractStatus::identity_missing);

    CenterOutGuidanceConfig undeclared = make_config();
    undeclared.geometry_unit = static_cast<GeometryUnit>(200);
    CHECK(validate(undeclared) == ContractStatus::enum_undeclared);

    CenterOutGuidanceConfig config = make_config();
    config.max_speed = 0.0;
    CHECK(validate(config) == ContractStatus::parameter_out_of_range);
    config = make_config();
    config.max_speed = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(config) == ContractStatus::value_not_finite);
    config = make_config();
    config.acceleration = -1.0;
    CHECK(validate(config) == ContractStatus::parameter_out_of_range);
    config = make_config();
    config.deceleration = 0.0;
    CHECK(validate(config) == ContractStatus::parameter_out_of_range);
    config = make_config();
    config.precision = -0.5;
    CHECK(validate(config) == ContractStatus::parameter_out_of_range);
    config = make_config();
    config.precision = 0.0;
    CHECK(validate(config) == ContractStatus::ok);

    CHECK(validate(CenterOutGuidanceState{}) == ContractStatus::ok);
    CHECK(validate(moving(10.0, -3.0, CenterOutGuidancePhase::speeding_up)) == ContractStatus::ok);
    // A braking run above the configured maximum is legal state: the clamp is
    // only applied while speeding up.
    CHECK(validate(moving(1e6, 0.0, CenterOutGuidancePhase::slowing_down)) == ContractStatus::ok);
    CHECK(validate(moving(std::numeric_limits<double>::infinity(), 0.0,
                          CenterOutGuidancePhase::speeding_up)) ==
          ContractStatus::value_not_finite);
    CHECK(validate(CenterOutGuidanceState{kTarget, WorkspaceVelocity{1.0, 0.0},
                                          static_cast<CenterOutGuidancePhase>(9)}) ==
          ContractStatus::enum_undeclared);
    // Legal field by field, impossible together.
    CHECK(validate(moving(1.0, 0.0, CenterOutGuidancePhase::idle, kUnsetTargetId)) ==
          ContractStatus::outcome_invalid);
    CHECK(validate(moving(0.0, 0.0, CenterOutGuidancePhase::speeding_up, kUnsetTargetId)) ==
          ContractStatus::outcome_invalid);
    CHECK(validate(moving(0.0, 0.0, CenterOutGuidancePhase::idle, kTarget)) ==
          ContractStatus::outcome_invalid);
    return failures == 0 ? 0 : 1;
}

CenterOut2DConfig make_session() noexcept
{
    RadialLayoutRequest request{};
    request.radius = 100.0;
    request.count = 8;
    request.center_id = 1;
    for (std::size_t i = 0; i < 8; ++i)
    {
        request.ids[i] = static_cast<TargetId>(i + 2);
        request.spokes[i] = static_cast<std::uint32_t>(i);
    }

    CenterOut2DConfig config{};
    config.geometry_unit = GeometryUnit::millimetres;
    if (build_radial_layout(request, config.layout) != ContractStatus::ok)
        std::cerr << "FAILED: could not build the test layout\n";
    config.acceptance = AcceptanceRegion{5.0, 4.0};
    config.cursor = CursorGeometry{1.0};
    config.movement_timeout = PhaseDurations{1'000'000'000, 1'000'000'000};
    config.hold_ns = 0;
    config.selection = TargetSelectionPolicy::repeat_until_success;
    config.seed = 0xBEEF;
    return config;
}

int check_cross_validation()
{
    const CenterOut2DConfig session = make_session();

    // The smaller margin is 4.0 - 1.0 = 3.0. A precision radius at it is
    // admissible, because arriving then still implies containment.
    CenterOutGuidanceConfig guidance = make_config();
    guidance.precision = 3.0;
    CHECK(validate_against(guidance, session) == ContractStatus::ok);
    guidance.precision = 3.0 + 1.0 / 1024.0;
    CHECK(validate_against(guidance, session) == ContractStatus::parameter_out_of_range);

    guidance = make_config();
    guidance.geometry_unit = GeometryUnit::metres;
    CHECK(validate_against(guidance, session) == ContractStatus::outcome_invalid);

    // The implication the bound exists for: at the admissible radius, every
    // point at exactly that distance is contained.
    guidance = make_config();
    guidance.precision = 3.0;
    CHECK(validate_against(guidance, session) == ContractStatus::ok);
    const WorkspacePoint centre = session.layout.center.pos;
    for (int step = 0; step <= 16; ++step)
    {
        const double angle = 2.0 * std::numbers::pi * static_cast<double>(step) / 16.0;
        const WorkspacePoint probe{centre.x + guidance.precision * std::cos(angle),
                                   centre.y + guidance.precision * std::sin(angle)};
        bool inside = false;
        CHECK(contains_cursor(session.acceptance, session.cursor, centre, probe, inside) ==
              ContractStatus::ok);
        CHECK(inside);
    }

    // Each record's own validator runs first.
    CenterOut2DConfig broken = session;
    broken.selection = TargetSelectionPolicy::unspecified;
    CHECK(validate_against(make_config(), broken) == ContractStatus::identity_missing);
    return failures == 0 ? 0 : 1;
}

int check_profile()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};

    // From rest, far from the target: one step of acceleration, pointed east.
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kEast, kOrigin, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(consistent(sample));
    CHECK(sample.state.phase == CenterOutGuidancePhase::speeding_up);
    CHECK(sample.vel.x == kAcceleration * kStepSeconds);
    CHECK(sample.vel.y == 0.0);
    CHECK(sample.state.target == kTarget);
    CHECK(sample.distance == 100.0);
    CHECK(sample.speed_towards_target == 0.0);
    CHECK(!sample.retargeted);

    // The same step northwards: the direction comes from the observation, so the
    // magnitude is identical and only the axis changes.
    CenterOutGuidanceSample north{};
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kNorth, kOrigin, kStepNs, north) ==
          ContractStatus::ok);
    CHECK(north.vel.x == 0.0);
    CHECK(north.vel.y == sample.vel.x);

    // A 3-4-5 offset: the velocity is the speed along the unit offset.
    CenterOutGuidanceSample diagonal{};
    const TargetPlacement close{kTarget, WorkspacePoint{3.0, 4.0}};
    CHECK(evaluate_guidance(config, moving(3.125, 0.0, CenterOutGuidancePhase::speeding_up), close,
                            kOrigin, 0, diagonal) == ContractStatus::ok);
    CHECK(diagonal.distance == 5.0);
    // The retained velocity pointed east; projected onto (0.6, 0.8) it keeps
    // three fifths of itself, and a zero step adds nothing to it.
    CHECK(diagonal.speed_towards_target == 3.125 * 0.6);
    CHECK(std::abs(diagonal.vel.x - 3.125 * 0.6 * 0.6) <= 1e-15);
    CHECK(std::abs(diagonal.vel.y - 3.125 * 0.6 * 0.8) <= 1e-15);

    // Accelerating in a straight line: the speed rises by one step's worth each
    // time and is pinned at the maximum once it arrives there. The clamp is a
    // clamp, not a phase.
    CenterOutGuidanceState state{};
    for (int step = 1; step <= 40; ++step)
    {
        CHECK(evaluate_guidance(config, state, kEast, kOrigin, kStepNs, sample) ==
              ContractStatus::ok);
        state = sample.state;
        CHECK(state.phase == CenterOutGuidancePhase::speeding_up);
        const double expected = kAcceleration * kStepSeconds * static_cast<double>(step);
        CHECK(state.vel.x == (expected < kMaxSpeed ? expected : kMaxSpeed));
    }

    // A zero step re-projects and leaves the speed alone.
    CenterOutGuidanceSample still{};
    CHECK(evaluate_guidance(config, state, kEast, kOrigin, 0, still) == ContractStatus::ok);
    CHECK(still.vel.x == kMaxSpeed);
    CHECK(still.state.phase == CenterOutGuidancePhase::speeding_up);
    return failures == 0 ? 0 : 1;
}

int check_braking_decision()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};

    // The test is `distance / speed <= speed / deceleration`. At speed 100 and
    // deceleration 400 that is a distance of 100 * 100 / 400 = 25 -- twice what
    // a constant-deceleration stop needs, and measured to the target itself
    // rather than to the edge of the precision region.
    const CenterOutGuidanceState fast = moving(kMaxSpeed, 0.0, CenterOutGuidancePhase::speeding_up);
    CHECK(evaluate_guidance(config, fast, kEast, WorkspacePoint{75.0, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.distance == 25.0);
    CHECK(sample.speed_towards_target == kMaxSpeed);
    CHECK(sample.state.phase == CenterOutGuidancePhase::slowing_down);
    CHECK(sample.vel.x == kMaxSpeed - kDeceleration * kStepSeconds);

    // A quarter of a millimetre further out it is still speeding up, and the
    // clamp holds it at the maximum.
    CHECK(evaluate_guidance(config, fast, kEast, WorkspacePoint{74.75, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.distance == 25.25);
    CHECK(sample.state.phase == CenterOutGuidancePhase::speeding_up);
    CHECK(sample.vel.x == kMaxSpeed);

    // The comparison is closed on the braking side: exactly equal times brake.
    // At speed 50 the switch distance is 50 * 50 / 400 = 6.25.
    const CenterOutGuidanceState half = moving(50.0, 0.0, CenterOutGuidancePhase::speeding_up);
    CHECK(evaluate_guidance(config, half, kEast, WorkspacePoint{93.75, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.distance == 6.25);
    CHECK(sample.state.phase == CenterOutGuidancePhase::slowing_down);

    // Braking never runs the speed below zero, and a profile standing still
    // outside the precision region is still braking, not arrived. Reaching that
    // at all needs a precision radius smaller than the switch distance the speed
    // implies -- at speed 3.125 the profile only brakes within
    // 3.125 * 3.125 / 400 = 0.0244 of the target, which a radius of two
    // millimetres swallows whole. That the two thresholds can hide each other is
    // the recovered behaviour, not a test artefact.
    CenterOutGuidanceConfig exact = make_config();
    exact.precision = 0.0;
    const CenterOutGuidanceState crawling =
        moving(kAcceleration * kStepSeconds, 0.0, CenterOutGuidancePhase::slowing_down);
    CHECK(evaluate_guidance(exact, crawling, kEast, WorkspacePoint{99.98, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.state.phase == CenterOutGuidancePhase::slowing_down);
    CHECK(sample.vel.x == 0.0);
    CHECK(sample.vel.y == 0.0);
    // With the ordinary radius the same geometry is simply arrival.
    CHECK(evaluate_guidance(config, crawling, kEast, WorkspacePoint{99.98, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.state.phase == CenterOutGuidancePhase::at_target);

    // No maximum-speed clamp while braking: a run above the maximum is already
    // on its way down, and one step of braking is all that happens to it.
    CHECK(evaluate_guidance(config, moving(1000.0, 0.0, CenterOutGuidancePhase::slowing_down),
                            kEast, kOrigin, kStepNs, sample) == ContractStatus::ok);
    CHECK(sample.state.phase == CenterOutGuidancePhase::slowing_down);
    CHECK(sample.vel.x == 1000.0 - kDeceleration * kStepSeconds);

    // The two rates are independent.
    CenterOutGuidanceConfig gentle = make_config();
    gentle.deceleration = 50.0;
    CHECK(evaluate_guidance(gentle, fast, kEast, WorkspacePoint{75.0, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.vel.x == kMaxSpeed - 50.0 * kStepSeconds);
    return failures == 0 ? 0 : 1;
}

int check_projection_and_overshoot()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};

    // Carried past the target: the retained velocity now points away from it, so
    // the projection is negative, it is clamped to rest, and the approach begins
    // again rather than reversing at speed.
    const CenterOutGuidanceState fast = moving(kMaxSpeed, 0.0, CenterOutGuidancePhase::speeding_up);
    CHECK(evaluate_guidance(config, fast, kEast, WorkspacePoint{200.0, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.speed_towards_target == -kMaxSpeed);
    CHECK(sample.state.phase == CenterOutGuidancePhase::speeding_up);
    CHECK(sample.vel.x == -kAcceleration * kStepSeconds);
    CHECK(sample.vel.y == 0.0);

    // Exactly sideways is the same story: no component along the direction
    // survives, so the profile starts from rest.
    CHECK(evaluate_guidance(config, moving(0.0, kMaxSpeed, CenterOutGuidancePhase::speeding_up),
                            kEast, kOrigin, kStepNs, sample) == ContractStatus::ok);
    CHECK(sample.speed_towards_target == 0.0);
    CHECK(sample.vel.x == kAcceleration * kStepSeconds);
    CHECK(sample.vel.y == 0.0);

    // The perpendicular part is discarded rather than carried: after the
    // projection nothing off the direction survives at all.
    CHECK(evaluate_guidance(config, moving(10.0, 40.0, CenterOutGuidancePhase::speeding_up), kEast,
                            kOrigin, 0, sample) == ContractStatus::ok);
    CHECK(sample.speed_towards_target == 10.0);
    CHECK(sample.vel.x == 10.0);
    CHECK(sample.vel.y == 0.0);
    return failures == 0 ? 0 : 1;
}

int check_precision_region()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};
    const CenterOutGuidanceState fast = moving(kMaxSpeed, 0.0, CenterOutGuidancePhase::speeding_up);

    // The region is closed, as the acceptance region is, and direction never
    // enters the arrival test: past the target counts too.
    for (const double x : {98.0, 99.0, 100.0, 101.0, 102.0})
    {
        CHECK(evaluate_guidance(config, fast, kEast, WorkspacePoint{x, 0.0}, kStepNs, sample) ==
              ContractStatus::ok);
        CHECK(consistent(sample));
        CHECK(sample.state.phase == CenterOutGuidancePhase::at_target);
        CHECK(sample.vel.x == 0.0);
        CHECK(sample.vel.y == 0.0);
        CHECK(sample.speed_towards_target == 0.0);
    }
    CHECK(evaluate_guidance(config, fast, kEast, WorkspacePoint{97.5, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.state.phase != CenterOutGuidancePhase::at_target);

    // Having arrived, a cursor that drifts back out starts from rest.
    CHECK(evaluate_guidance(config, fast, kEast, WorkspacePoint{99.0, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CenterOutGuidanceSample resumed{};
    CHECK(evaluate_guidance(config, sample.state, kEast, kOrigin, kStepNs, resumed) ==
          ContractStatus::ok);
    CHECK(resumed.state.phase == CenterOutGuidancePhase::speeding_up);
    CHECK(resumed.vel.x == kAcceleration * kStepSeconds);

    // With a zero radius only the target itself is arrival.
    CenterOutGuidanceConfig exact = make_config();
    exact.precision = 0.0;
    CHECK(evaluate_guidance(exact, fast, kEast, kEast.pos, kStepNs, sample) == ContractStatus::ok);
    CHECK(sample.state.phase == CenterOutGuidancePhase::at_target);
    CHECK(evaluate_guidance(exact, fast, kEast, WorkspacePoint{99.9999, 0.0}, kStepNs, sample) ==
          ContractStatus::ok);
    CHECK(sample.state.phase == CenterOutGuidancePhase::slowing_down);
    return failures == 0 ? 0 : 1;
}

int check_retarget_and_reset()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};
    const CenterOutGuidanceState fast = moving(kMaxSpeed, 0.0, CenterOutGuidancePhase::speeding_up);

    // A different target discards the retained velocity and says so. Without
    // that, the projection onto a perpendicular direction would have kept
    // nothing anyway -- so the case that matters is a target the old velocity
    // still points towards.
    const TargetPlacement other{kOtherTarget, WorkspacePoint{200.0, 0.0}};
    CHECK(evaluate_guidance(config, fast, other, kOrigin, kStepNs, sample) == ContractStatus::ok);
    CHECK(sample.retargeted);
    CHECK(sample.speed_towards_target == 0.0);
    CHECK(sample.state.target == kOtherTarget);
    CHECK(sample.vel.x == kAcceleration * kStepSeconds);

    // The same identifier at a different place is not a retarget.
    const TargetPlacement moved{kTarget, WorkspacePoint{200.0, 0.0}};
    CHECK(evaluate_guidance(config, fast, moved, kOrigin, kStepNs, sample) == ContractStatus::ok);
    CHECK(!sample.retargeted);
    CHECK(sample.speed_towards_target == kMaxSpeed);

    // A first update after a reset is not a retarget, and produces exactly what
    // a target change produces.
    CenterOutGuidanceSample fresh{};
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, other, kOrigin, kStepNs, fresh) ==
          ContractStatus::ok);
    CHECK(!fresh.retargeted);
    CenterOutGuidanceSample changed{};
    CHECK(evaluate_guidance(config, fast, other, kOrigin, kStepNs, changed) == ContractStatus::ok);
    CHECK(changed.vel.x == fresh.vel.x);
    CHECK(changed.vel.y == fresh.vel.y);
    CHECK(changed.state.phase == fresh.state.phase);

    // The generator agrees with the function it wraps.
    CenterOutGuidance generator{};
    CHECK(generator.update(kEast, kOrigin, kStepNs, sample) == ContractStatus::not_running);
    CHECK(!generator.configured());
    CHECK(generator.configure(config) == ContractStatus::ok);
    CHECK(generator.configured());
    CHECK(generator.state().phase == CenterOutGuidancePhase::idle);
    for (int step = 0; step < 4; ++step)
        CHECK(generator.update(kEast, kOrigin, kStepNs, sample) == ContractStatus::ok);
    CHECK(generator.state().vel.x == kAcceleration * kStepSeconds * 4.0);
    CHECK(generator.update(other, kOrigin, kStepNs, sample) == ContractStatus::ok);
    CHECK(sample.retargeted);
    CHECK(generator.state().vel.x == kAcceleration * kStepSeconds);
    generator.reset();
    CHECK(generator.state().target == kUnsetTargetId);
    CHECK(generator.state().vel.x == 0.0);
    CHECK(generator.state().phase == CenterOutGuidancePhase::idle);
    CHECK(generator.configuration().max_speed == kMaxSpeed);

    // A refused update leaves the run where it was.
    CHECK(generator.update(kEast, kOrigin, kStepNs, sample) == ContractStatus::ok);
    const CenterOutGuidanceState before = generator.state();
    CHECK(generator.update(TargetPlacement{kUnsetTargetId, WorkspacePoint{1.0, 0.0}}, kOrigin,
                           kStepNs, sample) == ContractStatus::identity_missing);
    CHECK(generator.state().target == before.target);
    CHECK(generator.state().vel.x == before.vel.x);
    CHECK(generator.state().phase == before.phase);
    return failures == 0 ? 0 : 1;
}

int check_cadence_and_determinism()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};

    // Repeating the same observation sequence reproduces it exactly, whatever
    // else has happened in between.
    const WorkspacePoint script[] = {
        WorkspacePoint{0.0, 0.0},  WorkspacePoint{10.0, 0.0}, WorkspacePoint{40.0, 0.0},
        WorkspacePoint{70.0, 0.0}, WorkspacePoint{85.5, 0.0}, WorkspacePoint{92.0, 0.0},
        WorkspacePoint{97.0, 0.0}, WorkspacePoint{98.5, 0.0}, WorkspacePoint{60.0, 0.0},
        WorkspacePoint{20.0, 0.0},
    };

    double first_x[10]{};
    CenterOutGuidancePhase first_phase[10]{};
    CenterOutGuidance generator{};
    CHECK(generator.configure(config) == ContractStatus::ok);
    for (std::size_t i = 0; i < 10; ++i)
    {
        CHECK(generator.update(kEast, script[i], kStepNs, sample) == ContractStatus::ok);
        first_x[i] = sample.vel.x;
        first_phase[i] = sample.state.phase;
    }

    // A copy of a run is a run.
    CenterOutGuidance forked = generator;
    CenterOutGuidanceSample from_original{};
    CenterOutGuidanceSample from_fork{};
    CHECK(generator.update(kEast, kOrigin, kStepNs, from_original) == ContractStatus::ok);
    CHECK(forked.update(kEast, kOrigin, kStepNs, from_fork) == ContractStatus::ok);
    CHECK(from_original.vel.x == from_fork.vel.x);

    generator.reset();
    for (std::size_t i = 0; i < 10; ++i)
    {
        CHECK(generator.update(kEast, script[i], kStepNs, sample) == ContractStatus::ok);
        CHECK(sample.vel.x == first_x[i]);
        CHECK(sample.state.phase == first_phase[i]);
    }

    // Nothing counts invocations: a hundred unrelated evaluations between two
    // updates change neither.
    generator.reset();
    for (std::size_t i = 0; i < 10; ++i)
    {
        for (int noise = 0; noise < 100; ++noise)
        {
            CenterOutGuidanceSample ignored{};
            CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kNorth,
                                    WorkspacePoint{static_cast<double>(noise), 3.0}, kStepNs,
                                    ignored) == ContractStatus::ok);
        }
        CHECK(generator.update(kEast, script[i], kStepNs, sample) == ContractStatus::ok);
        CHECK(sample.vel.x == first_x[i]);
    }
    return failures == 0 ? 0 : 1;
}

int check_rejections_and_input_integrity()
{
    const CenterOutGuidanceConfig config = make_config();
    CenterOutGuidanceSample sample{};

    CHECK(evaluate_guidance(config, CenterOutGuidanceState{},
                            TargetPlacement{kUnsetTargetId, WorkspacePoint{1.0, 0.0}}, kOrigin,
                            kStepNs, sample) == ContractStatus::identity_missing);
    const double infinity = std::numeric_limits<double>::infinity();
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kEast, WorkspacePoint{infinity, 0.0},
                            kStepNs, sample) == ContractStatus::value_not_finite);
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kEast,
                            WorkspacePoint{0.0, std::numeric_limits<double>::quiet_NaN()}, kStepNs,
                            sample) == ContractStatus::value_not_finite);
    // Both endpoints finite, their squared separation not.
    const double huge = std::numeric_limits<double>::max();
    CHECK(evaluate_guidance(
              config, CenterOutGuidanceState{}, TargetPlacement{kTarget, WorkspacePoint{huge, 0.0}},
              WorkspacePoint{-huge, 0.0}, kStepNs, sample) == ContractStatus::value_not_finite);
    CHECK(evaluate_guidance(CenterOutGuidanceConfig{}, CenterOutGuidanceState{}, kEast, kOrigin,
                            kStepNs, sample) == ContractStatus::identity_missing);
    CHECK(evaluate_guidance(config, moving(4.0, 0.0, CenterOutGuidancePhase::idle, kUnsetTargetId),
                            kEast, kOrigin, kStepNs, sample) == ContractStatus::outcome_invalid);

    // A rejected evaluation leaves the caller's sample untouched.
    CenterOutGuidanceSample kept{};
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kEast, kOrigin, kStepNs, kept) ==
          ContractStatus::ok);
    const double kept_x = kept.vel.x;
    CHECK(evaluate_guidance(config, CenterOutGuidanceState{}, kEast, WorkspacePoint{infinity, 0.0},
                            kStepNs, kept) == ContractStatus::value_not_finite);
    CHECK(kept.vel.x == kept_x);

    // Nothing modifies what it was given.
    CenterOutGuidanceConfig owned = make_config();
    CenterOutGuidanceState previous = moving(12.5, -3.0, CenterOutGuidancePhase::speeding_up);
    TargetPlacement target = kEast;
    WorkspacePoint pos{30.0, -7.5};
    CHECK(evaluate_guidance(owned, previous, target, pos, kStepNs, sample) == ContractStatus::ok);
    CHECK(owned.geometry_unit == GeometryUnit::millimetres);
    CHECK(owned.max_speed == kMaxSpeed && owned.acceleration == kAcceleration);
    CHECK(owned.deceleration == kDeceleration && owned.precision == kPrecision);
    CHECK(previous.target == kTarget && previous.vel.x == 12.5);
    CHECK(previous.vel.y == -3.0 && previous.phase == CenterOutGuidancePhase::speeding_up);
    CHECK(target.id == kTarget && target.pos.x == 100.0 && target.pos.y == 0.0);
    CHECK(pos.x == 30.0 && pos.y == -7.5);
    return failures == 0 ? 0 : 1;
}

int run()
{
    // A first evaluation faults in whatever the standard library initializes
    // lazily; the count is about everything after it.
    {
        CenterOutGuidanceSample warmup{};
        (void)evaluate_guidance(make_config(), CenterOutGuidanceState{}, kEast, kOrigin, kStepNs,
                                warmup);
    }
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);

    if (const int failure = check_reference_trajectory(); failure != 0)
        return failure;
    if (const int failure = check_validation(); failure != 0)
        return failure;
    if (const int failure = check_cross_validation(); failure != 0)
        return failure;
    if (const int failure = check_profile(); failure != 0)
        return failure;
    if (const int failure = check_braking_decision(); failure != 0)
        return failure;
    if (const int failure = check_projection_and_overshoot(); failure != 0)
        return failure;
    if (const int failure = check_precision_region(); failure != 0)
        return failure;
    if (const int failure = check_retarget_and_reset(); failure != 0)
        return failure;
    if (const int failure = check_cadence_and_determinism(); failure != 0)
        return failure;
    if (const int failure = check_rejections_and_input_integrity(); failure != 0)
        return failure;

    const std::size_t allocated = allocations.load(std::memory_order_relaxed) - baseline;
    if (allocated != 0)
    {
        std::cerr << "FAILED: center-out guidance allocated " << allocated << " times\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main()
{
    return run();
}
