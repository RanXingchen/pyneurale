/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/center_out.h>
#include <neurale/experiments/contract.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#include "allocation_counter.h"
#include "check_counts.h"
#include <span>

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::center_out;

// Configuration is a value: it is copied into a session, compared between two
// sessions, and digested. A field that stopped being trivially copyable would
// have grown owning storage and none of those would still hold.
static_assert(std::is_trivially_copyable_v<WorkspacePoint>);
static_assert(std::is_trivially_copyable_v<TargetPlacement>);
static_assert(std::is_trivially_copyable_v<CenterOut2DLayout>);
static_assert(std::is_trivially_copyable_v<CenterOut2DConfig>);
static_assert(std::is_trivially_copyable_v<RadialLayoutRequest>);
static_assert(std::is_standard_layout_v<CenterOut2DConfig>);

// Stated so that a layout growing a per-target acceptance region, or a
// configuration growing a string, is a decision somebody makes here.
static_assert(sizeof(TargetPlacement) <= 24);
static_assert(sizeof(CenterOut2DLayout) <= 448);
static_assert(sizeof(CenterOut2DConfig) <= 640);

int failures = 0;

bool close(double left, double right) noexcept
{
    return std::fabs(left - right) <= 1e-12 * (1.0 + std::fabs(right));
}

RadialLayoutRequest radial_request(double radius, std::uint8_t count) noexcept
{
    RadialLayoutRequest request{};
    request.radius = radius;
    request.count = count;
    request.center_id = 1;
    for (std::size_t i = 0; i < count; ++i)
    {
        request.ids[i] = static_cast<TargetId>(i + 2);
        request.spokes[i] = static_cast<std::uint32_t>(i);
    }
    return request;
}

CenterOut2DLayout eight_target_ring() noexcept
{
    CenterOut2DLayout layout{};
    const ContractStatus status = build_radial_layout(radial_request(100.0, 8), layout);
    CHECK(status == ContractStatus::ok);
    return layout;
}

CenterOut2DConfig valid_config() noexcept
{
    CenterOut2DConfig config{};
    config.geometry_unit = GeometryUnit::millimetres;
    config.layout = eight_target_ring();
    config.acceptance = AcceptanceRegion{5.0, 5.0};
    config.cursor = CursorGeometry{2.0};
    config.movement_timeout = PhaseDurations{1'000'000'000ULL, 1'000'000'000ULL};
    config.hold_ns = 100'000'000ULL;
    config.reward_dwell = PhaseDurations{100'000'000ULL, 100'000'000ULL};
    config.punish_dwell = PhaseDurations{200'000'000ULL, 200'000'000ULL};
    config.selection = TargetSelectionPolicy::sample_each_trial;
    config.seed = 0xC0FFEE;
    config.trial_limit = 0;
    return config;
}

int check_radial_geometry()
{
    const int before = failures;

    // The floor: eight or more surrounding targets divide the circle; fewer than
    // eight do not fill it. Four targets on spokes 0..3 sit in one half plane.
    CHECK(close(radial_spoke_step(8), 2.0 * std::numbers::pi / 8.0));
    CHECK(close(radial_spoke_step(4), 2.0 * std::numbers::pi / 8.0));
    CHECK(close(radial_spoke_step(1), 2.0 * std::numbers::pi / 8.0));
    CHECK(close(radial_spoke_step(12), 2.0 * std::numbers::pi / 12.0));

    const CenterOut2DLayout ring = eight_target_ring();
    CHECK(ring.count == 8);
    CHECK(ring.center.id == 1);
    CHECK(ring.center.pos.x == 0.0);
    CHECK(ring.center.pos.y == 0.0);

    const double diagonal = 100.0 * std::sqrt(0.5);
    const std::array<WorkspacePoint, 8> expected{
        WorkspacePoint{100.0, 0.0},  WorkspacePoint{diagonal, diagonal},
        WorkspacePoint{0.0, 100.0},  WorkspacePoint{-diagonal, diagonal},
        WorkspacePoint{-100.0, 0.0}, WorkspacePoint{-diagonal, -diagonal},
        WorkspacePoint{0.0, -100.0}, WorkspacePoint{diagonal, -diagonal}};
    for (std::size_t i = 0; i < 8; ++i)
    {
        CHECK(ring.surrounding[i].id == static_cast<TargetId>(i + 2));
        CHECK(close(ring.surrounding[i].pos.x, expected[i].x));
        CHECK(close(ring.surrounding[i].pos.y, expected[i].y));
    }

    // Four targets: 0, 45, 90, 135 degrees -- not 0, 90, 180, 270.
    CenterOut2DLayout quarter{};
    CHECK(build_radial_layout(radial_request(2.0, 4), quarter) == ContractStatus::ok);
    CHECK(quarter.count == 4);
    CHECK(close(quarter.surrounding[0].pos.x, 2.0));
    CHECK(close(quarter.surrounding[0].pos.y, 0.0));
    CHECK(close(quarter.surrounding[2].pos.x, 0.0));
    CHECK(close(quarter.surrounding[2].pos.y, 2.0));
    CHECK(close(quarter.surrounding[3].pos.x, -2.0 * std::sqrt(0.5)));
    CHECK(close(quarter.surrounding[3].pos.y, 2.0 * std::sqrt(0.5)));

    // One surrounding target is a legal layout, and it lands on the first axis.
    CenterOut2DLayout single{};
    CHECK(build_radial_layout(radial_request(7.5, 1), single) == ContractStatus::ok);
    CHECK(single.count == 1);
    CHECK(close(single.surrounding[0].pos.x, 7.5));
    CHECK(close(single.surrounding[0].pos.y, 0.0));

    // A spoke beyond the count wraps around the circle.
    RadialLayoutRequest sparse = radial_request(1.0, 2);
    sparse.spokes[1] = 10;
    CenterOut2DLayout sparse_layout{};
    CHECK(build_radial_layout(sparse, sparse_layout) == ContractStatus::ok);
    CHECK(close(sparse_layout.surrounding[1].pos.x, std::cos(radial_spoke_step(2) * 10.0)));

    // The full ring at the capacity boundary.
    CenterOut2DLayout full{};
    CHECK(build_radial_layout(radial_request(1.0, kMaxSurroundingTargets), full) ==
          ContractStatus::ok);
    CHECK(full.count == kMaxSurroundingTargets);

    return failures - before;
}

int check_radial_request_is_not_modified()
{
    const int before = failures;

    // Building a layout must not mutate the request.
    const RadialLayoutRequest original = radial_request(3.0, 5);
    RadialLayoutRequest request = original;
    // Descending identifiers on ascending spokes: a builder that sorted would
    // reorder these, and a builder that read the identifiers as angles would
    // place them somewhere else.
    for (std::size_t i = 0; i < 5; ++i)
        request.ids[i] = static_cast<TargetId>(10 - i);
    const RadialLayoutRequest before_call = request;

    CenterOut2DLayout layout{};
    CHECK(build_radial_layout(request, layout) == ContractStatus::ok);

    CHECK(request.radius == before_call.radius);
    CHECK(request.count == before_call.count);
    CHECK(request.center_id == before_call.center_id);
    for (std::size_t i = 0; i < kMaxSurroundingTargets; ++i)
    {
        CHECK(request.ids[i] == before_call.ids[i]);
        CHECK(request.spokes[i] == before_call.spokes[i]);
    }
    // Caller order is preserved and the angle follows the spoke, not the name.
    for (std::size_t i = 0; i < 5; ++i)
    {
        CHECK(layout.surrounding[i].id == static_cast<TargetId>(10 - i));
        CHECK(close(layout.surrounding[i].pos.x,
                    3.0 * std::cos(radial_spoke_step(5) * static_cast<double>(i))));
    }

    return failures - before;
}

int check_layout_validation()
{
    const int before = failures;

    const CenterOut2DLayout ring = eight_target_ring();
    CHECK(validate(ring) == ContractStatus::ok);

    CenterOut2DLayout empty{};
    CHECK(validate(empty) == ContractStatus::target_set_invalid);

    CenterOut2DLayout oversized = ring;
    oversized.count = kMaxSurroundingTargets + 1;
    CHECK(validate(oversized) == ContractStatus::target_set_invalid);

    // "Missing centre semantics": the centre is a field, so it cannot be left
    // out of the structure, but it can be left unnamed and that is refused.
    CenterOut2DLayout no_center = ring;
    no_center.center.id = kUnsetTargetId;
    CHECK(validate(no_center) == ContractStatus::identity_missing);

    CenterOut2DLayout unnamed = ring;
    unnamed.surrounding[3].id = kUnsetTargetId;
    CHECK(validate(unnamed) == ContractStatus::identity_missing);

    CenterOut2DLayout duplicated = ring;
    duplicated.surrounding[5].id = duplicated.surrounding[2].id;
    CHECK(validate(duplicated) == ContractStatus::target_set_invalid);

    CenterOut2DLayout shadows_center = ring;
    shadows_center.surrounding[0].id = shadows_center.center.id;
    CHECK(validate(shadows_center) == ContractStatus::target_set_invalid);

    CenterOut2DLayout not_finite = ring;
    not_finite.surrounding[1].pos.y = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(not_finite) == ContractStatus::value_not_finite);

    CenterOut2DLayout padded = ring;
    padded.surrounding[8].id = 99;
    CHECK(validate(padded) == ContractStatus::target_set_invalid);

    CenterOut2DLayout padded_position = ring;
    padded_position.surrounding[9].pos.x = 1.0;
    CHECK(validate(padded_position) == ContractStatus::target_set_invalid);

    // Two identifiers at one place is a caller's choice, not a contract error.
    CenterOut2DLayout stacked = ring;
    stacked.surrounding[4].pos = stacked.surrounding[0].pos;
    CHECK(validate(stacked) == ContractStatus::ok);

    return failures - before;
}

int check_radial_request_validation()
{
    const int before = failures;

    CHECK(validate(radial_request(1.0, 4)) == ContractStatus::ok);

    RadialLayoutRequest zero_radius = radial_request(0.0, 4);
    CHECK(validate(zero_radius) == ContractStatus::parameter_out_of_range);

    RadialLayoutRequest negative_radius = radial_request(-1.0, 4);
    CHECK(validate(negative_radius) == ContractStatus::parameter_out_of_range);

    RadialLayoutRequest infinite_radius = radial_request(1.0, 4);
    infinite_radius.radius = std::numeric_limits<double>::infinity();
    CHECK(validate(infinite_radius) == ContractStatus::value_not_finite);

    RadialLayoutRequest no_targets = radial_request(1.0, 0);
    CHECK(validate(no_targets) == ContractStatus::target_set_invalid);

    RadialLayoutRequest too_many = radial_request(1.0, 4);
    too_many.count = kMaxSurroundingTargets + 1;
    CHECK(validate(too_many) == ContractStatus::target_set_invalid);

    RadialLayoutRequest no_center = radial_request(1.0, 4);
    no_center.center_id = kUnsetTargetId;
    CHECK(validate(no_center) == ContractStatus::identity_missing);

    RadialLayoutRequest unnamed = radial_request(1.0, 4);
    unnamed.ids[2] = kUnsetTargetId;
    CHECK(validate(unnamed) == ContractStatus::identity_missing);

    RadialLayoutRequest duplicated = radial_request(1.0, 4);
    duplicated.ids[3] = duplicated.ids[1];
    CHECK(validate(duplicated) == ContractStatus::target_set_invalid);

    RadialLayoutRequest shadows_center = radial_request(1.0, 4);
    shadows_center.ids[0] = shadows_center.center_id;
    CHECK(validate(shadows_center) == ContractStatus::target_set_invalid);

    RadialLayoutRequest padded = radial_request(1.0, 4);
    padded.ids[7] = 77;
    CHECK(validate(padded) == ContractStatus::target_set_invalid);

    // A rejected request leaves the caller's layout untouched.
    CenterOut2DLayout layout = eight_target_ring();
    const CenterOut2DLayout untouched = layout;
    CHECK(build_radial_layout(duplicated, layout) == ContractStatus::target_set_invalid);
    CHECK(layout.count == untouched.count);
    CHECK(layout.surrounding[0].id == untouched.surrounding[0].id);

    return failures - before;
}

int check_containment()
{
    const int before = failures;

    const AcceptanceRegion region{5.0, 3.0};
    const CursorGeometry cursor{1.0};
    const WorkspacePoint target{10.0, -4.0};
    bool inside = false;

    // Dead centre.
    CHECK(contains_cursor(region, cursor, target, target, inside) == ContractStatus::ok);
    CHECK(inside);

    // Exactly on the boundary of the reduced region: equality is inside.
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{14.0, -4.0}, inside) ==
          ContractStatus::ok);
    CHECK(inside);
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{10.0, -6.0}, inside) ==
          ContractStatus::ok);
    CHECK(inside);
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{6.0, -2.0}, inside) ==
          ContractStatus::ok);
    CHECK(inside);

    // One ulp past it is outside on each axis independently.
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{std::nextafter(14.0, 20.0), -4.0},
                          inside) == ContractStatus::ok);
    CHECK(!inside);
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{10.0, std::nextafter(-6.0, -20.0)},
                          inside) == ContractStatus::ok);
    CHECK(!inside);

    // Well outside, and outside on one axis only.
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{100.0, -4.0}, inside) ==
          ContractStatus::ok);
    CHECK(!inside);
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{10.0, 0.0}, inside) ==
          ContractStatus::ok);
    CHECK(!inside);

    // It is containment, not overlap: a cursor whose edge merely touches the
    // region is outside, where a distance or overlap test would accept it.
    CHECK(contains_cursor(region, cursor, target, WorkspacePoint{15.5, -4.0}, inside) ==
          ContractStatus::ok);
    CHECK(!inside);

    // Point containment is the extent-zero case, and its own boundary is the
    // region's own edge.
    CHECK(contains_point(region, target, WorkspacePoint{15.0, -1.0}, inside) == ContractStatus::ok);
    CHECK(inside);
    CHECK(contains_point(region, target, WorkspacePoint{15.0, -0.5}, inside) == ContractStatus::ok);
    CHECK(!inside);
    bool as_point = false;
    CHECK(contains_cursor(region, CursorGeometry{0.0}, target, WorkspacePoint{15.0, -1.0},
                          as_point) == ContractStatus::ok);
    CHECK(as_point == true);

    // A cursor wider than the region answers false everywhere rather than
    // failing: the predicate is total, and the configuration is what is refused.
    const CursorGeometry oversized{9.0};
    CHECK(contains_cursor(region, oversized, target, target, inside) == ContractStatus::ok);
    CHECK(!inside);

    // Reported failures.
    CHECK(contains_cursor(AcceptanceRegion{0.0, 3.0}, cursor, target, target, inside) ==
          ContractStatus::parameter_out_of_range);
    CHECK(contains_cursor(AcceptanceRegion{-1.0, 3.0}, cursor, target, target, inside) ==
          ContractStatus::parameter_out_of_range);
    CHECK(contains_cursor(region, CursorGeometry{-0.5}, target, target, inside) ==
          ContractStatus::parameter_out_of_range);
    CHECK(contains_cursor(region, cursor,
                          WorkspacePoint{std::numeric_limits<double>::quiet_NaN(), 0.0}, target,
                          inside) == ContractStatus::value_not_finite);
    CHECK(contains_cursor(region, cursor, target,
                          WorkspacePoint{0.0, std::numeric_limits<double>::infinity()},
                          inside) == ContractStatus::value_not_finite);

    // A rejected call leaves the answer alone rather than writing a default.
    bool untouched = true;
    CHECK(contains_cursor(AcceptanceRegion{0.0, 1.0}, cursor, target, target, untouched) ==
          ContractStatus::parameter_out_of_range);
    CHECK(untouched);

    return failures - before;
}

int check_config_validation()
{
    const int before = failures;

    const CenterOut2DConfig config = valid_config();
    CHECK(validate(config) == ContractStatus::ok);

    CenterOut2DConfig no_unit = config;
    no_unit.geometry_unit = GeometryUnit::unspecified;
    CHECK(validate(no_unit) == ContractStatus::identity_missing);

    CenterOut2DConfig bad_unit = config;
    bad_unit.geometry_unit = static_cast<GeometryUnit>(200);
    CHECK(validate(bad_unit) == ContractStatus::enum_undeclared);

    CenterOut2DConfig no_policy = config;
    no_policy.selection = TargetSelectionPolicy::unspecified;
    CHECK(validate(no_policy) == ContractStatus::identity_missing);

    CenterOut2DConfig bad_policy = config;
    bad_policy.selection = static_cast<TargetSelectionPolicy>(200);
    CHECK(validate(bad_policy) == ContractStatus::enum_undeclared);

    CenterOut2DConfig unwinnable = config;
    unwinnable.cursor = CursorGeometry{5.5};
    CHECK(validate(unwinnable) == ContractStatus::parameter_out_of_range);

    // Exactly as wide as the region is still winnable: only the target centre
    // satisfies it, but the boundary is inside.
    CenterOut2DConfig exact = config;
    exact.cursor = CursorGeometry{5.0};
    exact.acceptance = AcceptanceRegion{5.0, 5.0};
    CHECK(validate(exact) == ContractStatus::ok);

    CenterOut2DConfig no_time = config;
    no_time.movement_timeout.to_out = 0;
    CHECK(validate(no_time) == ContractStatus::parameter_out_of_range);

    // A zero hold is legal.
    CenterOut2DConfig instant = config;
    instant.hold_ns = 0;
    CHECK(validate(instant) == ContractStatus::ok);

    // So are zero reward and punishment dwells.
    CenterOut2DConfig no_dwell = config;
    no_dwell.reward_dwell = PhaseDurations{};
    no_dwell.punish_dwell = PhaseDurations{};
    CHECK(validate(no_dwell) == ContractStatus::ok);

    CenterOut2DConfig no_sampler = config;
    no_sampler.sampler_version = 0;
    CHECK(validate(no_sampler) == ContractStatus::identity_missing);

    // An unsupported sampler is accepted by validate(), as a schedule identity
    // is, because such a session still replays from its recorded schedule.
    CenterOut2DConfig future_sampler = config;
    future_sampler.sampler_version = 999;
    CHECK(validate(future_sampler) == ContractStatus::ok);

    CenterOut2DConfig broken_layout = config;
    broken_layout.layout.count = 0;
    CHECK(validate(broken_layout) == ContractStatus::target_set_invalid);

    CHECK(phase_duration(config.movement_timeout, CenterOutPhase::to_center) ==
          config.movement_timeout.to_center);
    CHECK(phase_duration(config.punish_dwell, CenterOutPhase::to_out) ==
          config.punish_dwell.to_out);

    CHECK(!trial_limit_reached(config, 0));
    CHECK(!trial_limit_reached(config, 1'000'000));
    CenterOut2DConfig limited = config;
    limited.trial_limit = 3;
    CHECK(!trial_limit_reached(limited, 2));
    CHECK(trial_limit_reached(limited, 3));
    CHECK(trial_limit_reached(limited, 4));

    return failures - before;
}

int check_target_selection()
{
    const int before = failures;

    CenterOut2DConfig cycling = valid_config();
    cycling.selection = TargetSelectionPolicy::repeat_until_success;

    std::uint8_t idx = 0;
    for (std::uint64_t successes = 0; successes < 20; ++successes)
    {
        CHECK(select_outward_target(cycling, successes * 3, successes, idx) == ContractStatus::ok);
        CHECK(idx == static_cast<std::uint8_t>(successes % cycling.layout.count));
    }

    // A failure does not advance the success count, so the target repeats.
    std::uint8_t first = 0;
    std::uint8_t retried = 0;
    CHECK(select_outward_target(cycling, 7, 4, first) == ContractStatus::ok);
    CHECK(select_outward_target(cycling, 8, 4, retried) == ContractStatus::ok);
    CHECK(first == retried);
    std::uint8_t after_success = 0;
    CHECK(select_outward_target(cycling, 9, 5, after_success) == ContractStatus::ok);
    CHECK(after_success != first);

    CenterOut2DConfig sampling = valid_config();
    std::array<std::uint8_t, 64> drawn{};
    for (std::uint64_t trial = 0; trial < drawn.size(); ++trial)
    {
        CHECK(select_outward_target(sampling, trial, 0, drawn[trial]) == ContractStatus::ok);
        CHECK(drawn[trial] < sampling.layout.count);
    }

    // Deterministic: the same key answers the same way however many times it is
    // asked, and the success count is not part of that key.
    for (std::uint64_t trial = 0; trial < drawn.size(); ++trial)
    {
        std::uint8_t again = 0;
        CHECK(select_outward_target(sampling, trial, 41, again) == ContractStatus::ok);
        CHECK(again == drawn[trial]);
    }

    // Resetting a schedule is returning the ordinals to where they started; the
    // sampler holds nothing to reset, so the same sequence follows.
    for (std::uint64_t trial = 0; trial < 8; ++trial)
    {
        std::uint8_t replayed = 0;
        CHECK(select_outward_target(sampling, trial, 0, replayed) == ContractStatus::ok);
        CHECK(replayed == drawn[trial]);
    }

    // A different seed is a different sequence. This comparison is itself
    // deterministic: both sides are fixed functions of their seeds.
    CenterOut2DConfig other_seed = sampling;
    other_seed.seed = sampling.seed + 1;
    bool differs = false;
    for (std::uint64_t trial = 0; trial < drawn.size(); ++trial)
    {
        std::uint8_t value = 0;
        CHECK(select_outward_target(other_seed, trial, 0, value) == ContractStatus::ok);
        differs = differs || value != drawn[trial];
    }
    CHECK(differs);

    CenterOut2DConfig balanced = sampling;
    balanced.selection = TargetSelectionPolicy::balanced_shuffled_cycles;
    // Freeze the original draw addressing when sharing the shuffle with SSVEP.
    auto frozen = balanced;
    frozen.seed = 42;
    CHECK(build_radial_layout(radial_request(100.0, 4), frozen.layout) == ContractStatus::ok);
    constexpr std::array<std::uint8_t, 12> golden{3, 2, 0, 1, 3, 2, 0, 1, 0, 3, 2, 1};
    for (std::size_t i = 0; i < golden.size(); ++i)
    {
        std::uint8_t value{};
        CHECK(select_outward_target(frozen, i, 0, value) == ContractStatus::ok);
        CHECK(value == golden[i]);
    }
    std::array<std::uint32_t, kMaxSurroundingTargets> counts{};
    for (std::uint64_t trial = 0; trial < 53; ++trial)
    {
        std::uint8_t value = 0;
        CHECK(select_outward_target(balanced, trial, 0, value) == ContractStatus::ok);
        CHECK(value < balanced.layout.count);
        ++counts[value];
        std::uint32_t least = counts[0];
        std::uint32_t most = counts[0];
        for (std::size_t slot = 1; slot < balanced.layout.count; ++slot)
        {
            least = (std::min)(least, counts[slot]);
            most = (std::max)(most, counts[slot]);
        }
        CHECK(most - least <= 1);
    }
    for (std::uint64_t cycle = 0; cycle < 4; ++cycle)
    {
        std::array<bool, kMaxSurroundingTargets> cycle_seen{};
        for (std::uint64_t offset = 0; offset < balanced.layout.count; ++offset)
        {
            std::uint8_t value = 0;
            const auto trial = cycle * balanced.layout.count + offset;
            CHECK(select_outward_target(balanced, trial, cycle, value) == ContractStatus::ok);
            CHECK(!cycle_seen[value]);
            cycle_seen[value] = true;
        }
    }

    // Every surrounding target is reachable and the centre never is: the answer
    // indexes the surrounding array, which the centre is not a member of.
    std::array<bool, kMaxSurroundingTargets> seen{};
    for (std::uint64_t trial = 0; trial < 512; ++trial)
    {
        std::uint8_t value = 0;
        CHECK(select_outward_target(sampling, trial, 0, value) == ContractStatus::ok);
        seen[value] = true;
    }
    for (std::size_t slot = 0; slot < sampling.layout.count; ++slot)
        CHECK(seen[slot]);
    for (std::size_t slot = sampling.layout.count; slot < kMaxSurroundingTargets; ++slot)
        CHECK(!seen[slot]);

    // With one surrounding target both policies answer the only slot there is.
    CenterOut2DConfig single = valid_config();
    CHECK(build_radial_layout(radial_request(1.0, 1), single.layout) == ContractStatus::ok);
    std::uint8_t only = 200;
    CHECK(select_outward_target(single, 17, 5, only) == ContractStatus::ok);
    CHECK(only == 0);
    single.selection = TargetSelectionPolicy::repeat_until_success;
    only = 200;
    CHECK(select_outward_target(single, 17, 5, only) == ContractStatus::ok);
    CHECK(only == 0);
    single.selection = TargetSelectionPolicy::balanced_shuffled_cycles;
    only = 200;
    CHECK(select_outward_target(single, 17, 5, only) == ContractStatus::ok);
    CHECK(only == 0);

    // An invalid configuration is refused before anything is drawn, and the
    // caller's output is left alone.
    CenterOut2DConfig invalid = valid_config();
    invalid.selection = TargetSelectionPolicy::unspecified;
    std::uint8_t untouched = 123;
    CHECK(select_outward_target(invalid, 0, 0, untouched) == ContractStatus::identity_missing);
    CHECK(untouched == 123);

    // A sampler this build cannot evaluate is refused only where a draw is
    // actually needed.
    CenterOut2DConfig future = valid_config();
    future.sampler_version = 999;
    std::uint8_t unused = 111;
    CHECK(select_outward_target(future, 0, 0, unused) == ContractStatus::version_unsupported);
    CHECK(unused == 111);
    future.selection = TargetSelectionPolicy::repeat_until_success;
    CHECK(select_outward_target(future, 0, 3, unused) == ContractStatus::ok);
    CHECK(unused == 3);

    return failures - before;
}

int check_fingerprints()
{
    const int before = failures;

    const CenterOut2DConfig config = valid_config();
    const std::uint64_t digest = configuration_fingerprint(config);
    CHECK(digest == configuration_fingerprint(valid_config()));
    CHECK(layout_fingerprint(config.layout) == layout_fingerprint(eight_target_ring()));

    // Every field participates: a session that changed one is a different
    // schedule and must not be mistaken for the one it came from.
    const auto differs = [digest](const CenterOut2DConfig& changed)
    { return configuration_fingerprint(changed) != digest; };

    CenterOut2DConfig changed = config;
    changed.geometry_unit = GeometryUnit::metres;
    CHECK(differs(changed));
    changed = config;
    changed.acceptance.half_extent_y = 5.5;
    CHECK(differs(changed));
    changed = config;
    changed.cursor.extent = 2.5;
    CHECK(differs(changed));
    changed = config;
    changed.movement_timeout.to_center += 1;
    CHECK(differs(changed));
    changed = config;
    changed.hold_ns += 1;
    CHECK(differs(changed));
    changed = config;
    changed.reward_dwell.to_out += 1;
    CHECK(differs(changed));
    changed = config;
    changed.punish_dwell.to_center += 1;
    CHECK(differs(changed));
    changed = config;
    changed.selection = TargetSelectionPolicy::repeat_until_success;
    CHECK(differs(changed));
    changed = config;
    changed.seed += 1;
    CHECK(differs(changed));
    changed = config;
    changed.sampler_version += 1;
    CHECK(differs(changed));
    changed = config;
    changed.trial_limit = 40;
    CHECK(differs(changed));
    changed = config;
    changed.layout.surrounding[2].id = 99;
    CHECK(differs(changed));
    changed = config;
    changed.layout.surrounding[2].pos.x += 1.0;
    CHECK(differs(changed));

    // Order is part of the layout: the same targets listed differently is a
    // different layout, because the selection index refers to a position.
    CenterOut2DLayout swapped = config.layout;
    const TargetPlacement held = swapped.surrounding[0];
    swapped.surrounding[0] = swapped.surrounding[1];
    swapped.surrounding[1] = held;
    CHECK(layout_fingerprint(swapped) != layout_fingerprint(config.layout));

    // Negative zero is the same place as zero.
    CenterOut2DLayout negative_zero = config.layout;
    negative_zero.surrounding[2].pos.x = -0.0;
    CenterOut2DLayout positive_zero = config.layout;
    positive_zero.surrounding[2].pos.x = 0.0;
    CHECK(layout_fingerprint(negative_zero) == layout_fingerprint(positive_zero));

    return failures - before;
}

int check_status_messages()
{
    const int before = failures;
    const char* message = contract_status_message(ContractStatus::target_set_invalid);
    CHECK(message != nullptr);
    CHECK(message[0] != '\0');
    CHECK(message != contract_status_message(ContractStatus::dimension_invalid));
    return failures - before;
}

int run()
{
    // A first call may fault in whatever the standard library lazily
    // initializes; the geometry that follows is what the count is about.
    CenterOut2DLayout warmup{};
    (void)build_radial_layout(radial_request(1.0, 8), warmup);
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);

    if (const int failure = check_radial_geometry(); failure != 0)
        return failure;
    if (const int failure = check_radial_request_is_not_modified(); failure != 0)
        return failure;
    if (const int failure = check_layout_validation(); failure != 0)
        return failure;
    if (const int failure = check_radial_request_validation(); failure != 0)
        return failure;
    if (const int failure = check_containment(); failure != 0)
        return failure;
    if (const int failure = check_config_validation(); failure != 0)
        return failure;
    if (const int failure = check_target_selection(); failure != 0)
        return failure;
    if (const int failure = check_fingerprints(); failure != 0)
        return failure;
    if (const int failure = check_status_messages(); failure != 0)
        return failure;

    const std::size_t allocated = allocations.load(std::memory_order_relaxed) - baseline;
    if (allocated != 0)
    {
        std::cerr << "FAILED: center-out geometry allocated " << allocated << " times\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

namespace machine_tests
{
namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::center_out;

// A session is a value. A machine that stopped being trivially copyable would
// have grown owning storage, and copying a session -- which is how a caller
// forks a replay -- would no longer be a copy.
static_assert(std::is_trivially_copyable_v<CenterOutMachine>);
static_assert(std::is_trivially_copyable_v<CenterOutSnapshot>);
static_assert(std::is_trivially_copyable_v<CenterOutTrial>);
static_assert(std::is_trivially_copyable_v<CenterOutStepResult>);

int failures = 0;

constexpr ExperimentTimeNs kMs = 1'000'000;
constexpr ParadigmId kParadigm = 7;

CenterOut2DConfig make_config(DurationNs hold_ns = 100 * kMs, DurationNs dwell_ns = 50 * kMs,
                              TrialOrdinal limit = 0) noexcept
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
    config.acceptance = AcceptanceRegion{5.0, 5.0};
    config.cursor = CursorGeometry{0.0};
    config.movement_timeout = PhaseDurations{1000 * kMs, 1000 * kMs};
    config.hold_ns = hold_ns;
    config.reward_dwell = PhaseDurations{dwell_ns, dwell_ns};
    config.punish_dwell = PhaseDurations{dwell_ns, dwell_ns};
    config.selection = TargetSelectionPolicy::sample_each_trial;
    config.seed = 0xBEEF;
    config.trial_limit = limit;
    return config;
}

// Far outside every acceptance region, so a leg spent here can only time out.
constexpr WorkspacePoint kNowhere{500.0, 500.0};
constexpr WorkspacePoint kCentre{0.0, 0.0};

WorkspacePoint outward_position(const CenterOut2DConfig& config, std::uint8_t idx) noexcept
{
    return config.layout.surrounding[idx].pos;
}

int check_lifecycle()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();

    CenterOutMachine machine{};
    CenterOutStepResult result{};
    CHECK(machine.state() == CenterOutState::idle);
    // Stepping a machine that was never started is refused rather than answered
    // with a snapshot of a session that never began.
    CHECK(machine.step(0, kCentre, result) == ContractStatus::not_running);

    CHECK(machine.start(kUnsetParadigmId, config, 0, result) == ContractStatus::identity_missing);
    CenterOut2DConfig broken = config;
    broken.selection = TargetSelectionPolicy::unspecified;
    CHECK(machine.start(kParadigm, broken, 0, result) == ContractStatus::identity_missing);
    CHECK(machine.state() == CenterOutState::idle);

    CHECK(machine.start(kParadigm, config, 5 * kMs, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::move_to_center);
    CHECK(machine.paradigm() == kParadigm);
    CHECK(result.n_transitions == 1);
    CHECK(result.transitions[0].from_state == static_cast<StateId>(CenterOutState::idle));
    CHECK(result.transitions[0].to_state == static_cast<StateId>(CenterOutState::move_to_center));
    CHECK(result.transitions[0].paradigm == kParadigm);
    CHECK(result.n_events == 2);
    CHECK(result.events[0].kind == ExperimentEventKind::session_start);
    CHECK(result.events[1].kind == ExperimentEventKind::trial_start);
    CHECK(!result.trial_decided);
    CHECK(result.snapshot.leg.start_ns == 5 * kMs);
    CHECK(result.snapshot.leg.end_ns == 1005 * kMs);
    CHECK(result.snapshot.active_target == config.layout.center.id);
    CHECK(result.snapshot.outward_target != kUnsetTargetId);
    CHECK(result.snapshot.outward_target != config.layout.center.id);

    // Time may not move backwards, and a refused step leaves the machine alone.
    const CenterOutSnapshot held = machine.snapshot();
    CHECK(machine.step(4 * kMs, kCentre, result) == ContractStatus::time_regressed);
    CHECK(machine.snapshot().state == held.state);
    CHECK(machine.snapshot().time_ns == held.time_ns);

    // A non-finite observation is reported, not folded into "outside".
    CHECK(machine.step(6 * kMs, WorkspacePoint{0.0, 1.0 / 0.0 - 1.0 / 0.0}, result) ==
          ContractStatus::value_not_finite);

    machine.reset();
    CHECK(machine.state() == CenterOutState::idle);
    CHECK(machine.paradigm() == kUnsetParadigmId);
    CHECK(machine.configuration().layout.count == 0);
    return failures - before;
}

int check_successful_cycle()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();

    CenterOutMachine machine{};
    CenterOutStepResult result{};
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    const std::uint8_t idx = result.snapshot.outward_idx;
    const WorkspacePoint target = outward_position(config, idx);

    // Arrive at the centre: the hold begins at the instant containment is first
    // observed, and not before.
    CHECK(machine.step(100 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::hold_center);
    CHECK(result.snapshot.hold.start_ns == 100 * kMs);
    CHECK(result.snapshot.hold.end_ns == 200 * kMs);
    CHECK(result.snapshot.contained);
    CHECK(result.n_transitions == 1);
    CHECK(result.transitions[0].cause ==
          static_cast<std::uint32_t>(CenterOutCause::containment_gained));

    // Still short of the hold.
    CHECK(machine.step(150 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::hold_center);
    CHECK(result.n_transitions == 0);

    // The hold completes, and its instant -- not the observation's -- is when.
    CHECK(machine.step(210 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::center_success_dwell);
    CHECK(result.n_transitions == 1);
    CHECK(result.transitions[0].time_ns == 200 * kMs);
    CHECK(!result.trial_decided);
    CHECK(result.snapshot.dwell.start_ns == 200 * kMs);
    CHECK(result.snapshot.dwell.end_ns == 250 * kMs);

    // The outward leg begins when the dwell ends, not when it was noticed.
    CHECK(machine.step(260 * kMs, target, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::hold_out);
    CHECK(result.n_transitions == 2);
    CHECK(result.transitions[0].to_state == static_cast<StateId>(CenterOutState::move_to_out));
    CHECK(result.transitions[0].time_ns == 250 * kMs);
    CHECK(result.snapshot.leg.start_ns == 250 * kMs);
    CHECK(result.snapshot.active_target == config.layout.surrounding[idx].id);

    CHECK(machine.step(400 * kMs, target, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::out_success_dwell);
    CHECK(result.trial_decided);
    CHECK(result.trial.record.outcome == TrialOutcome::success);
    CHECK(result.trial.record.reason ==
          static_cast<std::uint32_t>(CenterOutReason::outward_acquired));
    CHECK(result.trial.record.trial.ordinal == 0);
    CHECK(result.trial.record.interval.start_ns == 0);
    CHECK(result.trial.record.interval.end_ns == 360 * kMs);
    CHECK(result.trial.decided_phase == CenterOutPhase::to_out);
    CHECK(result.trial.outward_target == config.layout.surrounding[idx].id);
    CHECK(result.trial.center_acquire_ns == 200 * kMs);
    CHECK(result.trial.outward_acquire_ns == 110 * kMs);
    CHECK(result.snapshot.completed == 1);
    CHECK(result.snapshot.successes == 1);

    // The next trial begins when the dwell ends.
    CHECK(machine.step(420 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::move_to_center);
    CHECK(result.snapshot.trial.ordinal == 1);
    return failures - before;
}

int check_failures()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();

    // The centre leg times out without moving the caller-owned cursor.
    CenterOutMachine machine{};
    CenterOutStepResult result{};
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(machine.step(1200 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::center_failure_dwell);
    CHECK(result.trial_decided);
    CHECK(result.trial.record.outcome == TrialOutcome::timeout);
    CHECK(result.trial.decided_phase == CenterOutPhase::to_center);
    CHECK(result.trial.record.reason ==
          static_cast<std::uint32_t>(CenterOutReason::center_movement_timeout));
    CHECK(result.trial.record.interval.end_ns == 1000 * kMs);
    CHECK(result.trial.center_acquire_ns == 0);
    CHECK(result.trial.outward_acquire_ns == 0);
    CHECK(result.snapshot.completed == 1);
    CHECK(result.snapshot.successes == 0);
    // More had already elapsed than this step consumed, so it is not settled.
    CHECK(!result.settled);

    // The outward leg times out, and the centre acquisition it did make is kept.
    CenterOutMachine second{};
    CHECK(second.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(second.step(100 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(second.step(200 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(second.state() == CenterOutState::center_success_dwell);
    CHECK(second.step(2000 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(second.state() == CenterOutState::out_failure_dwell);
    CHECK(result.trial_decided);
    CHECK(result.trial.record.outcome == TrialOutcome::timeout);
    CHECK(result.trial.decided_phase == CenterOutPhase::to_out);
    CHECK(result.trial.record.reason ==
          static_cast<std::uint32_t>(CenterOutReason::outward_movement_timeout));
    // Outward leg started at 250 ms and allowed 1000 ms.
    CHECK(result.trial.record.interval.end_ns == 1250 * kMs);
    CHECK(result.trial.center_acquire_ns == 200 * kMs);
    CHECK(result.trial.outward_acquire_ns == 0);
    return failures - before;
}

int check_hold_reset()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();

    CenterOutMachine machine{};
    CenterOutStepResult result{};
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);

    CHECK(machine.step(100 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::hold_center);
    CHECK(result.snapshot.hold.end_ns == 200 * kMs);

    // Leaving discards the hold; it is not paused and not carried.
    CHECK(machine.step(150 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::move_to_center);
    CHECK(result.n_transitions == 1);
    CHECK(result.transitions[0].cause ==
          static_cast<std::uint32_t>(CenterOutCause::containment_lost));
    CHECK(result.snapshot.hold.empty());
    CHECK(!result.snapshot.contained);
    // The movement window is untouched: leaving the target does not buy time.
    CHECK(result.snapshot.leg.end_ns == 1000 * kMs);

    // Returning starts a new hold from the instant it is observed, so the
    // acquisition is at 160 + 100 rather than at the original 200.
    CHECK(machine.step(160 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::hold_center);
    CHECK(result.snapshot.hold.start_ns == 160 * kMs);
    CHECK(result.snapshot.hold.end_ns == 260 * kMs);

    CHECK(machine.step(259 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::hold_center);
    CHECK(machine.step(260 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.state() == CenterOutState::center_success_dwell);
    CHECK(result.transitions[0].time_ns == 260 * kMs);
    return failures - before;
}

int check_boundary_precedence()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();

    // The hold would complete at the very instant the movement window ends.
    // The window is half-open, so at that instant it is already over and the
    // timeout wins.
    CenterOutMachine tie{};
    CenterOutStepResult result{};
    CHECK(tie.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(tie.step(900 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(tie.state() == CenterOutState::hold_center);
    CHECK(result.snapshot.hold.end_ns == 1000 * kMs);
    CHECK(tie.step(1000 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(tie.state() == CenterOutState::center_failure_dwell);
    CHECK(result.trial.record.outcome == TrialOutcome::timeout);

    // One nanosecond earlier the hold completes strictly inside the window.
    CenterOutMachine inside{};
    CHECK(inside.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(inside.step(900 * kMs - 1, kCentre, result) == ContractStatus::ok);
    CHECK(inside.step(1000 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(inside.state() == CenterOutState::center_success_dwell);
    CHECK(result.transitions[0].time_ns == 1000 * kMs - 1);

    // Containment observed exactly at the window's end does not start a hold.
    CenterOutMachine late{};
    CHECK(late.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(late.step(1000 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(late.state() == CenterOutState::center_failure_dwell);
    CHECK(result.trial.record.interval.end_ns == 1000 * kMs);

    // A hold that completed before the window ended wins even when the cursor
    // has since left and the observation arrives long afterwards: it happened
    // first.
    CenterOutMachine earlier{};
    CHECK(earlier.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(earlier.step(100 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(earlier.step(5000 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(earlier.state() != CenterOutState::center_failure_dwell);
    CHECK(result.transitions[0].to_state ==
          static_cast<StateId>(CenterOutState::center_success_dwell));
    CHECK(result.transitions[0].time_ns == 200 * kMs);

    // A zero hold acquires on the first observation inside the region.
    const CenterOut2DConfig instant = make_config(0, 0);
    CenterOutMachine snap{};
    CHECK(snap.start(kParadigm, instant, 0, result) == ContractStatus::ok);
    CHECK(snap.step(10 * kMs, kCentre, result) == ContractStatus::ok);
    // Centre acquired and the outward leg opened, all on one observation; the
    // outward target is elsewhere, so the chain settles there.
    CHECK(snap.state() == CenterOutState::move_to_out);
    CHECK(result.snapshot.leg.start_ns == 10 * kMs);
    CHECK(result.settled);
    return failures - before;
}

int check_elapsed_time_not_invocations()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();
    CenterOutStepResult result{};

    // One step at the deadline, versus a thousand before it.
    CenterOutMachine sparse{};
    CHECK(sparse.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(sparse.step(1000 * kMs, kNowhere, result) == ContractStatus::ok);
    const CenterOutTrial sparse_trial = result.trial;
    CHECK(result.trial_decided);

    CenterOutMachine dense{};
    CHECK(dense.start(kParadigm, config, 0, result) == ContractStatus::ok);
    bool decided = false;
    CenterOutTrial dense_trial{};
    for (std::uint64_t tick = 1; tick <= 1000; ++tick)
    {
        CHECK(dense.step(tick * kMs, kNowhere, result) == ContractStatus::ok);
        if (result.trial_decided)
        {
            CHECK(!decided);
            decided = true;
            dense_trial = result.trial;
        }
    }
    CHECK(decided);
    CHECK(dense_trial.record.interval.end_ns == sparse_trial.record.interval.end_ns);
    CHECK(dense_trial.record.outcome == sparse_trial.record.outcome);
    CHECK(dense.state() == sparse.state());
    return failures - before;
}

int check_idempotence()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();
    CenterOutStepResult result{};

    CenterOutMachine machine{};
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(machine.step(100 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.step(200 * kMs, kCentre, result) == ContractStatus::ok);
    CHECK(machine.step(250 * kMs, kCentre, result) == ContractStatus::ok);
    const std::uint8_t idx = machine.snapshot().outward_idx;
    CHECK(machine.step(300 * kMs, outward_position(config, idx), result) == ContractStatus::ok);
    CHECK(machine.step(400 * kMs, outward_position(config, idx), result) == ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(result.settled);

    // Repeating a settled step at the same instant with the same observation
    // produces nothing at all -- no transition, no event, and above all no
    // second copy of the trial that was just decided.
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        CenterOutStepResult again{};
        CHECK(machine.step(400 * kMs, outward_position(config, idx), again) == ContractStatus::ok);
        CHECK(again.n_transitions == 0);
        CHECK(again.n_events == 0);
        CHECK(!again.trial_decided);
        CHECK(again.settled);
        CHECK(again.snapshot.completed == 1);
        CHECK(again.snapshot.state == CenterOutState::out_success_dwell);
    }
    return failures - before;
}

// One scripted observation of a session.
struct Observation
{
    ExperimentTimeNs time_ns;
    WorkspacePoint cursor;
};

// Everything one run emitted, recorded for comparison.
struct Recording
{
    std::size_t transitions{};
    std::array<StateTransition, 256> transition{};
    std::size_t trials{};
    std::array<CenterOutTrial, 64> trial{};
    CenterOutSnapshot final_snapshot{};
};

bool same_transition(const StateTransition& left, const StateTransition& right) noexcept
{
    return left.time_ns == right.time_ns && left.sequence == right.sequence &&
           left.trial.ordinal == right.trial.ordinal &&
           left.trial.target_id == right.trial.target_id && left.paradigm == right.paradigm &&
           left.from_state == right.from_state && left.to_state == right.to_state &&
           left.cause == right.cause;
}

bool same_trial_record(const CenterOutTrial& left, const CenterOutTrial& right) noexcept
{
    return left.record.trial.ordinal == right.record.trial.ordinal &&
           left.record.interval.start_ns == right.record.interval.start_ns &&
           left.record.interval.end_ns == right.record.interval.end_ns &&
           left.record.outcome == right.record.outcome &&
           left.record.reason == right.record.reason &&
           left.outward_target == right.outward_target && left.outward_idx == right.outward_idx &&
           left.decided_phase == right.decided_phase &&
           left.center_acquire_ns == right.center_acquire_ns &&
           left.outward_acquire_ns == right.outward_acquire_ns;
}

// Drives a machine over @p script, stepping until each observation is settled so
// that a backlog is drained rather than carried into the next observation.
void play(CenterOutMachine& machine, const CenterOut2DConfig& config,
          std::span<const Observation> script, Recording& recording)
{
    CenterOutStepResult result{};
    if (machine.start(kParadigm, config, script.empty() ? 0 : script[0].time_ns, result) !=
        ContractStatus::ok)
    {
        std::cerr << "FAILED: could not start the scripted machine\n";
        ++failures;
        return;
    }
    for (std::uint8_t i = 0; i < result.n_transitions; ++i)
        recording.transition[recording.transitions++] = result.transitions[i];

    for (const Observation& observation : script)
    {
        for (int drain = 0; drain < 8; ++drain)
        {
            if (machine.complete())
                break;
            if (machine.step(observation.time_ns, observation.cursor, result) != ContractStatus::ok)
            {
                std::cerr << "FAILED: scripted step refused\n";
                ++failures;
                return;
            }
            for (std::uint8_t i = 0; i < result.n_transitions; ++i)
                recording.transition[recording.transitions++] = result.transitions[i];
            if (result.trial_decided)
                recording.trial[recording.trials++] = result.trial;
            if (result.settled)
                break;
        }
    }
    recording.final_snapshot = machine.snapshot();
}

// A cursor that sits at the centre, then at the outward target, alternating on a
// fixed period. Piecewise constant, so the observations at any two cadences that
// both include the breakpoints agree.
WorkspacePoint scripted_cursor(const CenterOut2DConfig& config, ExperimentTimeNs time_ns) noexcept
{
    const std::uint64_t slot = time_ns / (400 * kMs);
    if (slot % 2 == 0)
        return kCentre;
    return outward_position(config, static_cast<std::uint8_t>(slot % config.layout.count));
}

int check_determinism_and_cadence()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config(100 * kMs, 50 * kMs, 6);

    // The same script twice, on two machines, produces the same records.
    static std::array<Observation, 1200> dense_script{};
    for (std::size_t i = 0; i < dense_script.size(); ++i)
    {
        const ExperimentTimeNs time_ns = static_cast<ExperimentTimeNs>(i) * 10 * kMs;
        dense_script[i] = Observation{time_ns, scripted_cursor(config, time_ns)};
    }

    static Recording first{};
    static Recording second{};
    CenterOutMachine left{};
    CenterOutMachine right{};
    play(left, config, dense_script, first);
    play(right, config, dense_script, second);
    CHECK(first.transitions == second.transitions);
    CHECK(first.trials == second.trials);
    CHECK(first.trials > 0);
    for (std::size_t i = 0; i < first.transitions && i < second.transitions; ++i)
        CHECK(same_transition(first.transition[i], second.transition[i]));
    for (std::size_t i = 0; i < first.trials && i < second.trials; ++i)
        CHECK(same_trial_record(first.trial[i], second.trial[i]));

    // Reset and replay: a machine returned to idle and started again reproduces
    // its own run exactly.
    static Recording replayed{};
    left.reset();
    CHECK(left.state() == CenterOutState::idle);
    play(left, config, dense_script, replayed);
    CHECK(replayed.transitions == first.transitions);
    CHECK(replayed.trials == first.trials);
    for (std::size_t i = 0; i < first.trials && i < replayed.trials; ++i)
        CHECK(same_trial_record(first.trial[i], replayed.trial[i]));

    // A sparser cadence over the same piecewise-constant trajectory, whose
    // observation instants are a subset that still lands on every breakpoint,
    // decides the same trials. The machine is driven by elapsed time, so the
    // observations in between changed nothing.
    static std::array<Observation, 240> sparse_script{};
    for (std::size_t i = 0; i < sparse_script.size(); ++i)
    {
        const ExperimentTimeNs time_ns = static_cast<ExperimentTimeNs>(i) * 50 * kMs;
        sparse_script[i] = Observation{time_ns, scripted_cursor(config, time_ns)};
    }
    static Recording sparse{};
    CenterOutMachine coarse{};
    play(coarse, config, sparse_script, sparse);
    CHECK(sparse.trials == first.trials);
    for (std::size_t i = 0; i < first.trials && i < sparse.trials; ++i)
        CHECK(same_trial_record(first.trial[i], sparse.trial[i]));

    // The session stopped at its configured limit.
    CHECK(first.final_snapshot.state == CenterOutState::complete);
    CHECK(first.final_snapshot.completed == 6);

    // Every outward target is the one the M8-04 schedule names, and never the
    // centre.
    for (std::size_t i = 0; i < first.trials; ++i)
    {
        std::uint8_t expected = 0;
        std::uint64_t successes = 0;
        for (std::size_t earlier = 0; earlier < i; ++earlier)
            if (first.trial[earlier].record.outcome == TrialOutcome::success)
                ++successes;
        CHECK(select_outward_target(config, static_cast<TrialOrdinal>(i), successes, expected) ==
              ContractStatus::ok);
        CHECK(first.trial[i].outward_idx == expected);
        CHECK(first.trial[i].outward_target != config.layout.center.id);
    }

    // Statistics are a function of the decided trials, not a running total the
    // machine kept beside them.
    CenterOutStatistics statistics{};
    CHECK(summarize(std::span<const CenterOutTrial>(first.trial.data(), first.trials),
                    statistics) == ContractStatus::ok);
    CHECK(statistics.decided == first.final_snapshot.completed);
    CHECK(statistics.successes == first.final_snapshot.successes);
    CHECK(statistics.decided ==
          statistics.successes + statistics.center_timeouts + statistics.outward_timeouts);
    if (statistics.successes != 0)
        CHECK(statistics.mean_time_to_target_ns ==
              statistics.total_time_to_target_ns / statistics.successes);
    else
        CHECK(statistics.mean_time_to_target_ns == 0);

    // A finished session refuses to step rather than quietly doing nothing.
    CenterOutStepResult tail{};
    CHECK(left.step(1'000'000 * kMs, kCentre, tail) == ContractStatus::not_running);
    return failures - before;
}

int check_summary_edges()
{
    const int before = failures;
    CenterOutStatistics statistics{};
    CHECK(summarize(std::span<const CenterOutTrial>{}, statistics) == ContractStatus::ok);
    CHECK(statistics.decided == 0);
    CHECK(statistics.mean_time_to_target_ns == 0);

    // Hand-built trials must be valid decided Center-Out trials: each is one of
    // the three field combinations validate(const CenterOutTrial&) accepts, with
    // the record's target identifier equal to the outward target, and with the
    // acquisition durations consistent with the interval.
    const TargetId target = 2;
    auto make_trial = [&](TrialOutcome outcome, CenterOutReason reason, CenterOutPhase phase,
                          ExperimentTimeNs interval_end, DurationNs center_acquire_ns,
                          DurationNs outward_acquire_ns)
    {
        CenterOutTrial t{};
        t.record.paradigm = kParadigm;
        t.record.trial.target_id = target;
        t.record.interval = TimeInterval{0, interval_end};
        t.record.outcome = outcome;
        t.record.reason = static_cast<std::uint32_t>(reason);
        t.outward_target = target;
        t.outward_idx = 0;
        t.decided_phase = phase;
        t.center_acquire_ns = center_acquire_ns;
        t.outward_acquire_ns = outward_acquire_ns;
        return t;
    };
    // Each interval covers the acquisition durations it reports, so the trials
    // are the same as machine-produced ones and not just records that happen to
    // pass a field-by-field check.
    std::array<CenterOutTrial, 3> trials{
        make_trial(TrialOutcome::success, CenterOutReason::outward_acquired, CenterOutPhase::to_out,
                   100, 0, 100),
        make_trial(TrialOutcome::success, CenterOutReason::outward_acquired, CenterOutPhase::to_out,
                   201, 0, 201),
        make_trial(TrialOutcome::timeout, CenterOutReason::center_movement_timeout,
                   CenterOutPhase::to_center, 50, 0, 0),
    };
    CHECK(summarize(std::span<const CenterOutTrial>(trials), statistics) == ContractStatus::ok);
    CHECK(statistics.decided == 3);
    CHECK(statistics.successes == 2);
    CHECK(statistics.center_timeouts == 1);
    CHECK(statistics.outward_timeouts == 0);
    CHECK(statistics.total_time_to_target_ns == 301);
    // Truncated, and exactly so: no floating-point running mean anywhere.
    CHECK(statistics.mean_time_to_target_ns == 150);
    return failures - before;
}

int check_summarize_rejects_invalid_trials()
{
    const int before = failures;
    const TargetId target = 2;
    auto make_trial = [&](TrialOutcome outcome, CenterOutReason reason, CenterOutPhase phase,
                          TargetId outward, TargetId record_target, bool empty_interval)
    {
        CenterOutTrial t{};
        t.record.paradigm = kParadigm;
        t.record.trial.target_id = record_target;
        t.record.interval =
            TimeInterval{0, empty_interval ? ExperimentTimeNs{0} : ExperimentTimeNs{1}};
        t.record.outcome = outcome;
        t.record.reason = static_cast<std::uint32_t>(reason);
        t.outward_target = outward;
        t.decided_phase = phase;
        return t;
    };
    auto check_one = [&](const CenterOutTrial& t, ContractStatus expected)
    {
        CenterOutStatistics stats{};
        CHECK(summarize(std::span<const CenterOutTrial>(&t, 1), stats) == expected);
        // A refused summary leaves the statistics untouched.
        CHECK(stats.decided == 0);
    };

    // An undeclared outcome is caught by the shared record validator.
    check_one(make_trial(static_cast<TrialOutcome>(255), CenterOutReason::unspecified,
                         CenterOutPhase::to_out, target, target, false),
              ContractStatus::enum_undeclared);
    // An unset outward target.
    check_one(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                         CenterOutPhase::to_out, kUnsetTargetId, target, false),
              ContractStatus::identity_missing);
    // A record target identifier that disagrees with the outward target.
    check_one(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                         CenterOutPhase::to_out, target, 3, false),
              ContractStatus::outcome_invalid);
    // A pending outcome is not a decided trial; the empty interval keeps the
    // record itself valid so the combination check is what refuses it.
    check_one(make_trial(TrialOutcome::pending, CenterOutReason::unspecified,
                         CenterOutPhase::to_out, target, target, true),
              ContractStatus::outcome_invalid);
    // A success on the wrong leg.
    check_one(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                         CenterOutPhase::to_center, target, target, false),
              ContractStatus::outcome_invalid);
    // A timeout with the wrong reason for its leg.
    check_one(make_trial(TrialOutcome::timeout, CenterOutReason::outward_movement_timeout,
                         CenterOutPhase::to_center, target, target, false),
              ContractStatus::outcome_invalid);

    // A valid trial is still summarized.
    {
        CenterOutStatistics stats{};
        const CenterOutTrial t =
            make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                       CenterOutPhase::to_out, target, target, false);
        CHECK(summarize(std::span<const CenterOutTrial>(&t, 1), stats) == ContractStatus::ok);
        CHECK(stats.decided == 1);
        CHECK(stats.successes == 1);
    }
    return failures - before;
}

int check_summary_rejects_acquisition_and_index()
{
    const int before = failures;
    const TargetId target = 2;
    // A fully-parameterised builder: every duration field is set explicitly so a
    // rejected combination is the one under test, not a default that happened to
    // disagree. The interval end is the trial's own duration.
    auto make_trial = [&](TrialOutcome outcome, CenterOutReason reason, CenterOutPhase phase,
                          ExperimentTimeNs interval_end, DurationNs center_acquire_ns,
                          DurationNs outward_acquire_ns, std::uint8_t outward_idx)
    {
        CenterOutTrial t{};
        t.record.paradigm = kParadigm;
        t.record.trial.target_id = target;
        t.record.interval = TimeInterval{0, interval_end};
        t.record.outcome = outcome;
        t.record.reason = static_cast<std::uint32_t>(reason);
        t.outward_target = target;
        t.outward_idx = outward_idx;
        t.decided_phase = phase;
        t.center_acquire_ns = center_acquire_ns;
        t.outward_acquire_ns = outward_acquire_ns;
        return t;
    };
    auto rejected = [&](const CenterOutTrial& t, ContractStatus expected)
    {
        CenterOutStatistics stats{};
        CHECK(summarize(std::span<const CenterOutTrial>(&t, 1), stats) == expected);
        // A refused summary leaves the statistics untouched.
        CHECK(stats.decided == 0);
    };
    auto accepted = [&](const CenterOutTrial& t)
    {
        CenterOutStatistics stats{};
        CHECK(summarize(std::span<const CenterOutTrial>(&t, 1), stats) == ContractStatus::ok);
        CHECK(stats.decided == 1);
    };

    // A success whose outward acquisition outlasts the trial it is reported in:
    // an interval of 100 ns cannot contain a 101 ns time to target.
    rejected(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 0, 101, 0),
             ContractStatus::outcome_invalid);
    // A success whose centre acquisition outlasts the trial.
    rejected(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 101, 100, 0),
             ContractStatus::outcome_invalid);
    // A centre timeout that nonetheless reports a centre acquisition: the centre
    // never acquired, so the duration is meaningless and must be zero.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::center_movement_timeout,
                        CenterOutPhase::to_center, 1000, 1, 0, 0),
             ContractStatus::outcome_invalid);
    // A centre timeout that reports an outward acquisition.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::center_movement_timeout,
                        CenterOutPhase::to_center, 1000, 0, 1, 0),
             ContractStatus::outcome_invalid);
    // An outward timeout that reports an outward acquisition: the outward leg
    // never acquired, so the duration is meaningless and must be zero.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::outward_movement_timeout,
                        CenterOutPhase::to_out, 1000, 200, 1, 0),
             ContractStatus::outcome_invalid);
    // An outward timeout whose centre acquisition outlasts the trial.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::outward_movement_timeout,
                        CenterOutPhase::to_out, 100, 101, 0, 0),
             ContractStatus::outcome_invalid);
    // An outward timeout whose centre acquisition fills the whole trial: the
    // outward movement window is strictly positive, so the trial must end
    // strictly after the centre was acquired.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::outward_movement_timeout,
                        CenterOutPhase::to_out, 100, 100, 0, 0),
             ContractStatus::outcome_invalid);
    // An outward timeout cannot be a zero-length trial either.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::outward_movement_timeout,
                        CenterOutPhase::to_out, 0, 0, 0, 0),
             ContractStatus::outcome_invalid);
    // A centre timeout cannot be a zero-length trial: the centre movement window
    // is strictly positive, so the machine always runs it before it times out.
    rejected(make_trial(TrialOutcome::timeout, CenterOutReason::center_movement_timeout,
                        CenterOutPhase::to_center, 0, 0, 0, 0),
             ContractStatus::outcome_invalid);
    // An outward index no layout can hold: 16 is at the capacity, not inside it.
    rejected(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 0, 100,
                        static_cast<std::uint8_t>(kMaxSurroundingTargets)),
             ContractStatus::outcome_invalid);

    // Cross-stage consistency on a success: the outward leg cannot begin before
    // the centre is acquired, so the two acquisitions must sum to no more than
    // the trial. A centre reward dwell only makes the trial longer. These two
    // each pass the per-duration bound yet fail the sum.
    rejected(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 50, 51, 0),
             ContractStatus::outcome_invalid);
    rejected(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 100, 1, 0),
             ContractStatus::outcome_invalid);

    // The boundary invariants admit the trials they should. A success may have
    // both acquisitions zero (a zero hold, an overlapping geometry), so only an
    // upper -- not a positive lower -- bound is enforced.
    accepted(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 1, 0, 0, 0));
    // A zero-duration success is admissible: zero hold, zero centre dwell, and a
    // geometry that lets both legs complete at the same instant.
    accepted(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 0, 0, 0, 0));
    // The two acquisitions sum exactly to the trial duration, each staying
    // inside it: the outward leg begins the instant the centre is acquired.
    accepted(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 50, 50, 0));
    // A centre acquisition equal to the trial duration, with the outward leg
    // completing at the same instant it began.
    accepted(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 100, 0, 0));
    // The last admissible surrounding index is one past the largest in use.
    accepted(make_trial(TrialOutcome::success, CenterOutReason::outward_acquired,
                        CenterOutPhase::to_out, 100, 0, 100,
                        static_cast<std::uint8_t>(kMaxSurroundingTargets - 1)));
    // An outward timeout keeps the centre it did acquire, ending strictly after.
    accepted(make_trial(TrialOutcome::timeout, CenterOutReason::outward_movement_timeout,
                        CenterOutPhase::to_out, 1000, 200, 0, 0));
    // A centre timeout runs for its strictly positive movement window.
    accepted(make_trial(TrialOutcome::timeout, CenterOutReason::center_movement_timeout,
                        CenterOutPhase::to_center, 1000, 0, 0, 0));
    return failures - before;
}

int check_start_lifecycle()
{
    const int before = failures;
    const CenterOut2DConfig config = make_config();

    // start() on a running session is refused and leaves the session exactly
    // where it was: no silent session replacement.
    CenterOutMachine machine{};
    CenterOutStepResult result{};
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(machine.step(100 * kMs, kCentre, result) == ContractStatus::ok);
    const CenterOutState state = machine.state();
    const TrialOrdinal completed = machine.snapshot().completed;
    const std::uint64_t successes = machine.snapshot().successes;
    CenterOutStepResult again{};
    CHECK(machine.start(kParadigm, config, 0, again) == ContractStatus::already_running);
    CHECK(machine.state() == state);
    CHECK(machine.snapshot().completed == completed);
    CHECK(machine.snapshot().successes == successes);
    // The refused call did not write the caller's result.
    CHECK(again.n_transitions == 0);
    CHECK(again.n_events == 0);
    CHECK(!again.trial_decided);

    // A completed session is refused too; a new one needs reset() first.
    CenterOutMachine finite{};
    const CenterOut2DConfig bounded = make_config(100 * kMs, 50 * kMs, 1);
    CHECK(finite.start(kParadigm, bounded, 0, result) == ContractStatus::ok);
    // The centre leg times out -- deciding the one allowed trial -- and the
    // step stops at that decision, leaving the failure dwell running.
    CHECK(finite.step(2000 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(finite.state() == CenterOutState::center_failure_dwell);
    // The dwell has already ended at this instant, so the next step ends it and
    // the trial limit completes the session.
    CHECK(finite.step(2000 * kMs, kNowhere, result) == ContractStatus::ok);
    CHECK(finite.complete());
    CHECK(finite.state() == CenterOutState::complete);
    CHECK(finite.start(kParadigm, bounded, 0, result) == ContractStatus::already_running);
    finite.reset();
    CHECK(finite.state() == CenterOutState::idle);
    CHECK(finite.start(kParadigm, bounded, 0, result) == ContractStatus::ok);
    CHECK(finite.state() == CenterOutState::move_to_center);
    return failures - before;
}

int run()
{
    // A first session faults in whatever the standard library initializes
    // lazily; the sessions after this are what the count is about.
    {
        CenterOutMachine warmup{};
        CenterOutStepResult result{};
        (void)warmup.start(kParadigm, make_config(), 0, result);
        (void)warmup.step(kMs, kCentre, result);
    }
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);

    if (const int failure = check_lifecycle(); failure != 0)
        return failure;
    if (const int failure = check_start_lifecycle(); failure != 0)
        return failure;
    if (const int failure = check_successful_cycle(); failure != 0)
        return failure;
    if (const int failure = check_failures(); failure != 0)
        return failure;
    if (const int failure = check_hold_reset(); failure != 0)
        return failure;
    if (const int failure = check_boundary_precedence(); failure != 0)
        return failure;
    if (const int failure = check_elapsed_time_not_invocations(); failure != 0)
        return failure;
    if (const int failure = check_idempotence(); failure != 0)
        return failure;
    if (const int failure = check_determinism_and_cadence(); failure != 0)
        return failure;
    if (const int failure = check_summary_edges(); failure != 0)
        return failure;
    if (const int failure = check_summarize_rejects_invalid_trials(); failure != 0)
        return failure;
    if (const int failure = check_summary_rejects_acquisition_and_index(); failure != 0)
        return failure;

    const std::size_t allocated = allocations.load(std::memory_order_relaxed) - baseline;
    if (allocated != 0)
    {
        std::cerr << "FAILED: the center-out machine allocated " << allocated << " times\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace
} // namespace machine_tests

int main()
{
    const int machine_status = machine_tests::run();
    return run() == 0 && machine_status == 0 ? 0 : 1;
}
