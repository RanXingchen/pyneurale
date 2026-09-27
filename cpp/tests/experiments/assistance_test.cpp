/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/assistance.h>
#include <neurale/experiments/command.h>
#include <neurale/experiments/contract.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "allocation_counter.h"
#include "check_counts.h"

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::assistance;

// Assistance carries values, not payloads. A record that stopped being
// trivially copyable would have grown owning storage, and a transform that
// took one would no longer be safe to run where it is meant to run.
static_assert(std::is_trivially_copyable_v<VelocityVector>);
static_assert(std::is_trivially_copyable_v<DesiredVelocitySet>);
static_assert(std::is_trivially_copyable_v<LinearAssistance>);
static_assert(std::is_trivially_copyable_v<OrthoImpedanceParameters>);
static_assert(std::is_trivially_copyable_v<LinearAssistanceRecord>);
static_assert(std::is_trivially_copyable_v<OrthoImpedanceRecord>);
static_assert(std::is_standard_layout_v<VelocityVector>);

// A velocity is one command's worth of doubles and its identity, and a trace
// record is at most three of those. These bounds are stated so that a record
// growing a fourth vector, or a string, is a decision somebody makes here.
static_assert(sizeof(VelocityVector) <= 80);
static_assert(sizeof(LinearAssistanceRecord) <= 288);
static_assert(sizeof(OrthoImpedanceRecord) <= 288);

int failures = 0;

bool close(double left, double right) noexcept
{
    return std::fabs(left - right) <= 1e-12 * (1.0 + std::fabs(right));
}

CommandSpace velocity_space(std::uint8_t dim) noexcept
{
    CommandSpace space{};
    space.id = 1;
    space.dim = dim;
    space.frame = CommandFrame::workspace_2d;
    constexpr std::array<CommandAxisName, kMaxCommandDim> names{
        CommandAxisName::x,     CommandAxisName::y,     CommandAxisName::z,
        CommandAxisName::roll,  CommandAxisName::pitch, CommandAxisName::yaw,
        CommandAxisName::grasp, CommandAxisName::grasp};
    for (std::size_t axis = 0; axis < dim; ++axis)
        space.axes[axis] = CommandAxis{names[axis], CommandUnit::metres_per_second};
    return space;
}

VelocityVector velocity(std::uint8_t dim, std::initializer_list<double> values) noexcept
{
    VelocityVector result{};
    result.space = 1;
    result.dim = dim;
    std::size_t axis = 0;
    for (const double value : values)
        result.values[axis++] = value;
    return result;
}

int check_velocity_values()
{
    const CommandSpace space = velocity_space(2);

    VelocityVector unset{};
    CHECK(validate(unset) == ContractStatus::identity_missing);

    VelocityVector command = velocity(2, {0.25, -0.5});
    CHECK(validate(command) == ContractStatus::ok);
    CHECK(validate_against(command, space) == ContractStatus::ok);

    // The same rules the command records state, for the same reason: an unused
    // slot is exactly zero so equal velocities have equal bytes.
    command.values[4] = 1.0;
    CHECK(validate(command) == ContractStatus::dimension_invalid);
    command.values[4] = 0.0;

    command.values[1] = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(command) == ContractStatus::value_not_finite);
    command.values[1] = std::numeric_limits<double>::infinity();
    CHECK(validate(command) == ContractStatus::value_not_finite);
    command.values[1] = -0.5;

    command.dim = 0;
    CHECK(validate(command) == ContractStatus::dimension_invalid);
    command.dim = static_cast<std::uint8_t>(kMaxCommandDim + 1);
    CHECK(validate(command) == ContractStatus::dimension_invalid);
    command.dim = 3;
    CHECK(validate_against(command, space) == ContractStatus::dimension_invalid);
    command.dim = 2;
    command.space = 2;
    CHECK(validate_against(command, space) == ContractStatus::identity_missing);
    return 0;
}

int check_desired_set()
{
    const CommandSpace space = velocity_space(2);

    DesiredVelocitySet desired{};
    CHECK(validate(desired) == ContractStatus::identity_missing);
    desired.space = 1;
    desired.dim = 2;
    CHECK(validate(desired) == ContractStatus::dimension_invalid);
    desired.count = 1;
    desired.vectors[0][0] = 1.0;
    CHECK(validate(desired) == ContractStatus::ok);
    CHECK(validate_against(desired, space) == ContractStatus::ok);

    // A row past `count` is a row nobody supplied, not a direction that happens
    // to be zero, so the two do not share a representation.
    desired.vectors[1][0] = 1.0;
    CHECK(validate(desired) == ContractStatus::dimension_invalid);
    desired.vectors[1][0] = 0.0;
    CHECK(validate(desired) == ContractStatus::ok);

    desired.count = static_cast<std::uint8_t>(kMaxDesiredVelocities + 1);
    CHECK(validate(desired) == ContractStatus::dimension_invalid);
    desired.count = 1;

    desired.vectors[0][1] = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(desired) == ContractStatus::value_not_finite);
    return 0;
}

int check_parameters()
{
    const CommandSpace space = velocity_space(2);

    LinearAssistance blend{};
    CHECK(validate(blend) == ContractStatus::ok);
    for (const double outside : {-1e-9, 1.0 + 1e-9, std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity()})
    {
        blend.assistance = outside;
        CHECK(validate(blend) == ContractStatus::assistance_out_of_range);
    }

    OrthoImpedanceParameters parameters{};
    CHECK(validate(parameters) == ContractStatus::dimension_invalid);
    parameters.dim = 2;
    // Fixturing no axis is a configuration mistake, not a way of asking for
    // nothing to happen.
    CHECK(validate(parameters) == ContractStatus::domain_mask_invalid);
    parameters.domain[0] = true;
    CHECK(validate(parameters) == ContractStatus::ok);
    parameters.domain[5] = true;
    CHECK(validate(parameters) == ContractStatus::domain_mask_invalid);
    parameters.domain[5] = false;

    parameters.impedance[1] = 1.5;
    CHECK(validate(parameters) == ContractStatus::assistance_out_of_range);
    parameters.impedance[1] = 0.25;
    CHECK(validate(parameters) == ContractStatus::ok);
    parameters.impedance[6] = 0.25;
    CHECK(validate(parameters) == ContractStatus::dimension_invalid);
    parameters.impedance[6] = 0.0;

    // A negative tolerance is finite, so it is parameter_out_of_range, not
    // value_not_finite; only a NaN or infinite value is value_not_finite.
    parameters.rank_tol = -1.0;
    CHECK(validate(parameters) == ContractStatus::parameter_out_of_range);
    parameters.rank_tol = 1e-8;
    parameters.conditioning_tol = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(parameters) == ContractStatus::value_not_finite);
    parameters.conditioning_tol = 5e-6;
    // conditioning_tol is a fraction of the largest eigenvalue, so it
    // lies in [0, 1]: one keeps only the largest-eigenvalue eigenspace, zero
    // disables the relative cut, and anything above one is refused because it
    // would drop every eigenvalue and zero the projector.
    parameters.conditioning_tol = 1.0;
    CHECK(validate(parameters) == ContractStatus::ok);
    parameters.conditioning_tol = 0.0;
    CHECK(validate(parameters) == ContractStatus::ok);
    parameters.conditioning_tol = 1.0 + 1e-9;
    CHECK(validate(parameters) == ContractStatus::parameter_out_of_range);
    parameters.conditioning_tol = -1e-9;
    CHECK(validate(parameters) == ContractStatus::parameter_out_of_range);
    parameters.conditioning_tol = 5e-6;

    parameters.dim = 3;
    CHECK(validate_against(parameters, space) == ContractStatus::dimension_invalid);
    parameters.dim = 2;
    CHECK(validate_against(parameters, space) == ContractStatus::ok);
    return 0;
}

int check_linear_blend()
{
    const CommandSpace space = velocity_space(2);
    const VelocityVector external = velocity(2, {0.25, -0.5});
    const VelocityVector guidance = velocity(2, {1.0, 0.0});

    VelocityVector assisted{};
    // The endpoints are the identity on the stored values, not an approximation
    // of it: a caller asking for no assistance gets its command back exactly.
    CHECK(blend_velocity(space, external, guidance, LinearAssistance{0.0}, assisted) ==
          ContractStatus::ok);
    CHECK(assisted.values == external.values);
    CHECK(blend_velocity(space, external, guidance, LinearAssistance{1.0}, assisted) ==
          ContractStatus::ok);
    CHECK(assisted.values == guidance.values);

    CHECK(blend_velocity(space, external, guidance, LinearAssistance{0.25}, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.75 * 0.25 + 0.25 * 1.0));
    CHECK(close(assisted.values[1], 0.75 * -0.5));
    CHECK(assisted.space == space.id && assisted.dim == space.dim);

    // Nothing is clipped, normalized, or remembered: a second call with the
    // same arguments produces the same bytes, and a guidance vector larger than
    // any bound comes through at its own size.
    VelocityVector again{};
    CHECK(blend_velocity(space, external, guidance, LinearAssistance{0.25}, again) ==
          ContractStatus::ok);
    CHECK(again.values == assisted.values);
    const VelocityVector large = velocity(2, {1e6, -1e6});
    CHECK(blend_velocity(space, external, large, LinearAssistance{1.0}, assisted) ==
          ContractStatus::ok);
    CHECK(assisted.values[0] == 1e6);

    // A rejected call leaves the caller's output untouched rather than writing
    // half an answer into it.
    const VelocityVector marker = velocity(2, {7.0, 7.0});
    assisted = marker;
    CHECK(blend_velocity(space, external, guidance, LinearAssistance{1.5}, assisted) ==
          ContractStatus::assistance_out_of_range);
    CHECK(assisted.values == marker.values);

    VelocityVector mismatched = guidance;
    mismatched.space = 2;
    CHECK(blend_velocity(space, external, mismatched, LinearAssistance{0.5}, assisted) ==
          ContractStatus::identity_missing);
    mismatched = guidance;
    mismatched.dim = 1;
    mismatched.values[1] = 0.0;
    CHECK(blend_velocity(space, external, mismatched, LinearAssistance{0.5}, assisted) ==
          ContractStatus::dimension_invalid);
    return 0;
}

int check_ortho_impedance()
{
    const CommandSpace space = velocity_space(2);

    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 2;
    desired.count = 1;
    desired.vectors[0][0] = 1.0;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 2;
    parameters.domain[0] = true;
    parameters.domain[1] = true;

    const VelocityVector external = velocity(2, {0.6, 0.8});
    VelocityVector assisted{};

    // Zero impedance leaves the orthogonal remainder alone, so the transform is
    // the identity however the manifold lies.
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.6) && close(assisted.values[1], 0.8));

    // Full impedance removes everything orthogonal to the manifold, leaving the
    // projection onto it.
    parameters.impedance[0] = 1.0;
    parameters.impedance[1] = 1.0;
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.6) && close(assisted.values[1], 0.0));

    // Half impedance halves the orthogonal component and leaves the parallel
    // one untouched.
    parameters.impedance[0] = 0.5;
    parameters.impedance[1] = 0.5;
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.6) && close(assisted.values[1], 0.4));

    // With nothing to project onto, every component is orthogonal and impedance
    // is all that is left to apply. A command opposing the only direction and a
    // manifold of zeros are the same case.
    parameters.impedance[0] = 1.0;
    parameters.impedance[1] = 1.0;
    const VelocityVector opposed = velocity(2, {-0.6, 0.8});
    CHECK(apply_ortho_impedance(space, desired, opposed, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.0) && close(assisted.values[1], 0.0));

    DesiredVelocitySet empty = desired;
    empty.vectors[0][0] = 0.0;
    CHECK(apply_ortho_impedance(space, empty, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.0) && close(assisted.values[1], 0.0));

    // A zero command stays zero whatever the manifold says.
    const VelocityVector still = velocity(2, {0.0, 0.0});
    CHECK(apply_ortho_impedance(space, desired, still, parameters, assisted) == ContractStatus::ok);
    CHECK(assisted.values[0] == 0.0 && assisted.values[1] == 0.0);

    // A rank-deficient manifold projects onto the span it actually has rather
    // than onto the number of rows it was given.
    DesiredVelocitySet repeated = desired;
    repeated.count = 3;
    repeated.vectors[1][0] = 2.0;
    repeated.vectors[2][0] = 0.5;
    CHECK(apply_ortho_impedance(space, repeated, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.6) && close(assisted.values[1], 0.0));

    // Repeated calls are deterministic: the transform holds no state.
    VelocityVector again{};
    CHECK(apply_ortho_impedance(space, repeated, external, parameters, again) ==
          ContractStatus::ok);
    CHECK(again.values == assisted.values);
    return 0;
}

int check_domain_ordering()
{
    // Three axes with only the outer two fixtured; the middle axis stays in place.
    const CommandSpace space = velocity_space(3);

    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 3;
    desired.count = 1;
    desired.vectors[0][0] = 1.0;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 3;
    parameters.domain[0] = true;
    parameters.domain[2] = true;
    parameters.impedance[0] = 1.0;
    parameters.impedance[2] = 1.0;

    const VelocityVector external = velocity(3, {0.6, 5.0, 0.8});
    VelocityVector assisted{};
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(close(assisted.values[0], 0.6));
    // Untouched, and untouched *in place*.
    CHECK(assisted.values[1] == 5.0);
    CHECK(close(assisted.values[2], 0.0));

    // The manifold's component on an axis outside the domain takes no part.
    DesiredVelocitySet leaning = desired;
    leaning.vectors[0][1] = 100.0;
    VelocityVector unchanged{};
    CHECK(apply_ortho_impedance(space, leaning, external, parameters, unchanged) ==
          ContractStatus::ok);
    CHECK(unchanged.values == assisted.values);
    return 0;
}

int check_inputs_are_not_mutated()
{
    const CommandSpace space = velocity_space(2);
    CommandSpace space_copy = space;

    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 2;
    desired.count = 2;
    desired.vectors[0][0] = 1.0;
    desired.vectors[1][1] = 1.0;
    const DesiredVelocitySet desired_copy = desired;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 2;
    parameters.domain[0] = true;
    parameters.domain[1] = true;
    parameters.impedance[0] = 0.75;
    const OrthoImpedanceParameters parameters_copy = parameters;

    const VelocityVector external = velocity(2, {0.25, -0.5});
    VelocityVector external_copy = external;
    VelocityVector assisted{};

    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(desired.vectors == desired_copy.vectors && desired.count == desired_copy.count);
    CHECK(parameters.impedance == parameters_copy.impedance &&
          parameters.domain == parameters_copy.domain);
    CHECK(external.values == external_copy.values);
    CHECK(space.dim == space_copy.dim && space.frame == space_copy.frame);
    for (std::size_t axis = 0; axis < kMaxCommandDim; ++axis)
        CHECK(space.axes[axis].name == space_copy.axes[axis].name &&
              space.axes[axis].unit == space_copy.axes[axis].unit);
    return 0;
}

int check_ortho_impedance_near_collinear()
{
    // Two nearly collinear desired directions make the span's Gram matrix
    // ill-conditioned: one eigenvalue is near zero. The conditioning cut must
    // drop that small eigenvalue and keep the dominant direction, not the
    // reverse. With full impedance the command, which lies in the span, comes
    // back essentially unchanged; the wrong-direction cut annihilated it.
    const CommandSpace space = velocity_space(2);

    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 2;
    desired.count = 2;
    desired.vectors[0][0] = 1.0;
    desired.vectors[1][0] = 1.0;
    desired.vectors[1][1] = 1e-3;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 2;
    parameters.domain[0] = true;
    parameters.domain[1] = true;
    parameters.impedance[0] = 1.0;
    parameters.impedance[1] = 1.0;

    const VelocityVector external = velocity(2, {2.0, 1e-3});
    VelocityVector assisted{};
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(std::fabs(assisted.values[0] - 2.0) < 1e-6);
    CHECK(std::fabs(assisted.values[1] - 1e-3) < 1e-6);
    return 0;
}

int check_ortho_impedance_max_dimension()
{
    // Eight orthonormal desired directions span the whole 8D space, so nothing
    // is orthogonal to the manifold and the projector is the identity: full
    // impedance returns the command unchanged. This also exercises the largest
    // manifold the contract allows (count = dimension = 8).
    const CommandSpace space = velocity_space(8);

    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 8;
    desired.count = 8;
    for (std::size_t row = 0; row < 8; ++row)
        desired.vectors[row][row] = 1.0;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 8;
    for (std::size_t axis = 0; axis < 8; ++axis)
    {
        parameters.domain[axis] = true;
        parameters.impedance[axis] = 1.0;
    }

    const VelocityVector external = velocity(8, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    VelocityVector assisted{};
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    for (std::size_t axis = 0; axis < 8; ++axis)
        CHECK(close(assisted.values[axis], static_cast<double>(axis + 1)));
    return 0;
}

int check_ortho_impedance_rank_tolerance_zero()
{
    // A zero rank_tol disables the absolute floor and a zero
    // conditioning_tol disables the relative cut. The near-collinear
    // manifold's tiny eigenvalue is then inverted with a large gain, but the
    // result must stay finite: a zero eigenvalue is never divided by, so no
    // infinity propagates. The span is still full-rank 2D, so the projector is
    // the identity and the command comes back.
    const CommandSpace space = velocity_space(2);

    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 2;
    desired.count = 2;
    desired.vectors[0][0] = 1.0;
    desired.vectors[1][0] = 1.0;
    desired.vectors[1][1] = 1e-3;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 2;
    parameters.domain[0] = true;
    parameters.domain[1] = true;
    parameters.impedance[0] = 1.0;
    parameters.impedance[1] = 1.0;
    parameters.rank_tol = 0.0;
    parameters.conditioning_tol = 0.0;

    const VelocityVector external = velocity(2, {2.0, 1e-3});
    VelocityVector assisted{};
    CHECK(apply_ortho_impedance(space, desired, external, parameters, assisted) ==
          ContractStatus::ok);
    CHECK(std::isfinite(assisted.values[0]));
    CHECK(std::isfinite(assisted.values[1]));
    CHECK(std::fabs(assisted.values[0] - 2.0) < 1e-6);
    CHECK(std::fabs(assisted.values[1] - 1e-3) < 1e-6);
    return 0;
}

int check_ortho_impedance_row_ordering()
{
    // Permuting the rows of a manifold does not change its span, so it must not
    // change the projection.
    const CommandSpace space = velocity_space(2);

    DesiredVelocitySet first{};
    first.space = 1;
    first.dim = 2;
    first.count = 2;
    first.vectors[0][0] = 1.0;
    first.vectors[1][1] = 1.0;

    DesiredVelocitySet second{};
    second.space = 1;
    second.dim = 2;
    second.count = 2;
    second.vectors[0][1] = 1.0;
    second.vectors[1][0] = 1.0;

    OrthoImpedanceParameters parameters{};
    parameters.dim = 2;
    parameters.domain[0] = true;
    parameters.domain[1] = true;
    parameters.impedance[0] = 0.5;
    parameters.impedance[1] = 0.25;

    const VelocityVector external = velocity(2, {0.6, 0.8});
    VelocityVector from_first{};
    VelocityVector from_second{};
    CHECK(apply_ortho_impedance(space, first, external, parameters, from_first) ==
          ContractStatus::ok);
    CHECK(apply_ortho_impedance(space, second, external, parameters, from_second) ==
          ContractStatus::ok);
    CHECK(from_first.values == from_second.values);
    return 0;
}

int check_records()
{
    LinearAssistanceRecord linear{};
    CHECK(validate(linear) == ContractStatus::identity_missing);
    linear.space = 1;
    linear.dim = 2;
    CHECK(validate(linear) == ContractStatus::ok);
    linear.method = AssistanceMethod::ortho_impedance;
    // A record that named the other method would be read as evidence of a step
    // that never ran.
    CHECK(validate(linear) == ContractStatus::outcome_invalid);
    linear.method = static_cast<AssistanceMethod>(200);
    CHECK(validate(linear) == ContractStatus::enum_undeclared);
    linear.method = AssistanceMethod::linear_blend;
    linear.version = 0;
    // Zero is no longer a special "unset" version: only the declared version is
    // replayable, so any other value -- zero included -- is version_unsupported.
    CHECK(validate(linear) == ContractStatus::version_unsupported);
    linear.version = 999;
    CHECK(validate(linear) == ContractStatus::version_unsupported);
    linear.version = kLinearBlendVersion1;
    linear.assistance = 1.5;
    CHECK(validate(linear) == ContractStatus::assistance_out_of_range);
    linear.assistance = 0.5;
    linear.guidance[3] = 1.0;
    CHECK(validate(linear) == ContractStatus::dimension_invalid);
    linear.guidance[3] = 0.0;
    CHECK(validate(linear) == ContractStatus::ok);

    OrthoImpedanceRecord ortho{};
    ortho.space = 1;
    ortho.dim = 2;
    ortho.domain[0] = true;
    CHECK(validate(ortho) == ContractStatus::ok);
    ortho.domain[0] = false;
    CHECK(validate(ortho) == ContractStatus::domain_mask_invalid);
    ortho.domain[0] = true;
    ortho.impedance[0] = 2.0;
    CHECK(validate(ortho) == ContractStatus::assistance_out_of_range);
    ortho.impedance[0] = 1.0;
    ortho.method = AssistanceMethod::linear_blend;
    CHECK(validate(ortho) == ContractStatus::outcome_invalid);
    ortho.method = AssistanceMethod::ortho_impedance;
    CHECK(validate(ortho) == ContractStatus::ok);
    ortho.version = 0;
    CHECK(validate(ortho) == ContractStatus::version_unsupported);
    ortho.version = 999;
    CHECK(validate(ortho) == ContractStatus::version_unsupported);
    ortho.version = kOrthoImpedanceVersion1;

    for (const AssistanceMethod method : {AssistanceMethod::none, AssistanceMethod::linear_blend,
                                          AssistanceMethod::ortho_impedance})
        CHECK(assistance_method_declared(method));
    CHECK(!assistance_method_declared(static_cast<AssistanceMethod>(3)));
    return 0;
}

int check_manifold_fingerprint()
{
    DesiredVelocitySet desired{};
    desired.space = 1;
    desired.dim = 2;
    desired.count = 1;
    desired.vectors[0][0] = 1.0;

    const std::uint64_t base = manifold_fingerprint(desired);
    CHECK(base == manifold_fingerprint(desired));

    DesiredVelocitySet moved = desired;
    moved.vectors[0][1] = 1.0;
    CHECK(manifold_fingerprint(moved) != base);

    DesiredVelocitySet counted = desired;
    counted.count = 2;
    CHECK(manifold_fingerprint(counted) != base);

    // Negative zero and zero are the same direction, so they are the same
    // manifold, so they fingerprint the same.
    DesiredVelocitySet signed_zero = desired;
    signed_zero.vectors[0][1] = -0.0;
    CHECK(manifold_fingerprint(signed_zero) == base);
    return 0;
}

int check_status_messages()
{
    for (const ContractStatus status :
         {ContractStatus::assistance_out_of_range, ContractStatus::domain_mask_invalid})
    {
        const char* message = contract_status_message(status);
        CHECK(message != nullptr);
        CHECK(message[0] != '\0');
    }
    CHECK(contract_status_message(ContractStatus::assistance_out_of_range) !=
          contract_status_message(ContractStatus::domain_mask_invalid));
    return 0;
}

int run()
{
    // Assistance runs on values only. An allocation here would mean some record
    // or workspace had grown owning storage, which is exactly what the bounded
    // path this header promises cannot afford.
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);

    if (const int failure = check_velocity_values(); failure != 0)
        return failure;
    if (const int failure = check_desired_set(); failure != 0)
        return failure;
    if (const int failure = check_parameters(); failure != 0)
        return failure;
    if (const int failure = check_linear_blend(); failure != 0)
        return failure;
    if (const int failure = check_ortho_impedance(); failure != 0)
        return failure;
    if (const int failure = check_ortho_impedance_near_collinear(); failure != 0)
        return failure;
    if (const int failure = check_ortho_impedance_max_dimension(); failure != 0)
        return failure;
    if (const int failure = check_ortho_impedance_rank_tolerance_zero(); failure != 0)
        return failure;
    if (const int failure = check_ortho_impedance_row_ordering(); failure != 0)
        return failure;
    if (const int failure = check_domain_ordering(); failure != 0)
        return failure;
    if (const int failure = check_inputs_are_not_mutated(); failure != 0)
        return failure;
    if (const int failure = check_records(); failure != 0)
        return failure;
    if (const int failure = check_manifold_fingerprint(); failure != 0)
        return failure;
    if (const int failure = check_status_messages(); failure != 0)
        return failure;

    const std::size_t allocated = allocations.load(std::memory_order_relaxed) - baseline;
    if (allocated != 0)
    {
        std::cerr << "FAILED: assistance allocated " << allocated << " times\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main()
{
    return run();
}
