/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>

/**
 * @file
 * @brief Semantic command requests and their synchronous application status.
 *
 * A command's dimension, axis names, units, and coordinate frame are explicit,
 * because generic velocity assistance transforms these values and a transform
 * that does not know what a number means cannot be reviewed. They are explicit
 * as enumerators rather than strings: a fixed-size record carries no text, and
 * nothing dispatches on a name.
 *
 * The names live in a CommandSpace prepared before the session. A realtime
 * CommandRequest refers to one by CommandSpaceId and carries only numbers.
 */
namespace neurale::experiments
{

/// Largest command dimension the fixed-size record holds.
inline constexpr std::size_t kMaxCommandDim = 8;

/// Identifier of a prepared CommandSpace. Zero means unset.
using CommandSpaceId = std::uint32_t;

/// Unset ::CommandSpaceId.
inline constexpr CommandSpaceId kUnsetCommandSpaceId = 0;

/// Coordinate frame the command values are expressed in.
enum class CommandFrame : std::uint8_t
{
    /// Unset.
    unspecified = 0,
    /// The paradigm's own 2D semantic workspace.
    workspace_2d,
    /// The paradigm's own 3D semantic workspace.
    workspace_3d,
    /// The receiving device's native frame.
    device_native,
};

/// Meaning of one command axis.
enum class CommandAxisName : std::uint8_t
{
    /// Unset.
    unspecified = 0,
    /// First workspace axis.
    x,
    /// Second workspace axis.
    y,
    /// Third workspace axis.
    z,
    /// Rotation about the first axis.
    roll,
    /// Rotation about the second axis.
    pitch,
    /// Rotation about the third axis.
    yaw,
    /// Effector aperture.
    grasp,
};

/// Unit of one command axis.
enum class CommandUnit : std::uint8_t
{
    /// Unset.
    unspecified = 0,
    /// A pure number with no unit.
    dimensionless,
    /// A value scaled to the workspace, conventionally in `[-1, 1]`.
    normalized,
    /// Metres.
    metres,
    /// Metres per second.
    metres_per_second,
    /// Radians.
    radians,
    /// Radians per second.
    radians_per_second,
};

/// Whether @p frame is one of the declared CommandFrame values.
[[nodiscard]] constexpr bool command_frame_declared(CommandFrame frame) noexcept
{
    return static_cast<std::uint8_t>(frame) <=
           static_cast<std::uint8_t>(CommandFrame::device_native);
}

/// Whether @p name is one of the declared CommandAxisName values.
[[nodiscard]] constexpr bool command_axis_name_declared(CommandAxisName name) noexcept
{
    return static_cast<std::uint8_t>(name) <= static_cast<std::uint8_t>(CommandAxisName::grasp);
}

/// Whether @p unit is one of the declared CommandUnit values.
[[nodiscard]] constexpr bool command_unit_declared(CommandUnit unit) noexcept
{
    return static_cast<std::uint8_t>(unit) <=
           static_cast<std::uint8_t>(CommandUnit::radians_per_second);
}

/// One axis of a command space.
struct CommandAxis
{
    /// What the axis means.
    CommandAxisName name{CommandAxisName::unspecified};
    /// What its values are measured in.
    CommandUnit unit{CommandUnit::unspecified};
};

/// Immutable description of one command space.
///
/// Prepared before a session and persisted with it. A realtime request never
/// carries this; it carries the identifier.
struct CommandSpace
{
    /// Identity of this space.
    CommandSpaceId id{kUnsetCommandSpaceId};
    /// Number of axes in use, at most ::kMaxCommandDim.
    std::uint8_t dim{};
    /// Frame the values are expressed in.
    CommandFrame frame{CommandFrame::unspecified};
    /// Axis descriptions. Entries at or beyond `dimension` are unset.
    std::array<CommandAxis, kMaxCommandDim> axes{};
};

/// One semantic command the experiment asks the runtime to apply.
///
/// Values at or beyond `dimension` are not read and must be zero, so that two
/// commands that mean the same thing also have the same bytes and can be
/// compared and fingerprinted without knowing the dimension first.
struct CommandRequest
{
    /// Experiment time the command was generated.
    ExperimentTimeNs generated_ns{};
    /// Instant from which applying the command is no longer meaningful, or
    /// ::kNoExpiryNs. This is the experiment-side statement of intent; the
    /// runtime enforces its own actuator expiry independently and remains the
    /// only place expiry is enforced against a device.
    ExperimentTimeNs valid_until_ns{kNoExpiryNs};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the command belongs to.
    TrialIdentity trial{};
    /// Space the values belong to.
    CommandSpaceId space{kUnsetCommandSpaceId};
    /// Number of meaningful entries in `values`.
    std::uint8_t dim{};
    /// Command values. Every meaningful entry must be finite.
    std::array<double, kMaxCommandDim> values{};
};

/// What the runtime could prove about one submitted command.
///
/// This is a synchronous application status and nothing more. It is not a
/// hardware acknowledgement: the experiment contract does not define a
/// universal asynchronous actuator acknowledgement protocol, and a device that
/// has one reports it through its own adapter in Devices.
enum class CommandApplication : std::uint8_t
{
    /// The command was never submitted.
    not_submitted = 0,
    /// The runtime accepted it for application.
    accepted,
    /// The runtime refused it before application.
    rejected,
    /// It had passed `valid_until_ns` when submission was attempted.
    expired,
};

/// Whether @p application is one of the declared CommandApplication values.
[[nodiscard]] constexpr bool command_application_declared(CommandApplication application) noexcept
{
    return static_cast<std::uint8_t>(application) <=
           static_cast<std::uint8_t>(CommandApplication::expired);
}

/// The result of submitting one command.
struct CommandOutcome
{
    /// `generated_ns` of the request this reports on.
    ///
    /// Corroboration, not identity: the time contract accepts equal instants,
    /// so a time never identifies a request. `request_sequence` does.
    ExperimentTimeNs generated_ns{};
    /// Experiment time submission was attempted. Equal to `generated_ns` when
    /// the command was never submitted.
    ExperimentTimeNs submitted_ns{};
    /// Emission ordinal of *this report* within the session.
    SequenceOrdinal sequence{};
    /// `sequence` of the CommandRequest this reports on.
    ///
    /// Separate from `sequence` on purpose, for the same reason as in
    /// PresentationOutcome: one field cannot mean both "which record is this"
    /// and "which record is this about".
    SequenceOrdinal request_sequence{};
    /// Trial the command belonged to.
    TrialIdentity trial{};
    /// What the runtime could prove.
    CommandApplication application{CommandApplication::not_submitted};
    /// The runtime's own status enumerator, stored as its integer value.
    ///
    /// The experiment contract must not depend on the streaming domain, so it
    /// records the code rather than the type. The integration layer converts,
    /// and it is the only place that knows which enumeration this came from.
    std::uint16_t status_code{};
};

/// Validate a command space.
[[nodiscard]] ContractStatus validate(const CommandSpace& space) noexcept;

/// Validate a command request, including that every meaningful value is finite.
[[nodiscard]] ContractStatus validate(const CommandRequest& request) noexcept;

/// Validate that @p request is expressed in @p space.
[[nodiscard]] ContractStatus validate_against(const CommandRequest& request,
                                              const CommandSpace& space) noexcept;

/// Validate a command outcome.
[[nodiscard]] ContractStatus validate(const CommandOutcome& outcome) noexcept;

} // namespace neurale::experiments
