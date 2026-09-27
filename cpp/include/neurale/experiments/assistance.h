/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <neurale/experiments/command.h>
#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>

/**
 * @file
 * @brief Shared velocity assistance: linear blending and orthogonal impedance.
 *
 * Velocity assistance is reusable across every BCI paradigm whose movement
 * command is a velocity vector. It is owned here rather than by Center-Out, by
 * WebGrid, by a decoder, or by an actuator, and it is deliberately blind: a
 * transform in this header receives vectors and parameters and returns a
 * vector. It never learns target geometry, trial state, task phase, GUI state,
 * or experiment identity, because a transform that could read any of those
 * would be a paradigm behaviour wearing a shared name.
 *
 * The velocity vocabulary is not restated here. A velocity belongs to a
 * CommandSpace, which already fixes the dimension, the per-axis names and their
 * order, the units, and the coordinate frame. Assistance validates its inputs
 * against one prepared space, so "dimensions, units and frame match exactly" is
 * a check against a single authority rather than a comparison of two records
 * that each believe something.
 *
 * A VelocityVector is deliberately *not* a CommandRequest. A CommandRequest
 * carries trial identity, emission ordinal, and times; handing that to a blind
 * transform would make the blindness unenforceable, and the reviewer of a
 * future change would have nothing but a comment stopping the transform from
 * reading them.
 *
 * Nothing here arms or releases a SafetyController, changes a deadline policy,
 * retries an expired command, or touches a device. Assistance produces a
 * velocity; the streaming runtime remains the only place that decides whether a
 * command is applied.
 */
namespace neurale::experiments::assistance
{

/// Version of one assistance method's transformation.
///
/// It changes whenever the transformation changes at all. A recorded session
/// states the method and this version, so a replay knows which mapping produced
/// the velocities it is reading rather than assuming the current one.
using AssistanceVersion = std::uint32_t;

/// Which shared-control method produced an assisted velocity.
enum class AssistanceMethod : std::uint8_t
{
    /// No assistance was applied.
    none = 0,
    /// Linear blend of an external command with one guidance velocity.
    linear_blend,
    /// Projection onto a desired manifold with orthogonal impedance.
    ortho_impedance,
};

/// Whether @p method is one of the declared AssistanceMethod values.
[[nodiscard]] constexpr bool assistance_method_declared(AssistanceMethod method) noexcept
{
    return static_cast<std::uint8_t>(method) <=
           static_cast<std::uint8_t>(AssistanceMethod::ortho_impedance);
}

/// The linear blend specified in this header.
///
/// Version 1 names exactly
/// `assisted = (1 - assistance) * external + assistance * guidance`, evaluated
/// per component, with no clipping, no normalization, no adaptation, and no state.
inline constexpr AssistanceVersion kLinearBlendVersion1 = 1;

/// The orthogonal-impedance projection specified in this header.
///
/// Version 1 names the whole of apply_ortho_impedance(): the positive-span
/// search, the regularized projector, the impedance applied to the orthogonal
/// remainder, and the tolerance defaults below.
inline constexpr AssistanceVersion kOrthoImpedanceVersion1 = 1;

/// Largest number of desired-velocity vectors a manifold may declare.
///
/// Bounded because the positive-span search below is bounded by it: every pass
/// consumes one vector, so a bounded manifold is what makes the search's
/// iteration count something this header can state rather than hope for.
inline constexpr std::size_t kMaxDesiredVelocities = 8;

/// Smallest and largest admissible value of a scalar assistance parameter.
///
/// Stated as an interval rather than assumed, because "the supported range"
/// is the first thing a method changes when it grows. Both the linear blend
/// coefficient and each orthogonal impedance lie in it.
/// @{
inline constexpr double kMinAssistance = 0.0;
inline constexpr double kMaxAssistance = 1.0;
/// @}

/// Whether @p value lies in `[kMinAssistance, kMaxAssistance]`.
///
/// A NaN fails every comparison and is therefore reported as out of range,
/// which is the answer that matters: it is not a usable assistance value.
[[nodiscard]] constexpr bool assistance_in_range(double value) noexcept
{
    return value >= kMinAssistance && value <= kMaxAssistance;
}

/// A velocity vector expressed in one prepared CommandSpace.
///
/// Components at or beyond `dimension` must be exactly zero, for the same
/// reason as in CommandRequest: two velocities that mean the same thing then
/// also have the same bytes and can be compared and fingerprinted without
/// knowing the dimension first.
struct VelocityVector
{
    /// Space the components belong to. Zero means unset.
    CommandSpaceId space{kUnsetCommandSpaceId};
    /// Number of meaningful components.
    std::uint8_t dim{};
    /// Components. Entries at or beyond `dimension` must be zero.
    std::array<double, kMaxCommandDim> values{};
};

/// A set of desired velocity vectors spanning the manifold to project onto.
///
/// The vectors are directions, not magnitudes: the projection normalizes each
/// one before use, so scaling a row changes nothing. A row that is entirely
/// zero names no direction and is ignored rather than rejected, because "this
/// slot currently has no desired direction" is an ordinary state for a manifold
/// assembled per trial.
struct DesiredVelocitySet
{
    /// Space the vectors belong to. Zero means unset.
    CommandSpaceId space{kUnsetCommandSpaceId};
    /// Number of meaningful components in each vector.
    std::uint8_t dim{};
    /// Number of meaningful vectors, at most ::kMaxDesiredVelocities.
    std::uint8_t count{};
    /// The vectors. Rows at or beyond `count`, and components at or beyond
    /// `dimension`, must be zero.
    std::array<std::array<double, kMaxCommandDim>, kMaxDesiredVelocities> vectors{};
};

/// Parameters of the linear blend.
///
/// A struct rather than a bare `double` so that the parameter set is a value
/// that can be validated, recorded, and extended without every call site
/// changing shape.
struct LinearAssistance
{
    /// Weight given to the guidance velocity, in `[0, 1]`.
    ///
    /// Zero returns the external command unchanged; one returns the guidance
    /// command unchanged. Nothing in between is clipped or renormalized.
    double assistance{kMinAssistance};
};

/// Parameters of the orthogonal-impedance projection.
///
/// The parameter set states its own dimension. Both fields below are per-axis,
/// and a parameter set that described fewer axes than the space it is used with
/// would silently leave the rest at "not in the domain, impedance zero" --
/// which is a decision, and one nobody made.
struct OrthoImpedanceParameters
{
    /// Number of axes these parameters describe.
    std::uint8_t dim{};
    /// Which axes take part in the projection.
    ///
    /// An axis outside the domain is passed through unchanged, *in place*. The
    /// domain is a per-axis flag rather than a count because the axes that
    /// should be fixtured are not in general the leading ones.
    std::array<bool, kMaxCommandDim> domain{};
    /// Per-axis orthogonal impedance, each in `[0, 1]`.
    ///
    /// This is the diagonal of the impedance matrix `R`. Only the diagonal is
    /// stored because only the diagonal was ever read: an impedance of zero
    /// leaves the component orthogonal to the manifold untouched, and an
    /// impedance of one removes it entirely.
    ///
    /// Every entry below `dimension` must be in range whether or not its axis
    /// is currently in the domain. An impedance is a property of an axis and
    /// the domain is what is being fixtured right now; demanding that the
    /// entries be zeroed outside the domain would make one parameter set
    /// unusable the moment the domain changed. Entries at or beyond `dimension`
    /// must be zero, as everywhere else in the contract.
    std::array<double, kMaxCommandDim> impedance{};
    /// Absolute floor on the eigenvalues of the span Gram matrix.
    ///
    /// An eigenvalue is inverted only when it is strictly positive and at or
    /// above this floor; anything else names a direction the span does not
    /// actually have and is dropped. The rows entering that matrix are unit
    /// vectors, so its eigenvalues lie in `[0, kMaxDesiredVelocities]` and an
    /// absolute tolerance is meaningful. A floor of zero disables the absolute
    /// cut, but a zero eigenvalue is still never divided by. It is non-negative:
    /// a negative value is ContractStatus::parameter_out_of_range and a
    /// non-finite one is ContractStatus::value_not_finite.
    double rank_tol{1e-8};
    /// Relative cut on the eigenvalues of the span Gram matrix.
    ///
    /// An eigenvalue is dropped when it is smaller than this fraction of the
    /// largest one, so the cut is made in the eigenvalue domain: the numerical
    /// risk is a small eigenvalue, whose inverse blows up, and the directions
    /// dropped are the small ones. The condition number of the inverted block
    /// is then bounded by the reciprocal of this value, which is how far the
    /// projector is allowed to amplify the worst-conditioned kept direction of
    /// a nearly degenerate span. It lies in `[0, 1]`, being a fraction of the
    /// largest eigenvalue: zero disables the relative cut and leaves only the
    /// absolute floor, and one keeps only the largest-eigenvalue eigenspace. A
    /// value outside the interval is ContractStatus::parameter_out_of_range and
    /// a non-finite one is ContractStatus::value_not_finite.
    double conditioning_tol{5e-6};
};

/// Validate a velocity vector, including that every meaningful value is finite.
[[nodiscard]] ContractStatus validate(const VelocityVector& vel) noexcept;

/// Validate that @p velocity is expressed in @p space.
[[nodiscard]] ContractStatus validate_against(const VelocityVector& vel,
                                              const CommandSpace& space) noexcept;

/// Validate a desired-velocity set.
[[nodiscard]] ContractStatus validate(const DesiredVelocitySet& desired) noexcept;

/// Validate that @p desired is expressed in @p space.
[[nodiscard]] ContractStatus validate_against(const DesiredVelocitySet& desired,
                                              const CommandSpace& space) noexcept;

/// Validate linear blend parameters.
[[nodiscard]] ContractStatus validate(const LinearAssistance& parameters) noexcept;

/// Validate orthogonal-impedance parameters.
[[nodiscard]] ContractStatus validate(const OrthoImpedanceParameters& parameters) noexcept;

/// Validate that @p parameters describe the axes of @p space.
[[nodiscard]] ContractStatus validate_against(const OrthoImpedanceParameters& parameters,
                                              const CommandSpace& space) noexcept;

/// Blend an external command with one guidance velocity.
///
/// Computes `assisted = (1 - assistance) * external + assistance * guidance`
/// per component. There is no clipping, no normalization, no adaptation, and no
/// state update: the transform is a pure function of its arguments, and calling
/// it twice with the same arguments produces the same bytes.
///
/// At `assistance == 0` the external command is returned exactly, and at
/// `assistance == 1` the guidance command is -- both as the identity on the
/// stored values, not merely up to rounding.
///
/// This path allocates nothing and executes a bounded number of operations.
///
/// @param space Prepared space both velocities must be expressed in.
/// @param external The external or decoded command.
/// @param guidance The reference velocity the paradigm supplied.
/// @param parameters Blend parameters.
/// @param assisted Receives the result on ContractStatus::ok, and is left
///                 unmodified otherwise.
/// @return ContractStatus::ok, or the first reason an input was rejected.
[[nodiscard]] ContractStatus blend_velocity(const CommandSpace& space,
                                            const VelocityVector& external,
                                            const VelocityVector& guidance,
                                            const LinearAssistance& parameters,
                                            VelocityVector& assisted) noexcept;

/// Constrain an external command towards a desired manifold.
///
/// Implements `v = S u + (I - R) (I - S) u` over the axes in the domain, where
/// `S` projects onto the span of the desired directions that the command
/// actually points along, and `R` is the diagonal orthogonal impedance. Axes
/// outside the domain are copied through unchanged and *in their own
/// positions*, so the result is expressed in the same coordinate order as the
/// input.
///
/// Degenerate inputs are decided rather than left to the arithmetic:
///
/// ```text
/// no desired vector points anywhere      -> S = 0, so v = (I - R) u
/// the command opposes every direction    -> S = 0, so v = (I - R) u
/// the command is zero                    -> v = 0
/// the span is rank deficient             -> S projects onto the span it has
/// ```
///
/// The first two cases are the same statement: with nothing to project onto,
/// every component is orthogonal to the manifold, and impedance is all that
/// remains to apply.
///
/// This path allocates nothing and executes a bounded number of operations, but
/// no latency measurement has been taken for it and none is claimed. It is not
/// on the realtime data plane.
///
/// @param space Prepared space the command and the manifold belong to.
/// @param desired The desired-velocity manifold.
/// @param external The external or decoded command.
/// @param parameters Domain, impedance, and tolerances.
/// @param assisted Receives the result on ContractStatus::ok, and is left
///                 unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::numerical_failure if the
///         internal eigensolver exhausts its sweep budget without converging,
///         or the first reason an input was rejected.
[[nodiscard]] ContractStatus apply_ortho_impedance(const CommandSpace& space,
                                                   const DesiredVelocitySet& desired,
                                                   const VelocityVector& external,
                                                   const OrthoImpedanceParameters& parameters,
                                                   VelocityVector& assisted) noexcept;

/// Digest identifying a desired-velocity manifold exactly.
///
/// A manifold is too large to inline in a trace record, so a record carries
/// this instead and a replay checks the manifold it reconstructs against it.
/// The digest is defined here rather than left to each caller, because two
/// callers digesting "the manifold" differently would make the check
/// unfalsifiable.
[[nodiscard]] std::uint64_t manifold_fingerprint(const DesiredVelocitySet& desired) noexcept;

/// One applied linear blend, emitted as control provenance.
///
/// Provenance, not task-state truth: nothing reads this record to decide what
/// the experiment does next. It carries the trial it belonged to for linkage
/// only, and the transform that produced it never saw that field.
struct LinearAssistanceRecord
{
    /// Experiment time the blend was applied.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the blend belonged to.
    TrialIdentity trial{};
    /// Space the three velocities are expressed in.
    CommandSpaceId space{kUnsetCommandSpaceId};
    /// Which method this record is about.
    ///
    /// Redundant with the record's own type in memory and not redundant at all
    /// once persisted: a record read back out of a control stream has to say
    /// what produced it.
    AssistanceMethod method{AssistanceMethod::linear_blend};
    /// Version of the blend that produced `assisted`.
    ///
    /// Validated to equal ::kLinearBlendVersion1: a record naming a version
    /// this build cannot replay is ContractStatus::version_unsupported.
    AssistanceVersion version{kLinearBlendVersion1};
    /// Number of meaningful components in each velocity.
    std::uint8_t dim{};
    /// The blend coefficient that was applied.
    double assistance{};
    /// The external command that went in.
    std::array<double, kMaxCommandDim> external{};
    /// The guidance velocity that went in.
    std::array<double, kMaxCommandDim> guidance{};
    /// The assisted velocity that came out.
    std::array<double, kMaxCommandDim> assisted{};
};

/// One applied orthogonal-impedance step, emitted as control provenance.
struct OrthoImpedanceRecord
{
    /// Experiment time the projection was applied.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the projection belonged to.
    TrialIdentity trial{};
    /// Digest of the manifold that went in, from manifold_fingerprint().
    std::uint64_t manifold{};
    /// Space the velocities are expressed in.
    CommandSpaceId space{kUnsetCommandSpaceId};
    /// Which method this record is about.
    AssistanceMethod method{AssistanceMethod::ortho_impedance};
    /// Version of the projection that produced `assisted`.
    ///
    /// Validated to equal ::kOrthoImpedanceVersion1: a record naming a version
    /// this build cannot replay is ContractStatus::version_unsupported.
    AssistanceVersion version{kOrthoImpedanceVersion1};
    /// Number of meaningful components in each velocity.
    std::uint8_t dim{};
    /// Rank tolerance that was in force.
    ///
    /// The tolerances are recorded rather than assumed to be the defaults. A
    /// session that ran with a different one and did not say so could not be
    /// regenerated from its own record, which is the whole point of holding it.
    double rank_tol{};
    /// Conditioning tolerance that was in force.
    double conditioning_tol{};
    /// Axes that took part in the projection.
    std::array<bool, kMaxCommandDim> domain{};
    /// Orthogonal impedance that was applied.
    std::array<double, kMaxCommandDim> impedance{};
    /// The external command that went in.
    std::array<double, kMaxCommandDim> external{};
    /// The assisted velocity that came out.
    std::array<double, kMaxCommandDim> assisted{};
};

/// Validate a linear-assistance trace record.
[[nodiscard]] ContractStatus validate(const LinearAssistanceRecord& record) noexcept;

/// Validate an orthogonal-impedance trace record.
[[nodiscard]] ContractStatus validate(const OrthoImpedanceRecord& record) noexcept;

} // namespace neurale::experiments::assistance
