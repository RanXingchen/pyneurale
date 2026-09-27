/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/assistance.h>

#include <neurale/experiments/schedule.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace neurale::experiments::assistance
{
namespace
{

// Largest square matrix the projection ever factorizes: one row per desired
// vector that survived the positive-span search.
constexpr std::size_t kMaxSpanRows = kMaxDesiredVelocities;

// Sweeps the cyclic Jacobi rotation is allowed before it stops regardless of
// progress. Convergence is quadratic once the off-diagonal mass is small, and
// the matrix is at most 8x8, so this is far above what a symmetric positive
// semidefinite Gram matrix needs. It exists so the iteration count is stated
// rather than assumed.
constexpr std::size_t kMaxJacobiSweeps = 24;

// Relative off-diagonal mass below which the rotation has nothing left to do.
// Squared quantities are compared, so this is the square of a per-entry
// relative tolerance a little above the double epsilon.
constexpr double kJacobiConverged = 1e-30;

using Row = std::array<double, kMaxCommandDim>;
using Square = std::array<Row, kMaxCommandDim>;
using SpanMatrix = std::array<Row, kMaxSpanRows>;
using Gram = std::array<std::array<double, kMaxSpanRows>, kMaxSpanRows>;

// Every meaningful entry finite, every entry past the dimension exactly zero.
// The second half is the same rule CommandRequest states: two velocities that
// mean the same thing then also have the same bytes.
ContractStatus check_components(const Row& values, std::uint8_t dim) noexcept
{
    for (std::size_t axis = 0; axis < kMaxCommandDim; ++axis)
    {
        if (axis < dim)
        {
            if (!std::isfinite(values[axis]))
                return ContractStatus::value_not_finite;
        }
        else if (values[axis] != 0.0)
        {
            return ContractStatus::dimension_invalid;
        }
    }
    return ContractStatus::ok;
}

ContractStatus check_dimension(std::uint8_t dim) noexcept
{
    return dim == 0 || dim > kMaxCommandDim ? ContractStatus::dimension_invalid
                                            : ContractStatus::ok;
}

// The projection's working state. Every buffer it needs is a member, so the
// transform allocates nothing and its memory is bounded by the header's stated
// maxima rather than by its inputs.
struct Workspace
{
    // Original axis index of each axis in the domain.
    std::array<std::size_t, kMaxCommandDim> axis_of{};
    std::size_t axes{};

    // Candidate directions, unit length, in domain coordinates.
    SpanMatrix unit{};
    std::size_t candidates{};

    // Directions the command actually points along, in the order they were
    // selected.
    SpanMatrix span{};
    std::size_t rows{};

    Row command{};
    Row residual{};
    Row assisted{};

    Gram gram{};
    Gram rotation{};
    std::array<double, kMaxSpanRows> inverse_eigenvalues{};
    Gram inverse{};

    Square projector{};
};

// Cyclic Jacobi eigendecomposition of the symmetric matrix in `gram`.
//
// On return `gram` is diagonal and holds the eigenvalues, and the columns of
// `rotation` hold the corresponding eigenvectors, so the input equals
// `rotation * diag(gram) * transpose(rotation)`. Written out rather than
// delegated to a linear-algebra backend: assistance links nothing, and a
// recorded session has to regenerate bit-for-bit on a machine whose LAPACK is
// not this one's.
//
// Returns true when the off-diagonal mass fell below the convergence criterion
// within the sweep cap, and false when the cap was exhausted without
// converging; the caller must not build a projector on a false return, since
// the diagonal no longer holds trustworthy eigenvalues.
bool jacobi_eigen(Gram& gram, Gram& rotation, std::size_t size) noexcept
{
    for (std::size_t row = 0; row < kMaxSpanRows; ++row)
        for (std::size_t column = 0; column < kMaxSpanRows; ++column)
            rotation[row][column] = row == column ? 1.0 : 0.0;

    for (std::size_t sweep = 0; sweep < kMaxJacobiSweeps; ++sweep)
    {
        double off_diagonal = 0.0;
        double scale = 0.0;
        for (std::size_t p = 0; p < size; ++p)
        {
            scale += gram[p][p] * gram[p][p];
            for (std::size_t q = p + 1; q < size; ++q)
                off_diagonal += gram[p][q] * gram[p][q];
        }
        if (off_diagonal <= kJacobiConverged * (scale > 0.0 ? scale : 1.0))
            return true;

        for (std::size_t p = 0; p + 1 < size; ++p)
        {
            for (std::size_t q = p + 1; q < size; ++q)
            {
                const double pivot = gram[p][q];
                if (pivot == 0.0)
                    continue;

                const double theta = (gram[q][q] - gram[p][p]) / (2.0 * pivot);
                const double magnitude = std::fabs(theta) + std::sqrt(theta * theta + 1.0);
                const double tangent = (theta >= 0.0 ? 1.0 : -1.0) / magnitude;
                const double cosine = 1.0 / std::sqrt(tangent * tangent + 1.0);
                const double sine = tangent * cosine;

                // gram <- transpose(J) * gram * J, then rotation <- rotation * J,
                // with J the plane rotation that annihilates entry (p, q).
                for (std::size_t row = 0; row < size; ++row)
                {
                    const double left = gram[row][p];
                    const double right = gram[row][q];
                    gram[row][p] = cosine * left - sine * right;
                    gram[row][q] = sine * left + cosine * right;
                }
                for (std::size_t column = 0; column < size; ++column)
                {
                    const double upper = gram[p][column];
                    const double lower = gram[q][column];
                    gram[p][column] = cosine * upper - sine * lower;
                    gram[q][column] = sine * upper + cosine * lower;
                }
                for (std::size_t row = 0; row < size; ++row)
                {
                    const double left = rotation[row][p];
                    const double right = rotation[row][q];
                    rotation[row][p] = cosine * left - sine * right;
                    rotation[row][q] = sine * left + cosine * right;
                }
            }
        }
    }
    return false;
}

// Collect the desired directions the command actually points along.
//
// One pass per selected direction, and every pass removes at least the vector
// it selected, so the loop runs at most `candidates` times -- bounded rather
// than recursive, so it cannot allocate an unbounded call stack.
void select_positive_span(Workspace& work) noexcept
{
    work.rows = 0;
    for (std::size_t axis = 0; axis < work.axes; ++axis)
        work.residual[axis] = work.command[axis];

    for (std::size_t pass = 0; pass < work.candidates; ++pass)
    {
        std::size_t best = 0;
        double best_dot = -std::numeric_limits<double>::infinity();
        for (std::size_t candidate = 0; candidate < work.candidates; ++candidate)
        {
            double dot = 0.0;
            for (std::size_t axis = 0; axis < work.axes; ++axis)
                dot += work.unit[candidate][axis] * work.residual[axis];
            if (dot > best_dot)
            {
                best_dot = dot;
                best = candidate;
            }
        }
        // Nothing left points along the remaining command. This is also how the
        // search ends when the command is zero, and when it opposes every
        // desired direction.
        if (!(best_dot > 0.0))
            return;

        double squared_norm = 0.0;
        for (std::size_t axis = 0; axis < work.axes; ++axis)
            squared_norm += work.unit[best][axis] * work.unit[best][axis];
        if (!(squared_norm > 0.0))
            return;

        for (std::size_t axis = 0; axis < work.axes; ++axis)
            work.span[work.rows][axis] = work.unit[best][axis];
        ++work.rows;

        // Remove the selected direction, and take its projection out of the
        // command so the next pass ranks directions against the remainder.
        const double weight = best_dot / squared_norm;
        for (std::size_t axis = 0; axis < work.axes; ++axis)
        {
            work.residual[axis] -= weight * work.unit[best][axis];
            work.unit[best][axis] = 0.0;
        }

        // A direction that now opposes the remainder cannot contribute to a
        // positive span and is dropped rather than reconsidered later.
        for (std::size_t candidate = 0; candidate < work.candidates; ++candidate)
        {
            double dot = 0.0;
            for (std::size_t axis = 0; axis < work.axes; ++axis)
                dot += work.unit[candidate][axis] * work.residual[axis];
            if (dot <= 0.0)
                for (std::size_t axis = 0; axis < work.axes; ++axis)
                    work.unit[candidate][axis] = 0.0;
        }
    }
}

// Orthogonal projector onto the span of the selected directions. Returns
// false only when the eigensolver exhausts its sweep budget; the caller then
// must not use the projector.
bool build_projector(Workspace& work, const OrthoImpedanceParameters& parameters) noexcept
{
    for (std::size_t row = 0; row < work.axes; ++row)
        for (std::size_t column = 0; column < work.axes; ++column)
            work.projector[row][column] = 0.0;
    if (work.rows == 0)
        return true;

    for (std::size_t row = 0; row < work.rows; ++row)
    {
        for (std::size_t column = 0; column < work.rows; ++column)
        {
            double sum = 0.0;
            for (std::size_t axis = 0; axis < work.axes; ++axis)
                sum += work.span[row][axis] * work.span[column][axis];
            work.gram[row][column] = sum;
        }
    }

    const bool converged = jacobi_eigen(work.gram, work.rotation, work.rows);
    if (!converged)
        return false;

    // The numerical risk is a small eigenvalue, whose inverse blows up, so the
    // cut is made in the eigenvalue domain and the directions dropped are the
    // small ones. An absolute floor (rank_tol) drops eigenvalues that
    // name no real direction -- including one that came out slightly negative
    // from rounding -- and a relative cut (conditioning_tol) drops those
    // below this fraction of the largest, bounding the condition number of the
    // inverted block by its reciprocal. The floor requires a strictly positive
    // eigenvalue before inverting, so a zero eigenvalue is never divided by,
    // however small the tolerance.
    double largest_eigenvalue = 0.0;
    for (std::size_t i = 0; i < work.rows; ++i)
        if (work.gram[i][i] > largest_eigenvalue)
            largest_eigenvalue = work.gram[i][i];

    for (std::size_t i = 0; i < work.rows; ++i)
    {
        const double eigenvalue = work.gram[i][i];
        const bool within_floor = eigenvalue > 0.0 && eigenvalue >= parameters.rank_tol;
        const bool within_condition =
            largest_eigenvalue > 0.0 &&
            eigenvalue >= parameters.conditioning_tol * largest_eigenvalue;
        work.inverse_eigenvalues[i] = within_floor && within_condition ? 1.0 / eigenvalue : 0.0;
    }

    // inverse = rotation * diag(inverse_eigenvalues) * transpose(rotation)
    for (std::size_t row = 0; row < work.rows; ++row)
    {
        for (std::size_t column = 0; column < work.rows; ++column)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < work.rows; ++i)
                sum +=
                    work.rotation[row][i] * work.inverse_eigenvalues[i] * work.rotation[column][i];
            work.inverse[row][column] = sum;
        }
    }

    // projector = transpose(span) * inverse * span
    for (std::size_t row = 0; row < work.axes; ++row)
    {
        for (std::size_t column = 0; column < work.axes; ++column)
        {
            double sum = 0.0;
            for (std::size_t left = 0; left < work.rows; ++left)
                for (std::size_t right = 0; right < work.rows; ++right)
                    sum +=
                        work.span[left][row] * work.inverse[left][right] * work.span[right][column];
            work.projector[row][column] = sum;
        }
    }
    return true;
}

// Absorb one component, with negative zero folded onto zero so that two
// manifolds nobody could tell apart do not fingerprint apart.
void absorb_component(FingerprintAccumulator& accumulator, double value) noexcept
{
    accumulator.absorb(std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
}

} // namespace

ContractStatus validate(const VelocityVector& vel) noexcept
{
    if (vel.space == kUnsetCommandSpaceId)
        return ContractStatus::identity_missing;
    if (const ContractStatus status = check_dimension(vel.dim); status != ContractStatus::ok)
        return status;
    return check_components(vel.values, vel.dim);
}

ContractStatus validate_against(const VelocityVector& vel, const CommandSpace& space) noexcept
{
    if (const ContractStatus status = validate(space); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(vel); status != ContractStatus::ok)
        return status;
    if (vel.space != space.id)
        return ContractStatus::identity_missing;
    if (vel.dim != space.dim)
        return ContractStatus::dimension_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const DesiredVelocitySet& desired) noexcept
{
    if (desired.space == kUnsetCommandSpaceId)
        return ContractStatus::identity_missing;
    if (const ContractStatus status = check_dimension(desired.dim); status != ContractStatus::ok)
        return status;
    if (desired.count == 0 || desired.count > kMaxDesiredVelocities)
        return ContractStatus::dimension_invalid;
    for (std::size_t row = 0; row < kMaxDesiredVelocities; ++row)
    {
        // A row past `count` is not a direction that happens to be zero; it is
        // a row nobody supplied, and the two must not share a representation.
        const std::uint8_t width = row < desired.count ? desired.dim : 0;
        if (const ContractStatus status = check_components(desired.vectors[row], width);
            status != ContractStatus::ok)
            return status;
    }
    return ContractStatus::ok;
}

ContractStatus validate_against(const DesiredVelocitySet& desired,
                                const CommandSpace& space) noexcept
{
    if (const ContractStatus status = validate(space); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(desired); status != ContractStatus::ok)
        return status;
    if (desired.space != space.id)
        return ContractStatus::identity_missing;
    if (desired.dim != space.dim)
        return ContractStatus::dimension_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const LinearAssistance& parameters) noexcept
{
    return assistance_in_range(parameters.assistance) ? ContractStatus::ok
                                                      : ContractStatus::assistance_out_of_range;
}

ContractStatus validate(const OrthoImpedanceParameters& parameters) noexcept
{
    if (const ContractStatus status = check_dimension(parameters.dim); status != ContractStatus::ok)
        return status;

    bool any = false;
    for (std::size_t axis = 0; axis < kMaxCommandDim; ++axis)
    {
        if (axis < parameters.dim)
        {
            any = any || parameters.domain[axis];
        }
        else if (parameters.domain[axis])
        {
            return ContractStatus::domain_mask_invalid;
        }
    }
    // Fixturing no axis at all is a configuration mistake, not a request to do
    // nothing: a caller that meant "no assistance" has AssistanceMethod::none.
    if (!any)
        return ContractStatus::domain_mask_invalid;

    for (std::size_t axis = 0; axis < kMaxCommandDim; ++axis)
    {
        if (axis < parameters.dim)
        {
            if (!assistance_in_range(parameters.impedance[axis]))
                return ContractStatus::assistance_out_of_range;
        }
        else if (parameters.impedance[axis] != 0.0)
        {
            return ContractStatus::dimension_invalid;
        }
    }

    if (!std::isfinite(parameters.rank_tol))
        return ContractStatus::value_not_finite;
    if (parameters.rank_tol < 0.0)
        return ContractStatus::parameter_out_of_range;
    if (!std::isfinite(parameters.conditioning_tol))
        return ContractStatus::value_not_finite;
    if (parameters.conditioning_tol < 0.0 || parameters.conditioning_tol > 1.0)
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

ContractStatus validate_against(const OrthoImpedanceParameters& parameters,
                                const CommandSpace& space) noexcept
{
    if (const ContractStatus status = validate(space); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(parameters); status != ContractStatus::ok)
        return status;
    if (parameters.dim != space.dim)
        return ContractStatus::dimension_invalid;
    return ContractStatus::ok;
}

ContractStatus blend_velocity(const CommandSpace& space, const VelocityVector& external,
                              const VelocityVector& guidance, const LinearAssistance& parameters,
                              VelocityVector& assisted) noexcept
{
    if (const ContractStatus status = validate_against(external, space);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate_against(guidance, space);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(parameters); status != ContractStatus::ok)
        return status;

    VelocityVector result{};
    result.space = space.id;
    result.dim = space.dim;
    // The endpoints are the identity on the stored values rather than an
    // arithmetic coincidence: 1 - 1 is exactly 0 and 0 * x is not x when x is
    // an infinity, so a caller asking for no assistance gets its own command
    // back byte for byte.
    if (parameters.assistance == kMinAssistance)
    {
        result.values = external.values;
    }
    else if (parameters.assistance == kMaxAssistance)
    {
        result.values = guidance.values;
    }
    else
    {
        const double external_weight = 1.0 - parameters.assistance;
        for (std::size_t axis = 0; axis < space.dim; ++axis)
            result.values[axis] = external_weight * external.values[axis] +
                                  parameters.assistance * guidance.values[axis];
    }

    assisted = result;
    return ContractStatus::ok;
}

ContractStatus apply_ortho_impedance(const CommandSpace& space, const DesiredVelocitySet& desired,
                                     const VelocityVector& external,
                                     const OrthoImpedanceParameters& parameters,
                                     VelocityVector& assisted) noexcept
{
    if (const ContractStatus status = validate_against(external, space);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate_against(desired, space);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate_against(parameters, space);
        status != ContractStatus::ok)
        return status;

    Workspace work{};
    for (std::size_t axis = 0; axis < space.dim; ++axis)
        if (parameters.domain[axis])
        {
            work.axis_of[work.axes] = axis;
            work.command[work.axes] = external.values[axis];
            ++work.axes;
        }

    // A desired vector is a direction. One whose domain components are all
    // negligible names no direction and takes no part; the rest are normalized,
    // so scaling a row changes nothing.
    constexpr double kNegligible = std::numeric_limits<double>::epsilon();
    for (std::size_t row = 0; row < desired.count; ++row)
    {
        double squared_norm = 0.0;
        bool meaningful = false;
        for (std::size_t i = 0; i < work.axes; ++i)
        {
            const double component = desired.vectors[row][work.axis_of[i]];
            meaningful = meaningful || std::fabs(component) > kNegligible;
            squared_norm += component * component;
        }
        if (!meaningful || !(squared_norm > 0.0))
            continue;

        const double norm = std::sqrt(squared_norm);
        for (std::size_t i = 0; i < work.axes; ++i)
            work.unit[work.candidates][i] = desired.vectors[row][work.axis_of[i]] / norm;
        ++work.candidates;
    }

    select_positive_span(work);
    if (!build_projector(work, parameters))
        return ContractStatus::numerical_failure;

    // transform = (I - R) * (I - projector) + projector, applied to the command
    // restricted to the domain. With no span the projector is zero and the
    // transform is the impedance alone, which is the stated degenerate answer.
    for (std::size_t row = 0; row < work.axes; ++row)
    {
        const double transparency = 1.0 - parameters.impedance[work.axis_of[row]];
        double sum = 0.0;
        for (std::size_t column = 0; column < work.axes; ++column)
        {
            const double identity = row == column ? 1.0 : 0.0;
            const double entry = transparency * (identity - work.projector[row][column]) +
                                 work.projector[row][column];
            sum += entry * work.command[column];
        }
        work.assisted[row] = sum;
    }

    VelocityVector result{};
    result.space = space.id;
    result.dim = space.dim;
    for (std::size_t axis = 0; axis < space.dim; ++axis)
        result.values[axis] = external.values[axis];
    // Scatter each result back to its original axis.
    for (std::size_t i = 0; i < work.axes; ++i)
        result.values[work.axis_of[i]] = work.assisted[i];

    assisted = result;
    return ContractStatus::ok;
}

std::uint64_t manifold_fingerprint(const DesiredVelocitySet& desired) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(desired.space);
    accumulator.absorb(desired.dim);
    accumulator.absorb(desired.count);
    for (std::size_t row = 0; row < desired.count; ++row)
        for (std::size_t axis = 0; axis < desired.dim; ++axis)
            absorb_component(accumulator, desired.vectors[row][axis]);
    return accumulator.value();
}

ContractStatus validate(const LinearAssistanceRecord& record) noexcept
{
    if (!assistance_method_declared(record.method))
        return ContractStatus::enum_undeclared;
    if (record.method != AssistanceMethod::linear_blend)
        return ContractStatus::outcome_invalid;
    if (record.space == kUnsetCommandSpaceId)
        return ContractStatus::identity_missing;
    if (record.version != kLinearBlendVersion1)
        return ContractStatus::version_unsupported;
    if (const ContractStatus status = check_dimension(record.dim); status != ContractStatus::ok)
        return status;
    if (!assistance_in_range(record.assistance))
        return ContractStatus::assistance_out_of_range;
    for (const Row* values : {&record.external, &record.guidance, &record.assisted})
        if (const ContractStatus status = check_components(*values, record.dim);
            status != ContractStatus::ok)
            return status;
    return ContractStatus::ok;
}

ContractStatus validate(const OrthoImpedanceRecord& record) noexcept
{
    if (!assistance_method_declared(record.method))
        return ContractStatus::enum_undeclared;
    if (record.method != AssistanceMethod::ortho_impedance)
        return ContractStatus::outcome_invalid;
    if (record.space == kUnsetCommandSpaceId)
        return ContractStatus::identity_missing;
    if (record.version != kOrthoImpedanceVersion1)
        return ContractStatus::version_unsupported;
    if (const ContractStatus status = check_dimension(record.dim); status != ContractStatus::ok)
        return status;

    OrthoImpedanceParameters parameters{};
    parameters.dim = record.dim;
    parameters.domain = record.domain;
    parameters.impedance = record.impedance;
    parameters.rank_tol = record.rank_tol;
    parameters.conditioning_tol = record.conditioning_tol;
    if (const ContractStatus status = validate(parameters); status != ContractStatus::ok)
        return status;

    for (const Row* values : {&record.external, &record.assisted})
        if (const ContractStatus status = check_components(*values, record.dim);
            status != ContractStatus::ok)
            return status;
    return ContractStatus::ok;
}

} // namespace neurale::experiments::assistance
