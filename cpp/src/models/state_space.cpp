/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/state_space.h>

#include <neurale/models/linear_model.h>

#include "linalg.h"
#include "span_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace neurale::models
{
namespace
{

using detail::matrix_index;

// Both covariance tolerances are relative to the largest magnitude in the
// matrix and neither has an absolute floor: a floor turns the check into an
// absolute one for small covariances and accepts a matrix whose negative
// eigenvalue dwarfs its positive ones. Symmetry is forgiving because the
// matrix is symmetrized before use anyway; definiteness is not, because a
// symmetric eigensolver resolves eigenvalues to about ``eps * n * scale``.
// The Python validator in ``neurale.models.state_space`` uses the same rule.
constexpr double kSymmetryRelativeTolerance = 1e-8;
constexpr double kPsdToleranceFactor = 64.0;

void symmetrize(std::span<double> matrix, std::size_t n)
{
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t col = row + 1; col < n; ++col)
        {
            const double mean =
                0.5 * (matrix[matrix_index(row, col, n)] + matrix[matrix_index(col, row, n)]);
            matrix[matrix_index(row, col, n)] = mean;
            matrix[matrix_index(col, row, n)] = mean;
        }
    }
}

void add_diagonal(std::span<double> matrix, std::size_t n, double value)
{
    if (value == 0.0)
    {
        return;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        matrix[matrix_index(i, i, n)] += value;
    }
}

void require_finite(std::span<const double> values, const char* name)
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(name) + " must contain finite values");
        }
    }
}

double matrix_magnitude(std::span<const double> values)
{
    double magnitude = 0.0;
    for (double value : values)
    {
        magnitude = std::max(magnitude, std::abs(value));
    }
    return magnitude;
}

void require_symmetric_psd(std::span<const double> matrix, std::size_t n, const char* name)
{
    if (n == 0 || matrix.size() != n * n)
    {
        throw std::invalid_argument(std::string(name) + " shape is invalid");
    }
    require_finite(matrix, name);

    const double scale = matrix_magnitude(matrix);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t col = row + 1; col < n; ++col)
        {
            const double difference =
                std::abs(matrix[matrix_index(row, col, n)] - matrix[matrix_index(col, row, n)]);
            if (difference > kSymmetryRelativeTolerance * scale)
            {
                throw std::invalid_argument(std::string(name) + " must be symmetric");
            }
        }
    }

    auto symmetric = std::vector<double>(matrix.begin(), matrix.end());
    symmetrize(symmetric, n);
    const auto eigen = symmetric_eigen(symmetric, n);
    const double smallest = *std::min_element(eigen.values.begin(), eigen.values.end());
    const double psd_tol = kPsdToleranceFactor * static_cast<double>(n) *
                           std::numeric_limits<double>::epsilon() * scale;
    if (smallest < -psd_tol)
    {
        throw std::invalid_argument(std::string(name) + " must be positive semidefinite");
    }
}

void require_model_state(const LinearGaussianModelState& state)
{
    const std::size_t k = state.state_dim;
    const std::size_t m = state.observation_dim;
    if (k == 0 || m == 0)
    {
        throw std::invalid_argument("state and observation dimensions must be positive");
    }
    if (state.transition.size() != k * k || state.transition_offset.size() != k ||
        state.observation.size() != m * k || state.observation_offset.size() != m ||
        state.initial_state.size() != k)
    {
        throw std::invalid_argument("state-space parameter shapes are invalid");
    }
    require_finite(state.transition, "transition");
    require_finite(state.transition_offset, "transition_offset");
    require_finite(state.observation, "observation");
    require_finite(state.observation_offset, "observation_offset");
    require_finite(state.initial_state, "initial_state");
    require_symmetric_psd(state.process_covariance, k, "process_covariance");
    require_symmetric_psd(state.observation_covariance, m, "observation_covariance");
    require_symmetric_psd(state.initial_covariance, k, "initial_covariance");
}

void require_jitter(double jitter)
{
    if (!(jitter >= 0.0) || !std::isfinite(jitter))
    {
        throw std::invalid_argument("jitter must be a finite non-negative number");
    }
}

std::vector<double> residual_covariance(std::span<const double> residuals, std::size_t n_rows,
                                        std::size_t n_cols)
{
    auto covariance = std::vector<double>(n_cols * n_cols, 0.0);
    for (std::size_t row = 0; row < n_rows; ++row)
    {
        for (std::size_t left = 0; left < n_cols; ++left)
        {
            const double value = residuals[matrix_index(row, left, n_cols)];
            for (std::size_t right = left; right < n_cols; ++right)
            {
                covariance[matrix_index(left, right, n_cols)] +=
                    value * residuals[matrix_index(row, right, n_cols)];
            }
        }
    }
    const double scale = 1.0 / static_cast<double>(n_rows);
    for (std::size_t left = 0; left < n_cols; ++left)
    {
        for (std::size_t right = left; right < n_cols; ++right)
        {
            const double value = covariance[matrix_index(left, right, n_cols)] * scale;
            covariance[matrix_index(left, right, n_cols)] = value;
            covariance[matrix_index(right, left, n_cols)] = value;
        }
    }
    return covariance;
}

/// Residuals of an affine map: ``targets - (design * coef^T + intercept)``.
std::vector<double> affine_residuals(std::span<const double> design,
                                     std::span<const double> targets, std::span<const double> coef,
                                     std::span<const double> intercept, std::size_t n_rows,
                                     std::size_t n_inputs, std::size_t n_outputs)
{
    auto residuals = std::vector<double>(n_rows * n_outputs, 0.0);
    gemm_nt(design, coef, n_rows, n_outputs, n_inputs, residuals);
    for (std::size_t row = 0; row < n_rows; ++row)
    {
        for (std::size_t output = 0; output < n_outputs; ++output)
        {
            const std::size_t idx = matrix_index(row, output, n_outputs);
            residuals[idx] = targets[idx] - residuals[idx] - intercept[output];
        }
    }
    return residuals;
}

// ---------------------------------------------------------------------------
// The recursion itself, on caller-owned buffers.
//
// Every kernel below writes only into spans it is given, so the same arithmetic
// serves the allocating offline filter and the prepared real-time one. The one
// step the two do not share is the gain solve: offline it goes through
// ``solve_spd``, which allocates and may reach LAPACK, and on a real-time
// thread it goes through the in-place factorizations, which may not. Everything
// around that solve -- the innovation, the innovation covariance, the
// correction, and the Joseph-form posterior -- exists once.
// ---------------------------------------------------------------------------

/// ``output = left * right^T`` for row-major operands, written out by hand.
///
/// The recursion's products are small -- the state dimension is the number of
/// decoded target channels, and even a wide observation makes a thin product --
/// so the vendor kernel would be called for its overhead rather than for its
/// throughput. On a real-time thread it would cost more than that: a BLAS call
/// enters the shared thread pool and synchronizes it on every step, which is
/// exactly the tail-latency pathology the small-FFT dispatch avoids. Keeping
/// the whole recursion scalar also means the offline filter and the prepared
/// real-time one compute the same bits on every build, MKL or not.
void gemm_nt_scalar(std::span<const double> left, std::span<const double> right, std::size_t rows,
                    std::size_t cols, std::size_t inner, std::span<double> output) noexcept
{
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            double value = 0.0;
            for (std::size_t k = 0; k < inner; ++k)
            {
                value += left[matrix_index(row, k, inner)] * right[matrix_index(col, k, inner)];
            }
            output[matrix_index(row, col, cols)] = value;
        }
    }
}

/// ``output = left * right`` for row-major operands, ``left`` sized
/// ``rows x inner`` and ``right`` sized ``inner x cols``.
///
/// A zero entry of ``left`` contributes nothing to any column, so its whole row
/// of the inner product is skipped. That matters here rather than in general:
/// the recursion multiplies by the transition and observation maps on every
/// step, and both are routinely sparse -- a random-walk transition is nearly
/// diagonal, and an observation matrix drops the features a decoder does not
/// select.
void gemm_nn_scalar(std::span<const double> left, std::span<const double> right, std::size_t rows,
                    std::size_t inner, std::size_t cols, std::span<double> output) noexcept
{
    std::fill(output.begin(), output.end(), 0.0);
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t k = 0; k < inner; ++k)
        {
            const double value = left[matrix_index(row, k, inner)];
            if (value == 0.0)
            {
                continue;
            }
            for (std::size_t col = 0; col < cols; ++col)
            {
                output[matrix_index(row, col, cols)] += value * right[matrix_index(k, col, cols)];
            }
        }
    }
}

[[nodiscard]] bool all_finite(std::span<const double> values) noexcept
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return true;
}

/// ``state <- A x + b`` and ``covariance <- A P A^T + Q``.
///
/// ``next_state`` is ``k`` wide and ``scaled`` and ``propagated`` are ``k * k``.
void kalman_predict(const LinearGaussianModelState& parameters, std::span<double> state,
                    std::span<double> covariance, std::span<double> next_state,
                    std::span<double> scaled, std::span<double> propagated) noexcept
{
    const std::size_t k = parameters.state_dim;
    for (std::size_t row = 0; row < k; ++row)
    {
        double value = parameters.transition_offset[row];
        for (std::size_t col = 0; col < k; ++col)
        {
            value += parameters.transition[matrix_index(row, col, k)] * state[col];
        }
        next_state[row] = value;
    }

    gemm_nn_scalar(parameters.transition, covariance, k, k, k, scaled);
    gemm_nt_scalar(scaled, parameters.transition, k, k, k, propagated);
    for (std::size_t i = 0; i < k * k; ++i)
    {
        propagated[i] += parameters.process_covariance[i];
    }
    symmetrize(propagated, k);

    std::copy(next_state.begin(), next_state.end(), state.begin());
    std::copy(propagated.begin(), propagated.end(), covariance.begin());
}

/// ``innovation = z - (H x + d)``, ``cross = H P``, ``S = H P H^T + R + jitter I``.
void kalman_innovation(const LinearGaussianModelState& parameters, std::span<const double> state,
                       std::span<const double> covariance, std::span<const double> observation,
                       double jitter, std::span<double> innovation, std::span<double> cross,
                       std::span<double> innovation_covariance) noexcept
{
    const std::size_t k = parameters.state_dim;
    const std::size_t m = parameters.observation_dim;
    for (std::size_t row = 0; row < m; ++row)
    {
        double predicted = parameters.observation_offset[row];
        for (std::size_t col = 0; col < k; ++col)
        {
            predicted += parameters.observation[matrix_index(row, col, k)] * state[col];
        }
        innovation[row] = observation[row] - predicted;
    }

    gemm_nn_scalar(parameters.observation, covariance, m, k, k, cross);
    gemm_nt_scalar(cross, parameters.observation, m, m, k, innovation_covariance);
    for (std::size_t i = 0; i < m * m; ++i)
    {
        innovation_covariance[i] += parameters.observation_covariance[i];
    }
    symmetrize(innovation_covariance, m);
    add_diagonal(innovation_covariance, m, jitter);
}

/// Apply a solved gain: ``x <- x + K v`` and the Joseph-form posterior.
///
/// ``gain_transposed`` is ``K^T``, held ``m x k``, which is the shape the
/// symmetric solve produces. ``factor``, ``left`` and ``posterior`` are
/// ``k * k`` and ``noise`` is ``m * k``.
void kalman_correct(const LinearGaussianModelState& parameters, std::span<double> state,
                    std::span<double> covariance, std::span<const double> gain_transposed,
                    std::span<const double> innovation, std::span<double> factor,
                    std::span<double> left, std::span<double> posterior,
                    std::span<double> noise) noexcept
{
    const std::size_t k = parameters.state_dim;
    const std::size_t m = parameters.observation_dim;
    for (std::size_t row = 0; row < k; ++row)
    {
        double correction = 0.0;
        for (std::size_t col = 0; col < m; ++col)
        {
            correction += gain_transposed[matrix_index(col, row, k)] * innovation[col];
        }
        state[row] += correction;
    }

    // Joseph form: P <- (I - K H) P (I - K H)^T + K R K^T
    std::fill(factor.begin(), factor.end(), 0.0);
    for (std::size_t row = 0; row < k; ++row)
    {
        factor[matrix_index(row, row, k)] = 1.0;
        for (std::size_t col = 0; col < k; ++col)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < m; ++i)
            {
                value += gain_transposed[matrix_index(i, row, k)] *
                         parameters.observation[matrix_index(i, col, k)];
            }
            factor[matrix_index(row, col, k)] -= value;
        }
    }

    gemm_nn_scalar(factor, covariance, k, k, k, left);
    gemm_nt_scalar(left, factor, k, k, k, posterior);

    // K R K^T, with the gain held transposed as an m x k matrix.
    gemm_nn_scalar(parameters.observation_covariance, gain_transposed, m, m, k, noise);
    for (std::size_t row = 0; row < k; ++row)
    {
        for (std::size_t col = 0; col < k; ++col)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < m; ++i)
            {
                value += gain_transposed[matrix_index(i, row, k)] * noise[matrix_index(i, col, k)];
            }
            posterior[matrix_index(row, col, k)] += value;
        }
    }
    symmetrize(posterior, k);
    std::copy(posterior.begin(), posterior.end(), covariance.begin());
}

std::vector<std::size_t> segment_offsets(std::span<const std::size_t> segment_lengths,
                                         std::size_t n_samples)
{
    auto offsets = std::vector<std::size_t>();
    offsets.reserve(segment_lengths.size());
    std::size_t total = 0;
    for (std::size_t length : segment_lengths)
    {
        if (length == 0)
        {
            throw std::invalid_argument("segment lengths must be positive");
        }
        offsets.push_back(total);
        total += length;
    }
    if (segment_lengths.empty() || total != n_samples)
    {
        throw std::invalid_argument("segment lengths must sum to the number of samples");
    }
    return offsets;
}

} // namespace

LinearGaussianModel::LinearGaussianModel(LinearGaussianModelState state)
{
    require_model_state(state);
    symmetrize(state.process_covariance, state.state_dim);
    symmetrize(state.observation_covariance, state.observation_dim);
    symmetrize(state.initial_covariance, state.state_dim);
    state_ = std::move(state);
}

std::size_t LinearGaussianModel::state_dim() const noexcept
{
    return state_.state_dim;
}

std::size_t LinearGaussianModel::observation_dim() const noexcept
{
    return state_.observation_dim;
}

const LinearGaussianModelState& LinearGaussianModel::parameters() const noexcept
{
    return state_;
}

void LinearGaussianModel::predict(std::span<double> state, std::span<double> covariance) const
{
    const std::size_t k = state_.state_dim;
    if (state.size() != k || covariance.size() != k * k)
    {
        throw std::invalid_argument("filter state shape is invalid");
    }

    auto next = std::vector<double>(k, 0.0);
    auto scaled = std::vector<double>(k * k, 0.0);
    auto propagated = std::vector<double>(k * k, 0.0);
    kalman_predict(state_, state, covariance, next, scaled, propagated);
}

void LinearGaussianModel::update(std::span<double> state, std::span<double> covariance,
                                 std::span<const double> observation, double jitter) const
{
    const std::size_t k = state_.state_dim;
    const std::size_t m = state_.observation_dim;
    if (state.size() != k || covariance.size() != k * k || observation.size() != m)
    {
        throw std::invalid_argument("filter update shape is invalid");
    }
    require_jitter(jitter);

    auto innovation = std::vector<double>(m, 0.0);
    auto cross = std::vector<double>(m * k, 0.0);
    auto innovation_covariance = std::vector<double>(m * m, 0.0);
    kalman_innovation(state_, state, covariance, observation, jitter, innovation, cross,
                      innovation_covariance);

    // K^T solves S K^T = H P, so no inverse is formed. This is the one step the
    // prepared real-time filter does not share: here the solve is free to
    // allocate and to reach LAPACK, and there it may do neither.
    // A gain that cannot be solved for and one that is not finite are the same
    // fault to a caller, and are reported with the same wording.
    constexpr const char* kSingularInnovation =
        "innovation covariance is singular; supply a positive-definite observation covariance or "
        "an explicit jitter";
    std::vector<double> gain_transposed;
    try
    {
        gain_transposed = solve_spd(innovation_covariance, cross, m, k);
    }
    catch (const std::runtime_error&)
    {
        throw std::runtime_error(kSingularInnovation);
    }
    if (!all_finite(gain_transposed))
    {
        throw std::runtime_error(kSingularInnovation);
    }

    auto factor = std::vector<double>(k * k, 0.0);
    auto left = std::vector<double>(k * k, 0.0);
    auto posterior = std::vector<double>(k * k, 0.0);
    auto noise = std::vector<double>(m * k, 0.0);
    kalman_correct(state_, state, covariance, gain_transposed, innovation, factor, left, posterior,
                   noise);
}

void LinearGaussianModel::filter(std::span<const double> observations, std::size_t n_samples,
                                 std::span<const std::uint8_t> observed, std::span<double> state,
                                 std::span<double> covariance, std::span<double> filtered_states,
                                 std::span<double> filtered_covariances, double jitter) const
{
    const std::size_t k = state_.state_dim;
    const std::size_t m = state_.observation_dim;
    if (n_samples == 0 || observations.size() != n_samples * m || observed.size() != n_samples ||
        filtered_states.size() != n_samples * k || filtered_covariances.size() != n_samples * k * k)
    {
        throw std::invalid_argument("filter batch shape is invalid");
    }

    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        predict(state, covariance);
        if (observed[sample] != 0)
        {
            update(state, covariance, observations.subspan(sample * m, m), jitter);
        }
        std::copy(state.begin(), state.end(),
                  filtered_states.begin() + static_cast<std::ptrdiff_t>(sample * k));
        std::copy(covariance.begin(), covariance.end(),
                  filtered_covariances.begin() + static_cast<std::ptrdiff_t>(sample * k * k));
    }
}

/// Prepared state of the information-form recursion.
///
/// The covariance form the offline `LinearGaussianModel` runs solves an
/// `m x m` system for the gain on every step, which is cubic in the number of
/// observed features. A decoder whose observation is a neural feature vector
/// makes `m` the large dimension: at `m = 512` that solve alone measured
/// 8.1 ms per step, past a 10 ms end-to-end budget on its own.
///
/// This uses the algebraically equivalent gain
/// `K = (P^-1 + H^T R~^-1 H)^-1 H^T R~^-1`, which is cubic in the *state*
/// dimension instead. What makes it worth the change is that `R~ = R + jitter I`
/// and `H` are both fixed when the filter is prepared, so
///
///   A = R~^-1 H          (m x k)
///   M = H^T A            (k x k)
///   G = A^T R A          (k x k)
///
/// are computed once, here, and every step is linear in `m`. With `X` standing
/// for `(P^-1 + M)^-1`, the gain never has to be formed: `K v = X (A^T v)`,
/// `K H = X M`, and `K R K^T = X G X^T` are all `k`-sized once `A^T v` is done.
///
/// The one `m x m` factorization left is the one for `A`, and it happens at
/// construction where allocating and throwing are both allowed.
struct PreparedKalmanFilter::Workspace
{
    Workspace(LinearGaussianModelState parameters, double filter_jitter)
        : model(std::move(parameters)), jitter(filter_jitter)
    {
        const std::size_t k = model.state_dim();
        const std::size_t m = model.observation_dim();
        const auto& fitted = model.parameters();
        state.assign(fitted.initial_state.begin(), fitted.initial_state.end());
        covariance.assign(fitted.initial_covariance.begin(), fitted.initial_covariance.end());
        next_state.assign(k, 0.0);
        scaled.assign(k * k, 0.0);
        propagated.assign(k * k, 0.0);
        innovation.assign(m, 0.0);
        projected.assign(k, 0.0);
        prior_gain.assign(k * k, 0.0);
        system.assign(k * k, 0.0);
        gain.assign(k * k, 0.0);
        factor.assign(k * k, 0.0);
        left.assign(k * k, 0.0);
        posterior.assign(k * k, 0.0);

        // A = R~^-1 H, by solving R~ A = H. Both solves destroy the matrix they
        // are given, so each attempt starts from a fresh copy -- the same
        // Cholesky-then-general order the step used to take per step, paid once.
        std::vector<double> whitened(fitted.observation_covariance.begin(),
                                     fitted.observation_covariance.end());
        add_diagonal(whitened, m, jitter);
        auto factorization = whitened;
        r_inverse_observation.assign(fitted.observation.begin(), fitted.observation.end());
        if (!solve_spd_in_place(factorization, r_inverse_observation, m, k))
        {
            factorization = whitened;
            r_inverse_observation.assign(fitted.observation.begin(), fitted.observation.end());
            if (!solve_linear_in_place(factorization, r_inverse_observation, m, k))
            {
                throw std::invalid_argument("observation covariance plus jitter is not invertible");
            }
        }
        if (!all_finite(r_inverse_observation))
        {
            throw std::invalid_argument("observation covariance whitening is not finite");
        }

        // M = H^T A and N = A^T A, then G = A^T R A = M - jitter N.
        information.assign(k * k, 0.0);
        gain_noise.assign(k * k, 0.0);
        for (std::size_t row = 0; row < k; ++row)
        {
            for (std::size_t col = 0; col < k; ++col)
            {
                double cross_term = 0.0;
                double whitened_term = 0.0;
                for (std::size_t i = 0; i < m; ++i)
                {
                    cross_term += fitted.observation[matrix_index(i, row, k)] *
                                  r_inverse_observation[matrix_index(i, col, k)];
                    whitened_term += r_inverse_observation[matrix_index(i, row, k)] *
                                     r_inverse_observation[matrix_index(i, col, k)];
                }
                information[matrix_index(row, col, k)] = cross_term;
                gain_noise[matrix_index(row, col, k)] = cross_term - jitter * whitened_term;
            }
        }
        symmetrize(information, k);
        symmetrize(gain_noise, k);
    }

    [[nodiscard]] std::size_t scalar_count() const noexcept
    {
        return state.size() + covariance.size() + next_state.size() + scaled.size() +
               propagated.size() + innovation.size() + projected.size() + prior_gain.size() +
               system.size() + gain.size() + factor.size() + left.size() + posterior.size() +
               r_inverse_observation.size() + information.size() + gain_noise.size();
    }

    LinearGaussianModel model;
    double jitter{};
    std::vector<double> state;
    std::vector<double> covariance;
    /// Predict scratch, reused by the update as the corrected state so that
    /// neither half of the update is committed before both are checked.
    std::vector<double> next_state;
    std::vector<double> scaled;
    std::vector<double> propagated;
    std::vector<double> innovation;
    /// A^T v, the only quantity a step reads out of the observation.
    std::vector<double> projected;
    /// P M, then I + P M in `system`.
    std::vector<double> prior_gain;
    std::vector<double> system;
    /// X = (P^-1 + M)^-1.
    std::vector<double> gain;
    std::vector<double> factor;
    std::vector<double> left;
    std::vector<double> posterior;
    /// Prepared: A = R~^-1 H, M = H^T A, G = A^T R A.
    std::vector<double> r_inverse_observation;
    std::vector<double> information;
    std::vector<double> gain_noise;
};

PreparedKalmanFilter::PreparedKalmanFilter(LinearGaussianModelState state, double jitter)
{
    require_jitter(jitter);
    workspace_ = std::make_unique<Workspace>(std::move(state), jitter);
}

PreparedKalmanFilter::~PreparedKalmanFilter() = default;
PreparedKalmanFilter::PreparedKalmanFilter(PreparedKalmanFilter&&) noexcept = default;
PreparedKalmanFilter& PreparedKalmanFilter::operator=(PreparedKalmanFilter&&) noexcept = default;

std::size_t PreparedKalmanFilter::state_dim() const noexcept
{
    return workspace_->model.state_dim();
}

std::size_t PreparedKalmanFilter::observation_dim() const noexcept
{
    return workspace_->model.observation_dim();
}

double PreparedKalmanFilter::jitter() const noexcept
{
    return workspace_->jitter;
}

const LinearGaussianModelState& PreparedKalmanFilter::parameters() const noexcept
{
    return workspace_->model.parameters();
}

std::size_t PreparedKalmanFilter::workspace_bytes() const noexcept
{
    return workspace_->scalar_count() * sizeof(double);
}

void PreparedKalmanFilter::reset() noexcept
{
    const auto& parameters = workspace_->model.parameters();
    std::copy(parameters.initial_state.begin(), parameters.initial_state.end(),
              workspace_->state.begin());
    std::copy(parameters.initial_covariance.begin(), parameters.initial_covariance.end(),
              workspace_->covariance.begin());
}

KalmanStepStatus PreparedKalmanFilter::step(std::span<const double> observation, bool observed,
                                            bool advance) noexcept
{
    auto& work = *workspace_;
    const auto& parameters = work.model.parameters();
    const std::size_t k = parameters.state_dim;
    const std::size_t m = parameters.observation_dim;
    if (observed && observation.size() != m)
    {
        return KalmanStepStatus::invalid_argument;
    }
    if (advance)
    {
        kalman_predict(parameters, work.state, work.covariance, work.next_state, work.scaled,
                       work.propagated);
    }
    if (!observed)
    {
        return KalmanStepStatus::ok;
    }

    // The two passes over the observation, and the only work that grows with m.
    // innovation = z - (H x + d); projected = A^T innovation.
    for (std::size_t row = 0; row < m; ++row)
    {
        double predicted = parameters.observation_offset[row];
        for (std::size_t col = 0; col < k; ++col)
        {
            predicted += parameters.observation[matrix_index(row, col, k)] * work.state[col];
        }
        work.innovation[row] = observation[row] - predicted;
    }
    for (std::size_t col = 0; col < k; ++col)
    {
        double value = 0.0;
        for (std::size_t row = 0; row < m; ++row)
        {
            value += work.r_inverse_observation[matrix_index(row, col, k)] * work.innovation[row];
        }
        work.projected[col] = value;
    }

    // X = (P^-1 + M)^-1, obtained as (I + P M)^-1 P so that a prior covariance
    // that has lost rank -- a state direction pinned to zero variance -- is
    // still handled. P M is a product of two positive semidefinite matrices, so
    // I + P M has eigenvalues at or above one and the system is well posed from
    // below whatever the data does.
    gemm_nn_scalar(work.covariance, work.information, k, k, k, work.prior_gain);
    std::copy(work.prior_gain.begin(), work.prior_gain.end(), work.system.begin());
    add_diagonal(work.system, k, 1.0);
    std::copy(work.covariance.begin(), work.covariance.end(), work.gain.begin());
    if (!solve_linear_in_place(work.system, work.gain, k, k))
    {
        return KalmanStepStatus::numerical_failure;
    }
    symmetrize(work.gain, k);
    if (!all_finite(work.gain))
    {
        return KalmanStepStatus::numerical_failure;
    }

    // x <- x + K v, with K v = X (A^T v) and the gain never formed.
    //
    // Written into next_state rather than in place. The finiteness check below
    // can still return numerical_failure, and on that path the state has to be
    // the prior it was on entry: KalmanDecoderAdapter maps the failure to
    // processor_failure without clearing the stream state, so a state that had
    // already absorbed a non-finite correction would poison every later frame
    // until something outside reset the filter. next_state is free here --
    // kalman_predict copies it into state before returning, and nothing else
    // in a step reads it -- so this costs no workspace.
    for (std::size_t row = 0; row < k; ++row)
    {
        double correction = 0.0;
        for (std::size_t col = 0; col < k; ++col)
        {
            correction += work.gain[matrix_index(row, col, k)] * work.projected[col];
        }
        work.next_state[row] = work.state[row] + correction;
    }

    // Joseph form, k-sized throughout: K H = X M and K R K^T = X G X^T.
    gemm_nn_scalar(work.gain, work.information, k, k, k, work.factor);
    for (std::size_t row = 0; row < k; ++row)
    {
        for (std::size_t col = 0; col < k; ++col)
        {
            const double value = work.factor[matrix_index(row, col, k)];
            work.factor[matrix_index(row, col, k)] = (row == col ? 1.0 : 0.0) - value;
        }
    }
    gemm_nn_scalar(work.factor, work.covariance, k, k, k, work.left);
    gemm_nt_scalar(work.left, work.factor, k, k, k, work.posterior);

    gemm_nn_scalar(work.gain, work.gain_noise, k, k, k, work.left);
    gemm_nt_scalar(work.left, work.gain, k, k, k, work.prior_gain);
    for (std::size_t i = 0; i < k * k; ++i)
    {
        work.posterior[i] += work.prior_gain[i];
    }
    symmetrize(work.posterior, k);
    if (!all_finite(work.posterior) || !all_finite(work.next_state))
    {
        return KalmanStepStatus::numerical_failure;
    }
    // Both halves of the update commit together, after both have been checked.
    std::copy(work.next_state.begin(), work.next_state.end(), work.state.begin());
    std::copy(work.posterior.begin(), work.posterior.end(), work.covariance.begin());
    return KalmanStepStatus::ok;
}

std::span<const double> PreparedKalmanFilter::state() const noexcept
{
    return workspace_->state;
}

std::span<const double> PreparedKalmanFilter::covariance() const noexcept
{
    return workspace_->covariance;
}

StateSpaceFitResult fit_linear_gaussian(std::span<const double> states,
                                        std::span<const double> observations,
                                        std::span<const std::size_t> segment_lengths,
                                        std::size_t state_dim, std::size_t observation_dim,
                                        bool fit_offsets, double jitter)
{
    if (state_dim == 0 || observation_dim == 0)
    {
        throw std::invalid_argument("state and observation dimensions must be positive");
    }
    if (states.size() % state_dim != 0)
    {
        throw std::invalid_argument("states shape is invalid");
    }
    const std::size_t n_samples = states.size() / state_dim;
    if (n_samples == 0 || observations.size() != n_samples * observation_dim)
    {
        throw std::invalid_argument("observations shape is invalid");
    }
    require_jitter(jitter);
    const auto offsets = segment_offsets(segment_lengths, n_samples);
    const std::size_t n_segments = offsets.size();
    if (n_samples <= n_segments)
    {
        throw std::invalid_argument("at least one segment must contain two samples");
    }
    const std::size_t n_transitions = n_samples - n_segments;

    // Stack the within-segment transition pairs; no pair crosses a boundary.
    auto previous = std::vector<double>(n_transitions * state_dim, 0.0);
    auto next = std::vector<double>(n_transitions * state_dim, 0.0);
    std::size_t pair = 0;
    for (std::size_t segment = 0; segment < n_segments; ++segment)
    {
        const std::size_t start = offsets[segment];
        const std::size_t stop = start + segment_lengths[segment];
        for (std::size_t sample = start + 1; sample < stop; ++sample)
        {
            for (std::size_t i = 0; i < state_dim; ++i)
            {
                previous[matrix_index(pair, i, state_dim)] =
                    states[matrix_index(sample - 1, i, state_dim)];
                next[matrix_index(pair, i, state_dim)] = states[matrix_index(sample, i, state_dim)];
            }
            ++pair;
        }
    }

    const auto transition_fit =
        fit_linear_model(previous, n_transitions, state_dim, next, state_dim, 0.0, fit_offsets);
    const auto observation_fit = fit_linear_model(states, n_samples, state_dim, observations,
                                                  observation_dim, 0.0, fit_offsets);

    const auto process_residuals =
        affine_residuals(previous, next, transition_fit.coef, transition_fit.intercept,
                         n_transitions, state_dim, state_dim);
    const auto observation_residuals =
        affine_residuals(states, observations, observation_fit.coef, observation_fit.intercept,
                         n_samples, state_dim, observation_dim);

    auto process_covariance = residual_covariance(process_residuals, n_transitions, state_dim);
    auto observation_covariance =
        residual_covariance(observation_residuals, n_samples, observation_dim);

    // Initial state and covariance come from the first sample of every segment.
    auto initial_state = std::vector<double>(state_dim, 0.0);
    for (std::size_t segment = 0; segment < n_segments; ++segment)
    {
        for (std::size_t i = 0; i < state_dim; ++i)
        {
            initial_state[i] += states[matrix_index(offsets[segment], i, state_dim)];
        }
    }
    for (double& value : initial_state)
    {
        value /= static_cast<double>(n_segments);
    }
    auto initial_deviations = std::vector<double>(n_segments * state_dim, 0.0);
    for (std::size_t segment = 0; segment < n_segments; ++segment)
    {
        for (std::size_t i = 0; i < state_dim; ++i)
        {
            initial_deviations[matrix_index(segment, i, state_dim)] =
                states[matrix_index(offsets[segment], i, state_dim)] - initial_state[i];
        }
    }
    auto initial_covariance = residual_covariance(initial_deviations, n_segments, state_dim);

    add_diagonal(process_covariance, state_dim, jitter);
    add_diagonal(observation_covariance, observation_dim, jitter);
    add_diagonal(initial_covariance, state_dim, jitter);

    StateSpaceFitResult result;
    result.n_samples = n_samples;
    result.n_segments = n_segments;
    result.n_transitions = n_transitions;
    result.model.state_dim = state_dim;
    result.model.observation_dim = observation_dim;
    result.model.transition = transition_fit.coef;
    result.model.transition_offset = transition_fit.intercept;
    result.model.observation = observation_fit.coef;
    result.model.observation_offset = observation_fit.intercept;
    result.model.process_covariance = std::move(process_covariance);
    result.model.observation_covariance = std::move(observation_covariance);
    result.model.initial_state = std::move(initial_state);
    result.model.initial_covariance = std::move(initial_covariance);
    return result;
}

} // namespace neurale::models
