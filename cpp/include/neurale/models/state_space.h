/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace neurale::models
{

/// Immutable parameters of a linear-Gaussian state-space model.
///
/// The model is affine in both equations:
///
///     x[t] = transition * x[t-1] + transition_offset + w,  w ~ N(0, process_covariance)
///     z[t] = observation * x[t]  + observation_offset + v, v ~ N(0, observation_covariance)
struct LinearGaussianModelState
{
    std::size_t state_dim{};
    std::size_t observation_dim{};
    /// Row-major ``state_dim x state_dim``.
    std::vector<double> transition;
    /// Length ``state_dim``.
    std::vector<double> transition_offset;
    /// Row-major ``observation_dim x state_dim``.
    std::vector<double> observation;
    /// Length ``observation_dim``.
    std::vector<double> observation_offset;
    /// Row-major ``state_dim x state_dim``, symmetric positive semidefinite.
    std::vector<double> process_covariance;
    /// Row-major ``observation_dim x observation_dim``, symmetric positive semidefinite.
    std::vector<double> observation_covariance;
    /// Length ``state_dim``.
    std::vector<double> initial_state;
    /// Row-major ``state_dim x state_dim``, symmetric positive semidefinite.
    std::vector<double> initial_covariance;
};

/// Estimated parameters and the counts the estimator derived them from.
struct StateSpaceFitResult
{
    LinearGaussianModelState model;
    std::size_t n_samples{};
    std::size_t n_segments{};
    std::size_t n_transitions{};
};

/// Immutable model parameters plus the Kalman recursion that operates on
/// caller-owned mutable state.
///
/// The filter state never lives in this class: ``predict`` and ``update`` take
/// the state vector and covariance as mutable spans and advance them in place,
/// so one set of fitted parameters can drive any number of independent filters.
///
/// The recursion allocates its temporaries on every step. This is an offline
/// and near-line path, not a real-time one; a streaming adapter needs a
/// prepared workspace rather than a copy of the arithmetic below.
class LinearGaussianModel
{
  public:
    explicit LinearGaussianModel(LinearGaussianModelState state);

    [[nodiscard]] std::size_t state_dim() const noexcept;
    [[nodiscard]] std::size_t observation_dim() const noexcept;
    [[nodiscard]] const LinearGaussianModelState& parameters() const noexcept;

    /// Advance one step through the transition equation.
    void predict(std::span<double> state, std::span<double> covariance) const;

    /// Correct the predicted state with one fully observed measurement.
    ///
    /// The gain is obtained from a symmetric linear solve of the innovation
    /// covariance, and the posterior covariance uses the Joseph form, which
    /// stays symmetric and positive semidefinite under roundoff. ``jitter`` is
    /// added to the diagonal of the innovation covariance before the solve;
    /// nothing is added when it is zero.
    void update(std::span<double> state, std::span<double> covariance,
                std::span<const double> observation, double jitter) const;

    /// Run predict/update over a batch, writing the posterior of every step.
    ///
    /// Every row is one step: the incoming ``state`` is the posterior of the
    /// step before the first row, each row is that step's measurement, and each
    /// written row is that step's posterior. ``observed`` selects the steps that
    /// carry a measurement; a zero entry performs the predict step only.
    /// ``state`` and ``covariance`` are advanced in place and hold the last
    /// posterior when the call returns. A caller whose state is already the
    /// prior of the first row corrects that row with ``update`` and passes the
    /// rest here.
    void filter(std::span<const double> observations, std::size_t n_samples,
                std::span<const std::uint8_t> observed, std::span<double> state,
                std::span<double> covariance, std::span<double> filtered_states,
                std::span<double> filtered_covariances, double jitter) const;

  private:
    LinearGaussianModelState state_;
};

/// Outcome of one step of the prepared Kalman recursion.
enum class KalmanStepStatus : std::uint8_t
{
    ok,
    /// The observation span does not match the prepared observation dimension.
    invalid_argument,
    /// The innovation covariance could not be factorized, or the gain it
    /// produced was not finite.
    numerical_failure,
};

/// The Kalman recursion with every buffer it needs allocated up front.
///
/// This is the same arithmetic ``LinearGaussianModel`` runs -- the predict, the
/// symmetric gain solve, and the Joseph-form posterior all come from the same
/// kernels -- with the temporaries hoisted out of the step and into the object.
/// That is what makes it usable from a real-time thread: after construction a
/// step allocates nothing, throws nothing, and takes no lock.
///
/// The filter owns its state, unlike ``LinearGaussianModel``, because the
/// caller that owns the state on a real-time thread would otherwise have to own
/// the workspace too. ``reset`` restores the prepared initial state and
/// covariance, which is where a fitted model starts a segment.
///
/// Failure is a return value rather than an exception. A model whose innovation
/// covariance cannot be factorized is a fault the caller reports through its own
/// path; nothing here unwinds, and the state is left where the last successful
/// step put it.
class PreparedKalmanFilter
{
  public:
    /// Validate the parameters and allocate the whole workspace.
    ///
    /// Throws exactly what ``LinearGaussianModel`` throws for an invalid model
    /// or a negative jitter. This is the control-plane half of the class; every
    /// other member is the real-time half.
    PreparedKalmanFilter(LinearGaussianModelState state, double jitter);
    ~PreparedKalmanFilter();

    PreparedKalmanFilter(const PreparedKalmanFilter&) = delete;
    PreparedKalmanFilter& operator=(const PreparedKalmanFilter&) = delete;
    PreparedKalmanFilter(PreparedKalmanFilter&&) noexcept;
    PreparedKalmanFilter& operator=(PreparedKalmanFilter&&) noexcept;

    [[nodiscard]] std::size_t state_dim() const noexcept;
    [[nodiscard]] std::size_t observation_dim() const noexcept;
    [[nodiscard]] double jitter() const noexcept;
    [[nodiscard]] const LinearGaussianModelState& parameters() const noexcept;

    /// Bytes of state and scratch the prepared filter holds.
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;

    /// Restore the prepared initial state and covariance.
    void reset() noexcept;

    /// Advance one step over one observation row.
    ///
    /// ``advance`` selects the prior of this row: ``true`` runs the transition
    /// first, which is what continuing a stream costs, and ``false`` treats the
    /// current state as this row's own prior, which is the first row after a
    /// ``reset``. ``observed`` false runs the predict step alone, which is what
    /// an absent measurement means.
    [[nodiscard]] KalmanStepStatus step(std::span<const double> observation, bool observed,
                                        bool advance) noexcept;

    [[nodiscard]] std::span<const double> state() const noexcept;
    [[nodiscard]] std::span<const double> covariance() const noexcept;

  private:
    struct Workspace;
    std::unique_ptr<Workspace> workspace_;
};

/// Estimate the model from aligned state and observation sequences.
///
/// ``segment_lengths`` splits the rows into independent segments; no transition
/// is formed across a segment boundary. The transition and observation maps are
/// least-squares fits, the covariances are the maximum-likelihood residual
/// covariances, and the initial state and covariance are the mean and
/// maximum-likelihood covariance of the per-segment first states. ``jitter`` is
/// added to the diagonal of every estimated covariance; nothing is added when it
/// is zero, so a single segment yields an exactly zero initial covariance.
StateSpaceFitResult fit_linear_gaussian(std::span<const double> states,
                                        std::span<const double> observations,
                                        std::span<const std::size_t> segment_lengths,
                                        std::size_t state_dim, std::size_t observation_dim,
                                        bool fit_offsets, double jitter);

} // namespace neurale::models
