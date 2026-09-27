/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace neurale::models
{

namespace cuda
{
class GaussianKdeModel;
}

enum class BandwidthRule
{
    Scott,
    Silverman,
};

class GaussianKdeModel
{
  public:
    GaussianKdeModel(const GaussianKdeModel&) = delete;
    GaussianKdeModel(GaussianKdeModel&&) noexcept = default;
    GaussianKdeModel& operator=(const GaussianKdeModel&) = delete;
    GaussianKdeModel& operator=(GaussianKdeModel&&) noexcept = default;

    [[nodiscard]] std::size_t n_samples() const noexcept;
    [[nodiscard]] std::size_t n_features() const noexcept;
    [[nodiscard]] double bandwidth_factor() const noexcept;

    void pdf(std::span<const double> X, std::size_t nX, std::span<double> output) const;

    void logpdf(std::span<const double> X, std::size_t nX, std::span<double> output) const;

  private:
    struct State
    {
        std::size_t n_samples{};
        std::size_t n_features{};
        double bandwidth_factor{};
        double log_normalizer{};
        std::vector<double> mean;
        std::vector<double> whitened_samples;
        std::vector<double> lower_cholesky;
    };

    explicit GaussianKdeModel(State state);

    void evaluate_logpdf(std::span<const double> X, std::size_t nX, std::span<double> output) const;

    void evaluate_logpdf_impl(std::span<const double> X, std::size_t nX,
                              std::span<double> output) const;

    [[nodiscard]] double point_logpdf(std::span<const double> whitened_point) const;

    State state_;

    friend GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                             std::size_t n_features, BandwidthRule bandwidth_rule);

    friend GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                             std::size_t n_features, double bandwidth_factor);

    friend class cuda::GaussianKdeModel;
};

[[nodiscard]]
GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                  std::size_t n_features, BandwidthRule bandwidth_rule);

[[nodiscard]]
GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                  std::size_t n_features, double bandwidth_factor);

} // namespace neurale::models
