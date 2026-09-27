/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/density.h>

#include <cstddef>
#include <memory>
#include <span>

namespace neurale::models::cuda
{

class GaussianKdeModel
{
  public:
    GaussianKdeModel(GaussianKdeModel&&) noexcept;
    GaussianKdeModel& operator=(GaussianKdeModel&&) noexcept;
    ~GaussianKdeModel();

    GaussianKdeModel(const GaussianKdeModel&) = delete;
    GaussianKdeModel& operator=(const GaussianKdeModel&) = delete;

    [[nodiscard]] std::size_t n_samples() const noexcept;
    [[nodiscard]] std::size_t n_features() const noexcept;
    [[nodiscard]] double bandwidth_factor() const noexcept;

    void pdf(std::span<const double> X, std::size_t nX, std::span<double> output);

    void logpdf(std::span<const double> X, std::size_t nX, std::span<double> output);

  private:
    struct Impl;

    explicit GaussianKdeModel(neurale::models::GaussianKdeModel model, int device_id);

    std::unique_ptr<Impl> impl_;

    friend GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                             std::size_t n_features, BandwidthRule bandwidth_rule,
                                             int device_id);

    friend GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                             std::size_t n_features, double bandwidth_factor,
                                             int device_id);
};

[[nodiscard]] GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                                std::size_t n_features,
                                                BandwidthRule bandwidth_rule, int device_id);

[[nodiscard]] GaussianKdeModel fit_gaussian_kde(std::span<const double> X, std::size_t n_samples,
                                                std::size_t n_features, double bandwidth_factor,
                                                int device_id);

} // namespace neurale::models::cuda
