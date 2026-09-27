/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/neighbors.h>

#include <cstddef>
#include <span>
#include <vector>

namespace neurale::models
{

enum class LppProjectionMethod
{
    Opp,
    Lpp,
};

struct LppFitResult
{
    std::size_t n_samples{};
    std::size_t n_features{};
    std::size_t n_components{};
    std::vector<double> components;
};

struct LppModelState
{
    std::size_t n_features{};
    std::size_t n_components{};
    std::vector<double> components;
};

class LppModel
{
  public:
    explicit LppModel(LppModelState state);

    [[nodiscard]] std::size_t n_features() const noexcept;
    [[nodiscard]] std::size_t n_components() const noexcept;
    [[nodiscard]] const std::vector<double>& components() const noexcept;

    void transform(std::span<const double> X, std::size_t n_samples,
                   std::span<double> output) const;

  private:
    LppModelState state_;
};

LppFitResult fit_lpp(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t n_components, std::size_t n_neighbors, DistanceMetric metric,
                     LppProjectionMethod method);

} // namespace neurale::models
