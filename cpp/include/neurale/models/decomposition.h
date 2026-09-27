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

struct PcaFitResult
{
    std::size_t n_samples{};
    std::size_t n_features{};
    std::size_t n_components{};
    bool center{};
    std::vector<double> mean;
    std::vector<double> components;
    std::vector<double> offset;
    std::vector<double> singular_values;
    std::vector<double> explained_variance;
    std::vector<double> explained_variance_ratio;
};

struct PcaModelState
{
    std::size_t n_features{};
    std::size_t n_components{};
    bool center{};
    std::vector<double> mean;
    std::vector<double> components;
    std::vector<double> offset;
};

class PcaModel
{
  public:
    explicit PcaModel(PcaModelState state);

    [[nodiscard]] std::size_t n_features() const noexcept;
    [[nodiscard]] std::size_t n_components() const noexcept;
    [[nodiscard]] bool center() const noexcept;
    [[nodiscard]] const std::vector<double>& mean() const noexcept;
    [[nodiscard]] const std::vector<double>& components() const noexcept;
    [[nodiscard]] const std::vector<double>& offset() const noexcept;

    void transform(std::span<const double> X, std::size_t n_samples,
                   std::span<double> output) const;

  private:
    PcaModelState state_;
    bool has_offset_{};
};

PcaFitResult fit_pca(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                     std::size_t n_components, bool center);

void transform_pca(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                   std::span<const double> components, std::span<const double> offset,
                   std::size_t n_components, std::span<double> output);

} // namespace neurale::models
