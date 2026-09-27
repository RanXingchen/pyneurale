/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace neurale::models
{

struct LdaFitResult
{
    std::size_t n_features{};
    std::size_t n_classes{};
    std::size_t n_outputs{};
    std::vector<double> priors;
    std::vector<double> means;
    std::vector<double> covariance;
    std::vector<double> coef;
    std::vector<double> intercept;
};

struct LdaModelState
{
    std::size_t n_features{};
    std::size_t n_classes{};
    std::size_t n_outputs{};
    std::vector<double> coef;
    std::vector<double> intercept;
};

class LdaModel
{
  public:
    explicit LdaModel(LdaModelState state);

    [[nodiscard]] std::size_t n_features() const noexcept;
    [[nodiscard]] std::size_t n_classes() const noexcept;
    [[nodiscard]] std::size_t n_outputs() const noexcept;
    [[nodiscard]] const std::vector<double>& coef() const noexcept;
    [[nodiscard]] const std::vector<double>& intercept() const noexcept;

    void decision_function(std::span<const double> X, std::size_t n_samples,
                           std::span<double> output) const;

    void predict_proba(std::span<const double> X, std::size_t n_samples,
                       std::span<double> output) const;

  private:
    LdaModelState state_;
};

LdaFitResult fit_lda(std::span<const double> X, std::span<const std::size_t> y,
                     std::size_t n_samples, std::size_t n_features, std::size_t n_classes,
                     const std::string& shrinkage_kind, double shrinkage_value);

void lda_decision_function(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                           std::span<const double> coef, std::size_t n_outputs,
                           std::span<const double> intercept, std::span<double> output);

void lda_predict_proba(std::span<const double> X, std::size_t n_samples, std::size_t n_features,
                       std::span<const double> coef, std::size_t n_outputs,
                       std::span<const double> intercept, std::span<double> output);

} // namespace neurale::models
