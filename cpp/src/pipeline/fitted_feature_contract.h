/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The fitted-decoder vocabulary shared by the decoder adapters.
///
/// A decoder adapter carries the model parameters *and* the description of the
/// features those parameters were estimated on. That description is what makes
/// the parameters applicable, and checking it is identical work whatever the
/// model is: the Kalman recursion and the affine predictor both fail the same
/// way when handed features they were not fitted on -- every matrix still
/// multiplies, and the decoded values are silently wrong.
///
/// So the contract, the scaler vocabulary, and the two checks live here once.
/// The adapter-specific parts -- what a missing observation means, what the
/// output channels are -- stay with the adapter that has an opinion about them.
///
/// Nothing here runs on the real-time thread except `apply_feature_scaling`,
/// which is `noexcept` and touches only the caller's row. The throwing
/// functions are configuration- and prepare-time by construction.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <neurale/streaming/schema.h>

#include "adapter_support.h"

namespace neurale::pipeline
{

/// The fitted feature scaling a decoder applies before the model sees a row.
enum class FeatureScaling : std::uint8_t
{
    none,
    /// ``(x - center) / scale``, the fitted standard scaler.
    standard,
    /// ``x * scale + center``, the fitted min-max scaler.
    minmax,
};

/// The fitted feature-set contract a decoder was trained under, frozen into the
/// adapter at configuration time so ``prepare`` can compare it against the
/// runtime descriptor rather than trusting a session-local feature-set id.
///
/// Every field here is a value, not an identifier: feature-set and unit ids are
/// session-local and a different session may reuse the same number for a
/// different feature set. What a decoder needs is the content the Python
/// ``FeatureSchema`` checks -- names, units, rate, window, shift, source, and
/// algorithm identity -- because those are what make the model parameters and
/// scaler statistics applicable.
struct FittedFeatureContract
{
    /// Complete feature names in column order, including columns the decoder
    /// does not select. The Python decoders check the full schema before
    /// selection; a native adapter that checked only selected names would be
    /// weaker, and a reordering or renaming of unselected columns would pass.
    std::vector<std::string> feature_names;
    /// Canonical unit symbols, one per feature. Resolved from the unit registry
    /// at configuration time, so the comparison at prepare does not depend on
    /// session-local unit ids.
    std::vector<std::string> feature_unit_symbols;
    /// Observation rate of the feature stream the decoder was fitted on.
    streaming::RationalRate observation_rate{};
    /// Window length in nanoseconds.
    std::uint64_t window_length_ns{};
    /// Shift between adjacent observations in nanoseconds.
    std::uint64_t shift_ns{};
    /// Algorithm that produced the features.
    std::string algorithm_name;
    std::string algorithm_version;
    /// Name of the source stream in the session that produced the fit.
    std::string source_stream;
    streaming::FeatureTimestampReference timestamp_reference{
        streaming::FeatureTimestampReference::window_center};
};

/// Refuse a contract that cannot describe any input, naming @p adapter.
inline void validate_fitted_contract(const FittedFeatureContract& contract,
                                     std::string_view adapter)
{
    if (contract.feature_names.empty() ||
        contract.feature_unit_symbols.size() != contract.feature_names.size())
    {
        throw std::invalid_argument(std::string{adapter} +
                                    " adapter fitted feature contract is incomplete");
    }
    if (contract.observation_rate.denominator == 0 ||
        ((contract.observation_rate.numerator == 0) != (contract.shift_ns == 0)))
    {
        throw std::invalid_argument(
            std::string{adapter} +
            " adapter fitted feature contract observation rate must be positive");
    }
}

/// Refuse scaler statistics that do not describe @p n_selected features.
inline void validate_feature_scaler(FeatureScaling scaling, std::size_t n_selected,
                                    std::span<const double> center, std::span<const double> scale,
                                    std::string_view adapter)
{
    const auto width = scaling == FeatureScaling::none ? std::size_t{0} : n_selected;
    if (center.size() != width || scale.size() != width)
    {
        throw std::invalid_argument(std::string{adapter} +
                                    " adapter scaler statistics must match the selected feature "
                                    "count");
    }
    for (const double divisor : scale)
    {
        if (!std::isfinite(divisor) || divisor == 0.0)
        {
            throw std::invalid_argument(std::string{adapter} +
                                        " adapter scaler scale must be finite and nonzero");
        }
    }
    for (const double value : center)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string{adapter} +
                                        " adapter scaler center must be finite");
        }
    }
}

/// Refuse a runtime feature stream that is not the one @p contract describes.
///
/// This is the same check the Python ``FeatureSchema`` enforces, applied to
/// every column and not only to the selected ones: a native adapter weaker than
/// the offline decoder would accept inputs the Python one refuses.
inline void match_fitted_contract(const FittedFeatureContract& contract,
                                  const streaming::StreamSchema& schema,
                                  const streaming::FeatureSetDescriptor& descriptor,
                                  const streaming::SignalSchema& input,
                                  const adapter_support::Checked& checked, std::string_view adapter)
{
    const std::string name{adapter};
    if (descriptor.feature_names != contract.feature_names)
    {
        throw std::invalid_argument(
            name + " adapter input feature names do not match the fitted contract");
    }
    if (descriptor.timestamp_reference != contract.timestamp_reference)
    {
        throw std::invalid_argument(
            name + " adapter input feature timestamp reference does not match the fitted contract");
    }
    if (descriptor.window_length_ns != contract.window_length_ns)
    {
        throw std::invalid_argument(
            name + " adapter input feature window length does not match the fitted contract");
    }
    if (descriptor.shift_ns != contract.shift_ns)
    {
        throw std::invalid_argument(
            name + " adapter input feature shift does not match the fitted contract");
    }
    // Cross-multiplication rather than exact numerator/denominator equality:
    // ``StreamSchema`` does not require ``RationalRate`` to be reduced, so
    // ``1000/2`` and ``500/1`` are the same rate and must compare equal.
    const auto rate_left =
        checked.checked_multiply(input.fs.numerator, contract.observation_rate.denominator);
    const auto rate_right =
        checked.checked_multiply(contract.observation_rate.numerator, input.fs.denominator);
    if (rate_left != rate_right)
    {
        throw std::invalid_argument(
            name + " adapter input observation rate does not match the fitted contract");
    }
    if (descriptor.source_stream != contract.source_stream)
    {
        throw std::invalid_argument(
            name + " adapter input source stream does not match the fitted contract");
    }
    if (descriptor.algorithm_name != contract.algorithm_name)
    {
        throw std::invalid_argument(
            name + " adapter input algorithm name does not match the fitted contract");
    }
    if (descriptor.algorithm_version != contract.algorithm_version)
    {
        throw std::invalid_argument(
            name + " adapter input algorithm version does not match the fitted contract");
    }
    // Units are compared by canonical symbol, not by session-local unit id: a
    // different session may assign the same id to a different unit, and a
    // decoder cares about what the feature is measured in, not about the
    // registry it was registered under.
    for (std::size_t i = 0; i < descriptor.unit_ids.size(); ++i)
    {
        const auto* unit = schema.units().find(descriptor.unit_ids[i]);
        if (unit == nullptr)
        {
            throw std::invalid_argument(
                name + " adapter input feature-set descriptor references an unregistered unit");
        }
        if (unit->symbol != contract.feature_unit_symbols[i])
        {
            throw std::invalid_argument(
                name + " adapter input feature unit symbols do not match the fitted contract");
        }
    }
}

/// Scale one gathered observation row in place.
void apply_feature_scaling(FeatureScaling scaling, std::span<double> row,
                           std::span<const double> center, std::span<const double> scale) noexcept;

} // namespace neurale::pipeline
