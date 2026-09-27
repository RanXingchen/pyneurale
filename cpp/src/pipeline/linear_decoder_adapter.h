/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <neurale/models/linear_model.h>
#include <neurale/streaming/processor.h>

#include "decoder_training_capture.h"
#include "fitted_feature_contract.h"

namespace neurale::pipeline
{

/// Control-plane metadata and fixed geometry for the private linear adapter.
///
/// Everything a decoded frame depends on is here, already in native form: the
/// fitted coefficients, the fitted scaling, the feature selection, and the
/// names that selection is expected to resolve to. Parsing an artifact and
/// converting Python objects happen wherever this struct is built, which is
/// never on the data plane -- after ``prepare`` the adapter reads none of it
/// again.
///
/// Temporal context is deliberately absent. Stacking neighbouring observations
/// is an upstream rearrangement of the design matrix, not part of an affine
/// predictor; a fit that used it presents its stacked columns as ordinary
/// features here, and ``selection`` and ``selected_feature_names`` describe
/// them like any others. That is what keeps this adapter stateless.
struct LinearDecoderAdapterConfig
{
    streaming::SchemaId output_schema_id{};
    streaming::SignalId output_signal_id{};
    streaming::ChannelSetId output_channel_set_id{};
    /// Unit of the decoded target channels.
    streaming::PhysicalUnit output_physical_unit{streaming::PhysicalUnit::unspecified};

    /// Feature set the decoder was fitted on. The input signal must declare it.
    streaming::FeatureSetId feature_set_id{};
    /// Columns of one input observation the model consumes, in model order.
    std::vector<std::size_t> selection;
    /// Names those columns must carry, in the same order. Checked against the
    /// input feature-set descriptor at prepare: a selection of the right length
    /// pointing at the wrong columns feeds the model different neural features
    /// under the names the decoder reports, and every matrix still multiplies.
    std::vector<std::string> selected_feature_names;

    /// The full feature-set contract the decoder was fitted under.
    FittedFeatureContract fitted_feature_contract;

    FeatureScaling scaling{FeatureScaling::none};
    /// Per-selected-feature ``mean_`` (standard) or ``min_`` (min-max).
    std::vector<double> scaler_center;
    /// Per-selected-feature ``scale_``.
    std::vector<double> scaler_scale;

    /// Fitted affine parameters, ``n_outputs x n_features`` row-major.
    models::LinearModelState model;
    /// Non-empty for LDA classification: model-order labels, exactly representable as float64.
    std::vector<double> classes;
};

/// Internal adapter decoding one prepared feature stream with a linear or LDA model.
///
/// The input is one fixed regular ``SignalKind::feature`` stream whose
/// descriptor must match the contract the decoder was fitted under; the output
/// is one fixed sampled stream carrying the decoded target, one channel per
/// regression output or one numeric class label, on the input's clock and rate.
/// Every observation in an input block is decoded and the whole block leaves as
/// one output frame that keeps the input's observation timing and index.
///
/// Prediction is stateless, which is the substantive difference from
/// ``KalmanDecoderAdapter``: the same observations decode to the same values in
/// any order, ``reset`` has nothing to return to, and a discontinuity costs
/// nothing to absorb. A non-finite selected feature is a fault rather than a
/// missing measurement -- there is no recursion to carry a state forward
/// through an absent row, so an affine model has nothing to answer with.
///
/// The arithmetic is ``models::LinearModel::predict`` or
/// ``models::LdaModel::decision_function``, called once for the
/// whole block -- the adapter holds none of its own. Under MKL that reaches
/// ``cblas_dgemm`` with the local thread count pinned, which is what makes it
/// admissible here; see the declaration of ``predict``. After ``prepare``
/// nothing here allocates, logs, calls Python, or touches a device.
class LinearDecoderAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit LinearDecoderAdapter(LinearDecoderAdapterConfig config);
    ~LinearDecoderAdapter() override;

    LinearDecoderAdapter(const LinearDecoderAdapter&) = delete;
    LinearDecoderAdapter& operator=(const LinearDecoderAdapter&) = delete;
    LinearDecoderAdapter(LinearDecoderAdapter&&) = delete;
    LinearDecoderAdapter& operator=(LinearDecoderAdapter&&) = delete;

    [[nodiscard]] streaming::PreparedProcessorContract
    prepare(const streaming::ProcessorPrepareContext& context) override;

    [[nodiscard]] streaming::StreamStatus
    process(streaming::FrameBorrow& frame, streaming::FrameEmitter& emitter) noexcept override;

    [[nodiscard]] streaming::StreamStatus
    handle_discontinuity(const streaming::Discontinuity& discontinuity) noexcept override;

    [[nodiscard]] streaming::StreamStatus flush(streaming::FrameEmitter& emitter) noexcept override;

    [[nodiscard]] streaming::StreamStatus reset() noexcept override;

    void enable_training_capture(std::size_t capacity);
    [[nodiscard]] DecoderTrainingCapture* training_capture() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neurale::pipeline
