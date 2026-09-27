/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <neurale/models/state_space.h>
#include <neurale/streaming/processor.h>

#include "decoder_training_capture.h"
#include "fitted_feature_contract.h"

namespace neurale::pipeline
{

/// What a non-finite observation row means to the decoder.
enum class KalmanMissingPolicy : std::uint8_t
{
    /// Any non-finite value in a selected observation is a fault.
    error,
    /// A row that is *entirely* ``nan`` runs the predict step alone. A
    /// partially missing row, or any infinity, is still a fault.
    predict,
};

/// Control-plane metadata and fixed geometry for the private Kalman adapter.
///
/// Everything a decoded frame depends on is here, already in native form: the
/// model matrices, the fitted scaling, the feature selection, and the names the
/// selection is expected to resolve to. Parsing an artifact and converting
/// Python objects happen wherever this struct is built, which is never on the
/// data plane -- after ``prepare`` the adapter reads none of it again.
struct KalmanDecoderAdapterConfig
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

    /// The full feature-set contract the decoder was fitted under. Checked
    /// against the input descriptor at prepare: a feature-set id alone is
    /// session-local and cannot prove the runtime features are the ones the
    /// model parameters and scaler statistics were estimated on.
    FittedFeatureContract fitted_feature_contract;

    FeatureScaling scaling{FeatureScaling::none};
    /// Per-selected-feature ``mean_`` (standard) or ``min_`` (min-max).
    std::vector<double> scaler_center;
    /// Per-selected-feature ``scale_``.
    std::vector<double> scaler_scale;

    /// Fitted linear-Gaussian parameters, including the initial state and
    /// covariance a reset restores.
    models::LinearGaussianModelState model;
    /// Added to the diagonal of the innovation covariance before each solve.
    double innovation_jitter{};
    KalmanMissingPolicy missing{KalmanMissingPolicy::error};
};

/// Internal adapter decoding one prepared feature stream with a Kalman filter.
///
/// The input is one fixed regular ``SignalKind::feature`` stream whose
/// descriptor must match the contract the decoder was fitted under; the output
/// is one fixed sampled stream carrying the decoded target state, one channel
/// per state dimension, on the input's clock and at the input's observation
/// rate. Every observation in an input block is decoded, and the whole block
/// leaves as at most one output frame that keeps the input's observation timing
/// and index, so decoding a stream in chunks gives what decoding it whole
/// gives.
///
/// The recursion is ``models::PreparedKalmanFilter`` -- the adapter holds no
/// Kalman arithmetic of its own. After ``prepare`` nothing here allocates,
/// throws, logs, calls Python, or touches a device: a numerical failure, a
/// partially missing observation, or an output that will not fit are all return
/// values on the normal fault path.
class KalmanDecoderAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit KalmanDecoderAdapter(KalmanDecoderAdapterConfig config);
    ~KalmanDecoderAdapter() override;

    KalmanDecoderAdapter(const KalmanDecoderAdapter&) = delete;
    KalmanDecoderAdapter& operator=(const KalmanDecoderAdapter&) = delete;
    KalmanDecoderAdapter(KalmanDecoderAdapter&&) = delete;
    KalmanDecoderAdapter& operator=(KalmanDecoderAdapter&&) = delete;

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
