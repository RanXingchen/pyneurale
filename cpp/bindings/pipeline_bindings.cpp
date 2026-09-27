/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/linear_model.h>
#include <neurale/models/state_space.h>
#include <neurale/streaming/linear_processor_chain.h>
#include <neurale/streaming/realtime_config.h>

#include "bad_channel_removal_adapter.h"
#include "feature_stack_adapter.h"
#include "kalman_decoder_adapter.h"
#include "linear_decoder_adapter.h"
#include "resampler_adapter.h"
#include "sos_filter_adapter.h"
#include "spatial_reference_adapter.h"
#include "spike_detector_adapter.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using neurale::streaming::Discontinuity;
using neurale::streaming::FrameBorrow;
using neurale::streaming::FrameEmitter;
using neurale::streaming::NativeFrameProcessor;
using neurale::streaming::PreparedProcessorContract;
using neurale::streaming::ProcessorPrepareContext;
using neurale::streaming::StreamStatus;

/// Own the private algorithm adapters and expose only their generic processor
/// contract. The vector must outlive the chain because LinearProcessorChain
/// intentionally stores non-owning stage pointers.
class CompiledPipeline final : public NativeFrameProcessor
{
  public:
    explicit CompiledPipeline(std::vector<std::unique_ptr<NativeFrameProcessor>> stages)
        : stages_(std::move(stages))
    {
        if (stages_.empty())
        {
            throw std::invalid_argument("a compiled pipeline requires at least one stage");
        }
        std::vector<NativeFrameProcessor*> stage_views;
        stage_views.reserve(stages_.size());
        for (const auto& stage : stages_)
        {
            stage_views.push_back(stage.get());
        }
        chain_ = std::make_unique<neurale::streaming::LinearProcessorChain>(stage_views);
    }

    [[nodiscard]] PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return chain_->prepare(context);
    }

    [[nodiscard]] StreamStatus process(FrameBorrow& frame, FrameEmitter& output) noexcept override
    {
        return chain_->process(frame, output);
    }

    [[nodiscard]] StreamStatus handle_discontinuity(const Discontinuity& value) noexcept override
    {
        return chain_->handle_discontinuity(value);
    }

    [[nodiscard]] StreamStatus flush(FrameEmitter& output) noexcept override
    {
        return chain_->flush(output);
    }

    [[nodiscard]] StreamStatus reset() noexcept override
    {
        return chain_->reset();
    }

    [[nodiscard]] std::size_t stage_count() const noexcept
    {
        return stages_.size();
    }

    [[nodiscard]] neurale::streaming::StreamSchema
    output_schema(const neurale::streaming::StreamSchema& input,
                  const neurale::streaming::RealtimeConfig& config)
    {
        return prepare(ProcessorPrepareContext{
                           .input_schema = input,
                           .max_process_outputs = config.max_process_outputs,
                           .max_flush_outputs = config.max_flush_outputs,
                           .available_frame_pool_leases = config.pool_capacity.processor_owned,
                       })
            .output_schema;
    }

    void enable_training_capture(std::size_t capacity)
    {
        neurale::pipeline::LinearDecoderAdapter* linear_decoder = nullptr;
        neurale::pipeline::KalmanDecoderAdapter* kalman_decoder = nullptr;
        std::size_t decoders = 0;
        for (const auto& stage : stages_)
        {
            if (auto* linear = dynamic_cast<neurale::pipeline::LinearDecoderAdapter*>(stage.get()))
            {
                linear_decoder = linear;
                ++decoders;
            }
            if (auto* kalman = dynamic_cast<neurale::pipeline::KalmanDecoderAdapter*>(stage.get()))
            {
                kalman_decoder = kalman;
                ++decoders;
            }
        }
        if (decoders != 1)
            throw std::invalid_argument("training capture requires one decoder stage");
        if (linear_decoder != nullptr)
            linear_decoder->enable_training_capture(capacity);
        else
            kalman_decoder->enable_training_capture(capacity);
    }

    [[nodiscard]] neurale::pipeline::DecoderTrainingCapture* training_capture() noexcept
    {
        for (const auto& stage : stages_)
        {
            if (auto* linear = dynamic_cast<neurale::pipeline::LinearDecoderAdapter*>(stage.get()))
                return linear->training_capture();
            if (auto* kalman = dynamic_cast<neurale::pipeline::KalmanDecoderAdapter*>(stage.get()))
                return kalman->training_capture();
        }
        return nullptr;
    }

  private:
    std::vector<std::unique_ptr<NativeFrameProcessor>> stages_;
    std::unique_ptr<neurale::streaming::LinearProcessorChain> chain_;
};

class PipelineBuilder final
{
  public:
    void add_filter(std::string type, std::vector<double> cutoff_hz, std::size_t order,
                    std::string kind, double passband_ripple_db, double stopband_attenuation_db)
    {
        add<neurale::pipeline::SosFilterAdapter>(neurale::pipeline::FilterDesign{
            std::move(type), std::move(cutoff_hz), order, std::move(kind), passband_ripple_db,
            stopband_attenuation_db});
    }

    void add_line_noise_filter(double frequency_hz, double bandwidth_hz, std::size_t harmonics,
                               std::size_t order)
    {
        add<neurale::pipeline::SosFilterAdapter>(
            neurale::pipeline::LineNoiseFilterDesign{frequency_hz, bandwidth_hz, harmonics, order});
    }

    void add_resampler(neurale::streaming::SchemaId output_schema_id, std::size_t up,
                       std::size_t down)
    {
        add<neurale::pipeline::ResamplerAdapter>(output_schema_id, up, down);
    }

    void add_spatial_reference(const std::vector<std::size_t>& channels,
                               neurale::pipeline::SpatialReferenceStatistic statistic)
    {
        add<neurale::pipeline::SpatialReferenceAdapter>(channels, statistic);
    }

    void add_bad_channel_removal(const std::vector<std::size_t>& indices,
                                 const std::vector<std::string>& names,
                                 std::optional<double> min_impedance_ohm,
                                 std::optional<double> max_impedance_ohm)
    {
        add<neurale::pipeline::BadChannelRemovalAdapter>(indices, names, min_impedance_ohm,
                                                         max_impedance_ohm);
    }

    void add_feature(neurale::pipeline::FeatureStackAdapterConfig config)
    {
        add<neurale::pipeline::FeatureStackAdapter>(std::move(config));
    }

    void add_spike_detector(neurale::pipeline::SpikeDetectorAdapterConfig config)
    {
        add<neurale::pipeline::SpikeDetectorAdapter>(std::move(config));
    }

    void add_linear_decoder(neurale::pipeline::LinearDecoderAdapterConfig config)
    {
        add<neurale::pipeline::LinearDecoderAdapter>(std::move(config));
    }

    void add_kalman_decoder(neurale::pipeline::KalmanDecoderAdapterConfig config)
    {
        add<neurale::pipeline::KalmanDecoderAdapter>(std::move(config));
    }

    [[nodiscard]] std::unique_ptr<CompiledPipeline> build()
    {
        require_open();
        if (stages_.empty())
        {
            throw std::invalid_argument("a pipeline plan requires at least one stage");
        }
        built_ = true;
        return std::make_unique<CompiledPipeline>(std::move(stages_));
    }

  private:
    template <typename Stage, typename... Args> void add(Args&&... args)
    {
        require_open();
        stages_.push_back(std::make_unique<Stage>(std::forward<Args>(args)...));
    }

    void require_open() const
    {
        if (built_)
        {
            throw std::logic_error("a pipeline builder cannot be reused after build()");
        }
    }

    std::vector<std::unique_ptr<NativeFrameProcessor>> stages_;
    bool built_{};
};

template <typename Config> void bind_feature_common(py::class_<Config>& value)
{
    value.def(py::init<>())
        .def_readwrite("output_schema_id", &Config::output_schema_id)
        .def_readwrite("output_signal_id", &Config::output_signal_id)
        .def_readwrite("feature_set_id", &Config::feature_set_id)
        .def_readwrite("source_stream", &Config::source_stream)
        .def_readwrite("algorithm_version", &Config::algorithm_version)
        .def_readwrite("window_samples", &Config::window_samples)
        .def_readwrite("shift_samples", &Config::shift_samples);
}

void bind_fitted_contract(py::module_& module)
{
    py::enum_<neurale::pipeline::FeatureScaling>(module, "_FeatureScaling")
        .value("NONE", neurale::pipeline::FeatureScaling::none)
        .value("STANDARD", neurale::pipeline::FeatureScaling::standard)
        .value("MINMAX", neurale::pipeline::FeatureScaling::minmax);
    py::enum_<neurale::pipeline::KalmanMissingPolicy>(module, "_KalmanMissingPolicy")
        .value("ERROR", neurale::pipeline::KalmanMissingPolicy::error)
        .value("PREDICT", neurale::pipeline::KalmanMissingPolicy::predict);

    py::class_<neurale::pipeline::FittedFeatureContract>(module, "_FittedFeatureContract")
        .def(py::init<>())
        .def_readwrite("feature_names", &neurale::pipeline::FittedFeatureContract::feature_names)
        .def_readwrite("feature_unit_symbols",
                       &neurale::pipeline::FittedFeatureContract::feature_unit_symbols)
        .def_readwrite("observation_rate",
                       &neurale::pipeline::FittedFeatureContract::observation_rate)
        .def_readwrite("window_length_ns",
                       &neurale::pipeline::FittedFeatureContract::window_length_ns)
        .def_readwrite("shift_ns", &neurale::pipeline::FittedFeatureContract::shift_ns)
        .def_readwrite("algorithm_name", &neurale::pipeline::FittedFeatureContract::algorithm_name)
        .def_readwrite("algorithm_version",
                       &neurale::pipeline::FittedFeatureContract::algorithm_version)
        .def_readwrite("source_stream", &neurale::pipeline::FittedFeatureContract::source_stream)
        .def_readwrite("timestamp_reference",
                       &neurale::pipeline::FittedFeatureContract::timestamp_reference);

    py::class_<neurale::models::LinearModelState>(module, "_LinearModelState")
        .def(py::init<>())
        .def_readwrite("n_features", &neurale::models::LinearModelState::n_features)
        .def_readwrite("n_outputs", &neurale::models::LinearModelState::n_outputs)
        .def_readwrite("coef", &neurale::models::LinearModelState::coef)
        .def_readwrite("intercept", &neurale::models::LinearModelState::intercept);

    py::class_<neurale::models::LinearGaussianModelState>(module, "_LinearGaussianModelState")
        .def(py::init<>())
        .def_readwrite("state_dim", &neurale::models::LinearGaussianModelState::state_dim)
        .def_readwrite("observation_dim",
                       &neurale::models::LinearGaussianModelState::observation_dim)
        .def_readwrite("transition", &neurale::models::LinearGaussianModelState::transition)
        .def_readwrite("transition_offset",
                       &neurale::models::LinearGaussianModelState::transition_offset)
        .def_readwrite("observation", &neurale::models::LinearGaussianModelState::observation)
        .def_readwrite("observation_offset",
                       &neurale::models::LinearGaussianModelState::observation_offset)
        .def_readwrite("process_covariance",
                       &neurale::models::LinearGaussianModelState::process_covariance)
        .def_readwrite("observation_covariance",
                       &neurale::models::LinearGaussianModelState::observation_covariance)
        .def_readwrite("initial_state", &neurale::models::LinearGaussianModelState::initial_state)
        .def_readwrite("initial_covariance",
                       &neurale::models::LinearGaussianModelState::initial_covariance);
}

void bind_configs(py::module_& module)
{
    py::enum_<neurale::signal::MultitaperWeighting>(module, "_MultitaperWeighting")
        .value("UNITY", neurale::signal::MultitaperWeighting::unity)
        .value("EIGEN", neurale::signal::MultitaperWeighting::eigen)
        .value("ADAPTIVE", neurale::signal::MultitaperWeighting::adaptive);
    py::enum_<neurale::sorting::DetectionPolarity>(module, "_DetectionPolarity")
        .value("NEGATIVE", neurale::sorting::DetectionPolarity::Negative)
        .value("POSITIVE", neurale::sorting::DetectionPolarity::Positive)
        .value("BOTH", neurale::sorting::DetectionPolarity::Both);
    py::enum_<neurale::sorting::BoundaryBehavior>(module, "_BoundaryBehavior")
        .value("DROP", neurale::sorting::BoundaryBehavior::Drop)
        .value("RAISE", neurale::sorting::BoundaryBehavior::Raise);
    py::enum_<neurale::sorting::SpikeBlockOverflowPolicy>(module, "_SpikeOverflowPolicy")
        .value("FAULT", neurale::sorting::SpikeBlockOverflowPolicy::fault)
        .value("DROP_NEWEST", neurale::sorting::SpikeBlockOverflowPolicy::drop_newest);
    py::enum_<neurale::pipeline::SpatialReferenceStatistic>(module, "_SpatialReferenceStatistic")
        .value("MEAN", neurale::pipeline::SpatialReferenceStatistic::mean)
        .value("MEDIAN", neurale::pipeline::SpatialReferenceStatistic::median);

    py::enum_<neurale::features::Detrend>(module, "_FeatureDetrend")
        .value("NONE", neurale::features::Detrend::none)
        .value("MEAN", neurale::features::Detrend::mean)
        .value("LINEAR", neurale::features::Detrend::linear);
    py::enum_<neurale::signal::SpectralBackend>(module, "_FeatureSpectralBackend")
        .value("AUTO", neurale::signal::SpectralBackend::automatic)
        .value("BUILTIN", neurale::signal::SpectralBackend::builtin);
    py::class_<neurale::pipeline::MultitaperBand>(module, "_MultitaperBand")
        .def(py::init<>())
        .def_readwrite("name", &neurale::pipeline::MultitaperBand::name)
        .def_readwrite("low_hz", &neurale::pipeline::MultitaperBand::low_hz)
        .def_readwrite("high_hz", &neurale::pipeline::MultitaperBand::high_hz);
    py::class_<neurale::pipeline::MultitaperBandpowerBranchConfig>(
        module, "_MultitaperBandpowerBranchConfig")
        .def(py::init<>())
        .def_readwrite("bands", &neurale::pipeline::MultitaperBandpowerBranchConfig::bands)
        .def_readwrite("time_bandwidth",
                       &neurale::pipeline::MultitaperBandpowerBranchConfig::time_bandwidth)
        .def_readwrite("n_tapers", &neurale::pipeline::MultitaperBandpowerBranchConfig::n_tapers)
        .def_readwrite("detrend", &neurale::pipeline::MultitaperBandpowerBranchConfig::detrend)
        .def_readwrite("backend", &neurale::pipeline::MultitaperBandpowerBranchConfig::backend)
        .def_readwrite("weighting", &neurale::pipeline::MultitaperBandpowerBranchConfig::weighting);
    py::class_<neurale::pipeline::HilbertEnvelopeBranchConfig>(module,
                                                               "_HilbertEnvelopeBranchConfig")
        .def(py::init<>())
        .def_readwrite("bands", &neurale::pipeline::HilbertEnvelopeBranchConfig::bands)
        .def_readwrite("filter_order",
                       &neurale::pipeline::HilbertEnvelopeBranchConfig::filter_order)
        .def_readwrite("filter_kind", &neurale::pipeline::HilbertEnvelopeBranchConfig::filter_kind)
        .def_readwrite("passband_ripple_db",
                       &neurale::pipeline::HilbertEnvelopeBranchConfig::passband_ripple_db)
        .def_readwrite("stopband_attenuation_db",
                       &neurale::pipeline::HilbertEnvelopeBranchConfig::stopband_attenuation_db);
    py::class_<neurale::pipeline::LmpBranchConfig>(module, "_LmpBranchConfig")
        .def(py::init<>())
        .def_readwrite("cutoff_hz", &neurale::pipeline::LmpBranchConfig::cutoff_hz)
        .def_readwrite("filter_order", &neurale::pipeline::LmpBranchConfig::filter_order)
        .def_readwrite("filter_kind", &neurale::pipeline::LmpBranchConfig::filter_kind)
        .def_readwrite("passband_ripple_db",
                       &neurale::pipeline::LmpBranchConfig::passband_ripple_db)
        .def_readwrite("stopband_attenuation_db",
                       &neurale::pipeline::LmpBranchConfig::stopband_attenuation_db);
    py::class_<neurale::pipeline::FeatureStackAdapterConfig>(module, "_FeatureStackConfig")
        .def(py::init<>())
        .def_readwrite("algorithm_version",
                       &neurale::pipeline::FeatureStackAdapterConfig::algorithm_version)
        .def_readwrite("window_ns", &neurale::pipeline::FeatureStackAdapterConfig::window_ns)
        .def_readwrite("update_interval_ns",
                       &neurale::pipeline::FeatureStackAdapterConfig::update_interval_ns)
        .def_readwrite("branches", &neurale::pipeline::FeatureStackAdapterConfig::branches);

    py::class_<neurale::pipeline::SpikeDetectorAdapterConfig>(module, "_SpikeDetectorConfig")
        .def(py::init<>())
        .def_readwrite("output_schema_id",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::output_schema_id)
        .def_readwrite("output_signal_id",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::output_signal_id)
        .def_readwrite("block_capacity",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::block_capacity)
        .def_readwrite("refractory_samples",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::refractory_samples)
        .def_readwrite("alignment_search_radius",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::alignment_search_radius)
        .def_readwrite("pre_samples", &neurale::pipeline::SpikeDetectorAdapterConfig::pre_samples)
        .def_readwrite("post_samples", &neurale::pipeline::SpikeDetectorAdapterConfig::post_samples)
        .def_readwrite("polarity", &neurale::pipeline::SpikeDetectorAdapterConfig::polarity)
        .def_readwrite("boundary_behavior",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::boundary_behavior)
        .def_readwrite("overflow_policy",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::overflow_policy)
        .def_readwrite("channel_centers",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::channel_centers)
        .def_readwrite("channel_thresholds",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::channel_thresholds)
        .def_readwrite("electrode_groups",
                       &neurale::pipeline::SpikeDetectorAdapterConfig::electrode_groups);

    py::class_<neurale::pipeline::LinearDecoderAdapterConfig>(module, "_LinearDecoderConfig")
        .def(py::init<>())
        .def_readwrite("output_schema_id",
                       &neurale::pipeline::LinearDecoderAdapterConfig::output_schema_id)
        .def_readwrite("output_signal_id",
                       &neurale::pipeline::LinearDecoderAdapterConfig::output_signal_id)
        .def_readwrite("output_channel_set_id",
                       &neurale::pipeline::LinearDecoderAdapterConfig::output_channel_set_id)
        .def_readwrite("output_physical_unit",
                       &neurale::pipeline::LinearDecoderAdapterConfig::output_physical_unit)
        .def_readwrite("feature_set_id",
                       &neurale::pipeline::LinearDecoderAdapterConfig::feature_set_id)
        .def_readwrite("selection", &neurale::pipeline::LinearDecoderAdapterConfig::selection)
        .def_readwrite("selected_feature_names",
                       &neurale::pipeline::LinearDecoderAdapterConfig::selected_feature_names)
        .def_readwrite("fitted_feature_contract",
                       &neurale::pipeline::LinearDecoderAdapterConfig::fitted_feature_contract)
        .def_readwrite("scaling", &neurale::pipeline::LinearDecoderAdapterConfig::scaling)
        .def_readwrite("scaler_center",
                       &neurale::pipeline::LinearDecoderAdapterConfig::scaler_center)
        .def_readwrite("scaler_scale", &neurale::pipeline::LinearDecoderAdapterConfig::scaler_scale)
        .def_readwrite("model", &neurale::pipeline::LinearDecoderAdapterConfig::model)
        .def_readwrite("classes", &neurale::pipeline::LinearDecoderAdapterConfig::classes);

    py::class_<neurale::pipeline::KalmanDecoderAdapterConfig>(module, "_KalmanDecoderConfig")
        .def(py::init<>())
        .def_readwrite("output_schema_id",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::output_schema_id)
        .def_readwrite("output_signal_id",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::output_signal_id)
        .def_readwrite("output_channel_set_id",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::output_channel_set_id)
        .def_readwrite("output_physical_unit",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::output_physical_unit)
        .def_readwrite("feature_set_id",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::feature_set_id)
        .def_readwrite("selection", &neurale::pipeline::KalmanDecoderAdapterConfig::selection)
        .def_readwrite("selected_feature_names",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::selected_feature_names)
        .def_readwrite("fitted_feature_contract",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::fitted_feature_contract)
        .def_readwrite("scaling", &neurale::pipeline::KalmanDecoderAdapterConfig::scaling)
        .def_readwrite("scaler_center",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::scaler_center)
        .def_readwrite("scaler_scale", &neurale::pipeline::KalmanDecoderAdapterConfig::scaler_scale)
        .def_readwrite("model", &neurale::pipeline::KalmanDecoderAdapterConfig::model)
        .def_readwrite("innovation_jitter",
                       &neurale::pipeline::KalmanDecoderAdapterConfig::innovation_jitter)
        .def_readwrite("missing", &neurale::pipeline::KalmanDecoderAdapterConfig::missing);
}

} // namespace

void bind_pipeline_module(py::module_& module)
{
    auto pipeline = module.def_submodule(
        "pipeline", "Private constructors for compiled built-in realtime pipelines.");
    bind_fitted_contract(pipeline);
    bind_configs(pipeline);

    py::class_<CompiledPipeline, NativeFrameProcessor>(pipeline, "_CompiledPipeline",
                                                       py::is_final())
        .def_property_readonly("stage_count", &CompiledPipeline::stage_count)
        .def("output_schema", &CompiledPipeline::output_schema, py::arg("input_schema"),
             py::arg("runtime_config"))
        .def("enable_training_capture", &CompiledPipeline::enable_training_capture,
             py::arg("capacity"))
        .def("pop_training_observation",
             [](CompiledPipeline& self) -> py::object
             {
                 auto* capture = self.training_capture();
                 if (capture == nullptr)
                     return py::none();
                 neurale::pipeline::DecoderTrainingObservation observation{};
                 if (capture->try_pop(observation) != StreamStatus::ok)
                     return py::none();
                 py::tuple result(2);
                 py::tuple values(observation.n_features);
                 for (std::uint32_t i = 0; i < observation.n_features; ++i)
                     values[i] = observation.values[i];
                 result[0] = observation.sample_idx;
                 result[1] = std::move(values);
                 return result;
             })
        .def_property_readonly("training_capture_drops",
                               [](CompiledPipeline& self)
                               {
                                   auto* capture = self.training_capture();
                                   return capture == nullptr ? std::uint64_t{} : capture->dropped();
                               });

    py::class_<PipelineBuilder>(pipeline, "_PipelineBuilder", py::is_final())
        .def(py::init<>())
        .def("add_filter", &PipelineBuilder::add_filter)
        .def("add_line_noise_filter", &PipelineBuilder::add_line_noise_filter)
        .def("add_resampler", &PipelineBuilder::add_resampler)
        .def("add_spatial_reference", &PipelineBuilder::add_spatial_reference)
        .def("add_bad_channel_removal", &PipelineBuilder::add_bad_channel_removal)
        .def("add_feature", &PipelineBuilder::add_feature)
        .def("add_spike_detector", &PipelineBuilder::add_spike_detector)
        .def("add_linear_decoder", &PipelineBuilder::add_linear_decoder)
        .def("add_kalman_decoder", &PipelineBuilder::add_kalman_decoder)
        .def("build", &PipelineBuilder::build);
}
