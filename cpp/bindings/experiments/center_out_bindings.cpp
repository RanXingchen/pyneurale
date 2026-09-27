// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "adaptive_center_out_actuator.h"
#include "binding_helpers.h"
#include "center_out_controller.h"
#include "center_out_session_context.h"
#include "center_out_trace_writer.h"
#include "experiment_attachment.h"
#include "session.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <neurale/experiments/center_out.h>
#include <neurale/experiments/center_out_guidance.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/schema.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::center_out;

} // namespace

static void bind_center_out_guidance(py::module_& center_out)
{
    py::enum_<CenterOutGuidancePhase>(center_out, "CenterOutGuidancePhase")
        .value("IDLE", CenterOutGuidancePhase::idle)
        .value("SPEEDING_UP", CenterOutGuidancePhase::speeding_up)
        .value("SLOWING_DOWN", CenterOutGuidancePhase::slowing_down)
        .value("AT_TARGET", CenterOutGuidancePhase::at_target);

    center_out.def("center_out_guidance_phase_declared", &center_out_guidance_phase_declared,
                   py::arg("phase"));

    py::class_<WorkspaceVelocity>(center_out, "WorkspaceVelocity")
        .def(py::init([](double x, double y) { return WorkspaceVelocity{x, y}; }),
             py::arg("x") = 0.0, py::arg("y") = 0.0)
        .def_readonly("x", &WorkspaceVelocity::x)
        .def_readonly("y", &WorkspaceVelocity::y);

    // The native record includes the geometry unit required by persistence and
    // the controller.  Python's public config is unitless and resolves that
    // field from CenterOut2DConfig at composition time.
    py::class_<CenterOutGuidanceConfig>(center_out, "_NativeCenterOutGuidanceConfig")
        .def(py::init(
                 [](GeometryUnit geometry_unit, double max_speed, double acceleration,
                    double deceleration, double precision)
                 {
                     return CenterOutGuidanceConfig{geometry_unit, max_speed, acceleration,
                                                    deceleration, precision};
                 }),
             py::kw_only(), py::arg("geometry_unit") = GeometryUnit::unspecified,
             py::arg("max_speed") = 0.0, py::arg("acceleration") = 0.0,
             py::arg("deceleration") = 0.0, py::arg("precision") = 0.0)
        .def_readonly("geometry_unit", &CenterOutGuidanceConfig::geometry_unit)
        .def_readonly("max_speed", &CenterOutGuidanceConfig::max_speed)
        .def_readonly("acceleration", &CenterOutGuidanceConfig::acceleration)
        .def_readonly("deceleration", &CenterOutGuidanceConfig::deceleration)
        .def_readonly("precision", &CenterOutGuidanceConfig::precision);

    // A complete constructor, unlike CenterOutSnapshot: a caller genuinely
    // resumes a run from a state it is holding, and evaluate_guidance() takes one
    // as an argument, so a state nobody can build would make the pure function
    // unreachable from Python.
    py::class_<CenterOutGuidanceState>(center_out, "CenterOutGuidanceState")
        .def(
            py::init([](TargetId target, const WorkspaceVelocity& vel, CenterOutGuidancePhase phase)
                     { return CenterOutGuidanceState{target, vel, phase}; }),
            py::arg("target") = kUnsetTargetId, py::arg("vel") = WorkspaceVelocity{},
            py::arg("phase") = CenterOutGuidancePhase::idle)
        .def_readonly("target", &CenterOutGuidanceState::target)
        .def_readonly("vel", &CenterOutGuidanceState::vel)
        .def_readonly("phase", &CenterOutGuidanceState::phase);

    // No public constructor: only an update produces a sample.
    py::class_<CenterOutGuidanceSample>(center_out, "CenterOutGuidanceSample")
        .def_readonly("vel", &CenterOutGuidanceSample::vel)
        .def_readonly("state", &CenterOutGuidanceSample::state)
        .def_readonly("distance", &CenterOutGuidanceSample::distance)
        .def_readonly("speed_towards_target", &CenterOutGuidanceSample::speed_towards_target)
        .def_readonly("retargeted", &CenterOutGuidanceSample::retargeted);

    center_out.def(
        "validate", [](const CenterOutGuidanceConfig& value) { return validate(value); },
        py::arg("value"));
    center_out.def(
        "validate", [](const CenterOutGuidanceState& value) { return validate(value); },
        py::arg("value"));
    center_out.def(
        "validate_against",
        [](const CenterOutGuidanceConfig& guidance, const CenterOut2DConfig& config)
        { return validate_against(guidance, config); }, py::arg("guidance"), py::arg("config"));

    center_out.def(
        "evaluate_guidance",
        [](const CenterOutGuidanceConfig& config, const CenterOutGuidanceState& previous,
           const TargetPlacement& target, const WorkspacePoint& pos, DurationNs dt_ns)
        {
            CenterOutGuidanceSample sample{};
            const ContractStatus status =
                evaluate_guidance(config, previous, target, pos, dt_ns, sample);
            return std::pair{status, sample};
        },
        py::arg("config"), py::arg("previous"), py::arg("target"), py::arg("pos"),
        py::arg("dt_ns"));

    py::class_<CenterOutGuidance>(center_out, "_NativeCenterOutGuidance")
        .def(py::init<>())
        .def(
            "configure", [](CenterOutGuidance& guidance, const CenterOutGuidanceConfig& config)
            { return guidance.configure(config); }, py::arg("config"))
        .def("reset", &CenterOutGuidance::reset)
        .def(
            "update",
            [](CenterOutGuidance& guidance, const TargetPlacement& target,
               const WorkspacePoint& pos, DurationNs dt_ns)
            {
                CenterOutGuidanceSample sample{};
                const ContractStatus status = guidance.update(target, pos, dt_ns, sample);
                return std::pair{status, sample};
            },
            py::arg("target"), py::arg("pos"), py::arg("dt_ns"))
        // Methods rather than properties, so each call hands back a copy. A
        // property would alias the generator's own field, and a value read
        // before an update would change under the reader afterwards.
        .def("configuration", &CenterOutGuidance::configuration)
        .def("state", &CenterOutGuidance::state)
        .def_property_readonly("configured", &CenterOutGuidance::configured);
}

namespace neurale::bindings
{
namespace
{

class PyCenterOutSession final
{
  public:
    [[nodiscard]] streaming::StreamStatus
    prepare(const streaming::StreamSchema& decoded_schema,
            const ::neurale::execution::CenterOutControllerConfig& controller_config,
            streaming::HostTimeNs host_epoch_ns,
            ::neurale::execution::ExperimentSessionConfig session_config)
    {
        if (session_ != nullptr)
            return streaming::StreamStatus::invalid_state;
        const auto status = controller_.prepare(decoded_schema, controller_config);
        if (status != streaming::StreamStatus::ok)
            return status;
        writer_ = std::make_unique<::neurale::execution::CenterOutTraceWriter>(controller_,
                                                                               host_epoch_ns);
        writer_->declare_schemas(0, 0);
        session_ =
            std::make_unique<::neurale::execution::ExperimentSession>(*writer_, session_config);
        return streaming::StreamStatus::ok;
    }

    [[nodiscard]] streaming::NativeFrameConsumer& actuator() noexcept
    {
        return controller_;
    }

    [[nodiscard]] streaming::StreamStatus
    attach_runtime(streaming::NativeStreamRunner& runner) noexcept
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->attach_runtime(runner);
    }

    [[nodiscard]] streaming::StreamStatus start(::neurale::experiments::ExperimentTimeNs time_ns)
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->start(time_ns);
    }

    [[nodiscard]] streaming::StreamStatus stop(::neurale::experiments::ExperimentTimeNs time_ns,
                                               std::string_view reason) noexcept
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->stop(time_ns, reason);
    }

    [[nodiscard]] streaming::StreamStatus abort(::neurale::experiments::ExperimentTimeNs time_ns,
                                                std::string_view reason) noexcept
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->abort(time_ns, reason);
    }

    [[nodiscard]] streaming::StreamStatus
    emergency_stop(::neurale::experiments::ExperimentTimeNs time_ns,
                   std::string_view reason) noexcept
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->emergency_stop(time_ns, reason);
    }

    [[nodiscard]] recording::RecorderStatusCode close()
    {
        if (session_ == nullptr)
            return recording::RecorderStatusCode::wrong_state;
        const auto status = session_->close();
        controller_.close();
        return status;
    }

    [[nodiscard]] py::dict outcome() const
    {
        py::dict value;
        if (session_ == nullptr)
            return value;
        const auto outcome = session_->outcome();
        value["state"] = static_cast<std::uint8_t>(outcome.state);
        value["recorded"] = outcome.recorded;
        value["experiment_trace_complete"] = outcome.experiment_trace_complete;
        value["records_encoded"] = outcome.records_encoded;
        value["records_recorded"] = outcome.records_recorded;
        value["records_refused"] = outcome.records_refused;
        value["trace_losses"] = outcome.trace_losses;
        value["producer_trace_drops"] = outcome.producer_trace_drops;
        value["loss_faulted"] = outcome.loss_faulted;
        value["abnormal_conditions"] = outcome.abnormal_conditions;
        value["abnormal_trials_affected"] = outcome.abnormal_trials_affected;
        value["abnormal_session_aborted"] = outcome.abnormal_session_aborted;
        value["primary_abnormal"] = outcome.primary_abnormal;
        value["has_runtime_fault"] = outcome.has_runtime_fault;
        value["terminal_abort"] = outcome.terminal_abort;
        value["runtime_status"] = outcome.runtime_status;
        value["recorder_status"] = outcome.recorder_status;
        return value;
    }

    [[nodiscard]] ::neurale::experiments::center_out::CenterOutSnapshot snapshot() const noexcept
    {
        return controller_.snapshot();
    }

    [[nodiscard]] ::neurale::experiments::center_out::WorkspacePoint position() const noexcept
    {
        return controller_.position();
    }

    [[nodiscard]] py::object pop_training_label()
    {
        ::neurale::execution::CenterOutTrainingLabel label{};
        if (controller_.try_pop_training_label(label) != streaming::StreamStatus::ok)
            return py::none();
        py::dict value;
        value["sample_idx"] = label.sample_idx;
        value["time_ns"] = label.time_ns;
        value["trial"] = label.trial;
        value["target_position"] = py::make_tuple(label.target_position.x, label.target_position.y);
        value["cursor_position"] = py::make_tuple(label.cursor_position.x, label.cursor_position.y);
        value["cursor_velocity"] = py::make_tuple(label.cursor_velocity.x, label.cursor_velocity.y);
        value["guidance_velocity"] =
            py::make_tuple(label.guidance.values[0], label.guidance.values[1]);
        value["trial_stop"] = label.trial_stop;
        return value;
    }

    [[nodiscard]] std::uint64_t training_label_drops() const noexcept
    {
        return controller_.dropped_training_label_count();
    }

  private:
    ::neurale::execution::CenterOutController controller_{};
    std::unique_ptr<::neurale::execution::CenterOutTraceWriter> writer_{};
    std::unique_ptr<::neurale::execution::ExperimentSession> session_{};
};

class PyAdaptiveCenterOutSession final
{
  public:
    [[nodiscard]] streaming::StreamStatus
    prepare(const streaming::StreamSchema& feature_schema, py::object initial_decoder,
            std::uint64_t initial_version,
            const ::neurale::execution::CenterOutControllerConfig& controller_config,
            ::neurale::execution::ExperimentSessionConfig session_config)
    {
        if (session_ != nullptr)
            return streaming::StreamStatus::invalid_state;
        auto& decoder = initial_decoder.cast<streaming::NativeFrameProcessor&>();
        const auto status =
            actuator_.prepare(feature_schema, decoder, initial_version, controller_config);
        if (status != streaming::StreamStatus::ok)
            return status;
        initial_decoder_ = std::move(initial_decoder);
        writer_ =
            std::make_unique<::neurale::execution::CenterOutTraceWriter>(actuator_.controller(), 0);
        writer_->declare_schemas(0, 0);
        presentation_context_ = {writer_.get(), &actuator_.controller()};
        session_ =
            std::make_unique<::neurale::execution::ExperimentSession>(*writer_, session_config);
        return streaming::StreamStatus::ok;
    }

    [[nodiscard]] bool set_host_epoch(streaming::HostTimeNs host_epoch_ns) noexcept
    {
        return writer_ != nullptr && writer_->set_host_epoch(host_epoch_ns);
    }

    [[nodiscard]] streaming::StreamStatus prepare_candidate(py::object decoder,
                                                            std::uint64_t version)
    {
        auto& processor = decoder.cast<streaming::NativeFrameProcessor&>();
        auto status = actuator_.prepare_candidate(processor, version);
        if (status != streaming::StreamStatus::ok)
            return status;
        auto candidate = std::make_unique<::neurale::execution::PreparedDecoderCandidate>(
            ::neurale::execution::PreparedDecoderCandidate{&processor, version});
        status = actuator_.publish_candidate(*candidate);
        if (status != streaming::StreamStatus::ok)
            return status;
        candidate_decoders_.push_back(std::move(decoder));
        candidates_.push_back(std::move(candidate));
        return streaming::StreamStatus::ok;
    }

    [[nodiscard]] streaming::StreamStatus prepare_final_candidate(py::object decoder,
                                                                  std::uint64_t version)
    {
        if (runtime_ == nullptr || runtime_->state() != streaming::RuntimeState::stopped)
            return streaming::StreamStatus::invalid_state;
        const auto status = runtime_->join();
        if (status != streaming::StreamStatus::ok)
            return status;
        actuator_.discard_pending_candidate();
        return prepare_candidate(std::move(decoder), version);
    }

    [[nodiscard]] bool
    record_decoder_publication(std::uint64_t version, std::string_view plan_fingerprint,
                               std::uint64_t training_blocks, std::uint64_t training_trials,
                               ::neurale::experiments::TrialOrdinal completed_trials,
                               ::neurale::experiments::ExperimentTimeNs time_ns) noexcept
    {
        return writer_ != nullptr &&
               writer_->report_decoder_publication(version, plan_fingerprint, training_blocks,
                                                   training_trials, completed_trials, time_ns);
    }

    [[nodiscard]] streaming::NativeFrameConsumer& actuator() noexcept
    {
        return actuator_;
    }

    [[nodiscard]] streaming::StreamStatus
    attach_runtime(streaming::NativeStreamRunner& runner) noexcept
    {
        const auto status = session_ == nullptr ? streaming::StreamStatus::invalid_state
                                                : session_->attach_runtime(runner);
        if (status == streaming::StreamStatus::ok)
            runtime_ = &runner;
        return status;
    }

    [[nodiscard]] streaming::StreamStatus enable_recording(py::capsule capsule, py::object owner)
    {
        if (session_ == nullptr)
            return streaming::StreamStatus::invalid_state;
        auto* attachment = static_cast<recording::ExperimentRecorderAttachment*>(
            PyCapsule_GetPointer(capsule.ptr(), recording::kExperimentRecorderAttachmentCapsule));
        if (attachment == nullptr || attachment->recorder == nullptr ||
            attachment->plan == nullptr || attachment->spool == nullptr ||
            attachment->clock == nullptr || attachment->data_observer == nullptr)
            return streaming::StreamStatus::invalid_frame;
        const auto status = session_->enable_recording(
            {attachment->recorder, attachment->plan, attachment->spool, attachment->clock, true});
        if (status == streaming::StreamStatus::ok)
            recorder_owner_ = std::move(owner);
        return status;
    }

    [[nodiscard]] streaming::StreamStatus start(::neurale::experiments::ExperimentTimeNs time_ns)
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->start(time_ns);
    }

    [[nodiscard]] streaming::StreamStatus stop(::neurale::experiments::ExperimentTimeNs time_ns,
                                               std::string_view reason) noexcept
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->stop(time_ns, reason);
    }

    [[nodiscard]] streaming::StreamStatus abort(::neurale::experiments::ExperimentTimeNs time_ns,
                                                std::string_view reason) noexcept
    {
        return session_ == nullptr ? streaming::StreamStatus::invalid_state
                                   : session_->abort(time_ns, reason);
    }

    [[nodiscard]] recording::RecorderStatusCode close()
    {
        if (session_ == nullptr)
            return recording::RecorderStatusCode::wrong_state;
        const auto status = session_->close();
        actuator_.controller().close();
        return status;
    }

    [[nodiscard]] py::dict outcome() const
    {
        py::dict value;
        if (session_ == nullptr)
            return value;
        const auto outcome = session_->outcome();
        value["state"] = static_cast<std::uint8_t>(outcome.state);
        value["experiment_trace_complete"] = outcome.experiment_trace_complete;
        value["trace_losses"] = outcome.trace_losses;
        value["producer_trace_drops"] = outcome.producer_trace_drops;
        value["abnormal_conditions"] = outcome.abnormal_conditions;
        value["abnormal_trials_affected"] = outcome.abnormal_trials_affected;
        value["abnormal_session_aborted"] = outcome.abnormal_session_aborted;
        value["terminal_abort"] = outcome.terminal_abort;
        value["runtime_status"] = outcome.runtime_status;
        return value;
    }

    [[nodiscard]] py::object pop_training_observation()
    {
        ::neurale::execution::CenterOutTrainingLabel label{};
        if (actuator_.controller().try_pop_training_label(label) != streaming::StreamStatus::ok)
            return py::none();
        ::neurale::execution::AdaptiveFeatureObservation feature{};
        if (actuator_.try_pop_feature(feature) != streaming::StreamStatus::ok ||
            feature.sample_idx != label.sample_idx)
            throw std::runtime_error(
                "adaptive feature and Center-Out label captures are not aligned");
        py::tuple features(feature.values.size());
        for (std::size_t i = 0; i < feature.values.size(); ++i)
            features[i] = feature.values[i];
        py::dict value;
        value["sample_idx"] = feature.sample_idx;
        value["features"] = std::move(features);
        value["time_ns"] = label.time_ns;
        value["trial"] = label.trial;
        value["target_position"] = py::make_tuple(label.target_position.x, label.target_position.y);
        value["cursor_position"] = py::make_tuple(label.cursor_position.x, label.cursor_position.y);
        value["cursor_velocity"] = py::make_tuple(label.cursor_velocity.x, label.cursor_velocity.y);
        value["guidance_velocity"] =
            py::make_tuple(label.guidance.values[0], label.guidance.values[1]);
        value["trial_stop"] = label.trial_stop;
        return value;
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> training_capture_drops() const noexcept
    {
        return {actuator_.dropped_feature_count(),
                actuator_.controller().dropped_training_label_count()};
    }

    [[nodiscard]] py::object pop_presentation_state()
    {
        ::neurale::execution::CenterOutPresentationState state{};
        if (actuator_.controller().try_pop_presentation_state(state) != streaming::StreamStatus::ok)
            return py::none();
        py::dict value;
        value["snapshot"] = state.snapshot;
        value["cursor"] = state.cursor;
        value["source_ordinal"] = state.source_ordinal;
        value["time_ns"] = state.time_ns;
        return value;
    }

    [[nodiscard]] std::uint64_t presentation_state_drops() const noexcept
    {
        return actuator_.controller().dropped_presentation_state_count();
    }

    [[nodiscard]] py::capsule presentation_context()
    {
        if (presentation_context_.writer == nullptr || presentation_context_.controller == nullptr)
            throw std::runtime_error("adaptive Center-Out session is not prepared");
        return py::capsule(&presentation_context_,
                           ::neurale::bindings::experiments::kCenterOutSessionContextCapsule);
    }

    [[nodiscard]] ::neurale::experiments::center_out::CenterOutSnapshot snapshot() const noexcept
    {
        return actuator_.controller().snapshot();
    }

    [[nodiscard]] ::neurale::experiments::center_out::WorkspacePoint position() const noexcept
    {
        return actuator_.controller().position();
    }

    [[nodiscard]] py::dict get_intent() const
    {
        const auto context = actuator_.controller().intent_context();
        const auto& intent = context.intent;
        py::dict value;
        value["sequence"] = intent.sequence;
        value["intent_x"] = intent.intent_x;
        value["intent_y"] = intent.intent_y;
        value["context_ordinal"] = intent.context_ordinal;
        value["source_time_ns"] = intent.source_time_ns;
        value["source_frame_sequence"] = intent.source_frame_sequence;
        value["source_sample_index"] = intent.source_sample_index;
        value["trial_key"] = context.trial_key;
        value["target_id"] = context.target_id;
        value["valid"] = intent.valid;
        return value;
    }

    [[nodiscard]] py::capsule intent_source() const
    {
        using IntentSource = streaming::NeuralIntentSource;
        auto* owner = new std::shared_ptr<IntentSource>(actuator_.controller().intent_source());
        return py::capsule(owner, streaming::kNeuralIntentSourceCapsule,
                           [](PyObject* capsule)
                           {
                               auto* value =
                                   static_cast<std::shared_ptr<IntentSource>*>(PyCapsule_GetPointer(
                                       capsule, streaming::kNeuralIntentSourceCapsule));
                               delete value;
                           });
    }

    [[nodiscard]] std::uint64_t active_decoder_version() const noexcept
    {
        return actuator_.active_decoder_version();
    }

    [[nodiscard]] std::uint64_t completed_trials() const noexcept
    {
        return actuator_.controller().completed_trial_count();
    }

    [[nodiscard]] bool complete() const noexcept
    {
        return actuator_.controller().complete();
    }

  private:
    ::neurale::execution::AdaptiveCenterOutActuator actuator_{};
    streaming::NativeStreamRunner* runtime_{};
    py::object initial_decoder_{};
    std::vector<py::object> candidate_decoders_{};
    std::vector<std::unique_ptr<::neurale::execution::PreparedDecoderCandidate>> candidates_{};
    std::unique_ptr<::neurale::execution::CenterOutTraceWriter> writer_{};
    std::unique_ptr<::neurale::execution::ExperimentSession> session_{};
    ::neurale::bindings::experiments::CenterOutSessionContext presentation_context_{};
    py::object recorder_owner_{};
};

} // namespace

static void bind_center_out_session(py::module_& center_out)
{
    using ::neurale::execution::CenterOutAssistanceBlock;
    using ::neurale::execution::CenterOutController;
    using ::neurale::execution::CenterOutControllerConfig;
    using streaming::NativeFrameConsumer;
    using streaming::StreamSchema;

    py::enum_<::neurale::execution::TraceLossPolicy>(center_out, "_TraceLossPolicy")
        .value("FAULT", ::neurale::execution::TraceLossPolicy::fault)
        .value("MARK_INCOMPLETE", ::neurale::execution::TraceLossPolicy::mark_incomplete);

    py::class_<::neurale::execution::ExperimentSessionConfig>(
        center_out, "_ExperimentSessionConfig", py::is_final())
        .def(py::init<>())
        .def_readwrite("drain_budget", &::neurale::execution::ExperimentSessionConfig::drain_budget)
        .def_readwrite("bridge_poll_nanos",
                       &::neurale::execution::ExperimentSessionConfig::bridge_poll_nanos)
        .def_readwrite("control_clock_domain",
                       &::neurale::execution::ExperimentSessionConfig::control_clock_domain)
        .def_readwrite("max_control_body_bytes",
                       &::neurale::execution::ExperimentSessionConfig::max_control_body_bytes)
        .def_readwrite("trace_loss_policy",
                       &::neurale::execution::ExperimentSessionConfig::trace_loss_policy);

    py::class_<CenterOutAssistanceBlock>(center_out, "_CenterOutAssistanceBlock", py::is_final())
        .def(py::init<>())
        .def_readwrite("linear", &CenterOutAssistanceBlock::linear)
        .def_readwrite("trials", &CenterOutAssistanceBlock::trials);

    py::class_<CenterOutControllerConfig>(center_out, "_CenterOutControllerConfig", py::is_final())
        .def(py::init<>())
        .def_readwrite("decoded_signal_id", &CenterOutControllerConfig::decoded_signal_id)
        .def_readwrite("paradigm", &CenterOutControllerConfig::paradigm)
        .def_readwrite("task", &CenterOutControllerConfig::task)
        .def_readwrite("guidance", &CenterOutControllerConfig::guidance)
        .def_readwrite("guidance_resolver_version",
                       &CenterOutControllerConfig::guidance_resolver_version)
        .def_readwrite("velocity_space", &CenterOutControllerConfig::velocity_space)
        .def_readwrite("linear_assistance", &CenterOutControllerConfig::linear_assistance)
        .def_readwrite("assistance_blocks", &CenterOutControllerConfig::assistance_blocks)
        .def_readwrite("training_capture_capacity",
                       &CenterOutControllerConfig::training_capture_capacity)
        .def_readwrite("presentation_state_capacity",
                       &CenterOutControllerConfig::presentation_state_capacity)
        .def_readwrite("initial_position", &CenterOutControllerConfig::initial_position)
        .def_readwrite("cursor_min", &CenterOutControllerConfig::cursor_min)
        .def_readwrite("cursor_max", &CenterOutControllerConfig::cursor_max)
        .def_readwrite("trace_capacity", &CenterOutControllerConfig::trace_capacity)
        .def_readwrite("assistance_method", &CenterOutControllerConfig::assistance_method)
        .def_readwrite("abnormal", &CenterOutControllerConfig::abnormal)
        .def_readwrite("max_observation_interval_ns",
                       &CenterOutControllerConfig::max_observation_interval_ns);

    py::class_<CenterOutController, NativeFrameConsumer>(center_out, "_CenterOutController",
                                                         py::is_final())
        .def(py::init<>())
        .def("prepare", &CenterOutController::prepare, py::arg("schema"), py::arg("config"))
        .def("start", &CenterOutController::start, py::arg("host_epoch_ns"),
             py::arg("experiment_epoch_ns") = 0)
        .def("close", &CenterOutController::close)
        .def("finish_halt", &CenterOutController::finish_halt)
        .def_property_readonly("prepared", &CenterOutController::prepared)
        .def_property_readonly("halted", &CenterOutController::halted)
        .def_property_readonly("dropped_trace_count", &CenterOutController::dropped_trace_count)
        .def_property_readonly("abnormal_summary", &CenterOutController::abnormal_summary)
        .def_property_readonly("position", [](const CenterOutController& controller)
                               { return controller.position(); })
        .def_property_readonly("snapshot", &CenterOutController::snapshot);

    py::class_<PyCenterOutSession>(center_out, "_CenterOutSession", py::is_final())
        .def(py::init<>())
        .def("prepare", &PyCenterOutSession::prepare, py::arg("decoded_schema"),
             py::arg("controller_config"), py::arg("host_epoch_ns"), py::arg("session_config"))
        .def_property_readonly("actuator", &PyCenterOutSession::actuator,
                               py::return_value_policy::reference_internal)
        .def("attach_runtime", &PyCenterOutSession::attach_runtime, py::arg("runner"),
             py::keep_alive<1, 2>())
        .def("start", &PyCenterOutSession::start, py::arg("time_ns"))
        .def("stop", &PyCenterOutSession::stop, py::arg("time_ns"), py::arg("reason"))
        .def("abort", &PyCenterOutSession::abort, py::arg("time_ns"), py::arg("reason"))
        .def("emergency_stop", &PyCenterOutSession::emergency_stop, py::arg("time_ns"),
             py::arg("reason"))
        .def("close", &PyCenterOutSession::close)
        .def_property_readonly("outcome", &PyCenterOutSession::outcome)
        .def_property_readonly("snapshot", &PyCenterOutSession::snapshot)
        .def_property_readonly("position", &PyCenterOutSession::position)
        .def("pop_training_label", &PyCenterOutSession::pop_training_label)
        .def_property_readonly("training_label_drops", &PyCenterOutSession::training_label_drops);

    py::class_<PyAdaptiveCenterOutSession>(center_out, "_AdaptiveCenterOutSession", py::is_final())
        .def(py::init<>())
        .def("prepare", &PyAdaptiveCenterOutSession::prepare, py::arg("feature_schema"),
             py::arg("initial_decoder"), py::arg("initial_version"), py::arg("controller_config"),
             py::arg("session_config"))
        .def("_set_host_epoch", &PyAdaptiveCenterOutSession::set_host_epoch,
             py::arg("host_epoch_ns"))
        .def("prepare_candidate", &PyAdaptiveCenterOutSession::prepare_candidate,
             py::arg("decoder"), py::arg("version"))
        .def("_prepare_final_candidate", &PyAdaptiveCenterOutSession::prepare_final_candidate,
             py::arg("decoder"), py::arg("version"))
        .def("_record_decoder_publication", &PyAdaptiveCenterOutSession::record_decoder_publication,
             py::arg("version"), py::arg("plan_fingerprint"), py::arg("training_blocks"),
             py::arg("training_trials"), py::arg("completed_trials"), py::arg("time_ns"))
        .def_property_readonly("actuator", &PyAdaptiveCenterOutSession::actuator,
                               py::return_value_policy::reference_internal)
        .def("attach_runtime", &PyAdaptiveCenterOutSession::attach_runtime, py::arg("runner"),
             py::keep_alive<1, 2>())
        .def("_enable_recording", &PyAdaptiveCenterOutSession::enable_recording,
             py::arg("attachment"), py::arg("owner"))
        .def("start", &PyAdaptiveCenterOutSession::start, py::arg("time_ns"))
        .def("stop", &PyAdaptiveCenterOutSession::stop, py::arg("time_ns"), py::arg("reason"))
        .def("abort", &PyAdaptiveCenterOutSession::abort, py::arg("time_ns"), py::arg("reason"))
        .def("close", &PyAdaptiveCenterOutSession::close)
        .def("pop_training_observation", &PyAdaptiveCenterOutSession::pop_training_observation)
        .def("pop_presentation_state", &PyAdaptiveCenterOutSession::pop_presentation_state)
        .def("_presentation_context", &PyAdaptiveCenterOutSession::presentation_context)
        .def_property_readonly("training_capture_drops",
                               &PyAdaptiveCenterOutSession::training_capture_drops)
        .def_property_readonly("presentation_state_drops",
                               &PyAdaptiveCenterOutSession::presentation_state_drops)
        .def_property_readonly("outcome", &PyAdaptiveCenterOutSession::outcome)
        .def_property_readonly("snapshot", &PyAdaptiveCenterOutSession::snapshot)
        .def_property_readonly("position", &PyAdaptiveCenterOutSession::position)
        .def("get_intent", &PyAdaptiveCenterOutSession::get_intent)
        .def_property_readonly("intent_source", &PyAdaptiveCenterOutSession::intent_source)
        .def_property_readonly("active_decoder_version",
                               &PyAdaptiveCenterOutSession::active_decoder_version)
        .def_property_readonly("completed_trials", &PyAdaptiveCenterOutSession::completed_trials)
        .def_property_readonly("complete", &PyAdaptiveCenterOutSession::complete);
}

} // namespace neurale::bindings

namespace
{

using namespace neurale::experiments;
using neurale::bindings::experiments::bind_validate;
using neurale::bindings::experiments::prefix;
using namespace neurale::experiments::center_out;

using ScalarOrPair = std::variant<double, std::array<double, 2>>;
using ScalarOrAcceptance = std::variant<double, AcceptanceRegion>;

using neurale::bindings::experiments::duration_from_seconds;

PhaseDurations phase_durations_from_seconds(const ScalarOrPair& seconds, const char* name,
                                            bool allow_zero)
{
    if (const auto* scalar = std::get_if<double>(&seconds))
    {
        const DurationNs value = duration_from_seconds(*scalar, name, allow_zero);
        return PhaseDurations{value, value};
    }
    const auto& pair = std::get<std::array<double, 2>>(seconds);
    return PhaseDurations{
        duration_from_seconds(pair[0], std::string(name) + "[0] (to_center)", allow_zero),
        duration_from_seconds(pair[1], std::string(name) + "[1] (to_out)", allow_zero)};
}

AcceptanceRegion acceptance_from_python(const ScalarOrAcceptance& value)
{
    AcceptanceRegion acceptance{};
    if (const auto* scalar = std::get_if<double>(&value))
        acceptance = AcceptanceRegion{*scalar, *scalar};
    else
        acceptance = std::get<AcceptanceRegion>(value);

    if (!std::isfinite(acceptance.half_extent_x) || !std::isfinite(acceptance.half_extent_y))
        throw py::value_error("acceptance half-extents must be finite");
    if (acceptance.half_extent_x <= 0.0 || acceptance.half_extent_y <= 0.0)
        throw py::value_error("acceptance half-extents must be greater than zero");
    return acceptance;
}

CenterOut2DConfig make_public_config(GeometryUnit geometry_unit, const CenterOut2DLayout& layout,
                                     const ScalarOrAcceptance& acceptance_value,
                                     const ScalarOrPair& movement_timeout_seconds,
                                     TargetSelectionPolicy selection, double cursor_extent,
                                     double hold_seconds, const ScalarOrPair& reward_dwell_seconds,
                                     const ScalarOrPair& punish_dwell_seconds, ScheduleSeed seed)
{
    if (geometry_unit == GeometryUnit::unspecified)
        throw py::value_error("geometry_unit must be specified");
    if (selection == TargetSelectionPolicy::unspecified)
        throw py::value_error("selection must be specified");
    if (validate(layout) != ContractStatus::ok)
        throw py::value_error(
            "layout must contain a valid center and at least one uniquely identified surrounding "
            "target");
    if (!std::isfinite(cursor_extent))
        throw py::value_error("cursor_extent must be finite");
    if (cursor_extent < 0.0)
        throw py::value_error("cursor_extent must not be negative");

    const AcceptanceRegion acceptance = acceptance_from_python(acceptance_value);
    if (cursor_extent > acceptance.half_extent_x || cursor_extent > acceptance.half_extent_y)
        throw py::value_error("cursor_extent must not exceed either acceptance half-extent");

    CenterOut2DConfig config{};
    config.geometry_unit = geometry_unit;
    config.layout = layout;
    config.acceptance = acceptance;
    config.cursor = CursorGeometry{cursor_extent};
    config.movement_timeout =
        phase_durations_from_seconds(movement_timeout_seconds, "movement_timeout_seconds", false);
    config.hold_ns = duration_from_seconds(hold_seconds, "hold_seconds", true);
    config.reward_dwell =
        phase_durations_from_seconds(reward_dwell_seconds, "reward_dwell_seconds", true);
    config.punish_dwell =
        phase_durations_from_seconds(punish_dwell_seconds, "punish_dwell_seconds", true);
    config.selection = selection;
    config.seed = seed;
    return config;
}

// The layout and the radial request differ from the command records on purpose:
// their count is *derived* from the sequence rather than declared beside it, so
// there is no way to under-supply a declared slot and no padding for a caller to
// get wrong. That was the hazard the command records' covering rule exists to
// close, and deriving removes it instead of guarding it.
template <typename Element, std::size_t Capacity>
std::array<Element, Capacity> to_fixed(const std::vector<Element>& items, const char* what)
{
    if (items.size() > Capacity)
        throw py::value_error(std::string(what) + " holds " + std::to_string(items.size()) +
                              ", more than the " + std::to_string(Capacity) +
                              " a layout has room for");
    std::array<Element, Capacity> fixed{};
    for (std::size_t i = 0; i < items.size(); ++i)
        fixed[i] = items[i];
    return fixed;
}

[[noreturn]] void throw_radial_layout_error(const RadialLayoutRequest& request)
{
    if (!std::isfinite(request.radius))
        throw py::value_error("radius must be finite");
    if (request.radius <= 0.0)
        throw py::value_error("radius must be greater than zero");
    if (request.count == 0)
        throw py::value_error("a radial layout must have at least one surrounding target");
    if (request.count > kMaxSurroundingTargets)
        throw py::value_error("a radial layout has more surrounding targets than its capacity");
    if (request.center_id == kUnsetTargetId)
        throw py::value_error("center_id must not be 0");

    for (std::size_t i = 0; i < request.count; ++i)
    {
        const TargetId id = request.ids[i];
        if (id == kUnsetTargetId)
            throw py::value_error("ids[" + std::to_string(i) + "] must not be 0");
        if (id == request.center_id)
            throw py::value_error("ids[" + std::to_string(i) + "] reuses center_id " +
                                  std::to_string(request.center_id));
        for (std::size_t other = 0; other < i; ++other)
            if (request.ids[other] == id)
                throw py::value_error("ids contains duplicate target id " + std::to_string(id));
    }
    throw py::value_error("the radial layout request is invalid");
}

// The meaningful prefix of a fixed-capacity sequence. Returning the padding too
// would invite a reader to treat an unset slot as a target that happens to sit
// at the origin, and would stop a layout from round-tripping through its own
// constructor.
// The state machine and everything it produces. Split out only for length; it is
// the same submodule, because the machine is what the configuration above is
// for.
//
// The records here divide by who builds them. CenterOutTrial and
// CenterOutStatistics take complete constructors, because a caller genuinely
// builds trials to hand to summarize(); CenterOutSnapshot and
// CenterOutStepResult do not, because only the machine produces one and a
// hand-built snapshot would describe a session that never ran.
void bind_center_out_machine(py::module_& center_out)
{
    center_out.attr("MAX_STEP_TRANSITIONS") = kMaxStepTransitions;
    center_out.attr("MAX_STEP_EVENTS") = kMaxStepEvents;

    py::enum_<CenterOutState>(center_out, "CenterOutState")
        .value("IDLE", CenterOutState::idle)
        .value("MOVE_TO_CENTER", CenterOutState::move_to_center)
        .value("HOLD_CENTER", CenterOutState::hold_center)
        .value("CENTER_SUCCESS_DWELL", CenterOutState::center_success_dwell)
        .value("CENTER_FAILURE_DWELL", CenterOutState::center_failure_dwell)
        .value("MOVE_TO_OUT", CenterOutState::move_to_out)
        .value("HOLD_OUT", CenterOutState::hold_out)
        .value("OUT_SUCCESS_DWELL", CenterOutState::out_success_dwell)
        .value("OUT_FAILURE_DWELL", CenterOutState::out_failure_dwell)
        .value("COMPLETE", CenterOutState::complete);

    py::enum_<CenterOutCause>(center_out, "CenterOutCause")
        .value("UNSPECIFIED", CenterOutCause::unspecified)
        .value("SESSION_STARTED", CenterOutCause::session_started)
        .value("CONTAINMENT_GAINED", CenterOutCause::containment_gained)
        .value("CONTAINMENT_LOST", CenterOutCause::containment_lost)
        .value("HOLD_COMPLETED", CenterOutCause::hold_completed)
        .value("MOVEMENT_TIMED_OUT", CenterOutCause::movement_timed_out)
        .value("DWELL_ELAPSED", CenterOutCause::dwell_elapsed)
        .value("TRIAL_LIMIT_REACHED", CenterOutCause::trial_limit_reached);

    py::enum_<CenterOutReason>(center_out, "CenterOutReason")
        .value("UNSPECIFIED", CenterOutReason::unspecified)
        .value("OUTWARD_ACQUIRED", CenterOutReason::outward_acquired)
        .value("CENTER_MOVEMENT_TIMEOUT", CenterOutReason::center_movement_timeout)
        .value("OUTWARD_MOVEMENT_TIMEOUT", CenterOutReason::outward_movement_timeout);

    center_out.def("center_out_state_declared", &center_out_state_declared, py::arg("state"));
    center_out.def("center_out_phase_of", &center_out_phase_of, py::arg("state"));
    center_out.def("center_out_state_is_leg", &center_out_state_is_leg, py::arg("state"));
    center_out.def("center_out_state_is_dwell", &center_out_state_is_dwell, py::arg("state"));

    py::class_<CenterOutTrial>(center_out, "CenterOutTrial")
        .def(py::init(
                 [](const TrialRecord& record, TargetId outward_target, std::uint8_t outward_idx,
                    CenterOutPhase decided_phase, DurationNs center_acquire_ns,
                    DurationNs outward_acquire_ns)
                 {
                     return CenterOutTrial{record,        outward_target,    outward_idx,
                                           decided_phase, center_acquire_ns, outward_acquire_ns};
                 }),
             py::arg("record") = TrialRecord{}, py::arg("outward_target") = kUnsetTargetId,
             py::arg("outward_idx") = 0, py::arg("decided_phase") = CenterOutPhase::to_center,
             py::arg("center_acquire_ns") = 0, py::arg("outward_acquire_ns") = 0)
        .def_readonly("record", &CenterOutTrial::record)
        .def_readonly("outward_target", &CenterOutTrial::outward_target)
        .def_readonly("outward_idx", &CenterOutTrial::outward_idx)
        .def_readonly("decided_phase", &CenterOutTrial::decided_phase)
        .def_readonly("center_acquire_ns", &CenterOutTrial::center_acquire_ns)
        .def_readonly("outward_acquire_ns", &CenterOutTrial::outward_acquire_ns);

    // No public constructor: only the machine produces a snapshot, and a
    // hand-built one would describe a session that never ran. Return values
    // still cross the boundary without it.
    py::class_<CenterOutSnapshot>(center_out, "CenterOutSnapshot")
        .def_readonly("time_ns", &CenterOutSnapshot::time_ns)
        .def_readonly("state", &CenterOutSnapshot::state)
        .def_readonly("phase", &CenterOutSnapshot::phase)
        .def_readonly("trial", &CenterOutSnapshot::trial)
        .def_readonly("active_target", &CenterOutSnapshot::active_target)
        .def_readonly("active_position", &CenterOutSnapshot::active_position)
        .def_readonly("outward_target", &CenterOutSnapshot::outward_target)
        .def_readonly("outward_idx", &CenterOutSnapshot::outward_idx)
        .def_readonly("contained", &CenterOutSnapshot::contained)
        .def_readonly("leg", &CenterOutSnapshot::leg)
        .def_readonly("hold", &CenterOutSnapshot::hold)
        .def_readonly("dwell", &CenterOutSnapshot::dwell)
        .def_readonly("completed", &CenterOutSnapshot::completed)
        .def_readonly("successes", &CenterOutSnapshot::successes);

    // As CenterOutSnapshot above: only the machine produces a step result.
    py::class_<CenterOutStepResult>(center_out, "CenterOutStepResult")
        .def_readonly("snapshot", &CenterOutStepResult::snapshot)
        .def_readonly("settled", &CenterOutStepResult::settled)
        .def_readonly("trial_decided", &CenterOutStepResult::trial_decided)
        .def_readonly("trial", &CenterOutStepResult::trial)
        .def_property_readonly("transitions", [](const CenterOutStepResult& result)
                               { return prefix(result.transitions, result.n_transitions); })
        .def_property_readonly("events", [](const CenterOutStepResult& result)
                               { return prefix(result.events, result.n_events); });

    py::class_<CenterOutStatistics>(center_out, "CenterOutStatistics")
        .def(py::init(
                 [](std::uint64_t decided, std::uint64_t successes, std::uint64_t center_timeouts,
                    std::uint64_t outward_timeouts, DurationNs total_time_to_target_ns,
                    DurationNs mean_time_to_target_ns)
                 {
                     return CenterOutStatistics{decided,
                                                successes,
                                                center_timeouts,
                                                outward_timeouts,
                                                total_time_to_target_ns,
                                                mean_time_to_target_ns};
                 }),
             py::arg("decided") = 0, py::arg("successes") = 0, py::arg("center_timeouts") = 0,
             py::arg("outward_timeouts") = 0, py::arg("total_time_to_target_ns") = 0,
             py::arg("mean_time_to_target_ns") = 0)
        .def_readonly("decided", &CenterOutStatistics::decided)
        .def_readonly("successes", &CenterOutStatistics::successes)
        .def_readonly("center_timeouts", &CenterOutStatistics::center_timeouts)
        .def_readonly("outward_timeouts", &CenterOutStatistics::outward_timeouts)
        .def_readonly("total_time_to_target_ns", &CenterOutStatistics::total_time_to_target_ns)
        .def_readonly("mean_time_to_target_ns", &CenterOutStatistics::mean_time_to_target_ns);

    center_out.def(
        "summarize",
        [](const std::vector<CenterOutTrial>& trials)
        {
            CenterOutStatistics statistics{};
            const ContractStatus status =
                summarize(std::span<const CenterOutTrial>(trials), statistics);
            return std::pair{status, statistics};
        },
        py::arg("trials"));

    py::class_<CenterOutMachine>(center_out, "CenterOutMachine")
        .def(py::init<>())
        .def(
            "start",
            [](CenterOutMachine& machine, ParadigmId paradigm, const CenterOut2DConfig& config,
               ExperimentTimeNs time_ns)
            {
                CenterOutStepResult result{};
                const ContractStatus status = machine.start(paradigm, config, time_ns, result);
                return std::pair{status, result};
            },
            py::arg("paradigm"), py::arg("config"), py::arg("time_ns"))
        .def("reset", &CenterOutMachine::reset)
        .def(
            "step",
            [](CenterOutMachine& machine, ExperimentTimeNs time_ns, const WorkspacePoint& cursor)
            {
                CenterOutStepResult result{};
                const ContractStatus status = machine.step(time_ns, cursor, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"), py::arg("cursor"))
        .def("snapshot", &CenterOutMachine::snapshot)
        .def("configuration", &CenterOutMachine::configuration)
        .def_property_readonly("paradigm", &CenterOutMachine::paradigm)
        .def_property_readonly("state", &CenterOutMachine::state)
        .def_property_readonly("complete", &CenterOutMachine::complete);
}

} // namespace

void bind_experiments_center_out_module(py::module_& experiments)
{
    auto center_out = experiments.def_submodule(
        "center_out", "Center-Out 2D configuration, target geometry, deterministic target "
                      "schedule, and pure deterministic task state machine. No timer, no "
                      "device, and no renderer.");

    center_out.attr("MAX_SURROUNDING_TARGETS") = kMaxSurroundingTargets;
    center_out.attr("MIN_RADIAL_SPOKES") = kMinRadialSpokes;
    center_out.attr("TARGET_SELECTION_STREAM") = kTargetSelectionStream;

    py::enum_<GeometryUnit>(center_out, "GeometryUnit")
        .value("UNSPECIFIED", GeometryUnit::unspecified)
        .value("DIMENSIONLESS", GeometryUnit::dimensionless)
        .value("NORMALIZED", GeometryUnit::normalized)
        .value("MILLIMETRES", GeometryUnit::millimetres)
        .value("METRES", GeometryUnit::metres);

    py::enum_<CenterOutPhase>(center_out, "CenterOutPhase")
        .value("TO_CENTER", CenterOutPhase::to_center)
        .value("TO_OUT", CenterOutPhase::to_out);

    py::enum_<TargetSelectionPolicy>(center_out, "TargetSelectionPolicy")
        .value("UNSPECIFIED", TargetSelectionPolicy::unspecified)
        .value("REPEAT_UNTIL_SUCCESS", TargetSelectionPolicy::repeat_until_success)
        .value("SAMPLE_EACH_TRIAL", TargetSelectionPolicy::sample_each_trial)
        .value("BALANCED_SHUFFLED_CYCLES", TargetSelectionPolicy::balanced_shuffled_cycles);

    center_out.def("geometry_unit_declared", &geometry_unit_declared, py::arg("unit"));
    center_out.def("center_out_phase_declared", &center_out_phase_declared, py::arg("phase"));
    center_out.def("target_selection_policy_declared", &target_selection_policy_declared,
                   py::arg("policy"));

    py::class_<WorkspacePoint>(center_out, "WorkspacePoint")
        .def(py::init([](double x, double y) { return WorkspacePoint{x, y}; }), py::arg("x") = 0.0,
             py::arg("y") = 0.0)
        .def_readonly("x", &WorkspacePoint::x)
        .def_readonly("y", &WorkspacePoint::y);

    py::class_<TargetPlacement>(center_out, "TargetPlacement")
        .def(py::init([](TargetId id, const WorkspacePoint& pos)
                      { return TargetPlacement{id, pos}; }),
             py::arg("id") = kUnsetTargetId, py::arg("pos") = WorkspacePoint{})
        .def_readonly("id", &TargetPlacement::id)
        .def_readonly("pos", &TargetPlacement::pos);

    py::class_<CenterOut2DLayout>(center_out, "CenterOut2DLayout")
        .def(py::init(
                 [](const TargetPlacement& center, const std::vector<TargetPlacement>& surrounding)
                 {
                     CenterOut2DLayout layout{};
                     layout.center = center;
                     layout.count = static_cast<std::uint8_t>(surrounding.size());
                     layout.surrounding = to_fixed<TargetPlacement, kMaxSurroundingTargets>(
                         surrounding, "surrounding");
                     return layout;
                 }),
             py::arg("center") = TargetPlacement{},
             py::arg("surrounding") = std::vector<TargetPlacement>{})
        .def_readonly("center", &CenterOut2DLayout::center)
        .def_readonly("count", &CenterOut2DLayout::count)
        .def_property_readonly("surrounding", [](const CenterOut2DLayout& layout)
                               { return prefix(layout.surrounding, layout.count); })
        .def(
            "target",
            [](const CenterOut2DLayout& layout, std::size_t idx)
            {
                if (idx >= layout.count)
                    throw py::index_error("surrounding target index out of range");
                return layout.surrounding[idx];
            },
            py::arg("idx"));

    py::class_<AcceptanceRegion>(center_out, "AcceptanceRegion")
        .def(py::init([](double half_extent_x, double half_extent_y)
                      { return AcceptanceRegion{half_extent_x, half_extent_y}; }),
             py::arg("half_extent_x") = 0.0, py::arg("half_extent_y") = 0.0)
        .def_readonly("half_extent_x", &AcceptanceRegion::half_extent_x)
        .def_readonly("half_extent_y", &AcceptanceRegion::half_extent_y);

    py::class_<CursorGeometry>(center_out, "CursorGeometry")
        .def(py::init([](double extent) { return CursorGeometry{extent}; }),
             py::arg("extent") = 0.0)
        .def_readonly("extent", &CursorGeometry::extent);

    py::class_<PhaseDurations>(center_out, "PhaseDurations")
        .def(py::init([](DurationNs to_center, DurationNs to_out)
                      { return PhaseDurations{to_center, to_out}; }),
             py::arg("to_center") = 0, py::arg("to_out") = 0)
        .def_readonly("to_center", &PhaseDurations::to_center)
        .def_readonly("to_out", &PhaseDurations::to_out)
        .def(
            "of", [](const PhaseDurations& durations, CenterOutPhase phase)
            { return phase_duration(durations, phase); }, py::arg("phase"));

    py::class_<CenterOut2DConfig>(center_out, "CenterOutTask")
        .def(py::init(
                 [](GeometryUnit geometry_unit, const CenterOut2DLayout& layout,
                    const ScalarOrAcceptance& acceptance,
                    const ScalarOrPair& movement_timeout_seconds, TargetSelectionPolicy selection,
                    double cursor_extent, double hold_seconds,
                    const ScalarOrPair& reward_dwell_seconds,
                    const ScalarOrPair& punish_dwell_seconds, ScheduleSeed seed)
                 {
                     return make_public_config(geometry_unit, layout, acceptance,
                                               movement_timeout_seconds, selection, cursor_extent,
                                               hold_seconds, reward_dwell_seconds,
                                               punish_dwell_seconds, seed);
                 }),
             py::kw_only(), py::arg("geometry_unit"), py::arg("layout"), py::arg("acceptance"),
             py::arg("movement_timeout_seconds"), py::arg("selection"),
             py::arg("cursor_extent") = 0.0, py::arg("hold_seconds") = 0.0,
             py::arg("reward_dwell_seconds") = ScalarOrPair{0.0},
             py::arg("punish_dwell_seconds") = ScalarOrPair{0.0}, py::arg("seed") = 0,
             "Create a validated Center-Out task. Scalar acceptance creates an equal x/y "
             "region; duration scalars apply to both trial legs, while two-item sequences set "
             "the to-center and to-out values separately. Durations are expressed in seconds.")
        .def_readonly("geometry_unit", &CenterOut2DConfig::geometry_unit)
        .def_readonly("layout", &CenterOut2DConfig::layout)
        .def_readonly("acceptance", &CenterOut2DConfig::acceptance)
        .def_readonly("cursor", &CenterOut2DConfig::cursor)
        .def_readonly("movement_timeout", &CenterOut2DConfig::movement_timeout)
        .def_readonly("hold_ns", &CenterOut2DConfig::hold_ns)
        .def_readonly("reward_dwell", &CenterOut2DConfig::reward_dwell)
        .def_readonly("punish_dwell", &CenterOut2DConfig::punish_dwell)
        .def_readonly("selection", &CenterOut2DConfig::selection)
        .def_readonly("seed", &CenterOut2DConfig::seed)
        .def_readonly("sampler_version", &CenterOut2DConfig::sampler_version)
        .def_property_readonly("movement_timeout_seconds",
                               [](const CenterOut2DConfig& config)
                               {
                                   return std::array<double, 2>{
                                       static_cast<double>(config.movement_timeout.to_center) / 1e9,
                                       static_cast<double>(config.movement_timeout.to_out) / 1e9};
                               })
        .def_property_readonly("hold_seconds", [](const CenterOut2DConfig& config)
                               { return static_cast<double>(config.hold_ns) / 1e9; })
        .def_property_readonly("reward_dwell_seconds",
                               [](const CenterOut2DConfig& config)
                               {
                                   return std::array<double, 2>{
                                       static_cast<double>(config.reward_dwell.to_center) / 1e9,
                                       static_cast<double>(config.reward_dwell.to_out) / 1e9};
                               })
        .def_property_readonly("punish_dwell_seconds",
                               [](const CenterOut2DConfig& config)
                               {
                                   return std::array<double, 2>{
                                       static_cast<double>(config.punish_dwell.to_center) / 1e9,
                                       static_cast<double>(config.punish_dwell.to_out) / 1e9};
                               });

    // Internal construction paths preserve native/replay tests without making
    // sampler implementation versions or session length part of the public
    // task constructor.
    center_out.def(
        "_raw_center_out_config",
        [](GeometryUnit geometry_unit, const CenterOut2DLayout& layout,
           const AcceptanceRegion& acceptance, const CursorGeometry& cursor,
           const PhaseDurations& movement_timeout, DurationNs hold_ns,
           const PhaseDurations& reward_dwell, const PhaseDurations& punish_dwell,
           TargetSelectionPolicy selection, ScheduleSeed seed, SamplerVersion sampler_version,
           TrialOrdinal trial_limit)
        {
            return CenterOut2DConfig{geometry_unit,    layout,  acceptance,      cursor,
                                     movement_timeout, hold_ns, reward_dwell,    punish_dwell,
                                     selection,        seed,    sampler_version, trial_limit};
        },
        py::kw_only(), py::arg("geometry_unit") = GeometryUnit::unspecified,
        py::arg("layout") = CenterOut2DLayout{}, py::arg("acceptance") = AcceptanceRegion{},
        py::arg("cursor") = CursorGeometry{}, py::arg("movement_timeout") = PhaseDurations{},
        py::arg("hold_ns") = 0, py::arg("reward_dwell") = PhaseDurations{},
        py::arg("punish_dwell") = PhaseDurations{},
        py::arg("selection") = TargetSelectionPolicy::unspecified, py::arg("seed") = 0,
        py::arg("sampler_version") = kCurrentSamplerVersion, py::arg("trial_limit") = 0);

    center_out.def(
        "_with_trial_limit",
        [](const CenterOut2DConfig& config, TrialOrdinal trials)
        {
            CenterOut2DConfig result = config;
            result.trial_limit = trials;
            return result;
        },
        py::arg("config"), py::arg("trials"));

    py::class_<RadialLayoutRequest>(
        center_out, "RadialLayoutRequest",
        "Immutable radial Center-Out layout request. Omitting ids and spokes creates the "
        "classic eight-direction layout.")
        .def(py::init(
                 [](double radius, TargetId center_id,
                    const std::optional<std::vector<TargetId>>& ids,
                    const std::optional<std::vector<std::uint32_t>>& spokes)
                 {
                     if (ids.has_value() != spokes.has_value())
                         throw py::value_error("ids and spokes must be provided together");

                     std::vector<TargetId> resolved_ids;
                     std::vector<std::uint32_t> resolved_spokes;
                     if (!ids)
                     {
                         if (center_id != 1)
                             throw py::value_error(
                                 "center_id may only be changed when ids and spokes are provided");
                         resolved_ids = {2, 3, 4, 5, 6, 7, 8, 9};
                         resolved_spokes = {0, 1, 2, 3, 4, 5, 6, 7};
                     }
                     else
                     {
                         resolved_ids = *ids;
                         resolved_spokes = *spokes;
                     }

                     if (resolved_ids.size() != resolved_spokes.size())
                         throw py::value_error(
                             "ids holds " + std::to_string(resolved_ids.size()) +
                             " and spokes holds " + std::to_string(resolved_spokes.size()) +
                             "; a target without a spoke has no position, and a spoke without a "
                             "target names nothing");
                     RadialLayoutRequest request{};
                     request.radius = radius;
                     request.count = static_cast<std::uint8_t>(resolved_ids.size());
                     request.center_id = center_id;
                     request.ids = to_fixed<TargetId, kMaxSurroundingTargets>(resolved_ids, "ids");
                     request.spokes =
                         to_fixed<std::uint32_t, kMaxSurroundingTargets>(resolved_spokes, "spokes");
                     return request;
                 }),
             py::arg("radius"), py::arg("center_id") = 1, py::arg("ids") = py::none(),
             py::arg("spokes") = py::none(),
             "Create a request. Omit ids and spokes for center_id 1 and eight targets on "
             "spokes 0 through 7; provide both sequences for an advanced layout.")
        .def_readonly("radius", &RadialLayoutRequest::radius)
        .def_readonly("count", &RadialLayoutRequest::count)
        .def_readonly("center_id", &RadialLayoutRequest::center_id)
        .def_property_readonly("ids", [](const RadialLayoutRequest& request)
                               { return prefix(request.ids, request.count); })
        .def_property_readonly("spokes", [](const RadialLayoutRequest& request)
                               { return prefix(request.spokes, request.count); });

    center_out.def("radial_spoke_step", &radial_spoke_step, py::arg("count"));

    // Most functions mirror their native status-returning contract. Layout
    // construction is the ordinary Python setup path, so it raises ValueError
    // with the rejected field and returns the layout directly on success.
    center_out.def(
        "radial_position",
        [](double radius, std::uint32_t count, std::uint32_t spoke)
        {
            WorkspacePoint pos{};
            const ContractStatus status = radial_position(radius, count, spoke, pos);
            return std::pair{status, pos};
        },
        py::arg("radius"), py::arg("count"), py::arg("spoke"));

    center_out.def(
        "build_radial_layout",
        [](const RadialLayoutRequest& request)
        {
            CenterOut2DLayout layout{};
            const ContractStatus status = build_radial_layout(request, layout);
            if (status != ContractStatus::ok)
                throw_radial_layout_error(request);
            return layout;
        },
        py::arg("request"),
        "Build a radial layout, raising ValueError with the rejected field when invalid.");

    center_out.def(
        "contains_point",
        [](const AcceptanceRegion& region, const WorkspacePoint& target,
           const WorkspacePoint& cursor)
        {
            bool inside = false;
            const ContractStatus status = contains_point(region, target, cursor, inside);
            return std::pair{status, inside};
        },
        py::arg("region"), py::arg("target"), py::arg("cursor"));

    center_out.def(
        "contains_cursor",
        [](const AcceptanceRegion& region, const CursorGeometry& cursor_geometry,
           const WorkspacePoint& target, const WorkspacePoint& cursor)
        {
            bool inside = false;
            const ContractStatus status =
                contains_cursor(region, cursor_geometry, target, cursor, inside);
            return std::pair{status, inside};
        },
        py::arg("region"), py::arg("cursor_geometry"), py::arg("target"), py::arg("cursor"));

    center_out.def(
        "select_outward_target",
        [](const CenterOut2DConfig& config, TrialOrdinal trial, std::uint64_t successes)
        {
            std::uint8_t idx = 0;
            const ContractStatus status = select_outward_target(config, trial, successes, idx);
            return std::pair{status, idx};
        },
        py::arg("config"), py::arg("trial"), py::arg("successes"));

    center_out.def("trial_limit_reached", &trial_limit_reached, py::arg("config"),
                   py::arg("completed"));
    center_out.def("phase_duration", &phase_duration, py::arg("durations"), py::arg("phase"));
    center_out.def("layout_fingerprint", &layout_fingerprint, py::arg("layout"));
    center_out.def("configuration_fingerprint", &configuration_fingerprint, py::arg("config"));

    bind_validate<TargetPlacement>(center_out);
    bind_validate<CenterOut2DLayout>(center_out);
    bind_validate<AcceptanceRegion>(center_out);
    bind_validate<CursorGeometry>(center_out);
    bind_validate<RadialLayoutRequest>(center_out);
    bind_validate<CenterOut2DConfig>(center_out);

    bind_center_out_machine(center_out);
    bind_validate<CenterOutTrial>(center_out);
    bind_center_out_guidance(center_out);
    neurale::bindings::bind_center_out_session(center_out);
}
