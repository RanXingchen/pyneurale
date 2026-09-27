// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "binding_helpers.h"
#include "experiment_attachment.h"
#include "ssvep_controller.h"
#include <neurale/experiments/ssvep.h>
#include <neurale/streaming/runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace neurale::bindings
{
namespace
{
using namespace ::neurale::execution;
using streaming::StreamStatus;

class PySSVEPSession
{
  public:
    PySSVEPSession(const streaming::StreamSchema& schema,
                   ::neurale::experiments::ssvep::SSVEPConfig task, std::size_t calibration,
                   std::size_t trials)
        : processing_(schema, task.stimulation_duration_ns, trials),
          controller_(schema, with_trials(task, trials), calibration, processing_),
          session_(controller_, {})
    {
    }
    ~PySSVEPSession()
    {
        // Join before releasing the Python-owned native processor.
        controller_.stop_execution(true);
    }
    static ::neurale::experiments::ssvep::SSVEPConfig
    with_trials(::neurale::experiments::ssvep::SSVEPConfig task, std::size_t trials)
    {
        task.n_trials = trials;
        return task;
    }
    streaming::NativeFrameConsumer& actuator()
    {
        return processing_;
    }
    StreamStatus attach(streaming::NativeStreamRunner& runner)
    {
        return session_.attach_runtime(runner);
    }
    StreamStatus start(std::uint64_t epoch)
    {
        controller_.set_host_epoch(epoch);
        return session_.start(0);
    }
    StreamStatus stop(std::uint64_t now)
    {
        return session_.stop(now, "task-complete");
    }
    StreamStatus abort(std::uint64_t now)
    {
        return session_.abort(now, "ssvep-session-aborted");
    }
    recording::RecorderStatusCode close()
    {
        return session_.close();
    }
    bool complete() const
    {
        if (controller_.status() != StreamStatus::ok)
            throw std::runtime_error("SSVEP native task or interval processing failed");
        return controller_.complete();
    }
    bool training() const
    {
        return controller_.training();
    }
    std::uint64_t drops() const
    {
        return controller_.dropped_trace_count();
    }
    py::object pop_snapshot()
    {
        ::neurale::experiments::ssvep::SSVEPSnapshot snapshot;
        if (controller_.pop_snapshot(snapshot) != StreamStatus::ok)
            return py::none();
        return py::cast(snapshot);
    }
    py::list trials() const
    {
        py::list result;
        for (std::size_t i = 0; i < controller_.completed_trials(); ++i)
            result.append(py::cast(controller_.trial(i)));
        return result;
    }
    py::list features() const
    {
        py::list result;
        for (std::size_t i = 0; i < controller_.ready_trials(); ++i)
        {
            const auto row = controller_.features(i);
            result.append(std::vector<double>(row.begin(), row.end()));
        }
        return result;
    }
    streaming::StreamSchema feature_schema() const
    {
        return processing_.feature_schema().clone();
    }
    std::vector<std::uint64_t> feature_times() const
    {
        std::vector<std::uint64_t> result;
        for (std::size_t i = 0; i < processing_.ready(); ++i)
        {
            const auto interval = processing_.interval(i);
            result.push_back(interval.start_ns + (interval.end_ns - interval.start_ns) / 2);
        }
        return result;
    }
    void publish(py::object decoder, std::string plan, std::string fingerprint,
                 std::uint64_t time_ns)
    {
        if (decoder_)
            throw py::value_error("SSVEP decoder has already been published");
        if (!controller_.training())
            throw py::value_error("decoder publication requires completed calibration");
        processing_.publish_decoder(decoder.cast<streaming::NativeFrameProcessor&>());
        controller_.publish({std::move(plan), std::move(fingerprint), time_ns});
        decoder_ = std::move(decoder);
    }
    bool report_display(const ::neurale::experiments::ssvep::SSVEPSnapshot& snapshot,
                        std::uint64_t requested, std::uint64_t submitted, std::uint64_t presented,
                        bool expired)
    {
        return controller_.report_display({snapshot, requested, submitted, presented, expired});
    }
    StreamStatus enable_recording(py::capsule capsule, py::object owner)
    {
        auto* attachment = static_cast<recording::ExperimentRecorderAttachment*>(
            PyCapsule_GetPointer(capsule.ptr(), recording::kExperimentRecorderAttachmentCapsule));
        if (!attachment)
            throw py::error_already_set();
        const auto status = session_.enable_recording(
            {attachment->recorder, attachment->plan, attachment->spool, attachment->clock, true});
        if (status == StreamStatus::ok)
            recorder_ = std::move(owner);
        return status;
    }
    py::dict outcome() const
    {
        const auto value = session_.outcome();
        py::dict result;
        result["experiment_trace_complete"] = value.experiment_trace_complete;
        result["trace_losses"] = value.trace_losses;
        result["runtime_status"] = value.runtime_status;
        result["terminal_abort"] = value.terminal_abort;
        return result;
    }

  private:
    IntervalRunner processing_;
    SSVEPController controller_;
    py::object decoder_;
    ExperimentSession session_;
    py::object recorder_;
};
} // namespace
static void bind_ssvep_session(py::module_& module)
{
    py::class_<PySSVEPSession>(module, "_SSVEPSession")
        .def(py::init<const streaming::StreamSchema&, ::neurale::experiments::ssvep::SSVEPConfig,
                      std::size_t, std::size_t>())
        .def_property_readonly("actuator", &PySSVEPSession::actuator,
                               py::return_value_policy::reference_internal)
        .def("attach", &PySSVEPSession::attach, py::keep_alive<1, 2>())
        .def("start", &PySSVEPSession::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &PySSVEPSession::stop, py::call_guard<py::gil_scoped_release>())
        .def("abort", &PySSVEPSession::abort, py::call_guard<py::gil_scoped_release>())
        .def("close", &PySSVEPSession::close, py::call_guard<py::gil_scoped_release>())
        .def("publish", &PySSVEPSession::publish)
        .def("report_display", &PySSVEPSession::report_display)
        .def("_enable_recording", &PySSVEPSession::enable_recording)
        .def("pop_snapshot", &PySSVEPSession::pop_snapshot)
        .def_property_readonly("complete", &PySSVEPSession::complete)
        .def_property_readonly("training", &PySSVEPSession::training)
        .def_property_readonly("drops", &PySSVEPSession::drops)
        .def_property_readonly("trials", &PySSVEPSession::trials)
        .def_property_readonly("features", &PySSVEPSession::features)
        .def_property_readonly("feature_schema",
                               [](PySSVEPSession& self) { return self.feature_schema(); })
        .def_property_readonly("feature_times_ns", &PySSVEPSession::feature_times)
        .def_property_readonly("outcome", &PySSVEPSession::outcome);
}
} // namespace neurale::bindings

using namespace neurale::experiments;
using namespace neurale::experiments::ssvep;
using neurale::bindings::experiments::bind_validate;
using neurale::bindings::experiments::duration_from_seconds;
using neurale::bindings::experiments::prefix;

void bind_experiments_ssvep_module(py::module_& experiments)
{
    auto module = experiments.def_submodule("ssvep", "Pure explicit-time SSVEP selection task.");
    module.attr("MAX_SSVEP_TARGETS") = kMaxSSVEPTargets;
    py::enum_<SSVEPState>(module, "SSVEPState")
        .value("IDLE", SSVEPState::idle)
        .value("CUE", SSVEPState::cue)
        .value("STIMULATION", SSVEPState::stimulation)
        .value("AWAIT_DECISION", SSVEPState::await_decision)
        .value("FEEDBACK", SSVEPState::feedback)
        .value("INTER_TRIAL", SSVEPState::inter_trial)
        .value("COMPLETE", SSVEPState::complete);
    py::enum_<SSVEPPhase>(module, "SSVEPPhase")
        .value("NONE", SSVEPPhase::none)
        .value("CUE", SSVEPPhase::cue)
        .value("STIMULATION", SSVEPPhase::stimulation)
        .value("AWAIT_DECISION", SSVEPPhase::await_decision)
        .value("FEEDBACK", SSVEPPhase::feedback)
        .value("INTER_TRIAL", SSVEPPhase::inter_trial);
    py::enum_<SSVEPCause>(module, "SSVEPCause")
        .value("UNSPECIFIED", SSVEPCause::unspecified)
        .value("SESSION_STARTED", SSVEPCause::session_started)
        .value("CUE_ELAPSED", SSVEPCause::cue_elapsed)
        .value("STIMULATION_ELAPSED", SSVEPCause::stimulation_elapsed)
        .value("SELECTION_RECEIVED", SSVEPCause::selection_received)
        .value("DECISION_TIMEOUT", SSVEPCause::decision_timeout)
        .value("FEEDBACK_ELAPSED", SSVEPCause::feedback_elapsed)
        .value("REST_ELAPSED", SSVEPCause::rest_elapsed)
        .value("TRIAL_LIMIT_REACHED", SSVEPCause::trial_limit_reached)
        .value("STOPPED", SSVEPCause::stopped);
    py::enum_<SSVEPReason>(module, "SSVEPReason")
        .value("UNSPECIFIED", SSVEPReason::unspecified)
        .value("CORRECT_SELECTION", SSVEPReason::correct_selection)
        .value("INCORRECT_SELECTION", SSVEPReason::incorrect_selection)
        .value("DECISION_TIMEOUT", SSVEPReason::decision_timeout)
        .value("STOPPED", SSVEPReason::stopped);
    py::enum_<SSVEPMarker>(module, "SSVEPMarker")
        .value("UNSPECIFIED", SSVEPMarker::unspecified)
        .value("CUE_ONSET", SSVEPMarker::cue_onset)
        .value("CUE_OFFSET", SSVEPMarker::cue_offset)
        .value("STIMULATION_ONSET", SSVEPMarker::stimulation_onset)
        .value("STIMULATION_OFFSET", SSVEPMarker::stimulation_offset)
        .value("WAIT_ONSET", SSVEPMarker::wait_onset)
        .value("WAIT_OFFSET", SSVEPMarker::wait_offset)
        .value("FEEDBACK_ONSET", SSVEPMarker::feedback_onset)
        .value("FEEDBACK_OFFSET", SSVEPMarker::feedback_offset)
        .value("REST_ONSET", SSVEPMarker::rest_onset)
        .value("REST_OFFSET", SSVEPMarker::rest_offset);
    py::enum_<SSVEPSelectionDisposition>(module, "SSVEPSelectionDisposition")
        .value("NONE", SSVEPSelectionDisposition::none)
        .value("ACCEPTED", SSVEPSelectionDisposition::accepted)
        .value("EXPIRED", SSVEPSelectionDisposition::expired);
    py::class_<SSVEPTarget>(module, "SSVEPTarget")
        .def(py::init(
                 [](TargetId id, double frequency_hz)
                 {
                     if (id == 0)
                         throw py::value_error("id must be nonzero");
                     if (!std::isfinite(frequency_hz) || frequency_hz <= 0)
                         throw py::value_error("frequency_hz must be finite and greater than zero");
                     return SSVEPTarget{id, frequency_hz};
                 }),
             py::arg("id"), py::arg("frequency_hz"))
        .def_readonly("id", &SSVEPTarget::id)
        .def_readonly("frequency_hz", &SSVEPTarget::frequency_hz);
    py::class_<SSVEPTrialSchedule>(module, "SSVEPTrialSchedule")
        .def_readonly("ordinal", &SSVEPTrialSchedule::ordinal)
        .def_readonly("target_id", &SSVEPTrialSchedule::target_id);
    py::class_<SSVEPPhaseInterval>(module, "SSVEPPhaseInterval")
        .def_readonly("phase", &SSVEPPhaseInterval::phase)
        .def_readonly("interval", &SSVEPPhaseInterval::interval);
    py::class_<SSVEPPresentationRequest>(module, "SSVEPPresentationRequest")
        .def_readonly("request", &SSVEPPresentationRequest::request)
        .def_readonly("selected_id", &SSVEPPresentationRequest::selected_id)
        .def_readonly("outcome", &SSVEPPresentationRequest::outcome);
    py::class_<SSVEPConfig>(module, "SSVEPTask")
        .def(py::init(
                 [](const std::vector<SSVEPTarget>& targets, StimulusId stimulus_id,
                    double cue_duration, double stimulation_duration, double decision_timeout,
                    double feedback_duration, double inter_trial, ScheduleSeed seed)
                 {
                     if (targets.size() < 2 || targets.size() > kMaxSSVEPTargets)
                         throw py::value_error("targets must contain between 2 and 64 targets");
                     if (stimulus_id == 0)
                         throw py::value_error("stimulus_id must be nonzero");
                     SSVEPConfig config{};
                     config.n_targets = static_cast<std::uint8_t>(targets.size());
                     for (std::size_t i = 0; i < targets.size(); ++i)
                         config.targets[i] = targets[i];
                     config.stimulus_id = stimulus_id;
                     // Session length is supplied at start; task validation needs a positive count.
                     config.n_trials = 1;
                     config.seed = seed;
                     config.cue_duration_ns =
                         duration_from_seconds(cue_duration, "cue_duration", false);
                     config.stimulation_duration_ns =
                         duration_from_seconds(stimulation_duration, "stimulation_duration", false);
                     config.decision_timeout_ns =
                         duration_from_seconds(decision_timeout, "decision_timeout", false);
                     config.feedback_duration_ns =
                         duration_from_seconds(feedback_duration, "feedback_duration", false);
                     config.inter_trial_ns =
                         duration_from_seconds(inter_trial, "inter_trial", false);
                     if (validate(config) != ContractStatus::ok)
                         throw py::value_error(
                             "targets must have unique IDs and unique frequencies");
                     return config;
                 }),
             py::kw_only(), py::arg("targets"), py::arg("stimulus_id") = 1, py::arg("cue_duration"),
             py::arg("stimulation_duration"), py::arg("decision_timeout"),
             py::arg("feedback_duration"), py::arg("inter_trial"), py::arg("seed") = 0,
             "Validated SSVEP task. Durations are in seconds; trial count belongs to start().")
        .def_property_readonly("targets",
                               [](const SSVEPConfig& c) { return prefix(c.targets, c.n_targets); })
        .def_readonly("stimulus_id", &SSVEPConfig::stimulus_id)
        .def_readonly("seed", &SSVEPConfig::seed)
        .def_property_readonly("cue_duration", [](const SSVEPConfig& c)
                               { return static_cast<double>(c.cue_duration_ns) / 1e9; })
        .def_property_readonly("stimulation_duration", [](const SSVEPConfig& c)
                               { return static_cast<double>(c.stimulation_duration_ns) / 1e9; })
        .def_property_readonly("decision_timeout", [](const SSVEPConfig& c)
                               { return static_cast<double>(c.decision_timeout_ns) / 1e9; })
        .def_property_readonly("feedback_duration", [](const SSVEPConfig& c)
                               { return static_cast<double>(c.feedback_duration_ns) / 1e9; })
        .def_property_readonly("inter_trial", [](const SSVEPConfig& c)
                               { return static_cast<double>(c.inter_trial_ns) / 1e9; });
    py::class_<SSVEPTrial>(module, "SSVEPTrial")
        .def_property_readonly("phases",
                               [](const SSVEPTrial& v) { return prefix(v.phases, v.n_phases); })
        .def_readonly("record", &SSVEPTrial::record)
        .def_readonly("schedule", &SSVEPTrial::schedule)
        .def_readonly("has_selection", &SSVEPTrial::has_selection)
        .def_readonly("selection", &SSVEPTrial::selection)
        .def_readonly("has_decision", &SSVEPTrial::has_decision)
        .def_readonly("decision_ns", &SSVEPTrial::decision_ns);
    py::class_<SSVEPSnapshot>(module, "SSVEPSnapshot")
        .def_readonly("time_ns", &SSVEPSnapshot::time_ns)
        .def_readonly("state", &SSVEPSnapshot::state)
        .def_readonly("trial", &SSVEPSnapshot::trial)
        .def_readonly("active", &SSVEPSnapshot::active)
        .def_readonly("stimulation", &SSVEPSnapshot::stimulation)
        .def_readonly("decision_deadline_ns", &SSVEPSnapshot::decision_deadline_ns)
        .def_readonly("has_selection", &SSVEPSnapshot::has_selection)
        .def_readonly("selection", &SSVEPSnapshot::selection)
        .def_readonly("outcome", &SSVEPSnapshot::outcome)
        .def_readonly("completed", &SSVEPSnapshot::completed);
    py::class_<SSVEPStepResult>(module, "SSVEPStepResult")
        .def_readonly("snapshot", &SSVEPStepResult::snapshot)
        .def_readonly("selection_disposition", &SSVEPStepResult::selection_disposition)
        .def_readonly("trial_decided", &SSVEPStepResult::trial_decided)
        .def_readonly("trial", &SSVEPStepResult::trial)
        .def_readonly("settled", &SSVEPStepResult::settled)
        .def_property_readonly("transitions", [](const SSVEPStepResult& v)
                               { return prefix(v.transitions, v.n_transitions); })
        .def_property_readonly("events", [](const SSVEPStepResult& v)
                               { return prefix(v.events, v.n_events); })
        .def_property_readonly("requests", [](const SSVEPStepResult& v)
                               { return prefix(v.requests, v.n_requests); });
    bind_validate<SSVEPTarget>(module);
    bind_validate<SSVEPConfig>(module);
    bind_validate<SSVEPTrialSchedule>(module);
    bind_validate<SSVEPTrial>(module);
    bind_validate<SSVEPPresentationRequest>(module);
    module.def(
        "prepare_trial",
        [](const SSVEPConfig& config, TrialOrdinal ordinal)
        {
            SSVEPTrialSchedule value{};
            auto schedule_config = config;
            schedule_config.n_trials = std::numeric_limits<TrialOrdinal>::max();
            const auto status = prepare_trial(schedule_config, ordinal, value);
            return std::pair{status, value};
        },
        py::arg("config"), py::arg("ordinal"));
    py::class_<SSVEPMachine>(module, "SSVEPMachine")
        .def(py::init<>())
        .def(
            "start",
            [](SSVEPMachine& m, ParadigmId paradigm, const SSVEPConfig& task,
               ExperimentTimeNs time_ns, TrialOrdinal trials)
            {
                SSVEPStepResult result{};
                if (trials == 0)
                    throw py::value_error("trials must be greater than zero");
                auto config = task;
                config.n_trials = trials;
                const auto status = m.start(paradigm, config, time_ns, result);
                return std::pair{status, result};
            },
            py::arg("paradigm"), py::arg("task"), py::arg("time_ns"), py::kw_only(),
            py::arg("trials"))
        .def(
            "step",
            [](SSVEPMachine& m, ExperimentTimeNs time_ns)
            {
                SSVEPStepResult result{};
                const auto status = m.step(time_ns, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"))
        .def(
            "step",
            [](SSVEPMachine& m, ExperimentTimeNs time_ns, const SelectionEvent& selection)
            {
                SSVEPStepResult result{};
                const auto status = m.step(time_ns, selection, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"), py::arg("selection"))
        .def(
            "stop",
            [](SSVEPMachine& m, ExperimentTimeNs time_ns)
            {
                SSVEPStepResult result{};
                const auto status = m.stop(time_ns, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"))
        .def("reset", &SSVEPMachine::reset)
        .def("snapshot", &SSVEPMachine::snapshot)
        .def("configuration", &SSVEPMachine::configuration)
        .def_property_readonly("paradigm", &SSVEPMachine::paradigm)
        .def_property_readonly("state", &SSVEPMachine::state)
        .def_property_readonly("complete", &SSVEPMachine::complete);
    neurale::bindings::bind_ssvep_session(module);
}
