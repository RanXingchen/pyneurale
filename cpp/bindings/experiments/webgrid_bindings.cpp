// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "binding_helpers.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <neurale/experiments/webgrid.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::webgrid;

} // namespace

static void bind_webgrid_machine(py::module_& webgrid)
{
    py::enum_<WebGridState>(webgrid, "WebGridState")
        .value("IDLE", WebGridState::idle)
        .value("ACTIVE_TARGET", WebGridState::active_target)
        .value("COMPLETE", WebGridState::complete);
    py::enum_<WebGridReason>(webgrid, "WebGridReason")
        .value("UNSPECIFIED", WebGridReason::unspecified)
        .value("TARGET_SELECTED", WebGridReason::target_selected);

    webgrid.def("webgrid_state_declared", &webgrid_state_declared, py::arg("state"));

    py::class_<WebGridSelectionRecord>(webgrid, "WebGridSelectionRecord")
        .def(py::init(
                 [](const SelectionEvent& event, ExperimentTimeNs target_onset_ns,
                    DurationNs elapsed_since_target_onset_ns)
                 {
                     return WebGridSelectionRecord{event, target_onset_ns,
                                                   elapsed_since_target_onset_ns};
                 }),
             py::arg("event") = SelectionEvent{}, py::arg("target_onset_ns") = 0,
             py::arg("elapsed_since_target_onset_ns") = 0)
        .def_readonly("event", &WebGridSelectionRecord::event)
        .def_readonly("target_onset_ns", &WebGridSelectionRecord::target_onset_ns)
        .def_readonly("elapsed_since_target_onset_ns",
                      &WebGridSelectionRecord::elapsed_since_target_onset_ns);

    py::class_<WebGridTrial>(webgrid, "WebGridTrial")
        .def(py::init([](const TrialRecord& record, const WebGridSelectionRecord& selection,
                         DurationNs acquisition_ns)
                      { return WebGridTrial{record, selection, acquisition_ns}; }),
             py::arg("record") = TrialRecord{}, py::arg("selection") = WebGridSelectionRecord{},
             py::arg("acquisition_ns") = 0)
        .def_readonly("record", &WebGridTrial::record)
        .def_readonly("selection", &WebGridTrial::selection)
        .def_readonly("acquisition_ns", &WebGridTrial::acquisition_ns);

    py::class_<WebGridMetrics>(webgrid, "WebGridMetrics")
        .def_readonly("metric_version", &WebGridMetrics::metric_version)
        .def_readonly("correct_selections", &WebGridMetrics::correct_selections)
        .def_readonly("incorrect_selections", &WebGridMetrics::incorrect_selections)
        .def_readonly("elapsed_active_ns", &WebGridMetrics::elapsed_active_ns)
        .def_readonly("rates_defined", &WebGridMetrics::rates_defined)
        .def_readonly("correct_targets_per_minute", &WebGridMetrics::correct_targets_per_minute)
        .def_readonly("net_correct_targets_per_minute",
                      &WebGridMetrics::net_correct_targets_per_minute)
        .def_readonly("n_acquisitions", &WebGridMetrics::n_acquisitions)
        .def_readonly("total_acquisition_ns", &WebGridMetrics::total_acquisition_ns)
        .def_readonly("minimum_acquisition_ns", &WebGridMetrics::minimum_acquisition_ns)
        .def_readonly("maximum_acquisition_ns", &WebGridMetrics::maximum_acquisition_ns)
        .def_readonly("mean_acquisition_ns", &WebGridMetrics::mean_acquisition_ns);

    py::class_<WebGridSnapshot>(webgrid, "WebGridSnapshot")
        .def_readonly("time_ns", &WebGridSnapshot::time_ns)
        .def_readonly("state", &WebGridSnapshot::state)
        .def_readonly("trial", &WebGridSnapshot::trial)
        .def_readonly("active_target", &WebGridSnapshot::active_target)
        .def_readonly("target_onset_ns", &WebGridSnapshot::target_onset_ns)
        .def_readonly("completed", &WebGridSnapshot::completed)
        .def_readonly("metrics", &WebGridSnapshot::metrics);

    py::class_<WebGridStepResult>(webgrid, "WebGridStepResult")
        .def_readonly("snapshot", &WebGridStepResult::snapshot)
        .def_readonly("selection_processed", &WebGridStepResult::selection_processed)
        .def_readonly("selection", &WebGridStepResult::selection)
        .def_readonly("trial_decided", &WebGridStepResult::trial_decided)
        .def_readonly("trial", &WebGridStepResult::trial);

    webgrid.def(
        "validate", [](const WebGridSelectionRecord& value) { return validate(value); },
        py::arg("value"));
    webgrid.def(
        "validate", [](const WebGridTrial& value) { return validate(value); }, py::arg("value"));
    webgrid.def(
        "summarize",
        [](const std::vector<WebGridSelectionRecord>& selections, ExperimentTimeNs start_ns,
           ExperimentTimeNs end_ns, std::uint16_t metric_version)
        {
            WebGridMetrics metrics{};
            const ContractStatus status =
                summarize(std::span<const WebGridSelectionRecord>(selections), start_ns, end_ns,
                          metric_version, metrics);
            return std::pair{status, metrics};
        },
        py::arg("selections"), py::arg("start_ns"), py::arg("end_ns"), py::arg("metric_version"));

    py::class_<WebGridMachine>(webgrid, "WebGridMachine")
        .def(py::init<>())
        .def(
            "start",
            [](WebGridMachine& machine, ParadigmId paradigm, const WebGridConfig& config,
               ExperimentTimeNs time_ns)
            {
                WebGridStepResult result{};
                const ContractStatus status = machine.start(paradigm, config, time_ns, result);
                return std::pair{status, result};
            },
            py::arg("paradigm"), py::arg("config"), py::arg("time_ns"))
        .def("reset", &WebGridMachine::reset)
        .def(
            "step",
            [](WebGridMachine& machine, ExperimentTimeNs time_ns, const PointerPosition& pointer)
            {
                WebGridStepResult result{};
                const ContractStatus status = machine.step(time_ns, pointer, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"), py::arg("pointer"))
        .def(
            "step",
            [](WebGridMachine& machine, ExperimentTimeNs time_ns, const PointerPosition& pointer,
               const SelectionEvent& selection)
            {
                WebGridStepResult result{};
                const ContractStatus status = machine.step(time_ns, pointer, selection, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"), py::arg("pointer"), py::arg("selection"))
        .def("snapshot", &WebGridMachine::snapshot)
        .def("configuration", &WebGridMachine::configuration)
        .def_property_readonly("paradigm", &WebGridMachine::paradigm)
        .def_property_readonly("state", &WebGridMachine::state)
        .def_property_readonly("complete", &WebGridMachine::complete);
}

namespace
{

using namespace neurale::experiments;
using neurale::bindings::experiments::bind_validate;
using neurale::bindings::experiments::prefix;
using namespace neurale::experiments::webgrid;

template <typename Element, std::size_t Capacity>
std::array<Element, Capacity> to_fixed(const std::vector<Element>& items, const char* what)
{
    if (items.size() > Capacity)
        throw py::value_error(std::string(what) + " holds more than " + std::to_string(Capacity) +
                              " items");
    std::array<Element, Capacity> fixed{};
    for (std::size_t i = 0; i < items.size(); ++i)
        fixed[i] = items[i];
    return fixed;
}

} // namespace

void bind_experiments_webgrid_module(py::module_& experiments)
{
    auto webgrid = experiments.def_submodule(
        "webgrid", "Renderer-independent WebGrid configuration, geometry, target schedule, and "
                   "explicit discrete selection semantics.");

    webgrid.attr("MAX_WEBGRID_CELLS") = kMaxWebGridCells;
    webgrid.attr("TARGET_SELECTION_STREAM") = kTargetSelectionStream;
    webgrid.attr("METRIC_VERSION_1") = kMetricVersion1;
    webgrid.attr("CURRENT_METRIC_VERSION") = kCurrentMetricVersion;

    py::enum_<TargetScheduleKind>(webgrid, "TargetScheduleKind")
        .value("UNSPECIFIED", TargetScheduleKind::unspecified)
        .value("SEEDED", TargetScheduleKind::seeded)
        .value("EXPLICIT_SEQUENCE", TargetScheduleKind::explicit_sequence);
    py::enum_<ImmediateRepetitionPolicy>(webgrid, "ImmediateRepetitionPolicy")
        .value("UNSPECIFIED", ImmediateRepetitionPolicy::unspecified)
        .value("ALLOW", ImmediateRepetitionPolicy::allow)
        .value("FORBID", ImmediateRepetitionPolicy::forbid);
    py::enum_<CorrectSelectionPolicy>(webgrid, "CorrectSelectionPolicy")
        .value("UNSPECIFIED", CorrectSelectionPolicy::unspecified)
        .value("ADVANCE_TARGET", CorrectSelectionPolicy::advance_target);
    py::enum_<IncorrectSelectionPolicy>(webgrid, "IncorrectSelectionPolicy")
        .value("UNSPECIFIED", IncorrectSelectionPolicy::unspecified)
        .value("KEEP_CURRENT_TARGET", IncorrectSelectionPolicy::keep_current_target);

    webgrid.def("target_schedule_kind_declared", &target_schedule_kind_declared, py::arg("value"));
    webgrid.def("immediate_repetition_policy_declared", &immediate_repetition_policy_declared,
                py::arg("value"));
    webgrid.def("correct_selection_policy_declared", &correct_selection_policy_declared,
                py::arg("value"));
    webgrid.def("incorrect_selection_policy_declared", &incorrect_selection_policy_declared,
                py::arg("value"));

    py::class_<TaskBounds>(webgrid, "TaskBounds")
        .def(py::init([](double min_x, double max_x, double min_y, double max_y)
                      { return TaskBounds{min_x, max_x, min_y, max_y}; }),
             py::arg("min_x") = 0.0, py::arg("max_x") = 0.0, py::arg("min_y") = 0.0,
             py::arg("max_y") = 0.0)
        .def_readonly("min_x", &TaskBounds::min_x)
        .def_readonly("max_x", &TaskBounds::max_x)
        .def_readonly("min_y", &TaskBounds::min_y)
        .def_readonly("max_y", &TaskBounds::max_y);

    py::class_<PointerPosition>(webgrid, "PointerPosition")
        .def(py::init([](double x, double y) { return PointerPosition{x, y}; }), py::arg("x") = 0.0,
             py::arg("y") = 0.0)
        .def_readonly("x", &PointerPosition::x)
        .def_readonly("y", &PointerPosition::y);

    py::class_<CellBounds>(webgrid, "CellBounds")
        .def_readonly("min_x", &CellBounds::min_x)
        .def_readonly("max_x", &CellBounds::max_x)
        .def_readonly("min_y", &CellBounds::min_y)
        .def_readonly("max_y", &CellBounds::max_y);

    py::class_<GridCell>(webgrid, "GridCell")
        .def_readonly("id", &GridCell::id)
        .def_readonly("row", &GridCell::row)
        .def_readonly("column", &GridCell::column)
        .def_readonly("bounds", &GridCell::bounds);

    const WebGridConfig defaults{};
    py::class_<WebGridConfig>(webgrid, "WebGridConfig")
        .def(py::init(
                 [](std::uint16_t rows, std::uint16_t columns, const TaskBounds& bounds,
                    const std::vector<TargetId>& candidates, TargetScheduleKind schedule,
                    ImmediateRepetitionPolicy immediate_repetition,
                    CorrectSelectionPolicy correct_selection,
                    IncorrectSelectionPolicy incorrect_selection, ScheduleSeed seed,
                    SamplerVersion sampler_version, const std::vector<TargetId>& explicit_targets,
                    TargetId initial_target, DurationNs session_duration_ns,
                    TrialOrdinal target_count_limit, std::uint16_t metric_version)
                 {
                     WebGridConfig config{};
                     config.rows = rows;
                     config.columns = columns;
                     config.bounds = bounds;
                     config.n_candidates = static_cast<std::uint16_t>(candidates.size());
                     config.candidates =
                         to_fixed<TargetId, kMaxWebGridCells>(candidates, "candidates");
                     config.schedule = schedule;
                     config.immediate_repetition = immediate_repetition;
                     config.correct_selection = correct_selection;
                     config.incorrect_selection = incorrect_selection;
                     config.seed = seed;
                     config.sampler_version = sampler_version;
                     config.n_explicit = static_cast<std::uint16_t>(explicit_targets.size());
                     config.explicit_targets =
                         to_fixed<TargetId, kMaxWebGridCells>(explicit_targets, "explicit_targets");
                     config.initial_target = initial_target;
                     config.session_duration_ns = session_duration_ns;
                     config.target_count_limit = target_count_limit;
                     config.metric_version = metric_version;
                     return config;
                 }),
             py::arg("rows") = 0, py::arg("columns") = 0, py::arg("bounds") = TaskBounds{},
             py::arg("candidates") = std::vector<TargetId>{},
             py::arg("schedule") = TargetScheduleKind::unspecified,
             py::arg("immediate_repetition") = ImmediateRepetitionPolicy::unspecified,
             py::arg("correct_selection") = CorrectSelectionPolicy::unspecified,
             py::arg("incorrect_selection") = IncorrectSelectionPolicy::unspecified,
             py::arg("seed") = 0, py::arg("sampler_version") = defaults.sampler_version,
             py::arg("explicit_targets") = std::vector<TargetId>{},
             py::arg("initial_target") = kUnsetTargetId, py::arg("session_duration_ns") = 0,
             py::arg("target_count_limit") = 0, py::arg("metric_version") = defaults.metric_version)
        .def_readonly("rows", &WebGridConfig::rows)
        .def_readonly("columns", &WebGridConfig::columns)
        .def_readonly("bounds", &WebGridConfig::bounds)
        .def_readonly("n_candidates", &WebGridConfig::n_candidates)
        .def_property_readonly("candidates", [](const WebGridConfig& config)
                               { return prefix(config.candidates, config.n_candidates); })
        .def_readonly("schedule", &WebGridConfig::schedule)
        .def_readonly("immediate_repetition", &WebGridConfig::immediate_repetition)
        .def_readonly("correct_selection", &WebGridConfig::correct_selection)
        .def_readonly("incorrect_selection", &WebGridConfig::incorrect_selection)
        .def_readonly("seed", &WebGridConfig::seed)
        .def_readonly("sampler_version", &WebGridConfig::sampler_version)
        .def_readonly("n_explicit", &WebGridConfig::n_explicit)
        .def_property_readonly("explicit_targets", [](const WebGridConfig& config)
                               { return prefix(config.explicit_targets, config.n_explicit); })
        .def_readonly("initial_target", &WebGridConfig::initial_target)
        .def_readonly("session_duration_ns", &WebGridConfig::session_duration_ns)
        .def_readonly("target_count_limit", &WebGridConfig::target_count_limit)
        .def_readonly("metric_version", &WebGridConfig::metric_version);

    bind_validate<TaskBounds>(webgrid);
    bind_validate<PointerPosition>(webgrid);
    bind_validate<CellBounds>(webgrid);
    bind_validate<GridCell>(webgrid);
    bind_validate<WebGridConfig>(webgrid);

    webgrid.def(
        "grid_cell",
        [](const WebGridConfig& config, std::uint16_t row, std::uint16_t column)
        {
            GridCell cell{};
            return std::pair{grid_cell(config, row, column, cell), cell};
        },
        py::arg("config"), py::arg("row"), py::arg("column"));
    webgrid.def(
        "grid_cell_by_id",
        [](const WebGridConfig& config, TargetId id)
        {
            GridCell cell{};
            return std::pair{grid_cell(config, id, cell), cell};
        },
        py::arg("config"), py::arg("id"));
    webgrid.def(
        "locate_cell",
        [](const WebGridConfig& config, const PointerPosition& pointer)
        {
            TargetId id = kUnsetTargetId;
            return std::pair{locate_cell(config, pointer, id), id};
        },
        py::arg("config"), py::arg("pointer"));
    webgrid.def(
        "locate_selectable_cell",
        [](const WebGridConfig& config, const PointerPosition& pointer)
        {
            TargetId id = kUnsetTargetId;
            return std::pair{locate_selectable_cell(config, pointer, id), id};
        },
        py::arg("config"), py::arg("pointer"));
    webgrid.def("selectable", &selectable, py::arg("config"), py::arg("id"));
    webgrid.def(
        "select_target",
        [](const WebGridConfig& config, TrialOrdinal ordinal, TargetId previous)
        {
            TargetId target = kUnsetTargetId;
            return std::pair{select_target(config, ordinal, previous, target), target};
        },
        py::arg("config"), py::arg("ordinal"), py::arg("previous") = kUnsetTargetId);
    webgrid.def("target_limit_reached", &target_limit_reached, py::arg("config"),
                py::arg("completed"));
    webgrid.def(
        "session_duration_reached",
        [](const WebGridConfig& config, ExperimentTimeNs start_ns, ExperimentTimeNs time_ns)
        {
            bool reached = false;
            return std::pair{session_duration_reached(config, start_ns, time_ns, reached), reached};
        },
        py::arg("config"), py::arg("start_ns"), py::arg("time_ns"));
    webgrid.def(
        "make_selection_event",
        [](const WebGridConfig& config, const PointerPosition& pointer, TargetId intended,
           const TrialIdentity& trial, ParadigmId paradigm, ExperimentTimeNs time_ns,
           SequenceOrdinal sequence)
        {
            SelectionEvent event{};
            return std::pair{make_selection_event(config, pointer, intended, trial, paradigm,
                                                  time_ns, sequence, event),
                             event};
        },
        py::arg("config"), py::arg("pointer"), py::arg("intended"), py::arg("trial"),
        py::arg("paradigm"), py::arg("time_ns"), py::arg("sequence"));
    webgrid.def("configuration_fingerprint", &configuration_fingerprint, py::arg("config"));
    bind_webgrid_machine(webgrid);
}
