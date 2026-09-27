/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/devices/simulation.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace py = pybind11;
void bind_device_provider(py::module_& devices);

void bind_devices_module(py::module_& module)
{
    using neurale::devices::simulation::AcquisitionEvent;
    using neurale::devices::simulation::AcquisitionEventKind;
    using neurale::devices::simulation::AppliedNeuralControl;
    using neurale::devices::simulation::IntentDrivenNeuralSource;
    using neurale::devices::simulation::ManualHostClock;
    using neurale::devices::simulation::NeuralDriftSchedule;
    using neurale::devices::simulation::SimulatedNeuralSource;
    using neurale::devices::simulation::SimulatedNeuralSourceConfig;
    using neurale::signal::simulation::NeuralSignalConfig;
    using neurale::signal::simulation::SignalGenerator;
    using namespace neurale::streaming;

    auto devices = module.def_submodule("devices", "Native concrete device adapters.");
    bind_device_provider(devices);
    auto simulation = devices.def_submodule("simulation", "Native simulated acquisition devices.");

    py::enum_<AcquisitionEventKind>(simulation, "AcquisitionEventKind")
        .value("SAMPLE_LOSS", AcquisitionEventKind::sample_loss)
        .value("WOULD_BLOCK", AcquisitionEventKind::would_block)
        .value("STALL", AcquisitionEventKind::stall)
        .value("DISCONNECT", AcquisitionEventKind::disconnect)
        .value("SOURCE_FAULT", AcquisitionEventKind::source_fault)
        .value("DEVICE_TICK_JUMP", AcquisitionEventKind::device_tick_jump)
        .value("DEVICE_RESTART", AcquisitionEventKind::device_restart);

    py::class_<AcquisitionEvent>(simulation, "AcquisitionEvent")
        .def(py::init<>())
        .def_readwrite("frame_ordinal", &AcquisitionEvent::frame_ordinal)
        .def_readwrite("kind", &AcquisitionEvent::kind)
        .def_readwrite("n_samples", &AcquisitionEvent::n_samples)
        .def_readwrite("duration_ns", &AcquisitionEvent::duration_ns)
        .def_readwrite("device_tick_delta", &AcquisitionEvent::device_tick_delta)
        .def_readwrite("restart_tick", &AcquisitionEvent::restart_tick);

    py::class_<ManualHostClock, std::shared_ptr<ManualHostClock>>(simulation, "ManualHostClock",
                                                                  py::is_final())
        .def(py::init<HostTimeNs>(), py::arg("initial_time_ns") = 0)
        .def_property_readonly("now_ns", &ManualHostClock::now_ns)
        .def("set", &ManualHostClock::set, py::arg("time_ns"))
        .def("advance", &ManualHostClock::advance, py::arg("duration_ns"));

    py::class_<SimulatedNeuralSource, NativeFrameSource>(simulation, "SimulatedNeuralSource",
                                                         py::is_final())
        .def(py::init(
                 [](const SignalGenerator& generator, std::uint32_t samples_per_frame,
                    RationalRate fs, std::optional<std::uint64_t> total_sample_count,
                    SampleIndex initial_sample_idx, SignalDType sample_dtype,
                    PhysicalUnit physical_unit, bool device_ticks, DeviceTick initial_device_tick,
                    std::int64_t clock_offset_ns, std::int64_t clock_drift_ppm,
                    HostTimeNs clock_sync_uncertainty_ns, bool paced,
                    const std::vector<AcquisitionEvent>& events,
                    const std::vector<std::string>& channel_names,
                    const std::vector<double>& channel_impedances_ohm,
                    std::shared_ptr<ManualHostClock> clock)
                 {
                     SimulatedNeuralSourceConfig config{
                         .session_id = 0,
                         .schema_id = 1,
                         .signal_id = 1,
                         .clock_domain = 1,
                         .nominal_samples_per_frame = samples_per_frame,
                         .max_samples_per_frame = samples_per_frame,
                         .fs = fs,
                         .sample_dtype = sample_dtype,
                         .physical_unit = physical_unit,
                         .channel_set_id = 1,
                         .channel_names = channel_names,
                         .channel_impedances_ohm = channel_impedances_ohm,
                         .calibration_id = 0,
                         .reference_id = 0,
                         .initial_sample_idx = initial_sample_idx,
                         .total_sample_count = total_sample_count,
                         .device_ticks = device_ticks,
                         .initial_device_tick = initial_device_tick,
                         .clock_offset_ns = clock_offset_ns,
                         .clock_drift_ppm = clock_drift_ppm,
                         .clock_sync_uncertainty_ns = clock_sync_uncertainty_ns,
                         .paced = paced,
                         .events = events,
                     };
                     if (clock)
                         return std::make_unique<SimulatedNeuralSource>(
                             generator, std::move(config), std::move(clock));
                     return std::make_unique<SimulatedNeuralSource>(generator, std::move(config));
                 }),
             py::arg("generator"), py::arg("samples_per_frame"), py::arg("fs"),
             py::arg("total_sample_count") = py::none(), py::arg("initial_sample_idx") = 0,
             py::arg("sample_dtype") = SignalDType::float64,
             py::arg("physical_unit") = PhysicalUnit::volts, py::arg("device_ticks") = true,
             py::arg("initial_device_tick") = 0, py::arg("clock_offset_ns") = 0,
             py::arg("clock_drift_ppm") = 0, py::arg("clock_sync_uncertainty_ns") = 0,
             py::arg("paced") = true, py::arg("events") = std::vector<AcquisitionEvent>{},
             py::arg("channel_names") = std::vector<std::string>{},
             py::arg("channel_impedances_ohm") = std::vector<double>{},
             py::arg("clock") = py::none())
        .def_property_readonly("schema", [](const SimulatedNeuralSource& source)
                               { return source.schema().clone(); })
        .def_property_readonly("_session_id", &SimulatedNeuralSource::session_id)
        .def_property_readonly("frames_emitted", &SimulatedNeuralSource::frames_emitted)
        .def_property_readonly("samples_emitted", &SimulatedNeuralSource::samples_emitted)
        .def_property_readonly("closed", &SimulatedNeuralSource::closed)
        .def("cancel", &SimulatedNeuralSource::cancel)
        .def("close", &SimulatedNeuralSource::close)
        .def("reset", &SimulatedNeuralSource::reset);

    py::class_<NeuralDriftSchedule>(simulation, "NeuralDriftSchedule")
        .def(py::init<>())
        .def_readwrite("start_ordinal", &NeuralDriftSchedule::start_ordinal)
        .def_readwrite("end_ordinal", &NeuralDriftSchedule::end_ordinal)
        .def_readwrite("start_progress", &NeuralDriftSchedule::start_progress)
        .def_readwrite("end_progress", &NeuralDriftSchedule::end_progress);

    py::class_<IntentDrivenNeuralSource, SimulatedNeuralSource>(
        simulation, "IntentDrivenNeuralSource", py::is_final())
        .def(py::init(
                 [](NeuralSignalConfig neural_config, std::uint32_t samples_per_frame,
                    RationalRate fs, bool paced, std::optional<NeuralDriftSchedule> drift_schedule,
                    const std::vector<std::string>& channel_names, std::size_t evidence_capacity)
                 {
                     SimulatedNeuralSourceConfig config{
                         .session_id = 0,
                         .schema_id = 1,
                         .signal_id = 1,
                         .clock_domain = 1,
                         .nominal_samples_per_frame = samples_per_frame,
                         .max_samples_per_frame = samples_per_frame,
                         .fs = fs,
                         .sample_dtype = SignalDType::float64,
                         .physical_unit = PhysicalUnit::volts,
                         .channel_set_id = 1,
                         .channel_names = channel_names,
                         .device_ticks = true,
                         .paced = paced,
                     };
                     return std::make_unique<IntentDrivenNeuralSource>(
                         std::move(neural_config), std::move(config), drift_schedule,
                         evidence_capacity);
                 }),
             py::arg("generator_config"), py::arg("samples_per_frame"), py::arg("fs"),
             py::arg("paced") = true, py::arg("drift_schedule") = py::none(),
             py::arg("channel_names") = std::vector<std::string>{},
             py::arg("evidence_capacity") = 4'096)
        .def(
            "bind_intent_source",
            [](IntentDrivenNeuralSource& source, py::capsule capsule)
            {
                if (!PyCapsule_IsValid(capsule.ptr(),
                                       neurale::streaming::kNeuralIntentSourceCapsule))
                    return StreamStatus::invalid_frame;
                auto* value =
                    static_cast<std::shared_ptr<NeuralIntentSource>*>(PyCapsule_GetPointer(
                        capsule.ptr(), neurale::streaming::kNeuralIntentSourceCapsule));
                if (value == nullptr)
                    return StreamStatus::invalid_frame;
                return source.bind_intent_source(*value);
            },
            py::arg("source"))
        .def("publish_control", &IntentDrivenNeuralSource::publish_control, py::arg("intent_x"),
             py::arg("intent_y"), py::arg("drift_progress"), py::arg("context_ordinal"))
        .def("pop_applied_control",
             [](IntentDrivenNeuralSource& source) -> py::object
             {
                 AppliedNeuralControl control{};
                 if (source.try_pop_applied_control(control) != StreamStatus::ok)
                     return py::none();
                 py::dict value;
                 value["intent_sequence"] = control.intent_sequence;
                 value["intent_x"] = control.intent_x;
                 value["intent_y"] = control.intent_y;
                 value["context_ordinal"] = control.context_ordinal;
                 value["intent_valid"] = control.intent_valid;
                 value["drift_progress"] = control.drift_progress;
                 value["first_sample_index"] = control.first_sample_index;
                 value["last_sample_index"] = control.last_sample_index;
                 value["generated_frame_sequence"] = control.generated_frame_sequence;
                 return value;
             })
        .def_property_readonly("dropped_applied_control_count",
                               &IntentDrivenNeuralSource::dropped_applied_control_count)
        .def_property_readonly("drift_fingerprint", &IntentDrivenNeuralSource::drift_fingerprint);
}
