/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/neural_simulation.h>
#include <neurale/signal/simulation.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace py = pybind11;

namespace
{

using InputArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Generator = neurale::signal::simulation::SignalGenerator;
using Status = neurale::signal::simulation::GenerationStatus;
using NeuralConfig = neurale::signal::simulation::NeuralSignalConfig;
using NeuralGenerator = neurale::signal::simulation::NeuralSignalGenerator;
using NeuralStatus = neurale::signal::simulation::NeuralGenerationStatus;

[[nodiscard]] std::span<const double> values(const InputArray& arr)
{
    return {arr.data(), static_cast<std::size_t>(arr.size())};
}

void generate_into(const Generator& generator, std::uint64_t start, py::array output)
{
    if (!output.dtype().is(py::dtype::of<double>()))
    {
        throw py::value_error("output dtype must be float64");
    }
    if (output.ndim() != 2)
    {
        throw py::value_error("output must be a 2D sample-major array");
    }
    if ((output.flags() & py::array::c_style) == 0)
    {
        throw py::value_error("output must be C-contiguous");
    }
    if (!output.writeable())
    {
        throw py::value_error("output must be writable");
    }
    if (static_cast<std::size_t>(output.shape(1)) != generator.channel_count())
    {
        throw py::value_error("output channel dimension must match n_channels");
    }

    const auto count = static_cast<std::size_t>(output.shape(0));
    const auto output_span = std::span<double>(static_cast<double*>(output.mutable_data()),
                                               static_cast<std::size_t>(output.size()));
    Status status;
    {
        py::gil_scoped_release release;
        status = generator.generate(start, count, output_span);
    }
    switch (status)
    {
    case Status::ok:
        return;
    case Status::invalid_output:
        throw py::value_error("output shape is invalid");
    case Status::range_overflow:
        throw std::overflow_error("requested absolute sample range overflows uint64");
    case Status::source_exhausted:
        throw py::value_error("requested range exceeds finite supplied samples");
    }
}

void generate_neural_into(NeuralGenerator& generator, double intent_x, double intent_y,
                          double drift_progress, py::array output, py::object spike_output)
{
    if (!output.dtype().is(py::dtype::of<double>()) || output.ndim() != 2 ||
        (output.flags() & py::array::c_style) == 0 || !output.writeable())
        throw py::value_error("output must be a writable C-contiguous 2D float64 array");
    if (static_cast<std::size_t>(output.shape(1)) != generator.channel_count())
        throw py::value_error("output channel dimension must match n_channels");

    std::span<std::uint8_t> spikes;
    py::array spike_array;
    if (!spike_output.is_none())
    {
        spike_array = py::cast<py::array>(spike_output);
        if (!spike_array.dtype().is(py::dtype::of<std::uint8_t>()) || spike_array.ndim() != 2 ||
            (spike_array.flags() & py::array::c_style) == 0 || !spike_array.writeable())
            throw py::value_error("spike_output must be a writable C-contiguous 2D uint8 array");
        if (spike_array.shape(0) != output.shape(0) ||
            static_cast<std::size_t>(spike_array.shape(1)) != generator.unit_count())
            throw py::value_error("spike_output shape must be (n_samples, n_units)");
        spikes = {static_cast<std::uint8_t*>(spike_array.mutable_data()),
                  static_cast<std::size_t>(spike_array.size())};
    }
    const auto count = static_cast<std::size_t>(output.shape(0));
    const auto values = std::span<double>(static_cast<double*>(output.mutable_data()),
                                          static_cast<std::size_t>(output.size()));
    NeuralStatus status;
    {
        py::gil_scoped_release release;
        status = generator.generate(count, {intent_x, intent_y}, drift_progress, values, spikes);
    }
    switch (status)
    {
    case NeuralStatus::ok:
        return;
    case NeuralStatus::invalid_output:
        throw py::value_error("output shape is invalid");
    case NeuralStatus::invalid_spike_output:
        throw py::value_error("spike_output shape is invalid");
    case NeuralStatus::invalid_control:
        throw py::value_error(
            "intent and drift_progress must be finite; drift_progress must be in [0, 1]");
    case NeuralStatus::range_overflow:
        throw std::overflow_error("requested absolute sample range overflows uint64");
    }
}

} // namespace

void bind_signal_simulation(py::module_& module)
{
    py::class_<Generator>(module, "SignalGenerator")
        .def_static("zeros", &Generator::zeros, py::arg("n_channels"), py::arg("fs"))
        .def_static(
            "constant", [](std::size_t n_channels, double fs, const InputArray& x)
            { return Generator::constant(n_channels, fs, values(x)); }, py::arg("n_channels"),
            py::arg("fs"), py::arg("values"))
        .def_static(
            "tones",
            [](std::size_t n_channels, double fs, std::size_t n_tones, const InputArray& freqs,
               const InputArray& amps, const InputArray& phases)
            {
                return Generator::tones(n_channels, fs, n_tones, values(freqs), values(amps),
                                        values(phases));
            },
            py::arg("n_channels"), py::arg("fs"), py::arg("n_tones"), py::arg("freqs"),
            py::arg("amps"), py::arg("phases"))
        .def_static("noise", &Generator::noise, py::arg("n_channels"), py::arg("fs"),
                    py::arg("seed"), py::arg("low"), py::arg("high"))
        .def_static(
            "samples",
            [](std::size_t n_channels, double fs, std::size_t n_samples, const InputArray& x,
               bool repeat)
            { return Generator::samples(n_channels, fs, n_samples, values(x), repeat); },
            py::arg("n_channels"), py::arg("fs"), py::arg("n_samples"), py::arg("x"),
            py::arg("repeat"))
        .def_property_readonly("n_channels", &Generator::channel_count)
        .def_property_readonly("sample_rate", &Generator::sample_rate)
        .def("generate_into", &generate_into, py::arg("start"), py::arg("output"));

    py::class_<NeuralConfig>(module, "NeuralSignalConfig")
        .def(py::init<>())
        .def_readwrite("n_channels", &NeuralConfig::n_channels)
        .def_readwrite("fs", &NeuralConfig::fs)
        .def_readwrite("units_per_channel", &NeuralConfig::units_per_channel)
        .def_readwrite("seed", &NeuralConfig::seed)
        .def_readwrite("baseline_rate_hz", &NeuralConfig::baseline_rate_hz)
        .def_readwrite("intent_rate_gain_hz", &NeuralConfig::intent_rate_gain_hz)
        .def_readwrite("max_rate_hz", &NeuralConfig::max_rate_hz)
        .def_readwrite("refractory_seconds", &NeuralConfig::refractory_seconds)
        .def_readwrite("preferred_direction_jitter_radians",
                       &NeuralConfig::preferred_direction_jitter_radians)
        .def_readwrite("spike_amplitude_volts", &NeuralConfig::spike_amplitude_volts)
        .def_readwrite("spike_duration_seconds", &NeuralConfig::spike_duration_seconds)
        .def_readwrite("spike_spatial_decay_channels", &NeuralConfig::spike_spatial_decay_channels)
        .def_readwrite("lfp_frequency_hz", &NeuralConfig::lfp_frequency_hz)
        .def_readwrite("lfp_damping_time_seconds", &NeuralConfig::lfp_damping_time_seconds)
        .def_readwrite("lfp_std_volts", &NeuralConfig::lfp_std_volts)
        .def_readwrite("lfp_spatial_correlation", &NeuralConfig::lfp_spatial_correlation)
        .def_readwrite("background_noise_std_volts", &NeuralConfig::background_noise_std_volts)
        .def_readwrite("background_time_constant_seconds",
                       &NeuralConfig::background_time_constant_seconds)
        .def_readwrite("background_spatial_correlation",
                       &NeuralConfig::background_spatial_correlation)
        .def_readwrite("nonstationarity_std_fraction", &NeuralConfig::nonstationarity_std_fraction)
        .def_readwrite("nonstationarity_time_constant_seconds",
                       &NeuralConfig::nonstationarity_time_constant_seconds)
        .def_readwrite("rate_correlation", &NeuralConfig::rate_correlation)
        .def_readwrite("drift_rotation_degrees", &NeuralConfig::drift_rotation_degrees)
        .def_readwrite("drift_per_unit_rotation_std_degrees",
                       &NeuralConfig::drift_per_unit_rotation_std_degrees)
        .def_readwrite("drift_per_unit_rotation_limit_degrees",
                       &NeuralConfig::drift_per_unit_rotation_limit_degrees)
        .def_readwrite("drift_tuning_gain_scale", &NeuralConfig::drift_tuning_gain_scale)
        .def_readwrite("drift_baseline_rate_shift_hz", &NeuralConfig::drift_baseline_rate_shift_hz);

    py::class_<NeuralGenerator>(module, "NeuralSignalGenerator")
        .def(py::init<NeuralConfig>(), py::arg("config"))
        .def_property_readonly("n_channels", &NeuralGenerator::channel_count)
        .def_property_readonly("n_units", &NeuralGenerator::unit_count)
        .def_property_readonly("sample_rate", &NeuralGenerator::sample_rate)
        .def_property_readonly("sample_index", &NeuralGenerator::sample_index)
        .def_property_readonly("drift_fingerprint", &NeuralGenerator::drift_fingerprint)
        .def("generate_into", &generate_neural_into, py::arg("intent_x"), py::arg("intent_y"),
             py::arg("drift_progress"), py::arg("output"), py::arg("spike_output") = py::none())
        .def("reset", &NeuralGenerator::reset);
}
