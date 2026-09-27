/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_signal_windows(py::module_& module);
void bind_signal_filtering(py::module_& module);
void bind_signal_filter_design(py::module_& module);
void bind_signal_spatial(py::module_& module);
void bind_signal_resampling(py::module_& module);
void bind_signal_spectral(py::module_& module);
void bind_signal_transforms(py::module_& module);
void bind_signal_representations(py::module_& module);
void bind_signal_simulation(py::module_& module);

namespace
{

void bind_signal_api(py::module_& module)
{
    auto windows = module.def_submodule("windows", "Native signal window kernels.");
    bind_signal_windows(windows);
    auto filtering = module.def_submodule("filtering", "Native signal filtering kernels.");
    bind_signal_filtering(filtering);
    bind_signal_filter_design(filtering);
    bind_signal_spatial(filtering);
    auto representations =
        module.def_submodule("representations", "Native filter representation conversions.");
    bind_signal_representations(representations);
    auto resampling = module.def_submodule("resampling", "Native polyphase resampling kernels.");
    bind_signal_resampling(resampling);
    auto spectral = module.def_submodule("spectral", "Native spectral transform kernels.");
    bind_signal_spectral(spectral);
    auto transforms =
        module.def_submodule("transforms", "Native frequency-domain transform kernels.");
    bind_signal_transforms(transforms);
    auto simulation =
        module.def_submodule("simulation", "Native deterministic sampled-signal generation.");
    bind_signal_simulation(simulation);
}

} // namespace

void bind_signal_module(py::module_& module)
{
    auto signal = module.def_submodule("signal", "Native signal-processing kernels for PyNeurale.");
    bind_signal_api(signal);
}
