/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_runtime_module(py::module_& module);
void bind_streaming_module(py::module_& module);
void bind_pipeline_module(py::module_& module);
void bind_recording_module(py::module_& module);
void bind_signal_module(py::module_& module);
void bind_devices_module(py::module_& module);
void bind_features_module(py::module_& module);
void bind_models_module(py::module_& module);
void bind_sorting_module(py::module_& module);
void bind_experiments_module(py::module_& module);

PYBIND11_MODULE(_native, module)
{
    module.doc() = "Native runtime, signal-processing, model, and sorting kernels for PyNeurale.";

    bind_runtime_module(module);
    bind_streaming_module(module);
    // The compiled pipeline is a concrete implementation of the already
    // registered generic native processor contract.
    bind_pipeline_module(module);
    // After streaming: the recorder attaches to a `_NativeStreamRunner`, so the
    // runner's type is registered before anything can hand one over.
    bind_recording_module(module);
    bind_signal_module(module);
    // Devices compose registered streaming contracts and signal generators.
    bind_devices_module(module);
    bind_features_module(module);
    bind_models_module(module);
    bind_sorting_module(module);
    // Register domain values before sessions and optional presentation. Session
    // bindings also consume the streaming and recording types registered above.
    bind_experiments_module(module);
}
