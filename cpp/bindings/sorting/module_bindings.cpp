/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_sorting_detection(py::module_& module);
void bind_sorting_online_detection(py::module_& module);
void bind_sorting_valley_seeking(py::module_& module);

namespace
{

void bind_sorting_api(py::module_& module)
{
    auto detection = module.def_submodule("detection", "Native threshold spike-detection kernels.");
    bind_sorting_detection(detection);
    auto online_detection =
        module.def_submodule("online_detection", "Native fixed-capacity spike-block decoding.");
    bind_sorting_online_detection(online_detection);
    auto valley_seeking =
        module.def_submodule("valley_seeking", "Native Valley Seeking clustering kernels.");
    bind_sorting_valley_seeking(valley_seeking);
}

} // namespace

void bind_sorting_module(py::module_& module)
{
    auto sorting = module.def_submodule("sorting", "Native spike-sorting kernels.");
    bind_sorting_api(sorting);
}
