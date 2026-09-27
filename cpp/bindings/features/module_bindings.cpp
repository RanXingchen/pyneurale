/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_features_online(py::module_& module);

void bind_features_module(py::module_& module)
{
    auto features =
        module.def_submodule("features", "Native feature extraction kernels for PyNeurale.");
    auto online = features.def_submodule("online", "Stateful online feature processors.");
    bind_features_online(online);
}
