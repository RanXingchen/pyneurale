// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_experiments_contract(py::module_& experiments);
void bind_experiments_assistance_module(py::module_& experiments);
void bind_experiments_center_out_module(py::module_& experiments);
void bind_experiments_webgrid_module(py::module_& experiments);
void bind_experiments_speech_module(py::module_& experiments);
void bind_experiments_ssvep_module(py::module_& experiments);
#ifdef NEURALE_WITH_EXPERIMENT_PRESENTATION
void bind_experiments_presentation(py::module_& module);
#endif

void bind_experiments_module(py::module_& module)
{
    auto experiments = module.def_submodule(
        "experiments", "Shared experiment value contract: time, identity, schedule, presentation, "
                       "and command values.");
    bind_experiments_contract(experiments);
    bind_experiments_assistance_module(experiments);
    bind_experiments_center_out_module(experiments);
    bind_experiments_webgrid_module(experiments);
    bind_experiments_speech_module(experiments);
    bind_experiments_ssvep_module(experiments);
#ifdef NEURALE_WITH_EXPERIMENT_PRESENTATION
    auto presentation = experiments.def_submodule("presentation");
    bind_experiments_presentation(presentation);
#endif
}
