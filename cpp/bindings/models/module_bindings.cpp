/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_models_decomposition(py::module_& module);
void bind_models_linear_model(py::module_& module);
void bind_models_state_space(py::module_& module);
void bind_models_classification(py::module_& module);
void bind_models_neighbors(py::module_& module);
void bind_models_density(py::module_& module);
void bind_models_alignment(py::module_& module);
void bind_models_manifold(py::module_& module);
#ifdef NEURALE_MODELS_WITH_CUDA
void bind_models_cuda(py::module_& module);
#endif

namespace
{

void bind_models_api(py::module_& module)
{
    auto decomposition =
        module.def_submodule("decomposition", "Native decomposition model kernels.");
    bind_models_decomposition(decomposition);
    auto linear_model =
        module.def_submodule("linear_model", "Native linear and ridge regression kernels.");
    bind_models_linear_model(linear_model);
    auto state_space =
        module.def_submodule("state_space", "Native linear-Gaussian state-space kernels.");
    bind_models_state_space(state_space);
    auto classification =
        module.def_submodule("classification", "Native classification model kernels.");
    bind_models_classification(classification);
    auto neighbors = module.def_submodule("neighbors", "Native nearest-neighbor model kernels.");
    bind_models_neighbors(neighbors);
    auto density = module.def_submodule("density", "Native density estimation model kernels.");
    bind_models_density(density);
    auto alignment = module.def_submodule("alignment", "Native sequence alignment model kernels.");
    bind_models_alignment(alignment);
    auto manifold = module.def_submodule("manifold", "Native manifold projection model kernels.");
    bind_models_manifold(manifold);
#ifdef NEURALE_MODELS_WITH_CUDA
    auto cuda = module.def_submodule("cuda", "CUDA model kernels.");
    bind_models_cuda(cuda);
#endif
}

} // namespace

void bind_models_module(py::module_& module)
{
    auto models = module.def_submodule("models", "Native model kernels for PyNeurale.");
    bind_models_api(models);
}
