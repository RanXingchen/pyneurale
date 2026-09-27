/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/sorting/valley_seeking.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace py = pybind11;

namespace
{

using neurale::bindings::sorting::index_array;
using neurale::bindings::sorting::numeric_array;

void require_valley_seeking_arrays(const py::array& features, const py::array& labels)
{
    if (features.ndim() != 2)
    {
        throw py::value_error(
            "Valley Seeking features must have shape (n_observations, n_features)");
    }
    if (!features.dtype().is(py::dtype::of<double>()))
    {
        throw py::type_error("Valley Seeking features dtype must be float64");
    }
    if ((features.flags() & py::array::c_style) == 0)
    {
        throw py::value_error("Valley Seeking features must be C-contiguous");
    }
    if (labels.ndim() != 1 || labels.shape(0) != features.shape(0))
    {
        throw py::value_error(
            "Valley Seeking initial_labels must be 1D with one label per observation");
    }
    if (!labels.dtype().is(py::dtype::of<std::int64_t>()))
    {
        throw py::type_error("Valley Seeking initial_labels dtype must be int64");
    }
    if ((labels.flags() & py::array::c_style) == 0)
    {
        throw py::value_error("Valley Seeking initial_labels must be C-contiguous");
    }
}

const char* termination_name(neurale::sorting::ValleySeekingTermination termination)
{
    using Termination = neurale::sorting::ValleySeekingTermination;
    switch (termination)
    {
    case Termination::Empty:
        return "empty";
    case Termination::Converged:
        return "converged";
    case Termination::MaxIterations:
        return "max_iterations";
    }
    throw std::logic_error("unknown Valley Seeking termination");
}

py::dict valley_seeking(const py::array& features, const py::array& initial_labels, double radius,
                        std::size_t max_iterations)
{
    require_valley_seeking_arrays(features, initial_labels);
    const auto n_observations = static_cast<std::size_t>(features.shape(0));
    const auto n_features = static_cast<std::size_t>(features.shape(1));
    const auto* feature_data = static_cast<const double*>(features.data());
    const auto* label_data = static_cast<const std::int64_t*>(initial_labels.data());
    neurale::sorting::ValleySeekingResult result;
    {
        py::gil_scoped_release release;
        result = neurale::sorting::valley_seeking(
            std::span<const double>(feature_data, static_cast<std::size_t>(features.size())),
            n_observations, n_features,
            std::span<const std::int64_t>(label_data,
                                          static_cast<std::size_t>(initial_labels.size())),
            radius, max_iterations);
    }

    py::dict output;
    output["labels"] = numeric_array(result.labels);
    output["neighbor_counts"] =
        index_array(result.neighbor_counts, "Valley Seeking result is too large",
                    "Valley Seeking neighbor count does not fit int64");
    output["label_order"] = numeric_array(result.label_order);
    output["radius"] = result.radius;
    output["n_iterations"] = result.iterations;
    output["converged"] = result.converged;
    output["termination"] = termination_name(result.termination);
    output["neighbor_pairs"] = result.neighbor_pairs;
    output["workspace_bytes"] = result.workspace_bytes;
    return output;
}

} // namespace

void bind_sorting_valley_seeking(py::module_& module)
{
    module.def("_valley_seeking", &valley_seeking, py::arg("features"), py::arg("initial_labels"),
               py::arg("radius"), py::arg("max_iterations") = 1'000);
    module.attr("_valley_seeking_max_observations") =
        neurale::sorting::valley_seeking_max_observations;
    module.attr("_valley_seeking_max_features") = neurale::sorting::valley_seeking_max_features;
    module.attr("_valley_seeking_max_labels") = neurale::sorting::valley_seeking_max_labels;
    module.attr("_valley_seeking_max_neighbor_pairs") =
        neurale::sorting::valley_seeking_max_neighbor_pairs;
}
