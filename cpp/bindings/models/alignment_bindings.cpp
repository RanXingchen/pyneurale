/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/alignment.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace
{

using neurale::bindings::models::const_span;
using neurale::bindings::models::ConvertibleDoubleArray;
using neurale::bindings::models::to_size;

py::ssize_t checked_ssize(std::size_t value, const char* message)
{
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (value > maximum)
    {
        throw std::overflow_error(message);
    }
    return static_cast<py::ssize_t>(value);
}

py::array_t<py::ssize_t, py::array::c_style> path_array(const std::vector<std::size_t>& values)
{
    py::array_t<py::ssize_t, py::array::c_style> output(
        {checked_ssize(values.size(), "DTW path is too large")});
    auto* data = output.mutable_data();
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        data[i] = checked_ssize(values[i], "DTW path index is too large");
    }
    return output;
}

py::dict dtw(const ConvertibleDoubleArray& x, const ConvertibleDoubleArray& y,
             std::optional<std::size_t> radius, const std::string& metric)
{
    if (x.ndim() != 2 || y.ndim() != 2)
    {
        throw py::value_error("x and y must be 2D");
    }
    const auto x_span = const_span(x);
    const auto y_span = const_span(y);
    const auto n_x = to_size(x.shape(0));
    const auto x_dims = to_size(x.shape(1));
    const auto n_y = to_size(y.shape(0));
    const auto y_dims = to_size(y.shape(1));

    neurale::models::DtwResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::dtw(x_span, n_x, x_dims, y_span, n_y, y_dims, radius, metric);
    }
    py::dict output;
    output["cost"] = result.cost;
    output["path_x"] = path_array(result.path_x);
    output["path_y"] = path_array(result.path_y);
    return output;
}

} // namespace

void bind_models_alignment(py::module_& module)
{
    module.def("dtw", &dtw, py::arg("x"), py::arg("y"), py::arg("radius"), py::arg("metric"));
}
