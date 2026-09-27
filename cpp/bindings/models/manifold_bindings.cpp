/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "manifold_support.h"
#include "support.h"

#include <neurale/models/manifold.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace
{

using neurale::bindings::models::const_span;
using neurale::bindings::models::ConvertibleDoubleArray;
using neurale::bindings::models::make_lpp_fit_output;
using neurale::bindings::models::make_strict_double_output;
using neurale::bindings::models::mutable_span;
using neurale::bindings::models::parse_distance_metric;
using neurale::bindings::models::parse_lpp_method;
using neurale::bindings::models::readonly_array_view;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::require_2d_output_shape;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;

py::dict fit_lpp(const ConvertibleDoubleArray& X, std::size_t n_components, std::size_t n_neighbors,
                 const std::string& metric, const std::string& method)
{
    require_2d(X, "X");
    const auto metric_value = parse_distance_metric(metric);
    const auto method_value = parse_lpp_method(method);
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));
    neurale::models::LppFitResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::fit_lpp(const_span(X), n_samples, n_features, n_components,
                                          n_neighbors, metric_value, method_value);
    }

    return make_lpp_fit_output(std::move(result));
}

StrictDoubleArray model_transform(const neurale::models::LppModel& model,
                                  const StrictDoubleArray& X)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    StrictDoubleArray output = make_strict_double_output({
        X.shape(0),
        static_cast<py::ssize_t>(model.n_components()),
    });
    {
        py::gil_scoped_release release;
        model.transform(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

StrictDoubleArray model_transform_into(const neurale::models::LppModel& model,
                                       const StrictDoubleArray& X, StrictDoubleArray& output)
{
    require_2d(X, "X");
    require_2d_output_shape(output, X.shape(0), static_cast<py::ssize_t>(model.n_components()));
    require_writable(output, "out");
    {
        py::gil_scoped_release release;
        model.transform(const_span(X), to_size(X.shape(0)), mutable_span(output));
    }
    return output;
}

} // namespace

void bind_models_manifold(py::module_& module)
{
    py::class_<neurale::models::LppModel>(module, "LppModel")
        .def_property_readonly("n_features", &neurale::models::LppModel::n_features)
        .def_property_readonly("n_components", &neurale::models::LppModel::n_components)
        .def_property_readonly(
            "components",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::LppModel&>();
                return readonly_array_view(model.components(),
                                           {
                                               static_cast<py::ssize_t>(model.n_components()),
                                               static_cast<py::ssize_t>(model.n_features()),
                                           },
                                           self);
            })
        .def("transform", &model_transform, py::arg("X").noconvert())
        .def("transform", &model_transform_into, py::arg("X").noconvert(),
             py::arg("out").noconvert());

    module.def("fit_lpp", &fit_lpp, py::arg("X"), py::arg("n_components"), py::arg("n_neighbors"),
               py::arg("metric"), py::arg("proj_method"));
}
