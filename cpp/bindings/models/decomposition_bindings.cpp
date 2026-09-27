/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/decomposition.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>

namespace py = pybind11;

namespace
{

using neurale::bindings::models::array_from_vector;
using neurale::bindings::models::const_span;
using neurale::bindings::models::ConvertibleDoubleArray;
using neurale::bindings::models::make_strict_double_output;
using neurale::bindings::models::mutable_span;
using neurale::bindings::models::readonly_array_view;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::require_2d_output_shape;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;

py::dict fit_pca(const ConvertibleDoubleArray& X, std::size_t n_components, bool center)
{
    require_2d(X, "X");
    neurale::models::PcaFitResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::fit_pca(const_span(X), to_size(X.shape(0)), to_size(X.shape(1)),
                                          n_components, center);
    }

    py::dict output;
    py::object model = py::cast(neurale::models::PcaModel({
        result.n_features,
        result.n_components,
        result.center,
        std::move(result.mean),
        std::move(result.components),
        std::move(result.offset),
    }));
    output["model"] = model;
    output["n_samples"] = result.n_samples;
    output["mean"] = model.attr("mean");
    output["components"] = model.attr("components");
    output["offset"] = model.attr("offset");
    output["singular_values"] = array_from_vector(std::move(result.singular_values),
                                                  {static_cast<py::ssize_t>(result.n_components)});
    output["explained_variance"] = array_from_vector(
        std::move(result.explained_variance), {static_cast<py::ssize_t>(result.n_components)});
    output["explained_variance_ratio"] =
        array_from_vector(std::move(result.explained_variance_ratio),
                          {static_cast<py::ssize_t>(result.n_components)});
    return output;
}

StrictDoubleArray model_transform(const neurale::models::PcaModel& model,
                                  const StrictDoubleArray& X)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    StrictDoubleArray output = make_strict_double_output({
        static_cast<py::ssize_t>(n_samples),
        static_cast<py::ssize_t>(model.n_components()),
    });
    {
        py::gil_scoped_release release;
        model.transform(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

StrictDoubleArray model_transform_into(const neurale::models::PcaModel& model,
                                       const StrictDoubleArray& X, StrictDoubleArray& output)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    require_2d_output_shape(output, X.shape(0), static_cast<py::ssize_t>(model.n_components()));
    require_writable(output, "out");
    {
        py::gil_scoped_release release;
        model.transform(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

} // namespace

void bind_models_decomposition(py::module_& module)
{
    py::class_<neurale::models::PcaModel>(module, "PcaModel")
        .def_property_readonly("n_features", &neurale::models::PcaModel::n_features)
        .def_property_readonly("n_components", &neurale::models::PcaModel::n_components)
        .def_property_readonly("center", &neurale::models::PcaModel::center)
        .def_property_readonly(
            "mean",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::PcaModel&>();
                return readonly_array_view(model.mean(),
                                           {static_cast<py::ssize_t>(model.n_features())}, self);
            })
        .def_property_readonly(
            "components",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::PcaModel&>();
                return readonly_array_view(model.components(),
                                           {
                                               static_cast<py::ssize_t>(model.n_components()),
                                               static_cast<py::ssize_t>(model.n_features()),
                                           },
                                           self);
            })
        .def_property_readonly(
            "offset",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::PcaModel&>();
                return readonly_array_view(model.offset(),
                                           {static_cast<py::ssize_t>(model.n_components())}, self);
            })
        .def("transform", &model_transform, py::arg("X").noconvert())
        .def("transform", &model_transform_into, py::arg("X").noconvert(),
             py::arg("out").noconvert());

    module.def("fit_pca", &fit_pca, py::arg("X"), py::arg("n_components"), py::arg("center"));
}
