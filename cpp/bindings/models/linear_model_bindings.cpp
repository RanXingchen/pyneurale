/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/linear_model.h>

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
using neurale::bindings::models::require_coef_intercept;
using neurale::bindings::models::require_no_overlap;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;
using neurale::bindings::models::to_vector;

py::dict fit_linear_model(const ConvertibleDoubleArray& X, const ConvertibleDoubleArray& y,
                          double alpha, bool fit_intercept)
{
    require_2d(X, "X");
    require_2d(y, "y");
    if (X.shape(0) != y.shape(0))
    {
        throw py::value_error("X and y must have the same number of samples");
    }

    neurale::models::LinearFitResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::fit_linear_model(const_span(X), to_size(X.shape(0)),
                                                   to_size(X.shape(1)), const_span(y),
                                                   to_size(y.shape(1)), alpha, fit_intercept);
    }

    const auto n_components = static_cast<py::ssize_t>(result.singular_values.size());
    py::dict output;
    py::object model = py::cast(neurale::models::LinearModel({
        result.n_features,
        result.n_outputs,
        std::move(result.coef),
        std::move(result.intercept),
    }));
    output["model"] = model;
    output["n_samples"] = result.n_samples;
    output["rank"] = result.rank;
    output["coef"] = model.attr("coef");
    output["intercept"] = model.attr("intercept");
    output["singular_values"] =
        array_from_vector(std::move(result.singular_values), {n_components});
    return output;
}

/// Rebuild a fitted model from its parameters alone.
///
/// This is the read side of decoder persistence: an artifact stores the
/// coefficients and the intercept, and a load has to produce the *same*
/// predictor rather than a second implementation of the affine map that would
/// agree only to within rounding.
neurale::models::LinearModel linear_model_from_state(const ConvertibleDoubleArray& coef,
                                                     const ConvertibleDoubleArray& intercept)
{
    require_coef_intercept(coef, intercept);
    return neurale::models::LinearModel({
        to_size(coef.shape(1)),
        to_size(coef.shape(0)),
        to_vector(coef),
        to_vector(intercept),
    });
}

StrictDoubleArray model_predict(const neurale::models::LinearModel& model,
                                const StrictDoubleArray& X)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    StrictDoubleArray output = make_strict_double_output({
        static_cast<py::ssize_t>(n_samples),
        static_cast<py::ssize_t>(model.n_outputs()),
    });
    {
        py::gil_scoped_release release;
        model.predict(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

StrictDoubleArray model_predict_into(const neurale::models::LinearModel& model,
                                     const StrictDoubleArray& X, StrictDoubleArray& output)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    require_2d_output_shape(output, X.shape(0), static_cast<py::ssize_t>(model.n_outputs()));
    require_writable(output, "out");
    require_no_overlap(X, output, "out");
    {
        py::gil_scoped_release release;
        model.predict(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

} // namespace

void bind_models_linear_model(py::module_& module)
{
    py::class_<neurale::models::LinearModel>(module, "LinearModel")
        .def_property_readonly("n_features", &neurale::models::LinearModel::n_features)
        .def_property_readonly("n_outputs", &neurale::models::LinearModel::n_outputs)
        .def_property_readonly(
            "coef",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::LinearModel&>();
                return readonly_array_view(model.coef(),
                                           {
                                               static_cast<py::ssize_t>(model.n_outputs()),
                                               static_cast<py::ssize_t>(model.n_features()),
                                           },
                                           self);
            })
        .def_property_readonly(
            "intercept",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::LinearModel&>();
                return readonly_array_view(model.intercept(),
                                           {static_cast<py::ssize_t>(model.n_outputs())}, self);
            })
        .def("predict", &model_predict, py::arg("X").noconvert())
        .def("predict", &model_predict_into, py::arg("X").noconvert(), py::arg("out").noconvert());

    module.def("fit_linear_model", &fit_linear_model, py::arg("X"), py::arg("y"), py::arg("alpha"),
               py::arg("fit_intercept"));
    module.def("linear_model_from_state", &linear_model_from_state, py::arg("coef"),
               py::arg("intercept"));
}
