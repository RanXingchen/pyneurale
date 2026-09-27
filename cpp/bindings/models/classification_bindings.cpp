/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/classification.h>

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
using neurale::bindings::models::require_1d_output_shape;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::require_2d_output_shape;
using neurale::bindings::models::require_coef_intercept;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::SizeArray;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;
using neurale::bindings::models::to_vector;

py::dict fit_lda(const ConvertibleDoubleArray& X, const SizeArray& y, std::size_t n_classes,
                 const std::string& shrinkage_kind, double shrinkage_value)
{
    if (X.ndim() != 2 || y.ndim() != 1)
    {
        throw py::value_error("LDA fit inputs have invalid dimensions");
    }
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));
    neurale::models::LdaFitResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::fit_lda(const_span(X), const_span(y), n_samples, n_features,
                                          n_classes, shrinkage_kind, shrinkage_value);
    }

    py::dict output;
    py::object model = py::cast(neurale::models::LdaModel({
        result.n_features,
        result.n_classes,
        result.n_outputs,
        std::move(result.coef),
        std::move(result.intercept),
    }));
    output["model"] = model;
    output["priors"] =
        array_from_vector(std::move(result.priors), {static_cast<py::ssize_t>(result.n_classes)});
    output["means"] =
        array_from_vector(std::move(result.means), {
                                                       static_cast<py::ssize_t>(result.n_classes),
                                                       static_cast<py::ssize_t>(result.n_features),
                                                   });
    output["covariance"] = array_from_vector(std::move(result.covariance),
                                             {
                                                 static_cast<py::ssize_t>(result.n_features),
                                                 static_cast<py::ssize_t>(result.n_features),
                                             });
    output["coef"] = model.attr("coef");
    output["intercept"] = model.attr("intercept");
    return output;
}

/// Rebuild a fitted discriminant from its parameters alone.
///
/// The read side of decoder persistence; see ``linear_model_from_state``.
/// ``n_classes`` is carried separately because the binary case is fitted with a
/// single discriminant: ``n_outputs`` is one there while ``n_classes`` is two,
/// and it is ``n_classes`` that fixes the width of ``predict_proba``.
neurale::models::LdaModel lda_model_from_state(const ConvertibleDoubleArray& coef,
                                               const ConvertibleDoubleArray& intercept,
                                               std::size_t n_classes)
{
    require_coef_intercept(coef, intercept);
    if (n_classes < 2)
    {
        throw py::value_error("n_classes must be at least 2");
    }
    const auto n_outputs = to_size(coef.shape(0));
    const auto expected = n_classes == 2 ? std::size_t{1} : n_classes;
    if (n_outputs != expected)
    {
        throw py::value_error("coef must have one discriminant per class, or one for two classes");
    }
    return neurale::models::LdaModel({
        to_size(coef.shape(1)),
        n_classes,
        n_outputs,
        to_vector(coef),
        to_vector(intercept),
    });
}

StrictDoubleArray model_decision_function(const neurale::models::LdaModel& model,
                                          const StrictDoubleArray& X)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    StrictDoubleArray output =
        model.n_outputs() == 1 ? make_strict_double_output({static_cast<py::ssize_t>(n_samples)})
                               : make_strict_double_output({
                                     static_cast<py::ssize_t>(n_samples),
                                     static_cast<py::ssize_t>(model.n_outputs()),
                                 });
    {
        py::gil_scoped_release release;
        model.decision_function(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

StrictDoubleArray model_decision_function_into(const neurale::models::LdaModel& model,
                                               const StrictDoubleArray& X,
                                               StrictDoubleArray& output)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    if (model.n_outputs() == 1)
    {
        require_1d_output_shape(output, X.shape(0));
    }
    else
    {
        require_2d_output_shape(output, X.shape(0), static_cast<py::ssize_t>(model.n_outputs()));
    }
    require_writable(output, "out");
    {
        py::gil_scoped_release release;
        model.decision_function(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

StrictDoubleArray model_predict_proba(const neurale::models::LdaModel& model,
                                      const StrictDoubleArray& X)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    const auto n_classes = model.n_outputs() == 1 ? 2 : model.n_outputs();
    StrictDoubleArray output = make_strict_double_output({
        static_cast<py::ssize_t>(n_samples),
        static_cast<py::ssize_t>(n_classes),
    });
    {
        py::gil_scoped_release release;
        model.predict_proba(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

StrictDoubleArray model_predict_proba_into(const neurale::models::LdaModel& model,
                                           const StrictDoubleArray& X, StrictDoubleArray& output)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    const auto n_classes = model.n_outputs() == 1 ? 2 : model.n_outputs();
    require_2d_output_shape(output, X.shape(0), static_cast<py::ssize_t>(n_classes));
    require_writable(output, "out");
    {
        py::gil_scoped_release release;
        model.predict_proba(const_span(X), n_samples, mutable_span(output));
    }
    return output;
}

} // namespace

void bind_models_classification(py::module_& module)
{
    py::class_<neurale::models::LdaModel>(module, "LdaModel")
        .def_property_readonly("n_features", &neurale::models::LdaModel::n_features)
        .def_property_readonly("n_classes", &neurale::models::LdaModel::n_classes)
        .def_property_readonly("n_outputs", &neurale::models::LdaModel::n_outputs)
        .def_property_readonly(
            "coef",
            [](py::object self)
            {
                const auto& model = self.cast<const neurale::models::LdaModel&>();
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
                const auto& model = self.cast<const neurale::models::LdaModel&>();
                return readonly_array_view(model.intercept(),
                                           {static_cast<py::ssize_t>(model.n_outputs())}, self);
            })
        .def("decision_function", &model_decision_function, py::arg("X").noconvert())
        .def("decision_function", &model_decision_function_into, py::arg("X").noconvert(),
             py::arg("out").noconvert())
        .def("predict_proba", &model_predict_proba, py::arg("X").noconvert())
        .def("predict_proba", &model_predict_proba_into, py::arg("X").noconvert(),
             py::arg("out").noconvert());

    module.def("fit_lda", &fit_lda, py::arg("X"), py::arg("y"), py::arg("n_classes"),
               py::arg("shrinkage_kind"), py::arg("shrinkage_value"));
    module.def("lda_model_from_state", &lda_model_from_state, py::arg("coef"), py::arg("intercept"),
               py::arg("n_classes"));
}
