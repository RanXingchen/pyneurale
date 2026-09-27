/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/density.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>

namespace py = pybind11;

namespace
{

using neurale::bindings::models::const_span;
using neurale::bindings::models::make_strict_double_output;
using neurale::bindings::models::mutable_span;
using neurale::bindings::models::require_1d_output_shape;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::require_aligned;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;
using DensityEvaluator = decltype(&neurale::models::GaussianKdeModel::pdf);

void require_point_features(const StrictDoubleArray& X,
                            const neurale::models::GaussianKdeModel& model)
{
    if (to_size(X.shape(1)) != model.n_features())
    {
        throw py::value_error("X must have the same number of features as fit data");
    }
}

void require_point_matrix(const neurale::models::GaussianKdeModel& model,
                          const StrictDoubleArray& X)
{
    require_2d(X, "X");
    require_aligned(X, "X");
    require_point_features(X, model);
}

StrictDoubleArray evaluate_density(const neurale::models::GaussianKdeModel& model,
                                   const StrictDoubleArray& X, DensityEvaluator evaluate)
{
    require_point_matrix(model, X);
    const auto X_values = const_span(X);
    const auto nX = to_size(X.shape(0));
    StrictDoubleArray output = make_strict_double_output({static_cast<py::ssize_t>(nX)});
    auto output_values = mutable_span(output);
    {
        py::gil_scoped_release release;
        (model.*evaluate)(X_values, nX, output_values);
    }
    return output;
}

StrictDoubleArray evaluate_density_into(const neurale::models::GaussianKdeModel& model,
                                        const StrictDoubleArray& X, StrictDoubleArray& output,
                                        DensityEvaluator evaluate)
{
    require_point_matrix(model, X);
    const auto X_values = const_span(X);
    const auto nX = to_size(X.shape(0));
    require_1d_output_shape(output, X.shape(0));
    require_aligned(output, "out");
    require_writable(output, "out");
    auto output_values = mutable_span(output);
    {
        py::gil_scoped_release release;
        (model.*evaluate)(X_values, nX, output_values);
    }
    return output;
}

neurale::models::GaussianKdeModel fit_gaussian_kde(const StrictDoubleArray& X,
                                                   const std::string& bandwidth_kind,
                                                   double bandwidth_value)
{
    require_2d(X, "X");
    require_aligned(X, "X");
    const auto X_values = const_span(X);
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));

    if (bandwidth_kind == "scott")
    {
        py::gil_scoped_release release;
        return neurale::models::fit_gaussian_kde(X_values, n_samples, n_features,
                                                 neurale::models::BandwidthRule::Scott);
    }
    if (bandwidth_kind == "silverman")
    {
        py::gil_scoped_release release;
        return neurale::models::fit_gaussian_kde(X_values, n_samples, n_features,
                                                 neurale::models::BandwidthRule::Silverman);
    }
    if (bandwidth_kind == "scalar")
    {
        py::gil_scoped_release release;
        return neurale::models::fit_gaussian_kde(X_values, n_samples, n_features, bandwidth_value);
    }
    throw py::value_error("bw_method is invalid");
}

StrictDoubleArray model_pdf(const neurale::models::GaussianKdeModel& model,
                            const StrictDoubleArray& X)
{
    return evaluate_density(model, X, &neurale::models::GaussianKdeModel::pdf);
}

StrictDoubleArray model_pdf_into(const neurale::models::GaussianKdeModel& model,
                                 const StrictDoubleArray& X, StrictDoubleArray& output)
{
    return evaluate_density_into(model, X, output, &neurale::models::GaussianKdeModel::pdf);
}

StrictDoubleArray model_logpdf(const neurale::models::GaussianKdeModel& model,
                               const StrictDoubleArray& X)
{
    return evaluate_density(model, X, &neurale::models::GaussianKdeModel::logpdf);
}

StrictDoubleArray model_logpdf_into(const neurale::models::GaussianKdeModel& model,
                                    const StrictDoubleArray& X, StrictDoubleArray& output)
{
    return evaluate_density_into(model, X, output, &neurale::models::GaussianKdeModel::logpdf);
}

} // namespace

void bind_models_density(py::module_& module)
{
    py::class_<neurale::models::GaussianKdeModel>(module, "GaussianKdeModel")
        .def_property_readonly("n_samples", &neurale::models::GaussianKdeModel::n_samples)
        .def_property_readonly("n_features", &neurale::models::GaussianKdeModel::n_features)
        .def_property_readonly("bandwidth_factor",
                               &neurale::models::GaussianKdeModel::bandwidth_factor)
        .def("pdf", &model_pdf, py::arg("X").noconvert())
        .def("pdf", &model_pdf_into, py::arg("X").noconvert(), py::arg("out").noconvert())
        .def("logpdf", &model_logpdf, py::arg("X").noconvert())
        .def("logpdf", &model_logpdf_into, py::arg("X").noconvert(), py::arg("out").noconvert());

    module.def("fit_gaussian_kde", &fit_gaussian_kde, py::arg("X").noconvert(),
               py::arg("bandwidth_kind"), py::arg("bandwidth_value"));
}
