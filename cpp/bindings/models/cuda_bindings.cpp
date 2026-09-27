/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "manifold_support.h"
#include "support.h"

#include <neurale/models/cuda/density.h>
#include <neurale/models/cuda/errors.h>
#include <neurale/models/cuda/manifold.h>
#include <neurale/models/cuda/neighbors.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

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
using neurale::bindings::models::require_1d_output_shape;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::require_aligned;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;

using IndexArray = py::array_t<std::int64_t, py::array::c_style>;
using CudaKdeModel = neurale::models::cuda::GaussianKdeModel;
using CudaDensityEvaluator = void (CudaKdeModel::*)(std::span<const double>, std::size_t,
                                                    std::span<double>);

py::tuple knn(const ConvertibleDoubleArray& X, std::size_t k, const std::string& metric,
              bool include_self, int device_id)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));
    const auto metric_value = parse_distance_metric(metric);
    if (k > neurale::models::cuda::max_knn_neighbors)
    {
        throw py::value_error("CUDA KNN supports at most 64 neighbors");
    }
    const auto max_size = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (n_samples > max_size || k > max_size || (n_samples != 0 && k > max_size / n_samples))
    {
        throw py::value_error("KNN output shape is too large");
    }
    IndexArray indices({
        static_cast<py::ssize_t>(n_samples),
        static_cast<py::ssize_t>(k),
    });
    StrictDoubleArray distances({
        static_cast<py::ssize_t>(n_samples),
        static_cast<py::ssize_t>(k),
    });
    {
        py::gil_scoped_release release;
        neurale::models::cuda::knn(const_span(X), n_samples, n_features, k, metric_value,
                                   include_self, device_id, mutable_span(indices),
                                   mutable_span(distances));
    }
    return py::make_tuple(indices, distances);
}

py::dict fit_lpp(const ConvertibleDoubleArray& X, std::size_t n_components, std::size_t n_neighbors,
                 const std::string& metric, const std::string& method, int device_id)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));
    const auto metric_value = parse_distance_metric(metric);
    const auto method_value = parse_lpp_method(method);
    neurale::models::LppFitResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::cuda::fit_lpp(const_span(X), n_samples, n_features, n_components,
                                                n_neighbors, metric_value, method_value, device_id);
    }
    return make_lpp_fit_output(std::move(result));
}

void require_point_matrix(const CudaKdeModel& model, const StrictDoubleArray& X)
{
    require_2d(X, "X");
    require_aligned(X, "X");
    if (to_size(X.shape(1)) != model.n_features())
    {
        throw py::value_error("X must have the same number of features as fit data");
    }
}

StrictDoubleArray evaluate_density(CudaKdeModel& model, const StrictDoubleArray& X,
                                   CudaDensityEvaluator evaluate)
{
    require_point_matrix(model, X);
    const auto nX = to_size(X.shape(0));
    StrictDoubleArray output = make_strict_double_output({static_cast<py::ssize_t>(nX)});
    {
        py::gil_scoped_release release;
        (model.*evaluate)(const_span(X), nX, mutable_span(output));
    }
    return output;
}

StrictDoubleArray evaluate_density_into(CudaKdeModel& model, const StrictDoubleArray& X,
                                        StrictDoubleArray& output, CudaDensityEvaluator evaluate)
{
    require_point_matrix(model, X);
    const auto nX = to_size(X.shape(0));
    require_1d_output_shape(output, X.shape(0));
    require_aligned(output, "out");
    require_writable(output, "out");
    {
        py::gil_scoped_release release;
        (model.*evaluate)(const_span(X), nX, mutable_span(output));
    }
    return output;
}

CudaKdeModel fit_gaussian_kde(const StrictDoubleArray& X, const std::string& bandwidth_kind,
                              double bandwidth_value, int device_id)
{
    require_2d(X, "X");
    require_aligned(X, "X");
    const auto X_values = const_span(X);
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));
    if (bandwidth_kind == "scott")
    {
        py::gil_scoped_release release;
        return neurale::models::cuda::fit_gaussian_kde(
            X_values, n_samples, n_features, neurale::models::BandwidthRule::Scott, device_id);
    }
    if (bandwidth_kind == "silverman")
    {
        py::gil_scoped_release release;
        return neurale::models::cuda::fit_gaussian_kde(
            X_values, n_samples, n_features, neurale::models::BandwidthRule::Silverman, device_id);
    }
    if (bandwidth_kind == "scalar")
    {
        py::gil_scoped_release release;
        return neurale::models::cuda::fit_gaussian_kde(X_values, n_samples, n_features,
                                                       bandwidth_value, device_id);
    }
    throw py::value_error("bw_method is invalid");
}

StrictDoubleArray model_pdf(CudaKdeModel& model, const StrictDoubleArray& X)
{
    return evaluate_density(model, X, &CudaKdeModel::pdf);
}

StrictDoubleArray model_pdf_into(CudaKdeModel& model, const StrictDoubleArray& X,
                                 StrictDoubleArray& output)
{
    return evaluate_density_into(model, X, output, &CudaKdeModel::pdf);
}

StrictDoubleArray model_logpdf(CudaKdeModel& model, const StrictDoubleArray& X)
{
    return evaluate_density(model, X, &CudaKdeModel::logpdf);
}

StrictDoubleArray model_logpdf_into(CudaKdeModel& model, const StrictDoubleArray& X,
                                    StrictDoubleArray& output)
{
    return evaluate_density_into(model, X, output, &CudaKdeModel::logpdf);
}

void bind_cuda_density(py::module_& module)
{
    py::class_<CudaKdeModel>(module, "GaussianKdeModel")
        .def_property_readonly("n_samples", &CudaKdeModel::n_samples)
        .def_property_readonly("n_features", &CudaKdeModel::n_features)
        .def_property_readonly("bandwidth_factor", &CudaKdeModel::bandwidth_factor)
        .def("pdf", &model_pdf, py::arg("X").noconvert())
        .def("pdf", &model_pdf_into, py::arg("X").noconvert(), py::arg("out").noconvert())
        .def("logpdf", &model_logpdf, py::arg("X").noconvert())
        .def("logpdf", &model_logpdf_into, py::arg("X").noconvert(), py::arg("out").noconvert());
    module.def("fit_gaussian_kde", &fit_gaussian_kde, py::arg("X").noconvert(),
               py::arg("bandwidth_kind"), py::arg("bandwidth_value"), py::arg("device_id"));
}

} // namespace

void bind_models_cuda(py::module_& module)
{
    py::register_exception<neurale::models::cuda::DeviceUnavailableError>(
        module, "DeviceUnavailableError", PyExc_RuntimeError);

    auto neighbors = module.def_submodule("neighbors", "CUDA nearest-neighbor kernels.");
    neighbors.def("knn", &knn, py::arg("X"), py::arg("k"), py::arg("metric"),
                  py::arg("include_self"), py::arg("device_id"));

    auto density = module.def_submodule("density", "CUDA density estimation kernels.");
    bind_cuda_density(density);

    auto manifold =
        module.def_submodule("manifold", "CUDA-accelerated manifold projection kernels.");
    manifold.def("fit_lpp", &fit_lpp, py::arg("X"), py::arg("n_components"), py::arg("n_neighbors"),
                 py::arg("metric"), py::arg("proj_method"), py::arg("device_id"));
}
