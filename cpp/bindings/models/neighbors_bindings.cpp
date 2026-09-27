/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/neighbors.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace py = pybind11;

namespace
{

using neurale::bindings::models::const_span;
using neurale::bindings::models::ConvertibleDoubleArray;
using neurale::bindings::models::mutable_span;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;

using IndexArray = py::array_t<std::int64_t, py::array::c_style>;

neurale::models::DistanceMetric distance_metric(const std::string& metric)
{
    if (metric == "sqeuclidean")
    {
        return neurale::models::DistanceMetric::SquaredEuclidean;
    }
    if (metric == "euclidean")
    {
        return neurale::models::DistanceMetric::Euclidean;
    }
    throw py::value_error("metric is invalid");
}

void require_native_knn_shape(std::size_t n_samples, std::size_t n_features, std::size_t k,
                              bool include_self)
{
    if (n_samples == 0 || n_features == 0)
    {
        throw py::value_error("X shape is invalid");
    }
    if (k == 0)
    {
        throw py::value_error("k must be positive");
    }
    const std::size_t max_neighbors = include_self ? n_samples : n_samples - 1;
    if (k > max_neighbors)
    {
        throw py::value_error("k exceeds the available neighbor count");
    }
    const auto max_size = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (n_samples > max_size || k > max_size || k > max_size / n_samples)
    {
        throw py::value_error("KNN output shape is too large");
    }
}

void require_finite_input(const ConvertibleDoubleArray& X)
{
    const auto values = const_span(X);
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw py::value_error("X must contain finite values");
        }
    }
}

py::tuple knn(const ConvertibleDoubleArray& X, std::size_t k, const std::string& metric,
              bool include_self)
{
    require_2d(X, "X");
    const auto n_samples = to_size(X.shape(0));
    const auto n_features = to_size(X.shape(1));
    const auto metric_value = distance_metric(metric);
    require_native_knn_shape(n_samples, n_features, k, include_self);
    require_finite_input(X);
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
        neurale::models::knn(const_span(X), n_samples, n_features, k, metric_value, include_self,
                             mutable_span(indices), mutable_span(distances));
    }
    return py::make_tuple(indices, distances);
}

} // namespace

void bind_models_neighbors(py::module_& module)
{
    module.def("knn", &knn, py::arg("X"), py::arg("k"), py::arg("metric"), py::arg("include_self"));
}
