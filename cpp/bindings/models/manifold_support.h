/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/manifold.h>

#include <pybind11/pybind11.h>

#include <string>
#include <utility>

namespace py = pybind11;

namespace neurale::bindings::models
{

inline neurale::models::DistanceMetric parse_distance_metric(const std::string& metric)
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

inline neurale::models::LppProjectionMethod parse_lpp_method(const std::string& method)
{
    if (method == "OPP")
    {
        return neurale::models::LppProjectionMethod::Opp;
    }
    if (method == "LPP")
    {
        return neurale::models::LppProjectionMethod::Lpp;
    }
    throw py::value_error("proj_method is invalid");
}

inline py::dict make_lpp_fit_output(neurale::models::LppFitResult result)
{
    py::dict output;
    py::object model = py::cast(neurale::models::LppModel({
        result.n_features,
        result.n_components,
        std::move(result.components),
    }));
    output["model"] = model;
    output["n_samples"] = result.n_samples;
    output["components"] = model.attr("components");
    return output;
}

} // namespace neurale::bindings::models
