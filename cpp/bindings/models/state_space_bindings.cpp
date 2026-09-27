/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/models/state_space.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using neurale::bindings::models::const_span;
using neurale::bindings::models::ConvertibleDoubleArray;
using neurale::bindings::models::mutable_span;
using neurale::bindings::models::readonly_array_view;
using neurale::bindings::models::require_2d;
using neurale::bindings::models::require_shape;
using neurale::bindings::models::require_writable;
using neurale::bindings::models::SizeArray;
using neurale::bindings::models::StrictDoubleArray;
using neurale::bindings::models::to_size;
using neurale::bindings::models::to_vector;
using neurale::models::LinearGaussianModel;

using MaskArray = py::array_t<std::uint8_t, py::array::c_style>;

std::span<const std::uint8_t> mask_span(const MaskArray& arr)
{
    return {arr.data(), static_cast<std::size_t>(arr.size())};
}

std::span<const std::size_t> size_span(const SizeArray& arr)
{
    return {arr.data(), static_cast<std::size_t>(arr.size())};
}

// Every fitted parameter is exposed the same way -- a read-only view onto one
// member of the model state, sized by one or two of the model's two dimensions
// -- so a property is built from the member and the axes rather than written
// out eight times.
using StateMember = std::vector<double> neurale::models::LinearGaussianModelState::*;

enum class Axis : std::uint8_t
{
    state,
    observation,
};

std::size_t extent(const LinearGaussianModel& model, Axis axis)
{
    return axis == Axis::state ? model.state_dim() : model.observation_dim();
}

py::cpp_function vector_property(StateMember member, Axis size)
{
    return py::cpp_function(
        [member, size](py::object self)
        {
            const auto& model = self.cast<const LinearGaussianModel&>();
            return readonly_array_view(model.parameters().*member,
                                       {static_cast<py::ssize_t>(extent(model, size))}, self);
        });
}

py::cpp_function matrix_property(StateMember member, Axis rows, Axis cols)
{
    return py::cpp_function(
        [member, rows, cols](py::object self)
        {
            const auto& model = self.cast<const LinearGaussianModel&>();
            return readonly_array_view(model.parameters().*member,
                                       {
                                           static_cast<py::ssize_t>(extent(model, rows)),
                                           static_cast<py::ssize_t>(extent(model, cols)),
                                       },
                                       self);
        });
}

py::dict fit_linear_gaussian(const ConvertibleDoubleArray& states,
                             const ConvertibleDoubleArray& observations,
                             const SizeArray& segment_lengths, bool fit_offsets, double jitter)
{
    require_2d(states, "states");
    require_2d(observations, "observations");
    if (states.shape(0) != observations.shape(0))
    {
        throw py::value_error("states and observations must have the same number of samples");
    }

    neurale::models::StateSpaceFitResult result;
    {
        py::gil_scoped_release release;
        result = neurale::models::fit_linear_gaussian(
            const_span(states), const_span(observations), size_span(segment_lengths),
            to_size(states.shape(1)), to_size(observations.shape(1)), fit_offsets, jitter);
    }

    py::dict output;
    output["model"] = py::cast(LinearGaussianModel(std::move(result.model)));
    output["n_samples"] = result.n_samples;
    output["n_segments"] = result.n_segments;
    output["n_transitions"] = result.n_transitions;
    return output;
}

/// Rebuild a fitted linear-Gaussian model from its parameters alone.
///
/// The read side of decoder persistence; see ``linear_model_from_state``. Every
/// array is checked against the two dimensions the first two imply, so a state
/// assembled from a stored artifact cannot enter the recursion half-shaped.
LinearGaussianModel linear_gaussian_from_state(const ConvertibleDoubleArray& transition,
                                               const ConvertibleDoubleArray& transition_offset,
                                               const ConvertibleDoubleArray& observation,
                                               const ConvertibleDoubleArray& observation_offset,
                                               const ConvertibleDoubleArray& process_covariance,
                                               const ConvertibleDoubleArray& observation_covariance,
                                               const ConvertibleDoubleArray& initial_state,
                                               const ConvertibleDoubleArray& initial_covariance)
{
    require_2d(transition, "transition");
    require_2d(observation, "observation");
    if (transition.shape(0) == 0 || observation.shape(0) == 0)
    {
        throw py::value_error("transition and observation must not be empty");
    }
    const auto n_states = transition.shape(0);
    const auto n_observations = observation.shape(0);

    require_shape(transition, {n_states, n_states}, "transition");
    require_shape(transition_offset, {n_states}, "transition_offset");
    require_shape(observation, {n_observations, n_states}, "observation");
    require_shape(observation_offset, {n_observations}, "observation_offset");
    require_shape(process_covariance, {n_states, n_states}, "process_covariance");
    require_shape(observation_covariance, {n_observations, n_observations},
                  "observation_covariance");
    require_shape(initial_state, {n_states}, "initial_state");
    require_shape(initial_covariance, {n_states, n_states}, "initial_covariance");

    return LinearGaussianModel({
        to_size(n_states),
        to_size(n_observations),
        to_vector(transition),
        to_vector(transition_offset),
        to_vector(observation),
        to_vector(observation_offset),
        to_vector(process_covariance),
        to_vector(observation_covariance),
        to_vector(initial_state),
        to_vector(initial_covariance),
    });
}

void model_predict(const LinearGaussianModel& model, StrictDoubleArray& state,
                   StrictDoubleArray& covariance)
{
    require_writable(state, "state");
    require_writable(covariance, "covariance");
    py::gil_scoped_release release;
    model.predict(mutable_span(state), mutable_span(covariance));
}

void model_update(const LinearGaussianModel& model, StrictDoubleArray& state,
                  StrictDoubleArray& covariance, const StrictDoubleArray& observation,
                  double jitter)
{
    require_writable(state, "state");
    require_writable(covariance, "covariance");
    py::gil_scoped_release release;
    model.update(mutable_span(state), mutable_span(covariance), const_span(observation), jitter);
}

void model_filter(const LinearGaussianModel& model, const StrictDoubleArray& observations,
                  const MaskArray& observed, StrictDoubleArray& state,
                  StrictDoubleArray& covariance, StrictDoubleArray& filtered_states,
                  StrictDoubleArray& filtered_covariances, double jitter)
{
    require_2d(observations, "observations");
    require_writable(state, "state");
    require_writable(covariance, "covariance");
    require_writable(filtered_states, "filtered_states");
    require_writable(filtered_covariances, "filtered_covariances");
    const auto n_samples = to_size(observations.shape(0));
    py::gil_scoped_release release;
    model.filter(const_span(observations), n_samples, mask_span(observed), mutable_span(state),
                 mutable_span(covariance), mutable_span(filtered_states),
                 mutable_span(filtered_covariances), jitter);
}

} // namespace

void bind_models_state_space(py::module_& module)
{
    using State = neurale::models::LinearGaussianModelState;
    constexpr auto state = Axis::state;
    constexpr auto observed = Axis::observation;

    py::class_<LinearGaussianModel>(module, "LinearGaussianModel")
        .def_property_readonly("state_dim", &LinearGaussianModel::state_dim)
        .def_property_readonly("observation_dim", &LinearGaussianModel::observation_dim)
        .def_property_readonly("transition", matrix_property(&State::transition, state, state))
        .def_property_readonly("transition_offset",
                               vector_property(&State::transition_offset, state))
        .def_property_readonly("observation", matrix_property(&State::observation, observed, state))
        .def_property_readonly("observation_offset",
                               vector_property(&State::observation_offset, observed))
        .def_property_readonly("process_covariance",
                               matrix_property(&State::process_covariance, state, state))
        .def_property_readonly("observation_covariance",
                               matrix_property(&State::observation_covariance, observed, observed))
        .def_property_readonly("initial_state", vector_property(&State::initial_state, state))
        .def_property_readonly("initial_covariance",
                               matrix_property(&State::initial_covariance, state, state))
        .def("predict", &model_predict, py::arg("state").noconvert(),
             py::arg("covariance").noconvert())
        .def("update", &model_update, py::arg("state").noconvert(),
             py::arg("covariance").noconvert(), py::arg("observation").noconvert(),
             py::arg("jitter"))
        .def("filter", &model_filter, py::arg("observations").noconvert(),
             py::arg("observed").noconvert(), py::arg("state").noconvert(),
             py::arg("covariance").noconvert(), py::arg("filtered_states").noconvert(),
             py::arg("filtered_covariances").noconvert(), py::arg("jitter"));

    module.def("fit_linear_gaussian", &fit_linear_gaussian, py::arg("states"),
               py::arg("observations"), py::arg("segment_lengths"), py::arg("fit_offsets"),
               py::arg("jitter"));
    module.def("linear_gaussian_from_state", &linear_gaussian_from_state, py::arg("transition"),
               py::arg("transition_offset"), py::arg("observation"), py::arg("observation_offset"),
               py::arg("process_covariance"), py::arg("observation_covariance"),
               py::arg("initial_state"), py::arg("initial_covariance"));
}
