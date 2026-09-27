/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "binding_helpers.h"

#include <neurale/experiments/assistance.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using namespace neurale::experiments;
using neurale::bindings::experiments::bind_validate;
using namespace neurale::experiments::assistance;

// Same rule as the command records: a fixed-size record is padded out to its
// capacity only at or beyond the dimension the caller declared, and a sequence
// that leaves a declared slot out is refused rather than filled in. A velocity
// component nobody supplied and one deliberately set to zero are the same bytes
// once written, and this is the last point at which they can be told apart.
template <typename Element, std::size_t Capacity>
std::array<Element, Capacity> to_fixed(const std::vector<Element>& items, std::size_t declared,
                                       const char* what)
{
    if (items.size() > Capacity)
        throw py::value_error(std::string(what) + " exceeds the fixed command capacity");
    if (items.size() < declared)
        throw py::value_error(std::string(what) + " supplies " + std::to_string(items.size()) +
                              " of the " + std::to_string(declared) +
                              " declared; a slot left out is not a zero");
    std::array<Element, Capacity> fixed{};
    for (std::size_t i = 0; i < items.size(); ++i)
        fixed[i] = items[i];
    return fixed;
}

std::array<std::array<double, kMaxCommandDim>, kMaxDesiredVelocities>
to_fixed_vectors(const std::vector<std::vector<double>>& rows, std::size_t count, std::size_t dim)
{
    if (rows.size() > kMaxDesiredVelocities)
        throw py::value_error("vectors exceeds the fixed manifold capacity");
    if (rows.size() < count)
        throw py::value_error("vectors supplies " + std::to_string(rows.size()) + " of the " +
                              std::to_string(count) + " declared by count");
    std::array<std::array<double, kMaxCommandDim>, kMaxDesiredVelocities> fixed{};
    for (std::size_t row = 0; row < rows.size(); ++row)
        fixed[row] = to_fixed<double, kMaxCommandDim>(rows[row], row < count ? dim : 0,
                                                      "a desired velocity");
    return fixed;
}

void check_axis_index(std::size_t idx)
{
    if (idx >= kMaxCommandDim)
        throw py::index_error("velocity axis index out of range");
}

void check_vector_index(std::size_t idx)
{
    if (idx >= kMaxDesiredVelocities)
        throw py::index_error("desired velocity index out of range");
}

} // namespace

void bind_experiments_assistance_module(py::module_& experiments)
{
    auto assistance = experiments.def_submodule(
        "assistance",
        "Shared velocity assistance: linear blending and orthogonal impedance. The "
        "transforms are blind -- they receive vectors and parameters and return a vector.");

    assistance.attr("MAX_DESIRED_VELOCITIES") = kMaxDesiredVelocities;
    assistance.attr("MIN_ASSISTANCE") = kMinAssistance;
    assistance.attr("MAX_ASSISTANCE") = kMaxAssistance;
    assistance.attr("LINEAR_BLEND_VERSION_1") = kLinearBlendVersion1;
    assistance.attr("ORTHO_IMPEDANCE_VERSION_1") = kOrthoImpedanceVersion1;

    py::enum_<AssistanceMethod>(assistance, "AssistanceMethod")
        .value("NONE", AssistanceMethod::none)
        .value("LINEAR_BLEND", AssistanceMethod::linear_blend)
        .value("ORTHO_IMPEDANCE", AssistanceMethod::ortho_impedance);

    assistance.def("assistance_method_declared", &assistance_method_declared, py::arg("method"));
    assistance.def("assistance_in_range", &assistance_in_range, py::arg("value"));

    py::class_<VelocityVector>(assistance, "VelocityVector")
        .def(py::init(
                 [](CommandSpaceId space, std::uint8_t dim, const std::vector<double>& values)
                 {
                     return VelocityVector{space, dim,
                                           to_fixed<double, kMaxCommandDim>(values, dim, "values")};
                 }),
             py::arg("space") = kUnsetCommandSpaceId, py::arg("dim") = 0,
             py::arg("values") = std::vector<double>{})
        .def_readonly("space", &VelocityVector::space)
        .def_readonly("dim", &VelocityVector::dim)
        .def_readonly("values", &VelocityVector::values)
        .def(
            "value",
            [](const VelocityVector& vel, std::size_t idx)
            {
                check_axis_index(idx);
                return vel.values[idx];
            },
            py::arg("idx"));

    py::class_<DesiredVelocitySet>(assistance, "DesiredVelocitySet")
        .def(py::init(
                 [](CommandSpaceId space, std::uint8_t dim, std::uint8_t count,
                    const std::vector<std::vector<double>>& vectors)
                 {
                     return DesiredVelocitySet{space, dim, count,
                                               to_fixed_vectors(vectors, count, dim)};
                 }),
             py::arg("space") = kUnsetCommandSpaceId, py::arg("dim") = 0, py::arg("count") = 0,
             py::arg("vectors") = std::vector<std::vector<double>>{})
        .def_readonly("space", &DesiredVelocitySet::space)
        .def_readonly("dim", &DesiredVelocitySet::dim)
        .def_readonly("count", &DesiredVelocitySet::count)
        .def_readonly("vectors", &DesiredVelocitySet::vectors)
        .def(
            "vector",
            [](const DesiredVelocitySet& desired, std::size_t idx)
            {
                check_vector_index(idx);
                return desired.vectors[idx];
            },
            py::arg("idx"));

    py::class_<LinearAssistance>(assistance, "LinearAssistance")
        .def(py::init([](double value) { return LinearAssistance{value}; }),
             py::arg("assistance") = kMinAssistance)
        .def_readonly("assistance", &LinearAssistance::assistance);

    const OrthoImpedanceParameters tol_defaults{};
    py::class_<OrthoImpedanceParameters>(assistance, "OrthoImpedanceParameters")
        .def(py::init(
                 [](std::uint8_t dim, const std::vector<bool>& domain,
                    const std::vector<double>& impedance, double rank_tol, double conditioning_tol)
                 {
                     OrthoImpedanceParameters parameters{};
                     parameters.dim = dim;
                     parameters.domain = to_fixed<bool, kMaxCommandDim>(domain, dim, "domain");
                     parameters.impedance =
                         to_fixed<double, kMaxCommandDim>(impedance, dim, "impedance");
                     parameters.rank_tol = rank_tol;
                     parameters.conditioning_tol = conditioning_tol;
                     return parameters;
                 }),
             py::arg("dim") = 0, py::arg("domain") = std::vector<bool>{},
             py::arg("impedance") = std::vector<double>{},
             py::arg("rank_tol") = tol_defaults.rank_tol,
             py::arg("conditioning_tol") = tol_defaults.conditioning_tol)
        .def_readonly("dim", &OrthoImpedanceParameters::dim)
        .def_readonly("domain", &OrthoImpedanceParameters::domain)
        .def_readonly("impedance", &OrthoImpedanceParameters::impedance)
        .def_readonly("rank_tol", &OrthoImpedanceParameters::rank_tol)
        .def_readonly("conditioning_tol", &OrthoImpedanceParameters::conditioning_tol);

    py::class_<LinearAssistanceRecord>(assistance, "LinearAssistanceRecord")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                    CommandSpaceId space, AssistanceMethod method, AssistanceVersion version,
                    std::uint8_t dim, double value, const std::vector<double>& external,
                    const std::vector<double>& guidance, const std::vector<double>& assisted)
                 {
                     return LinearAssistanceRecord{
                         time_ns,
                         sequence,
                         trial,
                         space,
                         method,
                         version,
                         dim,
                         value,
                         to_fixed<double, kMaxCommandDim>(external, dim, "external"),
                         to_fixed<double, kMaxCommandDim>(guidance, dim, "guidance"),
                         to_fixed<double, kMaxCommandDim>(assisted, dim, "assisted")};
                 }),
             py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("space") = kUnsetCommandSpaceId,
             py::arg("method") = AssistanceMethod::linear_blend,
             py::arg("version") = kLinearBlendVersion1, py::arg("dim") = 0,
             py::arg("assistance") = 0.0, py::arg("external") = std::vector<double>{},
             py::arg("guidance") = std::vector<double>{},
             py::arg("assisted") = std::vector<double>{})
        .def_readonly("time_ns", &LinearAssistanceRecord::time_ns)
        .def_readonly("sequence", &LinearAssistanceRecord::sequence)
        .def_readonly("trial", &LinearAssistanceRecord::trial)
        .def_readonly("space", &LinearAssistanceRecord::space)
        .def_readonly("method", &LinearAssistanceRecord::method)
        .def_readonly("version", &LinearAssistanceRecord::version)
        .def_readonly("dim", &LinearAssistanceRecord::dim)
        .def_readonly("assistance", &LinearAssistanceRecord::assistance)
        .def_readonly("external", &LinearAssistanceRecord::external)
        .def_readonly("guidance", &LinearAssistanceRecord::guidance)
        .def_readonly("assisted", &LinearAssistanceRecord::assisted);

    py::class_<OrthoImpedanceRecord>(assistance, "OrthoImpedanceRecord")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                    std::uint64_t manifold, CommandSpaceId space, AssistanceMethod method,
                    AssistanceVersion version, std::uint8_t dim, double rank_tol,
                    double conditioning_tol, const std::vector<bool>& domain,
                    const std::vector<double>& impedance, const std::vector<double>& external,
                    const std::vector<double>& assisted)
                 {
                     return OrthoImpedanceRecord{
                         time_ns,
                         sequence,
                         trial,
                         manifold,
                         space,
                         method,
                         version,
                         dim,
                         rank_tol,
                         conditioning_tol,
                         to_fixed<bool, kMaxCommandDim>(domain, dim, "domain"),
                         to_fixed<double, kMaxCommandDim>(impedance, dim, "impedance"),
                         to_fixed<double, kMaxCommandDim>(external, dim, "external"),
                         to_fixed<double, kMaxCommandDim>(assisted, dim, "assisted")};
                 }),
             py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("manifold") = 0, py::arg("space") = kUnsetCommandSpaceId,
             py::arg("method") = AssistanceMethod::ortho_impedance,
             py::arg("version") = kOrthoImpedanceVersion1, py::arg("dim") = 0,
             py::arg("rank_tol") = tol_defaults.rank_tol,
             py::arg("conditioning_tol") = tol_defaults.conditioning_tol,
             py::arg("domain") = std::vector<bool>{}, py::arg("impedance") = std::vector<double>{},
             py::arg("external") = std::vector<double>{},
             py::arg("assisted") = std::vector<double>{})
        .def_readonly("time_ns", &OrthoImpedanceRecord::time_ns)
        .def_readonly("sequence", &OrthoImpedanceRecord::sequence)
        .def_readonly("trial", &OrthoImpedanceRecord::trial)
        .def_readonly("manifold", &OrthoImpedanceRecord::manifold)
        .def_readonly("space", &OrthoImpedanceRecord::space)
        .def_readonly("method", &OrthoImpedanceRecord::method)
        .def_readonly("version", &OrthoImpedanceRecord::version)
        .def_readonly("dim", &OrthoImpedanceRecord::dim)
        .def_readonly("rank_tol", &OrthoImpedanceRecord::rank_tol)
        .def_readonly("conditioning_tol", &OrthoImpedanceRecord::conditioning_tol)
        .def_readonly("domain", &OrthoImpedanceRecord::domain)
        .def_readonly("impedance", &OrthoImpedanceRecord::impedance)
        .def_readonly("external", &OrthoImpedanceRecord::external)
        .def_readonly("assisted", &OrthoImpedanceRecord::assisted);

    // The transforms report a status beside their result rather than raising:
    // the same code runs off a realtime thread, where an exception is not an
    // option, and a Python caller still learns exactly what was rejected.
    assistance.def(
        "blend_velocity",
        [](const CommandSpace& space, const VelocityVector& external,
           const VelocityVector& guidance, const LinearAssistance& parameters)
        {
            VelocityVector assisted{};
            const ContractStatus status =
                blend_velocity(space, external, guidance, parameters, assisted);
            return std::pair{status, assisted};
        },
        py::arg("space"), py::arg("external"), py::arg("guidance"), py::arg("parameters"));

    assistance.def(
        "apply_ortho_impedance",
        [](const CommandSpace& space, const DesiredVelocitySet& desired,
           const VelocityVector& external, const OrthoImpedanceParameters& parameters)
        {
            VelocityVector assisted{};
            const ContractStatus status =
                apply_ortho_impedance(space, desired, external, parameters, assisted);
            return std::pair{status, assisted};
        },
        py::arg("space"), py::arg("desired"), py::arg("external"), py::arg("parameters"));

    assistance.def("manifold_fingerprint", &manifold_fingerprint, py::arg("desired"));

    bind_validate<VelocityVector>(assistance);
    bind_validate<DesiredVelocitySet>(assistance);
    bind_validate<LinearAssistance>(assistance);
    bind_validate<OrthoImpedanceParameters>(assistance);
    bind_validate<LinearAssistanceRecord>(assistance);
    bind_validate<OrthoImpedanceRecord>(assistance);

    assistance.def(
        "validate_against", [](const VelocityVector& vel, const CommandSpace& space)
        { return validate_against(vel, space); }, py::arg("vel"), py::arg("space"));
    assistance.def(
        "validate_against", [](const DesiredVelocitySet& desired, const CommandSpace& space)
        { return validate_against(desired, space); }, py::arg("desired"), py::arg("space"));
    assistance.def(
        "validate_against",
        [](const OrthoImpedanceParameters& parameters, const CommandSpace& space)
        { return validate_against(parameters, space); }, py::arg("parameters"), py::arg("space"));
}
