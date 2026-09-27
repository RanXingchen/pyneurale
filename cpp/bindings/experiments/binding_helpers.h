/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Shared conversions and validation helpers for experiment bindings.
///
/// The experiment contract passes results in fixed-capacity arrays with a
/// separate count, because the runtime side may not allocate. Python wants a
/// list. Every paradigm crosses that boundary the same way, and every paradigm
/// exposes the same `validate` entry point, so both live here rather than once
/// per paradigm.

#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <neurale/experiments/contract.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace neurale::bindings::experiments
{

/// Convert user-facing seconds using the same rounding for every task.
inline neurale::experiments::DurationNs
duration_from_seconds(double seconds, const std::string& name, bool allow_zero)
{
    if (!std::isfinite(seconds))
        throw py::value_error(name + " must be finite");
    if (seconds < 0.0 || (!allow_zero && seconds == 0.0))
        throw py::value_error(
            name + (allow_zero ? " must not be negative" : " must be greater than zero"));

    constexpr long double nanos_per_second = 1'000'000'000.0L;
    const long double nanos = static_cast<long double>(seconds) * nanos_per_second;
    if (nanos >
        static_cast<long double>(std::numeric_limits<neurale::experiments::DurationNs>::max()))
        throw py::value_error(name + " is too large");
    const neurale::experiments::DurationNs rounded =
        static_cast<neurale::experiments::DurationNs>(std::round(nanos));
    if (!allow_zero && rounded == 0)
        throw py::value_error(name + " is too small to represent in nanoseconds");
    return rounded;
}

/// The first @p count elements of a fixed-capacity array, as a vector.
///
/// @p count is clamped to @p Capacity. The two are meant to agree -- the count
/// says how much of the array the runtime filled -- but the count arrives from
/// a struct a caller can also build by hand, and reading past the end to
/// discover it disagreed is not a way to find out.
template <typename Element, std::size_t Capacity>
std::vector<Element> prefix(const std::array<Element, Capacity>& items, std::size_t count)
{
    const std::size_t bounded = count < Capacity ? count : Capacity;
    return std::vector<Element>(items.begin(),
                                items.begin() + static_cast<std::ptrdiff_t>(bounded));
}

/// Expose `validate(Value)` on @p target.
///
/// Every configuration type in the contract answers the same question about
/// itself, and answers it with a ContractStatus rather than an exception.
///
/// Registered through a lambda rather than `py::overload_cast`: the native
/// `validate()` set is overloaded on record type and every member is `noexcept`,
/// which is exactly the combination that makes taking a plain address ambiguous.
template <typename Value> void bind_validate(py::module_& target)
{
    target.def("validate", [](const Value& value) { return validate(value); }, py::arg("value"));
}

} // namespace neurale::bindings::experiments
