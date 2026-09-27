/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstddef>
#include <span>
#include <string>

namespace py = pybind11;

namespace neurale::bindings::signal
{

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ComplexArray = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<std::size_t, py::array::c_style | py::array::forcecast>;

template <typename T, int Flags> std::span<const T> const_span(const py::array_t<T, Flags>& arr)
{
    return {arr.data(), static_cast<std::size_t>(arr.size())};
}

template <typename T, int Flags> std::span<T> mutable_span(py::array_t<T, Flags>& arr)
{
    return {arr.mutable_data(), static_cast<std::size_t>(arr.size())};
}

inline std::size_t checked_size(py::ssize_t value)
{
    return static_cast<std::size_t>(value);
}

/// Reject an argument whose rank is not what the kernel reads.
///
/// @param what Names the operation and the argument together, the way the
///             existing messages do ("multitaper PSD input", "cutoff"), because
///             a caller who passed the wrong array to a chain of spectral calls
///             needs to know which one refused.
inline void require_2d(const py::array& arr, const char* what)
{
    if (arr.ndim() != 2)
    {
        throw py::value_error(std::string(what) + " must be 2D");
    }
}

/// Reject an argument that is not a vector. See ::require_2d for @p what.
inline void require_1d(const py::array& arr, const char* what)
{
    if (arr.ndim() != 1)
    {
        throw py::value_error(std::string(what) + " must be 1D");
    }
}

} // namespace neurale::bindings::signal
