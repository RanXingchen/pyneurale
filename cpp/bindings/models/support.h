/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace neurale::bindings::models
{

using ConvertibleDoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using StrictDoubleArray = py::array_t<double, py::array::c_style>;
using SizeArray = py::array_t<std::size_t, py::array::c_style>;

template <typename T, int Flags> std::span<const T> const_span(const py::array_t<T, Flags>& arr)
{
    return {arr.data(), static_cast<std::size_t>(arr.size())};
}

template <typename T, int Flags> std::span<T> mutable_span(py::array_t<T, Flags>& arr)
{
    return {arr.mutable_data(), static_cast<std::size_t>(arr.size())};
}

inline std::size_t to_size(py::ssize_t value)
{
    return static_cast<std::size_t>(value);
}

inline void require_1d_output_shape(const py::array& output, py::ssize_t size)
{
    if (output.ndim() != 1 || output.shape(0) != size)
    {
        throw py::value_error("out has an invalid shape");
    }
}

inline void require_2d_output_shape(const py::array& output, py::ssize_t rows, py::ssize_t cols)
{
    if (output.ndim() != 2 || output.shape(0) != rows || output.shape(1) != cols)
    {
        throw py::value_error("out has an invalid shape");
    }
}

inline StrictDoubleArray make_strict_double_output(std::vector<py::ssize_t> shape)
{
    return StrictDoubleArray(std::move(shape));
}

inline StrictDoubleArray array_from_vector(std::vector<double> values,
                                           std::vector<py::ssize_t> shape)
{
    StrictDoubleArray output(shape);
    std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(double));
    return output;
}

/// Copy an array's elements into an owned vector.
///
/// Used by the ``*_from_state`` factories, which rebuild a fitted model from
/// caller-supplied parameter arrays: a model state owns its storage, so the
/// arrays are copied rather than viewed and the model never depends on a buffer
/// Python may free or mutate afterwards.
template <typename T, int Flags> std::vector<T> to_vector(const py::array_t<T, Flags>& arr)
{
    const auto values = const_span(arr);
    return {values.begin(), values.end()};
}

/// Reject the coefficient/intercept pair a ``*_from_state`` factory was given.
///
/// Every affine model state is spelled the same way -- an ``n_outputs x
/// n_features`` coefficient matrix and one intercept per output -- so the same
/// four checks, and the same wording, apply wherever one is rebuilt.
inline void require_coef_intercept(const py::array& coef, const py::array& intercept)
{
    if (coef.ndim() != 2)
    {
        throw py::value_error("coef must be 2D");
    }
    if (intercept.ndim() != 1)
    {
        throw py::value_error("intercept must be 1D");
    }
    if (coef.shape(0) == 0 || coef.shape(1) == 0)
    {
        throw py::value_error("coef must not be empty");
    }
    if (coef.shape(0) != intercept.shape(0))
    {
        throw py::value_error("coef and intercept must declare the same number of outputs");
    }
}

/// Reject an array that is not a matrix.
///
/// Every model kernel here reads a sample-major `n_samples x n_features` block;
/// a 1-D or 3-D argument is a caller error, and one wording for it means a
/// caller sees the same message whichever entry point they reached.
inline void require_2d(const py::array& arr, const char* name)
{
    if (arr.ndim() != 2)
    {
        throw py::value_error(std::string(name) + " must be 2D");
    }
}

/// Reject a parameter array whose shape is not the one a model state requires.
inline void require_shape(const py::array& arr, const std::vector<py::ssize_t>& shape,
                          const char* name)
{
    if (arr.ndim() != static_cast<py::ssize_t>(shape.size()))
    {
        throw py::value_error(std::string(name) + " has an invalid number of dimensions");
    }
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
    {
        if (arr.shape(static_cast<py::ssize_t>(axis)) != shape[axis])
        {
            throw py::value_error(std::string(name) + " has an invalid shape");
        }
    }
}

inline std::vector<py::ssize_t> c_strides(const std::vector<py::ssize_t>& shape)
{
    auto strides = std::vector<py::ssize_t>(shape.size());
    py::ssize_t stride = static_cast<py::ssize_t>(sizeof(double));
    for (std::size_t i = shape.size(); i > 0; --i)
    {
        strides[i - 1] = stride;
        stride *= shape[i - 1];
    }
    return strides;
}

inline StrictDoubleArray readonly_array_view(std::span<const double> values,
                                             std::vector<py::ssize_t> shape, py::handle base)
{
    StrictDoubleArray output(shape, c_strides(shape), const_cast<double*>(values.data()), base);
    output.attr("setflags")(py::arg("write") = false);
    return output;
}

inline void require_writable(const py::array& arr, const char* name)
{
    if (!arr.writeable())
    {
        throw py::value_error(std::string(name) + " must be writable");
    }
}

/// Reject an output buffer that shares storage with the input it is computed from.
///
/// The kernels write ``output`` while still reading ``input``, so an overlapping
/// pair silently produces a mix of old and new values rather than failing.
inline void require_no_overlap(const py::array& X, const py::array& output, const char* name)
{
    const auto X_begin = reinterpret_cast<std::uintptr_t>(X.data());
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data());
    const auto X_end = X_begin + static_cast<std::uintptr_t>(X.nbytes());
    const auto output_end = output_begin + static_cast<std::uintptr_t>(output.nbytes());
    if (X_begin < output_end && output_begin < X_end)
    {
        throw py::value_error(std::string(name) + " must not overlap the X array");
    }
}

inline void require_aligned(const py::array& arr, const char* name)
{
    if ((arr.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_) == 0)
    {
        throw py::value_error(std::string(name) + " must be aligned");
    }
}

} // namespace neurale::bindings::models
