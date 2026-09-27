/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace neurale::bindings::sorting
{

inline py::ssize_t checked_ssize(std::size_t value, const char* message)
{
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (value > maximum)
    {
        throw std::overflow_error(message);
    }
    return static_cast<py::ssize_t>(value);
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* message)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(message);
    }
    return left * right;
}

template <typename T> py::array_t<T> numeric_array(const std::vector<T>& values)
{
    py::array_t<T> output(checked_ssize(values.size(), "native result is too large"));
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

inline py::array_t<std::int64_t> index_array(const std::vector<std::size_t>& values,
                                             const char* size_message, const char* idx_message)
{
    py::array_t<std::int64_t> output(checked_ssize(values.size(), size_message));
    auto* destination = output.mutable_data();
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (values[i] > maximum)
        {
            throw std::overflow_error(idx_message);
        }
        destination[i] = static_cast<std::int64_t>(values[i]);
    }
    return output;
}

} // namespace neurale::bindings::sorting
