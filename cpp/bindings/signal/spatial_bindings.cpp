/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/spatial.h>

#include "support.h"

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <utility>

namespace nb = neurale::bindings::signal;

namespace
{

using Array = nb::DoubleArray;
using IndexArray = nb::IndexArray;

py::tuple common_reference(const Array& x, const IndexArray& reference_channels,
                           const std::string& method)
{
    if (x.ndim() != 2 || reference_channels.ndim() != 1)
    {
        throw py::value_error("common-reference input must be 2D and channels 1D");
    }

    Array output({x.shape(0), x.shape(1)});
    Array reference(x.shape(0));
    {
        py::gil_scoped_release release;
        neurale::signal::common_reference(nb::const_span(x), nb::checked_size(x.shape(0)),
                                          nb::checked_size(x.shape(1)),
                                          nb::const_span(reference_channels), method,
                                          nb::mutable_span(output), nb::mutable_span(reference));
    }
    return py::make_tuple(std::move(output), std::move(reference));
}

} // namespace

void bind_signal_spatial(py::module_& module)
{
    module.def("common_reference", &common_reference, py::arg("x"), py::arg("reference_channels"),
               py::arg("method"));
}
