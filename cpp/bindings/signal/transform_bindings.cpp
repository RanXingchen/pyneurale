/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/transforms.h>

#include "support.h"

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include <complex>

namespace nb = neurale::bindings::signal;

namespace
{

using RealArray = nb::DoubleArray;
using ComplexArray = nb::ComplexArray;

RealArray fft_integrate_real(const RealArray& x, double time_step, std::size_t order)
{
    nb::require_2d(x, "integration input");
    RealArray output({x.shape(0), x.shape(1)});
    {
        py::gil_scoped_release release;
        neurale::signal::fft_integrate(nb::const_span(x), nb::checked_size(x.shape(0)),
                                       nb::checked_size(x.shape(1)), time_step, order,
                                       nb::mutable_span(output));
    }
    return output;
}

ComplexArray fft_integrate_complex(const ComplexArray& x, double time_step, std::size_t order)
{
    nb::require_2d(x, "integration input");
    ComplexArray output({x.shape(0), x.shape(1)});
    {
        py::gil_scoped_release release;
        neurale::signal::fft_integrate(nb::const_span(x), nb::checked_size(x.shape(0)),
                                       nb::checked_size(x.shape(1)), time_step, order,
                                       nb::mutable_span(output));
    }
    return output;
}

} // namespace

void bind_signal_transforms(py::module_& module)
{
    module.def("fft_integrate", &fft_integrate_complex, py::arg("x"), py::arg("time_step"),
               py::arg("order") = 1);
    module.def("fft_integrate_real", &fft_integrate_real, py::arg("x"), py::arg("time_step"),
               py::arg("order") = 1);
}
