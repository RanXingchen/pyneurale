/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/windows.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <span>
#include <string>

namespace py = pybind11;

namespace
{

py::array_t<double> cosine_window(const std::string& kind, std::size_t length, bool symmetric)
{
    py::array_t<double> result(length);
    {
        py::gil_scoped_release release;
        neurale::signal::cosine_window(
            kind, std::span<double>(result.mutable_data(), result.size()), symmetric);
    }
    return result;
}

py::array_t<double> kaiser_window(std::size_t length, double beta, bool symmetric)
{
    py::array_t<double> result(length);
    {
        py::gil_scoped_release release;
        neurale::signal::kaiser_window(std::span<double>(result.mutable_data(), result.size()),
                                       beta, symmetric);
    }
    return result;
}

py::tuple dpss(std::size_t length, double nw, std::size_t n_tapers)
{
    py::array_t<double> tapers(
        {static_cast<py::ssize_t>(length), static_cast<py::ssize_t>(n_tapers)});
    py::array_t<double> ratios(n_tapers);
    {
        py::gil_scoped_release release;
        neurale::signal::multitap(length, nw, n_tapers,
                                  std::span<double>(tapers.mutable_data(), tapers.size()),
                                  std::span<double>(ratios.mutable_data(), ratios.size()));
    }
    return py::make_tuple(std::move(tapers), std::move(ratios));
}

} // namespace

void bind_signal_windows(py::module_& module)
{
    module.def("cosine_window", &cosine_window, py::arg("kind"), py::arg("length"),
               py::arg("symmetric") = true);
    module.def("kaiser_window", &kaiser_window, py::arg("length"), py::arg("beta"),
               py::arg("symmetric") = true);
    module.def("dpss", &dpss, py::arg("length"), py::arg("nw"), py::arg("n_tapers"));
}
