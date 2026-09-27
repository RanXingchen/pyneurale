/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/fir.h>
#include <neurale/signal/iir.h>

#include "support.h"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <utility>

namespace nb = neurale::bindings::signal;

namespace
{

using Array = nb::DoubleArray;
using ComplexArray = nb::ComplexArray;

py::tuple zpk_to_python(const neurale::signal::Zpk& zpk)
{
    ComplexArray z(zpk.z.size());
    ComplexArray p(zpk.p.size());
    std::copy(zpk.z.begin(), zpk.z.end(), z.mutable_data());
    std::copy(zpk.p.begin(), zpk.p.end(), p.mutable_data());
    return py::make_tuple(std::move(z), std::move(p), zpk.k);
}

Array firls(std::size_t order, const Array& bands, const Array& desired, const Array& weights)
{
    if (bands.ndim() != 1 || desired.ndim() != 1 || weights.ndim() != 1)
    {
        throw py::value_error("FIRLS inputs must be 1D");
    }

    Array output(order + 1);
    {
        py::gil_scoped_release release;
        neurale::signal::firls(order, nb::const_span(bands), nb::const_span(desired),
                               nb::const_span(weights), nb::mutable_span(output));
    }
    return output;
}

Array firwin_from_cutoff(std::size_t order, std::span<const double> cutoff, const std::string& band,
                         const std::string& window, bool scale)
{
    Array output(order + 1);
    {
        py::gil_scoped_release release;
        neurale::signal::firwin(order, cutoff, band, window, scale, nb::mutable_span(output));
    }
    return output;
}

Array firwin_scalar(std::size_t order, double cutoff, const std::string& band,
                    const std::string& window, bool scale)
{
    const std::array<double, 1> values{cutoff};
    return firwin_from_cutoff(order, values, band, window, scale);
}

Array firwin_pair(std::size_t order, double low, double high, const std::string& band,
                  const std::string& window, bool scale)
{
    const std::array<double, 2> values{low, high};
    return firwin_from_cutoff(order, values, band, window, scale);
}

Array firwin_array(std::size_t order, const Array& cutoff, const std::string& band,
                   const std::string& window, bool scale)
{
    nb::require_1d(cutoff, "cutoff");
    return firwin_from_cutoff(order, nb::const_span(cutoff), band, window, scale);
}

py::tuple iir_zpk(const std::string& kind, std::size_t order, const Array& cutoff,
                  const std::string& band, double fs, double rp, double rs)
{
    nb::require_1d(cutoff, "cutoff");
    return zpk_to_python(
        neurale::signal::iir_zpk(kind, order, nb::const_span(cutoff), band, fs, rp, rs));
}

py::tuple iir_tf(const std::string& kind, std::size_t order, const Array& cutoff,
                 const std::string& band, double fs, double rp, double rs)
{
    nb::require_1d(cutoff, "cutoff");
    const auto tf = neurale::signal::iir_tf(kind, order, nb::const_span(cutoff), band, fs, rp, rs);
    Array b(tf.num.size());
    Array a(tf.den.size());
    std::copy(tf.num.begin(), tf.num.end(), b.mutable_data());
    std::copy(tf.den.begin(), tf.den.end(), a.mutable_data());
    return py::make_tuple(std::move(b), std::move(a));
}

Array iir_sos(const std::string& kind, std::size_t order, const Array& cutoff,
              const std::string& band, double fs, double rp, double rs)
{
    nb::require_1d(cutoff, "cutoff");
    const auto values =
        neurale::signal::iir_sos(kind, order, nb::const_span(cutoff), band, fs, rp, rs);
    Array sos({static_cast<py::ssize_t>(values.size() / 6), py::ssize_t{6}});
    std::copy(values.begin(), values.end(), sos.mutable_data());
    return sos;
}

py::tuple iir_state_space(const std::string& kind, std::size_t order, const Array& cutoff,
                          const std::string& band, double fs, double rp, double rs)
{
    nb::require_1d(cutoff, "cutoff");
    const auto state =
        neurale::signal::iir_ss(kind, order, nb::const_span(cutoff), band, fs, rp, rs);
    const auto n = static_cast<py::ssize_t>(state.b.size());
    Array a({n, n});
    Array b(n);
    Array c(n);
    std::copy(state.a.begin(), state.a.end(), a.mutable_data());
    std::copy(state.b.begin(), state.b.end(), b.mutable_data());
    std::copy(state.c.begin(), state.c.end(), c.mutable_data());
    return py::make_tuple(std::move(a), std::move(b), std::move(c), state.d);
}

} // namespace

void bind_signal_filter_design(py::module_& module)
{
    module.def("firls", &firls, py::arg("order"), py::arg("bands"), py::arg("desired"),
               py::arg("weights"));
    module.def("firwin", &firwin_array, py::arg("order"), py::arg("cutoff"), py::arg("band"),
               py::arg("window"), py::arg("scale"));
    module.def("firwin_scalar", &firwin_scalar, py::arg("order"), py::arg("cutoff"),
               py::arg("band"), py::arg("window"), py::arg("scale"));
    module.def("firwin_pair", &firwin_pair, py::arg("order"), py::arg("low"), py::arg("high"),
               py::arg("band"), py::arg("window"), py::arg("scale"));
    module.def("notch_zpk", [](double freq, double bandwidth)
               { return zpk_to_python(neurale::signal::notch_zpk(freq, bandwidth)); });
    module.def("iir_zpk", &iir_zpk, py::arg("kind"), py::arg("order"), py::arg("cutoff"),
               py::arg("band"), py::arg("fs"), py::arg("rp"), py::arg("rs"));
    module.def("iir_tf", &iir_tf, py::arg("kind"), py::arg("order"), py::arg("cutoff"),
               py::arg("band"), py::arg("fs"), py::arg("rp"), py::arg("rs"));
    module.def("iir_sos", &iir_sos, py::arg("kind"), py::arg("order"), py::arg("cutoff"),
               py::arg("band"), py::arg("fs"), py::arg("rp"), py::arg("rs"));
    module.def("iir_ss", &iir_state_space, py::arg("kind"), py::arg("order"), py::arg("cutoff"),
               py::arg("band"), py::arg("fs"), py::arg("rp"), py::arg("rs"));
}
