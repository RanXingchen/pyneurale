/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/representations.h>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>

namespace py = pybind11;

namespace
{

using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ComplexArray = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

neurale::signal::StateSpace state_from_python(const RealArray& a, const RealArray& b,
                                              const RealArray& c, double d)
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1) || b.ndim() != 1 || c.ndim() != 1 ||
        b.shape(0) != a.shape(0) || c.shape(0) != a.shape(0))
    {
        throw py::value_error("invalid SISO state-space shapes");
    }
    return {static_cast<std::size_t>(a.shape(0)),
            std::vector<double>(a.data(), a.data() + a.size()),
            std::vector<double>(b.data(), b.data() + b.size()),
            std::vector<double>(c.data(), c.data() + c.size()), d};
}

py::tuple state_to_python(const neurale::signal::StateSpace& state)
{
    const auto n = static_cast<py::ssize_t>(state.order);
    py::array_t<double> a({n, n});
    py::array_t<double> b(n);
    py::array_t<double> c(n);
    std::copy(state.a.begin(), state.a.end(), a.mutable_data());
    std::copy(state.b.begin(), state.b.end(), b.mutable_data());
    std::copy(state.c.begin(), state.c.end(), c.mutable_data());
    return py::make_tuple(std::move(a), std::move(b), std::move(c), state.d);
}

py::tuple tf_to_python(const neurale::signal::TransferFunction& tf)
{
    py::array_t<double> num(tf.num.size());
    py::array_t<double> den(tf.den.size());
    std::copy(tf.num.begin(), tf.num.end(), num.mutable_data());
    std::copy(tf.den.begin(), tf.den.end(), den.mutable_data());
    return py::make_tuple(std::move(num), std::move(den));
}

py::tuple zpk_to_python(const neurale::signal::Zpk& zpk)
{
    py::array_t<std::complex<double>> z(zpk.z.size());
    py::array_t<std::complex<double>> p(zpk.p.size());
    std::copy(zpk.z.begin(), zpk.z.end(), z.mutable_data());
    std::copy(zpk.p.begin(), zpk.p.end(), p.mutable_data());
    return py::make_tuple(std::move(z), std::move(p), zpk.k);
}

double prewarp_or_nan(const py::object& value)
{
    return value.is_none() ? std::numeric_limits<double>::quiet_NaN() : value.cast<double>();
}

std::size_t trimmed_order(const RealArray& coefs)
{
    if (coefs.ndim() != 1)
    {
        throw py::value_error("coefficients must be 1D");
    }
    std::size_t first = 0;
    while (first < static_cast<std::size_t>(coefs.size()) && coefs.data()[first] == 0.0)
    {
        ++first;
    }
    if (first == static_cast<std::size_t>(coefs.size()))
    {
        throw py::value_error("coefficients must contain a nonzero value");
    }
    return static_cast<std::size_t>(coefs.size()) - first - 1;
}

} // namespace

void bind_signal_representations(py::module_& module)
{
    module.def("zpk2tf",
               [](const ComplexArray& z, const ComplexArray& p, std::complex<double> k)
               {
                   const auto size = static_cast<py::ssize_t>(p.size() + 1);
                   RealArray num(size);
                   RealArray den(size);
                   neurale::signal::zpk2tf(
                       {z.data(), static_cast<std::size_t>(z.size())},
                       {p.data(), static_cast<std::size_t>(p.size())}, k,
                       {num.mutable_data(), static_cast<std::size_t>(num.size())},
                       {den.mutable_data(), static_cast<std::size_t>(den.size())});
                   return py::make_tuple(std::move(num), std::move(den));
               });
    module.def("tf2ss",
               [](const RealArray& num, const RealArray& den)
               {
                   const auto n = static_cast<py::ssize_t>(trimmed_order(den));
                   RealArray a({n, n});
                   RealArray b(n);
                   RealArray c(n);
                   double d = 0.0;
                   neurale::signal::tf2ss({num.data(), static_cast<std::size_t>(num.size())},
                                          {den.data(), static_cast<std::size_t>(den.size())},
                                          {a.mutable_data(), static_cast<std::size_t>(a.size())},
                                          {b.mutable_data(), static_cast<std::size_t>(b.size())},
                                          {c.mutable_data(), static_cast<std::size_t>(c.size())},
                                          d);
                   return py::make_tuple(std::move(a), std::move(b), std::move(c), d);
               });
    module.def("zpk2ss",
               [](const ComplexArray& z, const ComplexArray& p, std::complex<double> k)
               {
                   const auto n = static_cast<py::ssize_t>(p.size());
                   RealArray a({n, n});
                   RealArray b(n);
                   RealArray c(n);
                   double d = 0.0;
                   neurale::signal::zpk2ss({z.data(), static_cast<std::size_t>(z.size())},
                                           {p.data(), static_cast<std::size_t>(p.size())}, k,
                                           {a.mutable_data(), static_cast<std::size_t>(a.size())},
                                           {b.mutable_data(), static_cast<std::size_t>(b.size())},
                                           {c.mutable_data(), static_cast<std::size_t>(c.size())},
                                           d);
                   return py::make_tuple(std::move(a), std::move(b), std::move(c), d);
               });
    module.def("ss2tf", [](const RealArray& a, const RealArray& b, const RealArray& c, double d)
               { return tf_to_python(neurale::signal::ss2tf(state_from_python(a, b, c, d))); });
    module.def("tf2zpk",
               [](const RealArray& num, const RealArray& den)
               {
                   return zpk_to_python(
                       neurale::signal::tf2zpk({num.data(), static_cast<std::size_t>(num.size())},
                                               {den.data(), static_cast<std::size_t>(den.size())}));
               });
    module.def("ss2zpk", [](const RealArray& a, const RealArray& b, const RealArray& c, double d)
               { return zpk_to_python(neurale::signal::ss2zpk(state_from_python(a, b, c, d))); });
    module.def("zpk2sos",
               [](const ComplexArray& z, const ComplexArray& p, double k)
               {
                   const auto sections = std::max<py::ssize_t>(1, (p.size() + 1) / 2);
                   py::array_t<double> result({sections, static_cast<py::ssize_t>(6)});
                   neurale::signal::zpk2sos(
                       {z.data(), static_cast<std::size_t>(z.size())},
                       {p.data(), static_cast<std::size_t>(p.size())}, k,
                       {result.mutable_data(), static_cast<std::size_t>(result.size())});
                   return result;
               });

    module.def(
        "lp2lp_ss",
        [](const RealArray& a, const RealArray& b, const RealArray& c, double d, double cutoff)
        {
            return state_to_python(
                neurale::signal::lp2lp_ss(state_from_python(a, b, c, d), cutoff));
        });
    module.def(
        "lp2hp_ss",
        [](const RealArray& a, const RealArray& b, const RealArray& c, double d, double cutoff)
        {
            return state_to_python(
                neurale::signal::lp2hp_ss(state_from_python(a, b, c, d), cutoff));
        });
    module.def("lp2bp_ss",
               [](const RealArray& a, const RealArray& b, const RealArray& c, double d,
                  double center, double bandwidth)
               {
                   return state_to_python(
                       neurale::signal::lp2bp_ss(state_from_python(a, b, c, d), center, bandwidth));
               });
    module.def("lp2bs_ss",
               [](const RealArray& a, const RealArray& b, const RealArray& c, double d,
                  double center, double bandwidth)
               {
                   return state_to_python(
                       neurale::signal::lp2bs_ss(state_from_python(a, b, c, d), center, bandwidth));
               });
    module.def("bilinear_zpk",
               [](const ComplexArray& z, const ComplexArray& p, std::complex<double> k, double rate,
                  const py::object& prewarp)
               {
                   if (z.ndim() != 1 || p.ndim() != 1)
                   {
                       throw py::value_error("z and p must be 1D");
                   }
                   ComplexArray digital_z(p.size());
                   ComplexArray digital_p(p.size());
                   std::complex<double> digital_k;
                   const auto prewarp_value = prewarp_or_nan(prewarp);
                   {
                       py::gil_scoped_release release;
                       neurale::signal::bilinear_zpk(
                           {z.data(), static_cast<std::size_t>(z.size())},
                           {p.data(), static_cast<std::size_t>(p.size())}, k, rate, prewarp_value,
                           {digital_z.mutable_data(), static_cast<std::size_t>(digital_z.size())},
                           {digital_p.mutable_data(), static_cast<std::size_t>(digital_p.size())},
                           digital_k);
                   }
                   return py::make_tuple(std::move(digital_z), std::move(digital_p), digital_k);
               });
    module.def("bilinear_ss",
               [](const RealArray& a, const RealArray& b, const RealArray& c, double d, double rate,
                  const py::object& prewarp)
               {
                   if (a.ndim() != 2 || a.shape(0) != a.shape(1))
                   {
                       throw py::value_error("invalid SISO state-space shapes");
                   }
                   const auto states = a.shape(0);
                   const auto valid_b = (b.ndim() == 1 && b.shape(0) == states) ||
                                        (b.ndim() == 2 && b.shape(0) == states && b.shape(1) == 1);
                   const auto valid_c = (c.ndim() == 1 && c.shape(0) == states) ||
                                        (c.ndim() == 2 && c.shape(0) == 1 && c.shape(1) == states);
                   if (!valid_b || !valid_c)
                   {
                       throw py::value_error("invalid SISO state-space shapes");
                   }
                   const auto n = static_cast<std::size_t>(a.shape(0));
                   RealArray digital_a({a.shape(0), a.shape(1)});
                   RealArray digital_b(states);
                   RealArray digital_c(states);
                   double digital_d;
                   const auto prewarp_value = prewarp_or_nan(prewarp);
                   {
                       py::gil_scoped_release release;
                       neurale::signal::bilinear_ss(
                           {a.data(), static_cast<std::size_t>(a.size())},
                           {b.data(), static_cast<std::size_t>(b.size())},
                           {c.data(), static_cast<std::size_t>(c.size())}, d, n, rate,
                           prewarp_value,
                           {digital_a.mutable_data(), static_cast<std::size_t>(digital_a.size())},
                           {digital_b.mutable_data(), static_cast<std::size_t>(digital_b.size())},
                           {digital_c.mutable_data(), static_cast<std::size_t>(digital_c.size())},
                           digital_d);
                   }
                   return py::make_tuple(std::move(digital_a), std::move(digital_b),
                                         std::move(digital_c), digital_d);
               });
    module.def(
        "bilinear_tf",
        [](const RealArray& num, const RealArray& den, double rate, const py::object& prewarp)
        {
            if (num.ndim() != 1 || den.ndim() != 1)
            {
                throw py::value_error("num and den must be 1D");
            }
            const auto size = neurale::signal::bilinear_tf_size(
                {num.data(), static_cast<std::size_t>(num.size())},
                {den.data(), static_cast<std::size_t>(den.size())});
            RealArray digital_num(static_cast<py::ssize_t>(size));
            RealArray digital_den(static_cast<py::ssize_t>(size));
            const auto prewarp_value = prewarp_or_nan(prewarp);
            {
                py::gil_scoped_release release;
                neurale::signal::bilinear_tf(
                    {num.data(), static_cast<std::size_t>(num.size())},
                    {den.data(), static_cast<std::size_t>(den.size())}, rate, prewarp_value,
                    {digital_num.mutable_data(), static_cast<std::size_t>(digital_num.size())},
                    {digital_den.mutable_data(), static_cast<std::size_t>(digital_den.size())});
            }
            return py::make_tuple(std::move(digital_num), std::move(digital_den));
        });
}
