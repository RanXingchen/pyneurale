/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/spectral.h>

#include "fft_dispatch.h"
#include "support.h"

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <string>
#include <string_view>
#include <vector>

namespace nb = neurale::bindings::signal;

namespace py = pybind11;

namespace
{

using RealArray = nb::DoubleArray;
using ComplexArray = nb::ComplexArray;

neurale::signal::MultitaperWeighting parse_multitaper_weighting(std::string_view weighting);

class NativeMultitaperPsdProcessor
{
  public:
    NativeMultitaperPsdProcessor(std::size_t n_samples, std::size_t n_channels,
                                 std::size_t fft_length, double density_scale,
                                 const RealArray& tapers, std::size_t n_tapers,
                                 const RealArray& concentration_ratios,
                                 const std::string& weighting, bool one_sided, bool real_input)
        : real_input_(real_input),
          processor_(n_samples, n_channels, fft_length, density_scale, nb::const_span(tapers),
                     n_tapers, nb::const_span(concentration_ratios),
                     parse_multitaper_weighting(weighting), one_sided, real_input)
    {
        nb::require_2d(tapers, "multitaper PSD tapers");
        nb::require_1d(concentration_ratios, "multitaper PSD ratios");
    }

    RealArray process_real(const RealArray& x)
    {
        if (!real_input_)
        {
            throw py::value_error("processor was constructed for complex input");
        }
        nb::require_2d(x, "multitaper PSD input");
        RealArray output({static_cast<py::ssize_t>(processor_.output_bins()), x.shape(1)});
        {
            py::gil_scoped_release release;
            processor_.process_real(nb::const_span(x), nb::mutable_span(output));
        }
        return output;
    }

    RealArray process_complex(const ComplexArray& x)
    {
        if (real_input_)
        {
            throw py::value_error("processor was constructed for real input");
        }
        nb::require_2d(x, "multitaper PSD input");
        RealArray output({static_cast<py::ssize_t>(processor_.output_bins()), x.shape(1)});
        {
            py::gil_scoped_release release;
            processor_.process_complex(nb::const_span(x), nb::mutable_span(output));
        }
        return output;
    }

    [[nodiscard]] std::size_t output_bins() const noexcept
    {
        return processor_.output_bins();
    }

    void reset_adaptive_state()
    {
        processor_.reset_adaptive_state();
    }

  private:
    bool real_input_;
    neurale::signal::MultitaperPsdProcessor processor_;
};

neurale::signal::detail::FftKernel parse_fft_kernel(std::string_view kernel)
{
    if (kernel == "auto")
    {
        return neurale::signal::detail::FftKernel::automatic;
    }
    if (kernel == "builtin-fft")
    {
        return neurale::signal::detail::FftKernel::builtin;
    }
    if (kernel == "mkl-dfti")
    {
        return neurale::signal::detail::FftKernel::mkl_dfti;
    }
    throw py::value_error("unknown FFT kernel: " + std::string(kernel));
}

neurale::signal::MultitaperWeighting parse_multitaper_weighting(std::string_view weighting)
{
    if (weighting == "unity")
    {
        return neurale::signal::MultitaperWeighting::unity;
    }
    if (weighting == "eigen")
    {
        return neurale::signal::MultitaperWeighting::eigen;
    }
    if (weighting == "adaptive")
    {
        return neurale::signal::MultitaperWeighting::adaptive;
    }
    throw py::value_error("unknown multitaper weighting: " + std::string(weighting));
}

ComplexArray czt(const ComplexArray& x, std::size_t output_length, std::complex<double> ratio,
                 std::complex<double> start)
{
    nb::require_2d(x, "CZT input");
    ComplexArray output({static_cast<py::ssize_t>(output_length), x.shape(1)});
    {
        py::gil_scoped_release release;
        neurale::signal::czt(nb::const_span(x), nb::checked_size(x.shape(0)),
                             nb::checked_size(x.shape(1)), output_length, ratio, start,
                             nb::mutable_span(output));
    }
    return output;
}

ComplexArray analytic_signal(const RealArray& x, std::size_t fft_length)
{
    nb::require_2d(x, "analytic signal input");
    ComplexArray output({static_cast<py::ssize_t>(fft_length), x.shape(1)});
    {
        py::gil_scoped_release release;
        neurale::signal::analytic_signal(nb::const_span(x), nb::checked_size(x.shape(0)),
                                         nb::checked_size(x.shape(1)), fft_length,
                                         nb::mutable_span(output));
    }
    return output;
}

py::tuple fft(const ComplexArray& x, bool inverse, const std::string& kernel)
{
    nb::require_2d(x, "FFT input");
    const auto length = nb::checked_size(x.shape(0));
    const auto channels = nb::checked_size(x.shape(1));
    const auto requested = parse_fft_kernel(kernel);
    if (!neurale::signal::detail::fft_kernel_available(requested))
    {
        throw py::value_error("requested FFT kernel is unavailable in this build: " + kernel);
    }
    const auto selected = requested == neurale::signal::detail::FftKernel::automatic
                              ? neurale::signal::detail::select_fft_kernel(length)
                              : requested;
    ComplexArray output({x.shape(0), x.shape(1)});
    {
        py::gil_scoped_release release;
        std::vector<std::complex<double>> buffer(length);
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            for (std::size_t sample = 0; sample < length; ++sample)
            {
                buffer[sample] = x.data()[sample * channels + channel];
            }
            neurale::signal::detail::fft_with_kernel(buffer, inverse, selected);
            for (std::size_t sample = 0; sample < length; ++sample)
            {
                output.mutable_data()[sample * channels + channel] = buffer[sample];
            }
        }
    }
    return py::make_tuple(std::move(output),
                          std::string(neurale::signal::detail::fft_kernel_name(selected)));
}

} // namespace

void bind_signal_spectral(py::module_& module)
{
    module.def("czt", &czt, py::arg("x"), py::arg("output_length"), py::arg("ratio"),
               py::arg("start"));
    module.def("analytic_signal", &analytic_signal, py::arg("x"), py::arg("fft_length"));
    py::class_<NativeMultitaperPsdProcessor>(module, "_MultitaperPsdProcessor")
        .def(py::init<std::size_t, std::size_t, std::size_t, double, const RealArray&, std::size_t,
                      const RealArray&, const std::string&, bool, bool>(),
             py::arg("n_samples"), py::arg("n_channels"), py::arg("fft_length"),
             py::arg("density_scale"), py::arg("tapers"), py::arg("n_tapers"),
             py::arg("concentration_ratios"), py::arg("weighting"), py::arg("one_sided"),
             py::arg("real_input"))
        .def("process_real", &NativeMultitaperPsdProcessor::process_real, py::arg("x"))
        .def("process_complex", &NativeMultitaperPsdProcessor::process_complex, py::arg("x"))
        .def_property_readonly("output_bins", &NativeMultitaperPsdProcessor::output_bins)
        .def("reset_adaptive_state", &NativeMultitaperPsdProcessor::reset_adaptive_state);
    module.def("fft", &fft, py::arg("x"), py::arg("inverse") = false, py::arg("kernel") = "auto");
    module.def("fft_kernel_name",
               [](std::size_t length)
               {
                   return std::string(neurale::signal::detail::fft_kernel_name(
                       neurale::signal::detail::select_fft_kernel(length)));
               });
    module.def("fft_available_kernels",
               []
               {
                   py::list kernels;
                   for (const auto kernel : {
                            neurale::signal::detail::FftKernel::builtin,
                            neurale::signal::detail::FftKernel::mkl_dfti,
                        })
                   {
                       if (neurale::signal::detail::fft_kernel_available(kernel))
                       {
                           kernels.append(
                               std::string(neurale::signal::detail::fft_kernel_name(kernel)));
                       }
                   }
                   return kernels;
               });
}
