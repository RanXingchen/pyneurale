/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/fir.h>
#include <neurale/signal/iir.h>

#include "fir_dispatch.h"
#include "fir_realtime.h"
#include "iir_dispatch.h"
#include "iir_realtime.h"
#include "sos_realtime.h"
#include "support.h"

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <string_view>

namespace nb = neurale::bindings::signal;

namespace
{

using Array = nb::DoubleArray;
using NativeFrame = py::array_t<double, py::array::c_style>;

std::size_t realtime_sample_count(const py::array& frame, std::size_t n_channels,
                                  std::string_view label)
{
    if (frame.ndim() == 2 && frame.shape(1) == static_cast<py::ssize_t>(n_channels))
    {
        return nb::checked_size(frame.shape(0));
    }
    if (frame.ndim() == 1)
    {
        const auto length = nb::checked_size(frame.shape(0));
        if (n_channels == 1)
        {
            return length;
        }
        if (length == n_channels)
        {
            return 1;
        }
    }
    throw py::value_error(std::string(label) + " frame shape does not match processor channels");
}

void require_writable_frame(const py::array& frame, std::string_view label)
{
    if (!frame.writeable())
    {
        throw py::value_error(std::string(label) + " frame must be writable");
    }
}

template <class Processor>
void process_realtime_frame(NativeFrame& frame, Processor& processor, std::string_view label)
{
    require_writable_frame(frame, label);
    const auto n_samples = realtime_sample_count(frame, processor.n_channels(), label);
    auto* data = frame.mutable_data();
    {
        py::gil_scoped_release release;
        processor.process({data, n_samples * processor.n_channels()}, n_samples);
    }
}

bool has_state_shape(const Array& state, std::size_t rows, std::size_t n_channels)
{
    return state.ndim() == 2 && state.shape(0) == static_cast<py::ssize_t>(rows) &&
           state.shape(1) == static_cast<py::ssize_t>(n_channels);
}

void require_state_shape(const Array& state, std::size_t rows, std::size_t n_channels)
{
    if (!has_state_shape(state, rows, n_channels))
    {
        throw py::value_error("state shape mismatch");
    }
}

Array processor_state(std::size_t rows, std::size_t n_channels)
{
    return Array({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(n_channels)});
}

class FirSampleProcessor
{
  public:
    FirSampleProcessor(const Array& taps, const Array& state)
        : processor_(make_processor(taps, state))
    {
    }

    void process(NativeFrame& frame)
    {
        process_realtime_frame(frame, processor_, "FIR");
    }

    void reset()
    {
        processor_.reset();
    }

    void set_state(const Array& state)
    {
        require_state_shape(state, processor_.order(), processor_.n_channels());
        processor_.set_state(nb::const_span(state));
    }

    Array state() const
    {
        Array result = processor_state(processor_.order(), processor_.n_channels());
        processor_.state(nb::mutable_span(result));
        return result;
    }

    std::string kernel() const
    {
        return std::string(processor_.kernel_name());
    }

  private:
    static neurale::signal::detail::FirRealtimeProcessor make_processor(const Array& taps,
                                                                        const Array& state)
    {
        if (taps.ndim() != 1 || taps.shape(0) == 0 || state.ndim() != 2 ||
            state.shape(0) != taps.shape(0) - 1 || state.shape(1) < 1)
        {
            throw py::value_error("FIR coefficient/state shape mismatch");
        }
        return neurale::signal::detail::FirRealtimeProcessor(
            nb::const_span(taps), nb::checked_size(state.shape(1)), nb::const_span(state));
    }

    neurale::signal::detail::FirRealtimeProcessor processor_;
};

class IirSampleProcessor
{
  public:
    IirSampleProcessor(const Array& b, const Array& a, const Array& state)
        : processor_(make_processor(b, a, state))
    {
    }

    void process(NativeFrame& frame)
    {
        process_realtime_frame(frame, processor_, "IIR");
    }

    void reset()
    {
        processor_.reset();
    }

    void set_state(const Array& state)
    {
        require_state_shape(state, processor_.order(), processor_.n_channels());
        processor_.set_state(nb::const_span(state));
    }

    Array state() const
    {
        Array result = processor_state(processor_.order(), processor_.n_channels());
        processor_.state(nb::mutable_span(result));
        return result;
    }

    std::string kernel() const
    {
        return std::string(processor_.kernel_name());
    }

  private:
    static neurale::signal::detail::IirRealtimeProcessor
    make_processor(const Array& b, const Array& a, const Array& state)
    {
        if (b.ndim() != 1 || a.ndim() != 1 || state.ndim() != 2 || b.shape(0) == 0 ||
            b.shape(0) != a.shape(0) || state.shape(0) != b.shape(0) - 1 || state.shape(1) < 1)
        {
            throw py::value_error("IIR coefficient/state shape mismatch");
        }
        if (a.data()[0] != 1.0)
        {
            throw py::value_error("IIR denominator must be normalized");
        }
        return neurale::signal::detail::IirRealtimeProcessor(nb::const_span(b), nb::const_span(a),
                                                             nb::checked_size(state.shape(1)),
                                                             nb::const_span(state));
    }

    neurale::signal::detail::IirRealtimeProcessor processor_;
};

class SosSampleProcessor
{
  public:
    SosSampleProcessor(const Array& sos, const Array& state)
        : processor_(make_processor(sos, state))
    {
    }

    void process(NativeFrame& frame)
    {
        process_realtime_frame(frame, processor_, "SOS");
    }

    void reset()
    {
        processor_.reset();
    }

    void set_state(const Array& state)
    {
        if (state.ndim() != 3 ||
            state.shape(0) != static_cast<py::ssize_t>(processor_.n_sections()) ||
            state.shape(1) != 2 ||
            state.shape(2) != static_cast<py::ssize_t>(processor_.n_channels()))
        {
            throw py::value_error("state shape mismatch");
        }
        processor_.set_state(nb::const_span(state));
    }

    Array state() const
    {
        Array result({static_cast<py::ssize_t>(processor_.n_sections()), py::ssize_t{2},
                      static_cast<py::ssize_t>(processor_.n_channels())});
        processor_.state(nb::mutable_span(result));
        return result;
    }

    std::string kernel() const
    {
        return std::string(processor_.kernel_name());
    }

  private:
    static neurale::signal::detail::SosRealtimeProcessor make_processor(const Array& sos,
                                                                        const Array& state)
    {
        if (sos.ndim() != 2 || sos.shape(1) != 6 || state.ndim() != 3 ||
            state.shape(0) != sos.shape(0) || state.shape(1) != 2 || state.shape(2) < 1)
        {
            throw py::value_error("SOS coefficient/state shape mismatch");
        }
        for (py::ssize_t section = 0; section < sos.shape(0); ++section)
        {
            if (sos.data()[section * 6 + 3] != 1.0)
            {
                throw py::value_error("SOS denominator must be normalized");
            }
        }
        return neurale::signal::detail::SosRealtimeProcessor(
            nb::const_span(sos), nb::checked_size(state.shape(2)), nb::const_span(state));
    }

    neurale::signal::detail::SosRealtimeProcessor processor_;
};

neurale::signal::detail::FirKernel parse_fir_kernel(std::string_view kernel)
{
    if (kernel == "auto")
    {
        return neurale::signal::detail::FirKernel::automatic;
    }
    if (kernel == "builtin-direct")
    {
        return neurale::signal::detail::FirKernel::builtin_direct;
    }
    if (kernel == "mkl-vsl-direct")
    {
        return neurale::signal::detail::FirKernel::mkl_vsl_direct;
    }
    if (kernel == "mkl-vsl-fft")
    {
        return neurale::signal::detail::FirKernel::mkl_vsl_fft;
    }
    throw py::value_error("unknown FIR kernel: " + std::string(kernel));
}

neurale::signal::detail::IirKernel parse_iir_kernel(std::string_view kernel)
{
    if (kernel == "auto")
    {
        return neurale::signal::detail::IirKernel::automatic;
    }
    if (kernel == "builtin-df2t")
    {
        return neurale::signal::detail::IirKernel::builtin_df2t;
    }
    if (kernel == "mkl-blas-df2t")
    {
        return neurale::signal::detail::IirKernel::mkl_blas_df2t;
    }
    throw py::value_error("unknown IIR kernel: " + std::string(kernel));
}

neurale::signal::detail::FirKernel checked_fir_kernel(const std::string& name)
{
    const auto kernel = parse_fir_kernel(name);
    if (!neurale::signal::detail::fir_kernel_available(kernel))
    {
        throw py::value_error("requested FIR kernel is unavailable in this build: " + name);
    }
    return kernel;
}

neurale::signal::detail::IirKernel checked_iir_kernel(const std::string& name, const char* label)
{
    const auto kernel = parse_iir_kernel(name);
    if (!neurale::signal::detail::iir_kernel_available(kernel))
    {
        throw py::value_error(std::string("requested ") + label +
                              " kernel is unavailable in this build: " + name);
    }
    return kernel;
}

std::string selected_fir_name(neurale::signal::detail::FirKernel kernel, std::size_t n_samples,
                              std::size_t n_channels, std::size_t n_taps)
{
    if (kernel == neurale::signal::detail::FirKernel::automatic)
    {
        kernel = neurale::signal::detail::select_fir_kernel(n_samples, n_channels, n_taps);
    }
    return std::string(neurale::signal::detail::fir_kernel_name(kernel));
}

std::string selected_iir_name(neurale::signal::detail::IirKernel kernel, std::size_t n_samples,
                              std::size_t n_channels, std::size_t complexity, bool sos)
{
    if (kernel == neurale::signal::detail::IirKernel::automatic)
    {
        kernel =
            sos ? neurale::signal::detail::select_sos_kernel(n_samples, n_channels, complexity)
                : neurale::signal::detail::select_iir_kernel(n_samples, n_channels, complexity);
    }
    return std::string(neurale::signal::detail::iir_kernel_name(kernel));
}

py::tuple fir_filter(const Array& x, const Array& taps, const Array& state,
                     const std::string& kernel)
{
    if (x.ndim() != 2 || taps.ndim() != 1 || state.ndim() != 2)
    {
        throw py::value_error("input/state must be 2D and taps must be 1D");
    }
    const auto n_samples = nb::checked_size(x.shape(0));
    const auto n_channels = nb::checked_size(x.shape(1));
    const auto n_taps = nb::checked_size(taps.shape(0));
    if (n_taps == 0 || state.shape(0) != static_cast<py::ssize_t>(n_taps - 1) ||
        state.shape(1) != x.shape(1))
    {
        throw py::value_error("state shape does not match taps and channels");
    }

    Array output({x.shape(0), x.shape(1)});
    Array final_state({state.shape(0), state.shape(1)});
    const auto selected = checked_fir_kernel(kernel);
    const auto name = selected_fir_name(selected, n_samples, n_channels, n_taps);
    {
        py::gil_scoped_release release;
        neurale::signal::detail::fir_filter_with_kernel(
            nb::const_span(x), n_samples, n_channels, nb::const_span(taps), nb::const_span(state),
            nb::mutable_span(output), nb::mutable_span(final_state), selected);
    }
    return py::make_tuple(std::move(output), std::move(final_state), name);
}

py::tuple iir_filter(const Array& x, const Array& b, const Array& a, const Array& state,
                     const std::string& kernel, bool check_finite)
{
    if (x.ndim() != 2 || b.ndim() != 1 || a.ndim() != 1 || state.ndim() != 2)
    {
        throw py::value_error("invalid IIR input dimensions");
    }
    if (b.shape(0) == 0 || x.shape(1) == 0)
    {
        throw py::value_error("IIR coefficients and channels must be nonempty");
    }
    const auto n_samples = nb::checked_size(x.shape(0));
    const auto n_channels = nb::checked_size(x.shape(1));
    const auto order = nb::checked_size(b.shape(0) - 1);
    if (b.shape(0) != a.shape(0) || state.shape(0) != static_cast<py::ssize_t>(order) ||
        state.shape(1) != x.shape(1))
    {
        throw py::value_error("IIR coefficient/state shape mismatch");
    }

    Array output({x.shape(0), x.shape(1)});
    Array final_state({state.shape(0), state.shape(1)});
    const auto selected = checked_iir_kernel(kernel, "IIR");
    const auto name = selected_iir_name(selected, n_samples, n_channels, order, false);
    {
        py::gil_scoped_release release;
        neurale::signal::detail::iir_filter_with_kernel(
            nb::const_span(x), n_samples, n_channels, nb::const_span(b), nb::const_span(a),
            nb::const_span(state), nb::mutable_span(output), nb::mutable_span(final_state),
            selected, check_finite);
    }
    return py::make_tuple(std::move(output), std::move(final_state), name);
}

py::tuple sos_filter(const Array& x, const Array& sos, const Array& state,
                     const std::string& kernel, bool check_finite)
{
    if (x.ndim() != 2 || sos.ndim() != 2 || sos.shape(1) != 6 || state.ndim() != 3 ||
        state.shape(0) != sos.shape(0) || state.shape(1) != 2 || state.shape(2) != x.shape(1))
    {
        throw py::value_error("SOS coefficient/state shape mismatch");
    }

    Array output({x.shape(0), x.shape(1)});
    Array final_state({state.shape(0), state.shape(1), state.shape(2)});
    const auto selected = checked_iir_kernel(kernel, "SOS");
    const auto name =
        selected_iir_name(selected, nb::checked_size(x.shape(0)), nb::checked_size(x.shape(1)),
                          nb::checked_size(sos.shape(0)), true);
    {
        py::gil_scoped_release release;
        neurale::signal::detail::sos_filter_with_kernel(
            nb::const_span(x), nb::checked_size(x.shape(0)), nb::checked_size(x.shape(1)),
            nb::const_span(sos), nb::checked_size(sos.shape(0)), nb::const_span(state),
            nb::mutable_span(output), nb::mutable_span(final_state), selected, check_finite);
    }
    return py::make_tuple(std::move(output), std::move(final_state), name);
}

Array sos_filtfilt(const Array& x, const Array& sos, bool check_finite)
{
    if (x.ndim() != 2 || sos.ndim() != 2 || sos.shape(1) != 6)
    {
        throw py::value_error("invalid zero-phase SOS input shape");
    }

    Array output({x.shape(0), x.shape(1)});
    {
        py::gil_scoped_release release;
        neurale::signal::detail::sos_filtfilt_with_kernel(
            nb::const_span(x), nb::checked_size(x.shape(0)), nb::checked_size(x.shape(1)),
            nb::const_span(sos), nb::checked_size(sos.shape(0)), nb::mutable_span(output),
            neurale::signal::detail::IirKernel::automatic, check_finite);
    }
    return output;
}

} // namespace

void bind_signal_filtering(py::module_& module)
{
    py::class_<FirSampleProcessor>(module, "_FirSampleProcessor")
        .def(py::init<const Array&, const Array&>())
        .def("process", &FirSampleProcessor::process, py::arg("frame").noconvert())
        .def("reset", &FirSampleProcessor::reset)
        .def("set_state", &FirSampleProcessor::set_state)
        .def("state", &FirSampleProcessor::state)
        .def_property_readonly("kernel", &FirSampleProcessor::kernel);

    py::class_<IirSampleProcessor>(module, "_IirSampleProcessor")
        .def(py::init<const Array&, const Array&, const Array&>())
        .def("process", &IirSampleProcessor::process, py::arg("frame").noconvert())
        .def("reset", &IirSampleProcessor::reset)
        .def("set_state", &IirSampleProcessor::set_state)
        .def("state", &IirSampleProcessor::state)
        .def_property_readonly("kernel", &IirSampleProcessor::kernel);

    py::class_<SosSampleProcessor>(module, "_SosSampleProcessor")
        .def(py::init<const Array&, const Array&>())
        .def("process", &SosSampleProcessor::process, py::arg("frame").noconvert())
        .def("reset", &SosSampleProcessor::reset)
        .def("set_state", &SosSampleProcessor::set_state)
        .def("state", &SosSampleProcessor::state)
        .def_property_readonly("kernel", &SosSampleProcessor::kernel);

    module.def("fir_filter", &fir_filter, py::arg("x"), py::arg("taps"), py::arg("state"),
               py::arg("kernel") = "auto");
    module.def("iir_filter", &iir_filter, py::arg("x"), py::arg("b"), py::arg("a"),
               py::arg("state"), py::arg("kernel") = "auto", py::arg("check_finite") = true);
    module.def("sos_filter", &sos_filter, py::arg("x"), py::arg("sos"), py::arg("state"),
               py::arg("kernel") = "auto", py::arg("check_finite") = true);
    module.def("sos_filtfilt", &sos_filtfilt, py::arg("x"), py::arg("sos"),
               py::arg("check_finite") = true);

    module.def("fir_kernel_name",
               [](std::size_t n_samples, std::size_t n_channels, std::size_t n_taps)
               {
                   return std::string(neurale::signal::detail::fir_kernel_name(
                       neurale::signal::detail::select_fir_kernel(n_samples, n_channels, n_taps)));
               });
    module.def("fir_available_kernels",
               []
               {
                   py::list kernels;
                   for (const auto kernel : {
                            neurale::signal::detail::FirKernel::builtin_direct,
                            neurale::signal::detail::FirKernel::mkl_vsl_direct,
                            neurale::signal::detail::FirKernel::mkl_vsl_fft,
                        })
                   {
                       if (neurale::signal::detail::fir_kernel_available(kernel))
                       {
                           kernels.append(
                               std::string(neurale::signal::detail::fir_kernel_name(kernel)));
                       }
                   }
                   return kernels;
               });
    module.def("iir_kernel_name",
               [](std::size_t n_samples, std::size_t n_channels, std::size_t order)
               {
                   return std::string(neurale::signal::detail::iir_kernel_name(
                       neurale::signal::detail::select_iir_kernel(n_samples, n_channels, order)));
               });
    module.def(
        "sos_kernel_name",
        [](std::size_t n_samples, std::size_t n_channels, std::size_t n_sections)
        {
            return std::string(neurale::signal::detail::iir_kernel_name(
                neurale::signal::detail::select_sos_kernel(n_samples, n_channels, n_sections)));
        });
    module.def("iir_available_kernels",
               []
               {
                   py::list kernels;
                   for (const auto kernel : {
                            neurale::signal::detail::IirKernel::builtin_df2t,
                            neurale::signal::detail::IirKernel::mkl_blas_df2t,
                        })
                   {
                       if (neurale::signal::detail::iir_kernel_available(kernel))
                       {
                           kernels.append(
                               std::string(neurale::signal::detail::iir_kernel_name(kernel)));
                       }
                   }
                   return kernels;
               });
}
