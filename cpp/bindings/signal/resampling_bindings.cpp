/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/resample.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace py = pybind11;

namespace
{

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Below this input size the per-call work is small enough that releasing the
// GIL costs more than it saves; only larger frames drop the GIL in process().
constexpr std::size_t kGilReleaseSampleThreshold = 4096;

py::array_t<double> output_array(const std::vector<double>& buffer, std::size_t n_samples,
                                 std::size_t n_channels)
{
    py::array_t<double> output(
        {static_cast<py::ssize_t>(n_samples), static_cast<py::ssize_t>(n_channels)});
    std::copy_n(buffer.data(), n_samples * n_channels, output.mutable_data());
    return output;
}

class PyResampler
{
  public:
    PyResampler(std::size_t up, std::size_t down, const Array& filter)
        : resampler_(up, down, std::span<const double>(filter.data(), filter.size()))
    {
    }

    [[nodiscard]] std::size_t output_length(std::size_t input_length) const
    {
        return resampler_.output_length(input_length);
    }

    [[nodiscard]] std::size_t max_output_length(std::size_t input_length) const
    {
        return resampler_.max_output_length(input_length);
    }

    [[nodiscard]] std::size_t max_flush_length() const
    {
        return resampler_.max_flush_length();
    }

    [[nodiscard]] std::size_t initial_output_trim() const
    {
        return resampler_.initial_output_trim();
    }

    void prepare(std::size_t n_channels, std::size_t max_input_samples)
    {
        resampler_.prepare(n_channels, max_input_samples);
        n_channels_ = n_channels;
    }

    py::array_t<double> process(const Array& x)
    {
        if (x.ndim() != 2)
        {
            throw py::value_error("input must be 2D");
        }
        const auto n_samples = static_cast<std::size_t>(x.shape(0));
        const auto n_channels = static_cast<std::size_t>(x.shape(1));
        const auto max_output = resampler_.max_output_length(n_samples);
        std::vector<double> buffer(max_output * n_channels);
        std::size_t written = 0;
        const auto x_span = std::span<const double>(x.data(), x.size());
        const auto output_span = std::span<double>(buffer.data(), buffer.size());
        if (x.size() > kGilReleaseSampleThreshold)
        {
            py::gil_scoped_release release;
            written = resampler_.process(x_span, n_samples, n_channels, output_span);
        }
        else
        {
            written = resampler_.process(x_span, n_samples, n_channels, output_span);
        }
        n_channels_ = n_channels;
        return output_array(buffer, written, n_channels);
    }

    py::array_t<double> flush()
    {
        if (n_channels_ == 0)
        {
            return py::array_t<double>(std::vector<py::ssize_t>{0, 0});
        }
        const auto max_output = resampler_.max_flush_length();
        std::vector<double> buffer(max_output * n_channels_);
        const auto output_span = std::span<double>(buffer.data(), buffer.size());
        const auto written = resampler_.flush(output_span);
        const auto n_channels = n_channels_;
        n_channels_ = 0;
        return output_array(buffer, written, n_channels);
    }

    void reset()
    {
        resampler_.reset();
        n_channels_ = 0;
    }

  private:
    neurale::signal::Resampler resampler_;
    std::size_t n_channels_ = 0;
};

} // namespace

void bind_signal_resampling(py::module_& module)
{
    py::class_<PyResampler>(module, "Resampler")
        .def(py::init<std::size_t, std::size_t, const Array&>(), py::arg("up"), py::arg("down"),
             py::arg("filter"))
        .def("output_length", &PyResampler::output_length, py::arg("input_length"))
        .def("max_output_length", &PyResampler::max_output_length, py::arg("input_length"))
        .def("max_flush_length", &PyResampler::max_flush_length)
        .def("initial_output_trim", &PyResampler::initial_output_trim)
        .def("prepare", &PyResampler::prepare, py::arg("n_channels"), py::arg("max_input_samples"))
        .def("process", &PyResampler::process, py::arg("x"))
        .def("flush", &PyResampler::flush)
        .def("reset", &PyResampler::reset);

    module.def("resampled_length", &neurale::signal::resampled_length, py::arg("input_length"),
               py::arg("up"), py::arg("down"));
}
