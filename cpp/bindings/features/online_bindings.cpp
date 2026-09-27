/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/features/online.h>

#include <cstddef>
#include <span>
#include <string_view>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<std::size_t, py::array::c_style | py::array::forcecast>;

neurale::signal::MultitaperWeighting parse_weighting(std::string_view value)
{
    if (value == "unity")
    {
        return neurale::signal::MultitaperWeighting::unity;
    }
    if (value == "eigen")
    {
        return neurale::signal::MultitaperWeighting::eigen;
    }
    if (value == "adaptive")
    {
        return neurale::signal::MultitaperWeighting::adaptive;
    }
    throw py::value_error("weighting must be 'unity', 'eigen', or 'adaptive'");
}

neurale::features::Detrend parse_detrend(std::string_view value)
{
    if (value == "none")
    {
        return neurale::features::Detrend::none;
    }
    if (value == "mean")
    {
        return neurale::features::Detrend::mean;
    }
    if (value == "linear")
    {
        return neurale::features::Detrend::linear;
    }
    throw py::value_error("detrend must be 'none', 'mean', or 'linear'");
}

neurale::signal::SpectralBackend parse_backend(std::string_view value)
{
    if (value == "auto")
    {
        return neurale::signal::SpectralBackend::automatic;
    }
    if (value == "builtin")
    {
        return neurale::signal::SpectralBackend::builtin;
    }
    throw py::value_error("backend must be 'auto' or 'builtin'");
}

template <typename Processor> RealArray process_block(Processor& processor, const RealArray& input)
{
    if (input.ndim() != 2)
    {
        throw py::value_error("online feature input must be 2D");
    }
    const auto input_samples = static_cast<std::size_t>(input.shape(0));
    const auto output_rows = processor.output_count(input_samples);
    RealArray output({
        static_cast<py::ssize_t>(output_rows),
        static_cast<py::ssize_t>(processor.feature_count()),
    });
    {
        py::gil_scoped_release release;
        processor.process(std::span<const double>(input.data(), input.size()), input_samples,
                          std::span<double>(output.mutable_data(), output.size()));
    }
    return output;
}

template <typename Processor> void bind_common(py::class_<Processor>& binding)
{
    binding.def("process", &process_block<Processor>, py::arg("input"))
        .def("reset", &Processor::reset)
        .def("output_count", &Processor::output_count, py::arg("input_samples"))
        .def_property_readonly("feature_count", &Processor::feature_count)
        .def_property_readonly("window_samples", &Processor::window_samples)
        .def_property_readonly("hop_samples", &Processor::hop_samples);
}

} // namespace

void bind_features_online(py::module_& module)
{
    py::class_<neurale::features::BandpowerProcessor> bandpower(module, "BandpowerProcessor");
    bandpower.def(
        py::init(
            [](std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
               std::size_t fft_length, double density_scale, const RealArray& tapers,
               std::size_t n_tapers, const RealArray& concentration_ratios,
               std::string_view weighting, const IndexArray& band_bins, double freq_step,
               std::string_view detrend, std::string_view backend)
            {
                if (tapers.ndim() != 2 || concentration_ratios.ndim() != 1)
                {
                    throw py::value_error("tapers must be 2D and ratios 1D");
                }
                if (band_bins.ndim() != 2 || band_bins.shape(1) != 2)
                {
                    throw py::value_error("band_bins must have shape (n_bands, 2)");
                }
                return neurale::features::BandpowerProcessor(
                    window_samples, hop_samples, n_channels, fft_length, density_scale,
                    std::span<const double>(tapers.data(), tapers.size()), n_tapers,
                    std::span<const double>(concentration_ratios.data(),
                                            concentration_ratios.size()),
                    parse_weighting(weighting),
                    std::span<const std::size_t>(band_bins.data(), band_bins.size()), freq_step,
                    parse_detrend(detrend), parse_backend(backend));
            }),
        py::arg("window_samples"), py::arg("hop_samples"), py::arg("n_channels"),
        py::arg("fft_length"), py::arg("density_scale"), py::arg("tapers"), py::arg("n_tapers"),
        py::arg("concentration_ratios"), py::arg("weighting"), py::arg("band_bins"),
        py::arg("freq_step"), py::arg("detrend"), py::arg("backend"));
    bind_common(bandpower);

    py::class_<neurale::features::HilbertEnvelopeProcessor> hilbert(module,
                                                                    "HilbertEnvelopeProcessor");
    hilbert.def(
        py::init(
            [](std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
               const RealArray& sos, std::size_t fft_length, std::string_view backend)
            {
                if (sos.ndim() != 3 || sos.shape(0) == 0 || sos.shape(1) == 0 || sos.shape(2) != 6)
                {
                    throw py::value_error(
                        "Hilbert SOS bank must have shape (n_bands, n_sections, 6)");
                }
                return neurale::features::HilbertEnvelopeProcessor(
                    window_samples, hop_samples, n_channels,
                    std::span<const double>(sos.data(), sos.size()),
                    static_cast<std::size_t>(sos.shape(0)), static_cast<std::size_t>(sos.shape(1)),
                    fft_length, parse_backend(backend));
            }),
        py::arg("window_samples"), py::arg("hop_samples"), py::arg("n_channels"), py::arg("sos"),
        py::arg("fft_length"), py::arg("backend"));
    bind_common(hilbert);

    py::class_<neurale::features::LmpProcessor> lmp(module, "LmpProcessor");
    lmp.def(py::init(
                [](std::size_t window_samples, std::size_t hop_samples, std::size_t n_channels,
                   const RealArray& sos)
                {
                    if (sos.ndim() != 2 || sos.shape(0) == 0 || sos.shape(1) != 6)
                    {
                        throw py::value_error("LMP SOS must have shape (n_sections, 6)");
                    }
                    return neurale::features::LmpProcessor(
                        window_samples, hop_samples, n_channels,
                        std::span<const double>(sos.data(), sos.size()),
                        static_cast<std::size_t>(sos.shape(0)));
                }),
            py::arg("window_samples"), py::arg("hop_samples"), py::arg("n_channels"),
            py::arg("sos"));
    bind_common(lmp);
}
