/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/sorting/detection.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

using neurale::bindings::sorting::checked_product;
using neurale::bindings::sorting::checked_ssize;
using neurale::bindings::sorting::index_array;
using neurale::bindings::sorting::numeric_array;

neurale::sorting::DetectionPolarity parse_polarity(const std::string& value)
{
    if (value == "negative")
    {
        return neurale::sorting::DetectionPolarity::Negative;
    }
    if (value == "positive")
    {
        return neurale::sorting::DetectionPolarity::Positive;
    }
    if (value == "both")
    {
        return neurale::sorting::DetectionPolarity::Both;
    }
    throw py::value_error("polarity must be 'negative', 'positive', or 'both'");
}

neurale::sorting::BoundaryBehavior parse_boundary(const std::string& value)
{
    if (value == "drop")
    {
        return neurale::sorting::BoundaryBehavior::Drop;
    }
    if (value == "raise")
    {
        return neurale::sorting::BoundaryBehavior::Raise;
    }
    throw py::value_error("boundary_behavior must be 'drop' or 'raise'");
}

void require_detection_array(const py::array& input)
{
    if (input.ndim() != 2)
    {
        throw py::value_error("threshold detection input must be 2D");
    }
    if (!input.dtype().is(py::dtype::of<double>()))
    {
        throw py::type_error("threshold detection input dtype must be float64");
    }
    if ((input.flags() & py::array::c_style) == 0)
    {
        throw py::value_error("threshold detection input must be C-contiguous sample-major data");
    }
}

neurale::sorting::ThresholdDetectionConfig
detection_config(double threshold_multiplier, std::size_t refractory_samples,
                 std::size_t alignment_search_radius, std::size_t pre_samples,
                 std::size_t post_samples, const std::string& polarity,
                 const std::string& boundary_behavior,
                 const std::optional<std::vector<std::vector<std::size_t>>>& electrode_groups)
{
    return neurale::sorting::ThresholdDetectionConfig{
        .threshold_multiplier = threshold_multiplier,
        .refractory_samples = refractory_samples,
        .alignment_search_radius = alignment_search_radius,
        .pre_samples = pre_samples,
        .post_samples = post_samples,
        .polarity = parse_polarity(polarity),
        .boundary_behavior = parse_boundary(boundary_behavior),
        .electrode_groups = electrode_groups.value_or(std::vector<std::vector<std::size_t>>{}),
    };
}

py::dict result_dict(const neurale::sorting::ThresholdDetectionResult& result)
{
    auto indices = [&](const std::vector<std::size_t>& values)
    {
        return index_array(values, "threshold detection result is too large",
                           "threshold detection index does not fit int64");
    };
    py::dict output;
    output["sample_indices"] = indices(result.sample_indices);
    output["crossing_indices"] = indices(result.crossing_indices);
    output["peak_channel_indices"] = indices(result.peak_channel_indices);
    output["electrode_group_ids"] = indices(result.electrode_group_ids);
    output["amps"] = numeric_array(result.amps);
    output["scores"] = numeric_array(result.scores);
    output["polarities"] = numeric_array(result.polarities);
    output["channel_centers"] = numeric_array(result.channel_centers);
    output["channel_noise"] = numeric_array(result.channel_noise);
    output["channel_thresholds"] = numeric_array(result.channel_thresholds);
    return output;
}

py::dict
detect_threshold(const py::array& input, double threshold_multiplier,
                 std::size_t refractory_samples, std::size_t alignment_search_radius,
                 std::size_t pre_samples, std::size_t post_samples, const std::string& polarity,
                 const std::string& boundary_behavior,
                 const std::optional<std::vector<std::vector<std::size_t>>>& electrode_groups)
{
    require_detection_array(input);
    const auto config =
        detection_config(threshold_multiplier, refractory_samples, alignment_search_radius,
                         pre_samples, post_samples, polarity, boundary_behavior, electrode_groups);
    const auto n_samples = static_cast<std::size_t>(input.shape(0));
    const auto n_channels = static_cast<std::size_t>(input.shape(1));
    const auto* data = static_cast<const double*>(input.data());
    neurale::sorting::ThresholdDetectionResult result;
    {
        py::gil_scoped_release release;
        result = neurale::sorting::detect_threshold(
            std::span<const double>(data, static_cast<std::size_t>(input.size())), n_samples,
            n_channels, config);
    }

    return result_dict(result);
}

py::array extract_waveforms(std::span<const std::span<const double>> chunks, std::size_t n_channels,
                            std::size_t pre_samples, std::size_t post_samples,
                            std::span<const std::size_t> sample_indices)
{
    if (post_samples == std::numeric_limits<std::size_t>::max() ||
        pre_samples > std::numeric_limits<std::size_t>::max() - post_samples - 1)
    {
        throw std::overflow_error("threshold waveform length is too large");
    }
    const std::size_t complete_waveform_samples = pre_samples + 1 + post_samples;
    const std::size_t values_per_waveform = checked_product(
        complete_waveform_samples, n_channels, "threshold waveform payload is too large");
    const std::size_t n_values = checked_product(sample_indices.size(), values_per_waveform,
                                                 "threshold waveform payload is too large");
    const std::size_t n_bytes =
        checked_product(n_values, sizeof(double), "threshold waveform payload is too large");
    const py::ssize_t python_byte_count =
        checked_ssize(n_bytes, "threshold waveform payload is too large");
    PyObject* raw_backing = PyBytes_FromStringAndSize(nullptr, python_byte_count);
    if (raw_backing == nullptr)
    {
        throw py::error_already_set();
    }
    py::bytes backing = py::reinterpret_steal<py::bytes>(raw_backing);
    auto* waveform_data = reinterpret_cast<double*>(PyBytes_AS_STRING(backing.ptr()));
    {
        py::gil_scoped_release release;
        neurale::sorting::extract_threshold_waveforms(chunks, n_channels, pre_samples, post_samples,
                                                      sample_indices,
                                                      std::span<double>(waveform_data, n_values));
    }
    const std::array<py::ssize_t, 3> shape = {
        checked_ssize(sample_indices.size(), "threshold event count is too large"),
        checked_ssize(complete_waveform_samples, "threshold waveform length is too large"),
        checked_ssize(n_channels, "threshold channel count is too large"),
    };
    const std::array<py::ssize_t, 3> strides = {
        checked_ssize(checked_product(values_per_waveform, sizeof(double),
                                      "threshold waveform stride is too large"),
                      "threshold waveform stride is too large"),
        checked_ssize(
            checked_product(n_channels, sizeof(double), "threshold waveform stride is too large"),
            "threshold waveform stride is too large"),
        static_cast<py::ssize_t>(sizeof(double)),
    };
    py::array waveforms(py::dtype::of<double>(), shape, strides, waveform_data, backing);
    waveforms.attr("setflags")(false);
    return waveforms;
}

py::array extract_threshold_waveforms(const py::array& input, const py::array& sample_idx_values,
                                      std::size_t pre_samples, std::size_t post_samples)
{
    require_detection_array(input);
    if (sample_idx_values.ndim() != 1 ||
        !sample_idx_values.dtype().is(py::dtype::of<std::int64_t>()))
    {
        throw py::type_error("threshold waveform sample_indices must be 1D int64");
    }
    if ((sample_idx_values.flags() & py::array::c_style) == 0)
    {
        throw py::value_error("threshold waveform sample_indices must be C-contiguous");
    }
    const auto* idx_data = static_cast<const std::int64_t*>(sample_idx_values.data());
    std::vector<std::size_t> sample_indices;
    sample_indices.reserve(static_cast<std::size_t>(sample_idx_values.size()));
    for (py::ssize_t i = 0; i < sample_idx_values.size(); ++i)
    {
        if (idx_data[i] < 0)
        {
            throw py::value_error("threshold waveform sample_indices must be non-negative");
        }
        sample_indices.push_back(static_cast<std::size_t>(idx_data[i]));
    }
    const std::size_t n_channels = static_cast<std::size_t>(input.shape(1));
    const std::array chunks = {std::span<const double>(static_cast<const double*>(input.data()),
                                                       static_cast<std::size_t>(input.size()))};
    return extract_waveforms(chunks, n_channels, pre_samples, post_samples, sample_indices);
}

py::dict detect_threshold_waveforms(
    const py::sequence& chunk_values, double threshold_multiplier, std::size_t refractory_samples,
    std::size_t alignment_search_radius, std::size_t pre_samples, std::size_t post_samples,
    const std::string& polarity, const std::string& boundary_behavior,
    const std::optional<std::vector<std::vector<std::size_t>>>& electrode_groups)
{
    if (chunk_values.empty())
    {
        throw py::value_error("threshold detection chunks must not be empty");
    }
    std::vector<py::array> arrays;
    arrays.reserve(static_cast<std::size_t>(chunk_values.size()));
    std::vector<std::span<const double>> chunks;
    chunks.reserve(static_cast<std::size_t>(chunk_values.size()));
    std::optional<std::size_t> n_channels;
    for (const py::handle value : chunk_values)
    {
        if (!py::isinstance<py::array>(value))
        {
            throw py::type_error("threshold detection chunks must contain numpy arrays");
        }
        arrays.push_back(py::reinterpret_borrow<py::array>(value));
        const py::array& arr = arrays.back();
        require_detection_array(arr);
        const auto channels = static_cast<std::size_t>(arr.shape(1));
        if (n_channels.has_value() && channels != *n_channels)
        {
            throw py::value_error("threshold detection chunks must have equal channel counts");
        }
        n_channels = channels;
        chunks.emplace_back(static_cast<const double*>(arr.data()),
                            static_cast<std::size_t>(arr.size()));
    }
    const auto config =
        detection_config(threshold_multiplier, refractory_samples, alignment_search_radius,
                         pre_samples, post_samples, polarity, boundary_behavior, electrode_groups);
    neurale::sorting::ThresholdDetectionResult result;
    {
        py::gil_scoped_release release;
        result = neurale::sorting::detect_threshold_chunk_events(chunks, *n_channels, config);
    }
    py::dict output = result_dict(result);
    output["waveforms"] =
        extract_waveforms(chunks, *n_channels, pre_samples, post_samples, result.sample_indices);
    return output;
}

} // namespace

void bind_sorting_detection(py::module_& module)
{
    module.def("_detect_threshold", &detect_threshold, py::arg("input"),
               py::arg("threshold_multiplier"), py::arg("refractory_samples"),
               py::arg("alignment_search_radius"), py::arg("pre_samples"), py::arg("post_samples"),
               py::arg("polarity"), py::arg("boundary_behavior"),
               py::arg("electrode_groups") = py::none());
    module.def("_detect_threshold_waveforms", &detect_threshold_waveforms, py::arg("chunks"),
               py::arg("threshold_multiplier"), py::arg("refractory_samples"),
               py::arg("alignment_search_radius"), py::arg("pre_samples"), py::arg("post_samples"),
               py::arg("polarity"), py::arg("boundary_behavior"),
               py::arg("electrode_groups") = py::none());
    module.def("_extract_threshold_waveforms", &extract_threshold_waveforms, py::arg("input"),
               py::arg("sample_indices"), py::arg("pre_samples"), py::arg("post_samples"));
}
