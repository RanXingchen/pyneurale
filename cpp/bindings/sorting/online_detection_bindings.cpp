/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "support.h"

#include <neurale/sorting/online_detection.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

using neurale::bindings::sorting::numeric_array;

std::string polarity_name(neurale::sorting::DetectionPolarity polarity)
{
    switch (polarity)
    {
    case neurale::sorting::DetectionPolarity::Negative:
        return "negative";
    case neurale::sorting::DetectionPolarity::Positive:
        return "positive";
    case neurale::sorting::DetectionPolarity::Both:
        return "both";
    }
    throw py::value_error("online spike block polarity is invalid");
}

py::dict decode(py::buffer payload)
{
    const auto info = payload.request();
    if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1)
        throw py::value_error("online spike payload must be a contiguous byte buffer");
    const auto snapshot = neurale::sorting::decode_spike_block(
        {static_cast<const std::byte*>(info.ptr), static_cast<std::size_t>(info.size)});
    auto waveforms =
        py::array_t<double>({static_cast<py::ssize_t>(snapshot.header.n_valid),
                             static_cast<py::ssize_t>(snapshot.header.waveform_samples),
                             static_cast<py::ssize_t>(snapshot.header.n_channels)});
    if (!snapshot.waveforms.empty())
        std::memcpy(waveforms.mutable_data(), snapshot.waveforms.data(),
                    snapshot.waveforms.size() * sizeof(double));
    py::dict result;
    result["waveforms"] = std::move(waveforms);
    result["sample_indices"] = numeric_array(snapshot.sample_indices);
    result["times"] = numeric_array(snapshot.times);
    result["peak_channel_indices"] = numeric_array(snapshot.peak_channel_indices);
    result["electrode_group_ids"] = numeric_array(snapshot.electrode_group_ids);
    result["amps"] = numeric_array(snapshot.amps);
    result["scores"] = numeric_array(snapshot.scores);
    result["spike_polarities"] = numeric_array(snapshot.polarities);
    result["segment_id"] = snapshot.header.segment_id;
    result["pre_samples"] = snapshot.header.pre_samples;
    result["post_samples"] = snapshot.header.post_samples;
    result["polarity"] = polarity_name(snapshot.header.detection_polarity);
    result["capacity"] = snapshot.header.capacity;
    result["overflow_count"] = snapshot.header.overflow_count;
    result["overflowed"] = snapshot.header.flags == neurale::sorting::SpikeBlockFlags::overflowed;
    return result;
}

} // namespace

void bind_sorting_online_detection(py::module_& module)
{
    module.def("decode_spike_block", &decode, py::arg("payload"),
               "Copy one fixed-capacity native spike block into offline arrays.");
}
