/* SPDX-License-Identifier: MIT */
#include "array_replay_bindings.h"
#include <cstring>
#include <neurale/streaming/array_replay.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace neurale::streaming;

void bind_array_replay(py::module_& module)
{
    py::class_<ArrayReplaySource, NativeFrameSource>(module, "ArrayReplaySource", py::is_final())
        .def(
            py::init(
                [](const StreamSchema& schema, py::array_t<double, py::array::c_style> data,
                   bool paced, SessionId session_id)
                {
                    if (data.ndim() != 2 || schema.signals().size() != 1 ||
                        data.shape(1) != schema.signals().front().n_channels)
                        throw py::value_error("data must have shape (samples, schema channels).");
                    return std::make_unique<ArrayReplaySource>(
                        schema,
                        std::span<const double>{data.data(), static_cast<std::size_t>(data.size())},
                        paced, session_id);
                }),
            py::arg("schema"), py::arg("data").noconvert(), py::arg("paced") = true,
            py::arg("session_id") = 1,
            "Copy float64 C-contiguous sampled data before running; emit complete blocks natively.")
        .def_property_readonly(
            "timings",
            [](const ArrayReplaySource& source)
            {
                const auto rows = source.timings();
                py::array_t<std::uint64_t> result(
                    {static_cast<py::ssize_t>(rows.size()), py::ssize_t{3}});
                auto out = result.mutable_unchecked<2>();
                for (std::size_t i = 0; i < rows.size(); ++i)
                {
                    out(i, 0) = rows[i].sample_idx_start;
                    out(i, 1) = rows[i].planned_ns;
                    out(i, 2) = rows[i].ready_ns;
                }
                return result;
            },
            "Copied rows: sample_idx_start, planned_ns, ready_ns. Reset only when quiescent.");

    py::class_<NativeResultSink, NativeFrameConsumer>(module, "NativeResultSink", py::is_final())
        .def(py::init<const StreamSchema&, std::size_t, std::size_t, std::size_t, std::size_t>(),
             py::arg("schema"), py::arg("capacity"), py::kw_only(), py::arg("window_samples") = 0,
             py::arg("hop_samples") = 0, py::arg("output_channels") = 5,
             "Capture values, or capture zero outputs at a window/hop cadence for a native "
             "null-path baseline.")
        .def(
            "snapshot",
            [](const NativeResultSink& sink)
            {
                const auto rows = sink.timings();
                py::array_t<std::uint64_t> timing(
                    {static_cast<py::ssize_t>(rows.size()), py::ssize_t{4}});
                auto out = timing.mutable_unchecked<2>();
                for (std::size_t i = 0; i < rows.size(); ++i)
                {
                    out(i, 0) = rows[i].observation_index;
                    out(i, 1) = rows[i].source_tick;
                    out(i, 2) = rows[i].source_received_ns;
                    out(i, 3) = rows[i].delivered_ns;
                }
                py::array_t<double> values({static_cast<py::ssize_t>(rows.size()),
                                            static_cast<py::ssize_t>(sink.n_channels())});
                const auto data = sink.values(rows.size());
                if (!data.empty())
                    std::memcpy(values.mutable_data(), data.data(), data.size_bytes());
                return py::make_tuple(values, timing);
            },
            "Copy (values, timing); timing columns: observation_index, source_tick, "
            "source_received_ns, delivered_ns.");
}
