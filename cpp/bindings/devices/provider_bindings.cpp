/* SPDX-License-Identifier: MIT */
#include <neurale/devices/provider.h>
#include <provider.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace neurale::devices;
using namespace neurale::streaming;
namespace
{
pn_block_v1 block_info(uint32_t signal, const py::dict& info)
{
    pn_block_v1 b{};
    b.struct_size = sizeof(b);
    b.signal_index = signal;
    for (auto item : info)
    {
        const auto key = py::cast<std::string>(item.first);
        const auto value = py::cast<uint64_t>(item.second);
        if (key == "sample_index")
        {
            b.sample_index = value;
            b.flags |= PN_HAS_SAMPLE_INDEX;
        }
        else if (key == "device_tick")
        {
            b.device_tick = value;
            b.flags |= PN_HAS_TICK;
        }
        else if (key == "missing_samples")
        {
            b.missing_samples = value;
            b.flags |= PN_KNOWN_LOSS;
        }
        else if (key == "tick_reference")
        {
            b.tick_reference = value;
            b.flags |= PN_HAS_CLOCK_SYNC;
        }
        else if (key == "host_reference_ns")
            b.host_reference_ns = value;
        else if (key == "tick_rate_numerator")
            b.tick_rate_numerator = value;
        else if (key == "tick_rate_denominator")
            b.tick_rate_denominator = value;
        else if (key == "uncertainty_ns")
            b.uncertainty_ns = value;
        else if (key == "clock_generation" && value <= UINT32_MAX)
            b.clock_generation = static_cast<uint32_t>(value);
        else if (key == "synchronized" && value <= 1)
            b.synchronized = static_cast<uint32_t>(value);
        else
            throw py::value_error("unknown or invalid device block metadata: " + key);
    }
    return b;
}
pn_block_v1 array_block(DeviceIngress& self, uint32_t signal_index, const py::array& data,
                        const py::dict& info)
{
    if (signal_index >= self.signals().size())
        throw py::value_error("invalid signal index");
    const auto& s = self.signals()[signal_index];
    const char* format[] = {"int16", "int32", "float32", "float64"};
    if (!data.dtype().is(py::dtype(format[static_cast<unsigned>(s.dtype)])) ||
        !(data.flags() & py::array::c_style) || data.ndim() != 2)
        throw py::value_error("device data must have exact native dtype and contiguous "
                              "two-dimensional layout");
    const auto channel_axis = s.layout == SignalLayout::sample_major ? 1 : 0;
    const auto samples = data.shape(1 - channel_axis);
    if (data.shape(channel_axis) != s.n_channels || samples <= 0 || samples > s.max_block_samples)
        throw py::value_error("device data shape exceeds declared signal");
    auto b = block_info(signal_index, info);
    b.samples = static_cast<uint32_t>(samples);
    return b;
}
void check(pn_status status)
{
    if (status != PN_OK)
        throw std::runtime_error("device ingress rejected block: " + std::to_string(status));
}
} // namespace
void bind_device_provider(py::module_& devices)
{
    devices.def("host_time_ns", [] { return default_native_clock().now_ns(); });
    py::class_<DeviceIngress, std::shared_ptr<DeviceIngress>>(devices, "DeviceIngress")
        .def(py::init<const std::string&, std::vector<SignalSchema>, std::size_t, bool>(),
             py::arg("path"), py::arg("signals"), py::arg("capacity"), py::arg("create") = false)
        .def(
            "publish",
            [](DeviceIngress& self, uint32_t signal_index, py::array data, py::dict info)
            {
                auto b = array_block(self, signal_index, data, info);
                pn_status status;
                {
                    py::gil_scoped_release release;
                    status = self.publish(b, data.data(), data.nbytes());
                }
                check(status);
            },
            py::arg("signal_index"), py::arg("data").noconvert(), py::arg("metadata") = py::dict())
        .def(
            "publish_frame",
            [](DeviceIngress& self, const py::sequence& items)
            {
                std::vector<py::array> arrays;
                std::vector<pn_block_v1> metadata;
                std::vector<DeviceIngress::Block> blocks;
                const auto count = py::len(items);
                if (!count || count > self.signals().size())
                    throw py::value_error("invalid frame block count");
                arrays.reserve(count);
                metadata.reserve(count);
                blocks.reserve(count);
                for (auto item : items)
                {
                    auto values = py::cast<py::tuple>(item);
                    if (values.size() != 3 || !py::isinstance<py::array>(values[1]))
                        throw py::value_error(
                            "blocks must be (signal_index, ndarray, metadata) tuples");
                    arrays.push_back(py::cast<py::array>(values[1]));
                    metadata.push_back(array_block(self, py::cast<uint32_t>(values[0]),
                                                   arrays.back(), py::cast<py::dict>(values[2])));
                    blocks.push_back({&metadata.back(), arrays.back().data(),
                                      static_cast<uint64_t>(arrays.back().nbytes())});
                }
                pn_status status;
                {
                    py::gil_scoped_release release;
                    status = self.publish_frame(blocks);
                }
                check(status);
            },
            py::arg("blocks"))
        .def(
            "gap",
            [](DeviceIngress& self, uint32_t signal, bool restart, py::dict info)
            {
                auto b = block_info(signal, info);
                b.kind = restart ? PN_RESTART : PN_GAP;
                check(self.publish(b, nullptr, 0));
            },
            py::arg("signal_index"), py::arg("restart") = false, py::arg("metadata") = py::dict())
        .def("finish", [](DeviceIngress& self) { self.finish(PN_OK); })
        .def("fail", [](DeviceIngress& self) { self.finish(PN_FAILED); })
        .def("cancel", &DeviceIngress::cancel)
        .def("remove_directory_on_destroy", &DeviceIngress::remove_directory_on_destroy)
        .def_property_readonly("cancelled", &DeviceIngress::cancelled)
        .def_property_readonly("stats",
                               [](const DeviceIngress& self)
                               {
                                   py::dict d;
                                   d["published"] = self.published();
                                   d["consumed"] = self.consumed();
                                   d["error"] = self.error();
                                   return d;
                               });
    py::class_<GenericDeviceSource, NativeFrameSource>(devices, "GenericDeviceSource")
        .def(py::init<const std::string&, const std::string&, const std::string&, std::size_t>())
        .def(py::init<std::vector<SignalSchema>, const std::string&, std::size_t>())
        .def_property_readonly("schema", [](const GenericDeviceSource& self)
                               { return self.schema().clone(); })
        .def_property_readonly("_native_session_id", &GenericDeviceSource::session_id)
        .def_property_readonly("ingress", &GenericDeviceSource::ingress)
        .def("start", &GenericDeviceSource::start, py::call_guard<py::gil_scoped_release>())
        .def("cancel", &GenericDeviceSource::cancel)
        .def("close", &GenericDeviceSource::close, py::call_guard<py::gil_scoped_release>())
        .def("reset", &GenericDeviceSource::reset, py::call_guard<py::gil_scoped_release>())
        .def("set_reset_action",
             [](GenericDeviceSource& self, py::function action)
             {
                 self.set_reset_action(
                     [action = std::move(action)]
                     {
                         py::gil_scoped_acquire acquire;
                         try
                         {
                             action();
                         }
                         catch (py::error_already_set& error)
                         {
                             const std::string message = error.what();
                             throw std::runtime_error(message);
                         }
                     });
             });
}
