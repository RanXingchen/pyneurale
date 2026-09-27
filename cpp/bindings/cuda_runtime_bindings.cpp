/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/runtime/runtime_info.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

py::dict cuda_info_dict(const neurale::runtime::CudaInfo& info)
{
    py::dict result;
    result["compiled"] = info.compiled;
    result["available"] = info.available;
    result["runtime_version"] = info.runtime_version ? py::cast(*info.runtime_version) : py::none();
    result["driver_version"] = info.driver_version ? py::cast(*info.driver_version) : py::none();
    result["reason"] = info.reason ? py::cast(*info.reason) : py::none();

    py::list devices;
    for (const auto& device : info.devices)
    {
        py::dict item;
        item["ordinal"] = device.ordinal;
        item["name"] = device.name;
        item["total_memory"] = device.total_memory;
        item["compute_capability"] =
            py::make_tuple(device.compute_capability_major, device.compute_capability_minor);
        item["multiprocessor_count"] = device.n_multiprocessors;
        devices.append(std::move(item));
    }
    result["devices"] = std::move(devices);
    result["device_count"] = info.devices.size();
    return result;
}

} // namespace

PYBIND11_MODULE(_native_cuda, module)
{
    module.doc() = "Lazy CUDA Runtime probing for PyNeurale.";
    module.def("info", [] { return cuda_info_dict(neurale::runtime::cuda_info()); });
    module.def("current_device", &neurale::runtime::cuda_current_device);
    module.def("set_device", &neurale::runtime::cuda_set_device);
}
