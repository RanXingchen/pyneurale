/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/runtime/runtime_info.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using neurale::runtime::BuildInfo;
using neurale::runtime::CpuInfo;
using neurale::runtime::ThreadingInfo;

namespace
{

py::dict build_info_dict(const BuildInfo& info)
{
    py::dict result;
    result["version"] = info.version;
    result["abi_version"] = info.abi_version;
    result["compiler"] = info.compiler;
    result["build_type"] = info.build_type;
    result["cpu_math_backend"] = info.cpu_math_backend;
    result["blas_available"] = info.blas_available;
    result["lapack_available"] = info.lapack_available;
    result["fft_backend"] = info.fft_backend;
    result["openmp_enabled"] = info.openmp_enabled;
    result["cuda_compiled"] = info.cuda_compiled;
    result["cuda_toolkit_version"] =
        info.cuda_toolkit_version ? py::cast(*info.cuda_toolkit_version) : py::none();
    return result;
}

py::dict cpu_info_dict(const CpuInfo& info)
{
    py::dict result;
    result["architecture"] = info.architecture;
    result["vendor"] = info.vendor ? py::cast(*info.vendor) : py::none();
    result["model"] = info.model ? py::cast(*info.model) : py::none();
    result["logical_cores"] = info.logical_cores;
    result["physical_cores"] = py::none();
    return result;
}

py::dict threading_info_dict(const ThreadingInfo& info)
{
    py::dict result;
    result["backend"] = info.backend;
    result["num_threads"] = info.num_threads;
    return result;
}

py::object load_cuda_extension()
{
    return py::module_::import("neurale._native_cuda");
}

py::dict cuda_unavailable(const std::string& reason)
{
    py::dict result;
    result["compiled"] = true;
    result["available"] = false;
    result["runtime_version"] = py::none();
    result["driver_version"] = py::none();
    result["devices"] = py::list();
    result["device_count"] = 0;
    result["reason"] = reason;
    return result;
}

py::dict cuda_info()
{
#ifdef NEURALE_WITH_CUDA_EXTENSION
    try
    {
        return load_cuda_extension().attr("info")().cast<py::dict>();
    }
    catch (const py::error_already_set& error)
    {
        return cuda_unavailable(error.what());
    }
#else
    py::dict result = cuda_unavailable("The native extension was built without CUDA support.");
    result["compiled"] = false;
    return result;
#endif
}

} // namespace

void bind_runtime_module(py::module_& module)
{
    module.def("build_info", [] { return build_info_dict(neurale::runtime::build_info()); });
    module.def("cpu_info", [] { return cpu_info_dict(neurale::runtime::cpu_info()); });
    module.def("threading_info",
               [] { return threading_info_dict(neurale::runtime::threading_info()); });
    module.def("set_num_threads", &neurale::runtime::set_num_threads);

    auto threading = module.def_submodule("threading");
    threading.def("info", [] { return threading_info_dict(neurale::runtime::threading_info()); });
    threading.def("get_num_threads", [] { return neurale::runtime::threading_info().num_threads; });
    threading.def("set_num_threads", &neurale::runtime::set_num_threads);
    threading.def("backend", [] { return neurale::runtime::threading_info().backend; });

    auto cuda = module.def_submodule("cuda");
    cuda.def("build_info",
             []
             {
                 const auto build = neurale::runtime::build_info();
                 py::dict result;
                 result["compiled"] = build.cuda_compiled;
                 result["toolkit_version"] = build.cuda_toolkit_version
                                                 ? py::cast(*build.cuda_toolkit_version)
                                                 : py::none();
                 return result;
             });
    cuda.def("info", &cuda_info);
    cuda.def("is_available", [] { return cuda_info()["available"].cast<bool>(); });
    cuda.def("device_count", [] { return cuda_info()["device_count"].cast<std::size_t>(); });
    cuda.def("devices", [] { return cuda_info()["devices"]; });
    cuda.def("current_device",
             []
             {
#ifdef NEURALE_WITH_CUDA_EXTENSION
                 return load_cuda_extension().attr("current_device")().cast<std::int32_t>();
#else
        throw std::runtime_error(
            "The native extension was built without CUDA support.");
#endif
             });
    cuda.def("set_device",
             [](std::int32_t ordinal)
             {
#ifdef NEURALE_WITH_CUDA_EXTENSION
                 load_cuda_extension().attr("set_device")(ordinal);
#else
        (void)ordinal;
        throw std::runtime_error(
            "The native extension was built without CUDA support.");
#endif
             });
}
