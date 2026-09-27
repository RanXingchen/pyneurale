/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/runtime/runtime_info.h>

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>

namespace neurale::runtime
{

CudaInfo cuda_info()
{
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess)
    {
        return CudaInfo{
            true,         false, std::nullopt,
            std::nullopt, {},    std::string(cudaGetErrorString(count_status)),
        };
    }

    int runtime_version = 0;
    int driver_version = 0;
    const cudaError_t runtime_status = cudaRuntimeGetVersion(&runtime_version);
    const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);

    std::vector<CudaDeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        cudaDeviceProp properties{};
        const cudaError_t status = cudaGetDeviceProperties(&properties, ordinal);
        if (status != cudaSuccess)
        {
            return CudaInfo{
                true,
                false,
                runtime_status == cudaSuccess ? std::optional<std::int32_t>(runtime_version)
                                              : std::nullopt,
                driver_status == cudaSuccess ? std::optional<std::int32_t>(driver_version)
                                             : std::nullopt,
                std::move(devices),
                std::string(cudaGetErrorString(status)),
            };
        }
        devices.push_back(CudaDeviceInfo{
            ordinal,
            properties.name,
            static_cast<std::uint64_t>(properties.totalGlobalMem),
            properties.major,
            properties.minor,
            properties.multiProcessorCount,
        });
    }

    return CudaInfo{
        true,
        count > 0,
        runtime_status == cudaSuccess ? std::optional<std::int32_t>(runtime_version) : std::nullopt,
        driver_status == cudaSuccess ? std::optional<std::int32_t>(driver_version) : std::nullopt,
        std::move(devices),
        count > 0 ? std::nullopt : std::optional<std::string>("No CUDA devices were found."),
    };
}

std::int32_t cuda_current_device()
{
    int ordinal = 0;
    const cudaError_t status = cudaGetDevice(&ordinal);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return ordinal;
}

void cuda_set_device(std::int32_t ordinal)
{
    if (ordinal < 0)
    {
        throw std::invalid_argument("CUDA device ordinal must be non-negative");
    }
    const cudaError_t status = cudaSetDevice(ordinal);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

} // namespace neurale::runtime
