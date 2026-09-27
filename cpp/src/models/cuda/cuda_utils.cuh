/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/models/cuda/errors.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace neurale::models::cuda::detail
{

inline bool device_unavailable(cudaError_t status) noexcept
{
    return status == cudaErrorNoDevice || status == cudaErrorInvalidDevice ||
           status == cudaErrorInsufficientDriver || status == cudaErrorInitializationError ||
           status == cudaErrorDevicesUnavailable;
}

inline void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
        if (device_unavailable(status))
        {
            throw DeviceUnavailableError(message);
        }
        throw std::runtime_error(message);
    }
}

class DeviceGuard
{
  public:
    explicit DeviceGuard(int requested_device)
    {
        check_cuda(cudaGetDevice(&previous_device_), "CUDA current-device query failed");
        device_id_ = requested_device < 0 ? previous_device_ : requested_device;
        if (device_id_ != previous_device_)
        {
            check_cuda(cudaSetDevice(device_id_), "CUDA device selection failed");
            restore_ = true;
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    ~DeviceGuard()
    {
        if (restore_)
        {
            cudaSetDevice(previous_device_);
        }
    }

    [[nodiscard]] int device_id() const noexcept
    {
        return device_id_;
    }

  private:
    int previous_device_{};
    int device_id_{};
    bool restore_{};
};

inline void free_device_memory(void* allocation, int device_id, bool pooled) noexcept
{
    if (allocation == nullptr)
    {
        return;
    }
    int previous_device = -1;
    const bool restore = cudaGetDevice(&previous_device) == cudaSuccess &&
                         previous_device != device_id && cudaSetDevice(device_id) == cudaSuccess;
#if CUDART_VERSION >= 11020
    if (pooled)
    {
        cudaFreeAsync(allocation, nullptr);
    }
    else
    {
        cudaFree(allocation);
    }
#else
    (void)pooled;
    cudaFree(allocation);
#endif
    if (restore)
    {
        cudaSetDevice(previous_device);
    }
}

template <typename T> class DeviceBuffer
{
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t size)
    {
        resize(size);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          device_id_(std::exchange(other.device_id_, -1))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            device_id_ = std::exchange(other.device_id_, -1);
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        release();
    }

    void resize(std::size_t size)
    {
        if (size <= size_)
        {
            return;
        }
        release();
        if (size == 0)
        {
            return;
        }
        int device_id = 0;
        check_cuda(cudaGetDevice(&device_id), "CUDA current-device query failed");
        void* allocation = nullptr;
        check_cuda(cudaMalloc(&allocation, size * sizeof(T)), "CUDA allocation failed");
        data_ = static_cast<T*>(allocation);
        size_ = size;
        device_id_ = device_id;
    }

    void copy_from(const T* source, std::size_t size)
    {
        resize(size);
        if (size != 0)
        {
            check_cuda(cudaMemcpy(data_, source, size * sizeof(T), cudaMemcpyHostToDevice),
                       "CUDA host-to-device copy failed");
        }
    }

    void copy_to(T* destination, std::size_t size) const
    {
        if (size != 0)
        {
            check_cuda(cudaMemcpy(destination, data_, size * sizeof(T), cudaMemcpyDeviceToHost),
                       "CUDA device-to-host copy failed");
        }
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] const T* data() const noexcept
    {
        return data_;
    }

  private:
    void release() noexcept
    {
        free_device_memory(data_, device_id_, false);
        data_ = nullptr;
        size_ = 0;
        device_id_ = -1;
    }

    T* data_{};
    std::size_t size_{};
    int device_id_{-1};
};

template <typename T> class PooledDeviceBuffer
{
  public:
    explicit PooledDeviceBuffer(std::size_t size)
    {
        if (size == 0)
        {
            return;
        }
        check_cuda(cudaGetDevice(&device_id_), "CUDA current-device query failed");
        void* allocation = nullptr;
#if CUDART_VERSION >= 11020
        check_cuda(cudaMallocAsync(&allocation, size * sizeof(T), nullptr),
                   "CUDA pooled allocation failed");
#else
        check_cuda(cudaMalloc(&allocation, size * sizeof(T)), "CUDA allocation failed");
#endif
        data_ = static_cast<T*>(allocation);
    }

    PooledDeviceBuffer(const PooledDeviceBuffer&) = delete;
    PooledDeviceBuffer& operator=(const PooledDeviceBuffer&) = delete;

    ~PooledDeviceBuffer()
    {
        free_device_memory(data_, device_id_, true);
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

  private:
    T* data_{};
    int device_id_{-1};
};

} // namespace neurale::models::cuda::detail
