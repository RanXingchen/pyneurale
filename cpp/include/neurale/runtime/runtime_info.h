/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * @brief Runtime build, CPU, threading, and CUDA capability interfaces.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace neurale::runtime
{

/**
 * @brief Build-time capabilities of the native extension.
 */
struct BuildInfo
{
    /// PyNeurale version embedded during compilation.
    std::string version;
    /// Native interface ABI version.
    std::int32_t abi_version;
    /// Compiler name and version.
    std::string compiler;
    /// CMake build configuration.
    std::string build_type;
    /// CPU math backend name, such as ``mkl`` or ``none``.
    std::string cpu_math_backend;
    /// Whether a BLAS implementation is available.
    bool blas_available;
    /// Whether a LAPACK implementation is available.
    bool lapack_available;
    /// FFT backend name, such as ``mkl`` or ``none``.
    std::string fft_backend;
    /// Whether OpenMP-compatible threading is enabled.
    bool openmp_enabled;
    /// Whether the optional CUDA extension was compiled.
    bool cuda_compiled;
    /// CUDA Toolkit version used to build the optional CUDA extension.
    std::optional<std::string> cuda_toolkit_version;
};

/**
 * @brief Host CPU information available without platform-specific Python packages.
 */
struct CpuInfo
{
    /// Normalized host architecture name.
    std::string architecture;
    /// CPU vendor identifier when detectable.
    std::optional<std::string> vendor;
    /// CPU model or brand string when detectable.
    std::optional<std::string> model;
    /// Number of logical processors visible to the process.
    std::uint32_t logical_cores;
};

/**
 * @brief Current native CPU threading state.
 */
struct ThreadingInfo
{
    /// Active threading backend name.
    std::string backend;
    /// Maximum number of threads configured for the backend.
    std::int32_t num_threads;
};

/**
 * @brief One CUDA device reported by the CUDA Runtime API.
 */
struct CudaDeviceInfo
{
    /// Zero-based CUDA device ordinal.
    std::int32_t ordinal;
    /// Device name reported by CUDA.
    std::string name;
    /// Total global device memory in bytes.
    std::uint64_t total_memory;
    /// Compute capability major version.
    std::int32_t compute_capability_major;
    /// Compute capability minor version.
    std::int32_t compute_capability_minor;
    /// Number of streaming multiprocessors.
    std::int32_t n_multiprocessors;
};

/**
 * @brief CUDA build and runtime state.
 */
struct CudaInfo
{
    /// Whether CUDA support was compiled into the queried extension.
    bool compiled;
    /// Whether at least one usable CUDA device is available.
    bool available;
    /// CUDA Runtime API version encoded as an integer.
    std::optional<std::int32_t> runtime_version;
    /// CUDA driver version encoded as an integer.
    std::optional<std::int32_t> driver_version;
    /// Devices successfully enumerated by the CUDA Runtime API.
    std::vector<CudaDeviceInfo> devices;
    /// Explanation when CUDA probing is unavailable or unsuccessful.
    std::optional<std::string> reason;
};

/**
 * @brief Return capabilities embedded in the native CPU extension.
 *
 * @return Build metadata and compiled feature flags.
 */
BuildInfo build_info();

/**
 * @brief Inspect the host CPU.
 *
 * @return Architecture, vendor, model, and logical processor information.
 */
CpuInfo cpu_info();

/**
 * @brief Return the active native threading backend and thread limit.
 *
 * @return Current native threading information.
 */
ThreadingInfo threading_info();

/**
 * @brief Set the native CPU math-library thread limit.
 *
 * @param count Positive number of threads to configure.
 * @throws std::invalid_argument If @p count is not positive.
 */
void set_num_threads(std::int32_t count);

/**
 * @brief Probe CUDA runtime and device availability.
 *
 * @return CUDA runtime, driver, and device information. Probe failures are
 *         represented in the returned object rather than thrown.
 */
CudaInfo cuda_info();

/**
 * @brief Return the current CUDA device ordinal.
 *
 * @return Current zero-based CUDA device ordinal.
 * @throws std::runtime_error If the CUDA Runtime API query fails.
 */
std::int32_t cuda_current_device();

/**
 * @brief Select the current CUDA device.
 *
 * @param ordinal Zero-based CUDA device ordinal.
 * @throws std::invalid_argument If @p ordinal is negative.
 * @throws std::runtime_error If CUDA rejects the requested device.
 */
void cuda_set_device(std::int32_t ordinal);

} // namespace neurale::runtime
