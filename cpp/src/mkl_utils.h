/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mkl.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

/// \file
/// MKL vocabulary shared by every target that links MKL.
///
/// It sits at the root of `cpp/src` rather than inside one domain because two
/// domains need it: `neurale_signal` for its transforms and filters, and
/// `neurale_models` for the decoder kernels. The same reason keeps
/// `allocation_tracker.{h,cpp}` at the root of the benchmark tree.
///
/// `LocalThreadLimit` is the important one. A BLAS or DFTI call on the
/// real-time thread otherwise enters MKL's shared pool and synchronizes it on
/// every invocation, which is a tail-latency cost with no throughput to show
/// for it at these sizes. In `neurale_models` it is also a *correctness*
/// requirement: that target links `OpenMP::OpenMP_CXX` (libgomp) while static
/// MKL brings its own `intel_thread` runtime (libiomp5), and with two OpenMP
/// runtimes in one process a threaded MKL kernel can return different results
/// for the same input.
namespace neurale::mkl
{

inline MKL_INT checked_mkl_int(std::size_t value, const char* name)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<MKL_INT>::max()))
    {
        throw std::overflow_error(std::string(name) + " is too large for MKL");
    }
    return static_cast<MKL_INT>(value);
}

inline MKL_LONG checked_mkl_long(std::size_t value, const char* name)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<MKL_LONG>::max()))
    {
        throw std::overflow_error(std::string(name) + " is too large for MKL");
    }
    return static_cast<MKL_LONG>(value);
}

inline void check_mkl_status(MKL_LONG status, const char* operation)
{
    if (status != DFTI_NO_ERROR)
    {
        throw std::runtime_error(std::string(operation) + " failed: " + DftiErrorMessage(status));
    }
}

inline void check_vsl_status(int status, const char* operation)
{
    if (status != VSL_STATUS_OK)
    {
        throw std::runtime_error(std::string(operation) + " failed with VSL status " +
                                 std::to_string(status));
    }
}

class LocalThreadLimit
{
  public:
    explicit LocalThreadLimit(bool enabled = true)
        : previous_(enabled ? mkl_set_num_threads_local(1) : 0), enabled_(enabled)
    {
    }

    ~LocalThreadLimit()
    {
        if (enabled_)
        {
            mkl_set_num_threads_local(previous_);
        }
    }

    LocalThreadLimit(const LocalThreadLimit&) = delete;
    LocalThreadLimit& operator=(const LocalThreadLimit&) = delete;

  private:
    int previous_;
    bool enabled_;
};

} // namespace neurale::mkl
