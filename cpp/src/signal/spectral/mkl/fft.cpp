/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fft.h"
#include "mkl_utils.h"

#include <mkl_dfti.h>

#include <array>
#include <complex>
#include <memory>

namespace neurale::signal::detail
{
namespace
{

class Descriptor
{
  public:
    explicit Descriptor(std::size_t length)
    {
        mkl::check_mkl_status(DftiCreateDescriptor(&handle_, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                                                   mkl::checked_mkl_long(length, "FFT length")),
                              "DftiCreateDescriptor");
        mkl::check_mkl_status(DftiSetValue(handle_, DFTI_PLACEMENT, DFTI_INPLACE),
                              "DftiSetValue(DFTI_PLACEMENT)");
        mkl::check_mkl_status(
            DftiSetValue(handle_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(length)),
            "DftiSetValue(DFTI_BACKWARD_SCALE)");
        mkl::check_mkl_status(DftiCommitDescriptor(handle_), "DftiCommitDescriptor");
    }

    ~Descriptor()
    {
        if (handle_ != nullptr)
        {
            DftiFreeDescriptor(&handle_);
        }
    }

    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;

    DFTI_DESCRIPTOR_HANDLE get() const noexcept
    {
        return handle_;
    }

  private:
    DFTI_DESCRIPTOR_HANDLE handle_ = nullptr;
};

struct CachedDescriptor
{
    std::size_t length = 0;
    std::size_t generation = 0;
    std::unique_ptr<Descriptor> descriptor;
};

Descriptor& cached_descriptor(std::size_t length)
{
    thread_local std::array<CachedDescriptor, 4> cache;
    thread_local std::size_t generation = 0;
    ++generation;

    for (auto& entry : cache)
    {
        if (entry.descriptor != nullptr && entry.length == length)
        {
            entry.generation = generation;
            return *entry.descriptor;
        }
    }

    auto* replacement = &cache.front();
    for (auto& entry : cache)
    {
        if (entry.descriptor == nullptr || entry.generation < replacement->generation)
        {
            replacement = &entry;
        }
    }
    replacement->descriptor = std::make_unique<Descriptor>(length);
    replacement->length = length;
    replacement->generation = generation;
    return *replacement->descriptor;
}

} // namespace

void fft_mkl_inplace(std::span<std::complex<double>> values, bool inverse)
{
    if (values.size() <= 1)
    {
        return;
    }
    static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                  "MKL complex layout requires adjacent real and imaginary doubles");
    auto& descriptor = cached_descriptor(values.size());
    const auto status = inverse ? DftiComputeBackward(descriptor.get(), values.data())
                                : DftiComputeForward(descriptor.get(), values.data());
    mkl::check_mkl_status(status, inverse ? "DftiComputeBackward" : "DftiComputeForward");
}

} // namespace neurale::signal::detail
