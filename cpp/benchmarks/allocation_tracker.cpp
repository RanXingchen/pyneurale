/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_tracker.h"

#include <atomic>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace
{

std::atomic<bool> tracking_enabled{};
std::atomic<std::uint64_t> allocations{};

void record_allocation() noexcept
{
    if (tracking_enabled.load(std::memory_order_relaxed))
    {
        allocations.fetch_add(1, std::memory_order_relaxed);
    }
}

[[nodiscard]] void* allocate(std::size_t size)
{
    record_allocation();
    if (void* ptr = std::malloc(size == 0 ? 1 : size))
    {
        return ptr;
    }
    throw std::bad_alloc{};
}

[[nodiscard]] void* allocate_aligned(std::size_t size, std::size_t alignment)
{
    record_allocation();
#ifdef _WIN32
    if (void* ptr = _aligned_malloc(size == 0 ? 1 : size, alignment))
    {
        return ptr;
    }
#else
    const auto rounded = (size + alignment - 1) / alignment * alignment;
    if (void* ptr = std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded))
    {
        return ptr;
    }
#endif
    throw std::bad_alloc{};
}

} // namespace

void* operator new(std::size_t size)
{
    return allocate(size);
}
void* operator new[](std::size_t size)
{
    return allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    try
    {
        return allocate(size);
    }
    catch (...)
    {
        return nullptr;
    }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    try
    {
        return allocate(size);
    }
    catch (...)
    {
        return nullptr;
    }
}
void* operator new(std::size_t size, std::align_val_t alignment)
{
    return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    try
    {
        return allocate_aligned(size, static_cast<std::size_t>(alignment));
    }
    catch (...)
    {
        return nullptr;
    }
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    try
    {
        return allocate_aligned(size, static_cast<std::size_t>(alignment));
    }
    catch (...)
    {
        return nullptr;
    }
}

void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept
{
    std::free(pointer);
}

#ifdef _WIN32
void operator delete(void* pointer, std::align_val_t) noexcept
{
    _aligned_free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept
{
    _aligned_free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept
{
    _aligned_free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept
{
    _aligned_free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
    _aligned_free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
    _aligned_free(pointer);
}
#else
void operator delete(void* pointer, std::align_val_t) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
    std::free(pointer);
}
#endif

#ifdef NEURALE_BENCHMARK_WRAP_MALLOC
extern "C" void* __real_malloc(std::size_t size);
extern "C" void* __real_calloc(std::size_t count, std::size_t size);
extern "C" void* __real_realloc(void* pointer, std::size_t size);

extern "C" void* __wrap_malloc(std::size_t size)
{
    record_allocation();
    return __real_malloc(size);
}

extern "C" void* __wrap_calloc(std::size_t count, std::size_t size)
{
    record_allocation();
    return __real_calloc(count, size);
}

extern "C" void* __wrap_realloc(void* pointer, std::size_t size)
{
    if (size != 0)
    {
        record_allocation();
    }
    return __real_realloc(pointer, size);
}
#endif

namespace neurale::benchmark
{

void reset_allocation_count() noexcept
{
    allocations.store(0, std::memory_order_relaxed);
}

void set_allocation_tracking(bool enabled) noexcept
{
    tracking_enabled.store(enabled, std::memory_order_release);
}

std::uint64_t allocation_count() noexcept
{
    return allocations.load(std::memory_order_acquire);
}

std::string_view allocation_tracking_backend() noexcept
{
#ifdef NEURALE_BENCHMARK_WRAP_MALLOC
    return "operator_new+malloc";
#else
    return "operator_new";
#endif
}

AllocationScope::AllocationScope() noexcept
{
    reset_allocation_count();
    set_allocation_tracking(true);
}

AllocationScope::~AllocationScope() noexcept
{
    stop();
}

std::uint64_t AllocationScope::count() const noexcept
{
    return allocation_count();
}

void AllocationScope::stop() noexcept
{
    if (active_)
    {
        set_allocation_tracking(false);
        active_ = false;
    }
}

} // namespace neurale::benchmark
