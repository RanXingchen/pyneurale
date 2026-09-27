/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// A global allocation counter, for tests that assert a path allocates nothing.
///
/// The realtime contract (docs/development/native_streaming.md) says the data
/// plane does not allocate once a session has started. The only way a test can
/// witness that from inside the process is to replace the global allocation
/// operators and count, which is what this does: `allocations` advances on every
/// `new`, and a test reads it either side of the region under test.
///
/// **Include this in exactly one translation unit per executable.** It defines
/// the replaceable global `operator new`/`operator delete`, and a second
/// definition in the same program is a link error -- which is also why the
/// counter lives beside them here rather than in each test. For the same
/// reason a target that links ``cpp/benchmarks/allocation_tracker.cpp``
/// must not include this: that tracker already replaces the same operators,
/// and those tests read their count through ``neurale::benchmark`` instead.
///
/// Deallocation is deliberately not counted. What the contract forbids is
/// acquiring memory on the realtime thread; a free that happens to land there
/// is not the failure being looked for, and counting it would only add noise to
/// the number a test compares.
///
/// The nothrow forms are not replaced here. Two presentation tests do replace
/// them, on top of this header, because the code they drive uses them; adding
/// them for everybody would change the count every other test asserts on.

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace
{
/// Allocations since the program started. Tests take a difference, not a total.
std::atomic<std::size_t> allocations{};
} // namespace

void* operator new(std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (auto* value = std::malloc(size == 0 ? 1 : size))
    {
        return value;
    }
    throw std::bad_alloc{};
}

void* operator new[](std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (auto* value = std::malloc(size == 0 ? 1 : size))
    {
        return value;
    }
    throw std::bad_alloc{};
}

void operator delete(void* value) noexcept
{
    std::free(value);
}

void operator delete[](void* value) noexcept
{
    std::free(value);
}

void operator delete(void* value, std::size_t) noexcept
{
    std::free(value);
}

void operator delete[](void* value, std::size_t) noexcept
{
    std::free(value);
}
