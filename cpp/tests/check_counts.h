/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// CHECK for a test that reports every failure it finds.
///
/// These suites run to the end and count: CHECK increments a `failures` the
/// including translation unit defines, and `main` returns it. One run then
/// names every assertion that failed instead of only the first, which is what
/// you want where one broken invariant shows up in many places at once. The
/// fail-fast shape is check_returns.h, and the two define CHECK differently on
/// purpose, so a translation unit includes exactly one.
///
/// The diagnostic goes to `stderr` through `std::fprintf` rather than
/// `std::cerr`: several of these suites override `operator new` to count
/// allocations, and an assertion that allocated on its way out would be an
/// assertion that changed what the test measured.

#ifdef NEURALE_TEST_CHECK_RETURNS
#error "check_counts.h and check_returns.h define CHECK differently; include one of them"
#endif
#define NEURALE_TEST_CHECK_COUNTS 1

#include <cstdio>

/// Record a failure at @p condition in the enclosing `failures` counter.
#define CHECK(condition)                                                                           \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::fprintf(stderr, "FAILED: %s at %s:%d\n", #condition, __FILE__, __LINE__);         \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)
