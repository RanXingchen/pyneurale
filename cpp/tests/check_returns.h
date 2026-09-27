/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// CHECK for a test that stops at its first failure.
///
/// The check functions in these suites return an `int`, and `main` returns the
/// first non-zero one; CHECK gives back `__LINE__`, so a failing run exits with
/// the line that failed and CTest's exit status locates it without a log. That
/// is the shape of most of the suite. The counting shape -- keep going, report
/// every failure -- is check_counts.h, and the two define CHECK differently on
/// purpose, so a translation unit includes exactly one.
///
/// The diagnostic goes to `stderr` through `std::fprintf` rather than
/// `std::cerr`: several of these suites override `operator new` to count
/// allocations, and an assertion that allocated on its way out would be an
/// assertion that changed what the test measured.

#ifdef NEURALE_TEST_CHECK_COUNTS
#error "check_returns.h and check_counts.h define CHECK differently; include one of them"
#endif
#define NEURALE_TEST_CHECK_RETURNS 1

#include <cstdio>

/// Fail the enclosing check function at @p condition, returning its line.
#define CHECK(condition)                                                                           \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::fprintf(stderr, "FAILED: %s at %s:%d\n", #condition, __FILE__, __LINE__);         \
            return __LINE__;                                                                       \
        }                                                                                          \
    } while (false)
