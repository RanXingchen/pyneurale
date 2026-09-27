# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Route a target's allocations through the linker's malloc wrappers, so
# cpp/benchmarks/allocation_tracker.cpp counts every acquisition rather than
# only the ones that reach `operator new`.
#
# The three settings are one setting. `NEURALE_BENCHMARK_WRAP_MALLOC=1` tells
# the tracker to define __wrap_malloc and friends; `-Wl,--wrap=...` is what
# redirects calls to them; and `-fno-builtin-malloc/-free` is what keeps the
# calls there to redirect. Under -O3 GCC treats std::malloc/std::free as
# builtins and dead-code-eliminates unused malloc/free pairs at compile time,
# before the linker wrap can act -- which would silently drop those allocations
# from the counter and break both the allocation_tracker self-test and the
# zero-allocation hot-path guards. A target that got two of the three would
# still build, and would still report a number; it would just be the wrong one.
#
# Whether a target *should* be wrapped is the caller's decision: the benchmarks
# wrap on any GNU/Clang Linux host, while the tests additionally refuse when a
# sanitizer is on, because a sanitizer replaces the allocator itself.
function(neurale_wrap_allocation target)
    target_compile_definitions(${target} PRIVATE NEURALE_BENCHMARK_WRAP_MALLOC=1)
    target_compile_options(${target} PRIVATE
        -fno-builtin-malloc
        -fno-builtin-free
    )
    target_link_options(${target} PRIVATE
        -Wl,--wrap=malloc
        -Wl,--wrap=calloc
        -Wl,--wrap=realloc
    )
endfunction()
