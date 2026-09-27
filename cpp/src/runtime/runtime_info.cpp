/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/runtime/runtime_info.h>

#include <stdexcept>

#ifdef NEURALE_WITH_MKL
#include <mkl_service.h>
#endif

#ifdef NEURALE_WITH_OPENMP
#include <omp.h>
#endif

namespace neurale::runtime
{

BuildInfo build_info()
{
    return BuildInfo{
        NEURALE_VERSION,
        NEURALE_NATIVE_ABI_VERSION,
        NEURALE_COMPILER,
        NEURALE_BUILD_TYPE,
#ifdef NEURALE_WITH_MKL
        "mkl",
        true,
        true,
        "mkl",
#else
        "none",
        false,
        false,
        "none",
#endif
#if defined(NEURALE_WITH_OPENMP) || defined(NEURALE_WITH_MKL)
        true,
#else
        false,
#endif
#ifdef NEURALE_WITH_CUDA_EXTENSION
        true,
        std::string(NEURALE_CUDA_TOOLKIT_VERSION),
#else
        false,
        std::nullopt,
#endif
    };
}

ThreadingInfo threading_info()
{
#ifdef NEURALE_WITH_MKL
    return ThreadingInfo{"mkl", mkl_get_max_threads()};
#elif defined(NEURALE_WITH_OPENMP)
    return ThreadingInfo{"openmp", omp_get_max_threads()};
#else
    return ThreadingInfo{"none", 1};
#endif
}

void set_num_threads(std::int32_t count)
{
    if (count <= 0)
    {
        throw std::invalid_argument("thread count must be positive");
    }
#ifdef NEURALE_WITH_MKL
    mkl_set_num_threads(count);
#endif
#ifdef NEURALE_WITH_OPENMP
    omp_set_num_threads(count);
#endif
}

} // namespace neurale::runtime
