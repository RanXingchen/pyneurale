# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_runtime
    STATIC
    cpu_info.cpp
    runtime_info.cpp
)
add_library(neurale::runtime ALIAS neurale_runtime)

target_include_directories(
    neurale_runtime
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
)

target_compile_definitions(
    neurale_runtime
    PRIVATE
    NEURALE_NATIVE_ABI_VERSION=${NEURALE_NATIVE_ABI_VERSION}
    NEURALE_VERSION="${SKBUILD_PROJECT_VERSION_FULL}"
    NEURALE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    NEURALE_BUILD_TYPE="$<CONFIG>"
)

if(NEURALE_MKL_FOUND)
    target_compile_definitions(neurale_runtime PRIVATE NEURALE_WITH_MKL=1)
    target_link_libraries(neurale_runtime PRIVATE MKL::MKL)
endif()

if(OpenMP_CXX_FOUND AND NOT NEURALE_MKL_FOUND)
    target_compile_definitions(neurale_runtime PRIVATE NEURALE_WITH_OPENMP=1)
    target_link_libraries(neurale_runtime PRIVATE OpenMP::OpenMP_CXX)
endif()

if(NEURALE_CUDA_FOUND)
    target_compile_definitions(
        neurale_runtime
        PRIVATE
        NEURALE_WITH_CUDA_EXTENSION=1
        NEURALE_CUDA_TOOLKIT_VERSION="${CUDAToolkit_VERSION}"
    )

    add_library(
        neurale_runtime_cuda
        STATIC
        cuda_info.cpp
    )
    add_library(neurale::runtime_cuda ALIAS neurale_runtime_cuda)
    target_include_directories(
        neurale_runtime_cuda
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
    )
    target_link_libraries(neurale_runtime_cuda PRIVATE CUDA::cudart)
endif()

if(WIN32 AND NEURALE_MKL_FOUND)
    set(_neurale_iomp_candidates "")
    if(DEFINED ENV{ONEAPI_ROOT})
        file(TO_CMAKE_PATH "$ENV{ONEAPI_ROOT}" _neurale_oneapi_root)
        list(APPEND _neurale_iomp_candidates
            "${_neurale_oneapi_root}/compiler/latest/bin/libiomp5md.dll"
        )
    endif()
    list(APPEND _neurale_iomp_candidates
        "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/libiomp5md.dll"
    )
    foreach(candidate IN LISTS _neurale_iomp_candidates)
        if(EXISTS "${candidate}")
            cmake_path(
                GET candidate
                PARENT_PATH
                _neurale_iomp_runtime_dir
            )

            set(
                NEURALE_IOMP_RUNTIME_DLL
                "${candidate}"
                CACHE INTERNAL
                "Intel OpenMP runtime DLL used by PyNeurale"
            )
            set(
                NEURALE_IOMP_RUNTIME_DIR
                "${_neurale_iomp_runtime_dir}"
                CACHE INTERNAL
                "Directory containing the Intel OpenMP runtime DLL"
            )

            install(
                FILES "${candidate}"
                DESTINATION neurale
            )
            break()
        endif()
    endforeach()

    if(NOT NEURALE_IOMP_RUNTIME_DLL)
        message(FATAL_ERROR
            "oneMKL was enabled with intel_thread, but "
            "libiomp5md.dll was not found"
        )
    endif()
endif()
