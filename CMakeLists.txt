# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME pyneurale)
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    set(SKBUILD_PROJECT_VERSION 0.0.0)
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION_FULL)
    set(SKBUILD_PROJECT_VERSION_FULL ${SKBUILD_PROJECT_VERSION})
endif()

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NEURALE_NATIVE_ABI_VERSION 1)
set(NEURALE_ENABLE_MKL "AUTO" CACHE STRING "Enable MKL runtime support: AUTO, ON, or OFF")
set(NEURALE_ENABLE_CUDA "AUTO" CACHE STRING "Enable CUDA runtime probing: AUTO, ON, or OFF")
set(
    NEURALE_ENABLE_EXPERIMENT_PRESENTATION
    "OFF"
    CACHE STRING
    "Build optional native experiment presentation support: ON or OFF"
)
set(NEURALE_CUDA_ARCHITECTURES "" CACHE STRING "CUDA architectures for native model kernels")
option(NEURALE_BUILD_CPP_TESTS "Build PyNeurale C++ unit tests" OFF)
option(NEURALE_BUILD_BENCHMARKS "Build PyNeurale native benchmarks" OFF)
option(NEURALE_ENABLE_ASAN "Instrument native targets and tests with AddressSanitizer" OFF)
option(NEURALE_ENABLE_UBSAN "Instrument native targets and tests with UndefinedBehaviorSanitizer" OFF)
option(NEURALE_ENABLE_LSAN "Instrument native targets and tests with LeakSanitizer" OFF)
option(NEURALE_ENABLE_TSAN "Instrument native targets and tests with ThreadSanitizer" OFF)
set_property(CACHE NEURALE_ENABLE_MKL PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE NEURALE_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE NEURALE_ENABLE_EXPERIMENT_PRESENTATION PROPERTY STRINGS ON OFF)

if(NEURALE_ENABLE_TSAN AND
   (NEURALE_ENABLE_ASAN OR NEURALE_ENABLE_UBSAN OR NEURALE_ENABLE_LSAN))
    message(FATAL_ERROR "ThreadSanitizer must use a separate build directory")
endif()

if((NEURALE_ENABLE_ASAN OR NEURALE_ENABLE_UBSAN OR NEURALE_ENABLE_LSAN OR
    NEURALE_ENABLE_TSAN) AND NOT NEURALE_BUILD_CPP_TESTS)
    message(FATAL_ERROR "Native sanitizers require NEURALE_BUILD_CPP_TESTS=ON")
endif()

function(neurale_normalize_feature_option option_name)
    if(DEFINED ENV{${option_name}} AND "${${option_name}}" STREQUAL "AUTO")
        set(${option_name} "$ENV{${option_name}}" PARENT_SCOPE)
    endif()
endfunction()

neurale_normalize_feature_option(NEURALE_ENABLE_MKL)
neurale_normalize_feature_option(NEURALE_ENABLE_CUDA)
string(TOUPPER "${NEURALE_ENABLE_MKL}" NEURALE_ENABLE_MKL)
string(TOUPPER "${NEURALE_ENABLE_CUDA}" NEURALE_ENABLE_CUDA)
string(
    TOUPPER
    "${NEURALE_ENABLE_EXPERIMENT_PRESENTATION}"
    NEURALE_ENABLE_EXPERIMENT_PRESENTATION
)

foreach(option_name IN ITEMS NEURALE_ENABLE_MKL NEURALE_ENABLE_CUDA)
    if(NOT ${option_name} MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR "${option_name} must be AUTO, ON, or OFF")
    endif()
endforeach()

if(NOT NEURALE_ENABLE_EXPERIMENT_PRESENTATION MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR
        "NEURALE_ENABLE_EXPERIMENT_PRESENTATION must be ON or OFF"
    )
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(NEURALE_MKL_FOUND OFF)
if(NOT NEURALE_ENABLE_MKL STREQUAL "OFF")
    if(NOT DEFINED MKL_DIR)
        set(_neurale_mkl_candidates "")
        if(DEFINED ENV{MKLROOT})
            list(APPEND _neurale_mkl_candidates "$ENV{MKLROOT}/lib/cmake/mkl")
        endif()
        if(DEFINED ENV{ONEAPI_ROOT})
            list(APPEND _neurale_mkl_candidates "$ENV{ONEAPI_ROOT}/mkl/latest/lib/cmake/mkl")
        endif()
        if(WIN32)
            list(APPEND _neurale_mkl_candidates
                "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/cmake/mkl"
            )
        else()
            list(APPEND _neurale_mkl_candidates
                "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl"
            )
        endif()
        foreach(candidate IN LISTS _neurale_mkl_candidates)
            if(EXISTS "${candidate}/MKLConfig.cmake")
                set(MKL_DIR "${candidate}" CACHE PATH "Path to MKL CMake configuration")
                break()
            endif()
        endforeach()
    endif()

    # Static MKL avoids shipping the complete set of dynamically dispatched MKL
    # libraries. Intel OpenMP remains a shared runtime on Windows.
    set(MKL_LINK static)
    set(MKL_THREADING intel_thread)
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        set(NEURALE_MKL_FOUND ON)
    elseif(NEURALE_ENABLE_MKL STREQUAL "ON")
        message(FATAL_ERROR
            "NEURALE_ENABLE_MKL=ON, but oneMKL was not found. "
            "Set MKL_DIR or MKLROOT, or configure with NEURALE_ENABLE_MKL=OFF."
        )
    endif()
endif()

if(NEURALE_MKL_FOUND)
    message(STATUS "PyNeurale MKL backend: enabled")
else()
    message(STATUS "PyNeurale MKL backend: disabled (${NEURALE_ENABLE_MKL})")
endif()

set(NEURALE_CUDA_FOUND OFF)
set(NEURALE_CUDA_KERNELS_FOUND OFF)
if(NOT NEURALE_ENABLE_CUDA STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(NEURALE_CUDA_FOUND ON)
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 20)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            set(CMAKE_CUDA_EXTENSIONS OFF)
            set(NEURALE_CUDA_KERNELS_FOUND ON)
        elseif(NEURALE_ENABLE_CUDA STREQUAL "ON")
            message(FATAL_ERROR
                "NEURALE_ENABLE_CUDA=ON, but no CUDA compiler was found"
            )
        endif()
    elseif(NEURALE_ENABLE_CUDA STREQUAL "ON")
        message(FATAL_ERROR "NEURALE_ENABLE_CUDA=ON, but the CUDA Toolkit was not found")
    endif()
endif()

find_package(OpenMP QUIET COMPONENTS CXX)

if(NEURALE_ENABLE_EXPERIMENT_PRESENTATION STREQUAL "ON")
    include(cpp/cmake/ExperimentPresentationDependencies.cmake)
    neurale_find_experiment_presentation_dependencies()
    message(STATUS "PyNeurale experiment presentation: enabled")
else()
    message(STATUS "PyNeurale experiment presentation: disabled")
endif()

if(NEURALE_BUILD_CPP_TESTS)
    enable_testing()
endif()

# The NRF v1 schemas have exactly one source of truth, `specifications/nrf/v1`.
# An installed wheel does not carry that tree, but a reader must still validate
# a session it did not write, so the build installs the normative files into the
# package as a runtime resource. Nothing copies them back into `src/`: a second
# on-disk copy is a second thing to keep in step, which is the drift this rule
# exists to make impossible.
set(NEURALE_NRF_V1_SPEC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/specifications/nrf/v1")
set(NEURALE_NRF_V1_SCHEMA_FILES
    "${NEURALE_NRF_V1_SPEC_DIR}/manifest.schema.json"
    "${NEURALE_NRF_V1_SPEC_DIR}/journal-record.schema.json"
    "${NEURALE_NRF_V1_SPEC_DIR}/checkpoint.schema.json"
    "${NEURALE_NRF_V1_SPEC_DIR}/head.schema.json"
    "${NEURALE_NRF_V1_SPEC_DIR}/recovery-report.schema.json"
)

foreach(schema_file IN LISTS NEURALE_NRF_V1_SCHEMA_FILES)
    if(NOT EXISTS "${schema_file}")
        # Failing here beats producing a wheel whose NRF reader raises on the
        # first manifest it is handed.
        message(FATAL_ERROR "Required NRF v1 schema is missing: ${schema_file}")
    endif()
endforeach()

install(
    FILES ${NEURALE_NRF_V1_SCHEMA_FILES}
    DESTINATION neurale/io/nrf/schemas
)

add_subdirectory(cpp)
