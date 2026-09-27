# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_models
    STATIC
    linalg.cpp
    decomposition.cpp
    linear_model.cpp
    state_space.cpp
    classification.cpp
    neighbors.cpp
    density.cpp
    alignment.cpp
    manifold.cpp
)
add_library(neurale::models ALIAS neurale_models)

target_include_directories(neurale_models PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
target_compile_features(neurale_models PUBLIC cxx_std_20)

set_source_files_properties(
    alignment.cpp
    neighbors.cpp
    PROPERTIES
    COMPILE_OPTIONS "${NEURALE_NO_FAST_MATH_COMPILE_OPTIONS}"
)

target_include_directories(
    neurale_models
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NEURALE_MKL_FOUND)
    target_compile_definitions(
        neurale_models
        PRIVATE
        NEURALE_MODELS_WITH_MKL=1
    )
    target_link_libraries(
        neurale_models
        PRIVATE
        MKL::MKL
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(
        neurale_models
        PRIVATE
        OpenMP::OpenMP_CXX
    )
endif()

if(NEURALE_CUDA_KERNELS_FOUND)
    add_subdirectory(cuda)
endif()

if(NEURALE_MKL_FOUND)
    message(STATUS "PyNeurale models linalg backend: mkl-lapack, builtin")
else()
    message(STATUS "PyNeurale models linalg backend: builtin")
endif()
