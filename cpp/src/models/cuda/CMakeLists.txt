# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_models_cuda
    STATIC
    neighbors.cu
    density.cu
    manifold.cpp
)
add_library(neurale::models_cuda ALIAS neurale_models_cuda)

if(NEURALE_CUDA_ARCHITECTURES)
    set(_neurale_cuda_architectures "${NEURALE_CUDA_ARCHITECTURES}")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set(_neurale_cuda_architectures 75-real 80-real 86-real 89-real 90-real 100-real 120)
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
    set(_neurale_cuda_architectures 75-real 80-real 86-real 89-real 90)
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.1)
    set(_neurale_cuda_architectures 75-real 80-real 86)
else()
    set(_neurale_cuda_architectures 75-real 80)
endif()

target_compile_features(neurale_models_cuda PUBLIC cxx_std_20)
set_source_files_properties(
    neighbors.cu
    PROPERTIES
    COMPILE_OPTIONS "--ftz=false"
)
if(MSVC)
    target_compile_options(
        neurale_models_cuda
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()
target_include_directories(
    neurale_models_cuda
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/src/models"
)
target_link_libraries(
    neurale_models_cuda
    PUBLIC
    neurale::models
    CUDA::cudart
)
set_target_properties(
    neurale_models_cuda
    PROPERTIES
    CUDA_ARCHITECTURES "${_neurale_cuda_architectures}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
