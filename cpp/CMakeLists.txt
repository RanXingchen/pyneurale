# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Compile flags that keep IEEE-precise numerics on sources whose results must
# not be reordered or fused by fast-math. Defined once here and reused by the
# sorting and models subdirectories via set_source_files_properties().
set(NEURALE_NO_FAST_MATH_COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-fast-math>;$<$<CXX_COMPILER_ID:MSVC>:/fp:precise>"
)

add_subdirectory(src/runtime)
add_subdirectory(src/streaming)
add_subdirectory(src/signal)
add_subdirectory(src/devices)
add_subdirectory(src/features)
add_subdirectory(src/sorting)
add_subdirectory(src/pipeline)
add_subdirectory(src/recording)
add_subdirectory(src/models)
add_subdirectory(src/experiments)
add_subdirectory(src/execution)
if(NEURALE_ENABLE_EXPERIMENT_PRESENTATION STREQUAL "ON")
    add_subdirectory(src/presentation)
endif()

include(cmake/DependencyChecks.cmake)
neurale_assert_internal_pipeline_dependencies()
neurale_assert_internal_recording_dependencies()
neurale_assert_devices_dependencies()
neurale_assert_experiments_dependencies()

add_subdirectory(bindings)
neurale_assert_cuda_dependencies()
neurale_assert_execution_dependencies()
neurale_assert_experiment_presentation_dependencies()

include(cmake/AllocationWrapping.cmake)

if(NEURALE_BUILD_CPP_TESTS)
    add_subdirectory(tests)
endif()

if(NEURALE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(cmake/Sanitizers.cmake)
neurale_configure_sanitizers()
