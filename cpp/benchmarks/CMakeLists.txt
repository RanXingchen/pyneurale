# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

find_package(Threads REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Register a native benchmark executable. SOURCES, INCLUDE_DIRS and LIBRARIES
# are forwarded to add_executable / target_include_directories /
# target_link_libraries. Benchmarks are separate targets that CTest runs but
# that are not built by default (NEURALE_BUILD_BENCHMARKS).
#
# PLATFORM_INFO defines the NEURALE_BENCHMARK_OS_* macros a record's environment
# fields are read from. WINDOWS_PROCESS_COUNTERS links Psapi, which is what
# GetProcessMemoryInfo needs; pass it wherever the benchmark reports process
# counters.
function(neurale_add_benchmark target_name)
    cmake_parse_arguments(PARSE_ARGV 1 _arg
        "PLATFORM_INFO;WINDOWS_PROCESS_COUNTERS" "" "SOURCES;INCLUDE_DIRS;LIBRARIES")
    add_executable(${target_name} ${_arg_SOURCES})
    if(_arg_INCLUDE_DIRS)
        target_include_directories(${target_name} PRIVATE ${_arg_INCLUDE_DIRS})
    endif()
    target_link_libraries(${target_name} PRIVATE ${_arg_LIBRARIES})
    if(_arg_PLATFORM_INFO)
        target_compile_definitions(${target_name} PRIVATE
            NEURALE_BENCHMARK_OS_NAME="${CMAKE_SYSTEM_NAME}"
            NEURALE_BENCHMARK_OS_VERSION="${CMAKE_SYSTEM_VERSION}"
            NEURALE_BENCHMARK_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
        )
    endif()
    if(_arg_WINDOWS_PROCESS_COUNTERS AND WIN32)
        target_link_libraries(${target_name} PRIVATE Psapi)
    endif()
endfunction()

neurale_add_benchmark(neurale_streaming_benchmark
    SOURCES allocation_tracker.cpp streaming/runtime_benchmark.cpp
    LIBRARIES neurale::runtime neurale::streaming Threads::Threads
)

neurale_add_benchmark(neurale_pipeline_multitaper_bandpower_benchmark
    SOURCES allocation_tracker.cpp pipeline/multitaper_bandpower_benchmark.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    LIBRARIES neurale::runtime neurale::streaming neurale_pipeline
)

neurale_add_benchmark(neurale_pipeline_kalman_decoder_adapter_benchmark
    SOURCES pipeline/kalman_decoder_adapter_benchmark.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    LIBRARIES neurale::runtime neurale::streaming neurale::models neurale_pipeline
)

neurale_add_benchmark(neurale_pipeline_strict_realtime_benchmark
    SOURCES pipeline/strict_realtime_benchmark.cpp
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
        "${PROJECT_SOURCE_DIR}/cpp/src/signal/spectral"
    LIBRARIES neurale::runtime neurale::streaming neurale_pipeline
)

# The SessionRecorder half of this simulation/device measurement lives in
# benchmarks/devices/benchmark_simulation.py.
neurale_add_benchmark(neurale_devices_simulation_benchmark
    SOURCES allocation_tracker.cpp devices/simulation_benchmark.cpp
    LIBRARIES
        neurale::runtime
        neurale::streaming
        neurale::signal
        neurale::devices
        Threads::Threads
    WINDOWS_PROCESS_COUNTERS
)

# Headless semantics characterization. The controllers are private
# execution types, so the benchmark includes their private headers without
# adding an installed/public target or Python surface.
neurale_add_benchmark(neurale_experiments_benchmark
    SOURCES allocation_tracker.cpp experiments/headless_benchmark.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/execution"
    LIBRARIES
        neurale::runtime
        neurale::streaming
        neurale::experiments
        neurale_execution
    PLATFORM_INFO
)

if(NEURALE_ENABLE_EXPERIMENT_PRESENTATION STREQUAL "ON")
    neurale_add_benchmark(neurale_experiment_presentation_benchmark
        SOURCES allocation_tracker.cpp presentation/presenter_benchmark.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES
            neurale::runtime
            neurale::experiments
            neurale_experiment_presentation
        PLATFORM_INFO
        WINDOWS_PROCESS_COUNTERS
    )
endif()

neurale_add_benchmark(neurale_sorting_valley_seeking_benchmark
    SOURCES sorting/valley_seeking_benchmark.cpp
    LIBRARIES neurale::runtime neurale::sorting
)

neurale_add_benchmark(neurale_pipeline_spike_detector_adapter_benchmark
    SOURCES pipeline/spike_detector_adapter_benchmark.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    LIBRARIES neurale::runtime neurale::streaming neurale_pipeline
)

neurale_add_benchmark(neurale_pipeline_spatial_reference_adapter_benchmark
    SOURCES pipeline/spatial_reference_adapter_benchmark.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    LIBRARIES neurale::runtime neurale::streaming neurale_pipeline
)

# Manual record/replay characterization helper. Python owns orchestration,
# finalization/recovery, and the aggregate JSONL artifact; this executable owns
# the deterministic native source and the two measurements whose critical paths
# cannot be timed faithfully from Python.
neurale_add_benchmark(neurale_recording_replay_benchmark
    SOURCES recording/replay_benchmark.cpp
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/cpp/src/recording"
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/replay"
    LIBRARIES
        neurale::runtime
        neurale::streaming
        neurale_recording
        neurale_recording_replay
        Threads::Threads
    WINDOWS_PROCESS_COUNTERS
)

# Every benchmark that reports an allocation count. Sanitizers do not
# instrument benchmark targets, so unlike the tests there is nothing to exclude.
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_neurale_wrapped_benchmarks
        neurale_streaming_benchmark
        neurale_pipeline_multitaper_bandpower_benchmark
        neurale_devices_simulation_benchmark
        neurale_experiments_benchmark
    )
    if(TARGET neurale_experiment_presentation_benchmark)
        list(APPEND _neurale_wrapped_benchmarks neurale_experiment_presentation_benchmark)
    endif()
    foreach(_target IN LISTS _neurale_wrapped_benchmarks)
        neurale_wrap_allocation(${_target})
    endforeach()
endif()

# CTest name matches the target name (benchmarks have no "_test" suffix).
function(neurale_add_benchmark_test target_name)
    cmake_parse_arguments(PARSE_ARGV 1 _arg "" "TIMEOUT" "LABELS")
    if(NOT DEFINED _arg_TIMEOUT)
        set(_arg_TIMEOUT 120)
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES TIMEOUT ${_arg_TIMEOUT})
    if(_arg_LABELS)
        set_tests_properties(${target_name} PROPERTIES LABELS "${_arg_LABELS}")
    endif()
endfunction()

set(_neurale_benchmark_test_targets
    neurale_streaming_benchmark
    neurale_pipeline_multitaper_bandpower_benchmark
    neurale_pipeline_strict_realtime_benchmark
    neurale_sorting_valley_seeking_benchmark
    neurale_pipeline_spike_detector_adapter_benchmark
    neurale_pipeline_spatial_reference_adapter_benchmark
    neurale_pipeline_kalman_decoder_adapter_benchmark
    neurale_devices_simulation_benchmark
    neurale_experiments_benchmark
)
if(TARGET neurale_experiment_presentation_benchmark)
    list(APPEND _neurale_benchmark_test_targets neurale_experiment_presentation_benchmark)
endif()

if(NEURALE_BUILD_CPP_TESTS)
    neurale_add_benchmark_test(neurale_streaming_benchmark
        LABELS benchmark TIMEOUT 120)
    neurale_add_benchmark_test(neurale_pipeline_multitaper_bandpower_benchmark
        LABELS benchmark TIMEOUT 120)
    neurale_add_benchmark_test(neurale_pipeline_strict_realtime_benchmark
        LABELS "benchmark;performance" TIMEOUT 1800)
    neurale_add_benchmark_test(neurale_sorting_valley_seeking_benchmark
        LABELS "benchmark;performance" TIMEOUT 120)
    neurale_add_benchmark_test(neurale_pipeline_spike_detector_adapter_benchmark
        LABELS "benchmark;performance" TIMEOUT 120)
    neurale_add_benchmark_test(neurale_pipeline_spatial_reference_adapter_benchmark
        LABELS "benchmark;performance" TIMEOUT 120)
    neurale_add_benchmark_test(neurale_pipeline_kalman_decoder_adapter_benchmark
        LABELS "benchmark;performance" TIMEOUT 120)
    neurale_add_benchmark_test(neurale_devices_simulation_benchmark
        LABELS "benchmark;performance" TIMEOUT 300)
    neurale_add_benchmark_test(neurale_experiments_benchmark
        LABELS "benchmark;performance" TIMEOUT 300)
    if(TARGET neurale_experiment_presentation_benchmark)
        neurale_add_benchmark_test(neurale_experiment_presentation_benchmark
            LABELS "benchmark;performance;presentation" TIMEOUT 300)
    endif()

    if(MSVC AND NEURALE_ENABLE_ASAN)
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _neurale_asan_runtime_dir)
        set_tests_properties(
            ${_neurale_benchmark_test_targets}
            PROPERTIES
            ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${_neurale_asan_runtime_dir}"
        )
    endif()
    if(WIN32 AND NEURALE_MKL_FOUND)
        if(NOT NEURALE_IOMP_RUNTIME_DIR)
            message(FATAL_ERROR
                "MKL-enabled benchmarks require libiomp5md.dll"
            )
        endif()
        set_tests_properties(
            ${_neurale_benchmark_test_targets}
            PROPERTIES
            ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${NEURALE_IOMP_RUNTIME_DIR}"
        )
    endif()
endif()
