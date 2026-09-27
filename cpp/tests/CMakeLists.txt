# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# The shared, owner-less test infrastructure -- check_returns.h / check_counts.h
# -- lives at this tree's root, the way cpp/benchmarks/allocation_tracker.h
# does, so a test in any owner subdirectory can include it by name.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Register a C++ unit test executable. SOURCES, INCLUDE_DIRS, and LIBRARIES are
# forwarded to add_executable / target_include_directories / target_link_libraries.
# The CTest name is derived from the target by stripping a trailing "_test"
# suffix, matching the convention used throughout this file; pass TEST_NAME to
# register the run under a different name. TIMEOUT defaults to 60s; pass TIMEOUT
# to override. Pass LABELS to attach CTest labels.
#
# Pass WINDOW to register a second CTest entry, "<target minus _test>_window",
# that runs the same executable with --window. That is how a presentation test
# is driven twice: once headless, and once against a real window, where exit
# code 77 means the host has no display and the run is skipped rather than
# failed. The .github/workflows/test.yml display jobs select exactly those
# entries by name, so the "_window" suffix is a published name, not a detail.
function(neurale_add_test target_name)
    cmake_parse_arguments(PARSE_ARGV 1 _arg
        "WINDOW" "TIMEOUT;TEST_NAME" "SOURCES;INCLUDE_DIRS;LIBRARIES;LABELS")
    if(NOT DEFINED _arg_TIMEOUT)
        set(_arg_TIMEOUT 60)
    endif()
    add_executable(${target_name} ${_arg_SOURCES})
    if(_arg_INCLUDE_DIRS)
        target_include_directories(${target_name} PRIVATE ${_arg_INCLUDE_DIRS})
    endif()
    target_link_libraries(${target_name} PRIVATE ${_arg_LIBRARIES})
    string(REGEX REPLACE "_test$" "" _base_name ${target_name})
    if(DEFINED _arg_TEST_NAME)
        set(_test_name ${_arg_TEST_NAME})
    else()
        set(_test_name ${_base_name})
    endif()
    add_test(NAME ${_test_name} COMMAND ${target_name})
    set_tests_properties(${_test_name} PROPERTIES TIMEOUT ${_arg_TIMEOUT})
    if(_arg_LABELS)
        set_tests_properties(${_test_name} PROPERTIES LABELS "${_arg_LABELS}")
    endif()
    if(_arg_WINDOW)
        add_test(NAME ${_base_name}_window COMMAND ${target_name} --window)
        set_tests_properties(${_base_name}_window
            PROPERTIES TIMEOUT ${_arg_TIMEOUT} SKIP_RETURN_CODE 77)
        if(_arg_LABELS)
            set_tests_properties(${_base_name}_window
                PROPERTIES LABELS "${_arg_LABELS}")
        endif()
    endif()
endfunction()

# Compile every documented domain header as the first include in an independent
# translation unit. This catches reliance on a consumer's include order without
# turning the source headers into an installed C++ SDK or linking any domain.
file(GLOB_RECURSE _neurale_public_headers CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/cpp/include/neurale/*.h"
)
set(_neurale_header_check_dir "${CMAKE_CURRENT_BINARY_DIR}/public_header_checks")
file(MAKE_DIRECTORY "${_neurale_header_check_dir}")
set(_neurale_header_check_sources "")
foreach(_header IN LISTS _neurale_public_headers)
    file(RELATIVE_PATH _relative_header "${PROJECT_SOURCE_DIR}/cpp/include" "${_header}")
    string(MAKE_C_IDENTIFIER "${_relative_header}" _header_identifier)
    set(_source "${_neurale_header_check_dir}/${_header_identifier}.cpp")
    file(GENERATE
        OUTPUT "${_source}"
        CONTENT "#include <${_relative_header}>\nnamespace { constexpr int ${_header_identifier}_included = 1; }\n"
    )
    list(APPEND _neurale_header_check_sources "${_source}")
endforeach()
add_library(
    neurale_public_header_self_containment
    OBJECT
    ${_neurale_header_check_sources}
)
target_compile_features(neurale_public_header_self_containment PRIVATE cxx_std_20)
target_include_directories(
    neurale_public_header_self_containment
    PRIVATE "${PROJECT_SOURCE_DIR}/cpp/include"
)

set(_neurale_pipeline_include_dirs
    "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    "${PROJECT_SOURCE_DIR}/cpp/src/signal/filtering"
)

neurale_add_test(neurale_signal_simulation_test
    SOURCES signal/simulation_test.cpp
    LIBRARIES neurale::signal)

neurale_add_test(neurale_signal_neural_simulation_test
    SOURCES signal/neural_simulation_test.cpp
    LIBRARIES neurale::signal)

neurale_add_test(neurale_devices_simulation_test
    SOURCES devices/simulation_test.cpp
    LIBRARIES neurale::devices neurale::streaming neurale::signal)

neurale_add_test(neurale_devices_provider_test
    SOURCES devices/provider_test.cpp
    LIBRARIES neurale::devices neurale::streaming)

foreach(_fixture IN ITEMS provider_fixture provider_bad_abi_fixture)
    add_library(${_fixture} SHARED devices/provider_fixture.cpp)
    target_include_directories(${_fixture} PRIVATE "${PROJECT_SOURCE_DIR}/src/neurale/devices/sdk")
    target_compile_features(${_fixture} PRIVATE cxx_std_17)
endforeach()
target_compile_definitions(provider_bad_abi_fixture PRIVATE PN_BAD_ABI)
target_compile_definitions(neurale_devices_provider_test PRIVATE
    PN_FIXTURE_PATH="$<TARGET_FILE:provider_fixture>"
    PN_BAD_FIXTURE_PATH="$<TARGET_FILE:provider_bad_abi_fixture>")
add_dependencies(neurale_devices_provider_test provider_fixture provider_bad_abi_fixture)

# Links only the value contract. If this ever needs another neurale target, the
# contract has stopped being a value contract.
neurale_add_test(neurale_experiments_contract_test
    SOURCES experiments/contract_test.cpp
    LIBRARIES neurale::experiments)

neurale_add_test(neurale_experiments_contract_status_test
    SOURCES experiments/contract_status_test.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/execution"
    LIBRARIES neurale::experiments neurale::streaming)

neurale_add_test(neurale_experiments_assistance_test
    SOURCES experiments/assistance_test.cpp
    LIBRARIES neurale::experiments)

neurale_add_test(neurale_experiments_center_out_test
    SOURCES experiments/center_out_test.cpp
    LIBRARIES neurale::experiments)

# Semantic replay links only the value contract too: a replay that needed the
# streaming runtime or the recorder would be re-executing something other than
# the paradigm's own decisions.
neurale_add_test(neurale_experiments_replay_test
    SOURCES experiments/replay_test.cpp
    LIBRARIES neurale::experiments)

neurale_add_test(neurale_experiments_center_out_guidance_test
    SOURCES experiments/center_out_guidance_test.cpp
    LIBRARIES neurale::experiments)

neurale_add_test(neurale_experiments_ssvep_test
    SOURCES experiments/ssvep_test.cpp
    LIBRARIES neurale::experiments)
neurale_add_test(neurale_experiments_speech_test
    SOURCES experiments/speech_test.cpp
    LIBRARIES neurale::experiments)

neurale_add_test(neurale_experiments_webgrid_test
    SOURCES experiments/webgrid_test.cpp
    LIBRARIES neurale::experiments)

neurale_add_test(neurale_experiments_headless_execution_test
    SOURCES experiments/headless_execution_test.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/execution"
    LIBRARIES neurale_execution neurale::experiments neurale::streaming)

neurale_add_test(neurale_experiments_session_test
    SOURCES experiments/session_test.cpp
    INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/cpp/src/execution"
        "${PROJECT_SOURCE_DIR}/cpp/src/recording"
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
    LIBRARIES neurale_execution neurale::experiments neurale::streaming
              neurale_recording neurale_recording_spool)

if(NEURALE_ENABLE_EXPERIMENT_PRESENTATION STREQUAL "ON")
    neurale_add_test(neurale_experiment_presentation_dependency_test
        SOURCES presentation/dependency_test.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES neurale_experiment_presentation
        LABELS presentation)

    neurale_add_test(neurale_experiment_presentation_surface_test
        SOURCES presentation/surface_test.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES neurale_experiment_presentation neurale::experiments
        TEST_NAME neurale_experiment_presentation_surface_contract
        LABELS presentation
        WINDOW)

    neurale_add_test(neurale_experiment_presentation_center_out_test
        SOURCES presentation/center_out_test.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES neurale_experiment_presentation neurale::experiments
        TEST_NAME neurale_experiment_presentation_center_out_contract
        LABELS presentation
        WINDOW)

    neurale_add_test(neurale_experiment_presentation_ssvep_test
        SOURCES presentation/ssvep_test.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES neurale_experiment_presentation neurale::experiments
        TEST_NAME neurale_experiment_presentation_ssvep_contract
        LABELS presentation
        WINDOW)

    neurale_add_test(neurale_experiment_presentation_webgrid_test
        SOURCES presentation/webgrid_test.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES neurale_experiment_presentation neurale::experiments
        TEST_NAME neurale_experiment_presentation_webgrid_contract
        LABELS presentation
        WINDOW)

    neurale_add_test(neurale_experiment_presentation_speech_test
        SOURCES presentation/speech_test.cpp
        INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
        LIBRARIES neurale_experiment_presentation neurale::experiments
        TEST_NAME neurale_experiment_presentation_speech_contract
        LABELS presentation
        WINDOW)

    neurale_add_test(
        neurale_experiment_presentation_recording_test
        SOURCES presentation/recording_test.cpp
        INCLUDE_DIRS
            "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
            "${PROJECT_SOURCE_DIR}/cpp/src/execution"
            "${PROJECT_SOURCE_DIR}/cpp/src/recording"
            "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
        LIBRARIES
            neurale_experiment_presentation_recording
            neurale_execution
            neurale_experiment_presentation
            neurale::experiments
            neurale::streaming
        LABELS presentation recording
        WINDOW
    )
endif()

neurale_add_test(neurale_sorting_detection_test
    SOURCES sorting/detection_test.cpp
    LIBRARIES neurale::sorting)

neurale_add_test(neurale_sorting_online_detection_test
    SOURCES sorting/online_detection_test.cpp
    LIBRARIES neurale::sorting)

neurale_add_test(neurale_sorting_valley_seeking_test
    SOURCES sorting/valley_seeking_test.cpp
    LIBRARIES neurale::sorting)

neurale_add_test(neurale_streaming_schema_test
    SOURCES streaming/schema_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_buffer_pool_test
    SOURCES streaming/buffer_pool_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_frame_validation_test
    SOURCES streaming/frame_validation_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_spsc_ring_test
    SOURCES streaming/spsc_ring_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_continuity_test
    SOURCES streaming/continuity_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_generic_source_test
    SOURCES streaming/generic_source_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_array_replay_test
    SOURCES streaming/array_replay_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_runtime_test
    SOURCES streaming/runtime_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_processor_chain_test
    SOURCES streaming/processor_chain_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_pipeline_sos_filter_adapter_test
    SOURCES pipeline/sos_filter_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_spatial_reference_adapter_test
    SOURCES pipeline/spatial_reference_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_bad_channel_removal_adapter_test
    SOURCES pipeline/bad_channel_removal_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::streaming)

neurale_add_test(neurale_pipeline_spike_detector_adapter_test
    SOURCES pipeline/spike_detector_adapter_test.cpp
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    LIBRARIES neurale_pipeline neurale::sorting neurale::streaming)

neurale_add_test(neurale_execution_interval_runner_test
    SOURCES experiments/interval_runner_test.cpp "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp/src/execution"
                 "${PROJECT_SOURCE_DIR}/cpp/src/recording"
                 "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
                 "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
    LIBRARIES neurale_execution neurale_pipeline neurale::streaming)

neurale_add_test(neurale_pipeline_linear_decoder_adapter_test
    SOURCES
        "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
        pipeline/linear_decoder_adapter_test.cpp
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
        ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::models neurale::streaming)

neurale_add_test(neurale_pipeline_fir_filter_adapter_test
    SOURCES pipeline/fir_filter_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::features neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_iir_filter_adapter_test
    SOURCES pipeline/iir_filter_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::features neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_resampler_adapter_test
    SOURCES pipeline/resampler_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_lmp_feature_adapter_test
    SOURCES pipeline/lmp_feature_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::features neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_kalman_decoder_adapter_test
    SOURCES
        "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
        pipeline/kalman_decoder_adapter_test.cpp
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
        ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::models neurale::features neurale::streaming
              neurale::signal)

neurale_add_test(neurale_integration_simulated_bci_test
    SOURCES
        "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
        integration/simulated_bci_test.cpp
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
        ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::devices neurale::models neurale::features
              neurale::streaming neurale::signal
    TIMEOUT 180)

neurale_add_test(neurale_pipeline_hilbert_envelope_adapter_test
    SOURCES pipeline/hilbert_envelope_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::features neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_multitaper_bandpower_adapter_test
    SOURCES pipeline/multitaper_bandpower_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::features neurale::streaming neurale::signal)

neurale_add_test(neurale_pipeline_feature_stack_adapter_test
    SOURCES pipeline/feature_stack_adapter_test.cpp
    INCLUDE_DIRS ${_neurale_pipeline_include_dirs}
    LIBRARIES neurale_pipeline neurale::features neurale::streaming neurale::signal)

# The strict-realtime contract test is special: one executable backs two CTest
# entries (--contract-only / --allocation-only) and it opts into the full
# malloc/calloc/realloc --wrap tracker when running outside sanitizers on a
# GNU/Clang Linux host. Kept inline rather than routed through neurale_add_test.
add_executable(
    neurale_pipeline_strict_realtime_contract_test
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
    pipeline/strict_realtime_contract_test.cpp
)
# The signal private include dirs are here so the test can assert, through the
# same predicates the kernels dispatch on, that its wide geometry really does
# cross the MKL thresholds. Without that the gate could silently degrade to
# builtin-only coverage.
target_include_directories(
    neurale_pipeline_strict_realtime_contract_test
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
    "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    "${PROJECT_SOURCE_DIR}/cpp/src/signal/filtering"
    "${PROJECT_SOURCE_DIR}/cpp/src/signal/spectral"
)
target_link_libraries(
    neurale_pipeline_strict_realtime_contract_test
    PRIVATE
    neurale::runtime
    neurale::streaming
    neurale::signal
    neurale_pipeline
)
set(_neurale_full_allocation_tracking OFF)
set(_neurale_allocation_tracking_targets
    neurale_pipeline_kalman_decoder_adapter_test
    neurale_pipeline_linear_decoder_adapter_test
    neurale_integration_simulated_bci_test
    neurale_pipeline_strict_realtime_contract_test
)
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
   NOT (NEURALE_ENABLE_ASAN OR NEURALE_ENABLE_UBSAN OR NEURALE_ENABLE_LSAN OR
        NEURALE_ENABLE_TSAN))
    set(_neurale_full_allocation_tracking ON)
    foreach(_target IN LISTS _neurale_allocation_tracking_targets)
        neurale_wrap_allocation(${_target})
    endforeach()
endif()
add_test(
    NAME neurale_pipeline_kalman_decoder_adapter_allocation
    COMMAND neurale_pipeline_kalman_decoder_adapter_test --allocation-only
)
add_test(
    NAME neurale_pipeline_linear_decoder_adapter_allocation
    COMMAND neurale_pipeline_linear_decoder_adapter_test --allocation-only
)
add_test(
    NAME neurale_pipeline_strict_realtime_contract
    COMMAND neurale_pipeline_strict_realtime_contract_test --contract-only
)
add_test(
    NAME neurale_pipeline_strict_realtime_allocation
    COMMAND neurale_pipeline_strict_realtime_contract_test --allocation-only
)
if(_neurale_full_allocation_tracking)
    set(_neurale_strict_allocation_labels "allocation-full")
else()
    set(_neurale_strict_allocation_labels "allocation-operator-new")
endif()
set_tests_properties(
    neurale_pipeline_kalman_decoder_adapter_allocation
    neurale_pipeline_linear_decoder_adapter_allocation
    PROPERTIES
    LABELS "${_neurale_strict_allocation_labels}"
    TIMEOUT 60
)
set_tests_properties(
    neurale_pipeline_strict_realtime_contract
    PROPERTIES
    LABELS "strict-realtime"
    TIMEOUT 300
)
set_tests_properties(
    neurale_pipeline_strict_realtime_allocation
    PROPERTIES
    LABELS "${_neurale_strict_allocation_labels}"
    TIMEOUT 300
)

# --- the private native spool ----------------------------------------------
#
# These link only neurale_recording_spool: the spool container depends on
# nothing above it, and a test that had to drag in the streaming runtime to
# exercise a writer would be evidence that it does.
set(_neurale_recording_include_dirs "${PROJECT_SOURCE_DIR}/cpp/src/recording")
set(_neurale_recorder_include_dirs
    "${PROJECT_SOURCE_DIR}/cpp/src/recording"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
)
set(_neurale_native_spool_vector_dir
    "${PROJECT_SOURCE_DIR}/specifications/native-spool/v1/vectors"
)

neurale_add_test(neurale_recording_spool_checksum_test
    SOURCES recording/spool_checksum_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_spool)

# Reads the committed vectors and index.json straight from the specification
# tree, the way any other language would.
neurale_add_test(neurale_recording_spool_vectors_test
    SOURCES recording/spool_vectors_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_spool)
target_compile_definitions(neurale_recording_spool_vectors_test
    PRIVATE "NEURALE_NATIVE_SPOOL_VECTOR_DIR=\"${_neurale_native_spool_vector_dir}\"")

# One executable, two CTest entries: the writer's behaviour and its
# steady-state allocation gate.
add_executable(
    neurale_recording_spool_writer_test
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
    recording/spool_writer_test.cpp
)
target_include_directories(
    neurale_recording_spool_writer_test
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
    ${_neurale_recording_include_dirs}
)
target_link_libraries(neurale_recording_spool_writer_test PRIVATE neurale_recording_spool)
target_compile_definitions(neurale_recording_spool_writer_test
    PRIVATE "NEURALE_NATIVE_SPOOL_VECTOR_DIR=\"${_neurale_native_spool_vector_dir}\"")
# Same malloc/calloc/realloc --wrap tracker the strict-realtime gate uses, on
# the same host conditions. Applied here rather than in the list above because
# this target is defined after that block runs.
if(_neurale_full_allocation_tracking)
    target_compile_definitions(neurale_recording_spool_writer_test
        PRIVATE NEURALE_BENCHMARK_WRAP_MALLOC=1)
    target_compile_options(neurale_recording_spool_writer_test PRIVATE
        -fno-builtin-malloc
        -fno-builtin-free
    )
    target_link_options(neurale_recording_spool_writer_test PRIVATE
        -Wl,--wrap=malloc
        -Wl,--wrap=calloc
        -Wl,--wrap=realloc
    )
endif()
add_test(
    NAME neurale_recording_spool_writer
    COMMAND neurale_recording_spool_writer_test --contract-only
)
add_test(
    NAME neurale_recording_spool_writer_allocation
    COMMAND neurale_recording_spool_writer_test --allocation-only
)
set_tests_properties(neurale_recording_spool_writer PROPERTIES TIMEOUT 60)
if(_neurale_full_allocation_tracking)
    set_tests_properties(neurale_recording_spool_writer_allocation
        PROPERTIES LABELS "allocation-full" TIMEOUT 60)
else()
    set_tests_properties(neurale_recording_spool_writer_allocation
        PROPERTIES LABELS "allocation-operator-new" TIMEOUT 60)
endif()

neurale_add_test(neurale_recording_spool_failure_test
    SOURCES recording/spool_failure_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_spool)

neurale_add_test(neurale_recording_spool_repair_test
    SOURCES recording/spool_repair_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_spool)

neurale_add_test(neurale_recording_memory_spool_file_test
    SOURCES recording/memory_spool_file_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_spool)

neurale_add_test(neurale_recording_mapped_spool_file_test
    SOURCES recording/mapped_spool_file_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_spool)

# --- the native replay path -------------------------------------------------
#
# The container reader links only the spool (for CRC-32C); the source links
# streaming too, because a source produces frames and discontinuities.
neurale_add_test(neurale_recording_replay_image_test
    SOURCES recording/replay_image_test.cpp
    INCLUDE_DIRS ${_neurale_recording_include_dirs}
    LIBRARIES neurale_recording_replay neurale_recording_spool)

# One executable, two CTest entries: the run's behaviour, and the gate that
# proves a steady-state replay read allocates nothing.
add_executable(
    neurale_recording_replay_source_test
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
    recording/replay_source_test.cpp
)
target_include_directories(
    neurale_recording_replay_source_test
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
    ${_neurale_recording_include_dirs}
)
target_link_libraries(
    neurale_recording_replay_source_test
    PRIVATE
    neurale_recording_replay
    neurale_recording_spool
    neurale::streaming
)
if(_neurale_full_allocation_tracking)
    target_compile_definitions(neurale_recording_replay_source_test
        PRIVATE NEURALE_BENCHMARK_WRAP_MALLOC=1)
    target_compile_options(neurale_recording_replay_source_test PRIVATE
        -fno-builtin-malloc
        -fno-builtin-free
    )
    target_link_options(neurale_recording_replay_source_test PRIVATE
        -Wl,--wrap=malloc
        -Wl,--wrap=calloc
        -Wl,--wrap=realloc
    )
endif()
add_test(
    NAME neurale_recording_replay_source
    COMMAND neurale_recording_replay_source_test --contract-only
)
add_test(
    NAME neurale_recording_replay_source_allocation
    COMMAND neurale_recording_replay_source_test --allocation-only
)
set_tests_properties(neurale_recording_replay_source PROPERTIES TIMEOUT 120)
if(_neurale_full_allocation_tracking)
    set_tests_properties(neurale_recording_replay_source_allocation
        PROPERTIES LABELS "allocation-full" TIMEOUT 60)
else()
    set_tests_properties(neurale_recording_replay_source_allocation
        PROPERTIES LABELS "allocation-operator-new" TIMEOUT 60)
endif()

# --- the whole lifecycle, end to end -----------------------------------------
#
# A harness more than a suite: the images it runs come from a real recording
# made by `tests/unit/recording/test_end_to_end.py`, so the CTest entry here
# is only its self-test. It carries the allocation tracker because one of its
# entries is the steady-state gate over a *real* image rather than a stated one.
add_executable(
    neurale_recording_end_to_end_test
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
    recording/end_to_end_test.cpp
)
target_include_directories(
    neurale_recording_end_to_end_test
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
    ${_neurale_recorder_include_dirs}
)
target_link_libraries(
    neurale_recording_end_to_end_test
    PRIVATE
    neurale_recording
    neurale_recording_replay
    neurale_recording_spool
    neurale::streaming
)
if(_neurale_full_allocation_tracking)
    target_compile_definitions(neurale_recording_end_to_end_test
        PRIVATE NEURALE_BENCHMARK_WRAP_MALLOC=1)
    target_compile_options(neurale_recording_end_to_end_test PRIVATE
        -fno-builtin-malloc
        -fno-builtin-free
    )
    target_link_options(neurale_recording_end_to_end_test PRIVATE
        -Wl,--wrap=malloc
        -Wl,--wrap=calloc
        -Wl,--wrap=realloc
    )
endif()
add_test(
    NAME neurale_recording_end_to_end
    COMMAND neurale_recording_end_to_end_test --self-test
)
set_tests_properties(neurale_recording_end_to_end PROPERTIES TIMEOUT 120)

# --- the native bounded recorder core --------------------------------------
#
# These do link the streaming runtime, because a critical recorder consumes its
# frames and discontinuities. One executable backs two CTest entries again: the
# recorder's behaviour, and the gate that proves the critical callback
# allocates nothing after prepare.
add_executable(
    neurale_recording_recorder_core_test
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks/allocation_tracker.cpp"
    recording/recorder_core_test.cpp
)
target_include_directories(
    neurale_recording_recorder_core_test
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/benchmarks"
    ${_neurale_recorder_include_dirs}
)
target_link_libraries(
    neurale_recording_recorder_core_test
    PRIVATE
    neurale_recording
    neurale_recording_spool
    neurale::streaming
)
if(_neurale_full_allocation_tracking)
    target_compile_definitions(neurale_recording_recorder_core_test
        PRIVATE NEURALE_BENCHMARK_WRAP_MALLOC=1)
    target_compile_options(neurale_recording_recorder_core_test PRIVATE
        -fno-builtin-malloc
        -fno-builtin-free
    )
    target_link_options(neurale_recording_recorder_core_test PRIVATE
        -Wl,--wrap=malloc
        -Wl,--wrap=calloc
        -Wl,--wrap=realloc
    )
endif()
add_test(
    NAME neurale_recording_recorder_core
    COMMAND neurale_recording_recorder_core_test --contract-only
)
if(MSVC)
    # The MSVC Debug runtime allocates once per instrumented submission, so its
    # operator-new count describes the debug runtime rather than the recorder.
    # Optimized MSVC builds, including ASan RelWithDebInfo, retain the strict
    # zero-allocation contract.
    add_test(
        NAME neurale_recording_recorder_core_allocation
        COMMAND neurale_recording_recorder_core_test --allocation-only
        CONFIGURATIONS Release RelWithDebInfo MinSizeRel
    )
else()
    add_test(
        NAME neurale_recording_recorder_core_allocation
        COMMAND neurale_recording_recorder_core_test --allocation-only
    )
endif()
set_tests_properties(neurale_recording_recorder_core PROPERTIES TIMEOUT 120)
if(_neurale_full_allocation_tracking)
    set_tests_properties(neurale_recording_recorder_core_allocation
        PROPERTIES LABELS "allocation-full" TIMEOUT 120)
else()
    set_tests_properties(neurale_recording_recorder_core_allocation
        PROPERTIES LABELS "allocation-operator-new" TIMEOUT 120)
endif()

neurale_add_test(neurale_recording_recorder_failure_test
    SOURCES recording/recorder_failure_test.cpp
    INCLUDE_DIRS ${_neurale_recorder_include_dirs}
    LIBRARIES neurale_recording neurale_recording_spool neurale::streaming
    TIMEOUT 120)

neurale_add_test(neurale_streaming_watchdog_test
    SOURCES streaming/watchdog_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_output_topology_test
    SOURCES streaming/output_topology_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_recording_lifecycle_test
    SOURCES streaming/recording_lifecycle_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_recording_runtime_integration_test
    SOURCES recording/runtime_integration_test.cpp
    INCLUDE_DIRS ${_neurale_recorder_include_dirs}
    LIBRARIES neurale_recording neurale_recording_spool neurale::streaming)

neurale_add_test(neurale_streaming_realtime_thread_test
    SOURCES streaming/realtime_thread_test.cpp
    LIBRARIES neurale::streaming)

neurale_add_test(neurale_streaming_fault_injection_test
    SOURCES streaming/fault_injection_test.cpp
    LIBRARIES neurale::streaming
    LABELS fault-injection
    TIMEOUT 120)

neurale_add_test(neurale_streaming_soak_test
    SOURCES streaming/soak_test.cpp
    LIBRARIES neurale::streaming
    LABELS soak
    TIMEOUT 300)
if(WIN32)
    target_link_libraries(neurale_streaming_soak_test PRIVATE Psapi)
endif()

set(_neurale_streaming_test_targets
    # The pure-contract experiment tests belong here even though they touch no
    # streaming runtime: they link neurale_experiments, which *is* instrumented,
    # so a sanitizer build compiles ubsan calls into the archive they link and
    # then leaves them undefined at link time unless the executable carries the
    # runtime too.
    neurale_experiments_contract_test
    neurale_experiments_contract_status_test
    neurale_experiments_assistance_test
    neurale_experiments_center_out_test
    neurale_experiments_center_out_guidance_test
    neurale_experiments_ssvep_test
    neurale_experiments_speech_test
    neurale_experiments_webgrid_test
    neurale_experiments_replay_test
    neurale_signal_simulation_test
    neurale_signal_neural_simulation_test
    neurale_devices_simulation_test
    neurale_devices_provider_test
    neurale_sorting_detection_test
    neurale_sorting_online_detection_test
    neurale_sorting_valley_seeking_test
    neurale_experiments_headless_execution_test
    neurale_execution_interval_runner_test
    neurale_experiments_session_test
    neurale_streaming_schema_test
    neurale_streaming_buffer_pool_test
    neurale_streaming_continuity_test
    neurale_streaming_generic_source_test
    neurale_streaming_frame_validation_test
    neurale_streaming_runtime_test
    neurale_streaming_processor_chain_test
    neurale_pipeline_sos_filter_adapter_test
    neurale_pipeline_spatial_reference_adapter_test
    neurale_pipeline_bad_channel_removal_adapter_test
    neurale_pipeline_fir_filter_adapter_test
    neurale_pipeline_iir_filter_adapter_test
    neurale_pipeline_resampler_adapter_test
    neurale_pipeline_lmp_feature_adapter_test
    neurale_pipeline_hilbert_envelope_adapter_test
    neurale_pipeline_multitaper_bandpower_adapter_test
    neurale_pipeline_feature_stack_adapter_test
    neurale_pipeline_spike_detector_adapter_test
    neurale_pipeline_kalman_decoder_adapter_test
    neurale_integration_simulated_bci_test
    neurale_pipeline_strict_realtime_contract_test
    neurale_recording_spool_checksum_test
    neurale_recording_spool_vectors_test
    neurale_recording_spool_writer_test
    neurale_recording_spool_failure_test
    neurale_recording_spool_repair_test
    neurale_recording_memory_spool_file_test
    neurale_recording_mapped_spool_file_test
    neurale_recording_recorder_core_test
    neurale_recording_recorder_failure_test
    neurale_streaming_spsc_ring_test
    neurale_streaming_watchdog_test
    neurale_streaming_output_topology_test
    neurale_streaming_recording_lifecycle_test
    neurale_recording_runtime_integration_test
    neurale_recording_replay_image_test
    neurale_recording_replay_source_test
    neurale_recording_end_to_end_test
    neurale_streaming_realtime_thread_test
    neurale_streaming_fault_injection_test
    neurale_streaming_soak_test
)
set(_neurale_streaming_test_names
    neurale_experiments_contract
    neurale_experiments_contract_status
    neurale_experiments_assistance
    neurale_experiments_center_out
    neurale_experiments_center_out_guidance
    neurale_experiments_ssvep
    neurale_experiments_speech
    neurale_experiments_webgrid
    neurale_experiments_replay
    neurale_signal_simulation
    neurale_signal_neural_simulation
    neurale_devices_simulation
    neurale_devices_provider
    neurale_sorting_detection
    neurale_sorting_online_detection
    neurale_sorting_valley_seeking
    neurale_experiments_headless_execution
    neurale_execution_interval_runner
    neurale_experiments_session
    neurale_streaming_schema
    neurale_streaming_buffer_pool
    neurale_streaming_continuity
    neurale_streaming_generic_source
    neurale_streaming_frame_validation
    neurale_streaming_array_replay
    neurale_streaming_runtime
    neurale_streaming_processor_chain
    neurale_pipeline_sos_filter_adapter
    neurale_pipeline_bad_channel_removal_adapter
    neurale_pipeline_spatial_reference_adapter
    neurale_pipeline_fir_filter_adapter
    neurale_pipeline_iir_filter_adapter
    neurale_pipeline_resampler_adapter
    neurale_pipeline_lmp_feature_adapter
    neurale_pipeline_hilbert_envelope_adapter
    neurale_pipeline_multitaper_bandpower_adapter
    neurale_pipeline_feature_stack_adapter
    neurale_pipeline_spike_detector_adapter
    neurale_pipeline_kalman_decoder_adapter
    neurale_pipeline_kalman_decoder_adapter_allocation
    neurale_pipeline_linear_decoder_adapter
    neurale_pipeline_linear_decoder_adapter_allocation
    neurale_integration_simulated_bci
    neurale_pipeline_strict_realtime_contract
    neurale_pipeline_strict_realtime_allocation
    neurale_recording_spool_checksum
    neurale_recording_spool_vectors
    neurale_recording_spool_writer
    neurale_recording_spool_writer_allocation
    neurale_recording_spool_failure
    neurale_recording_spool_repair
    neurale_recording_memory_spool_file
    neurale_recording_mapped_spool_file
    neurale_recording_recorder_core
    neurale_recording_recorder_core_allocation
    neurale_recording_recorder_failure
    neurale_streaming_spsc_ring
    neurale_streaming_watchdog
    neurale_streaming_output_topology
    neurale_streaming_recording_lifecycle
    neurale_recording_runtime_integration
    neurale_recording_replay_image
    neurale_recording_replay_source
    neurale_recording_replay_source_allocation
    neurale_recording_end_to_end
    neurale_streaming_realtime_thread
    neurale_streaming_fault_injection
    neurale_streaming_soak
)

get_property(_neurale_native_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER _neurale_native_test_targets INCLUDE REGEX "_test$")
set(NEURALE_NATIVE_TEST_TARGETS ${_neurale_native_test_targets} PARENT_SCOPE)

if(MSVC AND NEURALE_ENABLE_ASAN)
    cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _neurale_asan_runtime_dir)
    set_tests_properties(
        ${_neurale_streaming_test_names}
        PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${_neurale_asan_runtime_dir}"
    )

endif()

if(WIN32 AND NEURALE_MKL_FOUND)
    if(NOT NEURALE_IOMP_RUNTIME_DIR)
        message(FATAL_ERROR
            "MKL-enabled C++ tests require libiomp5md.dll"
        )
    endif()

    set_property(
        TEST ${_neurale_streaming_test_names}
        APPEND
        PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${NEURALE_IOMP_RUNTIME_DIR}"
    )
endif()
