# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Private orchestration seam above the pure experiment contract, the generic
# streaming runtime, and the recording core. It has no alias, public headers,
# binding, install, or export surface.
#
# It links neurale_recording because the recording bridge is native: the
# paradigm controllers publish bounded compact trace records from a realtime
# thread, and something has to turn those into control records without going
# anywhere near Python. That something is here, on a bridge thread, and it
# needs the recorder core's own submission surface to do it.
#
# neurale_recording exposes no public headers -- it is internal-only by design,
# like this target -- so the bridge includes its private ones by path. The set
# it may include is stated and checked in
# neurale_assert_execution_dependencies(); reaching further into
# the recording tree fails configuration rather than review.
add_library(
    neurale_execution
    STATIC
    adaptive_center_out_actuator.cpp
    center_out_controller.cpp
    center_out_trace_writer.cpp
    interval_runner.cpp
    records.cpp
    session.cpp
    ssvep_controller.cpp
    speech_controller.cpp
    speech_trace_writer.cpp
    webgrid_controller.cpp
    webgrid_trace_writer.cpp
)

set_property(TARGET neurale_execution PROPERTY NEURALE_INTERNAL_ONLY TRUE)

# Deterministic failure injection for the one bring-up step that fails by
# throwing rather than by returning a status. Mirrors neurale_streaming's own
# hook surface: absent when native tests are not built.
if(NEURALE_BUILD_CPP_TESTS)
    target_compile_definitions(
        neurale_execution
        PUBLIC
        NEURALE_EXPERIMENTS_TEST_HOOKS=1
    )
endif()
target_compile_features(neurale_execution PRIVATE cxx_std_20)
target_include_directories(
    neurale_execution
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
)
target_link_libraries(
    neurale_execution
    PRIVATE
    neurale::experiments
    neurale::streaming
    neurale_recording
    neurale_pipeline
)
