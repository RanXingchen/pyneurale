# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Two internal targets, split so that each one's dependency claim is checked
# rather than asserted. Like neurale_pipeline neither has a public alias, public
# headers, install(), or export(): the spool is a private staging store and the
# recorder core is not a public API, and a target nobody outside this repository
# can link is the strongest form of that statement.

# The container: framing, checksums, scanning, tail repair. It links nothing.
# It depends on the C++ standard library and the platform's file calls, and on
# neither streaming, signal, features, models, sorting, nor Python -- a writer
# that could reach any of them would be a writer that could be made to do work
# on somebody else's thread. Checked in cpp/cmake/DependencyChecks.cmake.
add_library(
    neurale_recording_spool
    STATIC
    crc32c.cpp
    memory_spool_file.cpp
    sha256.cpp
    spool_file.cpp
    spool_layout.cpp
    spool_repair.cpp
    spool_scanner.cpp
    spool_writer.cpp
)

set_property(TARGET neurale_recording_spool PROPERTY NEURALE_INTERNAL_ONLY TRUE)
target_compile_features(neurale_recording_spool PRIVATE cxx_std_20)
target_include_directories(neurale_recording_spool PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# The recorder core: observer callback, bounded queues, spool worker, lifecycle.
# It consumes neurale::streaming's frame, discontinuity, and fault types -- that
# is the whole point of a critical recorder -- and nothing above streaming. The
# direction stays one-way: streaming does not know this target exists.
add_library(
    neurale_recording
    STATIC
    core/record_payloads.cpp
    core/recorder.cpp
    core/recorder_status.cpp
    core/recording_plan.cpp
)

set_property(TARGET neurale_recording PROPERTY NEURALE_INTERNAL_ONLY TRUE)
target_compile_features(neurale_recording PRIVATE cxx_std_20)
target_include_directories(
    neurale_recording
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
)
target_link_libraries(neurale_recording PRIVATE neurale_recording_spool neurale::streaming)

# The replay side: the native reader for the private replay image, and the
# source that turns one into a run. It links the spool (for the container's
# CRC-32C, the only checksum this repository defines) and streaming (whose
# frame, discontinuity, schema, and clock types are what a source produces).
# It does not link neurale_recording: a replay consumes a committed session and
# has nothing to say to the recorder that wrote it, and a target that linked
# both would make that claim unfalsifiable. The direction stays one-way here
# too -- streaming does not know this target exists.
add_library(
    neurale_recording_replay
    STATIC
    replay/image.cpp
    replay/source.cpp
)

set_property(TARGET neurale_recording_replay PROPERTY NEURALE_INTERNAL_ONLY TRUE)
target_compile_features(neurale_recording_replay PRIVATE cxx_std_20)
target_include_directories(neurale_recording_replay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
    neurale_recording_replay
    PRIVATE
    neurale_recording_spool
    neurale::streaming
)

# The recorder failure suite needs deterministic barriers inside an in-flight
# callback. The internal header and branch exist only when native tests are
# built; production/library builds contain neither the hook nor hot-path cost.
if(NEURALE_BUILD_CPP_TESTS)
    target_compile_definitions(neurale_recording PUBLIC NEURALE_RECORDING_TEST_HOOKS=1)
endif()
